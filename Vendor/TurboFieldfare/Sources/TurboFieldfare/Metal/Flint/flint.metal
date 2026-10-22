#include <metal_stdlib>
using namespace metal;

// Flint kernels. Compiled into a private library (module "flint") so other
// architectures never pay for them.
//
// Embeddings are MLX affine 8-bit, group 64. Eight-bit values pack one per
// byte, so the stored row is already the unsigned codes. Scales and biases
// are BF16, one pair per group.
//
// Decode attention is one query token. K and V are [position, kvHeads, dim].
// A learned per-head sink is an extra softmax logit with no value: it changes
// the normalizer and nothing else. Query-key scores are scaled; the sink is not.

constant constexpr uint kFlintGroup = 64;

static inline float flint_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void flint_embed_lookup_q8(
    device const uchar*  W     [[buffer(0)]],
    device const ushort* S     [[buffer(1)]],
    device const ushort* B     [[buffer(2)]],
    device       half*   out   [[buffer(3)]],
    constant     uint&   token [[buffer(4)]],
    constant     uint&   dim   [[buffer(5)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) { return; }
    const uint groups = dim / kFlintGroup;
    const uint g = d / kFlintGroup;
    const float scale = flint_bf16(S[token * groups + g]);
    const float bias = flint_bf16(B[token * groups + g]);
    out[d] = half(float(W[token * dim + d]) * scale + bias);
}

// Split-KV decode attention for head dim 128, in two passes.
//
// Pass 1: one threadgroup per (KV head, chunk of positions) with one SIMD
// group per query head sharing that KV head. Each lane owns four of the 128
// dims, so a score is one simd_sum and the loop has no barriers. It writes
// the chunk's running max, normalizer and unnormalized output.
//
// Pass 2: one threadgroup per query head merges the chunks and adds the
// learned sink as one more logit with no value. Scores are scaled; the sink
// is not. K and V are [slot, kvHeads, 128], where a position's slot is the
// position itself, or `position % ringCapacity` when `ringCapacity` is set.
constant constexpr uint kFlintAttnDim = 128;

kernel void flint_attention_partial(
    device const half*  q            [[buffer(0)]],
    device const half*  k            [[buffer(1)]],
    device const half*  v            [[buffer(2)]],
    device       float* mOut         [[buffer(3)]],
    device       float* lOut         [[buffer(4)]],
    device       float* oOut         [[buffer(5)]],
    constant     uint&  numKVHeads   [[buffer(6)]],
    constant     uint&  queriesPerKV [[buffer(7)]],
    constant     uint&  seqLen       [[buffer(8)]],
    constant     uint&  kvStart      [[buffer(9)]],
    constant     uint&  chunkLen     [[buffer(10)]],
    constant     uint&  numChunks    [[buffer(11)]],
    constant     float& scale        [[buffer(12)]],
    constant     uint&  ringCapacity [[buffer(13)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    if (sg >= queriesPerKV) { return; }
    const uint kvHead = tg / numChunks;
    const uint chunk = tg % numChunks;
    const uint qHead = kvHead * queriesPerKV + sg;
    const uint d0 = lane * 4u;
    const float4 qv = float4(*(device const half4*)(q + qHead * kFlintAttnDim + d0));
    const uint kvStride = numKVHeads * kFlintAttnDim;
    const uint begin = kvStart + chunk * chunkLen;
    const uint end = min(begin + chunkLen, seqLen);

    float m = -INFINITY;
    float l = 0.0f;
    float4 acc = float4(0.0f);
    for (uint pos = begin; pos < end; ++pos) {
        const uint slot = ringCapacity > 0u ? pos % ringCapacity : pos;
        const uint base = slot * kvStride + kvHead * kFlintAttnDim + d0;
        const float score = simd_sum(dot(qv, float4(*(device const half4*)(k + base)))) * scale;
        const float next = max(m, score);
        const float alpha = exp(m - next);
        const float p = exp(score - next);
        l = fma(l, alpha, p);
        acc = acc * alpha + p * float4(*(device const half4*)(v + base));
        m = next;
    }
    const uint index = qHead * numChunks + chunk;
    if (lane == 0u) {
        mOut[index] = m;
        lOut[index] = l;
    }
    *(device float4*)(oOut + index * kFlintAttnDim + d0) = acc;
}

kernel void flint_attention_combine(
    device const float*  mIn       [[buffer(0)]],
    device const float*  lIn       [[buffer(1)]],
    device const float*  oIn       [[buffer(2)]],
    device const ushort* sinks     [[buffer(3)]],
    device       half*   out       [[buffer(4)]],
    constant     uint&   numChunks [[buffer(5)]],
    uint qHead [[threadgroup_position_in_grid]],
    uint d     [[thread_position_in_threadgroup]]
) {
    const float sink = flint_bf16(sinks[qHead]);
    const uint first = qHead * numChunks;
    float top = sink;
    for (uint c = 0u; c < numChunks; ++c) { top = max(top, mIn[first + c]); }
    float denom = exp(sink - top);
    float acc = 0.0f;
    for (uint c = 0u; c < numChunks; ++c) {
        const float w = exp(mIn[first + c] - top);
        denom = fma(lIn[first + c], w, denom);
        acc = fma(oIn[(first + c) * kFlintAttnDim + d], w, acc);
    }
    out[qHead * kFlintAttnDim + d] = half(acc / denom);
}

// BF16 router GEMV + sigmoid score: one SIMD group (32 lanes) per expert row,
// 8 SIMD groups (256 threads) per threadgroup. Top-k runs on the CPU from
// these scores (`FlintRouter.select(scores:...)`).
kernel void flint_router_gemv_bf16(
    device const ushort* W           [[buffer(0)]],
    device const half*   x           [[buffer(1)]],
    device       float*  out_scores  [[buffer(2)]],
    constant     uint&   num_experts [[buffer(3)]],
    constant     uint&   D           [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint e = tg_idx * 8u + sg_idx;
    if (e >= num_experts) { return; }
    device const ushort* w_row = W + e * D;
    float acc = 0.0f;
    for (uint i = lane; i < D; i += 32u) {
        acc = fma(flint_bf16(w_row[i]), float(x[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out_scores[e] = 1.0f / (1.0f + exp(-acc));
    }
}

// Replaces +/-inf with +/-65504 in place. NaN and finite values are untouched.
kernel void flint_clamp_inf_f16(
    device       ushort* x     [[buffer(0)]],
    constant     uint&   count [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) { return; }
    const ushort bits = x[i];
    if ((bits & 0x7FFFu) == 0x7C00u) { x[i] = (bits & 0x8000u) | 0x7BFFu; }
}



// Per-head Q and K RMSNorm (BF16 weights) and, when `rotate` is set, NeoX
// rotary encoding, in one dispatch for head dim 128. One threadgroup of 64
// threads per head: thread t owns the pair (t, t + 64). Q heads come first,
// then K heads. Normalized values round to half before rotating, as the
// separate norm and rope kernels do.
kernel void flint_qk_norm_rope(
    device       half*   q         [[buffer(0)]],
    device       half*   k         [[buffer(1)]],
    device const ushort* qWeight   [[buffer(2)]],
    device const ushort* kWeight   [[buffer(3)]],
    constant     uint&   numQHeads [[buffer(4)]],
    constant     uint&   position  [[buffer(5)]],
    constant     float&  theta     [[buffer(6)]],
    constant     uint&   rotate    [[buffer(7)]],
    constant     float&  eps       [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[2];
    const uint half_dim = kFlintAttnDim / 2u;
    const bool isQuery = head < numQHeads;
    device half* h = isQuery ? q + head * kFlintAttnDim
                             : k + (head - numQHeads) * kFlintAttnDim;
    device const ushort* w = isQuery ? qWeight : kWeight;
    const float x0 = float(h[tid]);
    const float x1 = float(h[tid + half_dim]);
    const float sum = simd_sum(fma(x1, x1, x0 * x0));
    if (lane == 0u) { partial[sg] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt((partial[0] + partial[1]) / float(kFlintAttnDim) + eps);
    const float y0 = float(half(x0 * inv * flint_bf16(w[tid])));
    const float y1 = float(half(x1 * inv * flint_bf16(w[tid + half_dim])));
    if (rotate == 0u) {
        h[tid] = half(y0);
        h[tid + half_dim] = half(y1);
        return;
    }
    const float exponent = -float(2u * tid) / float(kFlintAttnDim);
    const float angle = float(position) * pow(theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    h[tid] = half(y0 * cosine - y1 * sine);
    h[tid + half_dim] = half(y0 * sine + y1 * cosine);
}

static inline half flint_saturate(half value) {
    const ushort bits = as_type<ushort>(value);
    return (bits & 0x7FFFu) == 0x7C00u ? as_type<half>(ushort((bits & 0x8000u) | 0x7BFFu))
                                       : value;
}

// hidden += delta, then out = RMSNorm(hidden) * weight (BF16), with +/-inf
// saturated to +/-65504 in both. Dispatch one threadgroup of 256 threads:
// the same layout and reduction order as residual_add_fp16 then
// rmsnorm_bf16w, so finite results match them bit for bit. Each thread
// reads back only the elements it wrote.
kernel void flint_residual_rmsnorm(
    device       half*   hidden [[buffer(0)]],
    device const half*   delta  [[buffer(1)]],
    device const ushort* weight [[buffer(2)]],
    device       half*   out    [[buffer(3)]],
    constant     uint&   D      [[buffer(4)]],
    constant     float&  eps    [[buffer(5)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint sgs  [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[32];
    float acc = 0.0f;
    for (uint i = lid; i < D; i += size) {
        const half h = flint_saturate(half(float(hidden[i]) + float(delta[i])));
        hidden[i] = h;
        const float v = float(h);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { partial[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = lane < sgs ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) { partial[0] = rsqrt(v / float(D) + eps); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint i = lid; i < D; i += size) {
        out[i] = flint_saturate(half(float(hidden[i]) * inv * flint_bf16(weight[i])));
    }
}

// ---------------------------------------------------------------------------
// Prompt prefill: block versions of the decode kernels above, one row per
// token. Each row runs the same arithmetic as the decode kernel, so a row's
// result does not depend on how the prompt was split into chunks.

// Embedding rows for `tokens[t]`, grid (dim, T).
kernel void flint_embed_lookup_q8_block(
    device const uchar*  W      [[buffer(0)]],
    device const ushort* S      [[buffer(1)]],
    device const ushort* B      [[buffer(2)]],
    device       half*   out    [[buffer(3)]],
    device const uint*   tokens [[buffer(4)]],
    constant     uint&   dim    [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= dim) { return; }
    const uint token = tokens[t];
    const uint groups = dim / kFlintGroup;
    const uint g = d / kFlintGroup;
    const float scale = flint_bf16(S[token * groups + g]);
    const float bias = flint_bf16(B[token * groups + g]);
    out[t * dim + d] = half(float(W[token * dim + d]) * scale + bias);
}

// `flint_qk_norm_rope` for T tokens at positions `start + t`: threadgroup
// (head, t), Q rows `qStride` apart and K rows `kStride` apart.
kernel void flint_qk_norm_rope_block(
    device       half*   q         [[buffer(0)]],
    device       half*   k         [[buffer(1)]],
    device const ushort* qWeight   [[buffer(2)]],
    device const ushort* kWeight   [[buffer(3)]],
    constant     uint&   numQHeads [[buffer(4)]],
    constant     uint&   start     [[buffer(5)]],
    constant     float&  theta     [[buffer(6)]],
    constant     uint&   rotate    [[buffer(7)]],
    constant     float&  eps       [[buffer(8)]],
    constant     uint&   qStride   [[buffer(9)]],
    constant     uint&   kStride   [[buffer(10)]],
    uint2 group  [[threadgroup_position_in_grid]],
    uint2 local  [[thread_position_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg      [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[2];
    const uint head = group.x;
    const uint t = group.y;
    const uint tid = local.x;
    const uint half_dim = kFlintAttnDim / 2u;
    const bool isQuery = head < numQHeads;
    device half* h = isQuery ? q + t * qStride + head * kFlintAttnDim
                             : k + t * kStride + (head - numQHeads) * kFlintAttnDim;
    device const ushort* w = isQuery ? qWeight : kWeight;
    const float x0 = float(h[tid]);
    const float x1 = float(h[tid + half_dim]);
    const float sum = simd_sum(fma(x1, x1, x0 * x0));
    if (lane == 0u) { partial[sg] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt((partial[0] + partial[1]) / float(kFlintAttnDim) + eps);
    const float y0 = float(half(x0 * inv * flint_bf16(w[tid])));
    const float y1 = float(half(x1 * inv * flint_bf16(w[tid + half_dim])));
    if (rotate == 0u) {
        h[tid] = half(y0);
        h[tid + half_dim] = half(y1);
        return;
    }
    const float exponent = -float(2u * tid) / float(kFlintAttnDim);
    const float angle = float(start + t) * pow(theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    h[tid] = half(y0 * cosine - y1 * sine);
    h[tid + half_dim] = half(y0 * sine + y1 * cosine);
}

// `flint_router_gemv_bf16` for T rows: grid (ceil(E / 8), T) threadgroups.
kernel void flint_router_gemv_bf16_block(
    device const ushort* W           [[buffer(0)]],
    device const half*   x           [[buffer(1)]],
    device       float*  out_scores  [[buffer(2)]],
    constant     uint&   num_experts [[buffer(3)]],
    constant     uint&   D           [[buffer(4)]],
    uint2 tg     [[threadgroup_position_in_grid]],
    uint sg_idx  [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    const uint e = tg.x * 8u + sg_idx;
    const uint t = tg.y;
    if (e >= num_experts) { return; }
    device const ushort* w_row = W + e * D;
    device const half* x_row = x + t * D;
    float acc = 0.0f;
    for (uint i = lane; i < D; i += 32u) {
        acc = fma(flint_bf16(w_row[i]), float(x_row[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out_scores[t * num_experts + e] = 1.0f / (1.0f + exp(-acc));
    }
}

// act = silu(gate) * up over `count` elements. Inputs and output saturate to
// +/-65504, so an overflowed projection cannot turn into NaN.
kernel void flint_silu_mul(
    device const half* gate  [[buffer(0)]],
    device const half* up    [[buffer(1)]],
    device       half* act   [[buffer(2)]],
    constant     uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) { return; }
    const float g = float(flint_saturate(gate[i]));
    const float u = float(flint_saturate(up[i]));
    act[i] = flint_saturate(half(g / (1.0f + exp(-g)) * u));
}

// `flint_residual_rmsnorm` for T rows, one 256-thread threadgroup per row.
// `delta2` adds a second branch (the routed experts next to the shared one)
// when `hasDelta2` is set.
kernel void flint_residual_rmsnorm_block(
    device       half*   hidden    [[buffer(0)]],
    device const half*   delta     [[buffer(1)]],
    device const ushort* weight    [[buffer(2)]],
    device       half*   out       [[buffer(3)]],
    constant     uint&   D         [[buffer(4)]],
    constant     float&  eps       [[buffer(5)]],
    device const half*   delta2    [[buffer(6)]],
    constant     uint&   hasDelta2 [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint sgs  [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[32];
    device half* h_row = hidden + row * D;
    device const half* d_row = delta + row * D;
    device const half* d2_row = delta2 + row * D;
    device half* o_row = out + row * D;
    float acc = 0.0f;
    for (uint i = lid; i < D; i += size) {
        const float branch = hasDelta2 != 0u
            ? float(flint_saturate(half(float(d_row[i]) + float(d2_row[i]))))
            : float(d_row[i]);
        const half h = flint_saturate(half(float(h_row[i]) + branch));
        h_row[i] = h;
        const float v = float(h);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { partial[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = lane < sgs ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) { partial[0] = rsqrt(v / float(D) + eps); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint i = lid; i < D; i += size) {
        o_row[i] = flint_saturate(half(float(h_row[i]) * inv * flint_bf16(weight[i])));
    }
}

// Prefill attention for T tokens at positions `start + t`, head dim 128: one
// threadgroup per (KV head, token), one SIMD group per query head of that KV
// head, an online softmax over the token's own range and the learned sink.
// Window 0 means full attention. A token's result depends only on its own
// query and the cache, so it does not change with the chunking.
kernel void flint_attention_prefill(
    device const half*   q            [[buffer(0)]],
    device const half*   k            [[buffer(1)]],
    device const half*   v            [[buffer(2)]],
    device const ushort* sinks        [[buffer(3)]],
    device       half*   out          [[buffer(4)]],
    constant     uint&   numKVHeads   [[buffer(5)]],
    constant     uint&   queriesPerKV [[buffer(6)]],
    constant     uint&   start        [[buffer(7)]],
    constant     uint&   window       [[buffer(8)]],
    constant     float&  scale        [[buffer(9)]],
    constant     uint&   ringCapacity [[buffer(10)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    if (sg >= queriesPerKV) { return; }
    const uint kvHead = tg.x;
    const uint t = tg.y;
    const uint numQHeads = numKVHeads * queriesPerKV;
    const uint qHead = kvHead * queriesPerKV + sg;
    const uint seqLen = start + t + 1u;
    const uint kvStart = window > 0u && seqLen > window ? seqLen - window : 0u;
    const uint d0 = lane * 4u;
    const uint row = (t * numQHeads + qHead) * kFlintAttnDim + d0;
    const float4 qv = float4(*(device const half4*)(q + row));
    const uint kvStride = numKVHeads * kFlintAttnDim;

    float m = -INFINITY;
    float l = 0.0f;
    float4 acc = float4(0.0f);
    for (uint pos = kvStart; pos < seqLen; ++pos) {
        const uint slot = ringCapacity > 0u ? pos % ringCapacity : pos;
        const uint base = slot * kvStride + kvHead * kFlintAttnDim + d0;
        const float score = simd_sum(dot(qv, float4(*(device const half4*)(k + base)))) * scale;
        const float next = max(m, score);
        const float alpha = exp(m - next);
        const float p = exp(score - next);
        l = fma(l, alpha, p);
        acc = acc * alpha + p * float4(*(device const half4*)(v + base));
        m = next;
    }
    const float sink = flint_bf16(sinks[qHead]);
    const float top = max(m, sink);
    const float alpha = exp(m - top);
    const float denom = fma(l, alpha, exp(sink - top));
    *(device half4*)(out + row) = half4(acc * alpha / denom);
}
