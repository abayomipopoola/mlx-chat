#include <metal_stdlib>
using namespace metal;

// LFM2 MoE kernels. Compiled into a private library (module "lfm2") so other
// architectures never pay for them.
//
// 6-bit MLX affine weights pack values little-endian at bit 6*j, so one
// 64-value group occupies exactly 12 uint32 words. Scales and biases are BF16,
// one pair per 64-value group.

constant constexpr uint kLFMGroup = 64;
constant constexpr uint kLFMWordsPerGroup6 = 12;

static inline float lfm_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline uint lfm_q6(device const uint* words, uint j) {
    const uint bit = j * 6u;
    const uint word = bit >> 5;
    const uint shift = bit & 31u;
    uint value = words[word] >> shift;
    if (shift > 26u) {
        value |= words[word + 1] << (32u - shift);
    }
    return value & 63u;
}

// out[d] = dequant(W[token, d]) for one token row.
kernel void lfm2_embed_lookup_q6(
    device const uint*   W      [[buffer(0)]],
    device const ushort* S      [[buffer(1)]],
    device const ushort* B      [[buffer(2)]],
    device       half*   out    [[buffer(3)]],
    constant     uint&   token  [[buffer(4)]],
    constant     uint&   dim    [[buffer(5)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) { return; }
    const uint groups = dim / kLFMGroup;
    const uint rowWords = groups * kLFMWordsPerGroup6;
    device const uint* row = W + token * rowWords;
    const uint g = d / kLFMGroup;
    const float scale = lfm_bf16(S[token * groups + g]);
    const float bias = lfm_bf16(B[token * groups + g]);
    out[d] = half(float(lfm_q6(row, d)) * scale + bias);
}

// y[r] = sum_d dequant(W[r, d]) * x[d]. One SIMD group per row; each lane owns
// whole 64-value groups, so the affine terms fold into two dot products.
kernel void lfm2_q6_gemv(
    device const uint*   W      [[buffer(0)]],
    device const ushort* S      [[buffer(1)]],
    device const ushort* B      [[buffer(2)]],
    device const half*   x      [[buffer(3)]],
    device       half*   y      [[buffer(4)]],
    constant     uint&   rows   [[buffer(5)]],
    constant     uint&   dim    [[buffer(6)]],
    uint tg     [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]],
    uint sgs    [[simdgroups_per_threadgroup]]
) {
    const uint r = tg * sgs + sg;
    if (r >= rows) { return; }
    const uint groups = dim / kLFMGroup;
    device const uint* row = W + r * groups * kLFMWordsPerGroup6;
    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        device const uint* gw = row + g * kLFMWordsPerGroup6;
        device const half* gx = x + g * kLFMGroup;
        float qx = 0.0f;
        float sx = 0.0f;
        for (uint j = 0; j < kLFMGroup; ++j) {
            const float xv = float(gx[j]);
            qx = fma(float(lfm_q6(gw, j)), xv, qx);
            sx += xv;
        }
        const float scale = lfm_bf16(S[r * groups + g]);
        const float bias = lfm_bf16(B[r * groups + g]);
        acc = fma(scale, qx, fma(bias, sx, acc));
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[r] = half(acc); }
}

// One decode step of the gated short convolution.
//   [b, c, x] = bcx;  u = b * x
//   conv = sum_k w[d, k] * [state..., u][k]   (causal, kernel K)
//   y = c * conv; state shifts left and appends u.
// `state` holds the previous K-1 values of u, oldest first: [K-1, dim].
kernel void lfm2_short_conv_decode(
    device const half*   bcx    [[buffer(0)]],
    device const ushort* w      [[buffer(1)]],
    device       half*   state  [[buffer(2)]],
    device       half*   y      [[buffer(3)]],
    constant     uint&   dim    [[buffer(4)]],
    constant     uint&   kernelSize [[buffer(5)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) { return; }
    const float b = float(bcx[d]);
    const float c = float(bcx[dim + d]);
    const float xv = float(bcx[2u * dim + d]);
    const float u = b * xv;
    const uint taps = kernelSize - 1u;
    float acc = lfm_bf16(w[d * kernelSize + taps]) * u;
    for (uint k = 0; k < taps; ++k) {
        acc = fma(lfm_bf16(w[d * kernelSize + k]), float(state[k * dim + d]), acc);
    }
    for (uint k = 0; k + 1u < taps; ++k) {
        state[k * dim + d] = state[(k + 1u) * dim + d];
    }
    if (taps > 0u) {
        state[(taps - 1u) * dim + d] = half(u);
    }
    y[d] = half(c * acc);
}

// ---------------------------------------------------------------------------
// Prompt prefill: block versions of the decode kernels, one row per token. A
// row's result depends only on that token's inputs (and, for the short
// convolution, the tokens before it), never on how the prompt was chunked.

// Embedding rows for `tokens[t]`, grid (dim, T).
kernel void lfm2_embed_lookup_q6_block(
    device const uint*   W      [[buffer(0)]],
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
    const uint groups = dim / kLFMGroup;
    device const uint* row = W + token * groups * kLFMWordsPerGroup6;
    const uint g = d / kLFMGroup;
    const float scale = lfm_bf16(S[token * groups + g]);
    const float bias = lfm_bf16(B[token * groups + g]);
    out[t * dim + d] = half(float(lfm_q6(row, d)) * scale + bias);
}

// `lfm2_short_conv_decode` over T tokens in order, one thread per channel.
// Before token t, when `snapshotSlot[t] >= 0`, the state is copied to
// `snapshots + snapshotSlot[t] * snapshotStride` (the decode runner's
// snapshot before that position).
kernel void lfm2_short_conv_block(
    device const half*   bcx            [[buffer(0)]],
    device const ushort* w              [[buffer(1)]],
    device       half*   state          [[buffer(2)]],
    device       half*   y              [[buffer(3)]],
    constant     uint&   dim            [[buffer(4)]],
    constant     uint&   kernelSize     [[buffer(5)]],
    constant     uint&   count          [[buffer(6)]],
    device const int*    snapshotSlot   [[buffer(7)]],
    device       half*   snapshots      [[buffer(8)]],
    constant     uint&   snapshotStride [[buffer(9)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) { return; }
    const uint taps = kernelSize - 1u;
    for (uint t = 0; t < count; ++t) {
        const int slot = snapshotSlot[t];
        if (slot >= 0) {
            for (uint k = 0; k < taps; ++k) {
                snapshots[uint(slot) * snapshotStride + k * dim + d] = state[k * dim + d];
            }
        }
        device const half* row = bcx + t * 3u * dim;
        const float b = float(row[d]);
        const float c = float(row[dim + d]);
        const float xv = float(row[2u * dim + d]);
        const float u = b * xv;
        float acc = lfm_bf16(w[d * kernelSize + taps]) * u;
        for (uint k = 0; k < taps; ++k) {
            acc = fma(lfm_bf16(w[d * kernelSize + k]), float(state[k * dim + d]), acc);
        }
        for (uint k = 0; k + 1u < taps; ++k) {
            state[k * dim + d] = state[(k + 1u) * dim + d];
        }
        if (taps > 0u) {
            state[(taps - 1u) * dim + d] = half(u);
        }
        y[t * dim + d] = half(c * acc);
    }
}

// Causal prefill attention for T tokens at positions `start + t`, head dim 64,
// no sink: one threadgroup per (KV head, token), one SIMD group per query
// head, each lane owning two dims, an online softmax over the token's range.
kernel void lfm2_attention_prefill(
    device const half*  q            [[buffer(0)]],
    device const half*  k            [[buffer(1)]],
    device const half*  v            [[buffer(2)]],
    device       half*  out          [[buffer(3)]],
    constant     uint&  numKVHeads   [[buffer(4)]],
    constant     uint&  queriesPerKV [[buffer(5)]],
    constant     uint&  start        [[buffer(6)]],
    constant     float& scale        [[buffer(7)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint dim = 64u;
    if (sg >= queriesPerKV) { return; }
    const uint kvHead = tg.x;
    const uint t = tg.y;
    const uint numQHeads = numKVHeads * queriesPerKV;
    const uint qHead = kvHead * queriesPerKV + sg;
    const uint seqLen = start + t + 1u;
    const uint d0 = lane * 2u;
    const uint row = (t * numQHeads + qHead) * dim + d0;
    const float2 qv = float2(*(device const half2*)(q + row));
    const uint kvStride = numKVHeads * dim;
    float m = -INFINITY;
    float l = 0.0f;
    float2 acc = float2(0.0f);
    for (uint pos = 0; pos < seqLen; ++pos) {
        const uint base = pos * kvStride + kvHead * dim + d0;
        const float score = simd_sum(dot(qv, float2(*(device const half2*)(k + base)))) * scale;
        const float next = max(m, score);
        const float alpha = exp(m - next);
        const float p = exp(score - next);
        l = fma(l, alpha, p);
        acc = acc * alpha + p * float2(*(device const half2*)(v + base));
        m = next;
    }
    *(device half2*)(out + row) = half2(acc / l);
}

// Router scores for T rows from FP32 weights: grid (ceil(E / 8), T)
// threadgroups of 256 threads, one SIMD group per expert, sigmoid applied.
kernel void lfm2_router_f32_block(
    device const float* W           [[buffer(0)]],
    device const half*  x           [[buffer(1)]],
    device       float* out_scores  [[buffer(2)]],
    constant     uint&  num_experts [[buffer(3)]],
    constant     uint&  D           [[buffer(4)]],
    uint2 tg    [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint e = tg.x * 8u + sg_idx;
    const uint t = tg.y;
    if (e >= num_experts) { return; }
    device const float* w_row = W + e * D;
    device const half* x_row = x + t * D;
    float acc = 0.0f;
    for (uint i = lane; i < D; i += 32u) {
        acc = fma(w_row[i], float(x_row[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out_scores[t * num_experts + e] = 1.0f / (1.0f + exp(-acc));
    }
}

// act = silu(gate) * up over `count` elements.
kernel void lfm2_silu_mul(
    device const half* gate  [[buffer(0)]],
    device const half* up    [[buffer(1)]],
    device       half* act   [[buffer(2)]],
    constant     uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) { return; }
    const float g = float(gate[i]);
    act[i] = half(g / (1.0f + exp(-g)) * float(up[i]));
}
