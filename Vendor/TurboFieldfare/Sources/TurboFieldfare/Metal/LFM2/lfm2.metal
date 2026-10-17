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
