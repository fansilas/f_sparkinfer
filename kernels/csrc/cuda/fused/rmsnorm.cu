// Fused RMSNorm (+ optional residual add). One block per row; block-reduces the
// sum of squares, then writes the normalized, weighted row. A small CODA-style
// epilogue building block kept on the portable CUDA path.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float rn_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

template <int ADD_RESIDUAL>
__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ residual,
                               const __nv_bfloat16* __restrict__ weight,
                               __nv_bfloat16* __restrict__ out,
                               int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];

    float ss = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]);
        if (ADD_RESIDUAL) v += __bfloat162float(residual[base + c]);
        ss += v * v;
    }
    ss = rn_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = rn_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]);
        if (ADD_RESIDUAL) v += __bfloat162float(residual[base + c]);
        out[base + c] = __float2bfloat16(v * inv_rms * __bfloat162float(weight[c]));
    }
}

template __global__ void rmsnorm_kernel<0>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, float);
template __global__ void rmsnorm_kernel<1>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, float);

// Fused residual + RMSNorm that ALSO emits the residual sum:
//   sum = x + residual;  norm = (sum / rms(sum)) * weight
// One kernel replaces a residual_add + a rmsnorm (and keeps `sum` for the next
// residual), cutting the per-layer norm/residual kernel count from 4 to 2.
__global__ void add_rmsnorm2_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ residual,
                                    const __nv_bfloat16* __restrict__ weight,
                                    __nv_bfloat16* __restrict__ out_sum,
                                    __nv_bfloat16* __restrict__ out_norm,
                                    int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    float ss = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]) + __bfloat162float(residual[base + c]);
        out_sum[base + c] = __float2bfloat16(v);
        ss += v * v;
    }
    ss = rn_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = rn_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        out_norm[base + c] = __float2bfloat16(__bfloat162float(out_sum[base + c]) * inv_rms * __bfloat162float(weight[c]));
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"

void launch_rmsnorm(const void* x, const void* weight, void* out,
                    int rows, int cols, float eps, cudaStream_t stream) {
    rmsnorm_kernel<0><<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), nullptr,
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out), rows, cols, eps);
}

void launch_add_rmsnorm(const void* x, const void* residual, const void* weight, void* out,
                        int rows, int cols, float eps, cudaStream_t stream) {
    rmsnorm_kernel<1><<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out), rows, cols, eps);
}

void launch_add_rmsnorm2(const void* x, const void* residual, const void* weight,
                         void* out_sum, void* out_norm, int rows, int cols, float eps,
                         cudaStream_t stream) {
    add_rmsnorm2_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm), rows, cols, eps);
}
#endif

} // namespace kernels
} // namespace sparkinfer
