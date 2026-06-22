// Decode GEMV: y[N] = x[K] @ W^T, where W is [N, K] row-major (i.e. [out, in] —
// the GGUF-native linear layout). One warp computes one output row n: the warp
// streams W[n, :] (K contiguous bf16 → fully coalesced across lanes) and dots it
// with x (staged in shared memory). This replaces the M=1 tiled GEMM, which
// wasted ~16x of its threads on the empty batch dimension at decode time.
//
// Output is bf16 (projections) or fp32 (router / LM-head logits) via the OutT
// template. Portable CUDA — sm_89 .. sm_120/sm_121.

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

static constexpr int GEMV_WPB = 8;   // warps (output rows) per block

__device__ __forceinline__ void gemv_write(float* p, float v) { *p = v; }
__device__ __forceinline__ void gemv_write(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename OutT>
__global__ void gemv_kernel(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ W,
                            OutT* __restrict__ y, int N, int K) {
    extern __shared__ float s_x[];                 // K floats
    for (int i = threadIdx.x; i < K; i += blockDim.x) s_x[i] = __bfloat162float(x[i]);
    __syncthreads();

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n = blockIdx.x * GEMV_WPB + warp;
    if (n >= N) return;
    // 128-bit coalesced loads: each lane pulls a uint4 = 8 bf16 of the weight row.
    const uint4* row4 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    const int n4 = K / 8;
    float acc = 0.f;
    for (int i = lane; i < n4; i += 32) {
        uint4 v = row4[i];
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&v);
        const int base = i * 8;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(h2[j]);
            acc += f.x * s_x[base + 2*j] + f.y * s_x[base + 2*j + 1];
        }
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, m);
    if (lane == 0) gemv_write(y + n, acc);
}

template __global__ void gemv_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int);
template __global__ void gemv_kernel<float>(const __nv_bfloat16*, const __nv_bfloat16*, float*, int, int);

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/gemm.h"

void launch_gemv(const void* x, const void* W, void* y, int N, int K, cudaStream_t stream) {
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    gemv_kernel<__nv_bfloat16><<<grid, GEMV_WPB * 32, (size_t)K * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
}

void launch_gemv_f32(const void* x, const void* W, float* y, int N, int K, cudaStream_t stream) {
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    gemv_kernel<float><<<grid, GEMV_WPB * 32, (size_t)K * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W), y, N, K);
}
#endif

} // namespace kernels
} // namespace sparkinfer
