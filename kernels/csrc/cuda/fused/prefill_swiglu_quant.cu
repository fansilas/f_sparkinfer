// Fused SwiGLU + per-row int8 quantize for the long-context dense FFN down-projection input.
//
// The chunked int8 FFN computes gate/up GEMMs into ffg/ffu, then runs a standalone SwiGLU
// (silu(gate)*up -> ffg) and a standalone per-row int8 quantize (ffg -> A_i8, sx) before the down
// GEMM. Both are memory-bound passes over the ffn-wide (12288) intermediate; fusing them removes the
// ffg store (SwiGLU) and the ffg reload (quantize) -- ~2 x rows*ffn bf16 of DRAM traffic per chunk.
//
// One block per row: each thread strides the row computing h = silu(ffg)*ffu, tracking the row amax;
// a block reduction yields the per-row scale, then a second strided pass writes int8. Numerically
// identical to launch_prefill_swiglu followed by launch_prefill_quantize_rows_i8 (both round the
// SwiGLU result to bf16 first, so the int8 quant sees the same bf16 values).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "sparkinfer/kernels/prefill_fp8.h"

namespace sparkinfer { namespace kernels {

namespace {
__device__ __forceinline__ float sq_silu(float x) { return x / (1.f + __expf(-x)); }

__global__ void pf_swiglu_quant_i8_kernel(const __nv_bfloat16* __restrict__ gate,
                                          const __nv_bfloat16* __restrict__ up,
                                          signed char* __restrict__ q, float* __restrict__ scale,
                                          int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    // pass 1: compute h (bf16-rounded, matching the standalone SwiGLU) + row amax
    float amax = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float h = __bfloat162float(__float2bfloat16(sq_silu(__bfloat162float(gate[base + c]))
                                                           * __bfloat162float(up[base + c])));
        amax = fmaxf(amax, fabsf(h));
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
        if (threadIdx.x == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float d = (s_warp[0] == 0.f) ? 1.f : (s_warp[0] / 127.0f);
    if (threadIdx.x == 0) scale[row] = d;
    // pass 2: recompute h (cheap ALU) and store int8
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float h = __bfloat162float(__float2bfloat16(sq_silu(__bfloat162float(gate[base + c]))
                                                           * __bfloat162float(up[base + c])));
        q[base + c] = (signed char)((s_warp[0] == 0.f) ? 0 : (int)roundf(h / d));
    }
}
} // namespace

void launch_prefill_swiglu_quant_i8(const void* gate, const void* up, signed char* q, float* scale,
                                    int rows, int cols, cudaStream_t stream) {
    pf_swiglu_quant_i8_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        q, scale, rows, cols);
}

}} // namespace sparkinfer::kernels
