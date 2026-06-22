#pragma once
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

enum class GemmLayout { ROW_MAJOR, COL_MAJOR };
enum class GemmPrecision { FP16, BF16, FP8_E4M3, INT8 };

struct GemmConfig {
    GemmPrecision precision = GemmPrecision::BF16;
    GemmLayout    layout_a  = GemmLayout::ROW_MAJOR;
    GemmLayout    layout_b  = GemmLayout::COL_MAJOR;
    bool          use_tensor_cores = true;
    int           split_k = 1;
};

// C = alpha * A @ B + beta * C
// A: [M, K], B: [K, N], C: [M, N]
void launch_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    const GemmConfig& cfg,
    cudaStream_t stream = nullptr
);

// Batched GEMM: C[i] = A[i] @ B[i]
void launch_batched_gemm(
    const void** A, const void** B, void** C,
    int batch, int M, int N, int K,
    float alpha, float beta,
    const GemmConfig& cfg,
    cudaStream_t stream = nullptr
);

// Linear with fp32 output: C[M,N] = A[M,K] @ B[K,N]  (A,B bf16; C fp32).
// Used for the LM head (hidden -> vocab logits).
void launch_linear_f32(
    const void* A, const void* B, float* C,
    int M, int N, int K, cudaStream_t stream = nullptr);

// Decode GEMV: y[N] = x[K] @ W^T, W is [N,K] row-major ([out,in], GGUF-native).
// x,W bf16. y is bf16 (launch_gemv) or fp32 (launch_gemv_f32). One warp per row.
void launch_gemv(const void* x, const void* W, void* y, int N, int K,
                 cudaStream_t stream = nullptr);
void launch_gemv_f32(const void* x, const void* W, float* y, int N, int K,
                     cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
