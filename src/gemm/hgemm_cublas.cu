// hgemm_cublas.cu
// A100 上用 cuBLAS 做 FP16 GEMM (C = A * B) 的基准测试。
//
// 编译 (A100 是 sm_80):
//   nvcc -O3 -arch=sm_80 hgemm_cublas.cu -o hgemm_cublas -lcublas
//
// 运行 (默认 M=N=K=8192, 默认迭代 100 次):
//   ./hgemm_cublas
//   ./hgemm_cublas 8192 8192 8192 100      // 自定义 M N K iters
//
// 关键点:
//  - 用 cublasGemmEx 并显式开启 Tensor Core (CUBLAS_GEMM_DEFAULT_TENSOR_OP)。
//  - computeType = CUBLAS_COMPUTE_16F: FP16 累加, 这是冲 A100 312 TFLOPS 峰值的配置。
//    若需要 FP32 累加 (精度更高, 吞吐略低), 把它改成 CUBLAS_COMPUTE_32F。
//  - 有 warmup, 用 CUDA event 计时, 排除 cuBLAS 首次调用的 lazy init / 算法选择开销。

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = (call);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,       \
                    __LINE__, (int)st);                                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void fill_random(half* ptr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        ptr[i] = __float2half((rand() % 2000 - 1000) / 1000.0f);
    }
}

int main(int argc, char** argv) {
    int M = 8192, N = 8192, K = 8192;
    int repeats = 100;
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) repeats = atoi(argv[4]);

    printf("Matrix: M=%d N=%d K=%d  iters=%d\n", M, N, K, repeats);

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    half *h_A = (half*)malloc(sizeA * sizeof(half));
    half *h_B = (half*)malloc(sizeB * sizeof(half));

    srand(42);
    fill_random(h_A, sizeA);
    fill_random(h_B, sizeB);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB * sizeof(half), cudaMemcpyHostToDevice));

    half alpha = __float2half(1.0f);
    half beta  = __float2half(0.0f);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // 显式允许 Tensor Core
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // 列优先 GEMM: C[MxN] = A[MxK] * B[KxN], lda=M ldb=K ldc=M。
    // 用 GemmEx 显式指定数据类型 / 累加类型 / Tensor Core 算法。
    auto run_gemm = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, CUDA_R_16F, M,
            d_B, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_16F, M,
            CUBLAS_COMPUTE_16F,            // FP16 累加冲峰值; 要 FP32 累加改 CUBLAS_COMPUTE_32F
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // warmup
    for (int i = 0; i < 10; i++) run_gemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) run_gemm();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeats;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = flops / (ms * 1e-3) / 1e12;
    printf("Time:    %.3f ms (avg of %d)\n", ms, repeats);
    printf("TFLOPS:  %.2f\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    return 0;
}
