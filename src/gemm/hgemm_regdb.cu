// hgemm_regdb.cu
// 对比两个手写 WMMA HGEMM kernel:
//   V5 : 共享内存双缓冲 (寄存器单缓冲)
//   V6 : 共享内存双缓冲 + 寄存器双缓冲  <-- 本次新增, 用来验证寄存器双缓冲的影响
// 两者都用 cuBLAS(FP16累加) 做正确性校验, 并各自计时报 TFLOPS。
//
// 编译 (A100 = sm_80):
//   nvcc -O3 -arch=sm_80 hgemm_regdb.cu -o hgemm_regdb -lcublas
//
// 运行:
//   ./hgemm_regdb                      // 默认 4096^3, 50 次 (正确性 + 性能)
//   ./hgemm_regdb 8192 8192 8192 100   // 冲峰值时用这个
//
// 看寄存器用量 (关键, 判断是否 spill):
//   nvcc -O3 -arch=sm_80 --ptxas-options=-v hgemm_regdb.cu -o hgemm_regdb -lcublas
//   编译输出里找 "registers" 和 "spill stores/loads"。spill != 0 就是溢出了。

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err)); exit(1);                         \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = (call);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,    \
                    (int)st); exit(1);                                         \
        }                                                                      \
    } while (0)

// ===================== V5: 共享内存双缓冲, 寄存器单缓冲 =====================
__global__ void myHGEMMAlignedV5(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128, BN = 256, BK = 32;
    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    if (bx >= N / BN || by >= M / BM) return;

    const int APAD = 8, BPAD = 8;
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(frag_c[i][j], 0.0);

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * (int)sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * (int)sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * (int)sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * (int)sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * (int)sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * (int)sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    { // prologue: tile0 -> buf0
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_0),"l"(&a[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_1),"l"(&a[load_a_gmem_addr+K]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_0),"l"(&b[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_1),"l"(&b[load_b_gmem_addr+N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_2),"l"(&b[load_b_gmem_addr+2*N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_3),"l"(&b[load_b_gmem_addr+3*N]));
        asm("cp.async.commit_group;\n"::);
        asm("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {
        int smem_sel      = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_0 + smem_sel_next*s_a_db_offset*(int)sizeof(half)),"l"(&a[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_1 + smem_sel_next*s_a_db_offset*(int)sizeof(half)),"l"(&a[load_a_gmem_addr+K]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_0 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_1 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr+N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_2 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr+2*N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_3 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr+3*N]));

        #pragma unroll
        for (int k = 0; k < 2; k++) {
            #pragma unroll
            for (int m = 0; m < 4; m++)
                wmma::load_matrix_sync(frag_a[k][m], &s_a[smem_sel*s_a_db_offset + (comp_c_frag_m*64 + m*16)*(BK+APAD) + k*16], BK+APAD);
            #pragma unroll
            for (int n = 0; n < 4; n++)
                wmma::load_matrix_sync(frag_b[k][n], &s_b[smem_sel*s_b_db_offset + k*16*(BN+BPAD) + comp_c_frag_n*64 + n*16], BN+BPAD);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        asm("cp.async.commit_group;\n"::);
        asm("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        #pragma unroll
        for (int m = 0; m < 4; m++)
            wmma::load_matrix_sync(frag_a[k][m], &s_a[smem_sel*s_a_db_offset + (comp_c_frag_m*64 + m*16)*(BK+APAD) + k*16], BK+APAD);
        #pragma unroll
        for (int n = 0; n < 4; n++)
            wmma::load_matrix_sync(frag_b[k][n], &s_b[smem_sel*s_b_db_offset + k*16*(BN+BPAD) + comp_c_frag_n*64 + n*16], BN+BPAD);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i*16*N + j*16], frag_c[i][j], N, wmma::mem_row_major);
}

// ============ V6: 共享内存双缓冲 + 寄存器双缓冲 ============
// 区别: frag_a / frag_b 各开两份 [2][...], 用 reg_cur / reg_nxt 乒乓。
// 当前 tile 的 mma 用 frag[reg_cur], 同时把刚预取好的下一 tile 的
// 数据 load 进 frag[reg_nxt], 让 SMEM->REG 的 load 与 mma 重叠。
__global__ void myHGEMMAlignedV6(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128, BN = 256, BK = 32;
    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    if (bx >= N / BN || by >= M / BM) return;

    const int APAD = 8, BPAD = 8;
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // 寄存器双缓冲: 第一维 [2] = reg buffer, 第二维 [2] = K 子步, 第三维 [4] = m/n
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(frag_c[i][j], 0.0);

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * (int)sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * (int)sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * (int)sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * (int)sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * (int)sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * (int)sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    // 把 "从 smem sel 缓冲读 16 个 fragment 进 reg buffer R" 封成宏
    #define LOAD_FRAGS(R, SEL)                                                                                  \
        _Pragma("unroll")                                                                                       \
        for (int k = 0; k < 2; k++) {                                                                           \
            _Pragma("unroll")                                                                                   \
            for (int m = 0; m < 4; m++)                                                                         \
                wmma::load_matrix_sync(frag_a[R][k][m], &s_a[(SEL)*s_a_db_offset + (comp_c_frag_m*64 + m*16)*(BK+APAD) + k*16], BK+APAD); \
            _Pragma("unroll")                                                                                   \
            for (int n = 0; n < 4; n++)                                                                         \
                wmma::load_matrix_sync(frag_b[R][k][n], &s_b[(SEL)*s_b_db_offset + k*16*(BN+BPAD) + comp_c_frag_n*64 + n*16], BN+BPAD); \
        }
    #define DO_MMA(R)                                                                  \
        _Pragma("unroll")                                                              \
        for (int i = 0; i < 4; i++)                                                    \
            _Pragma("unroll")                                                          \
            for (int j = 0; j < 4; j++) {                                              \
                wmma::mma_sync(frag_c[i][j], frag_a[R][0][i], frag_b[R][0][j], frag_c[i][j]); \
                wmma::mma_sync(frag_c[i][j], frag_a[R][1][i], frag_b[R][1][j], frag_c[i][j]); \
            }

    { // prologue: tile0 -> smem buf0
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_0),"l"(&a[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_1),"l"(&a[load_a_gmem_addr+K]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_0),"l"(&b[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_1),"l"(&b[load_b_gmem_addr+N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_2),"l"(&b[load_b_gmem_addr+2*N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_3),"l"(&b[load_b_gmem_addr+3*N]));
        asm("cp.async.commit_group;\n"::);
        asm("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }

    LOAD_FRAGS(0, 0);   // tile0 进 reg buffer 0
    int reg_cur = 0;

    #pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {
        int smem_sel_next = ((bk - 1) & 1) ^ 1;   // tile_bk 预取去的 smem 缓冲
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // ① 异步预取 tile_bk 进 smem 另一块缓冲 (不阻塞)
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_0 + smem_sel_next*s_a_db_offset*(int)sizeof(half)),"l"(&a[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_1 + smem_sel_next*s_a_db_offset*(int)sizeof(half)),"l"(&a[load_a_gmem_addr+K]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_0 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_1 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr+N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_2 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr+2*N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_3 + smem_sel_next*s_b_db_offset*(int)sizeof(half)),"l"(&b[load_b_gmem_addr+3*N]));

        // ② 用已就绪的 reg_cur 计算当前 tile (tile_{bk-1})
        DO_MMA(reg_cur);

        // ③ 等预取完成
        asm("cp.async.commit_group;\n"::);
        asm("cp.async.wait_group 0;\n"::);
        __syncthreads();

        // ④ 把刚就绪的 tile_bk 读进另一个 reg buffer (与下一轮 mma 错开)
        int reg_nxt = reg_cur ^ 1;
        LOAD_FRAGS(reg_nxt, smem_sel_next);
        reg_cur = reg_nxt;
    }

    // epilogue: 算最后一块 tile (已在 reg_cur)
    DO_MMA(reg_cur);

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i*16*N + j*16], frag_c[i][j], N, wmma::mem_row_major);

    #undef LOAD_FRAGS
    #undef DO_MMA
}

// ===================== host 测试框架 =====================
static double run_and_time(void(*kernel)(half*,half*,half*,int,int,int),
                           const char *name, half *dA, half *dB, half *dC,
                           int M, int N, int K, size_t smem_bytes, int iters) {
    const int BM = 128, BN = 256;
    dim3 block(256);
    dim3 grid(N / BN, M / BM, 1);
    CUDA_CHECK(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    for (int i = 0; i < 5; i++)             // warmup
        kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; i++)
        kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= iters;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("%-6s  Time: %7.3f ms   TFLOPS: %7.2f\n", name, ms, tflops);
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    return tflops;
}

static void check_vs_cublas(half *dRef, half *dTest, int M, int N, const char *name) {
    size_t n = (size_t)M * N;
    half *hRef = (half*)malloc(n*sizeof(half));
    half *hTest = (half*)malloc(n*sizeof(half));
    CUDA_CHECK(cudaMemcpy(hRef,  dRef,  n*sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hTest, dTest, n*sizeof(half), cudaMemcpyDeviceToHost));
    double max_abs = 0, sum_rel = 0; int cnt = 0;
    for (size_t i = 0; i < n; i++) {
        double r = __half2float(hRef[i]), t = __half2float(hTest[i]);
        double d = fabs(r - t);
        if (d > max_abs) max_abs = d;
        if (fabs(r) > 1e-3) { sum_rel += d / fabs(r); cnt++; }
    }
    double mean_rel = cnt ? sum_rel / cnt : 0;
    printf("%-6s  vs cuBLAS:  max_abs=%.4f  mean_rel=%.4f%%  %s\n",
           name, max_abs, mean_rel*100,
           (mean_rel < 0.02) ? "OK" : "CHECK (fp16累加误差大可放宽)");
    free(hRef); free(hTest);
}

int main(int argc, char **argv) {
    int M = 4096, N = 4096, K = 4096, iters = 50;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5) iters = atoi(argv[4]);
    printf("Matrix: M=%d N=%d K=%d  iters=%d\n", M, N, K, iters);
    if (M % 128 || N % 256 || K % 32) {
        printf("要求 M%%128==0, N%%256==0, K%%32==0\n"); return 1;
    }

    size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
    half *hA=(half*)malloc(nA*sizeof(half)), *hB=(half*)malloc(nB*sizeof(half));
    srand(42);
    for (size_t i=0;i<nA;i++) hA[i]=__float2half((rand()%200-100)/100.0f*0.1f);
    for (size_t i=0;i<nB;i++) hB[i]=__float2half((rand()%200-100)/100.0f*0.1f);

    half *dA,*dB,*dC,*dRef;
    CUDA_CHECK(cudaMalloc(&dA,nA*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB,nB*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC,nC*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dRef,nC*sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dA,hA,nA*sizeof(half),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,nB*sizeof(half),cudaMemcpyHostToDevice));

    // cuBLAS 参考 (FP16 累加): 行优先 C=A*B  <=>  gemm(N,M,K, B, A)
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    half alpha=__float2half(1.f), beta=__float2half(0.f);
    auto cublas_ref = [&](half *out){
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K, &beta,
            out, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    for (int i=0;i<5;i++) cublas_ref(dRef);
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuBLAS 计时
    { cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s));CUDA_CHECK(cudaEventCreate(&e));
      CUDA_CHECK(cudaEventRecord(s));
      for(int i=0;i<iters;i++) cublas_ref(dRef);
      CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
      float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e)); ms/=iters;
      printf("%-6s  Time: %7.3f ms   TFLOPS: %7.2f\n","cuBLAS",ms,
             2.0*M*N*K/(ms*1e-3)/1e12);
      CUDA_CHECK(cudaEventDestroy(s));CUDA_CHECK(cudaEventDestroy(e)); }

    const int BM=128,BN=256,BK=32,APAD=8,BPAD=8;
    size_t smem_bytes = (2*BM*(BK+APAD) + 2*BK*(BN+BPAD))*sizeof(half);
    printf("dynamic smem = %zu bytes\n", smem_bytes);

    run_and_time(myHGEMMAlignedV5, "V5", dA,dB,dC, M,N,K, smem_bytes, iters);
    check_vs_cublas(dRef, dC, M, N, "V5");

    CUDA_CHECK(cudaMemset(dC, 0, nC*sizeof(half)));
    run_and_time(myHGEMMAlignedV6, "V6", dA,dB,dC, M,N,K, smem_bytes, iters);
    check_vs_cublas(dRef, dC, M, N, "V6");

    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dRef);
    free(hA);free(hB);
    return 0;
}
