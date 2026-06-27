#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void myHGEMM(
    half *__restrict__ a,half *__restrict__ b, half *__restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x + blockIdx.z * gridDim.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;                             // block 内线程号 0~255
    int wid = tid / 32;                                // tid/32 = warp 编号 0~7
    if (bx >= (N + BN - 1) / BN || by >= (M + BM - 1) / BM) return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM *(BK + APAD);
    int s_a_size = BM * (BK + APAD);
    int s_b_size = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator,16,16,16,half> frag_c[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    int load_a_smem_m = (tid / 4) * 2;
    int load_a_smem_k = (tid % 4) * 8;
    int load_b_smem_k = (tid / 32) * 4;
    int load_b_smem_n = (tid % 32) * 8;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid / 2;

    {
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_0),"l"(&a[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_1),"l"(&a[load_a_gmem_addr + K]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_0),"l"(&b[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_1),"l"(&b[load_b_gmem_addr + N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_2),"l"(&b[load_b_gmem_addr + 2 * N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_3),"l"(&b[load_b_gmem_addr + 3 * N]));
        asm("cp.async.commit_group;\n"::);
        asm("cp.async.wait_group 0;\n"::);
        __syncthreads();
    }

    #pragma unroll 32
    for(int bk = 1; bk < K / BK; bk++)
    {
        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = smem_sel ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_0 + smem_sel_next * s_a_size * (int)sizeof(half)),"l"(&a[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_a_smem_addr_1 + smem_sel_next * s_a_size * (int)sizeof(half)),"l"(&a[load_a_gmem_addr + K]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_0 + smem_sel_next * s_b_size * (int)sizeof(half)),"l"(&b[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_1 + smem_sel_next * s_b_size * (int)sizeof(half)),"l"(&b[load_b_gmem_addr + N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_2 + smem_sel_next * s_b_size * (int)sizeof(half)),"l"(&b[load_b_gmem_addr + 2 * N]));
        asm("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(load_b_smem_addr_3 + smem_sel_next * s_b_size * (int)sizeof(half)),"l"(&b[load_b_gmem_addr + 3 * N]));

        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

        int smem_sel = ((K / BK) & 1) ^ 1;

        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_size + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_size +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_size + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            #pragma unroll
            for (int j = 0; j < 4; j++) 
            {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }
        
        int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
        int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        #pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            #pragma unroll
            for (int j = 0; j < 4; j++) 
            {
                wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
            }
        }

    }