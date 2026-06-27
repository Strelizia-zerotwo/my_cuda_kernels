# GEMM 学习笔记

## 1. GEMM 是什么

GEMM 的计算形式是：

```text
C = A × B

其中：

A: M × K
B: K × N
C: M × N

每个输出元素：

C[i][j] = sum(A[i][k] * B[k][j])
2. 为什么 GEMM 重要

GEMM 是深度学习和高性能计算中最核心的算子之一。

例如：

全连接层；
Transformer 中的 QKV 投影；
Attention 中的 QK^T 和 PV；
卷积转 GEMM；
大模型推理和训练。
3. 优化重点

GEMM 的优化重点一般包括：

减少 global memory 访问；
使用 shared memory 做 tile 复用；
使用 register 保存局部结果；
提高访存合并程度；
提高计算访存重叠；
使用 Tensor Core；
减少 bank conflict；
提高 occupancy 和 warp 执行效率。
4. 当前仓库中的版本
my_hgemm.cu

根据学习资料复写的 HGEMM 版本。

hgemm_cublas.cu

cuBLAS 对照版本，用于性能 baseline。

hgemm_regdb.cu

加入寄存器双重缓存思想的版本。

gemm.cu

原始参考实现。

5. 后续需要记录的内容

每次优化后建议记录：

GPU:
M/N/K:
block size:
tile size:
shared memory usage:
register usage:
运行时间:
TFLOPS:
相比上个版本提升:
瓶颈分析:

