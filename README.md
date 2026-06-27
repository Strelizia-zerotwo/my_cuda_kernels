# CUDA 算子学习仓库

这个仓库主要用于学习、复写和优化 CUDA 算子。



1. 每个 kernel 都能跑；
2. 每个 kernel 的原理都能讲清楚；
3. 每个优化版本都能和前一个版本对比；
4. 每个版本都尽量保留测试和性能记录；
5. 最终能够形成一个自己完全理解的 CUDA 算子学习仓库。

## 当前内容

目前主要包含 GEMM 相关代码。

目录：

```text
src/gemm/
├── my_hgemm.cu
├── hgemm_cublas.cu
├── hgemm_regdb.cu
└── gemm.cu
GEMM 部分说明

GEMM 的核心计算形式是：

C = A × B

其中：

A: M × K
B: K × N
C: M × N

当前 GEMM 代码主要参考了知乎文章：

https://zhuanlan.zhihu.com/p/555339335

在学习过程中，我根据这篇文章和相关资料复写、整理了多个版本。

文件说明
my_hgemm.cu

这是我根据知乎文章复写的 HGEMM 实现。

主要目的：

理解 half 精度 GEMM 的基本写法；
理解 block / thread / tile 的划分方式；
理解 shared memory 在 GEMM 优化中的作用；
作为后续优化版本的基础。
hgemm_cublas.cu

这是使用 cuBLAS 的 HGEMM 对照版本。

主要目的：

调用 cuBLAS 作为性能参考；
对比自己写的 kernel 和成熟库之间的性能差距；
检查自己实现的结果是否正确；
作为 benchmark baseline。
hgemm_regdb.cu

这是加入寄存器双重缓存思想的 HGEMM 优化版本。

主要目的：

理解 register blocking；
理解寄存器级别的数据复用；
理解 double buffering 如何隐藏访存开销；
对比基础版本和寄存器优化版本的性能差异。
gemm.cu

这是原始 GEMM 参考实现。

主要目的：

保留原始代码；
作为学习和复写时的参考；
方便对比自己修改后的版本。
Attention 部分

Attention / FlashAttention 目前还没有正式加入。

后续计划：

FlashAttention 

目录已经预留：

src/attention/
编译

可以使用脚本编译 GEMM 代码：

bash scripts/build_gemm.sh

编译后的可执行文件会放在：

build/
运行
bash scripts/run_gemm.sh

学习目标

这个仓库的最终目标不是简单调用库，而是逐步理解高性能 CUDA 算子的实现方法。

