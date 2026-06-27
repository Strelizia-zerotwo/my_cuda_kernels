NVCC ?= nvcc
ARCH ?= sm_80

SRC_DIR := src/gemm
BUILD_DIR := build
RESULT_DIR := results

NVCC_FLAGS := -O3 -arch=$(ARCH) --use_fast_math -lineinfo
CUBLAS_LIBS := -lcublas

.PHONY: all clean run gemm hgemm_cublas hgemm_regdb my_hgemm_obj ptxas dirs

all: dirs gemm hgemm_cublas hgemm_regdb my_hgemm_obj

dirs:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(RESULT_DIR)

gemm: dirs
	$(NVCC) $(NVCC_FLAGS) $(SRC_DIR)/gemm.cu -o $(BUILD_DIR)/gemm

hgemm_cublas: dirs
	$(NVCC) $(NVCC_FLAGS) $(SRC_DIR)/hgemm_cublas.cu -o $(BUILD_DIR)/hgemm_cublas $(CUBLAS_LIBS)

hgemm_regdb: dirs
	$(NVCC) $(NVCC_FLAGS) $(SRC_DIR)/hgemm_regdb.cu -o $(BUILD_DIR)/hgemm_regdb $(CUBLAS_LIBS)

my_hgemm_obj: dirs
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_DIR)/my_hgemm.cu -o $(BUILD_DIR)/my_hgemm.o


ptxas:
	$(NVCC) $(NVCC_FLAGS) --ptxas-options=-v $(SRC_DIR)/hgemm_regdb.cu -o $(BUILD_DIR)/hgemm_regdb $(CUBLAS_LIBS)

clean:
	rm -rf $(BUILD_DIR)
