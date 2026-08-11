.PHONY:all
all: reduction_v0 reduction_v1 reduction_v1_coarsening reduction_v2 reduction_v2_coarsening reduction_v3 reduction_v3_coarsening

reduction_v0: reduction_v0.cu
	nvcc -o $@ $^ -O0 -G -g
reduction_v1: reduction_v1.cu
	nvcc -o $@ $^ -O0 -G -g
reduction_v1_coarsening: reduction_v1_coarsening.cu
	nvcc -o $@ $^ -O0 -G -g
reduction_v2: reduction_v2.cu
	nvcc -o $@ $^ -O0 -G -g
reduction_v2_coarsening: reduction_v2_coarsening.cu
	nvcc -o $@ $^ -O0 -G -g
reduction_v3: reduction_v3.cu
	nvcc -o $@ $^ -O0 -G -g
reduction_v3_coarsening: reduction_v3_coarsening.cu
	nvcc -o $@ $^ -O0 -G -g

.PHONY:clean
clean:
	rm -rf reduction