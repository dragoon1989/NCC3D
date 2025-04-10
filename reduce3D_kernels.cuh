/*
 * @file: reduce3D_kernels.cuh
 * @author: zhangyn
 * @date: 2024-11-14
 * @brief: device functions to implement 3D volume reduction
 */
#pragma once
#include <device_launch_parameters.h>
#include "util.cuh"


/* compute reduce sum of 2048 x elements 
* assert : blockDim = (1024, 1, 1)
*/
template<typename T, /* dtype */
	typename Prologue = NOP<T>	/* prologue op */>
__device__ __forceinline__ 
T reduce_sum_2048(T const* x, T smem[]) {
	// round-1
	// each thread loads 2 elements with a 1024 interval
	T x0 = x[threadIdx.x];
	T x1 = x[threadIdx.x + 1024];
	// prologue
	x0 = Prologue::op(x0);
	x1 = Prologue::op(x1);
	// use smem to share between threads
	smem[threadIdx.x] = x0 + x1;
	__syncthreads();

	// round-2~5
#pragma unroll 4
	for (int n = 512; n > 32; n >>= 1) {
		if (n > threadIdx.x) {
			x0 = smem[threadIdx.x];
			x1 = smem[threadIdx.x + n];
			smem[threadIdx.x] = x0 + x1;
		}
		__syncthreads();
	}

	// round-6 (single warp)
	if (threadIdx.x < 32) {
		x0 = smem[threadIdx.x];
		x1 = smem[threadIdx.x + 32];
		x0 += x1;

#pragma unroll
		for (int n = 16; n > 0; n >>= 1) {
			x0 += __shfl_xor_sync(unsigned(-1), x0, n, warpSize);
		}
	}
	// over
	return x0;
}

/* compute reduce sum input 3D tensor x 
* assert : blockDim = (1024, 1, 1)
* each threadblock computes sum of successive 16384 elements
*/
template<typename T, typename Layout, typename Prologue = NOP<T>>
__global__ void reduce_sum_16384_kernel(T const* x, /* input array */ T* sum /* output sum */) {
	const int xid = threadIdx.x + blockIdx.x * 16384;
	// accumulator register
	T res{};
	// to save smem, only 2048 elements are processed in each iteration (8 iterations in total)
	__shared__ T smem[1024];
	// main loop
	for (int i = 0; i < 8; i++) {
		// reduce sum over 2048 elements
		res += reduce_sum_2048<T, Prologue>(x + i * 2048, smem);
	}
	// output partital result over 16384 elements
	if (0 == threadIdx.x) {
		sum[blockIdx.x] = res;
	}
	// over
	return;
}

/* find the min/max element and its index in 2048 elements
*  if multiple minimum/maximum exist, return any one of them
*/
template<typename T, /* dtype */
	typename CMP	/* comparison op */>
__device__ __forceinline__
void find_minmax_2048(T const* x/* input x */, 
	const int offset /* offset of 2048 elements to be processed */, 
	T& minmax_val, int& minmax_idx, /* output */
	T smem_val[], int smem_idx[] /* smem buffers */) {
	// round-1
	int idx0 = offset + threadIdx.x;
	int idx1 = offset + threadIdx.x + 1024;
	T x0 = x[idx0];
	T x1 = x[idx1];
	if (CMP::cmp(x0, x1)) {
		smem_val[threadIdx.x] = x0;
		smem_idx[threadIdx.x] = idx0;
	} else {
		smem_val[threadIdx.x] = x1;
		smem_idx[threadIdx.x] = idx1;
	}
	__syncthreads();

	// round-2~5
#pragma unroll 4
	for (int n = 512; n > 32; n >>= 1) {
		if (n > threadIdx.x) {
			x0 = smem_val[threadIdx.x];
			x1 = smem_val[threadIdx.x + n];

			if (CMP::cmp(x1, x0)) {
				smem_val[threadIdx.x] = x1;
				smem_idx[threadIdx.x] = smem_idx[threadIdx.x + n];
			}
		}
		__syncthreads();
	}

	// round-6 (single warp)
	if (threadIdx.x < 32) {
		x0 = smem_val[threadIdx.x];
		idx0 = smem_idx[threadIdx.x];
		x1 = smem_val[threadIdx.x + 32];

		if (CMP::cmp(x1, x0)) {
			x0 = x1;
			idx0 = smem_idx[threadIdx.x + 32];
		}

#pragma unroll 5
		for (int n = 16; n > 0; n >>= 1) {
			x1 = __shfl_xor_sync(unsigned(-1), x0, n, warpSize);
			idx1 = __shfl_xor_sync(unsigned(-1), idx0, n, warpSize);
			if (CMP::cmp(x1, x0)) {
				x0 = x1;
				idx0 = idx1;
			}
		}
		// output result
		minmax_idx = idx0;
		minmax_val = x0;
	}
	// over
	return;
}


template <typename T, typename Layout, typename CMP>
__global__ void find_minmax_16384_kernel(T const* x, /* input array */
	T* val, /* output min/max element val */
	int* idx	/* output min/max element index */) {
	const int xid = threadIdx.x + blockIdx.x * 16384;
	T res = CMP::dummy;
	int res_idx = 0;
	// to save smem, only 2048 elements are processed in each iteration (8 iterations in total)
	__shared__ T smem_val[1024];
	__shared__ int smem_idx[1024];
	// main loop
	for (int i = 0; i < 8; i++) {
		T minmax_val;
		int minmax_idx;
		// find minmax of 2048 elements
		find_minmax_2048<T, CMP>(x, i * 2048, minmax_val, minmax_idx, smem_val, smem_idx);
		// only the 1st thread holds correct result!
		if (CMP::cmp(minmax_val, res)) {
			res = minmax_val;
			res_idx = minmax_idx;
		}
	}
	// output partital result
	if (0 == threadIdx.x) {
		val[blockIdx.x] = res;
		idx[blockIdx.x] = res_idx;
	}
	// over
	return;
}

