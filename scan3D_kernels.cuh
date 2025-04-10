/*
 * @file: scan3D_kernels.cuh
 * @author: zhangyn
 * @date: 2024-11-14
 * @brief: device functions to implement 3D volume sumtable computation
 */
#pragma once
#include <device_launch_parameters.h>
#include "util.cuh"


// 2D version (8 warps per threadblock, M32N128 input)
// blockDim = x128y2

/* generate 2D sum table on tensor x with layout = (128, 32):(1, 128)
*  compute inclusive prefix sum along the 1st mode of x
*  computation is in-place, and the result is stored in smem slice
*  1. input/output tensor x layout = (128, 32):(1, 128), reside on smem
*  2. use a threadblock with (128, 2):(1, 128) thread-layout
*/
template<typename T, /* dtype */
    typename SMEM_SWIZZLE = Swizzle<5, 0, 7>, /* smem swizzle function */
    typename Prologue = NOP<T> /* prologue op */>
__device__ __forceinline__
void scan_X_on_128x32_tensor(T* x /* input/output tensor x */) {
    const int lane_id = threadIdx.x & (32 - 1); // in-warp lane ID
    for (int i = threadIdx.y; i < 32; i += 2) {
        T res = x[SMEM_SWIZZLE::apply(i * 128 + threadIdx.x)];
        // perform prologue op for each element
        res = Prologue::op(res);
        // initialize partital sum
        T sum = res;
        // butterfly algorithm to get partial results
        int half_group_size_power = 0;
        for (int group_size = 2; group_size <= 32; group_size <<= 1) {
            T temp = __shfl_xor_sync(uint32_t(-1), sum, group_size >> 1, group_size);
            sum += temp;
            res += ((lane_id >> (half_group_size_power++)) & 1) ? temp : T{};
        }

        // store partial sum to smem
        const int warp_id = threadIdx.x >> 5;
        if (31 == lane_id) {
            x[SMEM_SWIZZLE::apply(warp_id + i * 128)] = res;
        }
        __syncthreads();
        // merge partial results
        // (1) round-1
        if (warp_id & 1) {
            res += x[SMEM_SWIZZLE::apply(warp_id - 1 + i * 128)];
        }
        if (63 == threadIdx.x) {
            x[SMEM_SWIZZLE::apply(1 + i * 128)] = res;
        }
        __syncthreads();
        // (2) round-2
        if (warp_id > 1) {
            res += x[SMEM_SWIZZLE::apply(1 + i * 128)];
        }
        __syncthreads(); // this barrier is unnecessary if we afford additional smem to exchange partial sums

        // output (to smem slice)
        x[SMEM_SWIZZLE::apply(threadIdx.x + i * 128)] = res;
    }
    // over
    return;
}

/* compute inclusive prefix sum along the 2nd mode of tensor x on smem
*  smem tensor x layout = (128, 32):(1, 128)
*  computation in-place
*  use a threadblock with 256 threads
*/
template<typename T, /* dtype */
    typename SMEM_SWIZZLE = Swizzle<5, 0, 7>/* smem swizzle function */>
__device__ __forceinline__
void scan_Y_on_128x32_tensor(T* x /* input/output tensor x on smem */) {
    const int lane_id = (threadIdx.x + threadIdx.y * blockDim.x) & (32 - 1);
    const int warp_id = (threadIdx.x + threadIdx.y * blockDim.x) >> 5;
    for (int j = 0; j < 128; j += 8) {
        T res = x[SMEM_SWIZZLE::apply(j + warp_id + 128 * lane_id)];
        T sum = res;
        // butterfly algorithm
        int half_group_size_power = 0;
        for (int group_size = 2; group_size <= 32; group_size <<= 1) {
            T temp = __shfl_xor_sync(uint32_t(-1), sum, group_size >> 1, group_size);
            sum += temp;
            res += ((lane_id >> (half_group_size_power++)) & 1) ? temp : T{};
        }
        // output
        x[SMEM_SWIZZLE::apply(j + warp_id + 128 * lane_id)] = res;
    }
    // over
    return;
}

/* compute 2D sum table on tensor x with layout = (128, 128):(1, StrideIn)
*  the result tensor s layout = (128, 128):(1, StrideOut)
*  if x and s are the same, compute in-place
*  for each input element, perform a prologue op before computing sumtable
*  use a threadblock with (128, 2):(1, 128) thread-layout
*/
template<typename T, /* dtype */
    int StrideIn = 128,   /* stride of the x tensor 2nd mode */
    int StrideOut = StrideIn,   /* stride of the s tensor 2nd mode */
    typename Prologue = NOP<T> /* prologue op */>
__device__ __forceinline__
void sumtable_on_128x128_tensor(T const* x, /* input tensor x (global memory) */
    T* s /* output tensor s (global memory) */) {
    // smem swizzle needed to avoid bank conflict when accessing a (128, 32):(1, 128) smem tensor
    using smem_swizzle_t = typename Swizzle<5, 0, 7>;

    __shared__ T smem[32 * 128];    // to save smem, each time only a (128, 32) sub-tensor is processed
    T base{};   // base is needed to accumulate sum along Y direction (2nd mode of tensor x)
    // partition input tensor into four (128, 32) sub-tensors
    for (int iter = 0; iter < 4; iter++) {
        // G2S copy
        for (int i = threadIdx.y; i < 32; i++) {
            smem[smem_swizzle_t::apply(threadIdx.x + i * 128)] = 
                x[(iter * 32 + i) * StrideIn + threadIdx.x];
        }
        // scan along X direction
        scan_X_on_128x32_tensor<T, smem_swizzle_t>(smem);
        __syncthreads();

        // add base
        if (0 == threadIdx.y) {
            smem[smem_swizzle_t::apply(threadIdx.x)] += base;
        }
        __syncthreads();

        // scan along Y direction
        scan_Y_on_128x32_tensor<T, smem_swizzle_t>(smem);
        __syncthreads();

        // output sumtable of sub-tensor
        T* sptr = s + iter * (32 * StrideOut);
        for (int i = threadIdx.y; i < 32; i++) {
            s[threadIdx.x + (iter * 32 + i) * StrideOut] = 
                smem[smem_swizzle_t::apply(threadIdx.x + i * 128)];
        }

        // update base value
        if (0 == threadIdx.y && iter < 3) {
            base = smem[smem_swizzle_t::apply(threadIdx.x + 31 * 128)];
        }
        __syncthreads();
    }

    // over
    return;
}

/* compute 2D sumtables along the 1st and 2nd mode of 3D tensor x
*  layout of x = (128, 128, Nz):(1, StrideYIn, StrideZIn)
*  output tensor s layout = (128, 128, Nz):(1, StrideYOut, StrideZOut)
*  if x = s, compute in-place
*  use Nz threadblocks, each threadblock is (128, 2):(1, 128) thread-layout
*/
template<typename T,
    int StrideYIn = 128, int StrideZIn = 128 * StrideYIn,   /* stride of x tensor 2nd and 3rd mode */
    int StrideYOut = StrideYIn, int StrideZOut = 128 * StrideYOut,/* stride of s tensor 2nd and 3rd mode */
    typename Prologue = NOP<T>>
    __global__ void sumtable2D_kernel(T const* x, T* s) {
    sumtable_on_128x128_tensor<T, StrideYIn, StrideYOut, Prologue>(
        x + blockIdx.x * StrideZIn, s + blockIdx.x * StrideZOut);
    // over
    return;
}

/* compute inclusive prefix sum along the 2nd mode of tensor x
   layout of tensor x = (128, 128):(1, StrideIn)
   the result tensor s layout = (128, 128):(1, StrideOut)
   if x and s are the same, compute in-place
*/
template<typename T,    /* dtype */
    int StrideIn = 128, int StrideOut = StrideIn>
__device__ __forceinline__
void scan_Y_on_128x128_tensor(T* x, /* input tensor x*/ T* s /* output tensor s */) {
    using smem_swizzle_t = typename Swizzle<5, 0, 7>;
    // to save smem, each time only a (128, 32) sub-tensor is processed
    __shared__ T smem[32 * 128];
    T base{};   // base is needed to accumulate sum along Y direction (2nd mode of tensor x)
    for (int iter = 0; iter < 4; iter++) {
        // G2S copy
        for (int i = threadIdx.y; i < 32; i++) {
            smem[smem_swizzle_t::apply(threadIdx.x + i * 128)] =
                x[(iter * 32 + i) * StrideIn + threadIdx.x];
        }
        __syncthreads();

        // add base
        if (0 == threadIdx.y) {
            smem[smem_swizzle_t::apply(threadIdx.x)] += base;
        }
        __syncthreads();

        // scan along Y direction (compute on smem)
        scan_Y_on_128x32_tensor<T, smem_swizzle_t>(smem);
        __syncthreads();

        // output
        for (int i = threadIdx.y; i < 32; i++) {
            s[threadIdx.x + (iter * 32 + i) * StrideOut] =
                smem[smem_swizzle_t::apply(threadIdx.x + i * 128)];
        }
        // update base value
        if (0 == threadIdx.y && iter < 3) {
            base = smem[smem_swizzle_t::apply(threadIdx.x + 31 * 128)];
        }
        __syncthreads();
    }

    // over
    return;
}

// split computation of 3D sum table of (128, 128, 128) tensor x into 2 stages
// stage-1: compute 2D sum table of each XY-slice (across the 1st and 2nd mode of x)
// if x and s are the same tensor, compute in-place
// we need 128 threadblocks, each is (128, 32):(1, 128) thread-layout
template<typename T,
    int StrideYIn = 128, int StrideZIn = 128 * StrideYIn,   /* stride of x tensor 2nd and 3rd mode */
    int StrideYOut = StrideYIn, int StrideZOut = 128 * StrideYOut,/* stride of s tensor 2nd and 3rd mode */
    typename Prologue = NOP<T>>
    __global__ void sumtable3D_stage1_kernel(T const* x, /* input 3D tensor */ T* s /* output 3D tensor */) {
    sumtable_on_128x128_tensor<T, StrideYIn, StrideYOut, Prologue>(
        x + blockIdx.x * StrideZIn, s + blockIdx.x * StrideZOut);
    // over
    return;
}
// stage-2: compute 3D sum table by reducing sum along Z-axis (3rd mode of s)
// we need 128 threadblocks, each is (128, 32):(1, 128) thread-layout
template<typename T,
    int StrideY = 128, int StrideZ = 128 * StrideY>
__global__ void sumtable3D_stage2_kernel(T* s /* input/output 3D tensor */) {
    scan_Y_on_128x128_tensor<T, StrideZ, StrideZ>(
        s + blockIdx.x * StrideY, s + blockIdx.x * StrideY);
    // over
    return;
}

/* normalize the 3D cross-correlation (CC) to get 3D normalized cross-correlation (NCC)
*  use 3D sum table of original image volume and energy image volume to accelerate
*  computation of CC window average and variance
*/
template <typename T,
    typename InputLayout,   /* input layout */
    typename SumtableLayout,    /* sumtables layout */
    typename OutputLayout, /* output layout */
    typename TemplateShape = Shape3D<SumtableLayout::Shape::ShapeX - OutputLayout::Shape::ShapeX + 1,
    SumtableLayout::Shape::ShapeY - OutputLayout::Shape::ShapeY + 1,
    SumtableLayout::Shape::ShapeZ - OutputLayout::Shape::ShapeZ + 1>
>
__global__ void normalize3DCC_kernel(T* input, /* input 3D cross-correlation */
    T const* sumtable, /* input 3D sum table of original image */
    T const* sumtable2, /* input 3D sum table of original energy image */
    T const template_img_sum, /* input sum of template volume */
    T const alpha, /* input normalization factor from template volume */
    T* output   /* output 3D NCC map */) {
    if (blockIdx.x * blockDim.x + threadIdx.x < OutputLayout::size()) {
        // get 3D coordinate in output tensor
        int xid, yid, zid;
        OutputLayout::coord1D_to_3D(blockIdx.x * blockDim.x + threadIdx.x, xid, yid, zid);

        // compute denominator of NCC formula
        T f = sumtable[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid + TemplateShape::ShapeY - 1, zid + TemplateShape::ShapeZ - 1)] -
            (xid ? sumtable[SumtableLayout::map(xid - 1, yid + TemplateShape::ShapeY - 1, zid + TemplateShape::ShapeZ - 1)] : T{}) -
            (yid ? sumtable[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid - 1, zid + TemplateShape::ShapeZ - 1)] : T{}) -
            (zid ? sumtable[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid + TemplateShape::ShapeY - 1, zid - 1)] : T{}) +
            (xid && yid ? sumtable[SumtableLayout::map(xid - 1, yid - 1, zid + TemplateShape::ShapeZ - 1)] : T{}) +
            (xid && zid ? sumtable[SumtableLayout::map(xid - 1, yid + TemplateShape::ShapeY - 1, zid - 1)] : T{}) +
            (yid && zid ? sumtable[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid - 1, zid - 1)] : T{}) -
            (xid && yid && zid ? sumtable[SumtableLayout::map(xid - 1, yid - 1, zid - 1)] : T{});
        T f2 = sumtable2[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid + TemplateShape::ShapeY - 1, zid + TemplateShape::ShapeZ - 1)] -
            (xid ? sumtable2[SumtableLayout::map(xid - 1, yid + TemplateShape::ShapeY - 1, zid + TemplateShape::ShapeZ - 1)] : T{}) -
            (yid ? sumtable2[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid - 1, zid + TemplateShape::ShapeZ - 1)] : T{}) -
            (zid ? sumtable2[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid + TemplateShape::ShapeY - 1, zid - 1)] : T{}) +
            (xid && yid ? sumtable2[SumtableLayout::map(xid - 1, yid - 1, zid + TemplateShape::ShapeZ - 1)] : T{}) +
            (xid && zid ? sumtable2[SumtableLayout::map(xid - 1, yid + TemplateShape::ShapeY - 1, zid - 1)] : T{}) +
            (yid && zid ? sumtable2[SumtableLayout::map(xid + TemplateShape::ShapeX - 1, yid - 1, zid - 1)] : T{}) -
            (xid && yid && zid ? sumtable2[SumtableLayout::map(xid - 1, yid - 1, zid - 1)] : T{});

        // normalize
        float factor = static_cast<float>(f2 - (f * f) / TemplateShape::size());
        factor = factor <= 0.0f ? 0.0f : alpha * __frsqrt_rn(factor); // use intrinsic math function in CUDA
        float numerator = static_cast<float>(input[InputLayout::map(xid, yid, zid)]) -
            static_cast<float>(template_img_sum * f) / TemplateShape::size();
        // output
        output[OutputLayout::map(xid, yid, zid)] = static_cast<T>(factor * numerator);
    }
    // over
    return;
}
