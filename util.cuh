/*
 * @file: util.cuh
 * @author: zhangyn
 * @date: 2024-11-12
 * @brief: 本文件包含了一些实用的CUDA内核工具函数，包括内存对齐、数据重排、形状和步长描述等。
 */
#pragma once
#include <device_launch_parameters.h>
#include <limits>

// align to 128 boundary
#define ALIGN128(n)  (((n) + 127) / 128) * 128

/* Our own smem swizzling to get rid of smem bank conflict */
template<int B, int M, int S>
struct Swizzle {
    static constexpr int msk = ((1 << B) - 1) << (M + S);
    template<typename T>
    __device__ __host__ // we may would like to test this on host side...
    static auto apply(T const& offset) { return offset ^ ((offset & msk) >> S); }
};

/* elementwise prologue ops used in reduce */
// 1. Do Nothing
template<typename T>
struct NOP {
    __device__ __host__ static T op(T const& x) { return x; };
};
// 2. Square op
template<typename T>
struct SQR {
    __device__ __host__ static T op(T const& x) { return x * x; }
};

/* simple layout description for 3D tensor (left-major) */
// 3D shape
template<int ShapeX_, int ShapeY_, int ShapeZ_>
struct Shape3D {
    static constexpr int ShapeX = ShapeX_;
    static constexpr int ShapeY = ShapeY_;
    static constexpr int ShapeZ = ShapeZ_;

    __device__ __host__ static constexpr int size() { return ShapeX * ShapeY * ShapeZ; }
};
// 3D stride
template<int StrideX_, int StrideY_, int StrideZ_>
struct Stride3D {
    static constexpr int StrideX = StrideX_;
    static constexpr int StrideY = StrideY_;
    static constexpr int StrideZ = StrideZ_;
};
// 3D layout = (Shape, Stride)
template<typename Shape_, 
    typename Stride_ = Stride3D<1,Shape_::ShapeX,Shape_::ShapeX*Shape_::ShapeY>>
struct Layout3D {
    using Shape = Shape_;
    using Stride = Stride_;
    // cvt 1D coordinate to 3D coordinate
    __device__ __host__ static void coord1D_to_3D(const int coord1D,
        int& x, int& y, int& z) {
        x = coord1D % Shape::ShapeX;
        y = (coord1D / Shape::ShapeX) % Shape::ShapeY;
        z = coord1D / (Shape::ShapeX * Shape::ShapeY);
    }
    // map 3D coordinate to underlying 1D index
    __device__ __host__ static int map(const int x, const int y, const int z) {
        return x * Stride::StrideX + y * Stride::StrideY + z * Stride::StrideZ;
    }
    // map 1D coordinate to underlying 1D index
    __device__ __host__ static int map(const int coord1D) {
        return (coord1D % Shape::ShapeX) * Stride::StrideX +
            ((coord1D / Shape::ShapeX) % Shape::ShapeY) * Stride::StrideY +
            (coord1D / (Shape::ShapeX * Shape::ShapeY)) * Stride::StrideZ;
    }
    // get tensor size, i.e. numel
    __device__ __host__ static constexpr int size() { return Shape::size(); }
};

/* comparison operations */
template <typename T>
struct LT {
    // for any x of type T, cmp(dummy, x) returns FALSE
    static constexpr T dummy = std::numeric_limits<T>::has_infinity ?
        std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
    // cmp return TRUE if and only if a should goes before b
    __device__ __host__ static bool cmp(T const& a, T const& b) { return a < b; };
};

template <typename T>
struct GT {
    // for any x of type T, cmp(dummy, x) returns FALSE
    static constexpr T dummy = std::numeric_limits<T>::has_infinity ?
        -(std::numeric_limits<T>::infinity()) : std::numeric_limits<T>::min();
    // cmp return TRUE if and only if a should goes before b
    __device__ __host__ static bool cmp(T const& a, T const& b) { return a > b; };
};

