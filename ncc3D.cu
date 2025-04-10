
#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>

#include "scan3D_kernels.cuh"
#include "reduce3D_kernels.cuh"

// compute Z = conv(X, Y) using FFT
template <typename Layout>
int conv3D_using_FFT(float* dX3D, float* dY3D, float* dZ3D);

// let's test NCC3D computation!
int main(int argc, char* argv[]) {
    // test data type
    using T = float;
    // image volume shape
    constexpr int IM = 128;
    constexpr int IN = 128;
    constexpr int IK = 128;
    // template volume shape
    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int TK = 64;
    // output volume shape (must be padded to avoid frequency aliasing)
    constexpr int OM = IM - TM + 1;
    constexpr int ON = IN - TN + 1;
    constexpr int OK = IK - TK + 1;

    // generate image volume
    std::vector<T> image(256 * 256 * 256, T{});
    std::ifstream ins;
    ins.open("input_image-128x128x128.raw", std::ios::binary);
    for (int k = 0; k < IK; k++) {
        for (int j = 0; j < IN; j++) {
            ins.read((char*)(image.data() + j * 256 + k * 256 * 256), sizeof(T) * IM);
        }
    }
    ins.close();

    // generate template volume
    std::vector<T> template_img(256 * 256 * 256, T{});
    ins.open("template_img-64x64x64.raw", std::ios::binary);
    for (int k = 0; k < TK; k++) {
        for (int j = 0; j < TN; j++) {
            ins.read((char*)(template_img.data() + j * 256 + k * 256 * 256), sizeof(T) * TM);
        }
    }
    ins.close();

    // allocate global memory
    cudaError_t err = cudaSuccess;
    T* dimage, * dimg_sumtable, * energy_sumtable;
    cudaMalloc(&dimage, ALIGN128(sizeof(T) * image.size()));
    cudaMalloc(&dimg_sumtable, ALIGN128(sizeof(T) * IM * IN * IK));
    cudaMalloc(&energy_sumtable, ALIGN128(sizeof(T) * IM * IN * IK));
    cudaMemcpy(dimage, image.data(), sizeof(T) * image.size(), cudaMemcpyHostToDevice);

    T* d_template_img, * d_sum_workspace, * d_sum2_workspace;
    std::vector<T> h_partital_sum((TM * TN * TK + 16383) / 16384);
    std::vector<T> h_partital_sum2((TM * TN * TK + 16383) / 16384);
    cudaMalloc(&d_template_img, ALIGN128(sizeof(T) * 256 * 256 * 256));
    cudaMalloc(&d_sum_workspace, ALIGN128(sizeof(T) * ((TM * TN * TK + 16383) / 16384)));
    cudaMalloc(&d_sum2_workspace, ALIGN128(sizeof(T) * ((TM * TN * TK + 16383) / 16384)));
    cudaMemcpy(d_template_img, template_img.data(), sizeof(T) * template_img.size(), cudaMemcpyHostToDevice);

    // generate convolved matrix
    std::vector<T> conv(256 * 256 * 256, T{});
    T* dconv, * doutput;
    cudaMalloc(&dconv, ALIGN128(sizeof(T) * conv.size()));
    doutput = dconv;
    cudaMemcpy(dconv, conv.data(), sizeof(T) * conv.size(), cudaMemcpyHostToDevice);
    if (0 != conv3D_using_FFT<Layout3D<Shape3D<256, 256, 256>>>(dimage, d_template_img, dconv)) {
        return -1;
    }

    // create CUDA streams:
    // compute sumtable --------s1------->|normalization----default stream---->|fetch ncc result
    // compute sumtable2 ------------s2------------>|
    // compute reduce sum ---s3-->|fetch result --->|
    // compute reduce sum2 --s4-->|fetch result --->|
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStream_t s3, s4;
    cudaStreamCreate(&s3);
    cudaStreamCreate(&s4);

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg, s1);   // this is inaccurate...

    // 1. generate sumtables
    // compute image volume sumtable
    sumtable3D_stage1_kernel<T, 256, 256 * 256, IM, IM* IN> << <IK, dim3(128, 2), 0, s1 >> > (dimage, dimg_sumtable);
    sumtable3D_stage2_kernel<T, IM, IM* IN> << <IK, dim3(128, 2), 0, s1 >> > (dimg_sumtable);
    // compute energy image volume sumtable
    sumtable3D_stage1_kernel<T, 256, 256 * 256, IM, IM* IN, SQR<T>> << <IK, dim3(128, 2), 0, s2 >> > (dimage, energy_sumtable);
    sumtable3D_stage2_kernel<T, IM, IM* IN> << <IK, dim3(128, 2), 0, s2 >> > (energy_sumtable);

    // 2. compute template mean values
    reduce_sum_16384_kernel<T, Layout3D<Shape3D<TM, TN, TK>, Stride3D<1, 256, 256 * 256>>> << <(TM * TN * TK + 16383) / 16384, 1024, 0, s3 >> > (d_template_img, d_sum_workspace);
    reduce_sum_16384_kernel<T, Layout3D<Shape3D<TM, TN, TK>, Stride3D<1, 256, 256 * 256>>, SQR<T>> << <(TM * TN * TK + 16383) / 16384, 1024, 0, s4 >> > (d_template_img, d_sum2_workspace);
    err = cudaStreamSynchronize(s3);
    if (cudaSuccess != err) {
        printf("Fail to compute template sum, CUDA error %d!\n", err);
        return -1;
    }
    err = cudaStreamSynchronize(s4);
    if (cudaSuccess != err) {
        printf("Fail to compute template sum, CUDA error %d!\n", err);
        return -1;
    }
    cudaMemcpy(h_partital_sum.data(), d_sum_workspace, sizeof(T) * h_partital_sum.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partital_sum2.data(), d_sum2_workspace, sizeof(T) * h_partital_sum2.size(), cudaMemcpyDeviceToHost);
    T template_img_sum = std::accumulate(h_partital_sum.begin(), h_partital_sum.end(), T{});
    T template_img2_sum = std::accumulate(h_partital_sum2.begin(), h_partital_sum2.end(), T{});

    // 3. normalize cross-correlation
    float alpha = static_cast<float>(template_img2_sum) -
        static_cast<float>(template_img_sum * template_img_sum) / (TM * TN * TK);
    alpha = alpha <= 0.0f ? 0.0f : 1 / sqrtf(alpha);
    err = cudaStreamSynchronize(s1);
    if (cudaSuccess != err) {
        printf("Fail to compute sum table, CUDA error %d!\n", err);
        return -1;
    }
    err = cudaStreamSynchronize(s2);
    if (cudaSuccess != err) {
        printf("Fail to compute sum table, CUDA error %d!\n", err);
        return -1;
    }
    normalize3DCC_kernel<T,
        Layout3D<Shape3D<OM, ON, OK>, Stride3D<1, 256, 256 * 256>>,
        Layout3D<Shape3D<IM, IN, IK>>,
        Layout3D<Shape3D<OM, ON, OK>, Stride3D<1, 256, 256 * 256>>> << <(OM * ON * OK + 1024 - 1) / 1024, 1024 >> > (dconv, dimg_sumtable, energy_sumtable, template_img_sum, static_cast<T>(alpha), doutput);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, beg, end);
    printf("elapsed time = %.2fms.\n", elapsedTime);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);

    // get NCC result
    std::vector<T> output(OM * ON * OK);
    for (int k = 0; k < OK; k++) {
        for (int j = 0; j < ON; j++) {
            cudaMemcpy(output.data() + j * OM + k * OM * ON, doutput + j * 256 + k * 256 * 256,
                sizeof(T) * OM, cudaMemcpyDeviceToHost);
        }
    }
    auto cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("Fail to comput 3D NCC! CUDA error: %d\n", cuda_error);
        return -1;
    }

    // find the max val in 3D NCC map
    T* d_max_val;
    int* d_max_idx;
    cudaMalloc(&d_max_val, ALIGN128(sizeof(T) * OM * ON * OK));
    cudaMalloc(&d_max_idx, ALIGN128(sizeof(int) * OM * ON * OK));

    find_minmax_16384_kernel<T, Layout3D<Shape3D<OM, ON, OK>, Stride3D<1, 256, 256 * 256>>, GT<T>> << <(OM * ON * OK + 16383) / 16384, 1024 >> > (doutput, d_max_val, d_max_idx);

    std::vector<T> h_max_val(((OM * ON * OK + 16383) / 16384));
    std::vector<int> h_max_idx(((OM * ON * OK + 16383) / 16384));
    cudaMemcpy(h_max_val.data(), d_max_val, sizeof(T) * h_max_val.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_idx.data(), d_max_idx, sizeof(int) * h_max_idx.size(), cudaMemcpyDeviceToHost);

    cuda_error = cudaGetLastError();
    if (cudaSuccess != cuda_error) {
        printf("Fail to find max val, CUDA error %d!\n", cuda_error);
        return -2;
    }
    T max_val = GT<T>::dummy;
    int max_idx = 0;
    for (int i = 0; i < h_max_val.size(); i++) {
        if (GT<T>::cmp(h_max_val[i], max_val)) {
            max_val = h_max_val[i];
            max_idx = h_max_idx[i];
        }
    }
    printf("NCC max val = %f, coordinate = (%d, %d, %d)\n", max_val,
        max_idx % OM, (max_idx / OM) % ON, max_idx / (OM * ON));

    // destroy CUDA streams
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);
    cudaStreamDestroy(s4);

    // output
    std::ofstream outs("ncc-65x65x65.raw", std::ios::binary);
    outs.write((char*)output.data(), sizeof(T) * output.size());
    outs.close();

    // release
    cudaFree(dimage);
    cudaFree(dimg_sumtable);
    cudaFree(energy_sumtable);
    cudaFree(d_template_img);
    cudaFree(d_sum_workspace);
    cudaFree(d_sum2_workspace);
    cudaFree(dconv);
    cudaFree(d_max_val);
    cudaFree(d_max_idx);

    // over
    return 0;
}

namespace { // use this kernel only in current compilation-unit!
    __global__ void xcorr_kernel(cuComplex* X, cuComplex* Y,
        const int N, const float alpha, cuComplex* Z) {
        cuComplex x, y;
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (N > idx) {
            x = X[idx];
            y = Y[idx];

            x.x *= alpha;
            x.y *= alpha;

            Z[idx] = cuCmulf(x, cuConjf(y));
        }
        // over
        return;
    }
}

// use cuFFT to perform 3D CC computation
template <typename Layout>
int conv3D_using_FFT(float* dX3D, float* dY3D, float* dZ3D) {
    // we use 3D FFT plans
    cufftHandle plan3D_fft, plan3D_ifft;
    cufftResult cufft_err;

    cufft_err = cufftPlan3d(&plan3D_fft,
        Layout::Shape::ShapeZ, Layout::Shape::ShapeY, Layout::Shape::ShapeX, cufftType::CUFFT_R2C);
    if (CUFFT_SUCCESS != cufft_err) {
        printf("Fail to make CUFFT plan! CUFFT error: %d\n", cufft_err);
        return -1;
    }

    cufft_err = cufftPlan3d(&plan3D_ifft,
        Layout::Shape::ShapeZ, Layout::Shape::ShapeY, Layout::Shape::ShapeX, cufftType::CUFFT_C2R);
    if (CUFFT_SUCCESS != cufft_err) {
        printf("Fail to make CUFFT plan for IFFT! CUFFT error: %d\n", cufft_err);
        return -2;
    }

    cufftComplex* dFX3D, * dFY3D;
    const int fft_size = Layout::Shape::ShapeZ * Layout::Shape::ShapeY * (Layout::Shape::ShapeX / 2 + 1);
    cudaMalloc(&dFX3D, ALIGN128(sizeof(cufftComplex) * fft_size));
    cudaMalloc(&dFY3D, ALIGN128(sizeof(cufftComplex) * fft_size));
    // FFT
    cufft_err = cufftExecR2C(plan3D_fft, dX3D, dFX3D);
    if (CUFFT_SUCCESS != cufft_err) {
        printf("Fail to transform image! CUFFT error: %d\n", cufft_err);
        return -3;
    }
    cufft_err = cufftExecR2C(plan3D_fft, dY3D, dFY3D);
    if (CUFFT_SUCCESS != cufft_err) {
        printf("Fail to transform template! CUFFT error: %d\n", cufft_err);
        return -4;
    }
    // apply convolution theorem in frequency domain
    xcorr_kernel << <(fft_size + 511) / 512, 512 >> > (dFX3D, dFY3D,
        fft_size, 1.0f / static_cast<float>(Layout::size()), dFX3D);
    // IFFT to get result
    cufft_err = cufftExecC2R(plan3D_ifft, dFX3D, dZ3D);
    if (CUFFT_SUCCESS != cufft_err) {
        printf("Fail to compute IFFT! CUFFT error: %d\n", cufft_err);
        return -5;
    }

    cudaDeviceSynchronize();
    auto cuda_err = cudaGetLastError();
    if (cudaSuccess != cuda_err) {
        printf("Fail to compute CC! CUDA error %d!\n", cuda_err);
        return -6;
    }
    // release
    cudaFree(dFX3D);
    cudaFree(dFY3D);

    cufftDestroy(plan3D_fft);
    cufftDestroy(plan3D_ifft);
    // over
    return 0;
}
