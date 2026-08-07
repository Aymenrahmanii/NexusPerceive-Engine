#include "nexus/core/cuda_preprocess.cu.h"
#include <cuda_fp16.h>
#include <stdio.h>

namespace nexus {
namespace cuda {

// FP32 Fused Grid Preprocessor Kernel
__global__ void preprocess_fused_fp32_kernel(
    const uint8_t* __restrict__ d_src_bgr,
    int src_width,
    int src_height,
    int src_pitch,
    float* __restrict__ d_dst_nchw,
    int dst_width,
    int dst_height,
    float scale,
    int pad_w,
    int pad_h,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_width || dy >= dst_height) return;

    float sx = (dx - pad_w) / scale;
    float sy = (dy - pad_h) / scale;

    int src_x = __float2int_rd(sx);
    int src_y = __float2int_rd(sy);

    float b = 114.0f, g = 114.0f, r = 114.0f;

    if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
        const uint8_t* pixel_ptr = d_src_bgr + src_y * src_pitch + src_x * 3;
        b = static_cast<float>(pixel_ptr[0]);
        g = static_cast<float>(pixel_ptr[1]);
        r = static_cast<float>(pixel_ptr[2]);
    }

    float norm_r = ((r / 255.0f) - mean0) / std0;
    float norm_g = ((g / 255.0f) - mean1) / std1;
    float norm_b = ((b / 255.0f) - mean2) / std2;

    int plane_stride = dst_width * dst_height;
    int spatial_idx = dy * dst_width + dx;

    d_dst_nchw[0 * plane_stride + spatial_idx] = norm_r;
    d_dst_nchw[1 * plane_stride + spatial_idx] = norm_g;
    d_dst_nchw[2 * plane_stride + spatial_idx] = norm_b;
}

// FP16 Half-Precision Fused Vectorized Kernel
__global__ void preprocess_fused_fp16_kernel(
    const uint8_t* __restrict__ d_src_bgr,
    int src_width,
    int src_height,
    int src_pitch,
    __half* __restrict__ d_dst_nchw_fp16,
    int dst_width,
    int dst_height,
    float scale,
    int pad_w,
    int pad_h,
    __half mean0, __half mean1, __half mean2,
    __half std0, __half std1, __half std2
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_width || dy >= dst_height) return;

    float sx = (dx - pad_w) / scale;
    float sy = (dy - pad_h) / scale;

    int src_x = __float2int_rd(sx);
    int src_y = __float2int_rd(sy);

    float b = 114.0f, g = 114.0f, r = 114.0f;

    if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
        const uint8_t* pixel_ptr = d_src_bgr + src_y * src_pitch + src_x * 3;
        b = static_cast<float>(pixel_ptr[0]);
        g = static_cast<float>(pixel_ptr[1]);
        r = static_cast<float>(pixel_ptr[2]);
    }

    __half h_r = __float2half(r / 255.0f);
    __half h_g = __float2half(g / 255.0f);
    __half h_b = __float2half(b / 255.0f);

    __half norm_r = __hdiv(__hsub(h_r, mean0), std0);
    __half norm_g = __hdiv(__hsub(h_g, mean1), std1);
    __half norm_b = __hdiv(__hsub(h_b, mean2), std2);

    int plane_stride = dst_width * dst_height;
    int spatial_idx = dy * dst_width + dx;

    d_dst_nchw_fp16[0 * plane_stride + spatial_idx] = norm_r;
    d_dst_nchw_fp16[1 * plane_stride + spatial_idx] = norm_g;
    d_dst_nchw_fp16[2 * plane_stride + spatial_idx] = norm_b;
}

void launchPreprocessKernelFP32(
    const uint8_t* d_src_bgr,
    int src_width,
    int src_height,
    int src_pitch,
    float* d_dst_nchw,
    int dst_width,
    int dst_height,
    const PreprocessParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (dst_width + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y,
        1
    );

    preprocess_fused_fp32_kernel<<<grid, block, 0, stream>>>(
        d_src_bgr, src_width, src_height, src_pitch,
        d_dst_nchw, dst_width, dst_height,
        params.scale, params.pad_w, params.pad_h,
        params.mean[0], params.mean[1], params.mean[2],
        params.std[0], params.std[1], params.std[2]
    );
}

void launchPreprocessKernelFP16(
    const uint8_t* d_src_bgr,
    int src_width,
    int src_height,
    int src_pitch,
    __half* d_dst_nchw_fp16,
    int dst_width,
    int dst_height,
    const PreprocessParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (dst_width + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y,
        1
    );

    preprocess_fused_fp16_kernel<<<grid, block, 0, stream>>>(
        d_src_bgr, src_width, src_height, src_pitch,
        d_dst_nchw_fp16, dst_width, dst_height,
        params.scale, params.pad_w, params.pad_h,
        __float2half(params.mean[0]), __float2half(params.mean[1]), __float2half(params.mean[2]),
        __float2half(params.std[0]), __float2half(params.std[1]), __float2half(params.std[2])
    );
}

} // namespace cuda
} // namespace nexus
