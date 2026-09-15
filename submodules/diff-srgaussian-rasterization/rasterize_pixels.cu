/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "cuda_rasterizer/config.h"
#include "rasterize_pixels.h"
#include "cuda_rasterizer/forward.h"
#include "cuda_rasterizer/backward.h"
#include <tuple>
#include <iostream>

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret) << std::endl; \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

std::tuple<torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &opacity,
    const torch::Tensor &means,
    const torch::Tensor &stds,
    const torch::Tensor &rhos,
    const torch::Tensor &colors,
    const int image_height,
    const int image_width,
    const float scale_factor,
    const float raster_ratio,
    const int num_channels,
    const bool debug)
{
    int batch_size = means.size(0);
    int num_gaussians = means.size(1);

    const int sH = scale_factor * image_height;
    const int sW = scale_factor * image_width;
    
    torch::Tensor out_color = torch::zeros({batch_size, sH, sW, num_channels}, opacity.options());

    dim3 block(BLOCK_X * BLOCK_Y);
    dim3 grid((num_gaussians + block.x - 1) / block.x);

    for (int b = 0; b < batch_size; b++) {
        CHECK_CUDA(FORWARD::render(
            grid, block,
            num_gaussians,
            opacity.contiguous().data_ptr<float>() + b * num_gaussians,
            reinterpret_cast<float2 *>(means.contiguous().data_ptr<float>() + b * num_gaussians * 2),
            reinterpret_cast<float2 *>(stds.contiguous().data_ptr<float>() + b * num_gaussians * 2),
            rhos.contiguous().data_ptr<float>() + b * num_gaussians,
            colors.contiguous().data_ptr<float>() + b * num_gaussians * num_channels,
            sH, sW,
            scale_factor,
            raster_ratio,
            num_channels,
            out_color.contiguous().data_ptr<float>() + b * sH * sW * num_channels), debug);
    }
    return std::make_tuple(out_color);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& opacity,
    const torch::Tensor& means,
    const torch::Tensor& stds,
    const torch::Tensor& rhos,
    const torch::Tensor& colors,
    const torch::Tensor& grad_output,
    const int image_height,
    const int image_width,
    const float scale_factor,
    const float raster_ratio,
    const int num_channels,
    const bool debug
)
{
    int batch_size = means.size(0);
    int num_gaussians = means.size(1);

    const int sH = scale_factor * image_height;
    const int sW = scale_factor * image_width;

    torch::Tensor dL_dopacity = torch::zeros_like(opacity);
    torch::Tensor dL_dmeans = torch::zeros_like(means);
    torch::Tensor dL_dstds = torch::zeros_like(stds);
    torch::Tensor dL_drhos = torch::zeros_like(rhos);
    torch::Tensor dL_dcolors = torch::zeros_like(colors);

    dim3 block(BLOCK_X * BLOCK_Y);
    dim3 grid((num_gaussians + block.x - 1) / block.x);
    
    for (int b = 0; b < batch_size; b++) {
        CHECK_CUDA(BACKWARD::render(
            grid, block,
            num_gaussians,
            opacity.contiguous().data_ptr<float>() + b * num_gaussians,
            reinterpret_cast<float2 *>(means.contiguous().data_ptr<float>() + b * num_gaussians * 2),
            reinterpret_cast<float2 *>(stds.contiguous().data_ptr<float>() + b * num_gaussians * 2),
            rhos.contiguous().data_ptr<float>() + b * num_gaussians,
            colors.contiguous().data_ptr<float>() + b * num_gaussians * num_channels,
            grad_output.contiguous().data_ptr<float>() + b * sH * sW * num_channels,
            sH, sW,
            scale_factor,
            raster_ratio,
            num_channels,
            dL_dopacity.contiguous().data_ptr<float>() + b * num_gaussians,
            reinterpret_cast<float2 *>(dL_dmeans.contiguous().data_ptr<float>() + b * num_gaussians * 2),
            reinterpret_cast<float2 *>(dL_dstds.contiguous().data_ptr<float>() + b * num_gaussians * 2),
            dL_drhos.contiguous().data_ptr<float>() + b * num_gaussians,
            dL_dcolors.contiguous().data_ptr<float>() + b * num_gaussians * num_channels
        ), debug);
    }
    return std::make_tuple(dL_dopacity, dL_dmeans, dL_dstds, dL_drhos, dL_dcolors);
}
