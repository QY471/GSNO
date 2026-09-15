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

#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& opacity,
    const torch::Tensor& means,
    const torch::Tensor& stds,
    const torch::Tensor& rhos,
    const torch::Tensor& colors,
    const int image_height,
    const int image_width,
    const float scale_factor,
    const float raster_ratio,
    const int num_channels,
    const bool debug
);

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
);
