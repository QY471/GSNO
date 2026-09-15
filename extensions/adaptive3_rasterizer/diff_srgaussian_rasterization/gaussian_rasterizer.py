# Check https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0/diff_gaussian_rasterization/__init__.py
# as a guide to implement this module

import torch
import torch.nn as nn
from . import _C

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        opacity,
        means,
        stds,
        rhos,
        colors,
        image_height,
        image_width,
        scale_factor,
        raster_ratio,
        num_channels,
        debug,
        adaptive_window,
        sigma_radius,
    ):
        args = (
            opacity,
            means,
            stds,
            rhos,
            colors,
            image_height,
            image_width,
            scale_factor,
            raster_ratio,
            num_channels,
            debug,
            adaptive_window,
            sigma_radius,
        )

        out_image = _C.rasterize_gaussians(*args)

        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.scale_factor = scale_factor
        ctx.raster_ratio = raster_ratio
        ctx.num_channels = num_channels
        ctx.debug = debug
        ctx.adaptive_window = adaptive_window
        ctx.sigma_radius = sigma_radius
        ctx.save_for_backward(opacity, means, stds, rhos, colors)        

        return out_image

    @staticmethod
    def backward(ctx, grad_output):
        opacity, means, stds, rhos, colors = ctx.saved_tensors
        image_height = ctx.image_height
        image_width = ctx.image_width
        scale_factor = ctx.scale_factor
        raster_ratio = ctx.raster_ratio
        num_channels = ctx.num_channels
        debug = ctx.debug
        adaptive_window = ctx.adaptive_window
        sigma_radius = ctx.sigma_radius

        args = (
            opacity,
            means,
            stds,
            rhos,
            colors,
            grad_output,
            image_height,
            image_width,
            scale_factor,
            raster_ratio,
            num_channels,
            debug,
            adaptive_window,
            sigma_radius,
        )

        # Invoke _C backward method
        grad_opacity, grad_means, grad_stds, grad_rhos, grad_colors = _C.rasterize_gaussians_backward(*args)
        
        grads = (
            grad_opacity,
            grad_means,
            grad_stds,
            grad_rhos,
            grad_colors,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
    
    def forward(self,
        opacity,
        means,
        stds,
        rhos,
        colors,
        image_height,
        image_width,
        scale_factor,
        raster_ratio,
        debug=False,
        adaptive_window=False,
        sigma_radius=3.0,
    ):
        return _RasterizeGaussians.apply(
            opacity,
            means,
            stds,
            rhos,
            colors,
            image_height,
            image_width,
            scale_factor,
            raster_ratio,
            self.num_channels,
            debug,
            adaptive_window,
            sigma_radius,
        )[0]
