#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_srgaussian_rasterization",
    packages=['diff_srgaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_srgaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_pixels.cu",
            "ext.cpp"],)
            # extra_compile_args={"nvcc": ["-g", "-G"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)