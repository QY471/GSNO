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

cudart_library_dir = os.environ.get("GSFUSION_CUDART_LIBRARY_DIR", "")

setup(
    name="diff_cell_aware_srgaussian_rasterization",
    packages=['diff_cell_aware_srgaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_cell_aware_srgaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_pixels.cu",
            "ext.cpp"],
            library_dirs=[cudart_library_dir] if cudart_library_dir else [],)
            # extra_compile_args={"nvcc": ["-g", "-G"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
