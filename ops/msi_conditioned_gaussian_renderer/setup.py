from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="msi_conditioned_gaussian_renderer_cuda",
    ext_modules=[
        CUDAExtension(
            name="msi_conditioned_gaussian_renderer_cuda",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "forward.cu"),
                str(ROOT / "csrc" / "backward.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
