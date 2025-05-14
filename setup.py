from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).parent

setup(
    name="flash",
    packages=["flash"],
    python_requires=">=3.10",
    ext_modules=[
        CUDAExtension(
            "flash._naive",
            [
                str(this_dir / "flash/kernels/naive.cu"),
            ],
            extra_compile_args={"nvcc": ["-O2", "--use_fast_math"]},
        ),
        CUDAExtension(
            "flash._flash1",
            [
                str(this_dir / "flash/kernels/flash1.cu"),
            ],
            extra_compile_args={"nvcc": ["-O3", "--use_fast_math"]},
        ),
        CUDAExtension(
            "flash._flash2",
            [
                str(this_dir / "flash/kernels/flash2.cu"),
            ],
            extra_compile_args={"nvcc": ["-O3", "--use_fast_math"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)