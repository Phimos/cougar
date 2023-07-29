import numpy as np
from setuptools import Extension, find_packages, setup

setup(
    name="cougar",
    version="0.1.0",
    author="Yunchong Gan",
    author_email="yunchong@pku.edu.cn",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "cougar.rolling",
            ["cougar/rolling.c"],
            include_dirs=["cougar", np.get_include()],
            extra_compile_args=["-O2"],
        )
    ],
)
