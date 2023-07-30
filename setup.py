import numpy as np
from setuptools import Extension, find_packages, setup

setup(
    name="cougar",
    version="0.2.0",
    author="Yunchong Gan",
    author_email="yunchong@pku.edu.cn",
    packages=find_packages(),
    requires=["numpy"],
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    ext_modules=[
        Extension(
            "cougar.rolling",
            ["cougar/rolling.c"],
            include_dirs=["cougar", np.get_include()],
            extra_compile_args=["-O2"],
        )
    ],
)
