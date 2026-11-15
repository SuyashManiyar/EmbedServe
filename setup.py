"""Builds the optional C++ hot tier (embedserve._hot). Metadata lives in pyproject.toml."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension("embedserve._hot", ["embedserve/_hot.cpp"], cxx_std=17,
                          extra_compile_args=["-O3"]),
    ],
    cmdclass={"build_ext": build_ext},
)
