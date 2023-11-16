#!/usr/bin/env python

# from distutils.core import run_setup
# from setuptools import find_packages

from setuptools import setup, find_packages

setup(
    name="MagneticClasses",
    version="0.0.1",
    description="Python package containing classes for magnetic field calculations",
    author="534rch",
    packages=["magnetic_classes"],
    install_requires=["numpy", "scipy", "matplotlib", "PyQt6", "pandas"]
)
