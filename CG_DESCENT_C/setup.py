#!/usr/bin/env python3
"""
Author: pokemer
Modern setup configuration for CG_DESCENT C++ extension with PyTorch integration
"""

import sys
import platform
from pathlib import Path
from setuptools import setup
from torch.utils import cpp_extension

def get_mkl_paths():
    """Detect Intel MKL installation paths"""
    intel_paths = [
        Path(r"C:\Program Files (x86)\Intel\oneAPI"),
        Path(r"C:\Program Files\Intel\oneAPI"),
        Path("/opt/intel/oneapi")
    ]
    
    for base_path in intel_paths:
        if base_path.exists():
            mkl_path = base_path / "mkl" / "latest"
            if mkl_path.exists():
                return {
                    'include': [
                        str(mkl_path / "include"),
                        str(mkl_path / "include" / "fftw"),
                        str(mkl_path / "include" / "mkl")
                    ],
                    'lib': [
                        str(base_path / "lib"),
                        str(mkl_path / "lib"),
                        str(base_path / "compiler" / "latest" / "lib")
                    ]
                }
    
    return None

def get_compile_args():
    """Get platform-specific compilation arguments"""
    if platform.system() == "Windows":
        return [
            '/std:c++17',
            '/O2',
            '/openmp',
            '/fp:fast',
            '/DMKL_ILP64',
            '/DWIN32_LEAN_AND_MEAN',
            '/DNOMINMAX'
        ]
    else:
        return [
            '-std=c++17',
            '-O3',
            '-fopenmp',
            '-ffast-math',
            '-DMKL_ILP64',
            '-march=native'
        ]

def get_link_args():
    """Get platform-specific linking arguments"""
    if platform.system() == "Windows":
        return []
    else:
        return ['-fopenmp']

def get_libraries():
    """Get MKL libraries for linking"""
    return [
        'mkl_intel_ilp64',
        'mkl_intel_thread', 
        'mkl_core',
        'libiomp5md' if platform.system() == "Windows" else 'iomp5'
    ]

def create_extension():
    """Create the C++ extension with optimized settings"""
    mkl_paths = get_mkl_paths()
    
    extension_kwargs = {
        'name': 'CG_descent',
        'sources': ['cpp/cg_bind.cpp', 'cpp/cg_descent.cpp'],
        'extra_compile_args': get_compile_args(),
        'extra_link_args': get_link_args(),
        'language': 'c++'
    }
    
    if mkl_paths:
        extension_kwargs.update({
            'include_dirs': mkl_paths['include'],
            'library_dirs': mkl_paths['lib'],
            'libraries': get_libraries()
        })
        print("✓ Intel MKL found and configured")
    else:
        print("⚠ Intel MKL not found, using default BLAS")
    
    return cpp_extension.CppExtension(**extension_kwargs)

if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise RuntimeError("Python >= 3.12 required")
    
    setup(
        name='CG_descent',
        version='2.0.0',
        author='pokemer',
        description='Modern C++17 Conjugate Gradient Optimization with PyTorch',
        long_description='High-performance conjugate gradient optimization library with PyTorch integration',
        ext_modules=[create_extension()],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
        python_requires='>=3.12',
        install_requires=[
            'torch>=2.0.0',
            'numpy>=1.20.0'
        ],
        zip_safe=False
    )