import os
import platform
from setuptools import setup
from torch.utils import cpp_extension


def get_compile_args():
    if platform.system() == "Windows":
        return [
            '/O2',
            '/openmp',
            '/std:c++17',
            '/fp:fast',
            '/Qvec-report:1',
            '/DMKL_ILP64',
        ]
    else:
        return [
            '-O3',
            '-fopenmp',
            '-std=c++17',
            '-ffast-math',
            '-march=native',
            '-DMKL_ILP64',
        ]


def get_link_args():
    if platform.system() == "Windows":
        return ['/openmp']
    else:
        return ['-fopenmp']


def create_extension():
    source_files = ['Proj_Simplex_dense.cpp']
    
    for src in source_files:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
    
    return cpp_extension.CppExtension(
        name='proj_simplex',
        sources=source_files,
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
        language='c++'
    )


def main():
    setup(
        name='proj_simplex',
        version='1.0.0',
        author='Wenjun Luo',
        author_email='luowenjunn@outlook.com',
        description='Efficient simplex projection operations for PyTorch',
        
        ext_modules=[create_extension()],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
        
        python_requires='>=3.12',
        
        install_requires=[
            'torch>=2.0.0',
            'numpy>=1.24.0',
        ],
        
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Programming Language :: C++',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        
        keywords='simplex projection optimization pytorch c++ extension',
        
        include_package_data=True,
        zip_safe=False,
    )


if __name__ == '__main__':
    main()
