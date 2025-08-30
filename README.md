# Fast Unconstraint Convex Symmetric Matrix for Semi-Supervised Learning

A high-performance implementation of "Fast Unconstraint Convex Symmetric Matrix for
Semi-Supervised Learning". This project provides efficient algorithms for large-scale clustering and classification problems using advanced optimization techniques.

## 🚀 Features

- **Fast Conjugate Gradient Optimization**: High-performance C++ implementation with PyTorch integration
- **Efficient Simplex Projections**: Optimized C++ extensions for simplex constraint handling
- **Semi-Supervised Learning**: Support for labeled data
- **Scalable Architecture**: Designed for large-scale datasets with anchor-based methods

## 📋 Requirements

- Python 3.12+
- PyTorch
- Intel MKL
- C++ compiler with C++17 support
- Microsoft Visual Studio 2022 (Windows) or GCC/Clang (Linux/macOS)

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### 1. Install uv (if not already installed)

<!-- ```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
``` -->

<!-- ### 2. Clone and Setup the Project

```bash
git clone <repository-url>
cd FUCSMF
``` -->

### 1. Install Python Dependencies

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv sync
```

### 2. Build CG_DESCENT_C Extension

The CG_DESCENT_C module provides high-performance conjugate gradient optimization:

```bash
cd CG_DESCENT_C

uv run setup.py build_ext --inplace

cd ..
```

### 3. Build ProjSimplex_cpp Extension

The ProjSimplex_cpp module provides efficient simplex projection operations:

```bash
cd Utils/ProjSimplex_cpp

uv run setup.py build_ext --inplace

cd ../..
```

## 🎯 Quick Start


### Running Demo

```bash
# Run the complete demo with COIL20 and COIL100 datasets
uv run demo.py
```


