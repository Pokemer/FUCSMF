/*
 * Author: Wenjun Luo
 * Date: 2025-08-30
 */

#include <torch/extension.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

int count_positive_and_sum(const double* x, double* output, int length) {
    if (!x || !output || length <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }
    
    int positive_count = 0;
    *output = 0.0;
    
    if (length > 1000) {
        double local_sum = 0.0;
        #pragma omp parallel for reduction(+:positive_count) reduction(+:local_sum)
        for (int i = 0; i < length; i++) {
            if (x[i] > 0.0) {
                positive_count++;
                local_sum += x[i];
            }
        }
        *output = local_sum;
    } else {
        for (int i = 0; i < length; i++) {
            if (x[i] > 0.0) {
                positive_count++;
                *output += x[i];
            }
        }
    }
    
    return positive_count;
}

double compute_mean(const double* x, int length) {
    if (!x || length <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }
    
    double sum = 0.0;
    
    for (int i = 0; i < length; i++) {
        sum += x[i];
    }
    
    return sum / static_cast<double>(length);
}

double find_min_after_subtract(double* x, int length, double k) {
    if (!x || length <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }
    
    x[0] -= k;
    double min_val = x[0];
    
    for (int i = 1; i < length; i++) {
        x[i] -= k;
        min_val = std::min(min_val, x[i]);
    }
    
    return min_val;
}

double find_min_after_subtract_v2(double* x, int length, double k) {
    return find_min_after_subtract(x, length, k);
}

void subtract_and_copy(const double* x, double k, double* output, int length) {
    if (!x || !output || length <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }
    
    for (int i = 0; i < length; i++) {
        output[i] = x[i] - k;
    }
}

void subtract_and_clamp_to_zero(double* x, double k, int length) {
    if (!x || length <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }
    
    for (int i = 0; i < length; i++) {
        x[i] = std::max(0.0, x[i] - k);
    }
}

void print_array_debug(const char* label, const double* x, long long n) {
    if (!label || !x || n <= 0) return;
    
    std::cout << label << ": ";
    for (long long i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
}

void proj_simplex_1d_cpu(double* x, double* v0, double k, long long length, 
                        int max_iterations, double tolerance) {
    if (!x || !v0 || length <= 0 || max_iterations <= 0 || tolerance <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }
    
    constexpr double initial_f = 1.0;
    double lambda_multiplier = 0.0;
    int positive_count;
    double positive_sum, residual, mean_val, min_val;

    mean_val = compute_mean(x, length);
    min_val = find_min_after_subtract(x, length, mean_val - k / static_cast<double>(length));
    
    if (min_val < 0.0) {
        for (int iter = 0; iter < max_iterations; iter++) {
            subtract_and_copy(x, lambda_multiplier, v0, length);
            positive_count = count_positive_and_sum(v0, &positive_sum, length);
            
            if (positive_count == 0) {
                break;
            }
            
            residual = positive_sum - k;
            
            if (std::abs(residual) <= tolerance) {
                break;
            }
            
            lambda_multiplier += residual / static_cast<double>(positive_count);
        }
        
        subtract_and_clamp_to_zero(x, lambda_multiplier, length);
    }
}

void proj_simplex_1d(torch::Tensor x, torch::Tensor v0, double k, 
                    long long length, int max_iterations, double tolerance) {
    if (!x.is_contiguous() || !v0.is_contiguous()) {
        throw std::runtime_error("Tensors must be contiguous");
    }
    
    if (x.dtype() != torch::kFloat64 || v0.dtype() != torch::kFloat64) {
        throw std::runtime_error("Tensors must be of type double (float64)");
    }
    
    auto x_ptr = x.data_ptr<double>();
    auto v0_ptr = v0.data_ptr<double>();
    
    proj_simplex_1d_cpu(x_ptr, v0_ptr, k, length, max_iterations, tolerance);
}

void proj_simplex_2d(torch::Tensor x, torch::Tensor v0, double k, 
                    int max_iterations, double tolerance) {
    if (!x.is_contiguous() || !v0.is_contiguous()) {
        throw std::runtime_error("Tensors must be contiguous");
    }
    
    if (x.dtype() != torch::kFloat64 || v0.dtype() != torch::kFloat64) {
        throw std::runtime_error("Tensors must be of type double (float64)");
    }
    
    const auto batch_size = v0.size(0);
    const auto vector_length = v0.size(1);
    
    auto x_ptr = x.data_ptr<double>();
    auto v0_ptr = v0.data_ptr<double>();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < batch_size; i++) {
        const auto offset = i * vector_length;
        try {
            proj_simplex_1d_cpu(&x_ptr[offset], &v0_ptr[offset], k, 
                              vector_length, max_iterations, tolerance);
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Error in batch " << i << ": " << e.what() << std::endl;
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Efficient simplex projection operations for PyTorch tensors";
    
    m.def("proj_simplex_2d", &proj_simplex_2d, 
          "Perform simplex projection on 2D tensor (batch processing)",
          py::arg("x"), py::arg("v0"), py::arg("k"), 
          py::arg("max_iterations") = 1000, py::arg("tolerance") = 1e-16);
          
    m.def("proj_simplex_1d", &proj_simplex_1d, 
          "Perform simplex projection on 1D tensor",
          py::arg("x"), py::arg("v0"), py::arg("k"), py::arg("length"),
          py::arg("max_iterations") = 1000, py::arg("tolerance") = 1e-16);
}