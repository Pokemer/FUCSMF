/*
 * Author: Wenjun Luo
 * Modern PyTorch C++ binding for CG_DESCENT optimization
 */

#include <torch/extension.h>
#include "cg_user.h"

namespace cg_torch_bindings {

inline double evaluate_function(
    const std::function<torch::Tensor(torch::Tensor)>& func,
    int64_t rows, int64_t cols,
    double* x, INT n) noexcept {
    
    auto tensor = torch::from_blob(x, {rows, cols}, torch::kFloat64);
    return func(tensor).template item<double>();
}

inline void evaluate_gradient(
    const std::function<void(torch::Tensor, torch::Tensor)>& grad_func,
    int64_t rows, int64_t cols,
    double* g, double* x, INT n) noexcept {
    
    auto x_tensor = torch::from_blob(x, {rows, cols}, torch::kFloat64);
    auto g_tensor = torch::from_blob(g, {rows, cols}, torch::kFloat64);
    grad_func(x_tensor, g_tensor);
}

inline double evaluate_value_grad(
    const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& valgrad_func,
    int64_t rows, int64_t cols,
    double* g, double* x, INT n) noexcept {
    
    auto x_tensor = torch::from_blob(x, {rows, cols}, torch::kFloat64);
    auto g_tensor = torch::from_blob(g, {rows, cols}, torch::kFloat64);
    return valgrad_func(x_tensor, g_tensor).template item<double>();
}

int optimize_no_workspace(
    torch::Tensor X, 
    const std::function<torch::Tensor(torch::Tensor)>& objective_func,
    const std::function<void(torch::Tensor, torch::Tensor)>& gradient_func,
    const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& valgrad_func,
    cg_stats* stats,
    cg_parameter* params,
    double tolerance) {
    
    const auto rows = X.sizes()[0];
    const auto cols = X.sizes()[1];
    const auto total_size = rows * cols;
    
    auto value_lambda = [&](double* x, INT n) -> double {
        return evaluate_function(objective_func, rows, cols, x, n);
    };
    
    auto grad_lambda = [&](double* g, double* x, INT n) -> void {
        evaluate_gradient(gradient_func, rows, cols, g, x, n);
    };
    
    auto valgrad_lambda = [&](double* g, double* x, INT n) -> double {
        return evaluate_value_grad(valgrad_func, rows, cols, g, x, n);
    };
    
    return cg_descent(
        X.data_ptr<double>(), 
        total_size, 
        stats, 
        params, 
        tolerance, 
        value_lambda, 
        grad_lambda, 
        valgrad_lambda, 
        nullptr
    );
}

int optimize_with_workspace(
    torch::Tensor X, 
    const std::function<torch::Tensor(torch::Tensor)>& objective_func,
    const std::function<void(torch::Tensor, torch::Tensor)>& gradient_func,
    const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& valgrad_func,
    cg_stats* stats,
    cg_parameter* params,
    double tolerance,
    torch::Tensor workspace) {
    
    const auto rows = X.sizes()[0];
    const auto cols = X.sizes()[1];
    const auto total_size = rows * cols;
    
    auto value_lambda = [&](double* x, INT n) -> double {
        return evaluate_function(objective_func, rows, cols, x, n);
    };
    
    auto grad_lambda = [&](double* g, double* x, INT n) -> void {
        evaluate_gradient(gradient_func, rows, cols, g, x, n);
    };
    
    auto valgrad_lambda = [&](double* g, double* x, INT n) -> double {
        return evaluate_value_grad(valgrad_func, rows, cols, g, x, n);
    };
    
    return cg_descent(
        X.data_ptr<double>(), 
        total_size, 
        stats, 
        params, 
        tolerance, 
        value_lambda, 
        grad_lambda, 
        valgrad_lambda, 
        workspace.data_ptr<double>()
    );
}

} // namespace cg_torch_bindings


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace cg_torch_bindings;
    
    py::class_<cg_parameter>(m, "CGParameter")
        .def(py::init<>())
        .def_readwrite("PrintFinal", &cg_parameter::PrintFinal)
        .def_readwrite("PrintLevel", &cg_parameter::PrintLevel)
        .def_readwrite("PrintParms", &cg_parameter::PrintParms)
        .def_readwrite("LBFGS", &cg_parameter::LBFGS)
        .def_readwrite("memory", &cg_parameter::memory)
        .def_readwrite("SubCheck", &cg_parameter::SubCheck)
        .def_readwrite("SubSkip", &cg_parameter::SubSkip)
        .def_readwrite("eta0", &cg_parameter::eta0)
        .def_readwrite("eta1", &cg_parameter::eta1)
        .def_readwrite("eta2", &cg_parameter::eta2)
        .def_readwrite("AWolfe", &cg_parameter::AWolfe)
        .def_readwrite("AWolfeFac", &cg_parameter::AWolfeFac)
        .def_readwrite("Qdecay", &cg_parameter::Qdecay)
        .def_readwrite("nslow", &cg_parameter::nslow)
        .def_readwrite("StopRule", &cg_parameter::StopRule)
        .def_readwrite("StopFac", &cg_parameter::StopFac)
        .def_readwrite("PertRule", &cg_parameter::PertRule)
        .def_readwrite("eps", &cg_parameter::eps)
        .def_readwrite("egrow", &cg_parameter::egrow)
        .def_readwrite("QuadStep", &cg_parameter::QuadStep)
        .def_readwrite("QuadCutOff", &cg_parameter::QuadCutOff)
        .def_readwrite("QuadSafe", &cg_parameter::QuadSafe)
        .def_readwrite("UseCubic", &cg_parameter::UseCubic)
        .def_readwrite("CubicCutOff", &cg_parameter::CubicCutOff)
        .def_readwrite("SmallCost", &cg_parameter::SmallCost)
        .def_readwrite("debug", &cg_parameter::debug)
        .def_readwrite("debugtol", &cg_parameter::debugtol)
        .def_readwrite("step", &cg_parameter::step)
        .def_readwrite("maxit", &cg_parameter::maxit)
        .def_readwrite("ntries", &cg_parameter::ntries)
        .def_readwrite("ExpandSafe", &cg_parameter::ExpandSafe)
        .def_readwrite("SecantAmp", &cg_parameter::SecantAmp)
        .def_readwrite("RhoGrow", &cg_parameter::RhoGrow)
        .def_readwrite("neps", &cg_parameter::neps)
        .def_readwrite("nshrink", &cg_parameter::nshrink)
        .def_readwrite("nline", &cg_parameter::nline)
        .def_readwrite("restart_fac", &cg_parameter::restart_fac)
        .def_readwrite("feps", &cg_parameter::feps)
        .def_readwrite("nan_rho", &cg_parameter::nan_rho)
        .def_readwrite("nan_decay", &cg_parameter::nan_decay)
        .def_readwrite("delta", &cg_parameter::delta)
        .def_readwrite("sigma", &cg_parameter::sigma)
        .def_readwrite("gamma", &cg_parameter::gamma)
        .def_readwrite("rho", &cg_parameter::rho)
        .def_readwrite("psi0", &cg_parameter::psi0)
        .def_readwrite("psi_lo", &cg_parameter::psi_lo)
        .def_readwrite("psi_hi", &cg_parameter::psi_hi)
        .def_readwrite("psi1", &cg_parameter::psi1)
        .def_readwrite("psi2", &cg_parameter::psi2)
        .def_readwrite("AdaptiveBeta", &cg_parameter::AdaptiveBeta)
        .def_readwrite("BetaLower", &cg_parameter::BetaLower)
        .def_readwrite("theta", &cg_parameter::theta)
        .def_readwrite("qeps", &cg_parameter::qeps)
        .def_readwrite("qrule", &cg_parameter::qrule)
        .def_readwrite("qrestart", &cg_parameter::qrestart);

    py::class_<cg_stats>(m, "CGStats")
        .def(py::init<>())
        .def_readwrite("f", &cg_stats::f)
        .def_readwrite("gnorm", &cg_stats::gnorm)
        .def_readwrite("iter", &cg_stats::iter)
        .def_readwrite("IterSub", &cg_stats::IterSub)
        .def_readwrite("NumSub", &cg_stats::NumSub)
        .def_readwrite("nfunc", &cg_stats::nfunc)
        .def_readwrite("ngrad", &cg_stats::ngrad);

    m.def(
        "optimize_no_workspace", 
        &optimize_no_workspace, 
        py::arg("X"),
        py::arg("objective_func"),
        py::arg("gradient_func"),
        py::arg("valgrad_func"),
        py::arg("stats") = nullptr,
        py::arg("params"),
        py::arg("tolerance"),
        R"pbdoc(
        Conjugate gradient optimization without workspace allocation.
        
        Args:
            X: Input tensor to optimize
            objective_func: Function to minimize
            gradient_func: Gradient computation function
            valgrad_func: Combined value and gradient function
            stats: Optional statistics output
            params: CG parameters
            tolerance: Convergence tolerance
        
        Returns:
            int: Optimization status code
        )pbdoc"
    );
    
    m.def(
        "optimize_with_workspace", 
        &optimize_with_workspace, 
        py::arg("X"),
        py::arg("objective_func"),
        py::arg("gradient_func"),
        py::arg("valgrad_func"),
        py::arg("stats") = nullptr,
        py::arg("params"),
        py::arg("tolerance"),
        py::arg("workspace"),
        R"pbdoc(
        Conjugate gradient optimization with pre-allocated workspace.
        
        Args:
            X: Input tensor to optimize
            objective_func: Function to minimize
            gradient_func: Gradient computation function
            valgrad_func: Combined value and gradient function
            stats: Optional statistics output
            params: CG parameters
            tolerance: Convergence tolerance
            workspace: Pre-allocated workspace tensor
        
        Returns:
            int: Optimization status code
        )pbdoc"
    );
    
    m.def("cg_default", &cg_default, "Get default CG parameters");
}   