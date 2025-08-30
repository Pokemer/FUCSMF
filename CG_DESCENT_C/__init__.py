"""
Author: pokemer
Modern Python interface for CG_DESCENT optimization library
"""

from typing import Optional, Callable, Union
import torch
import os
import sys

# 确保torch先被导入，这对PyTorch C++扩展是必需的
# 添加当前目录到路径以确保能找到编译好的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from . import CG_descent
except ImportError:
    try:
        import CG_descent
    except ImportError as e:
        # 提供更详细的错误信息
        raise ImportError(
            f"无法导入CG_descent模块。请确保已编译C++扩展。"
            f"运行: cd {current_dir} && uv run python setup.py build_ext --inplace"
        ) from e


class CGOptimizer:
    """Modern wrapper for CG_DESCENT optimization algorithm"""
    
    def __init__(self, 
                 tolerance: float = 1e-6,
                 max_iterations: Optional[int] = None,
                 print_level: int = 0,
                 use_lbfgs: bool = False,
                 memory: int = 5):
        """
        Initialize CG optimizer with modern interface
        
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            print_level: Verbosity level (0=silent, 1=minimal, 2=verbose)
            use_lbfgs: Whether to use L-BFGS variant
            memory: Memory parameter for L-BFGS
        """
        self.params = CG_descent.CGParameter()
        CG_descent.cg_default(self.params)
        self.tolerance = tolerance
        
        if max_iterations is not None:
            self.params.maxit = max_iterations
        
        self.params.PrintLevel = print_level
        self.params.LBFGS = use_lbfgs
        self.params.memory = memory if use_lbfgs else 0
        
    def optimize(self, 
                 x: torch.Tensor,
                 objective_fn: Callable[[torch.Tensor], torch.Tensor],
                 gradient_fn: Callable[[torch.Tensor, torch.Tensor], None],
                 valgrad_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 workspace: Optional[torch.Tensor] = None) -> tuple[int, dict]:
        """
        Optimize the given objective function
        
        Args:
            x: Initial point (will be modified in-place)
            objective_fn: Function to minimize f(x) -> scalar
            gradient_fn: Gradient computation g(x, grad) -> None
            valgrad_fn: Combined value/gradient function (optional)
            workspace: Pre-allocated workspace (optional)
            
        Returns:
            Tuple of (status_code, statistics_dict)
        """
        if not x.is_contiguous():
            x = x.contiguous()
        
        if x.dtype != torch.float64:
            raise ValueError("Input tensor must be float64")
        
        stats = CG_descent.CGStats()
        
        if valgrad_fn is None:
            def default_valgrad(x_tensor: torch.Tensor, g_tensor: torch.Tensor) -> torch.Tensor:
                f_val = objective_fn(x_tensor)
                gradient_fn(x_tensor, g_tensor)
                return f_val
            valgrad_fn = default_valgrad
        
        if workspace is not None:
            status = CG_descent.optimize_with_workspace(
                x, objective_fn, gradient_fn, valgrad_fn,
                stats, self.params, self.tolerance, workspace
            )
        else:
            status = CG_descent.optimize_no_workspace(
                x, objective_fn, gradient_fn, valgrad_fn,
                stats, self.params, self.tolerance
            )
        
        return status, {
            'final_value': stats.f,
            'gradient_norm': stats.gnorm,
            'iterations': stats.iter,
            'function_evaluations': stats.nfunc,
            'gradient_evaluations': stats.ngrad,
            'subspace_iterations': stats.IterSub,
            'num_subspaces': stats.NumSub
        }

def cg_descent(p, tol=1e-6, cg_stats=None, cg_parameter=None, work=None, **kwargs):
    """Legacy interface for backward compatibility"""
    if cg_parameter is None:
        cg_parameter = CG_descent.CGParameter()
        CG_descent.cg_default(cg_parameter)
    
    for key, value in kwargs.items():
        if hasattr(cg_parameter, key):
            setattr(cg_parameter, key, value)
    
    if work is None:
        return CG_descent.optimize_no_workspace(
            p.Z, p.evalobj, p.evalgrad, p.evalobjgrad, 
            cg_stats, cg_parameter, tol
        )
    else:
        return CG_descent.optimize_with_workspace(
            p.Z, p.evalobj, p.evalgrad, p.evalobjgrad, 
            cg_stats, cg_parameter, tol, work
        )

def get_default_parameters():
    """Get default CG parameters"""
    params = CG_descent.CGParameter()
    CG_descent.cg_default(params)
    return params

def optimize_function(x: torch.Tensor,
                      objective_fn: Callable[[torch.Tensor], torch.Tensor],
                      gradient_fn: Callable[[torch.Tensor, torch.Tensor], None],
                      tolerance: float = 1e-6,
                      **kwargs) -> tuple[int, dict]:
    """
    Convenience function for quick optimization
    
    Args:
        x: Initial point
        objective_fn: Function to minimize
        gradient_fn: Gradient computation function
        tolerance: Convergence tolerance
        **kwargs: Additional parameters for CGOptimizer
        
    Returns:
        Tuple of (status_code, statistics_dict)
    """
    optimizer = CGOptimizer(tolerance=tolerance, **kwargs)
    return optimizer.optimize(x, objective_fn, gradient_fn)

# 导出主要接口
try:
    CGParameter = CG_descent.CGParameter
    CGStats = CG_descent.CGStats
    cg_default = CG_descent.cg_default
except AttributeError as e:
    raise ImportError(f"CG_descent模块缺少必要的类或函数: {e}")

__all__ = [
    'CGOptimizer',
    'cg_descent',
    'get_default_parameters', 
    'optimize_function',
    'CGParameter',
    'CGStats',
    'cg_default'
]