import torch
from typing import Optional, Union

try:
    from .proj_simplex import proj_simplex_1d, proj_simplex_2d
except ImportError as e:
    raise ImportError(
        "Failed to import compiled C++ extension. "
        "Please ensure the module is properly compiled using setup.py"
    ) from e

__version__ = "1.0.0"
__author__ = "Wenjun Luo"
__email__ = "luowenjunn@outlook.com"

__all__ = [
    'ProjSimplex',
    'proj_simplex_1d', 
    'proj_simplex_2d',
    'project_to_simplex_1d',
    'project_to_simplex_2d'
]


def project_to_simplex_1d(
    x: torch.Tensor, 
    k: float = 1.0, 
    max_iterations: int = 1000, 
    tolerance: float = 1e-16,
    inplace: bool = True
) -> torch.Tensor:
    if x.dim() != 1:
        raise ValueError(f"Input must be 1D tensor, got {x.dim()}D")
    
    if x.dtype != torch.float64:
        raise ValueError(f"Input must be float64, got {x.dtype}")
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    if not inplace:
        x = x.clone()
    
    temp = torch.zeros_like(x)
    
    try:
        proj_simplex_1d(x, temp, k, len(x), max_iterations, tolerance)
        return x
    except Exception as e:
        raise RuntimeError(f"Simplex projection failed: {e}") from e


def project_to_simplex_2d(
    x: torch.Tensor, 
    k: float = 1.0, 
    max_iterations: int = 1000, 
    tolerance: float = 1e-16,
    inplace: bool = True
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Input must be 2D tensor, got {x.dim()}D")
    
    if x.dtype != torch.float64:
        raise ValueError(f"Input must be float64, got {x.dtype}")
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    if not inplace:
        x = x.clone()
    
    temp = torch.zeros_like(x)
    
    try:
        proj_simplex_2d(x, temp, k, max_iterations, tolerance)
        return x
    except Exception as e:
        raise RuntimeError(f"Simplex projection failed: {e}") from e


class ProjSimplex:
    
    def __init__(
        self, 
        shape: tuple[int, ...],
        k: float = 1.0, 
        tolerance: float = 1e-16, 
        max_iterations: int = 1000, 
        dtype: torch.dtype = torch.float64,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        if len(shape) not in (1, 2):
            raise ValueError(f"Only 1D and 2D tensors supported, got {len(shape)}D")
        
        if dtype != torch.float64:
            raise ValueError(f"Only float64 supported, got {dtype}")
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        self.k = float(k)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        
        self._workspace = torch.empty(
            self.shape, 
            dtype=self.dtype, 
            device=self.device
        )
        
        if len(shape) == 1:
            self._proj_func = proj_simplex_1d
            self._length = shape[0]
        else:
            self._proj_func = proj_simplex_2d
            self._length = None
    
    def __call__(self, x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        return self.project(x, inplace=inplace)
    
    def project(self, x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        self._validate_input(x)
        
        if not inplace:
            x = x.clone()
        
        if not x.is_contiguous():
            x = x.contiguous()
        
        if self._workspace.device != x.device:
            self._workspace = self._workspace.to(x.device)
        
        try:
            if len(self.shape) == 1:
                self._proj_func(
                    x, self._workspace, self.k, self._length, 
                    self.max_iterations, self.tolerance
                )
            else:
                self._proj_func(
                    x, self._workspace, self.k, 
                    self.max_iterations, self.tolerance
                )
            return x
        except Exception as e:
            raise RuntimeError(f"Simplex projection failed: {e}") from e
    
    def _validate_input(self, x: torch.Tensor) -> None:
        if x.shape != self.shape:
            raise ValueError(
                f"Input shape {x.shape} doesn't match expected {self.shape}"
            )
        
        if x.dtype != self.dtype:
            raise ValueError(
                f"Input dtype {x.dtype} doesn't match expected {self.dtype}"
            )
    
    def __repr__(self) -> str:
        return (
            f"ProjSimplex(shape={self.shape}, k={self.k}, "
            f"tolerance={self.tolerance}, max_iterations={self.max_iterations})"
        )