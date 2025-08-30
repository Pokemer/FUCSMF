import numpy as np
import torch
# from Archor_util import *
from Utils.util import *
from Utils.Graphs import generate_graph
from Utils.ProjSimplex_cpp import ProjSimplex
from CG_DESCENT_C import cg_descent, CGStats
from functools import partial
import time as t


# @torch.jit.script
def _evalobj(Z:torch.Tensor, alpha:float, ZZ:torch.Tensor, ZAS_plus_YS:torch.Tensor, AS:torch.Tensor, YS:torch.Tensor, C:torch.Tensor):
    ZZ[:]           = Z@Z.T
    ZAS_plus_YS[:]  = Z@AS
    ZAS_plus_YS    += YS
    objf            = alpha*torch.trace(ZZ)-(2+alpha)*torch.norm(ZAS_plus_YS).square()+torch.norm(ZZ+C).square()
    return objf


# @torch.jit.script
def _evalgrad(Z:torch.Tensor, grad:torch.Tensor, alpha:float, ZZ:torch.Tensor, ZAS_plus_YS:torch.Tensor, AS:torch.Tensor, YS:torch.Tensor, C:torch.Tensor, Ic:torch.Tensor):
    ZZ[:]           = Z@Z.T
    ZAS_plus_YS[:]  = Z@AS
    ZAS_plus_YS    += YS

    grad[:]         = (ZAS_plus_YS)@AS.T
    grad           *= -2*(2+alpha)
    grad           +=4*(ZZ+(alpha/2)*Ic+C)@Z


# @torch.jit.script
def _evalobjgrad(Z:torch.Tensor, grad:torch.Tensor, alpha:float, ZZ:torch.Tensor, ZAS_plus_YS:torch.Tensor, AS:torch.Tensor, YS:torch.Tensor, C:torch.Tensor, Ic:torch.Tensor):
    ZZ[:]           = Z@Z.T
    ZAS_plus_YS[:]  = Z@AS
    ZAS_plus_YS    += YS
    objf            = alpha*torch.trace(ZZ)-(2+alpha)*torch.norm(ZAS_plus_YS).square()+torch.norm(ZZ+C).square()
    
    grad[:]         = (ZAS_plus_YS)@AS.T
    grad           *= -2*(2+alpha)
    grad           +=4*(ZZ+(alpha/2)*Ic+C)@Z
    return objf

# @torch.jit.script
def _evalobj_a(Z:torch.Tensor, alpha:float, ZZ:torch.Tensor, ZAS_plus_YS:torch.Tensor, AS:torch.Tensor, YS:torch.Tensor, C:torch.Tensor):
    torch.mm(Z, Z.T, out = ZZ)
    torch.mm(AS.t(), Z.t(), out=ZAS_plus_YS.t())
    ZAS_plus_YS+=YS
    objf = torch.trace(ZZ)-(2/alpha+1)*torch.norm(ZAS_plus_YS).square()+torch.norm(ZZ+C).square()/alpha
    return objf

# @torch.jit.script
def _evalgrad_a(Z:torch.Tensor, grad:torch.Tensor, alpha:float, ZZ:torch.Tensor, ZAS_plus_YS:torch.Tensor, AS:torch.Tensor, YS:torch.Tensor, C:torch.Tensor, Ic:torch.Tensor):
    torch.mm(Z, Z.T, out = ZZ)
    torch.mm(AS.t(), Z.t(), out=ZAS_plus_YS.t())
    ZAS_plus_YS+=YS
    torch.mm(4/alpha*(ZZ+(alpha/2)*Ic+C), Z, out=grad)
    grad -= 2*(2/alpha+1)*(ZAS_plus_YS)@AS.T

# @torch.jit.script
def _evalobjgrad_a(Z:torch.Tensor, grad:torch.Tensor, alpha:float, ZZ:torch.Tensor, ZAS_plus_YS:torch.Tensor, AS:torch.Tensor, YS:torch.Tensor, C:torch.Tensor, Ic:torch.Tensor):
    torch.mm(Z, Z.T, out = ZZ)
    torch.mm(AS.t(), Z.t(), out=ZAS_plus_YS.t())
    ZAS_plus_YS+=YS
    objf = torch.trace(ZZ)-(2/alpha+1)*torch.norm(ZAS_plus_YS).square()+torch.norm(ZZ+C).square()/alpha
    torch.mm(4/alpha*(ZZ+(alpha/2)*Ic+C), Z, out=grad)
    grad -= 2*(2/alpha+1)*(ZAS_plus_YS)@AS.T
    return objf

# @profile
# @torch.jit.script
def _update_S(value:torch.Tensor, value_old:torch.Tensor, indices:torch.Tensor, H:torch.Tensor, HS:torch.Tensor, XUDist:torch.Tensor, gamma:float, k:int):
    value[:] = (H[:, indices[0]]*HS[:, indices[1]]).sum(dim=0)
    value -= H.norm(p = 2, dim=0, keepdim=True).square().T.repeat(1, k).flatten()*value_old
    value *= 0.01
    value -= XUDist
    value /= 2*gamma

    




class P():
    def __init__(self, X, U, Y, UnLabelInd, k, alpha, gamma, * , dtype=None) -> None:

        self.dtype = X.dtype if dtype is None else dtype
        self.c     = Y.shape[0]
        self.n     = X.shape[0]
        self.d     = X.shape[1]
        self.m     = U.shape[0]
        self.X     = X
        self.U     = U

        self.C  = Y.sum(1).values()
        self.C  = torch.sparse_coo_tensor(torch.arange(self.c).repeat(2,1), self.C)
        self.Ic = torch.sparse_coo_tensor(torch.arange(self.c).repeat(2,1), torch.ones(self.c, dtype=self.dtype))
        self.Y  = Y
        un_len  = UnLabelInd.shape[0]
        self.A2 = torch.sparse_coo_tensor(torch.vstack((torch.arange(un_len), torch.tensor(UnLabelInd))),
                                     torch.ones(un_len, dtype=self.dtype), (un_len, self.n))
        
        self.k      = k
        self.alpha  = alpha
        self.gamma  = gamma
        self.Lambda = torch.ones(self.m, dtype=self.dtype)

        self.proj = ProjSimplex(shape=(self.n, self.k))

        self.Z           = torch.zeros((self.c, un_len), dtype=self.dtype)
        self.ZZ          = torch.empty((self.c, self.c), dtype=self.dtype)
        self.ZAS_plus_YS = torch.empty((self.c, self.m), dtype=self.dtype)
        self.H           = torch.zeros((self.c, self.n), dtype=self.dtype)
        self.HS          = torch.zeros((self.c, self.m), dtype=self.dtype)
        self.work        = torch.empty(20*self.c*un_len, dtype=self.dtype)


    def init_XU(self):
        XU = torch.cdist(self.X, self.U)
        value, indices  = XU.topk(self.k, dim=1, largest=False)
        self.XUDist     = value.square()
        self.XUDist     = -torch.exp(-self.XUDist/self.XUDist.mean(1,keepdim=True))
        ind_0 = torch.arange(self.n).repeat(self.k,1).T.flatten()
        ind_1, ind_order = indices.sort(dim=1)
        self.XUDist     = self.XUDist[ind_0, ind_order.flatten()]
        self.value      = self.XUDist.clone()
        self.value_old  = torch.zeros_like(self.value)
        self.indices    = torch.vstack((ind_0, ind_1.flatten()))
        self.S          = torch.sparse_coo_tensor(self.indices, self.value, (self.n, self.m)).coalesce()
        self.value      = self.S.values()
        self.indices    = self.S.indices()
        
        
        self.YS_0 = 0
        self.AS_0 = 0

    
    def set_S(self, S):
        self.S = S
        self.Lambda = torch.zeros(self.m, dtype=self.dtype)
        Lsum   = self.S.sum(0)
        self.Lambda[Lsum.indices().flatten()] = Lsum.values()
        self.Lambda.clamp(min=1e-16)
        self.Lambda **= -1
        self.S   *= self.Lambda**(1/2)
        self.AS   = self.A2 @ self.S
        self.YS   = self.Y  @ self.S


    def update_S(self, iter_time):
        if iter_time != 0:
            self.HS[:] = self.Z@self.AS_0 + self.YS_0
            self.H[:]   =  self.Z@self.A2 + self.Y
        _update_S(self.value, self.value_old, self.indices, self.H, self.HS, self.XUDist, self.gamma, self.k)
        self.proj.project(self.value.view(self.n, self.k))

        self.set_S(self.S)

        self.S          = self.S.coalesce()
        self.value      = self.S.values()
        self.indices    = self.S.indices()

        self.AS_0       = self.AS * self.Lambda**(1/2)
        self.YS_0       = self.YS * self.Lambda**(1/2)
        self.value_old  = (self.S*self.Lambda**(1/2)).coalesce().values()



    def evalobj(self, Z):
        if self.alpha<=1:
            return _evalobj(Z, self.alpha, self.ZZ, self.ZAS_plus_YS, self.AS, self.YS, self.C)
        else:
            return _evalobj_a(Z, self.alpha, self.ZZ, self.ZAS_plus_YS, self.AS, self.YS, self.C)
    

    def evalgrad(self, Z, grad):
        if self.alpha<=1:
            return _evalgrad(Z, grad, self.alpha, self.ZZ, self.ZAS_plus_YS, self.AS, self.YS, self.C, self.Ic)
        else:
            return _evalgrad_a(Z, grad, self.alpha, self.ZZ, self.ZAS_plus_YS, self.AS, self.YS, self.C, self.Ic)
    

    def evalobjgrad(self, Z, grad):
        if self.alpha<=1:
            return _evalobjgrad(Z, grad, self.alpha, self.ZZ, self.ZAS_plus_YS, self.AS, self.YS, self.C, self.Ic)
        else:
            return _evalobjgrad_a(Z, grad, self.alpha, self.ZZ, self.ZAS_plus_YS, self.AS, self.YS, self.C, self.Ic)
    


def FUCSMF(X, U, Y, UnLabelInd, *, k = 5, alpha = 100, gamma = 1, max_iter = 5):
    if gamma is not None:
        gamma/=2

    # setup_seed(42)
    p = P(X, U, Y, UnLabelInd, k, alpha, gamma)
    cg_stats = CGStats()
    inner_iter = []
    time = t.time()
    if gamma is None:
        p.set_S(generate_graph(X, U, p=k, graph_type='nie'))
        max_iter = 1
    else:
        p.init_XU()
    for iter_time in range(max_iter):
        if gamma is not None:
            p.update_S(iter_time)
        cg_descent(p, cg_stats = cg_stats,
                   work = p.work,
                )
        if cg_stats.iter == 0:
            break
        inner_iter.append(cg_stats.iter)
    time = t.time() - time
    print()
    print('Iter_time:', iter_time, ', Inner_iter:', inner_iter, ', Time:', cg_stats.iter, time)
    Z = p.Z
    H = Z@p.A2+Y
    
    return H, time, p.S
       