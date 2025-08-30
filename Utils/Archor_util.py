import torch
import numpy as np
# import numba as nb
import math
try:
    from .util import *
except:
    from util import *




import os
os.environ["OMP_NUM_THREADS"] = '1'



def generate_Anchor(X, mArch, mode = 'sklearn', seed = None):
    if seed is not None:
        setup_seed(seed)
    match mode.lower():
        case 'sklearn'  | 'sk' | 'scikit-learn' | 'sl':
            # from sklearnex import patch_sklearn
            # patch_sklearn()
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=mArch, max_iter=10)
            km.fit(np.asarray(X))
            U = km.cluster_centers_
            U = torch.as_tensor(U)
        case 'fast_torch_kmeans' | 'ftk' | 'torch2':
            Warning('There is somethoing wrong with fast_torch_kmeans that it will generate a lot of points with only zeros.')
            from fast_pytorch_kmeans import KMeans
            km = KMeans(n_clusters=mArch,  mode = 'euclidean', max_iter=10)
            km.fit(torch.as_tensor(X))
            U = km.centroids
        case 'torch1':
            from .k_means import cluster
            U = cluster(X, mArch)[0]
        case 'BKHK':
            U = BKHK(X, mArch, mode = 'torch')
        case 'BKHK_nb':
            U = BKHK(X, mArch, mode = 'nb')
        case 'litekmeans':
            from .submodel.litekmeans import litekmeans
            U = litekmeans(X, mArch, random_state = seed)
        case 'my' | 'my_kmeans' | 'mykmeans':
            from .submodel.litekmeans import my_kmeans
            U = my_kmeans(X, mArch, random_state = seed)
    
    return U



def BalancedKM(X, ratio = 0.5):
    def TransformL(y, nclass, type=1):
        n = len(y)
        c = nclass
        if type == 1:
            Y = torch.eye(c)[y.long()]
        elif type == 2:
            Y = torch.eye(c)[y.long()]
            Y[Y==0] = -1
        elif type == 3:
            Y = torch.zeros((n, c))
            Y[:, 0] = 1
        else :
            raise('No other type')
        return Y

    class_num = 2
    nSmp = X.shape[1]
    eps = torch.as_tensor(np.spacing(0))
    StartInd = torch.randint(0, class_num, (nSmp,))
    InitF = TransformL(StartInd, class_num, 1)
    if ratio > 0.5:
        raise ValueError('ratio should not larger than 0.5')
    elif ratio < 0:
        ratio = 0

    a = math.floor(nSmp*ratio)
    b = math.floor(nSmp*(1-ratio))

    F = InitF
    for iterTime in range(10):
        C = X@F/(F+eps).sum(0)
        Q = torch.cdist(X.T, C.T)**2
        q = Q[:, 0] - Q[:, 1]
        idx = q.argsort()
        nn = (q<0).sum()
        cp = nn if (nn > a and nn <= b) else a if nn < a else b
        cp = 1 if cp < 1 else nSmp-1

        F = torch.zeros_like(F)
        F[idx[:cp], 0] = 1
        F[:, 1] = 1-F[:, 0]
    y = F.argmax(1)
    return C, Q, y


def hKM(X, idx0, k, count):
    # n = X.shape[0]

    X0 = X[:, idx0]
    if k == 1:
        centers, _, y = BalancedKM(X0, 0.0)
    else:
        centers, _, y = BalancedKM(X0, 0.5)
    ys = 2*count-1-y 
    if k > 1:
        id1 = torch.where(y==0)
        idx1 = idx0[id1]
        ys1, centers1 = hKM(X, idx1, k-1, 2*count)

        id2 = torch.where(y==1)
        idx2 = idx0[id2]
        ys2, centers2 = hKM(X, idx2, k-1, 2*count+1)

        ys[id1] = ys1
        ys[id2] = ys2
        centers = torch.hstack((centers1, centers2))
    return ys, centers

# @nb.njit
def _BalancedKM_nb(X, ratio = 0.5):

    def TransformL(y, nclass, type=1):
        n = len(y)
        c = nclass
        # class_set = torch.arange(c)

        if type == 1:
            Y = np.eye(c)[y]
        elif type == 2:
            Y = np.eye(c)[y]
            Y[Y==0] = -1
        elif type == 3:
            Y = np.zeros((n, c))
            Y[:, 0] = 1
        else :
            # raise('No other type')
            pass
        return Y
    def cdist2(X, Y):
        nSmp = X.shape[0]
        mArch = Y.shape[0]

        aa = np.square(X).sum(1)[:,np.newaxis]
        bb = np.sum(Y**2, 1).astype(np.float64)
        ab = (X@Y.T).astype(np.float64)

        D = aa+bb-2*ab
        # D[D<0] = 0
        return D


    class_num = 2
    nSmp = X.shape[1]
    eps = np.spacing(0)

    StartInd = np.random.randint(0, class_num, (nSmp,))
    # StartInd = torch.as_tensor(io.loadmat('./matlab/2022-TNNLS-FSSC-main/StartInd.mat')['StartInd'].reshape(-1)-1)
    InitF = TransformL(StartInd, class_num, 1)
    if ratio > 0.5:
        raise ValueError('ratio should not larger than 0.5')
    elif ratio < 0:
        ratio = 0

    a = math.floor(nSmp*ratio)
    b = math.floor(nSmp*(1-ratio))

    F = InitF
    for iterTime in range(10):
        C = X@F/(F+eps).sum(0)
        # F = torch.empty_like(F)
        
        Q = cdist2(X.T, C.T)
        q = Q[:, 0] - Q[:, 1]
        idx = q.argsort()
        nn = (q<0).sum()
        if nn > a and nn <= b:
            cp = nn
        elif nn <a:
            cp = a
        else:
            cp = b
        
        if cp < 1:
            cp = 1
        elif cp > nSmp-1:
            cp = nSmp-1

        F = np.zeros_like(F)
        F[idx[:cp], 0] = 1
        F[:, 1] = 1-F[:, 0]
    y = F.argmax(1)
    return C, Q, y

# @nb.njit
def _hKM_nb(X, idx0, k, count):

    X0 = X[:, idx0]
    if k == 1:
        centers, _, y = _BalancedKM_nb(X0, 0.0)
    else:
        centers, _, y = _BalancedKM_nb(X0, 0.5)
    ys = 2*count-1-y 
    if k > 1:
        id1 = y==0
        idx1 = idx0[id1]
        ys1, centers1 = _hKM_nb(X, idx1, k-1, 2*count)

        id2 = y==1
        idx2 = idx0[id2]
        ys2, centers2 = _hKM_nb(X, idx2, k-1, 2*count+1)

        ys[id1] = ys1
        ys[id2] = ys2
        centers = np.hstack((centers1, centers2))
    return ys, centers


def BKHK(X, k, mode = 'nb'):
    X = X.T
    n = X.shape[1]
    count = 1
    match mode.lower():
        case 'torch':
            idx0 = torch.arange(n)
            ys, centers = hKM(X, idx0, k, count)
        case 'nb' | 'numba':
            idx0 = np.arange(n)
            ys, centers = _hKM_nb(X.numpy(), idx0, k, count)
            centers = torch.from_numpy(centers)
    return centers.T