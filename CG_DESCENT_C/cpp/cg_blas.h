

#ifndef CG_BLAS_H
#define CG_BLAS_H

#include "mkl.h"
#include "mkl_cblas.h"

#ifdef MKL_ILP64
    #define BLAS_INT MKL_INT
#else
    #define BLAS_INT int
#endif

#define DDOT_START 100
#define DCOPY_START 100
#define DAXPY_START 6000
#define DSCAL_START 6000
#define IDAMAX_START 25
#define MATVEC_START 8000

#define CG_DGEMV cblas_dgemv
#define CG_DTRSV cblas_dtrsv
#define CG_DAXPY cblas_daxpy
#define CG_DDOT cblas_ddot
#define CG_DSCAL cblas_dscal
#define CG_DCOPY cblas_dcopy
#define CG_DNRM2 cblas_dnrm2

static inline BLAS_INT cg_idamax_wrapper(const BLAS_INT N, const double *X, const BLAS_INT incX) {
    return static_cast<BLAS_INT>(cblas_idamax(N, X, incX));
}

#define CG_IDAMAX cg_idamax_wrapper

#endif 
