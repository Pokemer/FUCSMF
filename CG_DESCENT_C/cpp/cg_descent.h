#include <cmath>
#include <climits>
#include <cfloat>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <functional>

#include <pybind11/functional.h>

namespace cg_modern {
    constexpr double ZERO = 0.0;
    constexpr double ONE = 1.0;

    template<typename T>
    constexpr T max(const T& a, const T& b) noexcept {
        return (a > b) ? a : b;
    }

    template<typename T>
    constexpr T min(const T& a, const T& b) noexcept {
        return (a < b) ? a : b;
    }
}

#define PRIVATE static
#define ZERO ((double) 0)
#define ONE ((double) 1)
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

typedef struct cg_com_struct
{

    INT              n ;
    INT             nf ;
    INT             ng ;
    int         QuadOK ;
    int       UseCubic ;
    int           neps ;
    int       PertRule ;

    int          QuadF ;
    double   SmallCost ;
    double       alpha ;
    double           f ;
    double          df ;
    double       fpert ;
    double         eps ;
    double         tol ;
    double          f0 ;
    double         df0 ;
    double          Ck ;

    double    wolfe_hi ;
    double    wolfe_lo ;
    double   awolfe_hi ;
    int         AWolfe ;

    int          Wolfe ;
    double         rho ;
    double    alphaold ;
    double          *x ;
    double      *xtemp ;
    double          *d ;
    double          *g ;
    double      *gtemp ;
    std::function<double(double*, INT)>  cg_value;
    std::function<void(double*, double*, INT)>  cg_grad;
    std::function<double(double*, double*, INT)>  cg_valgrad;
    cg_parameter *Parm ;
} cg_com ;

PRIVATE int cg_Wolfe
(
    double   alpha,
    double       f,
    double    dphi,
    cg_com    *Com
) ;

PRIVATE int cg_tol
(
    double     gnorm,
    cg_com    *Com
) ;

PRIVATE int cg_line
(
    cg_com   *Com
) ;

PRIVATE int cg_contract
(
    double    *A,
    double   *fA,
    double   *dA,
    double    *B,
    double   *fB,
    double   *dB,
    cg_com  *Com
) ;

PRIVATE int cg_evaluate
(
    char    *what,
    char     *nan,
    cg_com   *Com
) ;

PRIVATE double cg_cubic
(
    double  a,
    double fa,
    double da,
    double  b,
    double fb,
    double db
) ;

PRIVATE void cg_matvec
(
    double *y,
    double *A,
    double *x,
    int     n,
    INT     m,
    int     w
) ;

PRIVATE void cg_trisolve
(
    double *x,
    double *R,
    int     m,
    int     n,
    int     w
) ;

PRIVATE double cg_inf
(
    double *x,
    INT     n
) ;

PRIVATE void cg_scale0
(
    double *y,
    double *x,
    double  s,
    int     n
) ;

PRIVATE void cg_scale
(
    double *y,
    double *x,
    double  s,
    INT     n
) ;

PRIVATE void cg_daxpy0
(
    double     *x,
    double     *d,
    double  alpha,
    int         n
) ;

PRIVATE void cg_daxpy
(
    double     *x,
    double     *d,
    double  alpha,
    INT         n
) ;

PRIVATE double cg_dot0
(
    double *x,
    double *y,
    int     n
) ;

PRIVATE double cg_dot
(
    double *x,
    double *y,
    INT     n
) ;

PRIVATE void cg_copy0
(
    double *y,
    double *x,
    int     n
) ;

PRIVATE void cg_copy
(
    double *y,
    double *x,
    INT     n
) ;

PRIVATE void cg_step
(
    double *xtemp,
    double     *x,
    double     *d,
    double  alpha,
    INT         n
) ;

PRIVATE void cg_init
(
    double *x,
    double  s,
    INT     n
) ;

PRIVATE double cg_update_2
(
    double *gold,
    double *gnew,
    double    *d,
    INT        n
) ;

PRIVATE double cg_update_inf
(
    double *gold,
    double *gnew,
    double    *d,
    INT        n
) ;

PRIVATE double cg_update_ykyk
(
    double *gold,
    double *gnew,
    double *Ykyk,
    double *Ykgk,
    INT        n
) ;

PRIVATE double cg_update_inf2
(
    double   *gold,
    double   *gnew,
    double      *d,
    double *gnorm2,
    INT          n
) ;

PRIVATE double cg_update_d
(
    double      *d,
    double      *g,
    double    beta,
    double *gnorm2,
    INT          n
) ;

PRIVATE void cg_Yk
(
    double    *y,
    double *gold,
    double *gnew,
    double  *yty,
    INT        n
) ;

PRIVATE void cg_printParms
(
    cg_parameter  *Parm
) ;
