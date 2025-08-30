#include "cg_user.h"
#include "cg_descent.h"
#include "cg_blas.h"
#include <omp.h>

double one [1], zero [1] ;
BLAS_INT blas_one [1] ;

int cg_descent

(
    double            *x,
    INT                n,
    cg_stats       *Stat,
    cg_parameter  *UParm,
    double      grad_tol,

    std::function<double(double*, INT)> value,
    std::function<void(double*, double*, INT)> grad,
    std::function<double(double*, double*, INT)> valgrad,

    double         *Work

)
{
    INT     i, iter, IterRestart, maxit, n5, nrestart, nrestartsub ;
    int     nslow, slowlimit, IterQuad, status, PrintLevel, QuadF, StopRule ;
    double  delta2, Qk, Ck, fbest, gbest,
            f, ftemp, gnorm, xnorm, gnorm2, dnorm2, denom,
            t, dphi, dphi0, alpha,
            ykyk, ykgk, dkyk, beta, QuadTrust, tol,
           *d, *g, *xtemp, *gtemp, *work ;

    int     l1, l2, j, k, mem, memsq, memk, memk_begin, mlast, mlast_sub,
            mp, mp_begin, mpp, nsub, spp, spp1, SkFstart, SkFlast, Subspace,
            UseMemory, Restart, LBFGS, InvariantSpace, IterSub, NumSub,
            IterSubStart, IterSubRestart, FirstFull, SubSkip, SubCheck,
            StartSkip, StartCheck, DenseCol1, NegDiag, memk_is_mem,
           d0isg, qrestart ;
    double  gHg, scale, gsubnorm2,  ratio, stgkeep,
            alphaold, zeta, yty, ytg, t1, t2, t3, t4,
           *Rk, *Re, *Sk, *SkF, *stemp, *Yk, *SkYk,
           *dsub, *gsub, *gsubtemp, *gkeep, *tau, *vsub, *wsub ;

    cg_parameter *Parm, ParmStruc ;
    cg_com Com ;

    one [0] = (double) 1 ;
    zero [0] = (double) 0 ;
    blas_one [0] = (BLAS_INT) 1 ;

    if ( UParm == NULL )
    {
        Parm = &ParmStruc ;
        cg_default (Parm) ;
    }
    else Parm = UParm ;
    PrintLevel = Parm->PrintLevel ;
    qrestart = MIN (n, Parm->qrestart) ;
    Com.Parm = Parm ;
    Com.eps = Parm->eps ;
    Com.PertRule = Parm->PertRule ;
    Com.Wolfe = FALSE ;
    Com.nf = (INT) 0 ;
    Com.ng = (INT) 0 ;
    iter = (INT) 0 ;
    QuadF = FALSE ;
    NegDiag = FALSE ;
    mem = Parm->memory ;

    if ( Parm->PrintParms ) cg_printParms (Parm) ;
    if ( (mem != 0) && (mem < 3) )
    {
        status = 12 ;
        goto Exit ;
    }

    mem = MIN (mem, n) ;
    if ( Work == NULL )
    {
        if ( mem == 0 )
        {
            work = (double *) malloc (4*n*sizeof (double)) ;
        }
        else if ( Parm->LBFGS || (mem >= n) )
        {
            work = (double *) malloc ((2*mem*(n+1)+4*n)*sizeof (double)) ;
        }
        else
        {
            i = (mem+6)*n + (3*mem+9)*mem + 5 ;
            work = (double *) malloc (i*sizeof (double)) ;
        }
    }
    else work = Work ;
    if ( work == NULL )
    {
        status = 10 ;
        goto Exit ;
    }

    Com.x = x ;
    Com.xtemp = xtemp = work ;
    Com.d = d = xtemp+n ;
    Com.g = g = d+n ;
    Com.gtemp = gtemp = g+n ;
    Com.n = n ;
    Com.neps = 0 ;
    Com.AWolfe = Parm->AWolfe ;
    Com.cg_value = value ;
    Com.cg_grad = grad ;
    Com.cg_valgrad = valgrad ;
    StopRule = Parm->StopRule ;
    LBFGS = FALSE ;
    UseMemory = FALSE ;
    Subspace = FALSE ;
    FirstFull = FALSE ;
    memk = 0 ;

    nrestart = (INT) (((double) n)*Parm->restart_fac) ;

    if ( mem > 0 )
    {
        if ( (mem == n) || Parm->LBFGS )
        {
            LBFGS = TRUE ;
            mlast = -1 ;
            Sk = gtemp + n ;
            Yk = Sk + mem*n ;
            SkYk = Yk + mem*n ;
            tau = SkYk + mem ;
        }
        else
        {
            UseMemory = TRUE ;
            SubSkip = 0 ;
            SubCheck = mem*Parm->SubCheck ;
            StartCheck = 0 ;
            InvariantSpace = FALSE ;
            FirstFull = TRUE ;
            nsub = 0 ;
            memsq = mem*mem ;
            SkF = gtemp+n ;
            stemp = SkF + mem*n ;
            gkeep = stemp + n ;
            Sk = gkeep + n ;
            Rk = Sk + memsq ;

            cg_init (Rk, ZERO, memsq) ;
            Re = Rk + memsq ;
            Yk = Re + mem+1 ;
            SkYk = Yk + memsq+mem+2 ;
            tau = SkYk + mem ;
            dsub = tau + mem ;
            gsub = dsub + mem ;
            gsubtemp = gsub + mem+1 ;
            wsub = gsubtemp + mem ;
            vsub = wsub + mem+1 ;
        }
    }

    maxit = Parm->maxit ;

    f = ZERO ;
    fbest = INF ;
    gbest = INF ;
    nslow = 0 ;
    slowlimit = 2*n + Parm->nslow ;
    n5 = n % 5 ;

    Ck = ZERO ;
    Qk = ZERO ;

    Com.alpha = ZERO ;
    status = cg_evaluate ("fg", "n", &Com) ;
    f = Com.f ;
    if ( status )
    {
        if ( PrintLevel > 0 ) printf ("Function undefined at starting point\n");
        goto Exit ;
    }

    Com.f0 = f + f ;
    Com.SmallCost = fabs (f)*Parm->SmallCost ;
    xnorm = cg_inf (x, n) ;

    gnorm = cg_update_inf2 (g, g, d, &gnorm2, n) ;
    dnorm2 = gnorm2 ;

    if ( f != f )
    {
        status = 11 ;
        goto Exit ;
    }

    if ( Parm->StopRule ) tol = MAX (gnorm*Parm->StopFac, grad_tol) ;
    else                  tol = grad_tol ;
    Com.tol = tol ;

    if ( PrintLevel >= 1 )
    {
        printf ("iter: %5i f: %13.6e gnorm: %13.6e memk: %i\n",
        (int) 0, f, gnorm, memk) ;
    }

    if ( cg_tol (gnorm, &Com) )
    {
        iter = 0 ;
        status = 0 ;
        goto Exit ;
    }

    dphi0 = -gnorm2 ;
    delta2 = 2*Parm->delta - ONE ;
    alpha = Parm->step ;
    if ( alpha == ZERO )
    {
        if ( xnorm == ZERO )
        {
            if ( f != ZERO ) alpha = 2.*fabs (f)/gnorm2 ;
            else             alpha = ONE ;
        }
        else    alpha = Parm->psi0*xnorm/gnorm ;
    }

    Com.df0 = -2.0*fabs(f)/alpha ;

    Restart = FALSE ;
    IterRestart = 0 ;
    IterSub = 0 ;
    NumSub =  0 ;
    IterQuad = 0 ;

    scale = (double) 1 ;

    for (iter = 1; iter <= maxit; iter++)
    {

        alphaold = alpha ;
        Com.QuadOK = FALSE ;
        alpha = Parm->psi2*alpha ;
        if ( f != ZERO ) t = fabs ((f-Com.f0)/f) ;
        else             t = ONE ;
        Com.UseCubic = TRUE ;
        if ( (t < Parm->CubicCutOff) || !Parm->UseCubic ) Com.UseCubic = FALSE ;
        if ( Parm->QuadStep )
        {

            if ( ((t > Parm->QuadCutOff)&&(fabs(f) >= Com.SmallCost)) || QuadF )
            {
                if ( QuadF )
                {
                    Com.alpha = Parm->psi1*alpha ;
                    status = cg_evaluate ("g", "y", &Com) ;
                    if ( status ) goto Exit ;
                    if ( Com.df > dphi0 )
                    {
                        alpha = -dphi0/((Com.df-dphi0)/Com.alpha) ;
                        Com.QuadOK = TRUE ;
                    }
                    else if ( LBFGS )
                    {
                        if ( memk >= n )
                        {
                            alpha = ONE ;
                            Com.QuadOK = TRUE ;
                        }
                        else  alpha = 2. ;
                    }
                    else if ( Subspace )
                    {
                        if ( memk >= nsub )
                        {
                            alpha = ONE ;
                            Com.QuadOK = TRUE ;
                        }
                        else  alpha = 2. ;
                    }
                }
                else
                {
                    t = MAX (Parm->psi_lo, Com.df0/(dphi0*Parm->psi2)) ;
                    Com.alpha = MIN (t, Parm->psi_hi)*alpha ;
                    status = cg_evaluate ("f", "y", &Com) ;
                    if ( status ) goto Exit ;
                    ftemp = Com.f ;
                    denom = 2.*(((ftemp-f)/Com.alpha)-dphi0) ;
                    if ( denom > ZERO )
                    {
                        t = -dphi0*Com.alpha/denom ;

                        if ( ftemp >= f )
                              alpha = MAX (t, Com.alpha*Parm->QuadSafe) ;
                        else  alpha = t ;
                        Com.QuadOK = TRUE ;
                    }
                }
                if ( PrintLevel >= 1 )
                {
                    if ( denom <= ZERO )
                    {
                        printf ("Quad step fails (denom = %14.6e)\n", denom);
                    }
                    else if ( Com.QuadOK )
                    {
                        printf ("Quad step %14.6e OK\n", alpha);
                    }
                    else printf ("Quad step %14.6e done, but not OK\n", alpha) ;
                }
            }
            else if ( PrintLevel >= 1 )
            {
                printf ("No quad step (chg: %14.6e, cut: %10.2e)\n",
                         t, Parm->QuadCutOff) ;
            }
        }
        Com.f0 = f ;
        Com.df0 = dphi0 ;

        Qk = Parm->Qdecay*Qk + ONE ;
        Ck = Ck + (fabs (f) - Ck)/Qk ;

        if ( Com.PertRule ) Com.fpert = f + Com.eps*fabs (f) ;
        else                Com.fpert = f + Com.eps ;

        Com.wolfe_hi = Parm->delta*dphi0 ;
        Com.wolfe_lo = Parm->sigma*dphi0 ;
        Com.awolfe_hi = delta2*dphi0 ;
        Com.alpha = alpha ;

        status = cg_line (&Com) ;

        if ( (status > 0) && !Com.AWolfe )
        {
            if ( PrintLevel >= 1 )
            {
                 printf ("\nWOLFE LINE SEARCH FAILS\n") ;
            }
            if ( status != 3 )
            {
                Com.AWolfe = TRUE ;
                status = cg_line (&Com) ;
            }
        }

        alpha = Com.alpha ;
        f = Com.f ;
        dphi = Com.df ;

        if ( status ) goto Exit ;

        if ( -alpha*dphi0 <= Parm->feps*fabs (f) )
        {
            status = 1 ;
            goto Exit ;
        }

        t = alpha*(dphi+dphi0) ;
        if ( fabs (t) <= Parm->qeps*MIN (Ck, ONE) ) QuadTrust = ZERO ;
        else QuadTrust = fabs((2.0*(f-Com.f0)/t)-ONE) ;
        if ( QuadTrust <= Parm->qrule) IterQuad++ ;
        else                           IterQuad = 0 ;

        if ( IterQuad == qrestart ) QuadF = TRUE ;
        IterRestart++ ;
        if ( !Com.AWolfe )
        {
            if ( fabs (f-Com.f0) < Parm->AWolfeFac*Ck )
            {
                Com.AWolfe = TRUE ;
                if ( Com.Wolfe ) Restart = TRUE ;
            }
        }

        if ( (mem > 0) && !LBFGS )
        {
            if ( UseMemory )
            {
                if ( (iter - StartCheck > SubCheck) && !Subspace )
                {
                    StartSkip = iter ;
                    UseMemory = FALSE ;
                    if ( SubSkip == 0 ) SubSkip = mem*Parm->SubSkip ;
                    else                SubSkip *= 2 ;
                    if ( PrintLevel >= 1 )
                    {
                        printf ("skip subspace %i iterations\n", SubSkip) ;
                    }
                }
            }
            else
            {
                if ( iter - StartSkip > SubSkip )
                {
                    StartCheck = iter ;
                    UseMemory = TRUE ;
                    memk = 0 ;
                }
            }
        }

        if ( !UseMemory )
        {
            if ( !LBFGS )
            {
                if ( (IterRestart >= nrestart) || ((IterQuad == qrestart)
                     && (IterQuad != IterRestart)) ) Restart = TRUE ;
            }
        }
        else
        {
            if ( Subspace )
            {
                IterSubRestart++ ;

                gsubnorm2 = ZERO ;
                mp = SkFstart ;
                j = nsub - mp ;

                cg_matvec (wsub, SkF, gtemp, nsub, n, 0) ;

                cg_copy0 (gsubtemp, wsub+mp, j) ;
                cg_copy0 (gsubtemp+j, wsub, mp) ;

                cg_trisolve (gsubtemp, Rk, mem, nsub, 0) ;
                gsubnorm2 = cg_dot0 (gsubtemp, gsubtemp, nsub) ;
                gnorm2 = cg_dot (gtemp, gtemp, n);
                ratio = sqrt(gsubnorm2/gnorm2) ;
                if ( ratio < ONE - Parm->eta1  )
                {
                   if ( PrintLevel >= 1 )
                   {
                       printf ("iter: %i exit subspace\n", (int) iter) ;
                   }
                   FirstFull = TRUE ;
                   Subspace = FALSE ;
                   InvariantSpace = FALSE ;

                   StartCheck = iter ;
                   if ( IterSubRestart > 1 ) dnorm2 = cg_dot0 (dsub, dsub,nsub);
                }
                else
                {

                   if ( IterSubRestart == nrestartsub ) Restart = TRUE ;
                }
            }
            else
            {
                if ( (IterRestart == 1) || FirstFull ) memk = 0 ;
                if ( (memk == 1) && InvariantSpace )
                {
                     memk = 0 ;
                     InvariantSpace = FALSE ;
                }
                if (memk < mem )
                {
                    memk_is_mem = FALSE ;
                    SkFstart = 0 ;

                    if (memk == 0)
                    {
                        mlast = 0 ;
                        memk = 1 ;

                        t = sqrt(dnorm2) ;
                        zeta = alpha*t ;
                        Rk [0] = zeta ;
                        cg_scale (SkF, d, alpha, n) ;
                        Yk [0] = (dphi - dphi0)/t ;
                        gsub [0] = dphi/t ;
                        SkYk [0] = alpha*(dphi-dphi0) ;
                        FirstFull = FALSE ;
                        if ( IterRestart > 1 )
                        {

                           cg_copy (gkeep, g, n) ;

                           stgkeep = dphi0*alpha ;
                           d0isg = FALSE ;
                        }
                        else d0isg = TRUE ;
                    }
                    else
                    {
                        mlast = memk ;
                        memk++ ;
                        mpp = mlast*n ;
                        spp = mlast*mem ;
                        cg_scale (SkF+mpp, d, alpha, n) ;

                        if ((fabs(alpha-5.05)>4.95)||(fabs(alphaold-5.05)>4.95))
                        {

                            cg_matvec (Rk+spp, SkF, SkF+mpp, mlast, n, 0) ;

                            cg_trisolve (Rk+spp, Rk, mem, mlast, 0) ;
                        }
                        else
                        {
                            t1 = -alpha ;
                            t2 = beta*alpha/alphaold ;
                            for (j = 0; j < mlast; j++)
                            {
                                Rk [spp+j] = t1*gsub [j] + t2*Rk [spp-mem+j] ;
                            }
                        }
                        t = alpha*alpha*dnorm2 ;
                        t1 = cg_dot0 (Rk+spp, Rk+spp, mlast) ;
                        if (t <= t1)
                        {
                            zeta = t*1.e-12 ;
                            NegDiag = TRUE ;
                        }
                        else zeta = sqrt(t-t1);

                        Rk [spp+mlast] = zeta ;
                        t = - zeta/alpha ;
                        Yk [spp-mem+mlast] = t ;
                        gsub [mlast] = t ;

                        cg_matvec (wsub, SkF, gtemp, mlast, n, 0) ;

                        wsub [mlast] = alpha*dphi ;

                        cg_trisolve (wsub, Rk, mem, memk, 0) ;

                        cg_Yk (Yk+spp, gsub, wsub, NULL, memk) ;

                        SkYk [mlast] = alpha*(dphi-dphi0) ;
                    }
                }
                else
                {
                    memk_is_mem = TRUE ;
                    mlast = mem-1 ;
                    cg_scale (stemp, d, alpha, n) ;

                    if ((fabs(alpha-5.05)>4.95)||(fabs(alphaold-5.05)>4.95))
                    {
                        mp = SkFstart ;
                        j = mem - mp ;

                        cg_matvec (wsub, SkF, stemp, mem, n, 0) ;

                        cg_copy0 (Re, wsub+mp, j) ;
                        cg_copy0 (Re+j, wsub, mp) ;

                        cg_trisolve (Re, Rk, mem, mem, 0) ;
                    }
                    else
                    {
                        t1 = -alpha ;
                        t2 = beta*alpha/alphaold ;
                        for (j = 0; j < mem; j++)
                        {
                            Re [j] = t1*gsub [j] + t2*Re [j-mem] ;
                        }
                    }

                    t = alpha*alpha*dnorm2 ;

                    t1 = cg_dot0 (Re, Re, mem) ;
                    if (t <= t1)
                    {
                        zeta = t*1.e-12 ;
                        NegDiag = TRUE ;
                    }
                    else zeta = sqrt(t-t1);

                    Re [mem] = zeta ;

                    t = -zeta/alpha ;
                    gsub [mem] = t ;
                    Yk [memsq] = t ;

                    spp = memsq + 1 ;
                    mp = SkFstart ;
                    j = mem - mp ;

                    cg_matvec (vsub, SkF, gtemp, mem, n, 0) ;

                    cg_copy0 (wsub, vsub+mp, j) ;
                    cg_copy0 (wsub+j, vsub, mp) ;

                    cg_trisolve (wsub, Rk, mem, mem, 0) ;
                    wsub [mem] = (alpha*dphi - cg_dot0 (wsub, Re, mem))/zeta;

                    cg_Yk (Yk+spp, gsub, wsub, NULL, mem+1) ;

                    cg_copy (SkF+SkFstart*n, stemp, n) ;
                    SkFstart++ ;
                    if ( SkFstart == mem ) SkFstart = 0 ;

                    mp = SkFstart ;
                    for (k = 0; k < mem; k++)
                    {
                        spp = (k+1)*mem + k ;
                        t1 = Rk [spp] ;
                        t2 = Rk [spp+1] ;
                        t = sqrt(t1*t1 + t2*t2) ;
                        t1 = t1/t ;
                        t2 = t2/t ;

                        Rk [k*mem+k] = t ;
                        for (j = (k+2); j <= mem; j++)
                        {
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Rk [spp] ;
                            t4 = Rk [spp+1] ;
                            Rk [spp1] = t1*t3 + t2*t4 ;
                            Rk [spp+1] = t1*t4 - t2*t3 ;
                        }

                        if ( k < 2 )
                        {

                            spp = k ;
                            for (j = 1; j < mem; j++)
                            {
                                spp1 = spp ;
                                spp = j*mem + k ;
                                t3 = Yk [spp] ;
                                t4 = Yk [spp+1] ;
                                Yk [spp1] = t1*t3 + t2*t4 ;
                                Yk [spp+1] = t1*t4 -t2*t3 ;
                            }
                            spp1 = spp ;
                            spp = mem*mem + 1 + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                            Yk [spp+1] = t1*t4 -t2*t3 ;
                        }
                        else if ( (k == 2) && (2 < mem-1))
                        {
                            spp = k ;

                            j = 1 ;
                            spp1 = spp ;
                            spp = j*mem + k ;

                            t3 = Yk [spp] ;
                            Yk [spp1] = t1*t3 ;
                            Yk [spp+1] = -t2*t3 ;

                            for (j = 2; j < mem; j++)
                            {
                                spp1 = spp ;
                                spp = j*mem + k ;
                                t3 = Yk [spp] ;
                                t4 = Yk [spp+1] ;
                                Yk [spp1] = t1*t3 + t2*t4 ;
                                Yk [spp+1] = t1*t4 -t2*t3 ;
                            }
                            spp1 = spp ;
                            spp = mem*mem + 1 + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                            Yk [spp+1] = t1*t4 -t2*t3 ;
                        }
                        else if ( k < (mem-1) )
                        {
                            spp = k ;

                            j = 1 ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ;
                            Yk [spp1] = t1*t3 ;
                            Yk [spp+1] = -t2*t3 ;

                            j = k-1 ;
                            spp = (j-1)*mem+k ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ;
                            Yk [spp1] = t1*t3 ;

                            for (j = k; j < mem; j++)
                            {
                                spp1 = spp ;
                                spp = j*mem + k ;
                                t3 = Yk [spp] ;
                                t4 = Yk [spp+1] ;
                                Yk [spp1] = t1*t3 + t2*t4 ;
                                Yk [spp+1] = t1*t4 -t2*t3 ;
                            }
                            spp1 = spp ;
                            spp = mem*mem + 1 + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                            Yk [spp+1] = t1*t4 -t2*t3 ;
                        }
                        else
                        {
                            spp = k ;

                            j = 1 ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ;
                            Yk [spp1] = t1*t3 ;

                            j = k-1 ;
                            spp = (j-1)*mem+k ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ;
                            Yk [spp1] = t1*t3 ;

                            j = k ;
                            spp1 = spp ;
                            spp = j*mem + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;

                            spp1 = spp ;
                            spp = mem*mem + 1 + k ;
                            t3 = Yk [spp] ;
                            t4 = Yk [spp+1] ;
                            Yk [spp1] = t1*t3 + t2*t4 ;
                        }

                        if ( k < (mem-1) )
                        {
                            t3 = gsub [k] ;
                            t4 = gsub [k+1] ;
                            gsub [k] = t1*t3 + t2*t4 ;
                            gsub [k+1] = t1*t4 -t2*t3 ;
                        }
                        else
                        {
                            t3 = gsub [k] ;
                            t4 = gsub [k+1] ;
                            gsub [k] = t1*t3 + t2*t4 ;
                        }
                    }

                    for (k = 0; k < mlast; k++) SkYk [k] = SkYk [k+1] ;
                    SkYk [mlast] = alpha*(dphi-dphi0) ;
                }

                gsubnorm2 = cg_dot0 (gsub, gsub, memk) ;
                gnorm2 = cg_dot (gtemp, gtemp, n) ;
                ratio = sqrt (gsubnorm2/gnorm2) ;
                if ( ratio > ONE-Parm->eta2) InvariantSpace = TRUE ;

                if ( ((memk > 1) && InvariantSpace) ||
                     ((memk == mem) && (ratio > ONE-Parm->eta0)) )
                {
                    NumSub++ ;
                    if ( PrintLevel >= 1 )
                    {
                        if ( InvariantSpace )
                        {
                            printf ("iter: %i invariant space, "
                                    "enter subspace\n", (int) iter) ;
                        }
                        else
                        {
                            printf ("iter: %i enter subspace\n", (int) iter) ;
                        }
                    }

                    if ( !d0isg && !memk_is_mem )
                    {
                        wsub [0] = stgkeep ;

                        cg_matvec (wsub+1, SkF+n, gkeep, mlast, n, 0) ;

                        cg_trisolve (wsub, Rk, mem, memk, 0) ;

                        Yk [1] -= wsub [1] ;
                        cg_scale0 (Yk+2, wsub+2, -ONE, memk-2) ;
                    }
                    if ( d0isg && !memk_is_mem ) DenseCol1 = FALSE ;
                    else                         DenseCol1 = TRUE ;

                    Subspace = TRUE ;

                    SubSkip = 0 ;
                    IterSubRestart = 0 ;
                    IterSubStart = IterSub ;
                    nsub = memk ;
                    nrestartsub = (int) (((double) nsub)*Parm->restart_fac) ;
                    mp_begin = mlast ;
                    memk_begin = nsub ;
                    SkFlast = (SkFstart+nsub-1) % mem ;
                    cg_copy0 (gsubtemp, gsub, nsub) ;

                    cg_copy (Sk, Rk, (int) mem*nsub) ;
                }
                else
                {
                   if ( (IterRestart == nrestart) ||
                       ((IterQuad == qrestart) && (IterQuad != IterRestart)) )
                   {
                       Restart = TRUE ;
                   }
                }
            }
        }

        if ( LBFGS )
        {
            gnorm = cg_inf (gtemp, n) ;
            if ( cg_tol (gnorm, &Com) )
            {
                status = 0 ;
                cg_copy (x, xtemp, n) ;
                goto Exit ;
            }

            if ( IterRestart == nrestart )
            {
                IterRestart = 0 ;
                IterQuad = 0 ;
                mlast = -1 ;
                memk = 0 ;
                scale = (double) 1 ;

                cg_copy (x, xtemp, n) ;

                gnorm2 = cg_update_2 (g, gtemp, d, n) ;

                dnorm2 = gnorm2 ;
                dphi0 = -gnorm2 ;
            }
            else
            {
                mlast = (mlast+1) % mem ;
                spp = mlast*n ;
                cg_step (Sk+spp, xtemp, x, -ONE, n) ;
                cg_step (Yk+spp, gtemp, g, -ONE, n) ;
                SkYk [mlast] = alpha*(dphi-dphi0) ;
                if (memk < mem) memk++ ;

                cg_copy (x, xtemp, n) ;

                gnorm2 = cg_update_2 (g, gtemp, NULL, n) ;

                mp = mlast ;
                for (j = 0; j < memk; j++)
                {
                    mpp = mp*n ;
                    t = cg_dot (Sk+mpp, gtemp, n)/SkYk[mp] ;
                    tau [mp] = t ;
                    cg_daxpy (gtemp, Yk+mpp, -t, n) ;
                    mp -=  1;
                    if ( mp < 0 ) mp = mem-1 ;
                }

                t = cg_dot (Yk+mlast*n, Yk+mlast*n, n) ;
                if ( t > ZERO )
                {
                    scale = SkYk[mlast]/t ;
                }

                cg_scale (gtemp, gtemp, scale, n) ;

                for (j = 0; j < memk; j++)
                {
                    mp +=  1 ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*n ;
                    t = cg_dot (Yk+mpp, gtemp, n)/SkYk[mp] ;
                    cg_daxpy (gtemp, Sk+mpp, tau [mp]-t, n) ;
                }

                dnorm2 = cg_update_2 (NULL, gtemp, d, n) ;
                dphi0 = -cg_dot (g, gtemp, n) ;
            }
        }

        else if ( Subspace )
        {
            IterSub++ ;

            cg_copy (x, xtemp, n) ;

            gnorm = cg_update_inf (g, gtemp, NULL, n) ;

            if ( cg_tol (gnorm, &Com) )
            {
                status = 0 ;
                goto Exit ;
            }

            if ( Restart )
            {
                scale = (double) 1 ;
                Restart = FALSE ;
                IterRestart = 0 ;
                IterSubRestart = 0 ;
                IterQuad = 0 ;
                mp_begin = -1 ;
                memk_begin = 0 ;
                memk = 0 ;

                if ( PrintLevel >= 1 ) printf ("RESTART Sub-CG\n") ;

                cg_scale0 (dsub, gsubtemp, -ONE, nsub) ;
                cg_copy0 (gsub, gsubtemp, nsub) ;
                cg_copy0 (vsub, dsub, nsub) ;
                cg_trisolve (vsub, Rk, mem, nsub, 1) ;

                mp = SkFlast ;
                j = nsub - (mp+1) ;
                cg_copy0 (wsub, vsub+j, mp+1) ;
                cg_copy0 (wsub+(mp+1), vsub, j) ;
                cg_matvec (d, SkF, wsub, nsub, n, 1) ;

                dphi0 = -gsubnorm2 ;
                dnorm2 = gsubnorm2 ;
            }
            else
            {
                mlast_sub = (mp_begin + IterSubRestart) % mem ;

                if (IterSubRestart > 0 )
                {

                    spp = mlast_sub*mem ;
                    cg_scale0 (Sk+spp, dsub, alpha, nsub) ;

                    cg_Yk (Yk+spp, gsub, gsubtemp, &yty, nsub) ;
                    SkYk [mlast_sub] = alpha*(dphi - dphi0) ;
                    if ( yty > ZERO )
                    {
                        scale = SkYk [mlast_sub]/yty ;
                    }
                }
                else
                {
                    yty = cg_dot0 (Yk+mlast_sub*mem, Yk+mlast_sub*mem, nsub) ;
                    if ( yty > ZERO )
                    {
                        scale = SkYk [mlast_sub]/yty ;
                    }
                }

                mp = mlast_sub ;

                memk = MIN (memk_begin + IterSubRestart, mem) ;
                l1 = MIN (IterSubRestart, memk) ;

                l2 = memk - l1 ;

                l1++ ;
                l1 = MIN (l1, memk) ;

                for (j = 0; j < l1; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    tau [mp] = t ;

                    cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    mp-- ;
                    if ( mp < 0 ) mp = mem-1 ;
                }

                for (j = 1; j < l2; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, mp+1)/SkYk[mp] ;
                    tau [mp] = t ;

                    if ( mp == 0 && DenseCol1 )
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    }
                    else
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, MIN(mp+2,nsub)) ;
                    }
                    mp-- ;
                    if ( mp < 0 ) mp = mem-1 ;
                }
                cg_scale0 (gsubtemp, gsubtemp, scale, nsub) ;

                for (j = 1; j < l2; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    if ( mp == 0 && DenseCol1 )
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    }
                    else
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, MIN(mp+2,nsub))/SkYk[mp];
                    }

                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, mp+1) ;
                }

                for (j = 0; j < l1; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk [mp] ;

                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, nsub) ;
                }

                cg_scale0 (dsub, gsubtemp, -ONE, nsub) ;
                cg_copy0 (vsub, dsub, nsub) ;
                cg_trisolve (vsub, Rk, mem, nsub, 1) ;

                mp = SkFlast ;
                j = nsub - (mp+1) ;
                cg_copy0 (wsub, vsub+j, mp+1) ;
                cg_copy0 (wsub+(mp+1), vsub, j) ;

                cg_matvec (d, SkF, wsub, nsub, n, 1) ;
                dphi0 = -cg_dot0  (gsubtemp, gsub, nsub) ;
            }
        }
        else
        {
            if ( Restart )
            {
                Restart = FALSE ;
                IterRestart = 0 ;
                IterQuad = 0 ;
                if ( PrintLevel >= 1 ) printf ("RESTART CG\n") ;

                cg_copy (x, xtemp, n) ;

                if ( UseMemory )
                {

                   gnorm = cg_update_inf (g, gtemp, d, n) ;
                }
                else
                {

                    gnorm = cg_update_inf2 (g, gtemp, d, &gnorm2, n) ;
                }

                if ( cg_tol (gnorm, &Com) )
                {
                   status = 0 ;
                   goto Exit ;
                }
                dphi0 = -gnorm2 ;
                dnorm2 = gnorm2 ;
                beta = ZERO ;
            }
            else if ( !FirstFull )
            {

                cg_copy (x, xtemp, n) ;

                gnorm = cg_update_ykyk (g, gtemp, &ykyk, &ykgk, n) ;

                if ( cg_tol (gnorm, &Com) )
                {
                   status = 0 ;
                   goto Exit ;
                }

                dkyk = dphi - dphi0 ;
                if ( Parm->AdaptiveBeta ) t = 2. - ONE/(0.1*QuadTrust + ONE) ;
                else                      t = Parm->theta ;
                beta = (ykgk - t*dphi*ykyk/dkyk)/dkyk ;

                beta = MAX (beta, Parm->BetaLower*dphi0/dnorm2) ;

                if ( UseMemory )
                {

                    dnorm2 = cg_update_d (d, g, beta, NULL, n) ;
                }
                else
                {

                    dnorm2 = cg_update_d (d, g, beta, &gnorm2, n) ;
                }

                dphi0 = -gnorm2 + beta*dphi ;
                if ( Parm->debug )
                {
                    t = ZERO ;
                    for (i = 0; i < n; i++)  t = t + d [i]*g [i] ;
                    if ( fabs(t-dphi0) > Parm->debugtol*fabs(dphi0) )
                    {
                        printf("Warning, dphi0 != d'g!\n");
                        printf("dphi0:%13.6e, d'g:%13.6e\n",dphi0, t) ;
                    }
                }
            }
            else
            {

                cg_copy (x, xtemp, n) ;

                gnorm = cg_update_ykyk (g, gtemp, &ykyk, &ykgk, n) ;

                if ( cg_tol (gnorm, &Com) )
                {
                   status = 0 ;
                   goto Exit ;
                }

                mlast_sub = (mp_begin + IterSubRestart) % mem ;

                spp = mlast_sub*mem ;
                cg_scale0 (Sk+spp, dsub, alpha, nsub) ;

                cg_Yk (Yk+spp, gsub, gsubtemp, &yty, nsub) ;
                ytg = cg_dot0  (Yk+spp, gsub, nsub) ;
                t = alpha*(dphi - dphi0) ;
                SkYk [mlast_sub] = t ;

                if ( yty > ZERO )
                {
                    scale = t/yty ;
                }

                mp = mlast_sub ;

                memk = MIN (memk_begin + IterSubRestart, mem) ;
                l1 = MIN (IterSubRestart, memk) ;

                l2 = memk - l1 ;

                l1++ ;
                l1 = MIN (l1, memk) ;

                for (j = 0; j < l1; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    tau [mp] = t ;

                    cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    mp-- ;
                    if ( mp < 0 ) mp = mem-1 ;
                }

                for (j = 1; j < l2; j++)
                {
                    mpp = mp*mem ;
                    t = cg_dot0 (Sk+mpp, gsubtemp, mp+1)/SkYk[mp] ;
                    tau [mp] = t ;

                    if ( mp == 0 && DenseCol1 )
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, nsub) ;
                    }
                    else
                    {
                        cg_daxpy0 (gsubtemp, Yk+mpp, -t, MIN(mp+2,nsub)) ;
                    }
                    mp-- ;
                    if ( mp < 0 ) mp = mem-1 ;
                }
                cg_scale0 (gsubtemp, gsubtemp, scale, nsub) ;

                for (j = 1; j < l2; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    if ( mp == 0 && DenseCol1 )
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk[mp] ;
                    }
                    else
                    {
                        t = cg_dot0 (Yk+mpp, gsubtemp, MIN(mp+2,nsub))/SkYk[mp];
                    }

                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, mp+1) ;
                }

                for (j = 0; j < l1; j++)
                {
                    mp++ ;
                    if ( mp == mem ) mp = 0 ;
                    mpp = mp*mem ;
                    t = cg_dot0 (Yk+mpp, gsubtemp, nsub)/SkYk [mp] ;

                    cg_daxpy0 (gsubtemp, Sk+mpp, tau [mp] - t, nsub) ;
                }

                dkyk = dphi - dphi0 ;
                if ( Parm->AdaptiveBeta ) t = 2. - ONE/(0.1*QuadTrust + ONE) ;
                else                      t = Parm->theta ;
                t1 = MAX(ykyk-yty, ZERO) ;
                if ( ykyk > ZERO )
                {
                    scale = (alpha*dkyk)/ykyk ;
                }
                beta = scale*((ykgk - ytg) - t*dphi*t1/dkyk)/dkyk ;

                beta = MAX (beta, Parm->BetaLower*(dphi0*alpha)/dkyk) ;

                cg_scale0 (vsub, gsubtemp, -ONE, nsub) ;
                cg_daxpy0 (vsub, gsub, scale, nsub) ;
                cg_trisolve (vsub, Rk, mem, nsub, 1) ;

                mp = SkFlast ;
                j = nsub - (mp+1) ;
                cg_copy0 (wsub, vsub+j, mp+1) ;
                cg_copy0 (wsub+(mp+1), vsub, j) ;

                cg_copy (gtemp, d, n) ;

                cg_matvec (d, SkF, wsub, nsub, n, 1) ;

                cg_daxpy (d, g, -scale, n) ;
                cg_daxpy (d, gtemp, beta, n) ;

                gHg = cg_dot0  (gsubtemp, gsub, nsub) ;
                t1 = MAX(gnorm2 -gsubnorm2, ZERO) ;
                dphi0 = -gHg - scale*t1 + beta*dphi ;

                dnorm2 = cg_dot (d, d, n) ;
            }
        }

        if ( (f < fbest) || (gnorm2 < gbest) )
        {
            nslow = 0 ;
            if ( f < fbest ) fbest = f ;
            if ( gnorm2 < gbest ) gbest = gnorm2 ;
        }
        else nslow++ ;
        if ( nslow > slowlimit )
        {
            status = 9 ;
            goto Exit ;
        }

        if ( PrintLevel >= 1 )
        {
            printf ("\niter: %5i f = %13.6e gnorm = %13.6e memk: %i "
                    "Subspace: %i\n", (int) iter, f, gnorm, memk, Subspace) ;
        }

        if ( Parm->debug )
        {
            if ( f > Com.f0 + Parm->debugtol*Ck )
            {
                status = 8 ;
                goto Exit ;
            }
        }

        if ( dphi0 > ZERO )
        {
           status = 5 ;
           goto Exit ;
        }
    }
    status = 2 ;
Exit:
    if ( status == 11 ) gnorm = INF ;
    if ( Stat != NULL )
    {
        Stat->nfunc = Com.nf ;
        Stat->ngrad = Com.ng ;
        Stat->iter = iter ;
        Stat->NumSub = NumSub ;
        Stat->IterSub = IterSub ;
        if ( status < 10 )
        {
            Stat->f = f ;
            Stat->gnorm = gnorm ;
        }
    }

    if ( (status > 0) && (status < 10) )
    {
        cg_copy (x, xtemp, n) ;
        gnorm = ZERO ;
        for (i = 0; i < n; i++)
        {
            g [i] = gtemp [i] ;
            t = fabs (g [i]) ;
            gnorm = MAX (gnorm, t) ;
        }
        if ( Stat != NULL ) Stat->gnorm = gnorm ;
    }
    if ( Parm->PrintFinal || PrintLevel >= 1 )
    {
        const char mess1 [] = "Possible causes of this error message:" ;
        const char mess2 [] = "   - your tolerance may be too strict: "
                              "grad_tol = " ;
        const char mess3 [] = "Line search fails" ;
        const char mess4 [] = "   - your gradient routine has an error" ;
        const char mess5 [] = "   - the parameter epsilon is too small" ;

        printf ("\nTermination status: %i\n", status) ;

        if ( status && NegDiag )
        {
            printf ("Parameter eta2 may be too small\n") ;
        }

        if ( status == 0 )
        {
            printf ("Convergence tolerance for gradient satisfied\n\n") ;
        }
        else if ( status == 1 )
        {
            printf ("Terminating since change in function value "
                    "<= feps*|f|\n\n") ;
        }
        else if ( status == 2 )
        {
            printf ("Number of iterations exceed specified limit\n") ;
            printf ("Iterations: %10.0f maxit: %10.0f\n",
                    (double) iter, (double) maxit) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n\n", mess2, grad_tol) ;
        }
        else if ( status == 3 )
        {
            printf ("Slope always negative in line search\n") ;
            printf ("%s\n", mess1) ;
            printf ("   - your cost function has an error\n") ;
            printf ("%s\n\n", mess4) ;
        }
        else if ( status == 4 )
        {
            printf ("Line search fails, too many iterations\n") ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n\n", mess2, grad_tol) ;
        }
        else if ( status == 5 )
        {
            printf ("Search direction not a descent direction\n\n") ;
        }
        else if ( status == 6 )
        {
            printf ("%s due to excessive updating of eps\n", mess3) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
            printf ("%s\n\n", mess4) ;
        }
        else if ( status == 7 )
        {
            printf ("%s\n%s\n", mess3, mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
            printf ("%s\n%s\n\n", mess4, mess5) ;
        }
        else if ( status == 8 )
        {
            printf ("Debugger is on, function value does not improve\n") ;
            printf ("new value: %25.16e old value: %25.16e\n\n", f, Com.f0) ;
        }
        else if ( status == 9 )
        {
            printf ("%i iterations without strict improvement in cost "
                    "or gradient\n\n", nslow) ;
        }
        else if ( status == 10 )
        {
            printf ("Insufficient memory for specified problem dimension %e"
                    " in cg_descent\n", (double) n) ;
        }
        else if ( status == 11 )
        {
            printf ("Function nan and could not be repaired\n\n") ;
        }
        else if ( status == 12 )
        {
            printf ("memory = %i is an invalid choice for parameter memory\n",
                     Parm->memory) ;
            printf ("memory should be either 0 or greater than 2\n\n") ;
        }

        printf ("maximum norm for gradient: %13.6e\n", gnorm) ;
        printf ("function value:            %13.6e\n\n", f) ;
        printf ("iterations:              %10.0f\n", (double) iter) ;
        printf ("function evaluations:    %10.0f\n", (double) Com.nf) ;
        printf ("gradient evaluations:    %10.0f\n", (double) Com.ng) ;
        if ( IterSub > 0 )
        {
            printf ("subspace iterations:     %10.0f\n", (double) IterSub) ;
            printf ("number of subspaces:     %10.0f\n", (double) NumSub) ;
        }
        printf ("===================================\n\n") ;
    }
    if ( Work == NULL ) free (work) ;
    return (status) ;
}

PRIVATE int cg_Wolfe
(
    double   alpha,
    double       f,
    double    dphi,
    cg_com    *Com
)
{
    if ( dphi >= Com->wolfe_lo )
    {

        if ( f - Com->f0 <= alpha*Com->wolfe_hi )
        {
            if ( Com->Parm->PrintLevel >= 2 )
            {
                printf ("Wolfe conditions hold\n") ;

            }
            return (1) ;
        }

        else if ( Com->AWolfe )
        {

            if ( (f <= Com->fpert) && (dphi <= Com->awolfe_hi) )
            {
                if ( Com->Parm->PrintLevel >= 2 )
                {
                    printf ("Approximate Wolfe conditions hold\n") ;

                }
                return (1) ;
            }
        }
    }

    return (0) ;
}

PRIVATE int cg_tol
(
    double     gnorm,
    cg_com    *Com
)
{

    if ( Com->Parm->StopRule )
    {
        if ( gnorm <= Com->tol ) return (1) ;
    }
    else if ( gnorm <= Com->tol*(ONE + fabs (Com->f)) ) return (1) ;
    return (0) ;
}

PRIVATE int cg_line
(
    cg_com   *Com
)
{
    int AWolfe, iter, ngrow, PrintLevel, qb, qb0, status, toggle ;
    double alpha, a, a1, a2, b, bmin, B, da, db, d0, d1, d2, dB, df, f, fa, fb,
           fB, a0, b0, da0, db0, fa0, fb0, width, rho ;
    char *s1, *s2, *fmt1, *fmt2 ;
    cg_parameter *Parm ;

    AWolfe = Com->AWolfe ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    if ( PrintLevel >= 1 )
    {
        if ( AWolfe )
        {
            printf ("Approximate Wolfe line search\n") ;
            printf ("=============================\n") ;
        }
        else
        {
            printf ("Wolfe line search\n") ;
            printf ("=================\n") ;
        }
    }

    if ( Com->QuadOK )
    {
        status = cg_evaluate ("fg", "y", Com) ;
        fb = Com->f ;
        if ( !AWolfe ) fb -= Com->alpha*Com->wolfe_hi ;
        qb = TRUE ;
    }
    else
    {
        status = cg_evaluate ("g", "y", Com) ;
        qb = FALSE ;
    }
    if ( status ) return (status) ;
    b = Com->alpha ;

    if ( AWolfe )
    {
        db = Com->df ;
        d0 = da = Com->df0 ;
    }
    else
    {
        db = Com->df - Com->wolfe_hi ;
        d0 = da = Com->df0 - Com->wolfe_hi ;
    }
    a = ZERO ;
    a1 = ZERO ;
    d1 = d0 ;
    fa = Com->f0 ;
    if ( PrintLevel >= 1 )
    {
        fmt1 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb: %13.6e "
               "da: %13.6e db: %13.6e\n" ;
        fmt2 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb:  x.xxxxxxxxxx "
               "da: %13.6e db: %13.6e\n" ;
        if ( Com->QuadOK ) s2 = "OK" ;
        else               s2 = "" ;
        if ( qb ) printf (fmt1, "start    ", s2, a, b, fa, fb, da, db);
        else      printf (fmt2, "start    ", s2, a, b, fa, da, db) ;
    }

    if ( (Com->QuadOK) && (Com->f <= Com->f0) )
    {
        if ( cg_Wolfe (b, Com->f, Com->df, Com) ) return (0) ;
    }

    if ( !AWolfe ) Com->Wolfe = TRUE ;

    rho = Com->rho ;
    ngrow = 1 ;
    while ( db < ZERO )
    {
        if ( !qb )
        {
            status = cg_evaluate ("f", "n", Com) ;
            if ( status ) return (status) ;
            if ( AWolfe ) fb = Com->f ;
            else          fb = Com->f - b*Com->wolfe_hi ;
            qb = TRUE ;
        }
        if ( fb > Com->fpert )
        {
            status = cg_contract (&a, &fa, &da, &b, &fb, &db, Com) ;
            if ( status == 0 ) return (0) ;
            if ( status == -2 ) goto Line ;
            if ( Com->neps > Parm->neps ) return (6) ;
        }

        ngrow++ ;
        if ( ngrow > Parm->ntries ) return (3) ;

        a = b ;
        fa = fb ;
        da = db ;

        d2 = d1 ;
        d1 = da ;
        a2 = a1 ;
        a1 = a ;

        bmin = rho*b ;
        if ( (ngrow == 2) || (ngrow == 3) || (ngrow == 6) )
        {
            if ( d1 > d2 )
            {
                if ( ngrow == 2 )
                {
                    b = a1 - (a1-a2)*(d1/(d1-d2)) ;
                }
                else
                {
                    if ( (d1-d2)/(a1-a2) >= (d2-d0)/a2 )
                    {

                        b = a1 - (a1-a2)*(d1/(d1-d2)) ;
                    }
                    else
                    {

                        b = a1 - Parm->SecantAmp*(a1-a2)*(d1/(d1-d2)) ;
                    }
                }

                b = MIN (b, Parm->ExpandSafe*a1) ;
            }
            else rho *= Parm->RhoGrow ;
        }
        else rho *= Parm->RhoGrow ;
        b = MAX (bmin, b) ;
        Com->alphaold = Com->alpha ;
        Com->alpha = b ;
        status = cg_evaluate ("g", "p", Com) ;
        if ( status ) return (status) ;
        b = Com->alpha ;
        qb = FALSE ;
        if ( AWolfe ) db = Com->df ;
        else          db = Com->df - Com->wolfe_hi ;
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s2 = "OK" ;
            else               s2 = "" ;
            printf (fmt2, "expand   ", s2, a, b, fa, da, db) ;
        }
    }

Line:
    toggle = 0 ;
    width = b - a ;
    qb0 = FALSE ;
    for (iter = 0; iter < Parm->nline; iter++)
    {

        if ( (toggle == 0) || ((toggle == 2) && ((b-a) <= width)) )
        {
            Com->QuadOK = TRUE ;
            if ( Com->UseCubic && qb )
            {
                s1 = "cubic    " ;
                alpha = cg_cubic (a, fa, da, b, fb, db) ;
                if ( alpha < ZERO )
                {
                    s1 = "secant   " ;
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
            else
            {
                s1 = "secant   " ;
                if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                else                 alpha = -1. ;
            }
            width = Parm->gamma*(b - a) ;
        }
        else if ( toggle == 1 )
        {
            Com->QuadOK = TRUE ;
            if ( Com->UseCubic )
            {
                s1 = "cubic    " ;
                if ( Com->alpha == a )
                {
                    alpha = cg_cubic (a0, fa0, da0, a, fa, da) ;
                }
                else if ( qb0 )
                {
                    alpha = cg_cubic (b, fb, db, b0, fb0, db0) ;
                }
                else alpha = -1. ;

                if ( (alpha <= a) || (alpha >= b) )
                {
                    if ( qb ) alpha = cg_cubic (a, fa, da, b, fb, db) ;
                    else alpha = -1. ;
                }

                if ( alpha < ZERO )
                {
                    s1 = "secant   " ;
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
            else
            {
                s1 = "secant   " ;
                if ( (Com->alpha == a) && (da > da0) )
                {
                    alpha = a - (a-a0)*(da/(da-da0)) ;
                }
                else if ( db < db0 )
                {
                    alpha = b - (b-b0)*(db/(db-db0)) ;
                }
                else
                {
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }

                if ( (alpha <= a) || (alpha >= b) )
                {
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
        }
        else
        {
            alpha = .5*(a+b) ;
            s1 = "bisection" ;
            Com->QuadOK = FALSE ;
        }

        if ( (alpha <= a) || (alpha >= b) )
        {
            alpha = .5*(a+b) ;
            s1 = "bisection" ;
            if ( (alpha == a) || (alpha == b) ) return (7) ;
            Com->QuadOK = FALSE ;
        }

        if ( toggle == 0 )
        {
            a0 = a ;
            b0 = b ;
            da0 = da ;
            db0 = db ;
            fa0 = fa ;
            if ( qb )
            {
                fb0 = fb ;
                qb0 = TRUE ;
            }
        }

        toggle++ ;
        if ( toggle > 2 ) toggle = 0 ;

        Com->alpha = alpha ;
        status = cg_evaluate ("fg", "n", Com) ;
        if ( status ) return (status) ;
        Com->alpha = alpha ;
        f = Com->f ;
        df = Com->df ;
        if ( Com->QuadOK )
        {
            if ( cg_Wolfe (alpha, f, df, Com) )
            {
                if ( PrintLevel >= 2 )
                {
                    printf ("             a: %13.6e f: %13.6e df: %13.6e %1s\n",
                             alpha, f, df, s1) ;
                }
                return (0) ;
            }
        }
        if ( !AWolfe )
        {
            f -= alpha*Com->wolfe_hi ;
            df -= Com->wolfe_hi ;
        }
        if ( df >= ZERO )
        {
            b = alpha ;
            fb = f ;
            db = df ;
            qb = TRUE ;
        }
        else if ( f <= Com->fpert )
        {
            a = alpha ;
            da = df ;
            fa = f ;
        }
        else
        {
            B = b ;
            if ( qb ) fB = fb ;
            dB = db ;
            b = alpha ;
            fb = f ;
            db = df ;

            status = cg_contract (&a, &fa, &da, &b, &fb, &db, Com) ;
            if ( status == 0 ) return (0) ;
            if ( status == -1 )
            {
                if ( Com->neps > Parm->neps ) return (6) ;
                a = b ;
                fa = fb ;
                da = db ;
                b = B ;
                if ( qb ) fb = fB ;
                db = dB ;
            }
            else qb = TRUE ;
        }
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s2 = "OK" ;
            else               s2 = "" ;
            if ( !qb ) printf (fmt2, s1, s2, a, b, fa, da, db) ;
            else       printf (fmt1, s1, s2, a, b, fa, fb, da, db) ;
        }
    }
    return (4) ;
}

PRIVATE int cg_contract
(
    double    *A,
    double   *fA,
    double   *dA,
    double    *B,
    double   *fB,
    double   *dB,
    cg_com  *Com
)
{
    int AWolfe, iter, PrintLevel, toggle, status ;
    double a, alpha, b, old, da, db, df, d1, dold, f, fa, fb, f1, fold,
           t, width ;
    char *s ;
    cg_parameter *Parm ;

    AWolfe = Com->AWolfe ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    a = *A ;
    fa = *fA ;
    da = *dA ;
    b = *B ;
    fb = *fB ;
    db = *dB ;
    f1 = fb ;
    d1 = db ;
    toggle = 0 ;
    width = ZERO ;
    for (iter = 0; iter < Parm->nshrink; iter++)
    {
        if ( (toggle == 0) || ((toggle == 2) && ((b-a) <= width)) )
        {

            alpha = cg_cubic (a, fa, da, b, fb, db) ;
            toggle = 0 ;
            width = Parm->gamma*(b-a) ;
            if ( iter ) Com->QuadOK = TRUE ;
        }
        else if ( toggle == 1 )
        {
            Com->QuadOK = TRUE ;

            if ( old < a )
            {
                alpha = cg_cubic (a, fa, da, old, fold, dold) ;
            }
            else
            {
                alpha = cg_cubic (a, fa, da, b, fb, db) ;
            }
        }
        else
        {
            alpha = .5*(a+b) ;
            Com->QuadOK = FALSE ;
        }

        if ( (alpha <= a) || (alpha >= b) )
        {
            alpha = .5*(a+b) ;
            Com->QuadOK = FALSE ;
        }

        toggle++ ;
        if ( toggle > 2 ) toggle = 0 ;

        Com->alpha = alpha ;
        status = cg_evaluate ("fg", "n", Com) ;
        if ( status ) return (status) ;
        f = Com->f ;
        df = Com->df ;

        if ( Com->QuadOK )
        {
            if ( cg_Wolfe (alpha, f, df, Com) ) return (0) ;
        }
        if ( !AWolfe )
        {
            f -= alpha*Com->wolfe_hi ;
            df -= Com->wolfe_hi ;
        }
        if ( df >= ZERO )
        {
            *B = alpha ;
            *fB = f ;
            *dB = df ;
            *A = a ;
            *fA = fa ;
            *dA = da ;
            return (-2) ;
        }
        if ( f <= Com->fpert )
        {
            old = a ;
            a = alpha ;
            fold = fa ;
            fa = f ;
            dold = da ;
            da = df ;
        }
        else
        {
            old = b ;
            b = alpha ;
            fb = f ;
            db = df ;
        }
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s = "OK" ;
            else               s = "" ;
            printf ("contract  %2s a: %13.6e b: %13.6e fa: %13.6e fb: "
                    "%13.6e da: %13.6e db: %13.6e\n", s, a, b, fa, fb, da, db) ;
        }
    }

    if ( fabs (fb) <= Com->SmallCost ) Com->PertRule = FALSE ;

    t = Com->f0 ;
    if ( Com->PertRule )
    {
        if ( t != ZERO )
        {
            Com->eps = Parm->egrow*(f1-t)/fabs (t) ;
            Com->fpert = t + fabs (t)*Com->eps ;
        }
        else Com->fpert = 2.*f1 ;
    }
    else
    {
        Com->eps = Parm->egrow*(f1-t) ;
        Com->fpert = t + Com->eps ;
    }
    if ( PrintLevel >= 1 )
    {
        printf ("--increase eps: %e fpert: %e\n", Com->eps, Com->fpert) ;
    }
    Com->neps++ ;
    return (-1) ;
}

PRIVATE int cg_evaluate
(
    char    *what,
    char     *nan,
    cg_com   *Com
)
{
    INT n ;
    int i ;
    double alpha, *d, *gtemp, *x, *xtemp ;
    cg_parameter *Parm ;
    Parm = Com->Parm ;
    n = Com->n ;
    x = Com->x ;
    d = Com->d ;
    xtemp = Com->xtemp ;
    gtemp = Com->gtemp ;
    alpha = Com->alpha ;

    if ( !strcmp (nan, "y") || !strcmp (nan, "p") )
    {
        if ( !strcmp (what, "f") )
        {
            cg_step (xtemp, x, d, alpha, n) ;

            Com->f = Com->cg_value (xtemp, n) ;
            Com->nf++ ;

            if ( (Com->f != Com->f) || (Com->f >= INF) || (Com->f <= -INF) )
            {
                for (i = 0; i < Parm->ntries; i++)
                {
                    if ( !strcmp (nan, "p") )
                    {
                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
                    }
                    else
                    {
                        alpha *= Parm->nan_decay ;
                    }
                    cg_step (xtemp, x, d, alpha, n) ;
                    Com->f = Com->cg_value (xtemp, n) ;
                    Com->nf++ ;
                    if ( (Com->f == Com->f) && (Com->f < INF) &&
                         (Com->f > -INF) ) break ;
                }
                if ( i == Parm->ntries ) return (11) ;
            }
            Com->alpha = alpha ;
        }
        else if ( !strcmp (what, "g") )
        {
            cg_step (xtemp, x, d, alpha, n) ;
            Com->cg_grad (gtemp, xtemp, n) ;
            Com->ng++ ;
            Com->df = cg_dot (gtemp, d, n) ;

            if ( (Com->df != Com->df) || (Com->df >= INF) || (Com->df <= -INF) )
            {
                for (i = 0; i < Parm->ntries; i++)
                {
                    if ( !strcmp (nan, "p") )
                    {
                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
                    }
                    else
                    {
                        alpha *= Parm->nan_decay ;
                    }
                    cg_step (xtemp, x, d, alpha, n) ;
                    Com->cg_grad (gtemp, xtemp, n) ;
                    Com->ng++ ;
                    Com->df = cg_dot (gtemp, d, n) ;
                    if ( (Com->df == Com->df) && (Com->df < INF) &&
                         (Com->df > -INF) ) break ;
                }
                if ( i == Parm->ntries ) return (11) ;
                Com->rho = Parm->nan_rho ;
            }
            else Com->rho = Parm->rho ;
            Com->alpha = alpha ;
        }
        else
        {
            cg_step (xtemp, x, d, alpha, n) ;
            if ( Com->cg_valgrad != NULL )
            {
                Com->f = Com->cg_valgrad (gtemp, xtemp, n) ;
            }
            else
            {
                Com->cg_grad (gtemp, xtemp, n) ;
                Com->f = Com->cg_value (xtemp, n) ;
            }
            Com->df = cg_dot (gtemp, d, n) ;
            Com->nf++ ;
            Com->ng++ ;

            if ( (Com->df !=  Com->df) || (Com->f != Com->f) ||
                 (Com->df >=  INF)     || (Com->f >= INF)    ||
                 (Com->df <= -INF)     || (Com->f <= -INF))
            {
                for (i = 0; i < Parm->ntries; i++)
                {
                    if ( !strcmp (nan, "p") )
                    {
                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
                    }
                    else
                    {
                        alpha *= Parm->nan_decay ;
                    }
                    cg_step (xtemp, x, d, alpha, n) ;
                    if ( Com->cg_valgrad != NULL )
                    {
                        Com->f = Com->cg_valgrad (gtemp, xtemp, n) ;
                    }
                    else
                    {
                        Com->cg_grad (gtemp, xtemp, n) ;
                        Com->f = Com->cg_value (xtemp, n) ;
                    }
                    Com->df = cg_dot (gtemp, d, n) ;
                    Com->nf++ ;
                    Com->ng++ ;
                    if ( (Com->df == Com->df) && (Com->f == Com->f) &&
                         (Com->df <  INF)     && (Com->f <  INF)    &&
                         (Com->df > -INF)     && (Com->f > -INF) ) break ;
                }
                if ( i == Parm->ntries ) return (11) ;
                Com->rho = Parm->nan_rho ;
            }
            else Com->rho = Parm->rho ;
            Com->alpha = alpha ;
        }
    }
    else
    {
        if ( !strcmp (what, "fg") )
        {
            if ( alpha == ZERO )
            {

                cg_copy (xtemp, x, n) ;
                if ( Com->cg_valgrad != NULL )
                {
                    Com->f = Com->cg_valgrad (Com->g, xtemp, n) ;
                }
                else
                {
                    Com->cg_grad (Com->g, xtemp, n) ;
                    Com->f = Com->cg_value (xtemp, n) ;
                }
            }
            else
            {
                cg_step (xtemp, x, d, alpha, n) ;
                if ( Com->cg_valgrad != NULL )
                {
                    Com->f = Com->cg_valgrad (gtemp, xtemp, n) ;
                }
                else
                {
                    Com->cg_grad (gtemp, xtemp, n) ;
                    Com->f = Com->cg_value (xtemp, n) ;
                }
                Com->df = cg_dot (gtemp, d, n) ;
            }
            Com->nf++ ;
            Com->ng++ ;
            if ( (Com->df != Com->df) || (Com->f != Com->f) ||
                 (Com->df == INF)     || (Com->f == INF)    ||
                 (Com->df ==-INF)     || (Com->f ==-INF) ) return (11) ;
        }
        else if ( !strcmp (what, "f") )
        {
            cg_step (xtemp, x, d, alpha, n) ;
            Com->f = Com->cg_value (xtemp, n) ;
            Com->nf++ ;
            if ( (Com->f != Com->f) || (Com->f == INF) || (Com->f ==-INF) )
                return (11) ;
        }
        else
        {
            cg_step (xtemp, x, d, alpha, n) ;
            Com->cg_grad (gtemp, xtemp, n) ;
            Com->df = cg_dot (gtemp, d, n) ;
            Com->ng++ ;
            if ( (Com->df != Com->df) || (Com->df == INF) || (Com->df ==-INF) )
                return (11) ;
        }
    }
    return (0) ;
}

PRIVATE double cg_cubic
(
    double  a,
    double fa,
    double da,
    double  b,
    double fb,
    double db
)
{
    double c, d1, d2, delta, t, v, w ;
    delta = b - a ;
    if ( delta == ZERO ) return (a) ;
    v = da + db - 3.*(fb-fa)/delta ;
    t = v*v - da*db ;
    if ( t < ZERO )
    {
         if ( fabs (da) < fabs (db) ) c = a - (a-b)*(da/(da-db)) ;
         else if ( da != db )         c = b - (a-b)*(db/(da-db)) ;
         else                         c = -1 ;
         return (c) ;
    }

    if ( delta > ZERO ) w = sqrt(t) ;
    else                w =-sqrt(t) ;
    d1 = da + v - w ;
    d2 = db + v + w ;
    if ( (d1 == ZERO) && (d2 == ZERO) ) return (-1.) ;
    if ( fabs (d1) >= fabs (d2) ) c = a + delta*da/d1 ;
    else                          c = b - delta*db/d2 ;
    return (c) ;
}

PRIVATE void cg_matvec
(
    double *y,
    double *A,
    double *x,
    int     n,
    INT     m,
    int     w
)
{

#ifdef NOBLAS
    INT j, l ;
    l = 0 ;
    if ( w )
    {
        cg_scale0 (y, A, x [0], (int) m) ;
        for (j = 1; j < n; j++)
        {
            l += m ;
            cg_daxpy0 (y, A+l, x [j], (int) m) ;
        }
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            y [j] = cg_dot0 (A+l, x, (int) m) ;
            l += m ;
        }
    }

#endif

#ifndef NOBLAS
    INT j, l ;
    l = 0 ;
    if ( w )
    {
        cg_scale0 (y, A, x [0], (int) m) ;
        for (j = 1; j < n; j++)
        {
            l += m ;
            cg_daxpy0 (y, A+l, x [j], (int) m) ;
        }
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            y [j] = cg_dot0 (A+l, x, (int) m) ;
            l += m ;
        }
    }
#endif

    return ;
}

PRIVATE void cg_trisolve
(
    double *x,
    double *R,
    int     m,
    int     n,
    int     w
)
{
    int i, l ;
    if ( w )
    {
        l = m*n ;
        for (i = n; i > 0; )
        {
            i-- ;
            l -= (m-i) ;
            x [i] /= R [l] ;
            l -= i ;
            cg_daxpy0 (x, R+l, -x [i], i) ;
        }
    }
    else
    {
        l = 0 ;
        for (i = 0; i < n; i++)
        {
            x [i] = (x [i] - cg_dot0 (x, R+l, i))/R [l+i] ;
            l += m ;
        }
    }

    return ;
}

PRIVATE double cg_inf
(
    double *x,
    INT     n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    double t ;
    t = ZERO ;
    n5 = n % 5 ;

    for (i = 0; i < n5; i++) if ( t < fabs (x [i]) ) t = fabs (x [i]) ;
    for (; i < n; i += 5)
    {
        if ( t < fabs (x [i]  ) ) t = fabs (x [i]  ) ;
        if ( t < fabs (x [i+1]) ) t = fabs (x [i+1]) ;
        if ( t < fabs (x [i+2]) ) t = fabs (x [i+2]) ;
        if ( t < fabs (x [i+3]) ) t = fabs (x [i+3]) ;
        if ( t < fabs (x [i+4]) ) t = fabs (x [i+4]) ;
    }
    return (t) ;
#endif

#ifndef NOBLAS
    auto i = cblas_idamax(n, x, 1);
    return fabs(x[i]);
#endif
}

PRIVATE void cg_scale0
(
    double *y,
    double *x,
    double  s,
    int     n
)
{
#ifdef NOBLAS
    int i, n5 ;
    n5 = n % 5 ;
    if ( s == -ONE)
    {
       for (i = 0; i < n5; i++) y [i] = -x [i] ;
       for (; i < n;)
       {
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
           y [i] = -x [i] ;
           i++ ;
       }
    }
    else
    {
        for (i = 0; i < n5; i++) y [i] = s*x [i] ;
        for (; i < n;)
        {
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
        }
    }

#endif

#ifndef NOBLAS
    cblas_dcopy(n, x, 1, y, 1);
    cblas_dscal(n, s, y, 1);
#endif

    return ;
}

PRIVATE void cg_scale
(
    double *y,
    double *x,
    double  s,
    INT     n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    n5 = n % 5 ;
#endif
    if ( y == x)
    {
#ifdef NOBLAS
        for (i = 0; i < n5; i++) y [i] *= s ;
        for (; i < n;)
        {
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
            y [i] *= s ;
            i++ ;
        }
#endif
#ifndef NOBLAS
        cblas_dscal(n, s, y, 1);
#endif
    }
    else
    {
#ifdef NOBLAS
        for (i = 0; i < n5; i++) y [i] = s*x [i] ;
        for (; i < n;)
        {
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
            y [i] = s*x [i] ;
            i++ ;
        }
#endif

#ifndef NOBLAS
        cblas_dcopy(n, x, 1, y, 1);
        cblas_dscal(n, s, y, 1);
#endif
    }
    return ;

}

PRIVATE void cg_daxpy0
(
    double     *x,
    double     *d,
    double  alpha,
    int         n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    n5 = n % 5 ;
    if (alpha == -ONE)
    {
        for (i = 0; i < n5; i++) x [i] -= d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   -= d [i] ;
            x [i+1] -= d [i+1] ;
            x [i+2] -= d [i+2] ;
            x [i+3] -= d [i+3] ;
            x [i+4] -= d [i+4] ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++) x [i] += alpha*d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   += alpha*d [i] ;
            x [i+1] += alpha*d [i+1] ;
            x [i+2] += alpha*d [i+2] ;
            x [i+3] += alpha*d [i+3] ;
            x [i+4] += alpha*d [i+4] ;
        }
    }
#endif

#ifndef NOBLAS
    cblas_daxpy(n, alpha, d, 1, x, 1);
#endif
    return ;
}

PRIVATE void cg_daxpy
(
    double     *x,
    double     *d,
    double  alpha,
    INT         n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    n5 = n % 5 ;
    if (alpha == -ONE)
    {
        for (i = 0; i < n5; i++) x [i] -= d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   -= d [i] ;
            x [i+1] -= d [i+1] ;
            x [i+2] -= d [i+2] ;
            x [i+3] -= d [i+3] ;
            x [i+4] -= d [i+4] ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++) x [i] += alpha*d[i] ;
        for (; i < n; i += 5)
        {
            x [i]   += alpha*d [i] ;
            x [i+1] += alpha*d [i+1] ;
            x [i+2] += alpha*d [i+2] ;
            x [i+3] += alpha*d [i+3] ;
            x [i+4] += alpha*d [i+4] ;
        }
    }
#endif

#ifndef NOBLAS
    cblas_daxpy(n, alpha, d, 1, x, 1);
#endif

    return ;
}

PRIVATE double cg_dot0
(
    double *x,
    double *y,
    int     n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    double t ;
    t = ZERO ;
    if ( n <= 0 ) return (t) ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) t += x [i]*y [i] ;
    for (; i < n; i += 5)
    {
        t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2]
                        + x [i+3]*y [i+3] + x [i+4]*y [i+4] ;
    }
    return (t) ;
#endif

#ifndef NOBLAS
    return cblas_ddot(n, x, 1, y, 1);
#endif
}

PRIVATE double cg_dot
(
    double *x,
    double *y,
    INT     n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    double t ;
    t = ZERO ;
    if ( n <= 0 ) return (t) ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) t += x [i]*y [i] ;
    for (; i < n; i += 5)
    {
        t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2]
                        + x [i+3]*y [i+3] + x [i+4]*y [i+4] ;
    }
    return (t) ;
#endif

#ifndef NOBLAS
    return cblas_ddot(n, x, 1, y, 1);
#endif
}

PRIVATE void cg_copy0
(
    double *y,
    double *x,
    int     n
)
{
#ifdef NOBLAS
    int i, n5 ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) y [i] = x [i] ;
    for (; i < n; )
    {
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
    }
#endif

#ifndef NOBLAS
    cblas_dcopy(n, x, 1, y, 1);
#endif

    return ;
}

PRIVATE void cg_copy
(
    double *y,
    double *x,
    INT     n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) y [i] = x [i] ;
    for (; i < n; )
    {
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
        y [i] = x [i] ;
        i++ ;
    }
#endif

#ifndef NOBLAS
    cblas_dcopy(n, x, 1, y, 1);
#endif

    return ;
}

PRIVATE void cg_step
(
    double *xtemp,
    double     *x,
    double     *d,
    double  alpha,
    INT         n
)
{
#ifdef NOBLAS
    INT n5, i ;
    n5 = n % 5 ;
    if (alpha == -ONE)
    {
        for (i = 0; i < n5; i++) xtemp [i] = x[i] - d[i] ;
        for (; i < n; i += 5)
        {
            xtemp [i]   = x [i]   - d [i] ;
            xtemp [i+1] = x [i+1] - d [i+1] ;
            xtemp [i+2] = x [i+2] - d [i+2] ;
            xtemp [i+3] = x [i+3] - d [i+3] ;
            xtemp [i+4] = x [i+4] - d [i+4] ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++) xtemp [i] = x[i] + alpha*d[i] ;
        for (; i < n; i += 5)
        {
            xtemp [i]   = x [i]   + alpha*d [i] ;
            xtemp [i+1] = x [i+1] + alpha*d [i+1] ;
            xtemp [i+2] = x [i+2] + alpha*d [i+2] ;
            xtemp [i+3] = x [i+3] + alpha*d [i+3] ;
            xtemp [i+4] = x [i+4] + alpha*d [i+4] ;
        }
    }
#endif

#ifndef NOBLAS
    cblas_dcopy(n, x, 1, xtemp, 1);
    cblas_daxpy(n, alpha, d, 1, xtemp, 1);
#endif

    return ;
}

PRIVATE void cg_init
(
    double *x,
    double  s,
    INT     n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) x [i] = s ;
    for (; i < n;)
    {
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
        x [i] = s ;
        i++ ;
    }
#endif

#ifndef NOBLAS
#pragma omp parallel for
    for (INT i = 0; i < n; i++) {
        x[i] = s;
    }
#endif

    return ;
}

PRIVATE double cg_update_2
(
    double *gold,
    double *gnew,
    double    *d,
    INT        n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    double s, t ;
    t = ZERO ;
    n5 = n % 5 ;

    if ( d == NULL )
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            i++ ;
        }
    }
    else if ( gold != NULL )
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            gold [i] = s ;
            d [i] = -s ;
            i++ ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;

            s = gnew [i] ;
            t += s*s ;
            d [i] = -s ;
            i++ ;
        }
    }
    return (t) ;
#endif

#ifndef NOBLAS
    if (d == NULL) {
        cblas_dcopy(n, gnew, 1, gold, 1);
        auto t =  cblas_dnrm2(n, gnew, 1);
        return t*t;
    }
    else if (gold != NULL) {
        cblas_dcopy(n, gnew, 1, gold, 1);
        cblas_dcopy(n, gnew, 1, d, 1);
        cblas_dscal(n, -1.0, d, 1);
        auto t =  cblas_dnrm2(n, gnew, 1);
        return t*t;
    }
    else {
        cblas_dcopy(n, gnew, 1, d, 1);
        cblas_dscal(n, -1.0, d, 1);
        auto t =  cblas_dnrm2(n, gnew, 1);
        return t*t;
    }
#endif
}

PRIVATE double cg_update_inf
(
    double *gold,
    double *gnew,
    double    *d,
    INT        n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    double s, t ;
    t = ZERO ;
    n5 = n % 5 ;

    if ( d == NULL )
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;
        }
    }
    else
    {
        for (i = 0; i < n5; i++)
        {
            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
        }
        for (; i < n; )
        {
            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;

            s = gnew [i] ;
            gold [i] = s ;
            d [i] = -s ;
            if ( t < fabs (s) ) t = fabs (s) ;
            i++ ;
        }
    }
    return (t) ;
#endif

#ifndef NOBLAS
    if (d == NULL) {
        cblas_dcopy(n, gnew, 1, gold, 1);
        return gnew[cblas_idamax(n, gnew, 1)];
    }
    else {
        cblas_dcopy(n, gnew, 1, gold, 1);
        cblas_dcopy(n, gnew, 1, d, 1);
        cblas_dscal(n, -1.0, d, 1);
        return fabs(gnew[cblas_idamax(n, gnew, 1)]);
    }
#endif

}

PRIVATE double cg_update_ykyk
(
    double *gold,
    double *gnew,
    double *Ykyk,
    double *Ykgk,
    INT        n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    double t, gnorm, yk, ykyk, ykgk ;
    gnorm = ZERO ;
    ykyk = ZERO ;
    ykgk = ZERO ;
    n5 = n % 5 ;

    for (i = 0; i < n5; i++)
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
    }
    for (; i < n; )
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        yk = t - gold [i] ;
        gold [i] = t ;
        ykgk += yk*t ;
        ykyk += yk*yk ;
        i++ ;
    }
    *Ykyk = ykyk ;
    *Ykgk = ykgk ;
    return (gnorm) ;
#endif

#ifndef NOBLAS
    double ykyk, ykgk, yk;
    ykyk = ZERO ;
    ykgk = ZERO ;

#pragma omp parallel for reduction(+:ykyk,ykgk)
    for (INT i = 0; i < n; i++) {
        yk = gnew[i] - gold[i];
        ykgk += yk * gnew[i];
        ykyk += yk * yk;
    }
    cblas_dcopy(n, gnew, 1, gold, 1);
    *Ykyk = ykyk ;
    *Ykgk = ykgk ;
    return fabs(gnew[cblas_idamax(n, gnew, 1)]);

#endif
}

PRIVATE double cg_update_inf2
(
    double   *gold,
    double   *gnew,
    double      *d,
    double *gnorm2,
    INT          n
)
{
#ifdef NOBLAS
    INT i, n5 ;
    double gnorm, s, t ;
    gnorm = ZERO ;
    s = ZERO ;
    n5 = n % 5 ;

    for (i = 0; i < n5; i++)
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
    }
    for (; i < n; )
    {
        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;

        t = gnew [i] ;
        if ( gnorm < fabs (t) ) gnorm = fabs (t) ;
        s += t*t ;
        gold [i] = t ;
        d [i] = -t ;
        i++ ;
    }
    *gnorm2 = s ;
    return (gnorm) ;
#endif

#ifndef NOBLAS
    cblas_dcopy(n, gnew, 1, gold, 1);
    cblas_dcopy(n, gnew, 1, d, 1);
    cblas_dscal(n, -1.0, d, 1);
    *gnorm2 = cblas_dnrm2(n, gnew, 1);
    *gnorm2 *= *gnorm2;
    return fabs(gnew[cblas_idamax(n, gnew, 1)]);
#endif
}

PRIVATE double cg_update_d
(
    double      *d,
    double      *g,
    double    beta,
    double *gnorm2,
    INT          n
)

{
#ifdef NOBLAS
    INT i, n5 ;
    double dnorm2, s, t ;
    s = ZERO ;
    dnorm2 = ZERO ;
    n5 = n % 5 ;
    if ( gnorm2 == NULL )
    {
        for (i = 0; i < n5; i++)
        {
            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
        }
        for (; i < n; )
        {
            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;
        }
    }
    else
    {
        s = ZERO ;
        for (i = 0; i < n5; i++)
        {
            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
        }
        for (; i < n; )
        {
            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;

            t = g [i] ;
            s += t*t ;
            t = -t + beta*d [i] ;
            d [i] = t ;
            dnorm2 += t*t ;
            i++ ;
        }
        *gnorm2 = s ;
    }

    return (dnorm2) ;
#endif

#ifndef NOBLAS
    double dnorm2, gn2;
    dnorm2 = ZERO;
    gn2 = ZERO;
    if (gnorm2 == NULL) {
        #pragma omp parallel for reduction(+:dnorm2)
        for (INT i = 0; i < n; i++) {
            d[i] = -g[i] + beta * d[i];
            dnorm2 += d[i] * d[i];
        }
    }
    else {
        #pragma omp parallel for reduction(+:dnorm2, gn2)
        for (INT i = 0; i < n; i++) {
            d[i] = -g[i] + beta * d[i];
            dnorm2 += d[i] * d[i];
            gn2 += g[i] * g[i];
        }
        *gnorm2 = gn2;
    }
    return dnorm2;
#endif
}

PRIVATE void cg_Yk
(
    double    *y,
    double *gold,
    double *gnew,
    double  *yty,
    INT        n
)
{
#ifdef NOBLAS
    double s, t ;
    if ( (y != NULL) && (yty == NULL) ) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            y[i] = gnew[i] - gold[i];
            gold[i] = gnew[i];
        }
    }
    else if ( (y == NULL) && (yty != NULL) ) {
        s = ZERO ;
#pragma omp parallel for reduction(+:s)
        for (int i = 0; i < n; ++i) {
                t = gnew [i] - gold [i] ;
                gold [i] = gnew [i] ;
                s += t*t ;
        }
        *yty = s ;
    }
    else {
        s = ZERO ;
#pragma omp parallel for reduction(+:s)
        for (int i = 0; i < n; ++i) {
            y[i] = gnew[i] - gold[i];
            gold[i] = gnew[i];
            s += y[i]*y[i];
        }
        *yty = s;
    }

#endif

#ifndef NOBLAS
    if (y != NULL) {
        cblas_dcopy(n, gnew, 1, y, 1);
        cblas_daxpy(n, -1.0, gold, 1, y, 1);
    }
    if (yty != NULL) {
        *yty = cblas_dnrm2(n, y, 1);
        *yty *= *yty;
    }
    cblas_dcopy(n, gnew, 1, gold, 1);
#endif
    return ;
}

void cg_default
(
    cg_parameter   *Parm
)
{

    Parm->PrintFinal = FALSE ;

    Parm->PrintLevel = 0 ;

    Parm->PrintParms = FALSE ;

    Parm->LBFGS = FALSE ;

    Parm->memory = 11 ;

    Parm->SubCheck = 8 ;
    Parm->SubSkip = 4 ;

    Parm ->eta0 = 0.001 ;

    Parm->eta1 = 0.900 ;

    Parm->eta2 = 1.e-10 ;

    Parm->AWolfe = FALSE ;
    Parm->AWolfeFac = 1.e-3 ;

    Parm->Qdecay = .7 ;

    Parm->nslow = 1000 ;

    Parm->StopRule = TRUE ;
    Parm->StopFac = 0.e-12 ;

    Parm->PertRule = TRUE ;
    Parm->eps = 1.e-6 ;

    Parm->egrow = 10. ;

    Parm->QuadStep = TRUE ;
    Parm->QuadCutOff = 1.e-12 ;

    Parm->QuadSafe = 1.e-10 ;

    Parm->UseCubic = TRUE ;

    Parm->CubicCutOff = 1.e-12 ;

    Parm->SmallCost = 1.e-30 ;

    Parm->debug = FALSE ;
    Parm->debugtol = 1.e-10 ;

    Parm->step = ZERO ;

    Parm->maxit = INT_INF ;

    Parm->ntries = (int) 50 ;

    Parm->ExpandSafe = 200. ;

    Parm->SecantAmp = 1.05 ;

    Parm->RhoGrow = 2.0 ;

    Parm->neps = (int) 5 ;

    Parm->nshrink = (int) 10 ;

    Parm->nline = (int) 50 ;

    Parm->restart_fac = 6.0 ;

    Parm->feps = ZERO ;

    Parm->nan_rho = 1.3 ;

    Parm->nan_decay = 0.1 ;

    Parm->delta = .1 ;

    Parm->sigma = .9 ;

    Parm->gamma = .66 ;

    Parm->rho = 5. ;

    Parm->psi0 = .01 ;

    Parm->psi_lo = 0.1 ;
    Parm->psi_hi = 10. ;

    Parm->psi1 = 1.0 ;

    Parm->psi2 = 2. ;

    Parm->AdaptiveBeta = FALSE ;

    Parm->BetaLower = 0.4 ;

    Parm->theta = 1.0 ;

    Parm->qeps = 1.e-12 ;

    Parm->qrestart = 6 ;

    Parm->qrule = 1.e-8 ;
}

PRIVATE void cg_printParms
(
    cg_parameter  *Parm
)
{
    printf ("PARAMETERS:\n") ;
    printf ("\n") ;
    printf ("Wolfe line search parameter ..................... delta: %e\n",
             Parm->delta) ;
    printf ("Wolfe line search parameter ..................... sigma: %e\n",
             Parm->sigma) ;
    printf ("decay factor for bracketing interval ............ gamma: %e\n",
             Parm->gamma) ;
    printf ("growth factor for bracket interval ................ rho: %e\n",
             Parm->rho) ;
    printf ("growth factor for bracket interval after nan .. nan_rho: %e\n",
             Parm->nan_rho) ;
    printf ("decay factor for stepsize after nan ......... nan_decay: %e\n",
             Parm->nan_decay) ;
    printf ("parameter in lower bound for beta ........... BetaLower: %e\n",
             Parm->BetaLower) ;
    printf ("parameter describing cg_descent family .......... theta: %e\n",
             Parm->theta) ;
    printf ("perturbation parameter for function value ......... eps: %e\n",
             Parm->eps) ;
    printf ("factor by which eps grows if necessary .......... egrow: %e\n",
             Parm->egrow) ;
    printf ("factor for computing average cost .............. Qdecay: %e\n",
             Parm->Qdecay) ;
    printf ("relative change in cost to stop quadstep ... QuadCutOff: %e\n",
             Parm->QuadCutOff) ;
    printf ("maximum factor quadstep reduces stepsize ..... QuadSafe: %e\n",
             Parm->QuadSafe) ;
    printf ("skip quadstep if |f| <= SmallCost*start cost  SmallCost: %e\n",
             Parm->SmallCost) ;
    printf ("relative change in cost to stop cubic step  CubicCutOff: %e\n",
             Parm->CubicCutOff) ;
    printf ("terminate if no improvement over nslow iter ..... nslow: %i\n",
             Parm->nslow) ;
    printf ("factor multiplying gradient in stop condition . StopFac: %e\n",
             Parm->StopFac) ;
    printf ("cost change factor, approx Wolfe transition . AWolfeFac: %e\n",
             Parm->AWolfeFac) ;
    printf ("restart cg every restart_fac*n iterations . restart_fac: %e\n",
             Parm->restart_fac) ;
    printf ("cost error in quadratic restart is qeps*cost ..... qeps: %e\n",
             Parm->qeps) ;
    printf ("number of quadratic iterations before restart  qrestart: %i\n",
             Parm->qrestart) ;
    printf ("parameter used to decide if cost is quadratic ... qrule: %e\n",
             Parm->qrule) ;
    printf ("stop when cost change <= feps*|f| ................ feps: %e\n",
             Parm->feps) ;
    printf ("starting guess parameter in first iteration ...... psi0: %e\n",
             Parm->psi0) ;
    printf ("starting step in first iteration if nonzero ...... step: %e\n",
             Parm->step) ;
    printf ("lower bound factor in quad step ................ psi_lo: %e\n",
             Parm->psi_lo) ;
    printf ("upper bound factor in quad step ................ psi_hi: %e\n",
             Parm->psi_hi) ;
    printf ("initial guess factor for quadratic functions ..... psi1: %e\n",
             Parm->psi1) ;
    printf ("initial guess factor for general iteration ....... psi2: %e\n",
             Parm->psi2) ;
    printf ("max iterations .................................. maxit: %i\n",
             (int) Parm->maxit) ;
    printf ("max number of contracts in the line search .... nshrink: %i\n",
             Parm->nshrink) ;
    printf ("max expansions in line search .................. ntries: %i\n",
             Parm->ntries) ;
    printf ("maximum growth of secant step in expansion . ExpandSafe: %e\n",
             Parm->ExpandSafe) ;
    printf ("growth factor for secant step during expand . SecantAmp: %e\n",
             Parm->SecantAmp) ;
    printf ("growth factor for rho during expansion phase .. RhoGrow: %e\n",
             Parm->RhoGrow) ;
    printf ("distance threshhold for entering subspace ........ eta0: %e\n",
             Parm->eta0) ;
    printf ("distance threshhold for leaving subspace ......... eta1: %e\n",
             Parm->eta1) ;
    printf ("distance threshhold for invariant space .......... eta2: %e\n",
             Parm->eta2) ;
    printf ("number of vectors stored in memory ............. memory: %i\n",
             Parm->memory) ;
    printf ("check subspace condition mem*SubCheck its .... SubCheck: %i\n",
             Parm->SubCheck) ;
    printf ("skip subspace checking for mem*SubSkip its .... SubSkip: %i\n",
             Parm->SubSkip) ;
    printf ("max number of times that eps is updated .......... neps: %i\n",
             Parm->neps) ;
    printf ("max number of iterations in line search ......... nline: %i\n",
             Parm->nline) ;
    printf ("print level (0 = none, 3 = maximum) ........ PrintLevel: %i\n",
             Parm->PrintLevel) ;
    printf ("Logical parameters:\n") ;
    if ( Parm->PertRule )
        printf ("    Error estimate for function value is eps*Ck\n") ;
    else
        printf ("    Error estimate for function value is eps\n") ;
    if ( Parm->QuadStep )
        printf ("    Use quadratic interpolation step\n") ;
    else
        printf ("    No quadratic interpolation step\n") ;
    if ( Parm->UseCubic)
        printf ("    Use cubic interpolation step when possible\n") ;
    else
        printf ("    Avoid cubic interpolation steps\n") ;
    if ( Parm->AdaptiveBeta )
        printf ("    Adaptively adjust direction update parameter beta\n") ;
    else
        printf ("    Use fixed parameter theta in direction update\n") ;
    if ( Parm->PrintFinal )
        printf ("    Print final cost and statistics\n") ;
    else
        printf ("    Do not print final cost and statistics\n") ;
    if ( Parm->PrintParms )
        printf ("    Print the parameter structure\n") ;
    else
        printf ("    Do not print parameter structure\n") ;
    if ( Parm->AWolfe)
        printf ("    Approximate Wolfe line search\n") ;
    else
        printf ("    Wolfe line search") ;
        if ( Parm->AWolfeFac > 0. )
            printf (" ... switching to approximate Wolfe\n") ;
        else
            printf ("\n") ;
    if ( Parm->StopRule )
        printf ("    Stopping condition uses initial grad tolerance\n") ;
    else
        printf ("    Stopping condition weighted by absolute cost\n") ;
    if ( Parm->debug)
        printf ("    Check for decay of cost, debugger is on\n") ;
    else
        printf ("    Do not check for decay of cost, debugger is off\n") ;
}
