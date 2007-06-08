from __future__ import division
import pylinear.array as num
import pymbolic
from pytools import memoize




@memoize
def jacobi_polynomial(alpha, beta, N):
    """Return Jacobi Polynomial of type (alpha,beta) > -1
    (alpha+beta != -1) for order N.
    """

    # port of J. Hesthaven's JacobiP routine

    from math import sqrt
    from scipy.special import gamma
    from pymbolic import var, Polynomial

    x = Polynomial(var("x"))

    polys = []
    # Initial value P_0(x)
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)* \
            gamma(beta+1)/gamma(alpha+beta+1)
    polys.append(1/sqrt(gamma0))

    if N == 0: 
        return polys[-1]

    # Initial value P_1(x)
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    polys.append(((alpha+beta+2)/2*x + (alpha-beta)/2)/sqrt(gamma1))

    if N == 1: 
        return polys[-1]

    # Repeat value in recurrence.
    aold = 2/(2+alpha+beta)*sqrt((alpha+1)*(beta+1)/(alpha+beta+3))

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, N):
        h1 = 2*i+alpha+beta
        anew = 2/(h1+2)*sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*\
                (i+1+beta)/(h1+1)/(h1+3))
        bnew = - (alpha**2-beta**2)/h1/(h1+2)
        polys.append(1/anew*(-aold*polys[-2] + (x-bnew)*polys[-1]))
        aold = anew

    return polys[-1]




class jacobi_function:
    """Return Jacobi Polynomial of type (alpha,beta) > -1
    (alpha+beta != -1) for order N.
    """

    def __init__(self, alpha, beta, N):
        self.alpha = alpha
        self.beta = beta
        self.N = N

    def __call__(self, x):
        # port of J. Hesthaven's JacobiP routine

        from math import sqrt
        from scipy.special import gamma
        from pymbolic import var, Polynomial

        alpha = self.alpha
        beta = self.beta
        N = self.N

        polys = []
        # Initial value P_0(x)
        gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)* \
                gamma(beta+1)/gamma(alpha+beta+1)
        polys.append(1/sqrt(gamma0))

        if N == 0: 
            return polys[-1]

        # Initial value P_1(x)
        gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
        polys.append(((alpha+beta+2)/2*x + (alpha-beta)/2)/sqrt(gamma1))

        if N == 1: 
            return polys[-1]

        # Repeat value in recurrence.
        aold = 2/(2+alpha+beta)*sqrt((alpha+1)*(beta+1)/(alpha+beta+3))

        # Forward recurrence using the symmetry of the recurrence.
        for i in range(1, N):
            h1 = 2*i+alpha+beta
            anew = 2/(h1+2)*sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*\
                    (i+1+beta)/(h1+1)/(h1+3))
            bnew = - (alpha**2-beta**2)/h1/(h1+2)
            polys.append(1/anew*(-aold*polys[-2] + (x-bnew)*polys[-1]))
            aold = anew

        return polys[-1]




@memoize
def jacobi_function_2(alpha, beta, N):
    return pymbolic.compile(jacobi_polynomial(alpha, beta, N), ["x"])




@memoize
def diff_jacobi_polynomial(alpha, beta, N):
    from math import sqrt

    if N == 0:
        return 0
    else:
        return sqrt(N*(N+alpha+beta+1))*\
                jacobi_polynomial(alpha+1, beta+1, N-1)




class diff_jacobi_function:
    def __init__(self, alpha, beta, N):
        from math import sqrt
        if N == 0:
            self.jf = lambda x: 0
            self.factor = 0
        else:
            self.jf = jacobi_function(alpha+1, beta+1, N-1)
            self.factor = sqrt(N*(N+alpha+beta+1))

    def __call__(self, x):
        return self.factor*self.jf(x)

@memoize
def diff_jacobi_function_2(alpha, beta, N):
    return pymbolic.compile(diff_jacobi_polynomial(alpha, beta, N), ["x"])




def legendre_polynomial(N):
    return jacobi_polynomial(0, 0, N)




def diff_legendre_polynomial(N, derivative=1):
    return diff_jacobi_polynomial(0, 0, N, derivative)




def legendre_function(N):
    return jacobi_function(0, 0, N)




def diff_legendre_function(N):
    return diff_jacobi_function(0, 0, N)




def generic_vandermonde(points, functions):
    """Return a Vandermonde matrix
      V[i,j] := f[j](x[i])
    where functions=[f[j] for j] and points=[x[i] for i].
    """
    v = num.zeros((len(points), len(functions)))
    for i, x in enumerate(points):
        for j, f in enumerate(functions):
            v[i,j] = f(x)
    return v




def generic_multi_vandermonde(points, functions):
    """Return multiple Vandermonde matrices.
      V[i,j] := f[j](x[i])
    where functions=[f[j] for j] and points=[x[i] for i].
    The functions `f' are multi-valued, one matrix is returned
    for each return value.
    """
    count = len(functions[0](points[0]))
    result = [num.zeros((len(points), len(functions))) for n in range(count)]

    for i, x in enumerate(points):
        for j, f in enumerate(functions):
            for n, f_n in enumerate(f(x)):
                result[n][i,j] = f_n
    return result




def legendre_vandermonde(points, N):
    return generic_vandermonde(points, 
            [legendre_function(i) for i in range(N+1)])

