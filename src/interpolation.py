from __future__ import division
import pylinear.array as num
import pylinear.operator as op
from pytools import FunctionValueCache
import cProfile as profile




def newton_interpolation_coefficients(x, y):
    assert len(x) == len(y)
    n = len(y)
    divided_differences = [y]
    last = y

    for step in range(1, n):
        next = [(last[i+1]-last[i])/(x[i+step]-x[i])
                for i in range(n-step)]
        divided_differences.append(next)
        last = next

    return [dd_col[-1] for dd_col in divided_differences]




def newton_interpolation_polynomial(x, y):
    coeff = newton_interpolation_coefficients(x, y)

    import pymbolic

    var_x = pymbolic.var("x")
    linear_factors = [
            pymbolic.Polynomial(var_x, ((0, pt), (1, 1)))
            for pt in x]
    pyramid_linear_factors = [1]
    for l in linear_factors:
        pyramid_linear_factors.append(
                pyramid_linear_factors[-1]*l)

    return pymbolic.linear_combination(coeff, pyramid_linear_factors)



def newton_interpolation_function(x, y):
    import pymbolic
    return pymbolic.compile(newton_interpolation_polynomial(x, y))




@FunctionValueCache
def jacobi_polynomial(alpha, beta, N):
    """Return Jacobi Polynomial of type (alpha,beta) > -1
    (alpha+beta != -1) for order N.
    """

    # port of J. Hesthaven's JacobiP routine

    from math import sqrt
    from scipy.special import gamma
    from pymbolic import var, Polynomial

    one = Polynomial(var("x"), ((0, 1),))
    x = Polynomial(var("x"))

    polys = []
    # Initial value P_0(x)
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)* \
            gamma(beta+1)/gamma(alpha+beta+1)
    polys.append(1/sqrt(gamma0)*one)

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
        anew = 2/(h1+2)*sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)* \
                (i+1+beta)/(h1+1)/(h1+3));
        bnew = - (alpha^2-beta^2)/h1/(h1+2)
        polys.append(1/anew*( -aold*polys[-2] + (x-bnew)*polys[-1]))

        aold = anew

    return polys[-1]




def legendre_polynomial(N):
    return jacobi_polynomial(0, 0, N)




@FunctionValueCache
def legendre_polynomial_f(N):
    import pymbolic
    return pymbolic.compile(jacobi_polynomial(0, 0, N))




def generic_vandermonde(points, functions):
    v = num.zeros((len(points), len(functions)))
    for i, x in enumerate(points):
        for j, f in enumerate(functions):
            v[i,j] = f(x)
    return v




def legendre_vandermonde(points, N):
    return generic_vandermonde(points, 
            [legendre_polynomial_f(i) for i in range(N+1)])




def jacobi_gauss_quadrature(alpha, beta, N):
    """Compute the N'th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi
    polynomial, of type (alpha,beta) > -1 ( <> -0.5).

    Returns x, w.
    """

    from scipy.special.orthogonal import j_roots

    return j_roots(N+1, alpha, beta)




def legendre_gauss_quadrature(N):
    return jacobi_quadrature(0, 0, N)




def jacobi_gauss_lobatto_points(alpha, beta, N):
    """Compute the N'th order Gauss Lobatto quadrature
    points, x, associated with the Jacobi polynomial,
    of type (alpha,beta) > -1 ( <> -0.5).
    """

    x = num.zeros((N+1,))
    x[0] = -1
    x[-1] = 1

    if N == 1:
        return x

    xint, w = jacobi_gauss_quadrature(alpha+1,beta+1,N-2);
    x[1:-1] = num.array(xint).real
    return x




def legendre_gauss_lobatto_points(N):
    return jacobi_gauss_lobatto_points(0, 0, N)




class WarpFactorCalculator:
    """Calculator for Warburton's warp factor.

    See T. Warburton,
    "An explicit construction of interpolation nodes on the simplex"
    Journal of Engineering Mathematics Vol 56, No 3, p. 247-262, 2006
    """

    def __init__(self, N):
        # Find lgl and equidistant interpolation points
        r_lgl = legendre_gauss_lobatto_points(N)
        r_eq  = num.linspace(-1,1,N+1)
        self.int_f = newton_interpolation_function(r_eq, r_lgl - r_eq)

        assert abs(self.int_f(-1)) < 1e-10
        assert abs(self.int_f(1)) < 1e-10

    def __call__(self, x):
        if abs(x) > 1-1e-10:
            return 0
        else:
            return self.int_f(x)/(1-x**2)




def tri_nodes(N):
    """Compute (x,y) nodes in equilateral triangle for polynomials
    of order N
    """

    from math import sqrt, sin, cos, pi

    # port of J. Hesthaven's Nodes2D routine

    alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
              1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
              
    # Set optimized parameter alpha, depending on order N
    try:
        alpha = alpha_opt[N+1]
    except IndexError:
        alpha = 5/3

    warp = WarpFactorCalculator(N)

    c2 = cos(2*pi/3)
    s2 = sin(2*pi/3)
    c4 = cos(4*pi/3)
    s4 = sin(4*pi/3)

    points = []
    # Create equidistributed nodes on equilateral triangle
    for n in range(0, N+1):
        for m in range(0, N+1-n):
            lambda1 = n/N
            lambda3 = m/N
            lambda2 = 1-lambda1-lambda3

            x = -lambda2+lambda3
            y = (-lambda2-lambda3+2*lambda1)/sqrt(3.0)

            blend1 = 4*lambda2*lambda3
            blend2 = 4*lambda1*lambda3
            blend3 = 4*lambda1*lambda2

            # Amount of warp for each node, for each edge
            warp1 = blend1*warp(lambda3 - lambda2)*(1 + (alpha*lambda1)**2)
            warp2 = blend2*warp(lambda1 - lambda3)*(1 + (alpha*lambda2)**2)
            warp3 = blend3*warp(lambda2 - lambda1)*(1 + (alpha*lambda3)**2)

            points.append(num.array(
                    [   x + 1*warp1 + c2*warp2 + c4*warp3,
                        y           + s2*warp2 + s4*warp3]))

    return points



def plot_1d(f, a, b, steps=100):
    h = float(b - a)/steps

    points = []
    data = []
    for n in range(steps):
        x = a + h * n
        points.append(x)
        data.append(f(x))

    from Gnuplot import Gnuplot, Data
    gp = Gnuplot()
    gp.plot(Data(points, data))
    raw_input()




if __name__ == "__main__":
    outf = file("nodes.dat", "w")
    for x, y in tri_nodes(33):
        outf.write("%f\t%f\n" % (x,y))
    profile.run("list(tri_nodes(33))", "iprof")
    #print legendre_vandermonde([1,2,3,4], 4)
    #print num.linspace(-1, 1, N+1)
    #print legendre_gauss_lobatto_points(17)

    #x = [-1.5, -0.75, 0, 0.75, 1.5]
    #y = [-14.1014, -0.931596, 0, 0.931596, 14.1014]
    #nf = newton_interpolation_function(x,
            #[-14.1014, -0.931596, 0, 0.931596, 14.1014])


