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
    """Compute the N'th order Gauss-Lobatto quadrature
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
    """Compute the N'th order Gauss-Lobatto quadrature
    points, x, associated with the Legendre polynomials.
    """
    return jacobi_gauss_lobatto_points(0, 0, N)




