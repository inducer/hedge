import pylinear.array as num




class WarpFactorCalculator:
    """Calculator for Warburton's warp factor.

    See T. Warburton,
    "An explicit construction of interpolation nodes on the simplex"
    Journal of Engineering Mathematics Vol 56, No 3, p. 247-262, 2006
    """

    def __init__(self, N):
        from hedge.quadrature import legendre_gauss_lobatto_points
        from hedge.interpolation import newton_interpolation_function

        # Find lgl and equidistant interpolation points
        r_lgl = legendre_gauss_lobatto_points(N)
        r_eq  = num.linspace(-1,1,N+1)

        self.int_f = newton_interpolation_function(r_eq, r_lgl - r_eq)

    def __call__(self, x):
        if abs(x) > 1-1e-10:
            return 0
        else:
            return self.int_f(x)/(1-x**2)




class Triangle:
    """An arbitrary-order triangular finite element.

    Coordinate systems used:
    ------------------------

    unit triangle coordinates (r,s)

    C
    |\
    | \
    |  \
    |   \
    A----B

    equilateral unit coordinates (x,y)

            C
           / \
          /   \
         /     \
        /   O   \
       /         \
      A-----------B

    O = (0,0)
    A = (-
    """

    def __init__(self, order):
        self.order = order

    def face_indices(self):
        faces = [[], [], []]

        i = 0
        for n in range(0, self.order+1):
            for m in range(0, self.order+1-n):
                # face finding
                if n == 0:
                    faces[0].append(i)
                if n+m == self.order:
                    faces[1].append(i)
                if m == 0:
                    faces[2].append(i)

                i += 1

        # make sure faces are numbered counterclockwise
        face[2] = face[2][::-1]

        return faces

    def equidistant_barycentric_nodes(self):
        """Compute equidistant (x,y) nodes in barycentric coordinates
        of order N.
        """
        for n in range(0, self.order+1):
            for m in range(0, self.order+1-n):
                # compute barycentric coordinates
                lambda1 = n/self.order
                lambda3 = m/self.order
                lambda2 = 1-lambda1-lambda3

                yield lambda1, lambda2, lambda3

    @staticmethod
    def barycentric_to_equilateral((lambda1, lambda2, lambda3)):
        from math import sqrt

        return num.array([
            -lambda2+lambda3,
            (-lambda2-lambda3+2*lambda1)/sqrt(3.0)])

    #equilateral_to_unit = 

    def equidistant_equilateral_nodes(self):
        """Compute equidistant (x,y) nodes in equilateral triangle for polynomials
        of order N.
        """

        for bary in self.equidistant_barycentric_nodes():
            yield self.barycentric_to_equilateral(bary)

    def nodes(self):
        """Compute warped nodes in equilateral coordinates (x,y) for polynomials
        of order N.
        """

        # port of J. Hesthaven's Nodes2D routine

        from math import sqrt, sin, cos, pi

        # Set optimized parameter alpha, depending on order N
        alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                  1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
                  
        try:
            alpha = alpha_opt[self.order+1]
        except IndexError:
            alpha = 5/3

        warp = WarpFactorCalculator(self.order)

        edge1dir = num.array([1,0])
        edge2dir = num.array([cos(2*pi/3), sin(2*pi/3)])
        edge3dir = num.array([cos(4*pi/3), sin(4*pi/3)])

        for bary in self.equidistant_barycentric_nodes():
            lambda1, lambda2, lambda3 = bary

            # find equidistant (x,y) coordinates in equilateral triangle
            point = self.barycentric_to_equilateral(bary)

            # compute blend factors
            blend1 = 4*lambda2*lambda3
            blend2 = 4*lambda1*lambda3
            blend3 = 4*lambda1*lambda2

            # calculate amount of warp for each node, for each edge
            warp1 = blend1*warp(lambda3 - lambda2)*(1 + (alpha*lambda1)**2)
            warp2 = blend2*warp(lambda1 - lambda3)*(1 + (alpha*lambda2)**2)
            warp3 = blend3*warp(lambda2 - lambda1)*(1 + (alpha*lambda3)**2)

            # return warped point
            yield point + warp1*edge1dir + warp2*edge2dir + warp3*edge3dir





