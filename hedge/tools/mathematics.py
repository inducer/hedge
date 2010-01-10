"""Mathematical tools."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""

# If you were wondering: this module is called "mathematical" because
# "math" can erroneously pick up package-local "import math" statements.



import hedge._internal
import numpy




cyl_bessel_j = hedge._internal.cyl_bessel_j
cyl_neumann = hedge._internal.cyl_neumann




def cyl_bessel_j_prime(nu, z):
    if nu == 0:
        if z == 0:
            return 0
        else:
            return -cyl_bessel_j(nu+1, z)+nu/z*cyl_bessel_j(nu, z)
    else:
        return 0.5*(cyl_bessel_j(nu-1, z)-cyl_bessel_j(nu+1, z))




def levi_civita(tuple):
    """Compute an entry of the Levi-Civita tensor for the indices *tuple*.

    Only three-tuples are supported for now.
    """
    if len(tuple) == 3:
        if tuple in [(0,1,2), (2,0,1), (1,2,0)]:
            return 1
        elif tuple in [(2,1,0), (0,2,1), (1,0,2)]:
            return -1
        else:
            return 0
    else:
        raise NotImplementedError




class SubsettableCrossProduct:
    """A cross product that can operate on an arbitrary subsets of its
    two operands and return an arbitrary subset of its result.
    """

    full_subset = (True, True, True)

    def __init__(self, op1_subset=full_subset, op2_subset=full_subset, result_subset=full_subset):
        """Construct a subset-able cross product.

        :param op1_subset: The subset of indices of operand 1 to be taken into account.
          Given as a 3-sequence of bools.
        :param op2_subset: The subset of indices of operand 2 to be taken into account.
          Given as a 3-sequence of bools.
        :param result_subset: The subset of indices of the result that are calculated.
          Given as a 3-sequence of bools.
        """
        def subset_indices(subset):
            return [i for i, use_component in enumerate(subset)
                    if use_component]

        self.op1_subset = op1_subset
        self.op2_subset = op2_subset
        self.result_subset = result_subset

        import pymbolic
        op1 = pymbolic.var("x")
        op2 = pymbolic.var("y")

        self.functions = []
        self.component_lcjk = []
        for i, use_component in enumerate(result_subset):
            if use_component:
                this_expr = 0
                this_component = []
                for j, j_real in enumerate(subset_indices(op1_subset)):
                    for k, k_real in enumerate(subset_indices(op2_subset)):
                        lc = levi_civita((i, j_real, k_real))
                        if lc != 0:
                            this_expr += lc*op1[j]*op2[k]
                            this_component.append((lc, j, k))
                self.functions.append(pymbolic.compile(this_expr,
                    variables=[op1, op2]))
                self.component_lcjk.append(this_component)

    def __call__(self, x, y, three_mult=None):
        """Compute the subsetted cross product on the indexables *x* and *y*.

        :param three_mult: a function of three arguments *sign, xj, yk*
          used in place of the product *sign*xj*yk*. Defaults to just this
          product if not given.
        """
        from pytools.obj_array import join_fields
        if three_mult is None:
            return join_fields(*[f(x, y) for f in self.functions])
        else:
            return join_fields(
                    *[sum(three_mult(lc, x[j], y[k]) for lc, j, k in lcjk)
                    for lcjk in self.component_lcjk])




cross = SubsettableCrossProduct()




def normalize(v):
    return v/numpy.linalg.norm(v)




def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1




class Monomial:
    def __init__(self, exponents, factor=1):
        self.exponents = exponents
        self.ones = numpy.ones((len(self.exponents),))
        self.factor = factor

    def __call__(self, x):
        from operator import mul

        eps = 1e-15
        x = (x+self.ones)/2
        for xi in x:
            assert -eps <= xi <= 1+eps
        return self.factor* \
                reduce(mul, (x[i]**alpha
                    for i, alpha in enumerate(self.exponents)))

    def simplex_integral(self):
        """Integral over the unit simplex."""
        from pytools import factorial
        from operator import mul

        return (self.factor*2**len(self.exponents)*
            reduce(mul, (factorial(alpha) for alpha in self.exponents))
            /
            factorial(len(self.exponents)+sum(self.exponents)))

    def diff(self, coordinate):
        diff_exp = list(self.exponents)
        orig_exp = diff_exp[coordinate]
        if orig_exp == 0:
            return Monomial(diff_exp, 0)
        diff_exp[coordinate] = orig_exp-1
        return Monomial(diff_exp, self.factor*orig_exp)




def get_spherical_coord(x_vec):
    """
    :param x_vec: is an array whose leading dimension iterates over
        the X, Y, Z axes, and whose further dimensions may iterate over
        a number of points.

    :returns: object array of [r, phi, theta].
        phi is the angle in (x,y) in :math:`(-\\pi,\\pi)`.
    """

    if len(x_vec) != 3:
        raise ValueError("only 3-d arrays are supported")

    x = x_vec[0]
    y = x_vec[1]
    z = x_vec[2]

    r = numpy.sqrt(x**2+y**2+z**2)

    from warnings import warn
    if(numpy.any(r)<numpy.power(10.0,-10.0)):
        warn('spherical coordinate transformation ill-defined at r=0')

    phi = numpy.arctan2(y,x)
    theta = numpy.arccos(z/r)

    from pytools.obj_array import join_fields
    return join_fields(r,phi,theta)

def heaviside(x):
    """
    :param x: a list of numbers

    :returns: Heaviside step function where H(0)=0
    """
    return (x>0).astype(numpy.float64)

def heaviside_a(x,a):
    """
    :param x: a list of numbers
    :param a: real number such that H(0)=a

    :returns: Heaviside step function where H(0)=a
    """
    return a*(1.0 - heaviside(-x)) + (1.0 - a)*heaviside(x)





