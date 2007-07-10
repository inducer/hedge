# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




import pylinear.array as num
import hedge._internal




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




class AffineMap(object):
    def __init__(self, matrix, vector):
        """Construct an affine map given by f(x) = matrix * x + vector."""
        self.matrix = matrix
        self.vector = vector

    def __call__(self, x):
        """Apply this map."""
        return self.matrix*x + self.vector

    def inverted(self):
        """Return a new AffineMap that is the inverse of this one.
        """
        return AffineMap(1/self.matrix, -self.matrix <<num.solve>> self.vector)

    @property
    def jacobian(self):
        "Get the (constant) jacobian of the map."
        try:
            return self._jacobian
        except AttributeError:
            from pylinear.computation import determinant
            self._jacobian = determinant(self.matrix)
            return self._jacobian

class Rotation(AffineMap):
    def __init__(self, angle):
        # FIXME: Add axis, make multidimensional
        from math import sin, cos
        AffineMap.__init__(self,
                num.array([
                    [cos(angle), sin(angle)],
                    [-sin(angle), cos(angle)]]),
                num.zeros((2,)))





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




def reduction_matrix(indices, big_len):
    import pylinear.array as num
    result = num.zeros((len(indices), big_len), flavor=num.SparseBuildMatrix)
    for i, j in enumerate(indices):
        result[i,j] = 1
    return result




def dot(x, y): 
    from operator import add
    return reduce(add, (xi*yi for xi, yi in zip(x,y)))




def cross(a, b): 
    from pytools.arithmetic_container import ArithmeticList
    return ArithmeticList([
            a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]
            ])




def normalize(v):
    from pylinear.computation import norm_2

    return v/norm_2(v)




def sign(x):
    if x > 0: 
        return 1
    elif x == 0:
        return 0
    else: 
        return -1

# eoc estimation --------------------------------------------------------------
def estimate_order_of_convergence(abscissae, errors):
    """Assuming that abscissae and errors are connected by a law of the form

    error = constant * abscissa ^ (-order),

    this function finds, in a least-squares sense, the best approximation of
    constant and order for the given data set. It returns a tuple (constant, order).
    Both inputs must be PyLinear vectors.
    """
    import pylinear.toybox as toybox

    assert len(abscissae) == len(errors)
    if len(abscissae) <= 1:
        raise RuntimeError, "Need more than one value to guess order of convergence."

    coefficients = toybox.fit_polynomial(num.log10(abscissae), num.log10(errors), 1)
    return 10**coefficients[0], -coefficients[1]


  




class EOCRecorder:
    def __init__(self):
        self.history = []

    def add_data_point(self, abscissa, error):
        self.history.append((abscissa, error))

    def estimate_order_of_convergence(self, gliding_mean = None):
        abscissae = num.array([ a for a,e in self.history ])
        errors = num.array([ e for a,e in self.history ])

        size = len(abscissae)
        if gliding_mean is None:
            gliding_mean = size

        data_points = size - gliding_mean + 1
        result = num.zeros((data_points, 2), num.Float)
        for i in range(data_points):
            result[i,0], result[i,1] = estimate_order_of_convergence(
                abscissae[i:i+gliding_mean], errors[i:i+gliding_mean])
        return result

    def pretty_print(self, abscissa_label="N", error_label="Error", gliding_mean=2):
        from pytools import Table

        tbl = Table()
        tbl.add_row((abscissa_label, error_label, "Running EOC"))

        gm_eoc = self.estimate_order_of_convergence(gliding_mean)
        for i, (absc, err) in enumerate(self.history):
            if i < gliding_mean-1:
                tbl.add_row((str(absc), str(err), ""))
            else:
                tbl.add_row((str(absc), str(err), str(gm_eoc[i-gliding_mean+1,1])))

        if len(self.history) > 1:
            return str(tbl) + "\n\nOverall EOC: %s" % self.estimate_order_of_convergence()[0,1]
        else:
            return str(tbl)

    def write_gnuplot_file(self, filename):
        outfile = file(filename, "w")
        for absc, err in self.history:
            outfile.write("%f %f\n" % (absc, err))
        result = self.estimate_order_of_convergence()
        const = result[0,0]
        order = result[0,1]
        outfile.write("\n")
        for absc, err in self.history:
            outfile.write("%f %f\n" % (absc, const * absc**(-order)))

