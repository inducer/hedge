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
import pylinear.computation as comp
import hedge._internal
from pytools.arithmetic_container import work_with_arithmetic_containers




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




class AffineMap(hedge._internal.AffineMap):
    def __init__(self, matrix, vector):
        """Construct an affine map given by f(x) = matrix * x + vector."""
        from pylinear.computation import determinant
        hedge._internal.AffineMap.__init__(self,
                matrix, vector, determinant(matrix))

    def inverted(self):
        """Return a new AffineMap that is the inverse of this one.
        """
        return AffineMap(1/self.matrix, -self.matrix <<num.solve>> self.vector)

    def __getinitargs__(self):
        return self.matrix, self.vector





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




def find_matching_vertices_along_axis(axis, points_a, points_b, numbers_a, numbers_b):
    a_to_b = {}
    not_found = []


    for i, pi in enumerate(points_a):
        found = False
        for j, pj in enumerate(points_b):
            dist = pi-pj
            dist[axis] = 0
            if comp.norm_2(dist) < 1e-12:
                a_to_b[numbers_a[i]] = numbers_b[j]
                found = True
                break
        if not found:
            not_found.append(numbers_a[i])

    return a_to_b, not_found




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




# small utilities -------------------------------------------------------------
class Closable:
    def __init__(self):
        self.is_closed = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.is_closed:
            # even if close attempt fails, consider ourselves closed still
            try:
                self.do_close()
            finally:
                self.is_closed = True




# index map tools -------------------------------------------------------------
@work_with_arithmetic_containers
def apply_index_map(imap, vector):
    from hedge._internal import VectorTarget, perform_index_map

    result = num.zeros_like(vector, shape=(imap.to_length,))
    perform_index_map(imap, VectorTarget(vector, result))
    return result




def apply_inverse_index_map(imap, vector):
    from hedge._internal import VectorTarget, perform_inverse_index_map

    result = num.zeros_like(vector, shape=(imap.from_length,))
    perform_inverse_index_map(imap, VectorTarget(vector, result))
    return result




# mesh reorderings ------------------------------------------------------------
def cuthill_mckee(graph):
    """Return a Cuthill-McKee ordering for the given graph.

    See (for example)
    Y. Saad, Iterative Methods for Sparse Linear System,
    2nd edition, p. 76.

    `graph' is given as an adjacency mapping, i.e. each node is
    mapped to a list of its neighbors.
    """
    from pytools import argmin

    # this list is called "old_numbers" because it maps a 
    # "new number to its "old number"
    old_numbers = []
    visited_nodes = set()
    levelset = []

    all_nodes = set(graph.keys())

    def levelset_cmp(node_a, node_b):
        return cmp(len(graph[node_a]), len(graph[node_b]))

    while len(old_numbers) < len(graph):
        if not levelset:
            unvisited = list(set(graph.keys()) - visited_nodes)

            if not unvisited:
                break

            start_node = unvisited[
                    argmin(len(graph[node]) for node in unvisited)]
            visited_nodes.add(start_node)
            old_numbers.append(start_node)
            levelset = [start_node]

        next_levelset = set()
        levelset.sort(levelset_cmp)

        for node in levelset:
            for neighbor in graph[node]:
                if neighbor in visited_nodes:
                    continue

                visited_nodes.add(neighbor)
                next_levelset.add(neighbor)
                old_numbers.append(neighbor)

        levelset = list(next_levelset)

    return old_numbers




def reverse_lookup_table(lut):
    result = [None] * len(lut)
    for key, value in enumerate(lut):
        result[value] = key
    return result




# block matrix ----------------------------------------------------------------
class BlockMatrix:
    """A block matrix is the sum of different smaller
    matrices positioned within one big matrix.
    """

    def __init__(self, chunks):
        """Return a new block matrix made up of components (`chunks')
        given as triples (i,j,smaller_matrix), where the top left (0,0)
        corner of the smaller_matrix is taken to be at position (i,j).

        smaller_matrix may be anything that can be left-multiplied to
        a Pylinear vector, including BlockMatrix instances.
        """
        self.chunks = []
        for i, j, chunk in chunks:
            if isinstance(chunk, BlockMatrix):
                self.chunks.extend(
                        (i+subi, j+subj, subchunk)
                        for subi, subj, subchunk in chunk.chunks)
            else:
                self.chunks.append((i, j, chunk))

    @property
    def T(self):
        return BlockMatrix(
                (j, i, chunk.T) for i, j, chunk in self.chunks)

    @property
    def H(self):
        return BlockMatrix(
                (j, i, chunk.H) for i, j, chunk in self.chunks)

    @property
    def shape(self):
        return (
                max(i+chunk.shape[0] for i, j, chunk in self.chunks),
                max(j+chunk.shape[0] for i, j, chunk in self.chunks)
                )

    def __add__(self, other):
        if isinstance(other, BlockMatrix):
            return BlockMatrix(self.chunks + other.chunks)
        else:
            return NotImplemented

    def __neg__(self):
        return BlockMatrix((i,j,-chunk) for i, j, chunk in self.chunks)

    def __sub__(self, other):
        if isinstance(other, BlockMatrix):
            return BlockMatrix(self.chunks + (-other).chunks)
        else:
            return NotImplemented

    def __mul__(self, other):
        if num.Vector.is_a(other):
            h, w = self.shape
            assert len(other) == w
            result = num.zeros((h,))

            for i, j, chunk in self.chunks:
                ch, cw = chunk.shape
                result[i:i+ch] += chunk * other[j:j+cw]

            return result
        elif isinstance(other, (float, complex, int)):
            return BlockMatrix(
                    (i,j,other*chunk) 
                    for i, j, chunk in self.chunks)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (float, complex, int)):
            return BlockMatrix(
                    (i,j,other*chunk) 
                    for i, j, chunk in self.chunks)
        else:
            return NotImplemented








