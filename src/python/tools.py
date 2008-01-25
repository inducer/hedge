"""Miscellaneous helper facilities."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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





import pylinear.array as num
import pylinear.computation as comp
import pylinear.operator as op
import hedge._internal
from pytools.arithmetic_container import work_with_arithmetic_containers




try:
    cyl_bessel_j = hedge._internal.cyl_bessel_j
    cyl_neumann = hedge._internal.cyl_neumann
except AttributeError:
    # accept failure because of gcc ICE in boost math-toolkit
    pass




def cyl_bessel_j_prime(nu, z):
    if nu == 0:
        if z == 0:
            return 0
        else:
            return -cyl_bessel_j(nu+1, z)+nu/z*cyl_bessel_j(nu, z)
    else:
        return 0.5*(cyl_bessel_j(nu-1, z)-cyl_bessel_j(nu+1, z))




AffineMap = hedge._internal.AffineMap
def _affine_map_jacobian(self):
    try:
        return self._jacobian
    except AttributeError:
        self._jacobian = comp.determinant(self.matrix)
        return self._jacobian
AffineMap.jacobian = property(_affine_map_jacobian)

def _affine_map_inverted(self):
    """Return a new AffineMap that is the inverse of this one.
    """
    return AffineMap(1/self.matrix, -self.matrix <<num.solve>> self.vector)
AffineMap.inverted = _affine_map_inverted

def _affine_map___getinitargs__(self):
    return self.matrix, self.vector
AffineMap.__getinitargs__ = _affine_map___getinitargs__




class Rotation(AffineMap):
    def __init__(self, angle):
        # FIXME: Add axis, make multidimensional
        from math import sin, cos
        AffineMap.__init__(self,
                num.array([
                    [cos(angle), sin(angle)],
                    [-sin(angle), cos(angle)]]),
                num.zeros((2,)))




class Reflection(AffineMap):
    def __init__(self, axis, dim):
        mat = num.identity(dim)
        mat[axis,axis] = -1
        AffineMap.__init__(self, mat, num.zeros((dim,)))




def plot_1d(f, a, b, steps=100, driver=None):
    h = (b - a)/steps

    points = []
    data = []
    for n in range(steps):
        x = a + h * n
        points.append(x)
        data.append(f(x))

    # autodetect driver
    if driver is None:
        try:
            import pylab
            driver = "matplotlib"
        except ImportError:
            pass
    if driver is None:
        try:
            import Gnuplot
            driver = "gnuplot"
        except ImportError:
            pass

    # actually plot
    if driver == "matplotlib":
        from pylab import plot, show
        plot(points, data)
        show()
    elif driver == "gnuplot":
        from Gnuplot import Gnuplot, Data
        gp = Gnuplot()
        gp.plot(Data(points, data))
        raw_input()
    else:
        raise ValueError, "invalid plot driver '%s'" % driver




def dot(x, y, multiplication=None): 
    """Compute the dot product of the iterables C{x} and C{y}.

    @arg multiplication: If given, this specifies the binary function
      applied in place of multiplication. Defaults to C{operator.mul}.
    """
    if multiplication is None:
        from operator import add
        return reduce(add, (xi*yi for xi, yi in zip(x,y)))
    else:
        from operator import add
        return reduce(add, (multiplication(xi, yi) for xi, yi in zip(x,y)))




def levi_civita(tuple):
    """Compute an entry of the Levi-Civita tensor for the indices C{tuple}.

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

        @arg op1_subset: The subset of indices of operand 1 to be taken into account.
          Given as a 3-sequence of bools.
        @arg op2_subset: The subset of indices of operand 2 to be taken into account.
          Given as a 3-sequence of bools.
        @arg result_subset: The subset of indices of the result that are calculated.
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
                self.functions.append(pymbolic.compile(this_expr))
                self.component_lcjk.append(this_component)

    def __call__(self, x, y, three_mult=None):
        """Compute the subsetted cross product on the indexables C{x} and C{y}.

        @arg three_mult: a function of three arguments C{sign, xj, yk}
          used in place of the product C{sign*xj*yk}. Defaults to just this
          product if not given.
        """
        if three_mult is None:
            from pytools.arithmetic_container import ArithmeticList
            return ArithmeticList(f(x, y) for f in self.functions)
        else:
            from pytools.arithmetic_container import ArithmeticList
            return ArithmeticList(
                    sum(three_mult(lc, x[j], y[k]) for lc, j, k in lcjk)
                    for lcjk in self.component_lcjk)




cross = SubsettableCrossProduct()




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




def make_vector_target(argument, result):
    """Creates a VectorTarget for an OperatorTarget with `argument'
    and `result'. Normally, C{argument} and C{result} should be 
    vectors. However, C{argument} may also be the scalar 0, in which
    case a dummy operator is returned.
    """
    from hedge._internal import NullTarget, VectorTarget
    if isinstance(argument, (int, float)) and argument == 0:
        return NullTarget()
    else:
        return VectorTarget(argument, result)





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


  

class EOCRecorder(object):
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
class Closable(object):
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




def mem_checkpoint(name=None):
    """Invoke the garbage collector and wait for a keypress."""

    import gc
    gc.collect()
    if name:
        raw_input("%s -- hit Enter:" % name)
    else:
        raw_input("Enter:")





class FixedSizeSliceAdapter(object):
    """Adapts an indexable object C{idxable} so that C{idxable[i]}
    refers to the slice C{adaptee[i*unit:(i+1)*unit]}. This effectively
    turns one long vector into storage space of lots of identically-sized
    smaller ones.

    Slice operations on the adapter cause new list objects (shallow copies) to 
    be created.

    If A, B, and C are different vectors stored in this adapter, then their
    ordering in the adaptee will be C{A0A1A2B0B1B2}, i.e. each vector is contiguous.
    We refer to this as 'vector-major' order.
    """

    __slots__ = ["adaptee", "unit", "length"]

    def __init__(self, adaptee, unit, length=None):
        self.adaptee = adaptee
        self.unit = unit
        self.length = length

        technical_len, remainder = divmod(len(self.adaptee), self.unit)
        assert remainder == 0

        if self.length is not None:
            assert self.length <= technical_len

    def __len__(self):
        if self.length is not None:
            return self.length
        else:
            technical_len, remainder = divmod(len(self.adaptee), self.unit)
            assert remainder == 0
            return technical_len

    def __iter__(parent):
        class FSSAIterator:
            def __init__(self):
                self.idx = 0
    
            def next(self):
                if self.idx >= len(parent):
                    raise StopIteration
                result = parent[self.idx]
                self.idx += 1
                return result
    
        return FSSAIterator()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx >= len(self):
                raise IndexError, idx
            return self.adaptee[self.unit*idx:self.unit*(idx+1)]
        elif isinstance(idx, slice):
            range_args= idx.indices(self.__len__())
            return [self.adaptee[self.unit*i:self.unit*(i+1)]
                    for i in xrange(*range_args)]
        else:
            raise TypeError, "invalid index type"

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            self.adaptee[self.unit*idx:self.unit*(idx+1)] = value
        elif isinstance(idx, slice):
            range_args= idx.indices(self.__len__())
            for i, subval in zip(xrange(*range_args), values):
                self.adaptee[self.unit*i:self.unit*(i+1)] = subval
        else:
            raise TypeError, "invalid index type"

    def get_alist_of_components(self):
        """Return the adaptee's data as an ArithmeticList of
        each vectors for each component.
        """
        from pytools.arithmetic_container import ArithmeticList
        return ArithmeticList(
                self.adaptee[i:len(self)*self.unit:self.unit] for i in range(self.unit))

    def get_component_major_vector(self):
        """Return the adaptee's data in component-major order.
        
        This gives a vector of order C{A0B0C0A1B1C1...}
        """
        return num.hstack(self.get_alist_of_components())





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
class BlockMatrix(object):
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

    def add_to_build_matrix(self, bmat):
        for i, j, chunk in self.chunks:
            bmat.add_block(i, j, chunk)





import pylinear.operator
PylinearOperator = pylinear.operator.Operator(num.Float64)




# parallel cg -----------------------------------------------------------------
class CGStateContainer:
    def __init__(self, pcon, operator, precon=None):
        if precon is None:
            precon = op.IdentityOperator.make(operator.dtype, operator.size1())

        self.pcon = pcon
        self.operator = operator
        self.precon = precon

        if len(pcon.ranks) == 1:
            def inner(a, b):
                return a*num.conjugate(b)
        else:
            from boost.mpi import all_reduce
            from operator import add

            def inner(a, b):
                local = a*num.conjugate(b)
                return all_reduce(pcon.communicator, local, add)

        self.inner = inner

    def reset(self, rhs, x=None):
        self.rhs = rhs

        if x is None:
            x = num.zeros((self.operator.size1(),))
        self.x = x

        self.residual = rhs - self.operator(x)

        self.d = self.precon(self.residual)

        self.delta = self.inner(self.residual, self.d)
        return self.delta

    def one_iteration(self, compute_real_residual=False):
        # typed up from J.R. Shewchuk, 
        # An Introduction to the Conjugate Gradient Method
        # Without the Agonizing Pain, Edition 1 1/4 [8/1994]
        # Appendix B3

        q = self.operator(self.d)
        alpha = self.delta / self.inner(self.d, q)

        self.x += alpha * self.d

        if compute_real_residual:
            self.residual = self.rhs - self.operator(self.x)
        else:
            self.residual -= alpha*q

        s = self.precon(self.residual)
        delta_old = self.delta
        self.delta = self.inner(self.residual, s)

        beta = self.delta / delta_old;
        self.d = s + beta * self.d;

        return self.delta

    def run(self, max_iterations=None, tol=1e-7, debug_callback=None, debug=0):
        if max_iterations is None:
            max_iterations = 10 * self.operator.size1()

        if comp.norm_2(self.rhs) == 0:
            return self.rhs

        iterations = 0
        delta_0 = delta = self.delta
        while iterations < max_iterations:
            if debug_callback is not None:
                debug_callback(self.x, self.residual, self.d)

            compute_real_residual = \
                    iterations % 50 == 0 or \
                    abs(delta) < tol*tol * abs(delta_0)
            delta = self.one_iteration(
                    compute_real_residual=compute_real_residual)

            if compute_real_residual and abs(delta) < tol*tol * abs(delta_0):
                return self.x

            if debug and iterations % debug == 0 and self.pcon.is_head_rank:
                print "debug: delta=%g" % delta
            iterations += 1

        raise RuntimeError("cg failed to converge")
            



def parallel_cg(pcon, operator, b, precon=None, x=None, tol=1e-7, max_iterations=None, 
        debug=False, debug_callback=None):
    if x is None:
        x = num.zeros((operator.size1(),))

    if len(pcon.ranks) == 1 and debug_callback is None:
        # use canned single-processor cg if possible
        a_inv = op.CGOperator.make(operator, max_it=max_iterations, 
                tolerance=tol, precon_op=precon)
        if debug:
            a_inv.debug_level = 1
        a_inv.apply(b, x)
        return x

    cg = CGStateContainer(pcon, operator, precon)
    cg.reset(b, x)
    return cg.run(max_iterations, tol, debug_callback, debug)
