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





import numpy
import numpy.linalg as la
import pyublas
import hedge._internal



ZeroVector = hedge._internal.ZeroVector
cyl_bessel_j = hedge._internal.cyl_bessel_j
cyl_neumann = hedge._internal.cyl_neumann




def is_zero(x):
    return isinstance(x, (int, float)) and x == 0




def relative_error(norm_diff, norm_true):
    if norm_true == 0:
        if norm_diff == 0:
            return 0
        else:
            return float("inf")
    else:
        return norm_diff/norm_true




def is_mul_add_supported(vec):
    while isinstance(vec, numpy.ndarray) and vec.dtype == object:
        vec = vec[0]
    return hasattr(vec, "mul_add")




def mul_add(afac, a, bfac, b, add_timer=None):
    if isinstance(a, numpy.ndarray) and a.dtype == object:
        return numpy.array([
            mul_add(afac, a_i, bfac, b_i, add_timer=add_timer)
            for a_i, b_i in zip(a, b)],
            dtype=object)
    else:
        return a.mul_add(afac, b, bfac, add_timer=add_timer)




def cyl_bessel_j_prime(nu, z):
    if nu == 0:
        if z == 0:
            return 0
        else:
            return -cyl_bessel_j(nu+1, z)+nu/z*cyl_bessel_j(nu, z)
    else:
        return 0.5*(cyl_bessel_j(nu-1, z)-cyl_bessel_j(nu+1, z))




AffineMap = hedge._internal.AffineMap
def _affine_map___getinitargs__(self):
    return self.matrix, self.vector

AffineMap.__getinitargs__ = _affine_map___getinitargs__




class Rotation(AffineMap):
    def __init__(self, angle):
        # FIXME: Add axis, make multidimensional
        from math import sin, cos
        AffineMap.__init__(self,
                numpy.array([
                    [cos(angle), sin(angle)],
                    [-sin(angle), cos(angle)]]),
                numpy.zeros((2,)))




class Reflection(AffineMap):
    def __init__(self, axis, dim):
        mat = numpy.identity(dim)
        mat[axis,axis] = -1
        AffineMap.__init__(self, mat, numpy.zeros((dim,)))




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




# obj array helpers -----------------------------------------------------------
def is_obj_array(val):
    try:
        return val.dtype == object
    except AttributeError:
        return False




def to_obj_array(ary):
    ls = log_shape(ary)
    result = numpy.empty(ls, dtype=object)

    from pytools import indices_in_shape
    for i in indices_in_shape(ls):
        result[i] = ary[i]

    return result




def is_field_equal(a, b):
    if is_obj_array(a):
        return is_obj_array(b) and (a == b).all()
    else:
        return not is_obj_array(b) and a == b




def make_obj_array(res_list):
    result = numpy.empty((len(res_list),), dtype=object)
    for i, v in enumerate(res_list):
        result[i] = v

    return result




def setify_field(f):
    from hedge.tools import is_obj_array
    if is_obj_array(f):
        return set(f)
    else:
        return set([f])




def hashable_field(f):
    if is_obj_array(f):
        return tuple(f)
    else:
        return f




def field_equal(a, b):
    a_is_oa = is_obj_array(a)
    assert a_is_oa == is_obj_array(b)

    if a_is_oa:
        return (a == b).all()
    else:
        return a == b




def join_fields(*args):
    res_list = []
    for arg in args:
        if isinstance(arg, list):
            res_list.extend(arg)
        elif isinstance(arg, numpy.ndarray):
            if log_shape(arg) == ():
                res_list.append(arg)
            else:
                res_list.extend(arg)
        else:
            res_list.append(arg)

    return make_obj_array(res_list)




def log_shape(array):
    """Returns the "logical shape" of the array.

    The "logical shape" is the shape that's left when the node-depending
    dimension has been eliminated."""

    try:
        if array.dtype.char == "O":
            return array.shape
        else:
            return array.shape[:-1]
    except AttributeError:
        return ()




def with_object_array_or_scalar(f, field):
    ls = log_shape(field)
    if ls != ():
        from pytools import indices_in_shape
        result = numpy.zeros(ls, dtype=object)
        for i in indices_in_shape(ls):
            result[i] = f(field[i])
        return result
    else:
        return f(field)




def ptwise_mul(a, b):
    a_log_shape = log_shape(a)
    b_log_shape = log_shape(b)

    from pytools import indices_in_shape

    if a_log_shape == ():
        result = numpy.empty(b_log_shape, dtype=object)
        for i in indices_in_shape(b_log_shape):
            result[i] = a*b[i]
    elif b_log_shape == ():
        result = numpy.empty(a_log_shape, dtype=object)
        for i in indices_in_shape(a_log_shape):
            result[i] = a[i]*b
    else:
        raise ValueError, "ptwise_mul can't handle two non-scalars"

    return result
            



def ptwise_dot(logdims1, logdims2, a1, a2):
    a1_log_shape = a1.shape[:logdims1]
    a2_log_shape = a1.shape[:logdims2]

    assert a1_log_shape[-1] == a2_log_shape[0]
    len_k = a2_log_shape[0]

    result = numpy.empty(a1_log_shape[:-1]+a2_log_shape[1:], dtype=object)

    from pytools import indices_in_shape
    for a1_i in indices_in_shape(a1_log_shape[:-1]):
        for a2_i in indices_in_shape(a2_log_shape[1:]):
            result[a1_i+a2_i] = sum(
                    a1[a1_i+(k,)] * a2[(k,)+a2_i]
                    for k in xrange(len_k)
                    )

    if result.shape == ():
        return result[()]
    else: 
        return result




def amap(f, obj_array):
    return numpy.array([f(x) for x in obj_array], dtype=object)




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




def count_subset(subset):
    from pytools import len_iterable
    return len_iterable(uc for uc in subset if uc)




def full_to_subset_indices(subset, base=0):
    """Takes a sequence of bools and turns it into an array of indices
    to be used to extract the subset from the full set.

    Example:

    >>> full_to_subset_indices([False, True, True])
    array([1 2])
    """

    result = []
    for i, is_in in enumerate(subset):
        if is_in:
            result.append(i + base)

    return numpy.array(result, dtype=numpy.intp)



def full_to_all_subset_indices(subsets, base=0):
    """Takes a sequence of bools and generates it into an array of indices
    to be used to extract the subset from the full set.

    Example:

    >>> list(full_to_all_subset_indices([[False, True, True], [True,False,True]]))
    [array([1 2]), array([3 5]
    """

    for subset in subsets:
        result = []
        for i, is_in in enumerate(subset):
            if is_in:
                result.append(i + base)
        base += len(subset)

        yield numpy.array(result, dtype=numpy.intp)



def partial_to_all_subset_indices(subsets, base=0):
    """Takes a sequence of bools and generates it into an array of indices
    to be used to insert the subset into the full set.

    Example:

    >>> list(partial_to_all_subset_indices([[False, True, True], [True,False,True]]))
    [array([0 1]), array([2 3]
    """

    idx = base
    for subset in subsets:
        result = []
        for is_in in subset:
            if is_in:
                result.append(idx)
                idx += 1

        yield numpy.array(result, dtype=numpy.intp)



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
                self.functions.append(pymbolic.compile(this_expr, 
                    variables=[op1, op2]))
                self.component_lcjk.append(this_component)

    def __call__(self, x, y, three_mult=None):
        """Compute the subsetted cross product on the indexables C{x} and C{y}.

        @arg three_mult: a function of three arguments C{sign, xj, yk}
          used in place of the product C{sign*xj*yk}. Defaults to just this
          product if not given.
        """
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




def find_matching_vertices_along_axis(axis, points_a, points_b, numbers_a, numbers_b):
    a_to_b = {}
    not_found = []

    for i, pi in enumerate(points_a):
        found = False
        for j, pj in enumerate(points_b):
            dist = pi-pj
            dist[axis] = 0
            if la.norm(dist) < 1e-12:
                a_to_b[numbers_a[i]] = numbers_b[j]
                found = True
                break
        if not found:
            not_found.append(numbers_a[i])

    return a_to_b, not_found




make_vector_target = hedge._internal.make_vector_target




# linear algebra tools --------------------------------------------------------
def orthonormalize(vectors, discard_threshold=None):
    """Carry out a modified [1] Gram-Schmidt orthonormalization on
    vectors.

    If, during orthonormalization, the 2-norm of a vector drops 
    below C{discard_threshold}, then this vector is silently 
    discarded. If C{discard_threshold} is C{None}, then no vector
    will ever be dropped, and a zero 2-norm encountered during
    orthonormalization will throw an L{OrthonormalizationError}.

    [1] U{http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process}
    """

    from numpy import dot
    done_vectors = []

    for v in vectors:
        my_v = v.copy()
        for done_v in done_vectors:
            my_v = my_v - dot(my_v, done_v.conjugate()) * done_v
        v_norm = la.norm(my_v)

        if discard_threshold is None:
            if v_norm == 0:
                raise RuntimeError, "Orthogonalization failed"
        else:
            if v_norm < discard_threshold:
                continue

        my_v /= v_norm
        done_vectors.append(my_v)

    return done_vectors




def permutation_matrix(to_indices=None, from_indices=None, h=None, w=None,
        dtype=None, flavor=None):
    """Return a permutation matrix.

    If to_indices is specified, the resulting permutation 
    matrix P satisfies the condition

    P * e[i] = e[to_indices[i]] for i=1,...,len(to_indices)

    where e[i] is the i-th unit vector. The height of P is 
    determined either implicitly by the maximum of to_indices
    or explicitly by the parameter h.

    If from_indices is specified, the resulting permutation 
    matrix P satisfies the condition

    P * e[from_indices[i]] = e[i] for i=1,...,len(from_indices)
    
    where e[i] is the i-th unit vector. The width of P is
    determined either implicitly by the maximum of from_indices
    of explicitly by the parameter w.

    If both to_indices and from_indices is specified, a ValueError
    exception is raised.
    """
    if to_indices is not None and from_indices is not None:
        raise ValueError, "only one of to_indices and from_indices may " \
                "be specified"

    if to_indices is not None:
        if h is None:
            h = max(to_indices)+1
        w = len(to_indices)
    else:
        if w is None:
            w = max(from_indices)+1
        h = len(from_indices)

    if flavor is None:
        result = numpy.zeros((h,w), dtype=dtype)

        if to_indices is not None:
            for j, i in enumerate(to_indices):
                result[i,j] = 1
        else:
            for i, j in enumerate(from_indices):
                result[i,j] = 1
    else:
        result = pyublas.zeros((h,w), dtype=dtype, flavor=flavor)

        if to_indices is not None:
            for j, i in enumerate(to_indices):
                result.add_element(i, j, 1)
        else:
            for i, j in enumerate(from_indices):
                result.add_element(i, j, 1)

    return result




def leftsolve(A, B):
    return la.solve(A.T, B.T).T




def unit_vector(n, i, dtype=None):
    """Return the i-th unit vector of size n, with the given dtype."""
    result = numpy.zeros((n,), dtype=dtype)
    result[i] = 1
    return result




# eoc estimation --------------------------------------------------------------
def estimate_order_of_convergence(abscissae, errors):
    """Assuming that abscissae and errors are connected by a law of the form

    error = constant * abscissa ^ (-order),

    this function finds, in a least-squares sense, the best approximation of
    constant and order for the given data set. It returns a tuple (constant, order).
    Both inputs must be PyLinear vectors.
    """
    assert len(abscissae) == len(errors)
    if len(abscissae) <= 1:
        raise RuntimeError, "Need more than one value to guess order of convergence."

    coefficients = numpy.polyfit(numpy.log10(abscissae), numpy.log10(errors), 1)
    return 10**coefficients[-1], -coefficients[-2]


  

class EOCRecorder(object):
    def __init__(self):
        self.history = []

    def add_data_point(self, abscissa, error):
        self.history.append((abscissa, error))

    def estimate_order_of_convergence(self, gliding_mean = None):
        abscissae = numpy.array([ a for a,e in self.history ])
        errors = numpy.array([ e for a,e in self.history ])

        size = len(abscissae)
        if gliding_mean is None:
            gliding_mean = size

        data_points = size - gliding_mean + 1
        result = numpy.zeros((data_points, 2), float)
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





# index map tools -------------------------------------------------------------
def apply_index_map(imap, vector):
    from hedge._internal import VectorTarget, perform_index_map

    ls = log_shape(vector)
    if ls == ():
        result = numpy.zeros(shape=(imap.to_length,), dtype=float)
        perform_index_map(imap, VectorTarget(vector, result))
    else:
        result = numpy.zeros(shape=ls+(imap.to_length,), dtype=float)
        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            perform_index_map(imap, VectorTarget(vector[i], result[i]))
    return result




def apply_inverse_index_map(imap, vector):
    from hedge._internal import VectorTarget, perform_inverse_index_map

    ls = log_shape(vector)
    if ls == ():
        result = numpy.zeros(shape=(imap.from_length,), dtype=float)
        perform_inverse_index_map(imap, VectorTarget(vector, result))
    else:
        result = numpy.zeros(shape=ls+(imap.from_length,), dtype=float)
        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            perform_inverse_index_map(imap, VectorTarget(vector[i], result[i]))
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




class IndexListRegistry(object):
    def __init__(self, debug=False):
        self.index_lists = []
        self.il_id_to_number = {}
        self.il_to_number = {}
        self.debug = debug

    def register(self, identifier, generator):
        try:
            result = self.il_id_to_number[identifier]
            if self.debug:
                assert generator() == self.index_lists[result], (
                        "identifier %s used for two different index lists"
                        % str(identifier))
            return result
        except KeyError:
            il = generator()
            try:
                nbr = self.il_to_number[il]
            except KeyError:
                nbr = len(self.index_lists)
                self.index_lists.append(il)
                self.il_id_to_number[identifier] = nbr
                self.il_to_number[il] = nbr
            else:
                self.il_id_to_number[identifier] = nbr
            return nbr

    def get_list_length(self):
        from pytools import single_valued
        return single_valued(len(il) for il in self.index_lists)





# parallel cg -----------------------------------------------------------------
class OperatorBase(object):
    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    def __neg__(self):
        return NegOperator(self)

class NegOperator(OperatorBase):
    def __init__(self, sub_op):
        self.sub_op = sub_op

    @property
    def dtype(self):
        return self.sub_op.dtype

    @property
    def shape(self):
        return self.sub_op.shape

    def __call__(self, operand):
        return -self.sub_op(operand)

class IdentityOperator(OperatorBase):
    def __init__(self, dtype, n):
        self.my_dtype = dtype
        self.n = n

    @property
    def dtype(self):
        return self.my_dtype

    @property
    def shape(self):
        return self.n, self.n

    def __call__(self, operand):
        return operand




class CGStateContainer:
    def __init__(self, pcon, operator, precon=None, dot=None):
        if precon is None:
            precon = IdentityOperator(operator.dtype, operator.shape[0])

        self.pcon = pcon
        self.operator = operator
        self.precon = precon

        if dot is None:
            dot = numpy.dot

        def inner(a, b):
            return dot(a, b.conj())

        self.inner = inner

    def reset(self, rhs, x=None):
        self.rhs = rhs

        if x is None:
            x = numpy.zeros((self.operator.size1(),))
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
        myip = self.inner(self.d, q)
        alpha = self.delta / myip

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
            max_iterations = 10 * self.operator.shape[0]

        if self.inner(self.rhs, self.rhs) == 0:
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
                if debug:
                    print "%d iterations" % iterations
                return self.x

            if debug and iterations % debug == 0 and self.pcon.is_head_rank:
                print "debug: delta=%g" % delta
            iterations += 1

        raise RuntimeError("cg failed to converge")
            



def parallel_cg(pcon, operator, b, precon=None, x=None, tol=1e-7, max_iterations=None, 
        debug=False, debug_callback=None, dot=None):
    if x is None:
        x = numpy.zeros((operator.shape[1],))

    cg = CGStateContainer(pcon, operator, precon, dot=dot)
    cg.reset(b, x)
    return cg.run(max_iterations, tol, debug_callback, debug)




# diagnostics -----------------------------------------------------------------
def time_count_flop(func, timer, counter, flop_counter, flops, increment=1):
    def wrapped_f(*args, **kwargs):
        counter.add()
        flop_counter.add(flops)
        sub_timer = timer.start_sub_timer()
        try:
            return func(*args, **kwargs)
        finally:
            sub_timer.stop().submit()

    return wrapped_f




# flop counting ---------------------------------------------------------------
def diff_rst_flops(discr):
    result = 0
    for eg in discr.element_groups:
        ldis = eg.local_discretization
        result += (
                2 # mul+add
                * ldis.node_count() * len(eg.members)
                * ldis.node_count()
                )

    return result




def diff_rescale_one_flops(discr):
    result = 0
    for eg in discr.element_groups:
        ldis = eg.local_discretization
        result += (
                # x,y,z rescale
                2 # mul+add
                * discr.dimensions
                * len(eg.members) * ldis.node_count()
                )

    return result




def mass_flops(discr):
    result = 0
    for eg in discr.element_groups:
        ldis = eg.local_discretization
        result += (
                2 # mul+add
                * ldis.node_count() * len(eg.members)
                * ldis.node_count()
                )

    result += len(discr.nodes) # jacobian rescale

    return result




def lift_flops(discr):
    result = 0

    for eg in discr.element_groups:
        ldis = eg.local_discretization
        result += (
                2 # mul+add
                * ldis.face_node_count()
                * ldis.face_count()
                * ldis.node_count() 
                * len(eg.members)
                )

    return result




def gather_flops(discr):
    result = 0
    for eg in discr.element_groups:
        ldis = eg.local_discretization
        result += (
                ldis.face_node_count()
                * ldis.face_count()
                * len(eg.members)
                * (1 # facejac-mul
                    + 2 * # int+ext
                    3 # const-mul, normal-mul, add
                    )
                )

    return result




# flux creation ---------------------------------------------------------------
def make_lax_friedrichs_flux(wave_speed, state, flux_func, bdry_tags_and_states, strong):

    from hedge.flux import make_normal, FluxVectorPlaceholder
   
    fluxes = flux_func(state)

    n = len(state)
    d = len(fluxes)
    normal = make_normal(d)
    fvph = FluxVectorPlaceholder(len(state)*(1+d)+1)

    wave_speed_ph = fvph[0]
    state_ph = fvph[1:1+n]
    fluxes_ph = [fvph[1+i*n:1+(i+1)*n] for i in range(1, d+1)]

    penalty = wave_speed_ph.int*(state_ph.ext-state_ph.int) 

    if not strong:
        flux = 0.5*(sum(n_i*(f_i.int+f_i.ext) for n_i, f_i in zip(normal, fluxes_ph))
                    - penalty)
    else:
        flux = 0.5*(sum(n_i*(f_i.int-f_i.ext) for n_i, f_i in zip(normal, fluxes_ph))
                + penalty)

    from hedge.optemplate import get_flux_operator
    flux_op = get_flux_operator(flux)
    int_operand = join_fields(wave_speed, state, *fluxes)

    from hedge.optemplate import pair_with_boundary

    return (flux_op*int_operand
            + sum(
                flux_op*pair_with_boundary(int_operand,
                    join_fields(0, ext_state, *flux_func(ext_state)), tag)
                for tag, ext_state in bdry_tags_and_states))




# debug tools -----------------------------------------------------------------
def wait_for_keypress(discr):
    """MPI-aware keypress wait"""
    try:
        comm = discr.parallel_discr.context.communicator
    except AttributeError:
        raw_input("[Enter]")
    else:
        if comm.rank == 0:
            # OpenMPI connects mpirun's stdin to rank 0's stdin.
            print "[Enter]"
            raw_input()

        from hedge.mpi import broadcast
        broadcast(comm, value=0, root=0)

def get_rank(discr):
    """Rank query that works with and without MPI active."""
    try:
        comm = discr.parallel_discr.context.communicator
    except AttributeError:
        return 0
    else:
        return comm.rank




def typedump(value, max_seq=5, special_handlers={}):
    from pytools import typedump
    special_handlers = special_handlers.copy()
    special_handlers.update({
        numpy.ndarray: lambda x: "array(%s, %s)" % (len(x.shape), x.dtype)
        })
    return typedump(value, max_seq, special_handlers)




# futures ---------------------------------------------------------------------
class Future(object):
    """An abstract interface definition for futures.

    See http://en.wikipedia.org/wiki/Future_(programming)
    """
    def is_ready(self):
        raise NotImplementedError(self.__class__)

    def __call__(self):
        raise NotImplementedError(self.__class__)




class ImmediateFuture(Future):
    """A non-future that immediately has a value available."""
    def __init__(self, value):
        self.value = value

    def is_ready(self):
        return True

    def __call__(self):
        return self.value




class NestedFuture(Future):
    """A future that combines two sub-futures into one."""
    def __init__(self, outer_future_factory, inner_future):
        self.outer_future_factory = outer_future_factory
        self.inner_future = inner_future
        self.outer_future = None

    def is_ready(self):
        if self.inner_future.is_ready():
            self.outer_future = self.outer_future_factory(self.inner_future())
            self.is_ready = self.outer_future.is_ready()
            return self.is_ready()
        else:
            return False

    def __call__(self):
        if self.outer_future is None:
            return self.outer_future_factory(self.inner_future())()
        else:
            return self.outer_future()
