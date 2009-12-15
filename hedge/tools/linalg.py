"""Linear algebra tools."""

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




def orthonormalize(vectors, discard_threshold=None):
    """Carry out a modified [1] Gram-Schmidt orthonormalization on
    vectors.

    If, during orthonormalization, the 2-norm of a vector drops
    below *discard_threshold*, then this vector is silently
    discarded. If *discard_threshold* is *None*, then no vector
    will ever be dropped, and a zero 2-norm encountered during
    orthonormalization will throw a :exc:`RuntimeError`.

    [1] http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
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
        result = numpy.zeros((h,w), dtype=dtype, flavor=flavor)

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




