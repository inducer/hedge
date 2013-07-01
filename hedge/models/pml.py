# -*- coding: utf8 -*-
"""Models describing absorbing boundary layers."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""




import numpy

from pytools import memoize_method, Record
from hedge.models.em import \
        MaxwellOperator, \
        TMMaxwellOperator, \
        TEMaxwellOperator




class AbarbanelGottliebPMLMaxwellOperator(MaxwellOperator):
    """Implements a PML as in

    [1] S. Abarbanel and D. Gottlieb, "On the construction and analysis of absorbing
    layers in CEM," Applied Numerical Mathematics,  vol. 27, 1998, S. 331-340.
    (eq 3.7-3.11)

    [2] E. Turkel and A. Yefet, "Absorbing PML
    boundary layers for wave-like equations,"
    Applied Numerical Mathematics,  vol. 27,
    1998, S. 533-557.
    (eq. 4.10)

    [3] Abarbanel, D. Gottlieb, and J.S. Hesthaven, "Long Time Behavior of the
    Perfectly Matched Layer Equations in Computational Electromagnetics,"
    Journal of Scientific Computing,  vol. 17, Dez. 2002, S. 405-422.

    Generalized to 3D in doc/maxima/abarbanel-pml.mac.
    """

    class PMLCoefficients(Record):
        __slots__ = ["sigma", "sigma_prime", "tau"]
        # (tau=mu in [3] , to avoid confusion with permeability)

        def map(self, f):
            return self.__class__(
                    **dict((name, f(getattr(self, name)))
                        for name in self.fields))

    def __init__(self, *args, **kwargs):
        self.add_decay = kwargs.pop("add_decay", True)
        MaxwellOperator.__init__(self, *args, **kwargs)

    def pml_local_op(self, w):
        sub_e, sub_h, sub_p, sub_q = self.split_ehpq(w)

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]
        dim_subset = (True,) * self.dimensions + (False,) * (3-self.dimensions)

        def pad_vec(v, subset):
            result = numpy.zeros((3,), dtype=object)
            result[numpy.array(subset, dtype=bool)] = v
            return result

        from hedge.optemplate import make_sym_vector
        sig = pad_vec(
                make_sym_vector("sigma", self.dimensions),
                dim_subset)
        sig_prime = pad_vec(
                make_sym_vector("sigma_prime", self.dimensions),
                dim_subset)
        if self.add_decay:
            tau = pad_vec(
                    make_sym_vector("tau", self.dimensions),
                    dim_subset)
        else:
            tau = numpy.zeros((3,))

        e = pad_vec(sub_e, e_subset)
        h = pad_vec(sub_h, h_subset)
        p = pad_vec(sub_p, dim_subset)
        q = pad_vec(sub_q, dim_subset)

        rhs = numpy.zeros(12, dtype=object)

        for mx in range(3):
            my = (mx+1) % 3
            mz = (mx+2) % 3

            from hedge.tools.mathematics import levi_civita
            assert levi_civita((mx,my,mz)) == 1

            rhs[mx] += -sig[my]/self.epsilon*(2*e[mx]+p[mx]) - 2*tau[my]/self.epsilon*e[mx]
            rhs[my] += -sig[mx]/self.epsilon*(2*e[my]+p[my]) - 2*tau[mx]/self.epsilon*e[my]
            rhs[3+mz] += 1/(self.epsilon*self.mu) * (
              sig_prime[mx] * q[mx] - sig_prime[my] * q[my])

            rhs[6+mx] += sig[my]/self.epsilon*e[mx]
            rhs[6+my] += sig[mx]/self.epsilon*e[my]
            rhs[9+mx] += -sig[mx]/self.epsilon*q[mx] - (e[my] + e[mz])

        from hedge.tools import full_to_subset_indices
        sub_idx = full_to_subset_indices(e_subset+h_subset+dim_subset+dim_subset)

        return rhs[sub_idx]

    def op_template(self, w=None):
        from hedge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())
        if w is None:
            from hedge.optemplate import make_sym_vector
            w = make_sym_vector("w", fld_cnt+2*self.dimensions)

        from hedge.tools import join_fields
        return join_fields(
                MaxwellOperator.op_template(self, w[:fld_cnt]),
                numpy.zeros((2*self.dimensions,), dtype=object)
                ) + self.pml_local_op(w)

    def bind(self, discr, coefficients):
        return MaxwellOperator.bind(self, discr,
                sigma=coefficients.sigma,
                sigma_prime=coefficients.sigma_prime,
                tau=coefficients.tau)

    def assemble_ehpq(self, e=None, h=None, p=None, q=None, discr=None):
        if discr is None:
            def zero():
                return 0
        else:
            def zero():
                return discr.volume_zeros()

        from hedge.tools import count_subset
        e_components = count_subset(self.get_eh_subset()[0:3])
        h_components = count_subset(self.get_eh_subset()[3:6])

        def default_fld(fld, comp):
            if fld is None:
                return [zero() for i in xrange(comp)]
            else:
                return fld

        e = default_fld(e, e_components)
        h = default_fld(h, h_components)
        p = default_fld(p, self.dimensions)
        q = default_fld(q, self.dimensions)

        from hedge.tools import join_fields
        return join_fields(e, h, p, q)

    @memoize_method
    def partial_to_ehpq_subsets(self):
        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        dim_subset = [True] * self.dimensions + [False] * (3-self.dimensions)

        from hedge.tools import partial_to_all_subset_indices
        return tuple(partial_to_all_subset_indices(
            [e_subset, h_subset, dim_subset, dim_subset]))

    def split_ehpq(self, w):
        e_idx, h_idx, p_idx, q_idx = self.partial_to_ehpq_subsets()
        e, h, p, q = w[e_idx], w[h_idx], w[p_idx], w[q_idx]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=e), FVP(scalars=h)
        else:
            from hedge.tools import make_obj_array as moa
            return moa(e), moa(h), moa(p), moa(q)

    # sigma business ----------------------------------------------------------
    def _construct_scalar_coefficients(self, discr, node_coord,
            i_min, i_max, o_min, o_max, exponent):
        assert o_min < i_min <= i_max < o_max

        if o_min != i_min:
            l_dist = (i_min - node_coord) / (i_min-o_min)
            l_dist_prime = discr.volume_zeros(kind="numpy", dtype=node_coord.dtype)
            l_dist_prime[l_dist >= 0] = -1 / (i_min-o_min)
            l_dist[l_dist < 0] = 0
        else:
            l_dist = l_dist_prime = numpy.zeros_like(node_coord)

        if i_max != o_max:
            r_dist = (node_coord - i_max) / (o_max-i_max)
            r_dist_prime = discr.volume_zeros(kind="numpy", dtype=node_coord.dtype)
            r_dist_prime[r_dist >= 0] = 1 / (o_max-i_max)
            r_dist[r_dist < 0] = 0
        else:
            r_dist = r_dist_prime = numpy.zeros_like(node_coord)

        l_plus_r = l_dist+r_dist
        return l_plus_r**exponent, \
                (l_dist_prime+r_dist_prime)*exponent*l_plus_r**(exponent-1), \
                l_plus_r

    def coefficients_from_boxes(self, discr,
            inner_bbox, outer_bbox=None,
            magnitude=None, tau_magnitude=None,
            exponent=None, dtype=None):
        if outer_bbox is None:
            outer_bbox = discr.mesh.bounding_box()

        if exponent is None:
            exponent = 2

        if magnitude is None:
            magnitude = 20

        if tau_magnitude is None:
            tau_magnitude = 0.4

        # scale by free space conductivity
        from math import sqrt
        magnitude = magnitude*sqrt(self.epsilon/self.mu)
        tau_magnitude = tau_magnitude*sqrt(self.epsilon/self.mu)

        i_min, i_max = inner_bbox
        o_min, o_max = outer_bbox

        from hedge.tools import make_obj_array

        nodes = discr.nodes
        if dtype is not None:
            nodes = nodes.astype(dtype)

        sigma, sigma_prime, tau = zip(*[self._construct_scalar_coefficients(
            discr, nodes[:,i],
            i_min[i], i_max[i], o_min[i], o_max[i],
            exponent)
            for i in range(discr.dimensions)])

        def conv(f):
            return discr.convert_volume(f, kind=discr.compute_kind,
                    dtype=discr.default_scalar_type)

        return self.PMLCoefficients(
                sigma=conv(magnitude*make_obj_array(sigma)),
                sigma_prime=conv(magnitude*make_obj_array(sigma_prime)),
                tau=conv(tau_magnitude*make_obj_array(tau)))

    def coefficients_from_width(self, discr, width,
            magnitude=None, tau_magnitude=None, exponent=None,
            dtype=None):
        o_min, o_max = discr.mesh.bounding_box()
        return self.coefficients_from_boxes(discr,
                (o_min+width, o_max-width),
                (o_min, o_max),
                magnitude, tau_magnitude, exponent, dtype)




class AbarbanelGottliebPMLTEMaxwellOperator(
        TEMaxwellOperator, AbarbanelGottliebPMLMaxwellOperator):
    # not unimplemented--this IS the implementation.
    pass

class AbarbanelGottliebPMLTMMaxwellOperator(
        TMMaxwellOperator, AbarbanelGottliebPMLMaxwellOperator):
    # not unimplemented--this IS the implementation.
    pass
