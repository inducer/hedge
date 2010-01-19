"""Flop counting."""

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




def lift_flops(fg):
    ldis = fg.ldis_loc
    return (
            2 # mul+add
            * fg.face_length()
            * ldis.face_count()
            * ldis.node_count()
            * fg.element_count()
            )




def gather_flops(discr, quadrature_tag=None):
    result = 0
    for eg in discr.element_groups:
        ldis = eg.local_discretization

        if quadrature_tag is None:
            fnc = ldis.face_node_count()
        else:
            fnc = eg.quadrature_info[quadrature_tag] \
                    .ldis_quad_info.face_node_count()

        result += (
                fnc
                * ldis.face_count()
                * len(eg.members)
                * (1 # facejac-mul
                    + 2 * # int+ext
                    3 # const-mul, normal-mul, add
                    )
                )

    return result




def count_dofs(vec):
    try:
        dtype = vec.dtype
        size = vec.size
        shape = vec.shape
    except AttributeError:
        from warnings import warn
        warn("could not count dofs of vector")
        return 0

    if dtype == object:
        from pytools import indices_in_shape
        return sum(count_dofs(vec[i])
                for i in indices_in_shape(vec.shape))
    else:
        return size




