"""Flop counting."""

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




