"""Flux creation helpers."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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





def make_lax_friedrichs_flux(wave_speed, state, fluxes, bdry_tags_states_and_fluxes,
        strong):
    from pytools.obj_array import join_fields
    from hedge.flux import make_normal, FluxVectorPlaceholder, flux_max

    n = len(state)
    d = len(fluxes)
    normal = make_normal(d)
    fvph = FluxVectorPlaceholder(len(state)*(1+d)+1)

    wave_speed_ph = fvph[0]
    state_ph = fvph[1:1+n]
    fluxes_ph = [fvph[1+i*n:1+(i+1)*n] for i in range(1, d+1)]

    penalty = flux_max(wave_speed_ph.int,wave_speed_ph.ext)*(state_ph.ext-state_ph.int)

    if not strong:
        num_flux = 0.5*(sum(n_i*(f_i.int+f_i.ext) for n_i, f_i in zip(normal, fluxes_ph))
                - penalty)
    else:
        num_flux = 0.5*(sum(n_i*(f_i.int-f_i.ext) for n_i, f_i in zip(normal, fluxes_ph))
                + penalty)

    from hedge.optemplate import get_flux_operator
    flux_op = get_flux_operator(num_flux)
    int_operand = join_fields(wave_speed, state, *fluxes)

    from hedge.optemplate import BoundaryPair
    return (flux_op(int_operand)
            + sum(
                flux_op(BoundaryPair(int_operand,
                    join_fields(0, bdry_state, *bdry_fluxes), tag))
                for tag, bdry_state, bdry_fluxes in bdry_tags_states_and_fluxes))




def make_flux_bilinear_form(testee_expr, test_expr):
    """Create a hedge flux expression for the bilinear form

    .. math::

        \int_{\Gamma} u f(l_i^+,l_i^-),

    where :math:`u` is the (vector or scalar) unknown, given in *testee_epxr*,
    :math:`f` is a function given by *test_expr* in terms of the Lagrange polynomials
    :math:`l^_i^{\pm}` against which we are testing. In *test_expr*, :math:`l_i^{\pm}`
    are represented by :class:`hedge.flux.FieldComponent(-1, True/False)`.
    :math:`f` is required to be linear in :math:`l_i^{\pm}`.
    """
    # keeping this stub to not lose the design work that went into its signature.
    raise NotImplementedError



