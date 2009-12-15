"""Flux creation helpers."""

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

