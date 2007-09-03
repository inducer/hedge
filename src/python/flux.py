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




import hedge._internal as _internal
from pytools.arithmetic_container import ArithmeticList, \
        work_with_arithmetic_containers




Face = _internal.Face
Flux = _internal.Flux
Flux = _internal.ConstantFlux
ChainedFlux = _internal.ChainedFlux




def make_normal(dim):
    return ArithmeticList([
        getattr(_internal, "Normal%dFlux" % i)()
        for i in range(dim)])




@work_with_arithmetic_containers
def penalty(coefficient, exponent):
    return _internal.PenaltyTermFlux(coefficient, exponent)

zero = _internal.ZeroFlux()
local = _internal.ConstantFlux(1, 0)
neighbor = _internal.ConstantFlux(0, 1)
average = _internal.ConstantFlux(0.5, 0.5)
trace_sign = _internal.ConstantFlux(-1, 1)




# generic flux arithmetic -----------------------------------------------------
@work_with_arithmetic_containers
def _add_fluxes(fl1, fl2): 
    try:
        return _internal.add_fluxes(fl1, fl2)
    except TypeError:
        return _internal.SumFlux(fl1, fl2)


@work_with_arithmetic_containers
def _sub_fluxes(fl1, fl2): 
    try:
        return _internal.subtract_fluxes(fl1, fl2)
    except TypeError:
        return _internal.DifferenceFlux(fl1, fl2)

@work_with_arithmetic_containers
def _mul_fluxes(fl1, op2): 
    try:
        return _internal.multiply_fluxes(fl1, op2)
    except TypeError:
        return _internal.ConstantProductFlux(fl1, op2)

def _neg_flux(flux): 
    return _internal.NegativeFlux(flux)




Flux.__add__ = _add_fluxes
Flux.__sub__ = _sub_fluxes
Flux.__mul__ = _mul_fluxes
Flux.__rmul__ = _mul_fluxes
Flux.__neg__ = _neg_flux
