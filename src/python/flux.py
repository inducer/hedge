import hedge._internal as _internal
from pytools.arithmetic_container import ArithmeticList, \
        work_with_arithmetic_containers




Face = _internal.Face
Flux = _internal.Flux
ChainedFlux = _internal.ChainedFlux




def normal(dim):
    return ArithmeticList([
        _internal.NormalXFlux(), 
        _internal.NormalYFlux(),
        _internal.NormalZFlux(),
        ])[:dim]

def jump(dim):
    return ArithmeticList([
        _internal.JumpXFlux(), 
        _internal.JumpYFlux(),
        _internal.JumpZFlux(),
        ])[:dim]

@work_with_arithmetic_containers
def penalty(coefficient, exponent):
    return _internal.PenaltyTermFlux(coefficient, exponent)

zero = _internal.ZeroFlux()
local = _internal.LocalFlux()
neighbor = _internal.NeighborFlux()
average = _internal.AverageFlux()
trace_sign = _internal.TraceSignFlux()
neg_trace_sign = _internal.NegativeTraceSignFlux()

@work_with_arithmetic_containers
def _add_fluxes(fl1, fl2): 
    return _internal.SumFlux(fl1, fl2)

@work_with_arithmetic_containers
def _sub_fluxes(fl1, fl2): 
    return _internal.DifferenceFlux(fl1, fl2)

@work_with_arithmetic_containers
def _mul_fluxes(fl1, op2): 
    if isinstance(op2, Flux):
        return _internal.ProductFlux(fl1, op2)
    else:
        return _internal.ConstantProductFlux(fl1, op2)

def _neg_flux(flux): 
    return _internal.NegativeFlux(flux)




Flux.__add__ = _add_fluxes
Flux.__sub__ = _sub_fluxes
Flux.__mul__ = _mul_fluxes
Flux.__rmul__ = _mul_fluxes
Flux.__neg__ = _neg_flux

