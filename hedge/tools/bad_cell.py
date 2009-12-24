"""Bad-cell indicators."""

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



import numpy
import numpy.linalg as la
from hedge.optemplate.operators import (
        ElementwiseLinearOperator, StatelessOperator)




# Persson-Peraire -------------------------------------------------------------
def persson_peraire_filter_response_function(mode_idx, ldis):
    if sum(mode_idx) == ldis.order:
        return 0
    else:
        return 1



class PerssonPeraireDiscontinuitySensor(object):
    """
    see
    [1] P. Persson und J. Peraire,
    "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods,"
    Proc. of the 44th AIAA Aerospace Sciences Meeting and Exhibit, 2006.
    """

    def __init__(self, kappa, eps0, s_0):
        self.kappa = kappa
        self.eps0 = eps0
        self.s_0 = s_0

    def op_template(self, u=None):
        from pymbolic.primitives import IfPositive, Variable
        from hedge.optemplate.primitives import Field, ScalarParameter
        from hedge.tools.symbolic import make_common_subexpression as cse
        from math import pi

        if u is None:
            u = Field("u")

        from hedge.optemplate.operators import (
                MassOperator, FilterOperator, OnesOperator)

        mode_truncator = FilterOperator(
                persson_peraire_filter_response_function)

        truncated_u = mode_truncator(u)
        diff = u - truncated_u

        el_norm_squared_mass_diff_u = OnesOperator()(MassOperator()(diff)*diff)
        el_norm_squared_mass_u = OnesOperator()(MassOperator()(u)*u)

        capital_s_e = cse(el_norm_squared_mass_diff_u / el_norm_squared_mass_u,
                "S_e")

        sin = Variable("sin")
        log10 = Variable("log10")

        s_e = cse(log10(capital_s_e), "s_e")
        kappa = ScalarParameter("kappa")
        eps0 = ScalarParameter("eps0")
        s_0 = ScalarParameter("s_0")

        return IfPositive(s_0-self.kappa-s_e,
                0,
                IfPositive(s_e-self.kappa-s_0,
                    eps0,
                    eps0/2*(1+sin(pi*(s_e-s_0)/self.kappa))))

    def bind(self, discr):
        compiled = discr.compile(self.op_template())

        from pytools import match_precision
        scalar_type = match_precision(
                numpy.dtype(numpy.float64),
                discr.default_scalar_type).type

        kappa = scalar_type(self.kappa)
        eps0 = scalar_type(self.eps0)
        s_0 = scalar_type(self.s_0)

        def apply(u):
            return compiled(u=u, kappa=kappa, eps0=eps0, s_0=s_0)

        return apply





# exponential fit -------------------------------------------------------------
class DecayEstimateOperatorBase(ElementwiseLinearOperator):
    def __init__(self, ignored_modes):
        self.ignored_modes = ignored_modes

    def get_hash(self):
        return hash((self.__class__, self.ignored_modes))

    def is_equal(self, other):
        return (self.__class__ == other.__class__
                and self.ignored_modes == other.ignored_modes)

    def decay_fit_mat(self, ldis):
        im = self.ignored_modes
        node_cnt = ldis.node_count()

        node_number_vector = numpy.zeros(node_cnt-im, dtype=numpy.float64)
        for i, mid in enumerate(ldis.generate_mode_identifiers()):
            if i < im:
                continue
            node_number_vector[i-im] = sum(mid)

        a = numpy.zeros((node_cnt-im, 2), dtype=numpy.float64)
        a[:,0] = 1
        a[:,1] = numpy.log(node_number_vector)

        result = numpy.zeros((2, node_cnt))
        result[:,im:] = la.pinv(a)

        return result

class DecayExponentOperator(DecayEstimateOperatorBase):
    def matrix(self, eg):
        ldis = eg.local_discretization
        plsm = self.decay_fit_mat(ldis)
        a = numpy.zeros((ldis.node_count(), ldis.node_count()))
        for i in range(ldis.node_count()):
            a[i] = plsm[1]

        return a

class LogDecayConstantOperator(DecayEstimateOperatorBase):
    def matrix(self, eg):
        ldis = eg.local_discretization
        plsm = self.decay_fit_mat(ldis)
        a = numpy.zeros((ldis.node_count(), ldis.node_count()))
        for i in range(ldis.node_count()):
            a[i] = plsm[0]

        return a

def create_mode_number_vector(discr):
    """Create a vector of modal coefficients that exhibit 'optimal'
    (:math:`k^{-N}`) decay.
    """
    result = discr.volume_zeros(kind="numpy")
    for eg in discr.element_groups:
        ldis = eg.local_discretization

        modal_coefficients = numpy.zeros(ldis.node_count(), dtype=result.dtype)
        for i, mid in enumerate(ldis.generate_mode_identifiers()):
            msum = sum(mid)
            if msum != 0:
                modal_coefficients[i] = msum
            else:
                modal_coefficients[i] = 1

        for slc in eg.ranges:
            result[slc] = modal_coefficients

    return result

def create_decay_baseline(discr):
    """Create a vector of modal coefficients that exhibit 'optimal'
    (:math:`k^{-N}`) decay.
    """
    result = discr.volume_zeros(kind="numpy")
    for eg in discr.element_groups:
        ldis = eg.local_discretization

        modal_coefficients = numpy.zeros(ldis.node_count(), dtype=result.dtype)
        for i, mid in enumerate(ldis.generate_mode_identifiers()):
            msum = sum(mid)
            if msum != 0:
                modal_coefficients[i] = msum**(-ldis.order)
                #modal_coefficients[i] = 1e-7
            else:
                modal_coefficients[i] = 1 # irrelevant, just keeps log from NaNing

        for slc in eg.ranges:
            result[slc] = modal_coefficients

    return result




class BottomChoppingFilterResponseFunction:
    def __init__(self, ignored_modes):
        self.ignored_modes = ignored_modes

    def __call__(self, mode_idx, ldis):
        if sum(mode_idx) < self.ignored_modes:
            return 0
        else:
            return 1




class DecayFitDiscontinuitySensorBase(object):
    def decay_estimate_op_template(self, u, ignored_modes=0, with_baseline=True):
        from hedge.optemplate.operators import (FilterOperator,
                MassOperator, OnesOperator, InverseVandermondeOperator,
                InverseMassOperator)
        from hedge.optemplate.primitives import Field
        from hedge.optemplate.tools import get_flux_operator
        from hedge.tools.symbolic import make_common_subexpression as cse
        from pymbolic.primitives import Variable

        if u is None:
            u = Field("u")

        from hedge.flux import (
                FluxScalarPlaceholder, ElementOrder,
                ElementJacobian, FaceJacobian)
        from pymbolic.primitives import IfPositive

        u_flux = FluxScalarPlaceholder(0)

        # On the whole, this should scale like u.
        # Columns of lift scale like 1/N^2, compensate for that.
        # Further compensate for all geometric factors.

        jump_part = InverseMassOperator()(
                get_flux_operator(
                    ElementJacobian()**2/(ElementOrder()**2 * FaceJacobian())
                        *(u_flux.ext - u_flux.int))(u))

        baseline_squared = Field("baseline_squared")
        el_norm_u_squared = OnesOperator()(MassOperator()(u)*u)

        indicator_modal_coeffs = InverseVandermondeOperator()(u + jump_part)

        log, exp, sqrt = Variable("log"), Variable("exp"), Variable("sqrt")
        log_modal_coeffs = cse(
                log(indicator_modal_coeffs**2
                    + baseline_squared*el_norm_u_squared)/2,
                "log_modal_coeffs")

        # fit to c * n**s
        s = DecayExponentOperator(ignored_modes)(log_modal_coeffs)
        log_c = LogDecayConstantOperator(ignored_modes)(log_modal_coeffs)
        c = exp(LogDecayConstantOperator(ignored_modes)(log_modal_coeffs))

        log_mode_numbers = Field("log_mode_numbers")
        estimated_log_modal_coeffs = log_c + s*log_mode_numbers
        estimate_error = sqrt((estimated_log_modal_coeffs-log_modal_coeffs)**2)

        log_modal_coeffs_corrected = log_modal_coeffs + estimate_error
        s_corrected = DecayExponentOperator(ignored_modes)(log_modal_coeffs_corrected)

        from pytools import Record
        class DecayInformation(Record): pass

        return DecayInformation(
                decay_expt=s, c=c, log_modal_coeffs=log_modal_coeffs,
                estimated_log_modal_coeffs=estimated_log_modal_coeffs,
                decay_expt_corrected=s_corrected)

    def bind_quantity(self, discr, quantity_name, ignored_modes=1):
        baseline_squared = create_decay_baseline(discr)**2
        log_mode_numbers = numpy.log(create_mode_number_vector(discr))

        from hedge.optemplate import Field
        quantity = getattr(
                self.decay_estimate_op_template(Field("u"),
                    ignored_modes), quantity_name)

        compiled = discr.compile(quantity)

        def apply(u):
            return compiled(
                    u=u, 
                    baseline_squared=baseline_squared,
                    log_mode_numbers=log_mode_numbers)

        return apply





class DecayGatingDiscontinuitySensorBase(
        DecayFitDiscontinuitySensorBase):
    def __init__(self, max_viscosity=0.01):
        self.max_viscosity = max_viscosity

    def op_template(self, u=None):
        from pymbolic.primitives import IfPositive, Variable
        from hedge.optemplate import Field
        from math import pi
        from hedge.tools.symbolic import make_common_subexpression as cse

        if u is None:
            u = Field("u")

        decay_expt = self.decay_estimate_op_template(u, ignored_modes=1) \
                .decay_expt_corrected

        decay_expt = cse(decay_expt, "decay_expt")
        sin = Variable("sin")

        def flat_end_sin(x):
            return IfPositive(-pi/2-x,
                    -1, IfPositive(x-pi/2, 1, sin(x)))

        return 0.5*self.max_viscosity*(1+flat_end_sin((decay_expt+2)*pi/2))

    def bind(self, discr):
        baseline_squared = create_decay_baseline(discr)**2
        log_mode_numbers = numpy.log(create_mode_number_vector(discr))

        compiled = discr.compile(self.op_template())

        def apply(u):
            return compiled(u=u, 
                    baseline_squared=baseline_squared,
                    log_mode_numbers=log_mode_numbers)

        return apply
