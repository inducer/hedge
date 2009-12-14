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




# Persson-Peraire -------------------------------------------------------------
def persson_peraire_filter_response_function(mode_idx, ldis):
    if sum(mode_idx) == ldis.order - 1:
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

    def __init__(self, discr, kappa, eps0, s_0):
        from pytools import match_precision
        scalar_type = match_precision(
                numpy.dtype(numpy.float64),
                discr.default_scalar_type).type
        self.discr = discr
        self.kappa = scalar_type(kappa)
        self.eps0 = scalar_type(eps0)
        self.s_0 = scalar_type(s_0)

        from hedge.discretization import Filter
        self.mode_truncator = Filter(discr,
                persson_peraire_filter_response_function)
        self.ones_data_store = {}

        from hedge.optemplate import MassOperator, Field
        self.mass_op = discr.compile(MassOperator() * Field("f"))

        self.threshold_op = discr.compile(
                self.threshold_op_template())

    def threshold_op_template(self):
        from pymbolic.primitives import IfPositive, Variable
        from hedge.optemplate import Field, ScalarParameter
        from hedge.tools.symbolic import make_common_subexpression as cse
        from math import pi

        sin = Variable("sin")
        log10 = Variable("log10")

        capital_s_e = Field("S_e")
        s_e = cse(log10(capital_s_e), "s_e")
        kappa = ScalarParameter("kappa")
        eps0 = ScalarParameter("eps0")
        s_0 = ScalarParameter("s_0")

        return IfPositive(s_0-self.kappa-s_e,
                0,
                IfPositive(s_e-self.kappa-s_0,
                    eps0,
                    eps0/2*(1+sin(pi*(s_e-s_0)/self.kappa))))

    def capital_s_e(self, u):
        truncated_u = self.mode_truncator(u)
        diff = u - truncated_u

        mass_diff = self.mass_op(f=diff)
        mass_u = self.mass_op(f=u)

        def ones(eg):
            return numpy.ones(
                    (eg.local_discretization.node_count(), 
                        eg.local_discretization.node_count()),
                    dtype=self.discr.default_scalar_type)

        el_norm_squared_mass_diff_u = self.discr.apply_element_local_matrix(
                eg_to_matrix=ones, field=mass_diff*diff,
                prepared_data_store=self.ones_data_store)
        el_norm_squared_mass_u = self.discr.apply_element_local_matrix(
                eg_to_matrix=ones, field=mass_u*u,
                prepared_data_store=self.ones_data_store)

        return el_norm_squared_mass_diff_u / el_norm_squared_mass_u

    def __call__(self, u):
        return self.threshold_op(
                S_e=self.capital_s_e(u),
                kappa=self.kappa, eps0=self.eps0, s_0=self.s_0)





# exponential fit -------------------------------------------------------------
class DecayFitDiscontinuitySensorBase(object):
    """
    sort of (but not quite) like
    [1] H. Feng und C. Mavriplis, "Adaptive Spectral Element 
    Simulations of Thin Premixed Flame Sheet Deformations," 
    Journal of Scientific Computing,  vol. 17, Dec. 2002, 
    p. 385-395.
    """

    def __init__(self, discr):
        from pytools import match_precision
        self.discr = discr

        self.ones_data_store = {}
        self.inverse_vdm_data_store = {}
        self.exponent_eval_vdm_data_store = {}
        self.alpha_projection_data_store = {}
        self.log_c_projection_data_store = {}

        from hedge.optemplate import MassOperator, Field
        self.mass_op = discr.compile(MassOperator() * Field("f"))

        self.order = max(
                eg.local_discretization.order
                for eg in discr.element_groups)

    def estimate_decay(self, u, ignore_modes=0, debug=False):
        mass_u = self.mass_op(f=u)

        def ones(eg):
            return numpy.ones(
                    (eg.local_discretization.node_count(), 
                        eg.local_discretization.node_count()),
                    dtype=self.discr.default_scalar_type)

        el_norm_u_squared = self.discr.apply_element_local_matrix(
                eg_to_matrix=ones, field=mass_u*u,
                prepared_data_store=self.ones_data_store)

        def inverse_vandermonde(eg):
            return numpy.asarray(
                    la.inv(eg.local_discretization.vandermonde()),
                    order="C")

        modal_coeffs = self.discr.apply_element_local_matrix(
                eg_to_matrix=inverse_vandermonde, field=u,
                prepared_data_store=self.inverse_vdm_data_store)

        eps = numpy.finfo(u.dtype).eps
        log_modal_coeffs = numpy.log(modal_coeffs**2+eps*el_norm_u_squared)/2

        # find least-squares fit to c*exp(alpha*n)
        def pinv_least_squares_mat(eg):
            ldis = eg.local_discretization
            a = numpy.zeros((ldis.node_count(), 2))
            a[:,0] = 1
            a[:,1] = numpy.arange(ldis.node_count())
            pinv = la.pinv(a)
            pinv[:, :ignore_modes] = 0
            return pinv

        def alpha_projection(eg):
            ldis = eg.local_discretization
            plsm = pinv_least_squares_mat(eg)
            a = numpy.zeros((ldis.node_count(), ldis.node_count()))
            for i in range(ldis.node_count()):
                a[i] = plsm[1]

            return a

        alpha = self.discr.apply_element_local_matrix(
                eg_to_matrix=alpha_projection, field=log_modal_coeffs,
                prepared_data_store=self.alpha_projection_data_store)
        if debug:
            print alpha[::6]
            print repr(log_modal_coeffs.reshape((11,6)))
            raw_input()

        def log_c_projection(eg):
            ldis = eg.local_discretization
            plsm = pinv_least_squares_mat(eg)
            a = numpy.zeros((ldis.node_count(), ldis.node_count()))
            for i in range(ldis.node_count()):
                a[i] = plsm[0]

            return a

        c = numpy.exp(self.discr.apply_element_local_matrix(
                eg_to_matrix=log_c_projection, field=modal_coeffs,
                prepared_data_store=self.log_c_projection_data_store))

        return alpha, c, el_norm_u_squared




class DecayGatingDiscontinuitySensorBase(
        DecayFitDiscontinuitySensorBase):
    def __init__(self, discr, max_viscosity=0.01):
        DecayFitDiscontinuitySensorBase.__init__(self, discr)

        self.max_viscosity = max_viscosity

        self.threshold_op = discr.compile(
                self.threshold_op_template())

    def threshold_op_template(self):
        from pymbolic.primitives import IfPositive, Variable
        from hedge.optemplate import Field, ScalarParameter
        from hedge.tools.symbolic import make_common_subexpression as cse
        from math import pi

        alpha = Field("alpha")
        sin = Variable("sin")

        def flat_end_sin(x):
            return IfPositive(-pi/2-x,
                    -1, IfPositive(x-pi/2, 1, sin(x)))

        return 0.5*self.max_viscosity*(1+flat_end_sin((alpha+1)*pi/2))

    def __call__(self, u):
        alpha, c, el_norm_u_squared = self.estimate_decay(u, ignore_modes=1)

        return self.threshold_op(alpha=alpha)




class ErrorEstimatingDiscontinuitySensorBase(
        DecayFitDiscontinuitySensorBase):
    def __call__(self, u):
        alpha, c, el_norm_u_squared = self.estimate_decay(u, 
                ignore_modes=max(1, self.order-4))

        #alpha = numpy.minimum(-0.1, alpha)

        alpha_integral_np1_to_inf = c**2/2*(-numpy.exp(2*alpha*(self.order+1)))
        alpha_integral_0_to_np1 = c**2/2*(numpy.exp(2*alpha*(self.order+1))-1)

        indicator = 100*numpy.sqrt(numpy.abs(alpha_integral_np1_to_inf)
                / (numpy.abs(alpha)*el_norm_u_squared 
                    + numpy.abs(alpha_integral_0_to_np1 + alpha_integral_np1_to_inf)))

        #if numpy.isnan(indicator).any():
            #from pudb import set_trace; set_trace()

        return indicator
