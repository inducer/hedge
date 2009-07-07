# -*- coding: utf8 -*-

"""ODE solvers: timestepping support, such as Runge-Kutta, Adams-Bashforth, etc."""

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
from pytools import memoize




# coefficient generators ------------------------------------------------------
_RK4A = [0.0,
        -567301805773 /1357537059087,
        -2404267990393/2016746695238,
        -3550918686646/2091501179385,
        -1275806237668/ 842570457699,
        ]

_RK4B = [1432997174477/ 9575080441755,
        5161836677717 /13612068292357,
        1720146321549 / 2090206949498,
        3134564353537 / 4481467310338,
        2277821191437 /14882151754819,
        ]

_RK4C = [0.0,
        1432997174477/9575080441755,
        2526269341429/6820363962896,
        2006345519317/3224310063776,
        2802321613138/2924317926251,
        1,
        ]




def make_generic_ab_coefficients(levels, int_start, tap):
    """Find coefficients (αᵢ) such that
       ∑ᵢ αᵢ F(tᵢ) = ∫[int_start..tap] f(t) dt."""

    # explanations --------------------------------------------------------------
    # To calculate the AB coefficients this method makes use of the interpolation
    # connection of the Vandermonde matrix:
    #         
    #  Vᵀ * α = fe(t₀₊₁),                                    (1)
    #
    # with Vᵀ as the transposed Vandermonde matrix (with monomial base: xⁿ), 
    # 
    #  α = (..., α₋₂, α₋₁,α₀)ᵀ                               (2)
    #
    # a vector of interpolation coefficients and 
    # 
    #  fe(t₀₊₁) = (t₀₊₁⁰, t₀₊₁¹, t₀₊₁²,...,t₀₊₁ⁿ)ᵀ           (3)
    #
    # a vector of the evaluated interpolation polynomial f(t) at t₀₊₁ = t₀ ∓ h 
    # (h being any arbitrary stepsize).
    #
    # Solving the system (1) by knowing Vᵀ and fe(t₀₊₁) receiving α makes it 
    # possible for any function F(t) - the function which gets interpolated 
    # by the interpolation polynomial f(t) - to calculate f(t₀₊₁) by:
    #
    # f(t₀₊₁) =  ∑ᵢ αᵢ F(tᵢ)                                 (5)
    #
    # with F(tᵢ) being the values of F(t) at the sampling points tᵢ.
    # --------------------------------------------------------------------------
    # The Adams-Bashforth method is defined by:
    #
    #  y(t₀₊₁) = y(t₀) + Δt * ∫₀⁰⁺¹ f(t) dt                  (6)
    #
    # with:
    # 
    #  ∫₀⁰⁺¹ f(t) dt = ∑ᵢ ABcᵢ F(tᵢ),                        (8)
    #
    # with ABcᵢ = [AB coefficients], f(t) being the interpolation polynomial,
    # and F(tᵢ) being the values of F (= RHS) at the sampling points tᵢ.
    # --------------------------------------------------------------------------
    # For the AB method (1) becomes:
    #
    #  Vᵀ * ABc = ∫₀⁰⁺¹ fe(t₀₊₁)                             (7)
    #
    # with ∫₀⁰⁺¹ fe(t₀₊₁) being a vector evalueting the integral of the 
    # interpolation polynomial in the form oft 
    # 
    #  1/(n+1)*(t₀₊₁⁽ⁿ⁾-t₀⁽ⁿ⁾)                               (8)
    # 
    #  for n = 0,1,...,N sampling points, and 
    # 
    # ABc = [c₀,c₁, ... , cn]ᵀ                               (9)
    #
    # being the AB coefficients.
    # 
    # For example ∫₀⁰⁺¹ f(t₀₊₁) evaluated for the timestep [t₀,t₀₊₁] = [0,1]
    # is:
    #
    #  point_eval_vec = [1, 0.5, 0.333, 0.25, ... ,1/n]ᵀ.
    #
    # For substep levels the bounds of the integral has to be adapted to the
    # size and position of the substep interval: 
    # 
    #  [t₀,t₀₊₁] = [substep_int_start, substep_int_end] 
    # 
    # which is equal to the implemented [int_start, tap].
    #
    # Since Vᵀ and ∫₀⁰⁺¹ f(t₀₊₁) is known the AB coefficients c can be
    # predicted by solving system (7) and calculating:
    # 
    #  ∫₀⁰⁺¹ f(t) dt = ∑ᵢ ABcᵢ F(tᵢ),

    from hedge.polynomial import monomial_vdm
    point_eval_vec = numpy.array([
        1/(n+1)*(tap**(n+1)-int_start**(n+1)) for n in range(len(levels))])
    return la.solve(monomial_vdm(levels).T, point_eval_vec)




@memoize
def make_ab_coefficients(order):
    return make_generic_ab_coefficients(numpy.arange(0, -order, -1), 0, 1)




# time steppers ---------------------------------------------------------------
class TimeStepper(object):
    pass




class RK4TimeStepper(TimeStepper):
    dt_fudge_factor = 1

    def __init__(self, allow_jit=True):
        from pytools.log import IntervalTimer, EventCounter
        self.timer = IntervalTimer(
                "t_rk4", "Time spent doing algebra in RK4")
        self.flop_counter = EventCounter(
                "n_flops_rk4", "Floating point operations performed in RK4")
        self.coeffs = zip(_RK4A, _RK4B, _RK4C)

        self.allow_jit = allow_jit

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def __call__(self, y, t, dt, rhs):
        try:
            self.residual
        except AttributeError:
            self.residual = 0*rhs(t, y)
            from hedge.tools import count_dofs, has_data_in_numpy_arrays
            self.dof_count = count_dofs(self.residual)

            self.use_jit = self.allow_jit and has_data_in_numpy_arrays(y)

        if self.use_jit:
            from hedge.tools import numpy_linear_comb

            for a, b, c in self.coeffs:
                this_rhs = rhs(t + c*dt, y)

                sub_timer = self.timer.start_sub_timer()
                self.residual = numpy_linear_comb([(a, self.residual), (dt, this_rhs)])
                del this_rhs
                y = numpy_linear_comb([(1, y), (b, self.residual)])
                sub_timer.stop().submit()
        else:
            for a, b, c in self.coeffs:
                this_rhs = rhs(t + c*dt, y)

                sub_timer = self.timer.start_sub_timer()
                self.residual = a*self.residual + dt*this_rhs
                del this_rhs
                y = y + b * self.residual
                sub_timer.stop().submit()

        self.flop_counter.add(len(self.coeffs)*self.dof_count*5)

        return y




class AdamsBashforthTimeStepper(TimeStepper):
    dt_fudge_factor = 0.95

    def __init__(self, order, startup_stepper=None):
        self.coefficients = make_ab_coefficients(order)
        self.f_history = []

        if startup_stepper is not None:
            self.startup_stepper = startup_stepper
        else:
            self.startup_stepper = RK4TimeStepper()

    def __call__(self, y, t, dt, rhs):
        if len(self.f_history) == 0:
            # insert IC
            self.f_history.append(rhs(t, y))

        if len(self.f_history) < len(self.coefficients):
            ynew = self.startup_stepper(y, t, dt, rhs)
            if len(self.f_history) == len(self.coefficients) - 1:
                # here's some memory we won't need any more
                del self.startup_stepper

        else:
            from operator import add

            assert len(self.coefficients) == len(self.f_history)
            ynew = y + dt * reduce(add,
                    (coeff * f 
                        for coeff, f in 
                        zip(self.coefficients, self.f_history)))

            self.f_history.pop()

        self.f_history.insert(0, rhs(t+dt, ynew))
        return ynew




# helper functions ------------------------------------------------------------
def _rotate_insert(l, new_item):
    l.pop()
    l.insert(0, new_item)

def _linear_comb(coefficients, vectors):
    from operator import add
    return reduce(add,
            (coeff * v for coeff, v in
                zip(coefficients, vectors)))





class TwoRateAdamsBashforthTimeStepperBase(TimeStepper):
    """Simultaneously timesteps two parts of an ODE system,
    the first with a small timestep, the second with a large timestep.

    [1] C.W. Gear and D.R. Wells, "Multirate linear multistep methods," BIT
    Numerical Mathematics,  vol. 24, Dec. 1984, pg. 484-502.
    """

    def __init__(self, large_dt, step_ratio, order,method,
            startup_stepper=None):
        """

        If `substepping` is set to True, then an 
        exptrapolation of the state of the large-dt part of the
        system is computed for each small-dt timestep. If set to
        False, None will instead be passed to right-hand sides
        that would otherwise receive it.

        If `slowest_first` is set to True, then the time-stepper uses the
        "slowest-first" approach from [1].

        If `fastest_first` is set to True, then the time-stepper uses the 
        "fastest-first" approach from [1].
        """

        self.large_dt = large_dt
        self.small_dt = large_dt/step_ratio
        self.step_ratio = step_ratio

        #print "method:",method
        #raw_input()
        #self.large_dt_for_small = large_dt_for_small
        if method==[]:
            self.method = ["fastest_first"]
        else:
            self.method = method

        # get the "standard" AB coefficients extrapolating y for an entire
        # large_dt timestep:
        self.coefficients = make_ab_coefficients(order)

        # get the AB extrapolation coeffcients on
        # small_dt substep level:
        self.ab_extrapol_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    (i-1)/step_ratio,
                    i/step_ratio)
                for i in range(1, step_ratio+1)]

        # get the side effect free AB extrapolation 
        # coeffcients on small_dt substep level:
        self.ab_extrapol_side_effect_free_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    0,
                    i/step_ratio)
                for i in range(1, step_ratio+1)]

        # rhs_histories is row major--see documentation for 
        # rhss arg of __call__.
        self.rhs_histories = [[] for i in range(2*2)]

        if startup_stepper is not None:
            self.startup_stepper = startup_stepper
        else:
            self.startup_stepper = RK4TimeStepper()

        self.startup_history = []


    def __call__(self, ys, t, rhss):
        """
        @arg rhss: Matrix of right-hand sides, stored in row-major order, i.e.
        C{[s2f, s2f, f2s, s2s]}.
        """
        from hedge.tools import make_obj_array

        def finish_startup():
            # we're done starting up, pack data into split histories
            hist_f2f, hist_s2f, hist_f2s, hist_s2s = zip(*self.startup_history)

            n = len(self.coefficients)

            if "eval_s2f_on_substeplevel" in self.method:
                hist_s2f = list(hist_s2f[:n])
            else:
                hist_s2f = list(hist_s2f[::self.step_ratio])

            self.rhs_histories = [
                    list(hist_f2f[:n]),
                    hist_s2f,
                    list(hist_f2s[::self.step_ratio]),
                    list(hist_s2s[::self.step_ratio])
                    ]

            from pytools import single_valued
            assert single_valued(len(h) for h in self.rhs_histories) == n

            # here's some memory we won't need any more
            self.startup_stepper = None
            del self.startup_history

        def combined_rhs(t, y):
            return make_obj_array([rhs(t, *y) for rhs in rhss])

        def combined_summed_rhs(t, y):
            return numpy.sum(combined_rhs(t, y).reshape((2,2), order="C"), axis=1)


        if self.startup_stepper is not None:
            ys = make_obj_array(ys)

            if len(self.coefficients) == 1:
                # we're running forward Euler, no need for the startup stepper

                assert not self.startup_history
                self.startup_history.append(combined_rhs(t, ys))
                finish_startup()
                return run_ab()

            for i in range(self.step_ratio):
                ys = self.startup_stepper(ys, t+i*self.small_dt, self.small_dt, 
                        combined_summed_rhs)
                self.startup_history.insert(0, combined_rhs(t+(i+1)*self.small_dt, ys))

            if len(self.startup_history) == len(self.coefficients)*self.step_ratio:
                finish_startup()

            return ys
        else:
            return self.run_ab(ys, t, rhss)




class TwoRateAdamsBashforthTimeStepperFastestFirstMethod(TwoRateAdamsBashforthTimeStepperBase):
    def __init__(self, large_dt, step_ratio, order,method,
            startup_stepper=None):
        TwoRateAdamsBashforthTimeStepperBase.__init__(self, large_dt, step_ratio, order, method,
                startup_stepper)

    def run_ab(self, ys, t, rhss):
        y_fast, y_slow = ys
        rhs_f2f, rhs_s2f, rhs_f2s, rhs_s2s = rhss
        hist_f2f, hist_s2f, hist_f2s, hist_s2s = self.rhs_histories

        coeff = self.coefficients

        y_slow_substep = y_slow

        y_slow_start = y_slow

        # substep the faster component - y_fast -
        # first and extrapolate the the slow component -y_slow:
        for i in range(self.step_ratio):
            # Extrapolation of coupling RHS, l2s, on substep level.
            sub_ex_coeff = self.ab_extrapol_substep_coefficients[i]
            sub_ex_side_free_coeffs = self.ab_extrapol_side_effect_free_substep_coefficients[i]
#
            if "eval_s2f_on_substeplevel" in self.method:
                y_fast = y_fast + ( self.small_dt * _linear_comb(coeff, hist_f2f)
                        + self.small_dt * _linear_comb(coeff, hist_s2f))
            else:
                y_fast = y_fast + ( self.small_dt * _linear_comb(coeff, hist_f2f)
                        + self.large_dt * _linear_comb(sub_ex_coeff, hist_s2f))

            if i == self.step_ratio-1:
                break

            #if "large_dt_for_small" in self.method:
            #    y_slow_substep = y_slow_substep + (
            #            self.large_dt * _linear_comb(sub_ex_coeff, hist_s2s)
            #            + self.large_dt * _linear_comb(sub_ex_coeff, hist_f2s))



            if "eval_s2f_on_substeplevel" in self.method:
                y_slow_substep = y_slow_substep + (
                        self.large_dt * _linear_comb(sub_ex_coeff, hist_s2s)
                        + self.large_dt * _linear_comb(sub_ex_coeff, hist_f2s))

                _rotate_insert(hist_s2f, rhs_s2f(t+(i+1)*self.small_dt,
                    y_fast, y_slow_substep))

            #else:
            #    # If s2s only f(y_small) - PIC !!! - calculation of y_slow on 
            #    # substep level not required.
            #    y_slow_substep = None

            # compute s2s-RHS of fast component:
            #_rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
            #              y_fast, y_slow_substep))

            def y_slow_substep_func():
                return y_slow + (
                        self.large_dt * _linear_comb(sub_ex_side_free_coeffs, hist_s2s)
                        + self.large_dt * _linear_comb(sub_ex_side_free_coeffs, hist_f2s))

            _rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep_func()))

        if "large_dt_for_small" in self.method:
            sub_ex_coeffs = self.ab_extrapol_substep_coefficients[
                    self.step_ratio-1]
            #y_slow = y_slow_substep + (
            #        self.large_dt * _linear_comb(sub_ex_coeffs, hist_s2s)
            #        + self.large_dt * _linear_comb(sub_ex_coeffs, hist_f2s))
            y_slow = y_slow_start + (
                    self.large_dt * _linear_comb(coeff, hist_s2s)
                    + self.large_dt * _linear_comb(coeff, hist_f2s))

            # calculate all RHS running on dt_large level:
            _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast, y_slow))
            _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast, y_slow))

            # calculate all RHS running on substep - dt_small - level:
            _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow))
            _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow))

        elif "eval_s2f_on_substeplevel" in self.method:
            # step the 'large' part based on the substepped history of
            # s2l-RHS and y_large_substep. Extrapolation of l2l-RHS for
            # last substep reqired.
            sub_ex_coeff = self.ab_extrapol_substep_coefficients[
                    self.step_ratio-1]
            y_slow = y_slow_substep + (
                    self.large_dt * _linear_comb(sub_ex_coeff, hist_s2s)
                    + self.large_dt * _linear_comb(sub_ex_coeff, hist_f2s))

            # calculate all RHS running on dt_large level:
            _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast, y_slow))
            _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast, y_slow))

            # calculate all RHS running on substep - dt_small - level:
            _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow))
            _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow))


        else:
            # step the 'large' part
            y_slow = y_slow + self.large_dt * (
                    _linear_comb(coeff, hist_s2s)
                    + _linear_comb(coeff, hist_f2s))

            # calculate all RHS running on dt_large level:
            _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow))
            _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast, y_slow))
            _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast, y_slow))

            # calculate RHS running on substep - dt_small - level:
            _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow))

        from hedge.tools import make_obj_array
        return make_obj_array([y_fast, y_slow])






class TwoRateAdamsBashforthTimeStepperSlowestFirstMethod(TwoRateAdamsBashforthTimeStepperBase):
    def __init__(self, large_dt, step_ratio, order,method,
            startup_stepper=None):
        TwoRateAdamsBashforthTimeStepperBase.__init__(self, large_dt, step_ratio, order, method,
                startup_stepper)

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    (i-1)/step_ratio-1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

    def run_ab(self, ys, t, rhss):
        y_fast, y_slow = ys
        rhs_f2f, rhs_s2f, rhs_f2s, rhs_s2s = rhss
        hist_f2f, hist_s2f, hist_f2s, hist_s2s = self.rhs_histories

        coeff = self.coefficients

        y_slow_substep = y_slow

        # extrapolate y_slow from t=0 to t=1
        y_slow = y_slow + (
                self.large_dt * _linear_comb(coeff, hist_s2s)
                + self.large_dt * _linear_comb(coeff, hist_f2s))

        # update RHS's running 'large_dt' level:
        _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast, y_slow))
        _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast, y_slow))

        for i in range(self.step_ratio):
            # substep fast component - y_small - and interpolate
            # the values of the coupling l2s-RHS:
            sub_int_coeffs = self.ab_interp_substep_coefficients[i]
            y_fast = y_fast + (
                    self.small_dt * _linear_comb(coeff, hist_f2f)
                    + self.large_dt * _linear_comb(sub_int_coeffs, hist_s2f)
                    )

            if i == self.step_ratio-1:
                break

            if "large_dt_for_small" in self.method:
                # If s2s = f(y_fast, y_slow), inter/extra-polation [l2l/s2l]
                # of y_slow on substep level required:
                sub_ex_coeff =  self.ab_extrapol_substep_coefficients[i]
                y_slow_substep =  y_slow_substep + (
                        self.large_dt * _linear_comb(sub_int_coeffs, hist_s2s)
                        + self.large_dt * _linear_comb(sub_ex_coeff, hist_f2s))

            else:
                y_slow_substep=None

            # compute s2s-RHS of fast component:
            _rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep))

        if "large_dt_for_small" in self.method:
            sub_int_coeffs = self.ab_interp_substep_coefficients[self.step_ratio-1]
            sub_ex_coeffs =  self.ab_extrapol_substep_coefficients[self.step_ratio-1]
            y_slow = y_slow_substep + (
                    self.large_dt * _linear_comb(sub_int_coeffs, hist_s2s)
                    + self.large_dt * _linear_comb(sub_ex_coeffs, hist_f2s))

            # calculate missing RHS's:
            _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow))
            _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow))

        else:
            _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow_substep))
            _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow))

        from hedge.tools import make_obj_array
        return make_obj_array([y_fast, y_slow])





# bisection based method to find bounds of stability region on Imaginary axis only ---
def calculate_fudged_stability_region(stepper_class, *stepper_args):
    return calculate_stability_region(stepper_class, *stepper_args) \
            * stepper_class.dt_fudge_factor




@memoize
def calculate_stability_region(stepper_class, *stepper_args):
    def stepper_maker():
        return stepper_class(*stepper_args)

    prec = 1e-5

    def is_stable(stepper, k):
        y = 1
        for i in range(20):
            if abs(y) > 2:
                return False
            y = stepper(y, i, 1, lambda t, y: k*y)
        return True

    def make_k(angle, mag):
        from cmath import exp
        return -prec+mag*exp(1j*angle)

    def refine(stepper_maker, angle, stable, unstable):
        assert is_stable(stepper_maker(), make_k(angle, stable))
        assert not is_stable(stepper_maker(), make_k(angle, unstable))
        while abs(stable-unstable) > prec:
            mid = (stable+unstable)/2
            if is_stable(stepper_maker(), make_k(angle, mid)):
                stable = mid
            else:
                unstable = mid
        else:
            return stable

    def find_stable_k(stepper_maker, angle):
        mag = 1

        if is_stable(stepper_maker(), make_k(angle, mag)):
            mag *= 2
            while is_stable(stepper_maker(), make_k(angle, mag)):
                mag *= 2

                if mag > 2**8:
                    return mag
            return refine(stepper_maker, angle, mag/2, mag)
        else:
            mag /= 2
            while not is_stable(stepper_maker(), make_k(angle, mag)):
                mag /= 2

                if mag < prec:
                    return mag
            return refine(stepper_maker, angle, mag, mag*2)

    points = []
    from cmath import pi
    for angle in numpy.array([pi/2, 3/2*pi]):
        points.append(make_k(angle, find_stable_k(stepper_maker, angle)))

    points = numpy.array(points)

    return abs(points[0])
