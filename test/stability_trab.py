"""Automatic size finding for stability regions."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Stock, Andreas Kloeckner"

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



from ode_systems import Basic, \
        Full, \
        Real, \
        Comp, \
        CC,\
        Tria, \
        Inh, \
        Inh2

from math import sqrt, log, sin, cos, exp
import numpy

from hedge.timestep.multirate_ab.methods import methods
if False:
    methods_man = ['f_f_1a', 'f_f_1b',
            's_f_1', 's_f_1_nr',
            's_f_2a', 's_f_2a_nr',
            's_f_2b', 's_f_2b_nr',
            's_f_3a', 's_f_3a_nr',
            's_f_3b', 's_f_3b_nr',
            's_f_4', 's_f_4_nr']

else:
    methods_man = ['f_f_1a']







# bisection based method to find bounds of stability region on Imaginary axis only ---
def calculate_fudged_stability_region(stepper_class, *stepper_args):
    return calculate_stability_region(stepper_class, *stepper_args) \
            * stepper_class.dt_fudge_factor





def calc_stab_reg():

    from hedge.timestep.multirate_ab import \
                     TwoRateAdamsBashforthTimeStepper
    from hedge.timestep.ab import AdamsBashforthTimeStepper
    stepper_class = AdamsBashforthTimeStepper
    stepper_args = [3]
    order = 5
    step_ratio = 10
    ode = Inh
    #stepper = TwoRateAdamsBashforthTimeStepper(
    #                self.method, dt, self.step_ratio, self.order)
    find_sta_reg = calculate_stability_region('f_f_1a', order, step_ratio, ode)
    print "stabel_dt:", find_sta_reg()






#@memoize
class calculate_stability_region:
    def __init__(self, method, order, step_ratio, ode):
        self.method      = method
        self.order       = order
        self.step_ratio  = step_ratio
        self.ode         = ode()

        from hedge.timestep.multirate_ab import \
                TwoRateAdamsBashforthTimeStepper
        #self.stepper = TwoRateAdamsBashforthTimeStepper(
        #        self.method, dt, self.step_ratio, self.order)

        self.prec = 1e-2

    def is_stable(self, dt):
        from hedge.timestep.multirate_ab import \
                TwoRateAdamsBashforthTimeStepper
        stepper = TwoRateAdamsBashforthTimeStepper(self.method, dt, self.step_ratio, self.order)
        t = self.ode.t_start
        y = numpy.array([self.ode.soln_0(t),self.ode.soln_1(t)])
    print dt
        for i in range(20):
            err = abs(
                    sqrt(y[0]**2 + y[1]**2)
                    - sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2)
                    )
            print err, y, i
            raw_input()
            if err > 1:
                return False
            y = stepper(y, t, (self.ode.f2f_rhs, self.ode.s2f_rhs, self.ode.f2s_rhs, self.ode.s2s_rhs))
            t += dt
        return True


    def refine(self, stable_dt, unstable_dt):
        assert self.is_stable(stable_dt)
        assert not self.is_stable(unstable_dt)
        while abs(stable_dt-unstable_dt) > self.prec:
            mid_dt = (stable_dt+unstable_dt)/2
            print "refine", mid_dt
            if self.is_stable(mid_dt):
                stable = mid_dt
            else:
                unstable = mid_dt
        else:
            return stable

    def find_stable_dt(self):
        dt = 1/2

        if self.is_stable(dt):
            dt *= 2
            while self.is_stable(dt):
                dt *= 2

                if dt > 2**8:
                    return dt
            return self.refine(dt/2, dt)
        else:
            while not self.is_stable(dt):
                dt /= 2

                if dt < self.prec:
                    return dt
            return self.refine(dt, dt*2)

    # -----------------------------------------------------------------------------
    def __call__(self):
        #import rpdb2; rpdb2.start_embedded_debugger_interactive_password()
        return self.find_stable_dt()





if __name__ == "__main__":
    #from py.test.cmdline import main
    #main([__file__])
    #test_multirate_timestep_accuracy()
    calc_stab_reg()
