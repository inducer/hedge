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
import numpy
from pytools import memoize




@memoize
def approximate_rk4_relative_imag_stability_region(
        stepper=None, stepper_class=None, stepper_args=[]):
    if stepper is not None and stepper_class is not None:
        raise ValueError("only one of 'stepper' and 'stepper_class' "
                "may be specified")

    if stepper is not None:
        stepper_class = type(stepper)
        stepper_args = stepper.get_stability_relevant_init_args()
        stepper = None

    from hedge.timestep.runge_kutta import LSRK4TimeStepper
    if stepper_class is None or stepper_class == LSRK4TimeStepper:
        return 1
    else:
        assert isinstance(stepper_class, type)

        from hedge.timestep.stability import \
                approximate_imag_stability_region

        return (approximate_imag_stability_region(
                    stepper_class, *stepper_args)
                / approximate_imag_stability_region(LSRK4TimeStepper)
                * stepper_class.dt_fudge_factor
                / LSRK4TimeStepper.dt_fudge_factor)




@memoize
def approximate_imag_stability_region(stepper_class, *stepper_args):
    def stepper_maker():
        return stepper_class(*stepper_args, **{"dtype": numpy.complex128})

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

    def refine(angle, stable, unstable):
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

    def find_stable_k(angle):
        mag = 1

        if is_stable(stepper_maker(), make_k(angle, mag)):
            mag *= 2
            while is_stable(stepper_maker(), make_k(angle, mag)):
                mag *= 2

                if mag > 2**8:
                    return mag
            return refine(angle, mag/2, mag)
        else:
            mag /= 2
            while not is_stable(stepper_maker(), make_k(angle, mag)):
                mag /= 2

                if mag < prec:
                    return mag
            return refine(angle, mag, mag*2)

    from cmath import pi
    angle = pi/2
    return abs(make_k(angle, find_stable_k(angle)))
