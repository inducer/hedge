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
