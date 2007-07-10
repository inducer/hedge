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




_RK4A = [0.0,
        -567301805773.0/1357537059087.0,
        -2404267990393.0/2016746695238.0,
        -3550918686646.0/2091501179385.0,
        -1275806237668.0/ 842570457699.0,
        ]

_RK4B = [1432997174477.0/ 9575080441755.0,
        5161836677717.0/13612068292357.0,
        1720146321549.0/ 2090206949498.0,
        3134564353537.0/ 4481467310338.0,
        2277821191437.0/14882151754819.0,
        ]

_RK4C = [0.0,
        1432997174477.0/9575080441755.0,
        2526269341429.0/6820363962896.0,
        2006345519317.0/3224310063776.0,
        2802321613138.0/2924317926251.0,
        1.0,
        ]





class RK4TimeStepper:
    def __call__(self, y, t, dt, rhs):
        try:
            self.residual
        except AttributeError:
            self.residual = 0*rhs(t, y)

        for a, b, c in zip(_RK4A, _RK4B, _RK4C):
            self.residual = a*self.residual + dt*rhs(t + c*dt, y)
            y += b * self.residual

        return y

