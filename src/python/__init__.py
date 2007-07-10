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




"""
Hedge
-----

Hybrid'n'Easy Discontinuous Galerkin Environment
"""



if __name__ == "__main__":
    from hedge.element import Triangle
    tri = Triangle(17)
    #print tri.vandermonde()
    outf = file("nodes.dat", "w")
    nodes = list(tri.unit_nodes())
    face_idx = list(tri.face_indices())
    for fi, i in enumerate(face_idx[0]+face_idx[1]+face_idx[2]):
        outf.write("%f\t%f\t%f\n" % (nodes[i][0], nodes[i][1], fi))
    from hedge.polynomial import legendre_polynomial
    from pymbolic import differentiate, var

    #lp = legendre_polynomial(17)
    #print lp
    #print repr(lp.data[0][1])
    #print differentiate(lp, var("x"))
