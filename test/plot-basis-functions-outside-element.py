from __future__ import division
from hedge.element import TriangularElement

tri = TriangularElement(5)

import Gnuplot
gp = Gnuplot.Gnuplot()

import numpy

xpts = numpy.arange(-1.5, 1.5, 0.1)
ypts = numpy.arange(-1.5, 1.5, 0.1)

gp("set zrange [-3:3]")

for bfi, bf in zip(tri.generate_mode_identifiers(), tri.basis_functions()):
    lines = []
    for x in xpts:
        values = []
        for y in ypts:
            values.append((x, y, bf(numpy.array((x,y)))))
        lines.append(Gnuplot.Data(values, with_="lines"))

    for y in xpts:
        values = []
        for x in ypts:
            values.append((x, y, bf(numpy.array((x,y)))))
        lines.append(Gnuplot.Data(values, with_="lines"))

    tri = numpy.array([
        (-1,-1,0),
        (-1,1,0),
        (1,-1,0),
        (-1,-1,0),
        ])
    lines.append(Gnuplot.Data(tri, with_="lines"))

    gp.splot(*lines)
    raw_input(str(bfi))

