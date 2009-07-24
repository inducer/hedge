from math import pi, sqrt
import numpy
from numpy import newaxis
import pyublas
import pickle


# Create Axis: (case_1)
a_points = numpy.linspace(0, 1, 1)
b_points = numpy.linspace(0, 2*pi, 20)
c_points = numpy.linspace(0, 2*pi, 20)

data = numpy.empty(
        (len(a_points), len(b_points), len(c_points)),
        dtype=numpy.float64, order="F")

dataset = pickle.load(open("case_1_res_a_0.001.dat"))
data = dataset.pop("case_1_res")

#print numpy.shape(data)
#raw_input()

from pylo import SiloFile, DB_NODECENT
sf = SiloFile("stab_trab.silo")
sf.put_quadmesh("mesh", [a_points, b_points, c_points])
sf.put_quadvar1("TRAB_stability", "mesh", data, data.shape, DB_NODECENT)
sf.close()

data_2D = data[0][:][:]
sf = SiloFile("stab_trab_2d.silo")
sf.put_quadmesh("mesh", [b_points, c_points])
sf.put_quadvar1("TRAB_stability", "mesh", data_2D, data_2D.shape, DB_NODECENT)
sf.close()
