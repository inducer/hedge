class AffineMap(object):
    def __init__(self, matrix, vector):
        """Construct an affine map given by f(x) = matrix * x + vector."""
        self.matrix = matrix
        self.vector = vector

    def __call__(self, x):
        """Apply this map."""
        return self.matrix*x + self.vector

    def inverted(self):
        """Return a new AffineMap that is the inverse of this one.
        """
        return AffineMap(1/self.matrix, -self.matrix <<num.solve>> vector)

    def _jacobian(self):
        import pylinear.computation as comp
        result = comp.determinant(self.matrix)
        self.jacobian = result
        return result
    jacobian = property(_jacobian, doc="The jacobian of the map.")




def plot_1d(f, a, b, steps=100):
    h = float(b - a)/steps

    points = []
    data = []
    for n in range(steps):
        x = a + h * n
        points.append(x)
        data.append(f(x))

    from Gnuplot import Gnuplot, Data
    gp = Gnuplot()
    gp.plot(Data(points, data))
    raw_input()




