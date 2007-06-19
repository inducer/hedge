import pylinear.array as num




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
        return AffineMap(1/self.matrix, -self.matrix <<num.solve>> self.vector)

    @property
    def jacobian(self):
        "Get the (constant) jacobian of the map."
        try:
            return self._jacobian
        except AttributeError:
            from pylinear.computation import determinant
            self._jacobian = determinant(self.matrix)
            return self._jacobian




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




def reduction_matrix(indices, big_len):
    import pylinear.array as num
    result = num.zeros((len(indices), big_len), flavor=num.SparseBuildMatrix)
    for i, j in enumerate(indices):
        result[i,j] = 1
    return result
