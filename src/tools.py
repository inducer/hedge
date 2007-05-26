class AffineMap(object):
    def __init__(self, matrix, vector):
        self.matrix = matrix
        self.vector = vector

    def __call__(self, x):
        return self.matrix*x + self.vector

    def invert(self):
        return AffineMap(1/self.matrix, -self.matrix <<num.solve>> vector)

    def jacobian(self):
        import pylinear.computation as comp
        return comp.determinant(self.matrix)




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




