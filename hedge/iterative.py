"""Iterative solution of linear systems of equations."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""





import numpy




class OperatorBase(object):
    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    def matvec(self, x):
        return self(x)

    def __neg__(self):
        return NegOperator(self)

class NegOperator(OperatorBase):
    def __init__(self, sub_op):
        self.sub_op = sub_op

    @property
    def dtype(self):
        return self.sub_op.dtype

    @property
    def shape(self):
        return self.sub_op.shape

    def __call__(self, operand):
        return -self.sub_op(operand)

class IdentityOperator(OperatorBase):
    def __init__(self, dtype, n):
        self.my_dtype = dtype
        self.n = n

    @property
    def dtype(self):
        return self.my_dtype

    @property
    def shape(self):
        return self.n, self.n

    def __call__(self, operand):
        return operand




class DiagonalPreconditioner(OperatorBase):
    def __init__(self, diagonal):
        self.diagonal = diagonal

    @property
    def dtype(self):
        return self.diagonal.dtype

    @property
    def shape(self):
        n = self.diagonal.shape[0]
        return n, n

    def __call__(self, operand):
        return self.diagonal*operand




class ConvergenceError(RuntimeError):
    pass




class CGStateContainer:
    def __init__(self, operator, precon=None, dot=None):
        if precon is None:
            precon = IdentityOperator(operator.dtype, operator.shape[0])

        self.operator = operator
        self.precon = precon

        if dot is None:
            dot = numpy.dot

        def inner(a, b):
            return dot(a, b.conj())

        self.inner = inner

    def reset(self, rhs, x=None):
        self.rhs = rhs

        if x is None:
            x = numpy.zeros((self.operator.shape[0],))
        self.x = x

        ax = self.operator(x)
        self.residual = rhs - ax

        self.d = self.precon(self.residual)

        self.delta = self.inner(self.residual, self.d)
        return self.delta

    def one_iteration(self, compute_real_residual=False):
        # typed up from J.R. Shewchuk,
        # An Introduction to the Conjugate Gradient Method
        # Without the Agonizing Pain, Edition 1 1/4 [8/1994]
        # Appendix B3

        q = self.operator(self.d)
        myip = self.inner(self.d, q)
        alpha = self.delta / myip

        self.x += alpha * self.d

        if compute_real_residual:
            self.residual = self.rhs - self.operator(self.x)
        else:
            self.residual -= alpha*q

        s = self.precon(self.residual)
        delta_old = self.delta
        self.delta = self.inner(self.residual, s)

        beta = self.delta / delta_old
        self.d = s + beta * self.d

        return self.delta

    def run(self, max_iterations=None, tol=1e-7, debug_callback=None, debug=0):
        if max_iterations is None:
            max_iterations = 10 * self.operator.shape[0]

        if self.inner(self.rhs, self.rhs) == 0:
            return self.rhs

        iterations = 0
        delta_0 = delta = self.delta
        while iterations < max_iterations:
            compute_real_residual = \
                    iterations % 50 == 0 or \
                    abs(delta) < tol*tol * abs(delta_0)

            delta = self.one_iteration(
                    compute_real_residual=compute_real_residual)

            if debug_callback is not None:
                if compute_real_residual:
                    what = "it+residual"
                else:
                    what = "it"

                debug_callback(what, iterations, self.x,
                        self.residual, self.d, delta)

            if compute_real_residual and abs(delta) < tol*tol * abs(delta_0):
                if debug_callback is not None:
                    debug_callback("end", iterations, self.x, self.residual, self.d, delta)
                if debug:
                    print "%d iterations" % iterations
                return self.x

            if debug and iterations % debug == 0:
                print "debug: delta=%g" % delta
            iterations += 1

        raise ConvergenceError("cg failed to converge")




def parallel_cg(pcon, operator, b, precon=None, x=None, tol=1e-7, max_iterations=None,
        debug=False, debug_callback=None, dot=None):
    if x is None:
        x = numpy.zeros((operator.shape[1],))

    cg = CGStateContainer(operator, precon, dot=dot)
    cg.reset(b, x)

    if not pcon.is_head_rank:
        debug = False

    return cg.run(max_iterations, tol, debug_callback, debug)
