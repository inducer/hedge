"""1D Newton interpolation helper module."""

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




import pymbolic




def newton_interpolation_coefficients(x, y):
    assert len(x) == len(y)
    n = len(y)
    divided_differences = [y]
    last = y

    for step in range(1, n):
        next = [(last[i+1]-last[i])/(x[i+step]-x[i])
                for i in range(n-step)]
        divided_differences.append(next)
        last = next

    return [dd_col[-1] for dd_col in divided_differences]




def newton_interpolation_polynomial(x, y):
    coeff = newton_interpolation_coefficients(x, y)

    var_x = pymbolic.var("x")
    linear_factors = [
            pymbolic.Polynomial(var_x, ((0, pt), (1, 1)))
            for pt in x]
    pyramid_linear_factors = [1]
    for l in linear_factors:
        pyramid_linear_factors.append(
                pyramid_linear_factors[-1]*l)

    return pymbolic.linear_combination(coeff, pyramid_linear_factors)




def newton_interpolation_function(x, y):
    return pymbolic.compile(newton_interpolation_polynomial(x, y), ["x"])
