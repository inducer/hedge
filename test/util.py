from __future__ import division
import numpy




class Monomial:
    def __init__(self, exponents, factor=1):
        self.exponents = exponents
        self.ones = numpy.ones((len(self.exponents),))
        self.factor = factor

    def __call__(self, x):
        from operator import mul

        eps = 1e-15
        x = (x+self.ones)/2
        for xi in x:
            assert -eps <= xi <= 1+eps
        return self.factor* \
                reduce(mul, (x[i]**alpha 
                    for i, alpha in enumerate(self.exponents)))

    def theoretical_integral(self):
        from pytools import factorial
        from operator import mul

        return (self.factor*2**len(self.exponents)*
            reduce(mul, (factorial(alpha) for alpha in self.exponents))
            /
            factorial(len(self.exponents)+sum(self.exponents)))

    def diff(self, coordinate):
        diff_exp = list(self.exponents)
        orig_exp = diff_exp[coordinate]
        if orig_exp == 0:
            return Monomial(diff_exp, 0)
        diff_exp[coordinate] = orig_exp-1
        return Monomial(diff_exp, self.factor*orig_exp)





