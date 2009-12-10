from __future__ import division
import math




def find_zero_by_newton(f, fprime, x_start, tolerance = 1e-12, maxit = 10, debug=False):
    it = 0
    xs = []
    while it < maxit:
        it += 1
        f_value = f(x_start)
        if math.fabs(f_value) < tolerance:
            return x_start
        x_start -= f_value / fprime(x_start)
        if len(xs) > 5: xs.pop(0)
        xs.append(x_start)
    if debug:
        last_few = xs[-5:]
        print "Last few tangents of failed Newton:"
        for x in last_few:
            print "%g: %g*x+%g" % (x, fprime(x), f(x)-x*fprime(x))
    raise RuntimeError, "Newton iteration failed, a zero was not found"




class CLawNoShockExactSolution:
    def __init__(self, u0_expr, f_expr):
        import pymbolic as p

        var = p.var

        self.U0 = p.compile(u0_expr)

        fprime_expr = p.differentiate(f_expr, "u")

        phi_expr = var("x") \
                + p.substitute(fprime_expr, {var("u"): u0_expr})*var("t") \
                - var("xtarget")
        self.Phi =  p.compile(phi_expr, ["xtarget", "t", "x"])

        phiprime_expr = p.differentiate(phi_expr, "x")
        self.PhiPrime =  p.compile(phiprime_expr, ["xtarget", "t", "x"])
        
    def __call__(self, x, t):
        if t == 0:
            return self.U0(x)

        def phi(x0): return self.Phi(x, t, x0)
        def phiprime(x0): return self.PhiPrime(x, t, x0)

        x0 = find_zero_by_newton(phi, phiprime, x, tolerance=1e-6, maxit=10000)

        return self.U0(x0)
