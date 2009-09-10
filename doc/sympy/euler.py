import sympy as sy
import sympy.matrices as syma


d = 2

rho = sy.Symbol("rho")
rhouvec = [sy.Symbol("rho"+chr(ord("u")+i)) for i in range(d)]
uvec = [rhou_i/rho for rhou_i in rhouvec]
E = sy.Symbol("E")
p = sy.Symbol("p")
gamma = sy.Symbol("gamma")

p_solved = sy.solve(- E + p/(gamma -1) + rho/2*sum(uvec[i]**2 for i in range(d)), p)[0]
print p_solved

flux_rho = syma.Matrix(1, d, [rho*uvec[i] for i in range(d)])
flux_E = syma.Matrix(1, d, [(E+p_solved)*uvec[i] for i in range(d)])
flux_rho_u = syma.Matrix(1, d, [(E+p_solved)*uvec[i] for i in range(d)])

print flux_E

