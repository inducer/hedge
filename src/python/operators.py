from __future__ import division




def _double_cross(tbl, field):
    from pytools.arithmetic_container import ArithmeticList
    return ArithmeticList([
          tbl[0][1]*field[1] + tbl[0][2]*field[2]
        - tbl[1][1]*field[0] - tbl[2][2]*field[0],

          tbl[1][0]*field[0] + tbl[1][2]*field[2] 
        - tbl[0][0]*field[1] - tbl[2][2]*field[1],
          
          tbl[2][1]*field[1] + tbl[2][0]*field[0] 
        - tbl[0][0]*field[2] - tbl[1][1]*field[2],
        ])




class MaxwellOperator:
    """A 3D Maxwell operator with PEC boundaries everywhere.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    def __init__(self, discr, epsilon, mu, upwind_alpha=1):
        from hedge.flux import make_normal, local, neighbor
        from hedge.discretization import \
            bind_flux, \
            bind_nabla, \
            bind_inverse_mass_matrix, \
            pair_with_boundary

        self.discr = discr

        self.epsilon = epsilon
        self.mu = mu
        self.alpha = upwind_alpha

        normal = make_normal(discr.dimensions)

        self.n_jump = bind_flux(discr, 1/2*normal*(local-neighbor))
        self.n_n_jump_tbl = [[bind_flux(discr, 1/2*normal[i]*normal[j]*(local-neighbor))
                for i in range(discr.dimensions)]
                for j in range(discr.dimensions)]

        self.nabla = bind_nabla(discr)
        self.m_inv = bind_inverse_mass_matrix(discr)


    def rhs(self, t, y):
        from hedge.tools import cross
        from hedge.discretization import pair_with_boundary
        from pytools.arithmetic_container import ArithmeticList
        from math import sqrt

        e = y[0:3]
        h = y[3:6]

        def curl(field):
            return cross(self.nabla, field)

        bc_e = -self.discr.boundarize_volume_field(e)
        bc_h = self.discr.boundarize_volume_field(h)

        h_pair = pair_with_boundary(h, bc_h)
        e_pair = pair_with_boundary(e, bc_e)

        rhs = ArithmeticList([])

        Z = sqrt(self.mu/self.epsilon)
        Y = 1/Z

        # rhs e
        rhs.extend(
                1/self.epsilon*(
                    curl(h)
                    - (self.m_inv*(
                        cross(self.n_jump, h)
                        + cross(self.n_jump, h_pair)
                        - 1/Z*self.alpha*_double_cross(self.n_n_jump_tbl, e)
                        - 1/Z*self.alpha*_double_cross(self.n_n_jump_tbl, e_pair)
                        ))))
        # rhs h
        rhs.extend(
                1/self.mu*(
                    - curl(e)
                    + (self.m_inv*(
                        cross(self.n_jump, e)
                        + cross(self.n_jump, e_pair)
                        + 1/Y*self.alpha*_double_cross(self.n_n_jump_tbl, h)
                        + 1/Y*self.alpha*_double_cross(self.n_n_jump_tbl, h_pair)
                        ))))

        return rhs

