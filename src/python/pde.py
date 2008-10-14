"""Canned operators for several PDEs, such as Maxwell's, heat, Poisson, etc."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""





import numpy
import numpy.linalg as la
import pyublas
import hedge.tools
import hedge.mesh
import hedge.data
from pytools import memoize_method




class Operator(object):
    """A base class for Discontinuous Galerkin operators.

    You may derive your own operators from this class, but, at present
    this class provides no functionality. Its function is merely as 
    documentation, to group related classes together in an inheritance
    tree.
    """
    pass




class TimeDependentOperator(Operator):
    """A base class for time-dependent Discontinuous Galerkin operators.

    You may derive your own operators from this class, but, at present
    this class provides no functionality. Its function is merely as 
    documentation, to group related classes together in an inheritance
    tree.
    """
    pass




# operator binding ------------------------------------------------------------
class GradientOperator(Operator):
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def flux(self):
        from hedge.flux import make_normal, FluxScalarPlaceholder
        u = FluxScalarPlaceholder()

        normal = make_normal(self.dimensions)
        return u.int*normal - u.avg*normal

    def op_template(self):
        from hedge.mesh import TAG_ALL
        from hedge.optemplate import Field, pair_with_boundary, \
                make_nabla, InverseMassOperator, get_flux_operator

        u = Field("u")
        bc = Field("bc")

        nabla = make_nabla(self.dimensions)
        flux_op = get_flux_operator(self.flux())

        return nabla*u - InverseMassOperator()*(
                flux_op * u + 
                flux_op * pair_with_boundary(u, bc, TAG_ALL)
                )

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def op(u):
            from hedge.mesh import TAG_ALL

            return compiled_op_template(u=u, 
                    bc=discr.boundarize_volume_field(u, TAG_ALL))

        return op




class DivergenceOperator(Operator):
    def __init__(self, dimensions, subset=None):
        self.dimensions = dimensions

        if subset is None:
            self.subset = dimensions * [True,]
        else:
            # chop off any extra dimensions
            self.subset = subset[:dimensions]

    def flux(self):
        from hedge.flux import make_normal, FluxVectorPlaceholder

        v = FluxVectorPlaceholder(self.dimensions)

        normal = make_normal(self.dimensions)

        flux = 0
        idx = 0

        for i, i_enabled in enumerate(self.subset):
            if i_enabled and i < self.dimensions:
                flux += (v.int-v.avg)[idx]*normal[i]
                idx += 1

        return flux

    def op_template(self):
        from hedge.mesh import TAG_ALL
        from hedge.optemplate import make_vector_field, pair_with_boundary, \
                get_flux_operator, make_nabla, InverseMassOperator
                
        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        v = make_vector_field("v", self.dimensions)
        bc = make_vector_field("bc", self.dimensions)

        local_op_result = 0
        idx = 0
        for i, i_enabled in enumerate(self.subset):
            if i_enabled and i < self.dimensions:
                local_op_result += nabla[i]*v[idx]
                idx += 1

        flux_op = get_flux_operator(self.flux())
        
        return local_op_result - m_inv*(
                flux_op * v + 
                flux_op * pair_with_boundary(v, bc, TAG_ALL))
        
    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def op(v):
            from hedge.mesh import TAG_ALL
            return compiled_op_template(v=v, 
                    bc=discr.boundarize_volume_field(v, TAG_ALL))

        return op





class AdvectionOperatorBase(TimeDependentOperator):
    flux_types = [
            "central",
            "upwind",
            "lf"
            ]

    def __init__(self, v, 
            inflow_tag="inflow",
            inflow_u=hedge.data.make_tdep_constant(0),
            outflow_tag="outflow",
            flux_type="central"
            ):
        self.dimensions = len(v)
        self.v = v
        self.inflow_tag = inflow_tag
        self.inflow_u = inflow_u
        self.outflow_tag = outflow_tag
        self.flux_type = flux_type

    def weak_flux(self):
        from hedge.flux import make_normal, FluxScalarPlaceholder, IfPositive

        u = FluxScalarPlaceholder(0)
        normal = make_normal(self.dimensions)

        if self.flux_type == "central":
            return u.avg*numpy.dot(normal, self.v)
        elif self.flux_type == "lf":
            return u.avg*numpy.dot(normal, self.v) \
                    + 0.5*la.norm(self.v)*(u.int - u.ext)
        elif self.flux_type == "upwind":
            return (numpy.dot(normal, self.v)*
                    IfPositive(numpy.dot(normal, self.v),
                        u.int, # outflow
                        u.ext, # inflow
                        ))
        else:
            raise ValueError, "invalid flux type"

    def max_eigenvalue(self):
        return la.norm(self.v)




class StrongAdvectionOperator(AdvectionOperatorBase):
    def flux(self):
        from hedge.flux import make_normal, FluxScalarPlaceholder

        u = FluxScalarPlaceholder(0)
        normal = make_normal(self.dimensions)

        return u.int * numpy.dot(normal, self.v) - self.weak_flux()

    def op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, \
                get_flux_operator, make_nabla, InverseMassOperator

        u = Field("u")
        bc_in = Field("bc_in")

        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        return -numpy.dot(self.v, nabla*u) + m_inv*(
                flux_op * u
                + flux_op * pair_with_boundary(u, bc_in, self.inflow_tag)
                #+ flux_op * pair_with_boundary(u, bc_out, self.outflow_tag)
                )

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.inflow_tag, self.outflow_tag])

        def rhs(t, u):
            bc_in = self.inflow_u.boundary_interpolant(t, discr, self.inflow_tag)
            #bc_out = 0.5*discr.boundarize_volume_field(u, self.outflow_tag)
            return compiled_op_template(u=u, bc_in=bc_in)

        return rhs




class WeakAdvectionOperator(AdvectionOperatorBase):
    def flux(self):
        return self.weak_flux()

    def op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, \
                get_flux_operator, make_minv_stiffness_t, InverseMassOperator

        u = Field("u")
        bc_in = Field("bc_in")
        bc_out = Field("bc_out")

        minv_st = make_minv_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        return numpy.dot(self.v, minv_st*u) - m_inv*(
                    flux_op*u
                    + flux_op * pair_with_boundary(u, bc_in, self.inflow_tag)
                    + flux_op * pair_with_boundary(u, bc_out, self.outflow_tag)
                    )

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.inflow_tag, self.outflow_tag])

        def rhs(t, u):
            bc_in = self.inflow_u.boundary_interpolant(t, discr, self.inflow_tag)
            bc_out = discr.boundarize_volume_field(u, self.outflow_tag)

            return compiled_op_template(u=u, bc_in=bc_in, bc_out=bc_out)

        return rhs





class StrongWaveOperator:
    """This operator discretizes the Wave equation S{part}tt u = c^2 S{Delta} u.

    To be precise, we discretize the hyperbolic system

      * S{part}t u - c div v = 0
      * S{part}t v - c grad u = 0

    Note that this is not unique--we could also choose a different sign for M{v}.
    """

    def __init__(self, c, dimensions, source_f=None, 
            flux_type="upwind",
            dirichlet_tag=hedge.mesh.TAG_ALL,
            neumann_tag=hedge.mesh.TAG_NONE,
            radiation_tag=hedge.mesh.TAG_NONE):
        assert isinstance(dimensions, int)

        self.c = c
        self.dimensions = dimensions
        self.source_f = source_f

        if self.c > 0:
            self.sign = 1
        else:
            self.sign = -1

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.flux_type = flux_type

    def flux(self):
        from hedge.flux import FluxVectorPlaceholder, make_normal

        dim = self.dimensions
        w = FluxVectorPlaceholder(1+dim)
        u = w[0]
        v = w[1:]
        normal = make_normal(dim)

        from hedge.tools import join_fields
        flux_weak = join_fields(
                numpy.dot(v.avg, normal),
                u.avg * normal)

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            # see doc/notes/hedge-notes.tm
            flux_weak -= self.sign*join_fields(
                    0.5*(u.int-u.ext),
                    0.5*(normal * numpy.dot(normal, v.int-v.ext)))
        else:
            raise ValueError, "invalid flux type '%s'" % self.flux_type

        flux_strong = join_fields(
                numpy.dot(v.int, normal),
                u.int * normal) - flux_weak

        return -self.c*flux_strong

    def op_template(self):
        from hedge.optemplate import \
                make_vector_field, \
                pair_with_boundary, \
                get_flux_operator, \
                make_nabla, \
                InverseMassOperator

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]

        # boundary conditions -------------------------------------------------
        from hedge.flux import make_normal
        normal = make_normal(d)

        from hedge.tools import join_fields

        dir_bc = join_fields(-u, v)
        neu_bc = join_fields(u, -v)
        rad_bc = join_fields(
                0.5*(u - self.sign*numpy.dot(normal, v)),
                0.5*normal*(numpy.dot(normal, v) - self.sign*u)
                )

        # entire operator -----------------------------------------------------
        nabla = make_nabla(d)
        flux_op = get_flux_operator(self.flux())

        from hedge.tools import join_fields
        return (-join_fields(
            -self.c*numpy.dot(nabla, v), 
            -self.c*(nabla*u)
            ) + InverseMassOperator() * (
                flux_op*w 
                + flux_op * pair_with_boundary(w, dir_bc, self.dirichlet_tag)
                + flux_op * pair_with_boundary(w, neu_bc, self.neumann_tag)
                + flux_op * pair_with_boundary(w, rad_bc, self.radiation_tag)
                ))

    
    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w):
            from hedge.tools import join_fields, ptwise_dot

            rhs = compiled_op_template(w=w)

            if self.source_f is not None:
                rhs[0] += self.source_f(t)

            return rhs

        return rhs

    def max_eigenvalue(self):
        return abs(self.c)




class MaxwellOperator(TimeDependentOperator):
    """A 3D Maxwell operator with PEC boundaries.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    _default_dimensions = 3

    def __init__(self, epsilon, mu, 
            flux_type,
            bdry_flux_type=None,
            pec_tag=hedge.mesh.TAG_ALL, 
            absorb_tag=hedge.mesh.TAG_NONE,
            current=None, dimensions=None):
        """
        @arg flux_type: can be in [0,1] for anything between central and upwind, 
          or "lf" for Lax-Friedrichs.
        """
        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        from hedge.tools import SubsettableCrossProduct
        e_cross = self.e_cross = SubsettableCrossProduct(
                op2_subset=e_subset, result_subset=h_subset)
        h_cross = self.h_cross = SubsettableCrossProduct(
                op2_subset=h_subset, result_subset=e_subset)

        from math import sqrt

        self.epsilon = epsilon
        self.mu = mu
        self.c = 1/sqrt(mu*epsilon)

        self.Z = sqrt(mu/epsilon)
        self.Y = 1/self.Z

        self.flux_type = flux_type
        if bdry_flux_type is None:
            self.bdry_flux_type = flux_type
        else:
            self.bdry_flux_type = bdry_flux_type

        self.pec_tag = pec_tag
        self.absorb_tag = absorb_tag

        self.current = current

        self.dimensions = dimensions or self._default_dimensions

    def flux(self, flux_type):
        from math import sqrt
        from hedge.flux import make_normal, FluxVectorPlaceholder
        from hedge.tools import join_fields

        normal = make_normal(self.dimensions)

        w = FluxVectorPlaceholder(self.count_subset(self.get_eh_subset()))
        e, h = self.split_eh(w)

        if flux_type == "lf":
            return join_fields(
                    # flux e, 
                    1/2*(
                        -1/self.epsilon*self.h_cross(normal, h.int-h.ext)
                        -self.c/2*(e.int-e.ext)
                    ),
                    # flux h
                    1/2*(
                        1/self.mu*self.e_cross(normal, e.int-e.ext)
                        -self.c/2*(h.int-h.ext))
                    )
        elif isinstance(flux_type, (int, float)):
            # see doc/maxima/maxwell.mac
            return join_fields(
                    # flux e, 
                    1/self.epsilon*(
                        -1/2*self.h_cross(normal, 
                            h.int-h.ext
                            -flux_type/self.Z*self.e_cross(normal, e.int-e.ext))
                        ),
                    # flux h
                    1/self.mu*(
                        1/2*self.e_cross(normal, 
                            e.int-e.ext
                            +flux_type/(self.Y)*self.h_cross(normal, h.int-h.ext))
                        ),
                    )
        else:
            raise ValueError, "maxwell: invalid flux_type (%s)" % self.flux_type

    def local_op(self, e, h):
        # in conservation form: u_t + A u_x = 0
        def e_curl(field):
            return self.e_cross(nabla, field)

        def h_curl(field):
            return self.h_cross(nabla, field)

        from hedge.optemplate import make_nabla
        from hedge.tools import join_fields

        nabla = make_nabla(self.dimensions)

        return join_fields(
                - 1/self.epsilon * h_curl(h),
                1/self.mu * e_curl(e),
                )

    def op_template(self):
        from hedge.optemplate import make_vector_field, pair_with_boundary, \
                InverseMassOperator, get_flux_operator

        fld_cnt = self.count_subset(self.get_eh_subset())
        w = make_vector_field("w", fld_cnt)
        e,h = self.split_eh(w)

        # boundary conditions -------------------------------------------------
        from hedge.tools import join_fields
        pec_bc = join_fields(-e, h)

        from hedge.flux import make_normal
        normal = make_normal(self.dimensions)

        flux_op = get_flux_operator(self.flux(self.flux_type))
        bdry_flux_op = get_flux_operator(self.flux(self.bdry_flux_type))

        absorb_bc = w + 1/2*join_fields(
                self.h_cross(normal, self.e_cross(normal, e)) 
                - self.Z*self.h_cross(normal, h),
                self.e_cross(normal, self.h_cross(normal, h)) 
                + self.Y*self.e_cross(normal, e)
                )

        # actual operator template --------------------------------------------
        m_inv = InverseMassOperator()

        return - self.local_op(e, h) \
                + m_inv*(
                    flux_op * w
                    +bdry_flux_op * pair_with_boundary(w, pec_bc, self.pec_tag)
                    +bdry_flux_op * pair_with_boundary(w, absorb_bc, self.absorb_tag)
                    )

    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.pec_tag, self.absorb_tag])

        compiled_op_template = discr.compile(self.op_template())

        if self.current is None:
            def rhs(t, w):
                return compiled_op_template(w=w)
        else:
            def rhs(t, w):
                j = self.current.volume_interpolant(t, self.discr)
                j_source = []
                for j_idx, use_component in enumerate(self.get_eh_subset()[0:3]):
                    if use_component:
                        j_source.append(-j[j_idx])
                return compiled_op_template(w=w) + self.assemble_fields(e=j_source)

        return rhs

    def assemble_fields(self, e=None, h=None, discr=None):
        e_components = self.count_subset(self.get_eh_subset()[0:3])
        h_components = self.count_subset(self.get_eh_subset()[3:6])

        if discr is None:
            if e is None:
                e = [0 for i in xrange(e_components)]
            if h is None:
                h = [0 for i in xrange(h_components)]
        else:
            if e is None:
                e = [discr.volume_zeros() for i in xrange(e_components)]
            if h is None:
                h = [discr.volume_zeros() for i in xrange(h_components)]

        from hedge.tools import join_fields
        return join_fields(e, h)

    def split_eh(self, w):
        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        idx = 0

        e = []
        for use_component in e_subset:
            if use_component:
                e.append(w[idx])
                idx += 1

        h = []
        for use_component in h_subset:
            if use_component:
                h.append(w[idx])
                idx += 1

        from hedge.flux import FluxVectorPlaceholder
        from hedge.tools import join_fields

        if isinstance(w, FluxVectorPlaceholder):
            return FluxVectorPlaceholder(scalars=e), FluxVectorPlaceholder(scalars=h)
        elif isinstance(w, numpy.ndarray):
            return join_fields(*e), join_fields(*h)
        else:
            return e, h

    @staticmethod
    def count_subset(subset):
        from pytools import len_iterable
        return len_iterable(uc for uc in subset if uc)

    def get_eh_subset(self):
        """Return a 6-tuple of C{bool}s indicating whether field components 
        are to be computed. The fields are numbered in the order specified
        in the class documentation.
        """
        return 6*(True,)

    def max_eigenvalue(self):
        """Return the largest eigenvalue of Maxwell's equations as a hyperbolic system."""
        from math import sqrt
        return 1/sqrt(self.mu*self.epsilon)




class TMMaxwellOperator(MaxwellOperator):
    """A 2D TM Maxwell operator with PEC boundaries.

    Field order is [Ez Hx Hy].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (False,False,True) # only ez
                +
                (True,True,False) # hx and hy
                )




class TEMaxwellOperator(MaxwellOperator):
    """A 2D TE Maxwell operator with PEC boundaries.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (True,True,False) # ex and ey
                +
                (False,False,True) # only hz
                )




class WeakPoissonOperator(Operator, ):
    """Implements the Local Discontinuous Galerkin (LDG) Method for elliptic
    operators.

    See P. Castillo et al., 
    Local discontinuous Galerkin methods for elliptic problems", 
    Communications in Numerical Methods in Engineering 18, no. 1 (2002): 69-75.
    """

    def __init__(self, dimensions, diffusion_tensor=None, 
            dirichlet_bc=hedge.data.ConstantGivenFunction(), dirichlet_tag="dirichlet",
            neumann_bc=hedge.data.ConstantGivenFunction(), neumann_tag="neumann",
            flux="ip"):
        """Initialize the weak Poisson operator.

        @arg flux: Either C{"ip"} or C{"ldg"} to indicate which type of flux is 
        to be used. IP tends to be faster, and is therefore the default.
        """
        self.dimensions = dimensions
        assert isinstance(dimensions, int)

        self.flux_type = flux

        from math import sqrt

        # treat diffusion tensor
        if diffusion_tensor is None:
            diffusion_tensor = hedge.data.ConstantGivenFunction(
                    numpy.eye(dimensions))

        self.diffusion_tensor = diffusion_tensor

        self.dirichlet_bc = dirichlet_bc
        self.dirichlet_tag = dirichlet_tag
        self.neumann_bc = neumann_bc
        self.neumann_tag = neumann_tag

    # fluxes ------------------------------------------------------------------
    def get_weak_flux_set(self, flux):
        class FluxSet: pass
        fs = FluxSet()

        if flux == "ldg":
            ldg_terms = True
        elif flux == "ip":
            ldg_terms = False
        else:
            raise "Invalid flux type '%s'" % flux

        from hedge.flux import \
                FluxVectorPlaceholder, FluxScalarPlaceholder, \
                make_normal, PenaltyTerm
        from numpy import dot

        dim = self.dimensions
        vec = FluxVectorPlaceholder(1+dim)
        fs.u = u = vec[0]
        fs.v = v = vec[1:]
        normal = make_normal(dim)

        # central flux
        fs.flux_u = u.avg*normal
        fs.flux_v = dot(v.avg, normal)

        if ldg_terms:
            # ldg terms
            ldg_beta = numpy.array([1]*dim)

            fs.flux_u = fs.flux_u - (u.int-u.ext)*0.5*ldg_beta
            fs.flux_v = fs.flux_v + dot((v.int-v.ext)*0.5, ldg_beta)

        # penalty term
        stab_term = PenaltyTerm() * (u.int - u.ext)
        fs.flux_v -= stab_term

        # boundary fluxes
        fs.flux_u_dbdry = normal * u.ext
        fs.flux_v_dbdry = dot(v.int, normal) - stab_term

        fs.flux_u_nbdry = normal * u.int
        fs.flux_v_nbdry = dot(normal, v.ext)

        return fs

    # operator application, rhs prep ------------------------------------------
    def grad_op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, get_flux_operator, \
                make_stiffness_t, InverseMassOperator

        stiff_t = make_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        u = Field("u")

        fs = self.get_weak_flux_set(self.flux_type)

        flux_u = get_flux_operator(fs.flux_u)
        flux_u_dbdry = get_flux_operator(fs.flux_u_dbdry)
        flux_u_nbdry = get_flux_operator(fs.flux_u_nbdry)

        return m_inv * (
                - (stiff_t * u)
                + flux_u*u
                + flux_u_dbdry*pair_with_boundary(u, 0, self.dirichlet_tag)
                + flux_u_nbdry*pair_with_boundary(u, 0, self.neumann_tag)
                )

    def div_op_template(self, apply_minv):
        from hedge.optemplate import make_vector_field, pair_with_boundary, \
                make_stiffness_t, InverseMassOperator, get_flux_operator

        d = self.dimensions
        w = make_vector_field("w", 1+d)
        v = w[1:]
        dir_bc_w = make_vector_field("dir_bc_w", 1+d)
        neu_bc_w = make_vector_field("neu_bc_w", 1+d)

        stiff_t = make_stiffness_t(d)
        m_inv = InverseMassOperator()

        fs = self.get_weak_flux_set(self.flux_type)

        flux_v = get_flux_operator(fs.flux_v)
        flux_v_dbdry = get_flux_operator(fs.flux_v_dbdry)
        flux_v_nbdry = get_flux_operator(fs.flux_v_nbdry)

        result = (
                -numpy.dot(stiff_t, v)
                + flux_v * w
                + flux_v_dbdry * pair_with_boundary(w, dir_bc_w, self.dirichlet_tag)
                + flux_v_nbdry * pair_with_boundary(w, neu_bc_w, self.neumann_tag)
                )

        if apply_minv:
            return InverseMassOperator() * result
        else:
            return result

    @memoize_method
    def grad_bc_op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, \
                InverseMassOperator, get_flux_operator

        flux_u_dbdry = get_flux_operator(
                self.get_weak_flux_set(self.flux_type).flux_u_dbdry)

        return InverseMassOperator() * (
                flux_u_dbdry*pair_with_boundary(0, Field("dir_bc_u"), 
                    self.dirichlet_tag))

    # bound operator ----------------------------------------------------------
    class BoundPoissonOperator(hedge.tools.OperatorBase):
        def __init__(self, poisson_op, discr):
            hedge.tools.OperatorBase.__init__(self)
            self.discr = discr

            pop = self.poisson_op = poisson_op

            self.grad_c = discr.compile(pop.grad_op_template())
            self.div_c = discr.compile(pop.div_op_template(False))
            self.minv_div_c = discr.compile(pop.div_op_template(True))
            self.grad_bc_c = discr.compile(pop.grad_bc_op_template())

            self.neumann_normals = discr.boundary_normals(poisson_op.neumann_tag)

            if isinstance(pop.diffusion_tensor, hedge.data.ConstantGivenFunction):
                self.diffusion = self.neu_diff = pop.diffusion_tensor.value
            else:
                self.diffusion = pop.diffusion_tensor.volume_interpolant(discr)
                self.neu_diff = pop.diffusion_tensor.boundary_interpolant(discr, neumann_tag)

        # pyublasext operator compatibility
        def size1(self):
            return len(self.discr)

        def size2(self):
            return len(self.discr)

        def apply(self, before, after):
            after[:] = self.op(before)

        # actual functionality
        def grad(self, u):
            return self.grad_c(u=u)

        def div(self, v, u=None, apply_minv=True):
            """Compute the divergence of v using an LDG operator.

            The divergence computation is unaffected by the scaling
            effected by the diffusion tensor.

            @param apply_minv: Bool specifying whether to compute a complete 
              divergence operator. If False, the final application of the inverse
              mass operator is skipped. This is used in L{op}() in order to reduce
              the scheme M{M^{-1} S u = f} to M{S u = M f}, so that the mass operator
              only needs to be applied once, when preparing the right hand side
              in @L{prepare_rhs}.
            """
            from hedge.tools import join_fields

            dim = self.discr.dimensions

            if u is None:
                u = self.discr.volume_zeros()
            w = join_fields(u, v)

            dir_bc_w = join_fields(0, [0]*dim)
            neu_bc_w = join_fields(0, [0]*dim)

            if apply_minv:
                div_tpl = self.minv_div_c
            else:
                div_tpl = self.div_c

            return div_tpl(w=w, dir_bc_w=dir_bc_w, neu_bc_w=neu_bc_w)

        def op(self, u, apply_minv=False):
            from hedge.tools import ptwise_dot

            return self.div(
                    ptwise_dot(2, 1, self.diffusion, self.grad(u)), 
                    u, apply_minv=apply_minv)

        def prepare_rhs(self, rhs):
            """Perform the rhs(*) function in the class description, i.e.
            return a right hand side for the linear system op(u)=rhs(f).
            
            In matrix form, LDG looks like this:
            
            Mv = Cu + g
            Mf = Av + Bu + h

            where v is the auxiliary vector, u is the argument of the operator, f
            is the result of the operator and g and h are inhom boundary data, and
            A,B,C are some operator+lifting matrices

            M f = A Minv(Cu + g) + Bu + h

            so the linear system looks like

            M f = A Minv Cu + A Minv g + Bu + h
            M f - A Minv g - h = (A Minv C + B)u

            So the right hand side we're putting together here is really

            M f - A Minv g - h
            """

            dim = self.discr.dimensions

            pop = self.poisson_op

            dtag = pop.dirichlet_tag
            ntag = pop.neumann_tag

            dir_bc_u = pop.dirichlet_bc.boundary_interpolant(self.discr, dtag)
            vpart = self.grad_bc_c(dir_bc_u=dir_bc_u)

            from hedge.tools import ptwise_dot
            diff_v = ptwise_dot(2, 1, self.diffusion, vpart)

            def neu_bc_v():
                return ptwise_dot(2, 1, self.neu_diff, 
                        self.neumann_normals*
                            pop.neumann_bc.boundary_interpolant(self.discr, ntag))

            from hedge.tools import join_fields
            w = join_fields(0, diff_v)
            dir_bc_w = join_fields(dir_bc_u, [0]*dim)
            neu_bc_w = join_fields(0, neu_bc_v())

            from hedge.optemplate import MassOperator
            return (MassOperator().apply(self.discr, rhs.volume_interpolant(self.discr))
                - self.div_c(w=w, dir_bc_w=dir_bc_w, neu_bc_w=neu_bc_w))

    def bind(self, discr):
        assert self.dimensions == discr.dimensions

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.dirichlet_tag, self.neumann_tag])

        return self.BoundPoissonOperator(self, discr)

    # matrix creation ---------------------------------------------------------
    def grad_matrix(self):
        # broken
        discr = self.discr
        dim = discr.dimensions

        def assemble_local_vstack(operators):
            n = len(operators)
            dof = len(discr)
            result = pyublas.zeros((n*dof, dof), flavor=pyublas.SparseBuildMatrix)

            from hedge._internal import MatrixTarget
            tgt = MatrixTarget(result, 0, 0)

            for i, op in enumerate(operators):
                op.perform_on(tgt.rebased_target(i*dof, 0))
            return result

        def assemble_local_hstack(operators):
            n = len(operators)
            dof = len(discr)
            result = pyublas.zeros((dof, n*dof), flavor=pyublas.SparseBuildMatrix)

            from hedge._internal import MatrixTarget
            tgt = MatrixTarget(result, 0, 0)

            for i, op in enumerate(operators):
                op.perform_on(tgt.rebased_target(0, i*dof))
            return result

        def assemble_local_diag(operators):
            n = len(operators)
            dof = len(discr)
            result = pyublas.zeros((n*dof, n*dof), flavor=pyublas.SparseBuildMatrix)

            from hedge._internal import MatrixTarget
            tgt = MatrixTarget(result, 0, 0)

            for i, op in enumerate(operators):
                op.perform_on(tgt.rebased_target(i*dof, i*dof))
            return result

        def fast_mat(mat):
            return pyublas.asarray(mat, flavor=pyublas.SparseExecuteMatrix)

        def assemble_grad():
            n = self.discr.dimensions
            dof = len(discr)

            minv = fast_mat(assemble_local_diag([self.m_inv] * dim))

            m_local_grad = fast_mat(-assemble_local_vstack(self.discr.minv_stiffness_t))

            fluxes = pyublas.zeros((n*dof, dof), flavor=pyublas.SparseBuildMatrix)
            from hedge._internal import MatrixTarget
            fluxes_tgt = MatrixTarget(fluxes, 0, 0)
            self.flux_u.perform_inner(fluxes_tgt)
            self.flux_u_dbdry.perform_int_bdry(self.dirichlet_tag, fluxes_tgt)
            self.flux_u_nbdry.perform_int_bdry(self.neumann_tag, fluxes_tgt)

            return m_local_grad + minv * fast_mat(fluxes)

        return assemble_grad()




class StrongHeatOperator(TimeDependentOperator):
    def __init__(self, dimensions, coeff=hedge.data.ConstantGivenFunction(1), 
            dirichlet_bc=hedge.data.ConstantGivenFunction(), dirichlet_tag="dirichlet",
            neumann_bc=hedge.data.ConstantGivenFunction(), neumann_tag="neumann",
            ldg=True):
        self.dimensions = dimensions
        assert isinstance(dimensions, int)

        self.coeff = coeff
        self.ldg = ldg

        self.dirichlet_bc = dirichlet_bc
        self.dirichlet_tag = dirichlet_tag
        self.neumann_bc = neumann_bc
        self.neumann_tag = neumann_tag

    # fluxes ------------------------------------------------------------------
    def get_weak_flux_set(self, ldg):
        class FluxSet: pass
        fs = FluxSet()

        from hedge.flux import FluxVectorPlaceholder, FluxScalarPlaceholder, make_normal

        # note here:

        # local DG is unlike the other kids in that the computation of the flux
        # of u depends *only* on u, whereas the computation of the flux of v
        # (yielding the final right hand side) may also depend on u. That's why
        # we use the layout [u,v], where v is simply omitted for the u flux
        # computation.

        dim = self.dimensions
        vec = FluxVectorPlaceholder(1+dim)
        fs.u = u = vec[0]
        fs.v = v = vec[1:]
        normal = fs.normal = make_normal(dim)

        # central
        fs.flux_u = u.avg*normal
        fs.flux_v = numpy.dot(v.avg, normal)

        # dbdry is "dirichlet boundary"
        # nbdry is "neumann boundary"
        fs.flux_u_dbdry = fs.flux_u
        fs.flux_u_nbdry = fs.flux_u

        fs.flux_v_dbdry = fs.flux_v
        fs.flux_v_nbdry = fs.flux_v

        if ldg:
            ldg_beta = numpy.ones((dim,))

            fs.flux_u = fs.flux_u - (u.int-u.ext)*0.5*ldg_beta
            fs.flux_v = fs.flux_v + numpy.dot((v.int-v.ext)*0.5, ldg_beta)

        return fs

    def get_strong_flux_set(self, ldg):
        fs = self.get_weak_flux_set(ldg)

        u = fs.u
        v = fs.v
        normal = fs.normal

        fs.flux_u = u.int*normal - fs.flux_u
        fs.flux_v = numpy.dot(v.int, normal) - fs.flux_v
        fs.flux_u_dbdry = u.int*normal - fs.flux_u_dbdry
        fs.flux_v_dbdry = numpy.dot(v.int, normal) - fs.flux_v_dbdry
        fs.flux_u_nbdry = u.int*normal - fs.flux_u_nbdry
        fs.flux_v_nbdry = numpy.dot(v.int, normal) - fs.flux_v_nbdry

        return fs

    # right-hand side ---------------------------------------------------------
    def grad_op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, \
                InverseMassOperator, make_stiffness, get_flux_operator

        stiff = make_stiffness(self.dimensions)
        
        u = Field("u")
        sqrt_coeff_u = Field("sqrt_coeff_u")
        dir_bc_u = Field("dir_bc_u")
        neu_bc_u = Field("neu_bc_u")

        fs = self.get_strong_flux_set(self.ldg)
        flux_u = get_flux_operator(fs.flux_u)
        flux_u_dbdry = get_flux_operator(fs.flux_u_dbdry)
        flux_u_nbdry = get_flux_operator(fs.flux_u_nbdry)

        return InverseMassOperator() * (
                stiff * u
                - flux_u*sqrt_coeff_u
                - flux_u_dbdry*pair_with_boundary(sqrt_coeff_u, dir_bc_u, self.dirichlet_tag)
                - flux_u_nbdry*pair_with_boundary(sqrt_coeff_u, neu_bc_u, self.neumann_tag)
                )

    def div_op_template(self):
        from hedge.optemplate import make_vector_field, pair_with_boundary, \
                InverseMassOperator, get_flux_operator, make_stiffness

        stiff = make_stiffness(self.dimensions)
        
        d = self.dimensions
        w = make_vector_field("w", 1+d)
        v = w[1:]

        dir_bc_w = make_vector_field("dir_bc_w", 1+d)
        neu_bc_w = make_vector_field("neu_bc_w", 1+d)

        fs = self.get_strong_flux_set(self.ldg)
        flux_v = get_flux_operator(fs.flux_v)
        flux_v_dbdry = get_flux_operator(fs.flux_v_dbdry)
        flux_v_nbdry = get_flux_operator(fs.flux_v_nbdry)

        return InverseMassOperator() * (
                numpy.dot(stiff, v)
                - flux_v * w
                - flux_v_dbdry * pair_with_boundary(w, dir_bc_w, self.dirichlet_tag)
                - flux_v_nbdry * pair_with_boundary(w, neu_bc_w, self.neumann_tag)
                )

    # boundary conditions -----------------------------------------------------
    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.dirichlet_tag, self.neumann_tag])

        return self.BoundHeatOperator(self, discr)

    class BoundHeatOperator:
        def __init__(self, heat_op, discr):
            hop = self.heat_op = heat_op
            self.discr = discr

            self.sqrt_coeff = numpy.sqrt(
                    hop.coeff.volume_interpolant(discr))
            self.dir_sqrt_coeff = numpy.sqrt(
                    hop.coeff.boundary_interpolant(discr, hop.dirichlet_tag))
            self.neu_sqrt_coeff = numpy.sqrt(
                    hop.coeff.boundary_interpolant(discr, hop.neumann_tag))

            self.neumann_normals = discr.boundary_normals(hop.neumann_tag)

            self.grad_c = discr.compile(hop.grad_op_template())
            self.div_c = discr.compile(hop.div_op_template())

        def dirichlet_bc_u(self, t, sqrt_coeff_u):
            hop = self.heat_op

            return (
                    -self.discr.boundarize_volume_field(sqrt_coeff_u, hop.dirichlet_tag)
                    +2*self.dir_sqrt_coeff*hop.dirichlet_bc.boundary_interpolant(
                        t, self.discr, hop.dirichlet_tag)
                    )

        def dirichlet_bc_v(self, t, sqrt_coeff_v):
            hop = self.heat_op

            return self.discr.boundarize_volume_field(sqrt_coeff_v, hop.dirichlet_tag)

        def neumann_bc_u(self, t, sqrt_coeff_u):
            hop = self.heat_op

            return self.discr.boundarize_volume_field(sqrt_coeff_u, hop.neumann_tag)

        def neumann_bc_v(self, t, sqrt_coeff_v):
            hop = self.heat_op

            return (
                    -self.discr.boundarize_volume_field(sqrt_coeff_v, hop.neumann_tag)
                    +
                    2*self.neumann_normals*
                    hop.neumann_bc.boundary_interpolant(t, self.discr, hop.neumann_tag)
                    )

        def __call__(self, t, u):
            from math import sqrt
            from hedge.tools import join_fields

            hop = self.heat_op

            dtag = hop.dirichlet_tag
            ntag = hop.neumann_tag

            sqrt_coeff_u = self.sqrt_coeff * u

            dir_bc_u = self.dirichlet_bc_u(t, sqrt_coeff_u)
            neu_bc_u = self.neumann_bc_u(t, sqrt_coeff_u)

            v = self.grad_c(
                    u=u, sqrt_coeff_u=sqrt_coeff_u,
                    dir_bc_u=dir_bc_u, neu_bc_u=neu_bc_u)

            from hedge.tools import ptwise_mul
            sqrt_coeff_v = ptwise_mul(self.sqrt_coeff, v)

            dir_bc_v = self.dirichlet_bc_v(t, sqrt_coeff_v)
            neu_bc_v = self.neumann_bc_v(t, sqrt_coeff_v)

            w = join_fields(sqrt_coeff_u, sqrt_coeff_v)
            dir_bc_w = join_fields(dir_bc_u, dir_bc_v)
            neu_bc_w = join_fields(neu_bc_u, neu_bc_v)

            return self.div_c(w=w, dir_bc_w=dir_bc_w, neu_bc_w=neu_bc_w)
