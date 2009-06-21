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
from pytools import memoize_method, Record




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

        from hedge.tools import count_subset
        self.arg_count = count_subset(self.subset)

    def flux(self):
        from hedge.flux import make_normal, FluxVectorPlaceholder

        v = FluxVectorPlaceholder(self.arg_count)

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

        v = make_vector_field("v", self.arg_count)
        bc = make_vector_field("bc", self.arg_count)

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

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.inflow_tag, self.outflow_tag])

        def rhs(t, u):
            bc_in = self.inflow_u.boundary_interpolant(t, discr, self.inflow_tag)
            bc_out = discr.boundarize_volume_field(u, self.outflow_tag)

            return compiled_op_template(u=u, bc_in=bc_in, bc_out=bc_out)

        return rhs

    def bind_interdomain(self, 
            my_discr, my_part_data,
            nb_discr, nb_part_data):
        from hedge.partition import compile_interdomain_flux
        compiled_op_template, from_nb_indices = compile_interdomain_flux(
                self.op_template(), "u", "nb_bdry_u",
                my_discr, my_part_data, nb_discr, nb_part_data,
                use_stupid_substitution=True)

        from hedge.tools import with_object_array_or_scalar, is_zero

        def nb_bdry_permute(fld):
            if is_zero(fld):
                return 0
            else:
                return fld[from_nb_indices]

        def rhs(t, u, u_neighbor):
            return compiled_op_template(u=u, 
                    nb_bdry_u=with_object_array_or_scalar(nb_bdry_permute, u_neighbor))

        return rhs




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

        return (
                -numpy.dot(self.v, nabla*u) 
                + m_inv*(
                flux_op * u
                + flux_op * pair_with_boundary(u, bc_in, self.inflow_tag)
                #+ flux_op * pair_with_boundary(u, bc_out, self.outflow_tag)
                )
                )




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


class VariableCoefficientAdvectionOperator:
    """A class for space- and time-dependent DG-advection operators.

    `advec_v` is a callable expecting two arguments `(x, t)` representing space and time, 
    and returning an n-dimensional vector representing the velocity at x. 
    `bc_u_f` is a callable expecting `(x, t)` representing space and time, 
    and returning an 1-dimensional vector representing the state on the boundary.
    Both `advec_v` and `bc_u_f` conform to the 
    `hedge.data.ITimeDependentGivenFunction` interface.
    """

    flux_types = [
            "central",
            "upwind",
            "lf"
            ]

    def __init__(self, 
            dimensions, 
	    advec_v,
            bc_u_f="None",
	    flux_type="central"
	    ):
        self.dimensions = dimensions
        self.advec_v = advec_v
        self.bc_u_f = bc_u_f
        self.flux_type = flux_type

    def flux(self, ):
        from hedge.flux import \
	                make_normal, \
			FluxScalarPlaceholder, \
			FluxVectorPlaceholder, \
			IfPositive, flux_max, norm
        
        d = self.dimensions

        w = FluxVectorPlaceholder((1+d)+1)
	u = w[0]
        v = w[1:d+1]
        c = w[1+d]

        normal = make_normal(self.dimensions)

        if self.flux_type == "central":
	    return (u.int*numpy.dot(v.int, normal )
	            + u.ext*numpy.dot(v.ext, normal)) * 0.5
        elif self.flux_type == "lf":
            n_vint = numpy.dot(normal, v.int)
            n_vext = numpy.dot(normal, v.ext)
            return 0.5 * (n_vint * u.int + n_vext * u.ext) \
                   - 0.5 * (u.ext - u.int) \
                   * flux_max(c.int, c.ext)

        elif self.flux_type == "upwind": 
            return (
                    IfPositive(numpy.dot(normal, v.avg),
                        numpy.dot(normal, v.int) * u.int, # outflow
                        numpy.dot(normal, v.ext) * u.ext, # inflow
                        ))
            return f
        else:
            raise ValueError, "invalid flux type"

    def op_template(self):
        from hedge.optemplate import \
	        Field, \
		pair_with_boundary, \
                get_flux_operator, \
		make_minv_stiffness_t, \
		InverseMassOperator,\
		make_vector_field

        from hedge.tools import join_fields, \
                                ptwise_dot
        
        from hedge.optemplate import ElementwiseMaxOperator, BoundarizeOperator


        u = Field("u")
	v = make_vector_field("v", self.dimensions)
        c = ElementwiseMaxOperator()*ptwise_dot(1, 1, v, v)
        w = join_fields(u, v, c)

        # boundary conditions -------------------------------------------------
        from hedge.mesh import TAG_ALL
        bc_c = BoundarizeOperator(TAG_ALL) * c
        bc_u = Field("bc_u")
        bc_v = BoundarizeOperator(TAG_ALL) * v
        if self.bc_u_f is "None":
            bc_w = join_fields(0, bc_v, bc_c)
        else:
            bc_w = join_fields(bc_u, bc_v, bc_c)

        minv_st = make_minv_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        result = numpy.dot(minv_st, v*u) - m_inv*(
                    flux_op * w
                    + flux_op * pair_with_boundary(w, bc_w, TAG_ALL)
                    )
        return result

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())
        
        from hedge.mesh import check_bc_coverage, TAG_ALL
        check_bc_coverage(discr.mesh, [TAG_ALL])

        def rhs(t, u):
	    v = self.advec_v.volume_interpolant(t, discr)
            
            if self.bc_u_f is not "None":
                bc_u = self.bc_u_f.boundary_interpolant(t, discr, tag=TAG_ALL)
                return compiled_op_template(u=u, v=v, bc_u=bc_u)
            else:
                return compiled_op_template(u=u, v=v)

        return rhs

    def max_eigenvalue(self, t, discr):
        # Gives the max eigenvalue of a vector of eigenvalues.
        # As the velocities of each node is stored in the velocity-vector-field 
        # a pointwise dot product of this vector has to be taken to get the 
        # magnitude of the velocity at each node. From this vector the maximum 
        # values limits the timestep.
        
        from hedge.tools import ptwise_dot
        v = self.advec_v.volume_interpolant(t, discr)
        return (ptwise_dot(1, 1, v, v)**0.5).max()




class StrongWaveOperator:
    """This operator discretizes the Wave equation S{part}tt u = c^2 S{Delta} u.

    To be precise, we discretize the hyperbolic system

      * S{part}t u - c div v = 0
      * S{part}t v - c grad u = 0

    The sign of M{v} determines whether we discretize the forward or the
    backward wave equation.
    
    c is assumed to be constant across all space.
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
                InverseMassOperator, \
                BoundarizeOperator

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]

        # boundary conditions -------------------------------------------------

        from hedge.tools import join_fields

        dir_bc = join_fields(-u, v)
        neu_bc = join_fields(u, -v)

        from hedge.optemplate import make_normal
        rad_normal = make_normal(self.radiation_tag, d)

        rad_u = BoundarizeOperator(self.radiation_tag) * u
        rad_v = BoundarizeOperator(self.radiation_tag) * v

        rad_bc = join_fields(
                0.5*(rad_u - self.sign*numpy.dot(rad_normal, rad_v)),
                0.5*rad_normal*(numpy.dot(rad_normal, rad_v) - self.sign*rad_u)
                )

        # entire operator -----------------------------------------------------
        nabla = make_nabla(d)
        flux_op = get_flux_operator(self.flux())

        from hedge.tools import join_fields
        return (
                - join_fields(
                    -self.c*numpy.dot(nabla, v), 
                    -self.c*(nabla*u)
                    ) 
                + 
                InverseMassOperator() * (
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
            rhs = compiled_op_template(w=w)

            if self.source_f is not None:
                rhs[0] += self.source_f(t)

            return rhs

        return rhs

    def max_eigenvalue(self):
        return abs(self.c)




class VariableVelocityStrongWaveOperator:
    """This operator discretizes the Wave equation S{part}tt u = c^2 S{Delta} u.

    To be precise, we discretize the hyperbolic system

      * S{part}t u - c div v = 0
      * S{part}t v - c grad u = 0
    """

    def __init__(self, c, dimensions, source=None, 
            flux_type="upwind",
            dirichlet_tag=hedge.mesh.TAG_ALL,
            neumann_tag=hedge.mesh.TAG_NONE,
            radiation_tag=hedge.mesh.TAG_NONE,
            time_sign=1):
        """`c` is assumed to be positive and conforms to the
        `hedge.data.ITimeDependentGivenFunction` interface.

        `source` also conforms to the 
        `hedge.data.ITimeDependentGivenFunction` interface.
        """
        assert isinstance(dimensions, int)

        self.c = c
        self.time_sign = time_sign
        self.dimensions = dimensions
        self.source = source

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.flux_type = flux_type

    def flux(self):
        from hedge.flux import FluxVectorPlaceholder, make_normal

        dim = self.dimensions
        w = FluxVectorPlaceholder(2+dim)
        c = w[0]
        u = w[1]
        v = w[2:]
        normal = make_normal(dim)

        from hedge.tools import join_fields
        flux = self.time_sign*1/2*join_fields(
                c.ext * numpy.dot(v.ext, normal)
                - c.int * numpy.dot(v.int, normal),
                normal*(c.ext*u.ext - c.int*u.int))

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            flux += join_fields(
                    c.ext*u.ext - c.int*u.int,
                    c.ext*normal*numpy.dot(normal, v.ext)
                    - c.int*normal*numpy.dot(normal, v.int)
                    )
        else:
            raise ValueError, "invalid flux type '%s'" % self.flux_type

        return flux

    def op_template(self):
        from hedge.optemplate import \
                Field, \
                make_vector_field, \
                pair_with_boundary, \
                get_flux_operator, \
                make_nabla, \
                InverseMassOperator

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]
        
        from hedge.tools import join_fields
        c = Field("c")
        flux_w = join_fields(c, w)

        # boundary conditions -------------------------------------------------
        from hedge.flux import make_normal
        normal = make_normal(d)

        from hedge.tools import join_fields

        dir_bc = join_fields(c, -u, v)
        neu_bc = join_fields(c, u, -v)
        rad_bc = join_fields(
                c,
                0.5*(u - self.time_sign*numpy.dot(normal, v)),
                0.5*normal*(numpy.dot(normal, v) - self.time_sign*u)
                )

        # entire operator -----------------------------------------------------
        nabla = make_nabla(d)
        flux_op = get_flux_operator(self.flux())

        return (
                - join_fields(
                    -numpy.dot(nabla, self.time_sign*c*v), 
                    -(nabla*(self.time_sign*c*u))
                    ) 
                + 
                InverseMassOperator() * (
                    flux_op*flux_w 
                    + flux_op * pair_with_boundary(flux_w, dir_bc, self.dirichlet_tag)
                    + flux_op * pair_with_boundary(flux_w, neu_bc, self.neumann_tag)
                    + flux_op * pair_with_boundary(flux_w, rad_bc, self.radiation_tag)
                    ))

    
    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w):
            rhs = compiled_op_template(w=w,
                    c=self.c.volume_interpolant(t, discr))

            if self.source is not None:
                rhs[0] += self.source.volume_interpolant(t, discr)

            return rhs

        return rhs

    #def max_eigenvalue(self):
        #return abs(self.c)




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
            incident_tag=hedge.mesh.TAG_NONE,
            incident_bc=None, current=None, dimensions=None):
        """
        @arg flux_type: can be in [0,1] for anything between central and upwind, 
          or "lf" for Lax-Friedrichs.
        """
        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        from hedge.tools import SubsettableCrossProduct
        self.e_cross = SubsettableCrossProduct(
                op2_subset=e_subset, result_subset=h_subset)
        self.h_cross = SubsettableCrossProduct(
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
        self.incident_tag = incident_tag

        self.current = current
        self.incident_bc = incident_bc

        self.dimensions = dimensions or self._default_dimensions

    def flux(self, flux_type):
        from hedge.flux import make_normal, FluxVectorPlaceholder
        from hedge.tools import join_fields

        normal = make_normal(self.dimensions)

        from hedge.tools import count_subset
        w = FluxVectorPlaceholder(count_subset(self.get_eh_subset()))
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
        def e_curl(field):
            return self.e_cross(nabla, field)

        def h_curl(field):
            return self.h_cross(nabla, field)

        from hedge.optemplate import make_nabla
        from hedge.tools import join_fields, count_subset

        nabla = make_nabla(self.dimensions)

        if self.current is not None:
            from hedge.optemplate import make_vector_field
            j = make_vector_field("j", 
                    count_subset(self.get_eh_subset()[:3]))
        else:
            j = 0

        # in conservation form: u_t + A u_x = 0
        return join_fields(
                1/self.epsilon * (j - h_curl(h)),
                1/self.mu * e_curl(e),
                )

    def op_template(self, w=None):
        from hedge.optemplate import pair_with_boundary, \
                InverseMassOperator, get_flux_operator, \
                BoundarizeOperator

        from hedge.optemplate import make_vector_field

        from hedge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())
        if w is None:
            w = make_vector_field("w", fld_cnt)

        e, h = self.split_eh(w)

        # boundary conditions -------------------------------------------------
        from hedge.tools import join_fields
        pec_e = BoundarizeOperator(self.pec_tag) * e
        pec_h = BoundarizeOperator(self.pec_tag) * h
        pec_bc = join_fields(-pec_e, pec_h)

        from hedge.flux import make_normal
        normal = make_normal(self.dimensions)

        absorb_bc = w + 1/2*join_fields(
                self.h_cross(normal, self.e_cross(normal, e)) 
                - self.Z*self.h_cross(normal, h),
                self.e_cross(normal, self.h_cross(normal, h)) 
                + self.Y*self.e_cross(normal, e)
                )

        if self.incident_bc is not None:
            from hedge.optemplate import make_common_subexpression
            incident_bc = make_common_subexpression(
                        -make_vector_field("incident_bc", fld_cnt))

        else:
            from hedge.tools import make_obj_array
            incident_bc = make_obj_array([0]*fld_cnt)

        # actual operator template --------------------------------------------
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux(self.flux_type))
        bdry_flux_op = get_flux_operator(self.flux(self.bdry_flux_type))

        return - self.local_op(e, h) \
                + m_inv*(
                    flux_op * w
                    +bdry_flux_op * pair_with_boundary(w, pec_bc, self.pec_tag)
                    +bdry_flux_op * pair_with_boundary(w, absorb_bc, self.absorb_tag)
                    +bdry_flux_op * pair_with_boundary(w, incident_bc, self.incident_tag))

    def bind(self, discr, **extra_context):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.pec_tag, self.absorb_tag, self.incident_tag])

        compiled_op_template = discr.compile(self.op_template())

        from hedge.tools import full_to_subset_indices
        e_indices = full_to_subset_indices(self.get_eh_subset()[0:3])
        all_indices = full_to_subset_indices(self.get_eh_subset())

        def rhs(t, w):
            if self.current is not None:
                j = self.current.volume_interpolant(t, discr)[e_indices]
            else:
                j = 0

            if self.incident_bc is not None:
                incident_bc = self.incident_bc.boundary_interpolant(
                        t, discr, self.incident_tag)[all_indices]
            else:
                incident_bc = 0

            return compiled_op_template(
                    w=w, j=j, incident_bc=incident_bc, **extra_context)

        return rhs

    def assemble_eh(self, e=None, h=None, discr=None):
        if discr is None:
            def zero(): return 0
        else:
            def zero(): return discr.volume_zeros()

        from hedge.tools import count_subset
        e_components = count_subset(self.get_eh_subset()[0:3])
        h_components = count_subset(self.get_eh_subset()[3:6])

        def default_fld(fld, comp):
            if fld is None:
                return [zero() for i in xrange(comp)]
            else:
                return fld

        e = default_fld(e, e_components)
        h = default_fld(h, h_components)

        from hedge.tools import join_fields
        return join_fields(e, h)

    @memoize_method
    def partial_to_eh_subsets(self):
        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        from hedge.tools import partial_to_all_subset_indices
        return tuple(partial_to_all_subset_indices(
            [e_subset, h_subset]))

    def split_eh(self, w):
        e_idx, h_idx = self.partial_to_eh_subsets()
        e, h = w[e_idx], w[h_idx]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=e), FVP(scalars=h)
        else:
            from hedge.tools import make_obj_array as moa
            return moa(e), moa(h)

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




class AbarbanelGottliebPMLMaxwellOperator(MaxwellOperator):
    """Implements a PML as in 

    [1] S. Abarbanel and D. Gottlieb, "On the construction and analysis of absorbing
    layers in CEM," Applied Numerical Mathematics,  vol. 27, 1998, S. 331-340.
    (eq 3.7-3.11)

    [2] E. Turkel and A. Yefet, "Absorbing PML
    boundary layers for wave-like equations,"
    Applied Numerical Mathematics,  vol. 27,
    1998, S. 533-557.
    (eq. 4.10) 

    [3] Abarbanel, D. Gottlieb, and J.S. Hesthaven, "Long Time Behavior of the
    Perfectly Matched Layer Equations in Computational Electromagnetics,"
    Journal of Scientific Computing,  vol. 17, Dez. 2002, S. 405-422.

    Generalized to 3D in doc/maxima/abarbanel-pml.mac.
    """

    class PMLCoefficients(Record):
        __slots__ = ["sigma", "sigma_prime", "tau"] 
        # (tau=mu in [3] , to avoid confusion with permeability)

        def map(self, f):
            return self.__class__(
                    **dict((name, f(getattr(self, name)))
                        for name in self.fields))

    def __init__(self, *args, **kwargs):
        self.add_decay = kwargs.pop("add_decay", True)
        MaxwellOperator.__init__(self, *args, **kwargs)

    def pml_local_op(self, w):
        sub_e, sub_h, sub_p, sub_q = self.split_ehpq(w)

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]
        dim_subset = (True,) * self.dimensions + (False,) * (3-self.dimensions)

        def pad_vec(v, subset):
            result = numpy.zeros((3,), dtype=object)
            result[numpy.array(subset, dtype=bool)] = v
            return result

        from hedge.optemplate import make_vector_field
        sig = pad_vec(
                make_vector_field("sigma", self.dimensions), 
                dim_subset)
        sig_prime = pad_vec(
                make_vector_field("sigma_prime", self.dimensions), 
                dim_subset)
        if self.add_decay:
            tau = pad_vec(
                    make_vector_field("tau", self.dimensions), 
                    dim_subset)
        else:
            tau = numpy.zeros((3,))

        e = pad_vec(sub_e, e_subset)
        h = pad_vec(sub_h, h_subset)
        p = pad_vec(sub_p, dim_subset)
        q = pad_vec(sub_q, dim_subset)

        rhs = numpy.zeros(12, dtype=object)

        for mx in range(3):
            my = (mx+1) % 3
            mz = (mx+2) % 3

            from hedge.tools import levi_civita
            assert levi_civita((mx,my,mz)) == 1

            rhs[mx] += -sig[my]/self.epsilon*(2*e[mx]+p[mx]) - 2*tau[my]/self.epsilon*e[mx]
            rhs[my] += -sig[mx]/self.epsilon*(2*e[my]+p[my]) - 2*tau[mx]/self.epsilon*e[my]
            rhs[3+mz] += 1/(self.epsilon*self.mu) * (
              sig_prime[mx] * q[mx] - sig_prime[my] * q[my])

            rhs[6+mx] += sig[my]/self.epsilon*e[mx]
            rhs[6+my] += sig[mx]/self.epsilon*e[my]
            rhs[9+mx] += -sig[mx]/self.epsilon*q[mx] - (e[my] + e[mz])

        from hedge.tools import full_to_subset_indices
        sub_idx = full_to_subset_indices(e_subset+h_subset+dim_subset+dim_subset)

        return rhs[sub_idx]

    def op_template(self, w=None):
        from hedge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())
        if w is None:
            from hedge.optemplate import make_vector_field
            w = make_vector_field("w", fld_cnt+2*self.dimensions)

        from hedge.tools import join_fields
        return join_fields(
                MaxwellOperator.op_template(self, w[:fld_cnt]),
                numpy.zeros((2*self.dimensions,), dtype=object)
                ) + self.pml_local_op(w)

    def bind(self, discr, coefficients):
        return MaxwellOperator.bind(self, discr, 
                sigma=coefficients.sigma, 
                sigma_prime=coefficients.sigma_prime,
                tau=coefficients.tau)

    def assemble_ehpq(self, e=None, h=None, p=None, q=None, discr=None):
        if discr is None:
            def zero(): return 0
        else:
            def zero(): return discr.volume_zeros()

        from hedge.tools import count_subset
        e_components = count_subset(self.get_eh_subset()[0:3])
        h_components = count_subset(self.get_eh_subset()[3:6])

        def default_fld(fld, comp):
            if fld is None:
                return [zero() for i in xrange(comp)]
            else:
                return fld

        e = default_fld(e, e_components)
        h = default_fld(h, h_components)
        p = default_fld(p, self.dimensions)
        q = default_fld(q, self.dimensions)

        from hedge.tools import join_fields
        return join_fields(e, h, p, q)

    @memoize_method
    def partial_to_ehpq_subsets(self):
        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        dim_subset = [True] * self.dimensions + [False] * (3-self.dimensions)

        from hedge.tools import partial_to_all_subset_indices
        return tuple(partial_to_all_subset_indices(
            [e_subset, h_subset, dim_subset, dim_subset]))

    def split_ehpq(self, w):
        e_idx, h_idx, p_idx, q_idx = self.partial_to_ehpq_subsets()
        e, h, p, q = w[e_idx], w[h_idx], w[p_idx], w[q_idx]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=e), FVP(scalars=h)
        else:
            from hedge.tools import make_obj_array as moa
            return moa(e), moa(h), moa(p), moa(q)

    # sigma business ----------------------------------------------------------
    def _construct_scalar_coefficients(self, discr, node_coord, 
            i_min, i_max, o_min, o_max, exponent):
        assert o_min < i_min <= i_max < o_max 

        if o_min != i_min: 
            l_dist = (i_min - node_coord) / (i_min-o_min)
            l_dist_prime = discr.volume_zeros(kind="numpy", dtype=node_coord.dtype)
            l_dist_prime[l_dist >= 0] = -1 / (i_min-o_min)
            l_dist[l_dist < 0] = 0
        else:
            l_dist = l_dist_prime = numpy.zeros_like(node_coord)

        if i_max != o_max:
            r_dist = (node_coord - i_max) / (o_max-i_max)
            r_dist_prime = discr.volume_zeros(kind="numpy", dtype=node_coord.dtype)
            r_dist_prime[r_dist >= 0] = 1 / (o_max-i_max)
            r_dist[r_dist < 0] = 0
        else:
            r_dist = r_dist_prime = numpy.zeros_like(node_coord)

        l_plus_r = l_dist+r_dist
        return l_plus_r**exponent, \
                (l_dist_prime+r_dist_prime)*exponent*l_plus_r**(exponent-1), \
                l_plus_r

    def coefficients_from_boxes(self, discr, 
            inner_bbox, outer_bbox=None, 
            magnitude=None, tau_magnitude=None,
            exponent=None, dtype=None):
        if outer_bbox is None:
            outer_bbox = discr.mesh.bounding_box()

        if exponent is None:
            exponent = 2

        if magnitude is None:
            magnitude = 20

        if tau_magnitude is None:
            tau_magnitude = 0.4

        # scale by free space conductivity
        from math import sqrt
        magnitude = magnitude*sqrt(self.epsilon/self.mu)
        tau_magnitude = tau_magnitude*sqrt(self.epsilon/self.mu)

        i_min, i_max = inner_bbox
        o_min, o_max = outer_bbox

        from hedge.tools import make_obj_array

        nodes = discr.nodes
        if dtype is not None:
            nodes = nodes.astype(dtype)

        sigma, sigma_prime, tau = zip(*[self._construct_scalar_coefficients(
            discr, nodes[:,i], 
            i_min[i], i_max[i], o_min[i], o_max[i],
            exponent)
            for i in range(discr.dimensions)])

        return self.PMLCoefficients(
                sigma=magnitude*make_obj_array(sigma),
                sigma_prime=magnitude*make_obj_array(sigma_prime),
                tau=tau_magnitude*make_obj_array(tau))

    def coefficients_from_width(self, discr, width, 
            magnitude=None, tau_magnitude=None, exponent=None,
            dtype=None):
        o_min, o_max = discr.mesh.bounding_box()
        return self.coefficients_from_boxes(discr, 
                (o_min+width, o_max-width), 
                (o_min, o_max),
                magnitude, tau_magnitude, exponent, dtype)




class AbarbanelGottliebPMLTEMaxwellOperator(
        TEMaxwellOperator, AbarbanelGottliebPMLMaxwellOperator):
    # not unimplemented--this IS the implementation.
    pass

class AbarbanelGottliebPMLTMMaxwellOperator(
        TMMaxwellOperator, AbarbanelGottliebPMLMaxwellOperator):
    # not unimplemented--this IS the implementation.
    pass




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

        from hedge.flux import FluxVectorPlaceholder, \
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
        stab_term = 10 * PenaltyTerm() * (u.int - u.ext)
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
                self.neu_diff = pop.diffusion_tensor.boundary_interpolant(discr, 
                        poisson_op.neumann_tag)
            
            from hedge.mesh import TAG_ALL
            self.poincare_mean_value_hack = len(self.discr.get_boundary(TAG_ALL).nodes) 
        
        @property
        def dtype(self):
            return self.discr.default_scalar_type

        @property
        def shape(self):
            nodes = len(self.discr)
            return nodes, nodes

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
            if self.poincare_mean_value_hack:
                m_mean_state = 0
            else:
                from hedge.discretization import ones_on_volume
                B = self.discr.integral(ones_on_volume(self.discr)) 
                mean_state = self.discr.integral(u)/B
                m = ones_on_volume(self.discr)
                m_mean_state = m * mean_state

            return self.div(
                    ptwise_dot(2, 1, self.diffusion, self.grad(u)), 
                    u, apply_minv=apply_minv) \
                            - m_mean_state

        __call__ = op

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
 
            return (MassOperator().apply(self.discr, 
                rhs.volume_interpolant(self.discr))
                - self.div_c(w=w, dir_bc_w=dir_bc_w, neu_bc_w=neu_bc_w))
                        

    def bind(self, discr):
        assert self.dimensions == discr.dimensions

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.dirichlet_tag, self.neumann_tag])

        return self.BoundPoissonOperator(self, discr)

    # matrix creation ---------------------------------------------------------
    def grad_matrix(self):
        assert False, "this is broken"
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

        from hedge.flux import FluxVectorPlaceholder, make_normal

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

            from hedge.tools import to_obj_array
            return (
                    -self.discr.boundarize_volume_field(sqrt_coeff_v, hop.neumann_tag)
                    +
                    2*self.neumann_normals*
                        hop.neumann_bc.boundary_interpolant(
                            t, self.discr, hop.neumann_tag))

        def __call__(self, t, u):
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




class EulerOperator(TimeDependentOperator):
    """An nD Euler operator.

    Field order is [rho E rho_u_x rho_u_y ...].
    """
    def __init__(self, dimensions, gamma, bc):
        self.dimensions = dimensions
        self.gamma = gamma
        self.bc = bc

    def rho(self, q):
        return q[0]

    def e(self, q):
        return q[1]

    def rho_u(self, q):
        return q[2:2+self.dimensions]

    def u(self, q):
        from hedge.tools import make_obj_array
        return make_obj_array([
                rho_u_i/self.rho(q)
                for rho_u_i in self.rho_u(q)])

    def op_template(self):
        from hedge.optemplate import make_vector_field, \
                make_common_subexpression as cse

        def u(q):
            return cse(self.u(q))

        def p(q):
            return cse((self.gamma-1)*(self.e(q) - 0.5*numpy.dot(self.rho_u(q), u(q))))

        def flux(q):
            from pytools import delta
            from hedge.tools import make_obj_array, join_fields
            return [ # one entry for each flux direction
                    cse(join_fields(
                        # flux rho
                        self.rho_u(q)[i],

                        # flux E
                        cse(self.e(q)+p(q))*u(q)[i],

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q)[i]*self.u(q)[j] + delta(i,j) * p(q)
                            for j in range(self.dimensions)
                            ])
                        ))
                    for i in range(self.dimensions)]

        from hedge.optemplate import make_nabla, InverseMassOperator, \
                ElementwiseMaxOperator

        from pymbolic import var
        sqrt = var("sqrt")

        state = make_vector_field("q", self.dimensions+2)
        bc_state = make_vector_field("bc_q", self.dimensions+2)

        c = cse(sqrt(self.gamma*p(state)/self.rho(state)))

        speed = sqrt(numpy.dot(u(state), u(state))) + c

        from hedge.tools import make_lax_friedrichs_flux, join_fields
        from hedge.mesh import TAG_ALL
        return join_fields(
                (- numpy.dot(make_nabla(self.dimensions), flux(state))
                    + InverseMassOperator()*make_lax_friedrichs_flux(
                        wave_speed=ElementwiseMaxOperator()*c,
                        state=state, flux_func=flux,
                        bdry_tags_and_states=[
                            (TAG_ALL, bc_state)
                            ],
                        strong=True
                        )),
                    speed)

    def bind(self, discr):
        from hedge.mesh import TAG_ALL
        bound_op = discr.compile(self.op_template())

        def wrap(t, q):
            opt_result = bound_op(
                    q=q, 
                    bc_q=self.bc.boundary_interpolant(t, discr, TAG_ALL))
            max_speed = opt_result[-1]
            ode_rhs = opt_result[:-1]
            return ode_rhs, numpy.max(max_speed)

        return wrap



