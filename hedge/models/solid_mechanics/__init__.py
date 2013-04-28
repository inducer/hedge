# -*- coding: utf8 -*-
"""Hedge operator for solid mechanics using interior penalty stabilizer."""

from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

from pytools import memoize_method

import hedge.mesh
from hedge.models import HyperbolicOperator
from hedge.tools.symbolic import make_common_subexpression as cse
from hedge.tools import make_obj_array
from hedge.mesh import TAG_ALL, TAG_NONE

class SolidMechanicsOperator(HyperbolicOperator):
    """A 3D Solid Mechanics operator which supports constitutive
    behaviors specified in the constitutive_laws/ folder

    Field order is [Vx Vy Vz Ux Uy Uz]. 
    (V = dU/dt, U is displacement field)
    """

    _default_dimensions = 3

    def __init__(self, cstv_law,
            beta=6,
            init_displacement=None,
            init_velocity=None,
            dirichlet_tag=hedge.mesh.TAG_ALL,
            dirichlet_bc_data=None,
            traction_tag=hedge.mesh.TAG_NONE,
            traction_bc_data=None,
            dimensions=None):
        """
        :param flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs
        :param epsilon: can be a number, for fixed material throughout the computation
          domain, or a TimeConstantGivenFunction for spatially variable material coefficients
        :param mu: can be a number, for fixed material throughout the computation
          domain, or a TimeConstantGivenFunction for spatially variable material coefficients
        """

        self.dimensions = dimensions or self._default_dimensions
        self.material = cstv_law
        self.beta = beta
        self.init_displacement = init_displacement
        self.init_velocity = init_velocity
        self.dirichlet_tag = dirichlet_tag
        self.dirichlet_bc_data = dirichlet_bc_data
        self.traction_tag = traction_tag
        self.traction_bc_data = traction_bc_data

    def flux(self, beta, is_dirich):
        """The template for the numerical flux for variable coefficients.
            From Noels, Radovitzky 2007
        """
        from hedge.flux import (make_normal, FluxVectorPlaceholder,
                FluxConstantPlaceholder)
        from hedge.tools import join_fields
        
        dim = self.dimensions
        normal = make_normal(self.dimensions)
        w = FluxVectorPlaceholder(dim*2+9)

        # u is displacement field, v is its time derivative (velocity)
        u, v, F = self.split_grad_vars(w)
      
        P_int = self.material.stress(F.int, self.dimensions)
        C_int = self.material.tangent_moduli(F.int, self.dimensions, self.dimensions)

        # constitutive update for exterior face
        if is_dirich:
            P_ext = [0]*9  #[-3*p for p in P_int]
            C_ext = [0]*81 #C_int
        else:
            P_ext = self.material.stress(F.ext, self.dimensions)
            C_ext = self.material.tangent_moduli(F.ext, self.dimensions, self.dimensions)
        
        P_avg = [(P_int[i] + P_ext[i])/2 for i in range(dim*dim)]
        # 'force' flux
        v_flux = [0,]*self.dimensions
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                v_flux[i] = v_flux[i] + P_avg[3*i+j]*normal[j]
        
        from hedge.flux import make_penalty_term
        stab_factor = beta * make_penalty_term()
        C_avg = [stab_factor * (C_int[i] + C_ext[i]) / 2 for i in range(dim*dim*dim*dim)]
        # stabilization term
        u_jump = u.ext - u.int
        for i in range(self.dimensions):
            for j in range(self.dimensions):
               for k in range(self.dimensions):
                    for l in range(self.dimensions):
                        v_flux[i] = v_flux[i] - normal[j]* \
                                                C_avg[27*i+9*j+3*k+l]* \
                                                u_jump[k]* \
                                                normal[l]

        return join_fields(
                # u needs no flux term
                0,0,0,
                # flux for v
                v_flux[0], v_flux[1], v_flux[2]
                )

    def dirichlet_bc(self, w=None):
        """
        Flux term for dirichlet (displacement) boundary conditions
        """
        u, v = self.split_vars(self.field_placeholder(w))

        if self.dirichlet_bc_data is not None:
            from hedge.optemplate import make_vector_field
            dir_field = cse(
                    -make_vector_field("dirichlet_bc", 3))
        else:
            dir_field = make_obj_array([0,]*3)
        
        from hedge.tools import join_fields
        return join_fields(dir_field, [0]*3, [0,]*9)


    def local_derivatives(self, w=None):
        """Template for the volume terms of the time derivatives for
        U and V.  dU/dt = V, and dV/dt = div P.  Body forces not yet
        implemented"""
        u, v, F = self.split_grad_vars(w)
        dim = self.dimensions

        from hedge.optemplate import make_stiffness_t, make_vector_field
        from hedge.tools import join_fields
 
        P = self.material.stress(F, self.dimensions)
        
        stiffness = make_stiffness_t(dim)
        Dv = [0,]*3
        for i in range(dim):
            for j in range(dim):
                Dv[i] = Dv[i] - stiffness[j](P[3*j+i])

        # in conservation form: u_t + A u_x = 0
        return join_fields(
                # time derivative of u is v
                v[0], v[1], v[2],
                # time derivative of v is div P
                Dv[0], Dv[1], Dv[2]
                )
    
    def op_template(self, w=None):
        """The full operator template - the high level description of
        the nonlinear mechanics operator.

        Combines the relevant operator templates for spatial
        derivatives, flux, boundary conditions etc.

        NOTE: Only boundary conditions allowed currently are homogenous
        dirichlet and neumann, and I'm not sure dirichlet is done 
        properly
        """
        from hedge.optemplate import InverseMassOperator, Field, \
                make_vector_field

        from hedge.tools import join_fields
        w = self.field_placeholder(w)
       
        u,v = self.split_vars(w)
        from hedge.optemplate import make_nabla

        nabla = make_nabla(self.dimensions)
        ij = 0
        F = [0,]*9
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                F[ij] = nabla[j](u[i]) 
                if i == j:
                    F[ij] = F[ij] + 1
                ij = ij + 1
        
        w = join_fields(u,v,F)
        flux_w = w

        from hedge.optemplate import BoundaryPair, get_flux_operator

        flux_op = get_flux_operator(self.flux(self.beta, is_dirich=False))
        d_flux_op = get_flux_operator(self.flux(self.beta, is_dirich=True))
        
        from hedge.optemplate import make_normal, BoundarizeOperator

        dir_normal = make_normal(self.dirichlet_tag, self.dimensions)

        dir_bc = self.dirichlet_bc(w)

        return    self.local_derivatives(w) \
                - (flux_op(flux_w) + 
                   d_flux_op(BoundaryPair(flux_w, dir_bc, self.dirichlet_tag))
                  )
    
    def bind(self, discr, **extra_context):
        "Convert the abstract operator template into compiled code."
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.dirichlet_tag, self.traction_tag])

        compiled_op_template = discr.compile(self.op_template())
        
        def rhs(t, w):
            if self.dirichlet_bc_data is not None:
                dirichlet_bc_data = self.dirichlet_bc_data.boundary_interpolant(
                        t, discr, self.dirichlet_tag)[0:3]
            else:
                dirichlet_bc_data = 0

            kwargs = {}
            kwargs.update(extra_context)
            return compiled_op_template(w=w, 
                    dirichlet_bc=dirichlet_bc_data, **kwargs)

        return rhs

    def calculate_piola(self,u=None):
        from hedge.optemplate import make_nabla
        from hedge.optemplate import make_vector_field
        
        u = make_vector_field('u', 3)

        nabla = make_nabla(self.dimensions)
        ij = 0
        F = [0,]*9
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                F[ij] = nabla[i](u[j])
                if i == j:
                    F[ij] = F[ij] + 1
                ij = ij + 1
        
        return make_obj_array(self.material.stress(F, self.dimensions))
    
    def bind_stress_calculator(self, discr):
        compiled_calc = discr.compile(self.calculate_piola())
        
        def calc(u):
            return compiled_calc(u=u)
        
        return calc


    def field_placeholder(self, w=None):
        "A placeholder for u and v."
        fld_cnt = self.dimensions*2
        if w is None:
            from hedge.optemplate import make_vector_field
            w = make_vector_field("w", fld_cnt)
        return w
    
    def assemble_vars(self, u=None, v=None,F=None, discr=None):
        "Combines separate U and V vectors into a single array."
        if discr is None:
            def zero():
                return 0
        else:
            def zero():
                return discr.volume_zeros()

        dim = self.dimensions

        def default_fld(fld, comp):
            if fld is None:
                return [zero() for i in xrange(comp)]
            else:
                return fld

        if self.init_velocity is None or discr is None:
            v = default_fld(v, dim)
        else:
            v = self.init_velocity.volume_interpolant(discr)

        if self.init_displacement is None or discr is None:
            u = default_fld(u, dim)
        else:
            u = self.init_displacement.volume_interpolant(discr)

        from hedge.tools import join_fields
        return join_fields(u, v)

    assemble_fields = assemble_vars

    def split_grad_vars(self, w):
        "Splits an array into U, V, and F components"
        dim = self.dimensions
        u, v = w[:dim], w[dim:2*dim]
        F = w[2*dim:]
        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=u), FVP(scalars=v), FVP(scalars=F)
        else:
            return make_obj_array(u), make_obj_array(v), make_obj_array(F)
    
    def split_vars(self, w):
        "Splits an array into U and V components"
        dim = self.dimensions
        u, v = w[:dim], w[dim:]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=u), FVP(scalars=v)
        else:
            return make_obj_array(u), make_obj_array(v)

    def max_eigenvalue(self, t, fields=None, discr=None):
        """Return the wave speed in the material"""
        u,v = self.split_vars(fields)
        from hedge.optemplate import make_nabla
        nabla = make_nabla(self.dimensions)
        dim = self.dimensions
        F = [0,]*dim*dim
        ij = 0
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                F[ij] = nabla[i]*u[j]
                if i == j:
                    F[ij] = F[ij] + 1
                ij = ij + 1
        speed = self.material.celerity(F, self.dimensions)
        max_speed = discr.nodewise_max(speed)
        return max_speed * self.beta
