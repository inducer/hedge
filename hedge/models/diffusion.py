# -*- coding: utf8 -*-
"""Operators modeling diffusive phenomena."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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

import hedge.data
from hedge.models import TimeDependentOperator
from hedge.models.poisson import LaplacianOperatorBase
from hedge.models.nd_calculus import CentralSecondDerivative




class DiffusionOperator(TimeDependentOperator, LaplacianOperatorBase):
    def __init__(self, dimensions, diffusion_tensor=None,
            dirichlet_bc=hedge.data.make_tdep_constant(0), dirichlet_tag="dirichlet",
            neumann_bc=hedge.data.make_tdep_constant(0), neumann_tag="neumann",
            scheme=CentralSecondDerivative()):
        self.dimensions = dimensions

        self.scheme = scheme

        self.dirichlet_bc = dirichlet_bc
        self.dirichlet_tag = dirichlet_tag
        self.neumann_bc = neumann_bc
        self.neumann_tag = neumann_tag

        if diffusion_tensor is None:
            diffusion_tensor = numpy.eye(dimensions)
        self.diffusion_tensor = diffusion_tensor

    def bind(self, discr):
        """Return a :class:`BoundPoissonOperator`."""

        assert self.dimensions == discr.dimensions

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.dirichlet_tag, self.neumann_tag])

        return BoundDiffusionOperator(self, discr)

    def estimate_timestep(self, discr, 
            stepper=None, stepper_class=None, stepper_args=None,
            t=None, fields=None):
        u"""Estimate the largest stable timestep, given a time stepper
        `stepper_class`. If none is given, RK4 is assumed.
        """

        rk4_dt = 0.2 \
                * (discr.dt_non_geometric_factor()
                * discr.dt_geometric_factor())**2

        from hedge.timestep.stability import \
                approximate_rk4_relative_imag_stability_region
        return rk4_dt * approximate_rk4_relative_imag_stability_region(
                stepper, stepper_class, stepper_args)




class BoundDiffusionOperator(hedge.iterative.OperatorBase):
    """Returned by :meth:`DiffusionOperator.bind`."""

    def __init__(self, diffusion_op, discr):
        hedge.iterative.OperatorBase.__init__(self)
        self.discr = discr

        dop = self.diffusion_op = diffusion_op

        op = dop.op_template(apply_minv=True)

        self.compiled_op = discr.compile(op)

        # Check whether use of Poincaré mean-value method is required.
        # (for pure Neumann or pure periodic)

        from hedge.mesh import TAG_ALL
        self.poincare_mean_value_hack = (
                len(self.discr.get_boundary(TAG_ALL).nodes)
                == len(self.discr.get_boundary(diffusion_op.neumann_tag).nodes))

    def __call__(self, t, u):
        dop = self.diffusion_op

        context = {"u": u}

        if not isinstance(self.diffusion_op.diffusion_tensor, numpy.ndarray):
            self.diffusion = dop.diffusion_tensor.volume_interpolant(t, discr)
            self.neu_diff = dop.diffusion_tensor.boundary_interpolant(
                    t, discr, diffusion_op.neumann_tag)

            context["diffusion"] = self.diffusion
            context["neumann_diffusion"] = self.neu_diff

        if not self.discr.get_boundary(dop.dirichlet_tag).is_empty():
            context["dir_bc"] = dop.dirichlet_bc.boundary_interpolant(
                    t, self.discr, dop.dirichlet_tag)
        if not self.discr.get_boundary(dop.neumann_tag).is_empty():
            context["neu_bc"] = dop.neumann_bc.boundary_interpolant(
                    t, self.discr, dop.neumann_tag)

        return self.compiled_op(**context)
