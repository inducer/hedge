"""Code for operator execution shared among multiple backends."""

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




import hedge.optemplate




class ExecutionMapperBase(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin,
        hedge.optemplate.LocalOpReducerMixin):
    def __init__(self, context, executor):
        hedge.optemplate.Evaluator.__init__(self, context)
        self.discr = executor.discr
        self.executor = executor

    def map_normal_component(self, expr):
        return self.discr.boundary_normals(expr.tag)[expr.axis]

    def map_boundarize(self, op, field_expr):
        return self.discr.boundarize_volume_field(
                self.rec(field_expr), tag=op.tag,
                kind=self.discr.compute_kind)

    def map_scalar_parameter(self, expr):
        return self.context[expr.name]

    def map_jacobian(self, expr):
        return self.discr.volume_jacobians(expr.quadrature_tag)

    def map_forward_metric_derivative(self, expr):
        return (self.discr.forward_metric_derivatives(expr.quadrature_tag)
                    [expr.xyz_axis][expr.rst_axis])

    def map_inverse_metric_derivative(self, expr):
        return (self.discr.inverse_metric_derivatives(expr.quadrature_tag)
                    [expr.xyz_axis][expr.rst_axis])

