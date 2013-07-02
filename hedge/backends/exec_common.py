"""Code for operator execution shared among multiple backends."""

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


import numpy as np
import hedge.optemplate


class ExecutionMapperBase(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin,
        hedge.optemplate.LocalOpReducerMixin):
    def __init__(self, context, executor):
        hedge.optemplate.Evaluator.__init__(self, context)
        self.discr = executor.discr
        self.executor = executor

    def map_ones(self, expr):
        # FIXME
        if expr.quadrature_tag is not None:
            raise NotImplementedError("ones on quad. grids")

        result = self.discr.volume_empty(kind=self.discr.compute_kind)
        result.fill(1)
        return result

    def map_node_coordinate_component(self, expr):
        # FIXME
        if expr.quadrature_tag is not None:
            raise NotImplementedError("node coordinate components on quad. grids")
        # FIXME: Data transfer and strided CPU index every time, ugh.
        return self.discr.convert_volume(
                # Yes, that .copy() is necessary because much of hedge
                # doesn't check for striding. Ugh^2.
                self.discr.nodes[:, expr.axis].copy(),
                kind=self.discr.compute_kind)

    def map_normal_component(self, expr):
        if expr.quadrature_tag is not None:
            raise NotImplementedError("normal components on quad. grids")
        return self.discr.boundary_normals(expr.boundary_tag)[expr.axis]

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

    def map_call(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        func_name = expr.function.name

        try:
            func = self.discr.exec_functions[func_name]
        except KeyError:
            func = getattr(np, expr.function.name)

        return func(*[self.rec(p) for p in expr.parameters])
