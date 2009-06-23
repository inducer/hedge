"""Just-in-time compiling backend."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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
from pytools import memoize_method




class JitDifferentiator:
    def __init__(self, discr):
        self.discr = discr

    @memoize_method
    def make_diff(self, elgroup, dtype):
        from hedge._internal import UniformElementRanges
        assert isinstance(elgroup.ranges, UniformElementRanges)

        ldis = elgroup.local_discretization
        discr = self.discr
        from codepy.cgen import \
                FunctionDeclaration, FunctionBody, Typedef, \
                Const, Reference, Value, POD, \
                Statement, Include, Line, Block, Initializer, Assign, \
                For, \
                Define

        from pytools import to_uncomplex_dtype

        from codepy.bpl import BoostPythonModule
        mod = BoostPythonModule()

        S = Statement
        mod.add_to_preamble([
            Include("hedge/volume_operators.hpp"),
            Include("boost/foreach.hpp"),
            ])

        mod.add_to_module([
            S("namespace ublas = boost::numeric::ublas"),
            S("using namespace hedge"),
            S("using namespace pyublas"),
            Line(),
            Define("DOFS_PER_EL", ldis.node_count()),
            Define("DIMENSIONS", discr.dimensions),
            Line(),
            Typedef(POD(dtype, "value_type")),
            Typedef(POD(to_uncomplex_dtype(dtype), "uncomplex_type")),
            ])

        fdecl = FunctionDeclaration(
                    Value("void", "diff"),
                    [
                    Const(Reference(Value("uniform_element_ranges", "ers"))),
                    Value("numpy_array<value_type>", "field")
                    ]+[
                    Value("ublas::matrix<uncomplex_type>", "diffmat_rst%d" % rst)
                    for rst in range(discr.dimensions)
                    ]+[
                    Value("numpy_array<value_type>", "result%d" % i)
                    for i in range(discr.dimensions)
                    ]+[
                    Value("numpy_array<double>", "coeffs"),
                    Value("numpy_array<npy_uint32>", "el_nrs"),
                    POD(numpy.uint32, "total_el_count"),
                    ]
                    )

        def make_it(name, is_const=True, tpname="value_type"):
            if is_const:
                const = "const_"
            else:
                const = ""

            return Initializer(
                Value("numpy_array<%s>::%siterator" % (tpname, const), name+"_it"),
                "%s.begin()" % name)

        fbody = Block([
            S("assert(DOFS_PER_EL == ers.el_size())"),
            Line(),
            make_it("field"),
            ]+[
            make_it("result%d" % i, is_const=False)
            for i in range(discr.dimensions)
            ]+[
            Line(),
            make_it("coeffs", tpname="double"),
            make_it("el_nrs", tpname="npy_uint32"),
            Line(),
            For("element_number_t eg_el_nr = 0",
                "eg_el_nr < ers.size()",
                "++eg_el_nr",
                Block([
                    Initializer(
                        Value("node_number_t", "el_base"),
                        "ers.start() + eg_el_nr*DOFS_PER_EL"),
                    Initializer(
                        Value("element_number_t", "el_nr"),
                        "el_nrs_it[eg_el_nr]"),
                    Line(),
                    For("unsigned i = 0",
                        "i < DOFS_PER_EL",
                        "++i",
                        Block([
                            Initializer(Value("value_type", "drst_%d" % rst), 0)
                            for rst in range(discr.dimensions)
                            ]+[
                            Line(),
                            ]+[
                            For("unsigned j = 0",
                                "j < DOFS_PER_EL",
                                "++j",
                                Block([
                                    S("drst_%(rst)d += diffmat_rst%(rst)d(i, j)*field_it[el_base+j]"
                                        % {"rst":rst})
                                    for rst in range(discr.dimensions)
                                    ])
                                ),
                            Line(),
                            ]+[
                            Assign("result%d_it[el_base+i]" % xyz,
                                " + ".join(
                                    "uncomplex_type(coeffs_it[total_el_count*("
                                    "DIMENSIONS*%(xyz)d + %(rst)d) + el_nr])"
                                    " * drst_%(rst)d"
                                    % {"xyz":xyz, "rst":rst}
                                    for rst in range(discr.dimensions)
                                    )
                                )
                            for xyz in range(discr.dimensions)

                            ])
                        )
                    ])
                )
            ])

        mod.add_function(FunctionBody(fdecl, fbody))

        #print "----------------------------------------------------------------"
        #print mod.generate()
        #raw_input()

        compiled_func = mod.compile(self.discr.toolchain, wait_on_error=True).diff

        if self.discr.instrumented:
            from hedge.tools import time_count_flop

            compiled_func = time_count_flop(compiled_func,
                    discr.diff_timer, discr.diff_counter,
                    discr.diff_flop_counter,
                    flops=discr.dimensions*(
                        2 # mul+add
                        * ldis.node_count() * len(elgroup.members)
                        * ldis.node_count()
                        +
                        2 * discr.dimensions
                        * len(elgroup.members) * ldis.node_count()),
                    increment=discr.dimensions)

        return compiled_func

    def __call__(self, op_class, field, xyz_needed):
        result = [self.discr.volume_zeros(dtype=field.dtype) 
                for i in range(self.discr.dimensions)]
        from hedge.tools import is_zero
        if not is_zero(field):
            for eg in self.discr.element_groups:
                coeffs = op_class.coefficients(eg)

                from pytools import to_uncomplex_dtype
                uncomplex_dtype = to_uncomplex_dtype(field.dtype)
                args = ([eg.ranges, field]
                        + [m.astype(uncomplex_dtype) for m in op_class.matrices(eg)]
                        + result
                        + [coeffs, eg.member_nrs, coeffs.shape[2]])

                diff_routine = self.make_diff(eg, field.dtype)
                diff_routine(*args)

        return [result[i] for i in xyz_needed]

