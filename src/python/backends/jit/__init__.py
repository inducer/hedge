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




import hedge.backends.cpu_base
import hedge.discretization
from hedge.backends.cpu_base import ExecutorBase, ExecutionMapperBase
from pymbolic.mapper.c_code import CCodeMapper
import numpy




# flux to code mapper ---------------------------------------------------------
class FluxToCodeMapper(CCodeMapper):
    def __init__(self, fvi, is_flipped=False):
        CCodeMapper.__init__(self, repr, reverse=False)
        self.flux_var_info = fvi
        self.is_flipped = is_flipped

    def map_normal(self, expr, enclosing_prec):
        if self.is_flipped:
            where = "opp"
        else:
            where = "loc"
        return "fp.%s.normal[%d]" % (where, expr.axis)

    def map_penalty_term(self, expr, enclosing_prec):
        if self.is_flipped:
            where = "opp"
        else:
            where = "loc"
        return ("pow(fp.%(where)s.order*fp.%(where)s.order/fp.%(where)s.h, %(pwr)r)" 
                % {"pwr": expr.power, "where": where})

    def map_field_component(self, expr, enclosing_prec):
        if expr.is_local ^ self.is_flipped:
            where = "loc"
        else:
            where = "opp"

        arg_name = self.flux_var_info.flux_dep_to_arg_name[expr]
        
        if not arg_name:
            return "0"
        else:
            return "%s_it[%s_idx]" % (arg_name, where)




class ExecutionMapper(ExecutionMapperBase):
    # flux implementation -----------------------------------------------------
    def _get_flux_var_info(self, flux, field_expr, bfield_expr=None):
        if bfield_expr is None:
            bfield_expr = field_expr

        from pytools import Record
        class FluxVariableInfo(Record):
            pass

        fvi = FluxVariableInfo(
                arg_exprs = [],
                arg_names = [],
                flux_dep_to_arg_name = {}, # or 0 if zero
                )

        args = []
        field_expr_to_arg_name = {}
        from hedge.flux import FieldComponent, FluxDependencyMapper
        for fc in FluxDependencyMapper(composite_leaves=True)(flux):
            assert isinstance(fc, FieldComponent)
            if fc.is_local:
                this_field_expr = field_expr
            else:
                this_field_expr = bfield_expr

            from hedge.tools import is_obj_array
            if is_obj_array(this_field_expr):
                fc_field_expr = this_field_expr[fc.index]
            else:
                assert fc.index == 0
                fc_field_expr = this_field_expr

            if not fc.is_local and field_expr is not bfield_expr:
                prefix = "b"
            else:
                prefix = ""

            from pymbolic.primitives import is_zero
            if is_zero(fc_field_expr):
                fvi.flux_dep_to_arg_name[fc] = 0
            else:
                value = self.rec(fc_field_expr)
                if is_zero(value):
                    fvi.flux_dep_to_arg_name[fc] = 0
                else:
                    if fc_field_expr not in field_expr_to_arg_name:
                        arg_name = prefix+"field%d" % fc.index
                        field_expr_to_arg_name[fc_field_expr] = arg_name

                        args.append(value)
                        fvi.arg_names.append(arg_name)
                        fvi.arg_exprs.append(fc_field_expr)
                    else:
                        arg_name = field_expr_to_arg_name[fc_field_expr]

                    fvi.flux_dep_to_arg_name[fc] = arg_name

        return args, fvi

    def scalar_inner_flux(self, op, field_expr, is_lift, out=None):
        if out is None:
            out = self.discr.volume_zeros()

        if isinstance(field_expr, (int, float, complex)) and field_expr == 0:
            return 0

        args, fvi = self._get_flux_var_info(op.flux, field_expr)
        kernel = self.executor.make_inner_flux_kernel(op.flux, fvi)

        if len(args):
            for fg in self.discr.face_groups:
                fluxes_on_faces = numpy.zeros(
                        (fg.face_count*fg.face_length()*fg.element_count(),),
                        dtype=args[0].dtype)
                kernel(fg, fluxes_on_faces, *args)
                
                if is_lift:
                    self.executor.lift_flux(fg, fg.ldis_loc.lifting_matrix(),
                            fg.local_el_inverse_jacobians, fluxes_on_faces, out)
                else:
                    self.executor.lift_flux(fg, fg.ldis_loc.multi_face_mass_matrix(),
                            None, fluxes_on_faces, out)

        return out

    def scalar_bdry_flux(self, op, bpair, is_lift, out=None):
        if out is None:
            out = self.discr.volume_zeros()

        bdry = self.discr.get_boundary(bpair.tag)
        if not len(bdry.nodes):
            return 0

        args, fvi = self._get_flux_var_info(op.flux, bpair.field, bpair.bfield)
        kernel = self.executor.make_bdry_flux_extractor(op.flux, fvi)

        if args:
            for fg in bdry.face_groups:
                fluxes_on_faces = numpy.zeros(
                        (fg.face_count*fg.face_length()*fg.element_count(),),
                        dtype=self.discr.default_scalar_type)

                from pytools import typedump
                kernel(fg, fluxes_on_faces, *args)

                if is_lift:
                    self.executor.lift_flux(fg, fg.ldis_loc.lifting_matrix(),
                            fg.local_el_inverse_jacobians, 
                            fluxes_on_faces, out)
                else:
                    self.executor.lift_flux(fg, fg.ldis_loc.multi_face_mass_matrix(),
                            None, 
                            fluxes_on_faces, out)

        return out

    # entry points ------------------------------------------------------------
    def map_flux(self, op, field_expr, out=None, is_lift=False):
        from hedge.optemplate import BoundaryPair

        if isinstance(field_expr, BoundaryPair):
            bpair = field_expr
            return self.scalar_bdry_flux(op, bpair, is_lift, out)
        else:
            return self.scalar_inner_flux(op, field_expr, is_lift, out)

    def map_lift(self, op, field_expr, out=None):
        return self.map_flux(op, field_expr, out, is_lift=True)





class Executor(ExecutorBase):
    def __init__(self, discr, optemplate):
        ExecutorBase.__init__(self, discr)
        self.optemplate = optemplate

        self.inner_kernel_cache = {}
        self.bdry_kernel_cache = {}

    def __call__(self, **vars):
        from pdb import set_trace
        #if "w" in vars:
            #if numpy.linalg.norm(vars["w"][5]) > 1e-6:
                #set_trace()
        return ExecutionMapper(vars, self.discr, self)(self.optemplate)

    # flux code generators --------------------------------------------------------
    def make_inner_flux_kernel(self, flux, fvi):
        cache_key = (flux, frozenset(fvi.flux_dep_to_arg_name.iteritems()))
        try:
            return self.inner_kernel_cache[cache_key]
        except KeyError:
            pass 

        from codepy.cgen import \
                FunctionDeclaration, FunctionBody, \
                Const, Reference, Value, MaybeUnused, \
                Statement, Include, Line, Block, Initializer, Assign, \
                CustomLoop, For

        from codepy.bpl import BoostPythonModule
        mod = BoostPythonModule()

        S = Statement
        mod.add_to_module([
            Include("hedge/face_operators.hpp"), 
            Include("boost/foreach.hpp"), 
            Line(),
            S("using namespace hedge"),
            Line()
            ])

        fdecl = FunctionDeclaration(
                Value("void", "gather_flux"), 
                [
                    Const(Reference(Value("face_group", "fg"))),
                    Value("py_vector", "fluxes_on_faces"),
                    ]+[
                    Const(Reference(Value("py_vector", arg_name)))
                    for arg_name in fvi.arg_names
                    ]
                )

        from pytools import flatten

        from pymbolic.mapper.stringifier import PREC_PRODUCT

        fbody = Block([
            Initializer(
                Const(Value("py_vector::iterator", "fof_it")),
                "fluxes_on_faces.begin()"),
            ]+[
            Initializer(
                Const(Value("py_vector::const_iterator", "%s_it" % arg_name)),
                arg_name + ".begin()")
            for arg_name in fvi.arg_names
            ]+[
            Line(),
            CustomLoop("BOOST_FOREACH(const face_pair &fp, fg.face_pairs)", Block(
                list(flatten([
                Initializer(Value("node_number_t", "%s_ebi" % where),
                    "fp.%s.el_base_index" % where),
                Initializer(Value("index_lists_t::const_iterator", "%s_idx_list" % where),
                    "fg.index_list(fp.%s.face_index_list_number)" % where),
                Initializer(Value("node_number_t", "%s_fof_base" % where),
                    "fg.face_length()*(fp.%(where)s.local_el_number*fg.face_count"
                    " + fp.%(where)s.face_id)" % {"where": where}),
                Line(),
                ]
                for where in ["loc", "opp"]
                ))+[
                Initializer(Value("index_lists_t::const_iterator", "opp_write_map"),
                    "fg.index_list(fp.opp_native_write_map)"),
                Line(),
                For(
                    "unsigned i = 0",
                    "i < fg.face_length()",
                    "++i",
                    Block(
                        [
                        Initializer(MaybeUnused(Value("node_number_t", "%s_idx" % where)),
                            "%(where)s_ebi + %(where)s_idx_list[i]" 
                            % {"where": where})
                        for where in ["loc", "opp"]
                        ]+[
                        Assign("fof_it[%s_fof_base+%s]" % (where, tgt_idx),
                            "fp.loc.face_jacobian * " +
                            FluxToCodeMapper(fvi, is_flipped=is_flipped)(flux, PREC_PRODUCT))
                        for where, is_flipped, tgt_idx in [
                            ("loc", False, "i"),
                            ("opp", True, "opp_write_map[i]")
                            ]
                        ]
                        )
                    )
                ]))
            ])
        mod.add_function(FunctionBody(fdecl, fbody)) 

        result = mod.compile(
                self.discr.platform, wait_on_error=True).gather_flux

        if self.discr.instrumented:
            from hedge.tools import time_count_flop, gather_flops
            result = \
                    time_count_flop(
                            result,
                            self.discr.gather_timer,
                            self.discr.gather_counter,
                            self.discr.gather_flop_counter,
                            gather_flops(self.discr)*len(fvi.arg_names))

        self.inner_kernel_cache[cache_key] = result
        return result

    def make_bdry_flux_extractor(self, flux, fvi):
        cache_key = (flux, frozenset(fvi.flux_dep_to_arg_name.iteritems()))
        try:
            return self.bdry_kernel_cache[cache_key]
        except KeyError:
            pass 

        from codepy.cgen import \
                FunctionDeclaration, FunctionBody, Template, \
                Const, Reference, Value, MaybeUnused, \
                Statement, Include, Line, Block, Initializer, Assign, \
                CustomLoop, For

        from codepy.bpl import BoostPythonModule
        mod = BoostPythonModule()

        S = Statement
        mod.add_to_module([
            Include("hedge/face_operators.hpp"), 
            Include("boost/foreach.hpp"), 
            Line(),
            S("using namespace hedge"),
            Line()
            ])

        fdecl = FunctionDeclaration(
                    Value("void", "gather_flux"), 
                    [
                    Const(Reference(Value("face_group", "fg"))),
                    Value("py_vector", "fluxes_on_faces"),
                    ]+[
                    Const(Reference(Value("py_vector", arg_name)))
                    for arg_name in fvi.arg_names])

        from pytools import flatten

        from pymbolic.mapper.stringifier import PREC_PRODUCT

        fbody = Block([
            Initializer(
                Const(Value("py_vector::iterator", "fof_it")),
                "fluxes_on_faces.begin()"),
            ]+[
            Initializer(
                Const(Value("py_vector::const_iterator", 
                    "%s_it" % arg_name)),
                "%s.begin()" % arg_name)
            for arg_name in fvi.arg_names
            ]+[
            Line(),
            CustomLoop("BOOST_FOREACH(const face_pair &fp, fg.face_pairs)", Block(
                list(flatten([
                Initializer(Value("node_number_t", "%s_ebi" % where),
                    "fp.%s.el_base_index" % where),
                Initializer(Value("index_lists_t::const_iterator", "%s_idx_list" % where),
                    "fg.index_list(fp.%s.face_index_list_number)" % where),
                Line(),
                ]
                for where in ["loc", "opp"]
                ))+[
                Line(),
                Initializer(Value("node_number_t", "loc_fof_base"),
                    "fg.face_length()*(fp.%(where)s.local_el_number*fg.face_count"
                    " + fp.%(where)s.face_id)" % {"where": "loc"}),
                Line(),
                For(
                    "unsigned i = 0",
                    "i < fg.face_length()",
                    "++i",
                    Block(
                        [
                        Initializer(MaybeUnused(
                            Value("node_number_t", "%s_idx" % where)),
                            "%(where)s_ebi + %(where)s_idx_list[i]" 
                            % {"where": where})
                        for where in ["loc", "opp"]
                        ]+[
                        Assign("fof_it[loc_fof_base+i]",
                            "fp.loc.face_jacobian * " +
                            FluxToCodeMapper(fvi)(flux, PREC_PRODUCT))
                        ]
                        )
                    )
                ]))
            ])
        mod.add_function(FunctionBody(fdecl, fbody))

        #print "----------------------------------------------------------------"
        #print flux
        #print FunctionBody(fdecl, fbody)

        result = mod.compile(self.discr.platform, wait_on_error=True).gather_flux

        if self.discr.instrumented:
            from pytools.log import time_and_count_function
            result = time_and_count_function( result, self.discr.gather_timer)

        self.bdry_kernel_cache[cache_key] = result
        return result




class Discretization(hedge.discretization.Discretization):
    def __init__(self, *args, **kwargs):
        hedge.discretization.Discretization.__init__(self, *args, **kwargs)

        plat = kwargs.pop("platform", None)

        if plat is None:
            from codepy.jit import guess_platform
            plat = guess_platform()

        plat = plat.copy()
        
        from codepy.libraries import add_hedge
        add_hedge(plat)

        self.platform = plat

    def compile(self, optemplate):
        from hedge.optemplate import \
                OperatorBinder, \
                InverseMassContractor, \
                BCToFluxRewriter

        from hedge.optemplate import CommutativeConstantFoldingMapper

        result = (
                InverseMassContractor()(
                    CommutativeConstantFoldingMapper()(
                        BCToFluxRewriter()(
                            OperatorBinder()(
                                optemplate)))))

        from hedge.tools import is_obj_array
        if is_obj_array(result):
            for i, x_i in enumerate(result):
                print "XX", i, x_i
                print
        else:
            print "XX", result

        return Executor(self, result)


