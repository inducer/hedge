"""Just-in-time compiling backend: Flux code generation."""

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




from pymbolic.mapper.c_code import CCodeMapper
from hedge.flux import FluxIdentityMapper




# flux to code mapper ---------------------------------------------------------
class FluxConcretizer(FluxIdentityMapper):
    def __init__(self, flux_idx, fvi):
        self.flux_idx = flux_idx
        self.flux_var_info = fvi

    def map_field_component(self, expr):
        if expr.is_local:
            where = "loc"
        else:
            where = "opp"

        arg_name = self.flux_var_info.flux_idx_and_dep_to_arg_name[
                self.flux_idx, expr]

        if not arg_name:
            return 0
        else:
            from pymbolic import var
            return var(arg_name+"_it")[var(where+"_idx")]

    def map_scalar_parameter(self, expr):
        from pymbolic import var
        return var("args._scalar_arg_%d" 
                % self.flux_var_info.scalar_parameters.index(expr))




class FluxToCodeMapper(CCodeMapper):
    def map_normal(self, expr, enclosing_prec):
        return "uncomplex_type(fp.loc.normal[%d])" % (expr.axis)

    def map_penalty_term(self, expr, enclosing_prec):
        return ("uncomplex_type(pow(fp.loc.order*fp.loc.order/fp.loc.h, %(pwr)r))"
                % {"pwr": expr.power})

    def map_function_symbol(self, expr, enclosing_prec):
        from hedge.flux import FluxFunctionSymbol, \
                flux_abs, flux_min, flux_max

        assert isinstance(expr, FluxFunctionSymbol)

        return {
                flux_abs: "std::abs",
                flux_max: "std::max",
                flux_min: "std::min",
                }[expr]

    def map_constant(self, x, enclosing_prec):
        import numpy
        if isinstance(x, complex):
            return "std::complex<uncomplex_type>(%s, %s)" % (
                    repr(x.real), repr(x.imag))
        else:
            return "uncomplex_type(%s)" % repr(x)





def flux_to_code(f2c, is_flipped, flux_idx, fvi, flux, prec):
    if is_flipped:
        from hedge.flux import FluxFlipper
        flux = FluxFlipper()(flux)

    return f2c(FluxConcretizer(flux_idx, fvi)(flux), prec)




# flux variable info ----------------------------------------------------------
def get_flux_var_info(fluxes):
    from pytools import Record
    class FluxVariableInfo(Record):
        pass

    scalar_parameters = set()

    fvi = FluxVariableInfo(
            scalar_parameters=None,
            arg_specs=[],
            arg_names=[],
            flux_idx_and_dep_to_arg_name={}, # or 0 if zero
            )

    field_expr_to_arg_name = {}

    from hedge.flux import \
            FieldComponent, FluxDependencyMapper, \
            FluxScalarParameter

    from hedge.optemplate import BoundaryPair

    for flux_idx, flux_binding in enumerate(fluxes):
        for dep in FluxDependencyMapper(include_calls=False)(flux_binding.op.flux):
            if isinstance(dep, FluxScalarParameter):
                scalar_parameters.add(dep)
            elif isinstance(dep, FieldComponent):
                is_bdry = isinstance(flux_binding.field, BoundaryPair)
                if is_bdry:
                    if dep.is_local:
                        this_field_expr = flux_binding.field.field
                    else:
                        this_field_expr = flux_binding.field.bfield
                else:
                    this_field_expr = flux_binding.field

                from hedge.tools import is_obj_array
                if is_obj_array(this_field_expr):
                    fc_field_expr = this_field_expr[dep.index]
                else:
                    assert dep.index == 0
                    fc_field_expr = this_field_expr

                def set_or_check(dict_instance, key, value):
                    try:
                        existing_value = dict_instance[key]
                    except KeyError:
                        dict_instance[key] = value
                    else:
                        assert existing_value == value

                from pymbolic.primitives import is_zero
                if is_zero(fc_field_expr):
                    fvi.flux_idx_and_dep_to_arg_name[flux_idx, dep] = 0
                else:
                    if fc_field_expr not in field_expr_to_arg_name:
                        arg_name = "arg%d" % len(fvi.arg_specs)
                        field_expr_to_arg_name[fc_field_expr] = arg_name

                        fvi.arg_names.append(arg_name)
                        fvi.arg_specs.append((fc_field_expr, dep.is_local))
                    else:
                        arg_name = field_expr_to_arg_name[fc_field_expr]

                    set_or_check(
                            fvi.flux_idx_and_dep_to_arg_name,
                            (flux_idx, dep),
                            arg_name)

                    if not is_bdry:
                        # Interior fluxes are used flipped as well.
                        # Make sure we have assigned arg names for the
                        # flipped case as well.
                        set_or_check(
                                fvi.flux_idx_and_dep_to_arg_name,
                                (flux_idx,
                                    FieldComponent(dep.index, not dep.is_local)),
                                arg_name)
            else:
                raise ValueError("unknown flux dependency type: %s" % dep)

    fvi.scalar_parameters = list(scalar_parameters)

    return fvi




def get_flux_toolchain(discr, fluxes):
    from hedge.flux import FluxFlopCounter
    flop_count = sum(FluxFlopCounter()(flux.op.flux) for flux in fluxes)

    toolchain = discr.toolchain
    if flop_count > 250:
        if "jit_dont_optimize_large_exprs" in discr.debug:
            toolchain = toolchain.with_optimization_level(0)
        else:
            toolchain = toolchain.with_optimization_level(1)

    return toolchain




def get_interior_flux_mod(fluxes, fvi, discr, dtype):
    from codepy.cgen import \
            FunctionDeclaration, FunctionBody, \
            Const, Reference, Value, MaybeUnused, Typedef, POD, \
            Statement, Include, Line, Block, Initializer, Assign, \
            CustomLoop, For, Struct

    from codepy.bpl import BoostPythonModule
    mod = BoostPythonModule()

    from pytools import to_uncomplex_dtype, flatten

    S = Statement
    mod.add_to_preamble([
        Include("cstdlib"),
        Include("algorithm"),
        Line(),
        Include("boost/foreach.hpp"),
        Line(),
        Include("hedge/face_operators.hpp"),
        ])

    mod.add_to_module([
        S("using namespace hedge"),
        S("using namespace pyublas"),
        Line(),
        Typedef(POD(dtype, "value_type")),
        Typedef(POD(to_uncomplex_dtype(dtype), "uncomplex_type")),
        Line(),
        ])

    arg_struct = Struct("arg_struct", [
        Value("numpy_array<value_type>", "flux%d_on_faces" % i)
        for i in range(len(fluxes))
        ]+[
        Value("numpy_array<value_type>", arg_name)
        for arg_name in fvi.arg_names
        ]+[
        Value("value_type" if scalar_par.is_complex else "uncomplex_type", 
            "_scalar_arg_%d" % i)
        for i, scalar_par in enumerate(fvi.scalar_parameters)
        ])

    mod.add_struct(arg_struct, "ArgStruct")
    mod.add_to_module([Line()])

    fdecl = FunctionDeclaration(
            Value("void", "gather_flux"),
            [
                Const(Reference(Value("face_group", "fg"))),
                Reference(Value("arg_struct", "args"))
                ])

    from pymbolic.mapper.stringifier import PREC_PRODUCT

    def gen_flux_code():
        f2cm = FluxToCodeMapper(repr, reverse=False)

        result = [
                Assign("fof%d_it[%s_fof_base+%s]" % (flux_idx, where, tgt_idx),
                    "uncomplex_type(fp.loc.face_jacobian) * " +
                    flux_to_code(f2cm, is_flipped, flux_idx, fvi, flux.op.flux, PREC_PRODUCT))
                for flux_idx, flux in enumerate(fluxes)
                for where, is_flipped, tgt_idx in [
                    ("loc", False, "i"),
                    ("opp", True, "opp_write_map[i]")
                    ]]

        return [
            Initializer(Value("value_type", f2cm.cse_prefix+str(i)), cse)
            for i, cse in f2cm.cses] + result

    fbody = Block([
        Initializer(
            Const(Value("numpy_array<value_type>::iterator", "fof%d_it" % i)),
            "args.flux%d_on_faces.begin()" % i)
        for i in range(len(fluxes))
        ]+[
        Initializer(
            Const(Value("numpy_array<value_type>::const_iterator", "%s_it" % arg_name)),
            "args.%s.begin()" % arg_name)
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
                    ]+gen_flux_code()
                    )
                )
            ]))
        ])
    mod.add_function(FunctionBody(fdecl, fbody))

    #print "----------------------------------------------------------------"
    #print mod.generate()
    #raw_input("[Enter]")

    return mod.compile(get_flux_toolchain(discr, fluxes), 
            wait_on_error="jit_wait_on_compile_error" in discr.debug)




def get_boundary_flux_mod(fluxes, fvi, discr, dtype):
    from codepy.cgen import \
            FunctionDeclaration, FunctionBody, Typedef, Struct, \
            Const, Reference, Value, POD, MaybeUnused, \
            Statement, Include, Line, Block, Initializer, Assign, \
            CustomLoop, For

    from pytools import to_uncomplex_dtype, flatten

    from codepy.bpl import BoostPythonModule
    mod = BoostPythonModule()

    mod.add_to_preamble([
        Include("cstdlib"),
        Include("algorithm"),
        Line(),
        Include("boost/foreach.hpp"),
        Line(),
        Include("hedge/face_operators.hpp"),
	])

    S = Statement
    mod.add_to_module([
        S("using namespace hedge"),
        S("using namespace pyublas"),
        Line(),
        Typedef(POD(dtype, "value_type")),
        Typedef(POD(to_uncomplex_dtype(dtype), "uncomplex_type")),
        ])

    arg_struct = Struct("arg_struct", [
        Value("numpy_array<value_type>", "flux%d_on_faces" % i)
        for i in range(len(fluxes))
        ]+[
        Value("numpy_array<value_type>", arg_name)
        for arg_name in fvi.arg_names
        ])

    mod.add_struct(arg_struct, "ArgStruct")
    mod.add_to_module([Line()])

    fdecl = FunctionDeclaration(
                Value("void", "gather_flux"),
                [
                    Const(Reference(Value("face_group", "fg"))),
                    Reference(Value("arg_struct", "args"))
                    ])

    from pymbolic.mapper.stringifier import PREC_PRODUCT

    def gen_flux_code():
        f2cm = FluxToCodeMapper(repr, reverse=False)

        result = [
                Assign("fof%d_it[loc_fof_base+i]" % flux_idx,
                    "uncomplex_type(fp.loc.face_jacobian) * " +
                    flux_to_code(f2cm, False, flux_idx, fvi, flux.op.flux, PREC_PRODUCT))
                for flux_idx, flux in enumerate(fluxes)
                ]

        return [
            Initializer(Value("value_type", f2cm.cse_prefix+str(i)), cse)
            for i, cse in enumerate(f2cm.cses)] + result

    fbody = Block([
        Initializer(
            Const(Value("numpy_array<value_type>::iterator", "fof%d_it" % i)),
            "args.flux%d_on_faces.begin()" % i)
        for i in range(len(fluxes))
        ]+[
        Initializer(
            Const(Value("numpy_array<value_type>::const_iterator",
                "%s_it" % arg_name)),
            "args.%s.begin()" % arg_name)
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
                    ]+gen_flux_code()
                    )
                )
            ]))
        ])

    mod.add_function(FunctionBody(fdecl, fbody))

    #print "----------------------------------------------------------------"
    #print mod.generate()
    #raw_input("[Enter]")

    return mod.compile(get_flux_toolchain(discr, fluxes), 
            wait_on_error="jit_wait_on_compile_error" in discr.debug)
