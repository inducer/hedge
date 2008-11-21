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
from hedge.backends.codegen_base import FluxToCodeMapperBase
import numpy




# flux to code mapper ---------------------------------------------------------
class FluxToCodeMapper(FluxToCodeMapperBase):
    def __init__(self, is_flipped=False):
        FluxToCodeMapperBase.__init__(self, repr, reverse=False)
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

        return "op%(idx)d_it[%(where)s_idx]" % {
                "where": where, "idx": expr.index
                }




# flux code generator ---------------------------------------------------------
def make_double_sided_flux_extractor(platform, flux):
    from codepy.cgen import \
            FunctionDeclaration, FunctionBody, \
            Const, Reference, Value, \
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

    from hedge.flux import FluxDependencyMapper
    deps = list(FluxDependencyMapper(composite_leaves=True)(flux))
    dep_indices = list(set(fc.index for fc in deps))
    dep_indices.sort()

    fdecl = FunctionDeclaration(
            Value("void", "gather_flux"), 
            [
                Const(Reference(Value("face_group", "fg"))),
                Value("py_vector", "fluxes_on_faces"),
                ]+[
                Const(Reference(Value("py_vector", "field%d" % idx)))
                for idx in dep_indices
                ]
            )

    from pytools import flatten

    from pymbolic.mapper.stringifier import PREC_NONE

    fbody = Block([
        Initializer(
            Const(Value("py_vector::iterator", "fof_it")),
            "fluxes_on_faces.begin()"),
        ]+[
        Initializer(
            Const(Value("py_vector::const_iterator", "op%d_it" % idx)),
            "field%d.begin()" % idx)
        for idx in dep_indices
        ]+[
        CustomLoop("BOOST_FOREACH(const face_pair &fp, fg.face_pairs)", Block(
            list(flatten([
            Initializer(Value("node_number_t", "%s_ebi" % where),
                "fp.%s.el_base_index" % where),
            Initializer(Value("index_lists_t::const_iterator", "%s_idx_list" % where),
                "fg.index_list(fp.%s.face_index_list_number)" % where),
            Initializer(Value("node_number_t", "%s_fof_base" % where),
                "fg.face_length()*(fp.%(where)s.local_el_number*fg.face_count"
                " + fp.opp.face_id)" % {"where": where}),
            Line(),
            ]
            for where in ["loc", "opp"]
            ))+[
            Line(),
            Initializer(Value("index_lists_t::const_iterator", "opp_write_map"),
                "fg.index_list(fp.opp_native_write_map)"),
            Line(),
            For(
                "unsigned i = 0",
                "i < fg.face_length()",
                "++i",
                Block(
                    [
                    Initializer(Value("node_number_t", "%s_idx" % where),
                        "%(where)s_ebi + %(where)s_idx_list[i]" 
                        % {"where": where})
                    for where in ["loc", "opp"]
                    ]+[
                    Assign("fof_it[%s_fof_base+%s]" % (where, tgt_idx),
                        FluxToCodeMapper(is_flipped)(flux, PREC_NONE))
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

    return mod.compile(platform, wait_on_error=True).gather_flux, dep_indices




class ExecutionMapper(hedge.backends.cpu_base.ExecutionMapper):
    # implementation stuff ----------------------------------------------------
    def scalar_inner_flux(self, op, field, lift, out=None):
        if out is None:
            out = self.discr.volume_zeros()

        if isinstance(field, (int, float, complex)) and field == 0:
            return 0

        kernel, dep_indices = make_double_sided_flux_extractor(self.discr.platform, op.flux)

        from hedge.tools import is_obj_array
        if is_obj_array(field):
            field_args = self.rec(field)
        else:
            field_args = [self.rec(field)]

        if len(field_args):
            for fg in self.discr.face_groups:
                fluxes_on_faces = numpy.zeros(
                        (fg.face_count*fg.face_length()*fg.element_count(),),
                        dtype=field_args[0].dtype)
                kernel(fg, fluxes_on_faces, *field_args)
                
                if lift:
                    self.lift_flux(fg, fg.ldis_loc.lifting_matrix(),
                            fg.local_el_inverse_jacobians, fluxes_on_faces, out)
                else:
                    self.lift_flux(fg, fg.ldis_loc.multi_face_mass_matrix(),
                            None, fluxes_on_faces, out)

        return out

    def scalar_bdry_flux(self, int_coeff, ext_coeff, field, bfield, tag, lift, out=None):
        raise NotImplementedError

        if out is None:
            out = self.discr.volume_zeros()

        bdry = self.discr.get_boundary(tag)
        if not bdry.nodes:
            return 0

        from hedge._internal import \
                perform_single_sided_flux, ChainedFlux, ZeroVector, \
                lift_flux
        if isinstance(field, (int, float, complex)) and field == 0:
            field = ZeroVector()
            dtype = bfield.dtype
        else:
            dtype = field.dtype

        if isinstance(bfield, (int, float, complex)) and bfield == 0:
            bfield = ZeroVector()

        if bdry.nodes:
            for fg in bdry.face_groups:
                fluxes_on_faces = numpy.zeros(
                        (fg.face_count*fg.face_length()*fg.element_count(),),
                        dtype=dtype)

                perform_single_sided_flux(
                        fg, ChainedFlux(int_coeff), ChainedFlux(ext_coeff),
                        field, bfield, fluxes_on_faces)

                if lift:
                    lift_flux(fg, fg.ldis_loc.lifting_matrix(),
                            fg.local_el_inverse_jacobians, 
                            fluxes_on_faces, out)
                else:
                    lift_flux(fg, fg.ldis_loc.multi_face_mass_matrix(),
                            None, 
                            fluxes_on_faces, out)

        return out




    # entry points ------------------------------------------------------------
    def map_flux(self, op, field_expr, out=None, lift=False):
        from hedge.optemplate import BoundaryPair

        if isinstance(field_expr, BoundaryPair):
            bpair = field_expr
            return self.scalar_bdry_flux(op, bpair, lift, out)
        else:
            return self.scalar_inner_flux(op, field_expr, lift, out)

    def map_lift(self, op, field_expr, out=None):
        return self.map_flux(op, field_expr, out, lift=True)





class CompiledOpTemplate:
    def __init__(self, discr, pp_optemplate):
        self.discr = discr
        self.pp_optemplate = pp_optemplate

    def __call__(self, **vars):
        return ExecutionMapper(vars, self.discr)(self.pp_optemplate)




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

        from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper

        result = (
                InverseMassContractor()(
                    CommutativeConstantFoldingMapper()(
                        BCToFluxRewriter()(
                            OperatorBinder()(
                                optemplate)))))

        return CompiledOpTemplate(self, result)


