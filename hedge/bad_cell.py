# -*- coding: utf-8 -*-
"""Bad-cell indicators."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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



import numpy
import numpy.linalg as la
from hedge.optemplate.operators import (
        ElementwiseLinearOperator, StatelessOperator)
from pytools import Record, memoize_method




# {{{ Persson-Peraire ---------------------------------------------------------
def persson_peraire_filter_response_function(mode_idx, ldis):
    if sum(mode_idx) == ldis.order:
        return 0
    else:
        return 1



class PerssonPeraireDiscontinuitySensor(object):
    """
    see
    [1] P. Persson und J. Peraire,
    "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods,"
    Proc. of the 44th AIAA Aerospace Sciences Meeting and Exhibit, 2006.
    """

    def __init__(self, kappa, eps0, s_0):
        self.kappa = kappa
        self.eps0 = eps0
        self.s_0 = s_0

    def op_template(self, u=None):
        from pymbolic.primitives import IfPositive, Variable
        from hedge.optemplate.primitives import Field, ScalarParameter
        from hedge.tools.symbolic import make_common_subexpression as cse
        from math import pi

        if u is None:
            u = Field("u")

        from hedge.optemplate.operators import (
                MassOperator, FilterOperator, OnesOperator)

        mode_truncator = FilterOperator(
                persson_peraire_filter_response_function)

        truncated_u = mode_truncator(u)
        diff = u - truncated_u

        el_norm_squared_mass_diff_u = OnesOperator()(MassOperator()(diff)*diff)
        el_norm_squared_mass_u = OnesOperator()(MassOperator()(u)*u)

        capital_s_e = cse(el_norm_squared_mass_diff_u / el_norm_squared_mass_u,
                "S_e")

        sin = Variable("sin")
        log10 = Variable("log10")

        s_e = cse(log10(capital_s_e), "s_e")
        kappa = ScalarParameter("kappa")
        eps0 = ScalarParameter("eps0")
        s_0 = ScalarParameter("s_0")

        return IfPositive(s_0-self.kappa-s_e,
                0,
                IfPositive(s_e-self.kappa-s_0,
                    eps0,
                    eps0/2*(1+sin(pi*(s_e-s_0)/self.kappa))))

    def bind(self, discr):
        compiled = discr.compile(self.op_template())

        from pytools import match_precision
        scalar_type = match_precision(
                numpy.dtype(numpy.float64),
                discr.default_scalar_type).type

        kappa = scalar_type(self.kappa)
        eps0 = scalar_type(self.eps0)
        s_0 = scalar_type(self.s_0)

        def apply(u):
            return compiled(u=u, kappa=kappa, eps0=eps0, s_0=s_0)

        return apply





# }}}

# {{{ exponential fit ---------------------------------------------------------
# {{{ operators for basic fit
class DecayEstimateOperatorBase(ElementwiseLinearOperator):
    def __init__(self, ignored_modes, weight_mode):
        self.ignored_modes = ignored_modes
        self.weight_mode = weight_mode

    def __getinitargs__(self):
        return (self.ignored_modes, self.weight_mode)

    def make_mode_number_vector(self, ldis):
        im = self.ignored_modes
        node_cnt = ldis.node_count()

        mode_number_vector = numpy.zeros(node_cnt-im, dtype=numpy.float64)
        for i, mid in enumerate(ldis.generate_mode_identifiers()):
            if i < im:
                continue
            mode_number_vector[i-im] = sum(mid)

        return mode_number_vector

    def make_weight_vector(self, ldis):
        node_cnt = ldis.node_count()

        if self.weight_mode == "nd_weight":
            im = self.ignored_modes
            node_cnt = ldis.node_count()

            degree_count = {}
            for i, mid in enumerate(ldis.generate_mode_identifiers()):
                degree_count[sum(mid)] = \
                        degree_count.get(sum(mid), 0) + 1

            result = numpy.zeros(node_cnt-im, dtype=numpy.float64)
            for i, mid in enumerate(ldis.generate_mode_identifiers()):
                if i < im:
                    continue

                result[i-im] = 1/degree_count[sum(mid)]

            return result**0.5

        elif (isinstance(self.weight_mode, tuple)
                and self.weight_mode[0] == "exponential"):
            assert self.ignored_modes == 0

            weight_exponent = self.weight_mode[1]
            mode_number_vector = self.make_mode_number_vector(ldis)
            return mode_number_vector**self.weight_exponent

        elif self.weight_mode is None:
            return numpy.ones(node_cnt-self.ignored_modes,
                    dtype=numpy.float64)
        else:
            raise ValueError("invalid weight mode: "
                    + str(self.weight_mode))

    def decay_fit_mat(self, ldis):
        im = self.ignored_modes
        node_cnt = ldis.node_count()

        mode_number_vector = self.make_mode_number_vector(ldis)
        weight_vector = self.make_weight_vector(ldis)

        a = numpy.zeros((node_cnt-im, 2), dtype=numpy.float64)
        a[:,0] = weight_vector
        a[:,1] = weight_vector * numpy.log(mode_number_vector)

        if im == 0:
            assert not numpy.isfinite(a[0,1])
            a[0,1] = 0

        result = numpy.zeros((2, node_cnt))
        result[:,im:] = la.pinv(a)

        return result





class DecayExponentOperator(DecayEstimateOperatorBase):
    def matrix(self, eg):
        ldis = eg.local_discretization
        plsm = self.decay_fit_mat(ldis)
        a = numpy.zeros((ldis.node_count(), ldis.node_count()))
        for i in range(ldis.node_count()):
            a[i] = plsm[1]

        return a

class LogDecayConstantOperator(DecayEstimateOperatorBase):
    def matrix(self, eg):
        ldis = eg.local_discretization
        plsm = self.decay_fit_mat(ldis)
        a = numpy.zeros((ldis.node_count(), ldis.node_count()))
        for i in range(ldis.node_count()):
            a[i] = plsm[0]

        return a

# }}}

# {{{ data vector creation
def create_mode_number_vector(discr, nonzero):
    result = discr.volume_zeros(kind="numpy")
    for eg in discr.element_groups:
        ldis = eg.local_discretization

        modal_coefficients = numpy.zeros(ldis.node_count(), dtype=result.dtype)
        for i, mid in enumerate(ldis.generate_mode_identifiers()):
            msum = sum(mid)
            if msum == 0 and nonzero:
                modal_coefficients[i] = 1
            else:
                modal_coefficients[i] = msum

        eg.el_array_from_volume(result)[:,:] = modal_coefficients

    return result

def create_mode_weight_vector(discr, expt_op):
    result = discr.volume_zeros(kind="numpy")
    for eg in discr.element_groups:
        ldis = eg.local_discretization

        modal_coefficients = expt_op.make_weight_vector(ldis)

        eg.el_array_from_volume(result)[:,expt_op.ignored_modes:] = \
                    modal_coefficients

    return discr.convert_volume(result, kind=discr.compute_kind)

def create_decay_baseline(discr):
    """Create a vector of modal coefficients that exhibit 'optimal'
    (:math:`k^{-N}`) decay.
    """
    result = discr.volume_zeros(kind="numpy")
    for eg in discr.element_groups:
        ldis = eg.local_discretization

        modal_coefficients = numpy.zeros(ldis.node_count(), dtype=result.dtype)
        for i, mid in enumerate(ldis.generate_mode_identifiers()):
            msum = sum(mid)
            if msum != 0:
                modal_coefficients[i] = msum**(-ldis.order)
                #modal_coefficients[i] = 1e-7
            else:
                modal_coefficients[i] = 1 # irrelevant, just keeps log from NaNing

        modal_coefficients /= la.norm(modal_coefficients[1:])

        eg.el_array_from_volume(result)[:,:] = modal_coefficients

    return discr.convert_volume(result, kind=discr.compute_kind)




# }}}

# {{{ supporting classes
class BottomChoppingFilterResponseFunction:
    def __init__(self, ignored_modes):
        self.ignored_modes = ignored_modes

    def __call__(self, mode_idx, ldis):
        if sum(mode_idx) < self.ignored_modes:
            return 0
        else:
            return 1




class DecayInformation(Record):
    def __init__(self, **kwargs):
        from hedge.tools.symbolic import make_common_subexpression as cse

        Record.__init__(self, dict((name, cse(expr, name))
            for name, expr in kwargs.iteritems()))

# }}}

# {{{ supporting operators

class InverseChebyshevVandermondeOperator(ElementwiseLinearOperator, StatelessOperator):
    def matrix(self, eg):
        class ChebyshevMode:
            def __init__(self, n):
                from math import pi
                self.n = n
                if n == 0:
                    self.normalization = 1/pi
                else:
                    self.normalization = 1/(pi/2)

            def __call__(self, x):
                from math import acos, cos
                return cos(self.n*acos(x[0]))

        from hedge.discretization.local import IntervalDiscretization
        assert isinstance(eg.local_discretization, IntervalDiscretization)

        ldis = eg.local_discretization
        modes = [ChebyshevMode(i) for i in range(ldis.order+1)]

        from hedge.polynomial import generic_vandermonde
        vdm = generic_vandermonde(ldis.unit_nodes(), modes)

        return numpy.asarray(la.inv(vdm), order="C")

# }}}

# {{{ the actual sensor
class DecayFitDiscontinuitySensorBase(object):
    def __init__(self, mode_processor, weight_mode, ignored_modes):
        self.mode_processor = mode_processor
        self.weight_mode = weight_mode
        self.ignored_modes = ignored_modes

    def op_template_struct(self, u, with_baseline=True):
        from hedge.optemplate.operators import (
                MassOperator, OnesOperator, InverseVandermondeOperator,
                InverseMassOperator)
        from hedge.optemplate.primitives import Field
        from hedge.optemplate.tools import get_flux_operator
        from hedge.tools.symbolic import make_common_subexpression as cse
        from hedge.optemplate.primitives import CFunction
        from pymbolic.primitives import Variable

        if u is None:
            u = Field("u")

        from hedge.flux import (
                FluxScalarPlaceholder, ElementOrder,
                ElementJacobian, FaceJacobian, flux_abs)

        log, exp, sqrt = CFunction("log"), CFunction("exp"), CFunction("sqrt")

        if False:
            # On the whole, this should scale like u.
            # Columns of lift scale like 1/N^2, compensate for that.
            # Further compensate for all geometric factors.

            u_flux = FluxScalarPlaceholder(0)

            jump_part = InverseMassOperator()(
                    get_flux_operator(
                        ElementJacobian()/(ElementOrder()**2 * FaceJacobian())
                            *flux_abs(u_flux.ext - u_flux.int))(u))

        baseline_squared = Field("baseline_squared")
        el_norm_u_squared = cse(
                OnesOperator()(MassOperator()(u)*u),
                "l2_norm_u")

        indicator_modal_coeffs = cse(
                InverseVandermondeOperator()(u),
                #InverseChebyshevVandermondeOperator()(u),
                "u_modes")

        indicator_modal_coeffs_squared = indicator_modal_coeffs**2

        if self.mode_processor is not None:
            indicator_modal_coeffs_squared = \
                    Variable("mode_processor")(indicator_modal_coeffs_squared)

        log_modal_coeffs = cse(
                log(indicator_modal_coeffs_squared
                    + baseline_squared*el_norm_u_squared
                    )/2,
                "log_modal_coeffs")

        if False:
            modal_coeffs_jump = cse(
                    InverseVandermondeOperator()(jump_part),
                    "jump_modes")
            log_modal_coeffs_jump = cse(
                    log(modal_coeffs_jump**2)/2,
                    "lmc_jump")

        # fit to c * n**s
        expt_op = DecayExponentOperator(
                self.ignored_modes, self.weight_mode)
        log_const_op = LogDecayConstantOperator(
                self.ignored_modes, self.weight_mode)

        mode_weights = Field("mode_weights")

        weighted_log_modal_coeffs = mode_weights*log_modal_coeffs

        s = cse(expt_op(weighted_log_modal_coeffs),
                "first_decay_expt")
        log_c = cse(log_const_op(weighted_log_modal_coeffs),
                "first_decay_coeff")
        c = exp(log_const_op(weighted_log_modal_coeffs))

        log_mode_numbers = Field("log_mode_numbers")
        estimated_log_modal_coeffs = cse(
                log_c + s*log_mode_numbers,
                "estimated_log_modal_coeffs")
        estimate_error = cse(
                sqrt((estimated_log_modal_coeffs-weighted_log_modal_coeffs)**2),
                "estimate_error")

        log_modal_coeffs_corrected = log_modal_coeffs + estimate_error
        s_corrected = expt_op(mode_weights*log_modal_coeffs_corrected)

        return DecayInformation(
                indicator_modal_coeffs=indicator_modal_coeffs,
                decay_expt=s, c=c,
                log_modal_coeffs=log_modal_coeffs,
                weighted_log_modal_coeffs=weighted_log_modal_coeffs,
                estimated_log_modal_coeffs=estimated_log_modal_coeffs,
                decay_expt_corrected=s_corrected,
                )

    def bind_quantity(self, discr, quantity_name):
        baseline_squared = create_decay_baseline(discr)**2
        log_mode_numbers = numpy.log(create_mode_number_vector(discr, nonzero=True))
        mode_weights = create_mode_weight_vector(discr,
                DecayExponentOperator(
                    self.ignored_modes,
                    self.weight_mode))

        from hedge.optemplate import Field
        quantity = getattr(
                self.op_template_struct(Field("u")), quantity_name)

        compiled = discr.compile(quantity)

        if self.mode_processor is not None:
            discr.add_function("mode_processor", self.mode_processor.bind(discr))

        def apply(u, viscosity_scaling=None):
            kwargs = {}

            if viscosity_scaling is not None:
                kwargs["viscosity_scaling"] = viscosity_scaling
                if isinstance(viscosity_scaling, numpy.ndarray):
                    kwargs["max_viscosity_scaling"] = numpy.max(
                            viscosity_scaling[
                                ~numpy.isnan(viscosity_scaling)])
                else:
                    # FIXME
                    kwargs["max_viscosity_scaling"] = \
                            discr.nodewise_max(viscosity_scaling)
                    assert not numpy.isnan(kwargs["max_viscosity_scaling"])

            return compiled(
                    u=u,
                    baseline_squared=baseline_squared,
                    log_mode_numbers=log_mode_numbers,
                    mode_weights=mode_weights, **kwargs)

        return apply




class DecayGatingDiscontinuitySensorBase(
        DecayFitDiscontinuitySensorBase):
    def __init__(self,
            mode_processor,
            weight_mode,
            ignored_modes,
            correct_for_fit_error):
        DecayFitDiscontinuitySensorBase.__init__(
                self, mode_processor=mode_processor,
                weight_mode=weight_mode, ignored_modes=ignored_modes)
        self.correct_for_fit_error = correct_for_fit_error

    def op_template_struct(self, u=None):
        from hedge.optemplate import Field
        if u is None:
            u = Field("u")

        result = DecayFitDiscontinuitySensorBase\
                .op_template_struct(self, u)

        from pymbolic.primitives import IfPositive
        from hedge.optemplate.primitives import (
                CFunction, ScalarParameter)
        from math import pi
        from hedge.tools.symbolic import make_common_subexpression as cse

        if self.correct_for_fit_error:
            decay_expt = cse(result.decay_expt_corrected, "decay_expt")
        else:
            decay_expt = cse(result.decay_expt, "decay_expt")

        def flat_end_sin(x):
            return IfPositive(-pi/2-x,
                    -1, IfPositive(x-pi/2, 1, sin(x)))

        sin = CFunction("sin")
        isnan = CFunction("isnan")
        c_abs = CFunction("abs")

        visc_scale = Field("viscosity_scaling")

        result.sensor = IfPositive(c_abs(isnan(visc_scale)), 
                ScalarParameter("max_viscosity_scaling"),
                0.5*visc_scale
                * (1+flat_end_sin((decay_expt+2)*pi/2)))
        return result

    def bind(self, discr):
        return self.bind_quantity(discr, "sensor")
# }}}
# }}}

# {{{ mode processors
# {{{ elementwise code executor base class
class ElementwiseCodeExecutor(object):
    # {{{ CPU side
    @memoize_method
    def make_codepy_module(self, toolchain, dtype):
        from codepy.libraries import add_codepy
        toolchain = toolchain.copy()
        add_codepy(toolchain)

        from cgen import (Value, Include, Statement,
                Typedef, FunctionBody, FunctionDeclaration, Block, Const,
                Line, POD, Initializer, CustomLoop)
        S = Statement

        from codepy.bpl import BoostPythonModule
        mod = BoostPythonModule()

        mod.add_to_preamble([
            Include("vector"),
            Include("algorithm"),
            Include("hedge/base.hpp"),
            Include("hedge/volume_operators.hpp"),
            Include("boost/foreach.hpp"),
            Include("boost/numeric/ublas/io.hpp"),
            ]+self.get_cpu_extra_includes())

        mod.add_to_module([
            S("namespace ublas = boost::numeric::ublas"),
            S("using namespace hedge"),
            S("using namespace pyublas"),
            Line(),
            Typedef(POD(dtype, "value_type")),
            Line(),
            ])

        mod.add_function(FunctionBody(
            FunctionDeclaration(Value("void", "process_elements"), [
                Const(Value("uniform_element_ranges", "ers")),
                Const(Value("numpy_vector<value_type>", "field")),
                Value("numpy_vector<value_type>", "result"),
                ]+self.get_cpu_extra_parameter_declarators()),
            Block([
                Typedef(Value("numpy_vector<value_type>::iterator",
                    "it_type")),
                Typedef(Value("numpy_vector<value_type>::const_iterator",
                    "cit_type")),
                Line(),
                Initializer(Value("it_type", "result_it"),
                    "result.begin()"),
                Initializer(Value("cit_type", "field_it"),
                    "field.begin()"),
                Line() ]+self.get_cpu_extra_preamble()+[ Line(),
                CustomLoop(
                    "BOOST_FOREACH(const element_range er, ers)",
                    Block(self.get_cpu_per_element_code())
                    )
                ])))

        #print mod.generate()
        #toolchain = toolchain.copy()
        #toolchain.enable_debugging
        return mod.compile(toolchain)

    def bind_cpu(self, discr):
        def do(field):
            mod = self.make_codepy_module(discr.toolchain, field.dtype)

            out = discr.volume_empty(dtype=field.dtype)
            for eg in discr.element_groups:
                ldis = eg.local_discretization

                mod.process_elements(eg.ranges, field, out,
                        *self.get_cpu_extra_parameters(ldis))

            return out

        return do

    # }}}

    # {{{ CUDA side

    @memoize_method
    def make_cuda_kernel(self, discr, dtype, eg):
        given = discr.given
        ldis = eg.local_discretization

        microblocks_per_block = 1

        from cgen.cuda import CudaGlobal

        from cgen import (Module, Value, Include,
                Typedef, FunctionBody, FunctionDeclaration, Const,
                Line, POD, LiteralBlock,
                Define, Pointer)

        cmod = Module([
            Include("pycuda-helpers.hpp"),
            Line(),
            Typedef(POD(dtype, "value_type")),
            Line(),
            Define("DOFS_PER_EL", given.dofs_per_el()),
            Define("ALIGNED_DOFS_PER_MB", given.microblock.aligned_floats),
            Define("VERTICES_PER_EL", ldis.vertex_count()),
            Define("ELS_PER_MB", given.microblock.elements),
            Define("MBS_PER_BLOCK", microblocks_per_block),
            Line(),
            Define("DOF_IN_MB_IDX", "threadIdx.x"),
            Define("DOF_IN_EL_IDX", "(DOF_IN_MB_IDX-el_idx_in_mb*DOFS_PER_EL)"),
            Define("MB_IN_BLOCK_IDX", "threadIdx.y"),
            Define("BLOCK_IDX", "blockIdx.x"),
            Define("MB_NUMBER", "(BLOCK_IDX * MBS_PER_BLOCK + MB_IN_BLOCK_IDX)"),
            Define("BLOCK_DATA", "whole_block[MB_IN_BLOCK_IDX]")]
            + self.get_cuda_extra_preamble(discr, dtype, eg)
            + [FunctionBody(
            CudaGlobal(FunctionDeclaration(
                    Value("void", "elwise_kernel"), [
                    Pointer(Const(POD(dtype, "field"))),
                    Pointer(POD(dtype, "result")),
                    POD(numpy.uint32, "mb_count"),
                    ])),
                LiteralBlock("""
                int el_idx_in_mb = DOF_IN_MB_IDX / DOFS_PER_EL;

                if (MB_NUMBER >= mb_count)
                  return;

                int idx =  MB_NUMBER * ALIGNED_DOFS_PER_MB + DOF_IN_MB_IDX;
                int element_base_idx = ALIGNED_DOFS_PER_MB * MB_IN_BLOCK_IDX +
                    (DOF_IN_MB_IDX / DOFS_PER_EL) * DOFS_PER_EL;
                int dof_in_element = DOF_IN_MB_IDX-el_idx_in_mb*DOFS_PER_EL;

                __shared__ value_type whole_block[MBS_PER_BLOCK][ALIGNED_DOFS_PER_MB+1];
                int idx_in_block = ALIGNED_DOFS_PER_MB * MB_IN_BLOCK_IDX + DOF_IN_MB_IDX;
                BLOCK_DATA[idx_in_block] = field[idx];

                __syncthreads();

                %s

                result[idx] = node_result;
                """ % self.get_cuda_code(discr, dtype, eg)))
                ])


        if False:
            for i, l in enumerate(str(cmod).split("\n")):
                print i+1, l
            raw_input()

        from pycuda.compiler import SourceModule
        mod = SourceModule(
                cmod,
                keep="cuda_keep_kernels" in discr.debug,
                )
        func = mod.get_function("elwise_kernel")
        func.prepare(
            "PPI", block=(
                given.microblock.aligned_floats,
                microblocks_per_block, 1))

        mb_count = len(discr.blocks) * discr.given.microblocks_per_block
        grid_dim = (mb_count + microblocks_per_block - 1) \
                // microblocks_per_block

        from pytools import Record
        class KernelInfo(Record):
            pass

        return KernelInfo(
                func=func,
                grid_dim=grid_dim,
                mb_count=mb_count)

    def bind_cuda(self, discr):
        eg, = discr.element_groups
        knl_info = self.make_cuda_kernel(
                discr, discr.default_scalar_type, eg)

        def do(field):
            result = discr.volume_empty()
            knl_info.func.prepared_call((knl_info.grid_dim, 1),
                    field.gpudata,
                    result.gpudata,
                    knl_info.mb_count)

            return result

        return do

    def bind(self, discr):
        from hedge.backends.cuda import Discretization \
                as CUDADiscretization
        if isinstance(discr, CUDADiscretization):
            return self.bind_cuda(discr)
        else:
            return self.bind_cpu(discr)

    # }}}

    # {{{ user-overridable code

    def get_cpu_extra_includes(self):
        return []

    def get_cpu_extra_parameter_declarators(self):
        return []

    def get_cpu_extra_parameters(self, ldis):
        return []

    def get_cpu_extra_preamble(self):
        return []

    def get_cuda_extra_preamble(self, discr, dtype, eg):
        return []

    # }}}

# }}}




class SkylineModeProcessor(ElementwiseCodeExecutor):
    def get_cpu_extra_includes(self):
        from cgen import Include
        return [Include("boost/scoped_array.hpp")]

    def get_cpu_extra_parameter_declarators(self):
        from cgen import Value, POD
        return [
                Value("numpy_array<npy_uint32>", "mode_degrees"),
                POD(numpy.uint32, "max_degree")]

    @memoize_method
    def get_cpu_extra_parameters(self, ldis):
        return [
            numpy.array(
                [sum(mode_indices) for mode_indices in
                    ldis.generate_mode_identifiers()],
                dtype=numpy.uint32),
            ldis.order]

    def get_cpu_extra_preamble(self):
        from cgen import Initializer, Value, POD, Statement
        return [
                Initializer(Value("numpy_array<npy_uint32>::const_iterator",
                    "mode_degrees_iterator"),
                    "mode_degrees.begin()"),
                Initializer(POD(numpy.uint32, "mode_count"),
                    "mode_degrees.size()"),
                Statement("boost::scoped_array<value_type> reduced_modes"
                    "(new value_type[max_degree+1])"),
                ]

    def get_cpu_per_element_code(self):
        from cgen import (Value, Statement, Initializer, While,
                Comment, Block, For, Line, Pointer)
        S = Statement
        return [
                # assumes there is more than one coefficient
                Initializer(Value("cit_type", "el_modes"), "field_it+er.first"),

                Line(),
                Comment("zero out reduced_modes"),
                For("npy_uint32 mode_idx = 0",
                    "mode_idx < max_degree+1",
                    "++mode_idx",
                    S("reduced_modes[mode_idx] = 0")),

                Line(),
                Comment("gather modes by degree"),
                For("npy_uint32 mode_idx = 0",
                    "mode_idx < mode_count",
                    "++mode_idx",
                    S("reduced_modes[mode_degrees_iterator[mode_idx]]"
                        " += el_modes[mode_idx]")),

                Line(),
                Comment("perform skyline procedure"),
                Initializer(Pointer(Value("value_type", "start")),
                    "reduced_modes.get()"),
                Initializer(Pointer(Value("value_type", "end")),
                    "start+max_degree+1"),
                Initializer(Value("value_type", "cur_max"),
                    "std::max(*(end-1), *(end-2))"),

                Line(),
                While("end != start", Block([
                    S("--end"),
                    S("*end = std::max(cur_max, *end)"),
                    ])),

                Line(),
                Comment("scatter modes by degree"),
                Initializer(Value("it_type", "tgt_base"), "result_it+er.first"),
                For("npy_uint32 mode_idx = 0",
                    "mode_idx < mode_count",
                    "++mode_idx",
                    S("tgt_base[mode_idx] = "
                        "reduced_modes[mode_degrees_iterator[mode_idx]]")),
                ]

    def get_cuda_extra_preamble(self, discr, dtype, eg):
        from cgen import ArrayOf, Value, Initializer
        from cgen.cuda import CudaConstant

        ldis = eg.local_discretization
        mode_degrees = [sum(mode_indices) for mode_indices in
                ldis.generate_mode_identifiers()]

        return [Initializer(CudaConstant(
            ArrayOf(Value("unsigned", "mode_degrees"))),
            "{%s}" % ", ".join(str(i) for i in mode_degrees))
            ]

    def get_cuda_code(self, discr, dtype, eg):
        ldis = eg.local_discretization

        if dtype == numpy.float64:
            max_func = "fmax"
        elif dtype == numpy.float32:
            max_func = "fmaxf"
        else:
            raise TypeError("Could not find a maximum function"
                " due to unsupported field.dtype.")

        return """
                #define NUM_DEGREES %(num_degrees)d
                #define MAX_FUNC %(max_func)s
                #define RED_MODE_EL reduced_modes[MB_IN_BLOCK_IDX][el_idx_in_mb]

                __shared__ value_type reduced_modes[MBS_PER_BLOCK][ELS_PER_MB][NUM_DEGREES];
                if (DOF_IN_EL_IDX < NUM_DEGREES)
                    reduced_modes[MB_IN_BLOCK_IDX][el_idx_in_mb][DOF_IN_EL_IDX] = 0;

                if (dof_in_element == 0 && el_idx_in_mb < ELS_PER_MB)
                {
                  // gather modes by degree
                  for (int mode_idx = 0; mode_idx < DOFS_PER_EL; ++mode_idx)
                     RED_MODE_EL[mode_degrees[mode_idx]]
                       += BLOCK_DATA[
                         DOFS_PER_EL*el_idx_in_mb
                         + mode_idx];

                  // perform skyline procedure
                  value_type running_max = MAX_FUNC(
                    RED_MODE_EL[NUM_DEGREES-1],
                    RED_MODE_EL[NUM_DEGREES-2]);

                  int end_idx = NUM_DEGREES+1;
                  while (end_idx != 0)
                  {
                    --end_idx;
                    RED_MODE_EL[end_idx] = MAX_FUNC(
                      running_max,
                      RED_MODE_EL[end_idx]);
                  }
                }

                __syncthreads();

                value_type node_result = RED_MODE_EL
                  [mode_degrees[dof_in_element]];
                """ % {
                        "num_degrees": ldis.order+1,
                        "max_func": max_func,
                        }




class AveragingModeProcessor(ElementwiseCodeExecutor):
    def get_cpu_per_element_code(self):
        from cgen import (Value, Statement, Initializer, While, Block)
        S = Statement
        return [
                # assumes there is more than one coefficient
                Initializer(Value("cit_type", "start"), "field_it+er.first"),
                Initializer(Value("cit_type", "end"), "field_it+er.second"),
                Initializer(Value("it_type", "tgt"), "result_it+er.first"),

                Initializer(Value("cit_type", "cur"), "start"),
                While("cur != end",
                    Block([
                        Initializer(Value("cit_type", "avg_start"),
                            "std::max(start, cur-1)"),
                        Initializer(Value("cit_type", "avg_end"),
                            "std::min(end, cur+2)"),

                        S("*tgt++ = std::accumulate(avg_start, avg_end, value_type(0))"
                            "/std::distance(avg_start, avg_end)"),
                        S("++cur"),
                        ])
                    )
                ]

# }}}

# {{{ make h, h/n vector
def make_h_vector(discr):
    result = discr.volume_zeros(kind="numpy")

    for eg in discr.element_groups:
        for el, rng in zip(eg.members, eg.ranges):
            bbox_min, bbox_max = el.bounding_box(discr.mesh.points)
            h = numpy.max(bbox_max-bbox_min)
            result[rng] = h

    return discr.convert_volume(result, kind=discr.compute_kind)




def make_h_over_n_vector(discr):
    result = discr.volume_zeros(kind="numpy")

    for eg in discr.element_groups:
        for el, rng in zip(eg.members, eg.ranges):
            bbox_min, bbox_max = el.bounding_box(discr.mesh.points)
            h = numpy.max(bbox_max-bbox_min)
            result[rng] = h/(eg.local_discretization.order)

    return discr.convert_volume(result, kind=discr.compute_kind)

# }}}

# vim: foldmethod=marker
