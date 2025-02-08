# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional
import torch.fx as fx
import inspect

from .symbolic_constraints import SymbolicAlias

from ..compiler import builder, dispatch_codegen, kernel_codegen, host_codegen
from ..compiler.ir import Context, Operation
from .codegen import WaveEmitter
from .constraints import (
    Constraint,
    HardwareConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
    get_grid_shape,
)
from .codegen import WaveEmitter
from .expansion.expansion import expand_graph
from .promotion import promote_placeholders
from .hoisting import hoist_loop_invariant_ops
from .reuse_shared_allocs import reuse_shared_allocs
from .utils import (
    canonicalize_module,
    compile_to_vmfb,
    invoke_vmfb,
    safe_subs,
    remove_chained_getresult,
    remove_chained_extractslice,
    subs_idxc,
    delinearize_index,
    _write_file,
    initialize_iter_args,
    partial,
    print_trace,
)
from .minimize_global_loads import minimize_global_loads
from .decompose_reduce_ops import decompose_reduce_ops
from .decompose_vmma_ops import decompose_vmma_ops
from .barriers import add_shared_memory_barriers
from ..lang import Grid, IndexMapping
from ..lang.global_symbols import *
from ..ops import wave_ops
from ..ops.wave_ops import Reduction, CustomOp, get_custom, IterArg
from .index_sequence_analysis import (
    partition_ops_with_gpr_offsets,
    partition_strided_operators,
    set_node_indices,
    set_post_expansion_indices,
)
from .shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
    align_index_sizes,
)
from .scheduling.schedule import schedule_graph
from .._support.indexing import IndexingContext, IndexExpr
from .type_inference import infer_types
import iree.turbine.kernel.lang as tkl
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)
from .cache import is_cache_enabled, get_cache_manager, invoke_cached_kernel

import sympy

__all__ = ["wave", "wave_trace_only"]


def wave(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "LaunchableWave":
        return LaunchableWave(constraints, f.__name__, f)

    return decorator


def wave_trace_only(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "Callable[[], CapturedTrace]":
        wave = LaunchableWave(constraints, f.__name__, f)
        return wave._trace  # type: ignore

    return decorator


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: Optional[list[Constraint]],
        name: str,
        eager_function: Callable[[Any], Any],
    ):
        super().__init__(eager_function)

        self.constraints = constraints if constraints else []
        self.induction_vars: dict[CustomOp, IndexExpr] = {}
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

        self.grid_type = Grid[tuple(get_grid_shape(self.workgroup_constraints))]

    @property
    def workgroup_constraints(self) -> list[WorkgroupConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WorkgroupConstraint)
        ]

    @property
    def tiling_constraints(self) -> list[TilingConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, TilingConstraint)
        ]

    @property
    def wave_constraints(self) -> list[WaveConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WaveConstraint)
        ]

    @property
    def hardware_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, HardwareConstraint)
        ]

    @property
    def symbolic_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, SymbolicAlias)
        ]

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            # Get all explictly defined custom ops
            custom_ops: dict[str, wave_ops.CustomOp] = {
                cls.tkw_op_name: cls
                for _, cls in inspect.getmembers(wave_ops, inspect.isclass)
                if issubclass(cls, wave_ops.CustomOp) and hasattr(cls, "tkw_op_name")
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)

        return trace

    def create_induction_vars(self, trace: CapturedTrace) -> None:
        """
        Creates induction variables for all the reductions in the graph
        and associates tiling constraints all the reduction dimensions
        with the appropriate induction variables.

        """

        def is_reduction(node: fx.Node):
            custom = get_custom(node)
            return isinstance(custom, Reduction)

        reduction_nodes = trace.walk(is_reduction)
        for node in reduction_nodes:
            custom = get_custom(node)
            self.induction_vars[custom] = tkl.IndexSymbol(
                "$ARG" + str(custom.axis), integer=True, nonnegative=True
            )
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == custom.axis:
                    tiling_constraint.induction_var = self.induction_vars[custom]

    def initialize_wave_constraints(self, trace: CapturedTrace) -> None:
        """
        For each wave constraint, determines the appropriate wave id by looking
        for workgroup constraints along the same dimension and using information
        from the hardware constraints.

        """

        hardware_constraint = self.hardware_constraints[0]
        for wave_constraint in self.wave_constraints:
            for workgroup_constraint in self.workgroup_constraints:
                # The wave_id is the same as the thread_id, with the exception
                # of wave_id[0] = thread_id[0] / threads_per_wave. This is
                # a convention that we adopt.
                if wave_constraint.dim == workgroup_constraint.dim:
                    wave_constraint.wave_id = (
                        hardware_constraint.get_thread_id_from_workgroup_dim(
                            workgroup_constraint.workgroup_dim
                        )
                    )
                    if workgroup_constraint.workgroup_dim == 0:
                        wave_constraint.wave_id = sympy.floor(
                            wave_constraint.wave_id
                            / hardware_constraint.threads_per_wave
                        )

    def initialize_reductions(self, trace: CapturedTrace) -> None:
        """
        For each reduction, initializes the reduction count by looking at the
        tiling constraints associated with the reduction.

        """
        is_reduction = lambda node: isinstance(get_custom(node), Reduction)
        for reduction in trace.walk(is_reduction):
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == get_custom(reduction).axis:
                    reduction.count = subs_idxc(tiling_constraint.count)

    def get_workgroup_dims(self) -> list[int]:
        """
        Returns the workgroup dimensions that are not aliased.
        """
        # Ignore aliased variables. They will be handled separately.
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        workgroup_dims = [
            x for x in self.workgroup_constraints if x.dim not in aliased_dims
        ]
        return workgroup_dims

    def update_aliased_workgroup_constraints(
        self, workgroup_dims: dict[int, int]
    ) -> None:
        """
        This function updates the wg_dim for aliased workgroup constraints.
        """
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        # Update the workgroup constraints for aliases sources.
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliased_dims:
                constraint.wg_dim = workgroup_dims[constraint.workgroup_dim].wg_dim

    def initialize_workgroup_constraints(self, trace: CapturedTrace) -> None:
        """
        For kernels that distribute more than three dimensions among workgroups,
        we need to update the workgroup constraints for dimensions >= 2
        with the appropriate workgroup index.
        """

        workgroup_dims = self.get_workgroup_dims()
        # Filter to WG2 and above.
        dims_to_delinearize = [x for x in workgroup_dims if x.workgroup_dim >= 2]
        if all(x.workgroup_dim <= 2 for x in dims_to_delinearize):
            return
        # Only take account primary dim for delinearize shape.
        shape = [subs_idxc(x.count) for x in dims_to_delinearize if x.primary]
        new_workgroup_dims = delinearize_index(WORKGROUP_2, shape)
        for delinearize_dim in dims_to_delinearize:
            delinearize_dim.wg_dim = new_workgroup_dims[
                delinearize_dim.workgroup_dim - 2
            ]
        self.update_aliased_workgroup_constraints(workgroup_dims)

    def initialize_symbolic_constraints(self, trace: CapturedTrace) -> None:
        """
        For each symbolic constraint, create new constraints for the
        related symbolic values with appropriate substitutions.
        """
        new_wg_constraints, new_wave_constraints, new_tiling_constraints = [], [], []
        for symbolic_constraint in self.symbolic_constraints:
            new_wg_constraints += symbolic_constraint.create_new_constraints(
                self.workgroup_constraints
            )
            new_wave_constraints += symbolic_constraint.create_new_constraints(
                self.wave_constraints
            )
            new_tiling_constraints += symbolic_constraint.create_new_constraints(
                self.tiling_constraints
            )
        # Remove wave constraints with same tile size as workgroup constraints
        for wave_constraint in new_wave_constraints:
            for workgroup_constraint in new_wg_constraints:
                if (
                    wave_constraint.dim == workgroup_constraint.dim
                    and wave_constraint.tile_size == workgroup_constraint.tile_size
                ):
                    new_wave_constraints.remove(wave_constraint)
        self.constraints += (
            new_wg_constraints + new_wave_constraints + new_tiling_constraints
        )
        idxc = IndexingContext.current()
        for constraint in self.symbolic_constraints:
            if subs_idxc(constraint.target).is_number:
                idxc._bind_symbol(
                    constraint.source,
                    subs_idxc(constraint.source_to_target(constraint.target)),
                )

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ) -> tuple[
        builder.ModuleBuilder,
        CapturedTrace,
        dispatch_codegen.StreamExecutable,
        kernel_codegen.KernelSignature,
        str,
    ]:
        compile_config = kwargs.get("compile_config", {})
        print_ir_after = compile_config.get("print_ir_after", [])
        print_ir_before = compile_config.get("print_ir_before", [])
        if compile_config.get("print_trace_begin", False):
            print(f"\n***Tracing kernel {self._name}***")

        trace = self._trace()
        if (
            "all" in print_ir_after
            or "all" in print_ir_before
            or "trace" in print_ir_after
            or "first" in print_ir_before
        ):
            print(f"***After trace/Before first pass***\n")
            print_trace(trace)

        idxc = IndexingContext.current()

        def finalize_indices():
            idxc.finalize()

        def substitute_vector_shapes():
            self.hardware_constraints[0].subs_vector_shapes(idxc.subs)

        graph_passes = [
            partial(initialize_iter_args, trace),
            partial(self.create_induction_vars, trace),
            partial(self.initialize_wave_constraints, trace),
            partial(self.initialize_reductions, trace),
            partial(self.initialize_symbolic_constraints, trace),
            partial(self.initialize_workgroup_constraints, trace),
            finalize_indices,
            substitute_vector_shapes,
            partial(infer_types, trace),
            partial(promote_placeholders, trace, self.constraints),
            partial(set_node_indices, trace, self.constraints),
            partial(expand_graph, trace, self.constraints),
            partial(set_post_expansion_indices, trace, self.constraints),
            partial(remove_chained_getresult, trace),
        ]

        # Optimizations.
        graph_passes += [
            partial(decompose_vmma_ops, trace, self.constraints),
            partial(hoist_loop_invariant_ops, trace, self.constraints),
            partial(minimize_global_loads, trace, self.constraints),
            partial(reuse_shared_allocs, trace),
            partial(apply_shared_memory_indexing_corrections, trace, self.constraints),
        ]

        # Partition strided operators.
        graph_passes += [
            partial(partition_ops_with_gpr_offsets, trace, self.constraints),
            partial(partition_strided_operators, trace, self.constraints),
            partial(remove_chained_extractslice, trace),
        ]

        graph_passes += [
            partial(decompose_reduce_ops, trace, self.constraints, idxc.subs)
        ]

        # Schedule the reduction ops.
        # Scheduling should always be used with use_scheduling_barriers=True,
        # as this is the only way we can ensure that LLVM enforces our desired schedule.
        # However, due a bug in LLVM, you will need to patch your local LLVM repo
        # with the following commit: https://github.com/kerbowa/llvm-project/commit/ee52732cddae42deed2e3387a83b20ec05860b4e
        # Specifically:
        # git fetch https://github.com/kerbowa/llvm-project.git ee52732cddae42deed2e3387a83b20ec05860b4e
        # git cherry-pick ee52732cddae42deed2e3387a83b20ec05860b4e
        # [Manually resolve conflicts consistent with the PR]
        if kwargs.get("schedule", False):
            use_scheduling_barriers = kwargs.get("use_scheduling_barriers", False)
            graph_passes.append(
                partial(
                    schedule_graph, trace, self.constraints, use_scheduling_barriers
                )
            )

        graph_passes += [
            # Align sizes to WG/Tile sizes
            # This pass changes indexing keys, which can interfere with other passes,
            # so it should be called close to the end of pipeline.
            partial(align_index_sizes, trace, self.constraints),
            partial(add_shared_memory_barriers, trace),
        ]

        for p in graph_passes:
            if "all" in print_ir_before or p.__name__ in print_ir_before:
                print(f"***Before {p.__name__}***\n")
                print_trace(trace)
            try:
                p()
            except Exception:
                print(f"Error in pass: {p.__name__}\n")
                print_trace(trace)
                raise
            if "all" in print_ir_after or p.__name__ in print_ir_after:
                print(f"***After {p.__name__}***\n")
                print_trace(trace)

        if "all" in print_ir_after or "last" in print_ir_after:
            # Take advantage of Python leaking loop variables
            print(f"***After final pass {p.__name__}***\n")
            print_trace(trace)

        # Determine grid shape.
        self.grid_type.dims = [1, 1, 1]
        max_workgroup_dim = 2
        aliases = [x.source for x in self.constraints if isinstance(x, SymbolicAlias)]
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliases:
                continue
            if not constraint.primary:
                continue
            dim = (
                constraint.workgroup_dim
                if constraint.workgroup_dim < max_workgroup_dim
                else max_workgroup_dim
            )
            self.grid_type.dims[dim] *= safe_subs(constraint.count, idxc.subs)
        grid = self.grid_type
        if compile_config.get("print_grid", False):
            print(f"Grid: {grid}")

        root_graph = trace.get_root_graph()
        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(root_graph)
        dynamic_symbols = kwargs.get("dynamic_symbols", [])
        kernel_sig.add_from_dynamic_symbols(dynamic_symbols)
        kernel_sig.add_grid(self.grid_type)
        kernel_sig.determine_input_output_buffers(root_graph)
        if compile_config.get("print_signature", False):
            print(kernel_sig)

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        workgroup_size = self.hardware_constraints[0].threads_per_block
        subgroup_size = self.hardware_constraints[0].threads_per_wave

        # Setup LLVM func compilation configs.
        llvm_func_config = {}
        denorm_fp_math_f32 = compile_config.get("denorm_fp_math_f32", None)
        if denorm_fp_math_f32 != None:
            llvm_func_config["denormal-fp-math-f32"] = denorm_fp_math_f32

        waves_per_eu = compile_config.get("waves_per_eu", None)
        if waves_per_eu != None:
            llvm_func_config["amdgpu-waves-per-eu"] = waves_per_eu

        dispatch_entrypoint = exe.define_entrypoint(
            entrypoint_name,
            kernel_sig,
            grid,
            workgroup_size,
            subgroup_size,
            dynamic_symbols,
            llvm_func_config,
        )

        emitter = WaveEmitter(
            dispatch_entrypoint, trace, self.constraints, dynamic_symbols, kwargs
        )
        try:
            emitter.emit(trace.get_root_graph())
        except:
            print("Error in emitter")
            asm = mb.module_op.get_asm()
            print(asm)
            raise
        emitter.finish()

        if kwargs.get("canonicalize", False):
            canonicalize_module(mb.module_op)

        return mb, trace, exe, kernel_sig, entrypoint_name

    def test_execute(self, args, kwargs):
        run = kwargs.get("run", False)
        run_bench = kwargs.get("run_bench", False)
        create_vmfb_file = kwargs.get("create_vmfb_file", None)
        dynamic_symbols_map = kwargs.get("dynamic_symbols_map", {})
        dynamic_symbols = kwargs.get("dynamic_symbols", [])
        config = kwargs.get("run_config", None)
        use_scheduling = kwargs.get("schedule", False)
        use_scheduling_barriers = kwargs.get("use_scheduling_barriers", False)

        # Get cached kernel when available.
        cache_enabled = is_cache_enabled()
        if cache_enabled:
            cache_manager = get_cache_manager()
            # TODO: Move use_scheduling, use_scheduling_barriers, etc. into the config so everything is contained there.
            kernel_hash = cache_manager.get_hash(
                self.constraints,
                self._f,
                IndexingContext.current().subs,
                dynamic_symbols,
                config,
                use_scheduling=use_scheduling,
                use_scheduling_barriers=use_scheduling_barriers,
                run_bench=run_bench,
            )
            cached_kernel = cache_manager.load_kernel(kernel_hash)
            if cached_kernel and (run or run_bench):
                invoke_cached_kernel(
                    cached_kernel,
                    args,
                    config,
                    dynamic_symbols,
                    dynamic_symbols_map,
                    run,
                    run_bench,
                )
                return cached_kernel

        # Recompile kernel from scratch if not found in cache.
        (
            mb,
            graph,
            exe,
            kernel_sig,
            entrypoint_name,
        ) = self._trace_and_get_kernel_signature(args, kwargs)

        if run or run_bench or create_vmfb_file:
            host_codegen.isolated_test_call(
                mb, exe, kernel_sig, entrypoint_name, dynamic_symbols
            )
            asm = mb.module_op.get_asm()

            kernel_inputs = []
            kernel_outputs = []
            for arg, b in zip(args, kernel_sig.kernel_buffer_bindings):
                usage = b.kernel_buffer_type.usage
                if usage == kernel_codegen.KernelBufferUsage.INPUT:
                    kernel_inputs.append(arg)

                if usage == kernel_codegen.KernelBufferUsage.OUTPUT:
                    kernel_outputs.append(arg)

            dynamic_symbols_map = kwargs.get("dynamic_symbols_map", {})
            kernel_dynamic_dims = []
            if dynamic_symbols:
                kernel_dynamic_dims = dynamic_symbols_map.values()

            if not config:
                raise ValueError("no config provided")

            compiled_wave_vmfb = compile_to_vmfb(asm, config, run_bench)
            if create_vmfb_file is not None:
                _write_file(create_vmfb_file, "wb", compiled_wave_vmfb)

            kernel_usages = [
                binding.kernel_buffer_type.usage
                for binding in kernel_sig.kernel_buffer_bindings
            ]

            if cache_enabled:
                cache_manager.store_kernel(
                    compiled_wave_vmfb,
                    kernel_usages,
                    mb.module_op.get_asm(),
                    kernel_hash,
                )

            invoke_vmfb(
                compiled_wave_vmfb,
                "isolated_benchmark",
                config,
                kernel_inputs,
                kernel_outputs,
                kernel_dynamic_dims,
                run,
                run_bench,
                inplace=True,
            )

        return mb

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"
