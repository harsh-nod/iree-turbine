# RUN: python %s | FileCheck %s

import logging
import unittest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion import expand_graph
from iree.turbine.kernel.wave.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import run_test, print_trace
from iree.turbine.kernel.wave.constraints import MMAType


# Input sizes
B = tkl.sym.B
M = tkl.sym.M
M0 = tkl.sym.M0
N = tkl.sym.N
K1 = tkl.sym.K1
K2 = tkl.sym.K2
K3 = tkl.sym.K3
# Workgroup tile sizes
BLOCK_B = tkl.sym.BLOCK_B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_M0 = tkl.sym.BLOCK_M0
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K1 = tkl.sym.BLOCK_K1
BLOCK_K2 = tkl.sym.BLOCK_K2
BLOCK_K3 = tkl.sym.BLOCK_K3
# Induction variables
ARG_K1 = tkl.sym.ARG_K1
ARG_K2 = tkl.sym.ARG_K2
ARG_K3 = tkl.sym.ARG_K3

# Address space (for GPU, shared(1) or global(0))
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
# Other hyperparameters
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD


@tkw.wave_trace_only()
def nested_gemm(
    q: tkl.Memory[
        B, M0, K1, ADDRESS_SPACE, tkl.f16
    ],  # TODO: The compiler should rename these arguments.
    k: tkl.Memory[
        B, K3, K1, ADDRESS_SPACE, tkl.f16
    ],  # TODO: The compiler should rename these arguments.
    v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

    # In the outer reduction, K2 is the reduction dimension.
    @tkw.reduction(K2, init_args=[c_reg])
    def repeat(acc: tkl.Register[B, M, N, tkl.f32]) -> tkl.Register[B, M, N, tkl.f32]:
        imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
        # TODO: The compiler should insert these renames.
        imm_reg = tkw.rename(imm_reg, {K2: K3, M: M0})

        # In the inner reduction, K1 is the reduction dimension and K2 is a parallel dimension.
        # In this example, K2 is not distributed across workgroups or waves, but in general
        # it could be in which case we would have a tiling, workgroup and wave constraint on K2.
        # The vector_shape for K2 within this reduction in general will be different from the vector_shape
        # for K2 outside this reduction, allowing us to treat K2 inside the reduction as a different
        # variable as K2 outside. So before and after the reduction, we introduce a rename.
        @tkw.reduction(K1, init_args=[imm_reg])
        def inner_loop(
            inner_acc: tkl.Register[B, K3, M0, tkl.f32]
        ) -> tkl.Register[B, K3, M0, tkl.f32]:
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            inner_acc = tkw.mma(k_reg, q_reg, inner_acc)
            return inner_acc

        # TODO: The compiler should insert these renames.
        inner_loop = tkw.rename(inner_loop, {K3: K2, M0: M})
        imm_t = tkw.permute(inner_loop, target_shape=[B, M, K2])
        imm_f16 = tkw.cast(imm_t, tkl.f16)
        v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(imm_f16, v_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)


@run_test
def test_nested_gemm():
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(M0, BLOCK_M0, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K3, BLOCK_K3, ARG_K3)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2, ARG_K2)]
    constraints += [tkw.TilingConstraint(K1, BLOCK_K1, ARG_K1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=MMAType.F32_16x16x16_F16,
            vector_shapes={B: 0},
        )
    ]

    shape = [8, 128, 64, 256, 128]
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_M: 64,
        # BLOCK_M0 is a clone of BLOCK_M
        BLOCK_M0: 64,
        BLOCK_N: 64,
        BLOCK_K1: 32,
        BLOCK_K2: 32,
        # BLOCK_K3 is a clone of BLOCK_K2
        BLOCK_K3: 32,
        BLOCK_B: 1,
        B: shape[0],
        M: shape[1],
        # M0 is a clone of M
        M0: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
        # K3 is a clone of K2
        K3: shape[4],
    }
    with tk.gen.TestLaunchContext(hyperparams, canonicalize=True):
        graph = nested_gemm()
        IndexingContext.current().finalize()
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        breakpoint()
