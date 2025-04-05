# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils.general_utils import (
    torch_dtype_to_wave,
)
from .attention_common import *
import math
import torch
from typing import Optional


def get_gqa_bshd_attention_kernel(
    shape: AttentionShape,
    mfma_variant: tuple[MMAType, MMAType],
    input_dtype: Optional[torch.dtype] = torch.float16,
    output_dtype: Optional[torch.dtype] = torch.float32,
    is_causal: Optional[bool] = False,
    layer_scaling: Optional[float] = None,
    sliding_window_size: Optional[int] = -1,
):

    if sliding_window_size > 0 and not is_causal:
        raise NotImplementedError(
            "Sliding window is only supported for causal attention."
        )

    # Determine dtype of operands.
    wave_input_dtype = torch_dtype_to_wave(input_dtype)
    wave_output_dtype = torch_dtype_to_wave(output_dtype)

    LOG2E = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    layer_scaling = (layer_scaling or dk_sqrt) * LOG2E

    # Input sizes
    # BxS_QxHxD x BxS_KVxH_KVxD
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    H = tkl.sym.H
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_H = tkl.sym.BLOCK_H
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 3)]
    constraints += [tkw.WorkgroupConstraint(H_KV, BLOCK_H, 3, primary=False)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant[1],
            vector_shapes={B: 0, H: 0, H_KV: 0, M: Mvec, N: Nvec},
        )
    ]

    head_ratio = shape.num_query_heads // shape.num_kv_heads
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N: k, M: l},
        outputs={B: i, M: l, H: j, N: k},
    )
    q_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, M: k, K1: l},
        outputs={B: i, H: j, M: k, K1: l},
    )
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, K2: k, K1: l},
        outputs={B: i, H_KV: j, K2: k, K1: l},
    )
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, N: k, K2: l},
        outputs={B: i, H_KV: j, N: k, K2: l},
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, H, K1, GLOBAL_ADDRESS_SPACE, wave_input_dtype],
        k: tkl.Memory[B, K2, H_KV, K1, ADDRESS_SPACE, wave_input_dtype],
        v: tkl.Memory[B, K2, H_KV, N, ADDRESS_SPACE, wave_input_dtype],
        c: tkl.Memory[B, M, H, N, GLOBAL_ADDRESS_SPACE, wave_output_dtype],
    ):

        qkv_scaling = tkl.Register[B, H, M, K1, tkl.f16](dk_sqrt * layer_scaling)
        c_reg = tkl.Register[B, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, H, M, tkl.f32](-1e6)
        sliding_window = tkl.Register[M, K2, tkl.i64](sliding_window_size)
        ZEROF = tkl.Register[M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, M, tkl.f32],
            partial_sum: tkl.Register[B, H, M, tkl.f32],
            acc: tkl.Register[B, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, mapping=q_mapping)
            q_reg *= qkv_scaling
            k_reg = tkw.read(k, mapping=k_mapping)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, M, K2])
            k2_index = tkw.self_index(K2, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x < K2)
            mask = tkw.broadcast(mask, target_shape=[M, K2])
            if is_causal:
                m_index = tkw.self_index(M, tkl.i32)
                m_index = tkw.broadcast(m_index, target_shape=[M, K2])
                mask = (m_index >= k2_index) & mask
                if sliding_window_size > 0:
                    mask = (m_index - k2_index <= sliding_window) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, mapping=v_mapping)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_H: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape.num_seqs,
        H: shape.num_query_heads,
        H_KV: shape.num_kv_heads,
        M: shape.query_seq_len,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: shape.kv_seq_len,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map
