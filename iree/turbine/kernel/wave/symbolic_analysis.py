# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ops.wave_ops import (
    Write,
    ExtractSlice,
    get_custom,
    Reduction,
    MMA,
    Placeholder,
    IterArg,
    Allocate,
)
from .constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
    WaveConstraint,
)
from .._support.tracing import CapturedTrace, IndexingContext
from .._support.indexing import IndexSymbol, IndexSequence
from ..lang.global_symbols import *
from .utils import (
    simplify_index,
    get_mma_dimensional_mapping,
    get_hardware_constraint,
    subs_idxc,
    specialize_index_sequence,
)
import torch.fx as fx
import numpy as np
from functools import partial
from typing import Sequence
from ...support.logging import get_logger
from collections import defaultdict

logger = get_logger("turbine.wave.symbolic_analysis")


def identify_symbols_to_clone(
    trace: CapturedTrace, constraints: Sequence[Constraint]
) -> tuple[set[IndexSymbol], dict[IndexSymbol, list[fx.Node]]]:
    """
    A symbol is a candidate for cloning if:
        1. It is used in multiple MMA operations.
        2. Its vector shape is not the same in all MMA operations where it is used (as this
           would require different dim scalings along the same dimension) or
           its used as a different operand in the MMA operations (such as lhs in one, rhs in other
           as this would require different indices along the same dimension).
    """

    def is_mma(node):
        return isinstance(get_custom(node), MMA)

    mma_nodes = trace.walk(is_mma)

    mapping: dict[IndexSymbol, list[int]] = defaultdict(set)
    symbols_to_ops: dict[IndexSymbol, list[fx.Node]] = defaultdict(set)
    vector_shapes: dict[IndexSymbol, list[int]] = defaultdict(set)
    hardware_constraint = get_hardware_constraint(constraints)
    for node in mma_nodes:
        custom: MMA = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        lhs_shape = custom.lhs_type.symbolic_shape
        rhs_shape = custom.rhs_type.symbolic_shape
        acc_shape = custom.acc_type.symbolic_shape
        k = ((set(lhs_shape) & set(rhs_shape)) - set(acc_shape)).pop()
        M, N, K = hardware_constraint.mma_matrix_shapes
        vector_shapes[m].add(M)
        vector_shapes[n].add(N)
        vector_shapes[k].add(K)
        mapping[m].add(0)
        mapping[n].add(1)
        mapping[k].add(2)
        symbols_to_ops[m].add(node)
        symbols_to_ops[n].add(node)
        symbols_to_ops[k].add(node)

    symbols_to_clone: list[IndexSymbol] = []
    for symbol in vector_shapes:
        if len(vector_shapes[symbol]) > 1 or len(mapping[symbol]) > 1:
            logger.info(f"Symbol {symbol} is a candidate for cloning.")
            symbols_to_clone.append(symbol)

    symbols_to_clone = set(symbols_to_clone)
    return symbols_to_clone, {
        symbol: symbols_to_ops[symbol] for symbol in symbols_to_clone
    }


def clone_symbols(
    symbols_to_clone: set[IndexSymbol], symbols_to_ops: dict[IndexSymbol, list[fx.Node]]
) -> dict[IndexSymbol, IndexSymbol]:
    """
    Symbols are cloned by adding a suffix to their name.
    We create a new symbol for each operation where the symbol is used.
    This is conservative and may result in more clones than necessary.
    """
    cloned_symbols: dict[IndexSymbol, list[IndexSymbol]] = defaultdict(list)
    for symbol in symbols_to_clone:
        for i in range(len(symbols_to_ops[symbol]) - 1):
            cloned_symbols[symbol].append(index_symbol(f"${symbol.name}_clone_{i}"))
    return cloned_symbols


def replace_placeholders(
    trace: CapturedTrace, cloned_symbols: dict[IndexSymbol, IndexSymbol]
):
    def is_placeholder(node: fx.Node):
        return isinstance(get_custom(node), Placeholder) and not isinstance(
            get_custom(node), IterArg
        )

    placeholders = trace.walk(is_placeholder)
    for placeholder in placeholders:
        custom = get_custom(placeholder)
        symbolic_shape = custom._type.symbolic_shape
        new_symbolic_shape = [
            x if x not in cloned_symbols else cloned_symbols[x] for x in symbolic_shape
        ]
        custom._type.symbolic_shape = new_symbolic_shape
        print(
            f"Replaced {symbolic_shape} with {new_symbolic_shape} in {custom.fx_node}."
        )


def update_constraints(
    constraints: Sequence[Constraint], cloned_symbols: dict[IndexSymbol, IndexSymbol]
) -> Sequence[Constraint]:
    for symbol, cloned in cloned_symbols.items():
        for new_symbol in cloned:
            for constraint in constraints:
                if (
                    isinstance(constraint, WorkgroupConstraint)
                    and constraint.dim == symbol
                ):
                    constraints.append(
                        WorkgroupConstraint(
                            new_symbol,
                            constraint.tile_size,
                            constraint.workgroup_dim,
                        )
                    )
                elif (
                    isinstance(constraint, TilingConstraint)
                    and constraint.dim == symbol
                ):
                    constraints.append(
                        TilingConstraint(
                            new_symbol,
                            constraint.tile_size,
                            constraint.induction_var,
                        )
                    )
                elif (
                    isinstance(constraint, WaveConstraint) and constraint.dim == symbol
                ):
                    constraints.append(
                        WaveConstraint(
                            new_symbol,
                            constraint.tile_size,
                            constraint.wave_id,
                        )
                    )
    return constraints


def add_renames(
    cloned_symbols: dict[IndexSymbol, IndexSymbol],
    symbols_to_ops: dict[IndexSymbol, list[fx.Node]],
):
    """
    Renames are added to init_args and results of MMA reductions,
    starting from the innermost reduction going outward.
    """
    # Sort the MMA operations from innermost to outermost.
    breakpoint()


def partition_symbols(
    trace: CapturedTrace, constraints: Sequence[Constraint]
) -> dict[IndexSymbol, IndexSymbol]:
    # Identify the symbols that need to be partitioned.
    symbols_to_clone, symbols_to_ops = identify_symbols_to_clone(trace, constraints)
    if not symbols_to_clone:
        return {}
    # Clone the symbols.
    cloned_symbols = clone_symbols(symbols_to_clone, symbols_to_ops)
    # Add constraints and update indexing context.
    constraints = update_constraints(constraints, cloned_symbols)
    idxc = IndexingContext.current()
    for symbol, cloned_list in cloned_symbols.items():
        for cloned in cloned_list:
            idxc.subs.update({cloned: idxc.subs[symbol]})
    # Sort the mma ops from innermost to outermost and get the corresponding reductions.
    mma_ops = set()
    for symbol in symbols_to_ops:
        mma_ops = mma_ops.union(symbols_to_ops[symbol])
    breakpoint()
    for reduction in reductions:
        # Add renames.
        add_renames(cloned_symbols, symbols_to_ops)
        # Replace instances in function arguments.
        replace_placeholders(trace, cloned_symbols)


def combine_symbols(
    cloned_symbols: dict[IndexSymbol, IndexSymbol],
    trace: CapturedTrace,
    constraints: Sequence[Constraint],
):
    # Replace instances in function arguments.
    # Remove constraints and update indexing context.
    # Remove renames.
    # Update indices.
    pass
