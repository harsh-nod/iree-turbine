import torch
import json
import math
import os
import logging
import datetime
import random
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.utils.general_utils import get_default_scheduling_params
from iree.turbine.kernel.wave.utils.run_utils import set_default_run_config
from iree.turbine.kernel.wave.utils.torch_utils import device_randn, device_zeros
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.optimize_schedule import (
    ScheduleOptimizer,
    OptimizationAlgorithm,
    OptimizationResult,
)
from iree.turbine.kernel.wave.scheduling.verifier import ScheduleModifier
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType

try:
    from rpdTracerControl import rpdTracerControl
except ImportError:
    print("rpdTracerControl not found, falling back to CUDA events for timing")
    RPD_AVAILABLE = False
else:
    RPD_AVAILABLE = True


@dataclass
class AttentionConfig:
    """Configuration for attention kernel tuning."""

    batch_size: int
    num_heads: int
    seq_len_q: int
    seq_len_k: int
    head_dim: int
    head_dim_kv: int
    mfma_variant: Tuple[MMAType, MMAType]
    enable_scheduling: SchedulingType = SchedulingType.MODULO
    dynamic_dims: bool = False
    num_warmup: int = 10
    num_iterations: int = 100


def get_attention_shape(config: AttentionConfig) -> AttentionShape:
    """Convert config to AttentionShape."""
    return AttentionShape(
        num_query_heads=config.num_heads,
        num_kv_heads=config.num_heads,
        query_seq_len=config.seq_len_q,
        head_size_kv=config.head_dim_kv,
        head_size=config.head_dim,
        kv_seq_len=config.seq_len_k,
    )


@dataclass
class TimingResult:
    """Results from timing a kernel execution."""

    latency_ms: float
    throughput_tflops: float
    trace_file: Optional[str] = None


def calculate_throughput(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    latency_seconds: float,
) -> float:
    """Calculate theoretical throughput in TFLOPs.

    Args:
        batch_size: Batch dimension
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        head_dim: Head dimension
        latency_seconds: Execution time in seconds

    Returns:
        Throughput in TFLOPs
    """
    # For attention, we have:
    # 1. QK matmul: 2 * B * H * M * N * K operations
    # 2. Softmax: 2 * B * H * M * N operations
    # 3. V matmul: 2 * B * H * M * N * K operations
    # Total: 4 * B * H * M * N * K + 2 * B * H * M * N operations
    total_ops = (
        4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
        + 2 * batch_size * num_heads * seq_len_q * seq_len_k
    )
    return total_ops / (latency_seconds * 1e12)  # Convert to TFLOPs


def measure_with_rpd(
    kernel_fn,
    *args,
    num_warmup: int,
    num_iterations: int,
    output_filename: str,
    config: AttentionConfig,
) -> TimingResult:
    """Measure kernel performance using RPD tracer.

    Args:
        kernel_fn: The kernel function to measure
        *args: Arguments to pass to the kernel
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        output_filename: Path to save RPD trace
        config: Attention configuration for throughput calculation

    Returns:
        TimingResult with latency and throughput
    """
    if not RPD_AVAILABLE:
        raise RuntimeError("RPD tracer not available")

    # Warmup
    for _ in range(num_warmup):
        _ = kernel_fn(*args)

    # Synchronize GPU
    torch.cuda.synchronize()

    # Initialize RPD tracer
    rpdTracerControl.setFilename(name=output_filename, append=False)
    tracer = rpdTracerControl()
    tracer.start()

    # Benchmark with profiling
    for _ in range(num_iterations):
        _ = kernel_fn(*args)
    torch.cuda.synchronize()

    # Stop profiling and get results
    tracer.stop()
    tracer.flush()

    # Calculate statistics from RPD trace
    conn = sqlite3.connect(output_filename)
    df_top = pd.read_sql_query("SELECT * from top", conn)
    conn.close()

    avg_time = df_top["Ave_us"][0] / 1e6  # Convert to seconds
    throughput = calculate_throughput(
        config.batch_size,
        config.num_heads,
        config.seq_len_q,
        config.seq_len_k,
        config.head_dim,
        avg_time,
    )

    return TimingResult(
        latency_ms=avg_time * 1000,
        throughput_tflops=throughput,
        trace_file=output_filename,
    )


def measure_with_cuda_events(
    kernel_fn, *args, num_warmup: int, num_iterations: int, config: AttentionConfig
) -> TimingResult:
    """Measure kernel performance using CUDA events.

    Args:
        kernel_fn: The kernel function to measure
        *args: Arguments to pass to the kernel
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        config: Attention configuration for throughput calculation

    Returns:
        TimingResult with latency and throughput
    """
    # Warmup
    for _ in range(num_warmup):
        _ = kernel_fn(*args)

    # Synchronize GPU
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        _ = kernel_fn(*args)
    end_event.record()

    # Synchronize and get timing
    torch.cuda.synchronize()
    latency = (
        start_event.elapsed_time(end_event) / num_iterations / 1000.0
    )  # Convert to seconds

    throughput = calculate_throughput(
        config.batch_size,
        config.num_heads,
        config.seq_len_q,
        config.seq_len_k,
        config.head_dim,
        latency,
    )

    return TimingResult(latency_ms=latency * 1000, throughput_tflops=throughput)


def measure_attention_latency(
    config: AttentionConfig,
    schedule: Optional[Dict] = None,
    log_dir: Optional[Path] = None,
    iteration: Optional[int] = None,
) -> float:
    """Measure the latency of vanilla attention kernel with given schedule.

    Args:
        config: Attention configuration
        schedule: Optional schedule to use. If None, uses default schedule
        validator: Optional ScheduleModifier for validating schedules
        log_dir: Optional directory to save RPD traces
        iteration: Optional iteration number for trace file naming
        compiled_kernel: Optional pre-compiled kernel to use
        initial_schedule: Optional initial schedule to use for override

    Returns:
        Average latency in seconds
    """
    shape = get_attention_shape(config)

    # Get the kernel and hyperparameters
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(
        shape, config.mfma_variant, config.dynamic_dims, is_v_transposed=True
    )

    # Update hyperparameters with scheduling parameters
    hyperparams.update(get_default_scheduling_params())

    # Create input tensors
    q_shape = (config.num_heads, config.seq_len_q, config.head_dim)
    k_shape = (config.num_heads, config.seq_len_k, config.head_dim)
    v_shape = (config.num_heads, config.seq_len_k, config.head_dim_kv)
    o_shape = (config.num_heads, config.seq_len_q, config.head_dim_kv)

    torch.manual_seed(0)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)
    output = device_zeros(o_shape, dtype=torch.float32)

    # Set up compilation options
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.MODULO,
        use_scheduling_barriers=True,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=config.num_iterations,
        benchmark_repetitions=1,
    )

    if schedule is None:
        options.dump_schedule = initial_schedule
    else:
        options.override_schedule = schedule

    options = set_default_run_config(options)
    compiled_kernel = wave_compile(options, base_attention)

    # Prepare kernel arguments
    kernel_args = (q, k, v.permute([0, 2, 1]), output)

    # Measure performance
    if RPD_AVAILABLE and log_dir is not None and iteration is not None:
        # Use RPD tracer if available and we have a log directory
        trace_file = log_dir / "traces" / f"trace_{iteration:04d}.rpd"
        trace_file.parent.mkdir(parents=True, exist_ok=True)

        timing_result = measure_with_rpd(
            compiled_kernel,
            *kernel_args,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            output_filename=str(trace_file),
            config=config,
        )
    else:
        # Fall back to CUDA events
        timing_result = measure_with_cuda_events(
            compiled_kernel,
            *kernel_args,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            config=config,
        )

    return timing_result.latency_ms / 1000.0


def setup_logging(config: AttentionConfig) -> Tuple[logging.Logger, Path]:
    """Set up logging for the tuning process.

    Args:
        config: Attention configuration

    Returns:
        Tuple of (logger, log_dir)
    """
    # Create timestamp for unique directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories
    base_dir = Path("attention_tuning")
    log_dir = base_dir / f"tune_{timestamp}"
    schedules_dir = log_dir / "schedules"

    # Create directories if they don't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    schedules_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = logging.getLogger("attention_tuner")
    logger.setLevel(logging.INFO)

    # File handler for detailed log
    log_file = log_dir / "tuning.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial configuration
    logger.info("Starting attention kernel tuning")
    logger.info(f"Configuration:\n{json.dumps(asdict(config), indent=2)}")

    return logger, log_dir


def save_schedule(
    schedule: Dict,
    latency: float,
    iteration: int,
    schedules_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save a schedule to a JSON file.

    Args:
        schedule: The schedule to save
        latency: The latency achieved with this schedule
        iteration: The iteration number
        schedules_dir: Directory to save schedules
        logger: Logger instance
    """
    schedule_file = schedules_dir / f"schedule_{iteration:04d}.json"
    schedule_data = {
        "iteration": iteration,
        "latency_ms": latency * 1000,
        "schedule": {str(k): v for k, v in schedule.items()},
    }

    with open(schedule_file, "w") as f:
        json.dump(schedule_data, f, indent=2)

    logger.debug(f"Saved schedule for iteration {iteration} to {schedule_file}")


class TuningLogger:
    """Custom logger for the optimization process."""

    def __init__(self, logger: logging.Logger, schedules_dir: Path):
        self.logger = logger
        self.schedules_dir = schedules_dir
        self.best_latency = float("inf")
        self.best_iteration = -1
        self.history = []
        self.current_iteration = 0

    def log_iteration(
        self, iteration: int, schedule: Dict, latency: float, is_improvement: bool
    ) -> None:
        """Log an optimization iteration.

        Args:
            iteration: Current iteration number
            schedule: Current schedule
            latency: Achieved latency
            is_improvement: Whether this is an improvement
        """
        self.history.append(
            {
                "iteration": iteration,
                "latency_ms": latency * 1000,
                "is_improvement": is_improvement,
            }
        )

        if is_improvement:
            self.best_latency = latency
            self.best_iteration = iteration
            self.logger.info(
                f"Iteration {iteration}: Found improvement! "
                f"Latency: {latency*1000:.2f} ms"
            )
            save_schedule(schedule, latency, iteration, self.schedules_dir, self.logger)
        else:
            self.logger.debug(
                f"Iteration {iteration}: No improvement. "
                f"Latency: {latency*1000:.2f} ms"
            )

    def log_summary(self) -> None:
        """Log a summary of the tuning process."""
        self.logger.info("\nTuning Summary:")
        self.logger.info(f"Best latency: {self.best_latency*1000:.2f} ms")
        self.logger.info(f"Best iteration: {self.best_iteration}")
        self.logger.info(f"Total iterations: {len(self.history)}")

        # Save history
        history_file = self.schedules_dir.parent / "tuning_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        self.logger.info(f"Tuning history saved to {history_file}")


def tune_attention(config: AttentionConfig) -> Tuple[Dict, float]:
    """Tune the vanilla attention kernel using ScheduleOptimizer."""
    # Set up logging
    logger, log_dir = setup_logging(config)
    tuning_logger = TuningLogger(logger, log_dir / "schedules")

    if RPD_AVAILABLE:
        logger.info("Using RPD tracer for performance measurement")
    else:
        logger.warning("RPD tracer not available, falling back to CUDA events")

    shape = get_attention_shape(config)

    # Get initial kernel and validator
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(
        shape, config.mfma_variant, config.dynamic_dims, is_v_transposed=True
    )

    # Create validator
    validator = ScheduleModifier(
        initial_schedule={},  # Will be populated by the kernel
        T=hyperparams.get("T", 128),  # Default to 128 if not specified
        nodes=base_attention.nodes,
        effective_latencies={},  # Will be populated during optimization
        resource_limits=[],  # Will be populated during optimization
        node_rrt_getter=lambda x: {},  # Will be populated during optimization
        raw_edges_list=[],  # Will be populated during optimization
    )

    # Get initial schedule and compiled kernel
    logger.info("Getting initial schedule...")
    initial_latency = measure_attention_latency(
        config, validator=validator, log_dir=log_dir, iteration=0
    )

    logger.info(f"Initial latency: {initial_latency*1000:.2f} ms")

    # Create measurement function that captures config, validator, and compiled kernel
    def measure_fn(schedule: Dict) -> float:
        latency = measure_attention_latency(
            config,
            schedule,
            validator,
            log_dir=log_dir,
            iteration=tuning_logger.current_iteration,
        )
        return latency

    # Create optimizer with custom logging
    class LoggingOptimizer(ScheduleOptimizer):
        def optimize(
            self,
            max_iterations: int = 100,
            max_no_improvement: int = 20,
            verbose: bool = True,
        ) -> OptimizationResult:
            iteration = 0
            no_improvement_streak = 0
            current_best_schedule = initial_schedule.copy()
            current_best_latency = initial_latency

            while iteration < max_iterations:
                # Get candidate schedule
                schedulable_nodes = [
                    n
                    for n in self.validator.nodes
                    if get_custom_operation_type_val(get_custom(n)) != Operation.NOOP
                ]

                if not schedulable_nodes:
                    logger.info("No schedulable nodes to move. Stopping.")
                    break

                node_to_move = random.choice(schedulable_nodes)
                original_cycle = current_best_schedule[node_to_move]
                delta_cycle = random.randint(-self.validator.T, self.validator.T)
                if delta_cycle == 0:
                    delta_cycle = 1 if random.random() < 0.5 else -1
                new_target_cycle = max(0, original_cycle + delta_cycle)

                # Create candidate schedule by copying current best and modifying one node
                candidate_schedule = current_best_schedule.copy()
                candidate_schedule[node_to_move] = new_target_cycle

                # Try the move
                is_valid_move, _, _ = self.validator.attempt_move(
                    node_to_move, new_target_cycle
                )

                if is_valid_move:
                    candidate_latency = self.measure_fn(candidate_schedule)
                    is_improvement = candidate_latency < current_best_latency

                    # Log the iteration
                    tuning_logger.log_iteration(
                        iteration, candidate_schedule, candidate_latency, is_improvement
                    )

                    if is_improvement:
                        current_best_latency = candidate_latency
                        current_best_schedule = candidate_schedule
                        no_improvement_streak = 0
                    else:
                        no_improvement_streak += 1
                else:
                    no_improvement_streak += 1
                    tuning_logger.log_iteration(
                        iteration, current_best_schedule, current_best_latency, False
                    )

                if no_improvement_streak >= max_no_improvement:
                    logger.info(
                        f"Stopping early: No improvement in {max_no_improvement} iterations."
                    )
                    break

                iteration += 1
                tuning_logger.current_iteration = iteration

            # Log final summary
            tuning_logger.log_summary()

            return OptimizationResult(
                schedule=current_best_schedule,
                latency=current_best_latency,
                iterations=iteration,
                algorithm=self.algorithm,
                improvement_history=[h["latency_ms"] for h in tuning_logger.history],
            )

    # Create and run optimizer
    optimizer = LoggingOptimizer(
        validator=validator,
        measure_fn=measure_fn,
        algorithm=OptimizationAlgorithm.HILL_CLIMBING,
    )

    # Run optimization
    result = optimizer.optimize(max_iterations=100, max_no_improvement=20, verbose=True)

    # Save final results
    final_results = {
        "config": asdict(config),
        "initial_latency_ms": initial_latency * 1000,
        "initial_schedule": {str(k): v for k, v in initial_schedule.items()},
        "best_latency_ms": result.latency * 1000,
        "best_schedule": {str(k): v for k, v in result.schedule.items()},
        "total_iterations": result.iterations,
        "improvement_history": result.improvement_history,
    }

    results_file = log_dir / "final_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Final results saved to {results_file}")

    return result.schedule, result.latency


def main():
    # Example configuration
    config = AttentionConfig(
        batch_size=1,
        num_heads=32,
        seq_len_q=512,
        seq_len_k=512,
        head_dim=64,
        head_dim_kv=64,
        mfma_variant=(MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        enable_scheduling=SchedulingType.MODULO,
        dynamic_dims=False,
        num_warmup=10,
        num_iterations=100,
    )

    best_schedule, best_latency = tune_attention(config)


if __name__ == "__main__":
    main()
