from .verifier import ScheduleModifier
import random
from typing import Callable, Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum, auto


class OptimizationAlgorithm(Enum):
    HILL_CLIMBING = auto()
    # Add more algorithms here as needed


@dataclass
class OptimizationResult:
    schedule: Dict
    latency: float
    iterations: int
    algorithm: OptimizationAlgorithm
    improvement_history: List[float]


class ScheduleOptimizer:
    def __init__(
        self,
        validator: ScheduleModifier,
        measure_fn: Callable[[Dict], float],
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.HILL_CLIMBING,
    ):
        """Initialize the schedule optimizer.

        Args:
            validator: A ScheduleModifier instance that validates and modifies schedules
            measure_fn: A function that takes a schedule and returns its latency
            algorithm: The optimization algorithm to use
        """
        self.validator = validator
        self.measure_fn = measure_fn
        self.algorithm = algorithm
        self.current_best_schedule = None
        self.current_best_latency = float("inf")
        self.improvement_history = []

    def _run_hill_climbing(
        self,
        max_iterations: int = 100,
        max_no_improvement: int = 20,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run hill climbing optimization algorithm.

        Args:
            max_iterations: Maximum number of iterations to run
            max_no_improvement: Maximum number of iterations without improvement before stopping
            verbose: Whether to print progress information

        Returns:
            OptimizationResult containing the best schedule and optimization metrics
        """
        if verbose:
            print("Starting Hill Climbing Optimization...")

        current_best_schedule, _ = self.validator.get_current_schedule_state()
        current_best_latency = self.measure_fn(current_best_schedule)
        self.improvement_history = [current_best_latency]

        if verbose:
            print(
                f"Initial Best Latency: {current_best_latency:.2f} for schedule: {{ { {n.name: s for n,s in current_best_schedule.items()} } }}"
            )

        no_improvement_streak = 0
        iteration = 0

        while iteration < max_iterations:
            if verbose:
                print(f"\nIteration {iteration + 1}/{max_iterations}")

            schedulable_nodes = [
                n
                for n in self.validator.nodes
                if get_custom_operation_type_val(get_custom(n)) != Operation.NOOP
            ]

            if not schedulable_nodes:
                if verbose:
                    print("  No schedulable (non-NOOP) nodes to move. Stopping.")
                break

            node_to_move = random.choice(schedulable_nodes)
            original_cycle = current_best_schedule[node_to_move]
            delta_cycle = random.randint(-self.validator.T, self.validator.T)
            if delta_cycle == 0:
                delta_cycle = 1 if random.random() < 0.5 else -1
            new_target_cycle = max(0, original_cycle + delta_cycle)

            if verbose:
                print(
                    f"  Attempting to move node {node_to_move.name} from {original_cycle} to {new_target_cycle}"
                )

            (
                is_valid_move,
                candidate_schedule,
                candidate_rt,
            ) = self.validator.attempt_move(node_to_move, new_target_cycle)

            if is_valid_move and candidate_schedule is not None:
                candidate_latency = self.measure_fn(candidate_schedule)
                if candidate_latency < current_best_latency:
                    if verbose:
                        print(
                            f"  *** Improvement found! New latency: {candidate_latency:.2f} (old: {current_best_latency:.2f}) ***"
                        )
                    current_best_latency = candidate_latency
                    current_best_schedule = candidate_schedule
                    self.validator.commit_move(candidate_schedule, candidate_rt)
                    no_improvement_streak = 0
                    self.improvement_history.append(candidate_latency)
                else:
                    no_improvement_streak += 1
            else:
                no_improvement_streak += 1

            if no_improvement_streak >= max_no_improvement:
                if verbose:
                    print(
                        f"\nStopping early: No improvement in {max_no_improvement} iterations."
                    )
                break

            iteration += 1

        if verbose:
            print("\nOptimization Finished.")
            print(f"Final Best Latency: {current_best_latency:.2f}")
            print(
                f"Final Best Schedule: {{ { {n.name: s for n,s in current_best_schedule.items()} } }}"
            )

        return OptimizationResult(
            schedule=current_best_schedule,
            latency=current_best_latency,
            iterations=iteration,
            algorithm=OptimizationAlgorithm.HILL_CLIMBING,
            improvement_history=self.improvement_history,
        )

    def optimize(
        self,
        max_iterations: int = 100,
        max_no_improvement: int = 20,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run the selected optimization algorithm.

        Args:
            max_iterations: Maximum number of iterations to run
            max_no_improvement: Maximum number of iterations without improvement before stopping
            verbose: Whether to print progress information

        Returns:
            OptimizationResult containing the best schedule and optimization metrics
        """
        if self.algorithm == OptimizationAlgorithm.HILL_CLIMBING:
            return self._run_hill_climbing(
                max_iterations=max_iterations,
                max_no_improvement=max_no_improvement,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported optimization algorithm: {self.algorithm}")


if __name__ == "__main__":
    print("--- Optimizing Schedule ---")
    validator = ScheduleModifier(
        initial_schedule=initial_schedule_cycles,
        T=T_val,
        nodes=all_graph_nodes,
        effective_latencies=effective_latencies_dict,
        resource_limits=resource_limits_arr,
        node_rrt_getter=get_node_rrt_from_user_spec,
        raw_edges_list=raw_edges,
    )
    print("Validator initialized successfully.")

    # Create optimizer instance
    optimizer = ScheduleOptimizer(
        validator=validator,
        measure_fn=measure_on_hardware,
        algorithm=OptimizationAlgorithm.HILL_CLIMBING,
    )

    # Run optimization
    result = optimizer.optimize(max_iterations=50, max_no_improvement=15, verbose=True)

    print(f"\nOptimization completed with {result.algorithm.name}")
    print(f"Final latency: {result.latency:.2f}")
    print(f"Total iterations: {result.iterations}")
    print(f"Improvement history: {result.improvement_history}")
