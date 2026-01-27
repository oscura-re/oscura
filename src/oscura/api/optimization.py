"""Parameter optimization for signal analysis.

This module provides parameter optimization utilities including
grid search, parameter space definition, and optimization result tracking.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "GridSearch",
    "OptimizationResult",
    "ParameterSpace",
    "optimize_parameters",
]


@dataclass
class ParameterSpace:
    """Definition of parameter search space.

    Attributes:
        name: Parameter name
        values: List of values to try
        low: Low bound (for continuous params)
        high: High bound (for continuous params)
        log_scale: Use logarithmic scale
        num_samples: Number of samples for continuous

    Example:
        >>> # Discrete parameter
        >>> window = ParameterSpace("window", values=["hann", "hamming", "blackman"])
        >>> # Continuous parameter
        >>> cutoff = ParameterSpace("cutoff", low=1e3, high=1e6, num_samples=10)

    References:
        API-014: Parameter Optimization
    """

    name: str
    values: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    log_scale: bool = False
    num_samples: int = 10

    def __post_init__(self) -> None:
        """Generate values if continuous parameter."""
        if self.values is None:
            if self.low is not None and self.high is not None:
                if self.log_scale:
                    self.values = list(
                        np.logspace(np.log10(self.low), np.log10(self.high), self.num_samples)
                    )
                else:
                    self.values = list(np.linspace(self.low, self.high, self.num_samples))
            else:
                raise ValueError(
                    f"Parameter {self.name}: must specify either values or (low, high) bounds"
                )

    def __iter__(self) -> Iterator[Any]:
        """Iterate over parameter values."""
        return iter(self.values or [])

    def __len__(self) -> int:
        """Number of parameter values."""
        return len(self.values or [])


@dataclass
class OptimizationResult:
    """Result of parameter optimization.

    Attributes:
        best_params: Best parameter combination
        best_score: Best objective score
        all_results: All evaluated combinations
        elapsed_time: Total optimization time
        num_evaluations: Number of combinations evaluated

    References:
        API-014: Parameter Optimization
    """

    best_params: dict[str, Any]
    best_score: float
    all_results: list[tuple[dict[str, Any], float]] = field(default_factory=list)
    elapsed_time: float = 0.0
    num_evaluations: int = 0

    def top_n(self, n: int = 5) -> list[tuple[dict[str, Any], float]]:
        """Get top N parameter combinations.

        Args:
            n: Number of top results

        Returns:
            List of (params, score) tuples
        """
        sorted_results = sorted(self.all_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:n]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "num_evaluations": self.num_evaluations,
            "elapsed_time": self.elapsed_time,
        }


class GridSearch:
    """Grid search optimization for parameter tuning.

    Exhaustively searches all combinations of parameters to find
    the best combination based on an objective function.

    Example:
        >>> def objective(params, data):
        ...     result = analyze(data, **params)
        ...     return result.snr
        >>>
        >>> search = GridSearch([
        ...     ParameterSpace("nfft", values=[1024, 2048, 4096, 8192]),
        ...     ParameterSpace("window", values=["hann", "hamming"]),
        ...     ParameterSpace("overlap", low=0.25, high=0.75, num_samples=5)
        ... ])
        >>>
        >>> result = search.fit(objective, data)
        >>> print(f"Best params: {result.best_params}")

    References:
        API-014: Parameter Optimization
    """

    def __init__(self, param_spaces: list[ParameterSpace], verbose: bool = True):
        """Initialize grid search.

        Args:
            param_spaces: List of parameter spaces
            verbose: Print progress
        """
        self.param_spaces = param_spaces
        self.verbose = verbose
        self._progress_callback: Callable[[int, int], None] | None = None

    @property
    def num_combinations(self) -> int:
        """Total number of parameter combinations."""
        total = 1
        for space in self.param_spaces:
            total *= len(space)
        return total

    def on_progress(self, callback: Callable[[int, int], None]) -> GridSearch:
        """Set progress callback.

        Args:
            callback: Function called with (current, total)

        Returns:
            Self (for chaining)
        """
        self._progress_callback = callback
        return self

    def fit(
        self,
        objective: Callable[[dict[str, Any], Any], float],
        data: Any,
        *,
        maximize: bool = True,
        early_stop: float | None = None,
    ) -> OptimizationResult:
        """Run grid search optimization.

        Args:
            objective: Objective function (params, data) -> score
            data: Data to pass to objective
            maximize: If True, maximize score; if False, minimize
            early_stop: Stop if score reaches this threshold

        Returns:
            Optimization result with best parameters and all evaluated combinations.
        """
        start_time = time.time()
        all_results: list[tuple[dict[str, Any], float]] = []
        best_params: dict[str, Any] = {}
        best_score = self._initial_best_score(maximize)

        param_names, param_values = self._prepare_parameter_grid()
        total = self.num_combinations

        if self.verbose:
            logger.info(f"Grid search: {total} combinations")

        for i, values in enumerate(itertools.product(*param_values)):
            params = dict(zip(param_names, values, strict=False))
            score = self._evaluate_objective(objective, params, data, maximize)

            all_results.append((params, score))
            best_score, best_params = self._update_best(
                score, params, best_score, best_params, maximize
            )

            self._report_progress(i + 1, total)

            if self._should_stop_early(score, early_stop, maximize, i + 1, total):
                break

        return self._build_result(best_params, best_score, all_results, start_time)

    def _initial_best_score(self, maximize: bool) -> float:
        """Get initial best score for optimization.

        Args:
            maximize: Whether maximizing or minimizing.

        Returns:
            Initial best score value.
        """
        return float("-inf") if maximize else float("inf")

    def _prepare_parameter_grid(self) -> tuple[list[str], list[list[Any]]]:
        """Prepare parameter grid for iteration.

        Returns:
            Tuple of (parameter names, parameter values).
        """
        param_names = [s.name for s in self.param_spaces]
        param_values = [list(s) for s in self.param_spaces]
        return param_names, param_values

    def _evaluate_objective(
        self,
        objective: Callable[[dict[str, Any], Any], float],
        params: dict[str, Any],
        data: Any,
        maximize: bool,
    ) -> float:
        """Evaluate objective function with error handling.

        Args:
            objective: Objective function to evaluate.
            params: Parameter combination.
            data: Data for objective.
            maximize: Whether maximizing.

        Returns:
            Score from objective or fallback on error.
        """
        try:
            return objective(params, data)
        except Exception as e:
            logger.warning(f"Objective failed for {params}: {e}")
            return float("-inf") if maximize else float("inf")

    def _update_best(
        self,
        score: float,
        params: dict[str, Any],
        best_score: float,
        best_params: dict[str, Any],
        maximize: bool,
    ) -> tuple[float, dict[str, Any]]:
        """Update best score and parameters if improved.

        Args:
            score: Current score.
            params: Current parameters.
            best_score: Current best score.
            best_params: Current best parameters.
            maximize: Whether maximizing.

        Returns:
            Tuple of (updated best score, updated best params).
        """
        is_better = score > best_score if maximize else score < best_score
        if is_better:
            # NECESSARY COPY: Preserves best parameters snapshot.
            # Each iteration's best params must be isolated from current mutations.
            return score, params.copy()
        return best_score, best_params

    def _report_progress(self, current: int, total: int) -> None:
        """Report progress to callback if registered.

        Args:
            current: Current iteration.
            total: Total iterations.
        """
        if self._progress_callback:
            self._progress_callback(current, total)

    def _should_stop_early(
        self,
        score: float,
        early_stop: float | None,
        maximize: bool,
        current: int,
        total: int,
    ) -> bool:
        """Check if early stopping criteria met.

        Args:
            score: Current score.
            early_stop: Early stop threshold.
            maximize: Whether maximizing.
            current: Current iteration.
            total: Total iterations.

        Returns:
            True if should stop early.
        """
        if early_stop is None:
            return False

        should_stop = (maximize and score >= early_stop) or (not maximize and score <= early_stop)

        if should_stop and self.verbose:
            logger.info(f"Early stop at {current}/{total}")

        return should_stop

    def _build_result(
        self,
        best_params: dict[str, Any],
        best_score: float,
        all_results: list[tuple[dict[str, Any], float]],
        start_time: float,
    ) -> OptimizationResult:
        """Build optimization result object.

        Args:
            best_params: Best parameters found.
            best_score: Best score achieved.
            all_results: All evaluated combinations.
            start_time: Start time of optimization.

        Returns:
            OptimizationResult object.
        """
        elapsed = time.time() - start_time

        if self.verbose:
            logger.info(f"Completed: best_score={best_score:.4f}, time={elapsed:.2f}s")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            elapsed_time=elapsed,
            num_evaluations=len(all_results),
        )


class RandomSearch:
    """Random search optimization.

    Samples random combinations from parameter space.

    References:
        API-014: Parameter Optimization
    """

    def __init__(
        self,
        param_spaces: list[ParameterSpace],
        n_iterations: int = 100,
        random_state: int | None = None,
    ):
        """Initialize random search.

        Args:
            param_spaces: Parameter spaces
            n_iterations: Number of random samples
            random_state: Random seed
        """
        self.param_spaces = param_spaces
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(
        self,
        objective: Callable[[dict[str, Any], Any], float],
        data: Any,
        *,
        maximize: bool = True,
    ) -> OptimizationResult:
        """Run random search.

        Args:
            objective: Objective function
            data: Data for objective
            maximize: Maximize or minimize

        Returns:
            Optimization result
        """
        rng = np.random.default_rng(self.random_state)
        start_time = time.time()
        all_results: list[tuple[dict[str, Any], float]] = []
        best_params: dict[str, Any] = {}
        best_score = float("-inf") if maximize else float("inf")

        for _ in range(self.n_iterations):
            # Sample random parameters
            params = {}
            for space in self.param_spaces:
                if space.values:
                    params[space.name] = rng.choice(space.values)

            try:
                score = objective(params, data)
            except Exception:
                score = float("-inf") if maximize else float("inf")

            all_results.append((params, score))

            if maximize:
                if score > best_score:
                    best_score = score
                    # NECESSARY COPY: Isolates best params from current/future mutations.
                    best_params = params.copy()
            elif score < best_score:
                best_score = score
                # NECESSARY COPY: Isolates best params from current/future mutations.
                best_params = params.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            elapsed_time=time.time() - start_time,
            num_evaluations=len(all_results),
        )


def optimize_parameters(
    objective: Callable[[dict[str, Any], Any], float],
    data: Any,
    param_spaces: list[ParameterSpace] | dict[str, list[Any]],
    *,
    method: str = "grid",
    maximize: bool = True,
    **kwargs: Any,
) -> OptimizationResult:
    """Optimize parameters for objective function.

    Convenience function for parameter optimization.

    Args:
        objective: Objective function (params, data) -> score
        data: Data to pass to objective
        param_spaces: Parameter spaces (list or dict)
        method: Optimization method ("grid", "random")
        maximize: Maximize or minimize
        **kwargs: Additional arguments for optimizer

    Returns:
        Optimization result

    Raises:
        ValueError: If method is not one of the supported types.

    Example:
        >>> result = optimize_parameters(
        ...     objective=lambda p, d: analyze(d, **p).snr,
        ...     data=trace,
        ...     param_spaces={
        ...         "nfft": [1024, 2048, 4096],
        ...         "window": ["hann", "hamming"]
        ...     }
        ... )

    References:
        API-014: Parameter Optimization
    """
    # Convert dict to ParameterSpace list
    if isinstance(param_spaces, dict):
        param_spaces = [
            ParameterSpace(name, values=values) for name, values in param_spaces.items()
        ]

    if method == "grid":
        optimizer = GridSearch(param_spaces, **kwargs)
    elif method == "random":
        optimizer = RandomSearch(param_spaces, **kwargs)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return optimizer.fit(objective, data, maximize=maximize)
