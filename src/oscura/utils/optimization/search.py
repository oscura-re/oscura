"""Parameter optimization via grid search and random search.

This module provides tools for finding optimal analysis parameters through
systematic or random search of the parameter space.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from oscura.analyzers.waveform.spectral import thd as compute_thd
from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace

    ScoringFunction = Callable[[WaveformTrace, dict[str, Any]], float]
else:
    ScoringFunction = Callable


@dataclass
class SearchResult:
    """Result from parameter search.

    Attributes:
        best_params: Dictionary of best parameters found.
        best_score: Best score achieved.
        all_results: DataFrame with all parameter combinations and scores.
        cv_scores: Cross-validation scores if CV was used.

    Example:
        >>> result = search.fit(traces)
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Best score: {result.best_score}")

    References:
        API-014: Parameter Grid Search
    """

    best_params: dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    cv_scores: NDArray[np.float64] | None = None


def _default_snr_scorer(trace: WaveformTrace, params: dict[str, Any]) -> float:
    """Default SNR scoring function.

    Args:
        trace: Waveform trace to analyze.
        params: Parameters to apply (not used in basic SNR).

    Returns:
        Signal-to-noise ratio in dB.
    """
    # Simple SNR: signal power / noise power
    # Assume first half is signal, second half is noise (oversimplified)
    data = trace.data
    mid = len(data) // 2
    signal_power = np.mean(data[:mid] ** 2)
    noise_power = np.mean((data[mid:] - np.mean(data[mid:])) ** 2)

    if noise_power == 0:
        return float("inf")

    snr = signal_power / noise_power
    return float(10 * np.log10(snr))


def _default_thd_scorer(trace: WaveformTrace, params: dict[str, Any]) -> float:
    """Default THD scoring function.

    Args:
        trace: Waveform trace to analyze.
        params: Parameters to apply (not used in basic THD).

    Returns:
        Negative THD percentage (negative because lower THD is better, but we maximize scores).
    """
    # Compute THD and return negative value (lower THD = better = higher score)
    thd_value = compute_thd(trace)
    return float(-thd_value)


class GridSearchCV:
    """Grid search over parameter space with optional cross-validation.

    Systematically evaluates all combinations of parameters to find the
    optimal configuration.

    Example:
        >>> from oscura.utils.optimization.search import GridSearchCV
        >>> param_grid = {
        ...     'cutoff': [1e5, 5e5, 1e6],
        ...     'order': [2, 4, 6]
        ... }
        >>> search = GridSearchCV(
        ...     param_grid=param_grid,
        ...     scoring='snr',
        ...     cv=3
        ... )
        >>> result = search.fit(traces, apply_filter)
        >>> print(result.best_params)

    References:
        API-014: Parameter Grid Search
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        scoring: Literal["snr", "thd"] | ScoringFunction = "snr",
        cv: int | None = None,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        use_threads: bool = True,
    ) -> None:
        """Initialize grid search.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values.
            scoring: Scoring function. Built-in: 'snr', 'thd', or custom callable.
            cv: Number of cross-validation folds. None for no CV.
            parallel: Enable parallel evaluation.
            max_workers: Maximum parallel workers.
            use_threads: Use threads instead of processes.

        Raises:
            AnalysisError: If scoring function is invalid.

        Example:
            >>> param_grid = {'cutoff': [1e6, 2e6], 'order': [4, 6]}
            >>> search = GridSearchCV(param_grid, scoring='snr', cv=3)
        """
        self.param_grid = param_grid
        self.cv = cv
        self.parallel = parallel
        self.max_workers = max_workers
        self.use_threads = use_threads

        # Set scoring function
        if scoring == "snr":
            self.scoring_fn = _default_snr_scorer
        elif scoring == "thd":
            self.scoring_fn = _default_thd_scorer
        elif callable(scoring):
            self.scoring_fn = scoring  # type: ignore[assignment]
        else:
            raise AnalysisError(f"Unknown scoring function: {scoring}")

        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.results_df_: pd.DataFrame | None = None

    def fit(
        self,
        traces: list[WaveformTrace] | WaveformTrace,
        transform_fn: Callable[[WaveformTrace, dict[str, Any]], WaveformTrace],
    ) -> SearchResult:
        """Fit grid search on traces.

        Evaluates all parameter combinations and finds the best.

        Args:
            traces: Trace or list of traces to evaluate on.
            transform_fn: Function that applies parameters to trace.
                Should accept (trace, **params) and return transformed trace.

        Returns:
            SearchResult with best parameters and all results.

        Example:
            >>> def apply_filter(trace, cutoff, order):
            ...     return lowpass_filter(trace, cutoff=cutoff, order=order)
            >>> result = search.fit(traces, apply_filter)

        References:
            API-014: Parameter Grid Search
        """
        # Convert single trace to list
        if not isinstance(traces, list):
            traces = [traces]

        # Generate all parameter combinations
        param_combinations = self._generate_combinations()

        # Evaluate each combination
        results = self._evaluate_combinations(param_combinations, traces, transform_fn)

        # Convert to DataFrame
        self.results_df_ = pd.DataFrame(results)

        # Find best
        best_idx = self.results_df_["mean_score"].idxmax()
        best_row = self.results_df_.iloc[best_idx]

        self.best_params_ = {k: best_row[k] for k in self.param_grid}
        self.best_score_ = float(best_row["mean_score"])

        # Collect CV scores if available
        cv_scores = None
        if self.cv:
            cv_cols = [c for c in self.results_df_.columns if c.startswith("cv_")]
            if cv_cols:
                cv_scores = self.results_df_.loc[best_idx, cv_cols].values

        return SearchResult(
            best_params=self.best_params_,
            best_score=self.best_score_,
            all_results=self.results_df_,
            cv_scores=cv_scores,
        )

    def _generate_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations from grid.

        Returns:
            List of parameter dictionaries.
        """
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo, strict=False)))

        return combinations

    def _evaluate_combinations(
        self,
        param_combinations: list[dict[str, Any]],
        traces: list[WaveformTrace],
        transform_fn: Callable[[WaveformTrace, dict[str, Any]], WaveformTrace],
    ) -> list[dict[str, Any]]:
        """Evaluate all parameter combinations.

        Args:
            param_combinations: List of parameter dicts to evaluate.
            traces: Traces to evaluate on.
            transform_fn: Transformation function.

        Returns:
            List of result dictionaries.
        """
        if self.parallel:
            return self._evaluate_parallel(param_combinations, traces, transform_fn)
        else:
            return self._evaluate_sequential(param_combinations, traces, transform_fn)

    def _evaluate_one(
        self,
        params: dict[str, Any],
        traces: list[WaveformTrace],
        transform_fn: Callable[[WaveformTrace, dict[str, Any]], WaveformTrace],
    ) -> dict[str, Any]:
        """Evaluate one parameter combination.

        Args:
            params: Parameters to evaluate.
            traces: Traces to evaluate on.
            transform_fn: Transformation function.

        Returns:
            Result dictionary with scores.
        """
        scores: list[float] = []

        if self.cv:
            # Cross-validation - split traces into folds
            fold_size = len(traces) // self.cv
            for i in range(self.cv):
                # Select fold
                start = i * fold_size
                end = start + fold_size if i < self.cv - 1 else len(traces)
                fold_traces = traces[start:end]

                # Evaluate on fold
                fold_scores = []
                for trace in fold_traces:
                    transformed = transform_fn(trace, **params)  # type: ignore[call-arg]
                    score = self.scoring_fn(transformed, params)
                    fold_scores.append(score)

                scores.append(float(np.mean(fold_scores)))

        else:
            # No CV - evaluate on all traces
            for trace in traces:
                transformed = transform_fn(trace, **params)  # type: ignore[call-arg]
                score = self.scoring_fn(transformed, params)
                scores.append(score)

        # Build result
        result = params.copy()
        result["mean_score"] = float(np.mean(scores))
        result["std_score"] = float(np.std(scores))

        if self.cv:
            for i, score in enumerate(scores):
                result[f"cv_{i}"] = float(score)

        return result

    def _evaluate_sequential(
        self,
        param_combinations: list[dict[str, Any]],
        traces: list[WaveformTrace],
        transform_fn: Callable[[WaveformTrace, dict[str, Any]], WaveformTrace],
    ) -> list[dict[str, Any]]:
        """Evaluate combinations sequentially.

        Args:
            param_combinations: Parameter combinations.
            traces: Traces to evaluate on.
            transform_fn: Transformation function.

        Returns:
            List of results.
        """
        results = []
        for params in param_combinations:
            result = self._evaluate_one(params, traces, transform_fn)
            results.append(result)
        return results

    def _evaluate_parallel(
        self,
        param_combinations: list[dict[str, Any]],
        traces: list[WaveformTrace],
        transform_fn: Callable[[WaveformTrace, dict[str, Any]], WaveformTrace],
    ) -> list[dict[str, Any]]:
        """Evaluate combinations in parallel.

        Args:
            param_combinations: Parameter combinations.
            traces: Traces to evaluate on.
            transform_fn: Transformation function.

        Returns:
            List of results.
        """
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_one, params, traces, transform_fn): params
                for params in param_combinations
            }

            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results


class RandomizedSearchCV:
    """Random search over parameter distributions.

    Samples random combinations from parameter distributions rather than
    exhaustively evaluating all combinations.

    Example:
        >>> from oscura.utils.optimization.search import RandomizedSearchCV
        >>> import numpy as np
        >>> param_distributions = {
        ...     'cutoff': lambda: np.random.uniform(1e5, 1e7),
        ...     'order': lambda: np.random.choice([2, 4, 6, 8])
        ... }
        >>> search = RandomizedSearchCV(
        ...     param_distributions=param_distributions,
        ...     n_iter=20,
        ...     scoring='snr'
        ... )
        >>> result = search.fit(traces, apply_filter)

    References:
        API-014: Parameter Grid Search
    """

    def __init__(
        self,
        param_distributions: dict[str, Callable[[], Any]],
        n_iter: int = 10,
        scoring: Literal["snr", "thd"] | ScoringFunction = "snr",
        cv: int | None = None,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        use_threads: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize randomized search.

        Args:
            param_distributions: Dict mapping parameter names to sampling functions.
            n_iter: Number of parameter combinations to sample.
            scoring: Scoring function.
            cv: Number of cross-validation folds.
            parallel: Enable parallel evaluation.
            max_workers: Maximum parallel workers.
            use_threads: Use threads instead of processes.
            random_state: Random seed for reproducibility.

        Raises:
            AnalysisError: If scoring function is invalid.

        Example:
            >>> param_dist = {'cutoff': lambda: np.random.uniform(1e5, 1e7)}
            >>> search = RandomizedSearchCV(param_dist, n_iter=50)
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.parallel = parallel
        self.max_workers = max_workers
        self.use_threads = use_threads

        if random_state is not None:
            np.random.seed(random_state)

        # Set scoring function
        if scoring == "snr":
            self.scoring_fn = _default_snr_scorer
        elif scoring == "thd":
            self.scoring_fn = _default_thd_scorer
        elif callable(scoring):
            self.scoring_fn = scoring  # type: ignore[assignment]
        else:
            raise AnalysisError(f"Unknown scoring function: {scoring}")

        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.results_df_: pd.DataFrame | None = None

    def fit(
        self,
        traces: list[WaveformTrace] | WaveformTrace,
        transform_fn: Callable[[WaveformTrace, dict[str, Any]], WaveformTrace],
    ) -> SearchResult:
        """Fit randomized search on traces.

        Args:
            traces: Trace or list of traces to evaluate on.
            transform_fn: Function that applies parameters to trace.

        Returns:
            SearchResult with best parameters.

        Example:
            >>> result = search.fit(traces, apply_filter)
            >>> print(f"Best cutoff: {result.best_params['cutoff']:.2e}")

        References:
            API-014: Parameter Grid Search
        """
        # Convert single trace to list
        if not isinstance(traces, list):
            traces = [traces]

        # Sample parameter combinations
        param_combinations = self._sample_combinations()

        # Reuse grid search evaluation logic
        grid_search = GridSearchCV(
            param_grid={},  # Not used
            scoring=self.scoring_fn,
            cv=self.cv,
            parallel=self.parallel,
            max_workers=self.max_workers,
            use_threads=self.use_threads,
        )

        results = grid_search._evaluate_combinations(param_combinations, traces, transform_fn)

        # Convert to DataFrame
        self.results_df_ = pd.DataFrame(results)

        # Find best
        best_idx = self.results_df_["mean_score"].idxmax()
        best_row = self.results_df_.iloc[best_idx]

        self.best_params_ = {k: best_row[k] for k in self.param_distributions}
        self.best_score_ = float(best_row["mean_score"])

        # Collect CV scores if available
        cv_scores = None
        if self.cv:
            cv_cols = [c for c in self.results_df_.columns if c.startswith("cv_")]
            if cv_cols:
                cv_scores = self.results_df_.loc[best_idx, cv_cols].values

        return SearchResult(
            best_params=self.best_params_,
            best_score=self.best_score_,
            all_results=self.results_df_,
            cv_scores=cv_scores,
        )

    def _sample_combinations(self) -> list[dict[str, Any]]:
        """Sample random parameter combinations.

        Returns:
            List of sampled parameter dictionaries.
        """
        combinations = []

        for _ in range(self.n_iter):
            params = {key: sampler() for key, sampler in self.param_distributions.items()}
            combinations.append(params)

        return combinations


__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
    "ScoringFunction",
    "SearchResult",
]
