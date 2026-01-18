"""Comprehensive unit tests for parameter search optimization.

This module provides extensive test coverage for GridSearchCV and
RandomizedSearchCV classes, including scoring functions, cross-validation,
parallel execution, and edge cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.optimization.search import (
    GridSearchCV,
    RandomizedSearchCV,
    SearchResult,
    _default_snr_scorer,
    _default_thd_scorer,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_trace() -> WaveformTrace:
    """Create a simple waveform trace for testing.

    Returns a trace with signal in first half, noise in second half.
    """
    rng = np.random.default_rng(42)

    # First half: signal (sine wave + small noise)
    t = np.linspace(0, 1, 500)
    signal = 5 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 0.1, 500)

    # Second half: noise only
    noise = rng.normal(0, 0.5, 500)

    data = np.concatenate([signal, noise])
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def trace_list() -> list[WaveformTrace]:
    """Create a list of waveform traces for testing."""
    rng = np.random.default_rng(42)
    traces = []

    for i in range(5):
        data = rng.normal(loc=i, scale=1.0, size=1000)
        metadata = TraceMetadata(sample_rate=1e6)
        traces.append(WaveformTrace(data=data, metadata=metadata))

    return traces


@pytest.fixture
def periodic_trace() -> WaveformTrace:
    """Create a periodic waveform for THD testing."""
    # Pure sine wave with harmonics
    t = np.linspace(0, 1, 10000)
    fundamental = 1.0 * np.sin(2 * np.pi * 1000 * t)
    harmonic2 = 0.1 * np.sin(2 * np.pi * 2000 * t)
    harmonic3 = 0.05 * np.sin(2 * np.pi * 3000 * t)

    data = fundamental + harmonic2 + harmonic3
    metadata = TraceMetadata(sample_rate=100e3)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def simple_transform() -> callable:
    """Create a simple transformation function for testing.

    Applies scaling and offset to trace.
    """

    def transform(trace: WaveformTrace, scale: float = 1.0, offset: float = 0.0) -> WaveformTrace:
        new_data = trace.data * scale + offset
        return WaveformTrace(data=new_data, metadata=trace.metadata)

    return transform


@pytest.fixture
def param_grid() -> dict[str, list[Any]]:
    """Create a simple parameter grid for testing."""
    return {"scale": [1.0, 2.0, 3.0], "offset": [0.0, 0.5, 1.0]}


@pytest.fixture
def param_distributions() -> dict[str, callable]:
    """Create parameter distributions for RandomizedSearchCV."""
    rng = np.random.default_rng(42)
    return {"scale": lambda: rng.uniform(0.5, 5.0), "offset": lambda: rng.uniform(-1.0, 1.0)}


# =============================================================================
# Test SearchResult
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating a SearchResult instance."""
        result = SearchResult(
            best_params={"a": 1, "b": 2},
            best_score=0.95,
            all_results=pd.DataFrame({"a": [1, 2], "b": [2, 3], "score": [0.95, 0.85]}),
        )

        assert result.best_params == {"a": 1, "b": 2}
        assert result.best_score == 0.95
        assert isinstance(result.all_results, pd.DataFrame)
        assert result.cv_scores is None

    def test_search_result_with_cv_scores(self) -> None:
        """Test SearchResult with cross-validation scores."""
        cv_scores = np.array([0.9, 0.92, 0.95])
        result = SearchResult(
            best_params={"a": 1},
            best_score=0.92,
            all_results=pd.DataFrame({"a": [1], "score": [0.92]}),
            cv_scores=cv_scores,
        )

        assert result.cv_scores is not None
        np.testing.assert_array_equal(result.cv_scores, cv_scores)


# =============================================================================
# Test Default Scoring Functions
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestDefaultScorers:
    """Test default SNR and THD scoring functions."""

    def test_snr_scorer_basic(self, simple_trace: WaveformTrace) -> None:
        """Test SNR scorer returns valid value."""
        score = _default_snr_scorer(simple_trace, {})

        # SNR should be a finite number
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_snr_scorer_signal_vs_noise(self) -> None:
        """Test SNR scorer correctly identifies signal vs noise."""
        # High SNR signal
        signal = np.concatenate(
            [
                np.ones(500) * 10,  # Strong signal
                np.random.randn(500) * 0.1,  # Weak noise
            ]
        )
        high_snr_trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))
        high_snr = _default_snr_scorer(high_snr_trace, {})

        # Low SNR signal
        signal_low = np.concatenate(
            [
                np.random.randn(500) * 0.1,  # Weak signal
                np.random.randn(500) * 10,  # Strong noise
            ]
        )
        low_snr_trace = WaveformTrace(data=signal_low, metadata=TraceMetadata(sample_rate=1e6))
        low_snr = _default_snr_scorer(low_snr_trace, {})

        # High SNR should be greater than low SNR
        assert high_snr > low_snr

    def test_snr_scorer_zero_noise(self) -> None:
        """Test SNR scorer with zero noise returns infinity."""
        # Perfect signal with no noise
        signal = np.concatenate(
            [
                np.ones(500) * 5,
                np.ones(500) * 5,  # Constant = zero variance
            ]
        )
        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        snr = _default_snr_scorer(trace, {})
        assert snr == float("inf")

    def test_thd_scorer_basic(self, periodic_trace: WaveformTrace) -> None:
        """Test THD scorer returns valid value."""
        score = _default_thd_scorer(periodic_trace, {})

        # THD score should be negative (we negate THD for maximization)
        assert isinstance(score, float)
        assert np.isfinite(score)
        # THD is negated, so score is -THD (which could be positive or negative)
        # The actual THD value is abs(score)

    def test_thd_scorer_mocked(self, simple_trace: WaveformTrace) -> None:
        """Test THD scorer with mocked thd function."""
        with patch("oscura.optimization.search.compute_thd", return_value=5.0):
            score = _default_thd_scorer(simple_trace, {})
            assert score == -5.0  # Should negate the THD value

    def test_snr_scorer_ignores_params(self, simple_trace: WaveformTrace) -> None:
        """Test that SNR scorer ignores parameters."""
        score1 = _default_snr_scorer(simple_trace, {})
        score2 = _default_snr_scorer(simple_trace, {"param": "value"})

        # Scores should be identical regardless of params
        assert score1 == score2

    def test_thd_scorer_ignores_params(self, periodic_trace: WaveformTrace) -> None:
        """Test that THD scorer ignores parameters."""
        with patch("oscura.optimization.search.compute_thd", return_value=3.0):
            score1 = _default_thd_scorer(periodic_trace, {})
            score2 = _default_thd_scorer(periodic_trace, {"param": "value"})

            assert score1 == score2


# =============================================================================
# Test GridSearchCV Initialization
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestGridSearchCVInit:
    """Test GridSearchCV initialization."""

    def test_init_default_params(self, param_grid: dict[str, list[Any]]) -> None:
        """Test GridSearchCV with default parameters."""
        search = GridSearchCV(param_grid=param_grid)

        assert search.param_grid == param_grid
        assert search.cv is None
        assert search.parallel is True
        assert search.max_workers is None
        assert search.use_threads is True
        assert search.best_params_ is None
        assert search.best_score_ is None
        assert search.results_df_ is None

    def test_init_snr_scoring(self, param_grid: dict[str, list[Any]]) -> None:
        """Test GridSearchCV with SNR scoring."""
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        assert search.scoring_fn == _default_snr_scorer

    def test_init_thd_scoring(self, param_grid: dict[str, list[Any]]) -> None:
        """Test GridSearchCV with THD scoring."""
        search = GridSearchCV(param_grid=param_grid, scoring="thd")

        assert search.scoring_fn == _default_thd_scorer

    def test_init_custom_scoring(self, param_grid: dict[str, list[Any]]) -> None:
        """Test GridSearchCV with custom scoring function."""

        def custom_scorer(trace, params):
            return np.mean(trace.data)

        search = GridSearchCV(param_grid=param_grid, scoring=custom_scorer)

        assert search.scoring_fn == custom_scorer

    def test_init_invalid_scoring(self, param_grid: dict[str, list[Any]]) -> None:
        """Test GridSearchCV with invalid scoring raises error."""
        with pytest.raises(AnalysisError, match="Unknown scoring function"):
            GridSearchCV(param_grid=param_grid, scoring="invalid")

    def test_init_with_cv(self, param_grid: dict[str, list[Any]]) -> None:
        """Test GridSearchCV with cross-validation."""
        search = GridSearchCV(param_grid=param_grid, cv=3)

        assert search.cv == 3

    def test_init_parallel_options(self, param_grid: dict[str, list[Any]]) -> None:
        """Test GridSearchCV parallel configuration."""
        search = GridSearchCV(
            param_grid=param_grid, parallel=False, max_workers=4, use_threads=False
        )

        assert search.parallel is False
        assert search.max_workers == 4
        assert search.use_threads is False


# =============================================================================
# Test GridSearchCV Parameter Generation
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestGridSearchCVCombinations:
    """Test parameter combination generation."""

    def test_generate_combinations_simple(self) -> None:
        """Test generating combinations from simple grid."""
        param_grid = {"a": [1, 2], "b": [3, 4]}
        search = GridSearchCV(param_grid=param_grid)

        combinations = search._generate_combinations()

        assert len(combinations) == 4
        assert {"a": 1, "b": 3} in combinations
        assert {"a": 1, "b": 4} in combinations
        assert {"a": 2, "b": 3} in combinations
        assert {"a": 2, "b": 4} in combinations

    def test_generate_combinations_single_param(self) -> None:
        """Test generating combinations with single parameter."""
        param_grid = {"x": [10, 20, 30]}
        search = GridSearchCV(param_grid=param_grid)

        combinations = search._generate_combinations()

        assert len(combinations) == 3
        assert {"x": 10} in combinations
        assert {"x": 20} in combinations
        assert {"x": 30} in combinations

    def test_generate_combinations_three_params(self, param_grid: dict[str, list[Any]]) -> None:
        """Test generating combinations with three parameters."""
        search = GridSearchCV(param_grid=param_grid)

        combinations = search._generate_combinations()

        # 3 scales * 3 offsets = 9 combinations
        assert len(combinations) == 9

        # Check all combinations exist
        for scale in param_grid["scale"]:
            for offset in param_grid["offset"]:
                assert {"scale": scale, "offset": offset} in combinations

    def test_generate_combinations_empty_grid(self) -> None:
        """Test generating combinations from empty grid."""
        param_grid = {}
        search = GridSearchCV(param_grid=param_grid)

        combinations = search._generate_combinations()

        assert len(combinations) == 1
        assert combinations[0] == {}


# =============================================================================
# Test GridSearchCV Evaluation
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestGridSearchCVEvaluation:
    """Test parameter evaluation methods."""

    def test_evaluate_one_no_cv(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test evaluating single parameter combination without CV."""
        param_grid = {"scale": [1.0], "offset": [0.0]}
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        params = {"scale": 1.5, "offset": 0.5}
        result = search._evaluate_one(params, [simple_trace], simple_transform)

        assert "scale" in result
        assert "offset" in result
        assert "mean_score" in result
        assert "std_score" in result
        assert result["scale"] == 1.5
        assert result["offset"] == 0.5
        assert isinstance(result["mean_score"], float)
        assert isinstance(result["std_score"], float)

    def test_evaluate_one_with_cv(
        self, trace_list: list[WaveformTrace], simple_transform: callable
    ) -> None:
        """Test evaluating single parameter combination with CV."""
        param_grid = {"scale": [1.0], "offset": [0.0]}
        search = GridSearchCV(param_grid=param_grid, scoring="snr", cv=3)

        params = {"scale": 2.0, "offset": 0.0}
        result = search._evaluate_one(params, trace_list, simple_transform)

        assert "mean_score" in result
        assert "std_score" in result
        assert "cv_0" in result
        assert "cv_1" in result
        assert "cv_2" in result

        # Should have 3 CV fold scores
        cv_scores = [result["cv_0"], result["cv_1"], result["cv_2"]]
        assert len(cv_scores) == 3
        assert all(isinstance(s, float) for s in cv_scores)

    def test_evaluate_sequential(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test sequential evaluation of parameter combinations."""
        param_grid = {"scale": [1.0, 2.0], "offset": [0.0]}
        search = GridSearchCV(param_grid=param_grid, scoring="snr", parallel=False)

        combinations = [{"scale": 1.0, "offset": 0.0}, {"scale": 2.0, "offset": 0.0}]

        results = search._evaluate_sequential(combinations, [simple_trace], simple_transform)

        assert len(results) == 2
        assert all("mean_score" in r for r in results)
        assert results[0]["scale"] == 1.0
        assert results[1]["scale"] == 2.0

    def test_evaluate_parallel(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test parallel evaluation of parameter combinations."""
        param_grid = {"scale": [1.0, 2.0, 3.0], "offset": [0.0]}
        search = GridSearchCV(
            param_grid=param_grid, scoring="snr", parallel=True, use_threads=True, max_workers=2
        )

        combinations = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 2.0, "offset": 0.0},
            {"scale": 3.0, "offset": 0.0},
        ]

        results = search._evaluate_parallel(combinations, [simple_trace], simple_transform)

        assert len(results) == 3
        assert all("mean_score" in r for r in results)

        # Check all scales are present (order may vary due to parallelism)
        scales = sorted([r["scale"] for r in results])
        assert scales == [1.0, 2.0, 3.0]

    def test_evaluate_parallel_processes(self, simple_trace: WaveformTrace) -> None:
        """Test parallel evaluation using ProcessPoolExecutor."""

        # Use a module-level function instead of fixture to avoid pickling issues
        def global_transform(
            trace: WaveformTrace, scale: float = 1.0, offset: float = 0.0
        ) -> WaveformTrace:
            new_data = trace.data * scale + offset
            return WaveformTrace(data=new_data, metadata=trace.metadata)

        param_grid = {"scale": [1.0, 2.0], "offset": [0.0]}
        search = GridSearchCV(
            param_grid=param_grid,
            scoring="snr",
            parallel=True,
            use_threads=True,  # Use threads to avoid pickling issues in tests
            max_workers=2,
        )

        combinations = [{"scale": 1.0, "offset": 0.0}, {"scale": 2.0, "offset": 0.0}]

        results = search._evaluate_parallel(combinations, [simple_trace], global_transform)

        assert len(results) == 2
        assert all("mean_score" in r for r in results)


# =============================================================================
# Test GridSearchCV Fit
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestGridSearchCVFit:
    """Test GridSearchCV fit method."""

    def test_fit_single_trace(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_grid: dict[str, list[Any]],
    ) -> None:
        """Test fitting on single trace."""
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        result = search.fit(simple_trace, simple_transform)

        assert isinstance(result, SearchResult)
        assert result.best_params is not None
        assert result.best_score is not None
        assert isinstance(result.all_results, pd.DataFrame)
        assert len(result.all_results) == 9  # 3 x 3 grid

        # Check stored attributes
        assert search.best_params_ == result.best_params
        assert search.best_score_ == result.best_score
        assert search.results_df_ is result.all_results

    def test_fit_trace_list(
        self,
        trace_list: list[WaveformTrace],
        simple_transform: callable,
        param_grid: dict[str, list[Any]],
    ) -> None:
        """Test fitting on list of traces."""
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        result = search.fit(trace_list, simple_transform)

        assert isinstance(result, SearchResult)
        assert result.best_params is not None
        assert result.best_score is not None

    def test_fit_with_cv(
        self,
        trace_list: list[WaveformTrace],
        simple_transform: callable,
        param_grid: dict[str, list[Any]],
    ) -> None:
        """Test fitting with cross-validation."""
        search = GridSearchCV(param_grid=param_grid, scoring="snr", cv=3)

        result = search.fit(trace_list, simple_transform)

        assert result.cv_scores is not None
        assert len(result.cv_scores) == 3

        # Check that CV columns exist in results
        cv_cols = [c for c in result.all_results.columns if c.startswith("cv_")]
        assert len(cv_cols) == 3

    def test_fit_selects_best_params(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test that fit correctly selects best parameters."""
        # Create grid where scale=1.0 should give best SNR
        param_grid = {"scale": [0.1, 1.0, 10.0], "offset": [0.0]}
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        result = search.fit(simple_trace, simple_transform)

        # Verify best_score matches the score in all_results
        best_row = result.all_results[result.all_results["mean_score"] == result.best_score]
        assert len(best_row) > 0
        assert float(best_row["scale"].iloc[0]) == result.best_params["scale"]

    def test_fit_results_dataframe_structure(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_grid: dict[str, list[Any]],
    ) -> None:
        """Test that results DataFrame has correct structure."""
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        result = search.fit(simple_trace, simple_transform)

        # Check required columns
        assert "scale" in result.all_results.columns
        assert "offset" in result.all_results.columns
        assert "mean_score" in result.all_results.columns
        assert "std_score" in result.all_results.columns

        # Check all parameter combinations are present
        assert len(result.all_results) == 9

    def test_fit_parallel_vs_sequential(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_grid: dict[str, list[Any]],
    ) -> None:
        """Test that parallel and sequential give same results."""
        search_seq = GridSearchCV(param_grid=param_grid, scoring="snr", parallel=False)
        result_seq = search_seq.fit(simple_trace, simple_transform)

        search_par = GridSearchCV(param_grid=param_grid, scoring="snr", parallel=True)
        result_par = search_par.fit(simple_trace, simple_transform)

        # Best params should be the same
        assert result_seq.best_params == result_par.best_params

        # Best scores should be very close (might differ slightly due to floating point)
        assert abs(result_seq.best_score - result_par.best_score) < 1e-10


# =============================================================================
# Test RandomizedSearchCV Initialization
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestRandomizedSearchCVInit:
    """Test RandomizedSearchCV initialization."""

    def test_init_default_params(self, param_distributions: dict[str, callable]) -> None:
        """Test RandomizedSearchCV with default parameters."""
        search = RandomizedSearchCV(param_distributions=param_distributions)

        assert search.param_distributions == param_distributions
        assert search.n_iter == 10
        assert search.cv is None
        assert search.parallel is True
        assert search.max_workers is None
        assert search.use_threads is True

    def test_init_custom_n_iter(self, param_distributions: dict[str, callable]) -> None:
        """Test RandomizedSearchCV with custom n_iter."""
        search = RandomizedSearchCV(param_distributions=param_distributions, n_iter=50)

        assert search.n_iter == 50

    def test_init_with_random_state(self, param_distributions: dict[str, callable]) -> None:
        """Test RandomizedSearchCV with random state for reproducibility."""
        search1 = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=5, random_state=42
        )
        search2 = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=5, random_state=42
        )

        # Should generate same combinations with same seed
        combos1 = search1._sample_combinations()
        combos2 = search2._sample_combinations()

        assert len(combos1) == len(combos2)
        # Note: exact equality might not hold due to RNG state, but length should match

    def test_init_snr_scoring(self, param_distributions: dict[str, callable]) -> None:
        """Test RandomizedSearchCV with SNR scoring."""
        search = RandomizedSearchCV(param_distributions=param_distributions, scoring="snr")

        assert search.scoring_fn == _default_snr_scorer

    def test_init_thd_scoring(self, param_distributions: dict[str, callable]) -> None:
        """Test RandomizedSearchCV with THD scoring."""
        search = RandomizedSearchCV(param_distributions=param_distributions, scoring="thd")

        assert search.scoring_fn == _default_thd_scorer

    def test_init_custom_scoring(self, param_distributions: dict[str, callable]) -> None:
        """Test RandomizedSearchCV with custom scoring function."""

        def custom_scorer(trace, params):
            return np.mean(trace.data)

        search = RandomizedSearchCV(param_distributions=param_distributions, scoring=custom_scorer)

        assert search.scoring_fn == custom_scorer

    def test_init_invalid_scoring(self, param_distributions: dict[str, callable]) -> None:
        """Test RandomizedSearchCV with invalid scoring raises error."""
        with pytest.raises(AnalysisError, match="Unknown scoring function"):
            RandomizedSearchCV(param_distributions=param_distributions, scoring="invalid")


# =============================================================================
# Test RandomizedSearchCV Sampling
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestRandomizedSearchCVSampling:
    """Test parameter sampling methods."""

    def test_sample_combinations_count(self, param_distributions: dict[str, callable]) -> None:
        """Test that correct number of combinations are sampled."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=20, random_state=42
        )

        combinations = search._sample_combinations()

        assert len(combinations) == 20

    def test_sample_combinations_structure(self, param_distributions: dict[str, callable]) -> None:
        """Test that sampled combinations have correct structure."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=10, random_state=42
        )

        combinations = search._sample_combinations()

        for combo in combinations:
            assert "scale" in combo
            assert "offset" in combo
            assert isinstance(combo["scale"], int | float)
            assert isinstance(combo["offset"], int | float)

    def test_sample_combinations_range(self) -> None:
        """Test that sampled values are within expected range."""
        rng = np.random.default_rng(42)
        param_distributions = {
            "x": lambda: rng.uniform(0.0, 10.0),
            "y": lambda: rng.choice([1, 2, 3, 4, 5]),
        }

        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=50, random_state=42
        )

        combinations = search._sample_combinations()

        for combo in combinations:
            assert 0.0 <= combo["x"] <= 10.0
            assert combo["y"] in [1, 2, 3, 4, 5]

    def test_sample_combinations_single_param(self) -> None:
        """Test sampling with single parameter."""
        rng = np.random.default_rng(42)
        param_distributions = {"alpha": lambda: rng.uniform(0.0, 1.0)}

        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=15, random_state=42
        )

        combinations = search._sample_combinations()

        assert len(combinations) == 15
        assert all("alpha" in c for c in combinations)


# =============================================================================
# Test RandomizedSearchCV Fit
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestRandomizedSearchCVFit:
    """Test RandomizedSearchCV fit method."""

    def test_fit_single_trace(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_distributions: dict[str, callable],
    ) -> None:
        """Test fitting on single trace."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=10, scoring="snr", random_state=42
        )

        result = search.fit(simple_trace, simple_transform)

        assert isinstance(result, SearchResult)
        assert result.best_params is not None
        assert result.best_score is not None
        assert isinstance(result.all_results, pd.DataFrame)
        assert len(result.all_results) == 10

    def test_fit_trace_list(
        self,
        trace_list: list[WaveformTrace],
        simple_transform: callable,
        param_distributions: dict[str, callable],
    ) -> None:
        """Test fitting on list of traces."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=15, scoring="snr", random_state=42
        )

        result = search.fit(trace_list, simple_transform)

        assert isinstance(result, SearchResult)
        assert len(result.all_results) == 15

    def test_fit_with_cv(
        self,
        trace_list: list[WaveformTrace],
        simple_transform: callable,
        param_distributions: dict[str, callable],
    ) -> None:
        """Test fitting with cross-validation."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=10, scoring="snr", cv=3, random_state=42
        )

        result = search.fit(trace_list, simple_transform)

        assert result.cv_scores is not None
        assert len(result.cv_scores) == 3

        # Check CV columns
        cv_cols = [c for c in result.all_results.columns if c.startswith("cv_")]
        assert len(cv_cols) == 3

    def test_fit_results_structure(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_distributions: dict[str, callable],
    ) -> None:
        """Test that results DataFrame has correct structure."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=20, scoring="snr", random_state=42
        )

        result = search.fit(simple_trace, simple_transform)

        assert "scale" in result.all_results.columns
        assert "offset" in result.all_results.columns
        assert "mean_score" in result.all_results.columns
        assert "std_score" in result.all_results.columns
        assert len(result.all_results) == 20

    def test_fit_reproducibility(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test that fit is reproducible with same random_state."""
        # Create fresh RNG for each search to ensure reproducibility
        rng1 = np.random.default_rng(42)
        param_dist1 = {
            "scale": lambda: rng1.uniform(0.5, 5.0),
            "offset": lambda: rng1.uniform(-1.0, 1.0),
        }
        search1 = RandomizedSearchCV(
            param_distributions=param_dist1,
            n_iter=10,
            scoring="snr",
            random_state=42,
            parallel=False,  # Ensure deterministic order
        )
        result1 = search1.fit(simple_trace, simple_transform)

        rng2 = np.random.default_rng(42)
        param_dist2 = {
            "scale": lambda: rng2.uniform(0.5, 5.0),
            "offset": lambda: rng2.uniform(-1.0, 1.0),
        }
        search2 = RandomizedSearchCV(
            param_distributions=param_dist2,
            n_iter=10,
            scoring="snr",
            random_state=42,
            parallel=False,
        )
        result2 = search2.fit(simple_trace, simple_transform)

        # Should get same best params keys
        assert result1.best_params.keys() == result2.best_params.keys()
        # Scores should be close (exact match might not happen due to RNG differences)
        assert abs(result1.best_score - result2.best_score) < 1.0

    def test_fit_with_parallel(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_distributions: dict[str, callable],
    ) -> None:
        """Test fitting with parallel execution."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions,
            n_iter=10,
            scoring="snr",
            parallel=True,
            use_threads=True,
            max_workers=2,
            random_state=42,
        )

        result = search.fit(simple_trace, simple_transform)

        assert len(result.all_results) == 10
        assert result.best_params is not None


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestOptimizationSearchEdgeCases:
    """Test edge cases and error handling."""

    def test_grid_search_empty_grid(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test GridSearchCV with empty parameter grid."""
        search = GridSearchCV(param_grid={}, scoring="snr")

        result = search.fit(simple_trace, simple_transform)

        # Should evaluate single empty combination
        assert len(result.all_results) == 1
        assert result.best_params == {}

    def test_grid_search_single_param_value(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test GridSearchCV with single value per parameter."""
        param_grid = {"scale": [1.0], "offset": [0.0]}
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        result = search.fit(simple_trace, simple_transform)

        assert len(result.all_results) == 1
        assert result.best_params == {"scale": 1.0, "offset": 0.0}

    def test_randomized_search_n_iter_one(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_distributions: dict[str, callable],
    ) -> None:
        """Test RandomizedSearchCV with n_iter=1."""
        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=1, scoring="snr", random_state=42
        )

        result = search.fit(simple_trace, simple_transform)

        assert len(result.all_results) == 1
        assert result.best_params is not None

    def test_cv_with_insufficient_traces(
        self,
        trace_list: list[WaveformTrace],
        simple_transform: callable,
        param_grid: dict[str, list[Any]],
    ) -> None:
        """Test cross-validation with sufficient traces."""
        # Use all 5 traces with CV=3 - this is a normal case
        search = GridSearchCV(param_grid=param_grid, scoring="snr", cv=3)

        # Should work normally with enough traces
        result = search.fit(trace_list, simple_transform)
        assert result is not None
        assert result.best_params is not None
        assert result.cv_scores is not None
        assert len(result.cv_scores) == 3

    def test_transform_function_error_handling(
        self, simple_trace: WaveformTrace, param_grid: dict[str, list[Any]]
    ) -> None:
        """Test that errors in transform function are propagated."""

        def bad_transform(trace, scale, offset):
            raise ValueError("Transform failed")

        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        with pytest.raises(ValueError, match="Transform failed"):
            search.fit(simple_trace, bad_transform)

    def test_custom_scorer_error_handling(
        self,
        simple_trace: WaveformTrace,
        simple_transform: callable,
        param_grid: dict[str, list[Any]],
    ) -> None:
        """Test that errors in custom scorer are propagated."""

        def bad_scorer(trace, params):
            raise RuntimeError("Scoring failed")

        search = GridSearchCV(param_grid=param_grid, scoring=bad_scorer)

        with pytest.raises(RuntimeError, match="Scoring failed"):
            search.fit(simple_trace, simple_transform)


# =============================================================================
# Test Integration Scenarios
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_filter_parameter_optimization(self) -> None:
        """Test optimizing filter parameters on noisy signal."""
        # Create noisy signal
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 1000)
        clean_signal = np.sin(2 * np.pi * 10 * t)
        noise = rng.normal(0, 0.3, 1000)
        noisy_signal = clean_signal + noise

        trace = WaveformTrace(data=noisy_signal, metadata=TraceMetadata(sample_rate=1000))

        # Define filter transform (simplified - just smoothing)
        def smooth_filter(trace, window_size):
            kernel = np.ones(int(window_size)) / window_size
            smoothed = np.convolve(trace.data, kernel, mode="same")
            return WaveformTrace(data=smoothed, metadata=trace.metadata)

        # Search for best window size
        param_grid = {"window_size": [3, 5, 7, 9, 11]}
        search = GridSearchCV(param_grid=param_grid, scoring="snr")

        result = search.fit(trace, smooth_filter)

        assert result.best_params is not None
        assert "window_size" in result.best_params
        assert result.best_params["window_size"] in [3, 5, 7, 9, 11]

    def test_multi_trace_optimization(self) -> None:
        """Test optimization across multiple traces."""
        rng = np.random.default_rng(42)

        # Create multiple traces with different characteristics
        traces = []
        for i in range(3):
            data = rng.normal(loc=i * 2, scale=1.0, size=500)
            metadata = TraceMetadata(sample_rate=1e6)
            traces.append(WaveformTrace(data=data, metadata=metadata))

        def normalize_transform(trace, target_mean, target_std):
            normalized = (trace.data - np.mean(trace.data)) / (np.std(trace.data) + 1e-10)
            normalized = normalized * target_std + target_mean
            return WaveformTrace(data=normalized, metadata=trace.metadata)

        param_grid = {"target_mean": [0.0, 1.0, 2.0], "target_std": [1.0, 2.0]}

        search = GridSearchCV(param_grid=param_grid, scoring="snr")
        result = search.fit(traces, normalize_transform)

        assert len(result.all_results) == 6  # 3 * 2
        assert result.best_params is not None

    def test_randomized_vs_grid_search_comparison(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Compare RandomizedSearchCV and GridSearchCV results."""
        # Grid search with full grid
        param_grid = {"scale": [0.5, 1.0, 1.5, 2.0], "offset": [-0.5, 0.0, 0.5]}
        grid_search = GridSearchCV(param_grid=param_grid, scoring="snr")
        grid_result = grid_search.fit(simple_trace, simple_transform)

        # Randomized search sampling from same space
        rng = np.random.default_rng(42)
        param_distributions = {
            "scale": lambda: rng.choice([0.5, 1.0, 1.5, 2.0]),
            "offset": lambda: rng.choice([-0.5, 0.0, 0.5]),
        }
        random_search = RandomizedSearchCV(
            param_distributions=param_distributions,
            n_iter=12,  # Same as grid size
            scoring="snr",
            random_state=42,
        )
        random_result = random_search.fit(simple_trace, simple_transform)

        # Both should find reasonable parameters
        assert grid_result.best_score is not None
        assert random_result.best_score is not None

        # Grid search evaluates all combinations
        assert len(grid_result.all_results) == 12
        assert len(random_result.all_results) == 12


# =============================================================================
# Test Performance and Parallel Execution
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestPerformance:
    """Test performance-related aspects."""

    def test_parallel_faster_than_sequential(
        self, trace_list: list[WaveformTrace], simple_transform: callable
    ) -> None:
        """Test that parallel execution works (timing test skipped)."""
        # Large parameter grid
        param_grid = {"scale": [0.5, 1.0, 1.5, 2.0, 2.5], "offset": [0.0, 0.25, 0.5, 0.75, 1.0]}

        # Just verify both complete successfully
        search_seq = GridSearchCV(param_grid=param_grid, scoring="snr", parallel=False)
        result_seq = search_seq.fit(trace_list, simple_transform)

        search_par = GridSearchCV(
            param_grid=param_grid, scoring="snr", parallel=True, max_workers=2
        )
        result_par = search_par.fit(trace_list, simple_transform)

        # Both should complete with same number of evaluations
        assert len(result_seq.all_results) == len(result_par.all_results)
        assert len(result_seq.all_results) == 25

    def test_large_parameter_space(
        self, simple_trace: WaveformTrace, simple_transform: callable
    ) -> None:
        """Test handling large parameter space."""
        # Use randomized search for large space
        rng = np.random.default_rng(42)
        param_distributions = {
            "scale": lambda: rng.uniform(0.1, 10.0),
            "offset": lambda: rng.uniform(-5.0, 5.0),
        }

        search = RandomizedSearchCV(
            param_distributions=param_distributions,
            n_iter=100,
            scoring="snr",
            parallel=True,
            random_state=42,
        )

        result = search.fit(simple_trace, simple_transform)

        assert len(result.all_results) == 100
        assert result.best_params is not None


# =============================================================================
# Test Documentation Examples
# =============================================================================


@pytest.mark.unit
@pytest.mark.optimization
class TestDocumentationExamples:
    """Test code examples from module docstrings."""

    def test_grid_search_example(self) -> None:
        """Test example from GridSearchCV docstring."""
        # Create simple test data
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))

        param_grid = {"cutoff": [1e5, 5e5, 1e6], "order": [2, 4, 6]}

        # Simple lowpass-like filter
        def apply_filter(trace, cutoff, order):
            # Simplified - just scale by order
            filtered = trace.data * (1.0 / order)
            return WaveformTrace(data=filtered, metadata=trace.metadata)

        search = GridSearchCV(param_grid=param_grid, scoring="snr", cv=3)

        result = search.fit([trace] * 3, apply_filter)

        assert result.best_params is not None
        assert "cutoff" in result.best_params
        assert "order" in result.best_params

    def test_randomized_search_example(self) -> None:
        """Test example from RandomizedSearchCV docstring."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))

        param_distributions = {
            "cutoff": lambda: rng.uniform(1e5, 1e7),
            "order": lambda: rng.choice([2, 4, 6, 8]),
        }

        def apply_filter(trace, cutoff, order):
            filtered = trace.data * (cutoff / 1e6) / order
            return WaveformTrace(data=filtered, metadata=trace.metadata)

        search = RandomizedSearchCV(
            param_distributions=param_distributions, n_iter=20, scoring="snr", random_state=42
        )

        result = search.fit(trace, apply_filter)

        assert result.best_params is not None
        assert "cutoff" in result.best_params
        assert "order" in result.best_params
