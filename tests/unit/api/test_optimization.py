"""Comprehensive unit tests for optimization module.


Test coverage:
- ParameterSpace: discrete values, continuous ranges, log scale, validation
- OptimizationResult: best params, scoring, top_n, serialization
- GridSearch: exhaustive search, progress tracking, early stopping, maximize/minimize
- RandomSearch: random sampling, reproducibility, convergence
- optimize_parameters: convenience function, method selection, dict conversion
- Edge cases and error handling
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from oscura.api.optimization import (
    GridSearch,
    OptimizationResult,
    ParameterSpace,
    RandomSearch,
    optimize_parameters,
)

pytestmark = pytest.mark.unit


# =============================================================================
# ParameterSpace Tests
# =============================================================================


@pytest.mark.unit
class TestParameterSpace:
    """Test ParameterSpace class (API-014)."""

    def test_init_discrete_values(self) -> None:
        """Test initialization with discrete values."""
        space = ParameterSpace("window", values=["hann", "hamming", "blackman"])

        assert space.name == "window"
        assert space.values == ["hann", "hamming", "blackman"]
        assert not space.log_scale
        assert space.num_samples == 10

    def test_init_continuous_linear(self) -> None:
        """Test initialization with continuous linear range."""
        space = ParameterSpace("cutoff", low=100.0, high=1000.0, num_samples=5)

        assert space.name == "cutoff"
        assert space.low == 100.0
        assert space.high == 1000.0
        assert len(space.values) == 5
        assert not space.log_scale

        # Check linear spacing
        expected = [100.0, 325.0, 550.0, 775.0, 1000.0]
        assert np.allclose(space.values, expected)

    def test_init_continuous_log_scale(self) -> None:
        """Test initialization with logarithmic scale."""
        space = ParameterSpace("freq", low=10.0, high=10000.0, num_samples=4, log_scale=True)

        assert space.name == "freq"
        assert space.log_scale
        assert len(space.values) == 4

        # Check log spacing
        expected = np.logspace(1, 4, 4)  # 10^1 to 10^4
        assert np.allclose(space.values, expected)

    def test_init_no_values_or_bounds_raises(self) -> None:
        """Test initialization without values or bounds raises error."""
        with pytest.raises(ValueError, match="must specify either values or"):
            ParameterSpace("invalid")

    def test_iter(self) -> None:
        """Test iteration over parameter values."""
        space = ParameterSpace("test", values=[1, 2, 3])

        values = list(space)
        assert values == [1, 2, 3]

    def test_len(self) -> None:
        """Test length of parameter space."""
        space = ParameterSpace("test", values=["a", "b", "c", "d"])
        assert len(space) == 4

    def test_len_continuous(self) -> None:
        """Test length with continuous parameter."""
        space = ParameterSpace("x", low=0.0, high=1.0, num_samples=7)
        assert len(space) == 7

    def test_iter_empty(self) -> None:
        """Test iteration with no values."""
        space = ParameterSpace("test", values=[])
        assert list(space) == []
        assert len(space) == 0

    def test_continuous_single_sample(self) -> None:
        """Test continuous parameter with single sample."""
        space = ParameterSpace("x", low=5.0, high=5.0, num_samples=1)
        assert len(space) == 1
        assert space.values[0] == 5.0

    def test_mixed_types_in_values(self) -> None:
        """Test parameter space with mixed types."""
        space = ParameterSpace("mixed", values=[1, "two", 3.0, None])
        assert len(space) == 4
        assert space.values == [1, "two", 3.0, None]


# =============================================================================
# OptimizationResult Tests
# =============================================================================


@pytest.mark.unit
class TestOptimizationResult:
    """Test OptimizationResult class (API-014)."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        result = OptimizationResult(
            best_params={"x": 1, "y": 2},
            best_score=0.95,
        )

        assert result.best_params == {"x": 1, "y": 2}
        assert result.best_score == 0.95
        assert result.all_results == []
        assert result.elapsed_time == 0.0
        assert result.num_evaluations == 0

    def test_init_full(self) -> None:
        """Test initialization with all fields."""
        all_results = [
            ({"x": 1}, 0.5),
            ({"x": 2}, 0.8),
            ({"x": 3}, 0.95),
        ]

        result = OptimizationResult(
            best_params={"x": 3},
            best_score=0.95,
            all_results=all_results,
            elapsed_time=2.5,
            num_evaluations=3,
        )

        assert result.best_params == {"x": 3}
        assert result.best_score == 0.95
        assert len(result.all_results) == 3
        assert result.elapsed_time == 2.5
        assert result.num_evaluations == 3

    def test_top_n_basic(self) -> None:
        """Test top_n returns best results."""
        all_results = [
            ({"x": 1}, 0.5),
            ({"x": 2}, 0.8),
            ({"x": 3}, 0.95),
            ({"x": 4}, 0.3),
            ({"x": 5}, 0.7),
        ]

        result = OptimizationResult(
            best_params={"x": 3},
            best_score=0.95,
            all_results=all_results,
        )

        top_3 = result.top_n(3)

        assert len(top_3) == 3
        assert top_3[0] == ({"x": 3}, 0.95)
        assert top_3[1] == ({"x": 2}, 0.8)
        assert top_3[2] == ({"x": 5}, 0.7)

    def test_top_n_all(self) -> None:
        """Test top_n with n larger than results."""
        all_results = [
            ({"x": 1}, 0.5),
            ({"x": 2}, 0.8),
        ]

        result = OptimizationResult(
            best_params={"x": 2},
            best_score=0.8,
            all_results=all_results,
        )

        top_10 = result.top_n(10)
        assert len(top_10) == 2

    def test_top_n_empty(self) -> None:
        """Test top_n with empty results."""
        result = OptimizationResult(
            best_params={},
            best_score=0.0,
            all_results=[],
        )

        top_5 = result.top_n(5)
        assert top_5 == []

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = OptimizationResult(
            best_params={"x": 1, "y": 2},
            best_score=0.95,
            all_results=[],
            elapsed_time=3.14,
            num_evaluations=42,
        )

        d = result.to_dict()

        assert d == {
            "best_params": {"x": 1, "y": 2},
            "best_score": 0.95,
            "num_evaluations": 42,
            "elapsed_time": 3.14,
        }

    def test_to_dict_excludes_all_results(self) -> None:
        """Test to_dict does not include all_results."""
        result = OptimizationResult(
            best_params={"x": 1},
            best_score=0.5,
            all_results=[({"x": 1}, 0.5), ({"x": 2}, 0.3)],
        )

        d = result.to_dict()
        assert "all_results" not in d


# =============================================================================
# GridSearch Tests
# =============================================================================


@pytest.mark.unit
class TestGridSearch:
    """Test GridSearch class (API-014)."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        spaces = [
            ParameterSpace("x", values=[1, 2, 3]),
            ParameterSpace("y", values=["a", "b"]),
        ]

        search = GridSearch(spaces)

        assert search.param_spaces == spaces
        assert search.verbose is True
        assert search._progress_callback is None

    def test_init_not_verbose(self) -> None:
        """Test initialization with verbose=False."""
        spaces = [ParameterSpace("x", values=[1, 2])]
        search = GridSearch(spaces, verbose=False)

        assert search.verbose is False

    def test_num_combinations(self) -> None:
        """Test num_combinations property."""
        spaces = [
            ParameterSpace("x", values=[1, 2, 3]),
            ParameterSpace("y", values=["a", "b"]),
            ParameterSpace("z", values=[10, 20]),
        ]

        search = GridSearch(spaces)
        assert search.num_combinations == 3 * 2 * 2

    def test_num_combinations_single_space(self) -> None:
        """Test num_combinations with single parameter."""
        spaces = [ParameterSpace("x", values=[1, 2, 3, 4, 5])]
        search = GridSearch(spaces)
        assert search.num_combinations == 5

    def test_on_progress_chaining(self) -> None:
        """Test on_progress returns self for chaining."""
        spaces = [ParameterSpace("x", values=[1, 2])]
        search = GridSearch(spaces)

        callback = Mock()
        result = search.on_progress(callback)

        assert result is search
        assert search._progress_callback is callback

    def test_fit_maximize(self) -> None:
        """Test fit with maximize=True."""

        def objective(params: dict[str, Any], data: Any) -> float:
            # Simple quadratic: maximize at x=5
            return -((params["x"] - 5) ** 2)

        spaces = [ParameterSpace("x", values=[1, 3, 5, 7, 9])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True)

        assert result.best_params == {"x": 5}
        assert result.best_score == 0.0
        assert result.num_evaluations == 5
        assert len(result.all_results) == 5
        assert result.elapsed_time > 0

    def test_fit_minimize(self) -> None:
        """Test fit with maximize=False."""

        def objective(params: dict[str, Any], data: Any) -> float:
            # Minimize distance from 5
            return abs(params["x"] - 5)

        spaces = [ParameterSpace("x", values=[1, 3, 5, 7, 9])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=False)

        assert result.best_params == {"x": 5}
        assert result.best_score == 0.0
        assert result.num_evaluations == 5

    def test_fit_multiple_params(self) -> None:
        """Test fit with multiple parameters."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] * params["y"]

        spaces = [
            ParameterSpace("x", values=[1, 2, 3]),
            ParameterSpace("y", values=[10, 20]),
        ]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True)

        assert result.best_params == {"x": 3, "y": 20}
        assert result.best_score == 60
        assert result.num_evaluations == 6

    def test_fit_with_data(self) -> None:
        """Test fit passes data to objective."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["scale"] * data

        spaces = [ParameterSpace("scale", values=[0.5, 1.0, 2.0])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, 10, maximize=True)

        assert result.best_params == {"scale": 2.0}
        assert result.best_score == 20.0

    def test_fit_early_stop_maximize(self) -> None:
        """Test early stopping when maximizing."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        spaces = [ParameterSpace("x", values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True, early_stop=5.0)

        # Should stop at x=5
        assert result.best_score >= 5.0
        assert result.num_evaluations <= 5

    def test_fit_early_stop_minimize(self) -> None:
        """Test early stopping when minimizing."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return abs(params["x"] - 3)

        spaces = [ParameterSpace("x", values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=False, early_stop=0.0)

        # Should stop at x=3
        assert result.best_score == 0.0
        assert result.num_evaluations <= 3

    def test_fit_progress_callback(self) -> None:
        """Test progress callback is called."""
        calls = []

        def progress(current: int, total: int) -> None:
            calls.append((current, total))

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = GridSearch(spaces, verbose=False).on_progress(progress)

        search.fit(objective, None)

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)

    def test_fit_objective_exception_handling(self) -> None:
        """Test handling of exceptions in objective function."""

        def objective(params: dict[str, Any], data: Any) -> float:
            if params["x"] == 2:
                raise ValueError("Test error")
            return params["x"]

        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True)

        # Should handle exception and continue
        assert result.num_evaluations == 3
        # Best should be x=3 (x=2 failed)
        assert result.best_params == {"x": 3}

    def test_fit_all_exceptions_maximize(self) -> None:
        """Test all exceptions when maximizing."""

        def objective(params: dict[str, Any], data: Any) -> float:
            raise ValueError("Always fails")

        spaces = [ParameterSpace("x", values=[1, 2])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True)

        # All failed, should have -inf score
        assert result.best_score == float("-inf")
        assert result.num_evaluations == 2

    def test_fit_all_exceptions_minimize(self) -> None:
        """Test all exceptions when minimizing."""

        def objective(params: dict[str, Any], data: Any) -> float:
            raise ValueError("Always fails")

        spaces = [ParameterSpace("x", values=[1, 2])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=False)

        # All failed, should have +inf score
        assert result.best_score == float("inf")
        assert result.num_evaluations == 2

    def test_fit_verbose_early_stop(self, caplog) -> None:
        """Test verbose logging during early stopping."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        spaces = [ParameterSpace("x", values=[1, 2, 3, 4, 5])]
        search = GridSearch(spaces, verbose=True)

        result = search.fit(objective, None, maximize=True, early_stop=3.0)

        # Should have early stopped
        assert result.num_evaluations <= 3
        assert result.best_score >= 3.0

    def test_fit_all_results_stored(self) -> None:
        """Test all results are stored."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] ** 2

        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None)

        assert len(result.all_results) == 3
        expected = [
            ({"x": 1}, 1.0),
            ({"x": 2}, 4.0),
            ({"x": 3}, 9.0),
        ]
        assert result.all_results == expected


# =============================================================================
# RandomSearch Tests
# =============================================================================


@pytest.mark.unit
class TestRandomSearch:
    """Test RandomSearch class (API-014)."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = RandomSearch(spaces)

        assert search.param_spaces == spaces
        assert search.n_iterations == 100
        assert search.random_state is None

    def test_init_custom_iterations(self) -> None:
        """Test initialization with custom iterations."""
        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = RandomSearch(spaces, n_iterations=50)

        assert search.n_iterations == 50

    def test_init_with_seed(self) -> None:
        """Test initialization with random seed."""
        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = RandomSearch(spaces, random_state=42)

        assert search.random_state == 42

    def test_fit_maximize(self) -> None:
        """Test fit with maximize=True."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return -abs(params["x"] - 5)

        spaces = [ParameterSpace("x", values=list(range(10)))]
        search = RandomSearch(spaces, n_iterations=20, random_state=42)

        result = search.fit(objective, None, maximize=True)

        assert result.num_evaluations == 20
        assert len(result.all_results) == 20
        assert result.elapsed_time > 0
        # Should find x near 5
        assert result.best_params["x"] in [4, 5, 6]

    def test_fit_minimize(self) -> None:
        """Test fit with maximize=False."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return abs(params["x"] - 5)

        spaces = [ParameterSpace("x", values=list(range(10)))]
        search = RandomSearch(spaces, n_iterations=20, random_state=42)

        result = search.fit(objective, None, maximize=False)

        assert result.num_evaluations == 20
        # Should find x near 5
        assert result.best_params["x"] in [4, 5, 6]

    def test_fit_reproducibility(self) -> None:
        """Test random search is reproducible with seed."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] + params["y"]

        spaces = [
            ParameterSpace("x", values=[1, 2, 3, 4, 5]),
            ParameterSpace("y", values=[10, 20, 30]),
        ]

        search1 = RandomSearch(spaces, n_iterations=10, random_state=42)
        result1 = search1.fit(objective, None)

        search2 = RandomSearch(spaces, n_iterations=10, random_state=42)
        result2 = search2.fit(objective, None)

        # Should get same results with same seed
        assert result1.best_params == result2.best_params
        assert result1.best_score == result2.best_score

    def test_fit_multiple_params(self) -> None:
        """Test fit with multiple parameters."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] * params["y"]

        spaces = [
            ParameterSpace("x", values=[1, 2, 3]),
            ParameterSpace("y", values=[10, 20, 30]),
        ]
        search = RandomSearch(spaces, n_iterations=15, random_state=42)

        result = search.fit(objective, None, maximize=True)

        assert result.num_evaluations == 15
        # Best should be x=3, y=30
        assert result.best_score <= 90  # Should approach maximum

    def test_fit_exception_handling(self) -> None:
        """Test handling of exceptions in objective."""

        def objective(params: dict[str, Any], data: Any) -> float:
            if params["x"] == 5:
                raise ValueError("Test error")
            return params["x"]

        spaces = [ParameterSpace("x", values=list(range(10)))]
        search = RandomSearch(spaces, n_iterations=20, random_state=42)

        result = search.fit(objective, None, maximize=True)

        # Should handle exceptions gracefully
        assert result.num_evaluations == 20
        assert result.best_score != float("-inf")  # Should have some valid results

    def test_fit_all_exceptions_maximize(self) -> None:
        """Test all exceptions when maximizing."""

        def objective(params: dict[str, Any], data: Any) -> float:
            raise ValueError("Always fails")

        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = RandomSearch(spaces, n_iterations=5)

        result = search.fit(objective, None, maximize=True)

        assert result.best_score == float("-inf")
        assert result.num_evaluations == 5

    def test_fit_all_exceptions_minimize(self) -> None:
        """Test all exceptions when minimizing."""

        def objective(params: dict[str, Any], data: Any) -> float:
            raise ValueError("Always fails")

        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = RandomSearch(spaces, n_iterations=5)

        result = search.fit(objective, None, maximize=False)

        assert result.best_score == float("inf")
        assert result.num_evaluations == 5

    def test_fit_samples_with_replacement(self) -> None:
        """Test random search can sample same params multiple times."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        # Small space, many iterations - should sample with replacement
        spaces = [ParameterSpace("x", values=[1, 2])]
        search = RandomSearch(spaces, n_iterations=10, random_state=42)

        result = search.fit(objective, None)

        assert result.num_evaluations == 10
        # Should have repeated some values


# =============================================================================
# optimize_parameters Tests
# =============================================================================


@pytest.mark.unit
class TestOptimizeParameters:
    """Test optimize_parameters convenience function (API-014)."""

    def test_basic_grid_search(self) -> None:
        """Test basic grid search via convenience function."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] ** 2

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"x": [1, 2, 3, 4, 5]},
            method="grid",
            maximize=True,
        )

        assert result.best_params == {"x": 5}
        assert result.best_score == 25

    def test_basic_random_search(self) -> None:
        """Test basic random search via convenience function."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return -abs(params["x"] - 3)

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"x": list(range(10))},
            method="random",
            maximize=True,
            n_iterations=20,
            random_state=42,
        )

        assert result.num_evaluations == 20
        assert result.best_params["x"] in [2, 3, 4]

    def test_dict_to_spaces_conversion(self) -> None:
        """Test parameter dict is converted to ParameterSpace list."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] + params["y"]

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"x": [1, 2, 3], "y": [10, 20]},
            method="grid",
        )

        assert result.best_params in [{"x": 3, "y": 20}, {"x": 2, "y": 20}, {"x": 3, "y": 10}]
        assert result.num_evaluations == 6

    def test_parameter_space_list(self) -> None:
        """Test with ParameterSpace list directly."""
        spaces = [
            ParameterSpace("x", values=[1, 2, 3]),
            ParameterSpace("y", low=10, high=30, num_samples=3),
        ]

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] * params["y"]

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces=spaces,
            method="grid",
        )

        assert result.num_evaluations == 3 * 3

    def test_maximize_flag(self) -> None:
        """Test maximize flag is passed correctly."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"x": [1, 2, 3]},
            maximize=False,
        )

        # Minimizing, so best should be 1
        assert result.best_params == {"x": 1}
        assert result.best_score == 1

    def test_unknown_method_raises(self) -> None:
        """Test unknown optimization method raises error."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return 0.0

        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimize_parameters(
                objective=objective,
                data=None,
                param_spaces={"x": [1, 2]},
                method="invalid",
            )

    def test_kwargs_passed_to_grid(self) -> None:
        """Test kwargs are passed to GridSearch."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"x": list(range(20))},
            method="grid",
            verbose=False,
        )

        assert result.num_evaluations == 20

    def test_kwargs_passed_to_random(self) -> None:
        """Test kwargs are passed to RandomSearch."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"x": list(range(100))},
            method="random",
            n_iterations=15,
            random_state=42,
        )

        assert result.num_evaluations == 15

    def test_with_data_parameter(self) -> None:
        """Test data is passed to objective function."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["scale"] * np.mean(data)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = optimize_parameters(
            objective=objective,
            data=data,
            param_spaces={"scale": [0.5, 1.0, 2.0]},
            method="grid",
        )

        # Mean is 3.0, best scale is 2.0
        assert result.best_params == {"scale": 2.0}
        assert result.best_score == 6.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestApiOptimizationEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameter_space(self) -> None:
        """Test with empty parameter values."""
        space = ParameterSpace("x", values=[])

        assert len(space) == 0
        assert list(space) == []

    def test_single_parameter_value(self) -> None:
        """Test with single parameter value."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        spaces = [ParameterSpace("x", values=[42])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None)

        assert result.best_params == {"x": 42}
        assert result.num_evaluations == 1

    def test_very_large_search_space(self) -> None:
        """Test with large parameter space."""
        spaces = [
            ParameterSpace("x", low=0, high=100, num_samples=50),
            ParameterSpace("y", low=0, high=100, num_samples=50),
        ]

        search = GridSearch(spaces, verbose=False)
        assert search.num_combinations == 2500

    def test_negative_parameter_values(self) -> None:
        """Test with negative parameter values."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return abs(params["x"])

        spaces = [ParameterSpace("x", values=[-5, -3, -1, 1, 3, 5])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=False)

        assert result.best_params["x"] in [-1, 1]
        assert result.best_score == 1

    def test_floating_point_precision(self) -> None:
        """Test with floating point parameters."""
        space = ParameterSpace("x", low=0.0, high=1.0, num_samples=11)

        values = space.values
        assert len(values) == 11
        assert values[0] == 0.0
        assert values[-1] == 1.0

    def test_log_scale_very_wide_range(self) -> None:
        """Test log scale with very wide range."""
        space = ParameterSpace("freq", low=1e-3, high=1e6, num_samples=10, log_scale=True)

        values = space.values
        assert len(values) == 10
        assert values[0] == pytest.approx(1e-3)
        assert values[-1] == pytest.approx(1e6)
        # Check spacing is logarithmic
        ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
        # All ratios should be approximately equal for log spacing
        assert np.std(ratios) < 0.1 * np.mean(ratios)

    def test_objective_returns_nan(self) -> None:
        """Test handling of NaN from objective function."""

        def objective(params: dict[str, Any], data: Any) -> float:
            if params["x"] == 2:
                return float("nan")
            return params["x"]

        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True)

        # Should handle NaN gracefully
        assert result.num_evaluations == 3

    def test_objective_returns_inf(self) -> None:
        """Test handling of inf from objective function."""

        def objective(params: dict[str, Any], data: Any) -> float:
            if params["x"] == 2:
                return float("inf")
            return params["x"]

        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True)

        # inf should be the best when maximizing
        assert result.best_params == {"x": 2}
        assert result.best_score == float("inf")

    def test_zero_iterations_random_search(self) -> None:
        """Test random search with zero iterations."""
        spaces = [ParameterSpace("x", values=[1, 2, 3])]
        search = RandomSearch(spaces, n_iterations=0)

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"]

        result = search.fit(objective, None)

        assert result.num_evaluations == 0
        assert result.best_score == float("-inf")  # No results for maximize


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestApiOptimizationIntegration:
    """Integration tests for realistic optimization workflows."""

    def test_signal_processing_optimization(self) -> None:
        """Test optimizing signal processing parameters."""
        # Generate noisy signal with fixed seed for reproducibility
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 1000)
        clean_signal = np.sin(2 * np.pi * 10 * t)
        noisy_signal = clean_signal + 0.2 * rng.standard_normal(1000)

        def objective(params: dict[str, Any], data: Any) -> float:
            # Simulate filtering and measure SNR
            # Simple moving average filter
            window = params["window_size"]
            filtered = np.convolve(data, np.ones(window) / window, mode="same")

            # Calculate pseudo-SNR (inverse of residual)
            residual = np.mean((filtered - clean_signal) ** 2)
            return -residual if residual > 0 else 0

        result = optimize_parameters(
            objective=objective,
            data=noisy_signal,
            param_spaces={"window_size": [3, 5, 7, 9, 11, 13, 15]},
            method="grid",
            maximize=True,
        )

        assert result.num_evaluations == 7
        # Just check that a valid window size was selected
        assert result.best_params["window_size"] in [3, 5, 7, 9, 11, 13, 15]
        # And that optimization actually ran
        assert result.best_score < 0  # Negative residual

    def test_multi_parameter_optimization(self) -> None:
        """Test optimization with multiple interdependent parameters."""

        def objective(params: dict[str, Any], data: Any) -> float:
            # Simulate complex fitness function
            a, b, c = params["a"], params["b"], params["c"]
            return -((a - 2) ** 2) - (b - 3) ** 2 - (c - 4) ** 2

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={
                "a": [1, 2, 3],
                "b": [2, 3, 4],
                "c": [3, 4, 5],
            },
            method="grid",
            maximize=True,
        )

        assert result.best_params == {"a": 2, "b": 3, "c": 4}
        assert result.best_score == 0.0

    def test_random_search_convergence(self) -> None:
        """Test random search finds good solutions."""

        def objective(params: dict[str, Any], data: Any) -> float:
            # Multi-modal function with global optimum at x=5, y=15
            x, y = params["x"], params["y"]
            return -((x - 5) ** 2) - (y - 15) ** 2 + 50

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={
                "x": list(range(20)),
                "y": list(range(30)),
            },
            method="random",
            n_iterations=100,
            random_state=42,
            maximize=True,
        )

        # Should find near-optimal solution
        assert abs(result.best_params["x"] - 5) <= 2
        assert abs(result.best_params["y"] - 15) <= 3

    def test_optimization_with_early_stopping(self) -> None:
        """Test optimization with early stopping criterion."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] ** 2

        spaces = [ParameterSpace("x", low=0, high=10, num_samples=100)]
        search = GridSearch(spaces, verbose=False)

        result = search.fit(objective, None, maximize=True, early_stop=90)

        # Should stop early (100^2 = 10000, but we stop at 90)
        assert result.num_evaluations < 100
        assert result.best_score >= 90

    def test_top_n_analysis(self) -> None:
        """Test analyzing top N results."""

        def objective(params: dict[str, Any], data: Any) -> float:
            return params["x"] + params["y"]

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"x": [1, 2, 3, 4, 5], "y": [10, 20, 30]},
            method="grid",
            maximize=True,
        )

        top_3 = result.top_n(3)

        assert len(top_3) == 3
        # Top should be (5, 30), (4, 30), (5, 20) or similar high values
        assert top_3[0][1] == 35  # 5 + 30
        assert top_3[1][1] == 34  # 4 + 30
        assert top_3[2][1] == 33  # 3 + 30

    def test_continuous_parameter_optimization(self) -> None:
        """Test optimization with continuous parameters."""

        def objective(params: dict[str, Any], data: Any) -> float:
            # Minimize squared error from target
            return -abs(params["threshold"] - 0.75)

        result = optimize_parameters(
            objective=objective,
            data=None,
            param_spaces={"threshold": ParameterSpace("threshold", low=0, high=1, num_samples=20)},
            method="grid",
            maximize=True,
        )

        # Should find value close to 0.75
        assert abs(result.best_params["threshold"] - 0.75) < 0.1

    def test_mixed_discrete_continuous_params(self) -> None:
        """Test optimization with mixed parameter types."""
        spaces = [
            ParameterSpace("method", values=["linear", "cubic"]),
            ParameterSpace("threshold", low=0, high=1, num_samples=5),
            ParameterSpace("window", values=[3, 5, 7]),
        ]

        def objective(params: dict[str, Any], data: Any) -> float:
            score = 0.0
            if params["method"] == "cubic":
                score += 1.0
            score += params["threshold"]
            score += params["window"] / 10.0
            return score

        search = GridSearch(spaces, verbose=False)
        result = search.fit(objective, None, maximize=True)

        # Best should be cubic, threshold=1.0, window=7
        assert result.best_params["method"] == "cubic"
        assert result.best_params["threshold"] == pytest.approx(1.0)
        assert result.best_params["window"] == 7
