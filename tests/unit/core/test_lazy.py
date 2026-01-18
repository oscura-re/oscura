"""Tests for lazy evaluation module.

Tests deferred computation, thread safety, and memory-efficient workflows.

Requirements tested:
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from oscura.core.lazy import (
    LazyAnalysisResult,
    LazyComputeStats,
    LazyDict,
    LazyResult,
    get_lazy_stats,
    lazy,
    reset_lazy_stats,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestLazyComputeStats:
    """Test LazyComputeStats statistics tracking."""

    def test_create_stats(self) -> None:
        """Test creating compute stats."""
        stats = LazyComputeStats(
            total_created=100,
            total_computed=80,
            total_invalidated=5,
            compute_time_total=12.5,
            cache_hits=120,
        )
        assert stats.total_created == 100
        assert stats.total_computed == 80
        assert stats.cache_hits == 120

    def test_hit_rate_calculation(self) -> None:
        """Test cache hit rate calculation."""
        stats = LazyComputeStats(
            total_computed=20,
            cache_hits=80,
        )
        # Total accesses = 20 (computed) + 80 (hits) = 100
        # Hit rate = 80 / 100 = 0.8
        assert stats.hit_rate == pytest.approx(0.8)

    def test_hit_rate_zero_accesses(self) -> None:
        """Test hit rate with no accesses."""
        stats = LazyComputeStats()
        assert stats.hit_rate == 0.0

    def test_str_formatting(self) -> None:
        """Test string formatting."""
        stats = LazyComputeStats(
            total_created=50,
            total_computed=40,
            cache_hits=60,
            compute_time_total=5.5,
        )
        output = str(stats)
        assert "Created: 50" in output
        assert "Computed: 40" in output
        assert "Cache Hits: 60" in output
        assert "5.500s" in output


class TestLazyResult:
    """Test LazyResult deferred computation."""

    def test_deferred_computation(self) -> None:
        """Test that computation is deferred until access."""
        compute_fn = MagicMock(return_value=42)
        lazy_result = LazyResult(compute_fn, name="test")

        # Should not compute yet
        assert not lazy_result.is_computed()
        compute_fn.assert_not_called()

        # Access triggers computation
        value = lazy_result.value
        assert value == 42
        assert lazy_result.is_computed()
        compute_fn.assert_called_once()

    def test_compute_once_semantics(self) -> None:
        """Test that computation happens only once."""
        compute_fn = MagicMock(return_value=42)
        lazy_result = LazyResult(compute_fn)

        # Access multiple times
        value1 = lazy_result.value
        value2 = lazy_result.value
        value3 = lazy_result.value

        assert value1 == value2 == value3 == 42
        # Should only compute once
        compute_fn.assert_called_once()

    def test_invalidate(self) -> None:
        """Test invalidation for delta analysis."""
        call_count = 0

        def compute_fn() -> int:
            nonlocal call_count
            call_count += 1
            return call_count * 10

        lazy_result = LazyResult(compute_fn)

        # First computation
        value1 = lazy_result.value
        assert value1 == 10
        assert lazy_result.is_computed()

        # Invalidate and recompute
        lazy_result.invalidate()
        assert not lazy_result.is_computed()

        value2 = lazy_result.value
        assert value2 == 20  # New computation
        assert lazy_result.is_computed()

    def test_get_if_computed(self) -> None:
        """Test getting result only if computed."""
        lazy_result = LazyResult(lambda: 42)

        # Not computed yet
        result = lazy_result.get_if_computed()
        assert result is None

        # Trigger computation
        _ = lazy_result.value

        # Now returns result
        result = lazy_result.get_if_computed()
        assert result == 42

    def test_peek(self) -> None:
        """Test peeking at computation status."""
        lazy_result = LazyResult(lambda: 42)

        # Not computed
        computed, result = lazy_result.peek()
        assert not computed
        assert result is None

        # Trigger computation
        _ = lazy_result.value

        # Computed
        computed, result = lazy_result.peek()
        assert computed
        assert result == 42

    def test_map_chaining(self) -> None:
        """Test chained lazy operations via map."""
        lazy_base = LazyResult(lambda: 10, name="base")
        lazy_doubled = lazy_base.map(lambda x: x * 2)
        lazy_squared = lazy_doubled.map(lambda x: x**2)

        # Nothing computed yet
        assert not lazy_base.is_computed()
        assert not lazy_doubled.is_computed()
        assert not lazy_squared.is_computed()

        # Access final result triggers chain
        result = lazy_squared.value
        assert result == 400  # (10 * 2)^2 = 400

        # All computed now
        assert lazy_base.is_computed()
        assert lazy_doubled.is_computed()
        assert lazy_squared.is_computed()

    def test_thread_safety(self) -> None:
        """Test thread-safe concurrent access."""
        compute_count = 0
        lock = threading.Lock()

        def compute_fn() -> int:
            nonlocal compute_count
            # Simulate expensive computation
            time.sleep(0.01)
            with lock:
                compute_count += 1
            return 42

        lazy_result = LazyResult(compute_fn)

        # Launch multiple threads accessing concurrently
        results: list[int] = []

        def access_value() -> None:
            results.append(lazy_result.value)

        threads = [threading.Thread(target=access_value) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get same result
        assert all(r == 42 for r in results)
        # But computation should happen only once
        assert compute_count == 1

    def test_exception_handling(self) -> None:
        """Test that exceptions don't cache."""

        def failing_compute() -> int:
            raise ValueError("Computation failed")

        lazy_result = LazyResult(failing_compute)

        # First access raises exception
        with pytest.raises(ValueError, match="Computation failed"):
            _ = lazy_result.value

        # Should not be marked as computed
        assert not lazy_result.is_computed()

        # Second access should retry (and fail again)
        with pytest.raises(ValueError, match="Computation failed"):
            _ = lazy_result.value

    def test_repr(self) -> None:
        """Test string representation."""
        lazy_result = LazyResult(lambda: 42, name="test_result")

        # Before computation
        repr_str = repr(lazy_result)
        assert "test_result" in repr_str
        assert "deferred" in repr_str

        # After computation
        _ = lazy_result.value
        repr_str = repr(lazy_result)
        assert "test_result" in repr_str
        assert "computed" in repr_str


class TestLazyDict:
    """Test LazyDict auto-evaluation dictionary."""

    def test_auto_evaluation(self) -> None:
        """Test automatic evaluation of LazyResult values."""
        lazy_dict = LazyDict()
        lazy_dict["result"] = LazyResult(lambda: 42)
        lazy_dict["constant"] = 100

        # Access auto-evaluates lazy result
        assert lazy_dict["result"] == 42
        assert lazy_dict["constant"] == 100

    def test_get_lazy(self) -> None:
        """Test getting raw LazyResult without evaluation."""
        lazy_dict = LazyDict()
        lazy_result = LazyResult(lambda: 42)
        lazy_dict["result"] = lazy_result

        # Get raw lazy result
        raw = lazy_dict.get_lazy("result")
        assert raw is lazy_result
        assert not raw.is_computed()

        # Normal access evaluates
        value = lazy_dict["result"]
        assert value == 42
        assert raw.is_computed()

    def test_is_computed(self) -> None:
        """Test checking if values are computed."""
        lazy_dict = LazyDict()
        lazy_dict["lazy"] = LazyResult(lambda: 42)
        lazy_dict["constant"] = 100

        # Constant is "computed" (not lazy)
        assert lazy_dict.is_computed("constant")
        # Lazy value not computed yet
        assert not lazy_dict.is_computed("lazy")

        # Access lazy value
        _ = lazy_dict["lazy"]
        assert lazy_dict.is_computed("lazy")

    def test_invalidate_single(self) -> None:
        """Test invalidating single lazy value."""
        lazy_dict = LazyDict()
        lazy_dict["result"] = LazyResult(lambda: 42)

        # Compute
        _ = lazy_dict["result"]
        assert lazy_dict.is_computed("result")

        # Invalidate
        lazy_dict.invalidate("result")
        assert not lazy_dict.is_computed("result")

    def test_invalidate_all(self) -> None:
        """Test invalidating all lazy values."""
        lazy_dict = LazyDict()
        lazy_dict["result1"] = LazyResult(lambda: 10)
        lazy_dict["result2"] = LazyResult(lambda: 20)
        lazy_dict["constant"] = 100

        # Compute all
        _ = lazy_dict["result1"]
        _ = lazy_dict["result2"]

        assert lazy_dict.is_computed("result1")
        assert lazy_dict.is_computed("result2")

        # Invalidate all
        lazy_dict.invalidate_all()

        assert not lazy_dict.is_computed("result1")
        assert not lazy_dict.is_computed("result2")
        # Constant unaffected
        assert lazy_dict.is_computed("constant")

    def test_computed_keys(self) -> None:
        """Test getting list of computed keys."""
        lazy_dict = LazyDict()
        lazy_dict["computed"] = LazyResult(lambda: 42)
        lazy_dict["deferred"] = LazyResult(lambda: 100)
        lazy_dict["constant"] = 200

        # Access one lazy value
        _ = lazy_dict["computed"]

        computed = lazy_dict.computed_keys()
        assert "computed" in computed
        assert "constant" in computed
        assert "deferred" not in computed

    def test_deferred_keys(self) -> None:
        """Test getting list of deferred keys."""
        lazy_dict = LazyDict()
        lazy_dict["computed"] = LazyResult(lambda: 42)
        lazy_dict["deferred"] = LazyResult(lambda: 100)
        lazy_dict["constant"] = 200

        # Access one lazy value
        _ = lazy_dict["computed"]

        deferred = lazy_dict.deferred_keys()
        assert "deferred" in deferred
        assert "computed" not in deferred
        assert "constant" not in deferred

    def test_chained_dependencies(self) -> None:
        """Test lazy dict with dependent computations."""
        lazy_dict = LazyDict()
        lazy_dict["a"] = LazyResult(lambda: 10)
        lazy_dict["b"] = LazyResult(lambda: lazy_dict["a"] * 2)
        lazy_dict["c"] = LazyResult(lambda: lazy_dict["b"] + 5)

        # Access final result triggers chain
        result = lazy_dict["c"]
        assert result == 25  # (10 * 2) + 5

        # All should be computed now
        assert lazy_dict.is_computed("a")
        assert lazy_dict.is_computed("b")
        assert lazy_dict.is_computed("c")


class TestLazyDecorator:
    """Test @lazy decorator."""

    def test_decorator_returns_lazy_result(self) -> None:
        """Test that decorator returns LazyResult."""

        @lazy
        def compute_value(x: int) -> int:
            return x * 2

        result = compute_value(21)
        assert isinstance(result, LazyResult)
        assert not result.is_computed()

        value = result.value
        assert value == 42

    def test_decorator_with_complex_function(self) -> None:
        """Test decorator with complex function."""
        compute_called = False

        @lazy
        def compute_fft(signal: np.ndarray, nfft: int) -> np.ndarray:
            nonlocal compute_called
            compute_called = True
            return np.fft.fft(signal, n=nfft)

        signal = np.random.randn(1000)
        lazy_fft = compute_fft(signal, 2048)

        # Not computed yet
        assert not compute_called
        assert not lazy_fft.is_computed()

        # Access triggers computation
        spectrum = lazy_fft.value
        assert compute_called
        assert len(spectrum) == 2048

    def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator preserves function metadata."""

        @lazy
        def my_function() -> int:
            """My docstring."""
            return 42

        lazy_result = my_function()
        # LazyResult name should be function name
        assert "my_function" in repr(lazy_result)


class TestLazyAnalysisResult:
    """Test LazyAnalysisResult for multi-domain analysis."""

    def test_partial_domain_evaluation(self) -> None:
        """Test that only accessed domains are computed."""

        class MockAnalyzer:
            def __init__(self) -> None:
                self.computed_domains: list[str] = []

            def analyze(self, data: Any, domain: str) -> dict[str, Any]:
                self.computed_domains.append(domain)
                return {f"{domain}_result": data * 10}

        analyzer = MockAnalyzer()
        lazy_results = LazyAnalysisResult(
            analyzer,
            data=5,
            domains=["time", "frequency", "statistics"],
        )

        # Nothing computed yet
        assert len(analyzer.computed_domains) == 0
        assert len(lazy_results.computed_domains()) == 0
        assert len(lazy_results.deferred_domains()) == 3

        # Access only frequency domain
        freq_result = lazy_results.get_domain("frequency")
        assert freq_result == {"frequency_result": 50}

        # Only frequency computed
        assert analyzer.computed_domains == ["frequency"]
        assert lazy_results.computed_domains() == ["frequency"]
        assert set(lazy_results.deferred_domains()) == {"time", "statistics"}

    def test_compute_all_domains(self) -> None:
        """Test computing all domains."""

        class MockAnalyzer:
            def analyze(self, data: Any, domain: str) -> int:
                return data * {"time": 1, "frequency": 2, "statistics": 3}[domain]

        analyzer = MockAnalyzer()
        lazy_results = LazyAnalysisResult(
            analyzer,
            data=10,
            domains=["time", "frequency", "statistics"],
        )

        # Compute all
        all_results = lazy_results.compute_all()

        assert all_results == {
            "time": 10,
            "frequency": 20,
            "statistics": 30,
        }

        # All domains computed
        assert len(lazy_results.deferred_domains()) == 0
        assert set(lazy_results.computed_domains()) == {
            "time",
            "frequency",
            "statistics",
        }

    def test_dictionary_style_access(self) -> None:
        """Test dictionary-style domain access."""

        class MockAnalyzer:
            def analyze(self, data: Any, domain: str) -> str:
                return f"{domain}_result"

        analyzer = MockAnalyzer()
        lazy_results = LazyAnalysisResult(
            analyzer,
            data=None,
            domains=["time", "frequency"],
        )

        # Dictionary-style access
        time_result = lazy_results["time"]
        assert time_result == "time_result"

    def test_invalidate_domain(self) -> None:
        """Test invalidating specific domain."""

        class MockAnalyzer:
            def __init__(self) -> None:
                self.call_count = 0

            def analyze(self, data: Any, domain: str) -> int:
                self.call_count += 1
                return self.call_count

        analyzer = MockAnalyzer()
        lazy_results = LazyAnalysisResult(
            analyzer,
            data=None,
            domains=["time", "frequency"],
        )

        # Compute both domains
        time1 = lazy_results["time"]  # call_count = 1
        freq1 = lazy_results["frequency"]  # call_count = 2

        assert time1 == 1
        assert freq1 == 2

        # Invalidate time domain
        lazy_results.invalidate_domain("time")

        # Recompute time domain
        time2 = lazy_results["time"]  # call_count = 3
        assert time2 == 3

        # Frequency domain still cached
        freq2 = lazy_results["frequency"]  # No new computation
        assert freq2 == 2

    def test_custom_compute_function(self) -> None:
        """Test custom compute function template."""

        class MockAnalyzer:
            def analyze_time(self, data: Any) -> str:
                return "time_result"

            def analyze_frequency(self, data: Any) -> str:
                return "freq_result"

        def custom_compute(engine: Any, data: Any, domain: str) -> str:
            method = getattr(engine, f"analyze_{domain}")
            return method(data)

        analyzer = MockAnalyzer()
        lazy_results = LazyAnalysisResult(
            analyzer,
            data=None,
            domains=["time", "frequency"],
            compute_fn_template=custom_compute,
        )

        assert lazy_results["time"] == "time_result"
        assert lazy_results["frequency"] == "freq_result"

    def test_repr(self) -> None:
        """Test string representation."""

        class MockAnalyzer:
            def analyze(self, data: Any, domain: str) -> None:
                return None

        analyzer = MockAnalyzer()
        lazy_results = LazyAnalysisResult(
            analyzer,
            data=None,
            domains=["time", "frequency"],
        )

        repr_str = repr(lazy_results)
        assert "time" in repr_str
        assert "frequency" in repr_str

        # Access one domain
        _ = lazy_results["time"]

        repr_str = repr(lazy_results)
        assert "computed" in repr_str.lower()
        assert "deferred" in repr_str.lower()


class TestGlobalStatistics:
    """Test global lazy computation statistics."""

    def test_get_lazy_stats(self) -> None:
        """Test getting global statistics."""
        reset_lazy_stats()

        # Create some lazy results
        lazy1 = LazyResult(lambda: 42)
        lazy2 = LazyResult(lambda: 100)

        stats = get_lazy_stats()
        assert stats.total_created == 2
        assert stats.total_computed == 0

        # Compute one
        _ = lazy1.value

        stats = get_lazy_stats()
        assert stats.total_computed == 1

        # Access again (cache hit)
        _ = lazy1.value

        stats = get_lazy_stats()
        assert stats.cache_hits == 1

    def test_reset_lazy_stats(self) -> None:
        """Test resetting global statistics."""
        reset_lazy_stats()

        # Create and compute
        lazy1 = LazyResult(lambda: 42)
        _ = lazy1.value

        stats = get_lazy_stats()
        assert stats.total_created > 0

        # Reset
        reset_lazy_stats()

        stats = get_lazy_stats()
        assert stats.total_created == 0
        assert stats.total_computed == 0
        assert stats.cache_hits == 0


class TestCoreLazyIntegration:
    """Integration tests for real-world usage patterns."""

    def test_lazy_fft_workflow(self) -> None:
        """Test lazy FFT computation workflow."""
        reset_lazy_stats()

        # Create signal
        signal = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 1000))

        # Define lazy FFT computation
        @lazy
        def compute_fft(sig: np.ndarray, nfft: int) -> np.ndarray:
            return np.fft.fft(sig, n=nfft)

        # Create lazy pipeline using map for chaining
        lazy_fft = compute_fft(signal, 2048)
        lazy_power = lazy_fft.map(lambda spectrum: np.abs(spectrum) ** 2)

        # Nothing computed yet
        assert not lazy_fft.is_computed()
        assert not lazy_power.is_computed()

        # Access power triggers both computations
        power = lazy_power.value

        assert len(power) == 2048
        assert lazy_fft.is_computed()
        assert lazy_power.is_computed()

        # Check statistics
        stats = get_lazy_stats()
        assert stats.total_computed == 2

    def test_memory_efficient_analysis(self) -> None:
        """Test memory-efficient multi-domain analysis."""

        class SignalAnalyzer:
            def analyze(self, data: np.ndarray, domain: str) -> dict[str, Any]:
                if domain == "time":
                    return {
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                    }
                elif domain == "frequency":
                    spectrum = np.fft.fft(data)
                    return {
                        "spectrum": spectrum,
                        "power": np.abs(spectrum) ** 2,
                    }
                elif domain == "statistics":
                    return {
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "median": float(np.median(data)),
                    }
                return {}

        # Large signal
        signal = np.random.randn(100000)

        # Wrap in lazy analysis
        analyzer = SignalAnalyzer()
        lazy_results = LazyAnalysisResult(
            analyzer,
            signal,
            domains=["time", "frequency", "statistics"],
        )

        # Only access time domain (frequency and statistics not computed)
        time_results = lazy_results["time"]
        assert "mean" in time_results
        assert "std" in time_results

        # Verify only time domain computed
        assert lazy_results.computed_domains() == ["time"]
        assert set(lazy_results.deferred_domains()) == {"frequency", "statistics"}

    def test_lazy_dict_analysis_results(self) -> None:
        """Test using LazyDict for analysis results."""
        signal = np.random.randn(1000)

        results = LazyDict()
        results["fft"] = LazyResult(lambda: np.fft.fft(signal))
        results["power"] = LazyResult(lambda: np.abs(results["fft"]) ** 2)
        results["peak_freq"] = LazyResult(lambda: np.argmax(results["power"]))

        # Access peak frequency triggers entire chain
        peak = results["peak_freq"]

        assert isinstance(peak, int | np.integer)
        assert results.is_computed("fft")
        assert results.is_computed("power")
        assert results.is_computed("peak_freq")
