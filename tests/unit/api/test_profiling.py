"""Unit tests for performance profiling API.

Tests API-012: Performance Profiling API
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oscura.api.profiling import (
    OperationProfile,
    Profiler,
    ProfileReport,
    disable_profiling,
    enable_profiling,
    get_profile_report,
    get_profiler,
    profile,
    reset_profiling,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestOperationProfile:
    """Test OperationProfile dataclass.

    Tests API-012: Performance Profiling API
    """

    def test_operation_profile_creation(self) -> None:
        """Test creating OperationProfile instance."""
        op = OperationProfile(name="test_op")

        assert op.name == "test_op"
        assert op.calls == 0
        assert op.total_time == 0.0
        assert op.min_time == float("inf")
        assert op.max_time == 0.0
        assert op.times == []
        assert op.memory_peak == 0
        assert op.input_size == 0

    def test_operation_profile_with_initial_values(self) -> None:
        """Test OperationProfile with initial values."""
        op = OperationProfile(
            name="test_op",
            calls=5,
            total_time=1.0,
            min_time=0.1,
            max_time=0.5,
            memory_peak=1000,
            input_size=100,
        )

        assert op.calls == 5
        assert op.total_time == 1.0
        assert op.min_time == 0.1
        assert op.max_time == 0.5
        assert op.memory_peak == 1000
        assert op.input_size == 100

    def test_mean_time_with_calls(self) -> None:
        """Test mean_time property with calls."""
        op = OperationProfile(name="test", calls=4, total_time=2.0)

        assert op.mean_time == 0.5

    def test_mean_time_with_zero_calls(self) -> None:
        """Test mean_time property with zero calls."""
        op = OperationProfile(name="test", calls=0, total_time=0.0)

        assert op.mean_time == 0.0

    def test_std_time_with_multiple_times(self) -> None:
        """Test std_time property with multiple times."""
        op = OperationProfile(name="test")
        op.times = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Standard deviation should be calculated correctly
        std = op.std_time
        assert std > 0.0
        assert abs(std - 1.5811388300841898) < 1e-10

    def test_std_time_with_single_time(self) -> None:
        """Test std_time property with single time."""
        op = OperationProfile(name="test")
        op.times = [1.0]

        assert op.std_time == 0.0

    def test_std_time_with_no_times(self) -> None:
        """Test std_time property with no times."""
        op = OperationProfile(name="test")

        assert op.std_time == 0.0

    def test_throughput_with_data(self) -> None:
        """Test throughput property with data."""
        op = OperationProfile(name="test", calls=10, total_time=1.0, input_size=100)

        # Throughput = (input_size * calls) / total_time
        assert op.throughput == 1000.0

    def test_throughput_with_zero_time(self) -> None:
        """Test throughput property with zero time."""
        op = OperationProfile(name="test", calls=10, total_time=0.0, input_size=100)

        assert op.throughput == 0.0

    def test_throughput_with_zero_size(self) -> None:
        """Test throughput property with zero input size."""
        op = OperationProfile(name="test", calls=10, total_time=1.0, input_size=0)

        assert op.throughput == 0.0

    def test_record_timing(self) -> None:
        """Test recording a timing."""
        op = OperationProfile(name="test")

        op.record(0.5, size=100)

        assert op.calls == 1
        assert op.total_time == 0.5
        assert op.min_time == 0.5
        assert op.max_time == 0.5
        assert op.times == [0.5]
        assert op.input_size == 100

    def test_record_multiple_timings(self) -> None:
        """Test recording multiple timings."""
        op = OperationProfile(name="test")

        op.record(0.3, size=100)
        op.record(0.5, size=200)
        op.record(0.2, size=150)

        assert op.calls == 3
        assert op.total_time == 1.0
        assert op.min_time == 0.2
        assert op.max_time == 0.5
        assert op.times == [0.3, 0.5, 0.2]
        # Last size is recorded
        assert op.input_size == 150

    def test_record_without_size(self) -> None:
        """Test recording timing without size."""
        op = OperationProfile(name="test")

        op.record(0.5, size=0)

        assert op.calls == 1
        assert op.total_time == 0.5
        assert op.input_size == 0

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        op = OperationProfile(name="test", calls=3, total_time=1.5)
        op.min_time = 0.4
        op.max_time = 0.6
        op.times = [0.4, 0.5, 0.6]

        result = op.to_dict()

        assert result["name"] == "test"
        assert result["calls"] == 3
        assert result["total_time"] == 1.5
        assert result["mean_time"] == 0.5
        assert result["min_time"] == 0.4
        assert result["max_time"] == 0.6
        assert "std_time" in result
        assert "throughput" in result

    def test_to_dict_with_inf_min_time(self) -> None:
        """Test to_dict with infinite min_time."""
        op = OperationProfile(name="test")

        result = op.to_dict()

        # Should convert inf to 0
        assert result["min_time"] == 0


@pytest.mark.unit
class TestProfileReport:
    """Test ProfileReport dataclass.

    Tests API-012: Performance Profiling API
    """

    def test_profile_report_creation(self) -> None:
        """Test creating ProfileReport instance."""
        report = ProfileReport()

        assert report.profiles == {}
        assert report.start_time == 0.0
        assert report.end_time == 0.0
        assert report.total_operations == 0

    def test_profile_report_with_data(self) -> None:
        """Test ProfileReport with data."""
        profiles = {
            "op1": OperationProfile(name="op1", calls=5, total_time=1.0),
            "op2": OperationProfile(name="op2", calls=3, total_time=0.5),
        }

        report = ProfileReport(
            profiles=profiles,
            start_time=100.0,
            end_time=102.0,
            total_operations=8,
        )

        assert len(report.profiles) == 2
        assert report.start_time == 100.0
        assert report.end_time == 102.0
        assert report.total_operations == 8

    def test_total_time_property(self) -> None:
        """Test total_time property."""
        profiles = {
            "op1": OperationProfile(name="op1", total_time=1.0),
            "op2": OperationProfile(name="op2", total_time=0.5),
            "op3": OperationProfile(name="op3", total_time=2.0),
        }

        report = ProfileReport(profiles=profiles)

        assert report.total_time == 3.5

    def test_total_time_empty_profiles(self) -> None:
        """Test total_time with empty profiles."""
        report = ProfileReport()

        assert report.total_time == 0.0

    def test_wall_time_property(self) -> None:
        """Test wall_time property."""
        report = ProfileReport(start_time=100.0, end_time=105.5)

        assert report.wall_time == 5.5

    def test_wall_time_zero_end_time(self) -> None:
        """Test wall_time with zero end time."""
        report = ProfileReport(start_time=100.0, end_time=0.0)

        assert report.wall_time == 0.0

    def test_get_slowest_operations(self) -> None:
        """Test get_slowest method."""
        profiles = {
            "fast": OperationProfile(name="fast", total_time=0.1),
            "medium": OperationProfile(name="medium", total_time=0.5),
            "slow": OperationProfile(name="slow", total_time=2.0),
            "very_slow": OperationProfile(name="very_slow", total_time=5.0),
        }

        report = ProfileReport(profiles=profiles)
        slowest = report.get_slowest(n=2)

        assert len(slowest) == 2
        assert slowest[0].name == "very_slow"
        assert slowest[1].name == "slow"

    def test_get_slowest_more_than_available(self) -> None:
        """Test get_slowest with n larger than available."""
        profiles = {
            "op1": OperationProfile(name="op1", total_time=1.0),
            "op2": OperationProfile(name="op2", total_time=0.5),
        }

        report = ProfileReport(profiles=profiles)
        slowest = report.get_slowest(n=10)

        assert len(slowest) == 2

    def test_get_most_called_operations(self) -> None:
        """Test get_most_called method."""
        profiles = {
            "rare": OperationProfile(name="rare", calls=2),
            "common": OperationProfile(name="common", calls=100),
            "moderate": OperationProfile(name="moderate", calls=50),
            "frequent": OperationProfile(name="frequent", calls=200),
        }

        report = ProfileReport(profiles=profiles)
        most_called = report.get_most_called(n=2)

        assert len(most_called) == 2
        assert most_called[0].name == "frequent"
        assert most_called[1].name == "common"

    def test_get_most_called_empty_profiles(self) -> None:
        """Test get_most_called with empty profiles."""
        report = ProfileReport()
        most_called = report.get_most_called(n=5)

        assert len(most_called) == 0

    def test_summary_generation(self) -> None:
        """Test summary method."""
        profiles = {
            "op1": OperationProfile(name="op1", calls=5, total_time=2.0),
            "op2": OperationProfile(name="op2", calls=10, total_time=1.0),
        }

        report = ProfileReport(
            profiles=profiles,
            start_time=100.0,
            end_time=103.0,
            total_operations=15,
        )

        summary = report.summary()

        assert "Performance Profile Report" in summary
        assert "Total operations: 15" in summary
        assert "Total profiled time: 3.0000s" in summary
        assert "Wall clock time: 3.0000s" in summary
        assert "Slowest Operations:" in summary
        assert "op1" in summary
        assert "op2" in summary

    def test_summary_empty_report(self) -> None:
        """Test summary with empty report."""
        report = ProfileReport()

        summary = report.summary()

        assert "Performance Profile Report" in summary
        assert "Total operations: 0" in summary

    def test_to_dict_conversion(self) -> None:
        """Test to_dict method."""
        profiles = {
            "op1": OperationProfile(name="op1", calls=5, total_time=1.0),
        }

        report = ProfileReport(
            profiles=profiles,
            start_time=100.0,
            end_time=105.0,
            total_operations=5,
        )

        result = report.to_dict()

        assert result["total_time"] == 1.0
        assert result["wall_time"] == 5.0
        assert result["total_operations"] == 5
        assert "profiles" in result
        assert "op1" in result["profiles"]


@pytest.mark.unit
class TestProfiler:
    """Test Profiler class.

    Tests API-012: Performance Profiling API
    """

    def test_profiler_creation(self) -> None:
        """Test creating Profiler instance."""
        profiler = Profiler()

        assert profiler._profiles == {}
        assert profiler._start_time == 0.0
        assert profiler._enabled is True
        assert profiler._stack == []

    def test_get_instance_singleton(self) -> None:
        """Test get_instance returns singleton."""
        # Reset singleton first
        Profiler._instance = None

        profiler1 = Profiler.get_instance()
        profiler2 = Profiler.get_instance()

        assert profiler1 is profiler2

    def test_enable_profiling(self) -> None:
        """Test enable method."""
        profiler = Profiler()
        profiler._enabled = False

        profiler.enable()

        assert profiler._enabled is True

    def test_disable_profiling(self) -> None:
        """Test disable method."""
        profiler = Profiler()
        profiler._enabled = True

        profiler.disable()

        assert profiler._enabled is False

    def test_reset_profiler(self) -> None:
        """Test reset method."""
        profiler = Profiler()
        profiler._profiles = {"test": OperationProfile(name="test")}
        profiler._start_time = 100.0

        profiler.reset()

        assert profiler._profiles == {}
        assert profiler._start_time == 0.0

    @patch("time.perf_counter")
    def test_profile_context_manager(self, mock_time: MagicMock) -> None:
        """Test profile context manager."""
        mock_time.side_effect = [100.0, 100.0, 100.5]  # _start_time, start, end

        profiler = Profiler()

        with profiler.profile("test_op"):
            pass

        assert "test_op" in profiler._profiles
        assert profiler._profiles["test_op"].calls == 1
        assert profiler._profiles["test_op"].total_time == 0.5

    @patch("time.perf_counter")
    def test_profile_with_input_size(self, mock_time: MagicMock) -> None:
        """Test profile with input size."""
        mock_time.side_effect = [100.0, 100.0, 100.3]

        profiler = Profiler()

        with profiler.profile("test_op", input_size=1000):
            pass

        assert profiler._profiles["test_op"].input_size == 1000

    @patch("time.perf_counter")
    def test_profile_multiple_calls(self, mock_time: MagicMock) -> None:
        """Test profile with multiple calls."""
        # First call: _start_time, start, end
        # Second call: start, end (no _start_time since already set)
        mock_time.side_effect = [100.0, 100.0, 100.2, 100.2, 100.5]

        profiler = Profiler()

        with profiler.profile("test_op"):
            pass

        with profiler.profile("test_op"):
            pass

        assert profiler._profiles["test_op"].calls == 2
        assert profiler._profiles["test_op"].total_time == 0.5

    def test_profile_disabled(self) -> None:
        """Test profile when disabled."""
        profiler = Profiler()
        profiler.disable()

        with profiler.profile("test_op"):
            pass

        assert "test_op" not in profiler._profiles

    @patch("time.perf_counter")
    def test_profile_sets_start_time(self, mock_time: MagicMock) -> None:
        """Test profile sets start time on first use."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        profiler = Profiler()

        with profiler.profile("test_op"):
            pass

        assert profiler._start_time == 100.0

    @patch("time.perf_counter")
    def test_profile_maintains_stack(self, mock_time: MagicMock) -> None:
        """Test profile maintains stack."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        profiler = Profiler()

        with profiler.profile("test_op"):
            assert "test_op" in profiler._stack

        assert "test_op" not in profiler._stack

    @patch("time.perf_counter")
    def test_profile_exception_handling(self, mock_time: MagicMock) -> None:
        """Test profile handles exceptions."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        profiler = Profiler()

        with pytest.raises(ValueError):
            with profiler.profile("test_op"):
                raise ValueError("Test error")

        # Profile should still be recorded
        assert profiler._profiles["test_op"].calls == 1
        assert profiler._stack == []

    def test_record_manual_timing(self) -> None:
        """Test manual record method."""
        profiler = Profiler()

        profiler.record("test_op", elapsed=0.5, input_size=100)

        assert "test_op" in profiler._profiles
        assert profiler._profiles["test_op"].calls == 1
        assert profiler._profiles["test_op"].total_time == 0.5

    def test_record_disabled(self) -> None:
        """Test record when disabled."""
        profiler = Profiler()
        profiler.disable()

        profiler.record("test_op", elapsed=0.5)

        assert "test_op" not in profiler._profiles

    def test_record_creates_new_profile(self) -> None:
        """Test record creates new profile if it doesn't exist."""
        profiler = Profiler()

        # Record on non-existent profile should create it
        profiler.record("new_op", elapsed=0.3, input_size=50)

        assert "new_op" in profiler._profiles
        assert profiler._profiles["new_op"].calls == 1
        assert profiler._profiles["new_op"].total_time == 0.3
        assert profiler._profiles["new_op"].input_size == 50

    def test_get_profile_existing(self) -> None:
        """Test get_profile for existing operation."""
        profiler = Profiler()
        profiler._profiles["test_op"] = OperationProfile(name="test_op")

        profile = profiler.get_profile("test_op")

        assert profile is not None
        assert profile.name == "test_op"

    def test_get_profile_nonexistent(self) -> None:
        """Test get_profile for nonexistent operation."""
        profiler = Profiler()

        profile = profiler.get_profile("nonexistent")

        assert profile is None

    @patch("time.perf_counter")
    def test_report_generation(self, mock_time: MagicMock) -> None:
        """Test report method."""
        mock_time.return_value = 200.0

        profiler = Profiler()
        profiler._start_time = 100.0
        profiler._profiles = {
            "op1": OperationProfile(name="op1", calls=5),
            "op2": OperationProfile(name="op2", calls=3),
        }

        report = profiler.report()

        assert len(report.profiles) == 2
        assert report.start_time == 100.0
        assert report.end_time == 200.0
        assert report.total_operations == 8


@pytest.mark.unit
class TestProfileDecorator:
    """Test profile decorator.

    Tests API-012: Performance Profiling API
    """

    @patch("time.perf_counter")
    def test_profile_decorator_basic(self, mock_time: MagicMock) -> None:
        """Test basic profile decorator."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        # Reset singleton
        Profiler._instance = None

        @profile()
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)

        assert result == 10

        profiler = Profiler.get_instance()
        assert "test_func" in profiler._profiles
        assert profiler._profiles["test_func"].calls == 1

    @patch("time.perf_counter")
    def test_profile_decorator_with_name(self, mock_time: MagicMock) -> None:
        """Test profile decorator with custom name."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        @profile(name="custom_name")
        def test_func(x: int) -> int:
            return x * 2

        test_func(5)

        profiler = Profiler.get_instance()
        assert "custom_name" in profiler._profiles

    def test_profile_decorator_preserves_function(self) -> None:
        """Test profile decorator preserves function metadata."""

        @profile()
        def test_func(x: int) -> int:
            """Test docstring."""
            return x * 2

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test docstring."

    @patch("time.perf_counter")
    def test_profile_decorator_with_input_size_arg(self, mock_time: MagicMock) -> None:
        """Test profile decorator with input_size_arg."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        @profile(input_size_arg="data")
        def process_data(data: list[int]) -> int:
            return sum(data)

        result = process_data([1, 2, 3, 4, 5])

        assert result == 15

        profiler = Profiler.get_instance()
        assert profiler._profiles["process_data"].input_size == 5

    @patch("time.perf_counter")
    def test_profile_decorator_input_size_positional(self, mock_time: MagicMock) -> None:
        """Test profile decorator extracts size from positional arg."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        @profile(input_size_arg="data")
        def process_data(data: list[int]) -> int:
            return sum(data)

        # Pass as positional
        result = process_data([1, 2, 3])

        assert result == 6

        profiler = Profiler.get_instance()
        assert profiler._profiles["process_data"].input_size == 3

    @patch("time.perf_counter")
    def test_profile_decorator_no_input_size(self, mock_time: MagicMock) -> None:
        """Test profile decorator without input size."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        @profile()
        def simple_func() -> int:
            return 42

        result = simple_func()

        assert result == 42

        profiler = Profiler.get_instance()
        assert profiler._profiles["simple_func"].input_size == 0

    @patch("time.perf_counter")
    def test_profile_decorator_multiple_calls(self, mock_time: MagicMock) -> None:
        """Test profile decorator with multiple calls."""
        # First call: _start_time, start, end
        # Second call: start, end
        mock_time.side_effect = [100.0, 100.0, 100.2, 100.2, 100.5]

        Profiler._instance = None

        @profile()
        def test_func(x: int) -> int:
            return x * 2

        test_func(5)
        test_func(10)

        profiler = Profiler.get_instance()
        assert profiler._profiles["test_func"].calls == 2

    @patch("time.perf_counter")
    def test_profile_decorator_with_kwargs(self, mock_time: MagicMock) -> None:
        """Test profile decorator with keyword arguments."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        @profile()
        def test_func(x: int, y: int = 10) -> int:
            return x + y

        result = test_func(5, y=20)

        assert result == 25

    @patch("time.perf_counter")
    def test_profile_decorator_with_no_args_no_size(self, mock_time: MagicMock) -> None:
        """Test profile decorator when no args and input_size_arg set."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        @profile(input_size_arg="data")
        def no_args_func() -> int:
            return 42

        result = no_args_func()

        assert result == 42
        profiler = Profiler.get_instance()
        assert profiler._profiles["no_args_func"].input_size == 0

    @patch("time.perf_counter")
    def test_profile_decorator_with_non_sized_arg(self, mock_time: MagicMock) -> None:
        """Test profile decorator with arg that has no __len__."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        @profile(input_size_arg="data")
        def process_int(data: int) -> int:
            return data * 2

        result = process_int(42)

        assert result == 84
        profiler = Profiler.get_instance()
        assert profiler._profiles["process_int"].input_size == 0


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions.

    Tests API-012: Performance Profiling API
    """

    def test_get_profiler_function(self) -> None:
        """Test get_profiler function."""
        # Reset singleton
        Profiler._instance = None

        profiler = get_profiler()

        assert isinstance(profiler, Profiler)
        assert profiler is Profiler.get_instance()

    def test_enable_profiling_function(self) -> None:
        """Test enable_profiling function."""
        Profiler._instance = None
        profiler = get_profiler()
        profiler.disable()

        enable_profiling()

        assert profiler._enabled is True

    def test_disable_profiling_function(self) -> None:
        """Test disable_profiling function."""
        Profiler._instance = None
        profiler = get_profiler()

        disable_profiling()

        assert profiler._enabled is False

    def test_reset_profiling_function(self) -> None:
        """Test reset_profiling function."""
        Profiler._instance = None
        profiler = get_profiler()
        profiler._profiles = {"test": OperationProfile(name="test")}

        reset_profiling()

        assert profiler._profiles == {}

    @patch("time.perf_counter")
    def test_get_profile_report_function(self, mock_time: MagicMock) -> None:
        """Test get_profile_report function."""
        mock_time.return_value = 200.0

        Profiler._instance = None
        profiler = get_profiler()
        profiler._start_time = 100.0
        profiler._profiles = {
            "op1": OperationProfile(name="op1", calls=5),
        }

        report = get_profile_report()

        assert isinstance(report, ProfileReport)
        assert len(report.profiles) == 1


@pytest.mark.unit
class TestIntegrationScenarios:
    """Test integration scenarios.

    Tests API-012: Performance Profiling API
    """

    @patch("time.perf_counter")
    def test_full_profiling_workflow(self, mock_time: MagicMock) -> None:
        """Test complete profiling workflow."""
        mock_time.side_effect = [
            100.0,
            100.0,
            100.1,  # First operation
            100.2,
            100.5,  # Second operation
            200.0,  # Report time
        ]

        Profiler._instance = None
        profiler = get_profiler()

        with profiler.profile("operation1"):
            pass

        with profiler.profile("operation2"):
            pass

        report = profiler.report()

        assert len(report.profiles) == 2
        assert report.total_operations == 2
        assert "operation1" in report.profiles
        assert "operation2" in report.profiles

    @patch("time.perf_counter")
    def test_nested_profiling(self, mock_time: MagicMock) -> None:
        """Test nested profiling operations."""
        mock_time.side_effect = [
            100.0,
            100.0,  # Outer start
            100.0,
            100.5,  # Inner1
            100.5,
            101.0,  # Inner2
            101.0,  # Outer end
        ]

        Profiler._instance = None
        profiler = get_profiler()

        with profiler.profile("outer"):
            with profiler.profile("inner1"):
                pass
            with profiler.profile("inner2"):
                pass

        assert profiler._profiles["outer"].calls == 1
        assert profiler._profiles["inner1"].calls == 1
        assert profiler._profiles["inner2"].calls == 1

    @patch("time.perf_counter")
    def test_decorator_and_context_manager_mix(self, mock_time: MagicMock) -> None:
        """Test mixing decorator and context manager."""
        mock_time.side_effect = [
            100.0,
            100.0,
            100.2,  # Decorated function
            100.3,
            100.5,  # Context manager
        ]

        Profiler._instance = None

        @profile()
        def decorated_func() -> int:
            return 42

        profiler = get_profiler()

        decorated_func()

        with profiler.profile("manual_op"):
            pass

        assert len(profiler._profiles) == 2
        assert "decorated_func" in profiler._profiles
        assert "manual_op" in profiler._profiles

    def test_profiling_disabled_scenario(self) -> None:
        """Test profiling when disabled."""
        Profiler._instance = None
        profiler = get_profiler()

        # Profile some operations
        with profiler.profile("enabled_op"):
            pass

        # Disable profiling
        disable_profiling()

        # This should not be recorded
        with profiler.profile("disabled_op"):
            pass

        assert "enabled_op" in profiler._profiles
        assert "disabled_op" not in profiler._profiles

    @patch("time.perf_counter")
    def test_report_summary_formatting(self, mock_time: MagicMock) -> None:
        """Test report summary formatting."""
        mock_time.side_effect = [
            100.0,
            100.0,
            100.5,  # fast_op
            101.0,
            102.0,  # slow_op
            200.0,  # report
        ]

        Profiler._instance = None
        profiler = get_profiler()

        with profiler.profile("fast_op"):
            pass

        with profiler.profile("slow_op"):
            pass

        report = profiler.report()
        summary = report.summary()

        # Check summary contains expected information
        assert "Performance Profile Report" in summary
        assert "Slowest Operations:" in summary
        assert "slow_op" in summary
        assert "fast_op" in summary

    @patch("time.perf_counter")
    def test_multiple_operations_with_different_sizes(self, mock_time: MagicMock) -> None:
        """Test multiple operations with different input sizes."""
        mock_time.side_effect = [
            100.0,
            100.0,
            100.1,  # small_op
            100.2,
            100.4,  # medium_op
            100.5,
            100.9,  # large_op
        ]

        Profiler._instance = None
        profiler = get_profiler()

        with profiler.profile("small_op", input_size=10):
            pass

        with profiler.profile("medium_op", input_size=100):
            pass

        with profiler.profile("large_op", input_size=1000):
            pass

        assert profiler._profiles["small_op"].input_size == 10
        assert profiler._profiles["medium_op"].input_size == 100
        assert profiler._profiles["large_op"].input_size == 1000


@pytest.mark.unit
class TestApiProfilingEdgeCases:
    """Test edge cases and error handling."""

    def test_operation_profile_empty_times_list(self) -> None:
        """Test OperationProfile with empty times list."""
        op = OperationProfile(name="test", times=[])

        assert op.std_time == 0.0

    def test_profile_report_empty_profiles(self) -> None:
        """Test ProfileReport with empty profiles."""
        report = ProfileReport(profiles={})

        assert report.total_time == 0.0
        assert report.get_slowest() == []
        assert report.get_most_called() == []

    def test_profiler_reset_while_enabled(self) -> None:
        """Test resetting profiler while enabled."""
        profiler = Profiler()
        profiler._enabled = True

        with profiler.profile("test_op"):
            pass

        profiler.reset()

        assert profiler._profiles == {}
        assert profiler._enabled is True  # Should remain enabled

    @patch("time.perf_counter")
    def test_profile_with_zero_elapsed_time(self, mock_time: MagicMock) -> None:
        """Test profile with zero elapsed time."""
        mock_time.side_effect = [100.0, 100.0, 100.0]  # Same time

        profiler = Profiler()

        with profiler.profile("instant_op"):
            pass

        assert profiler._profiles["instant_op"].total_time == 0.0

    def test_operation_profile_negative_values(self) -> None:
        """Test OperationProfile handles unusual values."""
        op = OperationProfile(name="test")

        # Record should work with any float values
        op.record(0.0001)
        op.record(0.0002)

        assert op.calls == 2
        assert op.min_time == 0.0001
        assert op.max_time == 0.0002

    def test_get_slowest_with_zero(self) -> None:
        """Test get_slowest with n=0."""
        report = ProfileReport(
            profiles={
                "op1": OperationProfile(name="op1", total_time=1.0),
            }
        )

        slowest = report.get_slowest(n=0)

        assert slowest == []

    def test_profile_decorator_without_parentheses(self) -> None:
        """Test that profile decorator requires parentheses."""
        # This is the expected usage - decorator factory pattern
        # profile() returns the actual decorator

        @profile()
        def test_func() -> int:
            return 42

        assert callable(test_func)

    @patch("time.perf_counter")
    def test_profiler_singleton_across_modules(self, mock_time: MagicMock) -> None:
        """Test that profiler singleton works across different access methods."""
        mock_time.side_effect = [100.0, 100.0, 100.5]

        Profiler._instance = None

        # Get profiler through different methods
        profiler1 = get_profiler()
        profiler2 = Profiler.get_instance()

        with profiler1.profile("test_op"):
            pass

        # Both should reference the same instance
        assert "test_op" in profiler2._profiles

    def test_throughput_calculation_accuracy(self) -> None:
        """Test throughput calculation accuracy."""
        op = OperationProfile(name="test", calls=100, total_time=2.0, input_size=1000)

        # Throughput should be (1000 * 100) / 2.0 = 50000
        assert op.throughput == 50000.0

    def test_to_dict_completeness(self) -> None:
        """Test to_dict includes all expected fields."""
        op = OperationProfile(name="test", calls=5, total_time=1.0)
        op.times = [0.1, 0.2, 0.3, 0.2, 0.2]

        result = op.to_dict()

        expected_keys = {
            "name",
            "calls",
            "total_time",
            "mean_time",
            "min_time",
            "max_time",
            "std_time",
            "throughput",
        }

        assert set(result.keys()) == expected_keys

    @patch("time.perf_counter")
    def test_manual_record_creates_profile(self, mock_time: MagicMock) -> None:
        """Test that manual record creates profile if not exists."""
        profiler = Profiler()

        profiler.record("new_op", elapsed=0.5, input_size=100)

        assert "new_op" in profiler._profiles
        assert profiler._profiles["new_op"].name == "new_op"
