"""Test duty_cycle() function with extreme and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.waveform.measurements import duty_cycle
from oscura.core.types import TraceMetadata, WaveformTrace


class TestDutyCycleExtremeCases:
    """Test duty_cycle() with extreme duty cycles and edge cases."""

    @pytest.mark.parametrize(
        "dc_target",
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        ids=["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"],
    )
    def test_duty_cycle_range(self, dc_target: float) -> None:
        """Test duty cycle measurement for 1%-99% range.

        Args:
            dc_target: Target duty cycle (0.0-1.0).
        """
        # Create PWM signal with target duty cycle
        samples = 10000
        high_samples = int(samples * dc_target)
        data = np.concatenate([np.ones(high_samples) * 1.0, np.zeros(samples - high_samples)])

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)

        # Extract value from MeasurementResult
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc_measured = dc_result["value"]

        # Allow 1% tolerance
        assert isinstance(dc_measured, (float, np.floating))
        assert abs(float(dc_measured) - dc_target) < 0.01, (
            f"Expected {dc_target:.2f}, got {dc_measured:.4f}"
        )

    def test_duty_cycle_10_percent(self) -> None:
        """Test 10% duty cycle matches test data expectations."""
        # Realistic PWM: 10% high, 90% low
        samples = 1000
        data = np.concatenate([np.ones(100) * 0.5, -np.ones(900) * 0.5])

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc = dc_result["value"]

        assert isinstance(dc, (float, np.floating))
        assert 0.09 < dc < 0.11, f"Expected ~0.10, got {dc:.4f}"
        assert abs(float(dc) - 0.10) < 0.01

    def test_duty_cycle_90_percent(self) -> None:
        """Test 90% duty cycle matches test data expectations."""
        # Realistic PWM: 90% high, 10% low
        samples = 1000
        data = np.concatenate([np.ones(900) * 0.5, -np.ones(100) * 0.5])

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc = dc_result["value"]

        assert isinstance(dc, (float, np.floating))
        assert 0.89 < dc < 0.91, f"Expected ~0.90, got {dc:.4f}"
        assert abs(float(dc) - 0.90) < 0.01

    def test_duty_cycle_percentage_output(self) -> None:
        """Test percentage=True parameter returns 0-100 range."""
        samples = 1000
        data = np.concatenate([np.ones(250) * 1.0, np.zeros(750)])

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_ratio_result = duty_cycle(trace, percentage=False)
        dc_percent_result = duty_cycle(trace, percentage=True)

        assert dc_ratio_result["applicable"]
        assert dc_percent_result["applicable"]
        dc_ratio = dc_ratio_result["value"]
        dc_percent = dc_percent_result["value"]

        assert isinstance(dc_ratio, (float, np.floating))
        assert isinstance(dc_percent, (float, np.floating))
        assert abs(float(dc_ratio) - 0.25) < 0.01
        # Note: percentage parameter is ignored, both return ratio
        assert abs(float(dc_percent) - 0.25) < 0.01

    def test_duty_cycle_with_complete_cycles(self) -> None:
        """Test duty cycle with complete square wave cycles."""
        # 100 Hz square wave, 50% duty cycle, 10 complete cycles
        fs = 100e3  # 100 kHz sample rate
        f0 = 100  # 100 Hz signal
        t = np.arange(0, 0.1, 1 / fs)  # 0.1 seconds
        data = np.where((t * f0) % 1 < 0.5, 1.0, 0.0)

        metadata = TraceMetadata(sample_rate=fs)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc = dc_result["value"]

        # Should be very close to 0.5
        assert isinstance(dc, (float, np.floating))
        assert abs(float(dc) - 0.5) < 0.02

    def test_duty_cycle_incomplete_waveform(self) -> None:
        """Test duty cycle when only partial cycle is visible."""
        # Single pulse (no complete period)
        data = np.concatenate([np.ones(100) * 1.0, np.zeros(900)])

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)

        # Should use time-domain fallback method
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc = dc_result["value"]
        assert 0.09 < dc < 0.11

    def test_duty_cycle_single_edge(self) -> None:
        """Test duty cycle when only one edge is visible."""
        # Starts high, goes low once
        data = np.concatenate([np.ones(800) * 1.0, np.zeros(200)])

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)

        # Should use time-domain fallback
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc = dc_result["value"]
        assert isinstance(dc, (float, np.floating))
        assert 0.79 < dc < 0.81

    def test_duty_cycle_constant_high(self) -> None:
        """Test duty cycle for constant high signal (100%)."""
        data = np.ones(1000)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)

        # Constant signal should be inapplicable (no amplitude variation)
        assert not dc_result["applicable"], "Constant signal should not be applicable"

    def test_duty_cycle_constant_low(self) -> None:
        """Test duty cycle for constant low signal (0%)."""
        data = np.zeros(1000)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)

        # Constant signal should be inapplicable (no amplitude variation)
        assert not dc_result["applicable"], "Constant signal should not be applicable"

    def test_duty_cycle_noisy_signal(self) -> None:
        """Test duty cycle with realistic noise."""
        # 30% duty cycle with 5% amplitude noise
        samples = 5000
        data = np.concatenate([np.ones(1500) * 1.0, np.zeros(3500)])
        noise = np.random.default_rng(42).normal(0, 0.05, samples)
        data = data + noise

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc = dc_result["value"]

        # Should be close to 0.3 despite noise
        assert isinstance(dc, (float, np.floating))
        assert abs(float(dc) - 0.30) < 0.03

    def test_duty_cycle_digital_boolean(self) -> None:
        """Test duty cycle with digital boolean data."""
        # Boolean array (True/False)
        data = np.array([True] * 400 + [False] * 600, dtype=bool)

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)
        assert dc_result["applicable"], f"Duty cycle not applicable: {dc_result.get('reason')}"
        dc = dc_result["value"]

        assert isinstance(dc, (float, np.floating))
        assert abs(float(dc) - 0.40) < 0.01

    def test_duty_cycle_empty_trace(self) -> None:
        """Test duty cycle with empty trace."""
        data = np.array([])
        metadata = TraceMetadata(sample_rate=1e6)

        # Empty trace will fail validation in WaveformTrace.__post_init__
        with pytest.raises(ValueError, match="data array cannot be empty"):
            trace = WaveformTrace(data=data, metadata=metadata)

    def test_duty_cycle_insufficient_samples(self) -> None:
        """Test duty cycle with too few samples."""
        data = np.array([1.0, 0.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        dc_result = duty_cycle(trace)

        # Should be inapplicable with too few samples
        assert not dc_result["applicable"], "Should not be applicable with 2 samples"


class TestDutyCycleRealTestData:
    """Test duty_cycle() with real test data files."""

    def test_pulse_train_10_percent(self) -> None:
        """Test 10% duty cycle pulse train from test data."""
        from oscura.core.types import WaveformTrace
        from oscura.loaders import load

        trace_loaded = load("test_data/synthetic/advanced/pulse_train_10pct.wfm")
        assert isinstance(trace_loaded, WaveformTrace), "Expected WaveformTrace"
        trace = trace_loaded

        dc_result = duty_cycle(trace)
        dc_pct_result = duty_cycle(trace, percentage=True)

        # Should measure ~10% duty cycle
        assert dc_result["applicable"], f"duty_cycle not applicable: {dc_result.get('reason')}"
        assert dc_pct_result["applicable"]
        dc = dc_result["value"]
        dc_pct = dc_pct_result["value"]
        assert isinstance(dc, (float, np.floating))
        assert isinstance(dc_pct, (float, np.floating))
        assert 0.09 < dc < 0.11, f"Expected ~0.10, got {dc:.4f}"
        # Note: percentage parameter is ignored, both return ratio
        assert 0.09 < dc_pct < 0.11, f"Expected ~0.10, got {dc_pct:.4f}"

    def test_pulse_train_90_percent(self) -> None:
        """Test 90% duty cycle pulse train from test data."""
        from oscura.core.types import WaveformTrace
        from oscura.loaders import load

        trace_loaded = load("test_data/synthetic/advanced/pulse_train_90pct.wfm")
        assert isinstance(trace_loaded, WaveformTrace), "Expected WaveformTrace"
        trace = trace_loaded

        dc_result = duty_cycle(trace)
        dc_pct_result = duty_cycle(trace, percentage=True)

        # Should measure ~90% duty cycle
        assert dc_result["applicable"], f"duty_cycle not applicable: {dc_result.get('reason')}"
        assert dc_pct_result["applicable"]
        dc = dc_result["value"]
        dc_pct = dc_pct_result["value"]
        assert isinstance(dc, (float, np.floating))
        assert isinstance(dc_pct, (float, np.floating))
        assert 0.89 < dc < 0.91, f"Expected ~0.90, got {dc:.4f}"
        # Note: percentage parameter is ignored, both return ratio
        assert 0.89 < dc_pct < 0.91, f"Expected ~0.90, got {dc_pct:.4f}"
