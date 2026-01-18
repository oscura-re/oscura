"""Tests for Oscura triggering module.

Tests edge triggering, pattern triggering, pulse width triggering,
glitch detection, and window triggering.
"""

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.triggering import (
    EdgeTrigger,
    PatternTrigger,
    ZoneTrigger,
    check_limits,
    find_all_edges,
    find_falling_edges,
    find_glitches,
    find_pattern,
    find_pulses,
    find_rising_edges,
    find_runt_pulses,
    find_triggers,
    find_window_violations,
)
from oscura.triggering.base import TriggerType
from oscura.triggering.edge import edge_count, edge_rate
from oscura.triggering.pulse import pulse_statistics
from oscura.triggering.window import Zone

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 1_000_000.0  # 1 MHz


@pytest.fixture
def square_wave(sample_rate: float) -> WaveformTrace:
    """Generate a 10 kHz square wave."""
    duration = 0.001  # 1 ms = 10 cycles
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # 10 kHz square wave, 0 to 3.3V
    data = (np.sin(2 * np.pi * 10000 * t) > 0).astype(np.float64) * 3.3

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def pulse_train(sample_rate: float) -> WaveformTrace:
    """Generate a pulse train with known pulse widths."""
    duration = 0.001
    n_samples = int(sample_rate * duration)
    data = np.zeros(n_samples)

    # Create pulses at known positions with known widths
    # 10us pulses every 100us
    for i in range(10):
        start = i * 100 + 10  # Every 100us, offset by 10us
        width = 10  # 10us pulse width
        data[start : start + width] = 3.3

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def glitchy_signal(sample_rate: float) -> WaveformTrace:
    """Generate a signal with glitches."""
    n_samples = 1000
    data = np.zeros(n_samples)

    # Normal pulses (100 samples wide)
    data[100:200] = 3.3
    data[400:500] = 3.3

    # Glitches (5 samples wide)
    data[250:255] = 3.3
    data[600:605] = 3.3

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def runt_signal(sample_rate: float) -> WaveformTrace:
    """Generate a signal with runt pulses."""
    n_samples = 1000
    data = np.zeros(n_samples)

    # Full transition (0 -> 3.3V)
    data[100:200] = 3.3

    # Runt pulse (only reaches 1.5V, not full 3.3V)
    data[300:350] = 1.5

    # Another full transition
    data[500:600] = 3.3

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


# =============================================================================
# Edge Trigger Tests
# =============================================================================


class TestEdgeTrigger:
    """Tests for edge triggering functionality."""

    def test_find_rising_edges(self, square_wave: WaveformTrace):
        """Test finding rising edges."""
        edges = find_rising_edges(square_wave, level=1.65)

        # 10 kHz for 1ms = 10 complete cycles = 10 rising edges
        assert len(edges) == 10

    def test_find_falling_edges(self, square_wave: WaveformTrace):
        """Test finding falling edges."""
        edges = find_falling_edges(square_wave, level=1.65)

        assert len(edges) == 10

    def test_find_all_edges(self, square_wave: WaveformTrace):
        """Test finding all edges with polarity."""
        timestamps, is_rising = find_all_edges(square_wave, level=1.65)

        # Should have both rising and falling
        assert len(timestamps) == 20
        assert np.sum(is_rising) == 10  # Half rising
        assert np.sum(~is_rising) == 10  # Half falling

    def test_edge_trigger_hysteresis(self, sample_rate: float):
        """Test edge trigger with hysteresis."""
        # Signal with noise around threshold
        n = 1000
        t = np.arange(n) / sample_rate
        data = np.sin(2 * np.pi * 1000 * t) * 2  # +/- 2V swing
        noise = np.random.randn(n) * 0.3  # 300mV noise
        data_noisy = data + noise

        trace = WaveformTrace(
            data=data_noisy,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Without hysteresis - may detect false edges
        trigger_no_hyst = EdgeTrigger(level=0, edge="rising", hysteresis=0)
        trigger_no_hyst.find_events(trace)

        # With hysteresis - should be more stable
        trigger_hyst = EdgeTrigger(level=0, edge="rising", hysteresis=0.5)
        events_hyst = trigger_hyst.find_events(trace)

        # Hysteresis should reduce false triggers
        # (In practice, the clean signal might work well either way)
        assert len(events_hyst) > 0

    def test_edge_count(self, square_wave: WaveformTrace):
        """Test edge counting."""
        count = edge_count(square_wave, level=1.65, edge="rising")
        assert count == 10

        count_both = edge_count(square_wave, level=1.65, edge="either")
        assert count_both == 20

    def test_edge_rate(self, square_wave: WaveformTrace):
        """Test edge rate calculation."""
        rate = edge_rate(square_wave, level=1.65, edge="rising")

        # 10 edges in 1ms = 10,000 edges/sec
        assert abs(rate - 10000) < 100

    def test_edge_interpolation(self, sample_rate: float):
        """Test sub-sample edge interpolation."""
        # Create a ramp that crosses threshold mid-sample
        data = np.array([0.0, 0.5, 1.5, 2.0])
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        trigger = EdgeTrigger(level=1.0, edge="rising")
        events = trigger.find_events(trace)

        assert len(events) == 1
        # Interpolated time should be between samples 1 and 2
        sample_period = 1 / sample_rate
        assert sample_period < events[0].timestamp < 2 * sample_period


# =============================================================================
# Pattern Trigger Tests
# =============================================================================


class TestPatternTrigger:
    """Tests for pattern triggering functionality."""

    def test_find_pattern_sequence(self, sample_rate: float):
        """Test finding a bit sequence pattern."""
        # Create digital signal with pattern
        data = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0])
        data = data.astype(np.float64) * 3.3

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Find pattern [1, 0, 1]
        events = find_pattern(trace, pattern=[1, 0, 1], level=1.65)

        # Should find the pattern at positions 2 and 10
        assert len(events) >= 2

    def test_pattern_with_dont_care(self, sample_rate: float):
        """Test pattern with don't care values."""
        data = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0]).astype(np.float64) * 3.3

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Pattern [1, None, 1] matches any middle bit
        trigger = PatternTrigger(pattern=[1, None, 1], levels=1.65)
        events = trigger.find_events(trace)

        # Should match [1, 0, 1] patterns
        assert len(events) >= 1


# =============================================================================
# Pulse Width Trigger Tests
# =============================================================================


class TestPulseWidthTrigger:
    """Tests for pulse width triggering."""

    def test_find_pulses_in_range(self, pulse_train: WaveformTrace):
        """Test finding pulses within width range."""
        # Pulses are 10us wide
        events = find_pulses(
            pulse_train,
            level=1.65,
            min_width=5e-6,
            max_width=15e-6,
        )

        assert len(events) >= 5  # Should find most pulses

    def test_find_pulses_polarity(self, sample_rate: float):
        """Test pulse polarity detection."""
        n = 500
        data = np.zeros(n)
        # Positive pulse
        data[100:150] = 3.3
        # Negative pulse (inverted logic)
        data[300:350] = 0.0
        data[250:300] = 3.3
        data[350:400] = 3.3

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        pos_events = find_pulses(trace, level=1.65, polarity="positive")
        assert len(pos_events) >= 1

    def test_pulse_statistics(self, pulse_train: WaveformTrace):
        """Test pulse width statistics."""
        stats = pulse_statistics(pulse_train, level=1.65)

        assert stats["count"] >= 5
        assert stats["mean_width"] > 0
        # All pulses should be ~10us
        assert abs(stats["mean_width"] - 10e-6) < 5e-6


class TestGlitchTrigger:
    """Tests for glitch detection."""

    def test_find_glitches(self, glitchy_signal: WaveformTrace):
        """Test finding glitches (narrow pulses)."""
        # Glitches are 5 samples, normal pulses are 100 samples
        # At 1MHz, 5 samples = 5us, 100 samples = 100us
        glitches = find_glitches(
            glitchy_signal,
            max_width=20e-6,  # 20us max for glitch
            level=1.65,
        )

        assert len(glitches) == 2  # Two glitches

        # Verify durations
        for g in glitches:
            assert g.duration is not None
            assert g.duration < 20e-6


class TestRuntTrigger:
    """Tests for runt pulse detection."""

    def test_find_runt_pulses(self, runt_signal: WaveformTrace):
        """Test finding runt pulses."""
        runts = find_runt_pulses(
            runt_signal,
            low_threshold=0.5,
            high_threshold=2.5,
        )

        # Should find the runt (reaches 1.5V but not 2.5V)
        assert len(runts) >= 1

        # Verify runt data
        for runt in runts:
            assert runt.event_type == TriggerType.RUNT
            assert runt.data.get("polarity") in ("positive", "negative")


# =============================================================================
# Window Trigger Tests
# =============================================================================


class TestWindowTrigger:
    """Tests for window triggering."""

    def test_window_violation_detection(self, sample_rate: float):
        """Test detecting when signal exits window."""
        n = 1000
        data = np.sin(np.linspace(0, 4 * np.pi, n)) * 2  # +/- 2V

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Window from -1V to +1V
        violations = find_window_violations(trace, low=-1.0, high=1.0)

        # Should detect exits when signal goes beyond +/- 1V
        assert len(violations) > 0

    def test_check_limits_pass(self, sample_rate: float):
        """Test check_limits when signal is within limits."""
        data = np.random.uniform(1.0, 2.0, 1000)  # Signal between 1 and 2V

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = check_limits(trace, low=0.5, high=2.5)

        assert result["passed"]
        assert len(result["violations"]) == 0

    def test_check_limits_fail(self, sample_rate: float):
        """Test check_limits when signal exceeds limits."""
        data = np.array([1.0, 1.5, 3.0, 1.5, 1.0])  # Spike to 3V

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = check_limits(trace, low=0.0, high=2.5)

        assert not result["passed"]
        assert result["max_value"] == 3.0

    def test_zone_trigger(self, sample_rate: float):
        """Test zone-based triggering."""
        n = 500
        data = np.sin(np.linspace(0, 4 * np.pi, n)) * 2

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        zones = [
            Zone(low=1.5, high=2.5, name="overvoltage"),
            Zone(low=-2.5, high=-1.5, name="undervoltage"),
        ]

        trigger = ZoneTrigger(zones=zones)
        events = trigger.find_events(trace)

        # Should detect entries into both zones
        assert len(events) > 0


# =============================================================================
# find_triggers Unified Function Tests
# =============================================================================


class TestFindTriggers:
    """Tests for unified find_triggers function."""

    def test_find_triggers_edge(self, square_wave: WaveformTrace):
        """Test find_triggers with edge type."""
        events = find_triggers(square_wave, "edge", level=1.65, edge="rising")

        assert len(events) == 10

    def test_find_triggers_pulse_width(self, pulse_train: WaveformTrace):
        """Test find_triggers with pulse_width type."""
        events = find_triggers(
            pulse_train, "pulse_width", level=1.65, min_width=5e-6, max_width=15e-6
        )

        assert len(events) >= 5

    def test_find_triggers_glitch(self, glitchy_signal: WaveformTrace):
        """Test find_triggers with glitch type."""
        events = find_triggers(glitchy_signal, "glitch", level=1.65, max_width=20e-6)

        assert len(events) == 2

    def test_find_triggers_window(self, sample_rate: float):
        """Test find_triggers with window type."""
        data = np.sin(np.linspace(0, 4 * np.pi, 1000)) * 2

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        events = find_triggers(trace, "window", low_threshold=-1.0, high_threshold=1.0)

        assert len(events) > 0
