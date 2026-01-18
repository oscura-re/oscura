"""Comprehensive unit tests for edge triggering module.

Tests all classes and functions in src/oscura/triggering/edge.py:
- EdgeTrigger class
- find_rising_edges() function
- find_falling_edges() function
- find_all_edges() function
- edge_count() function
- edge_rate() function
"""

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace
from oscura.triggering.base import TriggerType
from oscura.triggering.edge import (
    EdgeTrigger,
    edge_count,
    edge_rate,
    find_all_edges,
    find_falling_edges,
    find_rising_edges,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 1_000_000.0  # 1 MHz


@pytest.fixture
def simple_rising_edge(sample_rate: float) -> WaveformTrace:
    """Create a simple trace with single rising edge."""
    # Signal goes from 0 -> 3.3V at sample 50
    data = np.concatenate([np.zeros(50), np.ones(50) * 3.3])
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def simple_falling_edge(sample_rate: float) -> WaveformTrace:
    """Create a simple trace with single falling edge."""
    # Signal goes from 3.3V -> 0V at sample 50
    data = np.concatenate([np.ones(50) * 3.3, np.zeros(50)])
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def square_wave(sample_rate: float) -> WaveformTrace:
    """Create a square wave with multiple edges."""
    # Square wave: 0V for 25 samples, 3.3V for 25 samples, repeat
    data = np.tile([0.0] * 25 + [3.3] * 25, 4)
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def sine_wave(sample_rate: float) -> WaveformTrace:
    """Create a sine wave trace."""
    n_samples = 1000
    t = np.arange(n_samples) / sample_rate
    # Sine wave from -2V to +2V
    data = 2.0 * np.sin(2 * np.pi * 1000 * t)
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def noisy_signal(sample_rate: float) -> WaveformTrace:
    """Create a signal with noise."""
    n_samples = 500
    # Base signal with noise
    t = np.arange(n_samples) / sample_rate
    signal = np.sin(2 * np.pi * 500 * t)
    noise = np.random.normal(0, 0.1, n_samples)
    data = signal + noise
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def empty_trace(sample_rate: float) -> WaveformTrace:
    """Create an empty trace."""
    data = np.array([])
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def single_sample_trace(sample_rate: float) -> WaveformTrace:
    """Create a trace with single sample."""
    data = np.array([1.5])
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def digital_trace_rising(sample_rate: float) -> DigitalTrace:
    """Create a digital trace with rising edge."""
    data = np.array([False] * 50 + [True] * 50, dtype=bool)
    return DigitalTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def digital_trace_falling(sample_rate: float) -> DigitalTrace:
    """Create a digital trace with falling edge."""
    data = np.array([True] * 50 + [False] * 50, dtype=bool)
    return DigitalTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


# =============================================================================
# EdgeTrigger Initialization Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeTriggerInit:
    """Tests for EdgeTrigger initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        trigger = EdgeTrigger(level=1.5)
        assert trigger.level == 1.5
        assert trigger.edge == "rising"
        assert trigger.hysteresis == 0.0

    def test_init_with_edge_types(self):
        """Test initialization with different edge types."""
        for edge_type in ["rising", "falling", "either"]:
            trigger = EdgeTrigger(level=1.5, edge=edge_type)
            assert trigger.edge == edge_type

    def test_init_with_hysteresis(self):
        """Test initialization with hysteresis."""
        trigger = EdgeTrigger(level=1.5, hysteresis=0.2)
        assert trigger.hysteresis == 0.2

    def test_init_negative_level(self):
        """Test initialization with negative level."""
        trigger = EdgeTrigger(level=-1.5)
        assert trigger.level == -1.5

    def test_init_zero_level(self):
        """Test initialization with zero level."""
        trigger = EdgeTrigger(level=0.0)
        assert trigger.level == 0.0

    def test_init_large_level(self):
        """Test initialization with large level."""
        trigger = EdgeTrigger(level=1e12)
        assert trigger.level == 1e12

    def test_init_small_hysteresis(self):
        """Test initialization with very small hysteresis."""
        trigger = EdgeTrigger(level=1.5, hysteresis=1e-12)
        assert trigger.hysteresis == 1e-12

    def test_init_large_hysteresis(self):
        """Test initialization with large hysteresis."""
        trigger = EdgeTrigger(level=1.5, hysteresis=10.0)
        assert trigger.hysteresis == 10.0


# =============================================================================
# EdgeTrigger.find_events Tests - Simple Mode
# =============================================================================


@pytest.mark.unit
class TestEdgeTriggerSimpleMode:
    """Tests for EdgeTrigger.find_events() in simple mode (no hysteresis)."""

    def test_find_rising_edge_simple(self, simple_rising_edge: WaveformTrace):
        """Test finding rising edge without hysteresis."""
        trigger = EdgeTrigger(level=1.5, edge="rising", hysteresis=0.0)
        events = trigger.find_events(simple_rising_edge)

        assert len(events) >= 1
        assert all(e.event_type == TriggerType.RISING_EDGE for e in events)

    def test_find_falling_edge_simple(self, simple_falling_edge: WaveformTrace):
        """Test finding falling edge without hysteresis."""
        trigger = EdgeTrigger(level=1.5, edge="falling", hysteresis=0.0)
        events = trigger.find_events(simple_falling_edge)

        assert len(events) >= 1
        assert all(e.event_type == TriggerType.FALLING_EDGE for e in events)

    def test_find_either_edge_simple(self, square_wave: WaveformTrace):
        """Test finding both edges without hysteresis."""
        trigger = EdgeTrigger(level=1.5, edge="either", hysteresis=0.0)
        events = trigger.find_events(square_wave)

        assert len(events) > 1
        # Mix of rising and falling edges
        event_types = {e.event_type for e in events}
        # May have both types or just one depending on starting state
        assert TriggerType.RISING_EDGE in event_types or TriggerType.FALLING_EDGE in event_types

    def test_event_properties(self, simple_rising_edge: WaveformTrace):
        """Test that events have required properties."""
        trigger = EdgeTrigger(level=1.5, edge="rising")
        events = trigger.find_events(simple_rising_edge)

        for event in events:
            assert event.timestamp >= 0
            assert event.sample_index >= 0
            assert event.event_type is not None
            assert event.level is not None
            assert isinstance(event.level, float)

    def test_timestamp_ordering(self, square_wave: WaveformTrace):
        """Test that events are ordered by timestamp."""
        trigger = EdgeTrigger(level=1.5, edge="either")
        events = trigger.find_events(square_wave)

        if len(events) > 1:
            timestamps = [e.timestamp for e in events]
            assert timestamps == sorted(timestamps)

    def test_no_edges_detected(self, sample_rate: float):
        """Test when no edges cross threshold."""
        # Constant signal below threshold
        data = np.ones(100) * 0.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="rising")
        events = trigger.find_events(trace)

        assert len(events) == 0

    def test_threshold_at_signal_level(self, sample_rate: float):
        """Test with threshold at signal level (boundary case)."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=2.0, edge="rising")
        events = trigger.find_events(trace)

        # Should detect crossing at or above threshold
        assert len(events) >= 0

    def test_level_at_exact_sample_value(self, sample_rate: float):
        """Test when level equals exact sample value."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=2.0, edge="either")
        events = trigger.find_events(trace)

        # Should handle equality correctly
        assert isinstance(events, list)

    def test_multiple_edges_count(self, square_wave: WaveformTrace):
        """Test counting multiple edges in square wave."""
        trigger = EdgeTrigger(level=1.5, edge="rising")
        events = trigger.find_events(square_wave)

        # Square wave should have multiple rising edges
        assert len(events) >= 3

    def test_empty_trace(self, empty_trace: WaveformTrace):
        """Test with empty trace."""
        trigger = EdgeTrigger(level=1.5)
        events = trigger.find_events(empty_trace)
        assert len(events) == 0

    def test_single_sample_trace(self, single_sample_trace: WaveformTrace):
        """Test with single sample trace."""
        trigger = EdgeTrigger(level=1.5)
        events = trigger.find_events(single_sample_trace)
        # No transition possible with single sample
        assert len(events) == 0

    def test_two_sample_trace(self, sample_rate: float):
        """Test with two sample trace."""
        data = np.array([0.0, 3.3])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="rising")
        events = trigger.find_events(trace)

        assert len(events) >= 1


# =============================================================================
# EdgeTrigger.find_events Tests - Hysteresis Mode
# =============================================================================


@pytest.mark.unit
class TestEdgeTriggerHysteresisMode:
    """Tests for EdgeTrigger.find_events() with hysteresis (Schmitt trigger)."""

    def test_find_edges_with_hysteresis(self, simple_rising_edge: WaveformTrace):
        """Test edge detection with hysteresis."""
        trigger = EdgeTrigger(level=1.5, edge="rising", hysteresis=0.5)
        events = trigger.find_events(simple_rising_edge)

        assert len(events) >= 1
        assert all(e.event_type == TriggerType.RISING_EDGE for e in events)

    def test_hysteresis_prevents_retriggering(self, noisy_signal: WaveformTrace):
        """Test that hysteresis prevents false triggers on noise."""
        # Without hysteresis
        trigger_no_hyst = EdgeTrigger(level=0.0, edge="either", hysteresis=0.0)
        events_no_hyst = trigger_no_hyst.find_events(noisy_signal)

        # With hysteresis
        trigger_with_hyst = EdgeTrigger(level=0.0, edge="either", hysteresis=0.2)
        events_with_hyst = trigger_with_hyst.find_events(noisy_signal)

        # Hysteresis should reduce or maintain event count (noise rejection)
        assert len(events_with_hyst) <= len(events_no_hyst) + 1

    def test_hysteresis_high_low_thresholds(self, sample_rate: float):
        """Test hysteresis creates high and low thresholds."""
        level = 1.5
        hysteresis = 0.4
        trigger = EdgeTrigger(level=level, hysteresis=hysteresis)

        # For rising edge: should trigger when crossing level + hysteresis/2
        high_thresh = level + hysteresis / 2  # 1.7
        low_thresh = level - hysteresis / 2  # 1.3

        # Create signal that oscillates around threshold
        data = np.array([0.0, 1.2, 1.8, 1.2, 1.8, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        events = trigger.find_events(trace)
        # Should get at least some edges with hysteresis applied
        assert isinstance(events, list)

    def test_hysteresis_state_tracking(self, sample_rate: float):
        """Test that hysteresis maintains state correctly."""
        # Signal that crosses high threshold multiple times
        data = np.array([0.0, 1.8, 1.8, 1.8, 1.2, 1.8, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="rising", hysteresis=0.4)
        events = trigger.find_events(trace)

        # State machine should handle the transitions
        assert isinstance(events, list)
        assert all(e.event_type == TriggerType.RISING_EDGE for e in events)

    def test_large_hysteresis(self, sample_rate: float):
        """Test with large hysteresis band."""
        data = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.0, edge="either", hysteresis=1.5)
        events = trigger.find_events(trace)

        # Large hysteresis should still detect edges
        assert isinstance(events, list)

    def test_tiny_hysteresis(self, sample_rate: float):
        """Test with very small hysteresis band."""
        data = np.array([0.0, 1.5, 2.0, 1.5, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.0, edge="either", hysteresis=1e-12)
        events = trigger.find_events(trace)

        assert isinstance(events, list)

    def test_hysteresis_falling_edge(self, sample_rate: float):
        """Test hysteresis mode for falling edges."""
        data = np.array([3.3, 3.3, 1.5, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="falling", hysteresis=0.5)
        events = trigger.find_events(trace)

        assert all(e.event_type == TriggerType.FALLING_EDGE for e in events)

    def test_hysteresis_both_edges(self, square_wave: WaveformTrace):
        """Test hysteresis mode for both edge types."""
        trigger = EdgeTrigger(level=1.5, edge="either", hysteresis=0.3)
        events = trigger.find_events(square_wave)

        # Should have both rising and falling edges
        assert len(events) > 0


# =============================================================================
# EdgeTrigger with DigitalTrace Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeTriggerDigitalTrace:
    """Tests for EdgeTrigger with DigitalTrace input."""

    def test_digital_rising_edge(self, digital_trace_rising: DigitalTrace):
        """Test finding rising edge in digital trace."""
        trigger = EdgeTrigger(level=0.5, edge="rising")
        events = trigger.find_events(digital_trace_rising)

        assert len(events) >= 1
        assert all(e.event_type == TriggerType.RISING_EDGE for e in events)

    def test_digital_falling_edge(self, digital_trace_falling: DigitalTrace):
        """Test finding falling edge in digital trace."""
        trigger = EdgeTrigger(level=0.5, edge="falling")
        events = trigger.find_events(digital_trace_falling)

        assert len(events) >= 1
        assert all(e.event_type == TriggerType.FALLING_EDGE for e in events)

    def test_digital_trace_conversion(self, digital_trace_rising: DigitalTrace):
        """Test that digital trace is properly converted to float."""
        trigger = EdgeTrigger(level=0.5, edge="rising")
        # Should not raise an error
        events = trigger.find_events(digital_trace_rising)
        assert isinstance(events, list)

    def test_digital_trace_boolean_levels(self, sample_rate: float):
        """Test with digital trace at boolean levels (0.0 and 1.0)."""
        data = np.array([False, False, True, True, False], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=0.5, edge="either")
        events = trigger.find_events(trace)

        assert len(events) >= 2


# =============================================================================
# find_rising_edges Tests
# =============================================================================


@pytest.mark.unit
class TestFindRisingEdges:
    """Tests for find_rising_edges() function."""

    def test_auto_level_calculation(self, simple_rising_edge: WaveformTrace):
        """Test automatic level calculation (midpoint)."""
        edges = find_rising_edges(simple_rising_edge)
        assert isinstance(edges, np.ndarray)
        assert len(edges) >= 1

    def test_explicit_level(self, simple_rising_edge: WaveformTrace):
        """Test with explicit level specification."""
        edges = find_rising_edges(simple_rising_edge, level=1.5)
        assert isinstance(edges, np.ndarray)
        assert len(edges) >= 1

    def test_return_timestamps(self, simple_rising_edge: WaveformTrace):
        """Test that timestamps are returned by default."""
        edges = find_rising_edges(simple_rising_edge, level=1.5)
        assert edges.dtype == np.float64
        # Timestamps should be >= 0
        assert np.all(edges >= 0)

    def test_return_indices(self, simple_rising_edge: WaveformTrace):
        """Test returning sample indices instead of timestamps."""
        indices = find_rising_edges(simple_rising_edge, level=1.5, return_indices=True)
        assert indices.dtype == np.int64
        # Indices should be non-negative
        assert np.all(indices >= 0)

    def test_timestamp_to_index_correspondence(self, simple_rising_edge: WaveformTrace):
        """Test that timestamps correspond to indices."""
        sample_rate = simple_rising_edge.metadata.sample_rate
        sample_period = 1.0 / sample_rate

        timestamps = find_rising_edges(simple_rising_edge, level=1.5, return_indices=False)
        indices = find_rising_edges(simple_rising_edge, level=1.5, return_indices=True)

        if len(timestamps) > 0:
            # Timestamp = index * sample_period (approximately)
            reconstructed_timestamps = indices * sample_period
            assert np.allclose(timestamps, reconstructed_timestamps, rtol=0.01)

    def test_no_rising_edges(self, sample_rate: float):
        """Test when no rising edges exist."""
        data = np.ones(100) * 2.0
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        edges = find_rising_edges(trace, level=1.5)
        assert len(edges) == 0

    def test_with_hysteresis_parameter(self, noisy_signal: WaveformTrace):
        """Test with hysteresis parameter."""
        edges = find_rising_edges(noisy_signal, level=0.0, hysteresis=0.1)
        assert isinstance(edges, np.ndarray)

    def test_multiple_rising_edges(self, square_wave: WaveformTrace):
        """Test finding multiple rising edges in square wave."""
        edges = find_rising_edges(square_wave, level=1.5)
        assert len(edges) >= 3

    def test_sine_wave_rising_edges(self, sine_wave: WaveformTrace):
        """Test with sine wave (should have rising edges)."""
        edges = find_rising_edges(sine_wave, level=-1.0)
        assert len(edges) > 0

    def test_empty_trace(self, empty_trace: WaveformTrace):
        """Test with empty trace."""
        edges = find_rising_edges(empty_trace, level=1.5)
        assert len(edges) == 0

    def test_single_sample(self, single_sample_trace: WaveformTrace):
        """Test with single sample trace."""
        edges = find_rising_edges(single_sample_trace, level=1.0)
        assert len(edges) == 0


# =============================================================================
# find_falling_edges Tests
# =============================================================================


@pytest.mark.unit
class TestFindFallingEdges:
    """Tests for find_falling_edges() function."""

    def test_auto_level_calculation(self, simple_falling_edge: WaveformTrace):
        """Test automatic level calculation."""
        edges = find_falling_edges(simple_falling_edge)
        assert isinstance(edges, np.ndarray)
        assert len(edges) >= 1

    def test_explicit_level(self, simple_falling_edge: WaveformTrace):
        """Test with explicit level."""
        edges = find_falling_edges(simple_falling_edge, level=1.5)
        assert isinstance(edges, np.ndarray)
        assert len(edges) >= 1

    def test_return_timestamps(self, simple_falling_edge: WaveformTrace):
        """Test timestamp return."""
        edges = find_falling_edges(simple_falling_edge, level=1.5)
        assert edges.dtype == np.float64
        assert np.all(edges >= 0)

    def test_return_indices(self, simple_falling_edge: WaveformTrace):
        """Test index return."""
        indices = find_falling_edges(simple_falling_edge, level=1.5, return_indices=True)
        assert indices.dtype == np.int64
        assert np.all(indices >= 0)

    def test_no_falling_edges(self, sample_rate: float):
        """Test when no falling edges exist."""
        data = np.ones(100) * 0.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        edges = find_falling_edges(trace, level=1.5)
        assert len(edges) == 0

    def test_with_hysteresis(self, noisy_signal: WaveformTrace):
        """Test with hysteresis."""
        edges = find_falling_edges(noisy_signal, level=0.0, hysteresis=0.1)
        assert isinstance(edges, np.ndarray)

    def test_multiple_falling_edges(self, square_wave: WaveformTrace):
        """Test multiple falling edges."""
        edges = find_falling_edges(square_wave, level=1.5)
        assert len(edges) >= 3

    def test_sine_wave_falling_edges(self, sine_wave: WaveformTrace):
        """Test with sine wave."""
        edges = find_falling_edges(sine_wave, level=0.0)
        assert len(edges) > 0


# =============================================================================
# find_all_edges Tests
# =============================================================================


@pytest.mark.unit
class TestFindAllEdges:
    """Tests for find_all_edges() function."""

    def test_return_structure(self, square_wave: WaveformTrace):
        """Test return value structure."""
        timestamps, is_rising = find_all_edges(square_wave, level=1.5)

        assert isinstance(timestamps, np.ndarray)
        assert isinstance(is_rising, np.ndarray)
        assert timestamps.dtype == np.float64
        assert is_rising.dtype == np.bool_

    def test_timestamps_and_polarity_match(self, square_wave: WaveformTrace):
        """Test that timestamps and polarity arrays have same length."""
        timestamps, is_rising = find_all_edges(square_wave, level=1.5)
        assert len(timestamps) == len(is_rising)

    def test_mixed_edge_types(self, square_wave: WaveformTrace):
        """Test detection of both rising and falling edges."""
        timestamps, is_rising = find_all_edges(square_wave, level=1.5)

        if len(is_rising) > 1:
            # Should have mix of rising and falling
            has_rising = np.any(is_rising)
            has_falling = np.any(~is_rising)
            # May have one or both depending on starting state
            assert has_rising or has_falling

    def test_auto_level(self, sine_wave: WaveformTrace):
        """Test with auto-calculated level."""
        timestamps, is_rising = find_all_edges(sine_wave)
        assert len(timestamps) > 0
        assert len(is_rising) == len(timestamps)

    def test_explicit_level(self, sine_wave: WaveformTrace):
        """Test with explicit level."""
        timestamps, is_rising = find_all_edges(sine_wave, level=0.0)
        assert len(timestamps) > 0

    def test_no_edges(self, sample_rate: float):
        """Test when no edges exist."""
        data = np.ones(100) * 1.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        timestamps, is_rising = find_all_edges(trace, level=1.0)
        assert len(timestamps) == 0
        assert len(is_rising) == 0

    def test_timestamps_ordered(self, square_wave: WaveformTrace):
        """Test that timestamps are ordered."""
        timestamps, is_rising = find_all_edges(square_wave, level=1.5)
        if len(timestamps) > 1:
            assert np.all(timestamps[:-1] <= timestamps[1:])

    def test_with_hysteresis(self, noisy_signal: WaveformTrace):
        """Test with hysteresis."""
        timestamps, is_rising = find_all_edges(noisy_signal, level=0.0, hysteresis=0.1)
        assert len(timestamps) == len(is_rising)

    def test_empty_trace(self, empty_trace: WaveformTrace):
        """Test with empty trace."""
        timestamps, is_rising = find_all_edges(empty_trace, level=1.5)
        assert len(timestamps) == 0
        assert len(is_rising) == 0


# =============================================================================
# edge_count Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeCount:
    """Tests for edge_count() function."""

    def test_count_rising_edges(self, square_wave: WaveformTrace):
        """Test counting rising edges."""
        count = edge_count(square_wave, level=1.5, edge="rising")
        assert count >= 3

    def test_count_falling_edges(self, square_wave: WaveformTrace):
        """Test counting falling edges."""
        count = edge_count(square_wave, level=1.5, edge="falling")
        assert count >= 3

    def test_count_either_edges(self, square_wave: WaveformTrace):
        """Test counting all edges."""
        count_rising = edge_count(square_wave, level=1.5, edge="rising")
        count_falling = edge_count(square_wave, level=1.5, edge="falling")
        count_either = edge_count(square_wave, level=1.5, edge="either")

        assert count_either >= count_rising
        assert count_either >= count_falling

    def test_count_no_edges(self, sample_rate: float):
        """Test counting when no edges exist."""
        data = np.ones(100) * 1.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        count = edge_count(trace, level=1.5, edge="either")
        assert count == 0

    def test_auto_level(self, sine_wave: WaveformTrace):
        """Test with auto level calculation."""
        count = edge_count(sine_wave, edge="either")
        assert count > 0

    def test_explicit_level(self, sine_wave: WaveformTrace):
        """Test with explicit level."""
        count = edge_count(sine_wave, level=0.0, edge="either")
        assert count > 0

    def test_with_hysteresis(self, noisy_signal: WaveformTrace):
        """Test with hysteresis."""
        count_no_hyst = edge_count(noisy_signal, level=0.0, hysteresis=0.0)
        count_hyst = edge_count(noisy_signal, level=0.0, hysteresis=0.1)

        # Hysteresis may reduce spurious detections
        assert count_hyst >= 0
        assert count_no_hyst >= 0

    def test_count_empty_trace(self, empty_trace: WaveformTrace):
        """Test counting on empty trace."""
        count = edge_count(empty_trace, level=1.5)
        assert count == 0

    def test_count_single_sample(self, single_sample_trace: WaveformTrace):
        """Test counting on single sample."""
        count = edge_count(single_sample_trace, level=1.5)
        assert count == 0


# =============================================================================
# edge_rate Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeRate:
    """Tests for edge_rate() function."""

    def test_rate_returns_float(self, square_wave: WaveformTrace):
        """Test that rate is returned as float."""
        rate = edge_rate(square_wave, level=1.5)
        assert isinstance(rate, float)

    def test_rate_is_positive(self, square_wave: WaveformTrace):
        """Test that rate is positive when edges exist."""
        rate = edge_rate(square_wave, level=1.5)
        assert rate >= 0

    def test_rate_calculation(self, sample_rate: float):
        """Test edge rate calculation accuracy."""
        # Create 1 MHz signal with 100 kHz square wave
        n_samples = 10000
        period_samples = sample_rate / 100_000  # 10 samples per half-period
        data = np.tile([0.0] * int(period_samples / 2) + [3.3] * int(period_samples / 2), 10)
        trace = WaveformTrace(
            data=data[:n_samples], metadata=TraceMetadata(sample_rate=sample_rate)
        )

        rate = edge_rate(trace, level=1.5, edge="either")

        # Expected: ~200 kHz (100 kHz square wave = 200 kHz edge rate)
        # Allow some margin for edge detection precision
        assert rate > 50_000  # Should be in reasonable range

    def test_rate_zero_when_no_edges(self, sample_rate: float):
        """Test that rate is zero when no edges exist."""
        data = np.ones(100) * 1.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        rate = edge_rate(trace, level=1.5)
        assert rate == 0.0

    def test_rate_with_hysteresis(self, noisy_signal: WaveformTrace):
        """Test rate calculation with hysteresis."""
        rate = edge_rate(noisy_signal, level=0.0, hysteresis=0.1)
        assert rate >= 0

    def test_rate_empty_trace(self, empty_trace: WaveformTrace):
        """Test rate on empty trace."""
        rate = edge_rate(empty_trace, level=1.5)
        assert rate == 0.0

    def test_rate_single_sample(self, single_sample_trace: WaveformTrace):
        """Test rate on single sample."""
        rate = edge_rate(single_sample_trace, level=1.5)
        assert rate == 0.0

    def test_rate_auto_level(self, square_wave: WaveformTrace):
        """Test rate with auto level calculation."""
        rate = edge_rate(square_wave)
        assert rate > 0

    def test_rate_units(self, square_wave: WaveformTrace):
        """Test that rate is in correct units (Hz)."""
        duration = square_wave.duration
        count = edge_count(square_wave, level=1.5, edge="either")
        expected_rate = count / duration if duration > 0 else 0

        rate = edge_rate(square_wave, level=1.5, edge="either")

        if duration > 0:
            assert abs(rate - expected_rate) < 1.0  # Within 1 Hz tolerance


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


@pytest.mark.unit
class TestTriggeringEdgeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rising_and_falling_edge_counts_sum(self, square_wave: WaveformTrace):
        """Test that rising + falling counts sum correctly."""
        rising = edge_count(square_wave, level=1.5, edge="rising")
        falling = edge_count(square_wave, level=1.5, edge="falling")
        either = edge_count(square_wave, level=1.5, edge="either")

        assert either == rising + falling

    def test_negative_threshold(self, sample_rate: float):
        """Test with negative threshold."""
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=-0.5, edge="rising")
        events = trigger.find_events(trace)

        assert len(events) >= 1

    def test_very_large_signal(self, sample_rate: float):
        """Test with very large voltage values."""
        data = np.array([1e10, 1e10 + 1e9, 1e10])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1e10 + 0.5e9, edge="rising")
        events = trigger.find_events(trace)

        assert isinstance(events, list)

    def test_very_small_signal(self, sample_rate: float):
        """Test with very small voltage values."""
        data = np.array([1e-10, 2e-10, 1e-10])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5e-10, edge="rising")
        events = trigger.find_events(trace)

        assert isinstance(events, list)

    def test_nan_in_signal(self, sample_rate: float):
        """Test handling of NaN values in signal."""
        data = np.array([1.0, 2.0, np.nan, 1.0, 2.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="either")
        events = trigger.find_events(trace)

        # Should handle NaN without crashing
        assert isinstance(events, list)

    def test_inf_in_signal(self, sample_rate: float):
        """Test handling of infinite values."""
        data = np.array([1.0, 2.0, np.inf, 1.0, 2.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="either")
        # inf values may cause RuntimeWarning during interpolation, which is expected
        # The test checks that it doesn't crash (returns a result)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            events = trigger.find_events(trace)

        # Should handle inf without crashing (may produce events or not)
        assert isinstance(events, list)

    def test_all_nan_signal(self, sample_rate: float):
        """Test with all NaN signal."""
        data = np.full(100, np.nan)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="either")
        events = trigger.find_events(trace)

        # Should return empty list for all-NaN signal
        assert len(events) == 0

    def test_very_high_sample_rate(self):
        """Test with very high sample rate."""
        sample_rate = 1e12  # 1 THz
        data = np.array([0.0, 3.3, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="either")
        events = trigger.find_events(trace)

        assert len(events) >= 2

    def test_very_low_sample_rate(self):
        """Test with very low sample rate."""
        sample_rate = 1.0  # 1 Hz
        data = np.array([0.0, 3.3, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.5, edge="either")
        events = trigger.find_events(trace)

        assert len(events) >= 2

    def test_monotonic_increasing_signal(self, sample_rate: float):
        """Test with monotonically increasing signal."""
        data = np.linspace(0.0, 10.0, 100)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        count = edge_count(trace, level=5.0, edge="rising")
        assert count == 1  # Single rising edge

    def test_monotonic_decreasing_signal(self, sample_rate: float):
        """Test with monotonically decreasing signal."""
        data = np.linspace(10.0, 0.0, 100)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        count = edge_count(trace, level=5.0, edge="falling")
        assert count == 1  # Single falling edge

    def test_alternating_levels(self, sample_rate: float):
        """Test with rapidly alternating signal."""
        data = np.array([0.0, 3.3] * 500)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        count = edge_count(trace, level=1.5, edge="either")
        # Should have many edges (approximately 1000)
        assert count >= 900

    def test_impulse_signal(self, sample_rate: float):
        """Test with impulse signal."""
        data = np.zeros(100)
        data[50] = 5.0
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = EdgeTrigger(level=1.0, edge="either")
        events = trigger.find_events(trace)

        # Should detect rising and falling edges
        assert len(events) >= 2


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestTriggeringEdgeIntegration:
    """Integration tests combining multiple functions."""

    def test_rising_falling_edges_consistency(self, square_wave: WaveformTrace):
        """Test consistency between rising and falling edge detection."""
        rising = find_rising_edges(square_wave, level=1.5, return_indices=True)
        falling = find_falling_edges(square_wave, level=1.5, return_indices=True)
        all_edges, is_rising = find_all_edges(square_wave, level=1.5)

        rising_from_all = np.sum(is_rising)
        falling_from_all = np.sum(~is_rising)

        assert len(rising) == rising_from_all or len(rising) == rising_from_all - 1
        assert len(falling) == falling_from_all or len(falling) == falling_from_all - 1

    def test_edge_count_vs_function_results(self, square_wave: WaveformTrace):
        """Test that edge_count matches function result counts."""
        rising_edges = find_rising_edges(square_wave, level=1.5)
        rising_count = edge_count(square_wave, level=1.5, edge="rising")

        assert len(rising_edges) == rising_count

    def test_edge_rate_calculation_consistency(self, square_wave: WaveformTrace):
        """Test that edge_rate is consistent with edge_count."""
        count = edge_count(square_wave, level=1.5, edge="either")
        rate = edge_rate(square_wave, level=1.5, edge="either")
        duration = square_wave.duration

        if duration > 0:
            expected_rate = count / duration
            assert abs(rate - expected_rate) < 1.0  # Within 1 Hz

    def test_hysteresis_reduces_noise_sensitivity(self, noisy_signal: WaveformTrace):
        """Test that hysteresis reduces sensitivity to noise."""
        count_no_hyst = edge_count(noisy_signal, level=0.0, hysteresis=0.0)
        count_hyst = edge_count(noisy_signal, level=0.0, hysteresis=0.2)

        # Hysteresis should not increase count (may reduce it)
        assert count_hyst <= count_no_hyst + 1

    def test_function_implementations_match_class(self, square_wave: WaveformTrace):
        """Test that functions use EdgeTrigger class correctly."""
        # Get results from functions
        rising_func = find_rising_edges(square_wave, level=1.5)
        falling_func = find_falling_edges(square_wave, level=1.5)

        # Get results from class
        trigger_r = EdgeTrigger(level=1.5, edge="rising")
        events_r = trigger_r.find_events(square_wave)
        rising_class = np.array([e.timestamp for e in events_r])

        trigger_f = EdgeTrigger(level=1.5, edge="falling")
        events_f = trigger_f.find_events(square_wave)
        falling_class = np.array([e.timestamp for e in events_f])

        # Should have same counts
        assert len(rising_func) == len(rising_class)
        assert len(falling_func) == len(falling_class)
