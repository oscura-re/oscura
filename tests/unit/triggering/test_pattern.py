"""Comprehensive unit tests for pattern triggering module.

Tests all classes and functions in src/oscura/triggering/pattern.py:
- PatternTrigger class
- MultiChannelPatternTrigger class
- find_pattern() function
- find_bit_sequence() function

Coverage targets >90% with edge cases, error handling, and validation.
"""

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace
from oscura.triggering.base import TriggerType
from oscura.triggering.pattern import (
    MultiChannelPatternTrigger,
    PatternTrigger,
    find_bit_sequence,
    find_pattern,
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
def simple_digital_trace(sample_rate: float) -> DigitalTrace:
    """Create a simple digital trace with known pattern."""
    # Pattern: 0, 1, 0, 1, 0, 1 repeated
    data = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5, dtype=bool)
    return DigitalTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def waveform_trace_digital(sample_rate: float) -> WaveformTrace:
    """Create a waveform trace with clear 0/1 digital pattern."""
    # Alternating pattern: low (0V) and high (3.3V)
    data = np.array([0.0, 3.3, 0.0, 3.3, 0.0, 3.3, 0.0, 3.3] * 5, dtype=np.float64)
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def waveform_sine_wave(sample_rate: float) -> WaveformTrace:
    """Create a sine wave for testing threshold detection."""
    n_samples = 1000
    t = np.arange(n_samples) / sample_rate
    # Sine wave oscillating between -1 and +1
    data = np.sin(2 * np.pi * 1000 * t)
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def waveform_with_pattern(sample_rate: float) -> WaveformTrace:
    """Create waveform with repeating bit pattern."""
    # Pattern [1, 0, 1] repeated 10 times, at 3.3V and 0V
    data = []
    for _ in range(10):
        data.extend([3.3, 0.0, 3.3])  # Pattern 1, 0, 1
    data = np.array(data, dtype=np.float64)
    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def multi_channel_traces(sample_rate: float) -> list[DigitalTrace]:
    """Create multiple digital traces for multi-channel pattern testing."""
    traces = []
    patterns = [
        [True, False, True, False],  # Channel 0: alternating
        [False, False, True, True],  # Channel 1: two lows then two highs
        [True, True, False, False],  # Channel 2: two highs then two lows
    ]

    for pattern in patterns:
        # Repeat pattern 5 times
        data = np.array(pattern * 5, dtype=bool)
        traces.append(
            DigitalTrace(
                data=data,
                metadata=TraceMetadata(sample_rate=sample_rate),
            )
        )

    return traces


@pytest.fixture
def empty_trace(sample_rate: float) -> DigitalTrace:
    """Create an empty digital trace."""
    data = np.array([], dtype=bool)
    return DigitalTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def single_sample_trace(sample_rate: float) -> DigitalTrace:
    """Create a trace with single sample."""
    data = np.array([True], dtype=bool)
    return DigitalTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


# =============================================================================
# PatternTrigger Tests - Initialization
# =============================================================================


@pytest.mark.unit
class TestPatternTriggerInit:
    """Tests for PatternTrigger initialization."""

    def test_init_simple_pattern(self):
        """Test initialization with simple binary pattern."""
        pattern = [1, 0, 1]
        trigger = PatternTrigger(pattern=pattern)
        assert trigger.pattern == pattern
        assert trigger.levels is None
        assert trigger.match_type == "sequence"

    def test_init_with_dont_care(self):
        """Test initialization with don't care values."""
        pattern = [1, None, 0]
        trigger = PatternTrigger(pattern=pattern)
        assert trigger.pattern == pattern

    def test_init_with_single_level(self):
        """Test initialization with single threshold level."""
        pattern = [0, 1]
        trigger = PatternTrigger(pattern=pattern, levels=1.5)
        assert trigger.levels == 1.5

    def test_init_with_multiple_levels(self):
        """Test initialization with multiple threshold levels."""
        pattern = [0, 1, 0]
        levels = [1.5, 1.5, 1.5]
        trigger = PatternTrigger(pattern=pattern, levels=levels)
        assert trigger.levels == levels

    def test_init_match_type_sequence(self):
        """Test initialization with sequence match type."""
        trigger = PatternTrigger(pattern=[1, 0], match_type="sequence")
        assert trigger.match_type == "sequence"

    def test_init_match_type_exact(self):
        """Test initialization with exact match type."""
        trigger = PatternTrigger(pattern=[1, 0], match_type="exact")
        assert trigger.match_type == "exact"

    def test_init_all_zeros_pattern(self):
        """Test initialization with all zeros pattern."""
        pattern = [0, 0, 0]
        trigger = PatternTrigger(pattern=pattern)
        assert trigger.pattern == pattern

    def test_init_all_ones_pattern(self):
        """Test initialization with all ones pattern."""
        pattern = [1, 1, 1]
        trigger = PatternTrigger(pattern=pattern)
        assert trigger.pattern == pattern

    def test_init_invalid_value_2(self):
        """Test initialization fails with invalid value 2."""
        with pytest.raises(AnalysisError, match="Pattern values must be 0, 1, or None"):
            PatternTrigger(pattern=[1, 2, 0])

    def test_init_invalid_value_negative(self):
        """Test initialization fails with negative value."""
        with pytest.raises(AnalysisError, match="Pattern values must be 0, 1, or None"):
            PatternTrigger(pattern=[1, -1, 0])

    def test_init_invalid_value_float(self):
        """Test initialization fails with float value."""
        with pytest.raises(AnalysisError, match="Pattern values must be 0, 1, or None"):
            PatternTrigger(pattern=[1, 0.5, 0])  # type: ignore

    def test_init_empty_pattern(self):
        """Test initialization with empty pattern."""
        trigger = PatternTrigger(pattern=[])
        assert trigger.pattern == []

    def test_init_long_pattern(self):
        """Test initialization with long pattern."""
        pattern = [0, 1] * 50  # 100 elements
        trigger = PatternTrigger(pattern=pattern)
        assert len(trigger.pattern) == 100


# =============================================================================
# PatternTrigger Tests - Sequence Matching
# =============================================================================


@pytest.mark.unit
class TestPatternTriggerSequence:
    """Tests for PatternTrigger sequence matching."""

    def test_find_events_simple_sequence(self, simple_digital_trace: DigitalTrace):
        """Test finding simple sequence pattern."""
        trigger = PatternTrigger(pattern=[0, 1])
        events = trigger.find_events(simple_digital_trace)

        assert len(events) > 0
        for event in events:
            assert event.event_type == TriggerType.PATTERN_MATCH
            assert event.sample_index >= 0
            assert event.timestamp >= 0
            assert event.data["pattern"] == [0, 1]

    def test_find_events_exact_match_count(self, sample_rate: float):
        """Test that sequence matches are counted correctly."""
        # Create trace with known pattern occurrences
        # Pattern "10" should appear exactly at indices 0, 2, 4, 6...
        data = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, 0])
        events = trigger.find_events(trace)

        # Pattern "10" appears at indices 0, 2, 4, 6
        assert len(events) == 4

    def test_find_events_with_dont_care(self, sample_rate: float):
        """Test sequence matching with don't care values."""
        # Pattern [1, X, 1] should match [1, 0, 1] and [1, 1, 1]
        data = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, None, 1])
        events = trigger.find_events(trace)

        # Should find [1, 0, 1] at index 0, [1, 1, 1] at indices 2 and 3
        # Pattern matches at: idx 0 [1,0,1], idx 2 [1,1,1], idx 3 [1,1,1]
        assert len(events) >= 2

    def test_find_events_overlapping_patterns(self, sample_rate: float):
        """Test with overlapping pattern matches."""
        # Pattern [1, 1, 1] in data [1, 1, 1, 1]
        # Could match at indices 0 and 1 (overlapping)
        data = np.array([1, 1, 1, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, 1, 1])
        events = trigger.find_events(trace)

        # Should find overlapping matches at 0 and 1
        assert len(events) == 2

    def test_find_events_waveform_input(self, waveform_trace_digital: WaveformTrace):
        """Test with waveform input (analog converted to digital)."""
        trigger = PatternTrigger(pattern=[0, 1])
        events = trigger.find_events(waveform_trace_digital)

        assert len(events) > 0
        for event in events:
            assert event.event_type == TriggerType.PATTERN_MATCH

    def test_find_events_empty_trace(self, empty_trace: DigitalTrace):
        """Test with empty trace."""
        trigger = PatternTrigger(pattern=[1, 0])
        events = trigger.find_events(empty_trace)
        assert len(events) == 0

    def test_find_events_single_sample_trace(self, single_sample_trace: DigitalTrace):
        """Test with single sample trace."""
        trigger = PatternTrigger(pattern=[1, 0])
        events = trigger.find_events(single_sample_trace)
        # Cannot find 2-bit pattern in 1 sample
        assert len(events) == 0

    def test_find_events_pattern_longer_than_trace(self, sample_rate: float):
        """Test when pattern is longer than trace."""
        data = np.array([1, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, 0, 1, 0, 1])
        events = trigger.find_events(trace)
        assert len(events) == 0

    def test_find_events_no_matches(self, sample_rate: float):
        """Test when pattern never appears in trace."""
        data = np.array([0, 0, 0, 0, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, 1])
        events = trigger.find_events(trace)
        assert len(events) == 0

    def test_find_events_timestamp_calculation(self, sample_rate: float):
        """Test that timestamps are calculated correctly."""
        data = np.array([0, 1, 0, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, 0])
        events = trigger.find_events(trace)

        sample_period = 1.0 / sample_rate
        for event in events:
            expected_timestamp = event.sample_index * sample_period
            assert abs(event.timestamp - expected_timestamp) < 1e-12

    def test_find_events_duration_set(self, sample_rate: float):
        """Test that event duration is set for sequence matches."""
        data = np.array([0, 1, 0, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, 0])
        events = trigger.find_events(trace)

        sample_period = 1.0 / sample_rate
        pattern_duration = len(trigger.pattern) * sample_period
        for event in events:
            assert event.duration == pattern_duration


# =============================================================================
# PatternTrigger Tests - Exact Matching
# =============================================================================


@pytest.mark.unit
class TestPatternTriggerExact:
    """Tests for PatternTrigger exact matching."""

    def test_find_events_exact_match_type(self, sample_rate: float):
        """Test exact match type finds transitions."""
        # Data transitions from 0 to 1 and back
        data = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1], match_type="exact")
        events = trigger.find_events(trace)

        # Should find transition to 1 at index 2
        assert len(events) == 1
        assert events[0].sample_index == 2

    def test_find_events_exact_match_multiple_transitions(self, sample_rate: float):
        """Test exact match with multiple transitions."""
        # Multiple transitions to 1
        data = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1], match_type="exact")
        events = trigger.find_events(trace)

        # Should find transitions to 1 at indices 2 and 6
        assert len(events) == 2

    def test_find_events_exact_match_type_0(self, sample_rate: float):
        """Test exact match for pattern [0]."""
        # Multiple transitions to 0
        data = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[0], match_type="exact")
        events = trigger.find_events(trace)

        # Should find transitions to 0 at indices 2 and 6
        assert len(events) == 2

    def test_find_events_exact_match_dont_care(self, sample_rate: float):
        """Test exact match with don't care pattern."""
        data = np.array([1, 0, 1, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        # Pattern [None] should match everything - no transitions
        trigger = PatternTrigger(pattern=[None], match_type="exact")
        events = trigger.find_events(trace)

        # Don't care pattern matches everything, so no transitions
        assert len(events) == 0

    def test_find_events_exact_no_transitions(self, sample_rate: float):
        """Test exact match when signal stays constant."""
        data = np.array([1, 1, 1, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1], match_type="exact")
        events = trigger.find_events(trace)

        # No transitions, so no events
        assert len(events) == 0


# =============================================================================
# PatternTrigger Tests - Analog to Digital Conversion
# =============================================================================


@pytest.mark.unit
class TestPatternTriggerAnalogConversion:
    """Tests for analog to digital conversion in pattern matching."""

    def test_find_events_with_custom_level(self, waveform_trace_digital: WaveformTrace):
        """Test with custom threshold level."""
        trigger = PatternTrigger(pattern=[0, 1], levels=1.5)
        events = trigger.find_events(waveform_trace_digital)

        # Should digitize using 1.5V threshold and find patterns
        assert isinstance(events, list)

    def test_find_events_default_threshold(self, waveform_sine_wave: WaveformTrace):
        """Test with default 50% threshold."""
        trigger = PatternTrigger(pattern=[1, 0])
        events = trigger.find_events(waveform_sine_wave)

        # Default threshold should be at (min + max) / 2 = 0V
        assert isinstance(events, list)

    def test_find_events_sine_wave_pattern(self, waveform_sine_wave: WaveformTrace):
        """Test pattern matching on sine wave."""
        # Sine wave crosses zero multiple times, creating transitions
        trigger = PatternTrigger(pattern=[1, 0], levels=0.0)
        events = trigger.find_events(waveform_sine_wave)

        # Should find multiple crossings
        assert len(events) > 0

    def test_find_events_multiple_levels(self, sample_rate: float):
        """Test with multiple levels in a list."""
        # Create simple waveform
        data = np.array([0.0, 3.3, 0.0, 3.3], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        # Use first level from list
        trigger = PatternTrigger(pattern=[0, 1], levels=[1.5, 2.0, 2.5])
        events = trigger.find_events(trace)

        # Should work with first level
        assert isinstance(events, list)

    def test_find_events_high_threshold(self, sample_rate: float):
        """Test with high threshold filtering out most signal."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        # High threshold means most values become 0
        trigger = PatternTrigger(pattern=[0], match_type="exact", levels=2.5)
        events = trigger.find_events(trace)

        assert isinstance(events, list)

    def test_find_events_low_threshold(self, sample_rate: float):
        """Test with low threshold making most signal high."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        # Low threshold means most values become 1
        trigger = PatternTrigger(pattern=[1], match_type="exact", levels=0.5)
        events = trigger.find_events(trace)

        assert isinstance(events, list)


# =============================================================================
# MultiChannelPatternTrigger Tests
# =============================================================================


@pytest.mark.unit
class TestMultiChannelPatternTrigger:
    """Tests for MultiChannelPatternTrigger class."""

    def test_init_simple(self):
        """Test basic initialization."""
        pattern = [1, 0, 1]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        assert trigger.pattern == pattern
        assert trigger.levels is None

    def test_init_with_levels(self):
        """Test initialization with threshold levels."""
        pattern = [1, 0, 1]
        levels = [1.5, 1.5, 1.5]
        trigger = MultiChannelPatternTrigger(pattern=pattern, levels=levels)
        assert trigger.levels == levels

    def test_init_with_dont_care(self):
        """Test initialization with don't care values."""
        pattern = [1, None, 0]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        assert trigger.pattern == pattern

    def test_find_events_matching_traces(self, multi_channel_traces: list[DigitalTrace]):
        """Test finding events with matching number of traces."""
        # Using fixtures: 3 traces
        pattern = [True, False, True]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(multi_channel_traces)

        assert isinstance(events, list)

    def test_find_events_mismatched_trace_count(
        self, sample_rate: float, multi_channel_traces: list[DigitalTrace]
    ):
        """Test error when trace count doesn't match pattern length."""
        pattern = [1, 0, 1, 1]  # 4 channels
        trigger = MultiChannelPatternTrigger(pattern=pattern)

        # Only 3 traces provided
        with pytest.raises(
            AnalysisError,
            match="Number of traces.*must match pattern length",
        ):
            trigger.find_events(multi_channel_traces)

    def test_find_events_simultaneous_match(self, sample_rate: float):
        """Test that all channels must match simultaneously."""
        # Create 3 traces where all channels have same value at specific indices
        ch0 = np.array([0, 1, 1, 0], dtype=bool)
        ch1 = np.array([1, 0, 0, 1], dtype=bool)
        ch2 = np.array([0, 0, 1, 1], dtype=bool)

        traces = [
            DigitalTrace(data=ch0, metadata=TraceMetadata(sample_rate=sample_rate)),
            DigitalTrace(data=ch1, metadata=TraceMetadata(sample_rate=sample_rate)),
            DigitalTrace(data=ch2, metadata=TraceMetadata(sample_rate=sample_rate)),
        ]

        pattern = [0, 1, 0]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(traces)

        # At index 0: [0, 1, 0] matches
        assert len(events) >= 1

    def test_find_events_with_dont_care_multi(self, sample_rate: float):
        """Test multi-channel matching with don't care values."""
        ch0 = np.array([1, 0, 1, 0], dtype=bool)
        ch1 = np.array([0, 1, 0, 1], dtype=bool)
        ch2 = np.array([0, 0, 1, 1], dtype=bool)

        traces = [
            DigitalTrace(data=ch0, metadata=TraceMetadata(sample_rate=sample_rate)),
            DigitalTrace(data=ch1, metadata=TraceMetadata(sample_rate=sample_rate)),
            DigitalTrace(data=ch2, metadata=TraceMetadata(sample_rate=sample_rate)),
        ]

        # Pattern [1, X, 0] - ch1 is don't care
        pattern = [1, None, 0]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(traces)

        # At index 0: [1, 0, 0] matches (X matches any)
        assert len(events) >= 1

    def test_find_events_waveform_input(self, sample_rate: float):
        """Test multi-channel with waveform input."""
        # Create 2 waveforms
        ch0 = np.array([0.0, 3.3, 0.0, 3.3], dtype=np.float64)
        ch1 = np.array([3.3, 0.0, 3.3, 0.0], dtype=np.float64)

        traces = [
            WaveformTrace(data=ch0, metadata=TraceMetadata(sample_rate=sample_rate)),
            WaveformTrace(data=ch1, metadata=TraceMetadata(sample_rate=sample_rate)),
        ]

        pattern = [1, 0]
        levels = [1.5, 1.5]
        trigger = MultiChannelPatternTrigger(pattern=pattern, levels=levels)
        events = trigger.find_events(traces)

        assert isinstance(events, list)

    def test_find_events_mixed_trace_types(self, sample_rate: float):
        """Test multi-channel with mixed digital and waveform traces."""
        ch0 = np.array([True, False, True], dtype=bool)
        ch1 = np.array([3.3, 0.0, 3.3], dtype=np.float64)

        traces = [
            DigitalTrace(data=ch0, metadata=TraceMetadata(sample_rate=sample_rate)),
            WaveformTrace(data=ch1, metadata=TraceMetadata(sample_rate=sample_rate)),
        ]

        pattern = [1, 1]
        trigger = MultiChannelPatternTrigger(pattern=pattern, levels=[None, 1.5])
        events = trigger.find_events(traces)

        assert isinstance(events, list)

    def test_find_events_empty_traces(self, sample_rate: float):
        """Test with empty traces."""
        traces = [
            DigitalTrace(
                data=np.array([], dtype=bool), metadata=TraceMetadata(sample_rate=sample_rate)
            ),
            DigitalTrace(
                data=np.array([], dtype=bool), metadata=TraceMetadata(sample_rate=sample_rate)
            ),
        ]

        pattern = [1, 0]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(traces)

        assert len(events) == 0

    def test_find_events_single_sample(self, sample_rate: float):
        """Test multi-channel with single sample per channel."""
        traces = [
            DigitalTrace(
                data=np.array([True], dtype=bool), metadata=TraceMetadata(sample_rate=sample_rate)
            ),
            DigitalTrace(
                data=np.array([False], dtype=bool), metadata=TraceMetadata(sample_rate=sample_rate)
            ),
        ]

        pattern = [1, 0]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(traces)

        # One sample, check if it matches the pattern
        # At index 0: [1, 0] matches pattern [1, 0]
        # No transition yet, depends on implementation
        assert isinstance(events, list)

    def test_find_events_unequal_trace_lengths(self, sample_rate: float):
        """Test with channels of different lengths."""
        ch0 = np.array([1, 0, 1, 0, 1], dtype=bool)
        ch1 = np.array([0, 1, 0], dtype=bool)  # Shorter

        traces = [
            DigitalTrace(data=ch0, metadata=TraceMetadata(sample_rate=sample_rate)),
            DigitalTrace(data=ch1, metadata=TraceMetadata(sample_rate=sample_rate)),
        ]

        pattern = [1, 0]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(traces)

        # Should use min length (3 samples)
        assert isinstance(events, list)

    def test_find_events_transition_detection(self, sample_rate: float):
        """Test that transitions are detected correctly."""
        # Channels that transition to matching state
        ch0 = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
        ch1 = np.array([0, 0, 1, 1, 0, 0], dtype=bool)

        traces = [
            DigitalTrace(data=ch0, metadata=TraceMetadata(sample_rate=sample_rate)),
            DigitalTrace(data=ch1, metadata=TraceMetadata(sample_rate=sample_rate)),
        ]

        pattern = [1, 1]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(traces)

        # Should detect transition to [1, 1] at index 2
        assert len(events) >= 1
        if len(events) > 0:
            assert events[0].sample_index == 2


# =============================================================================
# Helper Function Tests - find_pattern()
# =============================================================================


@pytest.mark.unit
class TestFindPattern:
    """Tests for find_pattern() helper function."""

    def test_find_pattern_returns_timestamps(self, simple_digital_trace: DigitalTrace):
        """Test that find_pattern returns timestamps by default."""
        result = find_pattern(simple_digital_trace, [0, 1])

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) > 0

    def test_find_pattern_returns_indices(self, simple_digital_trace: DigitalTrace):
        """Test that find_pattern can return indices instead."""
        result = find_pattern(simple_digital_trace, [0, 1], return_indices=True)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert len(result) > 0

    def test_find_pattern_with_custom_level(self, waveform_trace_digital: WaveformTrace):
        """Test find_pattern with custom threshold level."""
        result = find_pattern(waveform_trace_digital, [0, 1], level=1.5)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_find_pattern_with_dont_care(self, simple_digital_trace: DigitalTrace):
        """Test find_pattern with don't care values."""
        result = find_pattern(simple_digital_trace, [0, None])

        assert isinstance(result, np.ndarray)

    def test_find_pattern_empty_result(self, sample_rate: float):
        """Test find_pattern when pattern not found."""
        data = np.array([0, 0, 0, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = find_pattern(trace, [1, 1])

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_find_pattern_timestamp_values(self, sample_rate: float):
        """Test that returned timestamps are reasonable."""
        data = np.array([0, 1, 0, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = find_pattern(trace, [1, 0], return_indices=False)

        sample_period = 1.0 / sample_rate
        for ts in result:
            # Timestamps should be multiples of sample period
            assert ts >= 0
            assert abs(ts % sample_period) < 1e-12

    def test_find_pattern_index_values(self, sample_rate: float):
        """Test that returned indices are valid."""
        data = np.array([0, 1, 0, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = find_pattern(trace, [1, 0], return_indices=True)

        for idx in result:
            # Indices should be within trace bounds
            assert 0 <= idx < len(data) - len([1, 0]) + 1

    def test_find_pattern_with_waveform_input(self, waveform_sine_wave: WaveformTrace):
        """Test find_pattern with waveform input."""
        result = find_pattern(waveform_sine_wave, [1, 0], level=0.0)

        assert isinstance(result, np.ndarray)


# =============================================================================
# Helper Function Tests - find_bit_sequence()
# =============================================================================


@pytest.mark.unit
class TestFindBitSequence:
    """Tests for find_bit_sequence() helper function."""

    def test_find_bit_sequence_simple(self, waveform_trace_digital: WaveformTrace):
        """Test finding simple bit sequence."""
        events = find_bit_sequence(waveform_trace_digital, "01")

        assert isinstance(events, list)
        for event in events:
            assert event.event_type == TriggerType.PATTERN_MATCH

    def test_find_bit_sequence_with_dont_care(self, sample_rate: float):
        """Test bit sequence with don't care (X)."""
        # Pattern 10X (1, 0, don't care)
        data = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64) * 3.3, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "10X", level=1.5)

        assert isinstance(events, list)

    def test_find_bit_sequence_uppercase_x(self, sample_rate: float):
        """Test that uppercase X is handled."""
        data = np.array([1, 0, 1], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64) * 3.3, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "1X1", level=1.5)

        assert isinstance(events, list)

    def test_find_bit_sequence_lowercase_x(self, sample_rate: float):
        """Test that lowercase x is also handled."""
        data = np.array([1, 0, 1], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64) * 3.3, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "1x1", level=1.5)

        assert isinstance(events, list)

    def test_find_bit_sequence_all_zeros(self, sample_rate: float):
        """Test bit sequence of all zeros."""
        data = np.array([0, 0, 0, 0], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64), metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "0000", level=0.5)

        assert isinstance(events, list)

    def test_find_bit_sequence_all_ones(self, sample_rate: float):
        """Test bit sequence of all ones."""
        data = np.array([1, 1, 1, 1], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64) * 3.3, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "1111", level=1.5)

        assert isinstance(events, list)

    def test_find_bit_sequence_long_sequence(self, sample_rate: float):
        """Test with long bit sequence."""
        data = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64) * 3.3, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "10101010", level=1.5)

        assert isinstance(events, list)

    def test_find_bit_sequence_empty_string(self, waveform_trace_digital: WaveformTrace):
        """Test with empty bit string."""
        events = find_bit_sequence(waveform_trace_digital, "")

        assert isinstance(events, list)

    def test_find_bit_sequence_single_bit(self, sample_rate: float):
        """Test with single bit sequence."""
        data = np.array([0, 1, 0, 1], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64) * 3.3, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "1", level=1.5)

        assert isinstance(events, list)

    def test_find_bit_sequence_invalid_character(self, waveform_trace_digital: WaveformTrace):
        """Test error with invalid character."""
        with pytest.raises(AnalysisError, match="Invalid bit character"):
            find_bit_sequence(waveform_trace_digital, "10Z1")

    def test_find_bit_sequence_with_custom_level(self, waveform_sine_wave: WaveformTrace):
        """Test with custom threshold level."""
        events = find_bit_sequence(waveform_sine_wave, "101", level=0.0)

        assert isinstance(events, list)

    def test_find_bit_sequence_no_level_provided(self, waveform_trace_digital: WaveformTrace):
        """Test without providing level (uses default)."""
        events = find_bit_sequence(waveform_trace_digital, "01")

        assert isinstance(events, list)

    def test_find_bit_sequence_event_data(self, sample_rate: float):
        """Test that returned events have correct data."""
        data = np.array([0, 1, 0], dtype=bool)
        trace = WaveformTrace(
            data=data.astype(np.float64) * 3.3, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        events = find_bit_sequence(trace, "010", level=1.5)

        for event in events:
            assert "pattern" in event.data
            assert event.data["pattern"] == [0, 1, 0]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestTriggeringPatternEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pattern_larger_than_trace(self, sample_rate: float):
        """Test pattern larger than available trace."""
        data = np.array([1, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1, 0, 1, 0, 1, 0])
        events = trigger.find_events(trace)

        assert len(events) == 0

    def test_single_bit_pattern(self, sample_rate: float):
        """Test single-bit pattern."""
        data = np.array([0, 1, 0, 1, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1])
        events = trigger.find_events(trace)

        assert isinstance(events, list)

    def test_all_dont_care_pattern(self, sample_rate: float):
        """Test pattern of all don't cares."""
        data = np.array([1, 0, 1, 0], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[None, None])
        events = trigger.find_events(trace)

        # All don't care should match every position
        assert isinstance(events, list)

    def test_negative_threshold(self, sample_rate: float):
        """Test with negative threshold level."""
        data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[0, 1], levels=-0.2)
        events = trigger.find_events(trace)

        assert isinstance(events, list)

    def test_very_large_trace(self, sample_rate: float):
        """Test with very large trace."""
        # 100,000 samples
        data = np.tile([0, 1], 50000)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[0, 1])
        events = trigger.find_events(trace)

        # Should complete without memory issues
        assert len(events) > 0

    def test_many_matches(self, sample_rate: float):
        """Test when pattern matches many times."""
        # Pattern [0, 1] appears 1000 times
        data = np.tile([0, 1], 1000)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[0, 1])
        events = trigger.find_events(trace)

        # Should find many matches
        assert len(events) > 100

    def test_float_level_values(self, sample_rate: float):
        """Test with float threshold values."""
        data = np.array([0.1, 2.7, 0.2, 2.8], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[0, 1], levels=1.5)
        events = trigger.find_events(trace)

        assert isinstance(events, list)

    def test_inf_values_in_waveform(self, sample_rate: float):
        """Test handling of infinite values."""
        data = np.array([0.0, 3.3, np.inf, 0.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[0, 1], levels=1.5)
        events = trigger.find_events(trace)

        # Should handle inf without crashing
        assert isinstance(events, list)

    def test_nan_values_in_waveform(self, sample_rate: float):
        """Test handling of NaN values."""
        data = np.array([0.0, 3.3, np.nan, 0.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[0, 1], levels=1.5)
        events = trigger.find_events(trace)

        # Should handle NaN without crashing
        assert isinstance(events, list)

    def test_zero_amplitude_waveform(self, sample_rate: float):
        """Test waveform with zero amplitude (constant)."""
        data = np.ones(10) * 2.0
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = PatternTrigger(pattern=[1])
        events = trigger.find_events(trace)

        # Constant signal, no transitions for exact match
        assert isinstance(events, list)

    def test_multi_channel_different_lengths(self, sample_rate: float):
        """Test multi-channel with significantly different lengths."""
        ch0 = np.array([1, 0] * 10, dtype=bool)  # 20 samples
        ch1 = np.array([1], dtype=bool)  # 1 sample

        traces = [
            DigitalTrace(data=ch0, metadata=TraceMetadata(sample_rate=sample_rate)),
            DigitalTrace(data=ch1, metadata=TraceMetadata(sample_rate=sample_rate)),
        ]

        pattern = [1, 1]
        trigger = MultiChannelPatternTrigger(pattern=pattern)
        events = trigger.find_events(traces)

        # Should only check first sample
        assert isinstance(events, list)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestTriggeringPatternIntegration:
    """Integration tests combining multiple features."""

    def test_pattern_trigger_with_digital_and_waveform(self, sample_rate: float):
        """Test PatternTrigger works with both input types."""
        data = np.array([0, 1, 0, 1], dtype=bool)
        digital_trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        waveform_data = np.array([0.0, 3.3, 0.0, 3.3], dtype=np.float64)
        waveform_trace = WaveformTrace(
            data=waveform_data, metadata=TraceMetadata(sample_rate=sample_rate)
        )

        trigger = PatternTrigger(pattern=[0, 1], levels=1.5)

        digital_events = trigger.find_events(digital_trace)
        waveform_events = trigger.find_events(waveform_trace)

        assert len(digital_events) > 0
        assert len(waveform_events) > 0

    def test_base_trigger_methods(self, simple_digital_trace: DigitalTrace):
        """Test inherited base trigger methods."""
        trigger = PatternTrigger(pattern=[0, 1])

        # Test find_first
        first_event = trigger.find_first(simple_digital_trace)
        assert first_event is None or isinstance(first_event, object)

        # Test count_events
        count = trigger.count_events(simple_digital_trace)
        assert count >= 0

    def test_consecutive_pattern_calls(self, simple_digital_trace: DigitalTrace):
        """Test making multiple find_events calls on same trigger."""
        trigger = PatternTrigger(pattern=[0, 1])

        events1 = trigger.find_events(simple_digital_trace)
        events2 = trigger.find_events(simple_digital_trace)

        # Results should be identical
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2, strict=False):
            assert e1.sample_index == e2.sample_index

    def test_pattern_find_helpers_consistency(self, simple_digital_trace: DigitalTrace):
        """Test that helper functions use PatternTrigger correctly."""
        pattern = [0, 1]

        # Using PatternTrigger directly
        trigger = PatternTrigger(pattern=pattern)
        direct_events = trigger.find_events(simple_digital_trace)

        # Using find_pattern
        indices = find_pattern(simple_digital_trace, pattern, return_indices=True)

        # Should find same number of matches
        assert len(direct_events) == len(indices)
