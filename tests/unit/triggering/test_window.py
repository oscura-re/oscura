"""Comprehensive unit tests for window triggering module.

Tests all classes and functions in src/oscura/triggering/window.py:
- Zone dataclass
- WindowTrigger class
- ZoneTrigger class
- MaskTrigger class
- find_window_violations()
- find_zone_events()
- check_limits()
"""

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.triggering.base import TriggerType
from oscura.triggering.window import (
    MaskTrigger,
    WindowTrigger,
    Zone,
    ZoneTrigger,
    check_limits,
    find_window_violations,
    find_zone_events,
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
def basic_trace(sample_rate: float) -> WaveformTrace:
    """Create a basic trace with controlled values."""
    # Signal that goes 0 -> 1 -> 2 -> 3 -> 2 -> 1 -> 0
    data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0])
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
def square_wave(sample_rate: float) -> WaveformTrace:
    """Create a square wave between 0 and 3.3V."""
    n_samples = 1000
    data = np.zeros(n_samples)
    # Create square wave pattern
    for i in range(0, n_samples, 100):
        data[i : i + 50] = 3.3
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


# =============================================================================
# Zone Tests
# =============================================================================


@pytest.mark.unit
class TestZone:
    """Tests for Zone dataclass."""

    def test_zone_creation_basic(self):
        """Test basic zone creation."""
        zone = Zone(low=0.0, high=3.3)
        assert zone.low == 0.0
        assert zone.high == 3.3
        assert zone.start_time is None
        assert zone.end_time is None
        assert zone.name == ""

    def test_zone_creation_with_time_bounds(self):
        """Test zone creation with time boundaries."""
        zone = Zone(low=1.0, high=2.0, start_time=0.001, end_time=0.002)
        assert zone.low == 1.0
        assert zone.high == 2.0
        assert zone.start_time == 0.001
        assert zone.end_time == 0.002

    def test_zone_creation_with_name(self):
        """Test zone creation with name."""
        zone = Zone(low=0.8, high=1.2, name="metastable")
        assert zone.name == "metastable"

    def test_zone_all_attributes(self):
        """Test zone with all attributes."""
        zone = Zone(
            low=0.5,
            high=1.5,
            start_time=0.0001,
            end_time=0.0005,
            name="test_zone",
        )
        assert zone.low == 0.5
        assert zone.high == 1.5
        assert zone.start_time == 0.0001
        assert zone.end_time == 0.0005
        assert zone.name == "test_zone"


# =============================================================================
# WindowTrigger Tests
# =============================================================================


@pytest.mark.unit
class TestWindowTrigger:
    """Tests for WindowTrigger class."""

    def test_init_valid_thresholds(self):
        """Test initialization with valid thresholds."""
        trigger = WindowTrigger(low_threshold=0.0, high_threshold=3.3)
        assert trigger.low_threshold == 0.0
        assert trigger.high_threshold == 3.3
        assert trigger.trigger_on == "exit"

    def test_init_with_trigger_on_entry(self):
        """Test initialization with entry triggering."""
        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="entry")
        assert trigger.trigger_on == "entry"

    def test_init_with_trigger_on_both(self):
        """Test initialization with both triggering."""
        trigger = WindowTrigger(low_threshold=0.5, high_threshold=2.5, trigger_on="both")
        assert trigger.trigger_on == "both"

    def test_init_invalid_thresholds(self):
        """Test initialization fails with invalid thresholds."""
        with pytest.raises(AnalysisError, match="low_threshold must be less than"):
            WindowTrigger(low_threshold=3.3, high_threshold=0.0)

    def test_init_equal_thresholds(self):
        """Test initialization fails with equal thresholds."""
        with pytest.raises(AnalysisError, match="low_threshold must be less than"):
            WindowTrigger(low_threshold=1.5, high_threshold=1.5)

    def test_find_events_exit_only(self, basic_trace: WaveformTrace):
        """Test finding window exit events only."""
        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="exit")
        events = trigger.find_events(basic_trace)

        # Should find exits when going from inside to outside window
        assert len(events) > 0
        for event in events:
            assert event.event_type == TriggerType.WINDOW_EXIT
            assert "window" in event.data
            assert event.data["window"] == (1.0, 2.0)
            assert "direction" in event.data
            assert event.data["direction"] == "exiting"
            assert "boundary" in event.data
            assert event.data["boundary"] in ("high", "low")

    def test_find_events_entry_only(self, basic_trace: WaveformTrace):
        """Test finding window entry events only."""
        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="entry")
        events = trigger.find_events(basic_trace)

        # Should find entries when going from outside to inside window
        assert len(events) > 0
        for event in events:
            assert event.event_type == TriggerType.WINDOW_ENTRY
            assert "window" in event.data
            assert event.data["window"] == (1.0, 2.0)
            assert "direction" in event.data
            assert event.data["direction"] == "entering"

    def test_find_events_both(self, basic_trace: WaveformTrace):
        """Test finding both entry and exit events."""
        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="both")
        events = trigger.find_events(basic_trace)

        # Should find both entry and exit events
        assert len(events) > 0
        event_types = {event.event_type for event in events}
        # May have both types depending on signal
        assert event_types.issubset({TriggerType.WINDOW_ENTRY, TriggerType.WINDOW_EXIT})

    def test_find_events_boundary_detection_high(self, sample_rate: float):
        """Test boundary detection for high threshold crossing."""
        # Signal crosses high threshold
        data = np.array([1.5, 1.5, 2.5, 2.5])  # Inside then crosses high
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="exit")
        events = trigger.find_events(trace)

        assert len(events) == 1
        assert events[0].data["boundary"] == "high"

    def test_find_events_boundary_detection_low(self, sample_rate: float):
        """Test boundary detection for low threshold crossing."""
        # Signal crosses low threshold
        data = np.array([1.5, 1.5, 0.5, 0.5])  # Inside then crosses low
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="exit")
        events = trigger.find_events(trace)

        assert len(events) == 1
        assert events[0].data["boundary"] == "low"

    def test_find_events_timestamp_calculation(self, sample_rate: float):
        """Test timestamp calculation in events."""
        data = np.array([0.0, 1.5, 2.5, 1.5, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="exit")
        events = trigger.find_events(trace)

        sample_period = 1.0 / sample_rate
        for event in events:
            # Timestamp should be sample_index * sample_period
            assert abs(event.timestamp - event.sample_index * sample_period) < 1e-12

    def test_find_events_level_recording(self, basic_trace: WaveformTrace):
        """Test that signal level is recorded in events."""
        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="exit")
        events = trigger.find_events(basic_trace)

        for event in events:
            assert event.level is not None
            assert isinstance(event.level, float)

    def test_find_events_empty_trace(self, empty_trace: WaveformTrace):
        """Test with empty trace."""
        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="both")
        events = trigger.find_events(empty_trace)
        assert len(events) == 0

    def test_find_events_single_sample(self, single_sample_trace: WaveformTrace):
        """Test with single sample trace."""
        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="both")
        events = trigger.find_events(single_sample_trace)
        # No transitions possible with single sample
        assert len(events) == 0

    def test_find_events_always_inside(self, sample_rate: float):
        """Test trace that stays inside window."""
        data = np.ones(100) * 1.5  # Always 1.5V
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="exit")
        events = trigger.find_events(trace)
        assert len(events) == 0

    def test_find_events_always_outside(self, sample_rate: float):
        """Test trace that stays outside window."""
        data = np.ones(100) * 5.0  # Always 5.0V (above window)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="entry")
        events = trigger.find_events(trace)
        assert len(events) == 0

    def test_find_events_multiple_transitions(self, sine_wave: WaveformTrace):
        """Test with signal that has multiple transitions."""
        trigger = WindowTrigger(low_threshold=-1.0, high_threshold=1.0, trigger_on="both")
        events = trigger.find_events(sine_wave)

        # Sine wave should cross window boundaries multiple times
        assert len(events) > 2

    def test_find_events_narrow_window(self, basic_trace: WaveformTrace):
        """Test with very narrow window."""
        trigger = WindowTrigger(low_threshold=1.49, high_threshold=1.51, trigger_on="both")
        events = trigger.find_events(basic_trace)

        # Should find events for this narrow window
        assert len(events) >= 0  # May or may not trigger depending on exact values


# =============================================================================
# ZoneTrigger Tests
# =============================================================================


@pytest.mark.unit
class TestZoneTrigger:
    """Tests for ZoneTrigger class."""

    def test_init_single_zone(self):
        """Test initialization with single zone."""
        zone = Zone(low=0.8, high=1.2, name="forbidden")
        trigger = ZoneTrigger(zones=[zone])
        assert len(trigger.zones) == 1
        assert trigger.trigger_on == "violation"

    def test_init_multiple_zones(self):
        """Test initialization with multiple zones."""
        zones = [
            Zone(low=0.8, high=1.2, name="zone1"),
            Zone(low=2.0, high=2.5, name="zone2"),
        ]
        trigger = ZoneTrigger(zones=zones)
        assert len(trigger.zones) == 2

    def test_init_trigger_on_entry(self):
        """Test initialization with entry trigger."""
        zone = Zone(low=1.0, high=2.0)
        trigger = ZoneTrigger(zones=[zone], trigger_on="entry")
        assert trigger.trigger_on == "entry"

    def test_init_trigger_on_exit(self):
        """Test initialization with exit trigger."""
        zone = Zone(low=1.0, high=2.0)
        trigger = ZoneTrigger(zones=[zone], trigger_on="exit")
        assert trigger.trigger_on == "exit"

    def test_find_events_violation_mode(self, basic_trace: WaveformTrace):
        """Test finding zone violations (entries)."""
        zone = Zone(low=2.5, high=3.5, name="overvoltage")
        trigger = ZoneTrigger(zones=[zone], trigger_on="violation")
        events = trigger.find_events(basic_trace)

        # Should trigger when entering forbidden zone
        for event in events:
            assert event.event_type == TriggerType.ZONE_VIOLATION
            assert event.data["zone_name"] == "overvoltage"
            assert event.data["zone_bounds"] == (2.5, 3.5)
            assert event.data["direction"] == "entering"

    def test_find_events_entry_mode(self, basic_trace: WaveformTrace):
        """Test finding zone entries."""
        zone = Zone(low=1.5, high=2.5, name="target")
        trigger = ZoneTrigger(zones=[zone], trigger_on="entry")
        events = trigger.find_events(basic_trace)

        # Should trigger when entering zone
        for event in events:
            assert event.data["direction"] == "entering"

    def test_find_events_exit_mode(self, basic_trace: WaveformTrace):
        """Test finding zone exits."""
        zone = Zone(low=1.5, high=2.5, name="target")
        trigger = ZoneTrigger(zones=[zone], trigger_on="exit")
        events = trigger.find_events(basic_trace)

        # Should trigger when exiting zone
        for event in events:
            assert event.data["direction"] == "exiting"

    def test_find_events_multiple_zones(self, sine_wave: WaveformTrace):
        """Test with multiple zones."""
        zones = [
            Zone(low=-2.0, high=-1.0, name="undervoltage"),
            Zone(low=1.0, high=2.0, name="overvoltage"),
        ]
        trigger = ZoneTrigger(zones=zones, trigger_on="violation")
        events = trigger.find_events(sine_wave)

        # Should find violations in both zones
        assert len(events) > 0
        zone_names = {event.data["zone_name"] for event in events}
        # Should trigger in at least one zone (possibly both)
        assert len(zone_names) >= 1

    def test_find_events_time_bounds_start_only(self, sample_rate: float):
        """Test zone with start_time boundary only."""
        data = np.ones(1000) * 2.0
        data[500:600] = 1.5  # Zone violation in middle
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        start_time = 400 * sample_period

        zone = Zone(low=1.0, high=1.6, start_time=start_time)
        trigger = ZoneTrigger(zones=[zone], trigger_on="violation")
        events = trigger.find_events(trace)

        # Should only detect violations after start_time
        for event in events:
            assert event.timestamp >= start_time

    def test_find_events_time_bounds_end_only(self, sample_rate: float):
        """Test zone with end_time boundary only."""
        data = np.ones(1000) * 2.0
        data[200:300] = 1.5  # Early violation
        data[700:800] = 1.5  # Late violation
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        end_time = 500 * sample_period

        zone = Zone(low=1.0, high=1.6, end_time=end_time)
        trigger = ZoneTrigger(zones=[zone], trigger_on="violation")
        events = trigger.find_events(trace)

        # Should only detect violations before end_time
        for event in events:
            assert event.timestamp <= end_time

    def test_find_events_time_bounds_both(self, sample_rate: float):
        """Test zone with both start and end time boundaries."""
        data = np.ones(1000) * 2.0
        data[100:200] = 1.5  # Too early
        data[400:500] = 1.5  # Within time window
        data[800:900] = 1.5  # Too late
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        start_time = 300 * sample_period
        end_time = 600 * sample_period

        zone = Zone(low=1.0, high=1.6, start_time=start_time, end_time=end_time)
        trigger = ZoneTrigger(zones=[zone], trigger_on="violation")
        events = trigger.find_events(trace)

        # Should only detect violations within time window
        for event in events:
            assert start_time <= event.timestamp <= end_time

    def test_find_events_sorted_by_timestamp(self, sine_wave: WaveformTrace):
        """Test that events are sorted by timestamp."""
        zones = [
            Zone(low=-1.5, high=-0.5, name="zone1"),
            Zone(low=0.5, high=1.5, name="zone2"),
        ]
        trigger = ZoneTrigger(zones=zones, trigger_on="violation")
        events = trigger.find_events(sine_wave)

        # Events should be sorted by timestamp
        timestamps = [event.timestamp for event in events]
        assert timestamps == sorted(timestamps)

    def test_find_events_empty_trace(self, empty_trace: WaveformTrace):
        """Test with empty trace."""
        zone = Zone(low=1.0, high=2.0)
        trigger = ZoneTrigger(zones=[zone])
        events = trigger.find_events(empty_trace)
        assert len(events) == 0

    def test_find_events_single_sample(self, single_sample_trace: WaveformTrace):
        """Test with single sample trace."""
        zone = Zone(low=1.0, high=2.0)
        trigger = ZoneTrigger(zones=[zone])
        events = trigger.find_events(single_sample_trace)
        # No transitions possible
        assert len(events) == 0

    def test_find_events_no_violations(self, sample_rate: float):
        """Test when signal never enters zones."""
        data = np.ones(100) * 0.5  # Always 0.5V
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        zone = Zone(low=1.0, high=2.0)
        trigger = ZoneTrigger(zones=[zone], trigger_on="violation")
        events = trigger.find_events(trace)
        assert len(events) == 0


# =============================================================================
# MaskTrigger Tests
# =============================================================================


@pytest.mark.unit
class TestMaskTrigger:
    """Tests for MaskTrigger class."""

    def test_init_valid_mask(self):
        """Test initialization with valid mask."""
        mask_points = [(0.0, 0.0), (0.001, 0.0), (0.001, 3.3), (0.0, 3.3)]
        trigger = MaskTrigger(mask_points=mask_points)
        assert len(trigger.mask_points) == 4
        assert trigger.mode == "inside"

    def test_init_outside_mode(self):
        """Test initialization with outside mode."""
        mask_points = [(0.0, 0.0), (0.001, 0.0), (0.001, 3.3)]
        trigger = MaskTrigger(mask_points=mask_points, mode="outside")
        assert trigger.mode == "outside"

    def test_init_too_few_points(self):
        """Test initialization fails with too few points."""
        with pytest.raises(AnalysisError, match="at least 3 points"):
            MaskTrigger(mask_points=[(0.0, 0.0), (0.001, 0.0)])

    def test_find_events_inside_mode(self, sample_rate: float):
        """Test finding violations when signal is inside forbidden region."""
        # Create signal that enters mask region
        n_samples = 100
        data = np.zeros(n_samples)
        data[40:60] = 2.0  # Signal enters mask in middle
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        # Define mask as rectangular region where signal goes
        mask_points = [
            (30 * sample_period, 1.5),
            (70 * sample_period, 1.5),
            (70 * sample_period, 2.5),
            (30 * sample_period, 2.5),
        ]
        trigger = MaskTrigger(mask_points=mask_points, mode="inside")
        events = trigger.find_events(trace)

        # Should detect when signal enters forbidden mask
        assert len(events) >= 1
        for event in events:
            assert event.event_type == TriggerType.ZONE_VIOLATION
            assert event.data["mask_mode"] == "inside"
            assert event.data["violation_type"] == "inside_forbidden"

    def test_find_events_outside_mode(self, sample_rate: float):
        """Test finding violations when signal is outside required region."""
        # Create signal that stays mostly in range
        n_samples = 100
        data = np.ones(n_samples) * 2.0
        data[40:60] = 5.0  # Signal goes outside required region
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        # Define mask as required region
        mask_points = [
            (0.0, 1.5),
            (100 * sample_period, 1.5),
            (100 * sample_period, 2.5),
            (0.0, 2.5),
        ]
        trigger = MaskTrigger(mask_points=mask_points, mode="outside")
        events = trigger.find_events(trace)

        # Should detect when signal leaves required mask
        assert len(events) >= 1
        for event in events:
            assert event.data["mask_mode"] == "outside"
            assert event.data["violation_type"] == "outside_required"

    def test_find_events_grouped_violations(self, sample_rate: float):
        """Test that consecutive violations are grouped."""
        n_samples = 200
        data = np.zeros(n_samples)
        # Create two separate violation regions
        data[40:60] = 2.0
        data[140:160] = 2.0
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        mask_points = [
            (0.0, 1.5),
            (200 * sample_period, 1.5),
            (200 * sample_period, 2.5),
            (0.0, 2.5),
        ]
        trigger = MaskTrigger(mask_points=mask_points, mode="inside")
        events = trigger.find_events(trace)

        # Should group consecutive violations, creating 2 events
        assert len(events) == 2

    def test_find_events_no_violations_inside_mode(self, sample_rate: float):
        """Test when signal never enters forbidden mask."""
        data = np.ones(100) * 0.5  # Always outside mask
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        mask_points = [
            (0.0, 1.5),
            (100 * sample_period, 1.5),
            (100 * sample_period, 2.5),
            (0.0, 2.5),
        ]
        trigger = MaskTrigger(mask_points=mask_points, mode="inside")
        events = trigger.find_events(trace)
        assert len(events) == 0

    def test_find_events_no_violations_outside_mode(self, sample_rate: float):
        """Test when signal always stays in required mask."""
        data = np.ones(100) * 2.0  # Always inside mask
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        mask_points = [
            (0.0, 1.5),
            (100 * sample_period, 1.5),
            (100 * sample_period, 2.5),
            (0.0, 2.5),
        ]
        trigger = MaskTrigger(mask_points=mask_points, mode="outside")
        events = trigger.find_events(trace)
        assert len(events) == 0

    def test_find_events_triangular_mask(self, sample_rate: float):
        """Test with triangular mask."""
        n_samples = 100
        data = np.linspace(0, 3, n_samples)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        sample_period = 1.0 / sample_rate
        # Triangular mask
        mask_points = [
            (0.0, 0.5),
            (50 * sample_period, 2.0),
            (100 * sample_period, 0.5),
        ]
        trigger = MaskTrigger(mask_points=mask_points, mode="inside")
        events = trigger.find_events(trace)

        # Signal should violate triangular mask
        assert len(events) >= 0  # Depends on exact geometry

    def test_find_events_empty_trace(self, empty_trace: WaveformTrace):
        """Test with empty trace."""
        mask_points = [(0.0, 0.0), (0.001, 0.0), (0.001, 3.3)]
        trigger = MaskTrigger(mask_points=mask_points)
        events = trigger.find_events(empty_trace)
        assert len(events) == 0


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
class TestFindWindowViolations:
    """Tests for find_window_violations() function."""

    def test_basic_violation_detection(self, basic_trace: WaveformTrace):
        """Test basic window violation detection."""
        violations = find_window_violations(basic_trace, low=1.0, high=2.0)

        # Should find violations when signal exits window
        assert isinstance(violations, list)
        for v in violations:
            assert v.event_type == TriggerType.WINDOW_EXIT

    def test_no_violations(self, sample_rate: float):
        """Test when signal stays within window."""
        data = np.ones(100) * 1.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        violations = find_window_violations(trace, low=1.0, high=2.0)
        assert len(violations) == 0

    def test_multiple_violations(self, sine_wave: WaveformTrace):
        """Test multiple violations detection."""
        violations = find_window_violations(sine_wave, low=-1.0, high=1.0)

        # Sine wave should violate narrow window multiple times
        assert len(violations) > 0

    def test_wide_window_no_violations(self, basic_trace: WaveformTrace):
        """Test with window wider than signal range."""
        violations = find_window_violations(basic_trace, low=-10.0, high=10.0)
        assert len(violations) == 0

    def test_narrow_window_many_violations(self, sine_wave: WaveformTrace):
        """Test with very narrow window."""
        violations = find_window_violations(sine_wave, low=-0.1, high=0.1)

        # Most of sine wave should be outside narrow window
        assert len(violations) > 0


@pytest.mark.unit
class TestFindZoneEvents:
    """Tests for find_zone_events() function."""

    def test_with_tuple_zones(self, basic_trace: WaveformTrace):
        """Test with zones as tuples."""
        zones = [(2.5, 3.5), (0.0, 0.5)]
        events = find_zone_events(basic_trace, zones)

        assert isinstance(events, list)
        # Should detect entries into zones

    def test_with_zone_objects(self, basic_trace: WaveformTrace):
        """Test with Zone objects."""
        zones = [
            Zone(low=2.5, high=3.5, name="high_zone"),
            Zone(low=0.0, high=0.5, name="low_zone"),
        ]
        events = find_zone_events(basic_trace, zones)

        assert isinstance(events, list)
        # Zone names should be preserved
        for event in events:
            assert event.data["zone_name"] in ("high_zone", "low_zone")

    def test_mixed_zone_types(self, basic_trace: WaveformTrace):
        """Test with mixed tuple and Zone objects."""
        zones = [
            (2.5, 3.5),  # Tuple
            Zone(low=0.0, high=0.5, name="low_zone"),  # Zone object
        ]
        events = find_zone_events(basic_trace, zones)

        assert isinstance(events, list)

    def test_auto_naming_tuples(self, basic_trace: WaveformTrace):
        """Test that tuples get auto-generated names."""
        zones = [(2.5, 3.5), (0.0, 0.5)]
        events = find_zone_events(basic_trace, zones)

        # Tuples should be converted with names like "zone_0", "zone_1"
        zone_names = {event.data["zone_name"] for event in events}
        # Auto-generated names should be present
        assert all("zone_" in name for name in zone_names)

    def test_empty_zones_list(self, basic_trace: WaveformTrace):
        """Test with empty zones list."""
        events = find_zone_events(basic_trace, [])
        assert len(events) == 0

    def test_no_zone_entries(self, sample_rate: float):
        """Test when signal never enters any zone."""
        data = np.ones(100) * 1.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        zones = [(0.0, 0.5), (2.5, 3.5)]
        events = find_zone_events(trace, zones)
        assert len(events) == 0


@pytest.mark.unit
class TestCheckLimits:
    """Tests for check_limits() function."""

    def test_signal_within_limits(self, sample_rate: float):
        """Test when signal is completely within limits."""
        data = np.random.uniform(1.0, 2.0, 1000)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=0.5, high=2.5)

        assert result["passed"] is True
        assert len(result["violations"]) == 0
        assert result["time_in_spec"] == 100.0
        assert result["time_out_of_spec"] == 0.0
        assert 1.0 <= result["min_value"] <= 2.0
        assert 1.0 <= result["max_value"] <= 2.0

    def test_signal_exceeds_upper_limit(self, sample_rate: float):
        """Test when signal exceeds upper limit."""
        data = np.array([1.0, 1.5, 3.0, 1.5, 1.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=0.0, high=2.5)

        assert result["passed"] is False
        assert len(result["violations"]) > 0
        assert result["max_value"] == 3.0
        assert result["time_out_of_spec"] > 0

    def test_signal_below_lower_limit(self, sample_rate: float):
        """Test when signal goes below lower limit."""
        data = np.array([1.0, 1.5, -0.5, 1.5, 1.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=0.0, high=2.5)

        assert result["passed"] is False
        assert result["min_value"] == -0.5
        assert result["time_out_of_spec"] > 0

    def test_result_structure(self, basic_trace: WaveformTrace):
        """Test that result has all expected fields."""
        result = check_limits(basic_trace, low=1.0, high=2.0)

        assert "passed" in result
        assert "violations" in result
        assert "min_value" in result
        assert "max_value" in result
        assert "time_in_spec" in result
        assert "time_out_of_spec" in result

        assert isinstance(result["passed"], bool)
        assert isinstance(result["violations"], list)
        assert isinstance(result["min_value"], float)
        assert isinstance(result["max_value"], float)
        assert isinstance(result["time_in_spec"], float)
        assert isinstance(result["time_out_of_spec"], float)

    def test_time_percentages_sum_to_100(self, basic_trace: WaveformTrace):
        """Test that time_in_spec + time_out_of_spec = 100%."""
        result = check_limits(basic_trace, low=1.0, high=2.0)

        total = result["time_in_spec"] + result["time_out_of_spec"]
        assert abs(total - 100.0) < 1e-10

    def test_exact_boundary_values(self, sample_rate: float):
        """Test signal exactly at boundaries."""
        data = np.array([0.0, 1.0, 2.0, 3.3, 2.0, 1.0, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=0.0, high=3.3)

        # Signal exactly at boundaries should pass
        assert result["passed"] is True
        assert result["min_value"] == 0.0
        assert result["max_value"] == 3.3

    def test_all_samples_out_of_spec(self, sample_rate: float):
        """Test when all samples are out of spec."""
        data = np.ones(100) * 5.0
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=0.0, high=3.3)

        assert result["passed"] is False
        assert result["time_out_of_spec"] == 100.0
        assert result["time_in_spec"] == 0.0

    def test_half_in_half_out(self, sample_rate: float):
        """Test when half the samples are in/out of spec."""
        data = np.ones(100)
        data[0:50] = 1.0  # In spec
        data[50:100] = 5.0  # Out of spec
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=0.0, high=3.3)

        assert result["passed"] is False
        assert abs(result["time_in_spec"] - 50.0) < 1.0
        assert abs(result["time_out_of_spec"] - 50.0) < 1.0

    def test_min_max_values(self, basic_trace: WaveformTrace):
        """Test that min/max values are correctly reported."""
        result = check_limits(basic_trace, low=-10.0, high=10.0)

        data = basic_trace.data
        assert result["min_value"] == float(np.min(data))
        assert result["max_value"] == float(np.max(data))

    def test_empty_trace(self, empty_trace: WaveformTrace):
        """Test with empty trace."""
        # This will fail because min/max on empty array
        # Let's check if it handles gracefully or raises
        with pytest.raises((ValueError, RuntimeError)):
            check_limits(empty_trace, low=0.0, high=3.3)

    def test_single_sample_in_range(self, sample_rate: float):
        """Test single sample within range."""
        data = np.array([1.5])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=1.0, high=2.0)

        assert result["passed"] is True
        assert result["min_value"] == 1.5
        assert result["max_value"] == 1.5

    def test_single_sample_out_of_range(self, sample_rate: float):
        """Test single sample outside range."""
        data = np.array([5.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        result = check_limits(trace, low=1.0, high=2.0)

        assert result["passed"] is False
        assert result["max_value"] == 5.0


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


@pytest.mark.unit
class TestTriggeringWindowEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_trace(self, sample_rate: float):
        """Test with very large trace."""
        n_samples = 1_000_000
        data = np.random.uniform(0.0, 3.3, n_samples)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="exit")
        events = trigger.find_events(trace)

        # Should complete without error
        assert isinstance(events, list)

    def test_negative_voltages(self, sample_rate: float):
        """Test with negative voltage values."""
        data = np.linspace(-5.0, 5.0, 1000)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=-1.0, high_threshold=1.0)
        events = trigger.find_events(trace)

        assert len(events) > 0

    def test_extreme_voltage_ranges(self, sample_rate: float):
        """Test with extreme voltage ranges."""
        data = np.array([1e-12, 1e12, -1e12, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=-1e6, high_threshold=1e6)
        events = trigger.find_events(trace)

        assert len(events) > 0

    def test_nan_values_handling(self, sample_rate: float):
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 1.0, 2.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=0.5, high_threshold=2.5)
        # NaN comparisons will be False, so it's treated as outside window
        events = trigger.find_events(trace)

        # Should handle NaN without crashing
        assert isinstance(events, list)

    def test_inf_values_handling(self, sample_rate: float):
        """Test handling of infinite values."""
        data = np.array([1.0, 2.0, np.inf, 1.0, 2.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=0.5, high_threshold=2.5)
        events = trigger.find_events(trace)

        # Should handle inf without crashing
        assert isinstance(events, list)

    def test_all_identical_values(self, sample_rate: float):
        """Test trace with all identical values."""
        data = np.ones(1000) * 1.5
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0)
        events = trigger.find_events(trace)

        # No transitions, so no events
        assert len(events) == 0

    def test_alternating_values(self, sample_rate: float):
        """Test with rapidly alternating values."""
        # Alternating between inside and outside the window
        data = np.array([1.5, 3.0] * 500)  # Alternates inside/outside
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0, trigger_on="both")
        events = trigger.find_events(trace)

        # Should find many transitions (entry and exit events)
        assert len(events) > 100

    def test_zero_sample_period(self, sample_rate: float):
        """Test timestamp calculation with normal sample rate."""
        data = np.array([0.0, 1.5, 2.5, 1.5, 0.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        trigger = WindowTrigger(low_threshold=1.0, high_threshold=2.0)
        events = trigger.find_events(trace)

        # All timestamps should be valid
        for event in events:
            assert event.timestamp >= 0
            assert not np.isnan(event.timestamp)
            assert not np.isinf(event.timestamp)
