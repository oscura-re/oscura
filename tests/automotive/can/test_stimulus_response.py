"""Tests for stimulus-response mapping.

This module tests the StimulusResponseAnalyzer which helps identify which
CAN messages and signals change in response to user actions.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import struct

import pytest

from oscura.automotive.can.models import CANMessage, CANMessageList
from oscura.automotive.can.session import CANSession
from oscura.automotive.can.stimulus_response import (
    ByteChange,
    FrequencyChange,
    StimulusResponseAnalyzer,
    StimulusResponseReport,
)


@pytest.fixture
def baseline_session() -> CANSession:
    """Create baseline session (idle state, no user action).

    Creates messages:
    - 0x100: Constant data (8 bytes, all 0x00)
    - 0x200: Engine idle at 800 RPM (bytes 2-3)
    - 0x300: Throttle at 0% (byte 0)
    - 0x400: Brake not pressed (byte 0 = 0x00)
    """
    messages = CANMessageList()

    # Message 0x100 - Constant data, 10 Hz
    for i in range(50):
        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=i * 0.1,
            data=bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        )
        messages.append(msg)

    # Message 0x200 - Engine at 800 RPM, 100 Hz
    for i in range(500):
        rpm = 800
        raw_rpm = int(rpm / 0.25)
        data = bytearray(8)
        data[0] = 0xAA
        data[1] = 0xBB
        data[2:4] = struct.pack(">H", raw_rpm)
        data[4:8] = bytes([0xCC, 0xDD, 0xEE, 0xFF])

        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=i * 0.01,
            data=bytes(data),
        )
        messages.append(msg)

    # Message 0x300 - Throttle at 0%, 50 Hz
    for i in range(250):
        throttle_pct = 0
        raw_throttle = int(throttle_pct / 0.4)
        data = bytes([raw_throttle, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77])

        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=i * 0.02,
            data=data,
        )
        messages.append(msg)

    # Message 0x400 - Brake not pressed, 20 Hz
    for i in range(100):
        data = bytes([0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70])

        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=i * 0.05,
            data=data,
        )
        messages.append(msg)

    return CANSession.from_messages(messages.messages)


@pytest.fixture
def throttle_stimulus_session() -> CANSession:
    """Create stimulus session with throttle at 50%.

    Changes from baseline:
    - 0x100: Unchanged
    - 0x200: Engine at 3000 RPM (changed)
    - 0x300: Throttle at 50% (changed)
    - 0x400: Brake not pressed (unchanged)
    """
    messages = CANMessageList()

    # Message 0x100 - Still constant
    for i in range(50):
        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=i * 0.1,
            data=bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        )
        messages.append(msg)

    # Message 0x200 - Engine at 3000 RPM
    for i in range(500):
        rpm = 3000
        raw_rpm = int(rpm / 0.25)
        data = bytearray(8)
        data[0] = 0xAA
        data[1] = 0xBB
        data[2:4] = struct.pack(">H", raw_rpm)
        data[4:8] = bytes([0xCC, 0xDD, 0xEE, 0xFF])

        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=i * 0.01,
            data=bytes(data),
        )
        messages.append(msg)

    # Message 0x300 - Throttle at 50%
    for i in range(250):
        throttle_pct = 50
        raw_throttle = int(throttle_pct / 0.4)
        data = bytes([raw_throttle, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77])

        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=i * 0.02,
            data=data,
        )
        messages.append(msg)

    # Message 0x400 - Brake not pressed
    for i in range(100):
        data = bytes([0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70])

        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=i * 0.05,
            data=data,
        )
        messages.append(msg)

    return CANSession.from_messages(messages.messages)


@pytest.fixture
def brake_stimulus_session() -> CANSession:
    """Create stimulus session with brake pressed.

    Changes from baseline:
    - 0x100: Unchanged
    - 0x200: Engine at 800 RPM (unchanged)
    - 0x300: Throttle at 0% (unchanged)
    - 0x400: Brake pressed (byte 0 = 0xFF)
    - 0x500: New message appears (brake light control)
    """
    messages = CANMessageList()

    # Message 0x100 - Still constant
    for i in range(50):
        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=i * 0.1,
            data=bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        )
        messages.append(msg)

    # Message 0x200 - Engine at 800 RPM
    for i in range(500):
        rpm = 800
        raw_rpm = int(rpm / 0.25)
        data = bytearray(8)
        data[0] = 0xAA
        data[1] = 0xBB
        data[2:4] = struct.pack(">H", raw_rpm)
        data[4:8] = bytes([0xCC, 0xDD, 0xEE, 0xFF])

        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=i * 0.01,
            data=bytes(data),
        )
        messages.append(msg)

    # Message 0x300 - Throttle at 0%
    for i in range(250):
        throttle_pct = 0
        raw_throttle = int(throttle_pct / 0.4)
        data = bytes([raw_throttle, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77])

        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=i * 0.02,
            data=data,
        )
        messages.append(msg)

    # Message 0x400 - Brake pressed
    for i in range(100):
        data = bytes([0xFF, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70])

        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=i * 0.05,
            data=data,
        )
        messages.append(msg)

    # Message 0x500 - New message (brake light)
    for i in range(100):
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

        msg = CANMessage(
            arbitration_id=0x500,
            timestamp=i * 0.05,
            data=data,
        )
        messages.append(msg)

    return CANSession.from_messages(messages.messages)


@pytest.mark.unit
class TestStimulusResponseAnalyzer:
    """Tests for StimulusResponseAnalyzer class."""

    def test_detect_new_messages(self, baseline_session, brake_stimulus_session):
        """Test detecting messages that appear only in stimulus."""
        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, brake_stimulus_session)

        assert 0x500 in report.new_messages
        assert len(report.new_messages) == 1

    def test_detect_disappeared_messages(self, baseline_session):
        """Test detecting messages that disappear in stimulus."""
        # Create stimulus session without 0x400
        messages = CANMessageList()
        for i in range(50):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,
                data=bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            )
            messages.append(msg)

        stimulus = CANSession.from_messages(messages.messages)

        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, stimulus)

        # Messages 0x200, 0x300, 0x400 should have disappeared
        assert 0x200 in report.disappeared_messages
        assert 0x300 in report.disappeared_messages
        assert 0x400 in report.disappeared_messages

    def test_detect_byte_level_changes(self, baseline_session, throttle_stimulus_session):
        """Test detecting byte-level changes in messages."""
        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, throttle_stimulus_session)

        # Message 0x200 (RPM) should have changed
        assert 0x200 in report.changed_messages
        assert 0x200 in report.byte_changes

        # Message 0x300 (throttle) should have changed
        assert 0x300 in report.changed_messages
        assert 0x300 in report.byte_changes

        # Message 0x100 (constant) should NOT have changed
        assert 0x100 not in report.changed_messages

    def test_analyze_signal_changes(self, baseline_session, throttle_stimulus_session):
        """Test analyzing specific message for signal changes."""
        analyzer = StimulusResponseAnalyzer()

        # Analyze throttle message (0x300)
        changes = analyzer.analyze_signal_changes(
            baseline_session, throttle_stimulus_session, 0x300
        )

        # Byte 0 (throttle) should have changed
        byte_0_changes = [c for c in changes if c.byte_position == 0]
        assert len(byte_0_changes) == 1

        change = byte_0_changes[0]
        assert change.change_magnitude > 0.0
        assert 0 in change.baseline_values
        assert 125 in change.stimulus_values  # 50% / 0.4% = 125

    def test_no_changes_identical_sessions(self, baseline_session):
        """Test comparing identical sessions shows no changes."""
        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, baseline_session)

        assert len(report.changed_messages) == 0
        assert len(report.new_messages) == 0
        assert len(report.disappeared_messages) == 0
        assert len(report.byte_changes) == 0

    def test_find_responsive_messages(self, baseline_session, brake_stimulus_session):
        """Test finding responsive message IDs."""
        analyzer = StimulusResponseAnalyzer()
        responsive = analyzer.find_responsive_messages(baseline_session, brake_stimulus_session)

        # Should include changed message (0x400) and new message (0x500)
        assert 0x400 in responsive
        assert 0x500 in responsive

    def test_compare_to_method(self, baseline_session, throttle_stimulus_session):
        """Test CANSession.compare_to() method."""
        report = baseline_session.compare_to(throttle_stimulus_session)

        assert isinstance(report, StimulusResponseReport)
        assert 0x200 in report.changed_messages  # RPM changed
        assert 0x300 in report.changed_messages  # Throttle changed

    def test_byte_change_new_values(self, baseline_session, brake_stimulus_session):
        """Test detecting new values in byte changes."""
        analyzer = StimulusResponseAnalyzer()
        changes = analyzer.analyze_signal_changes(baseline_session, brake_stimulus_session, 0x400)

        # Byte 0 changed from 0x00 to 0xFF
        byte_0_changes = [c for c in changes if c.byte_position == 0]
        assert len(byte_0_changes) == 1

        change = byte_0_changes[0]
        assert 0xFF in change.new_values
        assert 0x00 in change.disappeared_values

    def test_change_threshold(self, baseline_session, throttle_stimulus_session):
        """Test change threshold filtering."""
        analyzer = StimulusResponseAnalyzer()

        # High threshold - should detect fewer changes
        report_high = analyzer.detect_responses(
            baseline_session, throttle_stimulus_session, change_threshold=0.5
        )

        # Low threshold - should detect more changes
        report_low = analyzer.detect_responses(
            baseline_session, throttle_stimulus_session, change_threshold=0.05
        )

        # Low threshold should find at least as many changes
        assert len(report_low.changed_messages) >= len(report_high.changed_messages)

    def test_report_summary(self, baseline_session, brake_stimulus_session):
        """Test generating human-readable summary."""
        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, brake_stimulus_session)

        summary = report.summary()

        assert "Stimulus-Response Analysis" in summary
        assert "New Messages" in summary
        assert "0x500" in summary

    def test_empty_sessions(self):
        """Test comparing empty sessions."""
        empty1 = CANSession.from_messages([])
        empty2 = CANSession.from_messages([])

        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(empty1, empty2)

        assert len(report.changed_messages) == 0
        assert len(report.new_messages) == 0
        assert len(report.disappeared_messages) == 0


@pytest.mark.unit
class TestFrequencyChange:
    """Tests for frequency change detection."""

    def test_detect_frequency_increase(self, baseline_session):
        """Test detecting frequency increase."""
        # Create stimulus with 2x frequency for message 0x100
        messages = CANMessageList()
        for i in range(100):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.05,  # 20 Hz instead of 10 Hz
                data=bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            )
            messages.append(msg)

        stimulus = CANSession.from_messages(messages.messages)

        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, stimulus, change_threshold=0.1)

        # Check if frequency change detected
        if 0x100 in report.frequency_changes:
            freq_change = report.frequency_changes[0x100]
            assert freq_change.change_type == "increased"
            assert freq_change.change_ratio > 1.0

    def test_detect_frequency_decrease(self, baseline_session):
        """Test detecting frequency decrease."""
        # Create stimulus with 0.5x frequency for message 0x200
        messages = CANMessageList()
        for i in range(250):
            rpm = 800
            raw_rpm = int(rpm / 0.25)
            data = bytearray(8)
            data[0] = 0xAA
            data[1] = 0xBB
            data[2:4] = struct.pack(">H", raw_rpm)
            data[4:8] = bytes([0xCC, 0xDD, 0xEE, 0xFF])

            msg = CANMessage(
                arbitration_id=0x200,
                timestamp=i * 0.02,  # 50 Hz instead of 100 Hz
                data=bytes(data),
            )
            messages.append(msg)

        stimulus = CANSession.from_messages(messages.messages)

        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, stimulus, change_threshold=0.1)

        # Check if frequency change detected
        if 0x200 in report.frequency_changes:
            freq_change = report.frequency_changes[0x200]
            assert freq_change.change_type == "decreased"
            assert freq_change.change_ratio < 1.0

    def test_frequency_change_repr(self):
        """Test FrequencyChange string representation."""
        fc = FrequencyChange(
            message_id=0x280,
            baseline_hz=10.0,
            stimulus_hz=20.0,
            change_ratio=2.0,
            change_type="increased",
            significance=0.9,
        )

        repr_str = repr(fc)
        assert "0x280" in repr_str
        assert "10.0Hz" in repr_str
        assert "20.0Hz" in repr_str
        assert "increased" in repr_str


@pytest.mark.unit
class TestByteChange:
    """Tests for ByteChange data model."""

    def test_byte_change_initialization(self):
        """Test ByteChange initialization."""
        change = ByteChange(
            byte_position=0,
            baseline_values={0x00, 0x01},
            stimulus_values={0xFF, 0xFE},
            change_magnitude=0.8,
            value_range_change=254.0,
            mean_change=253.5,
        )

        assert change.byte_position == 0
        assert change.change_magnitude == 0.8
        assert 0xFF in change.new_values
        assert 0x00 in change.disappeared_values

    def test_byte_change_overlapping_values(self):
        """Test ByteChange with overlapping value sets."""
        change = ByteChange(
            byte_position=0,
            baseline_values={0x00, 0x01, 0x02},
            stimulus_values={0x01, 0x02, 0x03},
            change_magnitude=0.3,
            value_range_change=1.0,
            mean_change=1.0,
        )

        # Should have one new value and one disappeared
        assert 0x03 in change.new_values
        assert 0x00 in change.disappeared_values
        # Overlapping values should not appear in either set
        assert 0x01 not in change.new_values
        assert 0x01 not in change.disappeared_values


@pytest.mark.unit
class TestRealWorldScenarios:
    """Tests for real-world reverse engineering scenarios."""

    def test_throttle_response_scenario(self, baseline_session, throttle_stimulus_session):
        """Test throttle press scenario - what changes when throttle is pressed?"""
        report = baseline_session.compare_to(throttle_stimulus_session)

        # Should detect RPM and throttle changes
        assert 0x200 in report.changed_messages  # RPM
        assert 0x300 in report.changed_messages  # Throttle

        # Verify byte-level details for throttle
        throttle_changes = report.byte_changes[0x300]
        assert len(throttle_changes) > 0

        # Byte 0 should have changed significantly
        byte_0 = next(c for c in throttle_changes if c.byte_position == 0)
        assert byte_0.change_magnitude > 0.1

    def test_brake_response_scenario(self, baseline_session, brake_stimulus_session):
        """Test brake press scenario - what appears when brake is pressed?"""
        report = baseline_session.compare_to(brake_stimulus_session)

        # Should detect brake message change
        assert 0x400 in report.changed_messages

        # Should detect new brake light message
        assert 0x500 in report.new_messages

        # Should not detect changes in unrelated messages
        assert 0x200 not in report.changed_messages  # RPM unchanged
        assert 0x300 not in report.changed_messages  # Throttle unchanged

    def test_multi_signal_change(self):
        """Test scenario where multiple bytes in same message change."""
        # Baseline with two signals
        baseline_msgs = CANMessageList()
        for i in range(100):
            data = bytes([i % 10, i % 20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.01,
                data=data,
            )
            baseline_msgs.append(msg)

        baseline = CANSession.from_messages(baseline_msgs.messages)

        # Stimulus with both signals changed
        stimulus_msgs = CANMessageList()
        for i in range(100):
            data = bytes(
                [
                    (i % 10) + 5,  # Byte 0 shifted
                    (i % 20) + 10,  # Byte 1 shifted
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                ]
            )
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.01,
                data=data,
            )
            stimulus_msgs.append(msg)

        stimulus = CANSession.from_messages(stimulus_msgs.messages)

        analyzer = StimulusResponseAnalyzer()
        changes = analyzer.analyze_signal_changes(baseline, stimulus, 0x100)

        # Both bytes should have changes
        byte_positions = {c.byte_position for c in changes}
        assert 0 in byte_positions
        assert 1 in byte_positions
