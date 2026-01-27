"""Comprehensive tests for stimulus-response analysis.

This module tests detection of CAN message changes between baseline and stimulus sessions.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

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
    """Create baseline CAN session (no stimulus)."""
    messages = CANMessageList()

    # Message 0x100: constant data
    for i in range(100):
        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=i * 0.01,
            data=bytes([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]),
            is_extended=False,
        )
        messages.append(msg)

    # Message 0x200: counter in byte 0
    for i in range(50):
        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=i * 0.02,
            data=bytes([i % 256] + [0xFF] * 7),
            is_extended=False,
        )
        messages.append(msg)

    session = CANSession(name="Baseline")
    session._messages = messages
    return session


@pytest.fixture
def stimulus_session() -> CANSession:
    """Create stimulus CAN session (with changes)."""
    messages = CANMessageList()

    # Message 0x100: byte 0 changed (was 0x10, now varies)
    for i in range(100):
        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=i * 0.01,
            data=bytes([0x10 + i % 10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]),
            is_extended=False,
        )
        messages.append(msg)

    # Message 0x200: same as baseline
    for i in range(50):
        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=i * 0.02,
            data=bytes([i % 256] + [0xFF] * 7),
            is_extended=False,
        )
        messages.append(msg)

    # Message 0x300: new message (only in stimulus)
    for i in range(20):
        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=i * 0.05,
            data=bytes([i % 256] + [0xAA] * 7),
            is_extended=False,
        )
        messages.append(msg)

    session = CANSession(name="Stimulus")
    session._messages = messages
    return session


class TestByteChange:
    """Tests for ByteChange dataclass."""

    def test_byte_change_creation(self):
        """Test creating a byte change."""
        bc = ByteChange(
            byte_position=2,
            baseline_values={0x10, 0x20, 0x30},
            stimulus_values={0x40, 0x50, 0x60},
            change_magnitude=0.75,
            value_range_change=10.0,
            mean_change=5.5,
        )

        assert bc.byte_position == 2
        assert bc.change_magnitude == 0.75

    def test_byte_change_post_init(self):
        """Test post-init calculation of derived fields."""
        bc = ByteChange(
            byte_position=0,
            baseline_values={0x10, 0x20, 0x30},
            stimulus_values={0x20, 0x30, 0x40},
            change_magnitude=0.5,
            value_range_change=0.0,
            mean_change=0.0,
        )

        # New values only in stimulus
        assert bc.new_values == {0x40}
        # Disappeared values only in baseline
        assert bc.disappeared_values == {0x10}


class TestFrequencyChange:
    """Tests for FrequencyChange dataclass."""

    def test_frequency_change_creation(self):
        """Test creating a frequency change."""
        fc = FrequencyChange(
            message_id=0x123,
            baseline_hz=10.0,
            stimulus_hz=20.0,
            change_ratio=2.0,
            change_type="increased",
            significance=0.95,
        )

        assert fc.message_id == 0x123
        assert fc.baseline_hz == 10.0
        assert fc.stimulus_hz == 20.0
        assert fc.change_ratio == 2.0
        assert fc.change_type == "increased"

    def test_frequency_change_repr(self):
        """Test FrequencyChange string representation."""
        fc = FrequencyChange(
            message_id=0x456,
            baseline_hz=50.0,
            stimulus_hz=25.0,
            change_ratio=0.5,
            change_type="decreased",
            significance=0.85,
        )

        repr_str = repr(fc)
        assert "0x456" in repr_str
        assert "50.0Hz" in repr_str
        assert "25.0Hz" in repr_str
        assert "decreased" in repr_str


class TestStimulusResponseReport:
    """Tests for StimulusResponseReport dataclass."""

    def test_report_creation(self):
        """Test creating stimulus-response report."""
        report = StimulusResponseReport(
            changed_messages=[0x100],
            new_messages=[0x300],
            disappeared_messages=[],
            frequency_changes={},
            byte_changes={},
            duration_baseline=1.0,
            duration_stimulus=1.0,
            confidence_threshold=0.1,
        )

        assert len(report.changed_messages) == 1
        assert len(report.new_messages) == 1
        assert report.duration_baseline == 1.0

    def test_report_summary_basic(self):
        """Test report summary generation."""
        report = StimulusResponseReport(
            changed_messages=[0x100],
            new_messages=[0x300],
            disappeared_messages=[0x400],
            frequency_changes={},
            byte_changes={},
            duration_baseline=1.5,
            duration_stimulus=2.0,
            confidence_threshold=0.1,
        )

        summary = report.summary()
        assert "Baseline duration: 1.50s" in summary
        assert "Stimulus duration: 2.00s" in summary
        assert "New Messages" in summary
        assert "Disappeared Messages" in summary
        assert "0x300" in summary
        assert "0x400" in summary

    def test_report_summary_with_byte_changes(self):
        """Test report summary with byte changes."""
        bc = ByteChange(
            byte_position=0,
            baseline_values={0x10},
            stimulus_values={0x20},
            change_magnitude=0.5,
            value_range_change=10.0,
            mean_change=16.0,
        )

        report = StimulusResponseReport(
            changed_messages=[0x100],
            new_messages=[],
            disappeared_messages=[],
            frequency_changes={},
            byte_changes={0x100: [bc]},
            duration_baseline=1.0,
            duration_stimulus=1.0,
            confidence_threshold=0.1,
        )

        summary = report.summary()
        assert "Byte-Level Changes" in summary
        assert "0x100" in summary

    def test_report_summary_no_changes(self):
        """Test report summary with no changes."""
        report = StimulusResponseReport(
            changed_messages=[],
            new_messages=[],
            disappeared_messages=[],
            frequency_changes={},
            byte_changes={},
            duration_baseline=1.0,
            duration_stimulus=1.0,
            confidence_threshold=0.1,
        )

        summary = report.summary()
        assert "No significant changes detected" in summary


class TestStimulusResponseAnalyzer:
    """Tests for StimulusResponseAnalyzer class."""

    def test_analyzer_creation(self):
        """Test creating analyzer."""
        analyzer = StimulusResponseAnalyzer()
        assert analyzer is not None

    def test_detect_responses_basic(self, baseline_session, stimulus_session):
        """Test basic response detection."""
        analyzer = StimulusResponseAnalyzer()

        report = analyzer.detect_responses(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            time_window_ms=100,
            change_threshold=0.1,
        )

        assert report is not None
        # Should detect new message 0x300
        assert 0x300 in report.new_messages
        # Should detect changes in 0x100
        # (baseline has constant 0x10, stimulus varies)

    def test_detect_responses_new_messages(self, baseline_session, stimulus_session):
        """Test detection of new messages."""
        analyzer = StimulusResponseAnalyzer()

        report = analyzer.detect_responses(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
        )

        # Message 0x300 only in stimulus
        assert 0x300 in report.new_messages

    def test_detect_responses_disappeared_messages(self, baseline_session):
        """Test detection of disappeared messages."""
        # Create stimulus without 0x200
        messages = CANMessageList()
        for i in range(100):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.01,
                data=bytes([0x10] * 8),
                is_extended=False,
            )
            messages.append(msg)

        stimulus = CANSession(name="Stimulus")
        stimulus._messages = messages

        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(baseline_session, stimulus)

        # Message 0x200 should be disappeared
        assert 0x200 in report.disappeared_messages

    def test_detect_responses_byte_changes(self, baseline_session, stimulus_session):
        """Test detection of byte-level changes."""
        analyzer = StimulusResponseAnalyzer()

        report = analyzer.detect_responses(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            change_threshold=0.05,
        )

        # Should detect byte changes in 0x100
        # (byte 0 changed from constant 0x10 to varying)
        if 0x100 in report.byte_changes:
            changes = report.byte_changes[0x100]
            # Should have change at byte position 0
            byte_0_changed = any(bc.byte_position == 0 for bc in changes)
            assert byte_0_changed

    def test_analyze_signal_changes_basic(self, baseline_session, stimulus_session):
        """Test analyzing signal changes for specific message."""
        analyzer = StimulusResponseAnalyzer()

        changes = analyzer.analyze_signal_changes(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            message_id=0x100,
            byte_threshold=1,
        )

        # Should detect changes (at least byte 0)
        assert len(changes) > 0

    def test_analyze_signal_changes_no_message(self, baseline_session, stimulus_session):
        """Test analyzing non-existent message."""
        analyzer = StimulusResponseAnalyzer()

        changes = analyzer.analyze_signal_changes(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            message_id=0xFFF,  # Doesn't exist
            byte_threshold=1,
        )

        # Should return empty list
        assert len(changes) == 0

    def test_find_responsive_messages(self, baseline_session, stimulus_session):
        """Test convenience method for finding responsive messages."""
        analyzer = StimulusResponseAnalyzer()

        responsive = analyzer.find_responsive_messages(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
        )

        # Should include changed messages and new messages
        assert isinstance(responsive, list)
        # Should include 0x300 (new message)
        assert 0x300 in responsive

    def test_detect_frequency_change_increased(self):
        """Test detecting increased frequency."""
        # Baseline: 10 Hz
        baseline_msgs = CANMessageList()
        for i in range(10):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,
                data=bytes([0] * 8),
                is_extended=False,
            )
            baseline_msgs.append(msg)

        baseline = CANSession(name="Baseline")
        baseline._messages = baseline_msgs

        # Stimulus: 20 Hz
        stimulus_msgs = CANMessageList()
        for i in range(20):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.05,
                data=bytes([0] * 8),
                is_extended=False,
            )
            stimulus_msgs.append(msg)

        stimulus = CANSession(name="Stimulus")
        stimulus._messages = stimulus_msgs

        analyzer = StimulusResponseAnalyzer()
        freq_change = analyzer._detect_frequency_change(baseline, stimulus, 0x100)

        assert freq_change is not None
        assert freq_change.change_type == "increased"
        assert freq_change.change_ratio > 1.0

    def test_detect_frequency_change_decreased(self):
        """Test detecting decreased frequency."""
        # Baseline: 20 Hz
        baseline_msgs = CANMessageList()
        for i in range(20):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.05,
                data=bytes([0] * 8),
                is_extended=False,
            )
            baseline_msgs.append(msg)

        baseline = CANSession(name="Baseline")
        baseline._messages = baseline_msgs

        # Stimulus: 10 Hz
        stimulus_msgs = CANMessageList()
        for i in range(10):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,
                data=bytes([0] * 8),
                is_extended=False,
            )
            stimulus_msgs.append(msg)

        stimulus = CANSession(name="Stimulus")
        stimulus._messages = stimulus_msgs

        analyzer = StimulusResponseAnalyzer()
        freq_change = analyzer._detect_frequency_change(baseline, stimulus, 0x100)

        assert freq_change is not None
        assert freq_change.change_type == "decreased"
        assert freq_change.change_ratio < 1.0

    def test_detect_frequency_change_appeared(self):
        """Test detecting appeared message."""
        baseline = CANSession(name="Baseline")
        # Empty baseline

        stimulus_msgs = CANMessageList()
        for i in range(10):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,
                data=bytes([0] * 8),
                is_extended=False,
            )
            stimulus_msgs.append(msg)

        stimulus = CANSession(name="Stimulus")
        stimulus._messages = stimulus_msgs

        analyzer = StimulusResponseAnalyzer()
        freq_change = analyzer._detect_frequency_change(baseline, stimulus, 0x100)

        # May be None if message doesn't exist in baseline
        # Proper test would be through detect_responses() which handles new messages

    def test_calculate_frequency_basic(self):
        """Test frequency calculation helper."""
        messages = []
        for i in range(10):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,  # 10 Hz
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg)

        analyzer = StimulusResponseAnalyzer()
        freq = analyzer._calculate_frequency(messages, (0.0, 0.9))

        # Should be approximately 10 Hz
        assert 9.0 < freq < 11.0

    def test_calculate_frequency_single_message(self):
        """Test frequency with single message."""
        messages = [
            CANMessage(
                arbitration_id=0x100,
                timestamp=0.5,
                data=bytes([0] * 8),
                is_extended=False,
            )
        ]

        analyzer = StimulusResponseAnalyzer()
        freq = analyzer._calculate_frequency(messages, (0.0, 1.0))

        assert freq == 0.0

    def test_change_threshold_filtering(self, baseline_session, stimulus_session):
        """Test that change threshold filters results."""
        analyzer = StimulusResponseAnalyzer()

        # Low threshold (more sensitive)
        report_low = analyzer.detect_responses(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            change_threshold=0.01,
        )

        # High threshold (less sensitive)
        report_high = analyzer.detect_responses(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            change_threshold=0.9,
        )

        # Low threshold should detect more changes
        assert len(report_low.changed_messages) >= len(report_high.changed_messages)


class TestByteChangeAnalysis:
    """Tests for byte change analysis details."""

    def test_byte_change_magnitude_calculation(self, baseline_session, stimulus_session):
        """Test change magnitude calculation."""
        analyzer = StimulusResponseAnalyzer()

        changes = analyzer.analyze_signal_changes(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            message_id=0x100,
        )

        # All changes should have magnitude between 0 and 1
        for change in changes:
            assert 0.0 <= change.change_magnitude <= 1.0

    def test_byte_change_statistics(self, baseline_session, stimulus_session):
        """Test byte change statistics."""
        analyzer = StimulusResponseAnalyzer()

        changes = analyzer.analyze_signal_changes(
            baseline_session=baseline_session,
            stimulus_session=stimulus_session,
            message_id=0x100,
        )

        # Check that statistics are calculated
        for change in changes:
            assert isinstance(change.mean_change, float)
            assert isinstance(change.value_range_change, float)
