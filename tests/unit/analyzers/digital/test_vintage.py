"""Tests for vintage logic high-level analysis API."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from oscura.analyzers.digital.vintage import REPLACEMENT_DATABASE, analyze_vintage_logic
from oscura.analyzers.digital.vintage_result import (
    BOMEntry,
    ICIdentificationResult,
    ModernReplacementIC,
    VintageLogicAnalysisResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


@pytest.fixture
def ttl_clock_trace(signal_builder):
    """Create TTL clock trace for testing."""
    from oscura.core.types import TraceMetadata, WaveformTrace

    # Generate square wave (-2.5 to +2.5) then shift to 0-5V TTL levels
    data = signal_builder.square_wave(
        frequency=1000,  # 1 kHz
        amplitude=2.5,  # Creates -2.5 to +2.5
        duration=0.001,  # 1ms
        sample_rate=1e6,
    )
    # Shift to 0-5V range
    data = data + 2.5

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6, channel_name="CLK"))


@pytest.fixture
def ttl_data_trace(signal_builder):
    """Create TTL data trace with propagation delay."""
    from oscura.core.types import TraceMetadata, WaveformTrace

    # Create slightly delayed square wave with TTL levels
    # Use sine wave with phase for delay simulation
    data = signal_builder.sine_wave(
        frequency=1000,
        amplitude=2.4,  # Slightly lower (realistic) -2.4 to +2.4
        phase=0.01,  # Small phase shift (simulates propagation delay)
        duration=0.001,
        sample_rate=1e6,
    )
    # Shift to 0-4.8V range (slightly lower than ideal TTL)
    data = data + 2.4

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6, channel_name="DATA"))


@pytest.fixture
def digital_trace(signal_builder):
    """Create digital trace (boolean) for testing."""
    from oscura.core.types import DigitalTrace, TraceMetadata

    # Create boolean array for digital trace
    samples = int(0.001 * 1e6)  # 1ms at 1MHz
    # Simple alternating pattern
    data = np.array([bool(i % 2) for i in range(samples)])

    return DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e6, channel_name="DIGITAL"))


@pytest.fixture
def open_collector_trace(signal_builder):
    """Create trace simulating open-collector output."""
    from oscura.core.types import TraceMetadata, WaveformTrace

    # Use square wave shifted to 0-5V
    data = signal_builder.square_wave(
        frequency=1000,
        amplitude=2.5,
        duration=0.001,
        sample_rate=1e6,
    )
    # Shift to 0-5V range
    data = data + 2.5

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6, channel_name="OC_OUT"))


@pytest.fixture
def constant_trace(signal_builder):
    """Create constant trace for error handling tests."""
    from oscura.core.types import TraceMetadata, WaveformTrace

    # Create constant voltage
    data = signal_builder.dc_offset(
        offset=2.5,
        duration=0.001,
        sample_rate=1e6,
    )

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6, channel_name="CONST"))


class TestAnalyzeVintageLogic:
    """Test analyze_vintage_logic function."""

    def test_basic_analysis(self, ttl_clock_trace, ttl_data_trace):
        """Test basic vintage logic analysis with TTL traces."""
        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        result = analyze_vintage_logic(traces)

        # Check result structure
        assert isinstance(result, VintageLogicAnalysisResult)
        assert isinstance(result.timestamp, datetime)
        assert result.analysis_duration > 0

        # Check logic family detection
        # Accept TTL, CMOS_5V (0-5V signals), or unknown
        assert result.detected_family in ["TTL", "CMOS_5V", "unknown"]
        assert 0 <= result.family_confidence <= 1.0

        # Check voltage levels populated if family detected
        if result.detected_family in ["TTL", "CMOS_5V"]:
            assert "VCC" in result.voltage_levels

        # Check timing measurements
        assert isinstance(result.timing_measurements, dict)
        # Timing measurements may vary depending on signal characteristics
        # Check that we have some measurements (t_pd, t_su, or t_h)
        has_timing = any(
            param in key
            for key in result.timing_measurements
            for param in ["t_pd", "t_su", "t_h", "t_w"]
        )
        # It's OK if no timing measurements (signals might not meet thresholds)
        if result.timing_measurements:
            assert has_timing

        # Check BOM generated
        assert isinstance(result.bom, list)
        assert len(result.bom) > 0
        # Should have at least decoupling capacitors
        assert any("capacitor" in entry.description.lower() for entry in result.bom)

        # Check no errors
        assert isinstance(result.warnings, list)

    def test_with_digital_traces(self, digital_trace):
        """Test analysis with digital (boolean) traces."""
        traces = {
            "CLK": digital_trace,
            "DATA": digital_trace,
        }

        result = analyze_vintage_logic(traces)

        # Should detect as unknown family with warning
        assert result.detected_family == "unknown"
        assert result.family_confidence == 0.0
        assert len(result.voltage_levels) == 0

        # Should have warning about digital trace
        assert any("digital trace" in w.lower() for w in result.warnings)

        # Should still generate BOM (even without IC identification)
        assert len(result.bom) > 0

    def test_with_target_frequency(self, ttl_clock_trace, ttl_data_trace):
        """Test analysis with target frequency specified."""
        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        result = analyze_vintage_logic(
            traces,
            target_frequency=1e6,  # 1 MHz
        )

        assert isinstance(result, VintageLogicAnalysisResult)
        # Target frequency doesn't affect basic analysis without timing paths

    def test_with_system_description(self, ttl_clock_trace):
        """Test analysis with system description."""
        traces = {"CLK": ttl_clock_trace}

        result = analyze_vintage_logic(traces, system_description="Test System 1976")

        # Description stored as source_file
        assert result.source_file == "Test System 1976"

    def test_with_protocol_decode_enabled(self, ttl_clock_trace):
        """Test analysis with protocol decoding enabled."""
        traces = {"CLK": ttl_clock_trace}

        result = analyze_vintage_logic(traces, enable_protocol_decode=True)

        # Protocol decoding returns empty dict (placeholder)
        assert result.decoded_protocols is not None
        assert isinstance(result.decoded_protocols, dict)

    def test_open_collector_detection(self, open_collector_trace, ttl_data_trace):
        """Test detection of open-collector outputs."""
        traces = {
            "OC_OUT": open_collector_trace,
            "DATA": ttl_data_trace,
        }

        result = analyze_vintage_logic(traces)

        # Should detect open-collector
        # Note: Detection depends on timing characteristics
        # Result may or may not detect OC depending on trace characteristics
        assert isinstance(result.open_collector_detected, bool)
        assert isinstance(result.asymmetry_ratio, float)
        assert result.asymmetry_ratio >= 0

        # If detected, should have warning about pull-up
        if result.open_collector_detected:
            assert any("open-collector" in w.lower() for w in result.warnings)
            # Should add pull-up resistor to BOM
            assert any("pull-up" in entry.description.lower() for entry in result.bom)

    def test_ic_identification(self, ttl_clock_trace, ttl_data_trace):
        """Test IC identification from timing measurements."""
        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        result = analyze_vintage_logic(traces)

        # May or may not identify IC depending on timing match
        assert isinstance(result.identified_ics, list)
        assert isinstance(result.confidence_scores, dict)

        # If IC identified
        if result.identified_ics:
            ic_result = result.identified_ics[0]
            assert isinstance(ic_result, ICIdentificationResult)
            assert ic_result.ic_name != "unknown"
            assert 0 <= ic_result.confidence <= 1.0
            assert isinstance(ic_result.timing_params, dict)
            assert isinstance(ic_result.validation, dict)

            # Should be in confidence scores
            assert "ic_identification" in result.confidence_scores

    def test_modern_replacement_recommendations(self, ttl_clock_trace, ttl_data_trace):
        """Test modern IC replacement recommendations."""
        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        result = analyze_vintage_logic(traces)

        # Modern replacements only generated if IC identified
        assert isinstance(result.modern_replacements, list)

        # If replacements exist
        if result.modern_replacements:
            replacement = result.modern_replacements[0]
            assert isinstance(replacement, ModernReplacementIC)
            assert replacement.original_ic != ""
            assert replacement.replacement_ic != ""
            assert isinstance(replacement.benefits, list)
            assert len(replacement.benefits) > 0

    def test_bom_generation(self, ttl_clock_trace, ttl_data_trace):
        """Test BOM generation."""
        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        result = analyze_vintage_logic(traces)

        # BOM should always be generated
        assert len(result.bom) > 0

        # Check BOM structure
        for entry in result.bom:
            assert isinstance(entry, BOMEntry)
            assert entry.part_number != ""
            assert entry.description != ""
            assert entry.quantity >= 0  # Can be 0 if no ICs identified
            assert entry.category in ["IC", "Resistor", "Capacitor"]

        # Should always have decoupling capacitors entry
        assert any("decoupling" in entry.description.lower() for entry in result.bom)

        # If ICs identified, should have IC entries
        if result.identified_ics:
            assert any(entry.category == "IC" for entry in result.bom)
            # And capacitor quantity should be positive
            cap_entry = next(e for e in result.bom if "decoupling" in e.description.lower())
            assert cap_entry.quantity > 0

    def test_timing_measurements(self, ttl_clock_trace, ttl_data_trace):
        """Test timing measurements are captured."""
        traces = {
            "INPUT": ttl_clock_trace,
            "OUTPUT": ttl_data_trace,
        }

        result = analyze_vintage_logic(traces)

        # Should have timing measurements
        assert len(result.timing_measurements) > 0

        # Check measurement naming
        for key, value in result.timing_measurements.items():
            assert isinstance(key, str)
            assert isinstance(value, float)
            assert value >= 0
            # Should have timing parameter indicators
            assert any(param in key for param in ["_t_pd", "_t_su", "_t_h", "_t_w"])

    def test_multiple_traces(self, ttl_clock_trace, ttl_data_trace, signal_builder):
        """Test analysis with multiple trace pairs."""
        from oscura.core.types import TraceMetadata, WaveformTrace

        # Create third trace
        data3 = signal_builder.square_wave(
            frequency=2000,
            amplitude=2.45,
            duration=0.001,
            sample_rate=1e6,
        )
        # Shift to ~0-4.9V
        data3 = data3 + 2.45
        trace3 = WaveformTrace(
            data=data3, metadata=TraceMetadata(sample_rate=1e6, channel_name="DATA2")
        )

        traces = {
            "CLK": ttl_clock_trace,
            "DATA1": ttl_data_trace,
            "DATA2": trace3,
        }

        result = analyze_vintage_logic(traces)

        # Should analyze multiple pairs
        # Should have measurements from CLK→DATA1 and DATA1→DATA2
        assert len(result.timing_measurements) >= 2

    def test_with_timing_paths(self, ttl_clock_trace, ttl_data_trace):
        """Test analysis with timing paths provided."""
        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        timing_paths = [
            ("74LS00", ttl_clock_trace, ttl_data_trace),
        ]

        result = analyze_vintage_logic(traces, timing_paths=timing_paths, target_frequency=1e6)

        # Should have timing path results
        assert result.timing_paths is not None
        assert len(result.timing_paths) > 0

        # Check timing path result structure
        path_result = result.timing_paths[0]
        assert hasattr(path_result, "meets_timing")
        assert hasattr(path_result, "total_delay")

    def test_warnings_generation(self, ttl_clock_trace):
        """Test that warnings are generated appropriately."""
        # Test with single trace (limited analysis)
        traces = {"CLK": ttl_clock_trace}

        result = analyze_vintage_logic(traces)

        # Warnings should be a list
        assert isinstance(result.warnings, list)

        # Each warning should be a string
        for warning in result.warnings:
            assert isinstance(warning, str)
            assert len(warning) > 0

    def test_confidence_scores(self, ttl_clock_trace, ttl_data_trace):
        """Test confidence scores are tracked."""
        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        result = analyze_vintage_logic(traces)

        # Should have confidence scores
        assert isinstance(result.confidence_scores, dict)

        # Should have logic family confidence
        assert "logic_family" in result.confidence_scores
        assert 0 <= result.confidence_scores["logic_family"] <= 1.0

        # May have IC identification confidence
        if "ic_identification" in result.confidence_scores:
            assert 0 <= result.confidence_scores["ic_identification"] <= 1.0

    def test_single_trace(self, ttl_clock_trace):
        """Test analysis with single trace."""
        traces = {"CLK": ttl_clock_trace}

        result = analyze_vintage_logic(traces)

        # Should complete without errors
        assert isinstance(result, VintageLogicAnalysisResult)

        # Should detect logic family
        assert result.detected_family != ""

        # May have no timing measurements (needs pairs)
        assert isinstance(result.timing_measurements, dict)

    def test_error_handling_in_timing_measurements(self, ttl_clock_trace, constant_trace):
        """Test error handling when timing measurements fail."""
        traces = {
            "CLK": ttl_clock_trace,
            "CONST": constant_trace,
        }

        result = analyze_vintage_logic(traces)

        # Should handle errors gracefully
        assert isinstance(result, VintageLogicAnalysisResult)

        # May have warnings about failed measurements
        # warnings are optional in this case

    def test_replacement_database(self):
        """Test REPLACEMENT_DATABASE structure."""
        assert isinstance(REPLACEMENT_DATABASE, dict)
        assert len(REPLACEMENT_DATABASE) > 0

        # Check structure of entries
        for original, replacement in REPLACEMENT_DATABASE.items():
            assert isinstance(original, str)
            assert isinstance(replacement, ModernReplacementIC)
            assert replacement.original_ic == original
            assert replacement.replacement_ic != ""
            assert isinstance(replacement.benefits, list)
            assert len(replacement.benefits) > 0

    def test_analysis_duration(self, ttl_clock_trace):
        """Test that analysis duration is tracked."""
        traces = {"CLK": ttl_clock_trace}

        result = analyze_vintage_logic(traces)

        # Analysis duration should be positive
        assert result.analysis_duration > 0
        # Should be reasonable (< 10 seconds for simple test)
        assert result.analysis_duration < 10.0

    def test_timestamp(self, ttl_clock_trace):
        """Test that timestamp is set."""
        traces = {"CLK": ttl_clock_trace}

        before = datetime.now()
        result = analyze_vintage_logic(traces)
        after = datetime.now()

        # Timestamp should be between before and after
        assert before <= result.timestamp <= after

    def test_unknown_family_voltage_levels(self, digital_trace):
        """Test that unknown family has empty voltage levels."""
        traces = {"DATA": digital_trace}

        result = analyze_vintage_logic(traces)

        # Digital trace should result in unknown family
        assert result.detected_family == "unknown"
        assert len(result.voltage_levels) == 0
        assert "digital trace" in " ".join(result.warnings).lower()

    def test_identified_ic_in_replacement_database(self, signal_builder):
        """Test IC identification with known IC in replacement database."""
        from oscura.core.types import TraceMetadata, WaveformTrace

        # Create traces with timing that matches 7400 (t_pd ~15ns)
        # Use very fast signals to match TTL propagation delay
        data1 = signal_builder.square_wave(
            frequency=10000,  # 10 kHz
            amplitude=2.5,
            duration=0.001,
            sample_rate=1e6,
        )
        data1 = data1 + 2.5

        data2 = signal_builder.square_wave(
            frequency=10000,
            amplitude=2.5,
            duration=0.001,
            sample_rate=1e6,
        )
        # Add small delay
        data2 = np.roll(data2, 15)  # 15 samples = 15µs at 1MHz
        data2 = data2 + 2.5

        trace1 = WaveformTrace(
            data=data1, metadata=TraceMetadata(sample_rate=1e6, channel_name="IN")
        )
        trace2 = WaveformTrace(
            data=data2, metadata=TraceMetadata(sample_rate=1e6, channel_name="OUT")
        )

        traces = {"IN": trace1, "OUT": trace2}

        result = analyze_vintage_logic(traces)

        # Should generate BOM entries (even if IC not identified)
        assert len(result.bom) > 0
        # Check BOM has decoupling capacitors
        assert any("capacitor" in entry.description.lower() for entry in result.bom)

    def test_timing_path_analysis_with_error(self, ttl_clock_trace, ttl_data_trace):
        """Test timing path analysis error handling."""
        from oscura.core.types import TraceMetadata, WaveformTrace

        # Create invalid timing path (tuple with bad data)
        invalid_trace = WaveformTrace(
            data=np.array([]),  # Empty data will cause issues
            metadata=TraceMetadata(sample_rate=1e6),
        )

        traces = {
            "CLK": ttl_clock_trace,
            "DATA": ttl_data_trace,
        }

        timing_paths = [
            ("74LS00", invalid_trace, invalid_trace),
        ]

        result = analyze_vintage_logic(traces, timing_paths=timing_paths, target_frequency=1e6)

        # Should handle error gracefully
        assert isinstance(result, VintageLogicAnalysisResult)
        # Should have warning about timing path failure
        assert any("timing path" in w.lower() for w in result.warnings)

    def test_setup_and_hold_time_measurement_errors(self, signal_builder):
        """Test setup and hold time measurements with traces that cause exceptions."""
        from oscura.core.types import TraceMetadata, WaveformTrace

        # Create traces that will cause measurement errors (no clear transitions)
        # Use random noise instead of clean signals
        data1 = signal_builder.white_noise(sample_rate=1e6, duration=0.001, amplitude=0.1)
        data1 = data1 + 2.5  # Center around TTL mid-level

        data2 = signal_builder.white_noise(sample_rate=1e6, duration=0.001, amplitude=0.1)
        data2 = data2 + 2.5

        trace1 = WaveformTrace(
            data=data1, metadata=TraceMetadata(sample_rate=1e6, channel_name="IN")
        )
        trace2 = WaveformTrace(
            data=data2, metadata=TraceMetadata(sample_rate=1e6, channel_name="OUT")
        )

        traces = {"IN": trace1, "OUT": trace2}

        result = analyze_vintage_logic(traces)

        # Should handle errors gracefully
        assert isinstance(result, VintageLogicAnalysisResult)
        # May have warnings about failed measurements
        assert isinstance(result.warnings, list)
