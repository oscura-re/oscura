"""Comprehensive unit tests for digital buffer characterization workflow.

Requirements tested:

Coverage target: 90%+ of src/oscura/workflows/digital.py
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.workflows.digital import (
    _generate_buffer_report,
    _get_logic_specs,
    characterize_buffer,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestGetLogicSpecs:
    """Test _get_logic_specs helper function."""

    def test_ttl_specs(self):
        """Test TTL logic family specifications."""
        specs = _get_logic_specs("TTL")

        assert specs["vih"] == 2.0
        assert specs["vil"] == 0.8
        assert specs["max_rise_time"] == 10e-9
        assert specs["max_fall_time"] == 10e-9

    def test_cmos_5v_specs(self):
        """Test CMOS 5V logic family specifications."""
        specs = _get_logic_specs("CMOS_5V")

        assert specs["vih"] == 3.5
        assert specs["vil"] == 1.5
        assert specs["max_rise_time"] == 15e-9
        assert specs["max_fall_time"] == 15e-9

    def test_cmos_3v3_specs(self):
        """Test CMOS 3.3V logic family specifications."""
        specs = _get_logic_specs("CMOS_3V3")

        assert specs["vih"] == 2.0
        assert specs["vil"] == 0.8
        assert specs["max_rise_time"] == 5e-9
        assert specs["max_fall_time"] == 5e-9

    def test_lvttl_specs(self):
        """Test LVTTL logic family specifications."""
        specs = _get_logic_specs("LVTTL")

        assert specs["vih"] == 2.0
        assert specs["vil"] == 0.8
        assert specs["max_rise_time"] == 3e-9
        assert specs["max_fall_time"] == 3e-9

    def test_lvcmos_specs(self):
        """Test LVCMOS logic family specifications."""
        specs = _get_logic_specs("LVCMOS")

        assert specs["vih"] == 1.7
        assert specs["vil"] == 0.7
        assert specs["max_rise_time"] == 2e-9
        assert specs["max_fall_time"] == 2e-9

    def test_unknown_family_defaults_to_cmos_3v3(self):
        """Test that unknown logic family defaults to CMOS_3V3."""
        specs = _get_logic_specs("UNKNOWN_FAMILY")

        # Should default to CMOS_3V3
        assert specs["vih"] == 2.0
        assert specs["vil"] == 0.8
        assert specs["max_rise_time"] == 5e-9
        assert specs["max_fall_time"] == 5e-9


@pytest.mark.unit
class TestGenerateBufferReport:
    """Test _generate_buffer_report function."""

    def test_report_generation(self, tmp_path: Path):
        """Test HTML report generation."""
        result = {
            "logic_family": "CMOS_3V3",
            "confidence": 0.95,
            "rise_time": 5e-9,
            "fall_time": 6e-9,
            "overshoot_percent": 10.5,
            "undershoot_percent": 8.2,
            "status": "PASS",
        }

        output_path = tmp_path / "report.html"
        _generate_buffer_report(result, str(output_path))

        assert output_path.exists()

        # Verify HTML content
        content = output_path.read_text()
        assert "<html>" in content
        assert "Buffer Characterization Report" in content
        assert "CMOS_3V3" in content
        assert "95.0%" in content  # Confidence
        assert "5.00" in content  # Rise time in ns
        assert "6.00" in content  # Fall time in ns
        assert "10.5" in content  # Overshoot %
        assert "8.2" in content  # Undershoot %
        assert "PASS" in content

    def test_report_with_fail_status(self, tmp_path: Path):
        """Test report generation with FAIL status."""
        result = {
            "logic_family": "TTL",
            "confidence": 0.85,
            "rise_time": 15e-9,
            "fall_time": 12e-9,
            "overshoot_percent": 25.0,
            "undershoot_percent": 20.0,
            "status": "FAIL",
        }

        output_path = tmp_path / "fail_report.html"
        _generate_buffer_report(result, str(output_path))

        content = output_path.read_text()
        assert "FAIL" in content
        assert "TTL" in content


@pytest.mark.unit
class TestCharacterizeBuffer:
    """Test characterize_buffer main function."""

    @pytest.fixture
    def mock_trace(self) -> WaveformTrace:
        """Create a mock waveform trace for testing."""
        # Create a digital signal with transitions
        samples = 1000
        data = np.zeros(samples)
        # Add some transitions
        data[200:400] = 3.3  # High level
        data[600:800] = 3.3  # High level

        metadata = TraceMetadata(sample_rate=1e9)  # 1 GSa/s
        return WaveformTrace(data=data, metadata=metadata)

    @pytest.fixture
    def mock_reference_trace(self) -> WaveformTrace:
        """Create a mock reference trace for propagation delay testing."""
        samples = 1000
        data = np.zeros(samples)
        data[190:390] = 3.3  # Slightly earlier transitions
        data[590:790] = 3.3

        metadata = TraceMetadata(sample_rate=1e9)
        return WaveformTrace(data=data, metadata=metadata)

    def test_auto_detect_logic_family(self, mock_trace: WaveformTrace):
        """Test automatic logic family detection."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            # Setup mocks
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 3.5e-9
            mock_overshoot.return_value = 0.2
            mock_undershoot.return_value = 0.15

            result = characterize_buffer(mock_trace)

            # Verify auto-detection was called
            mock_detect.assert_called_once_with(mock_trace)

            # Verify result
            assert result["logic_family"] == "CMOS_3V3"
            assert result["confidence"] == 0.95
            assert result["voh"] == 3.2
            assert result["vol"] == 0.1
            assert result["rise_time"] == 4e-9
            assert result["fall_time"] == 3.5e-9
            assert result["status"] == "PASS"  # Within CMOS_3V3 specs

    def test_specified_logic_family(self, mock_trace: WaveformTrace):
        """Test with explicitly specified logic family."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_rise.return_value = 8e-9
            mock_fall.return_value = 7e-9
            mock_overshoot.return_value = 0.3
            mock_undershoot.return_value = 0.25

            result = characterize_buffer(mock_trace, logic_family="TTL")

            # Verify auto-detection was NOT called
            mock_detect.assert_not_called()

            # Verify result uses specified family
            assert result["logic_family"] == "TTL"
            assert result["confidence"] == 1.0  # Full confidence when specified
            assert result["rise_time"] == 8e-9
            assert result["fall_time"] == 7e-9

    def test_voh_vol_calculation_when_specified(self, mock_trace: WaveformTrace):
        """Test VOH/VOL are calculated from trace when logic family specified."""
        with (
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_rise.return_value = 5e-9
            mock_fall.return_value = 5e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            result = characterize_buffer(mock_trace, logic_family="CMOS_3V3")

            # VOH/VOL should be percentiles of the data
            expected_voh = np.percentile(mock_trace.data, 95)
            expected_vol = np.percentile(mock_trace.data, 5)

            assert abs(result["voh"] - expected_voh) < 0.01
            assert abs(result["vol"] - expected_vol) < 0.01

    def test_overshoot_undershoot_percentages(self, mock_trace: WaveformTrace):
        """Test overshoot/undershoot percentage calculation."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_5V",
                    "confidence": 0.9,
                    "voh": 5.0,
                    "vol": 0.0,
                }
            }
            mock_rise.return_value = 10e-9
            mock_fall.return_value = 10e-9
            mock_overshoot.return_value = 0.5  # 0.5V overshoot
            mock_undershoot.return_value = 0.25  # 0.25V undershoot

            result = characterize_buffer(mock_trace)

            # Swing = 5.0 - 0.0 = 5.0V
            # Overshoot % = (0.5 / 5.0) * 100 = 10%
            # Undershoot % = (0.25 / 5.0) * 100 = 5%
            assert result["overshoot_percent"] == 10.0
            assert result["undershoot_percent"] == 5.0

    def test_zero_swing_handling(self, mock_trace: WaveformTrace):
        """Test handling of zero swing (VOH == VOL)."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.5,
                    "voh": 1.5,
                    "vol": 1.5,  # Same as VOH
                }
            }
            mock_rise.return_value = 5e-9
            mock_fall.return_value = 5e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            result = characterize_buffer(mock_trace)

            # Zero swing should result in 0% overshoot/undershoot
            assert result["overshoot_percent"] == 0.0
            assert result["undershoot_percent"] == 0.0

    def test_noise_margin_calculation(self, mock_trace: WaveformTrace):
        """Test noise margin calculation."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "TTL",
                    "confidence": 0.92,
                    "voh": 3.0,
                    "vol": 0.2,
                }
            }
            mock_rise.return_value = 8e-9
            mock_fall.return_value = 8e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.05

            result = characterize_buffer(mock_trace)

            # TTL specs: VIH=2.0, VIL=0.8
            # Noise margin high = VOH - VIH = 3.0 - 2.0 = 1.0
            # Noise margin low = VIL - VOL = 0.8 - 0.2 = 0.6
            assert abs(result["noise_margin_high"] - 1.0) < 1e-9
            assert abs(result["noise_margin_low"] - 0.6) < 1e-9

    def test_propagation_delay_with_reference(
        self, mock_trace: WaveformTrace, mock_reference_trace: WaveformTrace
    ):
        """Test propagation delay measurement with reference trace."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
            patch("oscura.analyzers.digital.timing.propagation_delay") as mock_prop_delay,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1
            mock_prop_delay.return_value = 2.5e-9  # 2.5 ns delay

            result = characterize_buffer(mock_trace, reference_trace=mock_reference_trace)

            # Verify propagation delay was measured
            mock_prop_delay.assert_called_once_with(mock_reference_trace, mock_trace)

            assert result["propagation_delay"] == 2.5e-9
            assert isinstance(result["reference_comparison"], dict)
            assert "propagation_delay" in result["reference_comparison"]
            assert result["reference_comparison"]["propagation_delay"] == 2.5e-9

    def test_propagation_delay_failure_handling(
        self, mock_trace: WaveformTrace, mock_reference_trace: WaveformTrace
    ):
        """Test graceful handling of propagation delay measurement failure."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
            patch("oscura.analyzers.digital.timing.propagation_delay") as mock_prop_delay,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1
            mock_prop_delay.side_effect = Exception("Measurement failed")

            result = characterize_buffer(mock_trace, reference_trace=mock_reference_trace)

            # Should not raise, propagation_delay should be None
            assert result["propagation_delay"] is None
            assert result["reference_comparison"] is None

    def test_no_reference_trace(self, mock_trace: WaveformTrace):
        """Test characterization without reference trace."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            result = characterize_buffer(mock_trace)

            assert result["propagation_delay"] is None
            assert result["reference_comparison"] is None

    def test_custom_thresholds_pass(self, mock_trace: WaveformTrace):
        """Test pass/fail with custom thresholds (passing case)."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 8e-9
            mock_fall.return_value = 7e-9
            mock_overshoot.return_value = 0.5
            mock_undershoot.return_value = 0.3

            thresholds = {
                "rise_time": 10e-9,
                "fall_time": 10e-9,
                "overshoot_percent": 20.0,
            }

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            # All within thresholds
            assert result["status"] == "PASS"

    def test_custom_thresholds_fail_rise_time(self, mock_trace: WaveformTrace):
        """Test fail status when rise time exceeds threshold."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 15e-9  # Exceeds threshold
            mock_fall.return_value = 5e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            thresholds = {"rise_time": 10e-9}

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            assert result["status"] == "FAIL"

    def test_custom_thresholds_fail_fall_time(self, mock_trace: WaveformTrace):
        """Test fail status when fall time exceeds threshold."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 5e-9
            mock_fall.return_value = 12e-9  # Exceeds threshold
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            thresholds = {"fall_time": 10e-9}

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            assert result["status"] == "FAIL"

    def test_custom_thresholds_fail_overshoot(self, mock_trace: WaveformTrace):
        """Test fail status when overshoot exceeds threshold."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.0,
                    "vol": 0.0,
                }
            }
            mock_rise.return_value = 5e-9
            mock_fall.return_value = 5e-9
            mock_overshoot.return_value = 0.9  # 30% overshoot
            mock_undershoot.return_value = 0.1

            thresholds = {"overshoot_percent": 20.0}

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            assert result["status"] == "FAIL"

    def test_logic_family_default_thresholds_pass(self, mock_trace: WaveformTrace):
        """Test pass/fail with logic family default thresholds (pass)."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9  # Within 5ns limit
            mock_fall.return_value = 4.5e-9  # Within 5ns limit
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            result = characterize_buffer(mock_trace)

            # CMOS_3V3 max times are 5ns
            assert result["status"] == "PASS"

    def test_logic_family_default_thresholds_fail_rise(self, mock_trace: WaveformTrace):
        """Test fail with logic family default thresholds (rise time)."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 8e-9  # Exceeds 5ns limit
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            result = characterize_buffer(mock_trace)

            assert result["status"] == "FAIL"

    def test_logic_family_default_thresholds_fail_fall(self, mock_trace: WaveformTrace):
        """Test fail with logic family default thresholds (fall time)."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 7e-9  # Exceeds 5ns limit
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            result = characterize_buffer(mock_trace)

            assert result["status"] == "FAIL"

    def test_rise_fall_time_measurement_error(self, mock_trace: WaveformTrace):
        """Test error handling when rise/fall time measurement fails."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.side_effect = ValueError("No edges found")

            with pytest.raises(AnalysisError, match="Failed to measure rise/fall time"):
                characterize_buffer(mock_trace)

    def test_report_generation_integration(self, mock_trace: WaveformTrace, tmp_path: Path):
        """Test report generation through characterize_buffer."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "TTL",
                    "confidence": 0.88,
                    "voh": 3.5,
                    "vol": 0.2,
                }
            }
            mock_rise.return_value = 9e-9
            mock_fall.return_value = 8.5e-9
            mock_overshoot.return_value = 0.3
            mock_undershoot.return_value = 0.25

            report_path = tmp_path / "buffer_report.html"
            result = characterize_buffer(mock_trace, report=str(report_path))

            # Verify report was created
            assert report_path.exists()

            # Verify report contains expected data
            content = report_path.read_text()
            assert "TTL" in content
            assert "88.0%" in content  # Confidence
            assert "9.00" in content  # Rise time

    def test_complete_result_structure(self, mock_trace: WaveformTrace):
        """Test that result contains all expected fields."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "LVCMOS",
                    "confidence": 0.92,
                    "voh": 1.8,
                    "vol": 0.0,
                }
            }
            mock_rise.return_value = 1.5e-9
            mock_fall.return_value = 1.8e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.05

            result = characterize_buffer(mock_trace)

            # Verify all expected fields are present
            expected_fields = [
                "logic_family",
                "confidence",
                "rise_time",
                "fall_time",
                "propagation_delay",
                "overshoot",
                "overshoot_percent",
                "undershoot",
                "undershoot_percent",
                "noise_margin_high",
                "noise_margin_low",
                "voh",
                "vol",
                "status",
                "reference_comparison",
            ]

            for field in expected_fields:
                assert field in result, f"Missing field: {field}"

    def test_result_types(self, mock_trace: WaveformTrace):
        """Test that result values have correct types."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "TTL",
                    "confidence": 0.9,
                    "voh": 3.0,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 7e-9
            mock_fall.return_value = 6e-9
            mock_overshoot.return_value = 0.2
            mock_undershoot.return_value = 0.15

            result = characterize_buffer(mock_trace)

            # Check types
            assert isinstance(result["logic_family"], str)
            assert isinstance(result["confidence"], int | float)
            assert isinstance(result["rise_time"], int | float)
            assert isinstance(result["fall_time"], int | float)
            assert isinstance(result["overshoot"], int | float)
            assert isinstance(result["overshoot_percent"], int | float)
            assert isinstance(result["undershoot"], int | float)
            assert isinstance(result["undershoot_percent"], int | float)
            assert isinstance(result["noise_margin_high"], int | float)
            assert isinstance(result["noise_margin_low"], int | float)
            assert isinstance(result["voh"], int | float)
            assert isinstance(result["vol"], int | float)
            assert isinstance(result["status"], str)
            assert result["status"] in ["PASS", "FAIL"]

    def test_multiple_threshold_failures(self, mock_trace: WaveformTrace):
        """Test that status is FAIL when multiple thresholds are exceeded."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.0,
                    "vol": 0.0,
                }
            }
            mock_rise.return_value = 15e-9  # Exceeds threshold
            mock_fall.return_value = 20e-9  # Exceeds threshold
            mock_overshoot.return_value = 1.5  # 50% overshoot
            mock_undershoot.return_value = 0.1

            thresholds = {
                "rise_time": 10e-9,
                "fall_time": 10e-9,
                "overshoot_percent": 20.0,
            }

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            # Should still be FAIL (any failure causes FAIL)
            assert result["status"] == "FAIL"

    def test_reference_comparison_with_none_propagation_delay(
        self, mock_trace: WaveformTrace, mock_reference_trace: WaveformTrace
    ):
        """Test reference_comparison is None when propagation_delay is None."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
            patch("oscura.analyzers.digital.timing.propagation_delay") as mock_prop_delay,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1
            mock_prop_delay.side_effect = Exception("Failed")

            result = characterize_buffer(mock_trace, reference_trace=mock_reference_trace)

            # Both should be None
            assert result["propagation_delay"] is None
            assert result["reference_comparison"] is None

    def test_edge_case_negative_swing(self, mock_trace: WaveformTrace):
        """Test handling when VOL > VOH (negative swing)."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.5,
                    "voh": 1.0,
                    "vol": 3.0,  # VOL > VOH
                }
            }
            mock_rise.return_value = 5e-9
            mock_fall.return_value = 5e-9
            mock_overshoot.return_value = 0.5
            mock_undershoot.return_value = 0.3

            result = characterize_buffer(mock_trace)

            # Negative swing, should result in 0% percentages
            assert result["overshoot_percent"] == 0.0
            assert result["undershoot_percent"] == 0.0

    def test_all_logic_family_specs(self, mock_trace: WaveformTrace):
        """Test all supported logic families are handled correctly."""
        families = ["TTL", "CMOS_5V", "CMOS_3V3", "LVTTL", "LVCMOS"]

        for family in families:
            with (
                patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
                patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
                patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
                patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
            ):
                mock_rise.return_value = 2e-9
                mock_fall.return_value = 2e-9
                mock_overshoot.return_value = 0.1
                mock_undershoot.return_value = 0.05

                result = characterize_buffer(mock_trace, logic_family=family)

                # Should successfully characterize with each family
                assert result["logic_family"] == family
                assert result["confidence"] == 1.0
                assert "noise_margin_high" in result
                assert "noise_margin_low" in result

    def test_reference_comparison_populated_correctly(
        self, mock_trace: WaveformTrace, mock_reference_trace: WaveformTrace
    ):
        """Test that reference_comparison dict is populated correctly."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
            patch("oscura.analyzers.digital.timing.propagation_delay") as mock_prop_delay,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1
            mock_prop_delay.return_value = 3e-9

            result = characterize_buffer(mock_trace, reference_trace=mock_reference_trace)

            # Verify reference_comparison structure
            assert result["reference_comparison"] is not None
            assert "propagation_delay" in result["reference_comparison"]
            assert "timing_drift" in result["reference_comparison"]
            assert result["reference_comparison"]["propagation_delay"] == 3e-9
            # timing_drift is always None in current implementation
            assert result["reference_comparison"]["timing_drift"] is None

    def test_overshoot_undershoot_voltage_values(self, mock_trace: WaveformTrace):
        """Test that overshoot and undershoot voltage values are correctly returned."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "TTL",
                    "confidence": 0.9,
                    "voh": 3.5,
                    "vol": 0.3,
                }
            }
            mock_rise.return_value = 5e-9
            mock_fall.return_value = 5e-9
            mock_overshoot.return_value = 0.7  # Voltage value
            mock_undershoot.return_value = 0.4  # Voltage value

            result = characterize_buffer(mock_trace)

            # Verify voltage values are returned
            assert result["overshoot"] == 0.7
            assert result["undershoot"] == 0.4

    def test_voh_vol_included_in_result(self, mock_trace: WaveformTrace):
        """Test that VOH and VOL are included in the result."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_5V",
                    "confidence": 0.88,
                    "voh": 4.8,
                    "vol": 0.2,
                }
            }
            mock_rise.return_value = 10e-9
            mock_fall.return_value = 10e-9
            mock_overshoot.return_value = 0.2
            mock_undershoot.return_value = 0.1

            result = characterize_buffer(mock_trace)

            assert "voh" in result
            assert "vol" in result
            assert result["voh"] == 4.8
            assert result["vol"] == 0.2

    def test_custom_threshold_only_overshoot(self, mock_trace: WaveformTrace):
        """Test custom thresholds with only overshoot_percent specified."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.3,
                    "vol": 0.0,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.99  # 30% overshoot
            mock_undershoot.return_value = 0.1

            # Only specify overshoot threshold, not rise/fall
            thresholds = {"overshoot_percent": 25.0}

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            # Should fail due to overshoot
            assert result["status"] == "FAIL"

    def test_custom_threshold_only_rise_time(self, mock_trace: WaveformTrace):
        """Test custom thresholds with only rise_time specified."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.3,
                    "vol": 0.0,
                }
            }
            mock_rise.return_value = 3e-9
            mock_fall.return_value = 20e-9  # High fall time
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            # Only specify rise_time threshold
            thresholds = {"rise_time": 5e-9}

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            # Should pass - fall time not checked, rise time within threshold
            assert result["status"] == "PASS"

    def test_empty_threshold_dict(self, mock_trace: WaveformTrace):
        """Test with empty threshold dictionary."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.3,
                    "vol": 0.0,
                }
            }
            mock_rise.return_value = 4e-9
            mock_fall.return_value = 4e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            # Empty threshold dict means no thresholds to check
            # So it will pass (no failing conditions in the dict)
            thresholds = {}

            result = characterize_buffer(mock_trace, thresholds=thresholds)

            # With empty dict, no thresholds are checked, so should pass
            assert result["status"] == "PASS"

    def test_logic_family_without_max_rise_fall_specs(self, mock_trace: WaveformTrace):
        """Test behavior with unknown logic family (uses defaults)."""
        with (
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_rise.return_value = 100e-9  # Very slow
            mock_fall.return_value = 100e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            # Use unknown family that will default to CMOS_3V3
            result = characterize_buffer(mock_trace, logic_family="UNKNOWN_FAMILY_XYZ")

            # Should use CMOS_3V3 defaults and fail due to slow edges
            assert result["status"] == "FAIL"

    def test_fall_time_exception_handling(self, mock_trace: WaveformTrace):
        """Test error handling when fall_time measurement fails."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "CMOS_3V3",
                    "confidence": 0.95,
                    "voh": 3.2,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 5e-9
            mock_fall.side_effect = RuntimeError("No falling edges detected")

            with pytest.raises(AnalysisError, match="Failed to measure rise/fall time"):
                characterize_buffer(mock_trace)

    def test_percentile_calculation_for_specified_family(self, mock_trace: WaveformTrace):
        """Test that VOH/VOL use percentiles when family is specified."""
        with (
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_rise.return_value = 5e-9
            mock_fall.return_value = 5e-9
            mock_overshoot.return_value = 0.1
            mock_undershoot.return_value = 0.1

            # Specify family to trigger percentile calculation path
            result = characterize_buffer(mock_trace, logic_family="TTL")

            # VOH and VOL should come from np.percentile
            assert result["voh"] == np.percentile(mock_trace.data, 95)
            assert result["vol"] == np.percentile(mock_trace.data, 5)

    def test_report_path_handling(self, mock_trace: WaveformTrace, tmp_path: Path):
        """Test that report is saved to the correct path."""
        with (
            patch("oscura.inference.logic.detect_logic_family") as mock_detect,
            patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rise,
            patch("oscura.analyzers.waveform.measurements.fall_time") as mock_fall,
            patch("oscura.analyzers.waveform.measurements.overshoot") as mock_overshoot,
            patch("oscura.analyzers.waveform.measurements.undershoot") as mock_undershoot,
        ):
            mock_detect.return_value = {
                "primary": {
                    "name": "LVTTL",
                    "confidence": 0.92,
                    "voh": 3.0,
                    "vol": 0.1,
                }
            }
            mock_rise.return_value = 2.5e-9
            mock_fall.return_value = 2.8e-9
            mock_overshoot.return_value = 0.15
            mock_undershoot.return_value = 0.12

            # Create nested directory structure
            report_dir = tmp_path / "reports" / "buffers"
            report_dir.mkdir(parents=True)
            report_path = report_dir / "test_buffer.html"

            result = characterize_buffer(mock_trace, report=str(report_path))

            # Verify file was created at correct path
            assert report_path.exists()
            assert report_path.is_file()

            # Verify content
            content = report_path.read_text()
            assert "LVTTL" in content
            assert "92.0%" in content

    def test_module_all_exports(self):
        """Test that __all__ exports the expected functions."""
        from oscura.workflows.digital import __all__

        assert "characterize_buffer" in __all__
        assert len(__all__) >= 1


@pytest.mark.unit
class TestIntegrationScenarios:
    """Integration-style tests that exercise real code paths."""

    @pytest.fixture
    def realistic_trace(self) -> WaveformTrace:
        """Create a realistic digital waveform with rise/fall times."""
        sample_rate = 10e9  # 10 GSa/s
        samples = 2000
        data = np.zeros(samples)

        # Create transitions with realistic rise/fall times
        # Low period: 0-500
        # Rising edge: 500-550 (5ns rise time at 10 GSa/s)
        # High period: 550-1500
        # Falling edge: 1500-1550 (5ns fall time)
        # Low period: 1550-2000

        data[:500] = 0.1  # VOL
        data[500:550] = np.linspace(0.1, 3.2, 50)  # Rising edge
        data[550:1500] = 3.2  # VOH
        data[1500:1550] = np.linspace(3.2, 0.1, 50)  # Falling edge
        data[1550:] = 0.1  # VOL

        # Add small overshoot/undershoot
        data[550] = 3.4  # 0.2V overshoot
        data[1550] = -0.1  # 0.2V undershoot

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def test_integration_with_real_measurements(self, realistic_trace: WaveformTrace):
        """Test characterize_buffer with real measurement functions."""
        # This will actually execute the real measurement code
        result = characterize_buffer(realistic_trace, logic_family="CMOS_3V3")

        # Verify basic result structure
        assert isinstance(result, dict)
        assert result["logic_family"] == "CMOS_3V3"
        assert result["confidence"] == 1.0  # Specified family

        # Verify measurements were performed
        assert result["rise_time"] > 0
        assert result["fall_time"] > 0
        assert result["overshoot"] >= 0
        assert result["undershoot"] >= 0

        # Verify noise margins calculated
        assert isinstance(result["noise_margin_high"], int | float)
        assert isinstance(result["noise_margin_low"], int | float)

        # Verify status determination
        assert result["status"] in ["PASS", "FAIL"]

    def test_integration_auto_detect_family(self, realistic_trace: WaveformTrace):
        """Test with automatic logic family detection."""
        result = characterize_buffer(realistic_trace)

        # Should auto-detect a logic family
        assert result["logic_family"] is not None
        assert result["confidence"] > 0

    def test_integration_with_reference_trace(self, realistic_trace: WaveformTrace):
        """Test with reference trace for propagation delay."""
        # Create a reference trace (slightly earlier transitions)
        sample_rate = 10e9
        samples = 2000
        ref_data = np.zeros(samples)

        ref_data[:490] = 0.1
        ref_data[490:540] = np.linspace(0.1, 3.2, 50)
        ref_data[540:1490] = 3.2
        ref_data[1490:1540] = np.linspace(3.2, 0.1, 50)
        ref_data[1540:] = 0.1

        ref_metadata = TraceMetadata(sample_rate=sample_rate)
        reference = WaveformTrace(data=ref_data, metadata=ref_metadata)

        result = characterize_buffer(
            realistic_trace, reference_trace=reference, logic_family="CMOS_3V3"
        )

        # Propagation delay should be measured or None (if measurement fails)
        assert "propagation_delay" in result
        # Result is either a float or None

    def test_integration_with_custom_thresholds(self, realistic_trace: WaveformTrace):
        """Test with custom thresholds."""
        thresholds = {
            "rise_time": 1e-9,  # 1 ns - very strict
            "fall_time": 1e-9,
            "overshoot_percent": 5.0,
        }

        result = characterize_buffer(
            realistic_trace, logic_family="CMOS_3V3", thresholds=thresholds
        )

        # With strict thresholds, likely to fail
        assert result["status"] in ["PASS", "FAIL"]
        assert "rise_time" in result
        assert "fall_time" in result

    def test_integration_report_generation(self, realistic_trace: WaveformTrace, tmp_path: Path):
        """Test actual report generation."""
        report_path = tmp_path / "integration_report.html"

        result = characterize_buffer(realistic_trace, logic_family="TTL", report=str(report_path))

        # Report should be created
        assert report_path.exists()

        # Report should contain valid HTML
        content = report_path.read_text()
        assert "<html>" in content
        assert "Buffer Characterization Report" in content
        assert "TTL" in content

    def test_integration_all_logic_families(self, realistic_trace: WaveformTrace):
        """Test characterization with all supported logic families."""
        families = ["TTL", "CMOS_5V", "CMOS_3V3", "LVTTL", "LVCMOS"]

        for family in families:
            result = characterize_buffer(realistic_trace, logic_family=family)

            assert result["logic_family"] == family
            assert result["confidence"] == 1.0
            assert result["rise_time"] > 0
            assert result["fall_time"] > 0
            assert result["status"] in ["PASS", "FAIL"]

    def test_integration_simple_square_wave(self):
        """Test with a simple square wave pattern."""
        sample_rate = 1e9  # 1 GSa/s
        samples = 1000
        data = np.tile([0.0, 0.0, 3.3, 3.3], samples // 4)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = characterize_buffer(trace, logic_family="CMOS_3V3")

        assert result["logic_family"] == "CMOS_3V3"
        assert result["rise_time"] >= 0
        assert result["fall_time"] >= 0

    def test_integration_noisy_signal(self):
        """Test with a noisy digital signal."""
        sample_rate = 10e9
        samples = 2000
        rng = np.random.default_rng(42)

        # Base signal
        data = np.zeros(samples)
        data[:500] = 0.1
        data[500:550] = np.linspace(0.1, 3.2, 50)
        data[550:1500] = 3.2
        data[1500:1550] = np.linspace(3.2, 0.1, 50)
        data[1550:] = 0.1

        # Add noise
        noise = rng.normal(0, 0.05, samples)
        data = data + noise

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = characterize_buffer(trace, logic_family="CMOS_3V3")

        # Should still work despite noise
        assert isinstance(result, dict)
        assert result["logic_family"] == "CMOS_3V3"

    def test_get_logic_specs_directly(self):
        """Test _get_logic_specs function directly."""
        # Test each family
        ttl_specs = _get_logic_specs("TTL")
        assert ttl_specs["vih"] == 2.0
        assert ttl_specs["vil"] == 0.8

        cmos5v_specs = _get_logic_specs("CMOS_5V")
        assert cmos5v_specs["vih"] == 3.5
        assert cmos5v_specs["vil"] == 1.5

        # Test unknown family defaults to CMOS_3V3
        unknown_specs = _get_logic_specs("NONEXISTENT")
        cmos3v3_specs = _get_logic_specs("CMOS_3V3")
        assert unknown_specs == cmos3v3_specs

    def test_generate_buffer_report_directly(self, tmp_path: Path):
        """Test _generate_buffer_report function directly."""
        result = {
            "logic_family": "CMOS_3V3",
            "confidence": 0.95,
            "rise_time": 5e-9,
            "fall_time": 6e-9,
            "overshoot_percent": 10.0,
            "undershoot_percent": 8.0,
            "status": "PASS",
        }

        report_path = tmp_path / "direct_report.html"
        _generate_buffer_report(result, str(report_path))

        assert report_path.exists()
        content = report_path.read_text()

        # Check HTML structure
        assert content.count("<html>") == 1
        assert content.count("</html>") == 1

        # Check content
        assert "CMOS_3V3" in content
        assert "95.0%" in content
        assert "5.00" in content  # Rise time in ns
        assert "PASS" in content
