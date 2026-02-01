"""Tests for calibration metadata and traceability.

Validates CalibrationInfo and TraceMetadata integration for regulatory compliance.

References:
    ISO/IEC 17025:2017 - Testing and Calibration Laboratories
    21 CFR Part 11 - Electronic Records (FDA)
    NIST Handbook 150 - Laboratory Accreditation
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from oscura.core.types import CalibrationInfo, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestCalibrationInfo:
    """Test CalibrationInfo dataclass."""

    def test_basic_creation(self):
        """Test creating basic calibration info."""
        cal_info = CalibrationInfo(instrument="Tektronix DPO7254C")
        assert cal_info.instrument == "Tektronix DPO7254C"
        assert cal_info.serial_number is None
        assert cal_info.calibration_date is None

    def test_full_calibration_info(self):
        """Test creating complete calibration info."""
        cal_date = datetime(2024, 12, 15)
        due_date = datetime(2025, 12, 15)

        cal_info = CalibrationInfo(
            instrument="Tektronix DPO7254C",
            serial_number="C012345",
            calibration_date=cal_date,
            calibration_due_date=due_date,
            firmware_version="FV:1.0.0",
            calibration_lab="NIST-Traceable Cal Lab",
            calibration_cert_number="CAL-2024-12345",
            probe_attenuation=10.0,
            coupling="DC",
            bandwidth_limit=None,
            vertical_resolution=8,
        )

        assert cal_info.instrument == "Tektronix DPO7254C"
        assert cal_info.serial_number == "C012345"
        assert cal_info.calibration_date == cal_date
        assert cal_info.calibration_due_date == due_date
        assert cal_info.firmware_version == "FV:1.0.0"
        assert cal_info.probe_attenuation == 10.0
        assert cal_info.coupling == "DC"
        assert cal_info.vertical_resolution == 8

    def test_probe_attenuation_validation(self):
        """Test probe attenuation must be positive."""
        with pytest.raises(ValueError, match="probe_attenuation must be positive"):
            CalibrationInfo(instrument="Test", probe_attenuation=-1.0)

        with pytest.raises(ValueError, match="probe_attenuation must be positive"):
            CalibrationInfo(instrument="Test", probe_attenuation=0.0)

    def test_bandwidth_limit_validation(self):
        """Test bandwidth limit must be positive if set."""
        with pytest.raises(ValueError, match="bandwidth_limit must be positive"):
            CalibrationInfo(instrument="Test", bandwidth_limit=-1e6)

    def test_vertical_resolution_validation(self):
        """Test vertical resolution must be positive."""
        with pytest.raises(ValueError, match="vertical_resolution must be positive"):
            CalibrationInfo(instrument="Test", vertical_resolution=0)

    def test_is_calibration_current_valid(self):
        """Test checking if calibration is current."""
        cal_date = datetime.now() - timedelta(days=180)  # 6 months ago
        due_date = datetime.now() + timedelta(days=180)  # 6 months from now

        cal_info = CalibrationInfo(
            instrument="Test",
            calibration_date=cal_date,
            calibration_due_date=due_date,
        )

        assert cal_info.is_calibration_current is True

    def test_is_calibration_current_expired(self):
        """Test expired calibration detection."""
        cal_date = datetime.now() - timedelta(days=400)  # Over 1 year ago
        due_date = datetime.now() - timedelta(days=30)  # Expired 30 days ago

        cal_info = CalibrationInfo(
            instrument="Test",
            calibration_date=cal_date,
            calibration_due_date=due_date,
        )

        assert cal_info.is_calibration_current is False

    def test_is_calibration_current_no_dates(self):
        """Test calibration status when dates not set."""
        cal_info = CalibrationInfo(instrument="Test")
        assert cal_info.is_calibration_current is None

    def test_traceability_info_complete(self):
        """Test complete calibration traceability information."""
        cal_date = datetime(2024, 12, 15)
        due_date = datetime(2025, 12, 15)

        cal_info = CalibrationInfo(
            instrument="Tektronix DPO7254C",
            serial_number="C012345",
            calibration_date=cal_date,
            calibration_due_date=due_date,
            calibration_cert_number="CAL-2024-12345",
        )

        assert cal_info.instrument == "Tektronix DPO7254C"
        assert cal_info.serial_number == "C012345"
        assert cal_info.calibration_date == cal_date
        assert cal_info.calibration_due_date == due_date
        assert cal_info.calibration_cert_number == "CAL-2024-12345"

    def test_traceability_info_minimal(self):
        """Test minimal calibration traceability information."""
        cal_info = CalibrationInfo(instrument="Generic Scope")
        assert cal_info.instrument == "Generic Scope"
        assert cal_info.serial_number is None
        assert cal_info.calibration_cert_number is None


class TestTraceMetadataWithCalibration:
    """Test TraceMetadata with calibration info."""

    def test_metadata_with_calibration_info(self):
        """Test adding calibration info to trace metadata."""
        cal_info = CalibrationInfo(
            instrument="Keysight DSOX4024A",
            calibration_date=datetime(2024, 10, 1),
        )

        metadata = TraceMetadata(sample_rate=1e9, calibration=cal_info)

        assert metadata.sample_rate == 1e9
        assert metadata.calibration is not None
        assert metadata.calibration.instrument == "Keysight DSOX4024A"

    def test_metadata_without_calibration_info(self):
        """Test metadata without calibration info (backward compatible)."""
        metadata = TraceMetadata(sample_rate=1e6)
        assert metadata.sample_rate == 1e6
        assert metadata.calibration is None

    def test_waveform_trace_with_calibration(self):
        """Test complete waveform trace with calibration info."""
        cal_info = CalibrationInfo(
            instrument="Tektronix MSO64",
            serial_number="C987654",
            calibration_date=datetime(2024, 11, 20),
            probe_attenuation=10.0,
            vertical_resolution=12,
        )

        metadata = TraceMetadata(
            sample_rate=2.5e9,  # 2.5 GSa/s
            vertical_scale=0.1,  # 100 mV/div
            vertical_offset=0.0,
            calibration=cal_info,
        )

        data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 2500))
        trace = WaveformTrace(data=data, metadata=metadata)

        assert len(trace) == 2500
        assert trace.metadata.calibration is not None
        assert trace.metadata.calibration.instrument == "Tektronix MSO64"
        assert trace.metadata.calibration.probe_attenuation == 10.0


class TestRegulatoryComplianceScenarios:
    """Test scenarios for regulatory compliance (DOD, FDA, etc.)."""

    def test_dod_contractor_scenario(self):
        """Test metadata for DOD contractor with calibrated scope.

        Scenario: Using Tektronix scope with current calibration
        for measurements that will be audited.
        """
        # Use relative dates to ensure test doesn't fail due to time passing
        cal_date = datetime.now() - timedelta(days=180)  # 6 months ago
        due_date = datetime.now() + timedelta(days=180)  # 6 months from now

        cal_info = CalibrationInfo(
            instrument="Tektronix DPO7254C",
            serial_number="B040123",
            calibration_date=cal_date,
            calibration_due_date=due_date,
            calibration_lab="Tektronix Standards Lab",
            calibration_cert_number="TEK-CAL-2024-040123",
            firmware_version="FV:11.0.0",
            probe_attenuation=10.0,
            coupling="DC",
            bandwidth_limit=None,
            vertical_resolution=8,
        )

        metadata = TraceMetadata(
            sample_rate=2.5e9,
            vertical_scale=0.5,  # 500 mV/div
            vertical_offset=0.0,
            calibration=cal_info,
        )

        # Verify all required traceability fields are present
        assert metadata.calibration.calibration_date is not None
        assert metadata.calibration.calibration_due_date is not None
        assert metadata.calibration.calibration_cert_number is not None
        assert metadata.calibration.is_calibration_current is True

        # Verify traceability certificate is present
        assert metadata.calibration.calibration_cert_number == "TEK-CAL-2024-040123"

    def test_fda_regulated_scenario(self):
        """Test metadata for FDA-regulated medical device testing.

        Scenario: 21 CFR Part 11 compliance for electronic records.
        """
        # Use relative dates
        cal_date = datetime.now() - timedelta(days=120)  # 4 months ago
        due_date = datetime.now() + timedelta(days=245)  # ~8 months from now

        cal_info = CalibrationInfo(
            instrument="Keysight DSOX6004A",
            serial_number="MY54321234",
            calibration_date=cal_date,
            calibration_due_date=due_date,
            calibration_lab="Keysight Metrology Lab (ISO/IEC 17025)",
            calibration_cert_number="KS-2024-09-54321234",
            firmware_version="07.50.2021102800",
        )

        metadata = TraceMetadata(
            sample_rate=1e9,
            calibration=cal_info,
        )

        # Verify audit trail components
        assert metadata.calibration is not None

        # Verify calibration is current
        assert metadata.calibration.is_calibration_current is True

    def test_nist_traceability_chain(self):
        """Test metadata documenting NIST traceability.

        Scenario: Measurements requiring full NIST traceability chain.
        """
        cal_info = CalibrationInfo(
            instrument="Rohde & Schwarz RTO2044",
            serial_number="123456/789",
            calibration_date=datetime(2024, 8, 1),
            calibration_due_date=datetime(2025, 8, 1),
            calibration_lab="R&S Calibration Lab (NIST-traceable)",
            calibration_cert_number="RS-CAL-2024-123456",
        )

        metadata = TraceMetadata(
            sample_rate=20e9,  # 20 GSa/s
            calibration=cal_info,
        )

        # Document traceability
        assert metadata.calibration.calibration_cert_number == "RS-CAL-2024-123456"
        assert metadata.calibration.calibration_date == datetime(2024, 8, 1)


class TestCalibrationWarnings:
    """Test warnings for calibration issues."""

    def test_expired_calibration_warning(self):
        """Test that expired calibration is detectable."""
        cal_info = CalibrationInfo(
            instrument="Test Scope",
            calibration_date=datetime(2023, 1, 1),
            calibration_due_date=datetime(2024, 1, 1),  # Expired
        )

        # Should detect as expired
        assert cal_info.is_calibration_current is False

    def test_missing_calibration_info_allowed(self):
        """Test that missing calibration info is allowed (backward compatible)."""
        metadata = TraceMetadata(sample_rate=1e6)
        assert metadata.calibration is None

        # Should still be able to create trace
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=metadata)
        assert len(trace) == 3


class TestProbeAttenuation:
    """Test probe attenuation handling."""

    def test_probe_1x(self):
        """Test 1x probe (no attenuation)."""
        cal_info = CalibrationInfo(instrument="Test", probe_attenuation=1.0)
        assert cal_info.probe_attenuation == 1.0

    def test_probe_10x(self):
        """Test standard 10x probe."""
        cal_info = CalibrationInfo(instrument="Test", probe_attenuation=10.0)
        assert cal_info.probe_attenuation == 10.0

    def test_probe_100x(self):
        """Test 100x probe (high voltage)."""
        cal_info = CalibrationInfo(instrument="Test", probe_attenuation=100.0)
        assert cal_info.probe_attenuation == 100.0

    def test_probe_fractional(self):
        """Test fractional attenuation (e.g., 0.1x for gain)."""
        cal_info = CalibrationInfo(instrument="Test", probe_attenuation=0.1)
        assert cal_info.probe_attenuation == 0.1


class TestVerticalResolution:
    """Test ADC resolution tracking."""

    def test_8bit_adc(self):
        """Test 8-bit ADC (entry-level scopes)."""
        cal_info = CalibrationInfo(instrument="Test", vertical_resolution=8)
        assert cal_info.vertical_resolution == 8

    def test_12bit_adc(self):
        """Test 12-bit ADC (mid-range scopes)."""
        cal_info = CalibrationInfo(instrument="Test", vertical_resolution=12)
        assert cal_info.vertical_resolution == 12

    def test_16bit_adc(self):
        """Test 16-bit ADC (high-end digitizers)."""
        cal_info = CalibrationInfo(instrument="Test", vertical_resolution=16)
        assert cal_info.vertical_resolution == 16
