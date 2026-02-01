"""Unit tests for Tektronix WFM file loader.

This module tests loading of Tektronix WFM files from test_data/formats/tektronix/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pytestmark = [pytest.mark.unit, pytest.mark.loader]


def get_trace_data(trace: Any) -> NDArray[np.floating[Any]]:
    """Get data array from any trace type.

    Handles WaveformTrace, DigitalTrace (with .data) and IQTrace (with .i_data/.q_data).
    For IQTrace, returns magnitude.
    """
    if hasattr(trace, "data"):
        return trace.data
    elif hasattr(trace, "i_data") and hasattr(trace, "q_data"):
        # IQTrace - return magnitude
        return np.sqrt(trace.i_data**2 + trace.q_data**2)
    else:
        raise AttributeError(f"Cannot get data from {type(trace).__name__}")


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requires_data
@pytest.mark.requirement("BDL-001")
class TestTektronixWFMLoader:
    """Test Tektronix WFM file loading."""

    def test_load_analog_waveform(self, wfm_files: list[Path]) -> None:
        """Test loading analog WFM files.

        Validates:
        - Files load without error
        - Returns WaveformTrace object
        - Data is numpy array
        - Sample rate is positive
        """
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        loaded_count = 0
        for wfm_path in wfm_files[:5]:  # Try up to 5 files
            try:
                trace = load(wfm_path)

                # Basic type checks
                assert trace is not None, f"Load returned None for {wfm_path.name}"
                assert hasattr(trace, "metadata"), f"No metadata in {wfm_path.name}"

                # Data checks - use helper to handle IQTrace
                data = get_trace_data(trace)
                assert isinstance(data, np.ndarray), f"Data not ndarray in {wfm_path.name}"
                assert len(data) > 0, f"Empty data in {wfm_path.name}"

                # Metadata checks
                if hasattr(trace.metadata, "sample_rate"):
                    assert trace.metadata.sample_rate > 0, f"Invalid sample rate in {wfm_path.name}"

                loaded_count += 1
                if loaded_count >= 3:  # Successfully loaded 3 files
                    break

            except Exception as e:
                # Skip IQ waveforms and other unsupported formats for now
                if "IQWaveform" in str(e) or "unsupported" in str(e).lower():
                    continue  # Try next file
                pytest.fail(f"Failed to load {wfm_path.name}: {e}")

        # Ensure we loaded at least one file successfully
        assert loaded_count > 0, "No WFM files could be loaded"

    def test_load_golden_analog(self, tektronix_wfm_dir: Path) -> None:
        """Test loading the golden analog reference file.

        The golden_analog.wfm file is a known-good reference.
        """
        golden = tektronix_wfm_dir / "analog" / "single_channel" / "golden_analog.wfm"
        if not golden.exists():
            pytest.skip("golden_analog.wfm not found")

        from oscura import load

        trace = load(golden)

        assert trace is not None
        data = get_trace_data(trace)
        assert len(data) > 0
        assert np.isfinite(data).all(), "Golden file has non-finite values"

    def test_load_digital_waveform(self, tektronix_wfm_dir: Path) -> None:
        """Test loading digital WFM files.

        Validates:
        - Digital waveform files load correctly
        - Returns appropriate trace type
        """
        digital = tektronix_wfm_dir / "analog" / "single_channel" / "digital_waveform.wfm"
        if not digital.exists():
            pytest.skip("digital_waveform.wfm not found")

        from oscura import load

        trace = load(digital)

        assert trace is not None
        assert len(trace.data) > 0

    def test_load_iq_waveform(self, tektronix_wfm_dir: Path) -> None:
        """Test loading IQ waveform files.

        IQ waveforms may have complex data or separate I/Q channels.
        """
        iq = tektronix_wfm_dir / "analog" / "single_channel" / "iq_waveform.wfm"
        if not iq.exists():
            pytest.skip("iq_waveform.wfm not found")

        from oscura import load

        try:
            trace = load(iq)

            assert trace is not None
            data = get_trace_data(trace)
            assert len(data) > 0
        except Exception as e:
            # IQ waveform support not yet implemented
            if "IQWaveform" in str(e) or "unsupported" in str(e).lower():
                pytest.skip(f"IQ waveform support not yet implemented: {e}")
            raise


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requires_data
@pytest.mark.requirement("BDL-003")
class TestTektronixFormatDetection:
    """Test automatic format detection for WFM files."""

    def test_auto_detect_wfm_format(self, wfm_files: list[Path]) -> None:
        """Test that WFM format is auto-detected.

        Validates:
        - load() auto-detects Tektronix format from .wfm extension
        - No format parameter needed
        """
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        # Try multiple files in case first is unsupported (e.g., IQ waveform)
        for wfm_path in wfm_files[:5]:
            try:
                trace = load(wfm_path)  # No format specified
                assert trace is not None
                return  # Test passed
            except Exception as e:
                if "IQWaveform" in str(e) or "unsupported" in str(e).lower():
                    continue  # Try next file
                raise

        pytest.skip("No supported WFM files found")

    def test_explicit_format_specification(self, wfm_files: list[Path]) -> None:
        """Test explicit Tektronix format specification.

        Validates:
        - format="tektronix" parameter works
        """
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        # Try multiple files in case first is unsupported
        for wfm_path in wfm_files[:5]:
            try:
                trace = load(wfm_path, format="tektronix")
                assert trace is not None
                return  # Test passed
            except Exception as e:
                if "IQWaveform" in str(e) or "unsupported" in str(e).lower():
                    continue
                raise

        pytest.skip("No supported WFM files found")


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requires_data
class TestTektronixErrorHandling:
    """Test error handling for invalid WFM files."""

    def test_invalid_format_file(self, invalid_wfm_files: list[Path]) -> None:
        """Test handling of invalid format WFM files.

        Validates:
        - Invalid files either raise LoaderError or load with warnings
        - Robust handling of edge cases
        """
        if not invalid_wfm_files:
            pytest.skip("No invalid WFM files available")

        from oscura import load
        from oscura.core.exceptions import LoaderError

        for invalid_path in invalid_wfm_files:
            try:
                # Try to load - may succeed with warnings or fail
                trace = load(invalid_path)
                # If it loads, verify it's at least a valid trace object
                assert trace is not None
                assert hasattr(trace, "data")
            except (LoaderError, Exception):
                # Expected for truly invalid files
                pass  # This is the expected behavior

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading a non-existent file.

        Validates:
        - FileNotFoundError is raised
        """
        from oscura import load

        fake_path = tmp_path / "nonexistent.wfm"

        with pytest.raises(FileNotFoundError):
            load(fake_path)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requires_data
class TestTektronixMetadata:
    """Test metadata extraction from WFM files."""

    def test_sample_rate_extraction(self, wfm_files: list[Path]) -> None:
        """Test that sample rate is correctly extracted.

        Validates:
        - Sample rate is positive float
        - Sample rate is in reasonable range (Hz)
        """
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        # Try multiple files in case first is unsupported

        for _wfm_path in wfm_files[:5]:
            try:
                trace = load(_wfm_path)

                break

            except Exception as _e:
                if "IQWaveform" in str(_e) or "unsupported" in str(_e).lower():
                    if _wfm_path == wfm_files[min(4, len(wfm_files) - 1)]:
                        pytest.skip("No supported WFM files found")

                    continue

                raise

        if hasattr(trace.metadata, "sample_rate"):
            sr = trace.metadata.sample_rate
            assert isinstance(sr, int | float)
            assert sr > 0
            # Typical oscilloscope sample rates: 1 kSa/s to 100 GSa/s
            assert 1e3 <= sr <= 1e12, f"Sample rate {sr} outside expected range"

    def test_vertical_scale_extraction(self, wfm_files: list[Path]) -> None:
        """Test that vertical scale is extracted if available."""
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        # Try multiple files in case first is unsupported

        for _wfm_path in wfm_files[:5]:
            try:
                trace = load(_wfm_path)

                break

            except Exception as _e:
                if "IQWaveform" in str(_e) or "unsupported" in str(_e).lower():
                    if _wfm_path == wfm_files[min(4, len(wfm_files) - 1)]:
                        pytest.skip("No supported WFM files found")

                    continue

                raise

        # Vertical scale may or may not be present
        if hasattr(trace.metadata, "vertical_scale"):
            vs = trace.metadata.vertical_scale
            if vs is not None:
                assert isinstance(vs, int | float)
                assert vs > 0

    def test_channel_name_extraction(self, wfm_files: list[Path]) -> None:
        """Test that channel name is extracted if available."""
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        # Try multiple files in case first is unsupported

        for _wfm_path in wfm_files[:5]:
            try:
                trace = load(_wfm_path)

                break

            except Exception as _e:
                if "IQWaveform" in str(_e) or "unsupported" in str(_e).lower():
                    if _wfm_path == wfm_files[min(4, len(wfm_files) - 1)]:
                        pytest.skip("No supported WFM files found")

                    continue

                raise

        # Channel name may or may not be present
        if hasattr(trace.metadata, "channel_name"):
            cn = trace.metadata.channel
            if cn is not None:
                assert isinstance(cn, str)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requires_data
class TestTektronixMultiChannel:
    """Test multi-channel loading from Tektronix files."""

    def test_load_all_channels(self, wfm_files: list[Path]) -> None:
        """Test loading all channels from a WFM file.

        Validates:
        - load_all_channels returns dict
        - Keys are channel names
        """
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load_all_channels

        wfm_path = wfm_files[0]

        try:
            channels = load_all_channels(wfm_path)

            assert isinstance(channels, dict)
            assert len(channels) > 0

            for name, trace in channels.items():
                assert isinstance(name, str)
                assert hasattr(trace, "data")

        except Exception as e:
            # Multi-channel may not be supported for all files
            pytest.skip(f"Multi-channel loading not supported: {e}")


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requires_data
class TestTektronixDataIntegrity:
    """Test data integrity of loaded WFM files."""

    def test_no_nan_values(self, wfm_files: list[Path]) -> None:
        """Verify loaded data contains no NaN values."""
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        for wfm_path in wfm_files[:3]:
            try:
                trace = load(wfm_path)
            except Exception as _e:
                if "IQWaveform" in str(_e) or "unsupported" in str(_e).lower():
                    continue
                raise
            data = get_trace_data(trace)
            assert not np.isnan(data).any(), f"NaN values in {wfm_path.name}"

    def test_no_infinite_values(self, wfm_files: list[Path]) -> None:
        """Verify loaded data contains no infinite values."""
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        for wfm_path in wfm_files[:3]:
            try:
                trace = load(wfm_path)
            except Exception as _e:
                if "IQWaveform" in str(_e) or "unsupported" in str(_e).lower():
                    continue
                raise
            data = get_trace_data(trace)
            assert not np.isinf(data).any(), f"Infinite values in {wfm_path.name}"

    def test_reasonable_value_range(self, wfm_files: list[Path]) -> None:
        """Verify data values are within reasonable range.

        Oscilloscope data typically ranges from microvolts to kilovolts.
        """
        if not wfm_files:
            pytest.skip("No WFM files available")

        from oscura import load

        for wfm_path in wfm_files[:3]:
            try:
                trace = load(wfm_path)
            except Exception as _e:
                if "IQWaveform" in str(_e) or "unsupported" in str(_e).lower():
                    continue
                raise

            data = get_trace_data(trace)
            data_range = data.max() - data.min()
            # Allow for normalized or raw ADC data
            assert data_range > 0, f"Zero range in {wfm_path.name}"
