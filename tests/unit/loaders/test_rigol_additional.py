"""Additional tests for Rigol WFM loader to improve coverage.

This file adds tests for code paths not covered by existing tests,
particularly focusing on the basic loader path and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.rigol import (
    _load_basic,
    load_rigol_wfm,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestRigolBasicLoaderEdgeCases:
    """Test edge cases for basic Rigol WFM loader."""

    def test_load_basic_with_exactly_256_bytes(self, tmp_path: Path) -> None:
        """Test file with exactly 256 bytes (header only, no data)."""
        wfm_path = tmp_path / "header_only.wfm"
        header = b"\x00" * 256
        wfm_path.write_bytes(header)

        with pytest.raises(FormatError, match="No waveform data"):
            _load_basic(wfm_path, channel=0)

    def test_load_basic_with_255_bytes(self, tmp_path: Path) -> None:
        """Test file with less than 256 bytes."""
        wfm_path = tmp_path / "too_small.wfm"
        wfm_path.write_bytes(b"\x00" * 255)

        with pytest.raises(FormatError, match="File too small"):
            _load_basic(wfm_path, channel=0)

    def test_load_basic_int8_fallback_on_odd_bytes(self, tmp_path: Path) -> None:
        """Test that odd number of data bytes triggers int8 fallback."""
        wfm_path = tmp_path / "odd_data.wfm"
        header = b"\x00" * 256
        # 5 bytes of data (odd number - can't be int16)
        data = np.array([100, -50, 75, -100, 127], dtype=np.int8)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert len(trace.data) == 5
        assert trace.data.dtype == np.float64
        # Verify normalization for int8 (divide by 128.0)
        expected = data.astype(np.float64) / 128.0
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_load_basic_various_data_sizes(self, tmp_path: Path) -> None:
        """Test loading files with various data sizes."""
        for n_samples in [1, 10, 100, 1000, 10000]:
            wfm_path = tmp_path / f"size_{n_samples}.wfm"
            header = b"\x00" * 256
            data = np.random.randint(-32768, 32767, n_samples, dtype=np.int16)
            wfm_path.write_bytes(header + data.tobytes())

            trace = _load_basic(wfm_path, channel=0)

            assert len(trace.data) == n_samples
            assert trace.metadata.sample_rate == 1e6
            assert trace.data.dtype == np.float64

    def test_load_basic_alternating_values(self, tmp_path: Path) -> None:
        """Test with alternating min/max values."""
        wfm_path = tmp_path / "alternating.wfm"
        header = b"\x00" * 256
        data = np.array([32767, -32768] * 50, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert len(trace.data) == 100
        # Values should alternate between ~1.0 and ~-1.0
        assert np.all(np.abs(trace.data) >= 0.99)

    def test_load_basic_metadata_fields_complete(self, tmp_path: Path) -> None:
        """Test that all metadata fields are populated correctly."""
        wfm_path = tmp_path / "metadata_test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        for channel in [0, 1, 2, 3]:
            trace = _load_basic(wfm_path, channel=channel)

            assert trace.metadata.sample_rate == 1e6
            assert trace.metadata.vertical_scale is None
            assert trace.metadata.vertical_offset is None
            assert trace.metadata.source_file == str(wfm_path)
            assert trace.metadata.channel == f"CH{channel + 1}"

    def test_load_basic_exception_during_parse(self, tmp_path: Path) -> None:
        """Test that parsing exceptions are wrapped in LoaderError."""
        wfm_path = tmp_path / "parse_error.wfm"
        header = b"\x00" * 256
        data = b"some data"
        wfm_path.write_bytes(header + data)

        # Patch np.frombuffer to raise an exception
        with patch("numpy.frombuffer") as mock_frombuffer:
            mock_frombuffer.side_effect = Exception("Unexpected parse error")

            with pytest.raises(LoaderError, match="Failed to parse"):
                _load_basic(wfm_path, channel=0)


class TestRigolDispatch:
    """Test the dispatch logic between RigolWFM and basic loader."""

    def test_uses_basic_loader_when_rigolwfm_unavailable(self, tmp_path: Path) -> None:
        """Test that basic loader is used when RigolWFM is not available."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        # Should use basic loader
        trace = load_rigol_wfm(wfm_path, channel=0)

        assert trace is not None
        assert len(trace.data) == 3
        # Basic loader defaults
        assert trace.metadata.sample_rate == 1e6

    def test_file_path_conversion(self, tmp_path: Path) -> None:
        """Test that both string and Path inputs are handled."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        # Test with Path object
        trace1 = load_rigol_wfm(wfm_path, channel=0)
        assert trace1 is not None

        # Test with string
        trace2 = load_rigol_wfm(str(wfm_path), channel=0)
        assert trace2 is not None

        # Should produce identical results
        np.testing.assert_array_equal(trace1.data, trace2.data)


class TestRigolErrorHandling:
    """Test error handling in Rigol loader."""

    def test_nonexistent_file_error(self, tmp_path: Path) -> None:
        """Test that missing file raises LoaderError."""
        fake_path = tmp_path / "does_not_exist.wfm"

        with pytest.raises(LoaderError, match="File not found"):
            load_rigol_wfm(fake_path)

    def test_file_path_in_error_message(self, tmp_path: Path) -> None:
        """Test that error messages include file path."""
        fake_path = tmp_path / "missing.wfm"

        try:
            load_rigol_wfm(fake_path)
            pytest.fail("Should have raised LoaderError")
        except LoaderError as e:
            assert str(fake_path) in str(e) or "missing.wfm" in str(e)

    def test_permission_denied_error(self, tmp_path: Path) -> None:
        """Test that permission errors are wrapped correctly."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        # Simulate permission error
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = OSError("Permission denied")

            with pytest.raises(LoaderError, match="Failed to read"):
                _load_basic(wfm_path, channel=0)


class TestRigolDataValidation:
    """Test data validation and integrity."""

    def test_output_always_float64(self, tmp_path: Path) -> None:
        """Test that output data is always float64."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256

        # Test with int16 data
        data_int16 = np.array([100, 200, 300], dtype=np.int16)
        wfm_path.write_bytes(header + data_int16.tobytes())
        trace = _load_basic(wfm_path, channel=0)
        assert trace.data.dtype == np.float64

        # Test with int8 data (odd bytes)
        data_int8 = np.array([10, 20, 30, 40, 50], dtype=np.int8)
        wfm_path.write_bytes(header + data_int8.tobytes())
        trace = _load_basic(wfm_path, channel=0)
        assert trace.data.dtype == np.float64

    def test_normalization_preserves_sign(self, tmp_path: Path) -> None:
        """Test that normalization preserves sign of values."""
        wfm_path = tmp_path / "signed.wfm"
        header = b"\x00" * 256
        data = np.array([1000, -1000, 2000, -2000, 3000], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        # Check signs are preserved
        assert trace.data[0] > 0
        assert trace.data[1] < 0
        assert trace.data[2] > 0
        assert trace.data[3] < 0
        assert trace.data[4] > 0

    def test_zero_data_handling(self, tmp_path: Path) -> None:
        """Test handling of all-zero data."""
        wfm_path = tmp_path / "zeros.wfm"
        header = b"\x00" * 256
        data = np.zeros(100, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert len(trace.data) == 100
        assert np.allclose(trace.data, 0.0)


class TestRigolIntegration:
    """Integration tests for Rigol loader with synthetic files."""

    def test_load_multiple_channels_from_same_file(self, tmp_path: Path) -> None:
        """Test loading different channels from the same file."""
        wfm_path = tmp_path / "multi.wfm"
        header = b"\x00" * 256
        data = np.arange(1000, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        # Load as different channels - should gracefully fall back to basic loader
        traces = []
        for ch in range(4):
            trace = load_rigol_wfm(wfm_path, channel=ch)
            traces.append(trace)
            assert trace.metadata.channel == f"CH{ch + 1}"

        # All should have same data (basic loader doesn't split channels)
        for i in range(1, 4):
            np.testing.assert_array_equal(traces[0].data, traces[i].data)

    def test_realistic_waveform_data(self, tmp_path: Path) -> None:
        """Test with realistic oscilloscope waveform data."""
        wfm_path = tmp_path / "realistic.wfm"
        header = b"\x00" * 256

        # Simulate a sine wave captured by oscilloscope
        t = np.linspace(0, 1, 1000)
        sine_wave = (np.sin(2 * np.pi * 10 * t) * 32767 * 0.8).astype(np.int16)
        wfm_path.write_bytes(header + sine_wave.tobytes())

        # Should gracefully handle synthetic file by falling back to basic loader
        trace = load_rigol_wfm(wfm_path, channel=0)

        assert len(trace.data) == 1000
        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0
        # Check it's roughly sinusoidal
        assert np.max(trace.data) > 0.7  # Peak should be near 0.8
        assert np.min(trace.data) < -0.7


class TestRigolPublicAPI:
    """Test the public API of the rigol module."""

    def test_module_exports_only_public_function(self) -> None:
        """Test that __all__ exports only load_rigol_wfm."""
        from oscura.loaders import rigol

        assert hasattr(rigol, "__all__")
        assert "load_rigol_wfm" in rigol.__all__
        assert len(rigol.__all__) == 1

    def test_load_rigol_wfm_signature(self) -> None:
        """Test function signature."""
        import inspect

        sig = inspect.signature(load_rigol_wfm)
        params = sig.parameters

        assert "path" in params
        assert "channel" in params
        assert params["channel"].default == 0

    def test_load_rigol_wfm_has_docstring(self) -> None:
        """Test that function has comprehensive docstring."""
        assert load_rigol_wfm.__doc__ is not None
        assert len(load_rigol_wfm.__doc__) > 100
        assert "Rigol" in load_rigol_wfm.__doc__
        assert "WFM" in load_rigol_wfm.__doc__
