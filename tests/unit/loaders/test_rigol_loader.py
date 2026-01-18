"""Unit tests for Rigol WFM file loader.

Tests LOAD-003: Rigol WFM Loader
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import WaveformTrace
from oscura.loaders.rigol import (
    RIGOL_WFM_AVAILABLE,
    _extract_trigger_info,
    _load_basic,
    _load_with_rigolwfm,
    load_rigol_wfm,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestLoadRigolWFM:
    """Test Rigol WFM file loading with load_rigol_wfm function."""

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading a non-existent file raises LoaderError."""
        fake_path = tmp_path / "nonexistent.wfm"

        with pytest.raises(LoaderError, match="File not found"):
            load_rigol_wfm(fake_path)

    def test_load_file_path_as_string(self, tmp_path: Path) -> None:
        """Test that path can be passed as string."""
        # Create a minimal valid file
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        # Pass as string
        trace = load_rigol_wfm(str(wfm_path))
        assert trace is not None
        assert isinstance(trace, WaveformTrace)

    def test_load_file_path_as_pathlike(self, tmp_path: Path) -> None:
        """Test that path can be passed as PathLike."""
        # Create a minimal valid file
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        # Pass as Path object
        trace = load_rigol_wfm(wfm_path)
        assert trace is not None
        assert isinstance(trace, WaveformTrace)

    def test_load_with_channel_parameter(self, tmp_path: Path) -> None:
        """Test loading with channel parameter."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = load_rigol_wfm(wfm_path, channel=2)
        assert trace is not None
        # Channel parameter affects metadata
        assert trace.metadata.channel_name == "CH3"  # channel=2 -> CH3

    @pytest.mark.skipif(not RIGOL_WFM_AVAILABLE, reason="rigol-wfm library not available")
    def test_load_with_rigolwfm_library(self, tmp_path: Path) -> None:
        """Test that RigolWFM library is used when available."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        with patch("oscura.loaders.rigol._load_with_rigolwfm") as mock_rigol:
            mock_trace = Mock(spec=WaveformTrace)
            mock_rigol.return_value = mock_trace

            result = load_rigol_wfm(wfm_path)

            assert result is mock_trace
            mock_rigol.assert_called_once()

    def test_load_without_rigolwfm_library(self, tmp_path: Path) -> None:
        """Test that basic loader is used when RigolWFM unavailable."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        with patch("oscura.loaders.rigol._load_basic") as mock_basic:
            mock_trace = Mock(spec=WaveformTrace)
            mock_basic.return_value = mock_trace

            result = load_rigol_wfm(wfm_path)

            assert result == mock_trace
            mock_basic.assert_called_once()


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestLoadBasic:
    """Test basic Rigol WFM loader (without RigolWFM library)."""

    def test_load_minimal_file(self, tmp_path: Path) -> None:
        """Test loading minimal valid WFM file."""
        wfm_path = tmp_path / "minimal.wfm"
        header = b"\x00" * 256
        # Create simple waveform data (int16)
        data = np.array([100, -200, 300, -400, 500], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert trace is not None
        assert isinstance(trace, WaveformTrace)
        assert len(trace.data) == 5
        assert trace.metadata.sample_rate == 1e6  # Default
        assert trace.metadata.channel_name == "CH1"
        assert trace.metadata.source_file == str(wfm_path)

    def test_load_with_channel_parameter(self, tmp_path: Path) -> None:
        """Test channel parameter affects metadata."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=3)
        assert trace.metadata.channel_name == "CH4"  # channel=3 -> CH4

    def test_load_int16_data_normalization(self, tmp_path: Path) -> None:
        """Test that int16 data is normalized to [-1, 1] range."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        # Max positive int16
        data = np.array([32767, 0, -32768], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert trace.data[0] == pytest.approx(1.0, abs=0.01)
        assert trace.data[1] == pytest.approx(0.0, abs=0.01)
        assert trace.data[2] == pytest.approx(-1.0, abs=0.01)

    def test_load_int8_data_fallback(self, tmp_path: Path) -> None:
        """Test fallback to int8 when int16 fails."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        # Odd number of bytes - can't be int16
        data = np.array([127, 0, -128], dtype=np.int8)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert len(trace.data) == 3
        assert trace.data[0] == pytest.approx(127 / 128.0, abs=0.01)

    def test_load_file_too_small(self, tmp_path: Path) -> None:
        """Test that file smaller than 256 bytes raises FormatError."""
        wfm_path = tmp_path / "tiny.wfm"
        wfm_path.write_bytes(b"too small")

        with pytest.raises(FormatError, match="File too small"):
            _load_basic(wfm_path, channel=0)

    def test_load_no_waveform_data(self, tmp_path: Path) -> None:
        """Test that file with only header raises FormatError."""
        wfm_path = tmp_path / "header_only.wfm"
        header = b"\x00" * 256
        wfm_path.write_bytes(header)

        with pytest.raises(FormatError, match="No waveform data"):
            _load_basic(wfm_path, channel=0)

    def test_load_io_error(self, tmp_path: Path) -> None:
        """Test that I/O errors are wrapped in LoaderError."""
        wfm_path = tmp_path / "test.wfm"
        # Create file then make it unreadable
        wfm_path.write_bytes(b"\x00" * 300)

        with patch("builtins.open", side_effect=OSError("Permission denied")):
            with pytest.raises(LoaderError, match="Failed to read"):
                _load_basic(wfm_path, channel=0)

    def test_metadata_fields(self, tmp_path: Path) -> None:
        """Test that all expected metadata fields are populated."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=1)

        assert trace.metadata.sample_rate == 1e6
        assert trace.metadata.vertical_scale is None
        assert trace.metadata.vertical_offset is None
        assert trace.metadata.source_file == str(wfm_path)
        assert trace.metadata.channel_name == "CH2"

    def test_large_data_file(self, tmp_path: Path) -> None:
        """Test loading file with large waveform data."""
        wfm_path = tmp_path / "large.wfm"
        header = b"\x00" * 256
        # 10000 samples
        data = np.random.randint(-32768, 32767, 10000, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert len(trace.data) == 10000
        assert np.all(np.abs(trace.data) <= 1.0)  # All normalized


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
@pytest.mark.skipif(not RIGOL_WFM_AVAILABLE, reason="rigol-wfm library not available")
class TestLoadWithRigolWFM:
    """Test Rigol WFM loader with RigolWFM library."""

    def create_mock_wfm_single_channel(
        self,
        volts: list[float] | None = None,
        sample_rate: float = 1e6,
        volts_per_div: float | None = 1.0,
        volt_offset: float | None = 0.0,
    ) -> MagicMock:
        """Create mock WFM object for single channel format.

        Args:
            volts: Voltage data array.
            sample_rate: Sample rate in Hz.
            volts_per_div: Vertical scale.
            volt_offset: Vertical offset.

        Returns:
            Mock WFM object.
        """
        if volts is None:
            volts = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Create Mock without channels attribute (single channel format)
        mock_wfm = Mock(spec=["volts", "sample_rate", "volts_per_div", "volt_offset"])
        mock_wfm.volts = volts
        mock_wfm.sample_rate = sample_rate
        mock_wfm.volts_per_div = volts_per_div
        mock_wfm.volt_offset = volt_offset
        return mock_wfm

    def create_mock_wfm_multi_channel(
        self,
        n_channels: int = 2,
        volts_per_channel: list[list[float]] | None = None,
        sample_rate: float = 1e6,
    ) -> MagicMock:
        """Create mock WFM object for multi-channel format.

        Args:
            n_channels: Number of channels.
            volts_per_channel: List of voltage arrays for each channel.
            sample_rate: Sample rate in Hz.

        Returns:
            Mock WFM object.
        """
        if volts_per_channel is None:
            volts_per_channel = [[0.1, 0.2, 0.3]] * n_channels

        mock_wfm = MagicMock()
        mock_wfm.sample_rate = sample_rate

        channels = []
        for i, volts in enumerate(volts_per_channel):
            channel = MagicMock()
            channel.volts = volts
            channel.volts_per_div = 1.0 + i * 0.1
            channel.volt_offset = i * 0.05
            channels.append(channel)

        mock_wfm.channels = channels
        return mock_wfm

    def test_load_single_channel_format(self, tmp_path: Path) -> None:
        """Test loading single channel WFM format."""
        wfm_path = tmp_path / "single.wfm"
        wfm_path.write_bytes(b"dummy")  # File must exist

        mock_wfm = self.create_mock_wfm_single_channel(
            volts=[1.0, 2.0, 3.0],
            sample_rate=2e6,
            volts_per_div=2.5,
            volt_offset=0.5,
        )

        with patch("oscura.loaders.rigol.rigol_wfm.Wfm.from_file") as mock_from_file:
            mock_from_file.return_value = mock_wfm

            trace = _load_with_rigolwfm(wfm_path, channel=0)

            assert len(trace.data) == 3
            assert np.allclose(trace.data, [1.0, 2.0, 3.0])
            assert trace.metadata.sample_rate == 2e6
            assert trace.metadata.vertical_scale == 2.5
            assert trace.metadata.vertical_offset == 0.5
            assert trace.metadata.channel_name == "CH1"

    def test_load_multi_channel_format_channel_0(self, tmp_path: Path) -> None:
        """Test loading multi-channel WFM format, channel 0."""
        wfm_path = tmp_path / "multi.wfm"
        wfm_path.write_bytes(b"dummy")

        mock_wfm = self.create_mock_wfm_multi_channel(
            n_channels=2,
            volts_per_channel=[[1.0, 2.0], [3.0, 4.0]],
        )

        with patch("oscura.loaders.rigol.rigol_wfm.Wfm.from_file") as mock_from_file:
            mock_from_file.return_value = mock_wfm

            trace = _load_with_rigolwfm(wfm_path, channel=0)

            assert np.allclose(trace.data, [1.0, 2.0])
            assert trace.metadata.channel_name == "CH1"

    def test_load_multi_channel_format_channel_1(self, tmp_path: Path) -> None:
        """Test loading multi-channel WFM format, channel 1."""
        wfm_path = tmp_path / "multi.wfm"
        wfm_path.write_bytes(b"dummy")

        mock_wfm = self.create_mock_wfm_multi_channel(
            n_channels=2,
            volts_per_channel=[[1.0, 2.0], [3.0, 4.0]],
        )

        with patch("oscura.loaders.rigol.rigol_wfm.Wfm.from_file") as mock_from_file:
            mock_from_file.return_value = mock_wfm

            trace = _load_with_rigolwfm(wfm_path, channel=1)

            assert np.allclose(trace.data, [3.0, 4.0])
            assert trace.metadata.channel_name == "CH2"

    def test_load_no_waveform_data_raises_error(self, tmp_path: Path) -> None:
        """Test that WFM with no data raises FormatError."""
        wfm_path = tmp_path / "empty.wfm"
        wfm_path.write_bytes(b"dummy")

        # Create Mock with no data attributes
        mock_wfm = Mock(spec=[])

        with patch("oscura.loaders.rigol.rigol_wfm.Wfm.from_file") as mock_from_file:
            mock_from_file.return_value = mock_wfm

            with pytest.raises(FormatError, match="No waveform data found"):
                _load_with_rigolwfm(wfm_path, channel=0)

    def test_load_metadata_defaults_when_missing(self, tmp_path: Path) -> None:
        """Test that missing metadata fields get default values."""
        wfm_path = tmp_path / "minimal.wfm"
        wfm_path.write_bytes(b"dummy")

        # Create Mock with only volts attribute (missing optional metadata)
        mock_wfm = Mock(spec=["volts"])
        mock_wfm.volts = [1.0, 2.0, 3.0]

        with patch("oscura.loaders.rigol.rigol_wfm.Wfm.from_file") as mock_from_file:
            mock_from_file.return_value = mock_wfm

            trace = _load_with_rigolwfm(wfm_path, channel=0)

            assert trace.metadata.sample_rate == 1e6  # Default
            assert trace.metadata.vertical_scale is None
            assert trace.metadata.vertical_offset is None

    def test_load_trigger_info_extracted(self, tmp_path: Path) -> None:
        """Test that trigger info is extracted when available."""
        wfm_path = tmp_path / "trigger.wfm"
        wfm_path.write_bytes(b"dummy")

        mock_wfm = self.create_mock_wfm_single_channel()
        mock_wfm.trigger_level = 0.5
        mock_wfm.trigger_mode = "edge"
        mock_wfm.trigger_source = "CH1"

        with patch("oscura.loaders.rigol.rigol_wfm.Wfm.from_file") as mock_from_file:
            mock_from_file.return_value = mock_wfm

            trace = _load_with_rigolwfm(wfm_path, channel=0)

            assert trace.metadata.trigger_info is not None
            assert trace.metadata.trigger_info["level"] == 0.5
            assert trace.metadata.trigger_info["mode"] == "edge"
            assert trace.metadata.trigger_info["source"] == "CH1"

    def test_load_exception_handling(self, tmp_path: Path) -> None:
        """Test that exceptions are wrapped in LoaderError."""
        wfm_path = tmp_path / "error.wfm"
        wfm_path.write_bytes(b"dummy")

        with patch(
            "oscura.loaders.rigol.rigol_wfm.Wfm.from_file",
            side_effect=Exception("Parse error"),
        ):
            with pytest.raises(LoaderError, match="Failed to load"):
                _load_with_rigolwfm(wfm_path, channel=0)

    def test_load_preserves_loader_error(self, tmp_path: Path) -> None:
        """Test that LoaderError is re-raised without wrapping."""
        wfm_path = tmp_path / "error.wfm"
        wfm_path.write_bytes(b"dummy")

        original_error = LoaderError("Original error", file_path=str(wfm_path))

        with patch(
            "oscura.loaders.rigol.rigol_wfm.Wfm.from_file",
            side_effect=original_error,
        ):
            with pytest.raises(LoaderError, match="Original error"):
                _load_with_rigolwfm(wfm_path, channel=0)

    def test_load_preserves_format_error(self, tmp_path: Path) -> None:
        """Test that FormatError is re-raised without wrapping."""
        wfm_path = tmp_path / "error.wfm"
        wfm_path.write_bytes(b"dummy")

        original_error = FormatError("Bad format", file_path=str(wfm_path))

        with patch(
            "oscura.loaders.rigol.rigol_wfm.Wfm.from_file",
            side_effect=original_error,
        ):
            with pytest.raises(FormatError, match="Bad format"):
                _load_with_rigolwfm(wfm_path, channel=0)

    def test_data_type_conversion_to_float64(self, tmp_path: Path) -> None:
        """Test that data is converted to float64."""
        wfm_path = tmp_path / "test.wfm"
        wfm_path.write_bytes(b"dummy")

        # Return integer data
        mock_wfm = self.create_mock_wfm_single_channel(volts=[1, 2, 3])

        with patch("oscura.loaders.rigol.rigol_wfm.Wfm.from_file") as mock_from_file:
            mock_from_file.return_value = mock_wfm

            trace = _load_with_rigolwfm(wfm_path, channel=0)

            assert trace.data.dtype == np.float64


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestExtractTriggerInfo:
    """Test trigger information extraction."""

    def test_extract_all_trigger_fields(self) -> None:
        """Test extracting all available trigger fields."""
        mock_wfm = MagicMock()
        mock_wfm.trigger_level = 1.5
        mock_wfm.trigger_mode = "normal"
        mock_wfm.trigger_source = "CH2"

        result = _extract_trigger_info(mock_wfm)

        assert result is not None
        assert result["level"] == 1.5
        assert result["mode"] == "normal"
        assert result["source"] == "CH2"

    def test_extract_partial_trigger_fields(self) -> None:
        """Test extracting when only some fields are available."""

        # Create a simple object with only trigger_level
        class PartialTriggerWfm:
            trigger_level = 2.0

        result = _extract_trigger_info(PartialTriggerWfm())

        assert result is not None
        assert result["level"] == 2.0
        assert "mode" not in result
        assert "source" not in result

    def test_extract_no_trigger_fields(self) -> None:
        """Test extracting when no trigger fields are available."""

        # Create a simple object with no trigger fields
        class NoTriggerWfm:
            pass

        result = _extract_trigger_info(NoTriggerWfm())

        assert result is None

    def test_extract_trigger_with_none_values(self) -> None:
        """Test extracting trigger fields with None values."""
        mock_wfm = MagicMock()
        mock_wfm.trigger_level = None
        mock_wfm.trigger_mode = "auto"
        mock_wfm.trigger_source = None

        result = _extract_trigger_info(mock_wfm)

        assert result is not None
        assert result["level"] is None
        assert result["mode"] == "auto"
        assert result["source"] is None


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestRigolWFMAvailability:
    """Test RigolWFM availability flag."""

    def test_rigol_wfm_availability_is_boolean(self) -> None:
        """Test that RIGOL_WFM_AVAILABLE is a boolean."""
        assert isinstance(RIGOL_WFM_AVAILABLE, bool)

    @pytest.mark.skipif(not RIGOL_WFM_AVAILABLE, reason="rigol-wfm library not available")
    def test_rigol_wfm_library_importable(self) -> None:
        """Test that RigolWFM can be imported when available."""
        try:
            import RigolWFM.wfm  # type: ignore[import-not-found]

            assert RigolWFM.wfm is not None
        except ImportError:
            pytest.fail("RigolWFM marked available but cannot import")


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestRigolDataIntegrity:
    """Test data integrity of loaded Rigol files."""

    def test_no_nan_values_basic(self, tmp_path: Path) -> None:
        """Verify basic loader produces no NaN values."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert not np.isnan(trace.data).any()

    def test_no_infinite_values_basic(self, tmp_path: Path) -> None:
        """Verify basic loader produces no infinite values."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert not np.isinf(trace.data).any()

    def test_reasonable_value_range_basic(self, tmp_path: Path) -> None:
        """Verify basic loader produces values in reasonable range."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        # Data should be normalized to approximately [-1, 1]
        assert np.max(trace.data) <= 1.1  # Allow small margin
        assert np.min(trace.data) >= -1.1

    def test_data_shape_basic(self, tmp_path: Path) -> None:
        """Verify basic loader produces 1D array."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.random.randint(-32768, 32767, 100, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert trace.data.ndim == 1
        assert len(trace.data) == 100


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestRigolEdgeCases:
    """Test edge cases for Rigol loader."""

    def test_empty_header(self, tmp_path: Path) -> None:
        """Test file with empty header bytes."""
        wfm_path = tmp_path / "empty_header.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)
        assert trace is not None

    def test_single_sample(self, tmp_path: Path) -> None:
        """Test file with only one sample."""
        wfm_path = tmp_path / "single.wfm"
        header = b"\x00" * 256
        data = np.array([42], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)
        assert len(trace.data) == 1

    def test_very_large_channel_number(self, tmp_path: Path) -> None:
        """Test with very large channel number."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=99)
        assert trace.metadata.channel_name == "CH100"

    def test_zero_values(self, tmp_path: Path) -> None:
        """Test file with all zero values."""
        wfm_path = tmp_path / "zeros.wfm"
        header = b"\x00" * 256
        data = np.zeros(100, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)
        assert np.allclose(trace.data, 0.0)

    def test_max_int16_values(self, tmp_path: Path) -> None:
        """Test file with maximum int16 values."""
        wfm_path = tmp_path / "max.wfm"
        header = b"\x00" * 256
        data = np.array([32767] * 10, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)
        assert np.all(trace.data <= 1.0)
        assert np.all(trace.data >= 0.99)  # Should be close to 1.0

    def test_min_int16_values(self, tmp_path: Path) -> None:
        """Test file with minimum int16 values."""
        wfm_path = tmp_path / "min.wfm"
        header = b"\x00" * 256
        data = np.array([-32768] * 10, dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)
        assert np.all(trace.data >= -1.0)
        assert np.all(trace.data <= -0.99)  # Should be close to -1.0


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestRigolMetadata:
    """Test metadata extraction and handling."""

    def test_source_file_absolute_path(self, tmp_path: Path) -> None:
        """Test that source_file contains absolute path."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert trace.metadata.source_file == str(wfm_path)
        assert Path(trace.metadata.source_file).is_absolute()

    def test_metadata_sample_rate_type(self, tmp_path: Path) -> None:
        """Test that sample_rate is float type."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        assert isinstance(trace.metadata.sample_rate, float)

    def test_metadata_time_base_calculation(self, tmp_path: Path) -> None:
        """Test that time_base is correctly derived from sample_rate."""
        wfm_path = tmp_path / "test.wfm"
        header = b"\x00" * 256
        data = np.array([1, 2, 3], dtype=np.int16)
        wfm_path.write_bytes(header + data.tobytes())

        trace = _load_basic(wfm_path, channel=0)

        expected_time_base = 1.0 / trace.metadata.sample_rate
        assert trace.metadata.time_base == pytest.approx(expected_time_base)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-003")
class TestRigolPublicAPI:
    """Test the public API of the rigol module."""

    def test_module_exports(self) -> None:
        """Test that only public API is exported."""
        from oscura.loaders import rigol

        assert "load_rigol_wfm" in rigol.__all__
        assert len(rigol.__all__) == 1  # Only public function

    def test_function_signatures(self) -> None:
        """Test that function signatures are correct."""
        import inspect

        sig = inspect.signature(load_rigol_wfm)
        params = sig.parameters

        assert "path" in params
        assert "channel" in params
        assert params["channel"].default == 0

    def test_docstring_present(self) -> None:
        """Test that public functions have docstrings."""
        assert load_rigol_wfm.__doc__ is not None
        assert len(load_rigol_wfm.__doc__) > 50  # Meaningful docstring
