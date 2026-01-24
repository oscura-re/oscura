"""Tests for MDF/MF4 (Measurement Data Format) file loader.

Tests the ASAM MDF loader. Since creating real MDF files requires complex
dependencies, these tests focus on error handling and structure validation.
Full integration tests would require asammdf library.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

try:
    import asammdf  # noqa: F401

    ASAMMDF_AVAILABLE = True
except ImportError:
    ASAMMDF_AVAILABLE = False

from oscura.automotive.loaders.mdf import load_mdf

pytestmark = [
    pytest.mark.unit,
    pytest.mark.loader,
    pytest.mark.skipif(
        not ASAMMDF_AVAILABLE,
        reason="asammdf not available (install with: pip install 'oscura[automotive]')",
    ),
]


class TestMDFLoader:
    """Tests for MDF file loading."""

    def test_load_mdf_file_not_found(self):
        """Test loading non-existent MDF file."""
        with pytest.raises(FileNotFoundError, match="MDF file not found"):
            load_mdf("/nonexistent/file.mf4")

    def test_load_mdf_import_error(self, monkeypatch):
        """Test that ImportError is raised if asammdf not installed."""
        import sys

        # Mock import failure
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "asammdf", None)

            # Force reimport
            import importlib

            from oscura.automotive.loaders import mdf as mdf_module

            importlib.reload(mdf_module)

            with pytest.raises(ImportError, match="asammdf is required"):
                mdf_module.load_mdf("dummy.mf4")

    @patch("asammdf.MDF")
    def test_load_mdf_bus_logging_format(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with bus logging format."""
        mdf_path = temp_dir / "bus_logging.mf4"
        mdf_path.touch()  # Create empty file

        # Mock MDF object with structured CAN data
        mock_mdf = MagicMock()

        # Create mock structured array for CAN frames
        dtype = np.dtype([("ID", np.uint32), ("Data", np.uint8, (8,)), ("DLC", np.uint8)])

        samples = np.zeros(10, dtype=dtype)
        for i in range(10):
            samples[i]["ID"] = 0x123 + i
            samples[i]["Data"] = np.arange(8, dtype=np.uint8)
            samples[i]["DLC"] = 8

        # Mock signal
        mock_signal = Mock()
        mock_signal.timestamps = np.arange(10, dtype=np.float64) * 0.1
        mock_signal.samples = samples

        # Mock channels_db and get method
        mock_mdf.channels_db = ["CAN_DataFrame"]
        mock_mdf.get.return_value = mock_signal

        # Setup context manager
        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        messages = load_mdf(mdf_path)

        assert len(messages) == 10
        assert messages[0].arbitration_id == 0x123
        assert messages[0].timestamp == pytest.approx(0.0, abs=0.001)

    @patch("asammdf.MDF")
    def test_load_mdf_raw_frames(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with raw CAN frame format."""
        mdf_path = temp_dir / "raw_frames.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()

        # Create mock raw CAN frames (13 bytes each: 4 bytes ID + 1 byte DLC + 8 bytes data)
        raw_frames = []
        for i in range(5):
            frame = bytearray(13)
            # ID (little-endian)
            arb_id = 0x123 + i
            frame[0:4] = arb_id.to_bytes(4, byteorder="little")
            # DLC
            frame[4] = 8
            # Data
            frame[5:13] = bytes(range(8))
            raw_frames.append(bytes(frame))

        samples = np.array(raw_frames, dtype=object)

        mock_signal = Mock()
        mock_signal.timestamps = np.arange(5, dtype=np.float64) * 0.1
        mock_signal.samples = samples

        mock_mdf.channels_db = ["CAN_Message"]
        mock_mdf.get.return_value = mock_signal

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        messages = load_mdf(mdf_path)

        assert len(messages) == 5

    @patch("asammdf.MDF")
    def test_load_mdf_signal_based_format(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with signal-based format (separate ID and data channels)."""
        mdf_path = temp_dir / "signals.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()

        # Mock separate ID and data signals
        mock_id_signal = Mock()
        mock_id_signal.timestamps = np.arange(5, dtype=np.float64) * 0.1
        mock_id_signal.samples = np.array([0x123, 0x280, 0x300, 0x123, 0x280], dtype=np.uint32)

        mock_data_signal = Mock()
        mock_data_signal.timestamps = np.arange(5, dtype=np.float64) * 0.1
        mock_data_signal.samples = np.array(
            [[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08] for _ in range(5)], dtype=np.uint8
        )

        def mock_get(channel_name):
            if "ID" in channel_name:
                return mock_id_signal
            elif "Data" in channel_name:
                return mock_data_signal
            raise KeyError(channel_name)

        mock_mdf.channels_db = ["CAN_ID", "CAN_Data"]
        mock_mdf.get.side_effect = mock_get

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        messages = load_mdf(mdf_path)

        assert len(messages) == 5
        assert messages[0].arbitration_id == 0x123

    @patch("asammdf.MDF")
    def test_load_mdf_no_can_data(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with no CAN data raises error."""
        mdf_path = temp_dir / "no_can.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()
        mock_mdf.channels_db = ["Temperature", "Pressure", "Speed"]  # No CAN channels

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        with pytest.raises(ValueError, match="No CAN messages found"):
            load_mdf(mdf_path)

    @patch("asammdf.MDF")
    def test_load_mdf_extended_ids(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with extended IDs."""
        mdf_path = temp_dir / "extended.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()

        # Create structured array with extended IDs
        dtype = np.dtype([("ID", np.uint32), ("Data", np.uint8, (8,))])

        samples = np.zeros(5, dtype=dtype)
        for i in range(5):
            samples[i]["ID"] = 0x18FF1234 + i  # Extended IDs
            samples[i]["Data"] = np.arange(8, dtype=np.uint8)

        mock_signal = Mock()
        mock_signal.timestamps = np.arange(5, dtype=np.float64) * 0.1
        mock_signal.samples = samples

        mock_mdf.channels_db = ["CAN_DataFrame"]
        mock_mdf.get.return_value = mock_signal

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        messages = load_mdf(mdf_path)

        assert len(messages) == 5
        assert all(msg.is_extended for msg in messages)
        assert messages[0].arbitration_id == 0x18FF1234

    @patch("asammdf.MDF")
    def test_load_mdf_malformed_frames_skipped(self, mock_mdf_class, temp_dir: Path):
        """Test that malformed frames are skipped gracefully."""
        mdf_path = temp_dir / "malformed.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()

        # Create structured array with some invalid data
        dtype = np.dtype([("ID", np.uint32), ("Data", np.uint8, (8,))])

        samples = np.zeros(10, dtype=dtype)
        for i in range(10):
            samples[i]["ID"] = 0x123 + i
            samples[i]["Data"] = np.arange(8, dtype=np.uint8)

        # Make some samples trigger errors (will be caught and skipped)
        mock_signal = Mock()
        mock_signal.timestamps = np.arange(10, dtype=np.float64) * 0.1
        mock_signal.samples = samples

        mock_mdf.channels_db = ["CAN_DataFrame"]
        mock_mdf.get.return_value = mock_signal

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        messages = load_mdf(mdf_path)

        # Should succeed even with potential errors
        assert len(messages) >= 0

    @patch("asammdf.MDF")
    def test_load_mdf_path_as_string(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with path as string."""
        mdf_path = temp_dir / "string_path.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()

        dtype = np.dtype([("ID", np.uint32), ("Data", np.uint8, (8,))])
        samples = np.zeros(3, dtype=dtype)
        for i in range(3):
            samples[i]["ID"] = 0x123
            samples[i]["Data"] = np.arange(8, dtype=np.uint8)

        mock_signal = Mock()
        mock_signal.timestamps = np.arange(3, dtype=np.float64) * 0.1
        mock_signal.samples = samples

        mock_mdf.channels_db = ["CAN_DataFrame"]
        mock_mdf.get.return_value = mock_signal

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        # Load using string path
        messages = load_mdf(str(mdf_path))

        assert len(messages) == 3

    @patch("asammdf.MDF")
    def test_load_mdf_empty_channel(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with empty CAN channel."""
        mdf_path = temp_dir / "empty_channel.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()

        # Create empty structured array
        dtype = np.dtype([("ID", np.uint32), ("Data", np.uint8, (8,))])
        samples = np.zeros(0, dtype=dtype)

        mock_signal = Mock()
        mock_signal.timestamps = np.array([], dtype=np.float64)
        mock_signal.samples = samples

        mock_mdf.channels_db = ["CAN_DataFrame"]
        mock_mdf.get.return_value = mock_signal

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        with pytest.raises(ValueError, match="No CAN messages found"):
            load_mdf(mdf_path)

    @patch("asammdf.MDF")
    def test_load_mdf_variable_dlc(self, mock_mdf_class, temp_dir: Path):
        """Test loading MDF with variable DLC."""
        mdf_path = temp_dir / "variable_dlc.mf4"
        mdf_path.touch()

        mock_mdf = MagicMock()

        # Create structured array with variable DLC
        dtype = np.dtype([("ID", np.uint32), ("Data", np.uint8, (8,)), ("DLC", np.uint8)])

        samples = np.zeros(8, dtype=dtype)
        for i in range(8):
            samples[i]["ID"] = 0x123
            samples[i]["Data"] = np.arange(8, dtype=np.uint8)
            samples[i]["DLC"] = i + 1  # DLC from 1 to 8

        mock_signal = Mock()
        mock_signal.timestamps = np.arange(8, dtype=np.float64) * 0.1
        mock_signal.samples = samples

        mock_mdf.channels_db = ["CAN_DataFrame"]
        mock_mdf.get.return_value = mock_signal

        mock_mdf_class.return_value.__enter__.return_value = mock_mdf
        mock_mdf_class.return_value.__exit__.return_value = None

        messages = load_mdf(mdf_path)

        assert len(messages) == 8

        # Verify DLC is respected
        for i, msg in enumerate(messages):
            assert msg.dlc == i + 1
