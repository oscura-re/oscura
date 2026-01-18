"""Tests for BLF (Binary Logging Format) file loader.

Tests the Vector BLF loader using python-can's BLF writer to create
synthetic test files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oscura.automotive.loaders.blf import load_blf

pytestmark = pytest.mark.loader


# Check if python-can is available
pytest_importorskip = pytest.importorskip("can", reason="python-can required for BLF tests")


@pytest.mark.unit
@pytest.mark.loader
class TestBLFLoader:
    """Tests for BLF file loading."""

    def test_load_blf_basic(self, sample_can_data: list[dict], temp_dir: Path):
        """Test loading a basic BLF file."""
        import can

        # Create synthetic BLF file
        blf_path = temp_dir / "test.blf"

        with can.BLFWriter(str(blf_path)) as writer:
            for msg_data in sample_can_data[:10]:  # First 10 messages
                can_msg = can.Message(
                    arbitration_id=msg_data["id"],
                    timestamp=msg_data["timestamp"],
                    data=msg_data["data"],
                    is_extended_id=msg_data.get("is_extended", False),
                )
                writer.on_message_received(can_msg)

        # Load with our loader
        messages = load_blf(blf_path)

        # Verify
        assert len(messages) == 10
        assert messages[0].arbitration_id == 0x123
        assert messages[0].timestamp == pytest.approx(0.0, abs=0.001)
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

    def test_load_blf_extended_ids(self, sample_can_data: list[dict], temp_dir: Path):
        """Test loading BLF with extended IDs."""
        import can

        blf_path = temp_dir / "extended.blf"

        # Create BLF with extended ID messages
        extended_messages = [msg for msg in sample_can_data if msg.get("is_extended", False)]

        with can.BLFWriter(str(blf_path)) as writer:
            for msg_data in extended_messages:
                can_msg = can.Message(
                    arbitration_id=msg_data["id"],
                    timestamp=msg_data["timestamp"],
                    data=msg_data["data"],
                    is_extended_id=True,
                )
                writer.on_message_received(can_msg)

        messages = load_blf(blf_path)

        assert len(messages) > 0
        assert all(msg.is_extended for msg in messages)
        assert messages[0].arbitration_id == 0x18FF1234

    def test_load_blf_standard_and_extended(self, sample_can_data: list[dict], temp_dir: Path):
        """Test loading BLF with mixed standard and extended IDs."""
        import can

        blf_path = temp_dir / "mixed.blf"

        with can.BLFWriter(str(blf_path)) as writer:
            for msg_data in sample_can_data[:15]:  # Mix of standard and extended
                can_msg = can.Message(
                    arbitration_id=msg_data["id"],
                    timestamp=msg_data["timestamp"],
                    data=msg_data["data"],
                    is_extended_id=msg_data.get("is_extended", False),
                )
                writer.on_message_received(can_msg)

        messages = load_blf(blf_path)

        assert len(messages) == 15

        # Check we have both types
        standard = [msg for msg in messages if not msg.is_extended]
        extended = [msg for msg in messages if msg.is_extended]

        assert len(standard) > 0
        assert len(extended) > 0

    def test_load_blf_empty_file(self, temp_dir: Path):
        """Test loading an empty BLF file."""
        import can

        blf_path = temp_dir / "empty.blf"

        # Create empty BLF file
        with can.BLFWriter(str(blf_path)) as writer:
            pass  # No messages

        messages = load_blf(blf_path)

        assert len(messages) == 0

    def test_load_blf_file_not_found(self):
        """Test loading non-existent BLF file."""
        with pytest.raises(FileNotFoundError, match="BLF file not found"):
            load_blf("/nonexistent/file.blf")

    def test_load_blf_corrupted_file(self, temp_dir: Path):
        """Test loading corrupted BLF file."""
        import warnings

        blf_path = temp_dir / "corrupted.blf"

        # Create invalid BLF file (wrong format)
        with open(blf_path, "wb") as f:
            f.write(b"NOT A BLF FILE" * 100)

        # python-can may raise different exceptions for corrupted files
        # Suppress ResourceWarning from python-can's file handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            with pytest.raises((ValueError, Exception)):
                load_blf(blf_path)

    def test_load_blf_path_as_string(self, sample_can_data: list[dict], temp_dir: Path):
        """Test loading BLF with path as string."""
        import can

        blf_path = temp_dir / "test_string.blf"

        with can.BLFWriter(str(blf_path)) as writer:
            for msg_data in sample_can_data[:5]:
                can_msg = can.Message(
                    arbitration_id=msg_data["id"],
                    timestamp=msg_data["timestamp"],
                    data=msg_data["data"],
                )
                writer.on_message_received(can_msg)

        # Load using string path
        messages = load_blf(str(blf_path))

        assert len(messages) == 5

    def test_load_blf_varying_dlc(self, temp_dir: Path):
        """Test loading BLF with varying data lengths."""
        import can

        blf_path = temp_dir / "varying_dlc.blf"

        with can.BLFWriter(str(blf_path)) as writer:
            # Messages with different DLC values
            for dlc in range(9):
                can_msg = can.Message(
                    arbitration_id=0x100 + dlc,
                    timestamp=dlc * 0.1,
                    data=bytes(range(dlc)),
                )
                writer.on_message_received(can_msg)

        messages = load_blf(blf_path)

        assert len(messages) == 9

        # Verify DLC values
        for i, msg in enumerate(messages):
            assert msg.dlc == i
            assert len(msg.data) == i

    def test_load_blf_timestamp_order(self, temp_dir: Path):
        """Test that timestamps are preserved correctly."""
        import can

        blf_path = temp_dir / "timestamps.blf"

        # Create messages with monotonically increasing timestamps
        test_messages = []
        for i in range(10):
            test_messages.append(
                {
                    "id": 0x123,
                    "timestamp": i * 0.1,
                    "data": bytes([i, i + 1, i + 2, i + 3]),
                }
            )

        with can.BLFWriter(str(blf_path)) as writer:
            for msg_data in test_messages:
                can_msg = can.Message(
                    arbitration_id=msg_data["id"],
                    timestamp=msg_data["timestamp"],
                    data=msg_data["data"],
                )
                writer.on_message_received(can_msg)

        messages = load_blf(blf_path)

        # Check we got all messages
        assert len(messages) == 10

        # Check timestamps are preserved (with some tolerance)
        for i, msg in enumerate(messages):
            assert msg.timestamp == pytest.approx(test_messages[i]["timestamp"], abs=0.001)

    def test_load_blf_channel_info(self, sample_can_data: list[dict], temp_dir: Path):
        """Test that channel information is preserved."""
        import can

        blf_path = temp_dir / "channels.blf"

        with can.BLFWriter(str(blf_path)) as writer:
            for i, msg_data in enumerate(sample_can_data[:10]):
                can_msg = can.Message(
                    arbitration_id=msg_data["id"],
                    timestamp=msg_data["timestamp"],
                    data=msg_data["data"],
                    channel=i % 2,  # Alternate between channels 0 and 1
                )
                writer.on_message_received(can_msg)

        messages = load_blf(blf_path)

        # Verify channel information is preserved
        assert len(messages) == 10
        for i, msg in enumerate(messages):
            # Channel should be 0 or 1
            assert msg.channel in [0, 1]

    def test_load_blf_import_error(self, monkeypatch):
        """Test that ImportError is raised if python-can not installed."""
        import sys

        # Mock import failure
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "can", None)

            # Force reimport
            import importlib

            from oscura.automotive.loaders import blf as blf_module

            importlib.reload(blf_module)

            with pytest.raises(ImportError, match="python-can is required"):
                blf_module.load_blf("dummy.blf")
