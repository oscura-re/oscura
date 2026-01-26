"""Unit tests for ASC (Vector ASCII) file loader.

This module tests the ASC file parser, including:
- Valid ASC file parsing
- Message extraction and field parsing
- Error handling for malformed files
- Edge cases (empty files, invalid data, etc.)
"""

from pathlib import Path

import pytest

from oscura.automotive.can.models import CANMessage, CANMessageList
from oscura.automotive.loaders.asc import load_asc


class TestLoadAsc:
    """Test cases for load_asc function."""

    def test_load_valid_asc_file(self, tmp_path: Path) -> None:
        """Test loading a valid ASC file with multiple messages.

        Verifies:
        - Messages are correctly parsed
        - Timestamps are extracted
        - CAN IDs are parsed (hex)
        - Data bytes are correctly decoded
        - Extended ID detection works
        """
        asc_content = """date Mon Jul 15 10:30:45.123 2024
// This is a comment
0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
0.010000 1 280 Rx d 8 0A 0B 0C 0D 0E 0F 10 11
0.020000 1 1FF Tx d 4 AA BB CC DD
"""
        asc_file = tmp_path / "test.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[0].timestamp == 0.0
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
        assert not messages[0].is_extended

        assert messages[1].arbitration_id == 0x280
        assert messages[1].timestamp == 0.010000
        assert messages[1].data == bytes([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11])

        assert messages[2].arbitration_id == 0x1FF
        assert messages[2].timestamp == 0.020000
        assert messages[2].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_asc_extended_id(self, tmp_path: Path) -> None:
        """Test parsing extended CAN IDs (29-bit).

        Extended IDs are typically > 0x7FF (2047).
        """
        asc_content = """0.000000 1 18FF1234 Rx d 8 01 02 03 04 05 06 07 08
0.001000 1 800 Rx d 8 AA BB CC DD EE FF 00 11
"""
        asc_file = tmp_path / "test_ext.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 2
        # First message has extended ID
        assert messages[0].arbitration_id == 0x18FF1234
        assert messages[0].is_extended

        # Second message has extended ID (> 0x7FF)
        assert messages[1].arbitration_id == 0x800
        assert messages[1].is_extended

    def test_load_asc_lowercase_hex(self, tmp_path: Path) -> None:
        """Test parsing hex values in lowercase."""
        asc_content = """0.000000 1 1a3 Rx d 8 aa bb cc dd ee ff 00 11
"""
        asc_file = tmp_path / "test_lower.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 1
        assert messages[0].arbitration_id == 0x1A3
        assert messages[0].data == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11])

    def test_load_asc_mixed_case_hex(self, tmp_path: Path) -> None:
        """Test parsing mixed case hex values."""
        asc_content = """0.000000 1 AbC Rx d 8 aB Cd EF 01 23 45 67 89
"""
        asc_file = tmp_path / "test_mixed.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 1
        assert messages[0].arbitration_id == 0xABC
        assert messages[0].data == bytes([0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89])

    def test_load_asc_variable_dlc(self, tmp_path: Path) -> None:
        """Test parsing messages with various DLCs (0-8 bytes)."""
        asc_content = """0.000000 1 100 Rx d 0
0.001000 1 101 Rx d 1 AA
0.002000 1 102 Rx d 2 BB CC
0.003000 1 103 Rx d 8 01 02 03 04 05 06 07 08
"""
        asc_file = tmp_path / "test_dlc.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 4
        assert len(messages[0].data) == 0
        assert len(messages[1].data) == 1
        assert messages[1].data == bytes([0xAA])
        assert len(messages[2].data) == 2
        assert messages[2].data == bytes([0xBB, 0xCC])
        assert len(messages[3].data) == 8

    def test_load_asc_with_comments(self, tmp_path: Path) -> None:
        """Test that comment lines are properly skipped."""
        asc_content = """// Comment at start
date Mon Jul 15 10:30:45.123 2024
// Another comment
0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
// Comment in middle
0.001000 1 456 Rx d 8 AA BB CC DD EE FF 00 11
"""
        asc_file = tmp_path / "test_comments.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 2
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 0x456

    def test_load_asc_whitespace_variations(self, tmp_path: Path) -> None:
        """Test parsing with various whitespace patterns."""
        asc_content = """  0.000000   1   123   Rx   d   8   01 02 03 04 05 06 07 08
0.001000	1	456	Rx	d	8	AA BB CC DD EE FF 00 11
"""
        asc_file = tmp_path / "test_whitespace.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 2
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
        assert messages[1].data == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11])

    def test_load_asc_channel_number(self, tmp_path: Path) -> None:
        """Test parsing messages from different CAN channels."""
        asc_content = """0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
0.001000 2 456 Rx d 8 AA BB CC DD EE FF 00 11
0.002000 3 789 Rx d 8 11 22 33 44 55 66 77 88
"""
        asc_file = tmp_path / "test_channels.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 3
        assert messages[0].channel == 1
        assert messages[1].channel == 2
        assert messages[2].channel == 3

    def test_load_asc_tx_rx_direction(self, tmp_path: Path) -> None:
        """Test parsing both Tx and Rx messages."""
        asc_content = """0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
0.001000 1 456 Tx d 8 AA BB CC DD EE FF 00 11
"""
        asc_file = tmp_path / "test_direction.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        # Both Rx and Tx should be parsed
        assert len(messages) == 2

    def test_load_asc_empty_file(self, tmp_path: Path) -> None:
        """Test loading an empty ASC file."""
        asc_file = tmp_path / "empty.asc"
        asc_file.write_text("")

        messages = load_asc(asc_file)

        assert len(messages) == 0

    def test_load_asc_only_comments(self, tmp_path: Path) -> None:
        """Test loading file with only comments and headers."""
        asc_content = """date Mon Jul 15 10:30:45.123 2024
// Only comments here
// No actual CAN messages
"""
        asc_file = tmp_path / "only_comments.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 0

    def test_load_asc_file_not_found(self, tmp_path: Path) -> None:
        """Test error handling when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.asc"

        with pytest.raises(FileNotFoundError, match="ASC file not found"):
            load_asc(nonexistent)

    def test_load_asc_malformed_line_skipped(self, tmp_path: Path) -> None:
        """Test that malformed lines are skipped gracefully."""
        asc_content = """0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
this is not a valid line
0.001000 1 456 Rx d 8 AA BB CC DD EE FF 00 11
incomplete line without data
0.002000 1 789 Rx d 8 11 22 33 44 55 66 77 88
"""
        asc_file = tmp_path / "test_malformed.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        # Should parse 3 valid messages, skip 2 malformed lines
        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 0x456
        assert messages[2].arbitration_id == 0x789

    def test_load_asc_with_pathlib_path(self, tmp_path: Path) -> None:
        """Test loading with pathlib.Path object."""
        asc_content = """0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
"""
        asc_file = tmp_path / "test.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)  # Pass Path object

        assert len(messages) == 1

    def test_load_asc_with_string_path(self, tmp_path: Path) -> None:
        """Test loading with string path."""
        asc_content = """0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
"""
        asc_file = tmp_path / "test.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(str(asc_file))  # Pass string path

        assert len(messages) == 1

    def test_load_asc_precision_timestamps(self, tmp_path: Path) -> None:
        """Test parsing high-precision timestamps."""
        asc_content = """0.000000123 1 123 Rx d 8 01 02 03 04 05 06 07 08
1.234567890 1 456 Rx d 8 AA BB CC DD EE FF 00 11
"""
        asc_file = tmp_path / "test_precision.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert len(messages) == 2
        assert pytest.approx(messages[0].timestamp, abs=1e-9) == 0.000000123
        assert pytest.approx(messages[1].timestamp, abs=1e-9) == 1.234567890

    def test_load_asc_is_fd_false(self, tmp_path: Path) -> None:
        """Test that ASC messages are marked as non-FD."""
        asc_content = """0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
"""
        asc_file = tmp_path / "test.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        # ASC format typically doesn't indicate FD
        assert not messages[0].is_fd

    def test_load_asc_large_file(self, tmp_path: Path) -> None:
        """Test loading a larger file with many messages."""
        # Generate 1000 messages
        lines = ["date Mon Jul 15 10:30:45.123 2024"]
        for i in range(1000):
            timestamp = i * 0.001
            can_id = (i % 256) + 0x100
            data = " ".join(f"{(i + j) % 256:02X}" for j in range(8))
            lines.append(f"{timestamp:.6f} 1 {can_id:X} Rx d 8 {data}")

        asc_file = tmp_path / "test_large.asc"
        asc_file.write_text("\n".join(lines))

        messages = load_asc(asc_file)

        assert len(messages) == 1000
        assert messages[0].timestamp == 0.0
        assert messages[999].timestamp == pytest.approx(0.999, abs=1e-6)

    def test_load_asc_return_type(self, tmp_path: Path) -> None:
        """Test that return type is CANMessageList."""
        asc_content = """0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
"""
        asc_file = tmp_path / "test.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        assert isinstance(messages, CANMessageList)
        assert isinstance(messages[0], CANMessage)

    def test_load_asc_utf8_encoding(self, tmp_path: Path) -> None:
        """Test UTF-8 encoding handling."""
        asc_content = """// UTF-8 comment: 测试 тест
date Mon Jul 15 10:30:45.123 2024
0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
"""
        asc_file = tmp_path / "test_utf8.asc"
        asc_file.write_text(asc_content, encoding="utf-8")

        messages = load_asc(asc_file)

        assert len(messages) == 1

    def test_load_asc_data_no_spaces(self, tmp_path: Path) -> None:
        """Test parsing data bytes without spaces (should fail gracefully)."""
        asc_content = """0.000000 1 123 Rx d 8 0102030405060708
"""
        asc_file = tmp_path / "test_no_spaces.asc"
        asc_file.write_text(asc_content)

        messages = load_asc(asc_file)

        # Should still parse correctly
        assert len(messages) == 1
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
