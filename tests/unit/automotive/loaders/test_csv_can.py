"""Unit tests for CSV CAN file loader.

This module tests the CSV CAN parser, including:
- Various CSV format variations
- Column detection
- ID parsing (hex/decimal)
- Data parsing
- Error handling
"""

from pathlib import Path

import pytest

from oscura.automotive.can.models import CANMessage, CANMessageList
from oscura.automotive.loaders.csv_can import load_csv_can


class TestLoadCsvCan:
    """Test cases for load_csv_can function."""

    def test_load_valid_csv_basic(self, tmp_path: Path) -> None:
        """Test loading a basic CSV file with standard columns.

        Verifies:
        - Header detection
        - Timestamp parsing
        - CAN ID parsing (hex)
        - Data bytes parsing
        """
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
0.010000,0x280,0A0B0C0D0E0F1011
0.020000,0x1FF,AABBCCDD
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[0].timestamp == 0.0
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

        assert messages[1].arbitration_id == 0x280
        assert messages[1].timestamp == 0.010000

        assert messages[2].arbitration_id == 0x1FF
        assert messages[2].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_csv_decimal_id(self, tmp_path: Path) -> None:
        """Test parsing decimal CAN IDs."""
        csv_content = """timestamp,id,data
0.000000,291,0102030405060708
0.010000,640,0A0B0C0D0E0F1011
"""
        csv_file = tmp_path / "test_decimal.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 2
        assert messages[0].arbitration_id == 291  # 0x123
        assert messages[1].arbitration_id == 640  # 0x280

    def test_load_csv_mixed_id_formats(self, tmp_path: Path) -> None:
        """Test parsing mixed decimal and hex IDs."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
0.010000,640,0A0B0C0D0E0F1011
0.020000,0X1FF,AABBCCDD
"""
        csv_file = tmp_path / "test_mixed.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 640
        assert messages[2].arbitration_id == 0x1FF

    def test_load_csv_alternative_column_names(self, tmp_path: Path) -> None:
        """Test automatic detection of alternative column names."""
        csv_content = """time,can_id,payload
0.000000,0x123,0102030405060708
0.010000,0x280,0A0B0C0D0E0F1011
"""
        csv_file = tmp_path / "test_alt.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 2
        assert messages[0].timestamp == 0.0
        assert messages[0].arbitration_id == 0x123

    def test_load_csv_case_insensitive_headers(self, tmp_path: Path) -> None:
        """Test case-insensitive header matching."""
        csv_content = """TIMESTAMP,ID,DATA
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test_case.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].arbitration_id == 0x123

    def test_load_csv_arbitration_id_column(self, tmp_path: Path) -> None:
        """Test parsing with 'arbitration_id' column name."""
        csv_content = """timestamp,arbitration_id,data
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test_arbid.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].arbitration_id == 0x123

    def test_load_csv_data_with_spaces(self, tmp_path: Path) -> None:
        """Test parsing data bytes with spaces."""
        csv_content = """timestamp,id,data
0.000000,0x123,01 02 03 04 05 06 07 08
"""
        csv_file = tmp_path / "test_spaces.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

    def test_load_csv_data_with_colons(self, tmp_path: Path) -> None:
        """Test parsing data bytes with colon separators."""
        csv_content = """timestamp,id,data
0.000000,0x123,01:02:03:04:05:06:07:08
"""
        csv_file = tmp_path / "test_colons.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

    def test_load_csv_data_with_hyphens(self, tmp_path: Path) -> None:
        """Test parsing data bytes with hyphen separators."""
        csv_content = """timestamp,id,data
0.000000,0x123,01-02-03-04-05-06-07-08
"""
        csv_file = tmp_path / "test_hyphens.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

    def test_load_csv_data_with_0x_prefix(self, tmp_path: Path) -> None:
        """Test parsing data with 0x prefix."""
        csv_content = """timestamp,id,data
0.000000,0x123,0x0102030405060708
"""
        csv_file = tmp_path / "test_prefix.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

    def test_load_csv_semicolon_delimiter(self, tmp_path: Path) -> None:
        """Test parsing CSV with semicolon delimiter."""
        csv_content = """timestamp;id;data
0.000000;0x123;0102030405060708
"""
        csv_file = tmp_path / "test_semicolon.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file, delimiter=";")

        assert len(messages) == 1
        assert messages[0].arbitration_id == 0x123

    def test_load_csv_tab_delimiter(self, tmp_path: Path) -> None:
        """Test parsing CSV with tab delimiter."""
        csv_content = """timestamp\tid\tdata
0.000000\t0x123\t0102030405060708
"""
        csv_file = tmp_path / "test_tab.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file, delimiter="\t")

        assert len(messages) == 1
        assert messages[0].arbitration_id == 0x123

    def test_load_csv_extended_id(self, tmp_path: Path) -> None:
        """Test parsing extended CAN IDs (>0x7FF)."""
        csv_content = """timestamp,id,data
0.000000,0x800,0102030405060708
0.010000,0x18FF1234,0A0B0C0D0E0F1011
"""
        csv_file = tmp_path / "test_ext.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 2
        assert messages[0].arbitration_id == 0x800
        assert messages[0].is_extended
        assert messages[1].arbitration_id == 0x18FF1234
        assert messages[1].is_extended

    def test_load_csv_standard_id(self, tmp_path: Path) -> None:
        """Test that standard IDs are marked as non-extended."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
0.010000,0x7FF,0A0B0C0D0E0F1011
"""
        csv_file = tmp_path / "test_std.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 2
        assert not messages[0].is_extended
        assert not messages[1].is_extended  # 0x7FF is still standard

    def test_load_csv_empty_file(self, tmp_path: Path) -> None:
        """Test loading an empty CSV file (header only)."""
        csv_content = """timestamp,id,data
"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 0

    def test_load_csv_file_not_found(self, tmp_path: Path) -> None:
        """Test error handling when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            load_csv_can(nonexistent)

    def test_load_csv_no_header(self, tmp_path: Path) -> None:
        """Test error handling for CSV with no header."""
        csv_content = ""
        csv_file = tmp_path / "no_header.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="no header row"):
            load_csv_can(csv_file)

    def test_load_csv_missing_required_columns(self, tmp_path: Path) -> None:
        """Test error handling when required columns are missing."""
        csv_content = """timestamp,some_other_column
0.000000,value
"""
        csv_file = tmp_path / "missing_cols.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="missing required columns"):
            load_csv_can(csv_file)

    def test_load_csv_skip_malformed_rows(self, tmp_path: Path) -> None:
        """Test that malformed rows are skipped gracefully."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
invalid,row,here
0.010000,0x280,0A0B0C0D0E0F1011
0.020000,0x1FF,AABBCCDD
"""
        csv_file = tmp_path / "test_malformed.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        # Should parse 3 valid messages, skip malformed row
        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 0x280
        assert messages[2].arbitration_id == 0x1FF

    def test_load_csv_variable_length_data(self, tmp_path: Path) -> None:
        """Test parsing messages with various data lengths."""
        csv_content = """timestamp,id,data
0.000000,0x100,
0.001000,0x101,AA
0.002000,0x102,BBCC
0.003000,0x103,0102030405060708
"""
        csv_file = tmp_path / "test_dlc.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 4
        assert len(messages[0].data) == 0
        assert len(messages[1].data) == 1
        assert messages[1].data == bytes([0xAA])
        assert len(messages[2].data) == 2
        assert messages[2].data == bytes([0xBB, 0xCC])
        assert len(messages[3].data) == 8

    def test_load_csv_with_pathlib_path(self, tmp_path: Path) -> None:
        """Test loading with pathlib.Path object."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1

    def test_load_csv_with_string_path(self, tmp_path: Path) -> None:
        """Test loading with string path."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(str(csv_file))

        assert len(messages) == 1

    def test_load_csv_return_type(self, tmp_path: Path) -> None:
        """Test that return type is CANMessageList."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert isinstance(messages, CANMessageList)
        assert isinstance(messages[0], CANMessage)

    def test_load_csv_whitespace_in_data(self, tmp_path: Path) -> None:
        """Test parsing data with leading/trailing whitespace."""
        csv_content = """timestamp,id,data
0.000000,0x123,  0102030405060708
"""
        csv_file = tmp_path / "test_ws.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

    def test_load_csv_lowercase_hex(self, tmp_path: Path) -> None:
        """Test parsing lowercase hex data."""
        csv_content = """timestamp,id,data
0.000000,0x123,aabbccddeeff0011
"""
        csv_file = tmp_path / "test_lower.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11])

    def test_load_csv_mixed_case_hex(self, tmp_path: Path) -> None:
        """Test parsing mixed case hex data."""
        csv_content = """timestamp,id,data
0.000000,0x123,AaBbCcDdEeFf0011
"""
        csv_file = tmp_path / "test_mixed.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11])

    def test_load_csv_channel_default(self, tmp_path: Path) -> None:
        """Test that channel defaults to 0."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert messages[0].channel == 0

    def test_load_csv_is_fd_false(self, tmp_path: Path) -> None:
        """Test that messages are marked as non-FD."""
        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert not messages[0].is_fd

    def test_load_csv_large_file(self, tmp_path: Path) -> None:
        """Test loading a larger CSV file."""
        lines = ["timestamp,id,data"]
        for i in range(1000):
            timestamp = i * 0.001
            can_id = (i % 256) + 0x100
            data = "".join(f"{(i + j) % 256:02X}" for j in range(8))
            lines.append(f"{timestamp:.6f},0x{can_id:X},{data}")

        csv_file = tmp_path / "test_large.csv"
        csv_file.write_text("\n".join(lines))

        messages = load_csv_can(csv_file)

        assert len(messages) == 1000
        assert messages[0].timestamp == 0.0
        assert messages[999].timestamp == pytest.approx(0.999, abs=1e-6)

    def test_load_csv_single_char_column_name(self, tmp_path: Path) -> None:
        """Test parsing with single-character column names."""
        csv_content = """t,id,data
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test_short.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].timestamp == 0.0

    def test_load_csv_bytes_column_name(self, tmp_path: Path) -> None:
        """Test parsing with 'bytes' as data column name."""
        csv_content = """timestamp,id,bytes
0.000000,0x123,0102030405060708
"""
        csv_file = tmp_path / "test_bytes.csv"
        csv_file.write_text(csv_content)

        messages = load_csv_can(csv_file)

        assert len(messages) == 1
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
