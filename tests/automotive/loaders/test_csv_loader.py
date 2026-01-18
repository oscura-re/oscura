"""Tests for CSV CAN log file loader.

Tests the CSV CAN loader using synthetic CSV content created as strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oscura.automotive.loaders.csv_can import load_csv_can

pytestmark = pytest.mark.loader


@pytest.mark.unit
@pytest.mark.loader
class TestCSVCANLoader:
    """Tests for CSV CAN file loading."""

    def test_load_csv_basic(self, temp_dir: Path):
        """Test loading a basic CSV file."""
        csv_path = temp_dir / "test.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,0102030405060708
0.010000,0x280,0A0B0C0D0E0F1011
0.020000,0x300,AABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[0].timestamp == pytest.approx(0.0, abs=0.001)
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

        assert messages[1].arbitration_id == 0x280
        assert messages[1].data == bytes([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11])

        assert messages[2].arbitration_id == 0x300
        assert messages[2].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_csv_decimal_ids(self, temp_dir: Path):
        """Test loading CSV with decimal IDs."""
        csv_path = temp_dir / "decimal.csv"

        csv_content = """timestamp,id,data
0.000000,291,0102030405060708
0.010000,640,AABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2
        assert messages[0].arbitration_id == 291  # 0x123
        assert messages[1].arbitration_id == 640  # 0x280

    def test_load_csv_extended_ids(self, temp_dir: Path):
        """Test loading CSV with extended IDs."""
        csv_path = temp_dir / "extended.csv"

        csv_content = """timestamp,id,data
0.000000,0x18FF1234,11223344
0.010000,0x1FFFFFFF,AABBCCDD
0.020000,0x7FF,55667788
0.030000,0x800,99AABBCC
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 4

        # Extended IDs (>0x7FF)
        assert messages[0].arbitration_id == 0x18FF1234
        assert messages[0].is_extended

        assert messages[1].arbitration_id == 0x1FFFFFFF
        assert messages[1].is_extended

        # Standard ID
        assert messages[2].arbitration_id == 0x7FF
        assert not messages[2].is_extended

        # Extended ID
        assert messages[3].arbitration_id == 0x800
        assert messages[3].is_extended

    def test_load_csv_column_name_variations(self, temp_dir: Path):
        """Test CSV with various column name formats."""
        csv_path = temp_dir / "columns.csv"

        # Test case-insensitive column detection
        csv_content = """Time,CAN_ID,Payload
0.000000,0x123,01020304
0.010000,0x280,AABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2

    def test_load_csv_uppercase_columns(self, temp_dir: Path):
        """Test CSV with uppercase column names."""
        csv_path = temp_dir / "uppercase.csv"

        csv_content = """TIMESTAMP,ID,DATA
0.000000,0x123,01020304
0.010000,0x280,AABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2

    def test_load_csv_arbitration_id_column(self, temp_dir: Path):
        """Test CSV with 'arbitration_id' column name."""
        csv_path = temp_dir / "arb_id.csv"

        csv_content = """timestamp,arbitration_id,data
0.000000,0x123,01020304
0.010000,0x280,AABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2

    def test_load_csv_data_with_spaces(self, temp_dir: Path):
        """Test CSV with spaces in data field."""
        csv_path = temp_dir / "spaces.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,01 02 03 04
0.010000,0x280,AA BB CC DD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04])
        assert messages[1].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_csv_data_with_colons(self, temp_dir: Path):
        """Test CSV with colons in data field."""
        csv_path = temp_dir / "colons.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,01:02:03:04
0.010000,0x280,AA:BB:CC:DD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04])
        assert messages[1].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_csv_data_with_dashes(self, temp_dir: Path):
        """Test CSV with dashes in data field."""
        csv_path = temp_dir / "dashes.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,01-02-03-04
0.010000,0x280,AA-BB-CC-DD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04])
        assert messages[1].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_csv_data_with_0x_prefix(self, temp_dir: Path):
        """Test CSV with 0x prefix in data field."""
        csv_path = temp_dir / "prefix.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,0x01020304
0.010000,0x280,0xAABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04])
        assert messages[1].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_csv_malformed_rows_skipped(self, temp_dir: Path):
        """Test that malformed rows are skipped."""
        csv_path = temp_dir / "malformed.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,01020304
invalid,row,here
0.010000,0x280,AABBCCDD
another,bad,row
0.020000,0x300,11223344
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        # Should skip bad rows
        assert len(messages) == 3

    def test_load_csv_semicolon_delimiter(self, temp_dir: Path):
        """Test CSV with semicolon delimiter."""
        csv_path = temp_dir / "semicolon.csv"

        csv_content = """timestamp;id;data
0.000000;0x123;01020304
0.010000;0x280;AABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path, delimiter=";")

        assert len(messages) == 2

    def test_load_csv_tab_delimiter(self, temp_dir: Path):
        """Test CSV with tab delimiter."""
        csv_path = temp_dir / "tab.csv"

        csv_content = """timestamp\tid\tdata
0.000000\t0x123\t01020304
0.010000\t0x280\tAABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path, delimiter="\t")

        assert len(messages) == 2

    def test_load_csv_empty_file(self, temp_dir: Path):
        """Test loading an empty CSV file."""
        csv_path = temp_dir / "empty.csv"

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(ValueError, match="CSV file has no header row"):
            load_csv_can(csv_path)

    def test_load_csv_only_header(self, temp_dir: Path):
        """Test loading CSV with only header."""
        csv_path = temp_dir / "header_only.csv"

        csv_content = """timestamp,id,data
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 0

    def test_load_csv_missing_columns(self, temp_dir: Path):
        """Test loading CSV with missing required columns."""
        csv_path = temp_dir / "missing.csv"

        csv_content = """timestamp,data
0.000000,01020304
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        with pytest.raises(ValueError, match="CSV file missing required columns"):
            load_csv_can(csv_path)

    def test_load_csv_file_not_found(self):
        """Test loading non-existent CSV file."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            load_csv_can("/nonexistent/file.csv")

    def test_load_csv_path_as_string(self, temp_dir: Path):
        """Test loading CSV with path as string."""
        csv_path = temp_dir / "string_path.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,01020304
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        # Load using string path
        messages = load_csv_can(str(csv_path))

        assert len(messages) == 1

    def test_load_csv_varying_dlc(self, temp_dir: Path):
        """Test loading CSV with varying data lengths."""
        csv_path = temp_dir / "varying_dlc.csv"

        csv_content = """timestamp,id,data
0.000000,0x100,
0.010000,0x101,01
0.020000,0x102,0102
0.030000,0x103,010203
0.040000,0x104,01020304
0.050000,0x105,0102030405
0.060000,0x106,010203040506
0.070000,0x107,01020304050607
0.080000,0x108,0102030405060708
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 9

        # Verify DLC values
        for i, msg in enumerate(messages):
            assert msg.dlc == i

    def test_load_csv_lowercase_hex(self, temp_dir: Path):
        """Test CSV with lowercase hex data."""
        csv_path = temp_dir / "lowercase.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,aabbccdd
0.010000,0x280,eeff0011
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2
        assert messages[0].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])
        assert messages[1].data == bytes([0xEE, 0xFF, 0x00, 0x11])

    def test_load_csv_mixed_case_hex(self, temp_dir: Path):
        """Test CSV with mixed case hex data."""
        csv_path = temp_dir / "mixed_case.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,AaBbCcDd
0.010000,0x280,EeFf0011
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 2
        assert messages[0].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])
        assert messages[1].data == bytes([0xEE, 0xFF, 0x00, 0x11])

    def test_load_csv_timestamp_precision(self, temp_dir: Path):
        """Test that timestamp precision is preserved."""
        csv_path = temp_dir / "precision.csv"

        csv_content = """timestamp,id,data
0.123456,0x123,01020304
1.234567,0x280,AABBCCDD
12.345678,0x300,11223344
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        assert len(messages) == 3
        assert messages[0].timestamp == pytest.approx(0.123456, abs=1e-6)
        assert messages[1].timestamp == pytest.approx(1.234567, abs=1e-6)
        assert messages[2].timestamp == pytest.approx(12.345678, abs=1e-6)

    def test_load_csv_extra_columns(self, temp_dir: Path):
        """Test CSV with extra columns (should be ignored)."""
        csv_path = temp_dir / "extra.csv"

        csv_content = """timestamp,id,data,channel,direction,notes
0.000000,0x123,01020304,1,Rx,Test message
0.010000,0x280,AABBCCDD,2,Tx,Another test
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_csv_can(csv_path)

        # Should parse successfully, ignoring extra columns
        assert len(messages) == 2
