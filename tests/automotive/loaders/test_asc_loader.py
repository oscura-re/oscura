"""Tests for ASC (ASCII Format) file loader.

Tests the Vector ASC loader using synthetic ASC content created as strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oscura.automotive.loaders.asc import load_asc

pytestmark = pytest.mark.loader


@pytest.mark.unit
@pytest.mark.loader
class TestASCLoader:
    """Tests for ASC file loading."""

    def test_load_asc_basic(self, temp_dir: Path):
        """Test loading a basic ASC file."""
        asc_path = temp_dir / "test.asc"

        # Create synthetic ASC content
        asc_content = """date Mon Jul 15 10:30:45.123 2024
0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
0.010000 1 280 Rx d 8 0A 0B 0C 0D 0E 0F 10 11
0.020000 1 300 Tx d 4 AA BB CC DD
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[0].timestamp == pytest.approx(0.0, abs=0.001)
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])

        assert messages[1].arbitration_id == 0x280
        assert messages[1].timestamp == pytest.approx(0.01, abs=0.001)
        assert messages[1].data == bytes([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11])

        assert messages[2].arbitration_id == 0x300
        assert messages[2].timestamp == pytest.approx(0.02, abs=0.001)
        assert messages[2].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_load_asc_extended_ids(self, temp_dir: Path):
        """Test loading ASC with extended IDs."""
        asc_path = temp_dir / "extended.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 18FF1234 Rx d 4 11 22 33 44
0.010000 1 1FFFFFFF Rx d 8 AA BB CC DD EE FF 00 11
0.020000 1 7FF Rx d 2 12 34
0.030000 1 800 Rx d 2 56 78
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 4

        # Extended IDs (>0x7FF)
        assert messages[0].arbitration_id == 0x18FF1234
        assert messages[0].is_extended

        assert messages[1].arbitration_id == 0x1FFFFFFF
        assert messages[1].is_extended

        # Standard ID (exactly 0x7FF)
        assert messages[2].arbitration_id == 0x7FF
        assert not messages[2].is_extended

        # Extended ID (>0x7FF)
        assert messages[3].arbitration_id == 0x800
        assert messages[3].is_extended

    def test_load_asc_comments_and_headers(self, temp_dir: Path):
        """Test that comments and header lines are skipped."""
        asc_path = temp_dir / "with_comments.asc"

        asc_content = """// This is a comment
date Mon Jul 15 10:30:45 2024
// Another comment
0.000000 1 123 Rx d 4 01 02 03 04
// Comment in the middle
date Mon Jul 15 10:30:46 2024
0.010000 1 280 Rx d 2 AA BB
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        # Should only get the 2 actual message lines
        assert len(messages) == 2
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 0x280

    def test_load_asc_malformed_lines(self, temp_dir: Path):
        """Test that malformed lines are skipped."""
        asc_path = temp_dir / "malformed.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 123 Rx d 4 01 02 03 04
This is not a valid line
0.010000 1 280 Rx d 2 AA BB
Another invalid line with random text
0.020000 1 300 Rx d 1 FF
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        # Should skip invalid lines and only parse valid ones
        assert len(messages) == 3
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 0x280
        assert messages[2].arbitration_id == 0x300

    def test_load_asc_varying_spacing(self, temp_dir: Path):
        """Test parsing with varying whitespace."""
        asc_path = temp_dir / "spacing.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
  0.000000   1   123   Rx   d   4   01 02 03 04
0.010000 1 280 Rx d 8 AA  BB  CC  DD  EE  FF  00  11
    0.020000    1    300    Rx    d    2    12    34
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 3
        assert messages[0].data == bytes([0x01, 0x02, 0x03, 0x04])
        assert messages[1].data == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11])
        assert messages[2].data == bytes([0x12, 0x34])

    def test_load_asc_uppercase_lowercase_hex(self, temp_dir: Path):
        """Test parsing hex data with mixed case."""
        asc_path = temp_dir / "case.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 abc Rx d 4 aa BB cc DD
0.010000 1 DEF Rx d 4 EE ff 00 11
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 2
        assert messages[0].arbitration_id == 0xABC
        assert messages[0].data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

        assert messages[1].arbitration_id == 0xDEF
        assert messages[1].data == bytes([0xEE, 0xFF, 0x00, 0x11])

    def test_load_asc_tx_and_rx(self, temp_dir: Path):
        """Test parsing both Tx and Rx messages."""
        asc_path = temp_dir / "tx_rx.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 123 Rx d 2 01 02
0.010000 1 123 Tx d 2 03 04
0.020000 1 280 Rx d 2 05 06
0.030000 1 280 Tx d 2 07 08
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        # Both Rx and Tx should be parsed
        assert len(messages) == 4

    def test_load_asc_empty_file(self, temp_dir: Path):
        """Test loading an empty ASC file."""
        asc_path = temp_dir / "empty.asc"

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write("")

        messages = load_asc(asc_path)

        assert len(messages) == 0

    def test_load_asc_only_comments(self, temp_dir: Path):
        """Test loading ASC with only comments."""
        asc_path = temp_dir / "only_comments.asc"

        asc_content = """// Comment 1
// Comment 2
date Mon Jul 15 10:30:45 2024
// Comment 3
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 0

    def test_load_asc_file_not_found(self):
        """Test loading non-existent ASC file."""
        with pytest.raises(FileNotFoundError, match="ASC file not found"):
            load_asc("/nonexistent/file.asc")

    def test_load_asc_path_as_string(self, temp_dir: Path):
        """Test loading ASC with path as string."""
        asc_path = temp_dir / "string_path.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 123 Rx d 4 01 02 03 04
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        # Load using string path
        messages = load_asc(str(asc_path))

        assert len(messages) == 1

    def test_load_asc_multiple_channels(self, temp_dir: Path):
        """Test parsing messages from different channels."""
        asc_path = temp_dir / "channels.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 123 Rx d 2 01 02
0.010000 2 123 Rx d 2 03 04
0.020000 3 123 Rx d 2 05 06
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 3
        assert messages[0].channel == 1
        assert messages[1].channel == 2
        assert messages[2].channel == 3

    def test_load_asc_zero_length_data(self, temp_dir: Path):
        """Test parsing messages with zero data bytes."""
        asc_path = temp_dir / "zero_data.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 123 Rx d 0
0.010000 1 280 Rx d 4 01 02 03 04
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 2
        assert messages[0].dlc == 0
        assert messages[0].data == b""
        assert messages[1].dlc == 4

    def test_load_asc_timestamp_precision(self, temp_dir: Path):
        """Test that timestamp precision is preserved."""
        asc_path = temp_dir / "precision.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.123456 1 123 Rx d 2 01 02
1.234567 1 280 Rx d 2 03 04
12.345678 1 300 Rx d 2 05 06
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        assert len(messages) == 3
        assert messages[0].timestamp == pytest.approx(0.123456, abs=1e-6)
        assert messages[1].timestamp == pytest.approx(1.234567, abs=1e-6)
        assert messages[2].timestamp == pytest.approx(12.345678, abs=1e-6)

    def test_load_asc_utf8_encoding(self, temp_dir: Path):
        """Test loading ASC with UTF-8 encoding."""
        asc_path = temp_dir / "utf8.asc"

        # Include UTF-8 characters in comments
        asc_content = """// Test file with UTF-8: Ä Ö Ü ß
date Mon Jul 15 10:30:45 2024
0.000000 1 123 Rx d 4 01 02 03 04
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_asc(asc_path)

        # Should parse successfully despite UTF-8 in comments
        assert len(messages) == 1
