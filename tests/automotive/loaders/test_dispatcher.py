"""Tests for automotive log format detection and dispatcher.

Tests the automatic format detection and routing to appropriate loaders.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oscura.automotive.loaders.dispatcher import detect_format, load_automotive_log

pytestmark = pytest.mark.loader


@pytest.mark.unit
@pytest.mark.loader
class TestFormatDetection:
    """Tests for format detection."""

    def test_detect_blf_by_extension(self, temp_dir: Path):
        """Test BLF detection by file extension."""
        blf_path = temp_dir / "test.blf"
        blf_path.touch()

        assert detect_format(blf_path) == "blf"

    def test_detect_asc_by_extension(self, temp_dir: Path):
        """Test ASC detection by file extension."""
        asc_path = temp_dir / "test.asc"
        asc_path.touch()

        assert detect_format(asc_path) == "asc"

    def test_detect_mdf_by_extension(self, temp_dir: Path):
        """Test MDF detection by file extension."""
        mdf_path = temp_dir / "test.mdf"
        mdf_path.touch()

        assert detect_format(mdf_path) == "mdf"

    def test_detect_mf4_by_extension(self, temp_dir: Path):
        """Test MF4 detection by file extension."""
        mf4_path = temp_dir / "test.mf4"
        mf4_path.touch()

        assert detect_format(mf4_path) == "mdf"

    def test_detect_dat_by_extension(self, temp_dir: Path):
        """Test DAT detection (MDF format) by file extension."""
        dat_path = temp_dir / "test.dat"
        dat_path.touch()

        assert detect_format(dat_path) == "mdf"

    def test_detect_csv_by_extension(self, temp_dir: Path):
        """Test CSV detection by file extension."""
        csv_path = temp_dir / "test.csv"
        csv_path.touch()

        assert detect_format(csv_path) == "csv"

    def test_detect_pcap_by_extension(self, temp_dir: Path):
        """Test PCAP detection by file extension."""
        pcap_path = temp_dir / "test.pcap"
        pcap_path.touch()

        assert detect_format(pcap_path) == "pcap"

    def test_detect_pcapng_by_extension(self, temp_dir: Path):
        """Test PCAPNG detection by file extension."""
        pcapng_path = temp_dir / "test.pcapng"
        pcapng_path.touch()

        assert detect_format(pcapng_path) == "pcap"

    def test_detect_blf_by_magic(self, temp_dir: Path):
        """Test BLF detection by file magic bytes."""
        file_path = temp_dir / "unknown.bin"

        # Write BLF magic bytes
        with open(file_path, "wb") as f:
            f.write(b"LOGG" + b"\x00" * 100)

        assert detect_format(file_path) == "blf"

    def test_detect_mdf_by_magic(self, temp_dir: Path):
        """Test MDF detection by file magic bytes."""
        file_path = temp_dir / "unknown.bin"

        # Write MDF magic bytes
        with open(file_path, "wb") as f:
            f.write(b"MDF     " + b"\x00" * 100)

        assert detect_format(file_path) == "mdf"

    def test_detect_pcap_by_magic(self, temp_dir: Path):
        """Test PCAP detection by file magic bytes."""
        file_path = temp_dir / "unknown.bin"

        # Write PCAP magic bytes (little-endian)
        with open(file_path, "wb") as f:
            f.write(b"\xd4\xc3\xb2\xa1" + b"\x00" * 100)

        assert detect_format(file_path) == "pcap"

    def test_detect_asc_by_content(self, temp_dir: Path):
        """Test ASC detection by file content."""
        file_path = temp_dir / "unknown.txt"

        # Write ASC-style content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("date Mon Jul 15 10:30:45 2024\n")

        assert detect_format(file_path) == "asc"

    def test_detect_csv_by_content(self, temp_dir: Path):
        """Test CSV detection by file content."""
        file_path = temp_dir / "unknown.txt"

        # Write CSV header with CAN-related columns
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("timestamp,can_id,data\n")

        assert detect_format(file_path) == "csv"

    def test_detect_csv_with_id_column(self, temp_dir: Path):
        """Test CSV detection with 'id' column."""
        file_path = temp_dir / "unknown.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("timestamp,id,data\n")

        assert detect_format(file_path) == "csv"

    def test_detect_unknown_format(self, temp_dir: Path):
        """Test detection returns 'unknown' for unrecognized files."""
        file_path = temp_dir / "random.bin"

        # Write random data
        with open(file_path, "wb") as f:
            f.write(b"\x12\x34\x56\x78" * 50)

        assert detect_format(file_path) == "unknown"

    def test_detect_format_path_as_string(self, temp_dir: Path):
        """Test format detection with path as string."""
        blf_path = temp_dir / "test.blf"
        blf_path.touch()

        assert detect_format(str(blf_path)) == "blf"


@pytest.mark.unit
@pytest.mark.loader
class TestLoadAutomotiveLog:
    """Tests for load_automotive_log dispatcher."""

    def test_load_automotive_log_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_automotive_log("/nonexistent/file.blf")

    def test_load_automotive_log_unknown_format(self, temp_dir: Path):
        """Test loading file with unknown format."""
        file_path = temp_dir / "unknown.bin"

        # Write unrecognized data
        with open(file_path, "wb") as f:
            f.write(b"\xff" * 100)

        with pytest.raises(ValueError, match="Unknown or unsupported file format"):
            load_automotive_log(file_path)

    @patch("oscura.automotive.loaders.blf.load_blf")
    def test_load_automotive_log_blf(self, mock_load_blf, temp_dir: Path, expected_message_list):
        """Test dispatching to BLF loader."""
        blf_path = temp_dir / "test.blf"
        blf_path.touch()

        mock_load_blf.return_value = expected_message_list

        messages = load_automotive_log(blf_path)

        mock_load_blf.assert_called_once_with(blf_path)
        assert len(messages) == len(expected_message_list)

    @patch("oscura.automotive.loaders.asc.load_asc")
    def test_load_automotive_log_asc(self, mock_load_asc, temp_dir: Path, expected_message_list):
        """Test dispatching to ASC loader."""
        asc_path = temp_dir / "test.asc"
        asc_path.touch()

        mock_load_asc.return_value = expected_message_list

        messages = load_automotive_log(asc_path)

        mock_load_asc.assert_called_once_with(asc_path)
        assert len(messages) == len(expected_message_list)

    @patch("oscura.automotive.loaders.mdf.load_mdf")
    def test_load_automotive_log_mdf(self, mock_load_mdf, temp_dir: Path, expected_message_list):
        """Test dispatching to MDF loader."""
        mdf_path = temp_dir / "test.mf4"
        mdf_path.touch()

        mock_load_mdf.return_value = expected_message_list

        messages = load_automotive_log(mdf_path)

        mock_load_mdf.assert_called_once_with(mdf_path)
        assert len(messages) == len(expected_message_list)

    @patch("oscura.automotive.loaders.csv_can.load_csv_can")
    def test_load_automotive_log_csv(
        self, mock_load_csv_can, temp_dir: Path, expected_message_list
    ):
        """Test dispatching to CSV loader."""
        csv_path = temp_dir / "test.csv"
        csv_path.touch()

        mock_load_csv_can.return_value = expected_message_list

        messages = load_automotive_log(csv_path)

        mock_load_csv_can.assert_called_once_with(csv_path)
        assert len(messages) == len(expected_message_list)

    def test_load_automotive_log_pcap_not_implemented(self, temp_dir: Path):
        """Test that PCAP loading with empty file raises ValueError."""
        pytest.importorskip("scapy", reason="scapy required for PCAP tests")

        pcap_path = temp_dir / "test.pcap"

        # Write minimal valid PCAP header (24 bytes) but no packets
        with open(pcap_path, "wb") as f:
            # PCAP global header (24 bytes)
            f.write(b"\xd4\xc3\xb2\xa1")  # Magic number (little-endian)
            f.write(b"\x02\x00\x04\x00")  # Version 2.4
            f.write(b"\x00\x00\x00\x00")  # Thiszone (GMT)
            f.write(b"\x00\x00\x00\x00")  # Sigfigs
            f.write(b"\xff\xff\x00\x00")  # Snaplen (65535)
            f.write(b"\xe3\x00\x00\x00")  # Network (227 = Linux SLL)

        # Should raise ValueError because no CAN frames found
        with pytest.raises(ValueError, match="No CAN frames found in PCAP file"):
            load_automotive_log(pcap_path)

    def test_load_automotive_log_path_as_string(self, temp_dir: Path):
        """Test loading with path as string."""
        asc_path = temp_dir / "test.asc"

        # Create simple ASC file
        with open(asc_path, "w", encoding="utf-8") as f:
            f.write("date Mon Jul 15 10:30:45 2024\n")
            f.write("0.000000 1 123 Rx d 4 01 02 03 04\n")

        # Load using string path
        messages = load_automotive_log(str(asc_path))

        assert len(messages) == 1

    def test_load_automotive_log_mixed_format_detection(self, temp_dir: Path):
        """Test that content-based detection works when extension is ambiguous."""
        # Create file with .log extension but BLF magic
        file_path = temp_dir / "test.log"

        with open(file_path, "wb") as f:
            f.write(b"LOGG" + b"\x00" * 100)

        # Should detect as BLF based on magic bytes
        assert detect_format(file_path) == "blf"

    def test_load_automotive_log_asc_with_can_keyword(self, temp_dir: Path):
        """Test ASC detection with CAN keyword in header."""
        file_path = temp_dir / "test.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("CAN log file\n")
            f.write("0.000000 1 123 Rx d 4 01 02 03 04\n")

        assert detect_format(file_path) == "asc"

    def test_load_automotive_log_integration_asc(self, temp_dir: Path):
        """Integration test: Load actual ASC file through dispatcher."""
        asc_path = temp_dir / "integration.asc"

        asc_content = """date Mon Jul 15 10:30:45 2024
0.000000 1 123 Rx d 4 01 02 03 04
0.010000 1 280 Rx d 4 AA BB CC DD
"""

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(asc_content)

        messages = load_automotive_log(asc_path)

        assert len(messages) == 2
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 0x280

    def test_load_automotive_log_integration_csv(self, temp_dir: Path):
        """Integration test: Load actual CSV file through dispatcher."""
        csv_path = temp_dir / "integration.csv"

        csv_content = """timestamp,id,data
0.000000,0x123,01020304
0.010000,0x280,AABBCCDD
"""

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        messages = load_automotive_log(csv_path)

        assert len(messages) == 2
        assert messages[0].arbitration_id == 0x123
        assert messages[1].arbitration_id == 0x280
