"""Tests for oscura.__main__ CLI interface.

This module tests the command-line interface for downloading and managing
sample data files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.__main__ import (
    download_file,
    download_samples,
    generate_sample_file,
    get_sample_files,
    get_samples_dir,
    list_samples,
    main,
)


class TestGetSamplesDir:
    """Tests for get_samples_dir function."""

    def test_returns_path_object(self) -> None:
        """Should return Path object."""
        result = get_samples_dir()
        assert isinstance(result, Path)

    def test_returns_home_oscura_samples(self) -> None:
        """Should return ~/.oscura/samples/."""
        result = get_samples_dir()
        expected = Path.home() / ".oscura" / "samples"
        assert result == expected

    def test_path_components(self) -> None:
        """Should have correct path components."""
        result = get_samples_dir()
        parts = result.parts
        assert ".oscura" in parts
        assert "samples" in parts


class TestGetSampleFiles:
    """Tests for get_sample_files function."""

    def test_returns_dictionary(self) -> None:
        """Should return dictionary."""
        result = get_sample_files()
        assert isinstance(result, dict)

    def test_contains_expected_files(self) -> None:
        """Should contain standard sample files."""
        result = get_sample_files()
        expected_files = [
            "sine_1khz.csv",
            "square_wave.csv",
            "uart_9600.bin",
            "i2c_capture.bin",
            "spi_flash.bin",
            "noisy_signal.csv",
            "eye_diagram.npz",
        ]
        for filename in expected_files:
            assert filename in result

    def test_file_metadata_structure(self) -> None:
        """Each file should have required metadata fields."""
        result = get_sample_files()
        for metadata in result.values():
            assert "description" in metadata
            assert "format" in metadata
            assert "size" in metadata
            assert "url" in metadata
            assert isinstance(metadata["description"], str)
            assert isinstance(metadata["format"], str)
            assert isinstance(metadata["size"], int)
            assert isinstance(metadata["url"], str)

    def test_csv_format_files(self) -> None:
        """CSV files should have correct format."""
        result = get_sample_files()
        csv_files = [
            "sine_1khz.csv",
            "square_wave.csv",
            "noisy_signal.csv",
        ]
        for filename in csv_files:
            assert result[filename]["format"] == "csv"

    def test_binary_format_files(self) -> None:
        """Binary files should have correct format."""
        result = get_sample_files()
        bin_files = ["uart_9600.bin", "i2c_capture.bin", "spi_flash.bin"]
        for filename in bin_files:
            assert result[filename]["format"] == "binary"

    def test_npz_format_file(self) -> None:
        """NPZ file should have correct format."""
        result = get_sample_files()
        assert result["eye_diagram.npz"]["format"] == "npz"


class TestDownloadFile:
    """Tests for download_file function."""

    def test_successful_download(self, tmp_path: Path) -> None:
        """Should download file successfully."""
        dest = tmp_path / "test.txt"
        mock_data = b"test content"

        with patch("urllib.request.urlopen") as mock_open:
            mock_response = MagicMock()
            mock_response.read.return_value = mock_data
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_open.return_value = mock_response

            result = download_file("https://example.com/file.txt", dest)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == mock_data

    def test_checksum_validation_success(self, tmp_path: Path) -> None:
        """Should validate correct checksum."""
        dest = tmp_path / "test.txt"
        mock_data = b"test"

        import hashlib

        expected_checksum = hashlib.sha256(mock_data).hexdigest()

        with patch("urllib.request.urlopen") as mock_open:
            mock_response = MagicMock()
            mock_response.read.return_value = mock_data
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_open.return_value = mock_response

            result = download_file("https://example.com/file.txt", dest, expected_checksum)

        assert result is True
        assert dest.exists()

    def test_checksum_validation_failure(self, tmp_path: Path) -> None:
        """Should reject incorrect checksum."""
        dest = tmp_path / "test.txt"
        mock_data = b"test"
        wrong_checksum = "0" * 64  # Invalid checksum

        with patch("urllib.request.urlopen") as mock_open:
            mock_response = MagicMock()
            mock_response.read.return_value = mock_data
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_open.return_value = mock_response

            result = download_file("https://example.com/file.txt", dest, wrong_checksum)

        assert result is False
        assert not dest.exists()

    def test_network_error(self, tmp_path: Path) -> None:
        """Should handle network errors gracefully."""
        dest = tmp_path / "test.txt"

        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = download_file("https://example.com/file.txt", dest)

        assert result is False
        assert not dest.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Should create parent directories if needed."""
        dest = tmp_path / "nested" / "dir" / "test.txt"
        mock_data = b"test"

        with patch("urllib.request.urlopen") as mock_open:
            mock_response = MagicMock()
            mock_response.read.return_value = mock_data
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_open.return_value = mock_response

            result = download_file("https://example.com/file.txt", dest)

        assert result is True
        assert dest.exists()
        assert dest.parent.exists()


class TestGenerateSampleFile:
    """Tests for generate_sample_file function."""

    def test_generate_sine_1khz_csv(self, tmp_path: Path) -> None:
        """Should generate sine wave CSV file."""
        dest = tmp_path / "sine_1khz.csv"
        result = generate_sample_file("sine_1khz.csv", dest)

        assert result is True
        assert dest.exists()

        # Verify content
        data = np.loadtxt(dest, delimiter=",", skiprows=1)
        assert data.shape[1] == 2  # time, voltage
        assert len(data) > 0

    def test_generate_square_wave_csv(self, tmp_path: Path) -> None:
        """Should generate square wave CSV file."""
        dest = tmp_path / "square_wave.csv"
        result = generate_sample_file("square_wave.csv", dest)

        assert result is True
        assert dest.exists()

        # Verify content
        data = np.loadtxt(dest, delimiter=",", skiprows=1)
        assert data.shape[1] == 2
        assert len(data) > 0

    def test_generate_noisy_signal_csv(self, tmp_path: Path) -> None:
        """Should generate noisy signal CSV file."""
        dest = tmp_path / "noisy_signal.csv"
        result = generate_sample_file("noisy_signal.csv", dest)

        assert result is True
        assert dest.exists()

        # Verify content
        data = np.loadtxt(dest, delimiter=",", skiprows=1)
        assert data.shape[1] == 2
        assert len(data) > 0

    def test_generate_binary_file(self, tmp_path: Path) -> None:
        """Should generate binary files."""
        dest = tmp_path / "test.bin"
        result = generate_sample_file("test.bin", dest)

        assert result is True
        assert dest.exists()

        # Verify content
        data = np.fromfile(dest, dtype=np.uint8)
        assert len(data) == 1000

    def test_generate_npz_file(self, tmp_path: Path) -> None:
        """Should generate NPZ file with eye diagram data."""
        dest = tmp_path / "eye_diagram.npz"
        result = generate_sample_file("eye_diagram.npz", dest)

        assert result is True
        assert dest.exists()

        # Verify content
        data = np.load(dest)
        assert "time" in data
        assert "signal" in data
        assert "sample_rate" in data

    def test_unknown_file_type(self, tmp_path: Path) -> None:
        """Should handle unknown file types."""
        dest = tmp_path / "unknown.xyz"
        result = generate_sample_file("unknown.xyz", dest)

        assert result is False

    def test_generation_error(self, tmp_path: Path) -> None:
        """Should handle generation errors gracefully."""
        dest = tmp_path / "test.csv"

        with patch("numpy.savetxt", side_effect=Exception("Write error")):
            result = generate_sample_file("sine_1khz.csv", dest)

        assert result is False


class TestDownloadSamples:
    """Tests for download_samples function."""

    def test_skip_existing_files(self, tmp_path: Path) -> None:
        """Should skip files that already exist."""
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            # Create ALL sample files as existing
            sample_files = get_sample_files()
            for filename in sample_files:
                (tmp_path / filename).write_text("existing")

            with patch("oscura.__main__.download_file") as mock_download:
                result = download_samples(force=False, generate=False)

                # Should not attempt download for any file
                mock_download.assert_not_called()
                assert result == 0

    def test_force_redownload(self, tmp_path: Path) -> None:
        """Should redownload even if files exist when force=True."""
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            # Create existing file
            (tmp_path / "sine_1khz.csv").write_text("existing")

            with patch("oscura.__main__.download_file", return_value=False):
                with patch("oscura.__main__.generate_sample_file", return_value=True):
                    result = download_samples(force=True, generate=True)

                    # Should succeed with generation fallback
                    assert result == 0

    def test_fallback_to_generation(self, tmp_path: Path) -> None:
        """Should fall back to generation if download fails."""
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=False):
                with patch("oscura.__main__.generate_sample_file", return_value=True):
                    result = download_samples(force=True, generate=True)

                    assert result == 0

    def test_no_generation_fallback(self, tmp_path: Path) -> None:
        """Should not generate if generate=False."""
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=False):
                with patch("oscura.__main__.generate_sample_file") as mock_gen:
                    result = download_samples(force=True, generate=False)

                    # Should fail without calling generation
                    mock_gen.assert_not_called()
                    assert result == 1


class TestListSamples:
    """Tests for list_samples function."""

    def test_list_samples_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Should list all available samples."""
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            # Create one file
            (tmp_path / "sine_1khz.csv").write_text("data")

            result = list_samples()

            assert result == 0

            captured = capsys.readouterr()
            assert "sine_1khz.csv" in captured.out
            assert "[EXISTS]" in captured.out
            assert "[NOT DOWNLOADED]" in captured.out


class TestMain:
    """Tests for main CLI entry point."""

    def test_download_samples_command(self) -> None:
        """Should execute download_samples command."""
        with patch("sys.argv", ["oscura", "download_samples"]):
            with patch("oscura.__main__.download_samples", return_value=0) as mock_dl:
                result = main()

                mock_dl.assert_called_once()
                assert result == 0

    def test_download_alias(self) -> None:
        """Should accept 'download' as alias."""
        with patch("sys.argv", ["oscura", "download"]):
            with patch("oscura.__main__.download_samples", return_value=0) as mock_dl:
                result = main()

                mock_dl.assert_called_once()

    def test_list_samples_command(self) -> None:
        """Should execute list_samples command."""
        with patch("sys.argv", ["oscura", "list_samples"]):
            with patch("oscura.__main__.list_samples", return_value=0) as mock_list:
                result = main()

                mock_list.assert_called_once()
                assert result == 0

    def test_list_alias(self) -> None:
        """Should accept 'list' as alias."""
        with patch("sys.argv", ["oscura", "list"]):
            with patch("oscura.__main__.list_samples", return_value=0) as mock_list:
                result = main()

                mock_list.assert_called_once()

    def test_version_command(self) -> None:
        """Should display version information."""
        with patch("sys.argv", ["oscura", "version"]):
            with patch("oscura.__version__", "1.0.0"):
                result = main()

                assert result == 0

    def test_version_command_no_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Should handle missing version gracefully."""
        # Save original version attribute state
        import oscura

        had_version = hasattr(oscura, "__version__")
        original_version = getattr(oscura, "__version__", None)

        # Remove __version__ to simulate ImportError
        if had_version:
            delattr(oscura, "__version__")

        try:
            with patch("sys.argv", ["oscura", "version"]):
                result = main()

                assert result == 0
                captured = capsys.readouterr()
                assert "unknown" in captured.out.lower()
        finally:
            # Restore original state
            if had_version and original_version is not None:
                oscura.__version__ = original_version

    def test_no_command_shows_help(self) -> None:
        """Should show help when no command given."""
        with patch("sys.argv", ["oscura"]):
            result = main()

            assert result == 0

    def test_force_flag(self) -> None:
        """Should pass force flag to download_samples."""
        with patch("sys.argv", ["oscura", "download_samples", "--force"]):
            with patch("oscura.__main__.download_samples", return_value=0) as mock_dl:
                main()

                mock_dl.assert_called_once_with(force=True, generate=True)

    def test_no_generate_flag(self) -> None:
        """Should pass no-generate flag to download_samples."""
        with patch("sys.argv", ["oscura", "download_samples", "--no-generate"]):
            with patch("oscura.__main__.download_samples", return_value=0) as mock_dl:
                main()

                mock_dl.assert_called_once_with(force=False, generate=False)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_samples_directory(self, tmp_path: Path) -> None:
        """Should handle empty samples directory."""
        samples_dir = tmp_path / "empty"
        samples_dir.mkdir()

        assert samples_dir.exists()
        assert list(samples_dir.iterdir()) == []

    def test_permission_error_on_write(self, tmp_path: Path) -> None:
        """Should handle permission errors gracefully."""
        dest = tmp_path / "test.txt"

        with patch("pathlib.Path.write_bytes", side_effect=PermissionError):
            with patch("urllib.request.urlopen") as mock_open:
                mock_response = MagicMock()
                mock_response.read.return_value = b"data"
                mock_response.__enter__.return_value = mock_response
                mock_open.return_value = mock_response

                result = download_file("https://example.com/file", dest)

                assert result is False

    def test_generate_creates_parent_directory(self, tmp_path: Path) -> None:
        """Should create parent directories when generating."""
        dest = tmp_path / "nested" / "dir" / "test.bin"

        result = generate_sample_file("test.bin", dest)

        assert result is True
        assert dest.exists()
        assert dest.parent.exists()
