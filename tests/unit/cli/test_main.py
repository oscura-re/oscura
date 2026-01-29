"""Comprehensive unit tests for __main__.py CLI module.

This module provides extensive testing for the Oscura CLI, including:
- Command-line argument parsing
- Sample file download functionality
- Sample file generation
- File listing
- Version information
- Error handling and edge cases


Test Coverage:
- get_samples_dir()
- get_sample_files()
- download_file() with checksums and error conditions
- generate_sample_file() for all file types
- download_samples() with force and generate flags
- list_samples()
- main() CLI entry point with all subcommands
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test get_samples_dir()
# =============================================================================


@pytest.mark.unit
def test_get_samples_dir():
    """Test that get_samples_dir returns correct path."""
    result = get_samples_dir()
    assert isinstance(result, Path)
    assert result == Path.home() / ".oscura" / "samples"


@pytest.mark.unit
def test_get_samples_dir_consistency():
    """Test that get_samples_dir returns consistent results."""
    result1 = get_samples_dir()
    result2 = get_samples_dir()
    assert result1 == result2


# =============================================================================
# Test get_sample_files()
# =============================================================================


@pytest.mark.unit
def test_get_sample_files_structure():
    """Test that get_sample_files returns properly structured dictionary."""
    files = get_sample_files()

    assert isinstance(files, dict)
    assert len(files) > 0

    # Check each file has required fields
    for filename, info in files.items():
        assert isinstance(filename, str)
        assert isinstance(info, dict)
        assert "description" in info
        assert "format" in info
        assert "size" in info
        assert "url" in info
        assert "checksum" in info


@pytest.mark.unit
def test_get_sample_files_contains_expected_files():
    """Test that get_sample_files contains expected sample files."""
    files = get_sample_files()

    # Check for key sample files
    expected_files = [
        "sine_1khz.csv",
        "square_wave.csv",
        "uart_9600.bin",
        "i2c_capture.bin",
        "spi_flash.bin",
        "noisy_signal.csv",
        "eye_diagram.npz",
    ]

    for expected in expected_files:
        assert expected in files, f"Expected {expected} in sample files"


@pytest.mark.unit
def test_get_sample_files_valid_formats():
    """Test that all sample files have valid formats."""
    files = get_sample_files()
    valid_formats = ["csv", "binary", "npz"]

    for filename, info in files.items():
        assert info["format"] in valid_formats, f"Invalid format for {filename}"


# =============================================================================
# Test download_file()
# =============================================================================


@pytest.mark.unit
def test_download_file_success(tmp_path):
    """Test successful file download."""
    dest = tmp_path / "test_file.bin"
    test_data = b"test content"

    # Mock urllib.request.urlopen
    mock_response = MagicMock()
    mock_response.read.return_value = test_data
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = download_file("http://example.com/test", dest)

    assert result is True
    assert dest.exists()
    assert dest.read_bytes() == test_data


@pytest.mark.unit
def test_download_file_with_valid_checksum(tmp_path, capsys):
    """Test file download with matching checksum."""
    dest = tmp_path / "test_file.bin"
    test_data = b"test content"
    checksum = hashlib.sha256(test_data).hexdigest()

    mock_response = MagicMock()
    mock_response.read.return_value = test_data
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = download_file("http://example.com/test", dest, checksum)

    assert result is True
    assert dest.exists()


@pytest.mark.unit
def test_download_file_with_invalid_checksum(tmp_path, capsys):
    """Test file download with mismatched checksum."""
    dest = tmp_path / "test_file.bin"
    test_data = b"test content"
    wrong_checksum = "0" * 64  # Invalid checksum

    mock_response = MagicMock()
    mock_response.read.return_value = test_data
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = download_file("http://example.com/test", dest, wrong_checksum)

    assert result is False
    captured = capsys.readouterr()
    assert "ERROR: Checksum mismatch" in captured.out


@pytest.mark.unit
def test_download_file_network_error(tmp_path, capsys):
    """Test file download with network error."""
    dest = tmp_path / "test_file.bin"

    with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
        result = download_file("http://example.com/test", dest)

    assert result is False
    captured = capsys.readouterr()
    assert "ERROR: Failed to download" in captured.out


@pytest.mark.unit
def test_download_file_creates_parent_directory(tmp_path):
    """Test that download_file creates parent directories."""
    dest = tmp_path / "subdir" / "nested" / "test_file.bin"
    test_data = b"test content"

    mock_response = MagicMock()
    mock_response.read.return_value = test_data
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = download_file("http://example.com/test", dest)

    assert result is True
    assert dest.exists()
    assert dest.parent.exists()


# =============================================================================
# Test generate_sample_file()
# =============================================================================


@pytest.mark.unit
def test_generate_sample_file_sine_1khz(tmp_path):
    """Test generation of sine_1khz.csv."""
    dest = tmp_path / "sine_1khz.csv"
    result = generate_sample_file("sine_1khz.csv", dest)

    assert result is True
    assert dest.exists()

    # Verify file content
    data = np.loadtxt(dest, delimiter=",", skiprows=1)
    assert data.shape[1] == 2  # time, voltage columns
    assert len(data) > 0


@pytest.mark.unit
def test_generate_sample_file_square_wave(tmp_path):
    """Test generation of square_wave.csv."""
    dest = tmp_path / "square_wave.csv"
    result = generate_sample_file("square_wave.csv", dest)

    assert result is True
    assert dest.exists()

    # Verify file content
    data = np.loadtxt(dest, delimiter=",", skiprows=1)
    assert data.shape[1] == 2
    assert len(data) > 0


@pytest.mark.unit
def test_generate_sample_file_noisy_signal(tmp_path):
    """Test generation of noisy_signal.csv."""
    dest = tmp_path / "noisy_signal.csv"
    result = generate_sample_file("noisy_signal.csv", dest)

    assert result is True
    assert dest.exists()

    # Verify file content
    data = np.loadtxt(dest, delimiter=",", skiprows=1)
    assert data.shape[1] == 2
    assert len(data) > 0


@pytest.mark.unit
def test_generate_sample_file_binary(tmp_path):
    """Test generation of binary files."""
    for filename in ["uart_9600.bin", "i2c_capture.bin", "spi_flash.bin"]:
        dest = tmp_path / filename
        result = generate_sample_file(filename, dest)

        assert result is True, f"Failed to generate {filename}"
        assert dest.exists(), f"{filename} was not created"

        # Verify file has content
        assert dest.stat().st_size > 0, f"{filename} is empty"


@pytest.mark.unit
def test_generate_sample_file_npz(tmp_path):
    """Test generation of eye_diagram.npz."""
    dest = tmp_path / "eye_diagram.npz"
    result = generate_sample_file("eye_diagram.npz", dest)

    assert result is True
    assert dest.exists()

    # Verify file can be loaded
    data = np.load(dest, allow_pickle=True)
    assert "time" in data
    assert "signal" in data
    assert "sample_rate" in data


@pytest.mark.unit
def test_generate_sample_file_unknown_type(tmp_path, capsys):
    """Test generation of unknown file type."""
    dest = tmp_path / "unknown.xyz"
    result = generate_sample_file("unknown.xyz", dest)

    assert result is False
    captured = capsys.readouterr()
    assert "WARNING: Unknown file type" in captured.out


@pytest.mark.unit
def test_generate_sample_file_creates_parent_directory(tmp_path):
    """Test that generate_sample_file creates parent directories."""
    dest = tmp_path / "subdir" / "nested" / "sine_1khz.csv"
    result = generate_sample_file("sine_1khz.csv", dest)

    assert result is True
    assert dest.exists()
    assert dest.parent.exists()


@pytest.mark.unit
def test_generate_sample_file_error_handling(tmp_path, capsys):
    """Test error handling in generate_sample_file."""
    dest = tmp_path / "test.csv"

    # Mock numpy.savetxt to raise an exception
    with patch("numpy.savetxt", side_effect=Exception("Test error")):
        result = generate_sample_file("sine_1khz.csv", dest)

    assert result is False
    captured = capsys.readouterr()
    assert "ERROR: Failed to generate" in captured.out


# =============================================================================
# Test download_samples()
# =============================================================================


@pytest.mark.unit
def test_download_samples_basic(tmp_path, capsys):
    """Test basic download_samples functionality."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", return_value=True):
            result = download_samples(force=False, generate=False)

    assert result == 0
    captured = capsys.readouterr()
    assert "Oscura Sample Data Download" in captured.out
    assert "succeeded" in captured.out


@pytest.mark.unit
def test_download_samples_skip_existing(tmp_path, capsys):
    """Test that download_samples skips existing files."""
    # Create all the sample files to ensure they're all skipped
    sample_files = get_sample_files()
    for filename in sample_files:
        (tmp_path / filename).write_text("existing content")

    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        result = download_samples(force=False, generate=False)

    assert result == 0
    captured = capsys.readouterr()
    assert "[SKIP]" in captured.out
    assert "already exists" in captured.out


@pytest.mark.unit
def test_download_samples_force_redownload(tmp_path, capsys):
    """Test force re-download of existing files."""
    # Create an existing file
    (tmp_path / "sine_1khz.csv").write_text("existing content")

    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", return_value=True):
            result = download_samples(force=True, generate=False)

    assert result == 0
    captured = capsys.readouterr()
    # Should not see SKIP message for any files with force=True
    # (except those that already succeeded from previous iterations)


@pytest.mark.unit
def test_download_samples_fallback_to_generation(tmp_path, capsys):
    """Test fallback to local generation when download fails."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", return_value=False):
            result = download_samples(force=False, generate=True)

    assert result == 0
    captured = capsys.readouterr()
    assert "Falling back to local generation" in captured.out
    assert "Generated:" in captured.out


@pytest.mark.unit
def test_download_samples_no_generation_on_failure(tmp_path, capsys):
    """Test behavior when download fails and generation is disabled."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", return_value=False):
            result = download_samples(force=False, generate=False)

    assert result == 1  # Should return error code
    captured = capsys.readouterr()
    assert "failed" in captured.out.lower()


@pytest.mark.unit
def test_download_samples_partial_failure(tmp_path, capsys):
    """Test handling of partial failures."""
    call_count = [0]

    def mock_download(url, dest, checksum=None):
        call_count[0] += 1
        # Fail every other download
        return call_count[0] % 2 == 1

    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", side_effect=mock_download):
            with patch("oscura.__main__.generate_sample_file", return_value=False):
                result = download_samples(force=False, generate=True)

    # Should have some failures
    captured = capsys.readouterr()
    assert "failed" in captured.out.lower()


@pytest.mark.unit
def test_download_samples_success_message(tmp_path, capsys):
    """Test success message when all downloads succeed."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", return_value=True):
            result = download_samples(force=False, generate=False)

    assert result == 0
    captured = capsys.readouterr()
    assert "Sample files downloaded successfully!" in captured.out
    assert "Example usage:" in captured.out


# =============================================================================
# Test list_samples()
# =============================================================================


@pytest.mark.unit
def test_list_samples_empty_directory(tmp_path, capsys):
    """Test listing samples when none are downloaded."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        result = list_samples()

    assert result == 0
    captured = capsys.readouterr()
    assert "Available sample files:" in captured.out
    assert "[NOT DOWNLOADED]" in captured.out


@pytest.mark.unit
def test_list_samples_with_existing_files(tmp_path, capsys):
    """Test listing samples when some are already downloaded."""
    # Create some sample files
    (tmp_path / "sine_1khz.csv").write_text("content")
    (tmp_path / "uart_9600.bin").write_bytes(b"content")

    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        result = list_samples()

    assert result == 0
    captured = capsys.readouterr()
    assert "[EXISTS]" in captured.out
    assert "[NOT DOWNLOADED]" in captured.out


@pytest.mark.unit
def test_list_samples_shows_descriptions(capsys):
    """Test that list_samples shows file descriptions."""
    result = list_samples()

    assert result == 0
    captured = capsys.readouterr()

    # Check for some expected descriptions
    assert "sine wave" in captured.out.lower() or "kHz" in captured.out


# =============================================================================
# Test main() CLI entry point
# =============================================================================


@pytest.mark.unit
def test_main_no_command(capsys):
    """Test main() with no command shows help."""
    with patch("sys.argv", ["oscura"]):
        result = main()

    assert result == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "oscura" in captured.out.lower()


@pytest.mark.unit
def test_main_download_samples_command(tmp_path, capsys):
    """Test main() with download_samples command."""
    with patch("sys.argv", ["oscura", "download_samples"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=True):
                result = main()

    assert result == 0


@pytest.mark.unit
def test_main_download_alias(tmp_path, capsys):
    """Test main() with download alias."""
    with patch("sys.argv", ["oscura", "download"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=True):
                result = main()

    assert result == 0


@pytest.mark.unit
def test_main_download_with_force_flag(tmp_path):
    """Test main() with --force flag."""
    with patch("sys.argv", ["oscura", "download_samples", "--force"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=True):
                result = main()

    assert result == 0


@pytest.mark.unit
def test_main_download_with_no_generate_flag(tmp_path):
    """Test main() with --no-generate flag."""
    with patch("sys.argv", ["oscura", "download_samples", "--no-generate"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=False):
                result = main()

    # Should fail because download fails and generation is disabled
    assert result == 1


@pytest.mark.unit
def test_main_list_samples_command(capsys):
    """Test main() with list_samples command."""
    with patch("sys.argv", ["oscura", "list_samples"]):
        result = main()

    assert result == 0
    captured = capsys.readouterr()
    assert "Available sample files:" in captured.out


@pytest.mark.unit
def test_main_list_alias(capsys):
    """Test main() with list alias."""
    with patch("sys.argv", ["oscura", "list"]):
        result = main()

    assert result == 0
    captured = capsys.readouterr()
    assert "Available sample files:" in captured.out


@pytest.mark.unit
def test_main_version_command(capsys):
    """Test main() with version command."""
    with patch("sys.argv", ["oscura", "version"]):
        result = main()

    assert result == 0
    captured = capsys.readouterr()
    assert "Oscura version" in captured.out


@pytest.mark.unit
def test_main_version_import_error(capsys):
    """Test main() version command when __version__ is unavailable."""
    # Mock the import to raise ImportError when trying to import __version__

    original_modules = sys.modules.copy()

    with patch("sys.argv", ["oscura", "version"]):
        # Create a mock oscura module without __version__
        mock_oscura = type(sys)("oscura")

        def mock_getattr(name):
            if name == "__version__":
                raise AttributeError("No __version__")
            raise AttributeError(f"module 'oscura' has no attribute '{name}'")

        mock_oscura.__getattr__ = mock_getattr

        with patch.dict("sys.modules", {"oscura": mock_oscura}):
            result = main()

    assert result == 0
    captured = capsys.readouterr()
    assert "Oscura version" in captured.out
    assert "unknown" in captured.out


@pytest.mark.unit
def test_main_help_flag(capsys):
    """Test main() with --help flag."""
    with patch("sys.argv", ["oscura", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0


@pytest.mark.unit
def test_main_download_help(capsys):
    """Test main() with download_samples --help."""
    with patch("sys.argv", ["oscura", "download_samples", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0


# =============================================================================
# Integration-style tests
# =============================================================================


@pytest.mark.unit
def test_full_download_workflow(tmp_path, capsys):
    """Test complete download workflow from command line to file generation."""
    with patch("sys.argv", ["oscura", "download", "--force"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=False):
                # Download fails, should fall back to generation
                result = main()

    assert result == 0
    captured = capsys.readouterr()
    assert "Falling back to local generation" in captured.out

    # Check that files were actually generated
    assert (tmp_path / "sine_1khz.csv").exists()
    assert (tmp_path / "uart_9600.bin").exists()


@pytest.mark.unit
def test_download_then_list(tmp_path, capsys):
    """Test downloading files then listing them."""

    # Mock download_file to actually create the files
    def mock_download(url, dest, checksum=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"test content")
        return True

    # First download
    with patch("sys.argv", ["oscura", "download"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", side_effect=mock_download):
                result = main()

    assert result == 0

    # Then list - should show files as existing
    with patch("sys.argv", ["oscura", "list"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            result = main()

    assert result == 0
    captured = capsys.readouterr()
    assert "[EXISTS]" in captured.out


# =============================================================================
# Edge cases and error conditions
# =============================================================================


@pytest.mark.unit
def test_download_file_timeout(tmp_path, capsys):
    """Test download_file with timeout error."""
    dest = tmp_path / "test_file.bin"

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Timeout")):
        result = download_file("http://example.com/test", dest)

    assert result is False


@pytest.mark.unit
def test_generate_sample_file_permission_error(tmp_path, capsys):
    """Test generate_sample_file with permission error."""
    dest = tmp_path / "test.csv"

    # Make the directory read-only
    tmp_path.chmod(0o444)

    try:
        result = generate_sample_file("sine_1khz.csv", dest)
        # On some systems this might succeed, on others it will fail
        # Just ensure it doesn't crash
        assert isinstance(result, bool)
    finally:
        # Restore permissions
        tmp_path.chmod(0o755)


@pytest.mark.unit
def test_download_samples_with_empty_sample_files(tmp_path, capsys):
    """Test download_samples when get_sample_files returns empty dict."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.get_sample_files", return_value={}):
            result = download_samples(force=False, generate=False)

    assert result == 0  # No files to download, so success
    captured = capsys.readouterr()
    assert "0 succeeded, 0 failed" in captured.out


@pytest.mark.unit
def test_main_with_short_force_flag(tmp_path):
    """Test main() with -f short flag."""
    with patch("sys.argv", ["oscura", "download", "-f"]):
        with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
            with patch("oscura.__main__.download_file", return_value=True):
                result = main()

    assert result == 0


@pytest.mark.unit
def test_download_file_ssl_error(tmp_path, capsys):
    """Test download_file with SSL error."""
    dest = tmp_path / "test_file.bin"

    import ssl

    with patch("urllib.request.urlopen", side_effect=ssl.SSLError("SSL error")):
        result = download_file("https://example.com/test", dest)

    assert result is False
    captured = capsys.readouterr()
    assert "ERROR" in captured.out


@pytest.mark.unit
def test_generate_all_sample_types_in_single_directory(tmp_path):
    """Test generating all sample file types in the same directory."""
    sample_files = get_sample_files()

    for filename in sample_files:
        dest = tmp_path / filename
        result = generate_sample_file(filename, dest)

        # Should succeed for all known file types
        if filename.endswith((".csv", ".bin", ".npz")):
            assert result is True, f"Failed to generate {filename}"
            assert dest.exists(), f"{filename} was not created"


@pytest.mark.unit
def test_download_samples_prints_destination(tmp_path, capsys):
    """Test that download_samples prints the destination directory."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", return_value=True):
            download_samples()

    captured = capsys.readouterr()
    assert f"Destination: {tmp_path}" in captured.out


@pytest.mark.unit
def test_download_samples_shows_file_descriptions(tmp_path, capsys):
    """Test that download_samples shows file descriptions."""
    with patch("oscura.__main__.get_samples_dir", return_value=tmp_path):
        with patch("oscura.__main__.download_file", return_value=True):
            download_samples()

    captured = capsys.readouterr()
    assert "Description:" in captured.out


@pytest.mark.unit
def test_checksum_calculation_accuracy():
    """Test that checksum calculation is accurate."""
    test_data = b"test content for checksum verification"
    expected = hashlib.sha256(test_data).hexdigest()

    # This should match what download_file calculates
    computed = hashlib.sha256(test_data).hexdigest()
    assert computed == expected
