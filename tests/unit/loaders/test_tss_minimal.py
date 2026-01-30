"""Minimal tests for Tektronix Session File (.tss) Loader.

These tests verify error handling and file validation without requiring
real .wfm test data. Full integration tests will be added when proper
Tektronix session files are available for testing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.tss import load_tss

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestTSSErrorHandling:
    """Error handling and validation tests."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.tss"

        with pytest.raises(LoaderError, match="File not found"):
            load_tss(nonexistent)

    def test_not_zip_file(self, tmp_path: Path) -> None:
        """Test error when file is not a ZIP archive."""
        not_zip = tmp_path / "not_a_zip.tss"
        not_zip.write_text("This is not a ZIP file")

        with pytest.raises(FormatError, match="Not a valid ZIP archive"):
            load_tss(not_zip)

    def test_loader_registration(self) -> None:
        """Test that .tss loader is registered in SUPPORTED_FORMATS."""
        from oscura.loaders import _LOADER_REGISTRY, SUPPORTED_FORMATS

        # Check registration
        assert ".tss" in SUPPORTED_FORMATS
        assert SUPPORTED_FORMATS[".tss"] == "tss"
        assert "tss" in _LOADER_REGISTRY

        # Verify it points to the right loader
        module_name, func_name = _LOADER_REGISTRY["tss"]
        assert module_name == "oscura.loaders.tss"
        assert func_name == "load_tss"
