"""Tests for Rigol WFM loader using mocks to cover RigolWFM paths.

This file tests the RigolWFM library code paths using mocks,
allowing coverage even when the library is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.rigol import (
    _extract_trigger_info,
    _load_with_rigolwfm,
    load_rigol_wfm,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


def _mock_rigol_wfm_module(mock_wfm: MagicMock) -> tuple[Any, Any]:
    """Helper to inject mock RigolWFM module.

    Args:
        mock_wfm: Mock WFM object to return from from_file.

    Returns:
        Tuple of (rigol_module, original_rigol_wfm) for restoration.
    """
    import oscura.loaders.rigol as rigol_module

    # Create mock module
    mock_module = MagicMock()
    mock_module.Wfm.from_file.return_value = mock_wfm

    # Store original and inject mock
    original = getattr(rigol_module, "rigol_wfm", None)
    rigol_module.rigol_wfm = mock_module

    return rigol_module, original


def _restore_rigol_wfm_module(rigol_module: Any, original: Any) -> None:
    """Restore original rigol_wfm module state.

    Args:
        rigol_module: The rigol module to restore.
        original: Original rigol_wfm value (or None).
    """
    if original is None and hasattr(rigol_module, "rigol_wfm"):
        delattr(rigol_module, "rigol_wfm")
    elif original is not None:
        rigol_module.rigol_wfm = original


class TestRigolWFMWithMocks:
    """Test RigolWFM code paths using mocks."""

    def test_load_with_rigolwfm_single_channel_volts(self, tmp_path: Path) -> None:
        """Test loading single channel format with volts attribute."""
        wfm_path = tmp_path / "single.wfm"
        wfm_path.write_bytes(b"dummy")

        # Create mock WFM object with single channel format
        mock_wfm = MagicMock()
        mock_wfm.volts = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_wfm.sample_rate = 1e6
        mock_wfm.volts_per_div = 2.0
        mock_wfm.volt_offset = 0.5

        # Remove channels attribute to simulate single channel format
        type(mock_wfm).channels = property(lambda self: (_ for _ in ()).throw(AttributeError()))

        rigol_module, original = _mock_rigol_wfm_module(mock_wfm)

        try:
            trace = _load_with_rigolwfm(wfm_path, channel=0)

            assert len(trace.data) == 5
            np.testing.assert_array_almost_equal(trace.data, [1.0, 2.0, 3.0, 4.0, 5.0])
            assert trace.metadata.sample_rate == 1e6
            assert trace.metadata.vertical_scale == 2.0
            assert trace.metadata.vertical_offset == 0.5
            assert trace.metadata.channel_name == "CH1"
        finally:
            _restore_rigol_wfm_module(rigol_module, original)

    def test_load_with_rigolwfm_multi_channel_format(self, tmp_path: Path) -> None:
        """Test loading multi-channel format."""
        wfm_path = tmp_path / "multi.wfm"
        wfm_path.write_bytes(b"dummy")

        # Create mock channels
        ch0 = MagicMock()
        ch0.volts = [1.0, 2.0, 3.0]
        ch0.volts_per_div = 1.5
        ch0.volt_offset = 0.1

        ch1 = MagicMock()
        ch1.volts = [4.0, 5.0, 6.0]
        ch1.volts_per_div = 2.5
        ch1.volt_offset = 0.2

        # Create mock WFM object with channels
        mock_wfm = MagicMock()
        mock_wfm.channels = [ch0, ch1]
        mock_wfm.sample_rate = 2e6

        rigol_module, original = _mock_rigol_wfm_module(mock_wfm)

        try:
            # Load channel 0
            trace0 = _load_with_rigolwfm(wfm_path, channel=0)
            np.testing.assert_array_almost_equal(trace0.data, [1.0, 2.0, 3.0])
            assert trace0.metadata.channel_name == "CH1"
            assert trace0.metadata.vertical_scale == 1.5
            assert trace0.metadata.vertical_offset == 0.1

            # Load channel 1
            trace1 = _load_with_rigolwfm(wfm_path, channel=1)
            np.testing.assert_array_almost_equal(trace1.data, [4.0, 5.0, 6.0])
            assert trace1.metadata.channel_name == "CH2"
            assert trace1.metadata.vertical_scale == 2.5
            assert trace1.metadata.vertical_offset == 0.2
        finally:
            _restore_rigol_wfm_module(rigol_module, original)

    def test_load_with_rigolwfm_missing_sample_rate(self, tmp_path: Path) -> None:
        """Test handling when sample_rate attribute is missing."""
        wfm_path = tmp_path / "no_sr.wfm"
        wfm_path.write_bytes(b"dummy")

        mock_wfm = MagicMock()
        mock_wfm.volts = [1.0, 2.0, 3.0]
        # Remove sample_rate attribute
        del mock_wfm.sample_rate

        # Make channels raise AttributeError
        type(mock_wfm).channels = property(lambda self: (_ for _ in ()).throw(AttributeError()))

        rigol_module, original = _mock_rigol_wfm_module(mock_wfm)

        try:
            trace = _load_with_rigolwfm(wfm_path, channel=0)
            # Should default to 1e6
            assert trace.metadata.sample_rate == 1e6
        finally:
            _restore_rigol_wfm_module(rigol_module, original)

    def test_load_with_rigolwfm_no_data_raises_error(self, tmp_path: Path) -> None:
        """Test that missing both channels and volts raises FormatError."""
        wfm_path = tmp_path / "no_data.wfm"
        wfm_path.write_bytes(b"dummy")

        # Create a simple object without channels or volts attributes
        class NoDataWfm:
            pass

        mock_wfm = NoDataWfm()

        rigol_module, original = _mock_rigol_wfm_module(mock_wfm)  # type: ignore[arg-type]

        try:
            with pytest.raises(FormatError, match="No waveform data found"):
                _load_with_rigolwfm(wfm_path, channel=0)
        finally:
            _restore_rigol_wfm_module(rigol_module, original)

    def test_load_with_rigolwfm_exception_wrapping(self, tmp_path: Path) -> None:
        """Test that exceptions are wrapped in LoaderError."""
        wfm_path = tmp_path / "error.wfm"
        wfm_path.write_bytes(b"dummy")

        import oscura.loaders.rigol as rigol_module

        # Create mock that raises exception
        mock_module = MagicMock()
        mock_module.Wfm.from_file.side_effect = Exception("Parse failed")

        original = getattr(rigol_module, "rigol_wfm", None)
        rigol_module.rigol_wfm = mock_module

        try:
            with pytest.raises(LoaderError, match="Failed to load"):
                _load_with_rigolwfm(wfm_path, channel=0)
        finally:
            _restore_rigol_wfm_module(rigol_module, original)

    def test_load_with_rigolwfm_preserves_loader_error(self, tmp_path: Path) -> None:
        """Test that LoaderError is re-raised without wrapping."""
        wfm_path = tmp_path / "error.wfm"
        wfm_path.write_bytes(b"dummy")

        import oscura.loaders.rigol as rigol_module

        original_error = LoaderError("Original error", file_path=str(wfm_path))

        mock_module = MagicMock()
        mock_module.Wfm.from_file.side_effect = original_error

        original = getattr(rigol_module, "rigol_wfm", None)
        rigol_module.rigol_wfm = mock_module

        try:
            with pytest.raises(LoaderError, match="Original error"):
                _load_with_rigolwfm(wfm_path, channel=0)
        finally:
            _restore_rigol_wfm_module(rigol_module, original)

    def test_load_with_rigolwfm_preserves_format_error(self, tmp_path: Path) -> None:
        """Test that FormatError is re-raised without wrapping."""
        wfm_path = tmp_path / "error.wfm"
        wfm_path.write_bytes(b"dummy")

        import oscura.loaders.rigol as rigol_module

        original_error = FormatError("Bad format", file_path=str(wfm_path))

        mock_module = MagicMock()
        mock_module.Wfm.from_file.side_effect = original_error

        original = getattr(rigol_module, "rigol_wfm", None)
        rigol_module.rigol_wfm = mock_module

        try:
            with pytest.raises(FormatError, match="Bad format"):
                _load_with_rigolwfm(wfm_path, channel=0)
        finally:
            _restore_rigol_wfm_module(rigol_module, original)

    def test_dispatch_to_rigolwfm_when_available(self, tmp_path: Path) -> None:
        """Test that load_rigol_wfm uses RigolWFM when available."""
        wfm_path = tmp_path / "test.wfm"
        wfm_path.write_bytes(b"dummy")

        mock_wfm = MagicMock()
        mock_wfm.volts = [1.0, 2.0, 3.0]
        mock_wfm.sample_rate = 1e6

        # Make channels raise AttributeError
        type(mock_wfm).channels = property(lambda self: (_ for _ in ()).throw(AttributeError()))

        import oscura.loaders.rigol as rigol_module

        # Store originals
        original_available = rigol_module.RIGOL_WFM_AVAILABLE
        original_rigol_wfm = getattr(rigol_module, "rigol_wfm", None)

        # Set up mock
        mock_module = MagicMock()
        mock_module.Wfm.from_file.return_value = mock_wfm
        rigol_module.RIGOL_WFM_AVAILABLE = True
        rigol_module.rigol_wfm = mock_module

        try:
            trace = load_rigol_wfm(wfm_path, channel=0)
            assert trace is not None
            assert len(trace.data) == 3
        finally:
            # Restore state
            rigol_module.RIGOL_WFM_AVAILABLE = original_available
            _restore_rigol_wfm_module(rigol_module, original_rigol_wfm)


class TestExtractTriggerInfoDetailed:
    """Detailed tests for trigger info extraction."""

    def test_extract_trigger_info_with_all_fields(self) -> None:
        """Test extraction with all trigger fields present."""
        mock_wfm = MagicMock()
        mock_wfm.trigger_level = 1.5
        mock_wfm.trigger_mode = "normal"
        mock_wfm.trigger_source = "CH2"

        result = _extract_trigger_info(mock_wfm)

        assert result is not None
        assert result["level"] == 1.5
        assert result["mode"] == "normal"
        assert result["source"] == "CH2"
        assert len(result) == 3

    def test_extract_trigger_info_partial_fields(self) -> None:
        """Test extraction with only some fields."""
        mock_wfm = MagicMock()
        mock_wfm.trigger_level = 2.0
        del mock_wfm.trigger_mode
        del mock_wfm.trigger_source

        result = _extract_trigger_info(mock_wfm)

        assert result is not None
        assert "level" in result
        assert result["level"] == 2.0
        assert "mode" not in result
        assert "source" not in result

    def test_extract_trigger_info_no_fields(self) -> None:
        """Test extraction when no trigger fields exist."""
        mock_wfm = MagicMock()
        del mock_wfm.trigger_level
        del mock_wfm.trigger_mode
        del mock_wfm.trigger_source

        result = _extract_trigger_info(mock_wfm)

        assert result is None
