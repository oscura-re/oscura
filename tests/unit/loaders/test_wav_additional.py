"""Additional tests for WAV loader to improve coverage.

This file adds tests for edge cases and error paths not covered
by existing tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from scipy.io import wavfile

from oscura.core.exceptions import LoaderError
from oscura.loaders.wav import load_wav

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestWAVExceptionHandling:
    """Test exception handling in WAV loader."""

    def test_generic_exception_on_read(self, tmp_path: Path) -> None:
        """Test that generic exceptions during read are wrapped in LoaderError."""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()  # Create empty file

        # Patch wavfile.read to raise a generic exception
        with patch("oscura.loaders.wav.wavfile.read") as mock_read:
            mock_read.side_effect = OSError("Generic I/O error")

            with pytest.raises(LoaderError, match="Failed to read WAV file"):
                load_wav(wav_path)

    def test_runtime_error_on_read(self, tmp_path: Path) -> None:
        """Test that RuntimeError during read is wrapped in LoaderError."""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        with patch("oscura.loaders.wav.wavfile.read") as mock_read:
            mock_read.side_effect = RuntimeError("Unexpected runtime error")

            with pytest.raises(LoaderError, match="Failed to read WAV file"):
                load_wav(wav_path)


class TestWAVChannelEdgeCases:
    """Test edge cases for channel handling."""

    def test_channel_with_none_type_on_stereo(self, tmp_path: Path) -> None:
        """Test channel=None defaults to first channel on stereo file."""
        wav_path = tmp_path / "stereo.wav"
        import numpy as np

        data = np.column_stack(
            [np.array([1, 2, 3], dtype=np.int16), np.array([4, 5, 6], dtype=np.int16)]
        )
        wavfile.write(str(wav_path), 44100, data)

        # channel=None should default to first channel
        trace = load_wav(wav_path, channel=None)
        assert trace.metadata.channel == "left"
        assert len(trace.data) == 3
