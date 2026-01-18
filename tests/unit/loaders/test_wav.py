"""Unit tests for WAV audio loader.

Tests LOAD-012: WAV Audio Loader
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.wav import get_wav_info, load_wav

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestWAVLoader:
    """Test WAV audio file loader."""

    def create_wav_file(
        self,
        path: Path,
        sample_rate: int = 44100,
        duration: float = 1.0,
        n_channels: int = 1,
        dtype: np.dtype = np.int16,
        frequency: float = 440.0,
    ) -> None:
        """Create a test WAV file.

        Args:
            path: Output path.
            sample_rate: Sample rate in Hz.
            duration: Duration in seconds.
            n_channels: Number of channels.
            dtype: Data type for samples.
            frequency: Sine wave frequency.
        """
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, dtype=np.float64)
        data = np.sin(2 * np.pi * frequency * t)

        # Scale and convert to target dtype
        if dtype == np.int16:
            data = (data * 32767).astype(np.int16)
        elif dtype == np.int32:
            data = (data * 2147483647).astype(np.int32)
        elif dtype == np.uint8:
            data = ((data + 1) * 127.5).astype(np.uint8)
        elif dtype == np.float32:
            data = data.astype(np.float32)

        # Handle multi-channel
        if n_channels > 1:
            data = np.column_stack([data] * n_channels)

        wavfile.write(str(path), sample_rate, data)

    def test_load_basic_wav(self, tmp_path: Path) -> None:
        """Test loading a basic mono WAV file."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, sample_rate=44100, duration=0.1)

        trace = load_wav(wav_path)

        assert trace is not None
        assert len(trace.data) > 0
        assert trace.metadata.sample_rate == 44100
        assert trace.metadata.source_file == str(wav_path)

    def test_load_stereo_wav(self, tmp_path: Path) -> None:
        """Test loading a stereo WAV file."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        # Default should load left channel
        trace = load_wav(wav_path)
        assert trace.metadata.channel_name == "left"

        # Load right channel
        trace = load_wav(wav_path, channel="right")
        assert trace.metadata.channel_name == "right"

        # Load mono mix
        trace = load_wav(wav_path, channel="mono")
        assert trace.metadata.channel_name == "mono"

    def test_load_channel_by_index(self, tmp_path: Path) -> None:
        """Test loading channels by index."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel=0)
        assert trace.metadata.channel_name == "left"

        trace = load_wav(wav_path, channel=1)
        assert trace.metadata.channel_name == "right"

    def test_normalization(self, tmp_path: Path) -> None:
        """Test that samples are normalized to [-1, 1]."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.int16)

        trace = load_wav(wav_path, normalize=True)

        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0

    def test_no_normalization(self, tmp_path: Path) -> None:
        """Test loading without normalization."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.int16)

        trace = load_wav(wav_path, normalize=False)

        # Without normalization, int16 values are preserved as float
        assert np.max(np.abs(trace.data)) > 1.0

    def test_different_sample_rates(self, tmp_path: Path) -> None:
        """Test loading files with different sample rates."""
        for sample_rate in [8000, 22050, 44100, 48000, 96000]:
            wav_path = tmp_path / f"sr_{sample_rate}.wav"
            self.create_wav_file(wav_path, sample_rate=sample_rate, duration=0.1)

            trace = load_wav(wav_path)
            assert trace.metadata.sample_rate == sample_rate

    def test_different_dtypes(self, tmp_path: Path) -> None:
        """Test loading files with different data types."""
        for dtype in [np.int16, np.int32, np.float32]:
            wav_path = tmp_path / f"dtype_{dtype.__name__}.wav"
            self.create_wav_file(wav_path, dtype=dtype, duration=0.1)

            trace = load_wav(wav_path)
            assert trace is not None
            assert len(trace.data) > 0

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        with pytest.raises(LoaderError, match="not found"):
            load_wav(tmp_path / "nonexistent.wav")

    def test_invalid_wav(self, tmp_path: Path) -> None:
        """Test error on invalid WAV file."""
        bad_file = tmp_path / "bad.wav"
        bad_file.write_bytes(b"not a wav file")

        with pytest.raises((FormatError, LoaderError), match="(Invalid|not|format|WAV|RIFF)"):
            load_wav(bad_file)

    def test_channel_out_of_range(self, tmp_path: Path) -> None:
        """Test error on invalid channel index."""
        wav_path = tmp_path / "mono.wav"
        self.create_wav_file(wav_path, n_channels=1)

        with pytest.raises(LoaderError, match="out of range"):
            load_wav(wav_path, channel=5)

    def test_invalid_channel_name(self, tmp_path: Path) -> None:
        """Test error on invalid channel name."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        with pytest.raises(LoaderError, match="Invalid channel"):
            load_wav(wav_path, channel="invalid")

    def test_duration_calculation(self, tmp_path: Path) -> None:
        """Test that duration is correctly calculated."""
        wav_path = tmp_path / "test.wav"
        duration = 0.5
        sample_rate = 44100
        self.create_wav_file(wav_path, sample_rate=sample_rate, duration=duration)

        trace = load_wav(wav_path)

        # Allow small tolerance for rounding
        assert abs(trace.duration - duration) < 0.001


class TestGetWAVInfo:
    """Test get_wav_info function."""

    def test_basic_info(self, tmp_path: Path) -> None:
        """Test getting basic WAV file info."""
        wav_path = tmp_path / "test.wav"
        sample_rate = 44100
        duration = 0.5
        n_samples = int(sample_rate * duration)

        # Create test file
        data = np.sin(np.linspace(0, 1, n_samples)).astype(np.float32)
        wavfile.write(str(wav_path), sample_rate, data)

        info = get_wav_info(wav_path)

        assert info["sample_rate"] == sample_rate
        assert info["n_channels"] == 1
        assert abs(info["duration"] - duration) < 0.01

    def test_stereo_info(self, tmp_path: Path) -> None:
        """Test info for stereo file."""
        wav_path = tmp_path / "stereo.wav"
        sample_rate = 44100
        n_samples = 22050

        data = np.column_stack(
            [
                np.sin(np.linspace(0, 1, n_samples)),
                np.cos(np.linspace(0, 1, n_samples)),
            ]
        ).astype(np.float32)
        wavfile.write(str(wav_path), sample_rate, data)

        info = get_wav_info(wav_path)

        assert info["n_channels"] == 2
        assert info["n_samples"] == n_samples

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        with pytest.raises(LoaderError, match="not found"):
            get_wav_info(tmp_path / "nonexistent.wav")
