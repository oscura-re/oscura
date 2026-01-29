"""Unit tests for WAV audio file loader.

Tests LOAD-012: WAV Audio Loader
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.wav import get_wav_info, load_wav

pytestmark = [pytest.mark.unit, pytest.mark.loader]


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-012")
class TestLoadWAV:
    """Test WAV audio file loading with load_wav function."""

    def create_wav_file(
        self,
        path: Path,
        sample_rate: int = 44100,
        duration: float = 1.0,
        n_channels: int = 1,
        dtype: np.dtype | type = np.int16,
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
        elif dtype == np.float64:
            data = data.astype(np.float64)

        # Handle multi-channel
        if n_channels > 1:
            data = np.column_stack([data] * n_channels)

        wavfile.write(str(path), sample_rate, data)

    def test_load_basic_mono_wav(self, tmp_path: Path) -> None:
        """Test loading a basic mono WAV file."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, sample_rate=44100, duration=0.1)

        trace = load_wav(wav_path)

        assert trace is not None
        assert len(trace.data) > 0
        assert trace.metadata.sample_rate == 44100
        assert trace.metadata.source_file == str(wav_path)
        assert trace.metadata.channel_name == "mono"

    def test_load_stereo_default_channel(self, tmp_path: Path) -> None:
        """Test loading stereo WAV file defaults to left channel."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path)
        assert trace.metadata.channel_name == "left"
        assert trace.metadata.trigger_info["n_channels"] == 2

    def test_load_stereo_left_channel_explicit(self, tmp_path: Path) -> None:
        """Test loading left channel explicitly."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="left")
        assert trace.metadata.channel_name == "left"

    def test_load_stereo_right_channel(self, tmp_path: Path) -> None:
        """Test loading right channel."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="right")
        assert trace.metadata.channel_name == "right"

    def test_load_stereo_mono_mix(self, tmp_path: Path) -> None:
        """Test loading stereo file as mono mix."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="mono")
        assert trace.metadata.channel_name == "mono"

    def test_load_stereo_mix_alias(self, tmp_path: Path) -> None:
        """Test loading with 'mix' alias for mono."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="mix")
        assert trace.metadata.channel_name == "mono"

    def test_load_stereo_avg_alias(self, tmp_path: Path) -> None:
        """Test loading with 'avg' alias for mono."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="avg")
        assert trace.metadata.channel_name == "mono"

    def test_load_channel_by_index_0(self, tmp_path: Path) -> None:
        """Test loading channel by index 0."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel=0)
        assert trace.metadata.channel_name == "left"

    def test_load_channel_by_index_1(self, tmp_path: Path) -> None:
        """Test loading channel by index 1."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel=1)
        assert trace.metadata.channel_name == "right"

    def test_load_channel_l_shorthand(self, tmp_path: Path) -> None:
        """Test loading left channel with 'l' shorthand."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="l")
        assert trace.metadata.channel_name == "left"

    def test_load_channel_r_shorthand(self, tmp_path: Path) -> None:
        """Test loading right channel with 'r' shorthand."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="r")
        assert trace.metadata.channel_name == "right"

    def test_load_channel_numeric_string_0(self, tmp_path: Path) -> None:
        """Test loading channel with numeric string '0'."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="0")
        assert trace.metadata.channel_name == "left"

    def test_load_channel_numeric_string_1(self, tmp_path: Path) -> None:
        """Test loading channel with numeric string '1'."""
        wav_path = tmp_path / "stereo.wav"
        self.create_wav_file(wav_path, n_channels=2)

        trace = load_wav(wav_path, channel="1")
        assert trace.metadata.channel_name == "right"

    def test_load_multichannel_more_than_2(self, tmp_path: Path) -> None:
        """Test loading file with more than 2 channels."""
        wav_path = tmp_path / "multichannel.wav"
        self.create_wav_file(wav_path, n_channels=4)

        # Default to first channel
        trace = load_wav(wav_path)
        assert trace.metadata.channel_name == "ch0"
        assert trace.metadata.trigger_info["n_channels"] == 4

        # Load specific channel
        trace = load_wav(wav_path, channel=2)
        assert trace.metadata.channel_name == "ch2"

    def test_normalization_int16(self, tmp_path: Path) -> None:
        """Test normalization of int16 samples to [-1, 1]."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.int16)

        trace = load_wav(wav_path, normalize=True)

        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0
        assert trace.metadata.trigger_info["normalized"] is True
        assert trace.metadata.trigger_info["original_dtype"] == "int16"

    def test_normalization_int32(self, tmp_path: Path) -> None:
        """Test normalization of int32 samples."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.int32)

        trace = load_wav(wav_path, normalize=True)

        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0

    def test_normalization_uint8(self, tmp_path: Path) -> None:
        """Test normalization of uint8 samples."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.uint8)

        trace = load_wav(wav_path, normalize=True)

        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0

    def test_normalization_float32(self, tmp_path: Path) -> None:
        """Test normalization of float32 samples."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.float32)

        trace = load_wav(wav_path, normalize=True)

        # Float data should already be in range
        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0

    def test_normalization_float64_out_of_range(self, tmp_path: Path) -> None:
        """Test normalization of float64 samples that exceed [-1, 1]."""
        wav_path = tmp_path / "test.wav"
        # Create float data that exceeds [-1, 1]
        data = np.array([2.0, -3.0, 1.5, -2.5], dtype=np.float64)
        wavfile.write(str(wav_path), 44100, data)

        trace = load_wav(wav_path, normalize=True)

        # Should be normalized to [-1, 1]
        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0

    def test_no_normalization_int16(self, tmp_path: Path) -> None:
        """Test loading without normalization preserves int16 scale."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.int16)

        trace = load_wav(wav_path, normalize=False)

        # Without normalization, int16 values are preserved as float
        assert np.max(np.abs(trace.data)) > 1.0
        assert trace.metadata.trigger_info["normalized"] is False

    def test_no_normalization_float32(self, tmp_path: Path) -> None:
        """Test loading float32 without normalization."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.float32)

        trace = load_wav(wav_path, normalize=False)

        # Float data remains unchanged
        assert trace.metadata.trigger_info["normalized"] is False

    def test_different_sample_rates(self, tmp_path: Path) -> None:
        """Test loading files with different sample rates."""
        sample_rates = [8000, 22050, 44100, 48000, 96000]
        for sample_rate in sample_rates:
            wav_path = tmp_path / f"sr_{sample_rate}.wav"
            self.create_wav_file(wav_path, sample_rate=sample_rate, duration=0.1)

            trace = load_wav(wav_path)
            assert trace.metadata.sample_rate == sample_rate

    def test_different_dtypes(self, tmp_path: Path) -> None:
        """Test loading files with different data types."""
        dtypes = [np.int16, np.int32, np.uint8, np.float32]
        for dtype in dtypes:
            wav_path = tmp_path / f"dtype_{dtype.__name__}.wav"
            self.create_wav_file(wav_path, dtype=dtype, duration=0.1)

            trace = load_wav(wav_path)
            assert trace is not None
            assert len(trace.data) > 0
            assert trace.metadata.trigger_info["original_dtype"] == str(np.dtype(dtype))

    def test_duration_calculation_short(self, tmp_path: Path) -> None:
        """Test duration calculation for short file."""
        wav_path = tmp_path / "test.wav"
        duration = 0.1
        sample_rate = 44100
        self.create_wav_file(wav_path, sample_rate=sample_rate, duration=duration)

        trace = load_wav(wav_path)

        # Allow small tolerance for rounding
        assert abs(trace.duration - duration) < 0.001

    def test_duration_calculation_long(self, tmp_path: Path) -> None:
        """Test duration calculation for longer file."""
        wav_path = tmp_path / "test.wav"
        duration = 2.5
        sample_rate = 44100
        self.create_wav_file(wav_path, sample_rate=sample_rate, duration=duration)

        trace = load_wav(wav_path)

        assert abs(trace.duration - duration) < 0.001

    def test_data_type_is_float64(self, tmp_path: Path) -> None:
        """Test that loaded data is always float64."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, dtype=np.int16)

        trace = load_wav(wav_path)

        assert trace.data.dtype == np.float64

    def test_pathlib_path_input(self, tmp_path: Path) -> None:
        """Test that pathlib.Path input works."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, sample_rate=44100, duration=0.1)

        # Pass pathlib.Path directly
        trace = load_wav(wav_path)

        assert trace is not None
        assert trace.metadata.source_file == str(wav_path)

    def test_string_path_input(self, tmp_path: Path) -> None:
        """Test that string path input works."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, sample_rate=44100, duration=0.1)

        # Pass string path
        trace = load_wav(str(wav_path))

        assert trace is not None
        assert trace.metadata.source_file == str(wav_path)

    def test_metadata_contains_trigger_info(self, tmp_path: Path) -> None:
        """Test that metadata includes trigger_info dictionary."""
        wav_path = tmp_path / "test.wav"
        self.create_wav_file(wav_path, n_channels=2, dtype=np.int16)

        trace = load_wav(wav_path)

        assert "trigger_info" in trace.metadata.__dict__
        assert "original_dtype" in trace.metadata.trigger_info
        assert "n_channels" in trace.metadata.trigger_info
        assert "normalized" in trace.metadata.trigger_info

    def test_very_short_duration(self, tmp_path: Path) -> None:
        """Test loading very short WAV file."""
        wav_path = tmp_path / "short.wav"
        self.create_wav_file(wav_path, sample_rate=44100, duration=0.001)

        trace = load_wav(wav_path)

        assert trace is not None
        assert len(trace.data) > 0

    def test_high_frequency_signal(self, tmp_path: Path) -> None:
        """Test loading high frequency signal."""
        wav_path = tmp_path / "highfreq.wav"
        self.create_wav_file(
            wav_path,
            sample_rate=96000,
            duration=0.1,
            frequency=20000.0,
        )

        trace = load_wav(wav_path)

        assert trace is not None
        assert trace.metadata.sample_rate == 96000


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-012")
class TestLoadWAVErrors:
    """Test error handling in load_wav function."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(LoaderError, match="not found"):
            load_wav(tmp_path / "nonexistent.wav")

    def test_invalid_wav_file_binary_data(self, tmp_path: Path) -> None:
        """Test error on invalid WAV file (binary garbage)."""
        bad_file = tmp_path / "bad.wav"
        bad_file.write_bytes(b"not a wav file")

        with pytest.raises((FormatError, LoaderError)):
            load_wav(bad_file)

    def test_invalid_wav_file_text_data(self, tmp_path: Path) -> None:
        """Test error on invalid WAV file (text data)."""
        bad_file = tmp_path / "bad.wav"
        bad_file.write_text("This is not a WAV file")

        with pytest.raises((FormatError, LoaderError)):
            load_wav(bad_file)

    def test_channel_index_out_of_range_mono(self, tmp_path: Path) -> None:
        """Test error when channel index is out of range for mono file."""
        wav_path = tmp_path / "mono.wav"
        data = np.sin(np.linspace(0, 1, 100)).astype(np.int16)
        wavfile.write(str(wav_path), 44100, data)

        with pytest.raises(LoaderError, match="out of range"):
            load_wav(wav_path, channel=1)

    def test_channel_index_out_of_range_stereo(self, tmp_path: Path) -> None:
        """Test error when channel index is out of range for stereo file."""
        wav_path = tmp_path / "stereo.wav"
        data = np.column_stack([np.sin(np.linspace(0, 1, 100))] * 2).astype(np.int16)
        wavfile.write(str(wav_path), 44100, data)

        with pytest.raises(LoaderError, match="out of range"):
            load_wav(wav_path, channel=5)

    def test_channel_negative_index(self, tmp_path: Path) -> None:
        """Test error with negative channel index."""
        wav_path = tmp_path / "stereo.wav"
        data = np.column_stack([np.sin(np.linspace(0, 1, 100))] * 2).astype(np.int16)
        wavfile.write(str(wav_path), 44100, data)

        with pytest.raises(LoaderError, match="out of range"):
            load_wav(wav_path, channel=-1)

    def test_invalid_channel_name(self, tmp_path: Path) -> None:
        """Test error with invalid channel name."""
        wav_path = tmp_path / "stereo.wav"
        data = np.column_stack([np.sin(np.linspace(0, 1, 100))] * 2).astype(np.int16)
        wavfile.write(str(wav_path), 44100, data)

        with pytest.raises(LoaderError, match="Invalid channel"):
            load_wav(wav_path, channel="invalid")

    def test_right_channel_on_mono_file(self, tmp_path: Path) -> None:
        """Test that requesting right channel from mono file silently loads mono data.

        Note: This is current implementation behavior - string channel names
        on mono files are ignored and mono data is returned.
        """
        wav_path = tmp_path / "mono.wav"
        data = np.sin(np.linspace(0, 1, 100)).astype(np.int16)
        wavfile.write(str(wav_path), 44100, data)

        # Currently, string channel names on mono files are ignored
        trace = load_wav(wav_path, channel="right")
        assert trace.metadata.channel_name == "mono"

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test error with empty file."""
        empty_file = tmp_path / "empty.wav"
        empty_file.write_bytes(b"")

        with pytest.raises((FormatError, LoaderError)):
            load_wav(empty_file)

    def test_corrupt_header(self, tmp_path: Path) -> None:
        """Test error with corrupt WAV header."""
        corrupt_file = tmp_path / "corrupt.wav"
        # Create partially valid WAV header but corrupt
        corrupt_file.write_bytes(b"RIFF" + b"\x00" * 100)

        with pytest.raises((FormatError, LoaderError)):
            load_wav(corrupt_file)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-012")
class TestGetWAVInfo:
    """Test get_wav_info function."""

    def test_basic_mono_info(self, tmp_path: Path) -> None:
        """Test getting basic info from mono WAV file."""
        wav_path = tmp_path / "test.wav"
        sample_rate = 44100
        duration = 0.5
        n_samples = int(sample_rate * duration)

        data = np.sin(np.linspace(0, 1, n_samples)).astype(np.float32)
        wavfile.write(str(wav_path), sample_rate, data)

        info = get_wav_info(wav_path)

        assert info["sample_rate"] == sample_rate
        assert info["n_channels"] == 1
        assert info["n_samples"] == n_samples
        assert abs(info["duration"] - duration) < 0.01
        assert "dtype" in info

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
        assert info["sample_rate"] == sample_rate

    def test_multichannel_info(self, tmp_path: Path) -> None:
        """Test info for multichannel file."""
        wav_path = tmp_path / "multi.wav"
        sample_rate = 48000
        n_samples = 1000
        n_channels = 4

        data = np.column_stack([np.sin(np.linspace(0, 1, n_samples))] * n_channels).astype(np.int16)
        wavfile.write(str(wav_path), sample_rate, data)

        info = get_wav_info(wav_path)

        assert info["n_channels"] == n_channels
        assert info["n_samples"] == n_samples

    def test_int16_dtype_info(self, tmp_path: Path) -> None:
        """Test info includes int16 dtype."""
        wav_path = tmp_path / "test.wav"
        data = np.zeros(100, dtype=np.int16)
        wavfile.write(str(wav_path), 44100, data)

        info = get_wav_info(wav_path)

        assert "int16" in info["dtype"]

    def test_int32_dtype_info(self, tmp_path: Path) -> None:
        """Test info includes int32 dtype."""
        wav_path = tmp_path / "test.wav"
        data = np.zeros(100, dtype=np.int32)
        wavfile.write(str(wav_path), 44100, data)

        info = get_wav_info(wav_path)

        assert "int32" in info["dtype"]

    def test_float32_dtype_info(self, tmp_path: Path) -> None:
        """Test info includes float32 dtype."""
        wav_path = tmp_path / "test.wav"
        data = np.zeros(100, dtype=np.float32)
        wavfile.write(str(wav_path), 44100, data)

        info = get_wav_info(wav_path)

        assert "float32" in info["dtype"]

    def test_different_sample_rates_info(self, tmp_path: Path) -> None:
        """Test info with different sample rates."""
        sample_rates = [8000, 22050, 44100, 48000, 96000]
        for sr in sample_rates:
            wav_path = tmp_path / f"sr_{sr}.wav"
            data = np.zeros(100, dtype=np.int16)
            wavfile.write(str(wav_path), sr, data)

            info = get_wav_info(wav_path)
            assert info["sample_rate"] == sr

    def test_duration_calculation_accuracy(self, tmp_path: Path) -> None:
        """Test duration calculation accuracy."""
        wav_path = tmp_path / "test.wav"
        sample_rate = 44100
        n_samples = 88200  # Exactly 2 seconds

        data = np.zeros(n_samples, dtype=np.int16)
        wavfile.write(str(wav_path), sample_rate, data)

        info = get_wav_info(wav_path)

        assert abs(info["duration"] - 2.0) < 0.0001

    def test_pathlib_path_input(self, tmp_path: Path) -> None:
        """Test get_wav_info works with pathlib.Path."""
        wav_path = tmp_path / "test.wav"
        data = np.zeros(100, dtype=np.int16)
        wavfile.write(str(wav_path), 44100, data)

        info = get_wav_info(wav_path)

        assert info is not None
        assert "sample_rate" in info

    def test_string_path_input(self, tmp_path: Path) -> None:
        """Test get_wav_info works with string path."""
        wav_path = tmp_path / "test.wav"
        data = np.zeros(100, dtype=np.int16)
        wavfile.write(str(wav_path), 44100, data)

        info = get_wav_info(str(wav_path))

        assert info is not None
        assert "sample_rate" in info


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-012")
class TestGetWAVInfoErrors:
    """Test error handling in get_wav_info function."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(LoaderError, match="not found"):
            get_wav_info(tmp_path / "nonexistent.wav")

    def test_invalid_wav_file(self, tmp_path: Path) -> None:
        """Test error on invalid WAV file."""
        bad_file = tmp_path / "bad.wav"
        bad_file.write_bytes(b"not a wav file")

        with pytest.raises(LoaderError):
            get_wav_info(bad_file)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test error with empty file."""
        empty_file = tmp_path / "empty.wav"
        empty_file.write_bytes(b"")

        with pytest.raises(LoaderError):
            get_wav_info(empty_file)

    def test_corrupt_wav(self, tmp_path: Path) -> None:
        """Test error with corrupt WAV file."""
        corrupt_file = tmp_path / "corrupt.wav"
        corrupt_file.write_bytes(b"RIFF" + b"\x00" * 50)

        with pytest.raises(LoaderError):
            get_wav_info(corrupt_file)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-012")
class TestWAVIntegration:
    """Integration tests for WAV loader."""

    def test_load_and_info_consistency(self, tmp_path: Path) -> None:
        """Test that load_wav and get_wav_info return consistent data."""
        wav_path = tmp_path / "test.wav"
        sample_rate = 48000
        duration = 0.5
        n_samples = int(sample_rate * duration)

        data = np.sin(np.linspace(0, 2 * np.pi, n_samples)).astype(np.int16)
        wavfile.write(str(wav_path), sample_rate, data)

        info = get_wav_info(wav_path)
        trace = load_wav(wav_path)

        assert info["sample_rate"] == trace.metadata.sample_rate
        assert info["n_samples"] == len(trace.data)
        assert abs(info["duration"] - trace.duration) < 0.001

    def test_stereo_consistency(self, tmp_path: Path) -> None:
        """Test consistency between info and loaded data for stereo."""
        wav_path = tmp_path / "stereo.wav"
        n_samples = 1000
        data = np.column_stack([np.sin(np.linspace(0, 1, n_samples))] * 2).astype(np.int16)
        wavfile.write(str(wav_path), 44100, data)

        info = get_wav_info(wav_path)
        trace = load_wav(wav_path, channel=0)

        assert info["n_channels"] == 2
        assert trace.metadata.trigger_info["n_channels"] == 2
        assert info["n_samples"] == len(trace.data)

    def test_normalized_data_range(self, tmp_path: Path) -> None:
        """Test normalized data stays within expected range."""
        wav_path = tmp_path / "test.wav"
        # Create max amplitude signal
        data = np.array([32767, -32768, 0, 16000, -16000], dtype=np.int16)
        wavfile.write(str(wav_path), 44100, data)

        trace = load_wav(wav_path, normalize=True)

        # Check normalized values
        assert np.max(trace.data) <= 1.0
        assert np.min(trace.data) >= -1.0
        # Check that max values are close to ±1
        assert np.max(trace.data) > 0.99
        assert np.min(trace.data) < -0.99

    def test_channel_averaging_correctness(self, tmp_path: Path) -> None:
        """Test that channel averaging produces correct values."""
        wav_path = tmp_path / "stereo.wav"
        n_samples = 100
        left = np.ones(n_samples, dtype=np.float32)
        right = -np.ones(n_samples, dtype=np.float32)
        data = np.column_stack([left, right])
        wavfile.write(str(wav_path), 44100, data)

        # Average should be zero
        trace = load_wav(wav_path, channel="mono")

        assert np.allclose(trace.data, 0.0, atol=1e-6)

    def test_roundtrip_normalization(self, tmp_path: Path) -> None:
        """Test that normalization and denormalization preserve signal shape."""
        wav_path = tmp_path / "test.wav"
        # Create known signal
        t = np.linspace(0, 1, 1000)
        original = np.sin(2 * np.pi * 10 * t)
        data_int16 = (original * 32767).astype(np.int16)
        wavfile.write(str(wav_path), 44100, data_int16)

        trace = load_wav(wav_path, normalize=True)

        # Normalized data should closely match original sine wave
        assert np.allclose(trace.data, original, atol=0.001)
