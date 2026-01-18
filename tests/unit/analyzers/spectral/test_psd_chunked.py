"""Tests for chunked Welch PSD (MEM-005).

Requirements tested:
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.spectral]


class TestPsdChunked:
    """Tests for psd_chunked functionality."""

    def test_psd_chunked_basic(self) -> None:
        """Test basic chunked PSD computation."""
        from oscura.analyzers.waveform.spectral import psd_chunked

        # Create test signal: 1 kHz sine at 100 kHz sample rate
        sample_rate = 100_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        signal = 0.5 * np.sin(2 * np.pi * 1000 * t)

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
            ),
        )

        freq, psd_db = psd_chunked(trace, chunk_size=10_000, nperseg=256)

        # Should have valid output
        assert len(freq) > 0
        assert len(psd_db) == len(freq)

        # Find peak near 1 kHz (with some tolerance due to chunking)
        peak_idx = np.argmax(psd_db)
        peak_freq = freq[peak_idx]
        assert 500 < peak_freq < 1500  # Within 500 Hz of 1 kHz (chunking may cause some spreading)

    def test_psd_chunked_small_signal(self) -> None:
        """Test that small signals fall back to standard PSD."""
        from oscura.analyzers.waveform.spectral import psd, psd_chunked

        sample_rate = 10_000
        signal = np.sin(2 * np.pi * 100 * np.arange(0, 0.1, 1 / sample_rate))

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
            ),
        )

        # With chunk_size larger than signal, should return same as standard PSD
        freq_chunked, psd_chunked_db = psd_chunked(trace, chunk_size=len(signal) * 10)
        freq_std, psd_std_db = psd(trace)

        np.testing.assert_array_equal(freq_chunked, freq_std)
        np.testing.assert_allclose(psd_chunked_db, psd_std_db, rtol=1e-5)

    def test_psd_chunked_accessible(self) -> None:
        """Test psd_chunked is accessible from spectral module."""
        from oscura.analyzers.spectral import psd_chunked

        assert callable(psd_chunked)

    def test_psd_chunked_scaling(self) -> None:
        """Test chunked PSD with different scaling options."""
        from oscura.analyzers.waveform.spectral import psd_chunked

        sample_rate = 10_000
        signal = np.random.randn(50_000)

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
            ),
        )

        freq_density, psd_density = psd_chunked(trace, chunk_size=10_000, scaling="density")
        freq_spectrum, psd_spectrum = psd_chunked(trace, chunk_size=10_000, scaling="spectrum")

        # Both should produce valid output
        assert len(freq_density) > 0
        assert len(psd_density) == len(freq_density)
        assert len(freq_spectrum) > 0
        assert len(psd_spectrum) == len(freq_spectrum)

        # Frequency axes should match
        np.testing.assert_array_equal(freq_density, freq_spectrum)


class TestSampleDataDownload:
    """Tests for sample data download functionality (LOAD-007)."""

    def test_get_samples_dir(self) -> None:
        """Test sample directory path generation."""
        from oscura.__main__ import get_samples_dir

        samples_dir = get_samples_dir()
        assert samples_dir.name == "samples"
        assert ".oscura" in str(samples_dir)

    def test_get_sample_files(self) -> None:
        """Test sample files list."""
        from oscura.__main__ import get_sample_files

        files = get_sample_files()
        assert len(files) > 0

        # Check expected file
        assert "sine_1khz.csv" in files
        info = files["sine_1khz.csv"]
        assert "description" in info
        assert "format" in info

    def test_generate_sample_sine(self) -> None:
        """Test local sample file generation."""
        import tempfile
        from pathlib import Path

        from oscura.__main__ import generate_sample_file

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "sine_1khz.csv"
            result = generate_sample_file("sine_1khz.csv", dest)

            assert result is True
            assert dest.exists()

            # Verify content
            content = dest.read_text()
            assert "time,voltage" in content or len(content) > 0
