"""Tests for Tier 2 export modules (EXP-004, EXP-005).

Requirements tested:
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.exporter]


class TestNPZExport:
    """Tests for NPZ export functionality (EXP-004)."""

    def test_export_npz_trace(self) -> None:
        """Test NPZ export of waveform trace."""
        from oscura.exporters.npz_export import export_npz, load_npz

        # Create test trace
        sample_rate = 100_000
        signal = np.sin(2 * np.pi * 1000 * np.arange(1000) / sample_rate)

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            export_npz(trace, path)

            assert path.exists()

            # Load and verify
            data = load_npz(path)

            assert "signal" in data
            assert "sample_rate" in data
            np.testing.assert_array_equal(data["signal"], signal)
            assert float(data["sample_rate"]) == sample_rate

    def test_export_npz_dict(self) -> None:
        """Test NPZ export of dictionary."""
        from oscura.exporters.npz_export import export_npz, load_npz

        test_data = {
            "signal": np.random.randn(100),
            "time": np.linspace(0, 1, 100),
            "rate": 100.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            export_npz(test_data, path)

            data = load_npz(path)

            assert "signal" in data
            assert "time" in data
            assert len(data["signal"]) == 100

    def test_export_npz_compressed(self) -> None:
        """Test compressed NPZ export."""
        from oscura.exporters.npz_export import export_npz

        signal = np.random.randn(10000)

        with tempfile.TemporaryDirectory() as tmpdir:
            path_compressed = Path(tmpdir) / "compressed.npz"
            path_uncompressed = Path(tmpdir) / "uncompressed.npz"

            export_npz(signal, path_compressed, compressed=True)
            export_npz(signal, path_uncompressed, compressed=False)

            # Compressed should be smaller
            assert path_compressed.stat().st_size < path_uncompressed.stat().st_size

    def test_npz_accessible_from_package(self) -> None:
        """Test NPZ functions accessible from package."""
        from oscura.exporters import export_npz, load_npz

        assert callable(export_npz)
        assert callable(load_npz)


class TestSpiceExport:
    """Tests for SPICE PWL export functionality (EXP-005)."""

    def test_export_pwl_basic(self) -> None:
        """Test basic PWL export."""
        from oscura.exporters.spice_export import export_pwl

        sample_rate = 10_000
        signal = np.sin(2 * np.pi * 100 * np.arange(100) / sample_rate)

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stimulus.pwl"
            export_pwl(trace, path)

            assert path.exists()

            # Check content
            content = path.read_text()
            lines = content.strip().split("\n")

            # Should have time-value pairs
            assert len(lines) == len(signal)

            # Parse first line
            parts = lines[0].split()
            assert len(parts) == 2  # time, value

    def test_export_pwl_scaling(self) -> None:
        """Test PWL export with time/amplitude scaling."""
        from oscura.exporters.spice_export import export_pwl

        signal = np.array([0, 1, 0, 1, 0], dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scaled.pwl"
            export_pwl(
                (np.arange(len(signal)), signal),
                path,
                time_scale=1e-9,  # ns
                amplitude_scale=3.3,  # 3.3V
            )

            content = path.read_text()
            lines = content.strip().split("\n")

            # Check scaling applied
            time_0, val_0 = lines[0].split()
            assert float(time_0) == 0.0  # First time point
            assert float(val_0) == 0.0  # 0 * 3.3 = 0

            time_1, val_1 = lines[1].split()
            assert abs(float(time_1) - 1e-9) < 1e-12  # 1 * 1e-9
            assert abs(float(val_1) - 3.3) < 1e-6  # 1 * 3.3

    def test_generate_spice_source(self) -> None:
        """Test SPICE source definition generation."""
        from oscura.exporters.spice_export import generate_spice_source

        line = generate_spice_source("input.pwl", "in", "0", "V1")
        assert line == "V1 in 0 PWL file=input.pwl"

        line_current = generate_spice_source(
            "current.pwl", "out", "gnd", "src", source_type="current"
        )
        assert "Isrc" in line_current

    def test_spice_accessible_from_package(self) -> None:
        """Test SPICE functions accessible from package."""
        from oscura.exporters import export_pwl, generate_spice_source

        assert callable(export_pwl)
        assert callable(generate_spice_source)


class TestRMSJitter:
    """Tests for RMS jitter measurement (TIM-007)."""

    def test_rms_jitter_basic(self) -> None:
        """Test basic RMS jitter measurement."""
        from oscura.analyzers.digital.timing import rms_jitter

        # Create square wave with known jitter
        sample_rate = 1_000_000  # 1 MHz
        n_periods = 20
        samples_per_period = 100  # 10 kHz clock
        half_period = samples_per_period // 2

        # Build square wave with jittered edges
        np.random.seed(42)
        jitter_std = 1e-6  # 1 us RMS jitter

        signal_parts = []
        for _i in range(n_periods):
            # Add jitter to rising edge position
            jitter_samples = int(np.random.randn() * jitter_std * sample_rate)

            # Low portion (with jitter adjustment)
            low_samples = max(1, half_period + jitter_samples)
            signal_parts.extend([0.0] * low_samples)

            # High portion
            high_samples = max(1, samples_per_period - low_samples)
            signal_parts.extend([1.0] * high_samples)

        signal = np.array(signal_parts)

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
            ),
        )

        result = rms_jitter(trace)

        assert result.samples > 0
        assert result.rms > 0
        assert result.uncertainty > 0
        assert result.edge_type == "rising"

    def test_rms_jitter_insufficient_data(self) -> None:
        """Test RMS jitter with insufficient edges."""
        from oscura.analyzers.digital.timing import rms_jitter

        # Signal with no edges
        signal = np.ones(100)

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=1000,
            ),
        )

        result = rms_jitter(trace)

        assert np.isnan(result.rms)
        assert result.samples == 0

    def test_rms_jitter_accessible(self) -> None:
        """Test rms_jitter is accessible from timing module."""
        from oscura.analyzers.digital.timing import RMSJitterResult, rms_jitter

        assert callable(rms_jitter)
        assert RMSJitterResult is not None
