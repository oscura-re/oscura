"""Analyzer-specific test fixtures.

This module provides fixtures for analyzer tests:
- Signal generation fixtures
- Analysis helper fixtures
- Quality metrics fixtures
- Digital signal fixtures
- Protocol-specific signal fixtures
"""

from __future__ import annotations

from typing import Any

import pytest

# =============================================================================
# Basic Signal Generation Fixtures
# =============================================================================
# MIGRATION COMPLETE: All basic signal fixtures removed.
# Tests should use SignalBuilder from tests/fixtures/signal_builders.py instead.
#
# For signal generation, use the signal_builder fixture:
#   def test_something(signal_builder):
#       signal = signal_builder.square_wave(frequency=1e3, sample_rate=10e3)
#       signal = signal_builder.noisy_sine(snr_db=40)
#       signal = signal_builder.triangle_wave()
#       signal = signal_builder.sawtooth_wave()
#
# This eliminates duplication and provides consistent signal generation.
# See tests/fixtures/signal_builders.py for 20+ signal generation methods.


# =============================================================================
# Analysis Metadata Fixtures
# =============================================================================


@pytest.fixture
def analyzer_metadata() -> dict[str, Any]:
    """Common metadata for analyzer tests.

    Returns:
        Dictionary with sample_rate, duration, and signal_type.
    """
    return {
        "sample_rate": 1e6,
        "duration": 0.01,
        "signal_type": "test",
        "units": "V",
    }


@pytest.fixture
def timing_metadata() -> dict[str, Any]:
    """Timing-specific metadata for analyzer tests."""
    return {
        "sample_rate": 1e9,  # 1 GHz
        "time_resolution": 1e-9,  # 1 ns
        "edge_detection_threshold": 1.65,  # 3.3V / 2
        "hysteresis": 0.2,  # 200 mV
    }


# =============================================================================
# Digital Signal Analysis Fixtures
# =============================================================================
# MIGRATION COMPLETE: All digital signal fixtures removed (unused).
# Tests should use SignalBuilder from tests/fixtures/signal_builders.py instead.
#
# Previous fixtures replaced by SignalBuilder methods:
#   spi_signal → signal_builder.digital_pattern() for each channel
#   i2c_signal → signal_builder.digital_pattern() for SCL/SDA
#   can_signal → signal_builder.digital_pattern() for CAN bus
#
# =============================================================================
# Eye Diagram Fixtures
# =============================================================================
# MIGRATION COMPLETE: Eye diagram fixtures removed (unused).
# Use signal_builder.digital_pattern() with noise for eye diagram testing.
#
# =============================================================================
# Jitter Analysis Fixtures
# =============================================================================
# MIGRATION COMPLETE: Jitter fixtures removed (unused).
# Use signal_builder.square_wave() with custom jitter if needed.
#
# =============================================================================
# Power Analysis Fixtures
# =============================================================================
# MIGRATION COMPLETE: Power analysis fixtures removed (unused).
# Use signal_builder.sine_wave() + dc_offset() for power supply simulation.
#
# =============================================================================
# Spectral Analysis Fixtures
# =============================================================================
# MIGRATION COMPLETE: Spectral fixtures removed (unused).
# Previous fixtures replaced by SignalBuilder methods:
#   multi_tone_signal → signal_builder.multitone([10, 25, 50])
#   chirp_signal      → signal_builder.chirp(f0=1, f1=50)
#
# =============================================================================
# Statistical Analysis Fixtures
# =============================================================================
# MIGRATION COMPLETE: Statistical signal fixtures removed (unused).
# Previous fixtures replaced by SignalBuilder methods:
#   gaussian_noise → signal_builder.white_noise()
#   uniform_noise  → numpy.random.uniform()
#
# =============================================================================
# Pattern Detection Fixtures
# =============================================================================
# MIGRATION COMPLETE: Pattern detection fixtures removed (unused).
# Previous fixtures replaced by SignalBuilder methods:
#   repeating_pattern_signal → signal_builder.digital_pattern()
#   anomaly_signal          → signal_builder.sine_wave() + custom anomalies


# =============================================================================
# Quality Metrics Fixtures
# =============================================================================


@pytest.fixture
def quality_thresholds() -> dict[str, float]:
    """Quality metric thresholds for analyzer validation.

    Returns:
        Dictionary with SNR, THD, and other quality thresholds.
    """
    return {
        "min_snr_db": 40.0,  # Minimum SNR in dB
        "max_thd_percent": 1.0,  # Maximum THD in percent
        "min_sfdr_db": 60.0,  # Spurious-free dynamic range
        "max_jitter_ui": 0.1,  # Maximum jitter in UI
        "min_eye_height": 0.7,  # Minimum eye height (normalized)
        "min_eye_width": 0.8,  # Minimum eye width (normalized)
    }


# =============================================================================
# Edge Detection Fixtures
# =============================================================================
# MIGRATION COMPLETE: Edge detection fixtures removed (unused).
# Use signal_builder.step_response() or custom edge generation if needed.
