"""Synthetic data generation utilities for demonstrations."""

from __future__ import annotations

import numpy as np

from oscura.core.types import TraceMetadata, WaveformTrace


def generate_sine_wave(
    frequency: float = 1000.0,
    amplitude: float = 1.0,
    duration: float = 0.001,
    sample_rate: float = 1e6,
    offset: float = 0.0,
    phase: float = 0.0,
) -> WaveformTrace:
    """Generate a sine wave signal.

    Args:
        frequency: Frequency in Hz
        amplitude: Peak amplitude
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        offset: DC offset
        phase: Phase offset in radians

    Returns:
        WaveformTrace with sine wave
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="sine_wave",
    )
    return WaveformTrace(data=signal, metadata=metadata)


def generate_square_wave(
    frequency: float = 1000.0,
    amplitude: float = 1.0,
    duration: float = 0.001,
    sample_rate: float = 1e6,
    duty_cycle: float = 0.5,
    offset: float = 0.0,
) -> WaveformTrace:
    """Generate a square wave signal.

    Args:
        frequency: Frequency in Hz
        amplitude: Peak amplitude
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        duty_cycle: Duty cycle (0.0 to 1.0)
        offset: DC offset

    Returns:
        WaveformTrace with square wave
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Generate square wave
    phase = (t * frequency) % 1.0
    signal = np.where(phase < duty_cycle, amplitude, -amplitude) + offset

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="square_wave",
    )
    return WaveformTrace(data=signal, metadata=metadata)


def add_noise(trace: WaveformTrace, snr_db: float = 40.0) -> WaveformTrace:
    """Add white noise to a signal.

    Args:
        trace: Input trace
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Trace with added noise
    """
    signal = trace.data
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    noisy_signal = signal + noise

    return WaveformTrace(data=noisy_signal, metadata=trace.metadata)


def generate_pulse_train(
    pulse_width: float = 1e-6,
    period: float = 10e-6,
    amplitude: float = 5.0,
    duration: float = 0.001,
    sample_rate: float = 1e6,
    rise_time: float = 1e-9,
    fall_time: float = 1e-9,
) -> WaveformTrace:
    """Generate a pulse train with realistic rise/fall times.

    Args:
        pulse_width: Pulse width in seconds
        period: Period between pulses in seconds
        amplitude: Pulse amplitude
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        rise_time: Rise time in seconds
        fall_time: Fall time in seconds

    Returns:
        WaveformTrace with pulse train
    """
    num_samples = int(duration * sample_rate)
    _t = np.arange(num_samples) / sample_rate  # Time vector for reference
    signal = np.zeros(num_samples)

    # Calculate pulse positions
    pulse_start_times = np.arange(0, duration, period)

    for start_time in pulse_start_times:
        # Find sample indices
        start_idx = int(start_time * sample_rate)
        end_idx = int((start_time + pulse_width) * sample_rate)
        rise_samples = int(rise_time * sample_rate)
        fall_samples = int(fall_time * sample_rate)

        if end_idx >= num_samples:
            break

        # Rising edge
        if start_idx + rise_samples < num_samples:
            signal[start_idx : start_idx + rise_samples] = np.linspace(0, amplitude, rise_samples)

        # Flat top
        signal[start_idx + rise_samples : end_idx - fall_samples] = amplitude

        # Falling edge
        if end_idx < num_samples:
            signal[end_idx - fall_samples : end_idx] = np.linspace(amplitude, 0, fall_samples)

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="pulse_train",
    )
    return WaveformTrace(data=signal, metadata=metadata)


def generate_complex_signal(
    fundamentals: list[float],
    amplitudes: list[float],
    duration: float = 0.001,
    sample_rate: float = 1e6,
    snr_db: float | None = None,
) -> WaveformTrace:
    """Generate a complex signal with multiple frequency components.

    Args:
        fundamentals: List of frequencies in Hz
        amplitudes: List of amplitudes for each frequency
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        snr_db: Optional SNR in dB (adds noise if specified)

    Returns:
        WaveformTrace with complex signal
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    signal = np.zeros(num_samples)

    # Add each frequency component
    for freq, amp in zip(fundamentals, amplitudes, strict=True):
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Add noise if requested
    if snr_db is not None:
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        signal += noise

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="complex_signal",
    )
    return WaveformTrace(data=signal, metadata=metadata)
