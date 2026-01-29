#!/usr/bin/env python3
"""Generate optimal demo data for Mixed Signal demo.

Creates three high-speed serial test signals for IEEE-compliant analysis:
1. gigabit_ethernet_eye.wfm - 1.25 Gbps NRZ, PRBS-31, RJ=5ps, DJ=10ps
2. clock_oscillator_100mhz.wfm - 100 MHz clock, period jitter=2ps RMS
3. usb_differential_pair.wfm - USB 2.0 HS diff pair @ 480 Mbps

Usage:
    python generate_demo_data.py [--force]

Author: Oscura Development Team
Date: 2026-01-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{BLUE}INFO:{RESET} {msg}")


def generate_prbs31(num_bits: int) -> list[int]:
    """Generate PRBS-31 pattern (pseudo-random binary sequence).

    Polynomial: x^31 + x^28 + 1

    Args:
        num_bits: Number of bits to generate

    Returns:
        List of bits (0 or 1)
    """
    # PRBS-31 state (31 bits, initialized to all 1s)
    state = 0x7FFFFFFF
    bits = []

    for _ in range(num_bits):
        # Output current LSB
        bit = state & 1
        bits.append(bit)

        # Calculate feedback bit: XOR of bits 31 and 28
        feedback = ((state >> 30) ^ (state >> 27)) & 1

        # Shift and insert feedback
        state = ((state >> 1) | (feedback << 30)) & 0x7FFFFFFF

    return bits


def generate_gigabit_ethernet_eye(output_file: Path) -> None:
    """Generate Gigabit Ethernet eye diagram test signal.

    Signal: 1.25 Gbps NRZ data (PRBS-31 pattern)
    Characteristics: RJ = 5 ps RMS, DJ = 10 ps p-p, noise
    Sampling: 40 GS/s, 100 μs (125,000 bits)

    Sample rate: 40 GS/s
    Duration: 100 μs
    """
    print_info("Generating gigabit_ethernet_eye.wfm...")

    data_rate = 1.25e9  # 1.25 Gbps
    sample_rate = 40e9  # 40 GS/s (32 samples per bit)
    duration = 100e-6  # 100 μs
    num_samples = int(sample_rate * duration)
    samples_per_bit = int(sample_rate / data_rate)

    # Generate PRBS-31 pattern
    num_bits = num_samples // samples_per_bit
    bits = generate_prbs31(num_bits)

    # Create ideal NRZ signal (±1V differential)
    signal = np.zeros(num_samples)
    for i, bit in enumerate(bits):
        start_idx = i * samples_per_bit
        end_idx = min(start_idx + samples_per_bit, num_samples)
        signal[start_idx:end_idx] = 1.0 if bit else -1.0

    # Add rise/fall time (realistic edge slew rate)
    # Rise time ~100 ps (10%-90%)
    rise_time = 100e-12
    rise_samples = int(rise_time * sample_rate)

    # Apply edge filtering (simple moving average for rise/fall)
    if rise_samples > 1:
        kernel = np.ones(rise_samples) / rise_samples
        signal = np.convolve(signal, kernel, mode="same")

    # Add random jitter (RJ) - Gaussian distributed
    rj_sigma = 5e-12  # 5 ps RMS
    t = np.arange(num_samples) / sample_rate

    # Apply jitter by time-shifting the signal
    jittered_signal = np.zeros_like(signal)
    for i in range(num_samples):
        # Random jitter component
        time_jitter_rj = np.random.randn() * rj_sigma

        # Deterministic jitter (DJ) - sinusoidal component
        dj_amplitude = 10e-12 / 2  # 10 ps p-p = 5 ps amplitude
        dj_freq = 1e6  # 1 MHz modulation (typical)
        time_jitter_dj = dj_amplitude * np.sin(2 * np.pi * dj_freq * t[i])

        # Total jitter
        time_jitter = time_jitter_rj + time_jitter_dj

        # Sample signal at jittered time
        jittered_idx = int(i + time_jitter * sample_rate)
        if 0 <= jittered_idx < num_samples:
            jittered_signal[i] = signal[jittered_idx]
        else:
            jittered_signal[i] = signal[i]

    # Add inter-symbol interference (ISI) - realistic channel response
    # Model as exponential decay from previous bit
    isi_signal = jittered_signal.copy()
    isi_factor = 0.15  # 15% ISI
    for i in range(samples_per_bit, num_samples):
        isi_signal[i] = jittered_signal[i] + isi_factor * jittered_signal[i - samples_per_bit]

    # Normalize to maintain amplitude
    isi_signal = isi_signal / (1 + isi_factor)

    # Add noise
    # SNR ~20 dB (realistic for high-speed serial)
    signal_power = np.mean(isi_signal**2)
    noise_power = signal_power / (10 ** (20 / 10))
    noise = np.sqrt(noise_power) * np.random.randn(num_samples)
    final_signal = isi_signal + noise

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=final_signal,
        sample_rate=sample_rate,
        duration=duration,
        data_rate=data_rate,
        channel_names=["GbE_Signal"],
        metadata={
            "signal_type": "Gigabit Ethernet eye diagram",
            "data_rate": "1.25 Gbps",
            "pattern": "PRBS-31",
            "rj_target": "5 ps RMS",
            "dj_target": "10 ps p-p",
            "rise_time": "100 ps (10%-90%)",
            "isi": "15%",
            "standard": "IEEE 802.3 (Gigabit Ethernet)",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated gigabit_ethernet_eye.npz ({size_mb:.2f} MB)")


def generate_clock_oscillator_100mhz(output_file: Path) -> None:
    """Generate 100 MHz clock oscillator WFM file.

    Signal: 100 MHz clock with realistic phase noise
    Characteristics: Period jitter = 2 ps RMS, duty cycle = 50% ± 2%
    Sampling: 10 GS/s, 10 μs

    Sample rate: 10 GS/s
    Duration: 10 μs
    """
    print_info("Generating clock_oscillator_100mhz.wfm...")

    clock_freq = 100e6  # 100 MHz
    sample_rate = 10e9  # 10 GS/s (100 samples per clock cycle)
    duration = 10e-6  # 10 μs
    num_samples = int(sample_rate * duration)
    _samples_per_cycle = int(sample_rate / clock_freq)

    # Generate ideal clock (square wave)
    t = np.arange(num_samples) / sample_rate
    num_cycles = int(clock_freq * duration)

    # Initialize signal
    signal = np.zeros(num_samples)

    # Period jitter parameters
    period_jitter_rms = 2e-12  # 2 ps RMS
    nominal_period = 1 / clock_freq

    # Duty cycle variation
    duty_cycle_nominal = 0.5
    duty_cycle_variation = 0.02  # ±2%

    # Generate clock with jitter
    current_time = 0.0
    for _cycle in range(num_cycles):
        # Add period jitter (random walk in time)
        period_error = np.random.randn() * period_jitter_rms
        actual_period = nominal_period + period_error

        # Add duty cycle variation
        duty_cycle = duty_cycle_nominal + (np.random.rand() - 0.5) * duty_cycle_variation * 2
        high_time = actual_period * duty_cycle
        _low_time = actual_period * (1 - duty_cycle)

        # Find sample indices for this clock cycle
        start_idx = int(current_time * sample_rate)
        mid_idx = int((current_time + high_time) * sample_rate)
        end_idx = int((current_time + actual_period) * sample_rate)

        # Set high and low periods
        if start_idx < num_samples:
            signal[start_idx : min(mid_idx, num_samples)] = 1.0
        if mid_idx < num_samples:
            signal[mid_idx : min(end_idx, num_samples)] = 0.0

        current_time += actual_period

    # Add edge slew rate (realistic rise/fall time ~100 ps)
    rise_time = 100e-12
    rise_samples = int(rise_time * sample_rate)
    if rise_samples > 1:
        kernel = np.ones(rise_samples) / rise_samples
        signal = np.convolve(signal, kernel, mode="same")

    # Add phase noise (low-frequency jitter)
    # Model as random walk in phase
    phase_noise_freq = np.logspace(1, 7, 1000)  # 10 Hz to 10 MHz
    phase_noise_psd = 1e-12 / phase_noise_freq**1.5  # 1/f^1.5 noise
    phase_noise = np.zeros(num_samples)
    for freq, psd in zip(phase_noise_freq, phase_noise_psd, strict=False):
        amplitude = np.sqrt(psd * sample_rate)
        phase = np.random.uniform(0, 2 * np.pi)
        phase_noise += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Apply phase noise to signal (frequency modulation)
    signal_with_noise = signal * (1 + 0.001 * phase_noise)

    # Add voltage noise
    voltage_noise = 0.02 * np.random.randn(num_samples)
    final_signal = signal_with_noise + voltage_noise

    # Scale to realistic voltage levels (3.3V CMOS)
    final_signal = final_signal * 3.3

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=final_signal,
        sample_rate=sample_rate,
        duration=duration,
        clock_freq=clock_freq,
        channel_names=["Clock_100MHz"],
        metadata={
            "signal_type": "Clock oscillator",
            "frequency": "100 MHz",
            "period_jitter_target": "2 ps RMS",
            "duty_cycle": "50% ± 2%",
            "rise_time": "100 ps",
            "standard": "IEEE 2414-2020 (Jitter measurement)",
            "phase_noise": "1/f^1.5 model",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated clock_oscillator_100mhz.npz ({size_mb:.2f} MB)")


def generate_usb_differential_pair(output_file: Path) -> None:
    """Generate USB differential pair WFM file.

    Signal: USB 2.0 HS differential pair (D+, D-) @ 480 Mbps
    Characteristics: Common mode noise, differential skew, reflection artifacts
    Sampling: 10 GS/s, 50 μs

    Sample rate: 10 GS/s
    Duration: 50 μs
    """
    print_info("Generating usb_differential_pair.wfm...")

    data_rate = 480e6  # 480 Mbps
    sample_rate = 10e9  # 10 GS/s (~21 samples per bit)
    duration = 50e-6  # 50 μs
    num_samples = int(sample_rate * duration)
    samples_per_bit = int(sample_rate / data_rate)

    # Generate PRBS-31 pattern
    num_bits = num_samples // samples_per_bit
    bits = generate_prbs31(num_bits)

    # Generate NRZI encoding (USB uses NRZI)
    # NRZI: no transition = 1, transition = 0
    nrzi_signal = [0]  # Start with 0
    for bit in bits:
        if bit == 1:
            # No transition
            nrzi_signal.append(nrzi_signal[-1])
        else:
            # Transition
            nrzi_signal.append(1 - nrzi_signal[-1])

    # Create differential NRZ signal
    signal_diff = np.zeros(num_samples)
    for i, bit in enumerate(nrzi_signal[:num_bits]):
        start_idx = i * samples_per_bit
        end_idx = min(start_idx + samples_per_bit, num_samples)
        signal_diff[start_idx:end_idx] = 1.0 if bit else -1.0

    # Add rise/fall time (~2 ns for USB 2.0 HS)
    rise_time = 2e-9
    rise_samples = int(rise_time * sample_rate)
    if rise_samples > 1:
        kernel = np.ones(rise_samples) / rise_samples
        signal_diff = np.convolve(signal_diff, kernel, mode="same")

    # Scale to USB voltage levels (400 mV differential)
    signal_diff = signal_diff * 0.4  # ±400 mV differential

    # Generate D+ and D- from differential signal
    d_plus = signal_diff / 2
    d_minus = -signal_diff / 2

    # Add differential skew (~50 ps)
    skew_time = 50e-12
    skew_samples = int(skew_time * sample_rate)
    if skew_samples > 0:
        d_minus = np.roll(d_minus, skew_samples)

    # Add common mode voltage (1.65V for USB)
    common_mode = 1.65

    # Add common mode noise (coupling from other signals)
    t = np.arange(num_samples) / sample_rate
    cm_noise_freq = 10e6  # 10 MHz interference
    cm_noise = 0.05 * np.sin(2 * np.pi * cm_noise_freq * t)

    # Add reflection artifacts (impedance mismatch)
    # Model as delayed, attenuated copy of signal
    reflection_delay = 5e-9  # 5 ns (typical for 1m cable)
    reflection_samples = int(reflection_delay * sample_rate)
    reflection_amplitude = 0.2  # 20% reflection

    signal_diff_reflected = signal_diff.copy()
    if reflection_samples < num_samples:
        signal_diff_reflected[reflection_samples:] += (
            reflection_amplitude * signal_diff[: num_samples - reflection_samples]
        )

    # Recalculate D+ and D- with reflections
    d_plus = signal_diff_reflected / 2 + common_mode + cm_noise
    d_minus = -signal_diff_reflected / 2 + common_mode + cm_noise

    # Add independent noise to each signal
    noise_amplitude = 0.01
    d_plus += noise_amplitude * np.random.randn(num_samples)
    d_minus += noise_amplitude * np.random.randn(num_samples)

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        d_plus=d_plus,
        d_minus=d_minus,
        sample_rate=sample_rate,
        duration=duration,
        data_rate=data_rate,
        channel_names=["D+", "D-"],
        metadata={
            "signal_type": "USB 2.0 High-Speed differential pair",
            "data_rate": "480 Mbps",
            "encoding": "NRZI",
            "pattern": "PRBS-31",
            "differential_voltage": "400 mV",
            "common_mode": "1.65V",
            "skew": "50 ps",
            "reflections": "20% @ 5 ns",
            "standard": "USB 2.0 specification",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated usb_differential_pair.npz ({size_mb:.2f} MB)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demo data for mixed signal analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Create demo_data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "demo_data"
    data_dir.mkdir(exist_ok=True)

    # Define output files
    files_to_generate = [
        ("gigabit_ethernet_eye.wfm", generate_gigabit_ethernet_eye),
        ("clock_oscillator_100mhz.wfm", generate_clock_oscillator_100mhz),
        ("usb_differential_pair.wfm", generate_usb_differential_pair),
    ]

    print(f"\n{BOLD}{BLUE}Generating Mixed Signal Demo Data{RESET}")
    print("=" * 80)

    for filename, generator_func in files_to_generate:
        output_file = data_dir / filename

        if output_file.with_suffix(".npz").exists() and not args.force:
            print_info(f"Skipping {filename} (already exists, use --force to overwrite)")
            continue

        generator_func(output_file)

    print(f"\n{GREEN}{BOLD}✓ Demo data generation complete!{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
