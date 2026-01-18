#!/usr/bin/env python3
"""Generate synthetic EMC/EMI compliance test data.

This script generates realistic test waveforms for EMC compliance analysis:
- Conducted emissions (CE) per CISPR 32 (150 kHz - 30 MHz)
- Radiated emissions (RE) scan (30 MHz - 1 GHz)
- Power quality harmonics per IEC 61000-3-2
- ESD transient burst per IEC 61000-4-2
- MIL-STD-461 CE102 conducted emissions

These files are used by comprehensive_emc_demo.py.
"""

import csv
from pathlib import Path

import numpy as np

from oscura.core.types import TraceMetadata, WaveformTrace


def generate_conducted_emissions_ac_line() -> WaveformTrace:
    """Generate conducted emissions on AC line per CISPR 32.

    Simulates switching power supply EMI:
    - Switching fundamental at 100 kHz
    - Harmonics up to 30 MHz
    - Some exceeding CISPR 32 Class B limits
    """
    sample_rate = 200e6  # 200 MSa/s
    duration = 100e-6  # 100 us
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Start with noise floor
    signal = np.random.normal(0, 0.001, n_samples)

    # Switching fundamental at 100 kHz and harmonics
    switching_freq = 100e3
    for harmonic in range(1, 301):  # Up to 30 MHz
        freq = switching_freq * harmonic
        if freq > 30e6:
            break

        # Amplitude decreases with frequency but some peaks
        amplitude = 0.05 / (harmonic**0.5)

        # Add some resonance peaks (violate limits)
        if harmonic in [5, 17, 42, 89]:  # Resonances
            amplitude *= 8.0

        signal += amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

    # Add some broadband noise from switching edges
    noise_bw = 10e6
    noise = np.random.normal(0, 0.002, n_samples)
    # Low-pass filter the noise
    from scipy import signal as sp_signal

    sos = sp_signal.butter(4, noise_bw / (sample_rate / 2), output="sos")
    signal += sp_signal.sosfilt(sos, noise)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=signal, metadata=metadata)


def generate_radiated_emissions_scan() -> list[tuple[float, float]]:
    """Generate radiated emissions scan data per CISPR 32.

    Returns frequency-amplitude pairs for 30 MHz - 1 GHz scan.
    """
    # Create frequency points (logarithmic spacing)
    frequencies = np.logspace(np.log10(30e6), np.log10(1e9), 500)

    emissions = []
    for freq in frequencies:
        # Base noise floor decreases with frequency
        noise_floor = 20 - 10 * np.log10(freq / 30e6)

        # Add random variation
        level = noise_floor + np.random.normal(0, 3)

        # Add some emission peaks (clock harmonics, etc.)
        # 25 MHz crystal and harmonics
        for harmonic in [1, 2, 3, 4, 5, 8]:
            peak_freq = 25e6 * harmonic
            if abs(freq - peak_freq) < 2e6:
                # Gaussian peak
                peak_height = 35 - 5 * harmonic
                width = 500e3
                level += peak_height * np.exp(-((freq - peak_freq) ** 2) / (2 * width**2))

        # 100 MHz oscillator
        if abs(freq - 100e6) < 5e6:
            level += 30 * np.exp(-((freq - 100e6) ** 2) / (2 * (1e6**2)))

        # WiFi at 2.4 GHz (violates limits)
        if 2.4e9 - 50e6 < freq < 2.4e9 + 50e6:
            level += 25 * np.exp(-((freq - 2.4e9) ** 2) / (2 * (10e6**2)))

        emissions.append((freq, level))

    return emissions


def generate_power_quality_harmonics() -> WaveformTrace:
    """Generate AC power signal with harmonics per IEC 61000-3-2.

    Simulates non-linear load (rectifier):
    - 60 Hz fundamental
    - Odd harmonics (3rd, 5th, 7th, etc.)
    - Some exceeding IEC limits
    """
    sample_rate = 100e3  # 100 kSa/s
    duration = 0.2  # 200 ms (12 cycles at 60 Hz)
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # 60 Hz fundamental at 120 V RMS
    fundamental = 120 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)

    # Add odd harmonics (typical of rectifier loads)
    harmonics = fundamental.copy()
    harmonic_data = [
        (3, 0.30),  # 30% - exceeds limit for this class
        (5, 0.20),  # 20% - at limit
        (7, 0.12),  # 12%
        (9, 0.08),  # 8%
        (11, 0.06),  # 6%
        (13, 0.04),  # 4%
        (15, 0.03),  # 3%
    ]

    for h, amplitude_ratio in harmonic_data:
        harmonics += fundamental * amplitude_ratio * np.sin(2 * np.pi * 60 * h * t)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=harmonics, metadata=metadata)


def generate_esd_transient_burst() -> WaveformTrace:
    """Generate ESD transient burst per IEC 61000-4-2.

    Simulates 8 kV contact discharge event.
    """
    sample_rate = 10e9  # 10 GSa/s for fast transient
    duration = 1e-6  # 1 us
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # ESD pulse: fast rise, exponential decay
    # IEC 61000-4-2: rise time 0.7-1 ns, first peak at 30 ns

    # Double exponential pulse
    rise_time = 0.8e-9  # 0.8 ns
    decay_time = 30e-9  # 30 ns

    # Pulse starts at 100 ns
    pulse_start = 100e-9
    pulse_time = t - pulse_start
    pulse_time[pulse_time < 0] = 0

    # Double exponential
    amplitude = 8000  # 8 kV
    pulse = amplitude * (np.exp(-pulse_time / decay_time) - np.exp(-pulse_time / rise_time))

    # Add ringing (damped oscillation)
    ring_freq = 100e6  # 100 MHz
    ring_decay = 50e-9
    ringing = (
        0.3
        * amplitude
        * np.sin(2 * np.pi * ring_freq * pulse_time)
        * np.exp(-pulse_time / ring_decay)
    )

    signal = pulse + ringing

    # Add noise floor
    signal += np.random.normal(0, 1.0, n_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=signal, metadata=metadata)


def generate_mil_std_461_ce102() -> WaveformTrace:
    """Generate MIL-STD-461 CE102 conducted emissions (10 kHz - 10 MHz).

    Simulates military equipment power supply emissions.
    """
    sample_rate = 100e6  # 100 MSa/s
    duration = 100e-6  # 100 us
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Noise floor
    signal = np.random.normal(0, 0.0005, n_samples)

    # Switching frequency at 50 kHz and harmonics
    switching_freq = 50e3
    for harmonic in range(1, 201):  # Up to 10 MHz
        freq = switching_freq * harmonic
        if freq > 10e6:
            break

        # MIL-STD-461 has stricter limits
        amplitude = 0.03 / (harmonic**0.6)

        # Some peaks that challenge limits
        if harmonic in [3, 11, 27, 64]:
            amplitude *= 5.0

        signal += amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=signal, metadata=metadata)


def save_cispr32_class_b_limits(output_dir: Path) -> None:
    """Save CISPR 32 Class B conducted emission limits."""
    limits = [
        # Frequency (Hz), QP Limit (dBuV), AV Limit (dBuV)
        (150e3, 66, 56),
        (500e3, 56, 46),
        (5e6, 56, 46),
        (30e6, 60, 50),
    ]

    filepath = output_dir / "cispr32_class_b.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "qp_limit_dbuv", "av_limit_dbuv"])
        writer.writerows(limits)

    print(f"  Saved: {filepath.name}")


def save_fcc_part15_class_b_limits(output_dir: Path) -> None:
    """Save FCC Part 15 Class B radiated emission limits."""
    limits = [
        # Frequency (Hz), Distance (m), Limit (dBuV/m)
        (30e6, 3, 40.0),
        (88e6, 3, 40.0),
        (216e6, 3, 43.5),
        (960e6, 3, 46.0),
        (1000e6, 3, 46.0),
    ]

    filepath = output_dir / "fcc_part15_class_b.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "distance_m", "limit_dbuv_per_m"])
        writer.writerows(limits)

    print(f"  Saved: {filepath.name}")


def save_iec61000_3_2_limits(output_dir: Path) -> None:
    """Save IEC 61000-3-2 harmonic current limits (Class A)."""
    limits = [
        # Harmonic, Max Current (A)
        (3, 2.30),
        (5, 1.14),
        (7, 0.77),
        (9, 0.40),
        (11, 0.33),
        (13, 0.21),
        (15, 0.15),
        (17, 0.13),
        (19, 0.11),
    ]

    filepath = output_dir / "iec61000-3-2.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["harmonic", "max_current_a"])
        writer.writerows(limits)

    print(f"  Saved: {filepath.name}")


def save_mil_std_461g_ce102_limits(output_dir: Path) -> None:
    """Save MIL-STD-461G CE102 limits."""
    # Piecewise linear limits
    limits = [
        # Frequency (Hz), Limit (dBuV)
        (10e3, 80),
        (2e6, 80),
        (10e6, 73),
    ]

    filepath = output_dir / "mil_std_461g_ce102.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "limit_dbuv"])
        writer.writerows(limits)

    print(f"  Saved: {filepath.name}")


def main() -> None:
    """Generate all demo data files."""
    print("=" * 70)
    print("Generating EMC/EMI Compliance Demo Data")
    print("=" * 70)

    # Create output directory
    output_dir = Path(__file__).parent
    limits_dir = output_dir / "compliance_limits"
    limits_dir.mkdir(exist_ok=True)

    # Generate waveform files
    print("\n--- Generating Waveform Files ---")

    print("Generating conducted_emissions_ac_line.npz...")
    ce_signal = generate_conducted_emissions_ac_line()
    np.savez(
        output_dir / "conducted_emissions_ac_line.npz",
        data=ce_signal.data,
        sample_rate=ce_signal.metadata.sample_rate,
    )
    print(f"  Duration: {ce_signal.duration * 1e6:.1f} us")
    print(f"  Sample rate: {ce_signal.metadata.sample_rate / 1e6:.0f} MSa/s")
    print("  Frequency range: 150 kHz - 30 MHz")

    print("\nGenerating radiated_emissions_scan.csv...")
    re_data = generate_radiated_emissions_scan()
    with open(output_dir / "radiated_emissions_scan.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "level_dbuv_per_m"])
        writer.writerows(re_data)
    print(f"  Points: {len(re_data)}")
    print("  Frequency range: 30 MHz - 1 GHz")

    print("\nGenerating power_quality_harmonics.npz...")
    pq_signal = generate_power_quality_harmonics()
    np.savez(
        output_dir / "power_quality_harmonics.npz",
        data=pq_signal.data,
        sample_rate=pq_signal.metadata.sample_rate,
    )
    print(f"  Duration: {pq_signal.duration * 1e3:.1f} ms")
    print(f"  Sample rate: {pq_signal.metadata.sample_rate / 1e3:.0f} kSa/s")
    print("  Fundamental: 60 Hz with harmonics")

    print("\nGenerating esd_transient_burst.npz...")
    esd_signal = generate_esd_transient_burst()
    np.savez(
        output_dir / "esd_transient_burst.npz",
        data=esd_signal.data,
        sample_rate=esd_signal.metadata.sample_rate,
    )
    print(f"  Duration: {esd_signal.duration * 1e6:.1f} us")
    print(f"  Sample rate: {esd_signal.metadata.sample_rate / 1e9:.0f} GSa/s")
    print("  Level: 8 kV contact discharge")

    print("\nGenerating mil_std_461_ce102.npz...")
    mil_signal = generate_mil_std_461_ce102()
    np.savez(
        output_dir / "mil_std_461_ce102.npz",
        data=mil_signal.data,
        sample_rate=mil_signal.metadata.sample_rate,
    )
    print(f"  Duration: {mil_signal.duration * 1e6:.1f} us")
    print(f"  Sample rate: {mil_signal.metadata.sample_rate / 1e6:.0f} MSa/s")
    print("  Frequency range: 10 kHz - 10 MHz")

    # Generate compliance limit files
    print("\n--- Generating Compliance Limit Files ---")
    save_cispr32_class_b_limits(limits_dir)
    save_fcc_part15_class_b_limits(limits_dir)
    save_iec61000_3_2_limits(limits_dir)
    save_mil_std_461g_ce102_limits(limits_dir)

    print("\n" + "=" * 70)
    print("Demo data generation complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - conducted_emissions_ac_line.npz")
    print("  - radiated_emissions_scan.csv")
    print("  - power_quality_harmonics.npz")
    print("  - esd_transient_burst.npz")
    print("  - mil_std_461_ce102.npz")
    print("  - compliance_limits/cispr32_class_b.csv")
    print("  - compliance_limits/fcc_part15_class_b.csv")
    print("  - compliance_limits/iec61000-3-2.csv")
    print("  - compliance_limits/mil_std_461g_ce102.csv")
    print("\nRun comprehensive_emc_demo.py to analyze these files.")


if __name__ == "__main__":
    main()
