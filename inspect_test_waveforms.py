#!/usr/bin/env python3
"""Inspect test waveforms to understand actual signal characteristics.

This script loads test WFM files and displays their actual properties
to help debug validation failures.
"""

from pathlib import Path

import numpy as np

import oscura as osc
from oscura.core.types import WaveformTrace


def inspect_waveform(filepath: Path) -> None:
    """Inspect a single waveform file."""
    print(f"\n{'=' * 80}")
    print(f"File: {filepath.name}")
    print(f"{'=' * 80}")

    try:
        loaded = osc.load(filepath)
        if not isinstance(loaded, WaveformTrace):
            print(f"ERROR: Not a WaveformTrace, got {type(loaded)}")
            return

        data = loaded.data
        sr = loaded.metadata.sample_rate

        print(f"Samples: {len(data)}")
        print(f"Sample rate: {sr:.2e} Hz")
        print(f"Duration: {len(data) / sr:.6f} s")
        print("\nData Statistics:")
        print(f"  Min: {data.min():.6e}")
        print(f"  Max: {data.max():.6e}")
        print(f"  Mean: {np.mean(data):.6e}")
        print(f"  Std Dev: {np.std(data):.6e}")
        print(f"  Peak-to-Peak: {data.max() - data.min():.6e}")
        print(f"  RMS (numpy): {np.sqrt(np.mean(data**2)):.6e}")

        # Check for DC offset
        if abs(np.mean(data)) > 0.01:
            print(f"  DC Offset: {np.mean(data):.6e} V")

        # Simple frequency detection
        from scipy import signal as scipy_signal

        freqs, psd = scipy_signal.welch(data, sr, nperseg=min(1024, len(data)))
        peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
        dominant_freq = freqs[peak_idx]
        print(f"\nDominant Frequency (Welch): {dominant_freq:.3f} Hz")

    except Exception as e:
        print(f"ERROR: {e}")


def main() -> None:
    """Main entry point."""
    test_data_dir = Path("test_data/synthetic")

    test_files = [
        "basic/sine_1khz.wfm",
        "basic/square_5khz.wfm",
        "basic/triangle_2khz.wfm",
        "edge_cases/dc_signal.wfm",
        "edge_cases/noisy_signal_snr20.wfm",
        "edge_cases/sine_with_noise_snr30.wfm",
        "advanced/mixed_harmonics.wfm",
        "advanced/pulse_train_10pct.wfm",
    ]

    for filename in test_files:
        filepath = test_data_dir / filename
        if filepath.exists():
            inspect_waveform(filepath)
        else:
            print(f"\nSkipping {filename} (not found)")


if __name__ == "__main__":
    main()
