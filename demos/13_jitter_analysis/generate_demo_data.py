#!/usr/bin/env python3
"""Generate NPZ data files for jitter analysis demos.

This script generates test data files that can be used with the jitter demos
to demonstrate file loading capabilities.

Usage:
    python generate_demo_data.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def generate_prbs_with_ddj(
    sample_rate: float = 10e9,
    data_rate: float = 1e9,
    dcd_ps: float = 15.0,
    ddj_ps: float = 10.0,
    rj_rms_ps: float = 3.0,
    n_bits: int = 5000,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Generate PRBS pattern with DDJ and DCD.

    Args:
        sample_rate: Sampling rate in Hz.
        data_rate: Data rate in bps.
        dcd_ps: Duty cycle distortion in picoseconds.
        ddj_ps: Data-dependent jitter in picoseconds.
        rj_rms_ps: Random jitter RMS in picoseconds.
        n_bits: Number of bits to generate.

    Returns:
        Tuple of (waveform, bit_pattern).
    """
    bit_period = 1 / data_rate
    samples_per_bit = int(sample_rate / data_rate)

    # Generate PRBS-7 pattern
    prbs = np.zeros(n_bits, dtype=int)
    state = 0x7F  # Initial state
    for i in range(n_bits):
        bit = (state >> 6) ^ (state >> 5) & 1
        prbs[i] = state & 1
        state = ((state << 1) | bit) & 0x7F

    # Generate edge times with jitter
    edge_times = []
    current_time = 0.0
    prev_bit = 0
    run_length = 0

    for _i, bit in enumerate(prbs):
        if bit != prev_bit:
            # Add jitter to edge
            rj = rj_rms_ps * 1e-12 * np.random.randn()

            # DDJ: depends on run length (ISI)
            ddj_factor = min(run_length, 5) / 5.0
            ddj = ddj_ps * 1e-12 * ddj_factor * (1 if bit == 1 else -1)

            # DCD: rising edges early, falling edges late
            dcd = dcd_ps * 1e-12 / 2 * (1 if bit == 1 else -1)

            edge_time = current_time + rj + ddj + dcd
            edge_times.append((edge_time, bit, run_length))
            run_length = 0
        else:
            run_length += 1

        current_time += bit_period
        prev_bit = bit

    # Generate waveform from edges
    n_samples = int(n_bits * samples_per_bit)
    waveform = np.zeros(n_samples)
    state = 0

    for edge_time, bit, _ in edge_times:
        edge_idx = int(edge_time * sample_rate)
        if 0 <= edge_idx < n_samples:
            state = bit
            waveform[edge_idx:] = state

    # Add realistic rise/fall times
    rise_samples = int(0.1e-9 * sample_rate)  # 100 ps
    if rise_samples > 1:
        kernel = np.ones(rise_samples) / rise_samples
        waveform = np.convolve(waveform, kernel, mode="same")

    # Add noise
    waveform += 0.02 * np.random.randn(n_samples)

    # Scale to voltage
    waveform = waveform * 0.8 + 0.1  # 0.1V to 0.9V

    return waveform, prbs


def generate_ddj_dcd_data() -> None:
    """Generate DDJ/DCD demo data file."""
    print("Generating DDJ/DCD jitter data...")

    # Generation parameters
    sample_rate = 10e9  # 10 GHz
    data_rate = 1e9  # 1 Gbps
    dcd_ps = 15.0  # 15 ps DCD
    ddj_ps = 10.0  # 10 ps DDJ
    rj_rms_ps = 3.0  # 3 ps RMS RJ

    waveform, bit_pattern = generate_prbs_with_ddj(
        sample_rate=sample_rate,
        data_rate=data_rate,
        dcd_ps=dcd_ps,
        ddj_ps=ddj_ps,
        rj_rms_ps=rj_rms_ps,
        n_bits=5000,
    )

    # Save to NPZ
    output_path = Path(__file__).parent / "demo_data" / "ddj_dcd_jitter.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        waveform=waveform,
        bit_pattern=bit_pattern,
        sample_rate=sample_rate,
        data_rate=data_rate,
        dcd_ps=dcd_ps,
        ddj_ps=ddj_ps,
        rj_rms_ps=rj_rms_ps,
    )

    print(f"Created {output_path}")
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Bit pattern length: {len(bit_pattern)}")
    print(f"  Sample rate: {sample_rate / 1e9:.1f} GHz")
    print(f"  Data rate: {data_rate / 1e9:.1f} Gbps")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    generate_ddj_dcd_data()
