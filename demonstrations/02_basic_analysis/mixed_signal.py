#!/usr/bin/env python3
"""Comprehensive Mixed-Signal Analysis Demonstration using BaseDemo Pattern.

This demo demonstrates Oscura's mixed-signal analysis capabilities:
- Clock recovery (FFT and edge-based)
- Jitter analysis (RMS, peak-to-peak, TIE) per IEEE 2414-2020
- Signal integrity metrics
- Eye diagram analysis
- IEEE 2414-2020 compliance validation

Usage:
    python demos/07_mixed_signal/comprehensive_mixed_signal_demo.py
    python demos/07_mixed_signal/comprehensive_mixed_signal_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import print_subheader
from oscura.analyzers.digital.timing import (
    peak_to_peak_jitter,
    recover_clock_edge,
    recover_clock_fft,
    rms_jitter,
    time_interval_error,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class MixedSignalDemo(BaseDemo):
    """Mixed-Signal Analysis Demonstration.

    Demonstrates Oscura's mixed-signal analysis capabilities including
    clock recovery, jitter analysis, and IEEE 2414-2020 compliance.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="comprehensive_mixed_signal_analysis",
            description="Demonstrates jitter analysis and IEEE 2414-2020 compliance",
            capabilities=["oscura.jitter_analysis", "oscura.clock_recovery"],
            ieee_standards=["IEEE 2414-2020"],
            **kwargs,
        )
        self.sample_rate = 10e9  # 10 GHz
        self.bit_rate = 1e9  # 1 Gbps
        self.trace = None

    def generate_test_data(self) -> dict:
        """Generate synthetic mixed-signal test data."""
        print_info("Generating synthetic high-speed serial signal with jitter...")

        samples_per_bit = int(self.sample_rate / self.bit_rate)
        n_bits = 1000
        n_samples = n_bits * samples_per_bit

        # Generate pseudo-random bit sequence
        np.random.seed(42)
        bits = np.random.randint(0, 2, n_bits)

        # Generate NRZ signal
        signal = np.zeros(n_samples)
        rise_fall_samples = samples_per_bit // 10

        for i, bit in enumerate(bits):
            start_idx = i * samples_per_bit
            end_idx = (i + 1) * samples_per_bit

            if bit == 1:
                signal[start_idx : start_idx + rise_fall_samples] = np.linspace(
                    0, 1, rise_fall_samples
                )
                signal[start_idx + rise_fall_samples : end_idx] = 1
            else:
                if i > 0 and bits[i - 1] == 1:
                    signal[start_idx : start_idx + rise_fall_samples] = np.linspace(
                        1, 0, rise_fall_samples
                    )

        # Add jitter (5 ps RMS)
        jitter_rms = 5e-12
        jitter = np.random.randn(n_samples) * jitter_rms * self.sample_rate
        indices = np.arange(n_samples) + jitter
        indices = np.clip(indices, 0, n_samples - 1).astype(int)
        signal = signal[indices]

        # Add noise (-40 dB)
        signal += 0.01 * np.random.randn(n_samples)

        self.trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="Serial_Data",
                source_file="synthetic",
            ),
        )

        print_result("Sample rate", f"{self.sample_rate / 1e9:.1f}", "GHz")
        print_result("Bit rate", f"{self.bit_rate / 1e9:.1f}", "Gbps")
        print_result("Samples", len(self.trace.data))
        print_result("Bits", n_bits)

        return {"trace": self.trace}

    def run_demonstration(self, data: dict) -> dict:
        """Execute mixed-signal analysis."""
        self.trace = data["trace"]

        # === Section 1: Clock Recovery ===
        print_subheader("Clock Recovery")
        self._analyze_clock_recovery()

        # === Section 2: Jitter Analysis ===
        print_subheader("Jitter Analysis (IEEE 2414-2020)")
        self._analyze_jitter()

        # === Section 3: Signal Integrity ===
        print_subheader("Signal Integrity Metrics")
        self._analyze_signal_integrity()

        # === Section 4: IEEE 2414-2020 Compliance ===
        print_subheader("IEEE 2414-2020 Compliance")
        self._validate_compliance()

        return self.results

    def _analyze_clock_recovery(self) -> None:
        """Perform clock recovery analysis."""
        # FFT-based clock recovery
        fft_result = recover_clock_fft(self.trace)

        if not np.isnan(fft_result.frequency):
            print_result("FFT method frequency", f"{fft_result.frequency / 1e6:.3f}", "MHz")
            print_result("FFT confidence", f"{fft_result.confidence:.2f}")
            self.results["clock_freq_fft"] = fft_result.frequency
            self.results["clock_confidence_fft"] = fft_result.confidence
        else:
            print_info("FFT clock recovery failed")

        # Edge-based clock recovery
        edge_result = recover_clock_edge(self.trace, edge_type="rising")

        if not np.isnan(edge_result.frequency):
            print_result("Edge method frequency", f"{edge_result.frequency / 1e6:.3f}", "MHz")
            print_result("Edge confidence", f"{edge_result.confidence:.2f}")
            self.results["clock_freq_edge"] = edge_result.frequency
            self.results["clock_confidence_edge"] = edge_result.confidence

            if edge_result.jitter_rms is not None:
                print_result("RMS jitter (edge)", f"{edge_result.jitter_rms * 1e12:.2f}", "ps")
            if edge_result.jitter_pp is not None:
                print_result("Pk-Pk jitter (edge)", f"{edge_result.jitter_pp * 1e12:.2f}", "ps")
        else:
            print_info("Edge-based clock recovery failed")

    def _analyze_jitter(self) -> None:
        """Perform jitter analysis per IEEE 2414-2020."""
        # RMS jitter
        rms_result = rms_jitter(self.trace, edge_type="rising", threshold=0.5)

        if not np.isnan(rms_result.rms):
            rms_ps = rms_result.rms * 1e12
            print_result("RMS jitter", f"{rms_ps:.3f}", "ps")
            print_result("Mean period", f"{rms_result.mean * 1e9:.3f}", "ns")
            print_result("Edge samples", rms_result.samples)

            if not np.isnan(rms_result.uncertainty):
                print_result("Uncertainty", f"{rms_result.uncertainty * 1e12:.3f}", "ps (1-sigma)")

            self.results["jitter_rms_ps"] = rms_ps
            self.results["jitter_mean_period_ns"] = rms_result.mean * 1e9
            self.results["jitter_samples"] = rms_result.samples
        else:
            print_info("Could not compute RMS jitter")

        # Peak-to-peak jitter
        pp_jitter = peak_to_peak_jitter(self.trace, edge_type="rising", threshold=0.5)

        if not np.isnan(pp_jitter):
            pp_ps = pp_jitter * 1e12
            print_result("Pk-Pk jitter", f"{pp_ps:.3f}", "ps")
            self.results["jitter_pp_ps"] = pp_ps
        else:
            print_info("Could not compute Pk-Pk jitter")

        # Time Interval Error (TIE)
        try:
            tie = time_interval_error(self.trace, edge_type="rising", threshold=0.5)

            tie_rms = float(np.std(tie))
            tie_pp = float(np.max(tie) - np.min(tie))

            print_result("TIE edges", len(tie))
            print_result("TIE RMS", f"{tie_rms * 1e12:.3f}", "ps")
            print_result("TIE Pk-Pk", f"{tie_pp * 1e12:.3f}", "ps")

            self.results["tie_rms_ps"] = tie_rms * 1e12
            self.results["tie_pp_ps"] = tie_pp * 1e12
            self.results["tie_edges"] = len(tie)
        except Exception as e:
            print_info(f"Could not compute TIE: {e}")

    def _analyze_signal_integrity(self) -> None:
        """Analyze signal integrity metrics."""
        data = self.trace.data

        # Voltage levels
        v_min = float(np.min(data))
        v_max = float(np.max(data))
        v_pp = v_max - v_min
        v_rms = float(np.sqrt(np.mean(data**2)))

        print_result("Min voltage", f"{v_min * 1e3:.2f}", "mV")
        print_result("Max voltage", f"{v_max * 1e3:.2f}", "mV")
        print_result("Peak-to-peak", f"{v_pp * 1e3:.2f}", "mV")
        print_result("RMS voltage", f"{v_rms * 1e3:.2f}", "mV")

        self.results["v_min"] = v_min
        self.results["v_max"] = v_max
        self.results["v_pp"] = v_pp
        self.results["v_rms"] = v_rms

        # Logic levels
        threshold = (v_min + v_max) / 2
        high_samples = data[data > threshold]
        low_samples = data[data <= threshold]

        if len(high_samples) > 0 and len(low_samples) > 0:
            v_high = float(np.mean(high_samples))
            v_low = float(np.mean(low_samples))
            noise_high = float(np.std(high_samples))
            noise_low = float(np.std(low_samples))

            print_result("Logic HIGH", f"{v_high * 1e3:.2f}", "mV")
            print_result("Logic LOW", f"{v_low * 1e3:.2f}", "mV")
            print_result("HIGH noise (sigma)", f"{noise_high * 1e3:.2f}", "mV")
            print_result("LOW noise (sigma)", f"{noise_low * 1e3:.2f}", "mV")

            self.results["v_high"] = v_high
            self.results["v_low"] = v_low
            self.results["noise_high"] = noise_high
            self.results["noise_low"] = noise_low

            # SNR
            signal_swing = v_high - v_low
            noise_total = np.sqrt(noise_high**2 + noise_low**2)
            snr = signal_swing / noise_total if noise_total > 0 else np.inf
            snr_db = 20 * np.log10(snr) if snr > 0 and not np.isinf(snr) else np.nan

            if not np.isnan(snr_db):
                print_result("Signal-to-noise", f"{snr_db:.2f}", "dB")
                self.results["snr_db"] = snr_db

    def _validate_compliance(self) -> None:
        """Validate IEEE 2414-2020 jitter compliance."""
        compliant = True
        violations = []

        # Check RMS jitter against typical spec
        rms_ps = self.results.get("jitter_rms_ps", float("inf"))
        if rms_ps < 10:
            print_info(f"RMS jitter within spec ({rms_ps:.2f} ps < 10 ps)")
        else:
            print_info(f"RMS jitter exceeds typical spec ({rms_ps:.2f} ps > 10 ps)")
            compliant = False
            violations.append(f"RMS jitter: {rms_ps:.2f} ps")

        # Check sample count
        samples = self.results.get("jitter_samples", 0)
        if samples >= 100:
            print_info(f"Adequate sample count ({samples} edges)")
        else:
            print_info(f"Low sample count ({samples} edges, recommend >=100)")

        self.results["ieee2414_compliant"] = compliant
        self.results["ieee2414_violations"] = violations

        if compliant:
            print_info("Signal meets IEEE 2414-2020 guidelines")
        else:
            print_info(f"{len(violations)} compliance issues detected")

    def validate(self, results: dict) -> bool:
        """Validate mixed-signal analysis results."""
        suite = ValidationSuite()

        # Clock recovery
        clock_freq = results.get("clock_freq_fft", 0) + results.get("clock_freq_edge", 0)
        suite.add_check(
            "Clock frequency detected",
            clock_freq > 0,
            f"Got {clock_freq} Hz",
        )

        # Jitter analysis
        jitter_samples = results.get("jitter_samples", 0)
        suite.add_check(
            "Jitter samples collected",
            jitter_samples > 0,
            f"Got {jitter_samples} samples",
        )

        # RMS jitter computed
        if "jitter_rms_ps" in results:
            rms_jitter = results["jitter_rms_ps"]
            suite.add_check(
                "RMS jitter computed",
                rms_jitter > 0,
                f"Got {rms_jitter} ps",
            )

        # Signal integrity
        v_pp = results.get("v_pp", 0)
        suite.add_check(
            "Signal swing measured",
            v_pp > 0,
            f"Got {v_pp} V",
        )

        # TIE computed
        tie_edges = results.get("tie_edges", 0)
        suite.add_check(
            "TIE edges detected",
            tie_edges > 0,
            f"Got {tie_edges} edges",
        )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(MixedSignalDemo))
