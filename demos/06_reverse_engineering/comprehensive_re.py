#!/usr/bin/env python3
"""Comprehensive Signal Reverse Engineering Demo using BaseDemo Pattern.

This demo demonstrates Oscura's signal reverse engineering capabilities:
- Waveform measurements (amplitude, timing, quality)
- Spectral analysis (FFT, PSD, spectrogram)
- Digital analysis (conversion, edge detection, timing)
- Pattern discovery and sequence analysis
- Protocol detection and decoding

Usage:
    python demos/04_signal_reverse_engineering/comprehensive_re.py
    python demos/04_signal_reverse_engineering/comprehensive_re.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from oscura.utils.builders.signal_builder import SignalBuilder
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from oscura.core.types import TraceMetadata, WaveformTrace


class SignalReverseEngineeringDemo(BaseDemo):
    """Signal Reverse Engineering Demonstration.

    Demonstrates Oscura's complete reverse engineering workflow for
    analyzing unknown signals and discovering protocol characteristics.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="Comprehensive Signal Reverse Engineering",
            description="Demonstrates signal reverse engineering workflow",
            **kwargs,
        )
        self.sample_rate = 10e6  # 10 MHz
        self.trace = None

    def generate_test_data(self) -> dict:
        """Generate or load test signal for reverse engineering.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data file if exists
        3. Generate synthetic data using SignalBuilder
        """
        import numpy as np

        # Try loading from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("unknown_uart_like.npz"):
            data_file_to_load = default_file
            print_info(f"Loading data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load)
                # Use first channel (ch1 or uart)
                if "uart" in data:
                    signal_data = data["uart"]
                elif "ch1" in data:
                    signal_data = data["ch1"]
                else:
                    # Use first available array
                    signal_data = data[next(iter(data.keys()))]

                loaded_sample_rate = float(data["sample_rate"])

                self.trace = WaveformTrace(
                    data=signal_data,
                    metadata=TraceMetadata(
                        sample_rate=loaded_sample_rate,
                        channel_name="Mystery_Signal",
                        source_file=str(data_file_to_load),
                    ),
                )

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Sample rate", f"{loaded_sample_rate / 1e6:.1f}", "MHz")
                print_result("Samples", len(self.trace.data))
                print_result(
                    "Duration",
                    f"{len(self.trace.data) / loaded_sample_rate * 1e3:.3f}",
                    "ms",
                )
                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic generation")

        # 3. Generate synthetic data as fallback
        print_info("Generating synthetic mystery signal...")

        # Create a UART-like signal with some noise
        channels = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.01)
            .add_uart(baud_rate=115200, data=b"Hello Oscura!", amplitude=3.3)
            .add_noise(snr_db=30, channel="uart")
            .build_channels()
        )

        self.trace = channels["uart"]

        print_result("Sample rate", f"{self.sample_rate / 1e6:.1f}", "MHz")
        print_result("Samples", len(self.trace.data))
        print_result(
            "Duration",
            f"{len(self.trace.data) / self.sample_rate * 1e3:.3f}",
            "ms",
        )

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Execute reverse engineering analysis."""
        # === Phase 1: Waveform Measurements ===
        print_subheader("Phase 1: Waveform Measurements")
        self._analyze_waveform()

        # === Phase 2: Spectral Analysis ===
        print_subheader("Phase 2: Spectral Analysis")
        self._analyze_spectral()

        # === Phase 3: Digital Analysis ===
        print_subheader("Phase 3: Digital Analysis")
        self._analyze_digital()

        # === Phase 4: Pattern Discovery ===
        print_subheader("Phase 4: Pattern Discovery")
        self._analyze_patterns()

        # === Phase 5: Protocol Detection ===
        print_subheader("Phase 5: Protocol Detection")
        self._analyze_protocol()

        return self.results

    def _analyze_waveform(self) -> None:
        """Perform waveform measurements."""
        # Basic measurements
        mean_val = osc.mean(self.trace)
        rms_val = osc.rms(self.trace)
        amp_val = osc.amplitude(self.trace)
        min_val = float(np.min(self.trace.data))
        max_val = float(np.max(self.trace.data))

        print_result("Mean", f"{mean_val:.6f}", "V")
        print_result("RMS", f"{rms_val:.6f}", "V")
        print_result("Vpp", f"{amp_val:.6f}", "V")
        print_result("Range", f"[{min_val:.3f}, {max_val:.3f}]", "V")

        self.results["mean"] = mean_val
        self.results["rms"] = rms_val
        self.results["amplitude"] = amp_val

        # Quality metrics
        try:
            snr_val = osc.snr(self.trace)
            print_result("SNR", f"{snr_val:.2f}", "dB")
            self.results["snr"] = snr_val
        except Exception:
            print_info("SNR calculation N/A")

    def _analyze_spectral(self) -> None:
        """Perform spectral analysis."""
        # FFT
        freq, mag = osc.fft(self.trace)
        fft_peak_idx = np.argmax(mag[1 : len(mag) // 2]) + 1
        fft_peak_freq = freq[fft_peak_idx]

        print_result("FFT bins", len(freq))
        print_result("Peak frequency", f"{fft_peak_freq / 1e3:.3f}", "kHz")

        self.results["fft_bins"] = len(freq)
        self.results["fft_peak_freq"] = fft_peak_freq

        # Band power analysis
        total_power = float(np.sum(mag**2))
        low_power = float(np.sum(mag[freq < 1000] ** 2))
        mid_power = float(np.sum(mag[(freq >= 1000) & (freq < 100000)] ** 2))
        float(np.sum(mag[freq >= 100000] ** 2))

        print_result("Total power", f"{total_power:.2e}")
        print_result("Power <1kHz", f"{100 * low_power / total_power:.1f}%")
        print_result("Power 1-100kHz", f"{100 * mid_power / total_power:.1f}%")

        self.results["total_power"] = total_power

    def _analyze_digital(self) -> None:
        """Perform digital signal analysis."""
        # Auto-threshold conversion
        digital = osc.to_digital(self.trace, threshold="auto")
        duty_cycle = 100 * np.sum(digital.data) / len(digital.data)

        print_result("Duty cycle", f"{duty_cycle:.1f}%")
        self.results["duty_cycle"] = duty_cycle

        # Edge detection
        rising = osc.detect_edges(digital, edge_type="rising")
        falling = osc.detect_edges(digital, edge_type="falling")

        print_result("Rising edges", len(rising))
        print_result("Falling edges", len(falling))

        self.results["rising_edges"] = len(rising)
        self.results["falling_edges"] = len(falling)

        # Edge timing analysis
        if len(rising) > 1:
            periods = np.diff(rising)
            avg_period = float(np.mean(periods)) / self.sample_rate
            period_std = float(np.std(periods)) / self.sample_rate

            print_result("Avg period", f"{avg_period * 1e6:.3f}", "us")
            print_result("Period std", f"{period_std * 1e9:.1f}", "ns")

            self.results["avg_period_us"] = avg_period * 1e6
            self.results["period_jitter_pct"] = (
                100 * period_std / avg_period if avg_period > 0 else 0
            )

        # Logic family detection
        try:
            logic_family = osc.detect_logic_family(self.trace)
            print_result("Logic family", logic_family)
            self.results["logic_family"] = logic_family
        except Exception:
            print_info("Logic family detection N/A")

    def _analyze_patterns(self) -> None:
        """Discover patterns in the signal."""
        # Convert to bytes
        digital = osc.to_digital(self.trace, threshold="auto")
        bits = digital.data.astype(np.uint8)

        if len(bits) >= 8:
            num_bytes = len(bits) // 8
            bytes_data = np.zeros(num_bytes, dtype=np.uint8)
            for i in range(num_bytes):
                byte_val = 0
                for j in range(8):
                    byte_val = (byte_val << 1) | bits[i * 8 + j]
                bytes_data[i] = byte_val

            # Pattern statistics
            unique, counts = np.unique(bytes_data, return_counts=True)
            entropy = float(-np.sum((counts / num_bytes) * np.log2(counts / num_bytes + 1e-10)))

            print_result("Unique bytes", len(unique))
            print_result("Total bytes", num_bytes)
            print_result("Entropy", f"{entropy:.2f}", "bits")

            self.results["unique_bytes"] = len(unique)
            self.results["entropy"] = entropy

            # Top patterns
            counter = Counter(bytes_data)
            top_patterns = counter.most_common(3)
            for byte_val, count in top_patterns:
                pct = 100 * count / num_bytes
                print_info(f"  Pattern 0x{byte_val:02X}: {count} times ({pct:.1f}%)")

    def _analyze_protocol(self) -> None:
        """Detect and analyze protocol."""
        # Baud rate detection
        try:
            from oscura.utils.autodetect import detect_baud_rate

            baud_rate, confidence = detect_baud_rate(self.trace, return_confidence=True)
            print_result("Detected baud rate", baud_rate)
            print_result("Confidence", f"{confidence:.1%}")

            self.results["baud_rate"] = baud_rate
            self.results["baud_confidence"] = confidence

            # Try UART decoding
            if baud_rate:
                try:
                    frames = osc.decode_uart(self.trace, baud_rate=baud_rate)
                    print_result("UART frames decoded", len(frames))
                    self.results["uart_frames"] = len(frames)
                except Exception:
                    print_info("UART decoding failed")
        except Exception:
            print_info("Baud rate detection N/A")

        # Protocol detection
        try:
            protocol = osc.detect_protocol(self.trace)
            print_result("Detected protocol", protocol if protocol else "Unknown")
            self.results["detected_protocol"] = protocol
        except Exception:
            print_info("Protocol detection N/A")

    def validate(self, results: dict) -> bool:
        """Validate reverse engineering results."""
        suite = ValidationSuite()

        # Waveform measurements
        amplitude = results.get("amplitude", 0)
        suite.add_check("Amplitude measured", amplitude > 0, f"Got {amplitude} V")

        # Spectral analysis
        fft_bins = results.get("fft_bins", 0)
        suite.add_check("FFT computed", fft_bins > 0, f"Got {fft_bins} bins")

        # Digital analysis
        rising_edges = results.get("rising_edges", 0)
        suite.add_check("Edges detected", rising_edges > 0, f"Got {rising_edges} edges")

        # Pattern analysis
        unique_bytes = results.get("unique_bytes", 0)
        suite.add_check("Patterns discovered", unique_bytes > 0, f"Got {unique_bytes} unique bytes")

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(SignalReverseEngineeringDemo))
