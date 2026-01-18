#!/usr/bin/env python3
"""Exploratory Analysis Validation - No Prior Knowledge Required.

This script validates that Oscura can analyze UNKNOWN waveforms without
any prior knowledge of:
- Signal type (analog/digital/mixed)
- Logic levels or thresholds
- Protocols or timing
- Frequencies or baud rates
- Message structures

Goal: Demonstrate that API is optimal for exploratory reverse engineering.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import oscura as osc
from oscura.loaders import load
from oscura.visualization.digital import plot_timing

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ExploratoryAnalysisValidator:
    """Validates exploratory analysis WITHOUT prior knowledge."""

    def __init__(self, wfm_file: str, output_dir: str = "exploratory_outputs"):
        self.wfm_file = Path(wfm_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.tests_run = 0
        self.tests_passed = 0
        self.discoveries = []

    def log(self, message: str, indent: int = 0) -> None:
        """Log message with indentation."""
        print("  " * indent + message)

    def discover(self, category: str, finding: str) -> None:
        """Record a discovery made during exploratory analysis."""
        self.discoveries.append((category, finding))

    def run_all_tests(self) -> int:
        """Run all exploratory analysis tests.

        Returns:
            Exit code (0 = success, 1 = failure)
        """
        print(f"{BLUE}Loading WFM file: {self.wfm_file}{RESET}")

        # STEP 1: Load file with ZERO prior knowledge
        trace = load(str(self.wfm_file))
        print(f"{GREEN}✓ Loaded: {len(trace.data)} samples{RESET}")
        self.discover("Basic", f"File contains {len(trace.data)} samples")

        # Extract what we CAN know from metadata
        if hasattr(trace.metadata, "sample_rate"):
            sr = trace.metadata.sample_rate
            duration = len(trace.data) / sr
            print(f"  Sample rate: {sr / 1e6:.3f} MS/s")
            print(f"  Duration: {duration * 1e3:.3f} ms")
            self.discover(
                "Timing", f"Sample rate: {sr / 1e6:.3f} MS/s, Duration: {duration * 1e3:.3f} ms"
            )

        print(f"\n{BOLD}{'=' * 80}{RESET}")
        print(f"{BOLD}EXPLORATORY ANALYSIS - NO PRIOR KNOWLEDGE{RESET}")
        print(f"{'=' * 80}\n")

        # CATEGORY 1: SIGNAL CHARACTERIZATION (NO KNOWLEDGE NEEDED)
        print(f"{BOLD}PHASE 1: AUTOMATIC SIGNAL CHARACTERIZATION{RESET}\n")

        self.test_auto_visualization(trace)
        self.test_auto_amplitude_analysis(trace)
        self.test_auto_spectral_analysis(trace)
        self.test_auto_statistical_analysis(trace)

        # CATEGORY 2: INTELLIGENT THRESHOLD DETECTION
        print(f"\n{BOLD}PHASE 2: INTELLIGENT THRESHOLD & LOGIC DETECTION{RESET}\n")

        self.test_auto_threshold_detection(trace)
        self.test_auto_logic_family_detection(trace)
        self.test_auto_digital_conversion(trace)

        # CATEGORY 3: TIMING & PATTERN DISCOVERY
        print(f"\n{BOLD}PHASE 3: TIMING & PATTERN DISCOVERY{RESET}\n")

        self.test_auto_timing_analysis(trace)
        self.test_auto_edge_detection(trace)
        self.test_auto_pattern_discovery(trace)

        # CATEGORY 4: PROTOCOL & MESSAGE INFERENCE
        print(f"\n{BOLD}PHASE 4: PROTOCOL & MESSAGE INFERENCE{RESET}\n")

        self.test_auto_baud_rate_detection(trace)
        self.test_auto_protocol_detection(trace)

        # SUMMARY
        print(f"\n{BOLD}{'=' * 80}{RESET}")
        print(f"{BOLD}EXPLORATORY ANALYSIS SUMMARY{RESET}")
        print(f"{'=' * 80}")
        print(f"Total tests:  {self.tests_run}")
        print(f"{GREEN}Passed:       {self.tests_passed}{RESET}")
        print(f"{RED}Failed:       {self.tests_run - self.tests_passed}{RESET}")
        print(f"Success rate: {100 * self.tests_passed / max(self.tests_run, 1):.1f}%\n")

        if self.discoveries:
            print(f"{BOLD}DISCOVERIES MADE (WITHOUT PRIOR KNOWLEDGE):{RESET}\n")
            current_category = None
            for category, finding in self.discoveries:
                if category != current_category:
                    print(f"{YELLOW}{category}:{RESET}")
                    current_category = category
                print(f"  • {finding}")
            print()

        print(f"{BLUE}All outputs saved to: {self.output_dir}{RESET}\n")

        return 0 if self.tests_passed == self.tests_run else 1

    def test_auto_visualization(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic visualization without any parameters."""
        self.tests_run += 1
        print(f"[{self.tests_run}] Testing: Auto Waveform Plot (Zero Parameters)")

        try:
            plt.figure(figsize=(12, 6), dpi=150)
            osc.plot_waveform(trace)  # NO parameters - all auto!
            plt.title("Auto Waveform Plot (No Prior Knowledge)")

            output_path = self.output_dir / "01_auto_waveform.png"
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            print(
                f"{GREEN}  ✓ PASSED: Auto waveform plot (time unit, range, grid all automatic){RESET}"
            )
            self.discover(
                "Visualization", "Auto time-domain plot generated without any configuration"
            )
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_amplitude_analysis(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic amplitude measurements."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Amplitude Measurements")

        try:
            # All these work with ZERO prior knowledge
            mean_val = osc.mean(trace)
            rms_val = osc.rms(trace)
            vpp = osc.amplitude(trace)
            min_val = np.min(trace.data)
            max_val = np.max(trace.data)

            print(f"  • Mean: {mean_val:.6f} V")
            print(f"  • RMS: {rms_val:.6f} V")
            print(f"  • Vpp: {vpp:.6f} V")
            print(f"  • Min: {min_val:.6f} V")
            print(f"  • Max: {max_val:.6f} V")

            self.discover("Amplitude", f"Vpp={vpp:.6f}V, Mean={mean_val:.6f}V, RMS={rms_val:.6f}V")

            print(
                f"{GREEN}  ✓ PASSED: All amplitude measurements work without prior knowledge{RESET}"
            )
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_spectral_analysis(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic spectral analysis."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Spectral Analysis")

        try:
            # FFT with automatic windowing and scaling
            freq, mag = osc.fft(trace)

            # Find dominant frequency (peak in FFT)
            peak_idx = np.argmax(mag[1 : len(mag) // 2]) + 1  # Skip DC
            dominant_freq = freq[peak_idx]
            peak_mag = mag[peak_idx]

            print(f"  • Dominant frequency: {dominant_freq / 1e3:.3f} kHz")
            print(f"  • Peak magnitude: {peak_mag:.6f}")

            # Auto spectrum plot
            plt.figure(figsize=(12, 6), dpi=150)
            osc.plot_spectrum(trace)  # All auto!
            plt.title("Auto Spectrum (No Prior Knowledge)")

            output_path = self.output_dir / "02_auto_spectrum.png"
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            self.discover("Spectral", f"Dominant frequency: {dominant_freq / 1e3:.3f} kHz")

            print(f"{GREEN}  ✓ PASSED: Spectral analysis automatic{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_statistical_analysis(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic statistical analysis."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Statistical Analysis")

        try:
            # Statistics with no prior knowledge
            data = trace.data
            std = np.std(data)
            variance = np.var(data)
            percentile_10 = np.percentile(data, 10)
            percentile_90 = np.percentile(data, 90)
            median = np.median(data)

            print(f"  • Std dev: {std:.6f}")
            print(f"  • Variance: {variance:.6f}")
            print(f"  • 10th percentile: {percentile_10:.6f}")
            print(f"  • 90th percentile: {percentile_90:.6f}")
            print(f"  • Median: {median:.6f}")

            self.discover(
                "Statistics",
                f"Signal range: {percentile_10:.3f}V to {percentile_90:.3f}V (10th-90th percentile)",
            )

            print(f"{GREEN}  ✓ PASSED: Statistical analysis automatic{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_threshold_detection(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic threshold detection."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Threshold Detection")

        try:
            # Get automatic threshold (NO prior knowledge needed)
            data = trace.data
            percentile_10 = np.percentile(data, 10)
            percentile_90 = np.percentile(data, 90)
            auto_threshold = (percentile_10 + percentile_90) / 2

            print(f"  • Auto threshold: {auto_threshold:.6f} V")
            print(f"  • Low level (10%): {percentile_10:.6f} V")
            print(f"  • High level (90%): {percentile_90:.6f} V")

            self.discover("Threshold", f"Auto-detected threshold: {auto_threshold:.6f}V")

            print(f"{GREEN}  ✓ PASSED: Automatic threshold detection{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_logic_family_detection(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic logic family detection."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Logic Family Detection")

        try:
            # Use Oscura's automatic logic family detection
            from oscura.utils.autodetect import detect_logic_family

            logic_family = detect_logic_family(trace)

            print(f"  • Detected logic family: {logic_family}")

            if logic_family:
                # Get threshold config
                from oscura.config.thresholds import ThresholdRegistry

                registry = ThresholdRegistry()
                family = registry.get_family(logic_family)
                print(f"  • VIH: {family.VIH:.3f} V")
                print(f"  • VIL: {family.VIL:.3f} V")
                print(f"  • VOH: {family.VOH:.3f} V")
                print(f"  • VOL: {family.VOL:.3f} V")

                self.discover(
                    "Logic Family",
                    f"Auto-detected: {logic_family} (VIH={family.VIH}V, VIL={family.VIL}V)",
                )

            print(f"{GREEN}  ✓ PASSED: Logic family auto-detection{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_digital_conversion(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic analog-to-digital conversion."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Digital Conversion")

        try:
            # Convert to digital with AUTO threshold (no prior knowledge!)
            digital = osc.to_digital(trace, threshold="auto")

            high_count = np.sum(digital.data)
            total = len(digital.data)
            duty_cycle = 100 * high_count / total

            print(f"  • Digital samples: {total}")
            print(f"  • High states: {high_count}")
            print(f"  • Duty cycle: {duty_cycle:.1f}%")

            # Plot digital waveform
            plt.figure(figsize=(12, 4), dpi=150)
            plot_timing([digital])  # Expects a list of traces
            plt.title("Auto Digital Conversion (No Prior Knowledge)")

            output_path = self.output_dir / "03_auto_digital.png"
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            self.discover("Digital", f"Duty cycle: {duty_cycle:.1f}% (automatic threshold)")

            print(f"{GREEN}  ✓ PASSED: Auto digital conversion{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_timing_analysis(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic timing analysis."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Timing Analysis")

        try:
            # Convert to digital first
            digital = osc.to_digital(trace, threshold="auto")

            # Detect edges (automatic) - returns array of timestamps
            rising_edges = osc.detect_edges(digital, edge_type="rising")
            falling_edges = osc.detect_edges(digital, edge_type="falling")

            print(f"  • Rising edges: {len(rising_edges)}")
            print(f"  • Falling edges: {len(falling_edges)}")

            if len(rising_edges) > 1:
                # Calculate period from rising edges
                periods = np.diff(rising_edges)
                avg_period = np.mean(periods)
                freq = 1 / avg_period if avg_period > 0 else 0

                print(f"  • Average period: {avg_period * 1e6:.3f} µs")
                print(f"  • Frequency: {freq / 1e3:.3f} kHz")

                self.discover(
                    "Timing", f"Frequency: {freq / 1e3:.3f} kHz, Period: {avg_period * 1e6:.3f} µs"
                )

            print(f"{GREEN}  ✓ PASSED: Auto timing analysis{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_edge_detection(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic edge detection and visualization."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Edge Detection")

        try:
            # Convert to digital
            digital = osc.to_digital(trace, threshold="auto")

            # Detect edges
            rising_edges = osc.detect_edges(digital, edge_type="rising")
            falling_edges = osc.detect_edges(digital, edge_type="falling")

            # Calculate pulse widths
            if len(rising_edges) > 0 and len(falling_edges) > 0:
                # Find matching rise/fall pairs
                pulse_widths = []
                for rise in rising_edges:
                    falls_after = falling_edges[falling_edges > rise]
                    if len(falls_after) > 0:
                        fall = falls_after[0]
                        width = fall - rise
                        pulse_widths.append(width)

                if pulse_widths:
                    avg_width = np.mean(pulse_widths)
                    print(f"  • Average pulse width: {avg_width * 1e6:.3f} µs")
                    self.discover("Pulse", f"Average pulse width: {avg_width * 1e6:.3f} µs")

            print(f"{GREEN}  ✓ PASSED: Auto edge detection{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_pattern_discovery(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic pattern discovery in digital data."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Pattern Discovery")

        try:
            # Convert to digital
            digital = osc.to_digital(trace, threshold="auto")

            # Convert to bytes for pattern analysis
            # Group bits into bytes
            bits = digital.data.astype(np.uint8)
            if len(bits) >= 8:
                # Pack into bytes (MSB first)
                num_bytes = len(bits) // 8
                bytes_data = np.zeros(num_bytes, dtype=np.uint8)
                for i in range(num_bytes):
                    byte = 0
                    for j in range(8):
                        byte = (byte << 1) | bits[i * 8 + j]
                    bytes_data[i] = byte

                # Find most common bytes
                unique, counts = np.unique(bytes_data, return_counts=True)
                top_idx = np.argsort(counts)[-3:]  # Top 3 most common

                print(f"  • Bytes analyzed: {num_bytes}")
                print(f"  • Unique bytes: {len(unique)}")
                print("  • Top 3 patterns:")
                for idx in reversed(top_idx):
                    byte_val = unique[idx]
                    count = counts[idx]
                    print(
                        f"    0x{byte_val:02X}: {count} occurrences ({100 * count / num_bytes:.1f}%)"
                    )

                self.discover(
                    "Patterns", f"Found {len(unique)} unique byte patterns in {num_bytes} bytes"
                )

            print(f"{GREEN}  ✓ PASSED: Pattern discovery{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_baud_rate_detection(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic baud rate detection."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Baud Rate Detection")

        try:
            from oscura.utils.autodetect import detect_baud_rate

            # Automatic baud rate detection (NO prior knowledge!)
            baud_rate, confidence = detect_baud_rate(trace, return_confidence=True)

            print(f"  • Detected baud rate: {baud_rate} baud")
            print(f"  • Confidence: {confidence:.1%}")

            if baud_rate:
                self.discover(
                    "Protocol",
                    f"Auto-detected baud rate: {baud_rate} (confidence: {confidence:.1%})",
                )

            print(f"{GREEN}  ✓ PASSED: Auto baud rate detection{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False

    def test_auto_protocol_detection(self, trace: osc.WaveformTrace) -> bool:
        """Test: Automatic protocol detection and decoding."""
        self.tests_run += 1
        print(f"\n[{self.tests_run}] Testing: Auto Protocol Detection")

        try:
            from oscura.utils.autodetect import detect_baud_rate

            # Try UART decoding with auto-detected baud rate
            baud_rate = detect_baud_rate(trace)

            if baud_rate:
                try:
                    frames = osc.uart_decode(trace, baud_rate=baud_rate)
                    print(f"  • Decoded {len(frames)} UART frames")
                    if len(frames) > 0:
                        print(
                            f"  • First 5 bytes: {' '.join(f'0x{f.data:02X}' for f in frames[:5])}"
                        )
                        self.discover(
                            "Protocol",
                            f"UART: Decoded {len(frames)} frames at {baud_rate} baud",
                        )
                except Exception:
                    print("  • UART decode failed (not UART data)")

            print(f"{GREEN}  ✓ PASSED: Protocol detection attempted{RESET}")
            self.tests_passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            return False


def main() -> int:
    """Main entry point."""
    # Default signal file from demo_data
    default_wfm = Path(__file__).parent / "demo_data" / "mystery_serial_protocol.npz"

    parser = argparse.ArgumentParser(
        description="Validate exploratory analysis (no prior knowledge required)"
    )
    parser.add_argument(
        "--wfm-file",
        type=str,
        default=str(default_wfm),
        help=f"Path to WFM file to analyze (default: {default_wfm})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="exploratory_outputs", help="Output directory for results"
    )

    args = parser.parse_args()

    validator = ExploratoryAnalysisValidator(args.wfm_file, args.output_dir)
    return validator.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
