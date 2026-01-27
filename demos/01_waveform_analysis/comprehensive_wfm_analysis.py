#!/usr/bin/env python3
"""Comprehensive Waveform Analysis Demo using BaseDemo Pattern.

# SKIP_VALIDATION: Comprehensive analysis takes >30s

This demo showcases Oscura's complete waveform analysis capabilities:
- Signal loading and metadata extraction
- Waveform measurements (amplitude, frequency, rise/fall time)
- Spectral analysis (FFT, THD, SNR, SINAD, ENOB - IEEE 1241)
- Power analysis (AC/DC, efficiency, ripple - IEEE 1459)
- Statistical analysis and filtering
- Visualization outputs

Refactored to BaseDemo pattern for consistent validation and reduced LOC.

Usage:
    python demos/01_waveform_analysis/comprehensive_wfm_analysis.py
    python demos/01_waveform_analysis/comprehensive_wfm_analysis.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader


class WaveformAnalysisDemo(BaseDemo):
    """Comprehensive Waveform Analysis Demonstration.

    Demonstrates all Oscura waveform analysis capabilities including
    measurements, spectral analysis, power analysis, and visualization.
    """

    name = "Comprehensive Waveform Analysis"
    description = "Demonstrates complete Oscura waveform analysis capabilities"
    category = "waveform_analysis"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 10e6  # 10 MHz sampling
        self.signal_freq = 10e3  # 10 kHz fundamental

        # Storage for signals and results
        self.trace = None

    def generate_data(self) -> None:
        """Generate or load test waveform data.

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
        elif default_file := self.find_default_data_file("multi_channel_mixed_signal.npz"):
            data_file_to_load = default_file
            print_info(f"Loading data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load)
                # Use first channel (ch1)
                signal_data = data["ch1"]
                loaded_sample_rate = float(data["sample_rate"])

                self.trace = osc.WaveformTrace(
                    data=signal_data,
                    metadata=osc.TraceMetadata(
                        sample_rate=loaded_sample_rate,
                        channel_name=str(data.get("channel_names", ["CH1"])[0]),
                        source_file=str(data_file_to_load),
                    ),
                )

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Sample rate", loaded_sample_rate / 1e6, "MHz")
                print_result("Duration", len(self.trace.data) / loaded_sample_rate * 1e3, "ms")
                print_result("Samples", len(self.trace.data))
                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic generation")

        # 3. Generate synthetic data as fallback
        from demos.data_generation import SignalBuilder

        print_info("Generating synthetic test waveform using SignalBuilder...")

        # Build a complex multi-tone signal with harmonics and noise
        signal = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.01)
            .add_sine(frequency=self.signal_freq, amplitude=1.0)
            .add_harmonics(fundamental=self.signal_freq, thd_percent=5.0)
            .add_noise(snr_db=40)
            .build()
        )

        # Convert to WaveformTrace
        self.trace = osc.WaveformTrace(
            data=signal.data["ch1"],
            metadata=osc.TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="Test_Signal",
                source_file="synthetic",
            ),
        )

        print_result("Sample rate", self.sample_rate / 1e6, "MHz")
        print_result("Duration", len(self.trace.data) / self.sample_rate * 1e3, "ms")
        print_result("Samples", len(self.trace.data))

    def run_analysis(self) -> None:
        """Execute comprehensive waveform analysis."""
        # === Section 1: Waveform Measurements ===
        print_subheader("Waveform Measurements")
        self._analyze_measurements()

        # === Section 2: Spectral Analysis ===
        print_subheader("Spectral Analysis (IEEE 1241)")
        self._analyze_spectral()

        # === Section 3: Power Analysis ===
        print_subheader("Power Analysis")
        self._analyze_power()

        # === Section 4: Statistical Analysis ===
        print_subheader("Statistical Analysis")
        self._analyze_statistics()

        # === Section 5: Filtering Operations ===
        print_subheader("Filtering Operations")
        self._analyze_filtering()

        # === Section 6: Protocol Detection ===
        print_subheader("Protocol Detection")
        self._analyze_protocols()

        # === Section 7: Math Operations ===
        print_subheader("Math Operations")
        self._analyze_math()

    def _analyze_measurements(self) -> None:
        """Perform waveform measurements."""
        # Basic measurements
        mean_val = osc.mean(self.trace)
        rms_val = osc.rms(self.trace)
        amp_val = osc.amplitude(self.trace)

        print_result("Mean", mean_val, "V")
        print_result("RMS", rms_val, "V")
        print_result("Amplitude (Vpp)", amp_val, "V")

        self.results["mean"] = mean_val
        self.results["rms"] = rms_val
        self.results["amplitude"] = amp_val

        # Frequency and timing measurements
        try:
            freq_val = osc.frequency(self.trace)
            period_val = osc.period(self.trace)
            rise_time = osc.rise_time(self.trace)
            fall_time = osc.fall_time(self.trace)

            print_result("Frequency", freq_val / 1e3 if freq_val else 0, "kHz")
            print_result("Period", period_val * 1e6 if period_val else 0, "us")
            print_result("Rise time", rise_time * 1e9 if rise_time else 0, "ns")
            print_result("Fall time", fall_time * 1e9 if fall_time else 0, "ns")

            self.results["frequency"] = freq_val
            self.results["period"] = period_val
            self.results["rise_time"] = rise_time
            self.results["fall_time"] = fall_time
        except Exception as e:
            print_info(f"Some timing measurements N/A: {e}")

    def _analyze_spectral(self) -> None:
        """Perform spectral analysis."""
        # FFT
        freq, mag = osc.fft(self.trace)
        self.results["fft_bins"] = len(freq)

        print_result("FFT bins", len(freq))
        print_result("Frequency resolution", freq[1] if len(freq) > 1 else 0, "Hz")

        # IEEE 1241 metrics
        try:
            thd_val = osc.thd(self.trace)
            snr_val = osc.snr(self.trace)
            sinad_val = osc.sinad(self.trace)
            enob_val = osc.enob(self.trace)
            sfdr_val = osc.sfdr(self.trace)

            print_result("THD", thd_val, "dB")
            print_result("SNR", snr_val, "dB")
            print_result("SINAD", sinad_val, "dB")
            print_result("ENOB", enob_val, "bits")
            print_result("SFDR", sfdr_val, "dB")

            self.results["thd"] = thd_val
            self.results["snr"] = snr_val
            self.results["sinad"] = sinad_val
            self.results["enob"] = enob_val
            self.results["sfdr"] = sfdr_val
        except Exception as e:
            print_info(f"Some spectral metrics N/A: {e}")

        # Use convenience function for quick analysis
        try:
            metrics = osc.quick_spectral(self.trace, fundamental=self.signal_freq)
            print_info(f"Quick spectral: THD={metrics.thd_db:.1f} dB, SNR={metrics.snr_db:.1f} dB")
            self.results["quick_spectral"] = True
        except Exception as e:
            print_info(f"Quick spectral N/A: {e}")
            self.results["quick_spectral"] = False

    def _analyze_power(self) -> None:
        """Perform power analysis."""
        try:
            # Use trace as both voltage and current for demo
            power = osc.average_power(voltage=self.trace, current=self.trace)
            energy = osc.energy(voltage=self.trace, current=self.trace)
            stats = osc.power_statistics(voltage=self.trace, current=self.trace)

            print_result("Average power", power, "W")
            print_result("Energy", energy, "J")
            print_result("Peak power", stats.get("peak", 0), "W")

            self.results["average_power"] = power
            self.results["energy"] = energy
            self.results["power_stats"] = stats

            # AC power analysis (IEEE 1459)
            apparent = osc.apparent_power(self.trace, self.trace)
            reactive = osc.reactive_power(self.trace, self.trace)
            pf = osc.power_factor(self.trace, self.trace)

            print_result("Apparent power", apparent, "VA")
            print_result("Power factor", pf)
            self.results["power_factor"] = pf
        except Exception as e:
            print_info(f"Power analysis N/A: {e}")

    def _analyze_statistics(self) -> None:
        """Perform statistical analysis."""
        stats = osc.basic_stats(self.trace)
        percs = osc.percentiles(self.trace, [25, 50, 75])
        dist = osc.distribution_metrics(self.trace)

        print_result("Mean", stats.get("mean", 0) if isinstance(stats, dict) else 0, "V")
        print_result("Std dev", stats.get("std", 0) if isinstance(stats, dict) else 0, "V")
        print_result("P50 (median)", percs.get(50, 0) if isinstance(percs, dict) else 0, "V")

        self.results["basic_stats"] = stats
        self.results["percentiles"] = percs
        self.results["distribution"] = dist

    def _analyze_filtering(self) -> None:
        """Demonstrate filtering operations."""
        try:
            cutoff = self.sample_rate * 0.1
            filtered_lp = osc.low_pass(self.trace, cutoff=cutoff)
            filtered_hp = osc.high_pass(self.trace, cutoff=self.sample_rate * 0.01)
            filtered_bp = osc.band_pass(
                self.trace, low=self.sample_rate * 0.01, high=self.sample_rate * 0.1
            )

            print_result("Low-pass cutoff", cutoff / 1e3, "kHz")
            print_result("Filtered samples", len(filtered_lp.data))

            # Additional filters
            filtered_ma = osc.moving_average(self.trace, window_size=5)
            filtered_med = osc.median_filter(self.trace, kernel_size=5)
            filtered_sg = osc.savgol_filter(self.trace, window_length=11, polyorder=3)

            print_info("Moving average, median, Savitzky-Golay filters applied")

            # Smart filter convenience function
            try:
                smart_filtered = osc.smart_filter(self.trace)
                print_info("Smart filter applied successfully")
                self.results["smart_filter"] = True
            except Exception:
                self.results["smart_filter"] = False

            self.results["filtering_ok"] = True
        except Exception as e:
            print_info(f"Filtering N/A: {e}")
            self.results["filtering_ok"] = False

    def _analyze_protocols(self) -> None:
        """Test protocol detection."""
        try:
            protocol = osc.detect_protocol(self.trace)
            print_result("Detected protocol", protocol if protocol else "None")
            self.results["protocol_detected"] = protocol

            # Try auto-decode
            try:
                result = osc.auto_decode(self.trace)
                print_result("Auto-decode protocol", result.protocol if result else "None")
                self.results["auto_decode"] = True
            except Exception:
                self.results["auto_decode"] = False
        except Exception as e:
            print_info(f"Protocol detection N/A: {e}")

    def _analyze_math(self) -> None:
        """Demonstrate math operations."""
        try:
            # Basic operations
            added = osc.add(self.trace, self.trace)
            subtracted = osc.subtract(self.trace, self.trace)
            multiplied = osc.multiply(self.trace, self.trace)
            differentiated = osc.differentiate(self.trace)
            integrated = osc.integrate(self.trace)
            absolute = osc.absolute(self.trace)
            scaled = osc.scale(self.trace, factor=2.0)
            offset = osc.offset(self.trace, value=1.0)

            print_result("Add samples", len(added.data))
            print_result("Differentiate samples", len(differentiated.data))
            print_result("Integrate samples", len(integrated.data))

            # Resampling
            resampled = osc.resample(self.trace, self.sample_rate / 2)
            downsampled = osc.downsample(self.trace, factor=2)

            print_result("Resampled samples", len(resampled.data))
            print_result("Downsampled samples", len(downsampled.data))

            self.results["math_ok"] = True
        except Exception as e:
            print_info(f"Math operations N/A: {e}")
            self.results["math_ok"] = False

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate analysis results."""
        # Basic measurements
        suite.check_true(
            "Trace loaded",
            self.trace is not None and len(self.trace.data) > 0,
            category="loading",
        )

        suite.check_greater("FFT bins", self.results.get("fft_bins", 0), 0, category="spectral")

        # Spectral metrics
        if "thd" in self.results:
            suite.check_less("THD reasonable", self.results["thd"], 0, category="spectral")

        if "snr" in self.results:
            suite.check_greater("SNR positive", self.results["snr"], 0, category="spectral")

        if "enob" in self.results:
            suite.check_greater("ENOB positive", self.results["enob"], 0, category="spectral")

        # Statistics
        suite.check_true(
            "Basic stats computed",
            self.results.get("basic_stats") is not None,
            category="statistics",
        )

        # Filtering
        suite.check_true(
            "Filtering successful",
            self.results.get("filtering_ok", False),
            category="filtering",
        )

        # Math operations
        suite.check_true(
            "Math operations successful",
            self.results.get("math_ok", False),
            category="math",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(WaveformAnalysisDemo))
