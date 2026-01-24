"""Analysis Wizards

Demonstrates step-by-step guided analysis workflows:
- Interactive wizard-style workflows
- Automatic configuration based on signal type
- Decision tree-based guidance
- Progressive analysis with validation

This demonstration shows:
1. How to create guided analysis workflows
2. How to automatically configure analysis parameters
3. How to use decision trees for analysis selection
4. How to validate intermediate results
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_sine_wave,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class AnalysisWizardsDemo(BaseDemo):
    """Demonstrate wizard-style guided analysis."""

    def __init__(self) -> None:
        """Initialize analysis wizards demonstration."""
        super().__init__(
            name="analysis_wizards",
            description="Step-by-step guided analysis workflows with automatic configuration",
            capabilities=[
                "oscura.wizards.frequency_analysis",
                "oscura.wizards.power_quality",
                "oscura.wizards.signal_integrity",
                "oscura.wizards.automatic_configuration",
            ],
            related_demos=[
                "13_guidance/01_smart_recommendations.py",
                "13_guidance/03_onboarding_helpers.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for wizard demonstrations."""
        self.info("Creating test signals for wizard workflows...")

        # Clean sine wave for frequency analysis wizard
        sine = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        self.info("  ✓ Sine wave for frequency analysis")

        # Power signal with harmonics
        power = self._create_power_signal()
        self.info("  ✓ Power signal for power quality wizard")

        # Digital signal for integrity check
        digital = self._create_digital_signal()
        self.info("  ✓ Digital signal for signal integrity wizard")

        return {
            "sine": sine,
            "power": power,
            "digital": digital,
        }

    def _create_power_signal(self) -> WaveformTrace:
        """Create power signal with harmonics."""
        sample_rate = 10_000.0
        duration = 0.2
        t = np.arange(int(sample_rate * duration)) / sample_rate

        # 50 Hz fundamental + harmonics
        fundamental = np.sin(2 * np.pi * 50 * t)
        harmonic_3 = 0.15 * np.sin(2 * np.pi * 150 * t)
        harmonic_5 = 0.08 * np.sin(2 * np.pi * 250 * t)

        data = fundamental + harmonic_3 + harmonic_5
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_digital_signal(self) -> WaveformTrace:
        """Create digital signal with some noise."""
        sample_rate = 100_000.0
        duration = 0.01
        n_samples = int(sample_rate * duration)

        # Random digital pattern
        bit_period = int(sample_rate / 1000)  # 1 kHz bit rate
        data = np.zeros(n_samples)

        for i in range(0, n_samples, bit_period):
            end = min(i + bit_period, n_samples)
            data[i:end] = np.random.choice([0.0, 1.0])

        # Add noise and ringing
        noise = 0.05 * np.random.randn(n_samples)
        data = data + noise

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate analysis wizards."""
        results: dict[str, Any] = {}

        # Part 1: Frequency Analysis Wizard
        self.section("Part 1: Frequency Analysis Wizard")
        self.info("Starting frequency analysis wizard for sine wave...\n")

        freq_results = self._frequency_analysis_wizard(data["sine"])
        results["frequency_wizard"] = freq_results

        # Part 2: Power Quality Wizard
        self.section("Part 2: Power Quality Analysis Wizard")
        self.info("Starting power quality wizard for power signal...\n")

        power_results = self._power_quality_wizard(data["power"])
        results["power_quality_wizard"] = power_results

        # Part 3: Signal Integrity Wizard
        self.section("Part 3: Signal Integrity Wizard")
        self.info("Starting signal integrity wizard for digital signal...\n")

        integrity_results = self._signal_integrity_wizard(data["digital"])
        results["signal_integrity_wizard"] = integrity_results

        # Part 4: Automatic Configuration
        self.section("Part 4: Automatic Configuration Demo")

        for name, signal in data.items():
            self.subsection(f"Auto-configuring for: {name}")
            config = self._auto_configure(signal)

            self.info("Automatically configured parameters:")
            for param, value in config.items():
                self.info(f"  {param:20s}: {value}")

        return results

    def _frequency_analysis_wizard(self, signal: WaveformTrace) -> dict[str, Any]:
        """Step-by-step frequency analysis wizard."""
        results = {}

        # Step 1: Signal inspection
        self.info("Step 1: Inspecting signal characteristics")
        self.info(f"  Sample rate: {signal.metadata.sample_rate:.0f} Hz")
        self.info(f"  Duration: {len(signal.data) / signal.metadata.sample_rate * 1000:.1f} ms")
        self.info(f"  Samples: {len(signal.data)}")
        results["sample_rate"] = signal.metadata.sample_rate
        results["duration"] = len(signal.data) / signal.metadata.sample_rate

        # Step 2: Quality check
        self.info("\nStep 2: Checking signal quality")
        peak = np.max(np.abs(signal.data))
        rms = np.sqrt(np.mean(signal.data**2))
        self.info(f"  Peak amplitude: {peak:.3f}")
        self.info(f"  RMS amplitude: {rms:.3f}")

        quality = "Good" if peak < 0.95 and rms > 0.1 else "Check signal"
        self.info(f"  Quality assessment: {quality}")
        results["quality"] = quality

        # Step 3: FFT configuration
        self.info("\nStep 3: Configuring FFT parameters")
        fft_size = 2 ** int(np.ceil(np.log2(len(signal.data))))
        window = "Hann"
        self.info(f"  FFT size: {fft_size} (auto-selected)")
        self.info(f"  Window: {window} (recommended for general use)")
        results["fft_size"] = fft_size
        results["window"] = window

        # Step 4: Compute spectrum
        self.info("\nStep 4: Computing frequency spectrum")
        fft = np.fft.rfft(signal.data, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1 / signal.metadata.sample_rate)
        magnitude = np.abs(fft)

        peak_idx = np.argmax(magnitude[1:]) + 1
        peak_freq = freqs[peak_idx]
        self.info(f"  Peak frequency: {peak_freq:.2f} Hz")
        results["peak_frequency"] = peak_freq

        # Step 5: Results summary
        self.info("\nStep 5: Analysis complete")
        self.success(f"Dominant frequency identified: {peak_freq:.2f} Hz")

        return results

    def _power_quality_wizard(self, signal: WaveformTrace) -> dict[str, Any]:
        """Power quality analysis wizard."""
        results = {}

        # Step 1: Detect fundamental frequency
        self.info("Step 1: Detecting fundamental frequency")
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude = np.abs(fft)

        fundamental_idx = np.argmax(magnitude[1:]) + 1
        fundamental_freq = freqs[fundamental_idx]
        self.info(f"  Fundamental: {fundamental_freq:.1f} Hz")

        if abs(fundamental_freq - 50) < 2:
            grid_type = "50 Hz (Europe/Asia)"
        elif abs(fundamental_freq - 60) < 2:
            grid_type = "60 Hz (Americas)"
        else:
            grid_type = "Non-standard"

        self.info(f"  Grid type: {grid_type}")
        results["fundamental"] = fundamental_freq
        results["grid_type"] = grid_type

        # Step 2: Measure THD
        self.info("\nStep 2: Measuring Total Harmonic Distortion")
        fundamental_power = magnitude[fundamental_idx] ** 2

        harmonic_power = 0
        harmonics_found = []
        for n in range(2, 6):
            harmonic_freq = n * fundamental_freq
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if harmonic_idx < len(magnitude):
                h_power = magnitude[harmonic_idx] ** 2
                if h_power > fundamental_power * 0.001:  # >0.1% threshold
                    harmonic_power += h_power
                    harmonics_found.append((n, freqs[harmonic_idx], h_power))

        thd = 100 * np.sqrt(harmonic_power / fundamental_power)
        self.info(f"  THD: {thd:.2f}%")

        if harmonics_found:
            self.info(f"  Harmonics detected: {len(harmonics_found)}")
            for n, freq, power in harmonics_found:
                pct = 100 * np.sqrt(power / fundamental_power)
                self.info(f"    {n}th harmonic ({freq:.1f} Hz): {pct:.2f}%")

        results["thd"] = thd
        results["harmonics"] = len(harmonics_found)

        # Step 3: Power quality assessment
        self.info("\nStep 3: Power quality assessment")
        if thd < 5:
            quality = "Excellent (THD < 5%)"
        elif thd < 8:
            quality = "Good (THD < 8%)"
        elif thd < 12:
            quality = "Fair (THD < 12%)"
        else:
            quality = "Poor (THD >= 12%)"

        self.info(f"  Power quality: {quality}")
        results["power_quality"] = quality

        # Step 4: Recommendations
        self.info("\nStep 4: Recommendations")
        if thd > 5:
            self.info("  ⚠ Consider harmonic filtering")
        if len(harmonics_found) > 3:
            self.info("  ⚠ Multiple harmonics present - check loads")

        self.success("Power quality analysis complete")

        return results

    def _signal_integrity_wizard(self, signal: WaveformTrace) -> dict[str, Any]:
        """Signal integrity analysis wizard."""
        results = {}

        # Step 1: Detect signal levels
        self.info("Step 1: Detecting logic levels")
        data = signal.data

        # Find high and low levels using histogram
        hist, edges = np.histogram(data, bins=50)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append(edges[i])

        if len(peaks) >= 2:
            low_level = min(peaks)
            high_level = max(peaks)
        else:
            low_level = np.percentile(data, 10)
            high_level = np.percentile(data, 90)

        self.info(f"  Logic LOW: {low_level:.3f} V")
        self.info(f"  Logic HIGH: {high_level:.3f} V")
        self.info(f"  Swing: {high_level - low_level:.3f} V")

        results["low_level"] = low_level
        results["high_level"] = high_level

        # Step 2: Noise margin analysis
        self.info("\nStep 2: Analyzing noise margins")
        threshold = (low_level + high_level) / 2

        low_samples = data[data < threshold]
        high_samples = data[data >= threshold]

        if len(low_samples) > 0:
            low_noise = np.std(low_samples)
            self.info(f"  LOW state noise: {low_noise * 1000:.1f} mV")

        if len(high_samples) > 0:
            high_noise = np.std(high_samples)
            self.info(f"  HIGH state noise: {high_noise * 1000:.1f} mV")

        # Step 3: Edge analysis
        self.info("\nStep 3: Analyzing edges")

        # Find edges
        diff = np.diff(data)
        rising_edges = np.where(diff > (high_level - low_level) * 0.3)[0]
        falling_edges = np.where(diff < -(high_level - low_level) * 0.3)[0]

        self.info(f"  Rising edges: {len(rising_edges)}")
        self.info(f"  Falling edges: {len(falling_edges)}")

        results["rising_edges"] = len(rising_edges)
        results["falling_edges"] = len(falling_edges)

        # Step 4: Signal integrity assessment
        self.info("\nStep 4: Signal integrity assessment")

        issues = []
        if high_level - low_level < 0.5:
            issues.append("Low voltage swing")
        if len(low_samples) > 0 and low_noise > 0.1:
            issues.append("High noise on LOW state")
        if len(high_samples) > 0 and high_noise > 0.1:
            issues.append("High noise on HIGH state")

        if issues:
            self.info("  Issues detected:")
            for issue in issues:
                self.info(f"    ⚠ {issue}")
        else:
            self.success("  No integrity issues detected")

        results["issues"] = issues

        self.success("Signal integrity analysis complete")

        return results

    def _auto_configure(self, signal: WaveformTrace) -> dict[str, Any]:
        """Automatically configure analysis parameters."""
        config = {}

        # Determine FFT size
        config["fft_size"] = 2 ** int(np.ceil(np.log2(len(signal.data))))

        # Determine window based on signal length
        if len(signal.data) < 1000:
            config["window"] = "Rectangular"
        else:
            config["window"] = "Hann"

        # Determine if averaging is appropriate
        if len(signal.data) > 10000:
            config["averaging"] = "Recommended"
        else:
            config["averaging"] = "Not needed"

        # Frequency resolution
        config["freq_resolution_hz"] = signal.metadata.sample_rate / config["fft_size"]

        # Nyquist frequency
        config["nyquist_freq_hz"] = signal.metadata.sample_rate / 2

        return config

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating analysis wizards...")
        all_valid = True

        # Check all wizards ran
        required_wizards = [
            "frequency_wizard",
            "power_quality_wizard",
            "signal_integrity_wizard",
        ]

        for wizard in required_wizards:
            if wizard not in results:
                self.error(f"Missing {wizard} results")
                all_valid = False

        # Validate frequency wizard
        if "frequency_wizard" in results:
            fw = results["frequency_wizard"]
            if "peak_frequency" not in fw:
                self.error("Frequency wizard missing peak frequency")
                all_valid = False
            else:
                # Should be around 1000 Hz
                if not (900 < fw["peak_frequency"] < 1100):
                    self.warning(f"Peak frequency unexpected: {fw['peak_frequency']:.1f} Hz")

        # Validate power quality wizard
        if "power_quality_wizard" in results:
            pqw = results["power_quality_wizard"]
            if "thd" not in pqw:
                self.error("Power quality wizard missing THD")
                all_valid = False

        if all_valid:
            self.success("All analysis wizards validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = AnalysisWizardsDemo()
    success = demo.execute()
    exit(0 if success else 1)
