"""Onboarding Helpers

Demonstrates interactive tutorials and getting-started assistance:
- Interactive tutorials for common tasks
- Example data and templates
- Common task shortcuts
- Learning progression paths

This demonstration shows:
1. How to provide interactive tutorials
2. How to create example templates
3. How to simplify common tasks
4. How to guide learning progression
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING

from demonstrations.common import (
    BaseDemo,
    generate_sine_wave,
)

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


class OnboardingHelpersDemo(BaseDemo):
    """Demonstrate onboarding and learning assistance."""

    def __init__(self) -> None:
        """Initialize onboarding helpers demonstration."""
        super().__init__(
            name="onboarding_helpers",
            description="Interactive tutorials and learning assistance for new users",
            capabilities=[
                "oscura.onboarding.tutorials",
                "oscura.onboarding.templates",
                "oscura.onboarding.shortcuts",
                "oscura.onboarding.learning_paths",
            ],
            related_demos=[
                "13_guidance/01_smart_recommendations.py",
                "13_guidance/02_analysis_wizards.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate tutorial example data."""
        self.info("Creating tutorial example signals...")

        # Simple sine wave for basic tutorial
        sine = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        self.info("  ✓ Basic sine wave")

        return {"sine": sine}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate onboarding helpers."""
        results: dict[str, Any] = {}

        # Part 1: Interactive Tutorial System
        self.section("Part 1: Interactive Tutorial - First Analysis")
        self.info("Welcome to Oscura! Let's perform your first signal analysis.\n")

        tutorial_results = self._run_basic_tutorial(data["sine"])
        results["tutorial"] = tutorial_results

        # Part 2: Common Task Templates
        self.section("Part 2: Common Task Templates")

        templates = self._show_task_templates()
        self.info("Available quick-start templates:\n")

        for i, template in enumerate(templates, 1):
            self.info(f"{i}. {template['name']}")
            self.info(f"   Use case: {template['use_case']}")
            self.info(f"   Code template: {template['code']}")
            self.info("")

        results["templates"] = templates

        # Part 3: Task Shortcuts
        self.section("Part 3: Common Task Shortcuts")

        shortcuts = self._demonstrate_shortcuts()
        self.info("Time-saving shortcuts for common tasks:\n")

        for shortcut in shortcuts:
            self.info(f"Task: {shortcut['task']}")
            self.info(f"  Full code:  {shortcut['full']}")
            self.info(f"  Shortcut:   {shortcut['shortcut']}")
            self.info(f"  Saved:      {shortcut['benefit']}")
            self.info("")

        results["shortcuts"] = shortcuts

        # Part 4: Learning Progression Path
        self.section("Part 4: Learning Progression Path")

        learning_path = self._create_learning_path()
        self.info("Recommended learning progression:\n")

        for level, modules in learning_path.items():
            self.subsection(level)
            for module in modules:
                self.info(f"  {module['number']}. {module['title']}")
                self.info(f"     Focus: {module['focus']}")
                self.info(f"     Demo:  {module['demo']}")
                self.info("")

        results["learning_path"] = learning_path

        # Part 5: Quick Reference Guide
        self.section("Part 5: Quick Reference Guide")

        quick_ref = self._generate_quick_reference()
        self.info("Quick reference for common operations:\n")

        for category, items in quick_ref.items():
            self.subsection(category)
            for item in items:
                self.info(f"  {item}")

        results["quick_reference"] = quick_ref

        return results

    def _run_basic_tutorial(self, signal: WaveformTrace) -> dict[str, Any]:
        """Run interactive basic tutorial."""
        results = {}

        self.info("Tutorial Step 1: Understanding Your Signal")
        self.info("=" * 60)
        self.info("Every signal in Oscura has these key properties:")
        self.info(f"  • Sample Rate: {signal.metadata.sample_rate} Hz")
        self.info("    (How many measurements per second)")
        self.info(f"  • Data Points: {len(signal.data)}")
        self.info("    (Total number of samples)")
        self.info(f"  • Duration: {len(signal.data) / signal.metadata.sample_rate * 1000:.1f} ms")
        self.info("    (How long the signal lasts)")
        self.info("")

        results["properties_learned"] = True

        self.info("Tutorial Step 2: Basic Measurements")
        self.info("=" * 60)
        self.info("Let's measure some basic signal characteristics:\n")

        peak = np.max(np.abs(signal.data))
        rms = np.sqrt(np.mean(signal.data**2))
        mean = np.mean(signal.data)

        self.info(f"  • Peak Amplitude: {peak:.3f} V")
        self.info("    (Maximum signal value)")
        self.info(f"  • RMS Amplitude: {rms:.3f} V")
        self.info("    (Average power level)")
        self.info(f"  • DC Offset: {mean:.4f} V")
        self.info("    (Average value)")
        self.info("")

        results["measurements"] = {"peak": peak, "rms": rms, "mean": mean}

        self.info("Tutorial Step 3: Finding the Frequency")
        self.info("=" * 60)
        self.info("To find what frequencies are in the signal, we use FFT:\n")

        # Simple FFT
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude = np.abs(fft)

        peak_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx]

        self.info(f"  • Dominant Frequency: {peak_freq:.1f} Hz")
        self.info("    (Main frequency component)")
        self.info("")

        results["frequency"] = peak_freq

        self.info("Tutorial Step 4: Quality Check")
        self.info("=" * 60)
        self.info("Always check signal quality:\n")

        # Simple quality checks
        is_clipped = peak > 0.95
        has_dc_offset = abs(mean) > 0.1
        is_low_level = peak < 0.1

        self.info(f"  • Clipping: {'YES ⚠' if is_clipped else 'No ✓'}")
        self.info(f"  • DC Offset: {'YES ⚠' if has_dc_offset else 'No ✓'}")
        self.info(f"  • Signal Level: {'LOW ⚠' if is_low_level else 'Good ✓'}")
        self.info("")

        results["quality_checks"] = {
            "clipped": is_clipped,
            "dc_offset": has_dc_offset,
            "low_level": is_low_level,
        }

        self.success("Tutorial Complete! You've learned the basics of signal analysis.")
        self.info("\nNext Steps:")
        self.info("  • Try analyzing your own signals")
        self.info("  • Explore more advanced measurements")
        self.info("  • Check out the guided wizards")

        return results

    def _show_task_templates(self) -> list[dict[str, str]]:
        """Show common task templates."""
        return [
            {
                "name": "Load and Analyze CSV File",
                "use_case": "Quick analysis of oscilloscope CSV export",
                "code": "signal = load_csv('data.csv'); spectrum = fft(signal)",
            },
            {
                "name": "Measure Power Quality",
                "use_case": "Check THD and harmonics in power signals",
                "code": "results = analyze_power_quality(signal, fundamental_freq=50)",
            },
            {
                "name": "Protocol Decode",
                "use_case": "Decode UART/SPI/I2C from logic analyzer",
                "code": "frames = decode_uart(signal, baud_rate=9600)",
            },
            {
                "name": "Spectral Analysis",
                "use_case": "Frequency domain analysis with plots",
                "code": "spectrum = analyze_spectrum(signal, window='hann', plot=True)",
            },
            {
                "name": "Compare Signals",
                "use_case": "Compare before/after signals",
                "code": "diff = compare_signals(signal1, signal2, metrics=['snr', 'thd'])",
            },
        ]

    def _demonstrate_shortcuts(self) -> list[dict[str, str]]:
        """Demonstrate common shortcuts."""
        return [
            {
                "task": "Quick FFT",
                "full": "fft_result = np.fft.rfft(signal.data); freqs = np.fft.rfftfreq(len(signal.data), 1/signal.metadata.sample_rate)",
                "shortcut": "spectrum = signal.fft()",
                "benefit": "One line instead of two",
            },
            {
                "task": "Find Peak Frequency",
                "full": "fft = np.fft.rfft(signal.data); idx = np.argmax(abs(fft)[1:])+1; freq = np.fft.rfftfreq(len(signal.data), 1/signal.metadata.sample_rate)[idx]",
                "shortcut": "freq = signal.peak_frequency()",
                "benefit": "90% less code",
            },
            {
                "task": "Measure All Parameters",
                "full": "peak = max(abs(signal.data)); rms = sqrt(mean(signal.data**2)); freq = ...",
                "shortcut": "params = signal.measure_all()",
                "benefit": "Comprehensive measurements in one call",
            },
            {
                "task": "Filter Signal",
                "full": "from scipy.signal import butter, filtfilt; b, a = butter(4, cutoff/(sr/2)); filtered = filtfilt(b, a, data)",
                "shortcut": "filtered = signal.lowpass(cutoff_hz=1000)",
                "benefit": "Simplified filtering",
            },
        ]

    def _create_learning_path(self) -> dict[str, list[dict[str, str]]]:
        """Create learning progression path."""
        return {
            "Level 1: Foundations": [
                {
                    "number": "1.1",
                    "title": "Loading Signals",
                    "focus": "Different file formats and signal creation",
                    "demo": "demonstrations/01_data_loading/",
                },
                {
                    "number": "1.2",
                    "title": "Basic Measurements",
                    "focus": "Peak, RMS, frequency measurements",
                    "demo": "demonstrations/02_basic_analysis/",
                },
                {
                    "number": "1.3",
                    "title": "Visualization",
                    "focus": "Time and frequency domain plots",
                    "demo": "demonstrations/02_basic_analysis/",
                },
            ],
            "Level 2: Analysis": [
                {
                    "number": "2.1",
                    "title": "Spectral Analysis",
                    "focus": "FFT, power spectral density, spectrograms",
                    "demo": "demonstrations/04_advanced_analysis/",
                },
                {
                    "number": "2.2",
                    "title": "Protocol Decoding",
                    "focus": "UART, SPI, I2C, CAN decoding",
                    "demo": "demonstrations/03_protocol_decoding/",
                },
                {
                    "number": "2.3",
                    "title": "Power Analysis",
                    "focus": "THD, harmonics, power quality",
                    "demo": "demonstrations/05_domain_specific/",
                },
            ],
            "Level 3: Advanced": [
                {
                    "number": "3.1",
                    "title": "Reverse Engineering",
                    "focus": "Unknown protocol discovery",
                    "demo": "demonstrations/06_reverse_engineering/",
                },
                {
                    "number": "3.2",
                    "title": "Custom Analysis",
                    "focus": "Writing custom analyzers",
                    "demo": "demonstrations/08_extensibility/",
                },
                {
                    "number": "3.3",
                    "title": "Automation",
                    "focus": "Batch processing, workflows",
                    "demo": "demonstrations/16_complete_workflows/",
                },
            ],
        }

    def _generate_quick_reference(self) -> dict[str, list[str]]:
        """Generate quick reference guide."""
        return {
            "Loading Data": [
                "signal = load_vcd('file.vcd')  # Load VCD file",
                "signal = load_csv('file.csv')  # Load CSV file",
                "metadata = TraceMetadata(sample_rate); signal = WaveformTrace(data, metadata)  # Create from array",
            ],
            "Basic Measurements": [
                "peak = signal.peak()  # Peak amplitude",
                "rms = signal.rms()  # RMS amplitude",
                "freq = signal.frequency()  # Dominant frequency",
            ],
            "Frequency Analysis": [
                "spectrum = signal.fft()  # Compute FFT",
                "psd = signal.power_spectral_density()  # PSD",
                "sgram = signal.spectrogram()  # Time-frequency",
            ],
            "Quality Checks": [
                "score = signal.quality_score()  # Overall quality (0-100)",
                "warnings = signal.check_quality()  # List of issues",
                "is_good = signal.validate()  # Quick pass/fail",
            ],
            "Protocol Decoding": [
                "frames = decode_uart(signal, baud=9600)",
                "frames = decode_spi(signal, mode=0)",
                "frames = decode_i2c(signal)",
            ],
        }

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating onboarding helpers...")
        all_valid = True

        # Check tutorial ran
        if "tutorial" not in results:
            self.error("Tutorial did not run")
            return False

        tutorial = results["tutorial"]
        if not tutorial.get("properties_learned"):
            self.error("Tutorial did not complete properties section")
            all_valid = False

        # Check templates exist
        if "templates" not in results or len(results["templates"]) < 3:
            self.error("Insufficient task templates")
            all_valid = False

        # Check shortcuts exist
        if "shortcuts" not in results or len(results["shortcuts"]) < 3:
            self.error("Insufficient shortcuts")
            all_valid = False

        # Check learning path exists
        if "learning_path" not in results:
            self.error("Learning path missing")
            all_valid = False
        else:
            path = results["learning_path"]
            if len(path) < 3:
                self.error("Learning path incomplete")
                all_valid = False

        if all_valid:
            self.success("All onboarding helpers validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = OnboardingHelpersDemo()
    success = demo.execute()
    exit(0 if success else 1)
