"""Interactive analysis wizard for guided signal analysis.

This module provides step-by-step guided analysis wizards that help
non-expert users analyze their signals with intelligent recommendations.

  - Step-by-step guidance
  - Intelligent defaults
  - Context-aware recommendations
  - Result interpretation

Example:
    >>> from oscura.cli.onboarding import run_wizard
    >>> run_wizard(trace)
    Analysis Wizard
    Step 1: What type of signal is this?
    ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class WizardAction(Enum):
    """Actions the wizard can perform."""

    MEASURE = "measure"
    CHARACTERIZE = "characterize"
    DECODE = "decode"
    FILTER = "filter"
    SPECTRAL = "spectral"
    COMPARE = "compare"


@dataclass
class WizardStep:
    """A step in the analysis wizard.

    Attributes:
        title: Step title
        question: Question to ask user
        options: Available options
        action: Action to perform based on choice
        help_text: Additional help for this step
    """

    title: str
    question: str
    options: list[str]
    action: Callable[[int], None] | None = None
    help_text: str = ""
    skip_condition: Callable[[Any], bool] | None = None


@dataclass
class WizardResult:
    """Result from wizard analysis.

    Attributes:
        steps_completed: Number of steps completed
        measurements: Collected measurements
        recommendations: Analysis recommendations
        summary: Human-readable summary
    """

    steps_completed: int = 0
    measurements: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""


class AnalysisWizard:
    """Interactive analysis wizard.

    Guides users through signal analysis with intelligent
    recommendations and plain English explanations.
    """

    def __init__(self, trace: Any) -> None:
        """Initialize wizard with a trace.

        Args:
            trace: WaveformTrace or DigitalTrace to analyze
        """
        self.trace = trace
        self.result = WizardResult()
        self.steps: list[WizardStep] = self._build_steps()
        self.current_step = 0

    def _build_steps(self) -> list[WizardStep]:
        """Build the wizard steps based on trace characteristics."""
        steps = [
            WizardStep(
                title="Signal Type Detection",
                question="What type of analysis do you want to perform?",
                options=[
                    "Auto-detect (let Oscura figure it out)",
                    "Digital signal analysis",
                    "Analog/waveform analysis",
                    "Protocol decoding",
                    "Power analysis",
                ],
                action=self._handle_signal_type,
                help_text="Not sure? Choose 'Auto-detect' and we'll analyze your signal.",
            ),
            WizardStep(
                title="Basic Measurements",
                question="Would you like to run basic measurements?",
                options=[
                    "Yes, run all standard measurements",
                    "Yes, but only timing measurements",
                    "Yes, but only amplitude measurements",
                    "No, skip this step",
                ],
                action=self._handle_measurements,
                help_text="Basic measurements give you an overview of your signal.",
            ),
            WizardStep(
                title="Spectral Analysis",
                question="Would you like to analyze the frequency content?",
                options=[
                    "Yes, compute FFT spectrum",
                    "Yes, compute power spectral density",
                    "Yes, both FFT and PSD",
                    "No, skip spectral analysis",
                ],
                action=self._handle_spectral,
                help_text="Spectral analysis shows what frequencies are in your signal.",
            ),
            WizardStep(
                title="Signal Quality",
                question="Would you like to assess signal quality?",
                options=[
                    "Yes, measure THD and SNR",
                    "Yes, check for anomalies",
                    "Yes, full quality assessment",
                    "No, skip quality check",
                ],
                action=self._handle_quality,
                help_text="Quality metrics help identify issues with your signal.",
            ),
        ]
        return steps

    def run(self, interactive: bool = True) -> WizardResult:
        """Run the analysis wizard.

        Args:
            interactive: If True, prompt for user input

        Returns:
            WizardResult with all collected data
        """
        print("\n" + "=" * 60)
        print("Oscura Analysis Wizard")
        print("=" * 60)
        print("Let's analyze your signal step by step.\n")

        # Show trace summary
        self._show_trace_summary()

        for i, step in enumerate(self.steps):
            # Check skip condition
            if step.skip_condition and step.skip_condition(self.result):
                continue

            print(f"\n{'=' * 60}")
            print(f"Step {i + 1}/{len(self.steps)}: {step.title}")
            print("=" * 60)

            if step.help_text:
                print(f"Tip: {step.help_text}\n")

            print(step.question)
            for j, option in enumerate(step.options, 1):
                print(f"  {j}. {option}")

            if interactive:
                choice = self._get_user_choice(len(step.options))
            else:
                choice = 1  # Auto-select first option

            if step.action:
                step.action(choice)

            self.result.steps_completed = i + 1

        # Generate summary
        self._generate_summary()

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        print(self.result.summary)

        if self.result.recommendations:
            print("\nRecommendations:")
            for rec in self.result.recommendations:
                print(f"  - {rec}")

        return self.result

    def _show_trace_summary(self) -> None:
        """Show a summary of the loaded trace."""
        trace = self.trace
        print("Loaded trace summary:")

        if hasattr(trace, "data"):
            print(f"  Samples: {len(trace.data):,}")

        if hasattr(trace, "metadata"):
            meta = trace.metadata
            if hasattr(meta, "sample_rate") and meta.sample_rate:
                rate = meta.sample_rate
                if rate >= 1e9:
                    print(f"  Sample rate: {rate / 1e9:.3f} GSa/s")
                elif rate >= 1e6:
                    print(f"  Sample rate: {rate / 1e6:.3f} MSa/s")
                else:
                    print(f"  Sample rate: {rate / 1e3:.3f} kSa/s")

            if hasattr(meta, "channel_name") and meta.channel_name:
                print(f"  Channel: {meta.channel_name}")

        if hasattr(trace, "data"):
            import numpy as np

            data = trace.data
            print(f"  Value range: {np.min(data):.4g} to {np.max(data):.4g}")

    def _get_user_choice(self, max_options: int) -> int:
        """Get user's choice with validation."""
        while True:
            try:
                choice_str = input(f"\nEnter choice (1-{max_options}): ")
                choice = int(choice_str)
                if 1 <= choice <= max_options:
                    return choice
                print(f"Please enter a number between 1 and {max_options}")
            except ValueError:
                print("Please enter a valid number")

    def _handle_signal_type(self, choice: int) -> None:
        """Handle signal type selection."""
        if choice == 1:  # Auto-detect
            print("\nAuto-detecting signal type...")
            try:
                from oscura.discovery import characterize_signal

                result = characterize_signal(self.trace)
                self.result.measurements["signal_type"] = result.signal_type
                self.result.measurements["signal_confidence"] = result.confidence
                print(f"Detected: {result.signal_type} (confidence: {result.confidence:.0%})")

                if result.confidence < 0.8:
                    self.result.recommendations.append(
                        f"Signal type detection has low confidence. "
                        f"Consider alternatives: {[a.signal_type for a in result.alternatives[:2]]}"  # type: ignore[attr-defined]
                    )
            except Exception as e:
                print(f"Auto-detection failed: {e}")
                self.result.measurements["signal_type"] = "unknown"

        elif choice == 2:  # Digital
            self.result.measurements["signal_type"] = "digital"
            print("\nDigital analysis mode selected.")

        elif choice == 3:  # Analog
            self.result.measurements["signal_type"] = "analog"
            print("\nAnalog/waveform analysis mode selected.")

        elif choice == 4:  # Protocol
            self.result.measurements["signal_type"] = "protocol"
            print("\nProtocol decoding mode selected.")
            self.result.recommendations.append(
                "For protocol decoding, try: decode_uart(), decode_spi(), decode_i2c()"
            )

        elif choice == 5:  # Power
            self.result.measurements["signal_type"] = "power"
            print("\nPower analysis mode selected.")

    def _handle_measurements(self, choice: int) -> None:
        """Handle measurement selection."""
        import oscura as osc

        if choice == 4:  # Skip
            print("\nSkipping measurements.")
            return

        print("\nRunning measurements...")

        try:
            if choice == 1:  # All
                results = osc.measure(self.trace)
            elif choice == 2:  # Timing only
                results = {
                    "rise_time": osc.rise_time(self.trace),
                    "fall_time": osc.fall_time(self.trace),
                    "frequency": osc.frequency(self.trace),
                    "period": osc.period(self.trace),
                    "duty_cycle": osc.duty_cycle(self.trace),
                }
            elif choice == 3:  # Amplitude only
                results = {
                    "amplitude": osc.amplitude(self.trace),
                    "rms": osc.rms(self.trace),
                    "mean": osc.mean(self.trace),
                    "overshoot": osc.overshoot(self.trace),
                    "undershoot": osc.undershoot(self.trace),
                }

            self.result.measurements.update(results)

            print("\nMeasurement results:")
            for name, value in results.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.4g}")
                else:
                    print(f"  {name}: {value}")

        except Exception as e:
            print(f"Measurement error: {e}")

    def _handle_spectral(self, choice: int) -> None:
        """Handle spectral analysis selection."""
        import numpy as np

        import oscura as osc

        if choice == 4:  # Skip
            print("\nSkipping spectral analysis.")
            return

        print("\nRunning spectral analysis...")

        try:
            if choice in (1, 3):  # FFT
                freq, mag = osc.fft(self.trace)  # type: ignore[misc]
                peak_idx = np.argmax(mag)
                self.result.measurements["fft_peak_freq"] = freq[peak_idx]
                self.result.measurements["fft_peak_mag"] = mag[peak_idx]
                print(f"  FFT peak: {freq[peak_idx] / 1e6:.3f} MHz at {mag[peak_idx]:.1f} dB")

            if choice in (2, 3):  # PSD
                freq, _psd_vals = osc.psd(self.trace)
                self.result.measurements["psd_computed"] = True
                print(f"  PSD computed over {len(freq)} frequency bins")

        except Exception as e:
            print(f"Spectral analysis error: {e}")

    def _handle_quality(self, choice: int) -> None:
        """Handle quality assessment selection."""
        import oscura as osc

        if choice == 4:  # Skip
            print("\nSkipping quality assessment.")
            return

        print("\nAssessing signal quality...")

        try:
            if choice in (1, 3):  # THD and SNR
                thd_val = osc.thd(self.trace)
                snr_val = osc.snr(self.trace)
                self.result.measurements["thd"] = thd_val
                self.result.measurements["snr"] = snr_val
                print(f"  THD: {thd_val:.1f} dB")
                print(f"  SNR: {snr_val:.1f} dB")

                # Recommendations based on quality
                if thd_val > -40:
                    self.result.recommendations.append(
                        f"THD is {thd_val:.1f} dB - consider filtering to reduce distortion"
                    )
                if snr_val < 40:
                    self.result.recommendations.append(
                        f"SNR is {snr_val:.1f} dB - signal is noisy, try averaging or filtering"
                    )

            if choice in (2, 3):  # Anomalies
                from oscura.discovery import find_anomalies

                anomalies = find_anomalies(self.trace)
                self.result.measurements["anomaly_count"] = len(anomalies)
                print(f"  Found {len(anomalies)} anomalies")

                if anomalies:
                    self.result.recommendations.append(
                        f"Found {len(anomalies)} anomalies - review the anomaly list for issues"
                    )

        except Exception as e:
            print(f"Quality assessment error: {e}")

    def _generate_summary(self) -> None:
        """Generate a human-readable summary of the analysis."""
        lines = ["Analysis Summary:"]

        self._add_signal_type_summary(lines)
        self._add_frequency_summary(lines)
        self._add_rise_time_summary(lines)
        self._add_thd_summary(lines)
        self._add_snr_summary(lines)

        self.result.summary = "\n".join(lines)

    def _add_signal_type_summary(self, lines: list[str]) -> None:
        """Add signal type to summary."""
        if "signal_type" in self.result.measurements:
            lines.append(f"  Signal type: {self.result.measurements['signal_type']}")

    def _add_frequency_summary(self, lines: list[str]) -> None:
        """Add frequency to summary with appropriate units."""
        if "frequency" not in self.result.measurements:
            return

        freq = self._extract_value(self.result.measurements["frequency"])
        if freq >= 1e6:
            lines.append(f"  Frequency: {freq / 1e6:.3f} MHz")
        elif freq >= 1e3:
            lines.append(f"  Frequency: {freq / 1e3:.3f} kHz")
        else:
            lines.append(f"  Frequency: {freq:.1f} Hz")

    def _add_rise_time_summary(self, lines: list[str]) -> None:
        """Add rise time to summary."""
        if "rise_time" not in self.result.measurements:
            return

        rt = self._extract_value(self.result.measurements["rise_time"])
        lines.append(f"  Rise time: {rt * 1e9:.2f} ns")

    def _add_thd_summary(self, lines: list[str]) -> None:
        """Add THD to summary."""
        if "thd" not in self.result.measurements:
            return

        thd = self._extract_value(self.result.measurements["thd"])
        lines.append(f"  THD: {thd:.1f} dB")

    def _add_snr_summary(self, lines: list[str]) -> None:
        """Add SNR to summary."""
        if "snr" not in self.result.measurements:
            return

        snr = self._extract_value(self.result.measurements["snr"])
        lines.append(f"  SNR: {snr:.1f} dB")

    @staticmethod
    def _extract_value(measurement: Any) -> float:
        """Extract numeric value from measurement (handles dict format)."""
        if isinstance(measurement, dict):
            value: float = float(measurement.get("value", 0))
            return value
        result: float = float(measurement)
        return result


def run_wizard(trace: Any, interactive: bool = True) -> WizardResult:
    """Run the analysis wizard on a trace.

    This is the main entry point for guided analysis.

    Args:
        trace: WaveformTrace or DigitalTrace to analyze
        interactive: If True, prompt for user input

    Returns:
        WizardResult with measurements, recommendations, and summary

    Example:
        >>> import oscura as osc
        >>> from oscura.cli.onboarding import run_wizard
        >>> trace = osc.load("signal.csv")
        >>> result = run_wizard(trace)
    """
    wizard = AnalysisWizard(trace)
    return wizard.run(interactive=interactive)
