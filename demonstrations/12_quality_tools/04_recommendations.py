"""Smart Measurement Recommendations

Demonstrates intelligent analysis recommendations:
- Context-aware measurement suggestions
- Parameter optimization guidance
- Best practice recommendations
- Troubleshooting assistance

This demonstration shows:
1. How to provide smart measurement recommendations
2. How to suggest optimal analysis parameters
3. How to guide users toward best practices
4. How to assist with troubleshooting common issues
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
    add_noise,
    generate_sine_wave,
    generate_square_wave,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class RecommendationsDemo(BaseDemo):
    """Demonstrate smart measurement recommendations."""

    def __init__(self) -> None:
        """Initialize recommendations demonstration."""
        super().__init__(
            name="recommendations",
            description="Smart measurement recommendations and analysis guidance",
            capabilities=[
                "oscura.recommendations.analysis_suggestions",
                "oscura.recommendations.parameter_optimization",
                "oscura.recommendations.best_practices",
                "oscura.recommendations.troubleshooting",
            ],
            related_demos=[
                "12_quality_tools/02_quality_scoring.py",
                "13_guidance/01_smart_recommendations.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate various signal types for recommendations."""
        self.info("Creating diverse test signals...")

        # Sine wave - good for spectral analysis
        sine = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        self.info("  ✓ Sine wave (spectral analysis candidate)")

        # Square wave - good for timing analysis
        square = generate_square_wave(1000.0, 1.0, 100_000.0, 0.1)
        self.info("  ✓ Square wave (timing analysis candidate)")

        # Noisy signal - needs preprocessing
        noisy = add_noise(generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0), 0.2)
        self.info("  ✓ Noisy signal (preprocessing needed)")

        # Low sample rate - aliasing risk
        low_sr = generate_sine_wave(1000.0, 1.0, 5_000.0, 0.1)
        self.info("  ✓ Low sample rate signal (aliasing risk)")

        # Multi-tone signal
        multi = self._create_multitone_signal()
        self.info("  ✓ Multi-tone signal (complex spectrum)")

        # Transient signal
        transient = self._create_transient_signal()
        self.info("  ✓ Transient signal (time-domain analysis)")

        return {
            "sine": sine,
            "square": square,
            "noisy": noisy,
            "low_sample_rate": low_sr,
            "multitone": multi,
            "transient": transient,
        }

    def _create_multitone_signal(self) -> WaveformTrace:
        """Create multi-tone signal."""
        sample_rate = 100_000.0
        duration = 0.1
        t = np.arange(int(sample_rate * duration)) / sample_rate
        data = (
            np.sin(2 * np.pi * 1000 * t)
            + 0.5 * np.sin(2 * np.pi * 2500 * t)
            + 0.3 * np.sin(2 * np.pi * 4000 * t)
        )
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_transient_signal(self) -> WaveformTrace:
        """Create transient signal."""
        sample_rate = 100_000.0
        duration = 0.1
        t = np.arange(int(sample_rate * duration)) / sample_rate
        # Damped sinusoid
        data = np.exp(-20 * t) * np.sin(2 * np.pi * 1000 * t)
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate recommendation system."""
        results: dict[str, Any] = {}

        # Part 1: Signal characterization
        self.section("Part 1: Automatic Signal Characterization")

        all_characteristics = {}
        for name, signal in data.items():
            chars = self._characterize_signal(signal)
            all_characteristics[name] = chars

            self.subsection(f"Signal: {name}")
            self.info(f"Signal type:      {chars['signal_type']}")
            self.info(f"Complexity:       {chars['complexity']}")
            self.info(f"Quality:          {chars['quality']}")
            self.info(f"Primary domain:   {chars['domain']}")

        results["characteristics"] = all_characteristics

        # Part 2: Analysis recommendations
        self.section("Part 2: Recommended Analysis Methods")

        all_recommendations = {}
        for name, signal in data.items():
            recs = self._get_analysis_recommendations(signal, all_characteristics[name])
            all_recommendations[name] = recs

            self.subsection(f"Signal: {name}")
            self.info("Recommended analyses:")
            for i, rec in enumerate(recs["primary"], 1):
                self.info(f"  {i}. {rec}")

            if recs["secondary"]:
                self.info("\nAdditional options:")
                for rec in recs["secondary"]:
                    self.info(f"  - {rec}")

        results["recommendations"] = all_recommendations

        # Part 3: Parameter optimization suggestions
        self.section("Part 3: Parameter Optimization Suggestions")

        signal = data["noisy"]
        params = self._suggest_parameters(signal)

        self.info("Optimal parameters for noisy signal:")
        self.info(f"  FFT window size:    {params['fft_size']}")
        self.info(f"  Window type:        {params['window_type']}")
        self.info(f"  Averaging count:    {params['averaging']}")
        self.info(f"  Filter cutoff:      {params['filter_cutoff_hz']:.0f} Hz")
        self.info(f"  Filter order:       {params['filter_order']}")

        results["parameter_suggestions"] = params

        # Part 4: Best practice guidance
        self.section("Part 4: Best Practice Recommendations")

        for name, signal in data.items():
            chars = all_characteristics[name]
            best_practices = self._get_best_practices(signal, chars)

            if best_practices:
                self.subsection(f"Signal: {name}")
                for practice in best_practices:
                    self.info(f"  • {practice}")

        # Part 5: Troubleshooting assistance
        self.section("Part 5: Troubleshooting Assistance")

        # Identify issues and suggest fixes
        for name, signal in data.items():
            issues = self._identify_issues(signal)

            if issues:
                self.subsection(f"Signal: {name}")
                for issue in issues:
                    self.warning(f"{issue['problem']}")
                    self.info(f"  Solution: {issue['solution']}")
                    if "steps" in issue:
                        self.info("  Steps:")
                        for step in issue["steps"]:
                            self.info(f"    {step}")

        results["troubleshooting"] = {
            name: self._identify_issues(signal) for name, signal in data.items()
        }

        # Part 6: Workflow recommendations
        self.section("Part 6: Complete Workflow Recommendations")

        for name in ["sine", "noisy", "transient"]:
            signal = data[name]
            workflow = self._recommend_workflow(signal, all_characteristics[name])

            self.subsection(f"Signal: {name}")
            self.info("Recommended workflow:")
            for i, step in enumerate(workflow, 1):
                self.info(f"  {i}. {step}")

        return results

    def _characterize_signal(self, signal: WaveformTrace) -> dict[str, str]:
        """Automatically characterize signal."""
        data = signal.data

        # Determine signal type from spectrum
        fft = np.fft.rfft(data)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(data), 1 / signal.metadata.sample_rate)

        # Find peaks
        peaks = []
        for i in range(1, len(magnitude) - 1):
            if magnitude[i] > magnitude[i - 1] and magnitude[i] > magnitude[i + 1]:
                if magnitude[i] > 0.1 * np.max(magnitude):
                    peaks.append(i)

        # Classify signal type
        if len(peaks) == 1:
            signal_type = "Single tone"
        elif len(peaks) > 1:
            # Check if peaks are harmonically related
            if len(peaks) > 2:
                fundamental = freqs[peaks[0]]
                is_harmonic = all(
                    abs(freqs[p] - n * fundamental) < 50 for n, p in enumerate(peaks[:3], 1)
                )
                if is_harmonic:
                    signal_type = "Harmonic (square/pulse)"
                else:
                    signal_type = "Multi-tone"
            else:
                signal_type = "Multi-tone"
        else:
            signal_type = "Broadband/noise"

        # Determine complexity
        spectral_entropy = -np.sum(magnitude * np.log(magnitude + 1e-10)) / np.log(len(magnitude))
        if spectral_entropy < 0.3:
            complexity = "Simple"
        elif spectral_entropy < 0.6:
            complexity = "Moderate"
        else:
            complexity = "Complex"

        # Estimate quality
        if len(peaks) > 0:
            signal_power = magnitude[peaks[0]] ** 2
            noise_power = np.mean(magnitude**2)
            snr_db = 10 * np.log10(signal_power / noise_power)
            if snr_db > 40:
                quality = "Excellent"
            elif snr_db > 20:
                quality = "Good"
            else:
                quality = "Poor"
        else:
            quality = "Poor"

        # Primary domain
        _time_variation = np.std(np.abs(np.diff(data)))  # For domain classification
        freq_concentration = np.max(magnitude) / np.sum(magnitude)
        if freq_concentration > 0.3:
            domain = "Frequency"
        else:
            domain = "Time"

        return {
            "signal_type": signal_type,
            "complexity": complexity,
            "quality": quality,
            "domain": domain,
        }

    def _get_analysis_recommendations(
        self, signal: WaveformTrace, characteristics: dict[str, str]
    ) -> dict[str, list[str]]:
        """Get recommended analysis methods."""
        primary = []
        secondary = []

        signal_type = characteristics["signal_type"]
        domain = characteristics["domain"]

        if domain == "Frequency":
            primary.append("FFT / Spectral analysis")
            primary.append("Power spectral density")
            if "tone" in signal_type.lower():
                primary.append("Harmonic analysis / THD")
            secondary.append("Spectrogram for time-varying content")

        if domain == "Time" or "transient" in signal_type.lower():
            primary.append("Waveform measurements (peak, RMS, etc.)")
            primary.append("Rise/fall time analysis")
            secondary.append("Envelope detection")

        if "harmonic" in signal_type.lower() or "square" in signal_type.lower():
            primary.append("Pulse parameter measurement")
            primary.append("Duty cycle analysis")

        if characteristics["quality"] == "Poor":
            primary.insert(0, "Noise reduction / filtering")
            secondary.append("Ensemble averaging")

        if characteristics["complexity"] == "Complex":
            secondary.append("Wavelet analysis")
            secondary.append("Time-frequency analysis")

        return {"primary": primary, "secondary": secondary}

    def _suggest_parameters(self, signal: WaveformTrace) -> dict[str, Any]:
        """Suggest optimal analysis parameters."""
        # Estimate noise level
        fft = np.fft.rfft(signal.data)
        magnitude = np.abs(fft)
        noise_floor = np.median(magnitude)
        signal_peak = np.max(magnitude)
        snr_estimate = signal_peak / (noise_floor + 1e-10)

        # FFT size based on signal length and desired resolution
        fft_size = 2 ** int(np.ceil(np.log2(len(signal.data))))

        # Window type based on SNR
        if snr_estimate > 100:
            window_type = "Rectangular (high SNR)"
        elif snr_estimate > 10:
            window_type = "Hann (balanced)"
        else:
            window_type = "Blackman-Harris (low SNR)"

        # Averaging based on SNR
        if snr_estimate > 50:
            averaging = 1
        elif snr_estimate > 10:
            averaging = 4
        else:
            averaging = 16

        # Filter parameters
        # Find fundamental frequency
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        peak_idx = np.argmax(magnitude[1:]) + 1
        fundamental = freqs[peak_idx]

        filter_cutoff = fundamental * 5  # 5x fundamental
        filter_order = 4 if snr_estimate > 10 else 6

        return {
            "fft_size": fft_size,
            "window_type": window_type,
            "averaging": averaging,
            "filter_cutoff_hz": filter_cutoff,
            "filter_order": filter_order,
        }

    def _get_best_practices(
        self, signal: WaveformTrace, characteristics: dict[str, str]
    ) -> list[str]:
        """Get best practice recommendations."""
        practices = []

        if characteristics["quality"] == "Poor":
            practices.append("Consider increasing signal amplitude or reducing noise at source")
            practices.append("Use AC coupling to remove DC offset")

        if signal.metadata.sample_rate < 50_000:
            practices.append("Verify sample rate is >10x highest frequency component (Nyquist)")

        practices.append("Apply appropriate window function for FFT analysis")
        practices.append("Use calibration signals to validate measurement accuracy")

        if characteristics["complexity"] == "Complex":
            practices.append("Consider segmented analysis for long signals")

        return practices

    def _identify_issues(self, signal: WaveformTrace) -> list[dict[str, Any]]:
        """Identify signal issues and suggest solutions."""
        issues = []

        # Check sample rate vs signal content
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude = np.abs(fft)

        max_freq_idx = np.where(magnitude > 0.1 * np.max(magnitude))[0]
        if len(max_freq_idx) > 0:
            max_signal_freq = freqs[max_freq_idx[-1]]
            if signal.metadata.sample_rate < 5 * max_signal_freq:
                issues.append(
                    {
                        "problem": "Sample rate too low (aliasing risk)",
                        "solution": f"Increase sample rate to >{10 * max_signal_freq:.0f} Hz",
                        "steps": [
                            "Check signal bandwidth",
                            "Adjust ADC sample rate",
                            "Apply anti-aliasing filter if needed",
                        ],
                    }
                )

        # Check for DC offset
        dc_offset = np.mean(signal.data)
        if abs(dc_offset) > 0.1:
            issues.append(
                {
                    "problem": f"Significant DC offset ({dc_offset:.3f})",
                    "solution": "Enable AC coupling or remove DC computationally",
                    "steps": [
                        "Subtract mean from signal",
                        "Or enable AC coupling on acquisition hardware",
                    ],
                }
            )

        # Check for clipping
        if np.max(signal.data) > 0.99 or np.min(signal.data) < -0.99:
            issues.append(
                {
                    "problem": "Signal clipping detected",
                    "solution": "Reduce input amplitude",
                    "steps": [
                        "Decrease signal source amplitude",
                        "Adjust attenuator/gain settings",
                        "Increase ADC input range if possible",
                    ],
                }
            )

        return issues

    def _recommend_workflow(
        self, signal: WaveformTrace, characteristics: dict[str, str]
    ) -> list[str]:
        """Recommend complete analysis workflow."""
        workflow = []

        workflow.append("Load and inspect signal")
        workflow.append("Check signal quality (SNR, clipping, noise)")

        if characteristics["quality"] == "Poor":
            workflow.append("Apply preprocessing (filtering, DC removal)")

        if characteristics["domain"] == "Frequency":
            workflow.append("Perform FFT with appropriate window")
            workflow.append("Analyze frequency spectrum")
            if "tone" in characteristics["signal_type"].lower():
                workflow.append("Measure THD and harmonics")
        else:
            workflow.append("Measure time-domain parameters")
            workflow.append("Analyze transient characteristics")

        workflow.append("Validate results against expected values")
        workflow.append("Generate report with measurements and plots")

        return workflow

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating recommendations...")
        all_valid = True

        # Check characterization exists
        if "characteristics" not in results:
            self.error("Missing signal characterization")
            return False

        # Check recommendations exist
        if "recommendations" not in results:
            self.error("Missing recommendations")
            return False

        # Validate that each signal has recommendations
        chars = results["characteristics"]
        recs = results["recommendations"]

        for signal_name in chars:
            if signal_name not in recs:
                self.error(f"Missing recommendations for {signal_name}")
                all_valid = False
            else:
                if not recs[signal_name]["primary"]:
                    self.warning(f"No primary recommendations for {signal_name}")

        if all_valid:
            self.success("All recommendations validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = RecommendationsDemo()
    success = demo.execute()
    exit(0 if success else 1)
