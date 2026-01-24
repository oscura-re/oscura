"""Exploratory Signal Analysis

Demonstrates interactive exploration and hypothesis generation:
- Interactive exploration workflows
- Automatic hypothesis generation
- Automated insight discovery
- Iterative analysis refinement

This demonstration shows:
1. How to perform exploratory analysis on unknown signals
2. How to generate and test hypotheses automatically
3. How to discover insights from data
4. How to iteratively refine analysis
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
)
from oscura.core.types import TraceMetadata, WaveformTrace


class ExploratoryAnalysisDemo(BaseDemo):
    """Demonstrate exploratory analysis and insight discovery."""

    def __init__(self) -> None:
        """Initialize exploratory analysis demonstration."""
        super().__init__(
            name="exploratory_analysis",
            description="Interactive exploration and automated insight discovery",
            capabilities=[
                "oscura.exploratory.interactive_exploration",
                "oscura.exploratory.hypothesis_generation",
                "oscura.exploratory.insight_discovery",
                "oscura.exploratory.iterative_refinement",
            ],
            related_demos=[
                "14_exploratory/01_unknown_signals.py",
                "13_guidance/01_smart_recommendations.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate mystery signals for exploration."""
        self.info("Creating mystery signals for exploratory analysis...")

        # Mystery signal 1: Multi-component
        mystery1 = self._create_mystery_signal_1()
        self.info("  ✓ Mystery signal #1")

        # Mystery signal 2: Periodic with drift
        mystery2 = self._create_mystery_signal_2()
        self.info("  ✓ Mystery signal #2")

        # Mystery signal 3: Burst pattern
        mystery3 = self._create_mystery_signal_3()
        self.info("  ✓ Mystery signal #3")

        return {
            "mystery1": mystery1,
            "mystery2": mystery2,
            "mystery3": mystery3,
        }

    def _create_mystery_signal_1(self) -> WaveformTrace:
        """Create multi-component signal."""
        sample_rate = 100_000.0
        duration = 0.2
        t = np.arange(int(sample_rate * duration)) / sample_rate

        # Combine multiple components
        component1 = np.sin(2 * np.pi * 1000 * t)  # 1kHz
        component2 = 0.5 * np.sin(2 * np.pi * 2500 * t)  # 2.5kHz
        component3 = 0.3 * np.sin(2 * np.pi * 5000 * t)  # 5kHz

        # Add some noise
        noise = 0.1 * np.random.randn(len(t))

        data = component1 + component2 + component3 + noise
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_mystery_signal_2(self) -> WaveformTrace:
        """Create signal with frequency drift."""
        sample_rate = 100_000.0
        duration = 0.2
        t = np.arange(int(sample_rate * duration)) / sample_rate

        # Frequency chirp (1kHz to 2kHz)
        freq_start = 1000
        freq_end = 2000
        freq = freq_start + (freq_end - freq_start) * t / duration
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate

        data = np.sin(phase)
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_mystery_signal_3(self) -> WaveformTrace:
        """Create burst pattern signal."""
        sample_rate = 100_000.0
        duration = 0.1
        n_samples = int(sample_rate * duration)

        data = 0.05 * np.random.randn(n_samples)

        # Add periodic bursts
        burst_period = 2000  # samples
        burst_length = 200  # samples

        for i in range(0, n_samples, burst_period):
            if i + burst_length < n_samples:
                t = np.arange(burst_length) / sample_rate
                burst = np.sin(2 * np.pi * 3000 * t)
                # Apply envelope
                envelope = np.exp(-t / 0.01)
                data[i : i + burst_length] += burst * envelope

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate exploratory analysis."""
        results: dict[str, Any] = {}

        # Part 1: Initial exploration
        self.section("Part 1: Initial Signal Exploration")

        for name, signal in data.items():
            self.subsection(f"Exploring: {name}")

            # Quick characterization
            observations = self._initial_observations(signal)

            self.info("Initial observations:")
            for obs in observations:
                self.info(f"  • {obs}")

        # Part 2: Hypothesis generation
        self.section("Part 2: Automatic Hypothesis Generation")

        all_hypotheses = {}
        for name, signal in data.items():
            hypotheses = self._generate_hypotheses(signal)
            all_hypotheses[name] = hypotheses

            self.subsection(f"{name}")
            self.info("Generated hypotheses:")
            for i, hyp in enumerate(hypotheses, 1):
                self.info(f"  {i}. {hyp['hypothesis']}")
                self.info(f"     Confidence: {hyp['confidence']:.0%}")
                self.info(f"     Test: {hyp['test']}")

        results["hypotheses"] = all_hypotheses

        # Part 3: Hypothesis testing
        self.section("Part 3: Hypothesis Testing")

        test_results = {}
        for name, signal in data.items():
            hypotheses = all_hypotheses[name]

            self.subsection(f"{name}")

            signal_tests = []
            for hyp in hypotheses:
                result = self._test_hypothesis(signal, hyp)
                signal_tests.append(result)

                status = "CONFIRMED" if result["confirmed"] else "REJECTED"
                self.info(f"  {status}: {hyp['hypothesis']}")
                self.info(f"     Evidence: {result['evidence']}")

            test_results[name] = signal_tests

        results["test_results"] = test_results

        # Part 4: Insight discovery
        self.section("Part 4: Automated Insight Discovery")

        for name, signal in data.items():
            insights = self._discover_insights(signal, test_results[name])

            if insights:
                self.subsection(f"{name}")
                self.info("Discovered insights:")
                for insight in insights:
                    self.info(f"  💡 {insight}")

        # Part 5: Iterative refinement
        self.section("Part 5: Iterative Analysis Refinement")

        signal_name = "mystery1"
        signal = data[signal_name]

        self.info(f"Performing iterative analysis on {signal_name}:\n")

        iteration_results = []

        # Iteration 1: Broad analysis
        self.info("Iteration 1: Broad frequency analysis")
        fft1 = np.fft.rfft(signal.data)
        freqs1 = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        peaks1 = self._find_spectral_peaks(fft1, freqs1)
        self.info(f"  Found {len(peaks1)} frequency components")
        iteration_results.append({"iteration": 1, "peaks": len(peaks1)})

        # Iteration 2: Focus on dominant component
        dominant_freq = peaks1[0][0] if peaks1 else 1000
        self.info(f"\nIteration 2: Analyzing around {dominant_freq:.0f} Hz")
        focused = self._bandpass_filter(signal, dominant_freq - 100, dominant_freq + 100)
        amplitude = np.max(np.abs(focused.data))
        self.info(f"  Component amplitude: {amplitude:.3f}")
        iteration_results.append({"iteration": 2, "amplitude": amplitude})

        # Iteration 3: Check for modulation
        self.info("\nIteration 3: Checking for amplitude modulation")
        envelope = np.abs(focused.data)
        envelope_variation = np.std(envelope) / (np.mean(envelope) + 1e-10)
        is_modulated = envelope_variation > 0.1
        self.info(f"  Modulation detected: {is_modulated}")
        self.info(f"  Envelope variation: {envelope_variation:.3f}")

        results["iterative_refinement"] = iteration_results

        # Part 6: Analysis summary and recommendations
        self.section("Part 6: Analysis Summary")

        for name in data:
            self.subsection(f"{name}")

            summary = self._generate_analysis_summary(
                signal, all_hypotheses[name], test_results[name]
            )

            self.info(f"Signal type:         {summary['type']}")
            self.info(f"Complexity:          {summary['complexity']}")
            self.info(f"Key finding:         {summary['key_finding']}")
            self.info("\nNext steps:")
            for step in summary["next_steps"]:
                self.info(f"  • {step}")

        return results

    def _initial_observations(self, signal: WaveformTrace) -> list[str]:
        """Make initial observations about signal."""
        observations = []
        data = signal.data

        # Basic statistics
        peak = np.max(np.abs(data))
        observations.append(f"Peak amplitude: {peak:.3f}")

        rms = np.sqrt(np.mean(data**2))
        observations.append(f"RMS amplitude: {rms:.3f}")

        # Frequency content
        fft = np.fft.rfft(data)
        _freqs = np.fft.rfftfreq(len(data), 1 / signal.metadata.sample_rate)
        magnitude = np.abs(fft)

        n_significant = np.sum(magnitude > 0.1 * np.max(magnitude))
        observations.append(f"{n_significant} significant frequency components")

        # Time domain behavior
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        crossing_rate = zero_crossings / len(data)
        observations.append(f"Zero crossing rate: {crossing_rate:.4f}")

        # Stationarity
        mid = len(data) // 2
        std1 = np.std(data[:mid])
        std2 = np.std(data[mid:])
        is_stationary = abs(std1 - std2) / std1 < 0.2
        observations.append(f"Stationarity: {'Stationary' if is_stationary else 'Non-stationary'}")

        return observations

    def _generate_hypotheses(self, signal: WaveformTrace) -> list[dict[str, Any]]:
        """Generate testable hypotheses about signal."""
        hypotheses = []

        # Analyze spectrum
        fft = np.fft.rfft(signal.data)
        _freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude = np.abs(fft)

        # Hypothesis 1: Multi-tone signal
        n_peaks = np.sum(magnitude > 0.3 * np.max(magnitude))
        if n_peaks > 1:
            hypotheses.append(
                {
                    "hypothesis": "Signal contains multiple frequency components",
                    "confidence": 0.8,
                    "test": "Count spectral peaks above threshold",
                }
            )

        # Hypothesis 2: Periodic signal
        autocorr = np.correlate(signal.data, signal.data, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / autocorr[0]

        # Check for periodic peaks in autocorrelation
        has_periodic_peaks = np.any(autocorr[100:] > 0.5)
        if has_periodic_peaks:
            hypotheses.append(
                {
                    "hypothesis": "Signal is periodic",
                    "confidence": 0.7,
                    "test": "Check autocorrelation for periodic peaks",
                }
            )

        # Hypothesis 3: Non-stationary
        mid = len(signal.data) // 2
        mean1 = np.mean(signal.data[:mid])
        mean2 = np.mean(signal.data[mid:])
        std1 = np.std(signal.data[:mid])
        std2 = np.std(signal.data[mid:])

        is_nonstationary = abs(mean1 - mean2) > 0.1 * max(abs(mean1), abs(mean2)) or abs(
            std1 - std2
        ) > 0.3 * max(std1, std2)

        if is_nonstationary:
            hypotheses.append(
                {
                    "hypothesis": "Signal characteristics change over time",
                    "confidence": 0.6,
                    "test": "Compare statistics of first and second half",
                }
            )

        # Hypothesis 4: Contains bursts
        envelope = np.abs(signal.data)
        envelope_smooth = np.convolve(envelope, np.ones(100) / 100, mode="same")
        has_bursts = np.max(envelope_smooth) > 2 * np.median(envelope_smooth)

        if has_bursts:
            hypotheses.append(
                {
                    "hypothesis": "Signal contains periodic bursts",
                    "confidence": 0.5,
                    "test": "Check envelope for large variations",
                }
            )

        return hypotheses

    def _test_hypothesis(self, signal: WaveformTrace, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """Test a hypothesis about the signal."""
        hyp_text = hypothesis["hypothesis"]

        if "multiple frequency" in hyp_text.lower():
            # Count spectral peaks
            fft = np.fft.rfft(signal.data)
            magnitude = np.abs(fft)
            n_peaks = np.sum(magnitude > 0.3 * np.max(magnitude))

            confirmed = n_peaks >= 2
            evidence = f"Found {n_peaks} spectral peaks"

        elif "periodic" in hyp_text.lower():
            # Check autocorrelation
            autocorr = np.correlate(signal.data, signal.data, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            max_secondary = np.max(autocorr[100:]) if len(autocorr) > 100 else 0
            confirmed = max_secondary > 0.5
            evidence = f"Autocorrelation peak: {max_secondary:.2f}"

        elif "change over time" in hyp_text.lower():
            # Statistical test
            mid = len(signal.data) // 2
            std1 = np.std(signal.data[:mid])
            std2 = np.std(signal.data[mid:])

            variation = abs(std1 - std2) / (max(std1, std2) + 1e-10)
            confirmed = variation > 0.2
            evidence = f"Standard deviation variation: {variation:.1%}"

        elif "bursts" in hyp_text.lower():
            # Check envelope
            envelope = np.abs(signal.data)
            envelope_smooth = np.convolve(envelope, np.ones(100) / 100, mode="same")
            peak_ratio = np.max(envelope_smooth) / (np.median(envelope_smooth) + 1e-10)

            confirmed = peak_ratio > 2
            evidence = f"Envelope peak/median ratio: {peak_ratio:.1f}"

        else:
            confirmed = False
            evidence = "Test not implemented"

        return {
            "hypothesis": hyp_text,
            "confirmed": confirmed,
            "evidence": evidence,
        }

    def _discover_insights(
        self, signal: WaveformTrace, test_results: list[dict[str, Any]]
    ) -> list[str]:
        """Discover insights from test results."""
        insights = []

        # Check what was confirmed
        confirmed = [r for r in test_results if r["confirmed"]]

        if any("multiple frequency" in r["hypothesis"].lower() for r in confirmed):
            # Identify the frequencies
            fft = np.fft.rfft(signal.data)
            freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
            peaks = self._find_spectral_peaks(fft, freqs)

            freq_list = ", ".join([f"{f:.0f} Hz" for f, _ in peaks[:3]])
            insights.append(f"Signal composed of tones at: {freq_list}")

        if any("periodic" in r["hypothesis"].lower() for r in confirmed):
            # Calculate period
            autocorr = np.correlate(signal.data, signal.data, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            peak_idx = np.argmax(autocorr[100:]) + 100
            period_samples = peak_idx
            period_ms = period_samples / signal.metadata.sample_rate * 1000

            insights.append(f"Signal repeats every {period_ms:.1f} ms ({1000 / period_ms:.1f} Hz)")

        if any("change over time" in r["hypothesis"].lower() for r in confirmed):
            insights.append("Signal characteristics evolve - consider time-frequency analysis")

        if any("bursts" in r["hypothesis"].lower() for r in confirmed):
            insights.append(
                "Burst pattern detected - likely pulsed transmission or measurement artifact"
            )

        return insights

    def _find_spectral_peaks(self, fft: np.ndarray, freqs: np.ndarray) -> list[tuple[float, float]]:
        """Find spectral peaks."""
        magnitude = np.abs(fft)
        threshold = 0.2 * np.max(magnitude)

        peaks = []
        for i in range(1, len(magnitude) - 1):
            if (
                magnitude[i] > magnitude[i - 1]
                and magnitude[i] > magnitude[i + 1]
                and magnitude[i] > threshold
            ):
                peaks.append((freqs[i], magnitude[i]))

        # Sort by magnitude
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:5]  # Top 5

    def _bandpass_filter(self, signal: WaveformTrace, f_low: float, f_high: float) -> WaveformTrace:
        """Simple bandpass filter."""
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)

        # Zero out frequencies outside band
        mask = (freqs < f_low) | (freqs > f_high)
        fft[mask] = 0

        filtered = np.fft.irfft(fft, n=len(signal.data))
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=filtered, metadata=metadata)

    def _generate_analysis_summary(
        self,
        signal: WaveformTrace,
        hypotheses: list[dict[str, Any]],
        test_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate analysis summary."""
        confirmed = [r for r in test_results if r["confirmed"]]

        # Determine signal type
        if any("multiple frequency" in r["hypothesis"].lower() for r in confirmed):
            signal_type = "Multi-tone"
        elif any("bursts" in r["hypothesis"].lower() for r in confirmed):
            signal_type = "Burst/Pulsed"
        elif any("periodic" in r["hypothesis"].lower() for r in confirmed):
            signal_type = "Periodic"
        else:
            signal_type = "Broadband/Complex"

        # Determine complexity
        complexity = "High" if len(confirmed) >= 3 else "Medium" if len(confirmed) >= 1 else "Low"

        # Key finding
        if confirmed:
            key_finding = confirmed[0]["hypothesis"]
        else:
            key_finding = "No strong patterns detected"

        # Next steps
        next_steps = []
        if signal_type == "Multi-tone":
            next_steps.append("Perform harmonic analysis")
            next_steps.append("Measure individual tone amplitudes and phases")
        if any("change over time" in r["hypothesis"].lower() for r in confirmed):
            next_steps.append("Create spectrogram for time-frequency analysis")
        if signal_type == "Burst/Pulsed":
            next_steps.append("Analyze burst timing and patterns")

        if not next_steps:
            next_steps.append("Consider alternative analysis methods")

        return {
            "type": signal_type,
            "complexity": complexity,
            "key_finding": key_finding,
            "next_steps": next_steps,
        }

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating exploratory analysis...")
        all_valid = True

        # Check hypotheses generated
        if "hypotheses" not in results:
            self.error("Missing hypothesis generation")
            return False

        for signal_name, hyps in results["hypotheses"].items():
            if len(hyps) < 1:
                self.warning(f"No hypotheses generated for {signal_name}")

        # Check hypothesis testing
        if "test_results" not in results:
            self.error("Missing hypothesis test results")
            all_valid = False

        # Check iterative refinement
        if "iterative_refinement" in results:
            if len(results["iterative_refinement"]) < 2:
                self.warning("Expected more iteration steps")

        if all_valid:
            self.success("All exploratory analysis validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = ExploratoryAnalysisDemo()
    success = demo.execute()
    exit(0 if success else 1)
