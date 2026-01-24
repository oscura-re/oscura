"""Smart Analysis Recommendations

Demonstrates context-aware analysis guidance system:
- Automatic signal type detection
- Context-aware analysis suggestions
- Optimal workflow recommendations
- Parameter tuning guidance based on signal characteristics

This demonstration shows:
1. How to automatically detect signal characteristics
2. How to recommend appropriate analysis methods
3. How to suggest optimal analysis workflows
4. How to guide parameter selection
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
    generate_square_wave,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class SmartRecommendationsDemo(BaseDemo):
    """Demonstrate context-aware smart recommendations."""

    def __init__(self) -> None:
        """Initialize smart recommendations demonstration."""
        super().__init__(
            name="smart_recommendations",
            description="Context-aware analysis recommendations and workflow guidance",
            capabilities=[
                "oscura.guidance.signal_detection",
                "oscura.guidance.analysis_recommendations",
                "oscura.guidance.workflow_suggestions",
                "oscura.guidance.parameter_tuning",
            ],
            related_demos=[
                "12_quality_tools/04_recommendations.py",
                "13_guidance/02_analysis_wizards.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate various signal types."""
        self.info("Creating test signals representing different use cases...")

        # Power line signal (50/60 Hz)
        power_line = self._create_power_signal()
        self.info("  ✓ Power line signal (50 Hz)")

        # Audio signal (multiple frequencies)
        audio = self._create_audio_signal()
        self.info("  ✓ Audio signal (musical note)")

        # PWM signal
        pwm = self._create_pwm_signal()
        self.info("  ✓ PWM signal (motor control)")

        # RF burst
        rf_burst = self._create_rf_burst()
        self.info("  ✓ RF burst (communication)")

        # Sensor data (noisy DC with drift)
        sensor = self._create_sensor_data()
        self.info("  ✓ Sensor data (temperature)")

        return {
            "power_line": power_line,
            "audio": audio,
            "pwm": pwm,
            "rf_burst": rf_burst,
            "sensor": sensor,
        }

    def _create_power_signal(self) -> WaveformTrace:
        """Create power line signal."""
        sample_rate = 10_000.0
        duration = 0.2
        signal = generate_sine_wave(50.0, 1.0, sample_rate, duration)
        # Add 3rd harmonic
        t = np.arange(int(sample_rate * duration)) / sample_rate
        signal.data += 0.1 * np.sin(2 * np.pi * 150 * t)
        return signal

    def _create_audio_signal(self) -> WaveformTrace:
        """Create audio signal (A4 note = 440 Hz)."""
        sample_rate = 48_000.0
        duration = 0.1
        t = np.arange(int(sample_rate * duration)) / sample_rate
        # Fundamental + harmonics (sawtooth-like)
        data = np.sin(2 * np.pi * 440 * t)
        data += 0.5 * np.sin(2 * np.pi * 880 * t)
        data += 0.25 * np.sin(2 * np.pi * 1320 * t)
        # Envelope
        envelope = np.exp(-3 * t)
        data = data * envelope
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_pwm_signal(self) -> WaveformTrace:
        """Create PWM signal."""
        sample_rate = 100_000.0
        duration = 0.01  # 10ms
        freq = 1000.0  # 1kHz PWM
        duty_cycle = 0.3  # 30%
        signal = generate_square_wave(freq, 1.0, sample_rate, duration)
        # Adjust for duty cycle
        period_samples = int(sample_rate / freq)
        on_samples = int(period_samples * duty_cycle)
        for i in range(0, len(signal.data), period_samples):
            if i + period_samples < len(signal.data):
                signal.data[i : i + on_samples] = 1.0
                signal.data[i + on_samples : i + period_samples] = 0.0
        return signal

    def _create_rf_burst(self) -> WaveformTrace:
        """Create RF burst signal."""
        sample_rate = 1_000_000.0  # 1 MHz
        duration = 0.001  # 1ms
        t = np.arange(int(sample_rate * duration)) / sample_rate
        carrier = 100_000.0  # 100 kHz carrier
        # Burst envelope (Gaussian)
        burst_center = duration / 2
        burst_width = duration / 6
        envelope = np.exp(-((t - burst_center) ** 2) / (2 * burst_width**2))
        data = envelope * np.sin(2 * np.pi * carrier * t)
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_sensor_data(self) -> WaveformTrace:
        """Create sensor data with drift."""
        sample_rate = 10.0  # 10 Hz sampling
        duration = 10.0  # 10 seconds
        t = np.arange(int(sample_rate * duration)) / sample_rate
        # Slow drift + noise
        drift = 20 + 2 * t / duration  # Temperature drift
        noise = 0.1 * np.random.randn(len(t))
        data = drift + noise
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate smart recommendations."""
        results: dict[str, Any] = {}

        # Part 1: Signal characterization
        self.section("Part 1: Automatic Signal Characterization")

        all_profiles = {}
        for name, signal in data.items():
            profile = self._create_signal_profile(signal)
            all_profiles[name] = profile

            self.subsection(f"{name}")
            self.info(f"Application domain:  {profile['domain']}")
            self.info(f"Signal class:        {profile['class']}")
            self.info(f"Bandwidth:           {profile['bandwidth_hz']:.1f} Hz")
            self.info(f"Dynamic range:       {profile['dynamic_range_db']:.1f} dB")
            self.info(f"Stationarity:        {profile['stationarity']}")

        results["profiles"] = all_profiles

        # Part 2: Analysis method recommendations
        self.section("Part 2: Recommended Analysis Methods")

        for name, profile in all_profiles.items():
            recommendations = self._recommend_analysis_methods(profile)

            self.subsection(f"{name}")
            self.info("Recommended analyses:")
            for i, (method, reason) in enumerate(recommendations, 1):
                self.info(f"  {i}. {method}")
                self.info(f"     Reason: {reason}")

        # Part 3: Workflow guidance
        self.section("Part 3: Step-by-Step Workflow Guidance")

        signal_name = "audio"
        signal = data[signal_name]
        profile = all_profiles[signal_name]

        workflow = self._generate_workflow(profile)

        self.info(f"Recommended workflow for {signal_name}:\n")
        for i, step in enumerate(workflow, 1):
            self.info(f"Step {i}: {step['action']}")
            self.info(f"  Purpose: {step['purpose']}")
            if "parameters" in step:
                self.info(f"  Parameters: {step['parameters']}")
            self.info("")

        results["workflow"] = workflow

        # Part 4: Parameter recommendations
        self.section("Part 4: Optimal Parameter Selection")

        for name, signal in data.items():
            profile = all_profiles[name]
            params = self._recommend_parameters(signal, profile)

            self.subsection(f"{name}")
            for param_name, value in params.items():
                if isinstance(value, float):
                    self.info(f"  {param_name:25s}: {value:.1f}")
                else:
                    self.info(f"  {param_name:25s}: {value}")

        # Part 5: Context-aware tips
        self.section("Part 5: Context-Aware Tips and Warnings")

        for name, profile in all_profiles.items():
            tips = self._generate_contextual_tips(profile, data[name])

            if tips:
                self.subsection(f"{name}")
                for tip in tips:
                    self.info(f"  💡 {tip}")

        return results

    def _create_signal_profile(self, signal: WaveformTrace) -> dict[str, Any]:
        """Create comprehensive signal profile."""
        data = signal.data
        sample_rate = signal.metadata.sample_rate

        # Frequency domain analysis
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1 / sample_rate)
        magnitude = np.abs(fft)
        power = magnitude**2

        # Bandwidth (95% power)
        cumsum_power = np.cumsum(power)
        total_power = cumsum_power[-1]
        idx_95 = np.where(cumsum_power >= 0.95 * total_power)[0][0]
        bandwidth = freqs[idx_95]

        # Dynamic range
        peak_power = np.max(power[1:])  # Exclude DC
        noise_floor = np.median(power[len(power) // 2 :])
        dynamic_range_db = 10 * np.log10(peak_power / (noise_floor + 1e-10))

        # Stationarity check (compare first and second half)
        mid = len(data) // 2
        std1 = np.std(data[:mid])
        std2 = np.std(data[mid:])
        stationarity = "Stationary" if abs(std1 - std2) / std1 < 0.2 else "Non-stationary"

        # Classify signal
        domain = self._classify_domain(sample_rate, bandwidth)
        signal_class = self._classify_signal(data, freqs, magnitude)

        return {
            "domain": domain,
            "class": signal_class,
            "bandwidth_hz": bandwidth,
            "dynamic_range_db": dynamic_range_db,
            "stationarity": stationarity,
            "sample_rate": sample_rate,
        }

    def _classify_domain(self, sample_rate: float, bandwidth: float) -> str:
        """Classify application domain."""
        if sample_rate < 100:
            return "Sensor/Instrumentation"
        elif sample_rate < 100_000 and bandwidth < 20_000:
            return "Audio/Power"
        elif sample_rate < 100_000:
            return "Control/Automation"
        else:
            return "RF/Communications"

    def _classify_signal(self, data: np.ndarray, freqs: np.ndarray, magnitude: np.ndarray) -> str:
        """Classify signal type."""
        # Find dominant frequencies
        threshold = 0.1 * np.max(magnitude)
        peaks = np.where(magnitude > threshold)[0]

        if len(peaks) == 1:
            return "Single tone"
        elif len(peaks) > 1:
            # Check if harmonic
            if len(peaks) >= 2:
                fundamental = freqs[peaks[0]]
                is_harmonic = all(
                    abs(freqs[p] - n * fundamental) < fundamental * 0.1
                    for n, p in enumerate(peaks[:3], 1)
                )
                if is_harmonic:
                    return "Harmonic/Pulsed"
            return "Multi-tone"

        # Check time domain characteristics
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        if zero_crossings < len(data) * 0.1:
            return "DC/Slowly varying"

        return "Broadband"

    def _recommend_analysis_methods(self, profile: dict[str, Any]) -> list[tuple[str, str]]:
        """Recommend analysis methods with reasons."""
        recommendations = []

        signal_class = profile["class"]
        domain = profile["domain"]

        if "tone" in signal_class.lower():
            recommendations.append(
                (
                    "Spectral Analysis (FFT)",
                    "Signal has distinct frequency components",
                )
            )
            recommendations.append(
                ("Harmonic Distortion Analysis", "Useful for tone purity assessment")
            )

        if "Harmonic" in signal_class or "Pulsed" in signal_class:
            recommendations.append(
                ("Pulse Parameter Measurement", "Signal has periodic pulse structure")
            )

        if domain == "Audio/Power":
            recommendations.append(
                ("RMS and Peak Measurements", "Standard for audio/power signals")
            )

        if domain == "RF/Communications":
            recommendations.append(("Envelope Detection", "Important for modulated RF signals"))
            recommendations.append(("Burst Analysis", "Useful for pulsed RF signals"))

        if profile["stationarity"] == "Non-stationary":
            recommendations.append(
                (
                    "Time-Frequency Analysis",
                    "Signal characteristics vary over time",
                )
            )

        if domain == "Sensor/Instrumentation":
            recommendations.append(("Trend Analysis", "Important for slow-varying sensor data"))
            recommendations.append(("Statistical Analysis", "Useful for noise characterization"))

        return recommendations

    def _generate_workflow(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate step-by-step workflow."""
        workflow = []

        # Always start with inspection
        workflow.append(
            {
                "action": "Load and inspect signal",
                "purpose": "Verify data integrity and characteristics",
            }
        )

        # Quality check
        workflow.append(
            {
                "action": "Perform quality assessment",
                "purpose": "Check for clipping, noise, and anomalies",
            }
        )

        # Domain-specific preprocessing
        if profile["domain"] == "Sensor/Instrumentation":
            workflow.append(
                {
                    "action": "Remove DC offset and drift",
                    "purpose": "Isolate signal of interest from slow variations",
                }
            )

        # Main analysis
        if "tone" in profile["class"].lower():
            workflow.append(
                {
                    "action": "Compute FFT spectrum",
                    "purpose": "Identify frequency components",
                    "parameters": "Use appropriate window (Hann/Blackman)",
                }
            )

        if profile["stationarity"] == "Non-stationary":
            workflow.append(
                {
                    "action": "Compute spectrogram",
                    "purpose": "Visualize time-varying frequency content",
                }
            )

        # Measurements
        workflow.append(
            {
                "action": "Extract key measurements",
                "purpose": "Quantify signal parameters",
            }
        )

        # Validation
        workflow.append(
            {
                "action": "Validate results",
                "purpose": "Ensure measurements are within expected ranges",
            }
        )

        return workflow

    def _recommend_parameters(
        self, signal: WaveformTrace, profile: dict[str, Any]
    ) -> dict[str, Any]:
        """Recommend optimal parameters."""
        params = {}

        # FFT size
        params["FFT Size"] = 2 ** int(np.ceil(np.log2(len(signal.data))))

        # Window type
        if profile["dynamic_range_db"] > 60:
            params["Window Type"] = "Rectangular"
        elif profile["dynamic_range_db"] > 30:
            params["Window Type"] = "Hann"
        else:
            params["Window Type"] = "Blackman-Harris"

        # Resolution bandwidth
        rbw = signal.metadata.sample_rate / params["FFT Size"]
        params["Resolution BW (Hz)"] = rbw

        # Averaging
        if profile["stationarity"] == "Stationary":
            params["Averaging"] = "Recommended (10-100 averages)"
        else:
            params["Averaging"] = "Not recommended (non-stationary)"

        return params

    def _generate_contextual_tips(
        self, profile: dict[str, Any], signal: WaveformTrace
    ) -> list[str]:
        """Generate context-aware tips."""
        tips = []

        # Sample rate tips
        nyquist = signal.metadata.sample_rate / 2
        if profile["bandwidth_hz"] > nyquist * 0.8:
            tips.append(
                f"Signal bandwidth ({profile['bandwidth_hz']:.0f} Hz) is close to Nyquist limit - verify no aliasing"
            )

        # Dynamic range tips
        if profile["dynamic_range_db"] < 20:
            tips.append("Low dynamic range detected - consider improving SNR or signal amplitude")

        # Domain-specific tips
        if profile["domain"] == "Audio/Power":
            tips.append("For power measurements, use RMS values rather than peak values")

        if profile["domain"] == "RF/Communications":
            tips.append(
                "Consider using calibrated probes and 50Ω termination for accurate RF measurements"
            )

        if profile["class"] == "DC/Slowly varying":
            tips.append("For DC measurements, ensure sufficient averaging to reduce noise")

        return tips

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating smart recommendations...")
        all_valid = True

        # Check profiles exist
        if "profiles" not in results:
            self.error("Missing signal profiles")
            return False

        profiles = results["profiles"]

        # Validate each profile has required fields
        required_fields = [
            "domain",
            "class",
            "bandwidth_hz",
            "dynamic_range_db",
            "stationarity",
        ]

        for name, profile in profiles.items():
            for field in required_fields:
                if field not in profile:
                    self.error(f"Missing {field} in {name} profile")
                    all_valid = False

        # Validate workflow exists
        if "workflow" not in results:
            self.error("Missing workflow")
            return False

        if len(results["workflow"]) < 3:
            self.warning("Workflow seems too short")

        if all_valid:
            self.success("All smart recommendations validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = SmartRecommendationsDemo()
    success = demo.execute()
    exit(0 if success else 1)
