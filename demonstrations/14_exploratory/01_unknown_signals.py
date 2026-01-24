"""Unknown Signal Characterization

Demonstrates automatic characterization of unknown signals:
- Automatic signal type detection (analog/digital/mixed)
- Frequency and pattern detection
- Modulation detection
- Signal classification

This demonstration shows:
1. How to automatically characterize unknown signals
2. How to detect signal types and patterns
3. How to identify potential protocols
4. How to extract key signal features
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


class UnknownSignalsDemo(BaseDemo):
    """Demonstrate automatic unknown signal characterization."""

    def __init__(self) -> None:
        """Initialize unknown signals demonstration."""
        super().__init__(
            name="unknown_signals",
            description="Automatic characterization and classification of unknown signals",
            capabilities=[
                "oscura.exploratory.signal_detection",
                "oscura.exploratory.type_classification",
                "oscura.exploratory.pattern_recognition",
                "oscura.exploratory.feature_extraction",
            ],
            related_demos=[
                "14_exploratory/02_fuzzy_matching.py",
                "13_guidance/01_smart_recommendations.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate various unknown signal types."""
        self.info("Creating test signals with unknown characteristics...")

        # Analog signals
        sine = generate_sine_wave(1234.56, 0.8, 100_000.0, 0.05)
        self.info("  ✓ Unknown analog signal #1")

        # Digital signal
        digital = self._create_unknown_digital()
        self.info("  ✓ Unknown digital signal #2")

        # Modulated signal
        modulated = self._create_unknown_modulated()
        self.info("  ✓ Unknown modulated signal #3")

        # Pulse train
        pulses = self._create_unknown_pulses()
        self.info("  ✓ Unknown pulse train #4")

        # Noise with embedded signal
        embedded = self._create_embedded_signal()
        self.info("  ✓ Signal embedded in noise #5")

        return {
            "signal_1": sine,
            "signal_2": digital,
            "signal_3": modulated,
            "signal_4": pulses,
            "signal_5": embedded,
        }

    def _create_unknown_digital(self) -> WaveformTrace:
        """Create unknown digital signal."""
        sample_rate = 100_000.0
        duration = 0.02
        # Random digital pattern
        n_samples = int(sample_rate * duration)
        bit_rate = 2400  # baud
        samples_per_bit = int(sample_rate / bit_rate)

        data = np.zeros(n_samples)
        for i in range(0, n_samples, samples_per_bit):
            end = min(i + samples_per_bit, n_samples)
            data[i:end] = np.random.choice([0.0, 1.0])

        # Add some noise
        data = data + 0.02 * np.random.randn(n_samples)

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_unknown_modulated(self) -> WaveformTrace:
        """Create AM modulated signal."""
        sample_rate = 500_000.0
        duration = 0.01
        t = np.arange(int(sample_rate * duration)) / sample_rate

        carrier_freq = 50_000.0
        modulation_freq = 1_000.0

        # AM modulation
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulation = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * modulation_freq * t))
        data = carrier * modulation

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_unknown_pulses(self) -> WaveformTrace:
        """Create pulse train with varying intervals."""
        sample_rate = 100_000.0
        duration = 0.05
        n_samples = int(sample_rate * duration)

        data = np.zeros(n_samples)

        # Random pulse intervals (50-500 samples)
        pos = 0
        while pos < n_samples:
            pulse_width = 10
            if pos + pulse_width < n_samples:
                data[pos : pos + pulse_width] = 1.0

            interval = np.random.randint(50, 500)
            pos += interval

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_embedded_signal(self) -> WaveformTrace:
        """Create signal embedded in noise."""
        sample_rate = 100_000.0
        duration = 0.1

        # Weak sine wave in heavy noise
        signal = generate_sine_wave(5000.0, 0.3, sample_rate, duration)
        noise = 1.0 * np.random.randn(len(signal.data))

        signal.data = signal.data + noise
        return signal

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate unknown signal characterization."""
        results: dict[str, Any] = {}

        # Part 1: Basic signal type detection
        self.section("Part 1: Automatic Signal Type Detection")

        all_classifications = {}
        for name, signal in data.items():
            classification = self._classify_signal_type(signal)
            all_classifications[name] = classification

            self.subsection(f"{name}")
            self.info(f"Signal type:       {classification['type']}")
            self.info(f"Confidence:        {classification['confidence']:.1%}")
            self.info(f"Primary domain:    {classification['domain']}")
            self.info(f"Characteristics:   {', '.join(classification['characteristics'])}")

        results["classifications"] = all_classifications

        # Part 2: Frequency detection
        self.section("Part 2: Frequency and Pattern Detection")

        for name, signal in data.items():
            freq_analysis = self._detect_frequencies(signal)

            self.subsection(f"{name}")
            if freq_analysis["dominant_freq"] > 0:
                self.info(f"Dominant frequency: {freq_analysis['dominant_freq']:.2f} Hz")

            if freq_analysis["secondary_freqs"]:
                self.info("Secondary frequencies:")
                for freq, power in freq_analysis["secondary_freqs"]:
                    self.info(f"  {freq:.2f} Hz (rel. power: {power:.1%})")

            if freq_analysis["is_periodic"]:
                self.info(f"Pattern: Periodic (period={freq_analysis['period_ms']:.2f} ms)")
            else:
                self.info("Pattern: Aperiodic/Random")

        # Part 3: Modulation detection
        self.section("Part 3: Modulation Detection")

        for name, signal in data.items():
            modulation = self._detect_modulation(signal)

            if modulation["detected"]:
                self.subsection(f"{name}")
                self.info(f"Modulation type:    {modulation['type']}")
                self.info(f"Carrier frequency:  {modulation['carrier_freq']:.0f} Hz")
                if modulation["modulation_freq"] > 0:
                    self.info(f"Modulation rate:    {modulation['modulation_freq']:.0f} Hz")

        # Part 4: Signal feature extraction
        self.section("Part 4: Feature Extraction")

        all_features = {}
        for name, signal in data.items():
            features = self._extract_features(signal)
            all_features[name] = features

            self.subsection(f"{name}")
            self.info("Extracted features:")
            for feature_name, value in features.items():
                if isinstance(value, float):
                    self.info(f"  {feature_name:25s}: {value:.4f}")
                else:
                    self.info(f"  {feature_name:25s}: {value}")

        results["features"] = all_features

        # Part 5: Protocol hints
        self.section("Part 5: Potential Protocol Identification")

        for name, signal in data.items():
            hints = self._identify_potential_protocols(signal, all_classifications[name])

            if hints:
                self.subsection(f"{name}")
                self.info("Potential protocols:")
                for hint in hints:
                    self.info(
                        f"  • {hint['protocol']}: {hint['reason']} (confidence: {hint['confidence']:.0%})"
                    )

        return results

    def _classify_signal_type(self, signal: WaveformTrace) -> dict[str, Any]:
        """Classify signal as analog, digital, or mixed."""
        data = signal.data

        # Check for digital characteristics
        unique_values = len(np.unique(np.round(data, 2)))
        value_range = np.max(data) - np.min(data)

        # Digital signals typically have few distinct levels
        if unique_values <= 4 and value_range > 0.5:
            signal_type = "Digital"
            confidence = 0.9
            characteristics = ["Two-level" if unique_values <= 2 else "Multi-level"]
        else:
            # Check for modulation
            envelope = np.abs(signal.data)
            envelope_variation = np.std(envelope) / (np.mean(envelope) + 1e-10)

            if envelope_variation > 0.3:
                signal_type = "Modulated Analog"
                confidence = 0.7
                characteristics = ["Amplitude varying", "Carrier present"]
            else:
                signal_type = "Analog"
                confidence = 0.8
                characteristics = ["Continuous amplitude"]

        # Determine primary domain
        fft = np.fft.rfft(data)
        magnitude = np.abs(fft)
        spectral_concentration = np.max(magnitude) / np.sum(magnitude)

        if spectral_concentration > 0.1:
            domain = "Frequency"
        else:
            domain = "Time"

        return {
            "type": signal_type,
            "confidence": confidence,
            "domain": domain,
            "characteristics": characteristics,
        }

    def _detect_frequencies(self, signal: WaveformTrace) -> dict[str, Any]:
        """Detect dominant and secondary frequencies."""
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude = np.abs(fft)
        power = magnitude**2

        # Find dominant frequency (skip DC)
        dominant_idx = np.argmax(magnitude[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        dominant_power = power[dominant_idx]

        # Find secondary frequencies
        secondary = []
        threshold = dominant_power * 0.1  # 10% of dominant

        for i in range(1, len(magnitude)):
            if i != dominant_idx and power[i] > threshold:
                # Check if it's a peak
                if i > 0 and i < len(magnitude) - 1:
                    if magnitude[i] > magnitude[i - 1] and magnitude[i] > magnitude[i + 1]:
                        rel_power = power[i] / dominant_power
                        secondary.append((freqs[i], rel_power))

        # Sort by power
        secondary.sort(key=lambda x: x[1], reverse=True)
        secondary = secondary[:5]  # Top 5

        # Check periodicity using autocorrelation
        autocorr = np.correlate(
            signal.data - np.mean(signal.data), signal.data - np.mean(signal.data), mode="full"
        )
        autocorr = autocorr[len(autocorr) // 2 :]

        # Find first significant peak after lag 0
        peaks = []
        for i in range(10, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > 0.3 * autocorr[0]:  # >30% of max correlation
                    peaks.append(i)
                    break

        is_periodic = len(peaks) > 0
        period_ms = (peaks[0] / signal.metadata.sample_rate * 1000) if peaks else 0

        return {
            "dominant_freq": dominant_freq,
            "secondary_freqs": secondary,
            "is_periodic": is_periodic,
            "period_ms": period_ms,
        }

    def _detect_modulation(self, signal: WaveformTrace) -> dict[str, Any]:
        """Detect if signal is modulated."""
        data = signal.data

        # Check envelope variation
        analytic = np.fft.ifft(np.concatenate([np.fft.rfft(data), np.zeros(len(data) // 2)]))
        envelope = np.abs(analytic)

        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        variation_coeff = envelope_std / (envelope_mean + 1e-10)

        # If envelope varies significantly, likely modulated
        if variation_coeff > 0.2:
            # Find carrier frequency
            fft = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1 / signal.metadata.sample_rate)
            carrier_idx = np.argmax(np.abs(fft)[1:]) + 1
            carrier_freq = freqs[carrier_idx]

            # Find modulation frequency from envelope
            envelope_fft = np.fft.rfft(envelope - np.mean(envelope))
            envelope_freqs = np.fft.rfftfreq(len(envelope), 1 / signal.metadata.sample_rate)
            mod_idx = np.argmax(np.abs(envelope_fft)[1:]) + 1
            mod_freq = envelope_freqs[mod_idx]

            return {
                "detected": True,
                "type": "AM (Amplitude Modulation)",
                "carrier_freq": carrier_freq,
                "modulation_freq": mod_freq,
            }

        return {
            "detected": False,
            "type": "None",
            "carrier_freq": 0,
            "modulation_freq": 0,
        }

    def _extract_features(self, signal: WaveformTrace) -> dict[str, Any]:
        """Extract comprehensive signal features."""
        data = signal.data

        features = {}

        # Time domain features
        features["mean"] = np.mean(data)
        features["std"] = np.std(data)
        features["peak"] = np.max(np.abs(data))
        features["rms"] = np.sqrt(np.mean(data**2))
        features["crest_factor"] = features["peak"] / (features["rms"] + 1e-10)
        features["zero_crossings"] = np.sum(np.diff(np.sign(data)) != 0)

        # Frequency domain features
        fft = np.fft.rfft(data)
        magnitude = np.abs(fft)
        _power = magnitude**2  # For power spectral density if needed

        features["spectral_centroid"] = np.sum(np.arange(len(magnitude)) * magnitude) / (
            np.sum(magnitude) + 1e-10
        )
        features["spectral_spread"] = np.sqrt(
            np.sum(((np.arange(len(magnitude)) - features["spectral_centroid"]) ** 2) * magnitude)
            / (np.sum(magnitude) + 1e-10)
        )
        features["spectral_flatness"] = np.exp(np.mean(np.log(magnitude + 1e-10))) / (
            np.mean(magnitude) + 1e-10
        )

        # Complexity features
        features["sample_entropy"] = self._sample_entropy(data)

        return features

    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy (complexity measure)."""
        N = len(data)
        r = r * np.std(data)

        def _phi(m: int) -> float:
            patterns = np.array([data[i : i + m] for i in range(N - m)])
            count = 0
            for i in range(len(patterns)):
                dist = np.max(np.abs(patterns - patterns[i]), axis=1)
                count += np.sum(dist < r) - 1  # Exclude self-match
            return count / (N - m)

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        if phi_m == 0 or phi_m1 == 0:
            return 0.0

        return -np.log(phi_m1 / phi_m)

    def _identify_potential_protocols(
        self, signal: WaveformTrace, classification: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify potential protocols based on characteristics."""
        hints = []

        signal_type = classification["type"]

        # Digital signals
        if "Digital" in signal_type:
            # Check bit rate
            transitions = np.sum(np.abs(np.diff(signal.data)) > 0.3)
            duration = len(signal.data) / signal.metadata.sample_rate
            bit_rate = transitions / (2 * duration)  # Approximate

            if 9600 * 0.9 < bit_rate < 9600 * 1.1:
                hints.append(
                    {
                        "protocol": "UART (9600 baud)",
                        "reason": f"Bit rate ~{bit_rate:.0f} bps",
                        "confidence": 0.6,
                    }
                )
            elif 115200 * 0.9 < bit_rate < 115200 * 1.1:
                hints.append(
                    {
                        "protocol": "UART (115200 baud)",
                        "reason": f"Bit rate ~{bit_rate:.0f} bps",
                        "confidence": 0.6,
                    }
                )

            # Check for clock-like pattern
            fft = np.fft.rfft(signal.data)
            _freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
            magnitude = np.abs(fft)
            peaks = np.where(magnitude > 0.5 * np.max(magnitude))[0]

            if len(peaks) >= 2:
                hints.append(
                    {
                        "protocol": "SPI/I2C (clocked)",
                        "reason": "Multiple spectral peaks suggest clock",
                        "confidence": 0.5,
                    }
                )

        # Modulated signals
        if "Modulated" in signal_type:
            hints.append(
                {
                    "protocol": "RF Communication (AM/FM)",
                    "reason": "Amplitude modulation detected",
                    "confidence": 0.7,
                }
            )

        return hints

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating unknown signal characterization...")
        all_valid = True

        # Check classifications exist
        if "classifications" not in results:
            self.error("Missing signal classifications")
            return False

        classifications = results["classifications"]

        # Validate each signal was classified
        for name, classification in classifications.items():
            if "type" not in classification:
                self.error(f"Missing type for {name}")
                all_valid = False

            if "confidence" not in classification:
                self.error(f"Missing confidence for {name}")
                all_valid = False

        # Check features extracted
        if "features" not in results:
            self.error("Missing feature extraction")
            all_valid = False

        if all_valid:
            self.success("All unknown signal characterization validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = UnknownSignalsDemo()
    success = demo.execute()
    exit(0 if success else 1)
