"""Fuzzy Signal Pattern Matching

Demonstrates fuzzy matching and pattern discovery:
- Fuzzy protocol matching
- Similar pattern discovery
- Approximate signal comparison
- Template matching with tolerance

This demonstration shows:
1. How to find similar patterns across signals
2. How to match protocols with variations
3. How to handle noisy pattern matching
4. How to discover recurring patterns
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


class FuzzyMatchingDemo(BaseDemo):
    """Demonstrate fuzzy pattern matching and discovery."""

    def __init__(self) -> None:
        """Initialize fuzzy matching demonstration."""
        super().__init__(
            name="fuzzy_matching",
            description="Fuzzy pattern matching and similar signal discovery",
            capabilities=[
                "oscura.exploratory.fuzzy_matching",
                "oscura.exploratory.pattern_discovery",
                "oscura.exploratory.similarity_search",
                "oscura.exploratory.template_matching",
            ],
            related_demos=[
                "14_exploratory/01_unknown_signals.py",
                "14_exploratory/03_signal_recovery.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for fuzzy matching."""
        self.info("Creating test signals with patterns...")

        # Template pattern
        template = self._create_template_pattern()
        self.info("  ✓ Template pattern")

        # Exact match
        exact = self._create_template_pattern()
        self.info("  ✓ Exact match")

        # Noisy match
        noisy = self._add_pattern_noise(self._create_template_pattern())
        self.info("  ✓ Noisy match")

        # Scaled match
        scaled = self._scale_pattern(self._create_template_pattern(), 1.2)
        self.info("  ✓ Scaled match (20%)")

        # Shifted match
        shifted = self._shift_pattern(self._create_template_pattern(), 0.1)
        self.info("  ✓ Time-shifted match")

        # Different pattern
        different = self._create_different_pattern()
        self.info("  ✓ Different pattern")

        # Signal with embedded patterns
        embedded = self._create_signal_with_patterns()
        self.info("  ✓ Signal with multiple embedded patterns")

        return {
            "template": template,
            "exact": exact,
            "noisy": noisy,
            "scaled": scaled,
            "shifted": shifted,
            "different": different,
            "embedded": embedded,
        }

    def _create_template_pattern(self) -> WaveformTrace:
        """Create template pattern (rising edge followed by decay)."""
        sample_rate = 100_000.0
        duration = 0.01  # 10ms
        t = np.arange(int(sample_rate * duration)) / sample_rate

        # Rising edge + exponential decay
        rise_time = 0.001  # 1ms
        decay_time = 0.005  # 5ms

        data = np.zeros_like(t)
        rise_samples = int(rise_time * sample_rate)
        data[:rise_samples] = np.linspace(0, 1, rise_samples)
        data[rise_samples:] = np.exp(-(t[rise_samples:] - rise_time) / decay_time)

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _add_pattern_noise(self, signal: WaveformTrace, noise_level: float = 0.1) -> WaveformTrace:
        """Add noise to pattern."""
        noisy_data = signal.data + noise_level * np.random.randn(len(signal.data))
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=noisy_data, metadata=metadata)

    def _scale_pattern(self, signal: WaveformTrace, scale: float) -> WaveformTrace:
        """Scale pattern amplitude."""
        scaled_data = signal.data * scale
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=scaled_data, metadata=metadata)

    def _shift_pattern(self, signal: WaveformTrace, shift_fraction: float) -> WaveformTrace:
        """Time-shift pattern."""
        shift_samples = int(len(signal.data) * shift_fraction)
        shifted_data = np.roll(signal.data, shift_samples)
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=shifted_data, metadata=metadata)

    def _create_different_pattern(self) -> WaveformTrace:
        """Create completely different pattern."""
        sample_rate = 100_000.0
        duration = 0.01
        return generate_sine_wave(1000.0, 1.0, sample_rate, duration)

    def _create_signal_with_patterns(self) -> WaveformTrace:
        """Create signal with multiple embedded patterns."""
        sample_rate = 100_000.0
        duration = 0.1
        n_samples = int(sample_rate * duration)

        # Base noise
        data = 0.1 * np.random.randn(n_samples)

        # Embed template pattern multiple times
        template = self._create_template_pattern()
        pattern_length = len(template.data)

        # Embed at different locations
        positions = [1000, 3000, 7000]
        for pos in positions:
            if pos + pattern_length < n_samples:
                # Add some variation
                noise_level = 0.05 * np.random.rand()
                noisy_pattern = template.data + noise_level * np.random.randn(pattern_length)
                data[pos : pos + pattern_length] += noisy_pattern

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate fuzzy matching."""
        results: dict[str, Any] = {}

        # Part 1: Exact vs fuzzy matching
        self.section("Part 1: Similarity Scoring")

        template = data["template"]
        test_signals = {
            "exact": data["exact"],
            "noisy": data["noisy"],
            "scaled": data["scaled"],
            "shifted": data["shifted"],
            "different": data["different"],
        }

        similarity_scores = {}
        for name, signal in test_signals.items():
            score = self._calculate_similarity(template, signal)
            similarity_scores[name] = score

            self.info(f"{name:15s}: {score:.1%} similarity")

        results["similarity_scores"] = similarity_scores

        # Part 2: Fuzzy matching with tolerance
        self.section("Part 2: Fuzzy Matching with Tolerance")

        self.info("Testing match detection at different tolerance levels:\n")

        tolerances = [0.95, 0.85, 0.70, 0.50]
        for tolerance in tolerances:
            self.subsection(f"Tolerance: {tolerance:.0%}")

            for name, score in similarity_scores.items():
                match = score >= tolerance
                symbol = "✓" if match else "✗"
                self.info(f"  {symbol} {name:15s}: {score:.1%}")

        # Part 3: Pattern discovery in embedded signals
        self.section("Part 3: Pattern Discovery in Long Signals")

        embedded_signal = data["embedded"]
        matches = self._find_pattern_occurrences(template, embedded_signal)

        self.info(f"Found {len(matches)} pattern occurrences in signal:")
        for i, match in enumerate(matches, 1):
            time_ms = match["position"] / embedded_signal.metadata.sample_rate * 1000
            self.info(f"  {i}. Position: {match['position']:6d} samples ({time_ms:6.2f} ms)")
            self.info(f"     Similarity: {match['similarity']:.1%}")

        results["pattern_matches"] = matches

        # Part 4: Approximate pattern matching
        self.section("Part 4: DTW-Based Approximate Matching")

        self.info("Testing Dynamic Time Warping for time-shifted patterns:\n")

        # Compare shifted pattern using DTW
        dtw_distance = self._dtw_distance(template.data, data["shifted"].data)
        euclidean_distance = np.sqrt(np.sum((template.data - data["shifted"].data) ** 2))

        self.info("Template vs Shifted pattern:")
        self.info(f"  Euclidean distance: {euclidean_distance:.4f}")
        self.info(f"  DTW distance:       {dtw_distance:.4f}")
        self.info(
            f"  DTW advantage:      {((euclidean_distance - dtw_distance) / euclidean_distance * 100):.1f}%"
        )

        results["dtw_comparison"] = {
            "euclidean": euclidean_distance,
            "dtw": dtw_distance,
        }

        # Part 5: Multi-scale pattern matching
        self.section("Part 5: Scale-Invariant Matching")

        scales = [0.8, 1.0, 1.2, 1.5]
        scale_results = []

        self.info("Testing pattern matching across different scales:\n")

        for scale in scales:
            scaled_signal = self._scale_pattern(template, scale)
            # Normalize for fair comparison
            norm_template = template.data / np.max(np.abs(template.data))
            norm_scaled = scaled_signal.data / np.max(np.abs(scaled_signal.data))

            correlation = np.corrcoef(norm_template, norm_scaled)[0, 1]
            scale_results.append({"scale": scale, "correlation": correlation})

            self.info(f"  Scale {scale:4.1f}x: correlation = {correlation:.3f}")

        results["scale_invariance"] = scale_results

        return results

    def _calculate_similarity(self, template: WaveformTrace, signal: WaveformTrace) -> float:
        """Calculate similarity score (0-1) between two signals."""
        # Ensure same length
        if len(template.data) != len(signal.data):
            return 0.0

        # Normalize both signals
        template_norm = template.data / (np.max(np.abs(template.data)) + 1e-10)
        signal_norm = signal.data / (np.max(np.abs(signal.data)) + 1e-10)

        # Correlation coefficient
        correlation = np.corrcoef(template_norm, signal_norm)[0, 1]

        # Convert to 0-1 score (correlation is -1 to 1)
        similarity = (correlation + 1) / 2

        return similarity

    def _find_pattern_occurrences(
        self, template: WaveformTrace, signal: WaveformTrace, threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """Find all occurrences of pattern in signal."""
        template_data = template.data
        signal_data = signal.data

        # Normalize template
        template_norm = template_data / (np.max(np.abs(template_data)) + 1e-10)

        matches = []
        pattern_length = len(template_data)

        # Sliding window search
        for i in range(0, len(signal_data) - pattern_length, pattern_length // 4):
            window = signal_data[i : i + pattern_length]

            # Normalize window
            window_norm = window / (np.max(np.abs(window)) + 1e-10)

            # Calculate correlation
            correlation = np.corrcoef(template_norm, window_norm)[0, 1]
            similarity = (correlation + 1) / 2

            if similarity >= threshold:
                matches.append({"position": i, "similarity": similarity})

        # Remove overlapping matches (keep best)
        filtered_matches = []
        for match in matches:
            # Check if too close to existing match
            too_close = False
            for existing in filtered_matches:
                if abs(match["position"] - existing["position"]) < pattern_length:
                    if match["similarity"] > existing["similarity"]:
                        # Replace with better match
                        filtered_matches.remove(existing)
                    else:
                        too_close = True
                        break

            if not too_close:
                filtered_matches.append(match)

        return filtered_matches

    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Calculate Dynamic Time Warping distance (simplified)."""
        n, m = len(s1), len(s2)

        # Initialize DTW matrix
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m]

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating fuzzy matching...")
        all_valid = True

        # Check similarity scores exist
        if "similarity_scores" not in results:
            self.error("Missing similarity scores")
            return False

        scores = results["similarity_scores"]

        # Validate exact match has highest score
        if "exact" in scores and scores["exact"] < 0.95:
            self.error(f"Exact match score too low: {scores['exact']:.1%}")
            all_valid = False

        # Validate different pattern has low score
        if "different" in scores and scores["different"] > 0.6:
            self.warning(f"Different pattern score too high: {scores['different']:.1%}")

        # Check pattern matches found
        if "pattern_matches" not in results:
            self.error("Missing pattern matches")
            all_valid = False
        elif len(results["pattern_matches"]) < 2:
            self.warning("Expected more pattern matches in embedded signal")

        if all_valid:
            self.success("All fuzzy matching validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = FuzzyMatchingDemo()
    success = demo.execute()
    exit(0 if success else 1)
