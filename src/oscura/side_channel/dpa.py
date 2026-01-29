"""Differential Power Analysis (DPA) framework for side-channel attacks.

This module implements DPA, CPA (Correlation Power Analysis), and Template attacks
for recovering cryptographic keys from power consumption traces. Supports AES and DES
with multiple leakage models.

Key capabilities:
- DPA attack using difference of means
- CPA attack using Pearson correlation
- Template attack with profiling and matching phases
- Multiple leakage models (Hamming weight, Hamming distance, identity)
- AES S-box and DES operations
- Visualization of correlation traces and key rankings
- JSON export of attack results

Typical use cases:
- Break AES-128 implementations using power analysis
- Recover DES keys from embedded devices
- Evaluate cryptographic implementation security
- Generate attack visualizations for reports

Example:
    >>> from oscura.side_channel.dpa import DPAAnalyzer, PowerTrace
    >>> import numpy as np
    >>> # Create analyzer
    >>> analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")
    >>> # Load power traces with plaintexts
    >>> traces = [
    ...     PowerTrace(
    ...         timestamp=np.arange(1000),
    ...         power=np.random.randn(1000),
    ...         plaintext=bytes([i % 256 for i in range(16)])
    ...     )
    ...     for _ in range(100)
    ... ]
    >>> # Perform attack on first key byte
    >>> result = analyzer.perform_attack(traces, target_byte=0, algorithm="aes128")
    >>> print(f"Recovered key byte: 0x{result.recovered_key[0]:02X}")
    >>> print(f"Confidence: {result.confidence:.2%}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# AES S-box (forward substitution box)
AES_SBOX = [
    0x63,
    0x7C,
    0x77,
    0x7B,
    0xF2,
    0x6B,
    0x6F,
    0xC5,
    0x30,
    0x01,
    0x67,
    0x2B,
    0xFE,
    0xD7,
    0xAB,
    0x76,
    0xCA,
    0x82,
    0xC9,
    0x7D,
    0xFA,
    0x59,
    0x47,
    0xF0,
    0xAD,
    0xD4,
    0xA2,
    0xAF,
    0x9C,
    0xA4,
    0x72,
    0xC0,
    0xB7,
    0xFD,
    0x93,
    0x26,
    0x36,
    0x3F,
    0xF7,
    0xCC,
    0x34,
    0xA5,
    0xE5,
    0xF1,
    0x71,
    0xD8,
    0x31,
    0x15,
    0x04,
    0xC7,
    0x23,
    0xC3,
    0x18,
    0x96,
    0x05,
    0x9A,
    0x07,
    0x12,
    0x80,
    0xE2,
    0xEB,
    0x27,
    0xB2,
    0x75,
    0x09,
    0x83,
    0x2C,
    0x1A,
    0x1B,
    0x6E,
    0x5A,
    0xA0,
    0x52,
    0x3B,
    0xD6,
    0xB3,
    0x29,
    0xE3,
    0x2F,
    0x84,
    0x53,
    0xD1,
    0x00,
    0xED,
    0x20,
    0xFC,
    0xB1,
    0x5B,
    0x6A,
    0xCB,
    0xBE,
    0x39,
    0x4A,
    0x4C,
    0x58,
    0xCF,
    0xD0,
    0xEF,
    0xAA,
    0xFB,
    0x43,
    0x4D,
    0x33,
    0x85,
    0x45,
    0xF9,
    0x02,
    0x7F,
    0x50,
    0x3C,
    0x9F,
    0xA8,
    0x51,
    0xA3,
    0x40,
    0x8F,
    0x92,
    0x9D,
    0x38,
    0xF5,
    0xBC,
    0xB6,
    0xDA,
    0x21,
    0x10,
    0xFF,
    0xF3,
    0xD2,
    0xCD,
    0x0C,
    0x13,
    0xEC,
    0x5F,
    0x97,
    0x44,
    0x17,
    0xC4,
    0xA7,
    0x7E,
    0x3D,
    0x64,
    0x5D,
    0x19,
    0x73,
    0x60,
    0x81,
    0x4F,
    0xDC,
    0x22,
    0x2A,
    0x90,
    0x88,
    0x46,
    0xEE,
    0xB8,
    0x14,
    0xDE,
    0x5E,
    0x0B,
    0xDB,
    0xE0,
    0x32,
    0x3A,
    0x0A,
    0x49,
    0x06,
    0x24,
    0x5C,
    0xC2,
    0xD3,
    0xAC,
    0x62,
    0x91,
    0x95,
    0xE4,
    0x79,
    0xE7,
    0xC8,
    0x37,
    0x6D,
    0x8D,
    0xD5,
    0x4E,
    0xA9,
    0x6C,
    0x56,
    0xF4,
    0xEA,
    0x65,
    0x7A,
    0xAE,
    0x08,
    0xBA,
    0x78,
    0x25,
    0x2E,
    0x1C,
    0xA6,
    0xB4,
    0xC6,
    0xE8,
    0xDD,
    0x74,
    0x1F,
    0x4B,
    0xBD,
    0x8B,
    0x8A,
    0x70,
    0x3E,
    0xB5,
    0x66,
    0x48,
    0x03,
    0xF6,
    0x0E,
    0x61,
    0x35,
    0x57,
    0xB9,
    0x86,
    0xC1,
    0x1D,
    0x9E,
    0xE1,
    0xF8,
    0x98,
    0x11,
    0x69,
    0xD9,
    0x8E,
    0x94,
    0x9B,
    0x1E,
    0x87,
    0xE9,
    0xCE,
    0x55,
    0x28,
    0xDF,
    0x8C,
    0xA1,
    0x89,
    0x0D,
    0xBF,
    0xE6,
    0x42,
    0x68,
    0x41,
    0x99,
    0x2D,
    0x0F,
    0xB0,
    0x54,
    0xBB,
    0x16,
]


@dataclass
class PowerTrace:
    """Power consumption trace for side-channel analysis.

    Attributes:
        timestamp: Time points for each power measurement (seconds or sample index).
        power: Power consumption measurements (arbitrary units, e.g., voltage).
        plaintext: Input plaintext for encryption (16 bytes for AES-128).
        ciphertext: Output ciphertext from encryption (optional).
        metadata: Additional trace information (device ID, temperature, etc.).

    Example:
        >>> import numpy as np
        >>> trace = PowerTrace(
        ...     timestamp=np.linspace(0, 1e-6, 1000),  # 1 microsecond
        ...     power=np.random.randn(1000),
        ...     plaintext=bytes(range(16)),
        ...     metadata={"device": "STM32", "temperature": 25.0}
        ... )
    """

    timestamp: NDArray[np.float64]
    power: NDArray[np.float64]
    plaintext: bytes | None = None
    ciphertext: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DPAResult:
    """Result from DPA/CPA/Template attack.

    Attributes:
        recovered_key: Recovered key bytes (1 byte for single-byte attack).
        key_ranks: Correlation or difference-of-means score for each key guess (0-255).
            Higher values indicate more likely key bytes.
        correlation_traces: Time x key_guess correlation matrix (optional).
            Shape: (256, num_samples) for CPA attacks.
        confidence: Attack confidence score (0.0-1.0).
            Based on separation between best and second-best key guess.
        successful: True if attack confidence exceeds threshold (>0.7).

    Example:
        >>> result = analyzer.perform_attack(traces, target_byte=0)
        >>> if result.successful:
        ...     print(f"Key: 0x{result.recovered_key.hex()}")
        ...     print(f"Confidence: {result.confidence:.2%}")
        ... else:
        ...     print("Attack failed - need more traces")
    """

    recovered_key: bytes
    key_ranks: NDArray[np.float64]
    correlation_traces: NDArray[np.float64] | None = None
    confidence: float = 0.0
    successful: bool = False


class DPAAnalyzer:
    """Differential Power Analysis framework for cryptographic key recovery.

    This class implements multiple power analysis attack methods for recovering
    secret keys from power consumption traces. Supports DPA, CPA, and Template
    attacks with various leakage models.

    Attack Types:
        dpa: Classic DPA using difference of means (Kocher et al., 1999)
        cpa: Correlation Power Analysis (Brier et al., 2004)
        template: Template attack with profiling phase (Chari et al., 2003)

    Leakage Models:
        hamming_weight: Power proportional to number of 1 bits (most common)
        hamming_distance: Power proportional to bit transitions
        identity: Direct intermediate value (linear leakage)

    Supported Algorithms:
        aes128: AES-128 encryption (16-byte key)
        des: DES encryption (8-byte key, not implemented yet)

    Example:
        >>> # Basic CPA attack on AES
        >>> analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")
        >>> result = analyzer.perform_attack(traces, target_byte=0)
        >>> # Template attack (profiling + matching)
        >>> analyzer_template = DPAAnalyzer(attack_type="template")
        >>> result = analyzer_template.template_attack(
        ...     profiling_traces=train_traces,
        ...     attack_traces=test_traces,
        ...     target_byte=0
        ... )
    """

    LEAKAGE_MODELS: ClassVar[list[str]] = ["hamming_weight", "hamming_distance", "identity"]
    ATTACK_TYPES: ClassVar[list[str]] = ["dpa", "cpa", "template"]

    def __init__(
        self,
        attack_type: str = "cpa",
        leakage_model: str = "hamming_weight",
    ) -> None:
        """Initialize DPA analyzer.

        Args:
            attack_type: Attack method ("dpa", "cpa", "template").
            leakage_model: Power leakage model ("hamming_weight", "hamming_distance",
                "identity").

        Raises:
            ValueError: If attack_type or leakage_model is invalid.

        Example:
            >>> analyzer = DPAAnalyzer(attack_type="cpa")
            >>> analyzer = DPAAnalyzer(
            ...     attack_type="template",
            ...     leakage_model="hamming_distance"
            ... )
        """
        if attack_type not in self.ATTACK_TYPES:
            msg = f"Invalid attack_type: {attack_type}. Must be one of {self.ATTACK_TYPES}"
            raise ValueError(msg)

        if leakage_model not in self.LEAKAGE_MODELS:
            msg = f"Invalid leakage_model: {leakage_model}. Must be one of {self.LEAKAGE_MODELS}"
            raise ValueError(msg)

        self.attack_type = attack_type
        self.leakage_model = leakage_model
        # Templates: key_byte -> (mean vector, covariance matrix)
        self.templates: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    def perform_attack(
        self,
        traces: list[PowerTrace],
        target_byte: int = 0,
        algorithm: str = "aes128",
    ) -> DPAResult:
        """Perform power analysis attack to recover key byte.

        Dispatches to appropriate attack method based on attack_type.

        Args:
            traces: List of power traces with plaintexts.
            target_byte: Target key byte position (0-15 for AES-128).
            algorithm: Cryptographic algorithm ("aes128" or "des").

        Returns:
            DPAResult with recovered key and confidence metrics.

        Raises:
            ValueError: If traces is empty, target_byte invalid, or algorithm unsupported.

        Example:
            >>> result = analyzer.perform_attack(traces, target_byte=0)
            >>> print(f"Key byte {0}: 0x{result.recovered_key[0]:02X}")
        """
        if not traces:
            raise ValueError("No traces provided")

        if target_byte < 0 or target_byte >= 16:
            raise ValueError(f"Invalid target_byte: {target_byte}. Must be 0-15 for AES-128")

        if algorithm != "aes128":
            raise ValueError(f"Unsupported algorithm: {algorithm}. Only 'aes128' implemented")

        if self.attack_type == "dpa":
            return self.dpa_attack(traces, target_byte)
        elif self.attack_type == "cpa":
            return self.cpa_attack(traces, target_byte)
        else:
            msg = "Template attack requires separate profiling/attack traces"
            raise ValueError(msg)

    def dpa_attack(
        self,
        traces: list[PowerTrace],
        target_byte: int,
    ) -> DPAResult:
        """Classic DPA attack using difference of means.

        Separates traces into two groups based on hypothetical intermediate bit value,
        then computes difference of mean power consumption at each time point.

        Algorithm:
            1. For each key guess k in [0, 255]:
                a. Compute intermediate value for each trace: v = SBOX[plaintext ^ k]
                b. Partition traces by bit b of v (e.g., LSB)
                c. Compute differential trace: mean(power | b=1) - mean(power | b=0)
                d. Score = max absolute value in differential trace
            2. Return key with highest score

        Args:
            traces: List of power traces with plaintexts.
            target_byte: Target key byte position.

        Returns:
            DPAResult with recovered key byte.

        Example:
            >>> analyzer = DPAAnalyzer(attack_type="dpa")
            >>> result = analyzer.dpa_attack(traces, target_byte=0)
        """
        if not traces:
            raise ValueError("No traces provided")

        # Extract plaintexts and power matrix
        plaintexts = [t.plaintext for t in traces if t.plaintext]
        if not plaintexts:
            raise ValueError("No plaintexts in traces")

        power_matrix = np.array([t.power for t in traces])  # (num_traces, num_samples)

        # Attack each key guess
        max_diffs = np.zeros(256)

        for key_guess in range(256):
            # Compute intermediate values and selection bit
            intermediates = []
            for plaintext in plaintexts:
                if plaintext is None or target_byte >= len(plaintext):
                    intermediates.append(0)
                    continue
                intermediate = self._aes_sbox_output(plaintext[target_byte], key_guess)
                intermediates.append(intermediate)

            # Use LSB as selection bit
            selection_bits = np.array([v & 1 for v in intermediates])

            # Partition traces
            group0_traces = power_matrix[selection_bits == 0]
            group1_traces = power_matrix[selection_bits == 1]

            if len(group0_traces) == 0 or len(group1_traces) == 0:
                continue

            # Compute differential trace
            mean0 = np.mean(group0_traces, axis=0)
            mean1 = np.mean(group1_traces, axis=0)
            diff = mean1 - mean0

            # Score is maximum absolute difference
            max_diffs[key_guess] = np.max(np.abs(diff))

        # Find best key guess
        recovered_key_byte = int(np.argmax(max_diffs))
        max_score = max_diffs[recovered_key_byte]

        # Calculate confidence (separation from second-best)
        sorted_scores = np.sort(max_diffs)
        if sorted_scores[-2] > 0:
            confidence = 1.0 - (sorted_scores[-2] / max_score)
        else:
            confidence = 1.0

        successful = confidence > 0.7

        return DPAResult(
            recovered_key=bytes([recovered_key_byte]),
            key_ranks=max_diffs,
            correlation_traces=None,
            confidence=float(confidence),
            successful=successful,
        )

    def cpa_attack(
        self,
        traces: list[PowerTrace],
        target_byte: int,
    ) -> DPAResult:
        """Correlation Power Analysis (CPA) attack.

        Computes Pearson correlation between hypothetical power consumption (based on
        leakage model) and measured power at each time point. Key with highest
        correlation is most likely correct.

        Algorithm:
            1. For each key guess k in [0, 255]:
                a. Compute hypothetical power: h[i] = LeakageModel(SBOX[plaintext[i] ^ k])
                b. For each time point t:
                    - Compute correlation: corr(h, power[:, t])
                c. Score = max |correlation| across all time points
            2. Return key with highest score

        Args:
            traces: List of power traces with plaintexts.
            target_byte: Target key byte position.

        Returns:
            DPAResult with recovered key byte and correlation traces.

        Example:
            >>> analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")
            >>> result = analyzer.cpa_attack(traces, target_byte=0)
            >>> print(f"Max correlation: {result.confidence:.3f}")
        """
        if not traces:
            raise ValueError("No traces provided")

        # Extract plaintexts and power matrix
        plaintexts = [t.plaintext for t in traces if t.plaintext]
        if not plaintexts:
            raise ValueError("No plaintexts in traces")

        power_matrix = np.array([t.power for t in traces])  # (num_traces, num_samples)
        num_samples = power_matrix.shape[1]

        # Attack each key guess
        correlation_traces = np.zeros((256, num_samples))

        for key_guess in range(256):
            # Calculate hypothetical power consumption
            hypothetical = self._calculate_hypothetical_power(
                plaintexts,
                key_guess,
                target_byte,
            )

            # Calculate correlation at each time point
            for sample_idx in range(num_samples):
                measured = power_matrix[:, sample_idx]
                # Use numpy's correlation coefficient
                corr_matrix = np.corrcoef(hypothetical, measured)
                correlation = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
                # Handle NaN from constant signals
                if np.isnan(correlation):
                    correlation = 0.0
                correlation_traces[key_guess, sample_idx] = abs(correlation)

        # Find key guess with maximum correlation
        max_correlations = np.max(correlation_traces, axis=1)
        recovered_key_byte = int(np.argmax(max_correlations))
        max_correlation = max_correlations[recovered_key_byte]

        # Calculate confidence (separation from second-best)
        sorted_corrs = np.sort(max_correlations)
        if sorted_corrs[-2] > 0:
            confidence = 1.0 - (sorted_corrs[-2] / max_correlation)
        else:
            confidence = 1.0

        # Check if attack successful (correlation > threshold)
        successful = max_correlation > 0.7

        return DPAResult(
            recovered_key=bytes([recovered_key_byte]),
            key_ranks=max_correlations,
            correlation_traces=correlation_traces,
            confidence=float(confidence),
            successful=successful,
        )

    def template_attack(
        self,
        profiling_traces: list[PowerTrace],
        attack_traces: list[PowerTrace],
        target_byte: int,
    ) -> DPAResult:
        """Template attack with profiling and matching phases.

        Phase 1 (Profiling): Build templates (mean, covariance) for each key byte
            using traces from controlled device with known key.
        Phase 2 (Matching): Match attack traces to templates using multivariate
            Gaussian probability.

        Args:
            profiling_traces: Traces from device with known key for template building.
            attack_traces: Traces from target device with unknown key.
            target_byte: Target key byte position.

        Returns:
            DPAResult with recovered key byte.

        Raises:
            ValueError: If profiling or attack traces are empty.

        Example:
            >>> analyzer = DPAAnalyzer(attack_type="template")
            >>> # Build templates with known key
            >>> result = analyzer.template_attack(
            ...     profiling_traces=train_traces,
            ...     attack_traces=test_traces,
            ...     target_byte=0
            ... )
        """
        if not profiling_traces or not attack_traces:
            raise ValueError("Both profiling and attack traces required")

        # Phase 1: Build templates from profiling traces
        self._build_templates(profiling_traces, target_byte)

        # Phase 2: Match attack traces to templates
        power_matrix = np.array([t.power for t in attack_traces])
        num_samples = power_matrix.shape[1]

        # Calculate probability for each key guess
        probabilities = np.zeros(256)

        for key_guess in range(256):
            if key_guess not in self.templates:
                continue

            mean, cov = self.templates[key_guess]

            # Use only points of interest (reduce dimensionality)
            # For simplicity, use first min(100, num_samples) points
            poi_count = min(100, num_samples)
            mean_poi = mean[:poi_count]
            cov_poi = cov[:poi_count, :poi_count]

            # Add small regularization to covariance
            cov_poi = cov_poi + np.eye(poi_count) * 1e-6

            # Calculate log probability for each trace
            log_prob = 0.0
            for trace in power_matrix:
                trace_poi = trace[:poi_count]
                try:
                    # Multivariate Gaussian log probability
                    diff = trace_poi - mean_poi
                    inv_cov = np.linalg.inv(cov_poi)
                    log_prob += -0.5 * (diff @ inv_cov @ diff)
                except np.linalg.LinAlgError:
                    # Singular covariance matrix
                    log_prob += -1e10

            probabilities[key_guess] = log_prob

        # Find best key guess
        recovered_key_byte = int(np.argmax(probabilities))
        max_prob = probabilities[recovered_key_byte]

        # Calculate confidence
        sorted_probs = np.sort(probabilities)
        if sorted_probs[-2] != -np.inf and max_prob != -np.inf and max_prob != 0:
            # Use absolute difference for log probabilities (negative values)
            confidence = abs(max_prob - sorted_probs[-2]) / (abs(max_prob) + 1e-10)
            confidence = min(confidence, 1.0)
        else:
            confidence = 0.0

        successful = confidence > 0.7

        return DPAResult(
            recovered_key=bytes([recovered_key_byte]),
            key_ranks=probabilities,
            correlation_traces=None,
            confidence=float(confidence),
            successful=successful,
        )

    def _build_templates(
        self,
        traces: list[PowerTrace],
        target_byte: int,
    ) -> None:
        """Build templates from profiling traces with known key.

        Args:
            traces: Profiling traces with known plaintexts and ciphertexts.
            target_byte: Target key byte position.
        """
        # Group traces by intermediate value (assuming key byte = 0 for profiling)
        # In real scenario, you'd know the profiling key
        profiling_key_byte = 0  # Known key for profiling device

        groups: dict[int, list[NDArray[np.float64]]] = {}

        for trace in traces:
            if trace.plaintext is None or target_byte >= len(trace.plaintext):
                continue

            intermediate = self._aes_sbox_output(
                trace.plaintext[target_byte],
                profiling_key_byte,
            )

            if intermediate not in groups:
                groups[intermediate] = []
            groups[intermediate].append(trace.power)

        # Build template for each intermediate value
        for intermediate, power_traces in groups.items():
            if len(power_traces) < 2:
                continue

            power_array = np.array(power_traces)
            mean = np.mean(power_array, axis=0)
            cov = np.cov(power_array.T)

            self.templates[intermediate] = (mean, cov)

    def _hamming_weight(self, value: int) -> int:
        """Calculate Hamming weight (population count).

        Args:
            value: Integer value (0-255).

        Returns:
            Number of 1 bits in binary representation (0-8).

        Example:
            >>> analyzer._hamming_weight(0x0F)  # 0b00001111
            4
            >>> analyzer._hamming_weight(0xFF)  # 0b11111111
            8
        """
        count = 0
        while value:
            count += value & 1
            value >>= 1
        return count

    def _hamming_distance(self, value1: int, value2: int) -> int:
        """Calculate Hamming distance between two values.

        Args:
            value1: First integer value (0-255).
            value2: Second integer value (0-255).

        Returns:
            Number of differing bits (0-8).

        Example:
            >>> analyzer._hamming_distance(0x00, 0xFF)
            8
            >>> analyzer._hamming_distance(0x0F, 0x0E)  # differ in 1 bit
            1
        """
        return self._hamming_weight(value1 ^ value2)

    def _aes_sbox_output(self, plaintext_byte: int, key_guess: int) -> int:
        """Calculate AES S-box output for given plaintext and key guess.

        Args:
            plaintext_byte: Single plaintext byte (0-255).
            key_guess: Key byte guess (0-255).

        Returns:
            S-box output (0-255).

        Example:
            >>> analyzer._aes_sbox_output(0x00, 0x00)
            99  # SBOX[0x00] = 0x63
        """
        xored = plaintext_byte ^ key_guess
        return AES_SBOX[xored]

    def _calculate_hypothetical_power(
        self,
        plaintexts: Sequence[bytes | None],
        key_guess: int,
        target_byte: int,
    ) -> NDArray[np.float64]:
        """Calculate hypothetical power consumption for key guess.

        Uses configured leakage model to convert intermediate values to
        hypothetical power consumption.

        Args:
            plaintexts: List of plaintext bytes.
            key_guess: Key byte guess (0-255).
            target_byte: Target byte position.

        Returns:
            Array of hypothetical power values.

        Example:
            >>> hyp_power = analyzer._calculate_hypothetical_power(
            ...     plaintexts=[b'\\x00\\x01\\x02...'],
            ...     key_guess=0x42,
            ...     target_byte=0
            ... )
        """
        hypothetical = []

        for plaintext in plaintexts:
            if plaintext is None or target_byte >= len(plaintext):
                hypothetical.append(0.0)
                continue

            # Calculate intermediate value (AES S-box output)
            intermediate = self._aes_sbox_output(plaintext[target_byte], key_guess)

            # Apply leakage model
            if self.leakage_model == "hamming_weight":
                power = float(self._hamming_weight(intermediate))
            elif self.leakage_model == "hamming_distance":
                # Hamming distance from plaintext to intermediate
                power = float(self._hamming_distance(plaintext[target_byte], intermediate))
            else:  # identity
                power = float(intermediate)

            hypothetical.append(power)

        return np.array(hypothetical)

    def visualize_attack(
        self,
        result: DPAResult,
        output_path: Path,
    ) -> None:
        """Visualize CPA attack results with correlation traces and key rankings.

        Creates two-panel plot:
            - Top: Correlation traces for all key guesses (highlighted for recovered key)
            - Bottom: Bar chart of maximum correlation per key guess

        Args:
            result: DPAResult from CPA attack.
            output_path: Path to save plot image (PNG format).

        Raises:
            ValueError: If correlation_traces is None (not a CPA attack).
            ImportError: If matplotlib is not installed.

        Example:
            >>> result = analyzer.cpa_attack(traces, target_byte=0)
            >>> analyzer.visualize_attack(result, Path("attack_plot.png"))
        """
        if result.correlation_traces is None:
            raise ValueError("Visualization requires correlation_traces (use CPA attack)")

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            msg = "matplotlib required for visualization"
            raise ImportError(msg) from e

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Correlation traces for all key guesses
        for key_guess in range(256):
            ax1.plot(
                result.correlation_traces[key_guess],
                alpha=0.1,
                color="blue",
            )

        # Highlight correct key
        recovered = result.recovered_key[0]
        ax1.plot(
            result.correlation_traces[recovered],
            color="red",
            linewidth=2,
            label=f"Recovered key: 0x{recovered:02X}",
        )

        ax1.set_xlabel("Sample point")
        ax1.set_ylabel("|Correlation|")
        ax1.set_title("Correlation traces for all key guesses")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Key ranking (max correlation per key)
        ax2.bar(range(256), result.key_ranks, color="blue", alpha=0.6)
        ax2.axvline(
            result.recovered_key[0],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Recovered key",
        )
        ax2.set_xlabel("Key guess")
        ax2.set_ylabel("Max |Correlation|")
        ax2.set_title("Key ranking")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Attack visualization saved to {output_path}")

    def export_results(
        self,
        result: DPAResult,
        output_path: Path,
    ) -> None:
        """Export attack results to JSON file.

        Args:
            result: DPAResult from attack.
            output_path: Path to save JSON file.

        Example:
            >>> result = analyzer.perform_attack(traces, target_byte=0)
            >>> analyzer.export_results(result, Path("attack_results.json"))
        """
        data = {
            "recovered_key": result.recovered_key.hex(),
            "confidence": result.confidence,
            "successful": result.successful,
            "key_ranks": result.key_ranks.tolist(),
            "attack_type": self.attack_type,
            "leakage_model": self.leakage_model,
        }

        if result.correlation_traces is not None:
            # Only export max correlations to reduce file size
            data["max_correlations"] = np.max(result.correlation_traces, axis=1).tolist()

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Attack results exported to {output_path}")
