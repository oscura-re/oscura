"""Power analysis side-channel attacks.

This module implements Differential Power Analysis (DPA) and Correlation Power
Analysis (CPA) for extracting cryptographic keys from power consumption traces.

Example:
    >>> import numpy as np
    >>> from oscura.analyzers.side_channel.power import CPAAnalyzer
    >>>
    >>> # Perform CPA attack on AES
    >>> traces = np.array([...])  # Power traces (n_traces, n_samples)
    >>> plaintexts = np.array([...])  # Known plaintexts
    >>> cpa = CPAAnalyzer(leakage_model="hamming_weight", algorithm="aes_sbox")
    >>> result = cpa.analyze(traces, plaintexts)
    >>> print(f"Best key guess: 0x{result.key_guess:02X}")

References:
    Kocher et al. "Differential Power Analysis" (CRYPTO 1999)
    Brier et al. "Correlation Power Analysis with a Leakage Model" (CHES 2004)
    Mangard et al. "Power Analysis Attacks" (Springer 2007)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "CPAAnalyzer",
    "CPAResult",
    "DPAAnalyzer",
    "DPAResult",
    "LeakageModel",
    "hamming_distance",
    "hamming_weight",
]


class LeakageModel(Enum):
    """Power leakage models.

    Attributes:
        HAMMING_WEIGHT: Count of '1' bits in value.
        HAMMING_DISTANCE: Count of bit flips between states.
        IDENTITY: Raw value (linear leakage).
    """

    HAMMING_WEIGHT = "hamming_weight"
    HAMMING_DISTANCE = "hamming_distance"
    IDENTITY = "identity"


def hamming_weight(value: int | NDArray[np.integer[Any]]) -> int | NDArray[np.integer[Any]]:
    """Calculate Hamming weight (number of 1 bits).

    Args:
        value: Integer or array of integers.

    Returns:
        Hamming weight(s).

    Example:
        >>> hamming_weight(0x0F)
        4
        >>> hamming_weight(np.array([0x0F, 0xFF]))
        array([4, 8])
    """
    if isinstance(value, np.ndarray):
        # Optimized vectorized implementation using np.unpackbits
        # Handle different integer sizes by converting to uint8 representation
        arr = value.astype(np.uint64)

        # Convert to bytes and unpack bits (much faster than loop)
        # For each value, convert to 8 bytes (uint64) and count 1s
        result = np.zeros(arr.shape, dtype=np.int32)

        for byte_idx in range(8):  # Process 8 bytes for uint64
            byte_vals = ((arr >> (byte_idx * 8)) & 0xFF).astype(np.uint8)
            # Unpack bits and sum for each value
            for i, byte_val in enumerate(byte_vals):
                result[i] += bin(byte_val).count("1")

        return result
    else:
        # Scalar implementation using Python's built-in bit_count (Python 3.10+)
        # Fallback to bin().count() for compatibility
        count: int = bin(int(value)).count("1")
        return count


def hamming_distance(
    val1: int | NDArray[np.integer[Any]], val2: int | NDArray[np.integer[Any]]
) -> int | NDArray[np.integer[Any]]:
    """Calculate Hamming distance (number of differing bits).

    Args:
        val1: First value(s).
        val2: Second value(s).

    Returns:
        Hamming distance(s).

    Example:
        >>> hamming_distance(0x00, 0xFF)
        8
        >>> hamming_distance(0x0F, 0xF0)
        8
    """
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        v1 = np.asarray(val1)
        v2 = np.asarray(val2)
        return hamming_weight(np.bitwise_xor(v1, v2))
    else:
        return hamming_weight(val1 ^ val2)


# AES S-box for cryptographic modeling
AES_SBOX = np.array(
    [
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
    ],
    dtype=np.uint8,
)


@dataclass
class DPAResult:
    """Result of Differential Power Analysis attack.

    Attributes:
        key_guess: Most likely key byte value.
        differential_traces: Differential traces for each key hypothesis (256, n_samples).
        max_differential: Maximum differential value achieved.
        key_rank: Ranking of all key hypotheses by differential.
        peak_sample: Sample index where maximum differential occurs.
    """

    key_guess: int
    differential_traces: NDArray[np.floating[Any]]
    max_differential: float
    key_rank: NDArray[np.integer[Any]]
    peak_sample: int


@dataclass
class CPAResult:
    """Result of Correlation Power Analysis attack.

    Attributes:
        key_guess: Most likely key byte value.
        max_correlation: Maximum correlation coefficient achieved.
        correlations: Correlation values for each key hypothesis (256, n_samples).
        key_rank: Ranking of all key hypotheses by correlation.
        peak_sample: Sample index where maximum correlation occurs.
    """

    key_guess: int
    max_correlation: float
    correlations: NDArray[np.floating[Any]]
    key_rank: NDArray[np.integer[Any]]
    peak_sample: int


class DPAAnalyzer:
    """Differential Power Analysis (DPA) attack implementation.

    DPA exploits differences in power consumption based on data-dependent
    operations. Traces are partitioned by a selection function and averaged
    to reveal key-dependent differences.

    Args:
        target_bit: Bit position to target (0-7) for AES S-box output.
        byte_position: Byte position in key (0-15 for AES-128).

    Example:
        >>> dpa = DPAAnalyzer(target_bit=0, byte_position=0)
        >>> result = dpa.analyze(power_traces, known_plaintexts)
        >>> print(f"Key byte 0: 0x{result.key_guess:02X}")

    References:
        Kocher et al. "Differential Power Analysis" CRYPTO 1999
    """

    def __init__(self, target_bit: int = 0, byte_position: int = 0) -> None:
        """Initialize DPA analyzer.

        Args:
            target_bit: Target bit position (0-7).
            byte_position: Key byte position (0-15).

        Raises:
            ValueError: If parameters out of range.
        """
        if not 0 <= target_bit <= 7:
            raise ValueError(f"target_bit must be 0-7, got {target_bit}")
        if not 0 <= byte_position <= 15:
            raise ValueError(f"byte_position must be 0-15, got {byte_position}")

        self.target_bit = target_bit
        self.byte_position = byte_position

    def _selection_function(self, plaintext_byte: int, key_guess: int) -> int:
        """Selection function: bit value of S-box output.

        Args:
            plaintext_byte: Input plaintext byte.
            key_guess: Hypothetical key byte.

        Returns:
            Value of target bit (0 or 1).
        """
        sbox_out = AES_SBOX[plaintext_byte ^ key_guess]
        return int((sbox_out >> self.target_bit) & 1)

    def analyze(
        self,
        traces: NDArray[np.floating[Any]],
        plaintexts: NDArray[np.integer[Any]],
    ) -> DPAResult:
        """Perform DPA attack to recover key byte.

        Args:
            traces: Power traces (n_traces, n_samples).
            plaintexts: Known plaintexts (n_traces, 16) or (n_traces,) if single byte.

        Returns:
            DPAResult with key guess and differential traces.

        Raises:
            ValueError: If input shapes incompatible.

        Example:
            >>> traces = np.random.randn(1000, 5000)  # 1000 traces, 5000 samples
            >>> plaintexts = np.random.randint(0, 256, (1000, 16), dtype=np.uint8)
            >>> result = dpa.analyze(traces, plaintexts)
        """
        n_traces, n_samples = traces.shape

        # Extract target byte from plaintexts
        if plaintexts.ndim == 1:
            plaintext_bytes = plaintexts
        elif plaintexts.ndim == 2:
            plaintext_bytes = plaintexts[:, self.byte_position]
        else:
            raise ValueError(f"plaintexts must be 1D or 2D, got shape {plaintexts.shape}")

        if len(plaintext_bytes) != n_traces:
            raise ValueError(
                f"Number of plaintexts ({len(plaintext_bytes)}) must match traces ({n_traces})"
            )

        # Calculate differential for each key hypothesis
        differential_traces = np.zeros((256, n_samples), dtype=np.float64)

        for key_guess in range(256):
            # Partition traces by selection function
            selection_bits = np.array(
                [self._selection_function(pt, key_guess) for pt in plaintext_bytes]
            )

            set_0 = traces[selection_bits == 0]
            set_1 = traces[selection_bits == 1]

            if len(set_0) > 0 and len(set_1) > 0:
                # Differential = mean(set_1) - mean(set_0)
                differential_traces[key_guess] = np.mean(set_1, axis=0) - np.mean(set_0, axis=0)

        # Find key with maximum differential
        max_differentials = np.max(np.abs(differential_traces), axis=1)
        key_rank = np.argsort(max_differentials)[::-1]  # Descending order
        key_guess = key_rank[0]
        max_differential = max_differentials[key_guess]
        peak_sample = int(np.argmax(np.abs(differential_traces[key_guess])))

        return DPAResult(
            key_guess=int(key_guess),
            differential_traces=differential_traces,
            max_differential=float(max_differential),
            key_rank=key_rank,
            peak_sample=peak_sample,
        )


class CPAAnalyzer:
    """Correlation Power Analysis (CPA) attack implementation.

    CPA uses statistical correlation between power consumption and
    intermediate values predicted by a leakage model.

    Args:
        leakage_model: Leakage model ("hamming_weight", "hamming_distance", "identity").
        algorithm: Target algorithm ("aes_sbox", "des", "custom").
        byte_position: Key byte position to attack (0-15).

    Example:
        >>> cpa = CPAAnalyzer(leakage_model="hamming_weight", algorithm="aes_sbox")
        >>> result = cpa.analyze(power_traces, known_plaintexts)
        >>> print(f"Correlation: {result.max_correlation:.4f}")

    References:
        Brier et al. "Correlation Power Analysis" CHES 2004
    """

    def __init__(
        self,
        leakage_model: Literal["hamming_weight", "hamming_distance", "identity"] = "hamming_weight",
        algorithm: Literal["aes_sbox", "des", "custom"] = "aes_sbox",
        byte_position: int = 0,
    ) -> None:
        """Initialize CPA analyzer.

        Args:
            leakage_model: Power leakage model.
            algorithm: Target cryptographic algorithm.
            byte_position: Target key byte position.

        Raises:
            ValueError: If parameters invalid.
        """
        valid_models = ["hamming_weight", "hamming_distance", "identity"]
        if leakage_model not in valid_models:
            raise ValueError(f"leakage_model must be one of {valid_models}")

        if not 0 <= byte_position <= 15:
            raise ValueError(f"byte_position must be 0-15, got {byte_position}")

        self.leakage_model = leakage_model
        self.algorithm = algorithm
        self.byte_position = byte_position

        # Select leakage function
        if leakage_model == "hamming_weight":
            self._leakage_func: Callable[[NDArray[np.integer[Any]]], NDArray[np.integer[Any]]] = (
                hamming_weight  # type: ignore[assignment]
            )
        elif leakage_model == "identity":
            self._leakage_func = lambda x: x  # Identity function cannot be simplified
        else:
            self._leakage_func = hamming_weight  # type: ignore[assignment]

    def _compute_intermediate(
        self, plaintext_byte: NDArray[np.integer[Any]], key_guess: int
    ) -> NDArray[np.integer[Any]]:
        """Compute intermediate value for key hypothesis.

        Args:
            plaintext_byte: Plaintext byte values.
            key_guess: Hypothetical key byte.

        Returns:
            Intermediate values (e.g., S-box output).
        """
        if self.algorithm == "aes_sbox":
            result: NDArray[np.integer[Any]] = AES_SBOX[plaintext_byte ^ key_guess]
            return result
        else:
            # Default: XOR with key
            return plaintext_byte ^ key_guess

    def analyze(
        self,
        traces: NDArray[np.floating[Any]],
        plaintexts: NDArray[np.integer[Any]],
    ) -> CPAResult:
        """Perform CPA attack to recover key byte.

        Args:
            traces: Power traces (n_traces, n_samples).
            plaintexts: Known plaintexts (n_traces, 16) or (n_traces,).

        Returns:
            CPAResult with key guess and correlation matrix.

        Raises:
            ValueError: If input shapes incompatible.

        Example:
            >>> traces = np.random.randn(1000, 5000)
            >>> plaintexts = np.random.randint(0, 256, (1000, 16), dtype=np.uint8)
            >>> result = cpa.analyze(traces, plaintexts)
            >>> print(f"Best key: 0x{result.key_guess:02X}")
        """
        n_traces, n_samples = traces.shape

        # Extract target byte
        if plaintexts.ndim == 1:
            plaintext_bytes = plaintexts
        elif plaintexts.ndim == 2:
            plaintext_bytes = plaintexts[:, self.byte_position]
        else:
            raise ValueError(f"plaintexts must be 1D or 2D, got shape {plaintexts.shape}")

        if len(plaintext_bytes) != n_traces:
            raise ValueError(
                f"Number of plaintexts ({len(plaintext_bytes)}) must match traces ({n_traces})"
            )

        # Compute correlations for all key hypotheses
        correlations = np.zeros((256, n_samples), dtype=np.float64)

        for key_guess in range(256):
            # Compute intermediate values
            intermediates = self._compute_intermediate(plaintext_bytes, key_guess)

            # Apply leakage model
            hypothetical_power = self._leakage_func(intermediates).astype(np.float64)

            # Compute Pearson correlation for each sample point
            for sample_idx in range(n_samples):
                trace_sample = traces[:, sample_idx]

                # Pearson correlation coefficient
                correlations[key_guess, sample_idx] = np.corrcoef(hypothetical_power, trace_sample)[
                    0, 1
                ]

        # Handle NaN values (can occur with constant traces)
        correlations = np.nan_to_num(correlations, nan=0.0)

        # Find key with maximum absolute correlation
        max_correlations = np.max(np.abs(correlations), axis=1)
        key_rank = np.argsort(max_correlations)[::-1]
        key_guess = key_rank[0]
        max_correlation = max_correlations[key_guess]
        peak_sample = int(np.argmax(np.abs(correlations[key_guess])))

        return CPAResult(
            key_guess=int(key_guess),
            max_correlation=float(max_correlation),
            correlations=correlations,
            key_rank=key_rank,
            peak_sample=peak_sample,
        )
