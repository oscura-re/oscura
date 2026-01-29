"""Shannon entropy analysis for data classification and boundary detection.

    - RE-ENT-002: Byte Frequency Distribution

This module provides tools for computing Shannon entropy at both byte and bit
levels, analyzing entropy profiles over sliding windows, detecting entropy
transitions for field boundary identification, and classifying data types
based on entropy characteristics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for input data
DataType = Union[bytes, bytearray, "NDArray[np.uint8]"]


@dataclass
class EntropyResult:
    """Entropy analysis result.

    Attributes:
        entropy: Shannon entropy value (0-8 bits for byte-level)
        classification: Data type classification based on entropy
        confidence: Confidence score for classification (0-1)
    """

    entropy: float
    classification: Literal["structured", "text", "compressed", "random", "constant"]
    confidence: float


@dataclass
class EntropyTransition:
    """Detected entropy transition (potential field boundary).

    Attributes:
        offset: Byte offset where transition occurs
        entropy_before: Entropy value before transition
        entropy_after: Entropy value after transition
        delta: Change in entropy (entropy_after - entropy_before)
        transition_type: Direction of entropy change
    """

    offset: int
    entropy_before: float
    entropy_after: float
    delta: float
    transition_type: str  # 'low_to_high', 'high_to_low'

    @property
    def entropy_change(self) -> float:
        """Alias for delta - provides compatibility with test expectations."""
        return abs(self.delta)


@dataclass
class ByteFrequencyResult:
    """Result of byte frequency distribution analysis.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Attributes:
        counts: Byte value counts (256-element array).
        frequencies: Normalized frequencies (256-element array).
        entropy: Shannon entropy of distribution.
        unique_bytes: Number of unique byte values.
        most_common: List of (byte_value, count) for most common bytes.
        least_common: List of (byte_value, count) for least common bytes.
        uniformity_score: How uniform the distribution is (0-1).
        zero_byte_ratio: Proportion of zero bytes.
        printable_ratio: Proportion of printable ASCII.
    """

    counts: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    entropy: float
    unique_bytes: int
    most_common: list[tuple[int, int]]
    least_common: list[tuple[int, int]]
    uniformity_score: float
    zero_byte_ratio: float
    printable_ratio: float


@dataclass
class FrequencyAnomalyResult:
    """Result of frequency anomaly detection.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Attributes:
        anomalous_bytes: Byte values with unusual frequencies.
        z_scores: Z-score for each byte value.
        is_anomalous: Boolean mask for anomalous bytes.
        expected_frequency: Expected frequency for uniform distribution.
    """

    anomalous_bytes: list[int]
    z_scores: NDArray[np.float64]
    is_anomalous: NDArray[np.bool_]
    expected_frequency: float


@dataclass
class CompressionIndicator:
    """Indicators suggesting compression or encryption.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Attributes:
        is_compressed: Likely compressed data.
        is_encrypted: Likely encrypted data.
        compression_ratio_estimate: Estimated compression ratio.
        confidence: Confidence in classification (0-1).
        indicators: List of detected indicators.
    """

    is_compressed: bool
    is_encrypted: bool
    compression_ratio_estimate: float
    confidence: float
    indicators: list[str] = field(default_factory=list)


def shannon_entropy(data: DataType) -> float:
    """Calculate Shannon entropy in bits (0-8 for bytes).

    : Shannon Entropy Analysis

    Shannon entropy measures the average information content per byte.
    For byte data, maximum entropy is 8 bits (uniform distribution).

    Args:
        data: Input data as bytes, bytearray, or numpy array

    Returns:
        Entropy value in bits (0.0 to 8.0)

    Raises:
        ValueError: If data is empty

    Example:
        >>> shannon_entropy(b'\\x00' * 100)  # All zeros
        0.0
        >>> shannon_entropy(bytes(range(256)))  # Uniform
        8.0
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if not data:
        raise ValueError("Cannot calculate entropy of empty data")

    # Count byte frequencies
    counts = Counter(data)
    length = len(data)

    # Calculate Shannon entropy
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            prob = count / length
            entropy -= prob * np.log2(prob)

    return float(entropy)


def bit_entropy(data: DataType) -> float:
    """Calculate bit-level entropy (0-1).

    : Shannon Entropy Analysis

    Computes entropy of the bit distribution (0s vs 1s) across all bytes.

    Args:
        data: Input data as bytes, bytearray, or numpy array

    Returns:
        Bit-level entropy (0.0 to 1.0)

    Raises:
        ValueError: If data is empty

    Example:
        >>> bit_entropy(b'\\x00' * 100)  # All bits are 0
        0.0
        >>> bit_entropy(b'\\xAA' * 100)  # Equal 0s and 1s
        1.0
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if not data:
        raise ValueError("Cannot calculate entropy of empty data")

    # Count total bits
    total_bits = len(data) * 8

    # Count set bits
    ones = sum(bin(byte).count("1") for byte in data)
    zeros = total_bits - ones

    if ones == 0 or zeros == 0:
        return 0.0

    # Calculate bit entropy
    p_one = ones / total_bits
    p_zero = zeros / total_bits

    entropy = -(p_one * np.log2(p_one) + p_zero * np.log2(p_zero))

    return float(entropy)


def sliding_entropy(
    data: DataType, window: int = 256, step: int = 64, window_size: int | None = None
) -> NDArray[np.float64]:
    """Calculate sliding window entropy profile.

    : Shannon Entropy Analysis

    Computes entropy over a sliding window to create an entropy profile
    of the data, useful for visualization and boundary detection.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        window: Window size in bytes (default: 256)
        step: Step size for window movement (default: 64)
        window_size: Alias for window parameter (for compatibility)

    Returns:
        Array of entropy values at each window position

    Raises:
        ValueError: If window size is larger than data or step is invalid
    """
    # Support window_size alias
    if window_size is not None:
        window = window_size

    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if len(data) < window:
        raise ValueError(f"Window size ({window}) larger than data ({len(data)})")

    if step <= 0:
        raise ValueError(f"Step size must be positive, got {step}")

    # Calculate number of windows
    num_windows = (len(data) - window) // step + 1
    entropies = np.zeros(num_windows)

    for i in range(num_windows):
        start = i * step
        end = start + window
        window_data = data[start:end]
        # Use internal calculation to avoid ValueError for non-empty windows
        counts = Counter(window_data)
        length = len(window_data)
        entropy_val = 0.0
        for count in counts.values():
            if count > 0:
                prob = count / length
                entropy_val -= prob * np.log2(prob)
        entropies[i] = entropy_val

    return entropies


def detect_entropy_transitions(
    data: DataType,
    window: int = 256,
    threshold: float = 1.0,
    min_gap: int = 64,
    step: int | None = None,
) -> list[EntropyTransition]:
    """Detect significant entropy transitions (field boundaries).

    : Shannon Entropy Analysis

    Identifies locations where entropy changes significantly, which often
    correspond to transitions between different data types or field boundaries.

    The algorithm uses a dual-approach strategy:
    1. For each potential boundary point, compute entropy of regions BEFORE
       and AFTER (non-overlapping) to detect sharp transitions.
    2. Use sliding window for gradual transition detection.

    This approach properly handles sharp boundaries like low->high entropy
    transitions without blending across the boundary.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        window: Window size for entropy calculation (default: 256)
        threshold: Minimum entropy change to consider a transition (default: 1.0 bits)
        min_gap: Minimum gap between transitions to avoid duplicates (default: 64 bytes)
        step: Step size for sliding window (optional, defaults to window//4)

    Returns:
        List of detected entropy transitions, sorted by offset

    Example:
        >>> data = b'\\x00' * 1000 + b'\\xFF\\xEE\\xDD' * 333  # Low to high entropy
        >>> transitions = detect_entropy_transitions(data)
        >>> len(transitions) > 0
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    data_len = len(data)

    if data_len < 16:
        return []

    # Use boundary scanning approach - this works for both small and large data
    # by comparing non-overlapping regions before and after each potential boundary
    transitions = _detect_transitions_boundary_scan(bytes(data), window, threshold, min_gap)

    # If we found transitions via boundary scan, return them
    if transitions:
        return transitions

    # Fall back to sliding window approach for gradual transitions
    if data_len < window:
        return []

    if step is None:
        step = max(1, window // 4)

    effective_min_gap = min(min_gap, max(step * 2, data_len // 10))

    try:
        entropies = sliding_entropy(data, window=window, step=step)
    except ValueError:
        return []

    if len(entropies) < 2:
        return []

    last_offset = -effective_min_gap - 1

    # Find significant entropy changes between adjacent windows
    for i in range(1, len(entropies)):
        delta = entropies[i] - entropies[i - 1]

        if abs(delta) >= threshold:
            offset = i * step

            # Enforce minimum gap between transitions
            if offset - last_offset >= effective_min_gap:
                transition_type = "low_to_high" if delta > 0 else "high_to_low"

                transitions.append(
                    EntropyTransition(
                        offset=offset,
                        entropy_before=float(entropies[i - 1]),
                        entropy_after=float(entropies[i]),
                        delta=float(delta),
                        transition_type=transition_type,
                    )
                )
                last_offset = offset

    return transitions


def _detect_transitions_boundary_scan(
    data: bytes,
    window: int,
    threshold: float,
    min_gap: int,
) -> list[EntropyTransition]:
    """Detect entropy transitions using boundary scanning.

    For each potential boundary point, compare entropy of the region
    BEFORE the boundary to the region AFTER (non-overlapping regions).
    This properly detects sharp transitions without blending.

    Args:
        data: Input data as bytes
        window: Window size for region comparison
        threshold: Minimum entropy change to consider a transition
        min_gap: Minimum gap between transitions

    Returns:
        List of detected transitions
    """
    region_size = _determine_boundary_scan_region_size(len(data), window)
    if region_size < 4:
        return []

    scan_start, scan_end = _compute_boundary_scan_range(len(data), region_size)
    if scan_start >= scan_end:
        return []

    best_transition = _find_best_boundary_transition(
        data, region_size, scan_start, scan_end, threshold, min_gap
    )

    if best_transition is None:
        return []

    return _accumulate_all_boundary_transitions(data, best_transition, window, threshold, min_gap)


def _determine_boundary_scan_region_size(data_len: int, window: int) -> int:
    """Determine region size for boundary scanning."""
    region_size = min(window, data_len // 3)
    if region_size < 8:
        region_size = max(8, data_len // 4)
    return region_size


def _compute_boundary_scan_range(data_len: int, region_size: int) -> tuple[int, int]:
    """Compute scan range for boundary detection."""
    scan_start = region_size
    scan_end = data_len - region_size

    if scan_start >= scan_end:
        # Data too small for this region size, reduce it
        region_size = max(4, data_len // 4)
        scan_start = region_size
        scan_end = data_len - region_size

    return scan_start, scan_end


def _find_best_boundary_transition(
    data: bytes,
    region_size: int,
    scan_start: int,
    scan_end: int,
    threshold: float,
    min_gap: int,
) -> EntropyTransition | None:
    """Find the strongest boundary transition in scan range."""
    best_transition = None
    best_delta = 0.0
    last_offset = -min_gap - 1
    scan_step = max(1, region_size // 4)

    for offset in range(scan_start, scan_end + 1, scan_step):
        region_before = data[offset - region_size : offset]
        region_after = data[offset : offset + region_size]

        if len(region_before) < 4 or len(region_after) < 4:
            continue

        try:
            entropy_before = shannon_entropy(region_before)
            entropy_after = shannon_entropy(region_after)
        except ValueError:
            continue

        delta = entropy_after - entropy_before

        # Track the strongest transition that exceeds threshold
        if abs(delta) >= threshold and offset - last_offset >= min_gap:
            if abs(delta) > abs(best_delta):
                best_delta = delta
                best_transition = EntropyTransition(
                    offset=offset,
                    entropy_before=entropy_before,
                    entropy_after=entropy_after,
                    delta=delta,
                    transition_type="low_to_high" if delta > 0 else "high_to_low",
                )

    return best_transition


def _accumulate_all_boundary_transitions(
    data: bytes,
    best_transition: EntropyTransition,
    window: int,
    threshold: float,
    min_gap: int,
) -> list[EntropyTransition]:
    """Accumulate all boundary transitions including recursive finds."""
    transitions = [best_transition]
    last_offset = best_transition.offset

    # Continue scanning for more transitions after this one
    remaining_transitions = _detect_transitions_boundary_scan(
        data[best_transition.offset :],
        window,
        threshold,
        min_gap,
    )

    for t in remaining_transitions:
        adjusted_t = EntropyTransition(
            offset=t.offset + best_transition.offset,
            entropy_before=t.entropy_before,
            entropy_after=t.entropy_after,
            delta=t.delta,
            transition_type=t.transition_type,
        )
        if adjusted_t.offset - last_offset >= min_gap:
            transitions.append(adjusted_t)
            last_offset = adjusted_t.offset

    return transitions


def classify_by_entropy(data: DataType) -> EntropyResult:
    """Classify data type by entropy characteristics.

    : Shannon Entropy Analysis

    Classification criteria:
        - constant: entropy < 0.5 (highly repetitive)
        - text: entropy 0.5-6.0 AND high printable ratio (>= 0.9)
        - random: entropy >= 7.5 (encrypted or random data)
        - compressed: entropy 6.0-7.5 (compressed data)
        - structured: other (structured binary data)

    Args:
        data: Input data as bytes, bytearray, or numpy array

    Returns:
        EntropyResult with classification and confidence

    Raises:
        ValueError: If data is empty

    Example:
        >>> result = classify_by_entropy(b'\\x00' * 100)
        >>> result.classification
        'constant'
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if not data:
        raise ValueError("Cannot classify empty data")

    # Calculate entropy
    entropy_val = shannon_entropy(data)

    # Calculate printable ratio for text detection
    # Include standard printable ASCII (32-126) plus tab, newline, carriage return
    printable_count = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
    printable_ratio = printable_count / len(data)

    # Classify based on entropy and characteristics
    # Order matters: check specific cases first, then fall through to general

    # 1. Constant/repetitive data - very low entropy
    classification: Literal["structured", "text", "compressed", "random", "constant"]
    if entropy_val < 0.5:
        classification = "constant"
        confidence = 1.0 - (entropy_val / 0.5) * 0.2  # High confidence

    # 2. Random/encrypted data - very high entropy (near maximum)
    elif entropy_val >= 7.5:
        classification = "random"
        confidence = min(1.0, (entropy_val - 7.5) / 0.5 + 0.8)

    # 3. Compressed data - high entropy but not maximum
    elif entropy_val >= 6.0:
        classification = "compressed"
        confidence = min(1.0, (entropy_val - 6.0) / 1.5 + 0.6)

    # 4. Text data - high printable ratio (checked BEFORE structured)
    # Text can have entropy from ~2.5 to ~5.5 depending on language/content
    # We use a high printable threshold (0.9) to distinguish from structured binary
    elif printable_ratio >= 0.9 and entropy_val >= 0.5:
        classification = "text"
        confidence = min(1.0, printable_ratio)

    # 5. Structured binary - everything else
    else:
        classification = "structured"
        confidence = 0.7  # Medium confidence for default case

    return EntropyResult(
        entropy=float(entropy_val), classification=classification, confidence=float(confidence)
    )


def entropy_profile(data: DataType, window: int = 256) -> NDArray[np.float64]:
    """Generate entropy profile for visualization.

    : Shannon Entropy Analysis

    Creates a smoothed entropy profile suitable for plotting and visual analysis.
    Uses overlapping windows with a step size of window/4 for smoother results.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        window: Window size in bytes (default: 256)

    Returns:
        Array of entropy values across the data

    Example:
        >>> data = bytes(range(256)) * 10
        >>> profile = entropy_profile(data)
        >>> len(profile) > 0
        True
    """
    step = max(1, window // 4)  # Overlapping windows for smooth profile
    return sliding_entropy(data, window=window, step=step)


def entropy_histogram(data: DataType) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Generate byte frequency histogram.

    : Shannon Entropy Analysis

    Creates a histogram of byte values (0-255) showing their frequencies.
    Useful for visualizing data distribution and entropy characteristics.

    Args:
        data: Input data as bytes, bytearray, or numpy array

    Returns:
        Tuple of (bin_edges, frequencies) where:
            - bin_edges: Array of 256 byte values (0-255)
            - frequencies: Array of normalized frequencies (0-1)

    Example:
        >>> bins, freqs = entropy_histogram(b'\\x00' * 50 + b'\\xFF' * 50)
        >>> len(bins)
        256
        >>> sum(freqs)
        1.0
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if not data:
        return np.arange(256), np.zeros(256)

    # Count byte frequencies
    counts = np.zeros(256, dtype=np.int64)
    for byte in data:
        counts[byte] += 1

    # Normalize to frequencies
    frequencies = counts / len(data)

    # Bin edges are byte values
    bin_edges = np.arange(256)

    return bin_edges, frequencies


# =============================================================================
# RE-ENT-002: Byte Frequency Distribution
# =============================================================================


def byte_frequency_distribution(data: DataType, n_most_common: int = 10) -> ByteFrequencyResult:
    """Analyze byte frequency distribution in data.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Computes detailed byte frequency statistics including counts, frequencies,
    most/least common bytes, uniformity score, and characteristic ratios.

    Args:
        data: Input data as bytes, bytearray, or numpy array.
        n_most_common: Number of most/least common bytes to report.

    Returns:
        ByteFrequencyResult with comprehensive distribution analysis.

    Example:
        >>> data = b'\\x00\\x00\\x01\\x02\\x03'
        >>> result = byte_frequency_distribution(data)
        >>> result.unique_bytes
        4
        >>> result.most_common[0]
        (0, 2)
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if not data:
        return ByteFrequencyResult(
            counts=np.zeros(256, dtype=np.int64),
            frequencies=np.zeros(256, dtype=np.float64),
            entropy=0.0,
            unique_bytes=0,
            most_common=[],
            least_common=[],
            uniformity_score=0.0,
            zero_byte_ratio=0.0,
            printable_ratio=0.0,
        )

    # Count bytes
    counts = np.zeros(256, dtype=np.int64)
    for byte in data:
        counts[byte] += 1

    # Normalize frequencies
    length = len(data)
    frequencies = counts / length

    # Calculate entropy (use internal calculation to avoid ValueError)
    byte_counts = Counter(data)
    entropy_val = 0.0
    for count in byte_counts.values():
        if count > 0:
            prob = count / length
            entropy_val -= prob * np.log2(prob)

    # Count unique bytes
    unique_bytes = np.count_nonzero(counts)

    # Find most and least common bytes
    nonzero_indices = np.where(counts > 0)[0]
    sorted_indices = nonzero_indices[np.argsort(-counts[nonzero_indices])]

    most_common = [(int(i), int(counts[i])) for i in sorted_indices[:n_most_common]]
    least_common = [(int(i), int(counts[i])) for i in sorted_indices[-n_most_common:][::-1]]

    # Calculate uniformity score (1 = perfectly uniform, 0 = single byte)
    expected_freq = 1.0 / 256
    if unique_bytes > 0:
        # Chi-squared like uniformity measure
        observed_freqs = frequencies[frequencies > 0]
        deviation = np.sum((observed_freqs - expected_freq) ** 2)
        max_deviation = (1.0 - expected_freq) ** 2 + 255 * expected_freq**2
        uniformity_score = 1.0 - min(1.0, deviation / max_deviation)
    else:
        uniformity_score = 0.0

    # Calculate characteristic ratios
    zero_byte_ratio = counts[0] / length if length > 0 else 0.0

    # Printable ASCII range
    printable_count = sum(counts[i] for i in range(32, 127))
    printable_count += counts[9] + counts[10] + counts[13]  # Tab, LF, CR
    printable_ratio = printable_count / length if length > 0 else 0.0

    return ByteFrequencyResult(
        counts=counts,
        frequencies=frequencies,
        entropy=entropy_val,
        unique_bytes=unique_bytes,
        most_common=most_common,
        least_common=least_common,
        uniformity_score=uniformity_score,
        zero_byte_ratio=zero_byte_ratio,
        printable_ratio=printable_ratio,
    )


def detect_frequency_anomalies(data: DataType, z_threshold: float = 3.0) -> FrequencyAnomalyResult:
    """Detect bytes with anomalous frequencies.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Identifies byte values that occur with unusual frequency compared to
    expected distribution using z-score analysis.

    Args:
        data: Input data as bytes, bytearray, or numpy array.
        z_threshold: Z-score threshold for anomaly detection.

    Returns:
        FrequencyAnomalyResult with anomalous bytes.

    Example:
        >>> data = b'A' * 100 + bytes(range(256))
        >>> result = detect_frequency_anomalies(data)
        >>> 65 in result.anomalous_bytes  # 'A' is anomalous
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    length = len(data) if data else 0

    if length == 0:
        return FrequencyAnomalyResult(
            anomalous_bytes=[],
            z_scores=np.zeros(256),
            is_anomalous=np.zeros(256, dtype=bool),
            expected_frequency=0.0,
        )

    # Count bytes
    counts = np.zeros(256, dtype=np.int64)
    for byte in data:
        counts[byte] += 1

    # Expected frequency under uniform distribution
    expected_count = length / 256
    expected_freq = 1.0 / 256

    # Calculate z-scores
    # Using binomial approximation: std = sqrt(n * p * (1-p))
    std = np.sqrt(length * expected_freq * (1 - expected_freq))
    if std == 0:
        std = 1.0  # Avoid division by zero

    z_scores = (counts - expected_count) / std

    # Identify anomalies
    is_anomalous = np.abs(z_scores) > z_threshold
    anomalous_bytes = list(np.where(is_anomalous)[0])

    return FrequencyAnomalyResult(
        anomalous_bytes=[int(b) for b in anomalous_bytes],
        z_scores=z_scores,
        is_anomalous=is_anomalous,
        expected_frequency=expected_freq,
    )


def compare_byte_distributions(
    data_a: DataType, data_b: DataType
) -> tuple[float, float, NDArray[np.float64]]:
    """Compare byte frequency distributions between two data samples.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Computes chi-squared distance, Kullback-Leibler divergence, and
    per-byte frequency differences.

    Args:
        data_a: First data sample.
        data_b: Second data sample.

    Returns:
        Tuple of (chi_squared_distance, kl_divergence, frequency_diffs).

    Example:
        >>> data_a = bytes(range(256)) * 10
        >>> data_b = bytes(range(256)) * 10
        >>> chi_sq, kl_div, diffs = compare_byte_distributions(data_a, data_b)
        >>> chi_sq < 0.01  # Very similar
        True
    """
    # Get frequency distributions
    result_a = byte_frequency_distribution(data_a)
    result_b = byte_frequency_distribution(data_b)

    freq_a = result_a.frequencies
    freq_b = result_b.frequencies

    # Compute chi-squared distance
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    chi_squared = np.sum((freq_a - freq_b) ** 2 / (freq_a + freq_b + eps))

    # Compute KL divergence (symmetrized)
    freq_a_safe = np.clip(freq_a, eps, 1.0)
    freq_b_safe = np.clip(freq_b, eps, 1.0)

    kl_ab = np.sum(freq_a_safe * np.log(freq_a_safe / freq_b_safe))
    kl_ba = np.sum(freq_b_safe * np.log(freq_b_safe / freq_a_safe))
    kl_divergence = (kl_ab + kl_ba) / 2

    # Per-byte frequency differences
    frequency_diffs = freq_a - freq_b

    return float(chi_squared), float(kl_divergence), frequency_diffs


def sliding_byte_frequency(
    data: DataType, window: int = 256, step: int = 64, byte_value: int | None = None
) -> NDArray[np.float64]:
    """Compute sliding window byte frequency profile.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Tracks how byte frequency varies across the data, useful for
    detecting regions with different characteristics.

    Args:
        data: Input data.
        window: Window size in bytes.
        step: Step size for sliding window.
        byte_value: Specific byte to track (None for all).

    Returns:
        Array of frequencies at each window position.
        If byte_value is None, returns array of shape (n_windows, 256).

    Example:
        >>> data = b'\\x00' * 1000 + b'\\xFF' * 1000
        >>> profile = sliding_byte_frequency(data, byte_value=0)
        >>> profile[0] > profile[-1]  # More zeros at start
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if len(data) < window:
        if byte_value is not None:
            return np.array([])
        return np.zeros((0, 256))

    num_windows = (len(data) - window) // step + 1

    if byte_value is not None:
        # Track single byte value
        profile = np.zeros(num_windows)
        for i in range(num_windows):
            start = i * step
            window_data = data[start : start + window]
            profile[i] = window_data.count(byte_value) / window
        return profile
    else:
        # Track all byte values
        profile = np.zeros((num_windows, 256))
        for i in range(num_windows):
            start = i * step
            window_data = data[start : start + window]
            for byte in window_data:
                profile[i, byte] += 1
            profile[i] /= window
        return profile


def detect_compression_indicators(data: DataType) -> CompressionIndicator:
    """Detect indicators of compression or encryption.

    Implements RE-ENT-002: Byte Frequency Distribution.

    Analyzes byte frequency distribution to identify characteristics
    typical of compressed or encrypted data.

    Args:
        data: Input data to analyze.

    Returns:
        CompressionIndicator with detection results.

    Example:
        >>> import os
        >>> random_data = os.urandom(1000)
        >>> result = detect_compression_indicators(random_data)
        >>> result.is_encrypted
        True
    """
    freq_result = byte_frequency_distribution(data)
    _entropy_result = classify_by_entropy(data)

    indicators = []
    is_compressed = False
    is_encrypted = False
    confidence = 0.0
    compression_ratio_estimate = 1.0

    entropy = freq_result.entropy

    # High entropy (> 7.5) suggests encryption
    if entropy >= 7.5:
        is_encrypted = True
        confidence = min(1.0, (entropy - 7.5) / 0.5 + 0.7)
        indicators.append(f"Very high entropy: {entropy:.2f} bits")

    # Moderately high entropy (6.0-7.5) suggests compression
    elif entropy >= 6.0:
        is_compressed = True
        confidence = min(1.0, (entropy - 6.0) / 1.5 + 0.5)
        compression_ratio_estimate = 1.0 - (entropy - 6.0) / 2.0
        indicators.append(f"High entropy: {entropy:.2f} bits")

    # Check uniformity
    if freq_result.uniformity_score > 0.8:
        if not is_encrypted:
            is_encrypted = True
            confidence = max(confidence, 0.6)
        indicators.append(f"Uniform byte distribution: {freq_result.uniformity_score:.2f}")

    # Check for few unique bytes (suggests compression)
    if freq_result.unique_bytes < 128 and entropy > 5.0:
        if not is_compressed:
            is_compressed = True
            confidence = max(confidence, 0.5)
        indicators.append(f"Limited byte vocabulary: {freq_result.unique_bytes}")

    # Low printable ratio suggests binary/compressed
    if freq_result.printable_ratio < 0.1 and entropy > 5.0:
        indicators.append(f"Low printable ratio: {freq_result.printable_ratio:.2%}")

    return CompressionIndicator(
        is_compressed=is_compressed,
        is_encrypted=is_encrypted,
        compression_ratio_estimate=compression_ratio_estimate,
        confidence=confidence,
        indicators=indicators,
    )


class EntropyAnalyzer:
    """Object-oriented wrapper for entropy analysis functionality.

    Provides a class-based interface for entropy operations,
    wrapping the functional API for consistency with test expectations.



    Example:
        >>> analyzer = EntropyAnalyzer()
        >>> entropy = analyzer.calculate_entropy(data)
    """

    def __init__(
        self,
        entropy_type: Literal["byte", "bit"] = "byte",
        window_size: int = 256,
    ):
        """Initialize entropy analyzer.

        Args:
            entropy_type: Type of entropy calculation ('byte' or 'bit').
            window_size: Default window size for sliding operations.
        """
        self.entropy_type = entropy_type
        self.window_size = window_size

    def calculate_entropy(self, data: DataType) -> float:
        """Calculate Shannon entropy of data.

        Args:
            data: Input data to analyze.

        Returns:
            Shannon entropy value.

        Example:
            >>> analyzer = EntropyAnalyzer()
            >>> entropy = analyzer.calculate_entropy(b"Hello World")
        """
        if self.entropy_type == "byte":
            return shannon_entropy(data)
        else:
            return bit_entropy(data)

    def analyze(self, data: DataType) -> EntropyResult:
        """Analyze data and classify by entropy.

        Args:
            data: Input data to analyze.

        Returns:
            EntropyResult with classification.
        """
        return classify_by_entropy(data)

    def detect_transitions(
        self,
        data: DataType,
        threshold: float = 0.5,
        window: int | None = None,
        step: int | None = None,
    ) -> list[EntropyTransition]:
        """Detect entropy transitions in data.

        Args:
            data: Input data to analyze.
            threshold: Minimum entropy change to detect.
            window: Window size for sliding entropy (defaults to self.window_size).
            step: Step size between windows.

        Returns:
            List of detected transitions.
        """
        if window is None:
            window = self.window_size
        return detect_entropy_transitions(data, window=window, threshold=threshold, step=step)

    def analyze_blocks(self, data: DataType, block_size: int = 256) -> list[float]:
        """Analyze entropy of fixed-size blocks.

        Args:
            data: Input data to analyze.
            block_size: Size of each block in bytes.

        Returns:
            List of entropy values for each block.

        Example:
            >>> analyzer = EntropyAnalyzer()
            >>> entropies = analyzer.analyze_blocks(data, block_size=256)
        """
        if isinstance(data, np.ndarray):
            data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

        if not data:
            return []

        entropies = []
        for i in range(0, len(data), block_size):
            block = data[i : i + block_size]
            if len(block) >= block_size // 2:  # Only analyze blocks at least half size
                # Use internal calculation to avoid ValueError
                counts = Counter(block)
                length = len(block)
                entropy_val = 0.0
                for count in counts.values():
                    if count > 0:
                        prob = count / length
                        entropy_val -= prob * np.log2(prob)
                entropies.append(entropy_val)

        return entropies


__all__ = [
    # RE-ENT-002: Byte Frequency Distribution
    "ByteFrequencyResult",
    "CompressionIndicator",
    "EntropyAnalyzer",
    "EntropyResult",
    "EntropyTransition",
    "FrequencyAnomalyResult",
    "bit_entropy",
    "byte_frequency_distribution",
    "classify_by_entropy",
    "compare_byte_distributions",
    "detect_compression_indicators",
    "detect_entropy_transitions",
    "detect_frequency_anomalies",
    "entropy_histogram",
    "entropy_profile",
    "shannon_entropy",
    "sliding_byte_frequency",
    "sliding_entropy",
]
