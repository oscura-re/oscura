"""Jitter decomposition into random and deterministic components.

This module implements IEEE 2414-2020 compliant jitter decomposition
using the dual-Dirac model and spectral analysis techniques.


Example:
    >>> from oscura.analyzers.jitter.decomposition import extract_rj, extract_dj
    >>> rj_result = extract_rj(tie_data)
    >>> print(f"RJ RMS: {rj_result.rj_rms * 1e12:.2f} ps")

References:
    IEEE 2414-2020: Standard for Jitter and Phase Noise
    Dual-Dirac Model: JEDEC JESD65C
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import stats

from oscura.core.exceptions import InsufficientDataError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class RandomJitterResult:
    """Result of random jitter extraction.

    Attributes:
        rj_rms: Random jitter RMS value in seconds.
        method: Method used for extraction ("tail_fit" or "q_scale").
        confidence: Confidence score (0.0 to 1.0).
        sigma: Gaussian sigma parameter in seconds.
        mu: Gaussian mean offset in seconds.
        n_samples: Number of TIE samples used.
    """

    rj_rms: float
    method: str
    confidence: float
    sigma: float
    mu: float
    n_samples: int


@dataclass
class DeterministicJitterResult:
    """Result of deterministic jitter extraction.

    Attributes:
        dj_pp: Peak-to-peak deterministic jitter in seconds.
        dj_delta: Dual-Dirac delta (half-width) in seconds.
        method: Method used for extraction.
        confidence: Confidence score (0.0 to 1.0).
        histogram: Histogram counts for analysis.
        bin_centers: Bin centers for histogram.
    """

    dj_pp: float
    dj_delta: float
    method: str
    confidence: float
    histogram: NDArray[np.float64] | None = None
    bin_centers: NDArray[np.float64] | None = None


@dataclass
class PeriodicJitterResult:
    """Result of periodic jitter extraction.

    Attributes:
        components: List of (frequency_hz, amplitude_seconds) tuples.
        pj_pp: Total periodic jitter peak-to-peak in seconds.
        dominant_frequency: Most significant PJ frequency in Hz.
        dominant_amplitude: Amplitude at dominant frequency in seconds.
    """

    components: list[tuple[float, float]]
    pj_pp: float
    dominant_frequency: float | None
    dominant_amplitude: float | None


@dataclass
class DataDependentJitterResult:
    """Result of data-dependent jitter extraction.

    Attributes:
        ddj_pp: Peak-to-peak DDJ in seconds.
        pattern_histogram: Jitter vs bit pattern histogram.
        pattern_length: Length of bit patterns analyzed.
        isi_coefficient: ISI correlation coefficient.
    """

    ddj_pp: float
    pattern_histogram: dict[str, float]
    pattern_length: int
    isi_coefficient: float


@dataclass
class JitterDecomposition:
    """Complete jitter decomposition result.

    Attributes:
        rj: Random jitter component.
        dj: Deterministic jitter component.
        pj: Periodic jitter component (optional).
        ddj: Data-dependent jitter component (optional).
        tj_pp: Total jitter peak-to-peak at measured BER.
        ber_measured: BER at which TJ was measured.
    """

    rj: RandomJitterResult
    dj: DeterministicJitterResult
    pj: PeriodicJitterResult | None = None
    ddj: DataDependentJitterResult | None = None
    tj_pp: float | None = None
    ber_measured: float | None = None

    @property
    def rj_rms(self) -> float:
        """Convenience property for random jitter RMS."""
        return self.rj.rj_rms

    @property
    def dj_pp(self) -> float:
        """Convenience property for deterministic jitter peak-to-peak."""
        return self.dj.dj_pp


def extract_rj(
    tie_data: NDArray[np.float64],
    *,
    method: Literal["tail_fit", "q_scale", "auto"] = "auto",
    min_samples: int = 1000,
) -> RandomJitterResult:
    """Extract random jitter component from TIE data.

    Uses the dual-Dirac model to separate random (Gaussian) jitter
    from the total jitter distribution. RJ is the unbounded random
    component typically caused by thermal noise.

    Args:
        tie_data: Time Interval Error data array in seconds.
        method: Extraction method:
            - "tail_fit": Fit Gaussian to distribution tails
            - "q_scale": Q-scale (probabilistic) analysis
            - "auto": Automatically select best method
        min_samples: Minimum samples required for analysis.

    Returns:
        RandomJitterResult with RJ_rms and analysis details.

    Raises:
        InsufficientDataError: If tie_data has fewer than min_samples.
        ValueError: If method is not recognized.

    Example:
        >>> rj = extract_rj(tie_data)
        >>> print(f"RJ: {rj.rj_rms * 1e12:.2f} ps RMS")

    References:
        IEEE 2414-2020 Section 6.2
    """
    if len(tie_data) < min_samples:
        raise InsufficientDataError(
            f"RJ extraction requires at least {min_samples} samples",
            required=min_samples,
            available=len(tie_data),
            analysis_type="random_jitter_extraction",
        )

    # Remove NaN values
    valid_data = tie_data[~np.isnan(tie_data)]

    if len(valid_data) < min_samples:
        raise InsufficientDataError(
            f"RJ extraction requires at least {min_samples} valid samples",
            required=min_samples,
            available=len(valid_data),
            analysis_type="random_jitter_extraction",
        )

    # Select method
    if method == "auto":
        # Use Q-scale for large datasets, tail_fit for smaller
        method = "q_scale" if len(valid_data) > 10000 else "tail_fit"

    if method == "tail_fit":
        return _extract_rj_tail_fit(valid_data)
    elif method == "q_scale":
        return _extract_rj_q_scale(valid_data)
    else:
        raise ValueError(f"Unknown method: {method}")


def _extract_rj_tail_fit(tie_data: NDArray[np.float64]) -> RandomJitterResult:
    """Extract RJ using Gaussian tail fitting.

    Fits a Gaussian distribution to the outer tails of the TIE histogram
    where deterministic jitter has minimal effect.

    Args:
        tie_data: Time Interval Error data array in seconds.

    Returns:
        RandomJitterResult with RJ_rms and analysis details.
    """
    # For pure Gaussian data, the tails should follow a Gaussian perfectly.
    # The key insight is to use Q-Q plot analysis on the extreme tails.

    sorted_data = np.sort(tie_data)
    n = len(sorted_data)

    # Use percentiles to estimate Gaussian parameters
    # For a Gaussian: P16 = μ - σ, P50 = μ, P84 = μ + σ
    p16 = np.percentile(sorted_data, 16)
    p50 = np.percentile(sorted_data, 50)
    p84 = np.percentile(sorted_data, 84)

    # Estimate sigma from the 68% confidence interval
    sigma_estimate = (p84 - p16) / 2
    mu_estimate = p50

    # Refine estimate using tail data (beyond ±2 sigma)
    # For pure Gaussian, fit Q-Q plot in the tails
    tail_fraction = 0.025  # Use outer 2.5% on each side (beyond ~2 sigma)
    lower_tail_idx = int(n * tail_fraction)
    upper_tail_idx = int(n * (1 - tail_fraction))

    # Get tail indices
    tail_indices = np.concatenate([np.arange(0, lower_tail_idx), np.arange(upper_tail_idx, n)])

    if len(tail_indices) >= 10:
        # Q-Q plot analysis on tails
        tail_data = sorted_data[tail_indices]
        tail_probabilities = np.array(
            [i / (n - 1) if i < lower_tail_idx else (i + 1) / (n - 1) for i in tail_indices]
        )

        # Get theoretical quantiles
        theoretical_quantiles = stats.norm.ppf(tail_probabilities)
        valid_mask = np.isfinite(theoretical_quantiles)

        if np.sum(valid_mask) >= 10:
            # Linear regression: data = sigma * theoretical + mu
            slope, intercept, r_value, _, _ = stats.linregress(
                theoretical_quantiles[valid_mask], tail_data[valid_mask]
            )

            sigma = abs(slope)
            mu = intercept
            confidence = max(0.0, min(1.0, r_value**2))
        else:
            sigma = sigma_estimate
            mu = mu_estimate
            confidence = 0.6
    else:
        sigma = sigma_estimate
        mu = mu_estimate
        confidence = 0.6

    return RandomJitterResult(
        rj_rms=sigma,
        method="tail_fit",
        confidence=confidence,
        sigma=sigma,
        mu=mu,
        n_samples=len(tie_data),
    )


def _extract_rj_q_scale(tie_data: NDArray[np.float64]) -> RandomJitterResult:
    """Extract RJ using Q-scale (probability plot) analysis.

    Uses quantile-quantile analysis to separate Gaussian (random)
    from non-Gaussian (deterministic) components.

    Args:
        tie_data: Time Interval Error data array in seconds.

    Returns:
        RandomJitterResult with RJ_rms and analysis details.
    """
    n = len(tie_data)
    sorted_data = np.sort(tie_data)

    # Calculate theoretical Gaussian quantiles
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = stats.norm.ppf(probabilities)

    # Remove infinities from edges
    valid_mask = np.isfinite(theoretical_quantiles)
    theoretical_quantiles = theoretical_quantiles[valid_mask]
    sorted_data = sorted_data[valid_mask]

    # Focus on the linear (Gaussian) region in the tails
    # Use outer 30% of data for slope estimation
    n_valid = len(sorted_data)
    tail_frac = 0.15

    lower_idx = int(n_valid * tail_frac)
    upper_idx = int(n_valid * (1 - tail_frac))

    # Combine tail indices
    tail_indices = np.concatenate([np.arange(0, lower_idx), np.arange(upper_idx, n_valid)])

    if len(tail_indices) < 10:
        # Fall back to simple estimation
        sigma = np.std(sorted_data)
        mu = np.mean(sorted_data)
        confidence = 0.3
    else:
        # Linear regression on Q-Q tail data
        x_tail = theoretical_quantiles[tail_indices]
        y_tail = sorted_data[tail_indices]

        slope, intercept, r_value, _p_value, _std_err = stats.linregress(x_tail, y_tail)

        # Slope of Q-Q plot is sigma, intercept is mu
        sigma = abs(slope)
        mu = intercept
        confidence = min(1.0, max(0.0, r_value**2))

    return RandomJitterResult(
        rj_rms=sigma,
        method="q_scale",
        confidence=confidence,
        sigma=sigma,
        mu=mu,
        n_samples=n,
    )


def _prepare_dj_histogram(
    valid_data: NDArray[np.float64],
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Create histogram for DJ analysis.

    Args:
        valid_data: Valid TIE data array.

    Returns:
        Tuple of (histogram, bin_centers).
    """
    n_bins = min(100, len(valid_data) // 50)
    hist, bin_edges = np.histogram(valid_data, bins=n_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hist, bin_centers


def _detect_bimodal_peaks(hist: NDArray[np.int_], bin_centers: NDArray[np.float64]) -> float | None:
    """Detect bimodal peaks in histogram for DJ estimation.

    Args:
        hist: Histogram counts.
        bin_centers: Bin center positions.

    Returns:
        Peak separation distance or None if not bimodal.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    if len(hist) < 5:
        return None

    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
    peaks, properties = find_peaks(hist_smooth, prominence=np.max(hist_smooth) * 0.1)

    if len(peaks) >= 2:
        prominences = properties.get("prominences", np.ones(len(peaks)))
        sorted_peak_idx = np.argsort(prominences)[::-1][:2]
        top_peaks = peaks[sorted_peak_idx]
        top_peaks = np.sort(top_peaks)
        peak_positions = bin_centers[top_peaks]
        separation: float = float(abs(peak_positions[1] - peak_positions[0]))
        return separation

    return None


def _calculate_dj_from_quantiles(sorted_data: NDArray[np.float64], rj_rms: float) -> float:
    """Calculate DJ using quantile-based method.

    Args:
        sorted_data: Sorted TIE data.
        rj_rms: Random jitter RMS.

    Returns:
        DJ peak-to-peak value.
    """
    n = len(sorted_data)
    lower_idx = max(0, int(n * 0.0001))
    upper_idx = min(n - 1, int(n * 0.9999))
    tj_at_ber = sorted_data[upper_idx] - sorted_data[lower_idx]

    # For dual-Dirac + RJ: TJ = 2*Q*RJ + DJ at BER = 1e-4 (Q ≈ 3.72)
    q_factor = 3.72
    rj_contribution = 2 * q_factor * rj_rms
    dj_pp: float = float(max(0.0, tj_at_ber - rj_contribution))
    return dj_pp


def _determine_dj_confidence(
    peak_separation_dj: float | None, dj_pp: float, rj_rms: float, n_peaks: int
) -> float:
    """Determine confidence score for DJ extraction.

    Args:
        peak_separation_dj: Peak separation from histogram.
        dj_pp: Calculated DJ peak-to-peak.
        rj_rms: Random jitter RMS.
        n_peaks: Number of peaks found.

    Returns:
        Confidence score (0.0 to 1.0).
    """
    if peak_separation_dj is not None:
        return 0.9 if n_peaks == 2 else 0.7
    elif dj_pp > 2 * rj_rms:
        return 0.5
    else:
        return 0.2


def extract_dj(
    tie_data: NDArray[np.float64],
    rj_result: RandomJitterResult | None = None,
    *,
    min_samples: int = 1000,
) -> DeterministicJitterResult:
    """Extract deterministic jitter component from TIE data.

    DJ is the bounded, repeatable component of jitter. It is calculated
    as TJ - RJ contribution using the dual-Dirac model.

    Args:
        tie_data: Time Interval Error data array in seconds.
        rj_result: Pre-computed RJ result (computed if None).
        min_samples: Minimum samples required.

    Returns:
        DeterministicJitterResult with DJ_pp value.

    Raises:
        InsufficientDataError: If insufficient samples.

    Example:
        >>> dj = extract_dj(tie_data)
        >>> print(f"DJ: {dj.dj_pp * 1e12:.2f} ps peak-to-peak")

    References:
        IEEE 2414-2020 Section 6.3
    """
    if len(tie_data) < min_samples:
        raise InsufficientDataError(
            f"DJ extraction requires at least {min_samples} samples",
            required=min_samples,
            available=len(tie_data),
            analysis_type="deterministic_jitter_extraction",
        )

    # Setup: prepare data and compute RJ if needed
    valid_data = tie_data[~np.isnan(tie_data)]
    if rj_result is None:
        rj_result = extract_rj(valid_data, method="tail_fit", min_samples=min_samples)
    rj_rms = rj_result.rj_rms

    # Processing: analyze histogram and detect peaks
    hist, bin_centers = _prepare_dj_histogram(valid_data)
    peak_separation_dj = _detect_bimodal_peaks(hist, bin_centers)
    sorted_data = np.sort(valid_data)

    # Result building: calculate DJ and confidence
    if peak_separation_dj is not None and peak_separation_dj > 2 * rj_rms:
        dj_pp = peak_separation_dj
        n_peaks = 2
    else:
        dj_pp = _calculate_dj_from_quantiles(sorted_data, rj_rms)
        n_peaks = 0

    dj_delta = dj_pp / 2
    confidence = _determine_dj_confidence(peak_separation_dj, dj_pp, rj_rms, n_peaks)

    return DeterministicJitterResult(
        dj_pp=dj_pp,
        dj_delta=dj_delta,
        method="dual_dirac",
        confidence=confidence,
        histogram=hist.astype(np.float64),
        bin_centers=bin_centers,
    )


def extract_pj(
    tie_data: NDArray[np.float64],
    sample_rate: float,
    *,
    min_frequency: float = 1.0,
    max_frequency: float | None = None,
    n_components: int = 5,
) -> PeriodicJitterResult:
    """Extract periodic jitter components via spectral analysis.

    Uses FFT of TIE data to identify sinusoidal jitter components,
    typically caused by power supply noise or EMI.

    Args:
        tie_data: Time Interval Error data array in seconds.
        sample_rate: Sample rate of edge events (edges per second).
        min_frequency: Minimum PJ frequency to detect (Hz).
        max_frequency: Maximum PJ frequency (default: Nyquist).
        n_components: Number of periodic components to extract.

    Returns:
        PeriodicJitterResult with frequency/amplitude pairs.

    Example:
        >>> pj = extract_pj(tie_data, sample_rate=1e6)
        >>> for freq, amp in pj.components:
        ...     print(f"{freq/1e3:.1f} kHz: {amp*1e12:.2f} ps")

    References:
        IEEE 2414-2020 Section 6.4
    """
    valid_data = tie_data[~np.isnan(tie_data)]
    if len(valid_data) < 32:
        return _empty_pj_result()

    # Compute spectrum
    frequencies, magnitudes = _compute_pj_spectrum(valid_data, sample_rate)

    # Filter frequency range
    max_frequency = max_frequency or sample_rate / 2
    valid_freqs, valid_mags = _filter_frequency_range(
        frequencies, magnitudes, min_frequency, max_frequency
    )

    if len(valid_mags) < 3:
        return _empty_pj_result()

    # Extract peaks and create result
    components = _extract_pj_peaks(valid_freqs, valid_mags, n_components)
    return _create_pj_result(components)


def _empty_pj_result() -> PeriodicJitterResult:
    """Create empty PJ result for insufficient data."""
    return PeriodicJitterResult(
        components=[], pj_pp=0.0, dominant_frequency=None, dominant_amplitude=None
    )


def _compute_pj_spectrum(
    data: NDArray[np.float64], sample_rate: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute FFT spectrum of TIE data."""
    n = len(data)
    # Remove DC and apply window
    data_centered = data - np.mean(data)
    window = np.hanning(n)
    data_windowed = data_centered * window

    # Compute FFT
    nfft = int(2 ** np.ceil(np.log2(n)))
    spectrum = np.fft.rfft(data_windowed, n=nfft)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    magnitudes = np.abs(spectrum) * 2 / n  # Scale for amplitude
    return frequencies, magnitudes


def _filter_frequency_range(
    frequencies: NDArray[np.float64],
    magnitudes: NDArray[np.float64],
    min_freq: float,
    max_freq: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filter spectrum to valid frequency range."""
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    return frequencies[freq_mask], magnitudes[freq_mask]


def _extract_pj_peaks(
    frequencies: NDArray[np.float64], magnitudes: NDArray[np.float64], n_components: int
) -> list[tuple[float, float]]:
    """Extract top N periodic jitter peaks from spectrum."""
    from scipy.signal import find_peaks

    threshold = 3 * np.median(magnitudes)
    peak_indices, _ = find_peaks(magnitudes, height=threshold, distance=3)

    if len(peak_indices) == 0:
        return []

    # Sort by amplitude and take top n_components
    peak_heights = magnitudes[peak_indices]
    sorted_indices = np.argsort(peak_heights)[::-1][:n_components]
    top_peaks = peak_indices[sorted_indices]
    return [(float(frequencies[idx]), float(magnitudes[idx])) for idx in top_peaks]


def _create_pj_result(components: list[tuple[float, float]]) -> PeriodicJitterResult:
    """Create PJ result from extracted components."""
    if not components:
        return _empty_pj_result()

    pj_pp = 2 * sum(amp for _, amp in components)
    dominant_frequency = components[0][0]
    dominant_amplitude = components[0][1]

    return PeriodicJitterResult(
        components=components,
        pj_pp=pj_pp,
        dominant_frequency=dominant_frequency,
        dominant_amplitude=dominant_amplitude,
    )


def extract_ddj(
    tie_data: NDArray[np.float64],
    bit_pattern: NDArray[np.int_] | None = None,
    *,
    pattern_length: int = 3,
) -> DataDependentJitterResult:
    """Extract data-dependent jitter caused by ISI effects.

    Analyzes correlation between jitter and preceding bit patterns
    to identify inter-symbol interference (ISI) induced timing variations.

    Args:
        tie_data: Time Interval Error data array in seconds.
        bit_pattern: Associated bit pattern for each TIE sample.
        pattern_length: Number of preceding bits to correlate.

    Returns:
        DataDependentJitterResult with pattern-correlated jitter.

    Raises:
        ValueError: If bit_pattern length does not match tie_data length.

    Example:
        >>> ddj = extract_ddj(tie_data, bit_pattern=bits, pattern_length=3)
        >>> print(f"DDJ: {ddj.ddj_pp * 1e12:.2f} ps")

    References:
        IEEE 2414-2020 Section 6.5
    """
    valid_data = tie_data[~np.isnan(tie_data)]
    n = len(valid_data)

    if bit_pattern is None:
        # Without bit pattern data, estimate from TIE distribution
        # Use alternating pattern assumption
        pattern_histogram: dict[str, float] = {}

        # Simple estimation: look for bimodality in TIE
        median = np.median(valid_data)
        above_median = valid_data > median
        below_median = ~above_median

        pattern_histogram["above_median"] = float(np.mean(valid_data[above_median]))
        pattern_histogram["below_median"] = float(np.mean(valid_data[below_median]))

        ddj_pp = abs(pattern_histogram["above_median"] - pattern_histogram["below_median"])

        return DataDependentJitterResult(
            ddj_pp=ddj_pp,
            pattern_histogram=pattern_histogram,
            pattern_length=pattern_length,
            isi_coefficient=0.0,  # Unknown without pattern
        )

    # With bit pattern available
    if len(bit_pattern) != n:
        raise ValueError("bit_pattern length must match tie_data length")

    pattern_histogram = {}
    2**pattern_length

    # Create pattern strings and accumulate TIE values
    for i in range(pattern_length - 1, n):
        pattern_bits = bit_pattern[i - pattern_length + 1 : i + 1]
        pattern_str = "".join(str(int(b)) for b in pattern_bits)

        if pattern_str not in pattern_histogram:
            pattern_histogram[pattern_str] = []  # type: ignore[assignment]
        pattern_histogram[pattern_str].append(valid_data[i])  # type: ignore[attr-defined]

    # Calculate mean TIE for each pattern
    pattern_means: dict[str, float] = {}
    for pattern, values in pattern_histogram.items():
        if len(values) > 0:  # type: ignore[arg-type]
            pattern_means[pattern] = float(np.mean(values))

    # DDJ is the range of pattern-dependent means
    if len(pattern_means) > 1:
        mean_values = list(pattern_means.values())
        ddj_pp = max(mean_values) - min(mean_values)
    else:
        ddj_pp = 0.0

    # Calculate ISI correlation coefficient
    # Correlation between previous bit and current TIE
    if n > 1:
        prev_bits = bit_pattern[:-1].astype(float)
        curr_tie = valid_data[1:]
        correlation = np.corrcoef(prev_bits, curr_tie)[0, 1]
        isi_coefficient = correlation if np.isfinite(correlation) else 0.0
    else:
        isi_coefficient = 0.0

    return DataDependentJitterResult(
        ddj_pp=ddj_pp,
        pattern_histogram=pattern_means,
        pattern_length=pattern_length,
        isi_coefficient=isi_coefficient,
    )


def decompose_jitter(
    tie_data: NDArray[np.float64],
    *,
    edge_rate: float | None = None,
    include_pj: bool = True,
    include_ddj: bool = False,
    bit_pattern: NDArray[np.int_] | None = None,
    target_ber: float = 1e-12,
) -> JitterDecomposition:
    """Perform complete jitter decomposition.

    Decomposes total jitter into its constituent components:
    - Random Jitter (RJ): Unbounded Gaussian component
    - Deterministic Jitter (DJ): Bounded, repeatable component
    - Periodic Jitter (PJ): Sinusoidal components (optional)
    - Data-Dependent Jitter (DDJ): ISI-related component (optional)

    Args:
        tie_data: Time Interval Error data array in seconds.
        edge_rate: Rate of edges in Hz (required for PJ analysis).
        include_pj: Include periodic jitter analysis.
        include_ddj: Include data-dependent jitter analysis.
        bit_pattern: Bit pattern for DDJ analysis.
        target_ber: Target bit error rate for TJ calculation (default: 1e-12).

    Returns:
        JitterDecomposition with all component results and calculated TJ.

    Example:
        >>> decomp = decompose_jitter(tie_data, edge_rate=1e9)
        >>> print(f"RJ: {decomp.rj.rj_rms * 1e12:.2f} ps")
        >>> print(f"DJ: {decomp.dj.dj_pp * 1e12:.2f} ps")
        >>> print(f"TJ: {decomp.tj_pp * 1e12:.2f} ps")

    References:
        IEEE 2414-2020 Section 6
        Dual-Dirac model: TJ = DJ + 2*Q*RJ where Q = norm.ppf(1 - BER/2)
    """
    # Extract RJ first
    rj_result = extract_rj(tie_data)

    # Extract DJ using RJ result
    dj_result = extract_dj(tie_data, rj_result)

    # Optional: Extract PJ
    pj_result = None
    if include_pj and edge_rate is not None:
        pj_result = extract_pj(tie_data, edge_rate)

    # Optional: Extract DDJ
    ddj_result = None
    if include_ddj:
        ddj_result = extract_ddj(tie_data, bit_pattern)

    # Calculate Total Jitter using dual-Dirac model
    # TJ = DJ + 2 * Q * RJ
    # where Q is the Q-factor for the target BER
    q_factor = stats.norm.ppf(1 - target_ber / 2)
    tj_pp = dj_result.dj_pp + 2 * q_factor * rj_result.rj_rms

    return JitterDecomposition(
        rj=rj_result,
        dj=dj_result,
        pj=pj_result,
        ddj=ddj_result,
        tj_pp=tj_pp,
        ber_measured=target_ber,
    )


__all__ = [
    "DataDependentJitterResult",
    "DeterministicJitterResult",
    "JitterDecomposition",
    "PeriodicJitterResult",
    "RandomJitterResult",
    "decompose_jitter",
    "extract_ddj",
    "extract_dj",
    "extract_pj",
    "extract_rj",
]
