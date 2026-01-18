"""Spectral method auto-selection.

This module automatically selects appropriate spectral analysis methods
based on signal characteristics.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('signal.wfm')
    >>> config = osc.auto_spectral_config(trace)
    >>> print(f"Method: {config['method']}")
    >>> print(f"Window: {config['window']}")

References:
    Welch's method: IEEE Signal Processing Magazine (1986)
    Stationarity tests: Augmented Dickey-Fuller test
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


def auto_spectral_config(
    trace: WaveformTrace,
    *,
    target_resolution: float | None = None,
    dynamic_range_db: float = 60.0,
    log_rationale: bool = True,
) -> dict[str, Any]:
    """Auto-select spectral analysis method and parameters.

    Analyzes signal stationarity and characteristics to select:
    - Spectral method (Welch, Bartlett, Periodogram)
    - Window function (Hann, Hamming, Blackman, Kaiser)
    - Window size and overlap
    - FFT size

    Args:
        trace: Signal to analyze.
        target_resolution: Desired frequency resolution in Hz (optional).
        dynamic_range_db: Required dynamic range in dB (default 60 dB).
        log_rationale: If True, include selection rationale in result.

    Returns:
        Dictionary containing:
        - method: Recommended method ('welch', 'bartlett', 'periodogram')
        - window: Window function name ('hann', 'hamming', 'blackman', etc.)
        - nperseg: Segment length for Welch/Bartlett
        - noverlap: Overlap samples
        - nfft: FFT size
        - stationarity_score: Stationarity score (0-1, 1=stationary)
        - rationale: Explanation of selection (if log_rationale=True)

    Example:
        >>> trace = osc.load('noisy_signal.wfm')
        >>> config = osc.auto_spectral_config(trace, dynamic_range_db=80)
        >>> print(f"Method: {config['method']}")
        >>> print(f"Window: {config['window']}")
        >>> print(f"Rationale: {config['rationale']}")
        >>> # Use configuration
        >>> freq, psd = osc.psd(trace, **config)

    References:
        Welch, P. D. (1967): Use of FFT for estimation of power spectra
        Stoica & Moses (2005): Spectral Analysis of Signals
    """
    # Analyze signal stationarity
    stationarity_score = _assess_stationarity(trace.data)

    # Determine method based on stationarity
    if stationarity_score > 0.8:
        # Stationary - can use simpler methods
        method = "bartlett"
        rationale = f"Signal is stationary (score={stationarity_score:.2f}). Bartlett method provides good variance reduction."
    elif stationarity_score > 0.5:
        # Moderately stationary - Welch is safest
        method = "welch"
        rationale = f"Signal is moderately stationary (score={stationarity_score:.2f}). Welch method with overlap for variance reduction."
    else:
        # Non-stationary - need time-frequency analysis or careful windowing
        method = "welch"
        rationale = f"Signal is non-stationary (score={stationarity_score:.2f}). Welch method with short segments to track changes."

    # Select window function based on dynamic range requirements
    if dynamic_range_db > 80:
        window = "blackman-harris"
        window_rationale = "Blackman-Harris window for high dynamic range (>80 dB)"
    elif dynamic_range_db > 60:
        window = "blackman"
        window_rationale = "Blackman window for good dynamic range (60-80 dB)"
    elif dynamic_range_db > 40:
        window = "hamming"
        window_rationale = "Hamming window for moderate dynamic range (40-60 dB)"
    else:
        window = "hann"
        window_rationale = "Hann window for general purpose (<40 dB)"

    if log_rationale:
        rationale += " " + window_rationale

    # Determine segment size
    n_samples = len(trace.data)
    sample_rate = trace.metadata.sample_rate

    if target_resolution is not None:
        # User specified resolution
        nperseg = int(sample_rate / target_resolution)
        # Round to next power of 2 for efficiency
        nperseg = 2 ** int(np.ceil(np.log2(nperseg)))
    # Auto-select based on signal length and stationarity
    elif stationarity_score > 0.8:
        # Stationary - can use longer segments
        nperseg = min(n_samples, 2**14)  # Up to 16k samples
    else:
        # Non-stationary - shorter segments
        nperseg = min(n_samples // 8, 2**12)  # Up to 4k samples

    # Ensure nperseg is reasonable
    nperseg = max(256, min(nperseg, n_samples // 2))

    # Determine overlap
    if method == "welch":
        # Welch typically uses 50% overlap
        noverlap = nperseg // 2
    elif method == "bartlett":
        # Bartlett uses no overlap
        noverlap = 0
    else:
        # Periodogram doesn't use segments
        noverlap = 0

    # FFT size (usually same as segment size or next power of 2)
    nfft = 2 ** int(np.ceil(np.log2(nperseg)))

    config = {
        "method": method,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "stationarity_score": stationarity_score,
    }

    if log_rationale:
        config["rationale"] = rationale

    return config


def _assess_stationarity(data: NDArray[np.floating[Any]]) -> float:
    """Assess signal stationarity using windowed variance method.

    A simpler alternative to ADF test that works well for spectral analysis.

    Args:
        data: Signal data array.

    Returns:
        Stationarity score from 0 (non-stationary) to 1 (stationary).
    """
    # Divide signal into windows
    n_windows = 8
    window_size = len(data) // n_windows

    if window_size < 100:
        # Signal too short for meaningful analysis
        return 0.7  # Assume moderately stationary

    # Calculate statistics in each window
    means = []
    variances = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_data = data[start:end]

        means.append(np.mean(window_data))
        variances.append(np.var(window_data))

    # Stationarity check: mean and variance should be consistent
    # Use coefficient of variation, but handle near-zero means specially
    mean_abs = abs(np.mean(means))
    if mean_abs > 1e-6:
        mean_variation = np.std(means) / mean_abs
    else:
        # For zero-mean signals, use absolute variation
        mean_variation = np.std(means)

    var_mean = np.mean(variances)
    if var_mean > 1e-12:
        var_variation = np.std(variances) / var_mean
    else:
        # Near-zero variance signal (constant)
        var_variation = 0.0

    # Low variation = high stationarity
    mean_score = max(0, 1.0 - mean_variation * 5)
    var_score = max(0, 1.0 - var_variation * 5)

    # Combined score
    stationarity_score = (mean_score + var_score) / 2

    return float(np.clip(stationarity_score, 0, 1))


__all__ = ["auto_spectral_config"]
