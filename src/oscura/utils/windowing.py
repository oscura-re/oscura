"""Window function support for spectral analysis.

This module provides standard window functions for FFT and spectral
analysis, implementing the requirements for windowed spectral estimation.


Example:
    >>> from oscura.utils.windowing import get_window, WINDOW_FUNCTIONS
    >>> window = get_window("hann", 1024)
    >>> print(f"Available windows: {list(WINDOW_FUNCTIONS.keys())}")

References:
    Harris, F. J. (1978). "On the use of windows for harmonic analysis
    with the discrete Fourier transform." Proceedings of the IEEE, 66(1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for window function (using string annotation for TYPE_CHECKING compatibility)
WindowFunction = Callable[[int], "NDArray[np.float64]"]


def rectangular(n: int) -> NDArray[np.float64]:
    """Rectangular (boxcar) window.

    No tapering applied - all samples weighted equally.

    Args:
        n: Window length in samples.

    Returns:
        Window coefficients (all ones).

    Example:
        >>> w = rectangular(64)
        >>> assert np.all(w == 1.0)
    """
    return np.ones(n, dtype=np.float64)


def hann(n: int) -> NDArray[np.float64]:
    """Hann (raised cosine) window.

    Also known as Hanning window. Provides good frequency resolution
    with moderate sidelobe suppression.

    Args:
        n: Window length in samples.

    Returns:
        Window coefficients.

    Example:
        >>> w = hann(64)
        >>> assert w[0] == w[-1]  # Symmetric

    References:
        IEEE Std 1057-2017 Section 4.4.2
    """
    return np.hanning(n).astype(np.float64)


def hamming(n: int) -> NDArray[np.float64]:
    """Hamming window.

    Similar to Hann but with reduced first sidelobe at cost of
    slower rolloff.

    Args:
        n: Window length in samples.

    Returns:
        Window coefficients.

    Example:
        >>> w = hamming(64)
        >>> assert w[32] > w[0]  # Peak in center
    """
    return np.hamming(n).astype(np.float64)


def blackman(n: int) -> NDArray[np.float64]:
    """Blackman window.

    Three-term cosine window with excellent sidelobe suppression
    (-58 dB first sidelobe).

    Args:
        n: Window length in samples.

    Returns:
        Window coefficients.

    Example:
        >>> w = blackman(64)
        >>> assert w[32] > w[0]
    """
    return np.blackman(n).astype(np.float64)


def kaiser(n: int, beta: float = 8.6) -> NDArray[np.float64]:
    """Kaiser window with configurable shape parameter.

    Provides adjustable tradeoff between main lobe width and
    sidelobe attenuation.

    Args:
        n: Window length in samples.
        beta: Shape parameter (default 8.6 for ~60 dB sidelobe attenuation).
            - beta=0: Rectangular
            - beta=5: ~30 dB sidelobe attenuation
            - beta=8.6: ~60 dB sidelobe attenuation
            - beta=14: ~90 dB sidelobe attenuation

    Returns:
        Window coefficients.

    Example:
        >>> w = kaiser(64, beta=10)
        >>> assert 0 < w[0] < w[32]
    """
    return np.kaiser(n, beta).astype(np.float64)


def flattop(n: int) -> NDArray[np.float64]:
    """Flat-top window for accurate amplitude measurements.

    Provides minimal scalloping loss (<0.01 dB) at cost of
    wider main lobe. Best for amplitude accuracy when frequency
    resolution is not critical.

    Args:
        n: Window length in samples.

    Returns:
        Window coefficients.

    Example:
        >>> w = flattop(64)
        >>> # Flat-top has characteristic near-zero values at edges

    References:
        D'Antona, G. & Ferrero, A. (2006). "Digital Signal Processing
        for Measurement Systems."
    """
    # Flat-top coefficients per HP/Agilent standard
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368

    k = np.arange(n, dtype=np.float64)
    w = (
        a0
        - a1 * np.cos(2 * np.pi * k / (n - 1))
        + a2 * np.cos(4 * np.pi * k / (n - 1))
        - a3 * np.cos(6 * np.pi * k / (n - 1))
        + a4 * np.cos(8 * np.pi * k / (n - 1))
    )
    return np.asarray(w, dtype=np.float64)


def bartlett(n: int) -> NDArray[np.float64]:
    """Bartlett (triangular) window.

    Linear taper from zero at edges to maximum at center.

    Args:
        n: Window length in samples.

    Returns:
        Window coefficients.

    Example:
        >>> w = bartlett(64)
        >>> assert w[32] == 1.0  # Maximum at center
    """
    return np.bartlett(n).astype(np.float64)


def blackman_harris(n: int) -> NDArray[np.float64]:
    """Blackman-Harris window (4-term).

    Four-term cosine window with excellent sidelobe suppression
    (-92 dB first sidelobe).

    Args:
        n: Window length in samples.

    Returns:
        Window coefficients.

    Example:
        >>> w = blackman_harris(64)
        >>> assert w[32] > w[0]
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168

    k = np.arange(n, dtype=np.float64)
    w = (
        a0
        - a1 * np.cos(2 * np.pi * k / (n - 1))
        + a2 * np.cos(4 * np.pi * k / (n - 1))
        - a3 * np.cos(6 * np.pi * k / (n - 1))
    )
    return np.asarray(w, dtype=np.float64)


# Window function registry
WINDOW_FUNCTIONS: dict[str, WindowFunction] = {
    "rectangular": rectangular,
    "boxcar": rectangular,
    "rect": rectangular,
    "hann": hann,
    "hanning": hann,
    "hamming": hamming,
    "blackman": blackman,
    "kaiser": lambda n: kaiser(n, beta=8.6),
    "flattop": flattop,
    "flat_top": flattop,
    "bartlett": bartlett,
    "triangular": bartlett,
    "blackman_harris": blackman_harris,
    "blackmanharris": blackman_harris,
}


# Type for window names
WindowName = Literal[
    "rectangular",
    "boxcar",
    "rect",
    "hann",
    "hanning",
    "hamming",
    "blackman",
    "kaiser",
    "flattop",
    "flat_top",
    "bartlett",
    "triangular",
    "blackman_harris",
    "blackmanharris",
]


def get_window(
    window: str | WindowFunction | NDArray[np.floating[Any]],
    n: int,
    *,
    beta: float | None = None,
) -> NDArray[np.float64]:
    """Get window coefficients by name or callable.

    Args:
        window: Window specification. Can be:
            - A string name from WINDOW_FUNCTIONS
            - A callable that takes length and returns coefficients
            - A pre-computed array of coefficients
        n: Window length in samples.
        beta: Optional beta parameter for Kaiser window.

    Returns:
        Window coefficients array of length n.

    Raises:
        ValueError: If window name is unknown.

    Example:
        >>> w = get_window("hann", 1024)
        >>> w = get_window("kaiser", 1024, beta=10)
        >>> w = get_window(np.hamming, 1024)
    """
    if isinstance(window, np.ndarray):
        if len(window) != n:
            raise ValueError(f"Window array length {len(window)} != requested {n}")
        return window.astype(np.float64)

    if callable(window) and not isinstance(window, str):
        return np.asarray(window(n), dtype=np.float64)

    window_name = window.lower()

    if window_name == "kaiser" and beta is not None:
        return kaiser(n, beta)

    if window_name not in WINDOW_FUNCTIONS:
        available = ", ".join(sorted(set(WINDOW_FUNCTIONS.keys())))
        raise ValueError(f"Unknown window: {window}. Available: {available}")

    return WINDOW_FUNCTIONS[window_name](n)


def window_properties(window: str | NDArray[np.floating[Any]], n: int = 1024) -> dict[str, Any]:
    """Compute window properties for analysis.

    Args:
        window: Window name or coefficients.
        n: Window length for named windows.

    Returns:
        Dictionary with window properties:
            - coherent_gain: Sum of window / length
            - noise_bandwidth: Normalized equivalent noise bandwidth
            - scalloping_loss: Peak amplitude error in dB

    Example:
        >>> props = window_properties("hann")
        >>> print(f"ENBW: {props['noise_bandwidth']:.3f}")
    """
    if isinstance(window, str):
        w = get_window(window, n)
    else:
        w = np.asarray(window, dtype=np.float64)
        n = len(w)

    # Coherent gain (DC gain)
    coherent_gain = np.sum(w) / n

    # Noise equivalent bandwidth
    # ENBW = N * sum(w^2) / (sum(w))^2
    noise_bandwidth = n * np.sum(w**2) / np.sum(w) ** 2

    # Scalloping loss (worst-case amplitude error at bin edge)
    # Approximate by evaluating window at half-bin offset
    k = np.arange(n)
    w_shifted = w * np.exp(2j * np.pi * 0.5 * k / n)
    scalloping_loss = 20 * np.log10(np.abs(np.sum(w_shifted)) / np.abs(np.sum(w)))

    return {
        "coherent_gain": float(coherent_gain),
        "noise_bandwidth": float(noise_bandwidth),
        "scalloping_loss": float(scalloping_loss),
        "length": n,
    }


__all__ = [
    "WINDOW_FUNCTIONS",
    "bartlett",
    "blackman",
    "blackman_harris",
    "flattop",
    "get_window",
    "hamming",
    "hann",
    "kaiser",
    "rectangular",
    "window_properties",
]
