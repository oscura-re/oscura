"""Filter introspection and visualization for Oscura.

Provides filter analysis tools including Bode plots, impulse response,
step response, and pole-zero diagrams.


Example:
    >>> from oscura.utils.filtering import LowPassFilter, plot_bode
    >>> filt = LowPassFilter(cutoff=1e6, sample_rate=10e6, order=4)
    >>> fig = plot_bode(filt)
    >>> plt.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from oscura.utils.filtering.base import Filter, IIRFilter


class FilterIntrospection:
    """Mixin class providing filter introspection methods.

    Provides methods for analyzing filter characteristics including
    frequency response, impulse response, step response, and stability.
    """

    def __init__(self, filter_obj: Filter) -> None:
        """Initialize with a filter object.

        Args:
            filter_obj: Filter to introspect.
        """
        self._filter = filter_obj

    @property
    def filter(self) -> Filter:
        """The wrapped filter object.

        Returns:
            The filter being introspected.
        """
        return self._filter

    def magnitude_response(
        self,
        freqs: NDArray[np.float64] | None = None,
        db: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get magnitude response.

        Args:
            freqs: Frequencies in Hz. If None, auto-generate.
            db: If True, return magnitude in dB.

        Returns:
            Tuple of (frequencies, magnitude).

        Raises:
            ValueError: If freqs is None and filter has no sample_rate.
        """
        if freqs is None:
            if self._filter.sample_rate is None:
                raise ValueError(
                    "Either freqs must be provided or filter must have sample_rate set"
                )
            freqs = np.linspace(0, self._filter.sample_rate / 2, 512)

        h = self._filter.get_transfer_function(freqs)
        mag = np.abs(h)

        if db:
            mag = 20 * np.log10(np.maximum(mag, 1e-12))

        return freqs, mag

    def phase_response(
        self,
        freqs: NDArray[np.float64] | None = None,
        unwrap: bool = True,
        degrees: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get phase response.

        Args:
            freqs: Frequencies in Hz. If None, auto-generate.
            unwrap: If True, unwrap phase to remove discontinuities.
            degrees: If True, return phase in degrees.

        Returns:
            Tuple of (frequencies, phase).

        Raises:
            ValueError: If freqs is None and filter has no sample_rate.
        """
        if freqs is None:
            if self._filter.sample_rate is None:
                raise ValueError(
                    "Either freqs must be provided or filter must have sample_rate set"
                )
            freqs = np.linspace(0, self._filter.sample_rate / 2, 512)

        h = self._filter.get_transfer_function(freqs)
        phase = np.angle(h)

        if unwrap:
            phase = np.unwrap(phase)

        if degrees:
            phase = np.degrees(phase)

        return freqs, phase

    def group_delay_hz(
        self,
        freqs: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get group delay in seconds.

        Args:
            freqs: Frequencies in Hz. If None, auto-generate.

        Returns:
            Tuple of (frequencies in Hz, group delay in seconds).
        """
        w, gd_samples = self._filter.get_group_delay()

        if self._filter.sample_rate is not None:
            freqs_out = w * self._filter.sample_rate / (2 * np.pi)
            gd_seconds = gd_samples / self._filter.sample_rate
        else:
            freqs_out = w
            gd_seconds = gd_samples

        return freqs_out, gd_seconds

    def passband_ripple(
        self,
        passband_edge: float,
    ) -> float:
        """Calculate passband ripple in dB.

        Args:
            passband_edge: Passband edge frequency in Hz.

        Returns:
            Peak-to-peak ripple in dB within passband.

        Raises:
            ValueError: If filter sample_rate is not set.
        """
        if self._filter.sample_rate is None:
            raise ValueError("Sample rate must be set")

        freqs = np.linspace(0, passband_edge, 256)
        _, mag_db = self.magnitude_response(freqs, db=True)

        return float(np.max(mag_db) - np.min(mag_db))

    def stopband_attenuation(
        self,
        stopband_edge: float,
    ) -> float:
        """Calculate minimum stopband attenuation in dB.

        Args:
            stopband_edge: Stopband edge frequency in Hz.

        Returns:
            Minimum attenuation in stopband in dB (positive value).

        Raises:
            ValueError: If filter sample_rate is not set.
        """
        if self._filter.sample_rate is None:
            raise ValueError("Sample rate must be set")

        freqs = np.linspace(stopband_edge, self._filter.sample_rate / 2, 256)
        _, mag_db = self.magnitude_response(freqs, db=True)

        return float(-np.max(mag_db))

    def cutoff_frequency(
        self,
        threshold_db: float = -3.0,
    ) -> float:
        """Find -3dB cutoff frequency.

        Args:
            threshold_db: Threshold in dB (default -3dB).

        Returns:
            Cutoff frequency in Hz.

        Raises:
            ValueError: If filter sample_rate is not set.
        """
        if self._filter.sample_rate is None:
            raise ValueError("Sample rate must be set")

        freqs = np.linspace(0, self._filter.sample_rate / 2, 1000)
        _, mag_db = self.magnitude_response(freqs, db=True)

        # Normalize to 0dB at DC
        mag_db = mag_db - mag_db[0]

        # Find first crossing of threshold
        crossings = np.where(mag_db < threshold_db)[0]
        if len(crossings) == 0:
            return float(freqs[-1])

        return float(freqs[crossings[0]])


def plot_bode(
    filt: Filter,
    *,
    figsize: tuple[float, float] = (10, 8),
    freq_range: tuple[float, float] | None = None,
    n_points: int = 512,
    title: str | None = None,
) -> Figure:
    """Plot Bode diagram (magnitude and phase response).

    Args:
        filt: Filter to plot.
        figsize: Figure size in inches.
        freq_range: Frequency range (min, max) in Hz. None for auto.
        n_points: Number of frequency points.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If filter sample_rate is not set.

    Example:
        >>> fig = plot_bode(filt)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt

    if filt.sample_rate is None:
        raise ValueError("Filter sample rate must be set for plotting")

    if freq_range is None:
        freq_range = (1, filt.sample_rate / 2)

    freqs = np.geomspace(freq_range[0], freq_range[1], n_points)

    introspect = FilterIntrospection(filt)
    _, mag_db = introspect.magnitude_response(freqs, db=True)
    _, phase_deg = introspect.phase_response(freqs, degrees=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Magnitude plot
    ax1.semilogx(freqs, mag_db)
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.axhline(-3, color="r", linestyle="--", alpha=0.5, label="-3 dB")
    ax1.legend()

    # Phase plot
    ax2.semilogx(freqs, phase_deg)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True, which="both", alpha=0.3)

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"Bode Plot - Order {filt.order} Filter")

    plt.tight_layout()
    return fig


def plot_impulse(
    filt: Filter,
    *,
    n_samples: int = 256,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
) -> Figure:
    """Plot impulse response.

    Args:
        filt: Filter to plot.
        n_samples: Number of samples in response.
        figsize: Figure size in inches.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    impulse = filt.get_impulse_response(n_samples)

    fig, ax = plt.subplots(figsize=figsize)

    if filt.sample_rate is not None:
        t = np.arange(n_samples) / filt.sample_rate * 1e6  # microseconds
        ax.plot(t, impulse)
        ax.set_xlabel("Time (us)")
    else:
        ax.plot(impulse)
        ax.set_xlabel("Samples")

    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Impulse Response")

    plt.tight_layout()
    return fig


def plot_step(
    filt: Filter,
    *,
    n_samples: int = 256,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
) -> Figure:
    """Plot step response.

    Args:
        filt: Filter to plot.
        n_samples: Number of samples in response.
        figsize: Figure size in inches.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    step = filt.get_step_response(n_samples)

    fig, ax = plt.subplots(figsize=figsize)

    if filt.sample_rate is not None:
        t = np.arange(n_samples) / filt.sample_rate * 1e6  # microseconds
        ax.plot(t, step)
        ax.set_xlabel("Time (us)")
    else:
        ax.plot(step)
        ax.set_xlabel("Samples")

    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.axhline(1, color="r", linestyle="--", alpha=0.5, label="Final value")
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Step Response")

    plt.tight_layout()
    return fig


def plot_poles_zeros(
    filt: Filter,
    *,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
) -> Figure:
    """Plot pole-zero diagram for IIR filter.

    Args:
        filt: IIR filter to plot.
        figsize: Figure size in inches.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If filter is not an IIRFilter.
    """
    import matplotlib.pyplot as plt

    if not isinstance(filt, IIRFilter):
        raise ValueError("Pole-zero plot only available for IIR filters")

    poles = filt.poles
    zeros = filt.zeros

    fig, ax = plt.subplots(figsize=figsize)

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")

    # Plot poles and zeros
    ax.scatter(
        np.real(zeros),
        np.imag(zeros),
        marker="o",
        s=100,
        facecolors="none",
        edgecolors="b",
        linewidths=2,
        label="Zeros",
    )
    ax.scatter(
        np.real(poles),
        np.imag(poles),
        marker="x",
        s=100,
        c="r",
        linewidths=2,
        label="Poles",
    )

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Stability indicator
    is_stable = np.all(np.abs(poles) < 1.0)
    stability_text = "STABLE" if is_stable else "UNSTABLE"
    stability_color = "green" if is_stable else "red"
    ax.text(
        0.95,
        0.95,
        stability_text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=stability_color,
        ha="right",
        va="top",
    )

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Pole-Zero Plot (Order {filt.order})")

    plt.tight_layout()
    return fig


def plot_group_delay(
    filt: Filter,
    *,
    figsize: tuple[float, float] = (10, 4),
    freq_range: tuple[float, float] | None = None,
    n_points: int = 512,
    title: str | None = None,
) -> Figure:
    """Plot group delay.

    Args:
        filt: Filter to plot.
        figsize: Figure size in inches.
        freq_range: Frequency range (min, max) in Hz.
        n_points: Number of frequency points.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If filter sample_rate is not set.
    """
    import matplotlib.pyplot as plt

    if filt.sample_rate is None:
        raise ValueError("Filter sample rate must be set for plotting")

    introspect = FilterIntrospection(filt)
    freqs, gd = introspect.group_delay_hz()

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogx(freqs, gd * 1e6)  # Convert to microseconds
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group Delay (us)")
    ax.grid(True, which="both", alpha=0.3)

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Group Delay")

    plt.tight_layout()
    return fig


def compare_filters(
    filters: list[Filter],
    labels: list[str] | None = None,
    *,
    figsize: tuple[float, float] = (12, 10),
    freq_range: tuple[float, float] | None = None,
    n_points: int = 512,
) -> Figure:
    """Compare multiple filters on the same plots.

    Args:
        filters: List of filters to compare.
        labels: Labels for each filter. If None, uses "Filter 1", etc.
        figsize: Figure size in inches.
        freq_range: Frequency range (min, max) in Hz.
        n_points: Number of frequency points.

    Returns:
        Matplotlib Figure object with comparison plots.

    Raises:
        ValueError: If number of labels doesn't match number of filters or if filter sample_rate is not set.
    """
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [f"Filter {i + 1}" for i in range(len(filters))]

    if len(labels) != len(filters):
        raise ValueError("Number of labels must match number of filters")

    # Use first filter's sample rate for frequency axis
    sample_rate = filters[0].sample_rate
    if sample_rate is None:
        raise ValueError("Filter sample rate must be set for plotting")

    if freq_range is None:
        freq_range = (1, sample_rate / 2)

    freqs = np.geomspace(freq_range[0], freq_range[1], n_points)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for filt, label in zip(filters, labels, strict=False):
        introspect = FilterIntrospection(filt)
        _, mag_db = introspect.magnitude_response(freqs, db=True)
        _, phase_deg = introspect.phase_response(freqs, degrees=True)
        impulse = filt.get_impulse_response(256)
        step = filt.get_step_response(256)

        # Magnitude
        axes[0, 0].semilogx(freqs, mag_db, label=label)
        # Phase
        axes[0, 1].semilogx(freqs, phase_deg, label=label)
        # Impulse
        axes[1, 0].plot(impulse, label=label)
        # Step
        axes[1, 1].plot(step, label=label)

    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].set_title("Magnitude Response")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].axhline(-3, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_ylabel("Phase (degrees)")
    axes[0, 1].set_title("Phase Response")
    axes[0, 1].grid(True, which="both", alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_xlabel("Samples")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_title("Impulse Response")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_xlabel("Samples")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].set_title("Step Response")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(1, color="k", linestyle="--", alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle("Filter Comparison")
    plt.tight_layout()
    return fig


__all__ = [
    "FilterIntrospection",
    "compare_filters",
    "plot_bode",
    "plot_group_delay",
    "plot_impulse",
    "plot_poles_zeros",
    "plot_step",
]
