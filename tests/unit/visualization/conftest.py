"""Visualization-specific test fixtures.

This module provides fixtures for visualization tests:
- Matplotlib configuration fixtures
- Plotting helper fixtures
- Figure comparison fixtures
- Interactive plot fixtures
- Accessibility fixtures

NOTE: Matplotlib cleanup is handled by the root conftest.py
via the cleanup_matplotlib fixture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# Try to import matplotlib - these fixtures only work when matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    matplotlib = None  # type: ignore[assignment]

# Note: Visualization tests will be skipped individually via @pytest.mark.skipif
# decorators if matplotlib is not available. We don't skip at module level to
# allow other unit tests to run.


# =============================================================================
# Matplotlib Configuration Fixtures
# =============================================================================


@pytest.fixture
def plot_config() -> dict[str, Any]:
    """Common plotting configuration for tests.

    Returns:
        Dictionary with figsize, dpi, and style settings.
    """
    return {
        "figsize": (10, 6),
        "dpi": 100,
        "style": "default",
        "backend": "Agg",
    }


@pytest.fixture
def plot_style_params() -> dict[str, Any]:
    """Matplotlib style parameters for consistent test plots.

    Returns:
        Dictionary with rcParams settings.
    """
    return {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    }


@pytest.fixture(autouse=True)
def reset_matplotlib_rc():
    """Reset matplotlib rcParams after each test.

    This ensures test isolation by restoring default settings.
    """
    import matplotlib.pyplot as plt

    # Save original rcParams
    original_params = plt.rcParams.copy()

    yield

    # Restore original rcParams
    plt.rcParams.update(original_params)


# =============================================================================
# Figure and Axes Fixtures
# =============================================================================


@pytest.fixture
def empty_figure():
    """Create an empty matplotlib figure.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6), dpi=100)
    return fig


@pytest.fixture
def figure_with_axes():
    """Create figure with single axes.

    Returns:
        Tuple of (figure, axes).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    return fig, ax


@pytest.fixture
def figure_with_subplots():
    """Create figure with 2x2 subplot grid.

    Returns:
        Tuple of (figure, axes_array).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    return fig, axes


# =============================================================================
# Test Data for Plotting
# =============================================================================
# MIGRATION COMPLETE: Signal generation fixtures removed.
# Tests should use SignalBuilder from tests/fixtures/signal_builders.py instead.
#
# Example usage:
#   def test_plot(signal_builder):
#       signal = signal_builder.multitone([10, 25], sample_rate=1000)
#       clock = signal_builder.square_wave(frequency=1000, sample_rate=1e6)
#
# See tests/fixtures/signal_builders.py for 20+ signal generation methods.


# =============================================================================
# Color and Palette Fixtures
# =============================================================================


@pytest.fixture
def color_palettes() -> dict[str, list[str]]:
    """Color palettes for visualization tests.

    Returns:
        Dictionary mapping palette name to color list.
    """
    return {
        "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        "grayscale": ["#000000", "#555555", "#aaaaaa", "#ffffff"],
        "colorblind_safe": ["#0173b2", "#de8f05", "#029e73", "#cc78bc"],
        "high_contrast": ["#000000", "#ffffff", "#ff0000", "#00ff00"],
    }


@pytest.fixture
def accessibility_colors() -> dict[str, str]:
    """Accessibility-friendly color mappings.

    Returns:
        Dictionary with semantic color names.
    """
    return {
        "background": "#ffffff",
        "foreground": "#000000",
        "grid": "#cccccc",
        "trace1": "#0173b2",  # Blue
        "trace2": "#de8f05",  # Orange
        "trace3": "#029e73",  # Green
        "trace4": "#cc78bc",  # Purple
        "warning": "#f0ad4e",
        "error": "#d9534f",
        "success": "#5cb85c",
    }


# =============================================================================
# Annotation Fixtures
# =============================================================================


@pytest.fixture
def annotation_examples() -> list[dict[str, Any]]:
    """Example annotations for plot testing.

    Returns:
        List of annotation dictionaries.
    """
    return [
        {
            "type": "text",
            "x": 0.5,
            "y": 0.5,
            "text": "Test Annotation",
            "ha": "center",
            "va": "center",
        },
        {
            "type": "arrow",
            "x": 0.3,
            "y": 0.3,
            "dx": 0.1,
            "dy": 0.1,
            "text": "Arrow",
            "arrowprops": {"arrowstyle": "->"},
        },
        {
            "type": "vline",
            "x": 0.25,
            "color": "red",
            "linestyle": "--",
            "label": "Threshold",
        },
        {
            "type": "hline",
            "y": 0.75,
            "color": "green",
            "linestyle": "-.",
            "label": "Reference",
        },
    ]


# =============================================================================
# Axis Configuration Fixtures
# =============================================================================


@pytest.fixture
def time_axis_config() -> dict[str, Any]:
    """Time axis configuration for waveform plots.

    Returns:
        Dictionary with axis settings.
    """
    return {
        "xlabel": "Time",
        "ylabel": "Voltage (V)",
        "xlim": (0, 1e-3),  # 0 to 1 ms
        "ylim": (-1, 5),  # -1V to 5V
        "xscale": "linear",
        "yscale": "linear",
        "grid": True,
    }


@pytest.fixture
def frequency_axis_config() -> dict[str, Any]:
    """Frequency axis configuration for spectral plots.

    Returns:
        Dictionary with axis settings.
    """
    return {
        "xlabel": "Frequency (Hz)",
        "ylabel": "Magnitude (dB)",
        "xlim": (0, 1e6),  # 0 to 1 MHz
        "ylim": (-80, 0),  # -80 dB to 0 dB
        "xscale": "log",
        "yscale": "linear",
        "grid": True,
    }


# =============================================================================
# Interactive Plot Fixtures
# =============================================================================


@pytest.fixture
def zoom_window_config() -> dict[str, Any]:
    """Zoom window configuration for interactive plots.

    Returns:
        Dictionary with zoom settings.
    """
    return {
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.2,
        "border_color": "red",
        "border_width": 2,
    }


@pytest.fixture
def cursor_config() -> dict[str, Any]:
    """Cursor configuration for interactive measurements.

    Returns:
        Dictionary with cursor settings.
    """
    return {
        "cursor1_position": 0.3,
        "cursor2_position": 0.7,
        "color": "red",
        "linestyle": "--",
        "show_delta": True,
    }


# =============================================================================
# Rendering Quality Fixtures
# =============================================================================


@pytest.fixture
def rendering_tolerances() -> dict[str, float]:
    """Tolerances for rendering quality checks.

    Returns:
        Dictionary with tolerance values.
    """
    return {
        "position_tolerance": 0.01,  # 1% position error
        "color_tolerance": 5,  # RGB value difference
        "line_width_tolerance": 0.5,  # pixels
        "text_similarity": 0.95,  # String similarity
    }


# =============================================================================
# Export Format Fixtures
# =============================================================================


@pytest.fixture
def export_formats() -> list[str]:
    """Supported export formats for figure saving.

    Returns:
        List of file extensions.
    """
    return ["png", "pdf", "svg", "jpg"]


@pytest.fixture
def export_config() -> dict[str, Any]:
    """Export configuration for figure saving.

    Returns:
        Dictionary with export settings.
    """
    return {
        "dpi": 300,
        "bbox_inches": "tight",
        "transparent": False,
        "facecolor": "white",
        "edgecolor": "none",
    }


# =============================================================================
# Specialized Plot Fixtures
# =============================================================================


@pytest.fixture
def waterfall_data() -> NDArray[np.float64]:
    """Generate data for waterfall plot testing.

    Returns:
        2D array (time x frequency).
    """
    time_steps = 100
    freq_bins = 512

    # Generate time-varying spectrum
    data = np.zeros((time_steps, freq_bins))
    for i in range(time_steps):
        # Frequency content shifts over time
        center_freq = int(256 + 100 * np.sin(2 * np.pi * i / 50))
        data[i, center_freq - 20 : center_freq + 20] = np.random.uniform(0.5, 1.0, 40)

    return data


@pytest.fixture
def constellation_data() -> NDArray[np.complex128]:
    """Generate constellation diagram data.

    Returns:
        Complex-valued signal points.
    """
    # QPSK constellation
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

    # Generate random symbols with noise
    rng = np.random.default_rng(42)
    num_symbols = 1000
    symbols = constellation[rng.integers(0, 4, num_symbols)]

    # Add Gaussian noise
    noise = rng.normal(0, 0.1, num_symbols) + 1j * rng.normal(0, 0.1, num_symbols)
    return symbols + noise


@pytest.fixture
def persistence_data() -> list[NDArray[np.float64]]:
    """Generate data for persistence plot testing.

    Returns:
        List of waveform captures.
    """
    num_captures = 50
    samples = 1000
    t = np.linspace(0, 1, samples)

    captures = []
    rng = np.random.default_rng(42)
    for _ in range(num_captures):
        # Sine wave with varying amplitude and phase
        amplitude = rng.uniform(0.8, 1.2)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 0.05, samples)
        signal = amplitude * np.sin(2 * np.pi * 10 * t + phase) + noise
        captures.append(signal)

    return captures


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture
def large_plot_data() -> dict[str, NDArray[np.float64]]:
    """Generate large dataset for performance testing.

    Returns:
        Dictionary with time and signal arrays (1M samples).
    """
    samples = 1_000_000
    t = np.arange(samples) / 1e6
    signal = np.sin(2 * np.pi * 1000 * t)

    return {
        "time": t,
        "signal": signal,
    }


@pytest.fixture
def plot_performance_thresholds() -> dict[str, float]:
    """Performance thresholds for plotting operations.

    Returns:
        Dictionary with time limits in seconds.
    """
    return {
        "max_render_time": 1.0,  # Maximum time to render plot
        "max_save_time": 2.0,  # Maximum time to save figure
        "max_update_time": 0.1,  # Maximum time to update plot
        "max_memory_mb": 500,  # Maximum memory usage
    }


# =============================================================================
# Accessibility Testing Fixtures
# =============================================================================


@pytest.fixture
def accessibility_requirements() -> dict[str, Any]:
    """Accessibility requirements for plots.

    Returns:
        Dictionary with WCAG compliance settings.
    """
    return {
        "min_contrast_ratio": 4.5,  # WCAG AA standard
        "min_font_size": 9,  # Points
        "max_line_weight_ratio": 3.0,  # Thickest/thinnest
        "require_alt_text": True,
        "colorblind_safe": True,
    }


# =============================================================================
# Theme Fixtures
# =============================================================================


@pytest.fixture
def plot_themes() -> dict[str, dict[str, Any]]:
    """Plot themes for testing different styles.

    Returns:
        Dictionary mapping theme name to rcParams.
    """
    return {
        "light": {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": "black",
            "axes.edgecolor": "black",
            "grid.color": "#cccccc",
        },
        "dark": {
            "axes.facecolor": "#1e1e1e",
            "figure.facecolor": "#2d2d2d",
            "text.color": "white",
            "axes.edgecolor": "white",
            "grid.color": "#444444",
        },
        "high_contrast": {
            "axes.facecolor": "black",
            "figure.facecolor": "black",
            "text.color": "white",
            "axes.edgecolor": "white",
            "grid.color": "yellow",
        },
    }
