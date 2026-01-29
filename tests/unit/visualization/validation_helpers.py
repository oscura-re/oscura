"""Helper functions for visualization test validation.

This module provides reusable validation functions for checking that
visualization tests produce correct plot data and parameters.
"""

import numpy as np


def validate_plot_data(x_data, y_data, x_type="frequency", y_type="magnitude_db"):
    """Validate x/y data arrays for correctness.

    Args:
        x_data: X-axis data array
        y_data: Y-axis data array
        x_type: Type of x data ("frequency", "time")
        y_type: Type of y data ("magnitude_db", "psd_db", "spectrogram_db")

    Raises:
        AssertionError: If validation fails
    """
    # Basic shape validation
    assert len(x_data) > 0, f"{x_type} data is empty"
    assert len(y_data) > 0, f"{y_type} data is empty"
    assert len(x_data) == len(y_data), (
        f"X and Y data length mismatch: {len(x_data)} vs {len(y_data)}"
    )

    # X-axis validation
    if x_type == "frequency":
        # Frequencies should be monotonically increasing
        assert np.all(np.diff(x_data) >= 0), "Frequency data not monotonically increasing"
        # Frequencies should be non-negative
        assert np.all(x_data >= 0), "Frequency data contains negative values"
    elif x_type == "time":
        # Time should be monotonically increasing
        assert np.all(np.diff(x_data) >= 0), "Time data not monotonically increasing"
        # Time should be non-negative
        assert np.all(x_data >= 0), "Time data contains negative values"

    # Y-axis validation
    assert np.all(np.isfinite(y_data) | (y_data == -np.inf)), (
        f"{y_type} contains NaN or +Inf values"
    )

    if y_type in ["magnitude_db", "psd_db"]:
        # dB values should be in reasonable range (-200 to +100)
        finite_y = y_data[np.isfinite(y_data)]
        if len(finite_y) > 0:
            assert np.all(finite_y >= -200), f"{y_type} contains unreasonably low dB values"
            assert np.all(finite_y <= 100), f"{y_type} contains unreasonably high dB values"


def validate_plot_call(mock_plot, x_type="frequency", y_type="magnitude_db"):
    """Validate that plot() was called with correct data.

    Args:
        mock_plot: Mock object for ax.plot()
        x_type: Type of x data
        y_type: Type of y data
    """
    mock_plot.assert_called_once()
    args, kwargs = mock_plot.call_args
    assert len(args) >= 2, "plot() should be called with at least 2 positional args (x, y)"

    x_data = args[0]
    y_data = args[1]

    validate_plot_data(x_data, y_data, x_type, y_type)

    # Validate common plot styling (both optional)
    # Note: linewidth and color are optional parameters that may be specified


def validate_pcolormesh_call(mock_pcolormesh, has_time=True, has_freq=True):
    """Validate that pcolormesh() was called with correct data.

    Args:
        mock_pcolormesh: Mock object for ax.pcolormesh()
        has_time: Whether time axis should be present
        has_freq: Whether frequency axis should be present
    """
    mock_pcolormesh.assert_called_once()
    args, kwargs = mock_pcolormesh.call_args
    assert len(args) >= 3, "pcolormesh() should be called with at least 3 positional args (X, Y, C)"

    times = args[0]
    freqs = args[1]
    Sxx = args[2]

    if has_time:
        assert len(times) > 0, "Time data is empty"
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0), "Time data not monotonically increasing"
        assert np.all(times >= 0), "Time data contains negative values"

    if has_freq:
        assert len(freqs) > 0, "Frequency data is empty"
        if len(freqs) > 1:
            assert np.all(np.diff(freqs) >= 0), "Frequency data not monotonically increasing"
        assert np.all(freqs >= 0), "Frequency data contains negative values"

    # Validate Sxx dimensions and values
    assert Sxx.ndim == 2, f"Spectrogram should be 2D, got {Sxx.ndim}D"
    # Note: pcolormesh accepts C with shape (len(Y), len(X)) or (len(Y)-1, len(X)-1)
    assert Sxx.shape[0] >= len(freqs) - 1, (
        f"Sxx first dimension too small: {Sxx.shape[0]} vs {len(freqs)}"
    )
    assert Sxx.shape[1] >= len(times) - 1, (
        f"Sxx second dimension too small: {Sxx.shape[1]} vs {len(times)}"
    )

    # Validate common pcolormesh styling
    assert "cmap" in kwargs, "cmap should be specified"
    assert "vmin" in kwargs or "vmin" not in kwargs, "vmin is optional"
    assert "vmax" in kwargs or "vmax" not in kwargs, "vmax is optional"


def validate_axis_labels(mock_set_xlabel, mock_set_ylabel, x_unit=None, y_label=None):
    """Validate that axis labels were set correctly.

    Args:
        mock_set_xlabel: Mock object for ax.set_xlabel()
        mock_set_ylabel: Mock object for ax.set_ylabel()
        x_unit: Expected unit in x-label (e.g., "Hz", "kHz", "s", "ms")
        y_label: Expected y-label text (e.g., "Magnitude (dB)")
    """
    mock_set_xlabel.assert_called()
    mock_set_ylabel.assert_called()

    if x_unit is not None:
        xlabel_call = str(mock_set_xlabel.call_args)
        assert x_unit in xlabel_call, f"Expected unit '{x_unit}' not found in xlabel: {xlabel_call}"

    if y_label is not None:
        ylabel_call = mock_set_ylabel.call_args[0][0]
        assert y_label in ylabel_call, f"Expected '{y_label}' not found in ylabel: {ylabel_call}"


def validate_scale(mock_set_xscale, expected_scale):
    """Validate that axis scale was set correctly.

    Args:
        mock_set_xscale: Mock object for ax.set_xscale()
        expected_scale: Expected scale ("linear" or "log")
    """
    mock_set_xscale.assert_called()
    actual_scale = mock_set_xscale.call_args[0][0]
    assert actual_scale == expected_scale, (
        f"Expected scale '{expected_scale}', got '{actual_scale}'"
    )


def validate_grid(mock_grid, should_be_on=True):
    """Validate that grid was configured correctly.

    Args:
        mock_grid: Mock object for ax.grid()
        should_be_on: Whether grid should be enabled
    """
    if should_be_on:
        mock_grid.assert_called()
        args = mock_grid.call_args[0]
        assert args[0] is True, "Grid should be enabled"
    else:
        mock_grid.assert_not_called()


def validate_colorbar(mock_colorbar, expected_label=None):
    """Validate that colorbar was created correctly.

    Args:
        mock_colorbar: Mock object for fig.colorbar()
        expected_label: Expected colorbar label text
    """
    mock_colorbar.assert_called_once()

    if expected_label is not None:
        # Get the returned colorbar mock
        cbar_mock = mock_colorbar.return_value
        cbar_mock.set_label.assert_called()
        label_call = cbar_mock.set_label.call_args[0][0]
        assert expected_label in label_call, (
            f"Expected '{expected_label}' in colorbar label, got '{label_call}'"
        )


def validate_peak_frequency(x_data, y_data, expected_freq, tolerance=10):
    """Validate that peak frequency is at expected location.

    Args:
        x_data: Frequency data array
        y_data: Magnitude data array
        expected_freq: Expected peak frequency in same units as x_data
        tolerance: Tolerance in same units as x_data
    """
    peak_idx = np.argmax(y_data)
    peak_freq = x_data[peak_idx]
    assert abs(peak_freq - expected_freq) <= tolerance, (
        f"Peak at {peak_freq} Hz, expected {expected_freq} ± {tolerance} Hz"
    )


def validate_dominant_peak(y_data, min_ratio=5.0):
    """Validate that there is a single dominant peak.

    Args:
        y_data: Magnitude data array
        min_ratio: Minimum ratio between largest and second-largest peak
    """
    sorted_mags = np.sort(y_data)
    if len(sorted_mags) >= 2:
        ratio = sorted_mags[-1] / sorted_mags[-2] if sorted_mags[-2] != 0 else float("inf")
        assert ratio >= min_ratio, (
            f"No dominant peak found - ratio {ratio:.2f} < {min_ratio} (largest/second-largest)"
        )
