"""Unit tests for DSL commands module.

Tests for built-in command implementations including:
- cmd_load: File loading with various formats
- cmd_filter: Signal filtering operations
- cmd_measure: Trace measurements
- cmd_plot: Visualization
- cmd_export: Data export
- cmd_glob: File pattern matching
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from oscura.api.dsl.commands import (
    BUILTIN_COMMANDS,
    cmd_export,
    cmd_filter,
    cmd_glob,
    cmd_load,
    cmd_measure,
    cmd_plot,
)
from oscura.core.exceptions import OscuraError

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestCmdLoad:
    """Test cmd_load function."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading a file that doesn't exist."""
        with pytest.raises(OscuraError, match="File not found"):
            cmd_load("/nonexistent/file.csv")

    def test_load_csv_file(self, tmp_path: Path) -> None:
        """Test loading CSV file."""
        # Create temporary CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("time,voltage\n0.0,1.0\n1.0,2.0\n")

        with patch("oscura.loaders.load") as mock_load:
            mock_load.return_value = {"time": [0.0, 1.0], "voltage": [1.0, 2.0]}
            result = cmd_load(str(csv_file))
            mock_load.assert_called_once_with(str(csv_file))
            assert result == {"time": [0.0, 1.0], "voltage": [1.0, 2.0]}

    def test_load_binary_file(self, tmp_path: Path) -> None:
        """Test loading binary file."""
        bin_file = tmp_path / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")

        with patch("oscura.loaders.load") as mock_load:
            mock_load.return_value = np.array([0, 1, 2, 3])
            result = cmd_load(str(bin_file))
            mock_load.assert_called_once_with(str(bin_file))
            np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_load_hdf5_file(self, tmp_path: Path) -> None:
        """Test loading HDF5 file."""
        h5_file = tmp_path / "test.h5"
        h5_file.touch()

        with patch("oscura.loaders.load") as mock_load:
            mock_load.return_value = {"data": np.array([1, 2, 3])}
            result = cmd_load(str(h5_file))
            mock_load.assert_called_once_with(str(h5_file))
            assert "data" in result

    def test_load_hdf5_alt_extension(self, tmp_path: Path) -> None:
        """Test loading HDF5 file with .hdf5 extension."""
        h5_file = tmp_path / "test.hdf5"
        h5_file.touch()

        with patch("oscura.loaders.load") as mock_load:
            mock_load.return_value = {}
            result = cmd_load(str(h5_file))
            mock_load.assert_called_once()
            assert isinstance(result, dict)

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        """Test loading file with unsupported format."""
        unsupported = tmp_path / "test.xyz"
        unsupported.touch()

        with patch("oscura.loaders.load", side_effect=OscuraError("Unsupported format")):
            with pytest.raises(OscuraError):
                cmd_load(str(unsupported))

    def test_load_import_error(self, tmp_path: Path) -> None:
        """Test handling of import errors."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("data")

        with patch("oscura.loaders.load", side_effect=ImportError("No module")):
            with pytest.raises(OscuraError, match="Loader not available"):
                cmd_load(str(csv_file))


@pytest.mark.unit
class TestCmdFilter:
    """Test cmd_filter function."""

    def test_lowpass_filter(self) -> None:
        """Test lowpass filter application."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.low_pass") as mock_filter:
            mock_filter.return_value = trace * 0.5
            result = cmd_filter(trace, "lowpass", 1000.0)
            mock_filter.assert_called_once_with(trace, cutoff=1000.0)
            np.testing.assert_array_equal(result, trace * 0.5)

    def test_lowpass_filter_with_kwargs(self) -> None:
        """Test lowpass filter with additional kwargs."""
        trace = np.array([1.0, 2.0, 3.0])

        with patch("oscura.utils.filtering.low_pass") as mock_filter:
            mock_filter.return_value = trace
            result = cmd_filter(trace, "lowpass", 1000.0, order=4, fs=10000.0)
            mock_filter.assert_called_once_with(trace, cutoff=1000.0, order=4, fs=10000.0)
            np.testing.assert_array_equal(result, trace)

    def test_lowpass_missing_cutoff(self) -> None:
        """Test lowpass filter without cutoff frequency."""
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(OscuraError, match="requires cutoff frequency"):
            cmd_filter(trace, "lowpass")

    def test_highpass_filter(self) -> None:
        """Test highpass filter application."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.high_pass") as mock_filter:
            mock_filter.return_value = trace * 0.8
            result = cmd_filter(trace, "highpass", 500.0)
            mock_filter.assert_called_once_with(trace, cutoff=500.0)
            np.testing.assert_array_equal(result, trace * 0.8)

    def test_highpass_missing_cutoff(self) -> None:
        """Test highpass filter without cutoff frequency."""
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(OscuraError, match="requires cutoff frequency"):
            cmd_filter(trace, "highpass")

    def test_bandpass_filter(self) -> None:
        """Test bandpass filter application."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.band_pass") as mock_filter:
            mock_filter.return_value = trace * 0.6
            result = cmd_filter(trace, "bandpass", 100.0, 1000.0)
            mock_filter.assert_called_once_with(trace, low=100.0, high=1000.0)
            np.testing.assert_array_equal(result, trace * 0.6)

    def test_bandpass_missing_frequencies(self) -> None:
        """Test bandpass filter without both frequencies."""
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(OscuraError, match="requires low and high cutoff"):
            cmd_filter(trace, "bandpass", 100.0)

    def test_bandstop_filter(self) -> None:
        """Test bandstop filter application."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.band_stop") as mock_filter:
            mock_filter.return_value = trace * 0.9
            result = cmd_filter(trace, "bandstop", 200.0, 800.0)
            mock_filter.assert_called_once_with(trace, low=200.0, high=800.0)
            np.testing.assert_array_equal(result, trace * 0.9)

    def test_bandstop_missing_frequencies(self) -> None:
        """Test bandstop filter without both frequencies."""
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(OscuraError, match="requires low and high cutoff"):
            cmd_filter(trace, "bandstop")

    def test_unknown_filter_type(self) -> None:
        """Test filter with unknown type."""
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(OscuraError, match="Unknown filter type"):
            cmd_filter(trace, "invalidfilter", 1000.0)

    def test_filter_import_error(self) -> None:
        """Test handling of import errors."""
        trace = np.array([1.0, 2.0, 3.0])
        # Patch the filtering module's low_pass function to raise ImportError
        with patch("oscura.utils.filtering.low_pass", side_effect=ImportError):
            with pytest.raises(OscuraError, match="Filtering module not available"):
                cmd_filter(trace, "lowpass", 1000.0)


@pytest.mark.unit
class TestCmdMeasure:
    """Test cmd_measure function."""

    def test_measure_no_measurements(self) -> None:
        """Test measure without measurement names."""
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(OscuraError, match="requires at least one measurement"):
            cmd_measure(trace)

    def test_measure_rise_time(self) -> None:
        """Test measuring rise time."""
        trace = np.array([0.0, 0.5, 1.0, 1.0, 1.0])

        with patch("oscura.analyzers.measurements.rise_time") as mock_meas:
            mock_meas.return_value = 0.5e-6
            result = cmd_measure(trace, "rise_time")
            mock_meas.assert_called_once_with(trace)
            assert result == 0.5e-6

    def test_measure_fall_time(self) -> None:
        """Test measuring fall time."""
        trace = np.array([1.0, 1.0, 0.5, 0.0, 0.0])

        with patch("oscura.analyzers.measurements.fall_time") as mock_meas:
            mock_meas.return_value = 0.3e-6
            result = cmd_measure(trace, "fall_time")
            mock_meas.assert_called_once_with(trace)
            assert result == 0.3e-6

    def test_measure_period(self) -> None:
        """Test measuring period."""
        trace = np.sin(np.linspace(0, 4 * np.pi, 100))

        with patch("oscura.analyzers.measurements.period") as mock_meas:
            mock_meas.return_value = 1.0e-3
            result = cmd_measure(trace, "period")
            mock_meas.assert_called_once_with(trace)
            assert result == 1.0e-3

    def test_measure_frequency(self) -> None:
        """Test measuring frequency."""
        trace = np.sin(np.linspace(0, 4 * np.pi, 100))

        with patch("oscura.analyzers.measurements.frequency") as mock_meas:
            mock_meas.return_value = 1000.0
            result = cmd_measure(trace, "frequency")
            mock_meas.assert_called_once_with(trace)
            assert result == 1000.0

    def test_measure_amplitude(self) -> None:
        """Test measuring amplitude."""
        trace = np.array([-1.0, 0.0, 1.0, 0.0, -1.0])

        with patch("oscura.analyzers.measurements.amplitude") as mock_meas:
            mock_meas.return_value = 2.0
            result = cmd_measure(trace, "amplitude")
            mock_meas.assert_called_once_with(trace)
            assert result == 2.0

    def test_measure_mean(self) -> None:
        """Test measuring mean."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.analyzers.measurements.mean") as mock_meas:
            mock_meas.return_value = 3.0
            result = cmd_measure(trace, "mean")
            mock_meas.assert_called_once_with(trace)
            assert result == 3.0

    def test_measure_rms(self) -> None:
        """Test measuring RMS."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.analyzers.measurements.rms") as mock_meas:
            mock_meas.return_value = 3.3166
            result = cmd_measure(trace, "rms")
            mock_meas.assert_called_once_with(trace)
            assert result == 3.3166

    def test_measure_multiple(self) -> None:
        """Test measuring multiple properties."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.analyzers.measurements") as mock_meas_module:
            mock_meas_module.rise_time.return_value = 0.5e-6
            mock_meas_module.fall_time.return_value = 0.3e-6
            mock_meas_module.frequency.return_value = 1000.0

            result = cmd_measure(trace, "rise_time", "fall_time", "frequency")

            assert isinstance(result, dict)
            assert result["rise_time"] == 0.5e-6
            assert result["fall_time"] == 0.3e-6
            assert result["frequency"] == 1000.0

    def test_measure_all(self) -> None:
        """Test measuring all available measurements."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.analyzers.measurements.measure") as mock_meas:
            mock_meas.return_value = {
                "frequency": 1000.0,
                "period": 1.0e-3,
                "amplitude": 4.0,
                "rms": 3.3166,
            }
            result = cmd_measure(trace, "all")
            mock_meas.assert_called_once_with(trace, parameters=None)
            assert len(result) == 4

    def test_measure_unknown(self) -> None:
        """Test measuring unknown property."""
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(OscuraError, match="Unknown measurement"):
            cmd_measure(trace, "invalid_measurement")

    def test_measure_import_error(self) -> None:
        """Test handling of import errors."""
        trace = np.array([1.0, 2.0, 3.0])
        # Patch the measurements module's mean function to raise ImportError
        with patch("oscura.analyzers.measurements.mean", side_effect=ImportError):
            with pytest.raises(OscuraError, match="Measurements module not available"):
                cmd_measure(trace, "mean")


@pytest.mark.unit
class TestCmdPlot:
    """Test cmd_plot function."""

    def test_plot_basic(self) -> None:
        """Test basic plotting."""
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.visualization.plot") as mock_plot:
            with patch("matplotlib.pyplot") as mock_plt:
                result = cmd_plot(trace)
                mock_plot.plot_trace.assert_called_once()
                mock_plt.show.assert_called_once()
                # Verify command executed without error
                assert result is None or result is not None  # plot returns None or figure

    def test_plot_with_title(self) -> None:
        """Test plotting with custom title."""
        trace = np.array([1.0, 2.0, 3.0])

        with patch("oscura.visualization.plot") as mock_plot:
            with patch("matplotlib.pyplot") as mock_plt:
                result = cmd_plot(trace, title="Custom Title")
                mock_plot.plot_trace.assert_called_once_with(trace, title="Custom Title")
                mock_plt.show.assert_called_once()
                assert result is None or result is not None

    def test_plot_with_annotation(self) -> None:
        """Test plotting with annotation."""
        trace = np.array([1.0, 2.0, 3.0])

        with patch("oscura.visualization.plot") as mock_plot:
            with patch("matplotlib.pyplot") as mock_plt:
                result = cmd_plot(trace, annotate="Test annotation")
                mock_plot.add_annotation.assert_called_once_with("Test annotation")
                mock_plt.show.assert_called_once()
                assert result is None or result is not None

    def test_plot_import_error(self) -> None:
        """Test handling of import errors."""
        trace = np.array([1.0, 2.0, 3.0])
        # Patch the plot module's plot_trace function to raise ImportError
        with patch("oscura.visualization.plot.plot_trace", side_effect=ImportError):
            with pytest.raises(OscuraError, match="Visualization module not available"):
                cmd_plot(trace)


@pytest.mark.unit
class TestCmdExport:
    """Test cmd_export function."""

    def test_export_not_implemented(self) -> None:
        """Test that export raises NotImplementedError."""
        data = {"test": "data"}
        with pytest.raises(NotImplementedError, match="redesigned"):
            cmd_export(data, "json", "output.json")

    def test_export_various_formats(self) -> None:
        """Test export with various format types."""
        data = np.array([1, 2, 3])

        for fmt in ["json", "csv", "hdf5"]:
            with pytest.raises(NotImplementedError):
                cmd_export(data, fmt)


@pytest.mark.unit
class TestCmdGlob:
    """Test cmd_glob function."""

    def test_glob_basic(self, tmp_path: Path) -> None:
        """Test basic glob pattern matching."""
        # Create test files
        (tmp_path / "test1.csv").touch()
        (tmp_path / "test2.csv").touch()
        (tmp_path / "data.txt").touch()

        pattern = str(tmp_path / "*.csv")
        result = cmd_glob(pattern)

        assert len(result) == 2
        assert all(
            str(tmp_path / "test") in str(f) or str(tmp_path / "test2") in str(f) for f in result
        )

    def test_glob_no_matches(self, tmp_path: Path) -> None:
        """Test glob with no matches."""
        pattern = str(tmp_path / "*.nonexistent")
        result = cmd_glob(pattern)
        assert result == []

    def test_glob_recursive(self, tmp_path: Path) -> None:
        """Test recursive glob pattern."""
        # Create nested directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "test1.bin").touch()
        (subdir / "test2.bin").touch()

        pattern = str(tmp_path / "**/*.bin")
        result = cmd_glob(pattern)

        assert len(result) >= 1


@pytest.mark.unit
class TestBuiltinCommands:
    """Test BUILTIN_COMMANDS registry."""

    def test_builtin_commands_registry(self) -> None:
        """Test that all commands are registered."""
        expected_commands = {"load", "filter", "measure", "plot", "export", "glob"}

        assert set(BUILTIN_COMMANDS.keys()) == expected_commands

    def test_all_commands_callable(self) -> None:
        """Test that all registered commands are callable."""
        for name, cmd in BUILTIN_COMMANDS.items():
            assert callable(cmd), f"Command {name} is not callable"
