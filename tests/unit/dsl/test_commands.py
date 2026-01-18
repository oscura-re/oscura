"""Tests for DSL commands.

Requirements tested:
"""

import sys
from unittest.mock import Mock, patch

import pytest

from oscura.core.exceptions import OscuraError
from oscura.dsl.commands import (
    BUILTIN_COMMANDS,
    cmd_export,
    cmd_filter,
    cmd_glob,
    cmd_load,
    cmd_measure,
    cmd_plot,
)

pytestmark = pytest.mark.unit


class TestCmdLoad:
    """Test cmd_load function."""

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(OscuraError, match="File not found"):
            cmd_load("/nonexistent/file.csv")

    def test_load_csv_file(self, tmp_path):
        """Test loading CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("time,value\n0,1\n1,2\n")

        mock_trace = {"data": [1, 2]}

        with patch("oscura.loaders.csv.load_csv", return_value=mock_trace) as mock_load:
            result = cmd_load(str(csv_file))

            assert result == mock_trace
            mock_load.assert_called_once_with(str(csv_file))

    def test_load_binary_file(self, tmp_path):
        """Test loading binary file."""
        bin_file = tmp_path / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")

        mock_trace = {"data": [0, 1, 2, 3]}

        with patch("oscura.loaders.binary.load_binary", return_value=mock_trace) as mock_load:
            result = cmd_load(str(bin_file))

            assert result == mock_trace
            mock_load.assert_called_once_with(str(bin_file))

    def test_load_hdf5_file(self, tmp_path):
        """Test loading HDF5 file."""
        h5_file = tmp_path / "test.h5"
        h5_file.touch()

        mock_trace = {"data": [1, 2, 3]}

        with patch("oscura.loaders.hdf5.load_hdf5", return_value=mock_trace) as mock_load:
            result = cmd_load(str(h5_file))

            assert result == mock_trace
            mock_load.assert_called_once_with(str(h5_file))

    def test_load_hdf5_file_alternate_extension(self, tmp_path):
        """Test loading HDF5 file with .hdf5 extension."""
        h5_file = tmp_path / "test.hdf5"
        h5_file.touch()

        mock_trace = {"data": [1, 2, 3]}

        with patch("oscura.loaders.hdf5.load_hdf5", return_value=mock_trace) as mock_load:
            result = cmd_load(str(h5_file))

            assert result == mock_trace
            mock_load.assert_called_once_with(str(h5_file))

    def test_load_unsupported_format(self, tmp_path):
        """Test loading file with unsupported format."""
        unknown_file = tmp_path / "test.xyz"
        unknown_file.touch()

        with pytest.raises(OscuraError, match="Unsupported file format: .xyz"):
            cmd_load(str(unknown_file))

    def test_load_import_error(self, tmp_path):
        """Test handling ImportError when loader module is not available."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("time,value\n0,1\n")

        with patch("oscura.loaders.csv.load_csv", side_effect=ImportError("Module not found")):
            with pytest.raises(OscuraError, match="Loader not available for .csv"):
                cmd_load(str(csv_file))

    def test_load_case_insensitive_extension(self, tmp_path):
        """Test that file extensions are handled case-insensitively."""
        csv_file = tmp_path / "test.CSV"
        csv_file.write_text("time,value\n0,1\n")

        mock_trace = {"data": [1, 2]}

        with patch("oscura.loaders.csv.load_csv", return_value=mock_trace):
            result = cmd_load(str(csv_file))

            assert result == mock_trace


class TestCmdFilter:
    """Test cmd_filter function."""

    def test_filter_lowpass(self):
        """Test lowpass filter."""
        trace = {"data": [1, 2, 3]}
        filtered_trace = {"data": [1.0, 1.5, 2.0]}

        mock_filters = Mock()
        mock_filters.low_pass = Mock(return_value=filtered_trace)

        with patch("oscura.filtering.filters", mock_filters):
            result = cmd_filter(trace, "lowpass", 1000)

            assert result == filtered_trace
            mock_filters.low_pass.assert_called_once_with(trace, cutoff=1000)

    def test_filter_lowpass_case_insensitive(self):
        """Test lowpass filter with case-insensitive filter type."""
        trace = {"data": [1, 2, 3]}
        filtered_trace = {"data": [1.0, 1.5, 2.0]}

        mock_filters = Mock()
        mock_filters.low_pass = Mock(return_value=filtered_trace)

        with patch("oscura.filtering.filters", mock_filters):
            result = cmd_filter(trace, "LowPass", 1000)

            assert result == filtered_trace
            mock_filters.low_pass.assert_called_once_with(trace, cutoff=1000)

    def test_filter_lowpass_with_kwargs(self):
        """Test lowpass filter with additional keyword arguments."""
        trace = {"data": [1, 2, 3]}
        filtered_trace = {"data": [1.0, 1.5, 2.0]}

        mock_filters = Mock()
        mock_filters.low_pass = Mock(return_value=filtered_trace)

        with patch("oscura.filtering.filters", mock_filters):
            result = cmd_filter(trace, "lowpass", 1000, order=4)

            assert result == filtered_trace
            mock_filters.low_pass.assert_called_once_with(trace, cutoff=1000, order=4)

    def test_filter_lowpass_missing_cutoff(self):
        """Test lowpass filter without cutoff frequency."""
        trace = {"data": [1, 2, 3]}

        mock_filters = Mock()

        with patch("oscura.filtering.filters", mock_filters):
            with pytest.raises(OscuraError, match="lowpass filter requires cutoff frequency"):
                cmd_filter(trace, "lowpass")

    def test_filter_highpass(self):
        """Test highpass filter."""
        trace = {"data": [1, 2, 3]}
        filtered_trace = {"data": [0.5, 1.0, 1.5]}

        mock_filters = Mock()
        mock_filters.high_pass = Mock(return_value=filtered_trace)

        with patch("oscura.filtering.filters", mock_filters):
            result = cmd_filter(trace, "highpass", 500)

            assert result == filtered_trace
            mock_filters.high_pass.assert_called_once_with(trace, cutoff=500)

    def test_filter_highpass_missing_cutoff(self):
        """Test highpass filter without cutoff frequency."""
        trace = {"data": [1, 2, 3]}

        mock_filters = Mock()

        with patch("oscura.filtering.filters", mock_filters):
            with pytest.raises(OscuraError, match="highpass filter requires cutoff frequency"):
                cmd_filter(trace, "highpass")

    def test_filter_bandpass(self):
        """Test bandpass filter."""
        trace = {"data": [1, 2, 3]}
        filtered_trace = {"data": [1.2, 1.8, 2.2]}

        mock_filters = Mock()
        mock_filters.band_pass = Mock(return_value=filtered_trace)

        with patch("oscura.filtering.filters", mock_filters):
            result = cmd_filter(trace, "bandpass", 500, 1500)

            assert result == filtered_trace
            mock_filters.band_pass.assert_called_once_with(trace, low=500, high=1500)

    def test_filter_bandpass_missing_frequencies(self):
        """Test bandpass filter without both frequencies."""
        trace = {"data": [1, 2, 3]}

        mock_filters = Mock()

        with patch("oscura.filtering.filters", mock_filters):
            with pytest.raises(
                OscuraError, match="bandpass filter requires low and high cutoff frequencies"
            ):
                cmd_filter(trace, "bandpass", 500)

    def test_filter_bandstop(self):
        """Test bandstop filter."""
        trace = {"data": [1, 2, 3]}
        filtered_trace = {"data": [0.9, 1.9, 2.9]}

        mock_filters = Mock()
        mock_filters.band_stop = Mock(return_value=filtered_trace)

        with patch("oscura.filtering.filters", mock_filters):
            result = cmd_filter(trace, "bandstop", 400, 600)

            assert result == filtered_trace
            mock_filters.band_stop.assert_called_once_with(trace, low=400, high=600)

    def test_filter_bandstop_missing_frequencies(self):
        """Test bandstop filter without both frequencies."""
        trace = {"data": [1, 2, 3]}

        mock_filters = Mock()

        with patch("oscura.filtering.filters", mock_filters):
            with pytest.raises(
                OscuraError, match="bandstop filter requires low and high cutoff frequencies"
            ):
                cmd_filter(trace, "bandstop", 400)

    def test_filter_unknown_type(self):
        """Test unknown filter type."""
        trace = {"data": [1, 2, 3]}

        mock_filters = Mock()

        with patch("oscura.filtering.filters", mock_filters):
            with pytest.raises(OscuraError, match="Unknown filter type: notch"):
                cmd_filter(trace, "notch", 1000)

    def test_filter_import_error(self):
        """Test handling ImportError when filtering module is not available."""
        trace = {"data": [1, 2, 3]}

        # Mock the import at module level to trigger the ImportError in the try block

        # Save original module
        original_module = sys.modules.get("oscura.filtering")

        try:
            # Remove the module to trigger ImportError
            if "oscura.filtering" in sys.modules:
                del sys.modules["oscura.filtering"]
            if "oscura.filtering.filters" in sys.modules:
                del sys.modules["oscura.filtering.filters"]

            # Create a fake module that raises ImportError on attribute access
            class FakeModule:
                def __getattr__(self, name):
                    raise ImportError("Module not available")

            sys.modules["oscura.filtering"] = FakeModule()

            with pytest.raises(OscuraError, match="Filtering module not available"):
                cmd_filter(trace, "lowpass", 1000)
        finally:
            # Restore original module
            if original_module is not None:
                sys.modules["oscura.filtering"] = original_module
            elif "oscura.filtering" in sys.modules:
                del sys.modules["oscura.filtering"]


class TestCmdMeasure:
    """Test cmd_measure function."""

    def test_measure_rise_time(self):
        """Test measuring rise time."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.rise_time = Mock(return_value=1.5)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "rise_time")

            assert result == 1.5
            mock_meas.rise_time.assert_called_once_with(trace)

    def test_measure_fall_time(self):
        """Test measuring fall time."""
        trace = {"data": [3, 2, 1]}

        mock_meas = Mock()
        mock_meas.fall_time = Mock(return_value=2.1)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "fall_time")

            assert result == 2.1
            mock_meas.fall_time.assert_called_once_with(trace)

    def test_measure_period(self):
        """Test measuring period."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.period = Mock(return_value=10.0)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "period")

            assert result == 10.0
            mock_meas.period.assert_called_once_with(trace)

    def test_measure_frequency(self):
        """Test measuring frequency."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.frequency = Mock(return_value=100.0)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "frequency")

            assert result == 100.0
            mock_meas.frequency.assert_called_once_with(trace)

    def test_measure_amplitude(self):
        """Test measuring amplitude."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.amplitude = Mock(return_value=2.0)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "amplitude")

            assert result == 2.0
            mock_meas.amplitude.assert_called_once_with(trace)

    def test_measure_mean(self):
        """Test measuring mean."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.mean = Mock(return_value=2.0)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "mean")

            assert result == 2.0
            mock_meas.mean.assert_called_once_with(trace)

    def test_measure_rms(self):
        """Test measuring RMS."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.rms = Mock(return_value=2.16)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "rms")

            assert result == 2.16
            mock_meas.rms.assert_called_once_with(trace)

    def test_measure_multiple(self):
        """Test measuring multiple properties."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.rise_time = Mock(return_value=1.5)
        mock_meas.fall_time = Mock(return_value=2.1)
        mock_meas.period = Mock(return_value=10.0)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "rise_time", "fall_time", "period")

            assert result == {"rise_time": 1.5, "fall_time": 2.1, "period": 10.0}
            mock_meas.rise_time.assert_called_once_with(trace)
            mock_meas.fall_time.assert_called_once_with(trace)
            mock_meas.period.assert_called_once_with(trace)

    def test_measure_all(self):
        """Test measuring all properties."""
        trace = {"data": [1, 2, 3]}

        all_measurements = {
            "rise_time": 1.5,
            "fall_time": 2.1,
            "period": 10.0,
            "frequency": 0.1,
            "amplitude": 2.0,
            "mean": 2.0,
            "rms": 2.16,
        }

        mock_meas = Mock()
        mock_meas.measure_all = Mock(return_value=all_measurements)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "all")

            assert result == all_measurements
            mock_meas.measure_all.assert_called_once_with(trace)

    def test_measure_all_with_other_measurements(self):
        """Test that 'all' breaks the loop and ignores other measurements."""
        trace = {"data": [1, 2, 3]}

        all_measurements = {"rise_time": 1.5, "fall_time": 2.1}

        mock_meas = Mock()
        mock_meas.measure_all = Mock(return_value=all_measurements)
        mock_meas.period = Mock(return_value=10.0)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "all", "period")

            assert result == all_measurements
            mock_meas.measure_all.assert_called_once_with(trace)
            # period should not be called since 'all' breaks the loop
            mock_meas.period.assert_not_called()

    def test_measure_case_insensitive(self):
        """Test that measurement names are case-insensitive."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()
        mock_meas.rise_time = Mock(return_value=1.5)

        with patch("oscura.analyzers.measurements", mock_meas):
            result = cmd_measure(trace, "Rise_Time")

            assert result == 1.5
            mock_meas.rise_time.assert_called_once_with(trace)

    def test_measure_no_measurements(self):
        """Test calling measure without any measurement names."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()

        with patch("oscura.analyzers.measurements", mock_meas):
            with pytest.raises(
                OscuraError, match="measure command requires at least one measurement name"
            ):
                cmd_measure(trace)

    def test_measure_unknown_measurement(self):
        """Test unknown measurement name."""
        trace = {"data": [1, 2, 3]}

        mock_meas = Mock()

        with patch("oscura.analyzers.measurements", mock_meas):
            with pytest.raises(OscuraError, match="Unknown measurement: voltage"):
                cmd_measure(trace, "voltage")

    def test_measure_import_error(self):
        """Test handling ImportError when measurements module is not available."""
        trace = {"data": [1, 2, 3]}

        # Mock the import at module level to trigger the ImportError in the try block

        # Save original module
        original_module = sys.modules.get("oscura.analyzers")

        try:
            # Remove the module to trigger ImportError
            if "oscura.analyzers" in sys.modules:
                del sys.modules["oscura.analyzers"]

            # Create a fake module that raises ImportError on attribute access
            class FakeModule:
                def __getattr__(self, name):
                    raise ImportError("Module not available")

            sys.modules["oscura.analyzers"] = FakeModule()

            with pytest.raises(OscuraError, match="Measurements module not available"):
                cmd_measure(trace, "rise_time")
        finally:
            # Restore original module
            if original_module is not None:
                sys.modules["oscura.analyzers"] = original_module
            elif "oscura.analyzers" in sys.modules:
                del sys.modules["oscura.analyzers"]


class TestCmdPlot:
    """Test cmd_plot function."""

    def test_plot_basic(self):
        """Test basic plotting."""
        trace = {"data": [1, 2, 3]}

        mock_plot = Mock()
        mock_plot.plot_trace = Mock()
        mock_plot.show = Mock()

        with patch("oscura.visualization.plot", mock_plot):
            cmd_plot(trace)

            mock_plot.plot_trace.assert_called_once_with(trace, title="Trace Plot")
            mock_plot.show.assert_called_once()

    def test_plot_with_title(self):
        """Test plotting with custom title."""
        trace = {"data": [1, 2, 3]}

        mock_plot = Mock()
        mock_plot.plot_trace = Mock()
        mock_plot.show = Mock()

        with patch("oscura.visualization.plot", mock_plot):
            cmd_plot(trace, title="My Custom Plot")

            mock_plot.plot_trace.assert_called_once_with(trace, title="My Custom Plot")
            mock_plot.show.assert_called_once()

    def test_plot_with_annotation(self):
        """Test plotting with annotation."""
        trace = {"data": [1, 2, 3]}

        mock_plot = Mock()
        mock_plot.plot_trace = Mock()
        mock_plot.add_annotation = Mock()
        mock_plot.show = Mock()

        with patch("oscura.visualization.plot", mock_plot):
            cmd_plot(trace, annotate="Peak at 2")

            mock_plot.plot_trace.assert_called_once_with(trace, title="Trace Plot")
            mock_plot.add_annotation.assert_called_once_with("Peak at 2")
            mock_plot.show.assert_called_once()

    def test_plot_with_title_and_annotation(self):
        """Test plotting with both title and annotation."""
        trace = {"data": [1, 2, 3]}

        mock_plot = Mock()
        mock_plot.plot_trace = Mock()
        mock_plot.add_annotation = Mock()
        mock_plot.show = Mock()

        with patch("oscura.visualization.plot", mock_plot):
            cmd_plot(trace, title="Signal Analysis", annotate="Critical point")

            mock_plot.plot_trace.assert_called_once_with(trace, title="Signal Analysis")
            mock_plot.add_annotation.assert_called_once_with("Critical point")
            mock_plot.show.assert_called_once()

    def test_plot_without_annotation(self):
        """Test that annotation is not added when not provided."""
        trace = {"data": [1, 2, 3]}

        mock_plot = Mock()
        mock_plot.plot_trace = Mock()
        mock_plot.add_annotation = Mock()
        mock_plot.show = Mock()

        with patch("oscura.visualization.plot", mock_plot):
            cmd_plot(trace, title="Test")

            mock_plot.plot_trace.assert_called_once_with(trace, title="Test")
            mock_plot.add_annotation.assert_not_called()
            mock_plot.show.assert_called_once()

    def test_plot_import_error(self):
        """Test handling ImportError when visualization module is not available."""
        trace = {"data": [1, 2, 3]}

        # Mock the import at module level to trigger the ImportError in the try block

        # Save original module
        original_module = sys.modules.get("oscura.visualization")

        try:
            # Remove the module to trigger ImportError
            if "oscura.visualization" in sys.modules:
                del sys.modules["oscura.visualization"]
            if "oscura.visualization.plot" in sys.modules:
                del sys.modules["oscura.visualization.plot"]

            # Create a fake module that raises ImportError on attribute access
            class FakeModule:
                def __getattr__(self, name):
                    raise ImportError("Module not available")

            sys.modules["oscura.visualization"] = FakeModule()

            with pytest.raises(OscuraError, match="Visualization module not available"):
                cmd_plot(trace)
        finally:
            # Restore original module
            if original_module is not None:
                sys.modules["oscura.visualization"] = original_module
            elif "oscura.visualization" in sys.modules:
                del sys.modules["oscura.visualization"]


class TestCmdExport:
    """Test cmd_export function."""

    def test_export_json(self):
        """Test exporting to JSON format."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.json = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with patch("sys.stderr"):
                cmd_export(data, "json", "output.json")

            mock_exporters.json.assert_called_once_with(data, "output.json")

    def test_export_csv(self):
        """Test exporting to CSV format."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.csv = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with patch("sys.stderr"):
                cmd_export(data, "csv", "output.csv")

            mock_exporters.csv.assert_called_once_with(data, "output.csv")

    def test_export_hdf5(self):
        """Test exporting to HDF5 format."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.hdf5 = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with patch("sys.stderr"):
                cmd_export(data, "hdf5", "output.hdf5")

            mock_exporters.hdf5.assert_called_once_with(data, "output.hdf5")

    def test_export_h5_format(self):
        """Test exporting to HDF5 format with 'h5' extension."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.hdf5 = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with patch("sys.stderr"):
                cmd_export(data, "h5", "output.h5")

            mock_exporters.hdf5.assert_called_once_with(data, "output.h5")

    def test_export_case_insensitive(self):
        """Test that export format is case-insensitive."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.json = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with patch("sys.stderr"):
                cmd_export(data, "JSON", "output.json")

            mock_exporters.json.assert_called_once_with(data, "output.json")

    def test_export_auto_filename(self):
        """Test auto-generating filename when not provided."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.json = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with patch("sys.stderr"):
                cmd_export(data, "json")

            mock_exporters.json.assert_called_once_with(data, "export.json")

    def test_export_auto_filename_csv(self):
        """Test auto-generating filename for CSV format."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.csv = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with patch("sys.stderr"):
                cmd_export(data, "csv")

            mock_exporters.csv.assert_called_once_with(data, "export.csv")

    def test_export_prints_message(self, capsys):
        """Test that export prints confirmation message to stderr."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()
        mock_exporters.json = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            cmd_export(data, "json", "output.json")

        captured = capsys.readouterr()
        assert "Exported to output.json" in captured.err

    def test_export_unknown_format(self):
        """Test exporting to unknown format."""
        data = {"values": [1, 2, 3]}

        mock_exporters = Mock()

        with patch("oscura.exporters.exporters", mock_exporters):
            with pytest.raises(OscuraError, match="Unknown export format: xml"):
                cmd_export(data, "xml", "output.xml")

    def test_export_import_error(self):
        """Test handling ImportError when export module is not available."""
        data = {"values": [1, 2, 3]}

        # Mock the import at module level to trigger the ImportError in the try block

        # Save original module
        original_module = sys.modules.get("oscura.exporters")

        try:
            # Remove the module to trigger ImportError
            if "oscura.exporters" in sys.modules:
                del sys.modules["oscura.exporters"]
            if "oscura.exporters.exporters" in sys.modules:
                del sys.modules["oscura.exporters.exporters"]

            # Create a fake module that raises ImportError on attribute access
            class FakeModule:
                def __getattr__(self, name):
                    raise ImportError("Module not available")

            sys.modules["oscura.exporters"] = FakeModule()

            with pytest.raises(OscuraError, match="Export module not available"):
                cmd_export(data, "json", "output.json")
        finally:
            # Restore original module
            if original_module is not None:
                sys.modules["oscura.exporters"] = original_module
            elif "oscura.exporters" in sys.modules:
                del sys.modules["oscura.exporters"]


class TestCmdGlob:
    """Test cmd_glob function."""

    def test_glob_basic(self):
        """Test basic glob pattern matching."""
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["file1.csv", "file2.csv", "file3.csv"]

            result = cmd_glob("*.csv")

            assert result == ["file1.csv", "file2.csv", "file3.csv"]
            mock_glob.assert_called_once_with("*.csv")

    def test_glob_no_matches(self):
        """Test glob with no matches."""
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = []

            result = cmd_glob("*.xyz")

            assert result == []
            mock_glob.assert_called_once_with("*.xyz")

    def test_glob_complex_pattern(self):
        """Test glob with complex pattern."""
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["data/test1.bin", "data/test2.bin"]

            result = cmd_glob("data/*.bin")

            assert result == ["data/test1.bin", "data/test2.bin"]
            mock_glob.assert_called_once_with("data/*.bin")

    def test_glob_recursive_pattern(self):
        """Test glob with recursive pattern."""
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["dir1/file.csv", "dir2/subdir/file.csv"]

            result = cmd_glob("**/*.csv")

            assert result == ["dir1/file.csv", "dir2/subdir/file.csv"]
            mock_glob.assert_called_once_with("**/*.csv")

    def test_glob_returns_list(self):
        """Test that glob always returns a list."""
        with patch("glob.glob") as mock_glob:
            # Ensure we return a proper list even if glob_func returns an iterator
            mock_glob.return_value = iter(["file1.csv", "file2.csv"])

            result = cmd_glob("*.csv")

            assert isinstance(result, list)
            assert result == ["file1.csv", "file2.csv"]


class TestBuiltinCommands:
    """Test BUILTIN_COMMANDS registry."""

    def test_builtin_commands_registry(self):
        """Test that all commands are registered in BUILTIN_COMMANDS."""
        assert "load" in BUILTIN_COMMANDS
        assert "filter" in BUILTIN_COMMANDS
        assert "measure" in BUILTIN_COMMANDS
        assert "plot" in BUILTIN_COMMANDS
        assert "export" in BUILTIN_COMMANDS
        assert "glob" in BUILTIN_COMMANDS

    def test_builtin_commands_functions(self):
        """Test that registry maps to correct functions."""
        assert BUILTIN_COMMANDS["load"] is cmd_load
        assert BUILTIN_COMMANDS["filter"] is cmd_filter
        assert BUILTIN_COMMANDS["measure"] is cmd_measure
        assert BUILTIN_COMMANDS["plot"] is cmd_plot
        assert BUILTIN_COMMANDS["export"] is cmd_export
        assert BUILTIN_COMMANDS["glob"] is cmd_glob

    def test_builtin_commands_count(self):
        """Test that we have exactly 6 builtin commands."""
        assert len(BUILTIN_COMMANDS) == 6

    def test_all_commands_callable(self):
        """Test that all registered commands are callable."""
        for cmd_name, cmd_func in BUILTIN_COMMANDS.items():
            assert callable(cmd_func), f"Command '{cmd_name}' is not callable"
