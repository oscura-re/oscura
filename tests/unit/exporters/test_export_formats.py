"""Comprehensive tests for Oscura export API.

Tests all export formats (CSV, JSON, HDF5, MATLAB) with round-trip validation.
"""

import json
from datetime import datetime

import numpy as np
import pytest

import oscura as osc
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.exporter]


@pytest.fixture
def sample_waveform():
    """Create sample waveform trace for testing."""
    t = np.linspace(0, 1e-3, 1000)
    data = np.sin(2 * np.pi * 1e3 * t)
    metadata = TraceMetadata(
        sample_rate=1e6,
        vertical_scale=0.1,
        vertical_offset=0.0,
        acquisition_time=datetime.now(),
        source_file="test.wfm",
        channel_name="CH1",
    )
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sample_digital():
    """Create sample digital trace for testing."""
    data = np.array([False, False, True, True, False, True, False], dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    edges = [(2e-6, True), (4e-6, False), (5e-6, True), (6e-6, False)]
    return DigitalTrace(data=data, metadata=metadata, edges=edges)


@pytest.mark.unit
@pytest.mark.exporter
class TestCSVExport:
    """Test CSV export functionality."""

    def test_export_csv_waveform(self, sample_waveform, tmp_path):
        """Test exporting waveform to CSV."""
        output = tmp_path / "waveform.csv"
        osc.export_csv(sample_waveform, output)

        assert output.exists()

        # Read and verify content
        with open(output) as f:
            content = f.read()

        # Check metadata comments
        assert "# Oscura CSV Export" in content
        assert "# Sample Rate:" in content
        assert "# Duration:" in content

        # Check data rows
        lines = content.split("\n")
        data_lines = [line for line in lines if line and not line.startswith("#")]
        assert len(data_lines) > 1  # Header + data rows

    def test_export_csv_digital(self, sample_digital, tmp_path):
        """Test exporting digital trace to CSV."""
        output = tmp_path / "digital.csv"
        osc.export_csv(sample_digital, output)

        assert output.exists()

        with open(output) as f:
            content = f.read()

        assert "Digital" in content

    def test_export_csv_precision(self, sample_waveform, tmp_path):
        """Test CSV export with different precision."""
        output = tmp_path / "precise.csv"
        osc.export_csv(sample_waveform, output, precision=12)

        assert output.exists()

    def test_export_csv_delimiter(self, sample_waveform, tmp_path):
        """Test CSV export with custom delimiter."""
        output = tmp_path / "tabbed.csv"
        osc.export_csv(sample_waveform, output, delimiter="\t")

        assert output.exists()

        with open(output) as f:
            content = f.read()

        # Check tab delimiter in data (skip comment lines)
        data_lines = [line for line in content.split("\n") if line and not line.startswith("#")]
        assert "\t" in data_lines[1]  # First data row

    def test_export_csv_no_header(self, sample_waveform, tmp_path):
        """Test CSV export without header."""
        output = tmp_path / "no_header.csv"
        osc.export_csv(sample_waveform, output, header=False)

        assert output.exists()

        with open(output) as f:
            content = f.read()

        assert "#" not in content

    def test_export_csv_time_units(self, sample_waveform, tmp_path):
        """Test CSV export with different time units."""
        for unit in ["s", "ms", "us", "ns"]:
            output = tmp_path / f"time_{unit}.csv"
            osc.export_csv(sample_waveform, output, time_unit=unit)
            assert output.exists()

            with open(output) as f:
                content = f.read()

            assert f"Time ({unit})" in content


@pytest.mark.unit
@pytest.mark.exporter
class TestJSONExport:
    """Test JSON export functionality."""

    def test_export_json_waveform(self, sample_waveform, tmp_path):
        """Test exporting waveform to JSON."""
        output = tmp_path / "waveform.json"
        osc.export_json(sample_waveform, output)

        assert output.exists()

        # Load and verify
        with open(output) as f:
            data = json.load(f)

        assert "_metadata" in data
        assert data["_metadata"]["format"] == "oscura_json"
        assert "data" in data
        assert data["data"]["_type"] == "WaveformTrace"
        assert "metadata" in data["data"]
        assert data["data"]["metadata"]["sample_rate"] == 1e6

    def test_export_json_digital(self, sample_digital, tmp_path):
        """Test exporting digital trace to JSON."""
        output = tmp_path / "digital.json"
        osc.export_json(sample_digital, output)

        assert output.exists()

        with open(output) as f:
            data = json.load(f)

        assert data["data"]["_type"] == "DigitalTrace"
        assert "edges" in data["data"]

    def test_export_json_pretty(self, sample_waveform, tmp_path):
        """Test JSON export with pretty printing."""
        output = tmp_path / "pretty.json"
        osc.export_json(sample_waveform, output, pretty=True)

        with open(output) as f:
            content = f.read()

        # Pretty printed JSON has indentation
        assert "  " in content

    def test_export_json_compact(self, sample_waveform, tmp_path):
        """Test JSON export without pretty printing."""
        output = tmp_path / "compact.json"
        osc.export_json(sample_waveform, output, pretty=False)

        with open(output) as f:
            content = f.read()

        # Compact JSON has no extra whitespace
        assert "\n  " not in content

    def test_export_json_compressed(self, sample_waveform, tmp_path):
        """Test JSON export with compression."""
        output = tmp_path / "compressed.json.gz"
        osc.export_json(sample_waveform, output, compress=True)

        assert output.exists()
        assert output.suffix == ".gz"

        # Verify can be read
        import gzip

        with gzip.open(output, "rt") as f:
            data = json.load(f)

        assert data["data"]["_type"] == "WaveformTrace"

    def test_export_json_dict(self, tmp_path):
        """Test exporting dictionary to JSON."""
        output = tmp_path / "dict.json"
        data = {"measurement1": 1.23, "measurement2": 4.56}
        osc.export_json(data, output)

        assert output.exists()

        with open(output) as f:
            loaded = json.load(f)

        assert loaded["data"]["measurement1"] == 1.23


@pytest.mark.unit
@pytest.mark.exporter
class TestHDF5Export:
    """Test HDF5 export functionality."""

    def test_export_hdf5_waveform(self, sample_waveform, tmp_path):
        """Test exporting waveform to HDF5."""
        pytest.importorskip("h5py")
        import h5py

        output = tmp_path / "waveform.h5"
        osc.export_hdf5(sample_waveform, output)

        assert output.exists()

        # Verify content
        with h5py.File(output, "r") as f:
            assert "trace" in f
            assert f["trace"].attrs["sample_rate"] == 1e6
            assert f["trace"].attrs["trace_type"] == "waveform"
            assert len(f["trace"][:]) == len(sample_waveform.data)

    def test_export_hdf5_digital(self, sample_digital, tmp_path):
        """Test exporting digital trace to HDF5."""
        pytest.importorskip("h5py")
        import h5py

        output = tmp_path / "digital.h5"
        osc.export_hdf5(sample_digital, output)

        assert output.exists()

        with h5py.File(output, "r") as f:
            assert f["trace"].attrs["trace_type"] == "digital"

    def test_export_hdf5_multiple(self, sample_waveform, sample_digital, tmp_path):
        """Test exporting multiple traces to HDF5."""
        pytest.importorskip("h5py")
        import h5py

        output = tmp_path / "multi.h5"
        osc.export_hdf5({"ch1": sample_waveform, "ch2": sample_digital}, output)

        assert output.exists()

        with h5py.File(output, "r") as f:
            assert "ch1" in f
            assert "ch2" in f

    def test_export_hdf5_compression(self, sample_waveform, tmp_path):
        """Test HDF5 export with compression."""
        pytest.importorskip("h5py")

        output = tmp_path / "compressed.h5"
        osc.export_hdf5(sample_waveform, output, compression="gzip", compression_opts=9)

        assert output.exists()

    def test_export_hdf5_no_compression(self, sample_waveform, tmp_path):
        """Test HDF5 export without compression."""
        pytest.importorskip("h5py")

        output = tmp_path / "uncompressed.h5"
        osc.export_hdf5(sample_waveform, output, compression=None)

        assert output.exists()


@pytest.mark.unit
@pytest.mark.exporter
class TestMATLABExport:
    """Test MATLAB export functionality."""

    def test_export_mat_waveform(self, sample_waveform, tmp_path):
        """Test exporting waveform to MATLAB."""
        scipy = pytest.importorskip("scipy")

        output = tmp_path / "waveform.mat"
        osc.export_mat(sample_waveform, output)

        assert output.exists()

        # Load and verify
        mat_data = scipy.io.loadmat(str(output))
        assert "trace_data" in mat_data
        assert "trace_time" in mat_data
        assert "trace_metadata" in mat_data
        assert len(mat_data["trace_data"]) == len(sample_waveform.data)

    def test_export_mat_digital(self, sample_digital, tmp_path):
        """Test exporting digital trace to MATLAB."""
        scipy = pytest.importorskip("scipy")

        output = tmp_path / "digital.mat"
        osc.export_mat(sample_digital, output)

        assert output.exists()

        mat_data = scipy.io.loadmat(str(output))
        assert "trace_data" in mat_data

    def test_export_mat_version_5(self, sample_waveform, tmp_path):
        """Test MATLAB export with version 5 format."""
        pytest.importorskip("scipy")

        output = tmp_path / "v5.mat"
        osc.export_mat(sample_waveform, output, version="5")

        assert output.exists()

    def test_export_mat_version_73(self, sample_waveform, tmp_path):
        """Test MATLAB export with version 7.3 format."""
        pytest.importorskip("scipy")
        h5py = pytest.importorskip("h5py")

        output = tmp_path / "v73.mat"
        osc.export_mat(sample_waveform, output, version="7.3")

        assert output.exists()

        # Version 7.3 is HDF5 format - verify with h5py
        with h5py.File(output, "r") as f:
            assert "trace_data" in f
            assert "trace_time" in f

    def test_export_mat_multiple(self, sample_waveform, sample_digital, tmp_path):
        """Test exporting multiple traces to MATLAB."""
        scipy = pytest.importorskip("scipy")

        output = tmp_path / "multi.mat"
        osc.export_mat({"ch1": sample_waveform, "ch2": sample_digital}, output)

        assert output.exists()

        mat_data = scipy.io.loadmat(str(output))
        assert "ch1_data" in mat_data
        assert "ch2_data" in mat_data

    def test_export_mat_dict(self, tmp_path):
        """Test exporting dictionary to MATLAB."""
        scipy = pytest.importorskip("scipy")

        output = tmp_path / "dict.mat"
        data = {"value1": 1.23, "value2": np.array([1, 2, 3])}
        osc.export_mat(data, output)

        assert output.exists()

        mat_data = scipy.io.loadmat(str(output))
        assert "value1" in mat_data


@pytest.mark.unit
@pytest.mark.exporter
class TestRoundTripExport:
    """Test round-trip export/import for data integrity."""

    def test_roundtrip_csv(self, sample_waveform, tmp_path):
        """Test CSV round-trip preserves data."""
        output = tmp_path / "roundtrip.csv"
        osc.export_csv(sample_waveform, output)

        # Count comment lines to determine skiprows
        with open(output) as f:
            lines = f.readlines()

        # Count comment lines (start with #) plus header row
        comment_lines = sum(1 for line in lines if line.startswith("#"))
        header_line = 1  # The column header row
        skiprows = comment_lines + header_line

        # Read back and verify
        data = np.loadtxt(output, delimiter=",", skiprows=skiprows)
        time = data[:, 0]
        voltage = data[:, 1]

        # Check time vector
        np.testing.assert_allclose(time, sample_waveform.time_vector, rtol=1e-6)

        # Check voltage data
        np.testing.assert_allclose(voltage, sample_waveform.data, rtol=1e-6)

    def test_roundtrip_hdf5(self, sample_waveform, tmp_path):
        """Test HDF5 round-trip preserves data and metadata."""
        pytest.importorskip("h5py")
        import h5py

        output = tmp_path / "roundtrip.h5"
        osc.export_hdf5(sample_waveform, output)

        # Read back and verify
        with h5py.File(output, "r") as f:
            data = f["trace"][:]
            sample_rate = f["trace"].attrs["sample_rate"]

        np.testing.assert_array_equal(data, sample_waveform.data)
        assert sample_rate == sample_waveform.metadata.sample_rate

    def test_roundtrip_mat(self, sample_waveform, tmp_path):
        """Test MATLAB round-trip preserves data."""
        scipy = pytest.importorskip("scipy")

        output = tmp_path / "roundtrip.mat"
        osc.export_mat(sample_waveform, output)

        # Read back and verify
        mat_data = scipy.io.loadmat(str(output))
        data = mat_data["trace_data"].flatten()
        time = mat_data["trace_time"].flatten()

        np.testing.assert_allclose(data, sample_waveform.data, rtol=1e-6)
        np.testing.assert_allclose(time, sample_waveform.time_vector, rtol=1e-6)


@pytest.mark.unit
@pytest.mark.exporter
class TestExportErrors:
    """Test error handling in export functions."""

    def test_export_csv_invalid_type(self, tmp_path):
        """Test CSV export with invalid data type."""
        output = tmp_path / "invalid.csv"
        with pytest.raises(TypeError):
            osc.export_csv("invalid", output)

    def test_export_mat_no_scipy(self, sample_waveform, tmp_path, monkeypatch, request):
        """Test MATLAB export without scipy installed."""
        # Mock scipy import failure
        import sys

        monkeypatch.setitem(sys.modules, "scipy.io", None)

        output = tmp_path / "test.mat"
        # Re-import to trigger the ImportError check
        from importlib import reload

        import oscura.exporters.matlab_export as mat_module

        # Add cleanup to reload module after test
        def cleanup():
            # Ensure scipy is restored before reload
            if "scipy.io" in sys.modules and sys.modules["scipy.io"] is None:
                del sys.modules["scipy.io"]
            reload(mat_module)

        request.addfinalizer(cleanup)

        reload(mat_module)

        if not mat_module.HAS_SCIPY:
            with pytest.raises(ImportError, match="scipy is required"):
                mat_module.export_mat(sample_waveform, output)

    def test_export_hdf5_no_h5py(self, sample_waveform, tmp_path, monkeypatch, request):
        """Test HDF5 export without h5py installed."""
        import sys

        monkeypatch.setitem(sys.modules, "h5py", None)

        output = tmp_path / "test.h5"
        from importlib import reload

        import oscura.exporters.hdf5 as hdf5_module

        # Add cleanup to reload module after test
        def cleanup():
            # Ensure h5py is restored before reload
            if "h5py" in sys.modules and sys.modules["h5py"] is None:
                del sys.modules["h5py"]
            reload(hdf5_module)

        request.addfinalizer(cleanup)

        reload(hdf5_module)

        if not hdf5_module.HAS_H5PY:
            with pytest.raises(ImportError, match="h5py is required"):
                hdf5_module.export_hdf5(sample_waveform, output)


@pytest.mark.unit
@pytest.mark.exporter
class TestExportIntegration:
    """Integration tests for export API."""

    def test_all_formats_available(self):
        """Test that all export functions are accessible from top-level API."""
        assert hasattr(osc, "export_csv")
        assert hasattr(osc, "export_json")
        assert hasattr(osc, "export_hdf5")
        assert hasattr(osc, "export_mat")

    def test_export_multiple_formats(self, sample_waveform, tmp_path):
        """Test exporting same trace to multiple formats."""
        osc.export_csv(sample_waveform, tmp_path / "test.csv")
        osc.export_json(sample_waveform, tmp_path / "test.json")

        # Only test HDF5/MATLAB if libraries available
        try:
            osc.export_hdf5(sample_waveform, tmp_path / "test.h5")
        except ImportError:
            pass

        try:
            osc.export_mat(sample_waveform, tmp_path / "test.mat")
        except ImportError:
            pass

        # At least CSV and JSON should exist
        assert (tmp_path / "test.csv").exists()
        assert (tmp_path / "test.json").exists()
