"""Unit tests for HDF5 file loader.

Tests LOAD-006: HDF5 Loader
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders import hdf5_loader
from oscura.loaders.hdf5_loader import (
    get_attributes,
    list_datasets,
    load_hdf5,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


# Use this fixture to create real HDF5 files for testing
@pytest.fixture
def create_h5_file():
    """Create HDF5 files for testing."""

    def _create(path: Path, datasets: dict, root_attrs: dict | None = None) -> None:
        """Create an HDF5 file with specified datasets and attributes.

        Args:
            path: Path to create the file
            datasets: Dictionary mapping dataset paths to (data, attrs) tuples
            root_attrs: Optional root-level attributes
        """
        with h5py.File(path, "w") as f:
            if root_attrs:
                for key, value in root_attrs.items():
                    f.attrs[key] = value

            for ds_path, (data, attrs) in datasets.items():
                # Create groups if needed
                parts = ds_path.strip("/").split("/")
                if len(parts) > 1:
                    group_path = "/".join(parts[:-1])
                    if group_path not in f:
                        f.create_group(group_path)

                # Create dataset
                ds = f.create_dataset(ds_path, data=data)

                # Add attributes
                if attrs:
                    for key, value in attrs.items():
                        ds.attrs[key] = value

    return _create


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-006")
class TestLoadHDF5:
    """Test HDF5 file loading with load_hdf5 function."""

    @patch.object(hdf5_loader, "H5PY_AVAILABLE", False)
    def test_load_hdf5_without_h5py_raises_error(self, tmp_path: Path) -> None:
        """Test that loading without h5py raises LoaderError."""
        hdf5_path = tmp_path / "test.h5"
        hdf5_path.touch()

        with pytest.raises(LoaderError, match="HDF5 support not available"):
            load_hdf5(hdf5_path)

    def test_load_hdf5_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises LoaderError."""
        hdf5_path = tmp_path / "nonexistent.h5"

        with pytest.raises(LoaderError, match="File not found"):
            load_hdf5(hdf5_path)

    def test_load_hdf5_with_data_dataset(self, create_h5_file, tmp_path: Path) -> None:
        """Test loading HDF5 with standard 'data' dataset name."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.sin(np.linspace(0, 2 * np.pi, 1000))

        create_h5_file(
            hdf5_path,
            datasets={"/data": (sample_data, {"sample_rate": 1e6})},
        )

        trace = load_hdf5(hdf5_path)

        assert trace is not None
        assert len(trace.data) == 1000
        assert trace.metadata.sample_rate == 1e6
        assert trace.metadata.source_file == str(hdf5_path)
        assert np.allclose(trace.data, sample_data, rtol=1e-10)

    def test_load_hdf5_with_waveform_dataset(self, create_h5_file, tmp_path: Path) -> None:
        """Test loading HDF5 with 'waveform' dataset name."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.random.randn(500)

        create_h5_file(
            hdf5_path,
            datasets={"/waveform": (sample_data, {"sample_rate": 2e6})},
        )

        trace = load_hdf5(hdf5_path)

        assert len(trace.data) == 500
        assert trace.metadata.sample_rate == 2e6

    def test_load_hdf5_with_explicit_dataset(self, create_h5_file, tmp_path: Path) -> None:
        """Test loading HDF5 with explicitly specified dataset path."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.random.randn(500)

        create_h5_file(
            hdf5_path,
            datasets={
                "/measurements/ch1": (sample_data, {"sample_rate": 2e6}),
                "/measurements/ch2": (np.ones(100), {"sample_rate": 1e6}),
            },
        )

        trace = load_hdf5(hdf5_path, dataset="/measurements/ch1")

        assert len(trace.data) == 500
        assert trace.metadata.sample_rate == 2e6

    def test_load_hdf5_with_channel_parameter(self, create_h5_file, tmp_path: Path) -> None:
        """Test loading HDF5 using channel parameter (alias for dataset)."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        create_h5_file(
            hdf5_path,
            datasets={"/ch1": (sample_data, {"sample_rate": 1e6})},
        )

        trace = load_hdf5(hdf5_path, channel="ch1")

        assert len(trace.data) == 100
        assert trace.metadata.sample_rate == 1e6

    def test_load_hdf5_with_sample_rate_override(self, create_h5_file, tmp_path: Path) -> None:
        """Test loading HDF5 with sample rate override parameter."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.linspace(0, 1, 256)

        create_h5_file(
            hdf5_path,
            datasets={"/signal": (sample_data, {})},
        )

        trace = load_hdf5(hdf5_path, sample_rate=10e6)

        assert trace.metadata.sample_rate == 10e6

    def test_load_hdf5_multidimensional_data_flattened(
        self, create_h5_file, tmp_path: Path
    ) -> None:
        """Test that multidimensional data is flattened."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.random.randn(10, 10)

        create_h5_file(
            hdf5_path,
            datasets={"/data": (sample_data, {"sample_rate": 1e6})},
        )

        trace = load_hdf5(hdf5_path)

        # Data should be flattened
        assert trace.data.ndim == 1
        assert len(trace.data) == 100

    def test_load_hdf5_with_vertical_scale_and_offset(self, create_h5_file, tmp_path: Path) -> None:
        """Test loading HDF5 with vertical scale and offset attributes."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.zeros(50)

        create_h5_file(
            hdf5_path,
            datasets={
                "/data": (
                    sample_data,
                    {
                        "sample_rate": 1e6,
                        "vertical_scale": 0.5,
                        "vertical_offset": 1.2,
                    },
                )
            },
        )

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.vertical_scale == 0.5
        assert trace.metadata.vertical_offset == 1.2

    def test_load_hdf5_with_channel_name_attribute(self, create_h5_file, tmp_path: Path) -> None:
        """Test loading HDF5 with channel_name attribute."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        create_h5_file(
            hdf5_path,
            datasets={
                "/data": (
                    sample_data,
                    {"sample_rate": 1e6, "channel_name": "Channel 1"},
                )
            },
        )

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.channel == "Channel 1"

    def test_load_hdf5_channel_name_from_dataset_path(self, create_h5_file, tmp_path: Path) -> None:
        """Test that channel name defaults to dataset name from path."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        create_h5_file(
            hdf5_path,
            datasets={"/measurements/my_channel": (sample_data, {"sample_rate": 1e6})},
        )

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.channel == "my_channel"

    def test_load_hdf5_default_sample_rate(self, create_h5_file, tmp_path: Path) -> None:
        """Test that default sample rate is used when not found in attributes."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        create_h5_file(
            hdf5_path,
            datasets={"/data": (sample_data, {})},
        )

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.sample_rate == 1e6  # Default

    def test_load_hdf5_dataset_not_found_raises_error(self, create_h5_file, tmp_path: Path) -> None:
        """Test that requesting nonexistent dataset raises FormatError."""
        hdf5_path = tmp_path / "test.h5"

        create_h5_file(
            hdf5_path,
            datasets={"/data": (np.ones(10), {})},
        )

        with pytest.raises(FormatError, match="Dataset not found"):
            load_hdf5(hdf5_path, dataset="nonexistent")

    def test_load_hdf5_no_waveform_data_raises_error(self, create_h5_file, tmp_path: Path) -> None:
        """Test that HDF5 without waveform data raises FormatError."""
        hdf5_path = tmp_path / "test.h5"

        # Create file with non-standard dataset name
        create_h5_file(
            hdf5_path,
            datasets={"/metadata/info": (np.array([1, 2, 3]), {})},
        )

        with pytest.raises(FormatError, match="No waveform data found"):
            load_hdf5(hdf5_path)

    def test_load_hdf5_sample_rate_from_dt_attribute(self, create_h5_file, tmp_path: Path) -> None:
        """Test sample rate extraction from dt (sample_interval) attribute."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        # dt = 1e-6 means sample_rate = 1e6
        create_h5_file(
            hdf5_path,
            datasets={"/data": (sample_data, {"dt": 1e-6})},
        )

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.sample_rate == 1e6

    def test_load_hdf5_sample_rate_from_parent_group(self, create_h5_file, tmp_path: Path) -> None:
        """Test sample rate extraction from parent group attributes."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        # Create file with sample_rate on parent group
        with h5py.File(hdf5_path, "w") as f:
            grp = f.create_group("measurements")
            grp.attrs["sample_rate"] = 5e6
            grp.create_dataset("data", data=sample_data)

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.sample_rate == 5e6

    def test_load_hdf5_sample_rate_from_root_attributes(
        self, create_h5_file, tmp_path: Path
    ) -> None:
        """Test sample rate extraction from root file attributes."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        create_h5_file(
            hdf5_path,
            datasets={"/data": (sample_data, {})},
            root_attrs={"fs": 8e6},
        )

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.sample_rate == 8e6

    def test_load_hdf5_sample_rate_from_metadata_group(
        self, create_h5_file, tmp_path: Path
    ) -> None:
        """Test sample rate extraction from metadata group."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        # Create file with metadata group
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("data", data=sample_data)
            meta = f.create_group("metadata")
            meta.attrs["rate"] = 3e6

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.sample_rate == 3e6

    def test_load_hdf5_bytes_attribute_decoded(self, create_h5_file, tmp_path: Path) -> None:
        """Test that bytes attributes are properly decoded to strings."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        # Create file with bytes attribute
        with h5py.File(hdf5_path, "w") as f:
            ds = f.create_dataset("data", data=sample_data)
            ds.attrs["sample_rate"] = 1e6
            ds.attrs["name"] = b"Test Channel"

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.channel == "Test Channel"

    def test_load_hdf5_alternative_attribute_names(self, create_h5_file, tmp_path: Path) -> None:
        """Test that alternative attribute names are recognized."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        # Use alternative names
        create_h5_file(
            hdf5_path,
            datasets={
                "/data": (
                    sample_data,
                    {
                        "samplerate": 5e6,  # Alternative to sample_rate
                        "v_scale": 0.5,  # Alternative to vertical_scale
                        "v_offset": 1.0,  # Alternative to vertical_offset
                    },
                )
            },
        )

        trace = load_hdf5(hdf5_path)

        assert trace.metadata.sample_rate == 5e6
        assert trace.metadata.vertical_scale == 0.5
        assert trace.metadata.vertical_offset == 1.0

    def test_load_hdf5_find_dataset_by_partial_name(self, create_h5_file, tmp_path: Path) -> None:
        """Test finding dataset by partial name match."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        create_h5_file(
            hdf5_path,
            datasets={"/measurements/channel_1_data": (sample_data, {"sample_rate": 1e6})},
        )

        # Should find by partial match
        trace = load_hdf5(hdf5_path, dataset="channel_1")

        assert trace is not None
        assert len(trace.data) == 100

    def test_load_hdf5_case_insensitive_dataset_search(
        self, create_h5_file, tmp_path: Path
    ) -> None:
        """Test finding dataset with case-insensitive search."""
        hdf5_path = tmp_path / "test.h5"
        sample_data = np.ones(100)

        create_h5_file(
            hdf5_path,
            datasets={"/MyChannel": (sample_data, {"sample_rate": 1e6})},
        )

        # Should find with different case
        trace = load_hdf5(hdf5_path, dataset="mychannel")

        assert trace is not None
        assert len(trace.data) == 100

    def test_load_hdf5_multiple_common_names(self, create_h5_file, tmp_path: Path) -> None:
        """Test auto-detection with various common dataset names."""
        common_names = ["signal", "samples", "voltage", "trace", "ch1", "analog"]

        for name in common_names:
            hdf5_path = tmp_path / f"test_{name}.h5"
            sample_data = np.ones(50)

            create_h5_file(
                hdf5_path,
                datasets={f"/{name}": (sample_data, {"sample_rate": 1e6})},
            )

            trace = load_hdf5(hdf5_path)

            assert trace is not None
            assert len(trace.data) == 50


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-006")
class TestListDatasets:
    """Test list_datasets function."""

    @patch.object(hdf5_loader, "H5PY_AVAILABLE", False)
    def test_list_datasets_without_h5py_raises_error(self, tmp_path: Path) -> None:
        """Test that listing without h5py raises LoaderError."""
        hdf5_path = tmp_path / "test.h5"
        hdf5_path.touch()

        with pytest.raises(LoaderError, match="HDF5 support not available"):
            list_datasets(hdf5_path)

    def test_list_datasets_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that listing nonexistent file raises LoaderError."""
        hdf5_path = tmp_path / "nonexistent.h5"

        with pytest.raises(LoaderError, match="File not found"):
            list_datasets(hdf5_path)

    def test_list_datasets_empty_file(self, create_h5_file, tmp_path: Path) -> None:
        """Test listing datasets in empty HDF5 file."""
        hdf5_path = tmp_path / "empty.h5"

        # Create empty file
        with h5py.File(hdf5_path, "w"):
            pass

        datasets = list_datasets(hdf5_path)

        assert datasets == []

    def test_list_datasets_with_multiple_datasets(self, create_h5_file, tmp_path: Path) -> None:
        """Test listing multiple datasets in HDF5 file."""
        hdf5_path = tmp_path / "multi.h5"

        create_h5_file(
            hdf5_path,
            datasets={
                "/measurements/ch1": (np.ones(10), {}),
                "/measurements/ch2": (np.ones(10), {}),
                "/time": (np.ones(10), {}),
            },
        )

        datasets = list_datasets(hdf5_path)

        assert len(datasets) == 3
        assert "/measurements/ch1" in datasets
        assert "/measurements/ch2" in datasets
        assert "/time" in datasets


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-006")
class TestGetAttributes:
    """Test get_attributes function."""

    @patch.object(hdf5_loader, "H5PY_AVAILABLE", False)
    def test_get_attributes_without_h5py_raises_error(self, tmp_path: Path) -> None:
        """Test that getting attributes without h5py raises LoaderError."""
        hdf5_path = tmp_path / "test.h5"
        hdf5_path.touch()

        with pytest.raises(LoaderError, match="HDF5 support not available"):
            get_attributes(hdf5_path)

    def test_get_attributes_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that getting attributes from nonexistent file raises LoaderError."""
        hdf5_path = tmp_path / "nonexistent.h5"

        with pytest.raises(LoaderError, match="File not found"):
            get_attributes(hdf5_path)

    def test_get_attributes_root_attributes(self, create_h5_file, tmp_path: Path) -> None:
        """Test getting root attributes from HDF5 file."""
        hdf5_path = tmp_path / "test.h5"

        create_h5_file(
            hdf5_path,
            datasets={"/data": (np.ones(10), {})},
            root_attrs={"version": "1.0", "sample_rate": 1e6},
        )

        attrs = get_attributes(hdf5_path)

        assert attrs["version"] == "1.0"
        assert attrs["sample_rate"] == 1e6

    def test_get_attributes_dataset_attributes(self, create_h5_file, tmp_path: Path) -> None:
        """Test getting attributes from specific dataset."""
        hdf5_path = tmp_path / "test.h5"

        create_h5_file(
            hdf5_path,
            datasets={"/data": (np.ones(10), {"sample_rate": 2e6, "units": "V"})},
        )

        attrs = get_attributes(hdf5_path, dataset="/data")

        assert attrs["sample_rate"] == 2e6
        assert attrs["units"] == "V"

    def test_get_attributes_bytes_decoded(self, create_h5_file, tmp_path: Path) -> None:
        """Test that bytes attributes are decoded to strings."""
        hdf5_path = tmp_path / "test.h5"

        # Create file with bytes attributes
        with h5py.File(hdf5_path, "w") as f:
            f.attrs["name"] = b"my_channel"
            f.attrs["description"] = b"Test data"

        attrs = get_attributes(hdf5_path)

        assert attrs["name"] == "my_channel"
        assert attrs["description"] == "Test data"

    def test_get_attributes_numpy_array_converted(self, create_h5_file, tmp_path: Path) -> None:
        """Test that numpy array attributes are converted to lists."""
        hdf5_path = tmp_path / "test.h5"

        # Create file with numpy array attribute
        with h5py.File(hdf5_path, "w") as f:
            f.attrs["calibration"] = np.array([1.0, 2.0, 3.0])

        attrs = get_attributes(hdf5_path)

        assert attrs["calibration"] == [1.0, 2.0, 3.0]
