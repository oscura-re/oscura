"""Centralized mocking utilities for Oscura tests.

This module provides reusable mocking utilities to reduce duplication
across the test suite, particularly for optional dependencies and
external file format loaders.
"""

import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch


@contextmanager
def mock_optional_module(
    module_name: str,
    *,
    available: bool = False,
    mock_attrs: dict[str, Any] | None = None,
) -> Generator[MagicMock | None, None, None]:
    """Context manager for mocking optional module imports.

    This utility handles the common pattern of testing code paths when
    optional dependencies are either available or unavailable.

    Args:
        module_name: The fully-qualified module name to mock (e.g., "pywt", "scipy.io").
        available: If True, provides a mock module. If False, simulates ImportError.
        mock_attrs: Optional dictionary of attributes to set on the mock module.

    Yields:
        MagicMock if available=True, None if available=False.

    Example:
        >>> with mock_optional_module("pywt", available=False):
        ...     # Test code path when PyWavelets is not installed
        ...     pass

        >>> with mock_optional_module("h5py", available=True, mock_attrs={"File": MagicMock()}):
        ...     # Test code path with mocked h5py
        ...     pass
    """
    original_modules: dict[str, Any] = {}

    # Save all related modules
    for key in list(sys.modules.keys()):
        if key == module_name or key.startswith(f"{module_name}."):
            original_modules[key] = sys.modules[key]

    try:
        # Remove existing modules
        for key in original_modules:
            del sys.modules[key]

        if available:
            # Create a mock module
            mock_module = MagicMock()
            if mock_attrs:
                for attr, value in mock_attrs.items():
                    setattr(mock_module, attr, value)
            sys.modules[module_name] = mock_module
            yield mock_module
        else:
            # Make import fail by setting to None (Python's convention)
            sys.modules[module_name] = None  # type: ignore[assignment]
            yield None

    finally:
        # Restore original modules
        # First, remove any mocked entries
        for key in list(sys.modules.keys()):
            if key == module_name or key.startswith(f"{module_name}."):
                if key in sys.modules:
                    del sys.modules[key]

        # Then restore originals
        sys.modules.update(original_modules)


class FakeModuleRaiseOnAccess:
    """A fake module that raises ImportError on any attribute access.

    Useful for testing graceful degradation when optional modules are
    partially available but broken.
    """

    def __init__(self, error_message: str = "Module not available"):
        self._error_message = error_message

    def __getattr__(self, name: str) -> Any:
        raise ImportError(self._error_message)


@contextmanager
def mock_module_broken(
    module_name: str, error_message: str | None = None
) -> Generator[None, None, None]:
    """Mock a module that raises ImportError on attribute access.

    This simulates the case where a module is installed but broken or
    has missing dependencies.

    Args:
        module_name: The module name to mock as broken.
        error_message: Custom error message for the ImportError.

    Example:
        >>> with mock_module_broken("oscura.optional_feature"):
        ...     # Test code handles broken module gracefully
        ...     pass
    """
    msg = error_message or f"{module_name} is not available"
    original_module = sys.modules.get(module_name)

    try:
        sys.modules[module_name] = FakeModuleRaiseOnAccess(msg)  # type: ignore[assignment]
        yield
    finally:
        if original_module is not None:
            sys.modules[module_name] = original_module
        elif module_name in sys.modules:
            del sys.modules[module_name]


def mock_rigol_wfm(
    *,
    num_channels: int = 1,
    sample_rate: float = 1e9,
    num_samples: int = 10000,
    time_offset: float = 0.0,
) -> MagicMock:
    """Create a mock for Rigol WFM file reader.

    This centralizes the common pattern of mocking Rigol oscilloscope
    waveform files for testing loaders.

    Args:
        num_channels: Number of channels in the mock waveform.
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples per channel.
        time_offset: Time offset for the waveform.

    Returns:
        A configured MagicMock that behaves like a Rigol WFM reader.

    Example:
        >>> mock_wfm = mock_rigol_wfm(num_channels=2, sample_rate=1e9)
        >>> with patch("oscura.loaders.rigol.RigolWfmReader", return_value=mock_wfm):
        ...     # Test Rigol loader
        ...     pass
    """
    import numpy as np

    mock = MagicMock()

    # Configure header info
    mock.header = MagicMock()
    mock.header.sample_rate = sample_rate
    mock.header.time_offset = time_offset
    mock.header.num_samples = num_samples

    # Configure channels
    channels = []
    for i in range(num_channels):
        channel = MagicMock()
        channel.name = f"CH{i + 1}"
        channel.enabled = True
        channel.scale = 1.0
        channel.offset = 0.0
        channel.probe_ratio = 1.0
        # Generate synthetic data
        t = np.linspace(0, num_samples / sample_rate, num_samples)
        channel.data = np.sin(2 * np.pi * 1e6 * t + i * np.pi / 4)
        channels.append(channel)

    mock.channels = channels
    mock.num_channels = num_channels

    return mock


def mock_vcd_parser(
    *,
    signals: dict[str, list[tuple[int, int]]] | None = None,
    timescale: str = "1ns",
) -> MagicMock:
    """Create a mock for VCD (Value Change Dump) parser.

    Args:
        signals: Dictionary mapping signal names to list of (time, value) tuples.
        timescale: VCD timescale string.

    Returns:
        A configured MagicMock that behaves like a VCD parser.
    """
    mock = MagicMock()

    if signals is None:
        signals = {
            "clk": [(0, 0), (5, 1), (10, 0), (15, 1), (20, 0)],
            "data": [(0, 0), (10, 1), (20, 0)],
        }

    mock.timescale = timescale
    mock.signals = signals
    mock.get_signal_names.return_value = list(signals.keys())

    def get_signal_changes(name: str) -> list[tuple[int, int]]:
        return signals.get(name, [])

    mock.get_signal_changes = MagicMock(side_effect=get_signal_changes)

    return mock


@contextmanager
def mock_file_content(
    file_path: str,
    content: bytes | str,
    *,
    binary: bool = True,
) -> Generator[None, None, None]:
    """Mock file read operations to return specific content.

    Useful for testing file loaders without creating actual test files.

    Args:
        file_path: The path that will be mocked.
        content: The content to return when the file is read.
        binary: If True, mock binary read operations.

    Example:
        >>> with mock_file_content("/path/to/file.bin", b"\\x00\\x01\\x02"):
        ...     # Test file loader
        ...     pass
    """
    mock_open = MagicMock()

    if binary:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            content if isinstance(content, bytes) else content.encode()
        )
    else:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            content if isinstance(content, str) else content.decode()
        )

    with patch("builtins.open", mock_open):
        yield


def create_mock_waveform_data(
    num_samples: int = 1000,
    sample_rate: float = 1e6,
    frequency: float = 1e3,
    amplitude: float = 1.0,
    noise_level: float = 0.0,
) -> dict[str, Any]:
    """Create mock waveform data dictionary.

    Args:
        num_samples: Number of samples to generate.
        sample_rate: Sample rate in Hz.
        frequency: Signal frequency in Hz.
        amplitude: Signal amplitude.
        noise_level: Standard deviation of Gaussian noise to add.

    Returns:
        Dictionary with 'time', 'data', 'sample_rate', and metadata.
    """
    import numpy as np

    t = np.linspace(0, num_samples / sample_rate, num_samples)
    data = amplitude * np.sin(2 * np.pi * frequency * t)

    if noise_level > 0:
        rng = np.random.default_rng(42)
        data = data + rng.normal(0, noise_level, num_samples)

    return {
        "time": t,
        "data": data,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "frequency": frequency,
        "amplitude": amplitude,
    }
