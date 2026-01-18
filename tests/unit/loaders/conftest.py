"""Loader-specific test fixtures.

This module provides fixtures for loader tests:
- File path fixtures for various loader types
- Mock file content fixtures
- Loader error scenario fixtures
- Sample file structure fixtures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# =============================================================================
# Error Scenario Fixtures
# =============================================================================


@pytest.fixture
def loader_error_scenarios() -> dict[str, bytes]:
    """Common error scenarios for loader tests.

    Returns:
        Dictionary mapping error type to binary content that triggers it.
    """
    return {
        "empty_file": b"",
        "truncated_header": b"\x00" * 10,
        "invalid_magic": b"\xff\xff\xff\xff",
        "corrupt_checksum": b"\xaa\x55\x00\x01" + b"\x00" * 60 + b"\xff\xff",
        "malformed_structure": b"\x00\x01" + b"\xaa" * 50,
        "invalid_version": b"\xaa\x55\xff\xff" + b"\x00" * 100,
    }


@pytest.fixture
def mock_file_handle(tmp_path: Path):
    """Create a mock file with configurable content.

    Returns:
        Factory function that creates files in tmp_path.

    Example:
        >>> file_path = mock_file_handle(b"\\xaa\\x55\\x01\\x02", "test.bin")
        >>> assert file_path.exists()
    """

    def _create(content: bytes, filename: str = "test.bin") -> Path:
        """Create file with specified content.

        Args:
            content: Binary content to write.
            filename: Name of file to create.

        Returns:
            Path to created file.
        """
        file_path = tmp_path / filename
        file_path.write_bytes(content)
        return file_path

    return _create


# =============================================================================
# WFM File Fixtures (moved from root conftest - candidates)
# =============================================================================
# NOTE: These fixtures currently exist in root conftest.py
# They are candidates for migration in Phase 2B:
# - tektronix_wfm_dir
# - wfm_files
# - invalid_wfm_files


# =============================================================================
# PCAP File Fixtures (moved from root conftest - candidates)
# =============================================================================
# NOTE: These fixtures currently exist in root conftest.py
# They are candidates for migration in Phase 2B:
# - pcap_dir
# - pcap_files
# - http_pcap
# - modbus_pcap
# - mqtt_pcap_files


# =============================================================================
# Sigrok File Fixtures (moved from root conftest - candidates)
# =============================================================================
# NOTE: These fixtures currently exist in root conftest.py
# They are candidates for migration in Phase 2B:
# - sigrok_dir
# - sigrok_files
# - uart_sigrok_files
# - uart_hello_world_files


# =============================================================================
# Binary Packet Fixtures
# =============================================================================


@pytest.fixture
def simple_binary_packet() -> bytes:
    """Simple binary packet for basic loader tests.

    Format: [header:2][length:2][payload:variable][checksum:2]
    """
    header = b"\xaa\x55"
    payload = b"\x01\x02\x03\x04"
    length = len(payload).to_bytes(2, "big")
    # Simple XOR checksum
    checksum = (sum(payload) & 0xFF).to_bytes(1, "big") * 2
    return header + length + payload + checksum


@pytest.fixture
def malformed_packets() -> dict[str, bytes]:
    """Collection of malformed packets for error handling tests.

    Returns:
        Dictionary mapping error type to malformed packet data.
    """
    return {
        "truncated_payload": b"\xaa\x55\x00\x04\x01\x02",  # Claims 4 bytes, only 2
        "invalid_header": b"\xff\xff\x00\x04\x01\x02\x03\x04",
        "checksum_mismatch": b"\xaa\x55\x00\x04\x01\x02\x03\x04\xff\xff",
        "zero_length": b"\xaa\x55\x00\x00",
        "negative_length": b"\xaa\x55\xff\xff",
    }


# =============================================================================
# CSV Loader Fixtures
# =============================================================================


@pytest.fixture
def sample_csv_content() -> str:
    """Sample CSV content for CSV loader tests.

    Format: timestamp,channel,value
    """
    return """timestamp,channel,value
0.000000,CH1,0.0
0.000001,CH1,1.0
0.000002,CH1,0.0
0.000003,CH1,1.0
0.000000,CH2,0.5
0.000001,CH2,0.7
0.000002,CH2,0.3
0.000003,CH2,0.9
"""


@pytest.fixture
def malformed_csv_content() -> dict[str, str]:
    """Malformed CSV content for error handling tests."""
    return {
        "missing_header": "0.0,CH1,1.0\n0.1,CH1,0.0",
        "inconsistent_columns": "time,ch,val\n0.0,CH1\n0.1,CH1,0.0,extra",
        "invalid_numeric": "time,ch,val\n0.0,CH1,not_a_number",
        "empty_file": "",
        "header_only": "timestamp,channel,value\n",
    }


# =============================================================================
# HDF5 Loader Fixtures
# =============================================================================


@pytest.fixture
def hdf5_structure() -> dict[str, Any]:
    """Expected HDF5 structure for loader tests.

    Returns:
        Dictionary describing expected groups and datasets.
    """
    return {
        "groups": {
            "waveforms": {
                "attributes": {"sample_rate": 1e6, "duration": 0.01},
                "datasets": {
                    "CH1": {"shape": (10000,), "dtype": "float64"},
                    "CH2": {"shape": (10000,), "dtype": "float64"},
                },
            },
            "metadata": {
                "datasets": {
                    "acquisition_time": {"shape": (), "dtype": "S32"},
                    "instrument": {"shape": (), "dtype": "S64"},
                },
            },
        },
    }


# =============================================================================
# TDMS Loader Fixtures
# =============================================================================


@pytest.fixture
def tdms_metadata() -> dict[str, Any]:
    """Sample TDMS metadata for loader tests."""
    return {
        "root": {
            "name": "Root",
            "properties": {
                "description": "Test TDMS file",
            },
        },
        "groups": [
            {
                "name": "Group1",
                "channels": [
                    {
                        "name": "Channel1",
                        "properties": {
                            "unit_string": "Volts",
                            "wf_increment": 1e-6,
                        },
                    },
                ],
            },
        ],
    }


# =============================================================================
# Lazy Loading Fixtures
# =============================================================================


@pytest.fixture
def large_file_metadata() -> dict[str, Any]:
    """Metadata for large file lazy loading tests.

    Used to test lazy loading without creating actual large files.
    """
    return {
        "total_size": 1_000_000_000,  # 1 GB
        "chunk_size": 1_048_576,  # 1 MB
        "num_chunks": 954,
        "sample_rate": 1e9,
        "duration": 1.0,
    }


@pytest.fixture
def chunked_data_factory():
    """Factory for creating chunked data for lazy loading tests.

    Returns:
        Function that generates data chunks on demand.
    """

    def _generate_chunk(chunk_index: int, chunk_size: int = 1024) -> bytes:
        """Generate a data chunk for testing.

        Args:
            chunk_index: Index of chunk to generate.
            chunk_size: Size of chunk in bytes.

        Returns:
            Binary data chunk.
        """
        # Generate deterministic data based on chunk index
        import struct

        data = bytearray()
        for i in range(chunk_size // 4):
            value = (chunk_index * 1000 + i) & 0xFFFFFFFF
            data.extend(struct.pack("<I", value))
        return bytes(data)

    return _generate_chunk


# =============================================================================
# Rigol Loader Fixtures
# =============================================================================


@pytest.fixture
def rigol_waveform_header() -> dict[str, Any]:
    """Sample Rigol waveform file header structure."""
    return {
        "magic": b"RGLW",
        "version": 1,
        "sample_rate": 1e9,
        "num_samples": 1000,
        "vertical_scale": 1.0,
        "vertical_offset": 0.0,
        "time_scale": 1e-6,
        "time_offset": 0.0,
    }


# =============================================================================
# Configurable Loader Fixtures
# =============================================================================


@pytest.fixture
def packet_format_config() -> dict[str, Any]:
    """Sample packet format configuration for configurable loader tests.

    Provides a standard packet format configuration structure for testing.
    """
    return {
        "packet_format": {
            "header": {
                "sync_bytes": "0xaa55",
                "length": 2,
                "endianness": "big",
            },
            "fields": [
                {
                    "name": "sequence",
                    "type": "uint16",
                    "endianness": "big",
                },
                {
                    "name": "payload_length",
                    "type": "uint8",
                },
                {
                    "name": "payload",
                    "type": "bytes",
                    "length_field": "payload_length",
                },
            ],
            "footer": {
                "checksum": {
                    "type": "crc16",
                    "polynomial": "0x8005",
                },
            },
        },
    }


@pytest.fixture
def validation_rules() -> dict[str, Any]:
    """Sample validation rules for configurable loader tests."""
    return {
        "rules": [
            {
                "field": "sequence",
                "validation": "sequential",
                "allow_gaps": False,
            },
            {
                "field": "payload_length",
                "validation": "range",
                "min": 1,
                "max": 255,
            },
            {
                "field": "checksum",
                "validation": "verify",
            },
        ],
    }
