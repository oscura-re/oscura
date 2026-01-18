"""Integration test fixtures.

This module provides fixtures for integration tests:
- End-to-end workflow fixtures
- Multi-module coordination fixtures
- Integration test data fixtures
- Validation fixtures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def integration_test_data_dir(test_data_dir: Path) -> Path:
    """Directory for integration test data.

    Args:
        test_data_dir: Root test data directory from root conftest.

    Returns:
        Path to integration test data directory.
    """
    integration_dir = test_data_dir / "integration"
    if not integration_dir.exists():
        pytest.skip("Integration test data directory not found")
    return integration_dir


# =============================================================================
# Workflow Configuration Fixtures
# =============================================================================


@pytest.fixture
def end_to_end_workflow() -> dict[str, bool]:
    """Configuration for end-to-end workflow testing.

    Returns:
        Dictionary with workflow stage flags.
    """
    return {
        "load": True,
        "preprocess": True,
        "analyze": True,
        "infer": True,
        "visualize": True,
        "export": True,
    }


@pytest.fixture
def workflow_config() -> dict[str, Any]:
    """Complete workflow configuration for integration tests.

    Returns:
        Dictionary with settings for each workflow stage.
    """
    return {
        "loader": {
            "type": "configurable",
            "validation": True,
            "preprocessing": True,
        },
        "analyzer": {
            "edge_detection": True,
            "pattern_matching": True,
            "statistical_analysis": True,
        },
        "inference": {
            "message_format": True,
            "state_machine": True,
            "protocol_library": True,
        },
        "visualization": {
            "waveform_plots": True,
            "protocol_diagrams": True,
            "export_format": "png",
        },
        "export": {
            "format": "json",
            "include_metadata": True,
            "compress": False,
        },
    }


# =============================================================================
# Test Scenario Fixtures
# =============================================================================


@pytest.fixture
def wfm_to_analysis_scenario() -> dict[str, Any]:
    """Scenario for WFM file to analysis integration test.

    Returns:
        Dictionary with input file and expected outputs.
    """
    return {
        "input_file": "formats/tektronix/analog/single_channel/test.wfm",
        "expected_channels": 1,
        "expected_duration": 0.01,  # 10 ms
        "analysis_tasks": [
            "edge_detection",
            "frequency_analysis",
            "quality_metrics",
        ],
        "validation": {
            "min_edges": 10,
            "max_edges": 10000,
            "min_snr_db": 20,
        },
    }


@pytest.fixture
def pcap_to_inference_scenario() -> dict[str, Any]:
    """Scenario for PCAP file to protocol inference integration test.

    Returns:
        Dictionary with input file and expected results.
    """
    return {
        "input_file": "formats/pcap/tcp/http/http.pcap",
        "expected_protocol": "HTTP",
        "expected_packets": 10,
        "inference_tasks": [
            "message_clustering",
            "field_detection",
            "protocol_matching",
        ],
        "validation": {
            "min_confidence": 0.8,
            "expected_fields": ["method", "uri", "version"],
        },
    }


@pytest.fixture
def binary_to_protocol_scenario() -> dict[str, Any]:
    """Scenario for binary data to protocol inference integration test.

    Returns:
        Dictionary with input data and expected protocol structure.
    """
    return {
        "input_file": "synthetic/can/synthetic/fixed_length/clean_packets_512b.bin",
        "packet_size": 512,
        "expected_structure": {
            "header": {"offset": 0, "length": 2, "value": b"\xaa\x55"},
            "sequence": {"offset": 2, "length": 2},
            "payload": {"offset": 4, "length": 506},
            "checksum": {"offset": 510, "length": 2},
        },
        "validation": {
            "packet_count": 100,
            "sequential": True,
            "checksum_valid": True,
        },
    }


# =============================================================================
# Multi-Module Integration Fixtures
# =============================================================================


@pytest.fixture
def loader_analyzer_chain():
    """Configuration for loader → analyzer integration.

    Returns:
        Function that creates loader-analyzer chain.
    """

    def _create_chain(loader_type: str, analyzer_type: str) -> dict[str, Any]:
        """Create loader-analyzer processing chain.

        Args:
            loader_type: Type of loader ('wfm', 'pcap', 'binary').
            analyzer_type: Type of analyzer ('digital', 'spectral', 'statistical').

        Returns:
            Chain configuration dictionary.
        """
        return {
            "loader": {
                "type": loader_type,
                "options": {
                    "validate": True,
                    "preprocess": True,
                },
            },
            "analyzer": {
                "type": analyzer_type,
                "options": {
                    "auto_detect": True,
                },
            },
            "data_flow": {
                "buffer_size": 1_000_000,
                "streaming": False,
            },
        }

    return _create_chain


@pytest.fixture
def analyzer_inference_chain():
    """Configuration for analyzer → inference integration.

    Returns:
        Function that creates analyzer-inference chain.
    """

    def _create_chain(analyzer_output: str, inference_type: str) -> dict[str, Any]:
        """Create analyzer-inference processing chain.

        Args:
            analyzer_output: Output type from analyzer ('edges', 'patterns', 'statistics').
            inference_type: Type of inference ('message_format', 'state_machine').

        Returns:
            Chain configuration dictionary.
        """
        return {
            "analyzer_output": analyzer_output,
            "inference": {
                "type": inference_type,
                "options": {
                    "min_confidence": 0.8,
                    "max_iterations": 100,
                },
            },
            "data_flow": {
                "batch_size": 100,
                "parallel": False,
            },
        }

    return _create_chain


# =============================================================================
# Validation Fixtures
# =============================================================================


@pytest.fixture
def integration_validation_rules() -> dict[str, Any]:
    """Validation rules for integration test results.

    Returns:
        Dictionary with validation criteria.
    """
    return {
        "data_integrity": {
            "check_checksums": True,
            "check_sequence_numbers": True,
            "check_timestamps": True,
        },
        "analysis_quality": {
            "min_snr_db": 20.0,
            "max_error_rate": 0.01,
            "min_coverage": 0.95,
        },
        "inference_quality": {
            "min_confidence": 0.8,
            "min_field_coverage": 0.9,
            "max_false_positives": 0.05,
        },
        "performance": {
            "max_processing_time": 10.0,  # seconds
            "max_memory_mb": 1000,
        },
    }


@pytest.fixture
def expected_outputs() -> dict[str, Any]:
    """Expected outputs for integration test validation.

    Returns:
        Dictionary mapping test scenario to expected results.
    """
    return {
        "wfm_to_analysis": {
            "edges": {"min": 10, "max": 10000},
            "frequency": {"min_hz": 100, "max_hz": 1e6},
            "snr_db": {"min": 20},
        },
        "pcap_to_inference": {
            "protocols": ["TCP", "HTTP"],
            "packet_count": {"min": 5, "max": 1000},
            "fields": ["src_ip", "dst_ip", "method", "uri"],
        },
        "binary_to_protocol": {
            "packet_size": 512,
            "packet_count": 100,
            "field_boundaries": [(0, 2), (2, 4), (4, 510), (510, 512)],
        },
    }


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def integration_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for integration test outputs.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to integration test output directory.
    """
    output_dir = tmp_path / "integration_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def integration_report_dir(tmp_path: Path) -> Path:
    """Temporary directory for integration test reports.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to integration test report directory.
    """
    report_dir = tmp_path / "integration_reports"
    report_dir.mkdir()
    return report_dir


# =============================================================================
# Configuration File Fixtures
# =============================================================================


@pytest.fixture
def config_driven_workflow(project_root: Path) -> dict[str, Path]:
    """Configuration files for config-driven workflow tests.

    Args:
        project_root: Project root directory from root conftest.

    Returns:
        Dictionary mapping config type to config file path.
    """
    config_dir = project_root / "examples" / "configs"
    return {
        "packet_format": config_dir / "packet_format_example.yaml",
        "device_mapping": config_dir / "device_mapping_example.yaml",
        "bus_config": config_dir / "bus_config_example.yaml",
        "protocol_definition": config_dir / "protocol_definition_example.yaml",
    }


# =============================================================================
# Mock External Service Fixtures
# =============================================================================


@pytest.fixture
def mock_protocol_library():
    """Mock protocol library service for integration tests.

    Returns:
        Mock service that provides protocol definitions.
    """

    class MockProtocolLibrary:
        """Mock protocol library service."""

        def __init__(self):
            self.protocols = {
                "HTTP": {
                    "name": "HTTP",
                    "fields": ["method", "uri", "version", "headers", "body"],
                },
                "MODBUS": {
                    "name": "MODBUS RTU",
                    "fields": ["address", "function", "data", "crc"],
                },
            }

        def lookup(self, protocol_name: str) -> dict[str, Any] | None:
            """Look up protocol definition.

            Args:
                protocol_name: Name of protocol.

            Returns:
                Protocol definition or None if not found.
            """
            return self.protocols.get(protocol_name)

        def search(self, pattern: bytes) -> list[str]:
            """Search for protocols matching pattern.

            Args:
                pattern: Binary pattern to match.

            Returns:
                List of matching protocol names.
            """
            # Simple mock: return all protocols
            return list(self.protocols.keys())

    return MockProtocolLibrary()


# =============================================================================
# Data Pipeline Fixtures
# =============================================================================


@pytest.fixture
def data_pipeline_config() -> dict[str, Any]:
    """Configuration for data processing pipeline.

    Returns:
        Pipeline configuration with stages and parameters.
    """
    return {
        "pipeline": [
            {
                "stage": "load",
                "module": "loaders",
                "function": "load_file",
                "options": {"validate": True},
            },
            {
                "stage": "preprocess",
                "module": "loaders.preprocessing",
                "function": "remove_idle",
                "options": {"threshold": 0.1},
            },
            {
                "stage": "analyze",
                "module": "analyzers.digital",
                "function": "detect_edges",
                "options": {"threshold": 1.65},
            },
            {
                "stage": "infer",
                "module": "inference",
                "function": "infer_protocol",
                "options": {"min_confidence": 0.8},
            },
            {
                "stage": "export",
                "module": "exporters",
                "function": "export_json",
                "options": {"pretty": True},
            },
        ],
        "error_handling": {
            "stop_on_error": False,
            "log_errors": True,
            "retry_failed": False,
        },
    }


# =============================================================================
# Regression Testing Fixtures
# =============================================================================


@pytest.fixture
def integration_baseline() -> dict[str, Any]:
    """Baseline results for integration test regression checking.

    Returns:
        Dictionary with baseline metrics.
    """
    return {
        "wfm_to_analysis": {
            "processing_time": 1.5,  # seconds
            "edge_count": 1234,
            "snr_db": 42.5,
        },
        "pcap_to_inference": {
            "processing_time": 2.0,
            "packet_count": 156,
            "confidence": 0.95,
        },
        "binary_to_protocol": {
            "processing_time": 0.5,
            "packet_count": 100,
            "field_accuracy": 0.98,
        },
    }


@pytest.fixture
def regression_tolerance() -> dict[str, float]:
    """Acceptable tolerance for regression testing.

    Returns:
        Dictionary with tolerance values.
    """
    return {
        "processing_time": 1.2,  # 20% slower allowed
        "accuracy": 0.02,  # 2% absolute difference
        "confidence": 0.05,  # 5% absolute difference
    }
