import pytest

"""Comprehensive unit tests for src/oscura/core/provenance.py

Tests coverage for:

Covers all public classes, functions, properties, and methods with
edge cases, error handling, and validation.
"""

import json

import numpy as np

from oscura.core.provenance import (
    MeasurementResultWithProvenance,
    Provenance,
    compute_input_hash,
    create_provenance,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ==============================================================================
# Provenance Class Tests
# ==============================================================================


class TestProvenanceCreation:
    """Test Provenance dataclass creation and defaults."""

    def test_create_minimal(self) -> None:
        """Test creating Provenance with only required fields."""
        prov = Provenance(algorithm="test_algorithm")
        assert prov.algorithm == "test_algorithm"
        assert prov.parameters == {}
        assert prov.library_version == "0.6.0"
        assert prov.input_hash is None
        assert prov.metadata == {}
        # timestamp should be auto-generated
        assert prov.timestamp is not None
        assert isinstance(prov.timestamp, str)

    def test_create_with_parameters(self) -> None:
        """Test creating Provenance with parameters."""
        params = {"ref_levels": (10, 90), "threshold": 1.5}
        prov = Provenance(algorithm="rise_time", parameters=params)
        assert prov.parameters == params

    def test_create_with_custom_timestamp(self) -> None:
        """Test creating Provenance with custom timestamp."""
        timestamp = "2025-12-21T10:30:00Z"
        prov = Provenance(algorithm="test", timestamp=timestamp)
        assert prov.timestamp == timestamp

    def test_create_with_input_hash(self) -> None:
        """Test creating Provenance with input hash."""
        hash_val = "abc123def456"
        prov = Provenance(algorithm="test", input_hash=hash_val)
        assert prov.input_hash == hash_val

    def test_create_with_metadata(self) -> None:
        """Test creating Provenance with metadata."""
        metadata = {"source": "oscilloscope", "channel": 1}
        prov = Provenance(algorithm="test", metadata=metadata)
        assert prov.metadata == metadata

    def test_create_full(self) -> None:
        """Test creating Provenance with all fields."""
        prov = Provenance(
            algorithm="peak_to_peak",
            parameters={"window": (0, 1e-3)},
            timestamp="2025-12-21T10:30:00Z",
            library_version="0.6.0",
            input_hash="abc123",
            metadata={"channel": 1},
        )
        assert prov.algorithm == "peak_to_peak"
        assert prov.parameters == {"window": (0, 1e-3)}
        assert prov.timestamp == "2025-12-21T10:30:00Z"
        assert prov.library_version == "0.6.0"
        assert prov.input_hash == "abc123"
        assert prov.metadata == {"channel": 1}


class TestProvenanceMethods:
    """Test Provenance methods."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict() with minimal provenance."""
        prov = Provenance(algorithm="test", timestamp="2025-12-21T10:30:00Z")
        result = prov.to_dict()
        assert result == {
            "algorithm": "test",
            "parameters": {},
            "timestamp": "2025-12-21T10:30:00Z",
            "library_version": "0.6.0",
            "input_hash": None,
            "metadata": {},
        }

    def test_to_dict_full(self) -> None:
        """Test to_dict() with all fields."""
        prov = Provenance(
            algorithm="fft",
            parameters={"window": "hann", "n": 1024},
            timestamp="2025-12-21T10:30:00Z",
            library_version="0.6.0",
            input_hash="abc123",
            metadata={"source": "test"},
        )
        result = prov.to_dict()
        assert result["algorithm"] == "fft"
        assert result["parameters"] == {"window": "hann", "n": 1024}
        assert result["input_hash"] == "abc123"
        assert result["metadata"] == {"source": "test"}

    def test_to_dict_serializable(self) -> None:
        """Test that to_dict() output is JSON serializable."""
        prov = Provenance(
            algorithm="test",
            parameters={"value": 1.5},
            timestamp="2025-12-21T10:30:00Z",
        )
        result = prov.to_dict()
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_from_dict_minimal(self) -> None:
        """Test from_dict() with minimal data."""
        data = {"algorithm": "test"}
        prov = Provenance.from_dict(data)
        assert prov.algorithm == "test"
        assert prov.parameters == {}
        assert prov.timestamp == ""
        assert prov.library_version == "0.6.0"
        assert prov.input_hash is None
        assert prov.metadata == {}

    def test_from_dict_full(self) -> None:
        """Test from_dict() with all fields."""
        data = {
            "algorithm": "rise_time",
            "parameters": {"ref_levels": (10, 90)},
            "timestamp": "2025-12-21T10:30:00Z",
            "library_version": "0.6.0",
            "input_hash": "abc123",
            "metadata": {"channel": 1},
        }
        prov = Provenance.from_dict(data)
        assert prov.algorithm == "rise_time"
        assert prov.parameters == {"ref_levels": (10, 90)}
        assert prov.timestamp == "2025-12-21T10:30:00Z"
        assert prov.library_version == "0.6.0"
        assert prov.input_hash == "abc123"
        assert prov.metadata == {"channel": 1}

    def test_from_dict_round_trip(self) -> None:
        """Test that to_dict() -> from_dict() preserves data."""
        original = Provenance(
            algorithm="fft",
            parameters={"window": "hann"},
            timestamp="2025-12-21T10:30:00Z",
            input_hash="abc123",
        )
        data = original.to_dict()
        restored = Provenance.from_dict(data)
        assert restored.algorithm == original.algorithm
        assert restored.parameters == original.parameters
        assert restored.timestamp == original.timestamp
        assert restored.input_hash == original.input_hash

    def test_str_minimal(self) -> None:
        """Test __str__() with minimal provenance."""
        prov = Provenance(
            algorithm="test",
            timestamp="2025-12-21T10:30:00Z",
            library_version="0.6.0",
        )
        result = str(prov)
        assert "Algorithm: test" in result
        assert "Timestamp: 2025-12-21T10:30:00Z" in result
        assert "Version: 0.6.0" in result

    def test_str_with_parameters(self) -> None:
        """Test __str__() with parameters."""
        prov = Provenance(
            algorithm="fft",
            parameters={"window": "hann", "n": 1024},
            timestamp="2025-12-21T10:30:00Z",
        )
        result = str(prov)
        assert "Parameters:" in result
        assert "window=hann" in result
        assert "n=1024" in result

    def test_str_with_input_hash(self) -> None:
        """Test __str__() with input hash."""
        long_hash = "a" * 64  # SHA-256 hash length
        prov = Provenance(algorithm="test", timestamp="2025-12-21T10:30:00Z", input_hash=long_hash)
        result = str(prov)
        assert "Input Hash:" in result
        # Should truncate to first 16 chars
        assert long_hash[:16] in result
        assert "..." in result


# ==============================================================================
# MeasurementResultWithProvenance Class Tests
# ==============================================================================


class TestMeasurementResultCreation:
    """Test MeasurementResultWithProvenance creation."""

    def test_create_minimal(self) -> None:
        """Test creating result with only value."""
        result = MeasurementResultWithProvenance(value=3.3)
        assert result.value == 3.3
        assert result.units is None
        assert result.provenance is None
        assert result.confidence is None

    def test_create_with_units(self) -> None:
        """Test creating result with units."""
        result = MeasurementResultWithProvenance(value=5.0, units="V")
        assert result.value == 5.0
        assert result.units == "V"

    def test_create_with_provenance(self) -> None:
        """Test creating result with provenance."""
        prov = Provenance(algorithm="vpp")
        result = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov)
        assert result.provenance == prov
        assert result.provenance.algorithm == "vpp"

    def test_create_with_confidence(self) -> None:
        """Test creating result with confidence interval."""
        result = MeasurementResultWithProvenance(value=3.3, units="V", confidence=(3.2, 3.4))
        assert result.confidence == (3.2, 3.4)

    def test_create_full(self) -> None:
        """Test creating result with all fields."""
        prov = Provenance(algorithm="peak_to_peak", parameters={"window": (0, 1e-3)})
        result = MeasurementResultWithProvenance(
            value=3.3, units="V", provenance=prov, confidence=(3.2, 3.4)
        )
        assert result.value == 3.3
        assert result.units == "V"
        assert result.provenance == prov
        assert result.confidence == (3.2, 3.4)


class TestMeasurementResultIsEquivalent:
    """Test MeasurementResultWithProvenance.is_equivalent() method."""

    def test_equivalent_same_value(self) -> None:
        """Test equivalence with same value."""
        result1 = MeasurementResultWithProvenance(value=3.3, units="V")
        result2 = MeasurementResultWithProvenance(value=3.3, units="V")
        assert result1.is_equivalent(result2)

    def test_equivalent_within_tolerance(self) -> None:
        """Test equivalence within tolerance."""
        result1 = MeasurementResultWithProvenance(value=3.3, units="V")
        result2 = MeasurementResultWithProvenance(value=3.3000001, units="V")
        assert result1.is_equivalent(result2, rtol=1e-6)

    def test_not_equivalent_outside_tolerance(self) -> None:
        """Test not equivalent outside tolerance."""
        result1 = MeasurementResultWithProvenance(value=3.3, units="V")
        result2 = MeasurementResultWithProvenance(value=3.31, units="V")
        assert not result1.is_equivalent(result2, rtol=1e-6, atol=0.0)

    def test_not_equivalent_different_units(self) -> None:
        """Test not equivalent with different units."""
        result1 = MeasurementResultWithProvenance(value=3.3, units="V")
        result2 = MeasurementResultWithProvenance(value=3.3, units="A")
        assert not result1.is_equivalent(result2)

    def test_equivalent_no_units(self) -> None:
        """Test equivalence when both have no units."""
        result1 = MeasurementResultWithProvenance(value=3.3)
        result2 = MeasurementResultWithProvenance(value=3.3)
        assert result1.is_equivalent(result2)

    def test_equivalent_same_algorithm(self) -> None:
        """Test equivalence with same algorithm and parameters."""
        prov1 = Provenance(algorithm="vpp", parameters={"window": (0, 1e-3)})
        prov2 = Provenance(algorithm="vpp", parameters={"window": (0, 1e-3)})
        result1 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov1)
        result2 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov2)
        assert result1.is_equivalent(result2, check_parameters=True)

    def test_not_equivalent_different_algorithm(self) -> None:
        """Test not equivalent with different algorithm."""
        prov1 = Provenance(algorithm="vpp")
        prov2 = Provenance(algorithm="vrms")
        result1 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov1)
        result2 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov2)
        assert not result1.is_equivalent(result2, check_parameters=True)

    def test_not_equivalent_different_parameters(self) -> None:
        """Test not equivalent with different parameters."""
        prov1 = Provenance(algorithm="vpp", parameters={"window": (0, 1e-3)})
        prov2 = Provenance(algorithm="vpp", parameters={"window": (0, 2e-3)})
        result1 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov1)
        result2 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov2)
        assert not result1.is_equivalent(result2, check_parameters=True)

    def test_equivalent_ignore_parameters(self) -> None:
        """Test equivalence when ignoring parameter check."""
        prov1 = Provenance(algorithm="vpp", parameters={"window": (0, 1e-3)})
        prov2 = Provenance(algorithm="vrms", parameters={"window": (0, 2e-3)})
        result1 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov1)
        result2 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov2)
        assert result1.is_equivalent(result2, check_parameters=False)

    def test_equivalent_no_provenance(self) -> None:
        """Test equivalence when neither has provenance."""
        result1 = MeasurementResultWithProvenance(value=3.3, units="V")
        result2 = MeasurementResultWithProvenance(value=3.3, units="V")
        assert result1.is_equivalent(result2, check_parameters=True)

    def test_equivalent_one_provenance_missing(self) -> None:
        """Test equivalence when one has provenance, other doesn't."""
        prov = Provenance(algorithm="vpp")
        result1 = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov)
        result2 = MeasurementResultWithProvenance(value=3.3, units="V")
        # Should still be equivalent (provenance check skipped if either is None)
        assert result1.is_equivalent(result2, check_parameters=True)

    def test_equivalent_with_absolute_tolerance(self) -> None:
        """Test equivalence using absolute tolerance."""
        result1 = MeasurementResultWithProvenance(value=3.3, units="V")
        result2 = MeasurementResultWithProvenance(value=3.35, units="V")
        assert result1.is_equivalent(result2, rtol=0.0, atol=0.1)
        assert not result1.is_equivalent(result2, rtol=0.0, atol=0.01)


class TestMeasurementResultMethods:
    """Test MeasurementResultWithProvenance methods."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict() with minimal result."""
        result = MeasurementResultWithProvenance(value=3.3)
        data = result.to_dict()
        assert data == {"value": 3.3, "units": None}

    def test_to_dict_with_units(self) -> None:
        """Test to_dict() with units."""
        result = MeasurementResultWithProvenance(value=3.3, units="V")
        data = result.to_dict()
        assert data["value"] == 3.3
        assert data["units"] == "V"

    def test_to_dict_with_provenance(self) -> None:
        """Test to_dict() with provenance."""
        prov = Provenance(
            algorithm="vpp", parameters={"window": (0, 1e-3)}, timestamp="2025-12-21T10:30:00Z"
        )
        result = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov)
        data = result.to_dict()
        assert "provenance" in data
        assert data["provenance"]["algorithm"] == "vpp"
        assert data["provenance"]["parameters"] == {"window": (0, 1e-3)}

    def test_to_dict_with_confidence(self) -> None:
        """Test to_dict() with confidence interval."""
        result = MeasurementResultWithProvenance(value=3.3, units="V", confidence=(3.2, 3.4))
        data = result.to_dict()
        assert data["confidence"] == (3.2, 3.4)

    def test_to_dict_serializable(self) -> None:
        """Test that to_dict() output is JSON serializable."""
        prov = Provenance(algorithm="vpp", timestamp="2025-12-21T10:30:00Z")
        result = MeasurementResultWithProvenance(
            value=3.3, units="V", provenance=prov, confidence=(3.2, 3.4)
        )
        data = result.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_from_dict_minimal(self) -> None:
        """Test from_dict() with minimal data."""
        data = {"value": 3.3}
        result = MeasurementResultWithProvenance.from_dict(data)
        assert result.value == 3.3
        assert result.units is None
        assert result.provenance is None
        assert result.confidence is None

    def test_from_dict_with_units(self) -> None:
        """Test from_dict() with units."""
        data = {"value": 3.3, "units": "V"}
        result = MeasurementResultWithProvenance.from_dict(data)
        assert result.value == 3.3
        assert result.units == "V"

    def test_from_dict_with_provenance(self) -> None:
        """Test from_dict() with provenance."""
        data = {
            "value": 3.3,
            "units": "V",
            "provenance": {
                "algorithm": "vpp",
                "parameters": {"window": (0, 1e-3)},
                "timestamp": "2025-12-21T10:30:00Z",
                "library_version": "0.6.0",
                "input_hash": None,
                "metadata": {},
            },
        }
        result = MeasurementResultWithProvenance.from_dict(data)
        assert result.provenance is not None
        assert result.provenance.algorithm == "vpp"
        assert result.provenance.parameters == {"window": (0, 1e-3)}

    def test_from_dict_with_confidence(self) -> None:
        """Test from_dict() with confidence interval."""
        data = {"value": 3.3, "units": "V", "confidence": [3.2, 3.4]}
        result = MeasurementResultWithProvenance.from_dict(data)
        assert result.confidence == (3.2, 3.4)

    def test_from_dict_round_trip(self) -> None:
        """Test that to_dict() -> from_dict() preserves data."""
        prov = Provenance(
            algorithm="vpp", parameters={"window": (0, 1e-3)}, timestamp="2025-12-21T10:30:00Z"
        )
        original = MeasurementResultWithProvenance(
            value=3.3, units="V", provenance=prov, confidence=(3.2, 3.4)
        )
        data = original.to_dict()
        restored = MeasurementResultWithProvenance.from_dict(data)
        assert restored.value == original.value
        assert restored.units == original.units
        assert restored.confidence == original.confidence
        assert restored.provenance is not None
        assert restored.provenance.algorithm == original.provenance.algorithm

    def test_str_minimal(self) -> None:
        """Test __str__() with minimal result."""
        result = MeasurementResultWithProvenance(value=3.3)
        assert str(result) == "3.3"

    def test_str_with_units(self) -> None:
        """Test __str__() with units."""
        result = MeasurementResultWithProvenance(value=3.3, units="V")
        assert str(result) == "3.3 V"

    def test_str_with_provenance(self) -> None:
        """Test __str__() with provenance."""
        prov = Provenance(algorithm="vpp")
        result = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov)
        assert str(result) == "3.3 V (vpp)"

    def test_str_full(self) -> None:
        """Test __str__() with all fields."""
        prov = Provenance(algorithm="peak_to_peak")
        result = MeasurementResultWithProvenance(
            value=3.3, units="V", provenance=prov, confidence=(3.2, 3.4)
        )
        result_str = str(result)
        assert "3.3" in result_str
        assert "V" in result_str
        assert "peak_to_peak" in result_str

    def test_repr_minimal(self) -> None:
        """Test __repr__() with minimal result."""
        result = MeasurementResultWithProvenance(value=3.3)
        assert repr(result) == "MeasurementResultWithProvenance(value=3.3)"

    def test_repr_with_units(self) -> None:
        """Test __repr__() with units."""
        result = MeasurementResultWithProvenance(value=3.3, units="V")
        assert repr(result) == "MeasurementResultWithProvenance(value=3.3, units='V')"

    def test_repr_with_provenance(self) -> None:
        """Test __repr__() with provenance."""
        prov = Provenance(algorithm="vpp")
        result = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov)
        assert (
            repr(result) == "MeasurementResultWithProvenance(value=3.3, units='V', algorithm='vpp')"
        )

    def test_pretty_print_minimal(self) -> None:
        """Test pretty_print() with minimal result."""
        result = MeasurementResultWithProvenance(value=3.3)
        output = result.pretty_print()
        assert "Value: 3.3" in output

    def test_pretty_print_with_units(self) -> None:
        """Test pretty_print() with units."""
        result = MeasurementResultWithProvenance(value=3.3, units="V")
        output = result.pretty_print()
        assert "Value: 3.3 V" in output

    def test_pretty_print_with_confidence(self) -> None:
        """Test pretty_print() with confidence interval."""
        result = MeasurementResultWithProvenance(value=3.3, units="V", confidence=(3.2, 3.4))
        output = result.pretty_print()
        assert "Value: 3.3 V" in output
        assert "Confidence: (3.2, 3.4)" in output

    def test_pretty_print_with_provenance(self) -> None:
        """Test pretty_print() with provenance."""
        prov = Provenance(
            algorithm="vpp",
            parameters={"window": (0, 1e-3)},
            timestamp="2025-12-21T10:30:00Z",
        )
        result = MeasurementResultWithProvenance(value=3.3, units="V", provenance=prov)
        output = result.pretty_print()
        assert "Value: 3.3 V" in output
        assert "Algorithm: vpp" in output
        assert "Timestamp: 2025-12-21T10:30:00Z" in output
        assert "Parameters:" in output

    def test_pretty_print_full(self) -> None:
        """Test pretty_print() with all fields."""
        prov = Provenance(
            algorithm="peak_to_peak",
            parameters={"window": (0, 1e-3)},
            timestamp="2025-12-21T10:30:00Z",
        )
        result = MeasurementResultWithProvenance(
            value=3.3, units="V", provenance=prov, confidence=(3.2, 3.4)
        )
        output = result.pretty_print()
        assert "Value: 3.3 V" in output
        assert "Confidence: (3.2, 3.4)" in output
        assert "Algorithm: peak_to_peak" in output
        assert "Timestamp: 2025-12-21T10:30:00Z" in output


# ==============================================================================
# Utility Function Tests
# ==============================================================================


class TestComputeInputHash:
    """Test compute_input_hash() function."""

    def test_hash_simple_array(self) -> None:
        """Test hashing a simple array."""
        data = np.array([1.0, 2.0, 3.0])
        hash_val = compute_input_hash(data)
        assert isinstance(hash_val, str)
        # SHA-256 produces 64 hex characters
        assert len(hash_val) == 64

    def test_hash_deterministic(self) -> None:
        """Test that same data produces same hash."""
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([1.0, 2.0, 3.0])
        hash1 = compute_input_hash(data1)
        hash2 = compute_input_hash(data2)
        assert hash1 == hash2

    def test_hash_different_data(self) -> None:
        """Test that different data produces different hash."""
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([1.0, 2.0, 4.0])
        hash1 = compute_input_hash(data1)
        hash2 = compute_input_hash(data2)
        assert hash1 != hash2

    def test_hash_empty_array(self) -> None:
        """Test hashing an empty array."""
        data = np.array([])
        hash_val = compute_input_hash(data)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_hash_large_array(self) -> None:
        """Test hashing a large array."""
        data = np.random.randn(10000)
        hash_val = compute_input_hash(data)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_hash_2d_array(self) -> None:
        """Test hashing a 2D array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        hash_val = compute_input_hash(data)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64


class TestCreateProvenance:
    """Test create_provenance() function."""

    def test_create_minimal(self) -> None:
        """Test creating provenance with minimal arguments."""
        prov = create_provenance(algorithm="test")
        assert prov.algorithm == "test"
        assert prov.parameters == {}
        assert prov.input_hash is None
        assert prov.metadata == {}
        assert prov.library_version == "0.6.0"
        # Timestamp should be auto-generated
        assert prov.timestamp is not None

    def test_create_with_parameters(self) -> None:
        """Test creating provenance with parameters."""
        params = {"window": "hann", "n": 1024}
        prov = create_provenance(algorithm="fft", parameters=params)
        assert prov.algorithm == "fft"
        assert prov.parameters == params

    def test_create_with_input_data(self) -> None:
        """Test creating provenance with input data hashing."""
        data = np.array([1.0, 2.0, 3.0])
        prov = create_provenance(algorithm="mean", input_data=data)
        assert prov.input_hash is not None
        assert len(prov.input_hash) == 64
        # Should match direct hash
        assert prov.input_hash == compute_input_hash(data)

    def test_create_with_metadata(self) -> None:
        """Test creating provenance with metadata."""
        metadata = {"source": "oscilloscope", "channel": 1}
        prov = create_provenance(algorithm="test", metadata=metadata)
        assert prov.metadata == metadata

    def test_create_full(self) -> None:
        """Test creating provenance with all arguments."""
        data = np.array([1.0, 2.0, 3.0])
        params = {"axis": 0}
        metadata = {"source": "test"}
        prov = create_provenance(
            algorithm="mean", parameters=params, input_data=data, metadata=metadata
        )
        assert prov.algorithm == "mean"
        assert prov.parameters == params
        assert prov.input_hash is not None
        assert prov.metadata == metadata

    def test_create_without_input_data(self) -> None:
        """Test that input_hash is None when no input_data provided."""
        prov = create_provenance(algorithm="test", parameters={"x": 1})
        assert prov.input_hash is None

    def test_create_parameters_none(self) -> None:
        """Test that None parameters becomes empty dict."""
        prov = create_provenance(algorithm="test", parameters=None)
        assert prov.parameters == {}

    def test_create_metadata_none(self) -> None:
        """Test that None metadata becomes empty dict."""
        prov = create_provenance(algorithm="test", metadata=None)
        assert prov.metadata == {}


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCoreProvenanceIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from creation to serialization."""
        # Create provenance with input data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prov = create_provenance(
            algorithm="mean",
            parameters={"axis": 0},
            input_data=data,
            metadata={"source": "test"},
        )

        # Create measurement result
        result = MeasurementResultWithProvenance(
            value=3.0, units="V", provenance=prov, confidence=(2.5, 3.5)
        )

        # Serialize to dict
        result_dict = result.to_dict()
        assert result_dict["value"] == 3.0
        assert result_dict["units"] == "V"
        assert "provenance" in result_dict
        assert result_dict["confidence"] == (2.5, 3.5)

        # Restore from dict
        restored = MeasurementResultWithProvenance.from_dict(result_dict)
        assert restored.is_equivalent(result, check_parameters=True)

    def test_comparison_workflow(self) -> None:
        """Test comparing results from different algorithms."""
        data = np.array([1.0, 2.0, 3.0])

        # Create two results with same value but different algorithms
        prov1 = create_provenance(algorithm="method_a", input_data=data)
        result1 = MeasurementResultWithProvenance(value=2.0, units="V", provenance=prov1)

        prov2 = create_provenance(algorithm="method_b", input_data=data)
        result2 = MeasurementResultWithProvenance(value=2.0, units="V", provenance=prov2)

        # Should be equivalent when ignoring parameters
        assert result1.is_equivalent(result2, check_parameters=False)

        # Should NOT be equivalent when checking parameters
        assert not result1.is_equivalent(result2, check_parameters=True)

    def test_serialization_workflow(self) -> None:
        """Test full JSON serialization workflow."""
        data = np.array([1.0, 2.0, 3.0])
        prov = create_provenance(
            algorithm="fft",
            parameters={"window": "hann", "n": 1024},
            input_data=data,
        )
        result = MeasurementResultWithProvenance(value=42.5, units="Hz", provenance=prov)

        # Serialize to JSON
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)

        # Deserialize from JSON
        restored_dict = json.loads(json_str)
        restored = MeasurementResultWithProvenance.from_dict(restored_dict)

        # Verify restoration
        assert restored.value == result.value
        assert restored.units == result.units
        assert restored.provenance is not None
        assert restored.provenance.algorithm == result.provenance.algorithm
        assert restored.provenance.input_hash == result.provenance.input_hash

    def test_hash_consistency(self) -> None:
        """Test that input hash is consistent across operations."""
        data = np.array([1.0, 2.0, 3.0])

        # Create two separate provenance objects with same data
        prov1 = create_provenance(algorithm="test", input_data=data)
        prov2 = create_provenance(algorithm="test", input_data=data)

        # Hashes should match
        assert prov1.input_hash == prov2.input_hash

        # Both should match direct computation
        direct_hash = compute_input_hash(data)
        assert prov1.input_hash == direct_hash
        assert prov2.input_hash == direct_hash
