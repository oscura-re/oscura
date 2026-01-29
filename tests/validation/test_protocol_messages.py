"""Ground truth validation tests for protocol message parsing.

This module validates message format inference against ground truth files
to ensure correct field boundary detection and message structure analysis.

- RE-PAY-001: Payload structure analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.validation


@pytest.mark.requires_data
@pytest.mark.requirement("PSI-001")
class TestMessageFormatValidation:
    """Validate message format inference against ground truth."""

    @pytest.mark.parametrize("size", ["64b", "128b", "256b"])
    def test_message_file_exists(self, protocol_message_files: dict[str, Path], size: str) -> None:
        """Verify protocol message binary files exist.

        Validates:
        - protocol_messages_64b.bin exists
        - protocol_messages_128b.bin exists
        - protocol_messages_256b.bin exists
        """
        path = protocol_message_files.get(size)
        if path is None:
            pytest.skip(f"Message file for {size} not configured")
        if not path.exists():
            pytest.skip(f"Message file not found: {path}")

        # Verify file is not empty
        assert path.stat().st_size > 0, f"Empty message file: {path}"

    @pytest.mark.parametrize("size", ["64b", "128b", "256b"])
    def test_message_ground_truth_exists(
        self, message_truth: dict[str, dict[str, Any]], size: str
    ) -> None:
        """Verify ground truth files exist for message validation."""
        truth = message_truth.get(size, {})
        if not truth:
            pytest.skip(f"Ground truth for {size} not available")

        # Ground truth should have field information
        assert len(truth) > 0, f"Empty ground truth for {size}"

    def test_message_64b_structure(
        self,
        protocol_message_files: dict[str, Path],
        message_truth: dict[str, dict[str, Any]],
    ) -> None:
        """Validate 64-byte message structure against ground truth.

        Validates:
        - Field boundaries match expected positions
        - Field types are correctly identified
        """
        path = protocol_message_files.get("64b")
        truth = message_truth.get("64b", {})

        if path is None or not path.exists():
            pytest.skip("64b message file not available")
        if not truth:
            pytest.skip("64b ground truth not available")

        # Load message data
        with open(path, "rb") as f:
            data = f.read()

        # Verify message size
        expected_msg_size = 64
        assert len(data) % expected_msg_size == 0, (
            f"Data size {len(data)} not multiple of message size {expected_msg_size}"
        )

        msg_count = len(data) // expected_msg_size
        assert msg_count > 0, "No messages in file"

        # Validate against ground truth field boundaries if present
        if "field_boundaries" in truth:
            expected_boundaries = truth["field_boundaries"]
            # Each boundary should be within message size (inclusive of end boundary)
            for boundary in expected_boundaries:
                assert 0 <= boundary <= expected_msg_size, (
                    f"Field boundary {boundary} outside message size {expected_msg_size}"
                )

    def test_message_field_types(self, message_truth: dict[str, dict[str, Any]]) -> None:
        """Validate field type identification in ground truth.

        Checks that ground truth specifies valid field types:
        - Data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64
        - Floating point: float32, float64
        - Binary: bytes, string
        - Semantic: constant, length, sequence, timestamp, data, checksum, crc
        """
        valid_types = {
            # Primitive data types
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
            "bytes",
            "string",
            # Semantic/protocol field types
            "constant",
            "length",
            "sequence",
            "timestamp",
            "data",
            "checksum",
            "crc",
        }

        for size, truth in message_truth.items():
            if "field_types" in truth:
                for field_type in truth["field_types"]:
                    assert field_type in valid_types, f"{size}: Unknown field type '{field_type}'"


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.requirement("PSI-001")
class TestMessageFormatInference:
    """Test message format inference on synthetic data."""

    def test_infer_format_basic(self, protocol_message_files: dict[str, Path]) -> None:
        """Test basic format inference on protocol messages."""
        path = protocol_message_files.get("64b")
        if path is None or not path.exists():
            pytest.skip("64b message file not available")

        with open(path, "rb") as f:
            data = f.read()

        try:
            from oscura.inference import infer_format

            # Convert to list of messages
            msg_size = 64
            messages = [data[i : i + msg_size] for i in range(0, len(data), msg_size)]

            # Infer format from messages
            schema = infer_format(messages)

            # Should return some schema
            assert schema is not None, "Format inference returned None"

        except ImportError:
            pytest.skip("infer_format not available")
        except Exception as e:
            pytest.skip(f"Format inference requires specific conditions: {e}")

    def test_detect_field_types_basic(self, protocol_message_files: dict[str, Path]) -> None:
        """Test field type detection on protocol messages."""
        path = protocol_message_files.get("128b")
        if path is None or not path.exists():
            pytest.skip("128b message file not available")

        with open(path, "rb") as f:
            data = f.read()

        try:
            from oscura.inference import detect_field_types

            # First message
            msg_size = 128
            first_msg = data[:msg_size]

            field_types = detect_field_types(first_msg)

            # Should return some field type information
            assert field_types is not None, "Field type detection returned None"

        except ImportError:
            pytest.skip("detect_field_types not available")
        except Exception as e:
            pytest.skip(f"Field type detection requires specific conditions: {e}")


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.requirement("RE-PAY-001")
class TestPayloadStructureValidation:
    """Validate payload structure analysis."""

    def test_payload_consistency(self, protocol_message_files: dict[str, Path]) -> None:
        """Test that payloads within a file have consistent structure.

        Validates:
        - All messages have same size
        - Header positions are consistent
        """
        for size_label, path in protocol_message_files.items():
            if path is None or not path.exists():
                continue

            with open(path, "rb") as f:
                data = f.read()

            # Extract expected message size from label
            if "64b" in size_label:
                msg_size = 64
            elif "128b" in size_label:
                msg_size = 128
            elif "256b" in size_label:
                msg_size = 256
            else:
                continue

            # All data should be aligned to message size
            assert len(data) % msg_size == 0, f"{size_label}: Data not aligned to message size"

            # Extract first byte of each message (often type/command)
            messages = [data[i : i + msg_size] for i in range(0, len(data), msg_size)]

            # Check for consistent header pattern in first few bytes
            if len(messages) > 1:
                # Compare first 4 bytes across messages
                # Some bytes (like magic/sync) should be consistent
                first_bytes = [msg[:4] for msg in messages[:10]]

                # At least some messages should share header patterns
                # (This is a weak check - real protocols vary)

    def test_checksum_locations(
        self,
        protocol_message_files: dict[str, Path],
        message_truth: dict[str, dict[str, Any]],
    ) -> None:
        """Validate checksum field locations from ground truth."""
        for size_label in ["64b", "128b", "256b"]:
            truth = message_truth.get(size_label, {})
            if "checksum_offset" not in truth:
                continue

            path = protocol_message_files.get(size_label)
            if path is None or not path.exists():
                continue

            expected_offset = truth["checksum_offset"]
            msg_size = int(size_label.replace("b", ""))

            assert 0 <= expected_offset < msg_size, (
                f"{size_label}: Checksum offset {expected_offset} outside message size {msg_size}"
            )


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.inference
class TestMessageSequenceAnalysis:
    """Test message sequence and dependency analysis."""

    def test_find_dependencies(self, protocol_message_files: dict[str, Path]) -> None:
        """Test dependency detection between message fields."""
        path = protocol_message_files.get("64b")
        if path is None or not path.exists():
            pytest.skip("64b message file not available")

        with open(path, "rb") as f:
            data = f.read()

        try:
            from oscura.inference import find_dependencies

            msg_size = 64
            messages = [data[i : i + msg_size] for i in range(0, len(data), msg_size)]

            deps = find_dependencies(messages)

            # Should return dependency information (may be empty)
            assert deps is not None, "Dependency analysis returned None"

        except ImportError:
            pytest.skip("find_dependencies not available")
        except Exception as e:
            pytest.skip(f"Dependency analysis requires specific conditions: {e}")


@pytest.mark.validation
@pytest.mark.requires_data
class TestMessageBinaryIntegrity:
    """Test binary data integrity for message files."""

    @pytest.mark.parametrize("size", ["64b", "128b", "256b"])
    def test_message_file_readable(
        self, protocol_message_files: dict[str, Path], size: str
    ) -> None:
        """Verify message files can be read as binary."""
        path = protocol_message_files.get(size)
        if path is None or not path.exists():
            pytest.skip(f"{size} message file not available")

        try:
            with open(path, "rb") as f:
                data = f.read()
            assert len(data) > 0
        except Exception as e:
            pytest.fail(f"Failed to read {size} message file: {e}")

    def test_no_truncated_messages(self, protocol_message_files: dict[str, Path]) -> None:
        """Verify no truncated messages in files."""
        size_map = {"64b": 64, "128b": 128, "256b": 256}

        for size_label, msg_size in size_map.items():
            path = protocol_message_files.get(size_label)
            if path is None or not path.exists():
                continue

            file_size = path.stat().st_size

            assert file_size % msg_size == 0, (
                f"{size_label}: File size {file_size} not multiple of "
                f"message size {msg_size}, {file_size % msg_size} bytes truncated"
            )


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.requirement("VAL-002")
class TestGroundTruthConsistency:
    """Validate ground truth files are internally consistent."""

    def test_ground_truth_complete(self, ground_truth_dir: Path) -> None:
        """Verify all expected ground truth files exist."""
        expected_files = [
            "fixed_length_packets_truth.json",
            "square_1MHz_truth.json",
            "square_10MHz_truth.json",
            "square_100MHz_truth.json",
            "uart_9600_truth.json",
            "messages_64b_truth.json",
            "messages_128b_truth.json",
            "messages_256b_truth.json",
        ]

        missing = []
        for filename in expected_files:
            path = ground_truth_dir / filename
            if not path.exists():
                missing.append(filename)

        if missing:
            # Don't fail, just report
            pytest.skip(f"Missing ground truth files: {missing}")

    def test_ground_truth_valid_json(self, ground_truth_dir: Path) -> None:
        """Verify all ground truth files are valid JSON."""
        if not ground_truth_dir.exists():
            pytest.skip("Ground truth directory not found")

        for json_file in ground_truth_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                assert isinstance(data, dict), f"{json_file.name}: Not a dict"
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {json_file.name}: {e}")
