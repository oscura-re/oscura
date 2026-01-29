"""Inference-specific test fixtures.

This module provides fixtures for protocol inference tests:
- Protocol pattern fixtures
- State machine fixtures
- Message format fixtures
- Alignment algorithm fixtures
- Protocol DSL fixtures
"""

from __future__ import annotations

from typing import Any

import pytest

# =============================================================================
# Protocol Pattern Fixtures
# =============================================================================


@pytest.fixture
def uart_message_pattern() -> dict[str, Any]:
    """Common UART message pattern for inference tests.

    Returns:
        UART configuration dictionary.
    """
    return {
        "start_bit": 0,
        "data_bits": 8,
        "parity": None,
        "stop_bits": 1,
        "baud_rate": 9600,
    }


@pytest.fixture
def spi_message_pattern() -> dict[str, Any]:
    """Common SPI message pattern for inference tests.

    Returns:
        SPI configuration dictionary.
    """
    return {
        "mode": 0,  # CPOL=0, CPHA=0
        "bit_order": "MSB",
        "clock_polarity": 0,
        "clock_phase": 0,
        "word_size": 8,
    }


@pytest.fixture
def i2c_message_pattern() -> dict[str, Any]:
    """Common I2C message pattern for inference tests.

    Returns:
        I2C configuration dictionary.
    """
    return {
        "address_bits": 7,
        "speed": "standard",  # 100 kHz
        "start_condition": True,
        "stop_condition": True,
        "ack_after_byte": True,
    }


@pytest.fixture
def protocol_samples() -> dict[str, bytes]:
    """Sample protocol messages for inference.

    Returns:
        Dictionary mapping protocol name to sample message bytes.
    """
    return {
        "uart": b"\xaa\x55\x01\x02\x03\x04",
        "spi": b"\x00\x01\x02\x03\x04\x05",
        "i2c": b"\xa0\x00\x12\x34\x56",
        "modbus": b"\x01\x03\x00\x00\x00\x0a\xc5\xcd",
        "can": b"\x12\x34\x56\x78\x00\x08\x01\x02\x03\x04\x05\x06\x07\x08",
    }


# =============================================================================
# State Machine Fixtures
# =============================================================================


@pytest.fixture
def simple_state_machine() -> dict[str, Any]:
    """Simple state machine for inference testing.

    Returns:
        State machine with 3 states and simple transitions.
    """
    return {
        "states": ["IDLE", "ACTIVE", "DONE"],
        "transitions": [
            {"from": "IDLE", "to": "ACTIVE", "symbol": "START"},
            {"from": "ACTIVE", "to": "DONE", "symbol": "STOP"},
            {"from": "DONE", "to": "IDLE", "symbol": "RESET"},
        ],
        "initial_state": "IDLE",
        "final_states": ["DONE"],
    }


@pytest.fixture
def protocol_state_machine() -> dict[str, Any]:
    """Protocol-like state machine for RPNI testing.

    Returns:
        State machine representing a simple protocol.
    """
    return {
        "states": ["IDLE", "HEADER", "LENGTH", "PAYLOAD", "CHECKSUM"],
        "transitions": [
            {"from": "IDLE", "to": "HEADER", "symbol": 0xAA},
            {"from": "HEADER", "to": "LENGTH", "symbol": 0x55},
            {"from": "LENGTH", "to": "PAYLOAD", "symbol": "ANY"},
            {"from": "PAYLOAD", "to": "PAYLOAD", "symbol": "DATA"},
            {"from": "PAYLOAD", "to": "CHECKSUM", "symbol": "END"},
            {"from": "CHECKSUM", "to": "IDLE", "symbol": "ANY"},
        ],
        "initial_state": "IDLE",
    }


@pytest.fixture
def sample_traces() -> list[list[int]]:
    """Sample message traces for state machine inference.

    Returns:
        List of symbol sequences representing protocol messages.
    """
    return [
        [0xAA, 0x55, 0x04, 0x01, 0x02, 0x03, 0x04, 0xFF],
        [0xAA, 0x55, 0x02, 0x01, 0x02, 0xFF],
        [0xAA, 0x55, 0x06, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0xFF],
        [0xAA, 0x55, 0x01, 0x01, 0xFF],
    ]


# =============================================================================
# Message Format Fixtures
# =============================================================================


@pytest.fixture
def fixed_length_messages() -> list[bytes]:
    """Fixed-length protocol messages for field inference.

    Returns:
        List of 16-byte messages with consistent structure.
    """
    return [
        b"\xaa\x55\x00\x01\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff",
        b"\xaa\x55\x00\x02\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\xff",
        b"\xaa\x55\x00\x03\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\xff",
        b"\xaa\x55\x00\x04\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\xff",
    ]


@pytest.fixture
def variable_length_messages() -> list[bytes]:
    """Variable-length protocol messages for field inference.

    Returns:
        List of messages with length-prefixed payloads.
    """
    return [
        b"\xaa\x55\x04\x01\x02\x03\x04\xff",
        b"\xaa\x55\x02\x01\x02\xff",
        b"\xaa\x55\x08\x01\x02\x03\x04\x05\x06\x07\x08\xff",
        b"\xaa\x55\x01\x01\xff",
        b"\xaa\x55\x06\x01\x02\x03\x04\x05\x06\xff",
    ]


@pytest.fixture
def delimited_messages() -> list[bytes]:
    """Delimiter-based protocol messages for field inference.

    Returns:
        List of messages with field delimiters.
    """
    return [
        b"CMD:READ|ADDR:0x1000|LEN:256\r\n",
        b"CMD:WRITE|ADDR:0x2000|DATA:0xABCD\r\n",
        b"CMD:STATUS|CODE:0\r\n",
        b"CMD:ERROR|CODE:5|MSG:Timeout\r\n",
    ]


@pytest.fixture
def field_boundaries() -> dict[str, list[tuple[int, int]]]:
    """Expected field boundaries for message format tests.

    Returns:
        Dictionary mapping message type to list of (start, end) tuples.
    """
    return {
        "fixed_length": [
            (0, 2),  # Header
            (2, 4),  # Sequence
            (4, 14),  # Payload
            (14, 16),  # Footer
        ],
        "variable_length": [
            (0, 2),  # Header
            (2, 3),  # Length
            # Payload: variable
            # Footer: 1 byte at end
        ],
    }


# =============================================================================
# Alignment Algorithm Fixtures
# =============================================================================


@pytest.fixture
def alignment_sequences() -> dict[str, tuple[list[int], list[int]]]:
    """Sequence pairs for alignment algorithm testing.

    Returns:
        Dictionary mapping test case to (seq1, seq2) tuples.
    """
    return {
        "identical": ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        "one_insertion": ([1, 2, 3, 4, 5], [1, 2, 9, 3, 4, 5]),
        "one_deletion": ([1, 2, 3, 4, 5], [1, 2, 4, 5]),
        "one_substitution": ([1, 2, 3, 4, 5], [1, 2, 9, 4, 5]),
        "multiple_edits": ([1, 2, 3, 4, 5], [1, 9, 3, 5, 6]),
        "completely_different": ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
    }


@pytest.fixture
def needleman_wunsch_params() -> dict[str, Any]:
    """Parameters for Needleman-Wunsch alignment.

    Returns:
        Dictionary with match, mismatch, and gap scores.
    """
    return {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_penalty": -2,
    }


@pytest.fixture
def smith_waterman_params() -> dict[str, Any]:
    """Parameters for Smith-Waterman alignment.

    Returns:
        Dictionary with match, mismatch, and gap scores.
    """
    return {
        "match_score": 3,
        "mismatch_score": -3,
        "gap_penalty": -2,
    }


# =============================================================================
# Protocol DSL Fixtures
# =============================================================================


@pytest.fixture
def simple_protocol_dsl() -> str:
    """Simple protocol DSL definition.

    Returns:
        Protocol DSL string for parsing tests.
    """
    return """
    protocol SimpleProtocol {
        header: bytes[2] = 0xaa55;
        sequence: uint16;
        length: uint8;
        payload: bytes[length];
        checksum: uint8;
    }
    """


@pytest.fixture
def complex_protocol_dsl() -> str:
    """Complex protocol DSL definition with conditionals.

    Returns:
        Protocol DSL string with conditional fields.
    """
    return """
    protocol ComplexProtocol {
        header: bytes[2] = 0xaa55;
        message_type: uint8;

        if message_type == 0x01 {
            payload: CommandPayload;
        } else if message_type == 0x02 {
            payload: DataPayload;
        } else {
            payload: ErrorPayload;
        }

        checksum: crc16;
    }

    struct CommandPayload {
        command: uint8;
        address: uint32;
        length: uint16;
    }

    struct DataPayload {
        address: uint32;
        data: bytes[256];
    }

    struct ErrorPayload {
        error_code: uint8;
        message: string[0..64];
    }
    """


@pytest.fixture
def protocol_library() -> dict[str, dict[str, Any]]:
    """Library of known protocol definitions.

    Returns:
        Dictionary mapping protocol name to definition.
    """
    return {
        "modbus_rtu": {
            "name": "Modbus RTU",
            "fields": [
                {"name": "address", "type": "uint8", "range": (1, 247)},
                {"name": "function", "type": "uint8", "range": (1, 127)},
                {"name": "data", "type": "bytes", "length": "variable"},
                {"name": "crc", "type": "crc16", "polynomial": 0xA001},
            ],
        },
        "mqtt": {
            "name": "MQTT",
            "fields": [
                {"name": "fixed_header", "type": "uint8"},
                {"name": "remaining_length", "type": "varint"},
                {"name": "variable_header", "type": "bytes", "length": "conditional"},
                {"name": "payload", "type": "bytes", "length": "remaining"},
            ],
        },
        "custom_binary": {
            "name": "Custom Binary",
            "fields": [
                {"name": "magic", "type": "bytes", "value": b"\xde\xad\xbe\xef"},
                {"name": "version", "type": "uint16"},
                {"name": "length", "type": "uint32"},
                {"name": "payload", "type": "bytes", "length_field": "length"},
                {"name": "checksum", "type": "uint32"},
            ],
        },
    }


# =============================================================================
# Inference Quality Metrics
# =============================================================================


@pytest.fixture
def inference_thresholds() -> dict[str, float]:
    """Quality thresholds for inference validation.

    Returns:
        Dictionary with confidence and accuracy thresholds.
    """
    return {
        "min_confidence": 0.8,  # Minimum confidence for field detection
        "min_accuracy": 0.95,  # Minimum accuracy for state machine
        "max_false_positive_rate": 0.05,  # Maximum FP rate
        "min_field_coverage": 0.9,  # Minimum field coverage
        "max_alignment_distance": 0.2,  # Maximum normalized edit distance
    }


# =============================================================================
# Clustering Fixtures
# =============================================================================


@pytest.fixture
def message_clusters() -> dict[str, list[bytes]]:
    """Pre-clustered messages for cluster validation.

    Returns:
        Dictionary mapping cluster label to messages.
    """
    return {
        "read_command": [
            b"\x01\x03\x00\x00\x00\x0a",
            b"\x01\x03\x00\x10\x00\x0a",
            b"\x01\x03\x00\x20\x00\x0a",
        ],
        "write_command": [
            b"\x01\x06\x00\x00\x12\x34",
            b"\x01\x06\x00\x10\x56\x78",
            b"\x01\x06\x00\x20\xab\xcd",
        ],
        "error_response": [
            b"\x01\x83\x01",
            b"\x01\x83\x02",
            b"\x01\x83\x03",
        ],
    }


@pytest.fixture
def entropy_patterns() -> dict[str, bytes]:
    """Binary patterns with varying entropy for field inference.

    Returns:
        Dictionary mapping entropy type to binary data.
    """
    return {
        "constant": b"\x00" * 100,
        "low_entropy": b"\xaa\x55" * 50,
        "medium_entropy": bytes(range(100)),
        "high_entropy": bytes(
            [
                0xA7,
                0x3B,
                0xE2,
                0x19,
                0x8F,
                0x4C,
                0xD5,
                0x61,
                0x2F,
                0x9A,
                0x47,
                0xC3,
                0x18,
                0xE6,
                0x5D,
                0xB2,
            ]
            * 6
            + [0xA7, 0x3B, 0xE2, 0x19]
        ),
    }


# =============================================================================
# Protocol Inference Test Cases
# =============================================================================


@pytest.fixture
def inference_test_cases() -> list[dict[str, Any]]:
    """Complete test cases for protocol inference.

    Returns:
        List of test case dictionaries with messages and expected results.
    """
    return [
        {
            "name": "fixed_header_fixed_payload",
            "messages": [
                b"\xaa\x55\x01\x02\x03\x04",
                b"\xaa\x55\x05\x06\x07\x08",
                b"\xaa\x55\x09\x0a\x0b\x0c",
            ],
            "expected_fields": [
                {"name": "header", "offset": 0, "length": 2, "constant": True},
                {"name": "payload", "offset": 2, "length": 4, "constant": False},
            ],
        },
        {
            "name": "length_prefixed",
            "messages": [
                b"\xaa\x55\x02\x01\x02",
                b"\xaa\x55\x04\x01\x02\x03\x04",
                b"\xaa\x55\x01\x01",
            ],
            "expected_fields": [
                {"name": "header", "offset": 0, "length": 2, "constant": True},
                {"name": "length", "offset": 2, "length": 1, "type": "length"},
                {"name": "payload", "offset": 3, "length": "variable", "length_field": 2},
            ],
        },
        {
            "name": "with_checksum",
            "messages": [
                b"\xaa\x55\x01\x02\x03\x04\xff",
                b"\xaa\x55\x05\x06\x07\x08\xff",
            ],
            "expected_fields": [
                {"name": "header", "offset": 0, "length": 2, "constant": True},
                {"name": "payload", "offset": 2, "length": 4},
                {"name": "footer", "offset": 6, "length": 1, "constant": True},
            ],
        },
    ]
