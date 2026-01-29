"""Pytest configuration for validation tests.

    - RE-VAL-001: RE Module Validation Suite

Provides fixtures and configuration for:
    - Performance benchmarking
    - Coverage tracking
    - Fuzz testing configuration

NOTE: Markers and pytest_addoption are defined in the root conftest.py
and pyproject.toml. Do NOT duplicate them here to avoid conflicts.

NOTE: The `benchmark` fixture is defined in the root conftest.py.
Do NOT redefine it here - pytest will automatically make it available.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

# =============================================================================
# Test Data Generators
# =============================================================================


@pytest.fixture
def random_bytes_factory():
    """Factory for generating random byte sequences."""

    def generate(size: int, seed: int | None = None) -> bytes:
        if seed is not None:
            random.seed(seed)
        return bytes([random.randint(0, 255) for _ in range(size)])

    return generate


@pytest.fixture
def structured_message_factory():
    """Factory for generating structured protocol messages."""

    def generate(
        header: bytes = b"\xaa\x55",
        min_payload: int = 10,
        max_payload: int = 100,
        count: int = 10,
        seed: int | None = None,
    ) -> list[bytes]:
        if seed is not None:
            random.seed(seed)

        messages = []
        for i in range(count):
            payload_len = random.randint(min_payload, max_payload)
            seq = bytes([i & 0xFF, (i >> 8) & 0xFF])
            payload = bytes([random.randint(0, 255) for _ in range(payload_len)])
            msg = header + seq + bytes([payload_len & 0xFF]) + payload
            messages.append(msg)

        return messages

    return generate


@pytest.fixture
def analog_signal_factory():
    """Factory for generating analog signals for threshold testing."""

    def generate(
        frequency: float = 100.0,
        duration: float = 0.1,
        sample_rate: float = 10000.0,
        noise_level: float = 0.1,
        dc_offset: float = 0.0,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        t = np.arange(0, duration, 1.0 / sample_rate)
        signal = np.sin(2 * np.pi * frequency * t) + dc_offset
        noise = np.random.normal(0, noise_level, len(t))
        return signal + noise

    return generate


# =============================================================================
# Known Protocol Fixtures
# =============================================================================


@pytest.fixture
def modbus_request_factory():
    """Factory for generating Modbus RTU requests."""

    def generate(
        address: int = 1,
        function: int = 3,
        start_reg: int = 0,
        count: int = 10,
    ) -> bytes:
        import struct

        # Basic request without CRC
        request = struct.pack(">BBHH", address, function, start_reg, count)

        # CRC-16 (simplified placeholder - real CRC would need full implementation)
        crc = (sum(request) & 0xFFFF) ^ 0xFFFF
        return request + struct.pack("<H", crc)

    return generate


@pytest.fixture
def length_prefixed_factory():
    """Factory for generating length-prefixed messages."""

    def generate(
        payloads: list[bytes] | None = None,
        count: int = 5,
        length_bytes: int = 2,
        big_endian: bool = True,
        seed: int | None = None,
    ) -> bytes:
        import struct

        if seed is not None:
            random.seed(seed)

        if payloads is None:
            payloads = [
                bytes([random.randint(0, 255) for _ in range(random.randint(5, 50))])
                for _ in range(count)
            ]

        result = bytearray()
        for payload in payloads:
            if length_bytes == 1:
                result.append(len(payload) & 0xFF)
            elif length_bytes == 2:
                fmt = ">H" if big_endian else "<H"
                result.extend(struct.pack(fmt, len(payload)))
            elif length_bytes == 4:
                fmt = ">I" if big_endian else "<I"
                result.extend(struct.pack(fmt, len(payload)))
            result.extend(payload)

        return bytes(result)

    return generate


# =============================================================================
# Coverage Tracking
# =============================================================================


@pytest.fixture(scope="session")
def coverage_tracker():
    """Track test coverage for RE modules."""
    tracker: dict[str, Any] = {
        "modules_tested": set(),
        "requirements_covered": set(),
        "test_count": 0,
    }

    yield tracker

    # Print coverage summary at end of session
    print("\n" + "=" * 60)
    print("RE Validation Coverage Summary")
    print("=" * 60)
    print(f"Modules tested: {len(tracker['modules_tested'])}")
    print(f"Requirements covered: {len(tracker['requirements_covered'])}")
    print(f"Total tests: {tracker['test_count']}")


# =============================================================================
# Performance Thresholds
# =============================================================================


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for benchmark tests."""
    return {
        "entropy_per_kb": 0.001,  # Max seconds per KB
        "pattern_search_per_pattern": 0.01,  # Max seconds per pattern
        "clustering_per_100_messages": 1.0,  # Max seconds per 100 messages
        "field_inference_per_message": 0.1,  # Max seconds per message
    }


# =============================================================================
# Fuzz Test Configuration
# =============================================================================


@pytest.fixture
def fuzz_iterations(request: pytest.FixtureRequest) -> int:
    """Get fuzz test iteration count from command line or default."""
    # Use the --fuzz-iterations option if provided via root conftest
    # Otherwise default to 10
    return getattr(request.config.option, "fuzz_iterations", 10)
