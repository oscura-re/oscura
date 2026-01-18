"""Pytest fixtures for CAN analysis tests.

This module provides synthetic CAN message generation for testing,
following Oscura's pattern of using only synthetic test data.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from oscura.automotive.can.models import CANMessage, CANMessageList


@pytest.fixture
def sample_can_messages() -> CANMessageList:
    """Generate sample CAN messages for testing.

    Creates a collection of synthetic CAN messages with known patterns:
    - 0x123: Simple data (no pattern)
    - 0x280: Engine data with RPM signal
    - 0x300: Counter in byte 0, checksum in byte 7
    - 0x400: Multiple signals

    Returns:
        CANMessageList with synthetic messages.
    """
    messages = CANMessageList()

    # Message 0x123 - Random data, 10 Hz, 1 second
    for i in range(10):
        msg = CANMessage(
            arbitration_id=0x123,
            timestamp=i * 0.1,
            data=bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
            is_extended=False,
        )
        messages.append(msg)

    # Message 0x280 - Engine RPM (bytes 2-3, big-endian, scale 0.25)
    # RPM increases from 800 to 2000 over 1 second
    for i in range(100):
        timestamp = i * 0.01
        rpm = 800 + (1200 * i / 99)  # 800 to 2000
        raw_rpm = int(rpm / 0.25)  # Convert to raw value

        # Pack as big-endian uint16 at bytes 2-3
        data = bytearray(8)
        data[0] = 0xAA  # Constant
        data[1] = 0xBB  # Constant
        data[2:4] = struct.pack(">H", raw_rpm)  # RPM as big-endian
        data[4] = i % 256  # Counter
        data[5] = 0xCC  # Constant
        data[6] = 0xDD  # Constant
        data[7] = 0xEE  # Constant

        msg = CANMessage(
            arbitration_id=0x280,
            timestamp=timestamp,
            data=bytes(data),
            is_extended=False,
        )
        messages.append(msg)

    # Message 0x300 - Counter with XOR checksum
    for i in range(50):
        timestamp = i * 0.02
        counter = i % 256

        data = bytearray(8)
        data[0] = counter  # Counter byte
        data[1] = 0x10
        data[2] = 0x20
        data[3] = 0x30
        data[4] = 0x40
        data[5] = 0x50
        data[6] = 0x60

        # XOR checksum in byte 7
        xor_sum = 0
        for b in data[:7]:
            xor_sum ^= b
        data[7] = xor_sum

        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=timestamp,
            data=bytes(data),
            is_extended=False,
        )
        messages.append(msg)

    # Message 0x400 - Multiple signals
    # Byte 0-1: uint16 big-endian (vehicle speed, scale 0.01 km/h)
    # Byte 2: uint8 (throttle position, scale 0.4%)
    # Byte 3-6: float32 (temperature in °C)
    for i in range(20):
        timestamp = i * 0.05
        speed_kmh = 50 + (i * 2)  # 50 to 88 km/h
        throttle_pct = 25 + (i * 1.5)  # 25% to 53.5%
        temp_c = 85.0 + (i * 0.5)  # 85°C to 94.5°C

        data = bytearray(8)
        # Speed
        raw_speed = int(speed_kmh / 0.01)
        data[0:2] = struct.pack(">H", raw_speed)
        # Throttle
        raw_throttle = int(throttle_pct / 0.4)
        data[2] = raw_throttle & 0xFF
        # Temperature (float32 big-endian)
        data[3:7] = struct.pack(">f", temp_c)
        # Unused
        data[7] = 0xFF

        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=timestamp,
            data=bytes(data),
            is_extended=False,
        )
        messages.append(msg)

    return messages


@pytest.fixture
def engine_rpm_messages() -> CANMessageList:
    """Generate CAN messages with engine RPM signal.

    RPM is encoded in bytes 2-3 as big-endian uint16 with scale 0.25.
    RPM varies from 800 to 6000 RPM.

    Returns:
        CANMessageList with RPM messages.
    """
    messages = CANMessageList()

    for i in range(100):
        timestamp = i * 0.01
        # RPM varies sinusoidally between 800 and 6000
        import math

        rpm = 3400 + 2600 * math.sin(2 * math.pi * i / 100)
        raw_rpm = int(rpm / 0.25)

        data = bytearray(8)
        data[0] = 0x00
        data[1] = 0x00
        data[2:4] = struct.pack(">H", raw_rpm)
        data[4] = 0x00
        data[5] = 0x00
        data[6] = 0x00
        data[7] = 0x00

        msg = CANMessage(
            arbitration_id=0x280,
            timestamp=timestamp,
            data=bytes(data),
        )
        messages.append(msg)

    return messages


@pytest.fixture
def counter_messages() -> CANMessageList:
    """Generate CAN messages with a counter.

    Counter is in byte 0, wraps at 255.

    Returns:
        CANMessageList with counter messages.
    """
    messages = CANMessageList()

    for i in range(300):
        timestamp = i * 0.01
        counter = i % 256

        data = bytes([counter, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77])

        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=timestamp,
            data=data,
        )
        messages.append(msg)

    return messages


@pytest.fixture
def checksum_messages() -> CANMessageList:
    """Generate CAN messages with XOR checksum.

    Checksum is XOR of bytes 0-6, stored in byte 7.

    Returns:
        CANMessageList with checksum messages.
    """
    messages = CANMessageList()

    for i in range(50):
        timestamp = i * 0.02

        # Varying data
        data = bytearray(8)
        data[0] = (i * 7) % 256
        data[1] = (i * 11) % 256
        data[2] = (i * 13) % 256
        data[3] = (i * 17) % 256
        data[4] = (i * 19) % 256
        data[5] = (i * 23) % 256
        data[6] = (i * 29) % 256

        # XOR checksum
        xor_sum = 0
        for b in data[:7]:
            xor_sum ^= b
        data[7] = xor_sum

        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=timestamp,
            data=bytes(data),
        )
        messages.append(msg)

    return messages


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test files.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to temporary directory.
    """
    return tmp_path
