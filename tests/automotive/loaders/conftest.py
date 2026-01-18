"""Pytest fixtures for automotive loader tests.

This module provides synthetic data generation for testing automotive
file loaders (BLF, ASC, MDF, CSV).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from oscura.automotive.can.models import CANMessage, CANMessageList


@pytest.fixture
def sample_can_data() -> list[dict]:
    """Generate sample CAN message data for creating test files.

    Returns:
        List of message dicts with timestamp, id, data fields.
    """
    messages = []

    # Standard ID messages
    for i in range(10):
        messages.append(
            {
                "timestamp": i * 0.1,
                "id": 0x123,
                "data": bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
                "is_extended": False,
            }
        )

    # Extended ID messages
    for i in range(5):
        messages.append(
            {
                "timestamp": i * 0.2,
                "id": 0x18FF1234,
                "data": bytes([0xAA, 0xBB, 0xCC, 0xDD]),
                "is_extended": True,
            }
        )

    # Variable data messages
    for i in range(20):
        rpm = 800 + (i * 100)
        raw_rpm = int(rpm / 0.25)
        data = bytearray(8)
        data[0:2] = struct.pack(">H", raw_rpm)
        data[2] = i % 256
        data[3:8] = bytes([0x11, 0x22, 0x33, 0x44, 0x55])

        messages.append(
            {"timestamp": 1.0 + i * 0.05, "id": 0x280, "data": bytes(data), "is_extended": False}
        )

    return messages


@pytest.fixture
def expected_message_list(sample_can_data: list[dict]) -> CANMessageList:
    """Convert sample CAN data to expected CANMessageList.

    Args:
        sample_can_data: Sample CAN message data.

    Returns:
        Expected CANMessageList for validation.
    """
    msg_list = CANMessageList()

    for msg_data in sample_can_data:
        msg = CANMessage(
            arbitration_id=msg_data["id"],
            timestamp=msg_data["timestamp"],
            data=msg_data["data"],
            is_extended=msg_data.get("is_extended", False),
            is_fd=False,
            channel=0,
        )
        msg_list.append(msg)

    return msg_list


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test files.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to temporary directory.
    """
    return tmp_path
