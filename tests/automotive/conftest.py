"""Pytest fixtures for automotive CAN testing.

This module provides reusable fixtures for CAN bus analysis tests.
"""

from __future__ import annotations

import pytest

from oscura.automotive.can.models import CANMessage, CANMessageList


@pytest.fixture
def sample_can_messages() -> CANMessageList:
    """Create a sample CAN message collection for testing.

    Returns a CANMessageList with multiple IDs showing various patterns:
    - 0x280: Engine RPM (varying values, simulating RPM changes)
    - 0x400: Vehicle speed (varying values, simulating speed changes)
    - 0x180: Constant diagnostic message
    - 0x1C0: Status flags (some variation)

    This fixture provides realistic CAN traffic for correlation, analysis,
    and pattern detection tests.

    Returns:
        CANMessageList with ~200 messages across 4 different IDs.
    """
    messages = CANMessageList()

    # Create 50 message cycles with realistic patterns
    for i in range(50):
        timestamp = i * 0.01  # 10ms intervals

        # 0x280: Engine RPM - 16-bit value in bytes 2-3 (big endian)
        # Simulate RPM increasing from 1000 to 3000 RPM
        rpm = 1000 + (i * 40)  # Increases from 1000 to ~3000
        rpm_bytes = rpm.to_bytes(2, byteorder="big")
        data_280 = bytes([0x00, 0x00]) + rpm_bytes + bytes([0x00, 0x00, 0x00, 0x00])
        msg_280 = CANMessage(arbitration_id=0x280, timestamp=timestamp, data=data_280)
        messages.append(msg_280)

        # 0x400: Vehicle speed - 8-bit value in byte 0
        # Simulate speed increasing from 0 to 100 km/h
        speed = min(100, i * 2)  # Increases from 0 to 100
        data_400 = bytes([speed, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg_400 = CANMessage(arbitration_id=0x400, timestamp=timestamp + 0.002, data=data_400)
        messages.append(msg_400)

        # 0x180: Constant diagnostic message (always same value)
        data_180 = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
        msg_180 = CANMessage(arbitration_id=0x180, timestamp=timestamp + 0.004, data=data_180)
        messages.append(msg_180)

        # 0x1C0: Status flags - some variation but mostly constant
        # Byte 0: counter, Byte 1-7: mostly constant
        counter = i % 16  # Wraps at 16
        data_1c0 = bytes([counter, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00])
        msg_1c0 = CANMessage(arbitration_id=0x1C0, timestamp=timestamp + 0.006, data=data_1c0)
        messages.append(msg_1c0)

    return messages
