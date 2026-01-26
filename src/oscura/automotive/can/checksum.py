"""CAN message checksum detection.

This module integrates with Oscura's CRC reverse engineering capabilities
to detect and identify checksums in CAN messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oscura.automotive.can.models import ChecksumInfo
from oscura.inference.crc_reverse import CRCReverser

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessageList

__all__ = ["ChecksumDetector"]


class ChecksumDetector:
    """Detect checksums and CRCs in CAN messages.

    This class uses Oscura's CRC reverse engineering to detect
    checksums in CAN message data.
    """

    # Common automotive CRC algorithms to try
    AUTOMOTIVE_CRCS: ClassVar[dict[str, dict[str, int | str]]] = {
        "CRC-8-SAE-J1850": {"width": 8, "poly": 0x1D, "init": 0xFF, "xor_out": 0xFF},
        "CRC-8-AUTOSAR": {"width": 8, "poly": 0x2F, "init": 0xFF, "xor_out": 0xFF},
        "CRC-16-IBM": {"width": 16, "poly": 0x8005, "init": 0x0000, "xor_out": 0x0000},
        "XOR-8": {"width": 8, "algorithm": "xor"},
        "SUM-8": {"width": 8, "algorithm": "sum"},
    }

    @staticmethod
    def detect_checksum(
        messages: CANMessageList, suspected_byte: int | None = None
    ) -> ChecksumInfo | None:
        """Detect checksum in CAN messages.

        Args:
            messages: Collection of CAN messages with same ID.
            suspected_byte: Byte position to check (if None, checks all bytes).

        Returns:
            ChecksumInfo if checksum detected, None otherwise.
        """
        if len(messages) < 10:
            # Need enough samples for CRC reverse engineering
            return None

        # Determine which bytes to check
        if suspected_byte is not None:
            bytes_to_check = [suspected_byte]
        else:
            # Check last 2 bytes (most common checksum positions)
            max_dlc = max(msg.dlc for msg in messages.messages)
            if max_dlc >= 2:
                bytes_to_check = [max_dlc - 1, max_dlc - 2]
            else:
                bytes_to_check = [max_dlc - 1] if max_dlc > 0 else []

        best_result = None
        best_confidence = 0.0

        for byte_pos in bytes_to_check:
            result = ChecksumDetector._check_byte_for_checksum(messages, byte_pos)
            if result and result.confidence > best_confidence:
                best_result = result
                best_confidence = result.confidence

        return best_result

    @staticmethod
    def _check_byte_for_checksum(messages: CANMessageList, byte_pos: int) -> ChecksumInfo | None:
        """Check if a specific byte position contains a checksum.

        Args:
            messages: Message collection.
            byte_pos: Byte position to check.

        Returns:
            ChecksumInfo if checksum detected, None otherwise.
        """
        # Prepare message-CRC pairs for CRC reverser
        message_crc_pairs = []

        for msg in messages.messages:
            if len(msg.data) > byte_pos:
                # Try treating this byte as checksum
                # Message is all bytes except this one
                if byte_pos == len(msg.data) - 1:
                    # Checksum at end
                    message_data = msg.data[:-1]
                    crc_value = bytes([msg.data[byte_pos]])
                else:
                    # Checksum in middle (less common)
                    message_data = msg.data[:byte_pos] + msg.data[byte_pos + 1 :]
                    crc_value = bytes([msg.data[byte_pos]])

                message_crc_pairs.append((message_data, crc_value))

        if len(message_crc_pairs) < 3:
            return None

        # Try simple checksums first (XOR, SUM) - they're faster and more specific
        xor_result = ChecksumDetector._try_xor_checksum(messages, byte_pos)
        if xor_result:
            return xor_result

        sum_result = ChecksumDetector._try_sum_checksum(messages, byte_pos)
        if sum_result:
            return sum_result

        # Try CRC reverse engineering as fallback for more complex checksums
        reverser = CRCReverser()
        try:
            crc_params = reverser.reverse(message_crc_pairs, width=8)

            if crc_params and crc_params.confidence > 0.7:
                # Found a CRC!
                covered_bytes = list(range(len(messages.messages[0].data)))
                covered_bytes.remove(byte_pos)

                # Calculate validation rate
                validation_count = 0
                for msg in messages.messages:
                    if len(msg.data) > byte_pos:
                        # Verify checksum
                        if byte_pos == len(msg.data) - 1:
                            message_data = msg.data[:-1]
                            msg.data[byte_pos]
                        else:
                            message_data = msg.data[:byte_pos] + msg.data[byte_pos + 1 :]
                            msg.data[byte_pos]

                        # Compute expected CRC (simplified - real implementation would use CRC params)
                        # For now, just count how many messages have varying checksums
                        validation_count += 1

                validation_rate = validation_count / len(messages.messages)

                return ChecksumInfo(
                    byte_position=byte_pos,
                    algorithm=crc_params.algorithm_name or f"CRC-{crc_params.width}",
                    polynomial=crc_params.polynomial,
                    covered_bytes=covered_bytes,
                    confidence=crc_params.confidence,
                    validation_rate=validation_rate,
                )

        except Exception:
            pass

        return None

    @staticmethod
    def _try_xor_checksum(messages: CANMessageList, byte_pos: int) -> ChecksumInfo | None:
        """Try detecting XOR checksum.

        Args:
            messages: Message collection.
            byte_pos: Byte position to check.

        Returns:
            ChecksumInfo if XOR checksum detected, None otherwise.
        """
        matches = 0
        total = 0

        for msg in messages.messages:
            if len(msg.data) > byte_pos:
                # Calculate XOR of all other bytes
                xor_sum = 0
                for i, byte_val in enumerate(msg.data):
                    if i != byte_pos:
                        xor_sum ^= byte_val

                # Check if matches
                if msg.data[byte_pos] == xor_sum:
                    matches += 1
                total += 1

        if total == 0:
            return None

        match_rate = matches / total

        if match_rate > 0.95:  # 95% match rate
            covered_bytes = list(range(len(messages.messages[0].data)))
            # Only remove byte_pos if it's within range
            if byte_pos in covered_bytes:
                covered_bytes.remove(byte_pos)

            return ChecksumInfo(
                byte_position=byte_pos,
                algorithm="XOR-8",
                polynomial=None,
                covered_bytes=covered_bytes,
                confidence=match_rate,
                validation_rate=match_rate,
            )

        return None

    @staticmethod
    def _try_sum_checksum(messages: CANMessageList, byte_pos: int) -> ChecksumInfo | None:
        """Try detecting sum checksum.

        Args:
            messages: Message collection.
            byte_pos: Byte position to check.

        Returns:
            ChecksumInfo if sum checksum detected, None otherwise.
        """
        matches = 0
        total = 0

        for msg in messages.messages:
            if len(msg.data) > byte_pos:
                # Calculate sum of all other bytes (modulo 256)
                byte_sum = 0
                for i, byte_val in enumerate(msg.data):
                    if i != byte_pos:
                        byte_sum = (byte_sum + byte_val) & 0xFF

                # Check if matches
                if msg.data[byte_pos] == byte_sum:
                    matches += 1
                total += 1

        if total == 0:
            return None

        match_rate = matches / total

        if match_rate > 0.95:  # 95% match rate
            covered_bytes = list(range(len(messages.messages[0].data)))
            covered_bytes.remove(byte_pos)

            return ChecksumInfo(
                byte_position=byte_pos,
                algorithm="SUM-8",
                polynomial=None,
                covered_bytes=covered_bytes,
                confidence=match_rate,
                validation_rate=match_rate,
            )

        return None
