"""Packet validation and integrity checking.

This module provides comprehensive packet validation including sync markers,
sequence numbers, checksums, and structural integrity verification.


Example:
    >>> from oscura.loaders.validation import PacketValidator
    >>> validator = PacketValidator(sync_marker=0xFA, checksum_type="crc16")
    >>> result = validator.validate_packet(packet_data)
    >>> if result.is_valid:
    ...     print("Packet valid")
    >>> stats = validator.get_statistics()
    >>> print(f"Pass rate: {stats.pass_rate:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

# Logger for debug output
logger = logging.getLogger(__name__)


@dataclass
class SequenceGap:
    """Sequence gap information.



    Attributes:
        position: Packet position where gap was detected.
        expected: Expected sequence number.
        got: Actual sequence number received.
        gap_size: Size of the gap (number of missing packets).
    """

    position: int
    expected: int
    got: int
    gap_size: int


@dataclass
class SequenceValidation:
    """Sequence validation results.



    Attributes:
        total_packets: Total number of packets validated.
        sequence_gaps: List of detected sequence gaps.
        duplicates: Number of duplicate sequence numbers.
        valid: Whether sequence validation passed overall.
    """

    total_packets: int = 0
    sequence_gaps: list[SequenceGap] = field(default_factory=list)
    duplicates: int = 0
    valid: bool = True

    @property
    def gap_count(self) -> int:
        """Number of sequence gaps detected.

        Returns:
            Number of gaps.
        """
        return len(self.sequence_gaps)

    @property
    def total_missing_packets(self) -> int:
        """Total number of missing packets across all gaps.

        Returns:
            Sum of all gap sizes.
        """
        return sum(gap.gap_size for gap in self.sequence_gaps)


@dataclass
class ValidationResult:
    """Result of packet validation.



    Attributes:
        is_valid: Whether packet passed all validation checks.
        sync_valid: Sync marker validation result.
        sequence_valid: Sequence number validation result.
        checksum_valid: Checksum validation result.
        errors: List of validation error messages.
        warnings: List of validation warnings.
        packet_index: Index of validated packet.
    """

    is_valid: bool = True
    sync_valid: bool = True
    sequence_valid: bool = True
    checksum_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    packet_index: int = 0

    def add_error(self, message: str) -> None:
        """Add validation error.

        Args:
            message: Error message.
        """
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add validation warning.

        Args:
            message: Warning message.
        """
        self.warnings.append(message)


@dataclass
class ValidationStats:
    """Aggregate validation statistics.



    Attributes:
        total_packets: Total number of packets validated.
        valid_packets: Number of packets that passed all checks.
        sync_failures: Number of sync marker failures.
        sequence_gaps: Number of sequence gaps detected.
        sequence_duplicates: Number of duplicate sequences detected.
        checksum_failures: Number of checksum failures.
        error_types: Dictionary of error type counts.
    """

    total_packets: int = 0
    valid_packets: int = 0
    sync_failures: int = 0
    sequence_gaps: int = 0
    sequence_duplicates: int = 0
    checksum_failures: int = 0
    error_types: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Calculate validation pass rate.

        Returns:
            Fraction of packets that passed validation (0.0 to 1.0).
        """
        if self.total_packets == 0:
            return 0.0
        return self.valid_packets / self.total_packets

    @property
    def fail_rate(self) -> float:
        """Calculate validation fail rate.

        Returns:
            Fraction of packets that failed validation (0.0 to 1.0).
        """
        return 1.0 - self.pass_rate

    def add_error_type(self, error_type: str) -> None:
        """Increment error type counter.

        Args:
            error_type: Type of error (e.g., "sync_mismatch", "checksum_fail").
        """
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1


class PacketValidator:
    """Validate packet integrity and structure.



    Performs comprehensive validation including:
    - Sync/magic byte verification
    - Sequence number gap/duplicate detection
    - Checksum verification (CRC-8/16/32, sum, XOR)
    - Field value range validation

    Attributes:
        sync_marker: Expected sync marker value (optional).
        sync_field: Name of sync field in packet header (optional).
        sequence_field: Name of sequence field in packet header (optional).
        checksum_type: Checksum algorithm ("crc8", "crc16", "crc32", "sum", "xor", optional).
        checksum_field: Name of checksum field in packet header (optional).
        strictness: Validation strictness level ("strict", "normal", "lenient").
        stats: Validation statistics.
    """

    def __init__(
        self,
        *,
        sync_marker: int | bytes | None = None,
        sync_field: str = "sync_marker",
        sequence_field: str = "sequence",
        checksum_type: str | None = None,
        checksum_field: str = "checksum",
        strictness: str = "normal",
    ) -> None:
        """Initialize packet validator.

        Args:
            sync_marker: Expected sync marker value (optional).
            sync_field: Name of sync field in packet header (default: "sync_marker").
            sequence_field: Name of sequence field in packet header (default: "sequence").
            checksum_type: Checksum algorithm (optional).
            checksum_field: Name of checksum field in packet header (default: "checksum").
            strictness: Validation strictness ("strict", "normal", "lenient").
        """
        self.sync_marker = sync_marker
        self.sync_field = sync_field
        self.sequence_field = sequence_field
        self.checksum_type = checksum_type
        self.checksum_field = checksum_field
        self.strictness = strictness

        self.stats = ValidationStats()
        self._last_sequence: int | None = None

    def validate_packet(
        self, packet: dict[str, Any], packet_data: bytes | None = None
    ) -> ValidationResult:
        """Validate a single packet.



        Args:
            packet: Parsed packet dictionary.
            packet_data: Raw packet bytes (required for checksum validation).

        Returns:
            ValidationResult with validation outcome.

        Example:
            >>> validator = PacketValidator(sync_marker=0xFA)
            >>> result = validator.validate_packet(packet)
            >>> if not result.is_valid:
            ...     print(f"Errors: {result.errors}")
        """
        result = ValidationResult(packet_index=packet.get("index", 0))

        header = packet.get("header", {})

        # Validate sync marker
        if self.sync_marker is not None:
            result.sync_valid = self._validate_sync(header, result)

        # Validate sequence number
        if self.sequence_field in header:
            result.sequence_valid = self._validate_sequence(header, result)

        # Validate checksum
        if self.checksum_type is not None and packet_data is not None:
            result.checksum_valid = self._validate_checksum(header, packet_data, result)

        # Update statistics
        self.stats.total_packets += 1
        if result.is_valid:
            self.stats.valid_packets += 1

        return result

    def _validate_sync(self, header: dict[str, Any], result: ValidationResult) -> bool:
        """Validate sync marker.

        Args:
            header: Packet header dictionary.
            result: Validation result to update.

        Returns:
            True if sync is valid.
        """
        if self.sync_field not in header:
            if self.strictness == "strict":
                result.add_error(f"Missing sync field: {self.sync_field}")
                self.stats.sync_failures += 1
                self.stats.add_error_type("sync_missing")
                return False
            else:
                result.add_warning(f"Missing sync field: {self.sync_field}")
                return True

        sync_value = header[self.sync_field]

        if sync_value != self.sync_marker:
            # Convert bytes to int if needed for formatting
            if isinstance(sync_value, int):
                sync_val_hex = sync_value
            elif isinstance(sync_value, bytes):
                sync_val_hex = int.from_bytes(sync_value, "big")
            else:
                sync_val_hex = int.from_bytes(bytes([sync_value]), "big")

            # Convert sync_marker to int if needed for formatting
            if isinstance(self.sync_marker, int):
                expected_hex = self.sync_marker
            elif isinstance(self.sync_marker, bytes):
                expected_hex = int.from_bytes(self.sync_marker, "big")
            else:
                expected_hex = 0

            msg = f"Sync marker mismatch: expected {expected_hex:#x}, got {sync_val_hex:#x}"
            if self.strictness == "strict":
                result.add_error(msg)
            else:
                result.add_warning(msg)

            self.stats.sync_failures += 1
            self.stats.add_error_type("sync_mismatch")
            return False

        return True

    def _validate_sequence(self, header: dict[str, Any], result: ValidationResult) -> bool:
        """Validate sequence number.

        Args:
            header: Packet header dictionary.
            result: Validation result to update.

        Returns:
            True if sequence is valid.
        """
        sequence = header.get(self.sequence_field)
        if sequence is None:
            return True  # No sequence to validate

        if self._last_sequence is not None:
            expected = (self._last_sequence + 1) & 0xFFFFFFFF  # Handle rollover

            if sequence == self._last_sequence:
                # Duplicate sequence
                msg = f"Duplicate sequence number: {sequence}"
                result.add_warning(msg)
                self.stats.sequence_duplicates += 1
                self.stats.add_error_type("sequence_duplicate")
                return False

            elif sequence != expected:
                # Sequence gap
                gap = (sequence - expected) & 0xFFFFFFFF
                msg = f"Sequence gap detected: expected {expected}, got {sequence} (gap: {gap})"

                if self.strictness == "strict":
                    result.add_error(msg)
                else:
                    result.add_warning(msg)

                self.stats.sequence_gaps += 1
                self.stats.add_error_type("sequence_gap")

                if self.strictness == "strict":
                    self._last_sequence = sequence
                    return False

        self._last_sequence = sequence
        return True

    def _validate_checksum(
        self, header: dict[str, Any], packet_data: bytes, result: ValidationResult
    ) -> bool:
        """Validate packet checksum.

        Args:
            header: Packet header dictionary.
            packet_data: Raw packet bytes.
            result: Validation result to update.

        Returns:
            True if checksum is valid.
        """
        if self.checksum_field not in header:
            if self.strictness == "strict":
                result.add_error(f"Missing checksum field: {self.checksum_field}")
                self.stats.checksum_failures += 1
                self.stats.add_error_type("checksum_missing")
                return False
            return True

        expected_checksum = header[self.checksum_field]
        computed_checksum = self._compute_checksum(packet_data)

        if computed_checksum != expected_checksum:
            msg = f"Checksum mismatch: expected {expected_checksum:#x}, got {computed_checksum:#x}"

            if self.strictness == "strict":
                result.add_error(msg)
            else:
                result.add_warning(msg)

            self.stats.checksum_failures += 1
            self.stats.add_error_type("checksum_fail")
            return False

        return True

    def _compute_checksum(self, data: bytes) -> int:
        """Compute checksum using configured algorithm.

        Args:
            data: Data to checksum.

        Returns:
            Computed checksum value.
        """
        if self.checksum_type == "crc8":
            return self._crc8(data)
        elif self.checksum_type == "crc16":
            return self._crc16(data)
        elif self.checksum_type == "crc32":
            return self._crc32(data)
        elif self.checksum_type == "sum":
            return sum(data) & 0xFF
        elif self.checksum_type == "xor":
            result = 0
            for byte in data:
                result ^= byte
            return result
        else:
            logger.warning("Unknown checksum type: %s", self.checksum_type)
            return 0

    @staticmethod
    def _crc8(data: bytes, poly: int = 0x07) -> int:
        """Compute CRC-8 checksum.

        Args:
            data: Data to checksum.
            poly: CRC polynomial (default: 0x07).

        Returns:
            CRC-8 value.
        """
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ poly
                else:
                    crc <<= 1
            crc &= 0xFF
        return crc

    @staticmethod
    def _crc16(data: bytes, poly: int = 0x1021) -> int:
        """Compute CRC-16 checksum.

        Args:
            data: Data to checksum.
            poly: CRC polynomial (default: 0x1021 for CRC-16-CCITT).

        Returns:
            CRC-16 value.
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ poly
                else:
                    crc <<= 1
            crc &= 0xFFFF
        return crc

    @staticmethod
    def _crc32(data: bytes, poly: int = 0xEDB88320) -> int:
        """Compute CRC-32 checksum.

        Args:
            data: Data to checksum.
            poly: CRC polynomial (default: 0xEDB88320 for CRC-32).

        Returns:
            CRC-32 value.
        """
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
        return crc ^ 0xFFFFFFFF

    def get_statistics(self) -> ValidationStats:
        """Get aggregate validation statistics.



        Returns:
            ValidationStats with cumulative validation results.

        Example:
            >>> validator = PacketValidator()
            >>> # ... validate packets ...
            >>> stats = validator.get_statistics()
            >>> print(f"Pass rate: {stats.pass_rate:.1%}")
            >>> print(f"Sync failures: {stats.sync_failures}")
        """
        return self.stats

    def validate_sequence(self, packets: list[dict[str, Any]]) -> SequenceValidation:
        """Validate sequence numbers across multiple packets.



        Args:
            packets: List of parsed packets with headers.

        Returns:
            SequenceValidation with gap and duplicate detection results.

        Example:
            >>> validator = PacketValidator(sequence_field="sequence")
            >>> seq_validation = validator.validate_sequence(packets)
            >>> if seq_validation.gap_count > 0:
            ...     print(f"Found {seq_validation.gap_count} sequence gaps")
        """
        result = SequenceValidation(total_packets=len(packets))

        if not packets:
            return result

        last_seq: int | None = None

        for i, packet in enumerate(packets):
            header = packet.get("header", {})
            seq = header.get(self.sequence_field)

            if seq is None:
                continue

            if last_seq is not None:
                expected = (last_seq + 1) & 0xFFFFFFFF

                if seq == last_seq:
                    # Duplicate
                    result.duplicates += 1
                    result.valid = False

                elif seq != expected:
                    # Gap detected
                    gap_size = (seq - expected) & 0xFFFFFFFF
                    gap = SequenceGap(
                        position=i,
                        expected=expected,
                        got=seq,
                        gap_size=gap_size,
                    )
                    result.sequence_gaps.append(gap)
                    result.valid = False

            last_seq = seq

        return result

    def reset_statistics(self) -> None:
        """Reset validation statistics.

        Useful for validating multiple files or resetting state.
        """
        self.stats = ValidationStats()
        self._last_sequence = None


__all__ = [
    "PacketValidator",
    "SequenceGap",
    "SequenceValidation",
    "ValidationResult",
    "ValidationStats",
]
