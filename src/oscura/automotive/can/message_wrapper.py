"""CAN message wrapper for analysis and hypothesis testing.

This module provides the CANMessageWrapper class which wraps messages
with a specific CAN ID and provides analysis and hypothesis testing methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.automotive.can.models import DecodedSignal, SignalDefinition

if TYPE_CHECKING:
    from oscura.automotive.can.models import MessageAnalysis
    from oscura.automotive.can.session import CANSession

__all__ = ["CANMessageWrapper", "HypothesisResult"]


class HypothesisResult:
    """Result of testing a signal hypothesis.

    Attributes:
        signal_name: Name of the tested signal.
        definition: Signal definition tested.
        values: Decoded values from all messages.
        min_value: Minimum decoded value.
        max_value: Maximum decoded value.
        mean: Mean decoded value.
        std: Standard deviation.
        is_valid: Whether hypothesis appears valid.
        confidence: Confidence score (0.0-1.0).
        feedback: Human-readable feedback.
    """

    def __init__(
        self,
        signal_name: str,
        definition: SignalDefinition,
        values: list[float],
        min_value: float,
        max_value: float,
        mean: float,
        std: float,
        is_valid: bool,
        confidence: float,
        feedback: str,
    ):
        """Initialize hypothesis result."""
        self.signal_name = signal_name
        self.definition = definition
        self.values = values
        self.min_value = min_value
        self.max_value = max_value
        self.mean = mean
        self.std = std
        self.is_valid = is_valid
        self.confidence = confidence
        self.feedback = feedback

    def __repr__(self) -> str:
        """Human-readable representation."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"HypothesisResult({self.signal_name}: {status}, "
            f"confidence={self.confidence:.2f}, "
            f"range=[{self.min_value:.2f}, {self.max_value:.2f}])"
        )

    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            f"=== Hypothesis Test: {self.signal_name} ===",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
            f"Confidence: {self.confidence:.2f}",
            "",
            "Signal Definition:",
            f"  Start Byte: {self.definition.start_byte}",
            f"  Start Bit: {self.definition.start_bit}",
            f"  Length: {self.definition.length} bits",
            f"  Byte Order: {self.definition.byte_order}",
            f"  Value Type: {self.definition.value_type}",
            f"  Scale: {self.definition.scale}",
            f"  Offset: {self.definition.offset}",
            "",
            "Decoded Values:",
            f"  Min: {self.min_value:.2f} {self.definition.unit}",
            f"  Max: {self.max_value:.2f} {self.definition.unit}",
            f"  Mean: {self.mean:.2f} {self.definition.unit}",
            f"  Std Dev: {self.std:.2f} {self.definition.unit}",
            f"  Sample Count: {len(self.values)}",
            "",
            f"Feedback: {self.feedback}",
        ]
        return "\n".join(lines)


class CANMessageWrapper:
    """Wrapper for analyzing messages with a specific CAN ID.

    This class provides methods for analyzing and reverse engineering
    a specific CAN message ID.
    """

    def __init__(self, session: CANSession, arbitration_id: int):
        """Initialize message wrapper.

        Args:
            session: Parent CAN session.
            arbitration_id: CAN ID to wrap.
        """
        self._session = session
        self._arbitration_id = arbitration_id
        self._documented_signals: dict[str, SignalDefinition] = {}

    @property
    def arbitration_id(self) -> int:
        """Get CAN arbitration ID."""
        return self._arbitration_id

    def analyze(self, force_refresh: bool = False) -> MessageAnalysis:
        """Perform complete statistical analysis of this message.

        Args:
            force_refresh: Force re-analysis even if cached.

        Returns:
            MessageAnalysis with complete analysis results.

        Example:
            >>> msg = session.message(0x280)
            >>> analysis = msg.analyze()
            >>> print(analysis.summary())
        """
        return self._session.analyze_message(self._arbitration_id, force_refresh=force_refresh)

    def test_hypothesis(
        self,
        signal_name: str,
        start_byte: int,
        bit_length: int,
        byte_order: Literal["big_endian", "little_endian"] = "big_endian",
        value_type: Literal["unsigned", "signed", "float"] = "unsigned",
        scale: float = 1.0,
        offset: float = 0.0,
        unit: str = "",
        expected_min: float | None = None,
        expected_max: float | None = None,
    ) -> HypothesisResult:
        """Test a hypothesis about signal encoding.

        Args:
            signal_name: Name for the signal.
            start_byte: Starting byte position (0-7).
            bit_length: Signal length in bits.
            byte_order: Byte order ('big_endian' or 'little_endian').
            value_type: Value type ('unsigned', 'signed', 'float').
            scale: Scaling factor.
            offset: Offset value.
            unit: Physical unit.
            expected_min: Expected minimum value (for validation).
            expected_max: Expected maximum value (for validation).

        Returns:
            HypothesisResult with test results and feedback.

        Example:
            >>> msg = session.message(0x280)
            >>> result = msg.test_hypothesis(
            ...     signal_name="rpm",
            ...     start_byte=2,
            ...     bit_length=16,
            ...     scale=0.25,
            ...     unit="rpm",
            ...     expected_min=0,
            ...     expected_max=8000
            ... )
            >>> print(result.summary())
        """
        # Create signal definition
        definition = _create_signal_definition(
            signal_name, start_byte, bit_length, byte_order, value_type, scale, offset, unit
        )

        # Decode all values
        decoded_values = _decode_hypothesis_values(self._session, self._arbitration_id, definition)

        # Handle no decoded values
        if not decoded_values:
            return _create_failed_hypothesis_result(signal_name, definition)

        # Calculate statistics
        stats = _calculate_hypothesis_statistics(decoded_values)

        # Validate hypothesis
        is_valid, confidence, feedback = _validate_hypothesis(stats, expected_min, expected_max)

        return HypothesisResult(
            signal_name=signal_name,
            definition=definition,
            values=decoded_values,
            **stats,
            is_valid=is_valid,
            confidence=confidence,
            feedback=feedback,
        )

    def document_signal(
        self,
        name: str,
        start_bit: int,
        length: int,
        byte_order: Literal["big_endian", "little_endian"] = "big_endian",
        value_type: Literal["unsigned", "signed", "float"] = "unsigned",
        scale: float = 1.0,
        offset: float = 0.0,
        unit: str = "",
        comment: str = "",
    ) -> None:
        """Document a confirmed signal definition.

        Args:
            name: Signal name.
            start_bit: Starting bit position.
            length: Signal length in bits.
            byte_order: Byte order.
            value_type: Value type.
            scale: Scaling factor.
            offset: Offset value.
            unit: Physical unit.
            comment: Description or notes.

        Example:
            >>> msg = session.message(0x280)
            >>> msg.document_signal(
            ...     name="rpm",
            ...     start_bit=16,
            ...     length=16,
            ...     scale=0.25,
            ...     unit="rpm",
            ...     comment="Confirmed via OBD-II correlation"
            ... )
        """
        definition = SignalDefinition(
            name=name,
            start_bit=start_bit,
            length=length,
            byte_order=byte_order,
            value_type=value_type,
            scale=scale,
            offset=offset,
            unit=unit,
            comment=comment,
        )

        self._documented_signals[name] = definition

    def get_documented_signals(self) -> dict[str, SignalDefinition]:
        """Get all documented signal definitions.

        Returns:
            Dictionary mapping signal names to definitions.
        """
        return self._documented_signals.copy()

    def decode_signals(self) -> list[DecodedSignal]:
        """Decode all documented signals from all messages.

        Returns:
            List of DecodedSignal objects, one per message per signal.

        Example:
            >>> msg = session.message(0x280)
            >>> msg.document_signal("rpm", start_bit=16, length=16, scale=0.25, unit="rpm")
            >>> decoded = msg.decode_signals()
            >>> for sig in decoded[:5]:  # First 5
            ...     print(sig)
        """
        decoded_signals = []

        filtered = self._session._messages.filter_by_id(self._arbitration_id)

        for msg in filtered.messages:
            for sig_name, sig_def in self._documented_signals.items():
                try:
                    value = sig_def.decode(msg.data)
                    raw_value = sig_def.extract_raw(msg.data)

                    decoded_sig = DecodedSignal(
                        name=sig_name,
                        value=value,
                        unit=sig_def.unit,
                        timestamp=msg.timestamp,
                        raw_value=raw_value,
                        definition=sig_def,
                    )
                    decoded_signals.append(decoded_sig)

                except Exception:
                    # Skip messages that can't be decoded
                    pass

        return decoded_signals

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"CANMessageWrapper(id=0x{self._arbitration_id:03X}, documented_signals={len(self._documented_signals)})"


def _create_signal_definition(
    signal_name: str,
    start_byte: int,
    bit_length: int,
    byte_order: Literal["big_endian", "little_endian"],
    value_type: Literal["unsigned", "signed", "float"],
    scale: float,
    offset: float,
    unit: str,
) -> SignalDefinition:
    """Create signal definition from parameters."""
    return SignalDefinition(
        name=signal_name,
        start_bit=start_byte * 8,
        length=bit_length,
        byte_order=byte_order,
        value_type=value_type,
        scale=scale,
        offset=offset,
        unit=unit,
    )


def _decode_hypothesis_values(
    session: CANSession, arbitration_id: int, definition: SignalDefinition
) -> list[float]:
    """Decode all values using signal definition."""

    filtered = session._messages.filter_by_id(arbitration_id)
    decoded_values = []
    for msg in filtered.messages:
        try:
            value = definition.decode(msg.data)
            decoded_values.append(value)
        except Exception:
            pass
    return decoded_values


def _create_failed_hypothesis_result(
    signal_name: str, definition: SignalDefinition
) -> HypothesisResult:
    """Create result for failed decoding."""
    return HypothesisResult(
        signal_name=signal_name,
        definition=definition,
        values=[],
        min_value=0.0,
        max_value=0.0,
        mean=0.0,
        std=0.0,
        is_valid=False,
        confidence=0.0,
        feedback="Failed to decode any messages with this definition",
    )


def _calculate_hypothesis_statistics(decoded_values: list[float]) -> dict[str, float]:
    """Calculate statistics for decoded values."""
    arr = np.array(decoded_values)
    return {
        "min_value": float(np.min(arr)),
        "max_value": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _validate_hypothesis(
    stats: dict[str, float], expected_min: float | None, expected_max: float | None
) -> tuple[bool, float, str]:
    """Validate hypothesis and generate feedback."""
    is_valid = True
    confidence = 1.0
    feedback_parts = []

    # Check expected range
    if expected_min is not None and stats["min_value"] < expected_min:
        is_valid = False
        confidence *= 0.5
        feedback_parts.append(
            f"Min value {stats['min_value']:.2f} below expected {expected_min:.2f}"
        )

    if expected_max is not None and stats["max_value"] > expected_max:
        is_valid = False
        confidence *= 0.5
        feedback_parts.append(
            f"Max value {stats['max_value']:.2f} above expected {expected_max:.2f}"
        )

    # Check value distribution
    if stats["std"] == 0:
        confidence *= 0.7
        feedback_parts.append("Warning: All values are identical - might be a constant field")

    value_range = stats["max_value"] - stats["min_value"]
    if value_range > 1e6:
        confidence *= 0.6
        feedback_parts.append("Warning: Very large value range - check scaling factor")

    # Positive feedback
    if is_valid and not feedback_parts:
        feedback_parts.append(
            f"Values in expected range [{stats['min_value']:.2f}, {stats['max_value']:.2f}]"
        )
        if stats["std"] > 0:
            feedback_parts.append("Signal shows variation - likely represents real data")

    feedback = "; ".join(feedback_parts) if feedback_parts else "Hypothesis test passed"
    return is_valid, confidence, feedback
