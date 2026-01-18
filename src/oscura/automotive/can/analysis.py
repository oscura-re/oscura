"""CAN message analysis algorithms.

This module implements statistical analysis algorithms for CAN message reverse
engineering, including entropy analysis, counter detection, and signal boundary
suggestion.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.automotive.can.checksum import ChecksumDetector
from oscura.automotive.can.models import (
    ByteAnalysis,
    CounterPattern,
    MessageAnalysis,
)

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessageList

__all__ = ["MessageAnalyzer"]


class MessageAnalyzer:
    """Analyze CAN messages for reverse engineering.

    This class implements various statistical and pattern detection algorithms
    to aid in reverse engineering CAN bus protocols.
    """

    @staticmethod
    def calculate_entropy(values: list[int]) -> float:
        """Calculate Shannon entropy of byte values.

        Args:
            values: List of byte values (0-255).

        Returns:
            Shannon entropy in bits (0.0-8.0).
        """
        if not values:
            return 0.0

        # Count occurrences
        counts = Counter(values)
        total = len(values)

        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    def detect_counter(values: list[int], max_value: int = 255) -> CounterPattern | None:
        """Detect if a sequence of values represents a counter.

        Args:
            values: List of byte values.
            max_value: Maximum counter value before wrap.

        Returns:
            CounterPattern if counter detected, None otherwise.
        """
        if len(values) < 3:
            return None

        # Calculate differences (handling wraparound)
        diffs = []
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            # Handle wraparound
            if diff < 0:
                diff += max_value + 1
            diffs.append(diff)

        # Check if diffs are consistent (most diffs should be the same)
        diff_counts = Counter(diffs)
        if not diff_counts:
            return None

        # Most common difference
        most_common_diff, count = diff_counts.most_common(1)[0]

        # Calculate confidence
        confidence = count / len(diffs)

        # Must be reasonably consistent to be a counter
        if confidence < 0.7:
            return None

        # Common increment values: 1, 2, 4, etc.
        if most_common_diff not in [1, 2, 4, 8, 16]:
            # Could be a sequence but not a simple counter
            if confidence > 0.9:
                pattern_type: Literal["counter", "sequence", "toggle"] = "sequence"
            else:
                return None
        else:
            pattern_type = "counter"

        # Detect wrap value
        wrap_value = max(values)
        if wrap_value == max_value:
            wraps_at = max_value
        else:
            # Might wrap at a power of 2
            for candidate in [15, 31, 63, 127, 255]:
                if wrap_value <= candidate:
                    wraps_at = candidate
                    break
            else:
                wraps_at = 255

        return CounterPattern(
            byte_position=0,  # Will be set by caller
            increment=most_common_diff,
            wraps_at=wraps_at,
            confidence=confidence,
            pattern_type=pattern_type,
        )

    @staticmethod
    def analyze_byte(messages: CANMessageList, byte_position: int) -> ByteAnalysis:
        """Analyze a specific byte position across multiple messages.

        Args:
            messages: Collection of CAN messages with the same ID.
            byte_position: Byte position to analyze (0-7).

        Returns:
            ByteAnalysis with statistical information.
        """
        # Extract byte values
        values = []
        for msg in messages:
            if len(msg.data) > byte_position:
                values.append(msg.data[byte_position])

        if not values:
            # No data at this position
            return ByteAnalysis(
                position=byte_position,
                entropy=0.0,
                min_value=0,
                max_value=0,
                mean=0.0,
                std=0.0,
                is_constant=True,
                unique_values=0,
                most_common_value=0,
                change_rate=0.0,
            )

        # Calculate statistics
        arr = np.array(values)
        min_val = int(np.min(arr))
        max_val = int(np.max(arr))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))

        # Entropy
        entropy = MessageAnalyzer.calculate_entropy(values)

        # Unique values
        unique_vals = len(set(values))
        is_constant = unique_vals == 1

        # Most common value
        counter = Counter(values)
        most_common = counter.most_common(1)[0][0]

        # Change rate
        changes = sum(1 for i in range(1, len(values)) if values[i] != values[i - 1])
        change_rate = changes / (len(values) - 1) if len(values) > 1 else 0.0

        return ByteAnalysis(
            position=byte_position,
            entropy=entropy,
            min_value=min_val,
            max_value=max_val,
            mean=mean_val,
            std=std_val,
            is_constant=is_constant,
            unique_values=unique_vals,
            most_common_value=most_common,
            change_rate=change_rate,
        )

    @staticmethod
    def suggest_signal_boundaries(
        byte_analyses: list[ByteAnalysis],
    ) -> list[dict[str, Any]]:
        """Suggest likely signal boundaries based on entropy analysis.

        Args:
            byte_analyses: List of per-byte analyses.

        Returns:
            List of suggested signal definitions (as dicts).
        """
        suggestions = []

        # Group contiguous variable bytes
        i = 0
        while i < len(byte_analyses):
            ba = byte_analyses[i]

            # Skip constant bytes
            if ba.is_constant:
                i += 1
                continue

            # Found a variable byte - see how far it extends
            start_byte = i
            end_byte = i

            # Look ahead for contiguous variable bytes
            while end_byte + 1 < len(byte_analyses) and not byte_analyses[end_byte + 1].is_constant:
                end_byte += 1

            # Suggest signal
            num_bytes = end_byte - start_byte + 1
            suggestions.append(
                {
                    "start_byte": start_byte,
                    "num_bytes": num_bytes,
                    "start_bit": start_byte * 8,
                    "length_bits": num_bytes * 8,
                    "entropy_range": (
                        min(byte_analyses[j].entropy for j in range(start_byte, end_byte + 1)),
                        max(byte_analyses[j].entropy for j in range(start_byte, end_byte + 1)),
                    ),
                    "suggested_types": MessageAnalyzer._suggest_types(
                        byte_analyses[start_byte : end_byte + 1]
                    ),
                }
            )

            i = end_byte + 1

        return suggestions

    @staticmethod
    def _suggest_types(byte_analyses: list[ByteAnalysis]) -> list[str]:
        """Suggest possible data types based on byte patterns.

        Args:
            byte_analyses: Analyses for consecutive bytes.

        Returns:
            List of suggested type names.
        """
        num_bytes = len(byte_analyses)
        suggestions = []

        # Based on size, suggest common types
        if num_bytes == 1:
            suggestions.append("uint8")
            suggestions.append("int8")
        elif num_bytes == 2:
            suggestions.append("uint16")
            suggestions.append("int16")
        elif num_bytes == 4:
            suggestions.append("uint32")
            suggestions.append("int32")
            suggestions.append("float32")

        # Check if values suggest specific ranges
        if num_bytes == 2:
            # Common automotive scaling
            max_val = max(ba.max_value for ba in byte_analyses)
            if max_val <= 100:
                suggestions.append("percentage")
            elif max_val <= 8000:
                suggestions.append("rpm (if scaled by 0.25)")

        return suggestions

    @staticmethod
    def analyze_message_id(messages: CANMessageList, arbitration_id: int) -> MessageAnalysis:
        """Perform complete analysis on all messages with a specific ID.

        Args:
            messages: All messages (will be filtered by ID).
            arbitration_id: CAN ID to analyze.

        Returns:
            MessageAnalysis with complete analysis results.

        Raises:
            ValueError: If no messages found for the specified ID.
        """
        # Filter to this ID
        filtered = messages.filter_by_id(arbitration_id)

        if not filtered.messages:
            raise ValueError(f"No messages found for ID 0x{arbitration_id:03X}")

        # Calculate timing statistics
        timestamps = np.array([msg.timestamp for msg in filtered.messages])
        periods = np.diff(timestamps)

        if len(periods) > 0:
            period_ms = float(np.mean(periods) * 1000)
            period_jitter_ms = float(np.std(periods) * 1000)
            frequency_hz = 1.0 / np.mean(periods) if np.mean(periods) > 0 else 0.0
        else:
            period_ms = 0.0
            period_jitter_ms = 0.0
            frequency_hz = 0.0

        # Determine max DLC
        max_dlc = max(msg.dlc for msg in filtered.messages)

        # Analyze each byte position
        byte_analyses = []
        for byte_pos in range(max_dlc):
            analysis = MessageAnalyzer.analyze_byte(filtered, byte_pos)
            byte_analyses.append(analysis)

        # Detect counters
        detected_counters = []
        for byte_pos in range(max_dlc):
            values = [msg.data[byte_pos] for msg in filtered.messages if len(msg.data) > byte_pos]
            counter = MessageAnalyzer.detect_counter(values)
            if counter:
                counter.byte_position = byte_pos
                detected_counters.append(counter)

        # Suggest signal boundaries
        suggested_signals = MessageAnalyzer.suggest_signal_boundaries(byte_analyses)

        # Detect checksum
        detected_checksum = ChecksumDetector.detect_checksum(filtered)

        # Create analysis result
        return MessageAnalysis(
            arbitration_id=arbitration_id,
            message_count=len(filtered.messages),
            frequency_hz=frequency_hz,
            period_ms=period_ms,
            period_jitter_ms=period_jitter_ms,
            byte_analyses=byte_analyses,
            detected_counters=detected_counters,
            detected_checksum=detected_checksum,
            suggested_signals=suggested_signals,
            correlations={},  # Will be set by correlation analysis
        )
