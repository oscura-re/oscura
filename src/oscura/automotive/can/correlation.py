"""Signal and message correlation analysis.

This module provides correlation analysis for CAN signals to discover
relationships between different signals and messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessageList, SignalDefinition
    from oscura.automotive.can.session import CANSession

__all__ = ["CorrelationAnalyzer"]


class CorrelationAnalyzer:
    """Analyze correlations between CAN signals and messages."""

    @staticmethod
    def correlate_signals(
        messages1: CANMessageList,
        signal_def1: SignalDefinition,
        messages2: CANMessageList,
        signal_def2: SignalDefinition,
        max_time_shift: float = 0.0,
    ) -> dict[str, float | int]:
        """Compute correlation between two signals.

        Args:
            messages1: Messages containing first signal.
            signal_def1: First signal definition.
            messages2: Messages containing second signal.
            signal_def2: Second signal definition.
            max_time_shift: Maximum time shift to consider (seconds).

        Returns:
            Dictionary with correlation results:
                - correlation: Pearson correlation coefficient
                - p_value: Statistical significance
                - lag: Time lag with maximum correlation (if time shift enabled)
                - sample_count: Number of samples used
        """
        # Decode signals
        values1 = []
        timestamps1 = []
        for msg in messages1.messages:
            try:
                value = signal_def1.decode(msg.data)
                values1.append(value)
                timestamps1.append(msg.timestamp)
            except Exception:
                pass

        values2 = []
        timestamps2 = []
        for msg in messages2.messages:
            try:
                value = signal_def2.decode(msg.data)
                values2.append(value)
                timestamps2.append(msg.timestamp)
            except Exception:
                pass

        if len(values1) < 2 or len(values2) < 2:
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "lag": 0.0,
                "sample_count": 0,
            }

        # Align signals by timestamp (simple approach: nearest neighbor)
        aligned_values1 = []
        aligned_values2 = []

        for t1, v1 in zip(timestamps1, values1, strict=False):
            # Find closest timestamp in signal 2
            time_diffs = [abs(t2 - t1) for t2 in timestamps2]
            if min(time_diffs) <= max(0.1, max_time_shift):  # Within 100ms or max_time_shift
                closest_idx = time_diffs.index(min(time_diffs))
                aligned_values1.append(v1)
                aligned_values2.append(values2[closest_idx])

        if len(aligned_values1) < 2:
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "lag": 0.0,
                "sample_count": 0,
            }

        # Compute Pearson correlation
        arr1 = np.array(aligned_values1)
        arr2 = np.array(aligned_values2)

        correlation, p_value = pearsonr(arr1, arr2)

        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "lag": 0.0,
            "sample_count": len(aligned_values1),
        }

    @staticmethod
    def correlate_bytes(
        messages1: CANMessageList,
        byte_pos1: int,
        messages2: CANMessageList,
        byte_pos2: int,
    ) -> float:
        """Compute correlation between two byte positions.

        Args:
            messages1: First message collection.
            byte_pos1: Byte position in first message.
            messages2: Second message collection.
            byte_pos2: Byte position in second message.

        Returns:
            Pearson correlation coefficient (-1.0 to 1.0).
        """
        # Extract byte values
        values1 = []
        for msg in messages1.messages:
            if len(msg.data) > byte_pos1:
                values1.append(msg.data[byte_pos1])

        values2 = []
        for msg in messages2.messages:
            if len(msg.data) > byte_pos2:
                values2.append(msg.data[byte_pos2])

        if len(values1) < 2 or len(values2) < 2:
            return 0.0

        # Truncate to same length
        min_len = min(len(values1), len(values2))
        values1 = values1[:min_len]
        values2 = values2[:min_len]

        # Compute correlation
        arr1 = np.array(values1, dtype=float)
        arr2 = np.array(values2, dtype=float)

        # Check for zero variance
        if np.std(arr1) == 0 or np.std(arr2) == 0:
            return 0.0

        correlation, _ = pearsonr(arr1, arr2)

        return float(correlation)

    @staticmethod
    def find_correlated_messages(
        session: CANSession,
        arbitration_id: int,
        threshold: float = 0.7,
    ) -> dict[int, float]:
        """Find messages correlated with a given message ID.

        Args:
            session: CANSession to analyze.
            arbitration_id: CAN ID to find correlations for.
            threshold: Minimum correlation threshold (0.0-1.0).

        Returns:
            Dictionary mapping correlated message IDs to correlation scores.
        """

        correlations = {}

        # Get messages for the target ID
        target_messages = session._messages.filter_by_id(arbitration_id)
        if not target_messages.messages:
            return {}

        # Determine max DLC for target
        max_dlc_target = max(msg.dlc for msg in target_messages.messages)

        # Check all other unique IDs
        for other_id in session.unique_ids():
            if other_id == arbitration_id:
                continue

            other_messages = session._messages.filter_by_id(other_id)
            max_dlc_other = max(msg.dlc for msg in other_messages.messages)

            # Check byte-by-byte correlation
            max_correlation = 0.0

            for byte_pos_target in range(max_dlc_target):
                for byte_pos_other in range(max_dlc_other):
                    corr = CorrelationAnalyzer.correlate_bytes(
                        target_messages,
                        byte_pos_target,
                        other_messages,
                        byte_pos_other,
                    )

                    if abs(corr) > abs(max_correlation):
                        max_correlation = corr

            if abs(max_correlation) >= threshold:
                correlations[other_id] = max_correlation

        return correlations
