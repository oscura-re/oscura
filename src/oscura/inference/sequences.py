"""Sequence pattern detection and request-response correlation.

    - RE-SEQ-002: Sequence Pattern Detection
    - RE-SEQ-003: Request-Response Correlation

This module provides tools for detecting sequential patterns in message
streams, identifying request-response pairs, and analyzing communication
flows.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SequencePattern:
    """A detected sequence pattern.

    Implements RE-SEQ-002: Sequence pattern representation.

    Attributes:
        pattern: List of message identifiers in sequence.
        frequency: Number of occurrences.
        positions: Starting positions in stream.
        confidence: Detection confidence (0-1).
        avg_gap: Average gap between elements.
        gap_variance: Variance in inter-element gaps.
    """

    pattern: list[Any]
    frequency: int
    positions: list[int] = field(default_factory=list)
    confidence: float = 0.0
    avg_gap: float = 0.0
    gap_variance: float = 0.0


@dataclass
class RequestResponsePair:
    """A correlated request-response pair.

    Implements RE-SEQ-003: Request-response pair.

    Attributes:
        request_index: Index of request message.
        response_index: Index of response message.
        request: Request message data.
        response: Response message data.
        latency: Time between request and response.
        correlation_id: Detected correlation identifier.
        confidence: Correlation confidence (0-1).
    """

    request_index: int
    response_index: int
    request: Any
    response: Any
    latency: float
    correlation_id: bytes | int | None = None
    confidence: float = 0.0


@dataclass
class CommunicationFlow:
    """A complete communication flow.

    Implements RE-SEQ-003: Communication flow.

    Attributes:
        flow_id: Unique flow identifier.
        messages: List of messages in flow.
        pairs: Request-response pairs.
        direction: Primary direction ('request_first' or 'response_first').
        participants: Identified participants.
        duration: Total flow duration.
    """

    flow_id: int
    messages: list[Any]
    pairs: list["RequestResponsePair"]
    direction: str
    participants: list[str]
    duration: float


class SequencePatternDetector:
    """Detect sequential patterns in message streams.

    Implements RE-SEQ-002: Sequence Pattern Detection.

    Identifies recurring patterns of message types or values in
    communication streams.

    Example:
        >>> detector = SequencePatternDetector()
        >>> patterns = detector.detect_patterns(messages, key=lambda m: m.type)
    """

    def __init__(
        self,
        min_pattern_length: int = 2,
        max_pattern_length: int = 10,
        min_frequency: int = 2,
        max_gap: float | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            min_pattern_length: Minimum pattern length.
            max_pattern_length: Maximum pattern length.
            min_frequency: Minimum occurrences to consider.
            max_gap: Maximum allowed gap between pattern elements.
        """
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_frequency = min_frequency
        self.max_gap = max_gap

    def detect_patterns(
        self,
        messages: Sequence[Any],
        key: Callable[[Any], Any] | None = None,
        timestamp_key: Callable[[Any], float] | None = None,
    ) -> list["SequencePattern"]:
        """Detect sequential patterns in message stream.

        Implements RE-SEQ-002: Pattern detection workflow.

        Args:
            messages: Sequence of messages.
            key: Function to extract message identifier.
            timestamp_key: Function to extract timestamp.

        Returns:
            List of detected patterns.

        Example:
            >>> patterns = detector.detect_patterns(
            ...     messages,
            ...     key=lambda m: m.get('type'),
            ...     timestamp_key=lambda m: m.get('timestamp')
            ... )
        """
        if not messages:
            return []

        # Extract identifiers
        if key is not None:
            identifiers = [key(m) for m in messages]
        else:
            identifiers = list(messages)

        # Extract timestamps if provided
        timestamps = None
        if timestamp_key is not None:
            timestamps = [timestamp_key(m) for m in messages]

        # Find all n-grams
        candidates = self._find_ngram_patterns(identifiers, timestamps)

        # Filter and score
        patterns = self._score_patterns(candidates, identifiers, timestamps)

        # Sort by confidence
        patterns.sort(key=lambda p: (-p.confidence, -p.frequency))

        return patterns

    def find_repeating_sequences(
        self,
        messages: Sequence[Any],
        key: Callable[[Any], Any] | None = None,
    ) -> list[tuple[list[Any], int, list[int]]]:
        """Find exactly repeating message sequences.

        Implements RE-SEQ-002: Exact sequence detection.

        Args:
            messages: Sequence of messages.
            key: Function to extract message identifier.

        Returns:
            List of (sequence, count, positions) tuples.
        """
        if not messages:
            return []

        # Extract identifiers
        if key is not None:
            identifiers = tuple(key(m) for m in messages)
        else:
            identifiers = tuple(messages)

        results = []

        for length in range(self.min_pattern_length, self.max_pattern_length + 1):
            # Count n-grams
            ngram_positions = defaultdict(list)

            for i in range(len(identifiers) - length + 1):
                ngram = identifiers[i : i + length]
                ngram_positions[ngram].append(i)

            # Filter by frequency
            for ngram, positions in ngram_positions.items():
                if len(positions) >= self.min_frequency:
                    results.append((list(ngram), len(positions), positions))

        # Sort by frequency
        results.sort(key=lambda x: -x[1])

        return results

    def detect_periodic_patterns(
        self,
        messages: Sequence[Any],
        key: Callable[[Any], Any] | None = None,
        timestamp_key: Callable[[Any], float] | None = None,
    ) -> list["SequencePattern"]:
        """Detect patterns that occur at regular intervals.

        Implements RE-SEQ-002: Periodic pattern detection.

        Args:
            messages: Sequence of messages.
            key: Function to extract message identifier.
            timestamp_key: Function to extract timestamp.

        Returns:
            List of periodic patterns.
        """
        patterns = self.detect_patterns(messages, key, timestamp_key)

        # Filter for low gap variance (periodic)
        periodic = []
        for pattern in patterns:
            if pattern.frequency >= 3 and pattern.avg_gap > 0:
                # Calculate coefficient of variation
                if pattern.gap_variance > 0:
                    cv = (pattern.gap_variance**0.5) / pattern.avg_gap
                else:
                    cv = 0
                # Low CV indicates periodicity
                if cv < 0.2:  # Less than 20% variation
                    periodic.append(pattern)

        return periodic

    def _find_ngram_patterns(
        self,
        identifiers: list[Any],
        timestamps: list[float] | None,
    ) -> dict[tuple[Any, ...], list[int]]:
        """Find all n-gram patterns.

        Args:
            identifiers: Message identifiers.
            timestamps: Message timestamps.

        Returns:
            Dictionary mapping patterns to positions.
        """
        candidates = defaultdict(list)

        for length in range(self.min_pattern_length, self.max_pattern_length + 1):
            for i in range(len(identifiers) - length + 1):
                ngram = tuple(identifiers[i : i + length])

                # Check gap constraint
                if self.max_gap is not None and timestamps is not None:
                    gaps = [timestamps[i + j + 1] - timestamps[i + j] for j in range(length - 1)]
                    if any(g > self.max_gap for g in gaps):
                        continue

                candidates[ngram].append(i)

        return candidates

    def _score_patterns(
        self,
        candidates: dict[tuple[Any, ...], list[int]],
        identifiers: list[Any],
        timestamps: list[float] | None,
    ) -> list["SequencePattern"]:
        """Score candidate patterns.

        Args:
            candidates: Pattern -> positions mapping.
            identifiers: Original identifiers.
            timestamps: Message timestamps.

        Returns:
            List of scored patterns.
        """
        patterns = []

        for pattern_tuple, positions in candidates.items():
            if len(positions) < self.min_frequency:
                continue

            pattern_list = list(pattern_tuple)
            length = len(pattern_list)

            # Calculate gap statistics if timestamps available
            avg_gap = 0.0
            gap_variance = 0.0

            if timestamps is not None and len(positions) > 1:
                # Calculate gaps between pattern occurrences
                gaps = []
                for i in range(len(positions) - 1):
                    start_time = timestamps[positions[i]]
                    end_time = timestamps[positions[i + 1]]
                    gaps.append(end_time - start_time)

                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    gap_variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)

            # Calculate confidence
            # Higher confidence for: frequent, consistent gaps, longer patterns
            frequency_score = min(1.0, len(positions) / 10)
            length_score = min(1.0, length / 5)

            if gap_variance > 0 and avg_gap > 0:
                consistency_score = 1.0 / (1.0 + (gap_variance**0.5 / avg_gap))
            else:
                consistency_score = 0.5

            confidence = 0.4 * frequency_score + 0.3 * consistency_score + 0.3 * length_score

            patterns.append(
                SequencePattern(
                    pattern=pattern_list,
                    frequency=len(positions),
                    positions=positions,
                    confidence=confidence,
                    avg_gap=avg_gap,
                    gap_variance=gap_variance,
                )
            )

        return patterns


class RequestResponseCorrelator:
    """Correlate request and response messages.

    Implements RE-SEQ-003: Request-Response Correlation.

    Identifies matching request-response pairs in bidirectional
    communication streams.

    Example:
        >>> correlator = RequestResponseCorrelator()
        >>> pairs = correlator.correlate(
        ...     messages,
        ...     request_filter=lambda m: m.type == 'REQ',
        ...     response_filter=lambda m: m.type == 'RSP'
        ... )
    """

    def __init__(
        self,
        max_latency: float = 10.0,
        correlation_key: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize correlator.

        Args:
            max_latency: Maximum time between request and response.
            correlation_key: Function to extract correlation ID.
        """
        self.max_latency = max_latency
        self.correlation_key = correlation_key

    def correlate(
        self,
        messages: Sequence[Any],
        request_filter: Callable[[Any], bool] | None = None,
        response_filter: Callable[[Any], bool] | None = None,
        timestamp_key: Callable[[Any], float] | None = None,
    ) -> list["RequestResponsePair"]:
        """Correlate requests with responses.

        Implements RE-SEQ-003: Request-response correlation workflow.

        Args:
            messages: All messages in stream.
            request_filter: Function to identify requests.
            response_filter: Function to identify responses.
            timestamp_key: Function to extract timestamp.

        Returns:
            List of correlated pairs.

        Example:
            >>> pairs = correlator.correlate(
            ...     messages,
            ...     request_filter=lambda m: m['direction'] == 'out',
            ...     response_filter=lambda m: m['direction'] == 'in',
            ...     timestamp_key=lambda m: m['time']
            ... )
        """
        # Separate requests and responses
        requests = []
        responses = []

        for i, msg in enumerate(messages):
            ts = timestamp_key(msg) if timestamp_key else float(i)

            if request_filter is None or request_filter(msg):
                correlation_id = None
                if self.correlation_key is not None:
                    try:
                        correlation_id = self.correlation_key(msg)
                    except (KeyError, TypeError):
                        pass
                requests.append((i, msg, ts, correlation_id))

            if response_filter is None or response_filter(msg):
                correlation_id = None
                if self.correlation_key is not None:
                    try:
                        correlation_id = self.correlation_key(msg)
                    except (KeyError, TypeError):
                        pass
                responses.append((i, msg, ts, correlation_id))

        # Match pairs
        return self._match_pairs(requests, responses)

    def correlate_by_content(
        self,
        messages: Sequence[Any],
        content_key: Callable[[Any], bytes],
        timestamp_key: Callable[[Any], float] | None = None,
    ) -> list["RequestResponsePair"]:
        """Correlate by analyzing message content similarity.

        Implements RE-SEQ-003: Content-based correlation.

        Args:
            messages: All messages.
            content_key: Function to extract message content.
            timestamp_key: Function to extract timestamp.

        Returns:
            List of correlated pairs.
        """
        pairs = []
        used_responses = set()

        for i, msg in enumerate(messages):
            req_content = content_key(msg)
            req_ts = timestamp_key(msg) if timestamp_key else float(i)

            best_match = None
            best_score = 0.0

            for j in range(i + 1, len(messages)):
                if j in used_responses:
                    continue

                resp = messages[j]
                resp_ts = timestamp_key(resp) if timestamp_key else float(j)

                latency = resp_ts - req_ts
                if latency < 0 or latency > self.max_latency:
                    continue

                resp_content = content_key(resp)
                score = self._content_similarity(req_content, resp_content)

                if score > best_score:
                    best_score = score
                    best_match = (j, resp, latency)

            if best_match is not None and best_score > 0.3:
                j, resp, latency = best_match
                used_responses.add(j)
                pairs.append(
                    RequestResponsePair(
                        request_index=i,
                        response_index=j,
                        request=msg,
                        response=resp,
                        latency=latency,
                        confidence=best_score,
                    )
                )

        return pairs

    def extract_flows(
        self,
        pairs: Sequence["RequestResponsePair"],
        messages: Sequence[Any],
        flow_key: Callable[[Any], str] | None = None,
    ) -> list["CommunicationFlow"]:
        """Extract communication flows from pairs.

        Implements RE-SEQ-003: Flow extraction.

        Args:
            pairs: Correlated request-response pairs.
            messages: All messages.
            flow_key: Function to extract flow identifier.

        Returns:
            List of communication flows.
        """
        if flow_key is None:
            # Group all pairs into one flow
            return [
                CommunicationFlow(
                    flow_id=0,
                    messages=list(messages),
                    pairs=list(pairs),
                    direction="request_first",
                    participants=[],
                    duration=0.0,
                )
            ]

        # Group by flow key
        flow_groups = defaultdict(list)
        for pair in pairs:
            key = flow_key(pair.request)
            flow_groups[key].append(pair)

        flows = []
        for i, (key, group_pairs) in enumerate(flow_groups.items()):
            # Get all messages in this flow
            indices = set()
            for pair in group_pairs:
                indices.add(pair.request_index)
                indices.add(pair.response_index)

            flow_messages = [messages[j] for j in sorted(indices)]

            # Calculate duration
            if group_pairs:
                _start = min(p.latency for p in group_pairs)
                duration = max(p.latency for p in group_pairs)
            else:
                duration = 0.0

            flows.append(
                CommunicationFlow(
                    flow_id=i,
                    messages=flow_messages,
                    pairs=group_pairs,
                    direction="request_first",
                    participants=[str(key)],
                    duration=duration,
                )
            )

        return flows

    def _match_pairs(
        self,
        requests: list[tuple[int, Any, float, Any]],
        responses: list[tuple[int, Any, float, Any]],
    ) -> list["RequestResponsePair"]:
        """Match request and response messages.

        Args:
            requests: List of (index, message, timestamp, correlation_id).
            responses: List of (index, message, timestamp, correlation_id).

        Returns:
            List of matched pairs.
        """
        pairs = []
        used_responses = set()

        for req_idx, req_msg, req_ts, req_id in requests:
            best_match = None
            best_score = 0.0

            for resp_idx, resp_msg, resp_ts, resp_id in responses:
                if resp_idx in used_responses:
                    continue

                # Check timing
                latency = resp_ts - req_ts
                if latency < 0 or latency > self.max_latency:
                    continue

                # Check correlation ID
                if req_id is not None and resp_id is not None:
                    if req_id == resp_id:
                        score = 1.0
                    else:
                        score = 0.0
                else:
                    # Use timing proximity
                    score = 1.0 - (latency / self.max_latency)

                if score > best_score:
                    best_score = score
                    best_match = (resp_idx, resp_msg, latency, resp_id)

            if best_match is not None:
                resp_idx, resp_msg, latency, resp_id = best_match
                used_responses.add(resp_idx)

                pairs.append(
                    RequestResponsePair(
                        request_index=req_idx,
                        response_index=resp_idx,
                        request=req_msg,
                        response=resp_msg,
                        latency=latency,
                        correlation_id=req_id if req_id is not None else resp_id,
                        confidence=best_score,
                    )
                )

        return pairs

    def _content_similarity(self, content_a: bytes, content_b: bytes) -> float:
        """Calculate content similarity.

        Args:
            content_a: First content.
            content_b: Second content.

        Returns:
            Similarity score (0-1).
        """
        if not content_a or not content_b:
            return 0.0

        # Use byte set similarity (Jaccard-like)
        set_a = set(content_a)
        set_b = set(content_b)

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        if union == 0:
            return 0.0

        return intersection / union


# =============================================================================
# Convenience functions
# =============================================================================


def detect_sequence_patterns(
    messages: Sequence[Any],
    key: Callable[[Any], Any] | None = None,
    min_length: int = 2,
    max_length: int = 10,
    min_frequency: int = 2,
) -> list["SequencePattern"]:
    """Detect sequential patterns in messages.

    Implements RE-SEQ-002: Sequence Pattern Detection.

    Args:
        messages: Sequence of messages.
        key: Function to extract message identifier.
        min_length: Minimum pattern length.
        max_length: Maximum pattern length.
        min_frequency: Minimum occurrences.

    Returns:
        List of detected patterns.

    Example:
        >>> patterns = detect_sequence_patterns(
        ...     messages,
        ...     key=lambda m: m['type']
        ... )
    """
    detector = SequencePatternDetector(
        min_pattern_length=min_length,
        max_pattern_length=max_length,
        min_frequency=min_frequency,
    )
    return detector.detect_patterns(messages, key)


def correlate_requests(
    messages: Sequence[Any],
    request_filter: Callable[[Any], bool],
    response_filter: Callable[[Any], bool],
    timestamp_key: Callable[[Any], float] | None = None,
    max_latency: float = 10.0,
) -> list["RequestResponsePair"]:
    """Correlate request and response messages.

    Implements RE-SEQ-003: Request-Response Correlation.

    Args:
        messages: All messages.
        request_filter: Function to identify requests.
        response_filter: Function to identify responses.
        timestamp_key: Function to extract timestamp.
        max_latency: Maximum time between request and response.

    Returns:
        List of correlated pairs.

    Example:
        >>> pairs = correlate_requests(
        ...     messages,
        ...     request_filter=lambda m: m['dir'] == 'out',
        ...     response_filter=lambda m: m['dir'] == 'in',
        ...     timestamp_key=lambda m: m['ts']
        ... )
    """
    correlator = RequestResponseCorrelator(max_latency=max_latency)
    return correlator.correlate(messages, request_filter, response_filter, timestamp_key)


def find_message_dependencies(
    messages: Sequence[Any],
    key: Callable[[Any], Any],
    timestamp_key: Callable[[Any], float] | None = None,
) -> dict[Any, list[Any]]:
    """Find dependencies between message types.

    Implements RE-SEQ-002: Dependency detection.

    Args:
        messages: Sequence of messages.
        key: Function to extract message type.
        timestamp_key: Function to extract timestamp.

    Returns:
        Dictionary mapping message types to their typical successors.

    Example:
        >>> deps = find_message_dependencies(messages, key=lambda m: m['type'])
        >>> deps['REQ']  # ['RSP', 'ACK']
    """
    dependencies: dict[Any, list[Any]] = defaultdict(list)

    for i in range(len(messages) - 1):
        current = key(messages[i])
        next_msg = key(messages[i + 1])

        if next_msg not in dependencies[current]:
            dependencies[current].append(next_msg)

    return dict(dependencies)


def calculate_latency_stats(
    pairs: Sequence["RequestResponsePair"],
) -> dict[str, float]:
    """Calculate latency statistics for request-response pairs.

    Implements RE-SEQ-003: Latency analysis.

    Args:
        pairs: List of request-response pairs.

    Returns:
        Dictionary with min, max, mean, median, std latencies.
    """
    if not pairs:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }

    latencies = [p.latency for p in pairs]
    arr = np.array(latencies)

    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


__all__ = [
    "CommunicationFlow",
    "RequestResponseCorrelator",
    "RequestResponsePair",
    # Data classes
    "SequencePattern",
    # Classes
    "SequencePatternDetector",
    "calculate_latency_stats",
    "correlate_requests",
    # Functions
    "detect_sequence_patterns",
    "find_message_dependencies",
]
