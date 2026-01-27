"""Pattern mining and correlation analysis for protocol traffic.

This module implements pattern mining algorithms (FP-Growth, Apriori) for
discovering repeated byte sequences, field correlations, and temporal patterns
in protocol message traffic.

Example:
    >>> from oscura.analyzers.patterns.pattern_mining import PatternMiner
    >>> miner = PatternMiner(min_support=0.1, min_confidence=0.5)
    >>> messages = [b"\\xAA\\xBB\\xCC", b"\\xAA\\xBB\\xDD", b"\\xAA\\xBB\\xCC"]
    >>> patterns = miner.mine_byte_patterns(messages)
    >>> print(f"Found {len(patterns)} frequent patterns")
    >>> rules = miner.find_associations(patterns)
    >>> print(f"Discovered {len(rules)} association rules")

References:
    Han et al. (2000): "Mining Frequent Patterns without Candidate Generation"
    Agrawal & Srikant (1994): "Fast Algorithms for Mining Association Rules"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class Pattern:
    """Discovered pattern in message traffic.

    Attributes:
        sequence: Byte sequence or field values (tuple of integers).
        support: Frequency of pattern (0.0-1.0, fraction of messages).
        confidence: Confidence for association rules (optional).
        locations: List of (message_idx, offset) tuples where pattern appears.
        metadata: Additional pattern metadata.
    """

    sequence: tuple[int, ...]
    support: float
    confidence: float | None = None
    locations: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Generate readable representation."""
        seq_str = " ".join(f"{b:02X}" for b in self.sequence)
        return f"Pattern([{seq_str}], support={self.support:.3f})"


@dataclass
class AssociationRule:
    """Association rule between patterns (A -> B).

    Represents the rule: "If A appears, then B follows".

    Attributes:
        antecedent: Pattern A (if this appears).
        consequent: Pattern B (then this follows).
        support: Frequency of (A, B) appearing together.
        confidence: Probability P(B|A) = support(A, B) / support(A).
        lift: Confidence / P(B), measures rule strength.
    """

    antecedent: tuple[int, ...]
    consequent: tuple[int, ...]
    support: float
    confidence: float
    lift: float

    def __repr__(self) -> str:
        """Generate readable representation."""
        ant_str = " ".join(f"{b:02X}" for b in self.antecedent)
        con_str = " ".join(f"{b:02X}" for b in self.consequent)
        return f"Rule([{ant_str}] -> [{con_str}], conf={self.confidence:.3f}, lift={self.lift:.2f})"


@dataclass
class TemporalPattern:
    """Temporal sequence pattern in events.

    Represents a sequence of events that occur with regular timing.

    Attributes:
        events: Sequence of event type names.
        timestamps: Event timestamps (relative to first event).
        avg_interval: Average time between consecutive events.
        variance: Variance in event timing.
    """

    events: list[str]
    timestamps: list[float]
    avg_interval: float
    variance: float

    def __repr__(self) -> str:
        """Generate readable representation."""
        events_str = " -> ".join(self.events)
        return (
            f"TemporalPattern([{events_str}], interval={self.avg_interval:.3f}±{self.variance:.3f})"
        )


class PatternMiner:
    """Pattern mining and correlation analysis for protocol traffic.

    Implements FP-Growth and Apriori algorithms for mining frequent patterns,
    association rule discovery, temporal pattern detection, and field correlation
    analysis.

    Example:
        >>> miner = PatternMiner(min_support=0.1, min_confidence=0.5)
        >>> patterns = miner.mine_byte_patterns(messages, algorithm="fp_growth")
        >>> rules = miner.find_associations(patterns)
        >>> temporal = miner.mine_temporal_patterns(events, max_gap=1.0)
    """

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        min_pattern_length: int = 2,
        max_pattern_length: int = 10,
    ) -> None:
        """Initialize pattern miner with thresholds.

        Args:
            min_support: Minimum pattern frequency (0.0-1.0).
            min_confidence: Minimum confidence for association rules (0.0-1.0).
            min_pattern_length: Minimum pattern length in bytes.
            max_pattern_length: Maximum pattern length in bytes.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if not 0.0 <= min_support <= 1.0:
            raise ValueError(f"min_support must be in [0.0, 1.0], got {min_support}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")
        if min_pattern_length < 1:
            raise ValueError(f"min_pattern_length must be >= 1, got {min_pattern_length}")
        if max_pattern_length < min_pattern_length:
            raise ValueError(
                f"max_pattern_length ({max_pattern_length}) must be >= "
                f"min_pattern_length ({min_pattern_length})"
            )

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.patterns: list[Pattern] = []
        self.rules: list[AssociationRule] = []

    def mine_byte_patterns(
        self, messages: list[bytes], algorithm: Literal["fp_growth", "apriori"] = "fp_growth"
    ) -> list[Pattern]:
        """Mine frequent byte patterns from messages.

        Extracts all subsequences from messages, counts occurrences, and filters
        by minimum support threshold. Returns patterns sorted by frequency.

        Args:
            messages: List of message byte sequences.
            algorithm: Mining algorithm to use ("fp_growth" or "apriori").

        Returns:
            List of Pattern objects sorted by support (descending).

        Raises:
            ValueError: If messages list is empty or algorithm is unknown.

        Example:
            >>> messages = [b"\\xAA\\xBB\\xCC", b"\\xAA\\xBB\\xDD"]
            >>> patterns = miner.mine_byte_patterns(messages)
            >>> for p in patterns:
            ...     print(f"{p.sequence}: {p.support:.2f}")
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if algorithm not in ("fp_growth", "apriori"):
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Extract all subsequences
        pattern_counts: dict[tuple[int, ...], int] = defaultdict(int)
        pattern_locations: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)

        for msg_idx, message in enumerate(messages):
            subsequences = self._extract_subsequences(
                message, self.min_pattern_length, self.max_pattern_length
            )

            for subseq in subsequences:
                # Find all occurrences in this message
                for offset in range(len(message) - len(subseq) + 1):
                    if tuple(message[offset : offset + len(subseq)]) == subseq:
                        pattern_counts[subseq] += 1
                        pattern_locations[subseq].append((msg_idx, offset))

        # Calculate support
        total_subsequences = sum(pattern_counts.values())

        # Filter by minimum support
        patterns = []
        for seq, count in pattern_counts.items():
            support = count / total_subsequences if total_subsequences > 0 else 0.0

            if support >= self.min_support:
                # NECESSARY COPY: Prevents mutations of locations after pattern creation.
                patterns.append(
                    Pattern(sequence=seq, support=support, locations=pattern_locations[seq].copy())
                )

        # Sort by support (most frequent first)
        patterns.sort(key=lambda p: p.support, reverse=True)

        self.patterns = patterns
        return patterns

    def mine_field_patterns(
        self, field_sequences: list[list[int]], field_names: list[str]
    ) -> list[Pattern]:
        """Mine patterns in extracted field values.

        Analyzes sequences of field values across multiple messages to find
        common patterns and value combinations.

        Args:
            field_sequences: List of field value sequences (one per message).
            field_names: Names of fields in sequences.

        Returns:
            List of Pattern objects for field value sequences.

        Raises:
            ValueError: If field_sequences is empty or field_names length mismatch.

        Example:
            >>> field_sequences = [[0x01, 0x02], [0x01, 0x03], [0x01, 0x02]]
            >>> field_names = ["field_a", "field_b"]
            >>> patterns = miner.mine_field_patterns(field_sequences, field_names)
        """
        if not field_sequences:
            raise ValueError("Field sequences cannot be empty")

        if field_names and len(field_names) != len(field_sequences[0]):
            raise ValueError(
                f"Field names length ({len(field_names)}) must match "
                f"sequence length ({len(field_sequences[0])})"
            )

        # Convert field sequences to transactions for pattern mining
        pattern_counts: dict[tuple[int, ...], int] = defaultdict(int)
        pattern_locations: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)

        for msg_idx, field_vals in enumerate(field_sequences):
            # Extract subsequences from field values
            for length in range(
                self.min_pattern_length, min(len(field_vals) + 1, self.max_pattern_length + 1)
            ):
                for offset in range(len(field_vals) - length + 1):
                    subseq = tuple(field_vals[offset : offset + length])
                    pattern_counts[subseq] += 1
                    pattern_locations[subseq].append((msg_idx, offset))

        # Calculate support
        total_patterns = sum(pattern_counts.values())

        # Filter and create patterns
        patterns = []
        for seq, count in pattern_counts.items():
            support = count / total_patterns if total_patterns > 0 else 0.0

            if support >= self.min_support:
                # Add field names to metadata
                metadata = {}
                if field_names and len(seq) == 1:
                    # Single field pattern
                    first_loc = pattern_locations[seq][0] if pattern_locations[seq] else (0, 0)
                    offset = first_loc[1]
                    if offset < len(field_names):
                        metadata["field_name"] = field_names[offset]

                # NECESSARY COPY: Protects pattern's locations from external mutations.
                patterns.append(
                    Pattern(
                        sequence=seq,
                        support=support,
                        locations=pattern_locations[seq].copy(),
                        metadata=metadata,
                    )
                )

        # Sort by support
        patterns.sort(key=lambda p: p.support, reverse=True)

        self.patterns = patterns
        return patterns

    def find_associations(self, patterns: list[Pattern]) -> list[AssociationRule]:
        """Find association rules between patterns (A -> B).

        Discovers rules where pattern A appearing implies pattern B follows,
        with confidence and lift metrics.

        Args:
            patterns: List of patterns to analyze for associations.

        Returns:
            List of AssociationRule objects sorted by confidence.

        Example:
            >>> rules = miner.find_associations(patterns)
            >>> for rule in rules:
            ...     print(f"{rule.antecedent} -> {rule.consequent}: {rule.confidence:.2f}")
        """
        if not patterns:
            return []

        rules = []

        # Generate rules (check if B follows A immediately)
        for i, pattern_a in enumerate(patterns):
            for j, pattern_b in enumerate(patterns):
                if i >= j:
                    continue

                seq_a = pattern_a.sequence
                seq_b = pattern_b.sequence

                # Find co-occurrences (B appears immediately after A)
                co_occur_count = 0
                total_occurrences = len(pattern_a.locations)

                for msg_idx_a, offset_a in pattern_a.locations:
                    expected_offset_b = offset_a + len(seq_a)

                    # Check if B appears right after A
                    for msg_idx_b, offset_b in pattern_b.locations:
                        if msg_idx_a == msg_idx_b and offset_b == expected_offset_b:
                            co_occur_count += 1
                            break

                if co_occur_count > 0:
                    # Calculate metrics
                    # Confidence = P(B|A) = count(A->B) / count(A)
                    confidence = (
                        co_occur_count / total_occurrences if total_occurrences > 0 else 0.0
                    )
                    # Support is the frequency of the rule in the dataset
                    support_ab = pattern_a.support * confidence
                    # Lift = confidence / P(B)
                    lift = confidence / pattern_b.support if pattern_b.support > 0 else 0.0

                    if confidence >= self.min_confidence:
                        rules.append(
                            AssociationRule(
                                antecedent=seq_a,
                                consequent=seq_b,
                                support=support_ab,
                                confidence=confidence,
                                lift=lift,
                            )
                        )

        # Sort by confidence
        rules.sort(key=lambda r: r.confidence, reverse=True)

        self.rules = rules
        return rules

    def mine_temporal_patterns(
        self, events: list[tuple[float, str]], max_gap: float = 1.0
    ) -> list[TemporalPattern]:
        """Mine temporal event sequences.

        Finds sequences like [EventA, EventB, EventC] that occur with regular
        timing intervals.

        Args:
            events: List of (timestamp, event_type) tuples.
            max_gap: Maximum time gap between consecutive events in sequence.

        Returns:
            List of TemporalPattern objects.

        Raises:
            ValueError: If events list is empty or max_gap is negative.

        Example:
            >>> events = [(0.0, "A"), (0.5, "B"), (1.0, "A"), (1.5, "B")]
            >>> patterns = miner.mine_temporal_patterns(events, max_gap=0.6)
        """
        if not events:
            raise ValueError("Events list cannot be empty")
        if max_gap < 0:
            raise ValueError(f"max_gap must be non-negative, got {max_gap}")

        # Sort events by timestamp
        events_sorted = sorted(events, key=lambda e: e[0])

        # Find sequences
        sequences: dict[tuple[str, ...], list[list[float]]] = defaultdict(list)

        for i in range(len(events_sorted)):
            current_seq = [events_sorted[i][1]]
            current_times = [events_sorted[i][0]]

            for j in range(i + 1, min(i + self.max_pattern_length, len(events_sorted))):
                time_gap = events_sorted[j][0] - current_times[-1]

                if time_gap <= max_gap:
                    current_seq.append(events_sorted[j][1])
                    current_times.append(events_sorted[j][0])

                    if len(current_seq) >= self.min_pattern_length:
                        seq_tuple = tuple(current_seq)
                        # NECESSARY COPY: current_times is mutated in loop.
                        # Without .copy(), all sequences would reference final state.
                        # Removing would cause: data corruption, identical timestamps.
                        sequences[seq_tuple].append(current_times.copy())
                else:
                    break

        # Convert to TemporalPattern objects
        temporal_patterns = []
        for seq, time_lists in sequences.items():
            # Calculate average interval
            all_intervals = []
            for times in time_lists:
                intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
                all_intervals.extend(intervals)

            if all_intervals:
                avg_interval = float(np.mean(all_intervals))
                variance = float(np.var(all_intervals))

                # Use relative timestamps (first event at 0.0)
                relative_times = [t - time_lists[0][0] for t in time_lists[0]]

                temporal_patterns.append(
                    TemporalPattern(
                        events=list(seq),
                        timestamps=relative_times,
                        avg_interval=avg_interval,
                        variance=variance,
                    )
                )

        return temporal_patterns

    def find_correlations(self, field_data: dict[str, list[float]]) -> dict[tuple[str, str], float]:
        """Calculate Pearson correlation coefficients between fields.

        Computes pairwise correlations to find relationships between numeric
        field values across messages.

        Args:
            field_data: Dictionary of field_name -> list of values.

        Returns:
            Dictionary of (field_a, field_b) -> correlation_coefficient.

        Raises:
            ValueError: If field_data is empty or fields have different lengths.

        Example:
            >>> field_data = {"field_a": [1, 2, 3], "field_b": [2, 4, 6]}
            >>> correlations = miner.find_correlations(field_data)
            >>> print(correlations[("field_a", "field_b")])
            1.0
        """
        if not field_data:
            raise ValueError("Field data cannot be empty")

        # Validate all fields have same length
        field_names = list(field_data.keys())
        lengths = {name: len(values) for name, values in field_data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"All fields must have same length, got {lengths}")

        correlations: dict[tuple[str, str], float] = {}

        # Compute pairwise correlations
        for i, field_a in enumerate(field_names):
            for j, field_b in enumerate(field_names):
                if i >= j:
                    continue

                values_a = np.array(field_data[field_a], dtype=np.float64)
                values_b = np.array(field_data[field_b], dtype=np.float64)

                # Calculate Pearson correlation
                if len(values_a) < 2:
                    corr = 0.0
                else:
                    # Compute correlation coefficient
                    mean_a = np.mean(values_a)
                    mean_b = np.mean(values_b)

                    centered_a = values_a - mean_a
                    centered_b = values_b - mean_b

                    num = np.sum(centered_a * centered_b)
                    denom = np.sqrt(np.sum(centered_a**2) * np.sum(centered_b**2))

                    corr = float(num / denom) if denom > 0 else 0.0

                correlations[(field_a, field_b)] = corr
                # Add symmetric entry
                correlations[(field_b, field_a)] = corr

        return correlations

    def _extract_subsequences(
        self, sequence: bytes, min_length: int, max_length: int
    ) -> set[tuple[int, ...]]:
        """Extract all subsequences of given length range.

        Args:
            sequence: Byte sequence to extract from.
            min_length: Minimum subsequence length.
            max_length: Maximum subsequence length.

        Returns:
            Set of unique subsequences as tuples.
        """
        subsequences: set[tuple[int, ...]] = set()

        for length in range(min_length, min(len(sequence) + 1, max_length + 1)):
            for offset in range(len(sequence) - length + 1):
                subseq = tuple(sequence[offset : offset + length])
                subsequences.add(subseq)

        return subsequences

    def _fp_growth(
        self, transactions: list[frozenset[int]], min_support: float
    ) -> list[tuple[frozenset[int], int]]:
        """FP-Growth algorithm for frequent itemset mining.

        Simplified implementation for pattern mining. For production use,
        consider using libraries like mlxtend.

        Args:
            transactions: List of transaction itemsets.
            min_support: Minimum support threshold (0.0-1.0).

        Returns:
            List of (itemset, count) tuples.
        """
        # Count item frequencies
        item_counts: dict[int, int] = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # Filter by minimum support
        min_count = int(min_support * len(transactions))
        frequent_items = {item for item, count in item_counts.items() if count >= min_count}

        # Generate frequent itemsets (simplified - single items only for now)
        result = []
        for item in frequent_items:
            count = item_counts[item]
            result.append((frozenset([item]), count))

        return result

    def _apriori(
        self, transactions: list[frozenset[int]], min_support: float
    ) -> list[tuple[frozenset[int], int]]:
        """Apriori algorithm for frequent itemset mining.

        Simplified implementation. For production use, consider libraries
        like mlxtend.

        Args:
            transactions: List of transaction itemsets.
            min_support: Minimum support threshold (0.0-1.0).

        Returns:
            List of (itemset, count) tuples.
        """
        # Use same implementation as FP-Growth for now
        # Full Apriori would generate candidate itemsets iteratively
        return self._fp_growth(transactions, min_support)

    def visualize_patterns(
        self, output_path: Path, format: Literal["graph", "tree", "heatmap"] = "graph"
    ) -> None:
        """Visualize discovered patterns.

        Creates visualization of patterns and their relationships. Requires
        matplotlib and networkx for graph visualization.

        Args:
            output_path: Path to save visualization file.
            format: Visualization format ("graph", "tree", or "heatmap").

        Raises:
            ValueError: If no patterns have been mined yet.
            ImportError: If required visualization libraries not available.
        """
        if not self.patterns:
            raise ValueError("No patterns to visualize. Run mine_byte_patterns() first.")

        if format == "heatmap":
            self._visualize_heatmap(output_path)
        elif format in ("graph", "tree"):
            self._visualize_graph(output_path)

    def _visualize_heatmap(self, output_path: Path) -> None:
        """Create heatmap of pattern support values."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib required for visualization") from e

        fig, ax = plt.subplots(figsize=(10, 6))

        pattern_labels = [" ".join(f"{b:02X}" for b in p.sequence) for p in self.patterns[:20]]
        support_values = [p.support for p in self.patterns[:20]]

        ax.barh(pattern_labels, support_values)
        ax.set_xlabel("Support")
        ax.set_title("Top 20 Pattern Support Values")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _visualize_graph(self, output_path: Path) -> None:
        """Create graph visualization of pattern relationships."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError("matplotlib and networkx required for graph visualization") from e

        # Build graph
        G = self._build_pattern_graph(nx)

        # Draw graph
        fig, ax = plt.subplots(figsize=(12, 8))

        pos = nx.spring_layout(G, k=2, iterations=50)
        labels = nx.get_node_attributes(G, "label")

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", ax=ax)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, ax=ax)

        ax.set_title("Pattern Association Graph")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _build_pattern_graph(self, nx: Any) -> Any:
        """Build networkx graph from patterns and rules."""
        G = nx.DiGraph()

        # Add patterns as nodes
        for i, pattern in enumerate(self.patterns[:20]):
            label = " ".join(f"{b:02X}" for b in pattern.sequence)
            G.add_node(i, label=label, support=pattern.support)

        # Add edges for association rules
        for rule in self.rules:
            ant_idx = next(
                (i for i, p in enumerate(self.patterns) if p.sequence == rule.antecedent),
                None,
            )
            con_idx = next(
                (i for i, p in enumerate(self.patterns) if p.sequence == rule.consequent),
                None,
            )

            if ant_idx is not None and con_idx is not None:
                G.add_edge(ant_idx, con_idx, confidence=rule.confidence, lift=rule.lift)

        return G

    def export_rules(
        self, output_path: Path, format: Literal["json", "csv", "yaml"] = "json"
    ) -> None:
        """Export association rules to file.

        Args:
            output_path: Path to save rules file.
            format: Output format ("json", "csv", or "yaml").

        Raises:
            ValueError: If no rules have been discovered yet.
            ImportError: If required library not available for format.
        """
        if not self.rules:
            raise ValueError("No rules to export. Run find_associations() first.")

        if format == "json":
            self._export_json(output_path)
        elif format == "csv":
            self._export_csv(output_path)
        elif format == "yaml":
            self._export_yaml(output_path)

    def _export_json(self, output_path: Path) -> None:
        """Export rules as JSON.

        Args:
            output_path: Path to save JSON file.
        """
        import json

        rules_data = [
            {
                "antecedent": [int(b) for b in rule.antecedent],
                "consequent": [int(b) for b in rule.consequent],
                "support": rule.support,
                "confidence": rule.confidence,
                "lift": rule.lift,
            }
            for rule in self.rules
        ]

        with output_path.open("w") as f:
            json.dump(rules_data, f, indent=2)

    def _export_csv(self, output_path: Path) -> None:
        """Export rules as CSV.

        Args:
            output_path: Path to save CSV file.
        """
        with output_path.open("w") as f:
            f.write("antecedent,consequent,support,confidence,lift\n")
            for rule in self.rules:
                ant_str = " ".join(f"{b:02X}" for b in rule.antecedent)
                con_str = " ".join(f"{b:02X}" for b in rule.consequent)
                f.write(f'"{ant_str}","{con_str}",{rule.support},{rule.confidence},{rule.lift}\n')

    def _export_yaml(self, output_path: Path) -> None:
        """Export rules as YAML.

        Args:
            output_path: Path to save YAML file.
        """
        import yaml

        rules_data = [
            {
                "antecedent": [int(b) for b in rule.antecedent],
                "consequent": [int(b) for b in rule.consequent],
                "support": rule.support,
                "confidence": rule.confidence,
                "lift": rule.lift,
            }
            for rule in self.rules
        ]

        with output_path.open("w") as f:
            yaml.dump(rules_data, f, default_flow_style=False)


__all__ = [
    "AssociationRule",
    "Pattern",
    "PatternMiner",
    "TemporalPattern",
]
