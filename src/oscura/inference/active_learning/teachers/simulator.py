"""Simulator teacher for L* active learning.

This module implements an oracle that uses captured protocol traces as
ground truth. It's useful for learning DFAs from historical data without
requiring a live system.

The simulator teacher treats traces as examples of valid protocol sequences.
"""

from oscura.inference.active_learning.oracle import Oracle
from oscura.inference.state_machine import FiniteAutomaton


class SimulatorTeacher(Oracle):
    """Oracle that replays captured protocol traces.

    This teacher uses captured message traces as ground truth for learning.
    It answers queries based on whether sequences appear in the captured traces.

    Membership queries check if a word is a valid prefix of any trace.
    Equivalence queries test the hypothesis against all captured traces.

    This is useful for:
    - Learning from historical protocol captures
    - Offline protocol analysis
    - Testing L* implementation with known data

    Attributes:
        traces: List of message sequences (each trace is list of symbols)
        alphabet: Set of all symbols appearing in traces
        membership_query_count: Number of membership queries answered
        equivalence_query_count: Number of equivalence queries answered
    """

    def __init__(self, traces: list[list[str]]):
        """Initialize simulator teacher from captured traces.

        Args:
            traces: List of message sequences, where each sequence is a
                    list of symbols (strings). All traces are treated as
                    positive examples (accepted sequences).

        Raises:
            ValueError: If no traces provided
        """
        if not traces:
            raise ValueError("Need at least one trace")

        self.traces = traces
        self.alphabet = self._extract_alphabet()
        self.membership_query_count = 0
        self.equivalence_query_count = 0

        # Build set of all valid prefixes for efficient membership queries
        self._valid_prefixes: set[tuple[str, ...]] = set()
        self._build_prefix_set()

    def _extract_alphabet(self) -> set[str]:
        """Extract alphabet from all traces.

        Returns:
            Set of all unique symbols in traces
        """
        alphabet = set()
        for trace in self.traces:
            alphabet.update(trace)
        return alphabet

    def _build_prefix_set(self) -> None:
        """Build set of all valid prefixes from traces.

        Pre-computes all prefixes for efficient membership query answering.
        """
        for trace in self.traces:
            # Add empty prefix
            self._valid_prefixes.add(())

            # Add all prefixes of this trace
            for i in range(1, len(trace) + 1):
                prefix = tuple(trace[:i])
                self._valid_prefixes.add(prefix)

    def membership_query(self, word: tuple[str, ...]) -> bool:
        """Check if word is a valid prefix of any trace.

        A word is accepted if it appears as a prefix (including the full
        sequence) of at least one captured trace.

        Args:
            word: Sequence of symbols to test

        Returns:
            True if word is a prefix of any trace, False otherwise
        """
        self.membership_query_count += 1
        return word in self._valid_prefixes

    def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[str, ...] | None:
        """Check hypothesis against all traces.

        Tests whether the hypothesis DFA correctly accepts all captured traces
        and their prefixes. Returns a counterexample if the hypothesis disagrees
        with the trace data.

        Args:
            hypothesis: Proposed DFA to test

        Returns:
            None if hypothesis matches all trace data, otherwise a
            counterexample word that hypothesis classifies incorrectly
        """
        self.equivalence_query_count += 1

        # Check all valid prefixes
        for prefix in self._valid_prefixes:
            hypothesis_accepts = hypothesis.accepts(list(prefix))
            target_accepts = True  # All prefixes should be accepted

            if hypothesis_accepts != target_accepts:
                return prefix

        # Also check some strings not in traces (they should be rejected)
        # Generate some random invalid sequences
        invalid_sequences = self._generate_invalid_sequences(hypothesis)

        for seq in invalid_sequences:
            hypothesis_accepts = hypothesis.accepts(list(seq))
            target_accepts = seq in self._valid_prefixes

            if hypothesis_accepts != target_accepts:
                return seq

        return None

    def _generate_invalid_sequences(
        self, hypothesis: FiniteAutomaton, max_checks: int = 100
    ) -> list[tuple[str, ...]]:
        """Generate sequences not in traces to test hypothesis.

        Creates test sequences by:
        1. Taking prefixes from traces and extending with different symbols
        2. Creating short random sequences
        3. Concatenating partial traces

        Args:
            hypothesis: Current hypothesis (used to guide generation)
            max_checks: Maximum number of invalid sequences to generate

        Returns:
            List of sequences not in valid prefixes
        """
        invalid = []

        # Strategy 1: Extend valid prefixes with unexpected symbols
        for prefix in list(self._valid_prefixes)[:20]:  # Limit to avoid too many
            for symbol in self.alphabet:
                candidate = prefix + (symbol,)
                if candidate not in self._valid_prefixes:
                    invalid.append(candidate)
                    if len(invalid) >= max_checks:
                        return invalid

        # Strategy 2: Create sequences from alphabet permutations
        if len(self.alphabet) > 0:
            alphabet_list = list(self.alphabet)
            for i in range(min(5, len(alphabet_list))):
                for j in range(min(5, len(alphabet_list))):
                    candidate = (alphabet_list[i], alphabet_list[j])
                    if candidate not in self._valid_prefixes:
                        invalid.append(candidate)
                        if len(invalid) >= max_checks:
                            return invalid

        return invalid

    def get_alphabet(self) -> set[str]:
        """Get the alphabet of the target language.

        Returns:
            Set of symbols extracted from traces
        """
        return self.alphabet.copy()

    def get_query_counts(self) -> tuple[int, int]:
        """Get counts of membership and equivalence queries.

        Returns:
            Tuple of (membership_queries, equivalence_queries)
        """
        return (self.membership_query_count, self.equivalence_query_count)
