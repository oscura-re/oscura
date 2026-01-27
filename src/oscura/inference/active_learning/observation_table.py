"""L* observation table for DFA learning.

This module implements the observation table data structure used in
Angluin's L* algorithm for active learning of deterministic finite automata.

References:
    Angluin, D. (1987). Learning regular sets from queries and counterexamples.
    Information and Computation, 75(2), 87-106.
"""

from dataclasses import dataclass, field

from oscura.inference.state_machine import FiniteAutomaton, State, Transition


@dataclass
class ObservationTable:
    """L* observation table for DFA learning.

    The observation table is the core data structure in the L* algorithm.
    It maintains:
    - S: Set of prefixes (rows of upper part)
    - SA: Set of S + Σ (one-step extensions, lower part)
    - E: Set of suffixes (columns/experiments)
    - T: Table entries mapping (prefix + suffix) -> bool (accept/reject)

    A table is:
    - Closed: Every row in SA has an equivalent row in S
    - Consistent: Identical rows in S have identical one-step extensions

    Attributes:
        S: Set of prefixes (represented as tuples of symbols)
        E: Set of suffixes (represented as tuples of symbols)
        T: Mapping from string (prefix+suffix) to acceptance
        alphabet: Alphabet of symbols
    """

    S: set[tuple[str, ...]] = field(default_factory=lambda: {()})
    E: set[tuple[str, ...]] = field(default_factory=lambda: {()})
    T: dict[tuple[str, ...], bool] = field(default_factory=dict)
    alphabet: set[str] = field(default_factory=set)

    def row(self, s: tuple[str, ...]) -> tuple[bool, ...]:
        """Get row for prefix s.

        Returns the row in the observation table for prefix s, which
        consists of the membership query results for s·e for each suffix e in E.

        Args:
            s: Prefix (tuple of symbols)

        Returns:
            Tuple of boolean values representing the row
        """
        return tuple(self.T.get(s + e, False) for e in sorted(self.E))

    def is_closed(self) -> bool:
        """Check if table is closed.

        A table is closed if for every row in SA (one-step extensions),
        there exists an equivalent row in S.

        Returns:
            True if table is closed
        """
        return self.find_closing_counterexample() is None

    def is_consistent(self) -> bool:
        """Check if table is consistent.

        A table is consistent if for any two prefixes s1, s2 in S with
        identical rows, all one-step extensions s1·a and s2·a also have
        identical rows for each symbol a in the alphabet.

        Returns:
            True if table is consistent
        """
        return self.find_consistency_counterexample() is None

    def find_closing_counterexample(self) -> tuple[str, ...] | None:
        """Find row in SA without equivalent in S.

        Searches for a string s·a where s ∈ S and a ∈ Σ such that
        row(s·a) is not equal to row(s') for any s' ∈ S.

        Returns:
            String in SA without equivalent in S, or None if closed
        """
        # Get all rows in S
        s_rows = {s: self.row(s) for s in self.S}

        # Check all one-step extensions
        for s in self.S:
            for a in self.alphabet:
                sa = s + (a,)
                sa_row = self.row(sa)

                # Check if this row exists in S
                if sa_row not in s_rows.values():
                    return sa

        return None

    def find_consistency_counterexample(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...], str] | None:
        """Find inconsistency in table.

        Searches for two prefixes s1, s2 in S with identical rows and
        a symbol a such that row(s1·a) ≠ row(s2·a).

        Returns:
            Tuple (s1, s2, a) representing inconsistency, or None if consistent
        """
        # Build mapping from rows to prefixes
        row_to_prefixes: dict[tuple[bool, ...], list[tuple[str, ...]]] = {}
        for s in self.S:
            r = self.row(s)
            if r not in row_to_prefixes:
                row_to_prefixes[r] = []
            row_to_prefixes[r].append(s)

        # Check each equivalence class
        for prefixes in row_to_prefixes.values():
            if len(prefixes) < 2:
                continue

            # Check all pairs in this equivalence class
            for i in range(len(prefixes)):
                for j in range(i + 1, len(prefixes)):
                    s1 = prefixes[i]
                    s2 = prefixes[j]

                    # Check each symbol
                    for a in self.alphabet:
                        s1a = s1 + (a,)
                        s2a = s2 + (a,)

                        if self.row(s1a) != self.row(s2a):
                            return (s1, s2, a)

        return None

    def to_dfa(self) -> FiniteAutomaton:
        """Construct hypothesis DFA from closed, consistent table.

        Creates a DFA where:
        - Each distinct row in S corresponds to a state
        - Transitions are determined by row(s·a)
        - Initial state corresponds to row(ε)
        - Accepting states are those where T[s·ε] = True

        Returns:
            FiniteAutomaton constructed from table

        Raises:
            ValueError: If table is not closed or not consistent
        """
        if not self.is_closed():
            raise ValueError("Table must be closed to construct DFA")
        if not self.is_consistent():
            raise ValueError("Table must be consistent to construct DFA")

        # Map rows to state IDs
        row_to_state: dict[tuple[bool, ...], int] = {}
        state_to_row: dict[int, tuple[bool, ...]] = {}
        state_representatives: dict[int, tuple[str, ...]] = {}

        state_id = 0
        for s in sorted(self.S):
            r = self.row(s)
            if r not in row_to_state:
                row_to_state[r] = state_id
                state_to_row[state_id] = r
                state_representatives[state_id] = s
                state_id += 1

        # Create states
        states = []
        accepting_states = set()

        for sid in range(state_id):
            rep = state_representatives[sid]
            # State is accepting if T[rep] = True (i.e., rep is accepted)
            is_accepting = self.T.get(rep, False)
            is_initial = rep == ()

            state = State(
                id=sid,
                name=f"q{sid}",
                is_initial=is_initial,
                is_accepting=is_accepting,
            )
            states.append(state)

            if is_accepting:
                accepting_states.add(sid)

        # Create transitions
        transitions = []
        seen_transitions = set()

        for sid in range(state_id):
            rep = state_representatives[sid]

            for a in sorted(self.alphabet):
                # Find target state
                rep_a = rep + (a,)
                target_row = self.row(rep_a)
                target_state = row_to_state[target_row]

                # Add transition
                key = (sid, target_state, a)
                if key not in seen_transitions:
                    seen_transitions.add(key)
                    transitions.append(Transition(source=sid, target=target_state, symbol=a))

        # Find initial state
        initial_row = self.row(())
        initial_state = row_to_state[initial_row]

        # Type cast alphabet from set[str] to set[str | int] for FiniteAutomaton
        alphabet_union: set[str | int] = set(self.alphabet)

        return FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet=alphabet_union,
            initial_state=initial_state,
            accepting_states=accepting_states,
        )
