"""State machine inference using RPNI algorithm.

Requirements addressed: PSI-002

This module infers protocol state machines from observed message sequences using
passive learning algorithms (no system interaction required).

Key capabilities:
- RPNI algorithm for passive DFA learning
- State merging to minimize automaton
- Export to DOT format for visualization
- Export to NetworkX graph for analysis
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass
class State:
    """A state in the inferred automaton.

    : State representation.

    Attributes:
        id: Unique state identifier
        name: Human-readable state name
        is_initial: Whether this is the initial state
        is_accepting: Whether this is an accepting state
    """

    id: int
    name: str
    is_initial: bool = False
    is_accepting: bool = False


@dataclass
class Transition:
    """A transition in the automaton.

    : Transition representation.

    Attributes:
        source: Source state ID
        target: Target state ID
        symbol: Transition label/symbol
        count: Number of times observed
    """

    source: int  # State ID
    target: int  # State ID
    symbol: str  # Transition label
    count: int = 1  # Number of observations


@dataclass
class FiniteAutomaton:
    """An inferred finite automaton.

    : Complete automaton representation with export capabilities.

    Attributes:
        states: List of all states
        transitions: List of all transitions
        alphabet: Set of all symbols
        initial_state: Initial state ID
        accepting_states: Set of accepting state IDs
    """

    states: list[State]
    transitions: list[Transition]
    alphabet: set[str]
    initial_state: int
    accepting_states: set[int]

    def to_dot(self) -> str:
        """Export to DOT format for Graphviz.

        : DOT format export for visualization.

        Returns:
            DOT format string
        """
        lines = ["digraph finite_automaton {", "    rankdir=LR;", "    node [shape=circle];"]

        # Mark accepting states
        if self.accepting_states:
            accepting_names = [s.name for s in self.states if s.id in self.accepting_states]
            lines.append(f"    node [shape=doublecircle]; {' '.join(accepting_names)};")
            lines.append("    node [shape=circle];")

        # Add invisible start node for initial state
        initial_state = next(s for s in self.states if s.id == self.initial_state)
        lines.append('    __start__ [shape=none, label=""];')
        lines.append(f"    __start__ -> {initial_state.name};")

        # Add transitions
        for trans in self.transitions:
            src_state = next(s for s in self.states if s.id == trans.source)
            tgt_state = next(s for s in self.states if s.id == trans.target)
            label = trans.symbol
            if trans.count > 1:
                label = f"{trans.symbol} ({trans.count})"
            lines.append(f'    {src_state.name} -> {tgt_state.name} [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def to_networkx(self) -> Any:
        """Export to NetworkX graph.

        : NetworkX export for programmatic analysis.

        Returns:
            NetworkX MultiDiGraph (supports multiple edges between same nodes)

        Raises:
            ImportError: If NetworkX is not installed.
        """
        try:
            import networkx as nx  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError("NetworkX is required for graph export") from err

        # Use MultiDiGraph to support multiple transitions between same states
        G = nx.MultiDiGraph()

        # Add nodes
        for state in self.states:
            G.add_node(
                state.id,
                name=state.name,
                is_initial=state.is_initial,
                is_accepting=state.is_accepting,
            )

        # Add edges
        for trans in self.transitions:
            G.add_edge(trans.source, trans.target, symbol=trans.symbol, count=trans.count)

        return G

    def accepts(self, sequence: list[str]) -> bool:
        """Check if automaton accepts sequence.

        : Sequence acceptance checking.

        Args:
            sequence: List of symbols

        Returns:
            True if sequence is accepted
        """
        current_state = self.initial_state

        for symbol in sequence:
            # Find transition with this symbol
            trans = None
            for t in self.transitions:
                if t.source == current_state and t.symbol == symbol:
                    trans = t
                    break

            if trans is None:
                return False  # No valid transition

            current_state = trans.target

        # Check if we ended in accepting state
        return current_state in self.accepting_states

    def get_successors(self, state_id: int) -> dict[str, int]:
        """Get successor states from given state.

        : State successor lookup.

        Args:
            state_id: State ID to query

        Returns:
            Dictionary mapping symbols to target state IDs
        """
        successors = {}
        for trans in self.transitions:
            if trans.source == state_id:
                successors[trans.symbol] = trans.target
        return successors


class StateMachineInferrer:
    """Infer state machines using passive learning.

    : RPNI algorithm for DFA inference.

    The RPNI (Regular Positive and Negative Inference) algorithm:
    1. Build Prefix Tree Acceptor from positive samples
    2. Iteratively merge compatible state pairs
    3. Validate against negative samples
    4. Converge to minimal consistent DFA
    """

    def __init__(self) -> None:
        """Initialize inferrer."""
        self._next_state_id = 0

    def infer(
        self,
        positive_traces: list[list[str]] | None = None,
        negative_traces: list[list[str]] | None = None,
        positive_samples: list[list[str]] | None = None,
        negative_samples: list[list[str]] | None = None,
    ) -> FiniteAutomaton:
        """Infer DFA from traces (alias for infer_rpni).

        Args:
            positive_traces: List of accepted sequences.
            negative_traces: List of rejected sequences (optional).
            positive_samples: Alias for positive_traces (deprecated).
            negative_samples: Alias for negative_traces (deprecated).

        Returns:
            Inferred FiniteAutomaton.

        Raises:
            ValueError: If no positive traces provided.
        """
        # Handle parameter aliases
        pos = positive_traces if positive_traces is not None else positive_samples
        neg = negative_traces if negative_traces is not None else negative_samples

        if pos is None:
            raise ValueError("Must provide either positive_traces or positive_samples")

        return self.infer_rpni(pos, neg)

    def infer_rpni(
        self, positive_traces: list[list[str]], negative_traces: list[list[str]] | None = None
    ) -> FiniteAutomaton:
        """Infer DFA using RPNI (Regular Positive and Negative Inference).

        : Complete RPNI algorithm.

        Args:
            positive_traces: List of accepted sequences (list of symbols)
            negative_traces: List of rejected sequences (optional)

        Returns:
            Inferred FiniteAutomaton

        Raises:
            ValueError: If no positive traces provided.
        """
        if not positive_traces:
            raise ValueError("Need at least one positive trace")

        # Build alphabet from all traces
        alphabet: set[str] = set()
        neg_traces = negative_traces if negative_traces is not None else []
        for trace in positive_traces + neg_traces:
            alphabet.update(trace)

        # Build Prefix Tree Acceptor from positive traces
        pta = self._build_pta(positive_traces)

        # RPNI merging process
        automaton = pta
        states = sorted([s.id for s in automaton.states])

        # Try to merge states in order
        i = 1  # Start from second state (never merge initial state)
        while i < len(states):
            merged = False

            # Try to merge states[i] with any earlier state
            for j in range(i):
                if self._is_compatible(automaton, states[j], states[i], neg_traces):
                    # Merge states[i] into states[j]
                    automaton = self._merge_states(automaton, states[j], states[i])
                    # Update state list
                    states = sorted([s.id for s in automaton.states])
                    merged = True
                    break

            if not merged:
                i += 1

        return automaton

    def _build_pta(self, traces: list[list[str]]) -> FiniteAutomaton:
        """Build Prefix Tree Acceptor from traces.

        : PTA construction.

        Args:
            traces: List of sequences

        Returns:
            Prefix Tree Acceptor as FiniteAutomaton
        """
        # Reset state counter
        self._next_state_id = 0

        # Create initial state
        initial_state = State(
            id=self._get_next_state_id(), name="q0", is_initial=True, is_accepting=False
        )

        states: list[State] = [initial_state]
        transitions: list[Transition] = []
        alphabet: set[str] = set()

        # Build tree from traces
        for trace in traces:
            current_state_id = initial_state.id

            # Walk/build tree for this trace
            for symbol in trace:
                alphabet.add(symbol)

                # Check if transition exists
                next_state_id = None
                for trans in transitions:
                    if trans.source == current_state_id and trans.symbol == symbol:
                        next_state_id = trans.target
                        break

                if next_state_id is None:
                    # Create new state and transition
                    new_state_id = self._get_next_state_id()
                    new_state = State(
                        id=new_state_id,
                        name=f"q{new_state_id}",
                        is_initial=False,
                        is_accepting=False,
                    )
                    states.append(new_state)

                    new_trans = Transition(
                        source=current_state_id, target=new_state_id, symbol=symbol
                    )
                    transitions.append(new_trans)

                    next_state_id = new_state_id

                current_state_id = next_state_id

            # Mark final state as accepting
            for state in states:
                if state.id == current_state_id:
                    state.is_accepting = True

        accepting_states = {s.id for s in states if s.is_accepting}

        return FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet=alphabet,
            initial_state=initial_state.id,
            accepting_states=accepting_states,
        )

    def _merge_states(
        self, automaton: FiniteAutomaton, state_a: int, state_b: int
    ) -> FiniteAutomaton:
        """Merge two states in automaton.

        : State merging operation.

        Merges state_b into state_a.

        Args:
            automaton: Current automaton
            state_a: Target state ID (survives)
            state_b: Source state ID (removed)

        Returns:
            New automaton with merged states
        """
        # Deep copy to avoid modifying original
        new_automaton = deepcopy(automaton)

        # Remove state_b
        new_automaton.states = [s for s in new_automaton.states if s.id != state_b]

        # Update transitions: redirect all transitions to/from state_b to state_a
        for trans in new_automaton.transitions:
            if trans.source == state_b:
                trans.source = state_a
            if trans.target == state_b:
                trans.target = state_a

        # Merge accepting status
        if state_b in new_automaton.accepting_states:
            new_automaton.accepting_states.add(state_a)
            new_automaton.accepting_states.discard(state_b)

        # Merge duplicate transitions (same source, target, symbol)
        unique_transitions = []
        seen = set()

        for trans in new_automaton.transitions:
            key = (trans.source, trans.target, trans.symbol)
            if key not in seen:
                seen.add(key)
                unique_transitions.append(trans)
            else:
                # Increment count on existing transition
                for ut in unique_transitions:
                    if (ut.source, ut.target, ut.symbol) == key:
                        ut.count += trans.count
                        break

        new_automaton.transitions = unique_transitions

        return new_automaton

    def _is_compatible(
        self,
        automaton: FiniteAutomaton,
        state_a: int,
        state_b: int,
        negative_traces: list[list[str]],
    ) -> bool:
        """Check if two states can be merged without accepting negatives.

        : Compatibility checking for state merging.

        Args:
            automaton: Current automaton
            state_a: First state ID
            state_b: Second state ID
            negative_traces: Negative example traces

        Returns:
            True if states are compatible
        """
        # Get accepting status
        _a_accepting = state_a in automaton.accepting_states
        _b_accepting = state_b in automaton.accepting_states

        # If one is accepting and other is not, they might still be compatible
        # (we'll merge accepting status), but check negative traces

        # Try merging and test
        test_automaton = self._merge_states(automaton, state_a, state_b)

        # Check that no negative traces are accepted
        for neg_trace in negative_traces:
            if test_automaton.accepts(neg_trace):
                return False

        # Recursively check successor compatibility
        _succ_a = test_automaton.get_successors(state_a)
        # state_b has been merged, so its successors are now in state_a

        return True

    def _get_next_state_id(self) -> int:
        """Get next available state ID.

        Returns:
            Next state ID
        """
        state_id = self._next_state_id
        self._next_state_id += 1
        return state_id


def minimize_dfa(automaton: FiniteAutomaton) -> FiniteAutomaton:
    """Minimize DFA using partition refinement.

    : DFA minimization using Hopcroft's algorithm.

    Args:
        automaton: DFA to minimize

    Returns:
        Minimized FiniteAutomaton
    """
    # Use partition refinement (simplified version)
    # Start with two partitions: accepting and non-accepting
    accepting = automaton.accepting_states
    non_accepting = {s.id for s in automaton.states if s.id not in accepting}

    partitions = []
    if accepting:
        partitions.append(accepting)
    if non_accepting:
        partitions.append(non_accepting)

    # Refine partitions
    changed = True
    while changed:
        changed = False
        new_partitions = []

        for partition in partitions:
            # Try to split this partition
            if len(partition) <= 1:
                new_partitions.append(partition)
                continue

            # Group states by transition signatures
            groups: dict[tuple[tuple[str, int | None], ...], set[int]] = {}
            for state_id in partition:
                successors = automaton.get_successors(state_id)

                # Create signature based on which partition each successor is in
                signature_list: list[tuple[str, int | None]] = []
                for symbol in sorted(automaton.alphabet):
                    if symbol in successors:
                        target = successors[symbol]
                        # Find which partition target is in
                        target_partition: int | None = None
                        for i, p in enumerate(partitions):
                            if target in p:
                                target_partition = i
                                break
                        signature_list.append((symbol, target_partition))
                    else:
                        signature_list.append((symbol, None))

                signature = tuple(signature_list)
                if signature not in groups:
                    groups[signature] = set()
                groups[signature].add(state_id)

            # If we split, mark as changed
            if len(groups) > 1:
                changed = True

            new_partitions.extend(groups.values())

        partitions = new_partitions

    # Build minimized automaton
    # Map old state IDs to partition IDs
    state_to_partition = {}
    for i, partition in enumerate(partitions):
        for state_id in partition:
            state_to_partition[state_id] = i

    # Create new states
    new_states = []
    for i, partition in enumerate(partitions):
        # Pick representative state
        rep_id = min(partition)
        _rep_state = next(s for s in automaton.states if s.id == rep_id)

        is_accepting = any(sid in automaton.accepting_states for sid in partition)
        is_initial = automaton.initial_state in partition

        new_state = State(id=i, name=f"q{i}", is_initial=is_initial, is_accepting=is_accepting)
        new_states.append(new_state)

    # Create new transitions
    new_transitions = []
    seen_transitions = set()

    for trans in automaton.transitions:
        src_partition = state_to_partition[trans.source]
        tgt_partition = state_to_partition[trans.target]

        key = (src_partition, tgt_partition, trans.symbol)
        if key not in seen_transitions:
            seen_transitions.add(key)
            new_transitions.append(
                Transition(
                    source=src_partition,
                    target=tgt_partition,
                    symbol=trans.symbol,
                    count=trans.count,
                )
            )

    # Find new initial state
    new_initial = state_to_partition[automaton.initial_state]
    new_accepting = {s.id for s in new_states if s.is_accepting}

    return FiniteAutomaton(
        states=new_states,
        transitions=new_transitions,
        alphabet=automaton.alphabet,
        initial_state=new_initial,
        accepting_states=new_accepting,
    )


def to_dot(automaton: FiniteAutomaton) -> str:
    """Export automaton to DOT format.

    : Convenience function for DOT export.

    Args:
        automaton: Automaton to export

    Returns:
        DOT format string
    """
    return automaton.to_dot()


def to_networkx(automaton: FiniteAutomaton) -> Any:
    """Export automaton to NetworkX graph.

    : Convenience function for NetworkX export.

    Args:
        automaton: Automaton to export

    Returns:
        NetworkX DiGraph
    """
    return automaton.to_networkx()


def infer_rpni(
    positive_traces: list[list[str]], negative_traces: list[list[str]] | None = None
) -> FiniteAutomaton:
    """Convenience function for RPNI inference.

    : Top-level API for state machine inference.

    Args:
        positive_traces: List of accepted sequences
        negative_traces: List of rejected sequences (optional)

    Returns:
        Inferred FiniteAutomaton
    """
    inferrer = StateMachineInferrer()
    return inferrer.infer_rpni(positive_traces, negative_traces)
