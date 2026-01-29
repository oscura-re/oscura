"""Enhanced state machine inference using RPNI and EDSM algorithms.

Requirements addressed: PSI-002, Feature 29

This module infers protocol state machines from observed message sequences using
passive learning algorithms (no system interaction required).

Key capabilities:
- RPNI algorithm for passive DFA learning
- EDSM (Evidence-Driven State Merging) algorithm for improved inference
- State machines with guards and probabilistic transitions
- Export to DOT, PlantUML, and SMV formats
- Validation against captured sequences
- State merging to minimize automaton
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class State:
    """A state in the inferred state machine.

    Represents a state with metadata for protocol state machines.

    Attributes:
        id: Unique state identifier
        name: Human-readable state name
        is_initial: Whether this is the initial state
        is_accepting: Whether this is an accepting/final state (alias: is_final)
        is_error: Whether this is an error state
        metadata: Additional state information (dict)

    Example:
        >>> state = State(id=0, name="IDLE", is_initial=True)
        >>> state.is_initial
        True
        >>> state.metadata["description"] = "Waiting for connection"
    """

    id: int
    name: str
    is_initial: bool = False
    is_accepting: bool = False
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        """Alias for is_accepting (more intuitive for state machines).

        Returns:
            True if this is a final state
        """
        return self.is_accepting

    @is_final.setter
    def is_final(self, value: bool) -> None:
        """Set final state status.

        Args:
            value: Whether this is a final state
        """
        self.is_accepting = value


@dataclass
class Transition:
    """A transition in the state machine.

    Represents a state transition with optional guards and probabilities.

    Attributes:
        source: Source state ID (alias: from_state)
        target: Target state ID (alias: to_state)
        symbol: Transition label/event (alias: event)
        guard: Optional condition for transition (e.g., "x > 10")
        probability: Probability of transition (1.0 = deterministic)
        count: Number of times observed

    Example:
        >>> trans = Transition(source=0, target=1, symbol="CONNECT")
        >>> trans.probability
        1.0
        >>> trans2 = Transition(source=0, target=2, symbol="TIMEOUT",
        ...                     guard="timer > 5", probability=0.1)
    """

    source: int  # State ID
    target: int  # State ID
    symbol: str | int  # Transition label/event
    guard: str | None = None  # Condition for transition
    probability: float = 1.0  # Probability (1.0 = deterministic)
    count: int = 1  # Number of observations

    @property
    def from_state(self) -> int:
        """Alias for source state ID.

        Returns:
            Source state ID
        """
        return self.source

    @property
    def to_state(self) -> int:
        """Alias for target state ID.

        Returns:
            Target state ID
        """
        return self.target

    @property
    def event(self) -> str | int:
        """Alias for symbol/event.

        Returns:
            Transition event/symbol
        """
        return self.symbol


@dataclass
class FiniteAutomaton:
    """An inferred finite automaton / state machine.

    Complete automaton representation with export capabilities.

    Attributes:
        states: List of all states
        transitions: List of all transitions
        alphabet: Set of all symbols/events
        initial_state: Initial state ID
        accepting_states: Set of accepting/final state IDs (alias: final_states)

    Example:
        >>> states = [State(id=0, name="q0", is_initial=True),
        ...           State(id=1, name="q1", is_accepting=True)]
        >>> transitions = [Transition(source=0, target=1, symbol="A")]
        >>> fa = FiniteAutomaton(states=states, transitions=transitions,
        ...                      alphabet={"A"}, initial_state=0,
        ...                      accepting_states={1})
        >>> fa.accepts(["A"])
        True
    """

    states: list[State]
    transitions: list[Transition]
    alphabet: set[str | int]
    initial_state: int
    accepting_states: set[int]

    @property
    def final_states(self) -> set[int]:
        """Alias for accepting_states (more intuitive for state machines).

        Returns:
            Set of final state IDs
        """
        return self.accepting_states

    @final_states.setter
    def final_states(self, value: set[int]) -> None:
        """Set final states.

        Args:
            value: Set of final state IDs
        """
        self.accepting_states = value

    def to_dot(self) -> str:
        """Export to DOT format for Graphviz.

        Generates GraphViz DOT format with:
        - Double circles for final states
        - Red circles for error states
        - Transition labels with events and probabilities
        - Guard conditions shown in labels

        Returns:
            DOT format string

        Example:
            >>> fa = FiniteAutomaton(...)
            >>> dot = fa.to_dot()
            >>> Path("automaton.dot").write_text(dot)
        """
        lines = ["digraph StateMachine {", "    rankdir=LR;", "    node [shape=circle];"]

        # Mark accepting/final states
        if self.accepting_states:
            accepting_names = [s.name for s in self.states if s.id in self.accepting_states]
            lines.append(f"    node [shape=doublecircle]; {' '.join(accepting_names)};")
            lines.append("    node [shape=circle];")

        # Add invisible start node for initial state
        initial_state = next(s for s in self.states if s.id == self.initial_state)
        lines.append("    __start__ [shape=point];")
        lines.append(f"    __start__ -> {initial_state.name};")

        # Add states with colors
        for state in self.states:
            if state.is_error:
                lines.append(f"    {state.name} [color=red];")

        # Add transitions
        for trans in self.transitions:
            src_state = next(s for s in self.states if s.id == trans.source)
            tgt_state = next(s for s in self.states if s.id == trans.target)

            # Build label with event, guard, probability, count
            label_parts = [str(trans.symbol)]

            if trans.guard:
                label_parts.append(f"[{trans.guard}]")

            if trans.probability < 1.0:
                label_parts.append(f"(p={trans.probability:.2f})")

            if trans.count > 1:
                label_parts.append(f"(cnt={trans.count})")

            label = " ".join(label_parts)
            lines.append(f'    {src_state.name} -> {tgt_state.name} [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def to_networkx(self) -> Any:
        """Export to NetworkX graph.

        Returns NetworkX MultiDiGraph for programmatic analysis.

        Returns:
            NetworkX MultiDiGraph (supports multiple edges between same nodes)

        Raises:
            ImportError: If NetworkX is not installed.

        Example:
            >>> fa = FiniteAutomaton(...)
            >>> graph = fa.to_networkx()
            >>> graph.number_of_nodes()
            3
        """
        try:
            import networkx as nx
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
                is_error=state.is_error,
                metadata=state.metadata,
            )

        # Add edges
        for trans in self.transitions:
            G.add_edge(
                trans.source,
                trans.target,
                symbol=trans.symbol,
                guard=trans.guard,
                probability=trans.probability,
                count=trans.count,
            )

        return G

    def accepts(self, sequence: list[str | int]) -> bool:
        """Check if automaton accepts sequence.

        Simulates execution on the sequence, following deterministic transitions.

        Args:
            sequence: List of symbols/events

        Returns:
            True if sequence is accepted (ends in accepting state)

        Example:
            >>> fa.accepts(["CONNECT", "DATA", "DISCONNECT"])
            True
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

    def get_successors(self, state_id: int) -> dict[str | int, int]:
        """Get successor states from given state.

        Finds all outgoing transitions from a state.

        Args:
            state_id: State ID to query

        Returns:
            Dictionary mapping symbols to target state IDs

        Example:
            >>> fa.get_successors(0)
            {'A': 1, 'B': 2}
        """
        successors: dict[str | int, int] = {}
        for trans in self.transitions:
            if trans.source == state_id:
                successors[trans.symbol] = trans.target
        return successors


# Alias for backward compatibility
StateMachine = FiniteAutomaton


class StateMachineInferrer:
    """Infer state machines using passive learning algorithms.

    Implements RPNI and EDSM algorithms for DFA inference from traces.

    The RPNI (Regular Positive and Negative Inference) algorithm:
    1. Build Prefix Tree Acceptor from positive samples
    2. Iteratively merge compatible state pairs
    3. Validate against negative samples
    4. Converge to minimal consistent DFA

    The EDSM (Evidence-Driven State Merging) algorithm:
    1. Build Prefix Tree Acceptor
    2. Score state pairs by evidence (shared suffix behavior)
    3. Merge highest-scoring compatible pairs first
    4. More accurate than RPNI for noisy data

    Example:
        >>> inferrer = StateMachineInferrer(algorithm="edsm")
        >>> positive = [["CONNECT", "DATA", "CLOSE"], ["CONNECT", "CLOSE"]]
        >>> negative = [["DATA", "CONNECT"], ["CLOSE", "DATA"]]
        >>> sm = inferrer.extract(positive, negative)
        >>> sm.accepts(["CONNECT", "DATA", "CLOSE"])
        True
    """

    ALGORITHMS: ClassVar[list[str]] = ["rpni", "edsm"]  # Supported algorithms

    def __init__(self, algorithm: str = "rpni") -> None:
        """Initialize inferrer with algorithm choice.

        Args:
            algorithm: Algorithm to use ("rpni" or "edsm")

        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Algorithm '{algorithm}' not supported. Choose from: {self.ALGORITHMS}"
            )
        self.algorithm = algorithm
        self._next_state_id = 0

    def extract(
        self,
        positive_sequences: list[list[str | int]],
        negative_sequences: list[list[str | int]] | None = None,
    ) -> StateMachine:
        """Extract state machine from sequences.

        Main entry point for state machine extraction.

        Args:
            positive_sequences: Sequences that should be accepted
            negative_sequences: Sequences that should be rejected (optional)

        Returns:
            Inferred StateMachine

        Raises:
            ValueError: If no positive sequences provided

        Example:
            >>> inferrer = StateMachineInferrer()
            >>> positive = [["A", "B"], ["A", "C"]]
            >>> sm = inferrer.extract(positive)
        """
        if self.algorithm == "rpni":
            return self._rpni(positive_sequences, negative_sequences or [])
        elif self.algorithm == "edsm":
            return self._edsm(positive_sequences, negative_sequences or [])
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def infer(
        self,
        positive_traces: list[list[str | int]] | None = None,
        negative_traces: list[list[str | int]] | None = None,
        positive_samples: list[list[str | int]] | None = None,
        negative_samples: list[list[str | int]] | None = None,
    ) -> FiniteAutomaton:
        """Infer DFA from traces (backward compatibility API).

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

        return self.extract(pos, neg)

    def infer_rpni(
        self,
        positive_traces: list[list[str | int]],
        negative_traces: list[list[str | int]] | None = None,
    ) -> FiniteAutomaton:
        """Infer DFA using RPNI (backward compatibility API).

        Args:
            positive_traces: List of accepted sequences (list of symbols)
            negative_traces: List of rejected sequences (optional)

        Returns:
            Inferred FiniteAutomaton

        Raises:
            ValueError: If no positive traces provided.
        """
        return self._rpni(positive_traces, negative_traces or [])

    def _rpni(
        self,
        positive: list[list[str | int]],
        negative: list[list[str | int]],
    ) -> StateMachine:
        """RPNI algorithm: builds prefix tree, then merges states.

        Regular Positive and Negative Inference algorithm.

        Steps:
        1. Build prefix tree acceptor (PTA) from positive examples
        2. Order states (lexicographic by state ID)
        3. For each pair of states (red, blue):
           - Try merging blue into red
           - Check if merge still rejects all negative examples
           - If yes, keep merge; if no, color blue as red and continue
        4. Return resulting automaton

        Args:
            positive: Positive example sequences
            negative: Negative example sequences

        Returns:
            Inferred StateMachine

        Raises:
            ValueError: If no positive sequences provided
        """
        if not positive:
            raise ValueError("Need at least one positive trace")

        # Build alphabet from all traces
        alphabet: set[str | int] = set()
        for trace in positive + negative:
            alphabet.update(trace)

        # Build Prefix Tree Acceptor from positive traces
        pta = self._build_prefix_tree(positive)

        # RPNI merging process
        automaton = pta
        states = sorted([s.id for s in automaton.states])

        # Try to merge states in order
        i = 1  # Start from second state (never merge initial state)
        while i < len(states):
            merged = False

            # Try to merge states[i] with any earlier state
            for j in range(i):
                if self._is_merge_compatible(automaton, states[j], states[i], negative):
                    # Merge states[i] into states[j]
                    automaton = self._merge_states(automaton, states[j], states[i])
                    # Update state list
                    states = sorted([s.id for s in automaton.states])
                    merged = True
                    break

            if not merged:
                i += 1

        return automaton

    def _edsm(
        self,
        positive: list[list[str | int]],
        negative: list[list[str | int]],
    ) -> StateMachine:
        """EDSM algorithm: evidence-driven state merging.

        Evidence-Driven State Merging algorithm (more accurate than RPNI).

        Steps:
        1. Build prefix tree acceptor (PTA)
        2. Compute evidence scores for all state pairs
        3. Sort pairs by score (higher = more evidence they should merge)
        4. Try merging in score order, keeping compatible merges
        5. Return resulting automaton

        Evidence score = number of shared suffix behaviors (transitions)

        Args:
            positive: Positive example sequences
            negative: Negative example sequences

        Returns:
            Inferred StateMachine

        Raises:
            ValueError: If no positive sequences provided
        """
        if not positive:
            raise ValueError("Need at least one positive trace")

        # Build Prefix Tree Acceptor
        pta = self._build_prefix_tree(positive)
        automaton = pta

        # Iteratively merge states based on evidence
        while True:
            # Compute evidence scores for all state pairs
            state_ids = [s.id for s in automaton.states]
            best_pair: tuple[int, int] | None = None
            best_score = 0

            for i, state_a in enumerate(state_ids):
                for state_b in state_ids[i + 1 :]:
                    # Don't merge initial state
                    if state_a == automaton.initial_state or state_b == automaton.initial_state:
                        continue

                    # Compute evidence score
                    score = self._compute_evidence(automaton, state_a, state_b)

                    if score > best_score:
                        # Check compatibility before considering
                        if self._is_merge_compatible(automaton, state_a, state_b, negative):
                            best_score = score
                            best_pair = (state_a, state_b)

            # If no merge found, done
            if best_pair is None:
                break

            # Merge best pair
            automaton = self._merge_states(automaton, best_pair[0], best_pair[1])

        return automaton

    def _compute_evidence(self, automaton: StateMachine, state_a: int, state_b: int) -> int:
        """Compute evidence score for merging two states.

        Evidence = number of symbols for which both states have transitions
        to states that could also be merged (shared suffix behavior).

        Args:
            automaton: Current automaton
            state_a: First state ID
            state_b: Second state ID

        Returns:
            Evidence score (higher = more evidence for merge)
        """
        succ_a = automaton.get_successors(state_a)
        succ_b = automaton.get_successors(state_b)

        # Count shared symbols (evidence)
        shared_symbols = set(succ_a.keys()) & set(succ_b.keys())

        # Simple evidence: number of shared transition symbols
        return len(shared_symbols)

    def _build_prefix_tree(self, sequences: list[list[str | int]]) -> StateMachine:
        """Build Prefix Tree Automaton (PTA) from sequences.

        Each unique prefix gets a state. Transitions labeled with symbols.
        All sequences accepted (final states at end of each sequence).

        Args:
            sequences: List of symbol sequences

        Returns:
            Prefix Tree Automaton as StateMachine
        """
        # Reset state counter
        self._next_state_id = 0

        # Create initial state
        initial_state = State(
            id=self._get_next_state_id(), name="s0", is_initial=True, is_accepting=False
        )

        states: list[State] = [initial_state]
        transitions: list[Transition] = []
        alphabet: set[str | int] = set()

        # Build tree from sequences
        for seq in sequences:
            current_state_id = initial_state.id

            # Walk/build tree for this sequence
            for symbol in seq:
                alphabet.add(symbol)

                # Check if transition exists
                next_state_id = None
                for trans in transitions:
                    if trans.source == current_state_id and trans.symbol == symbol:
                        next_state_id = trans.target
                        trans.count += 1  # Increment observation count
                        break

                if next_state_id is None:
                    # Create new state and transition
                    new_state_id = self._get_next_state_id()
                    new_state = State(
                        id=new_state_id,
                        name=f"s{new_state_id}",
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

        return StateMachine(
            states=states,
            transitions=transitions,
            alphabet=alphabet,
            initial_state=initial_state.id,
            accepting_states=accepting_states,
        )

    def _merge_states(self, automaton: StateMachine, state_a: int, state_b: int) -> StateMachine:
        """Merge two states, updating all transitions.

        Merges state_b into state_a (state_b is removed).

        Args:
            automaton: Current state machine
            state_a: Target state ID (survives)
            state_b: Source state ID (removed)

        Returns:
            New state machine with merged states
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

    def _is_merge_compatible(
        self,
        automaton: StateMachine,
        state_a: int,
        state_b: int,
        negative: list[list[str | int]],
    ) -> bool:
        """Check if merging would accept negative sequences.

        Tests whether merging two states would cause the automaton
        to accept any negative example sequences.

        Args:
            automaton: Current state machine
            state_a: First state ID
            state_b: Second state ID
            negative: Negative example sequences

        Returns:
            True if states can be merged without accepting negatives
        """
        # Try merging and test
        test_automaton = self._merge_states(automaton, state_a, state_b)

        # Check that no negative traces are accepted
        return all(not test_automaton.accepts(neg_trace) for neg_trace in negative)

    def _get_next_state_id(self) -> int:
        """Get next available state ID.

        Returns:
            Next state ID
        """
        state_id = self._next_state_id
        self._next_state_id += 1
        return state_id


class StateMachineExtractor:
    """Enhanced state machine extraction with multiple algorithms.

    Provides high-level API for state machine extraction with
    export and validation capabilities.

    Example:
        >>> extractor = StateMachineExtractor(algorithm="edsm")
        >>> positive = [["CONNECT", "DATA", "CLOSE"]]
        >>> sm = extractor.extract(positive)
        >>> extractor.export_graphviz(sm, Path("machine.dot"))
        >>> extractor.export_plantuml(sm, Path("machine.puml"))
        >>> accepted, rejected = extractor.validate_sequences(sm, positive)
        >>> accepted == len(positive)
        True
    """

    ALGORITHMS: ClassVar[list[str]] = StateMachineInferrer.ALGORITHMS

    def __init__(self, algorithm: str = "rpni") -> None:
        """Initialize extractor with algorithm choice.

        Args:
            algorithm: Algorithm to use ("rpni" or "edsm")
        """
        self.algorithm = algorithm
        self._inferrer = StateMachineInferrer(algorithm=algorithm)

    def extract(
        self,
        positive_sequences: list[list[str | int]],
        negative_sequences: list[list[str | int]] | None = None,
    ) -> StateMachine:
        """Extract state machine from sequences.

        Args:
            positive_sequences: Sequences that should be accepted
            negative_sequences: Sequences that should be rejected (optional)

        Returns:
            Inferred StateMachine
        """
        return self._inferrer.extract(positive_sequences, negative_sequences)

    def export_graphviz(self, sm: StateMachine, output_path: Path) -> None:
        """Export as GraphViz DOT format.

        Args:
            sm: State machine to export
            output_path: Path to write DOT file

        Example:
            >>> extractor.export_graphviz(sm, Path("machine.dot"))
        """
        dot_content = sm.to_dot()
        output_path.write_text(dot_content)

    def export_plantuml(self, sm: StateMachine, output_path: Path) -> None:
        """Export as PlantUML state diagram.

        Generates PlantUML format for state diagrams.

        Args:
            sm: State machine to export
            output_path: Path to write PlantUML file

        Example:
            >>> extractor.export_plantuml(sm, Path("machine.puml"))
        """
        lines = ["@startuml"]

        # Find initial state
        initial_state = next(s for s in sm.states if s.id == sm.initial_state)
        lines.append(f"[*] --> {initial_state.name}")

        # Add states
        for state in sm.states:
            if state.is_error:
                lines.append(f"{state.name} : <<error>>")
            if state.metadata:
                for key, value in state.metadata.items():
                    lines.append(f"{state.name} : {key}={value}")

        # Add transitions
        for trans in sm.transitions:
            src_state = next(s for s in sm.states if s.id == trans.source)
            tgt_state = next(s for s in sm.states if s.id == trans.target)

            # Build label
            label = str(trans.symbol)
            if trans.guard:
                label += f" [{trans.guard}]"
            if trans.probability < 1.0:
                label += f" (p={trans.probability:.2f})"

            lines.append(f"{src_state.name} --> {tgt_state.name} : {label}")

        # Add final states
        for state_id in sm.accepting_states:
            state = next(s for s in sm.states if s.id == state_id)
            lines.append(f"{state.name} --> [*]")

        lines.append("@enduml")
        output_path.write_text("\n".join(lines))

    def export_smv(self, sm: StateMachine, output_path: Path) -> None:
        """Export as SMV (Symbolic Model Verifier) format.

        Generates NuSMV/SMV format for formal verification.

        Args:
            sm: State machine to export
            output_path: Path to write SMV file

        Example:
            >>> extractor.export_smv(sm, Path("machine.smv"))
        """
        lines = ["MODULE main", "VAR"]

        # State variable
        state_names = [s.name for s in sm.states]
        lines.append(f"  state : {{{', '.join(state_names)}}};")

        # Event/input variable
        alphabet_str = ", ".join(str(s) for s in sorted(sm.alphabet, key=str))
        lines.append(f"  event : {{{alphabet_str}}};")

        # Initial state
        lines.append("")
        lines.append("ASSIGN")
        initial_state_obj = next(s for s in sm.states if s.id == sm.initial_state)
        lines.append(f"  init(state) := {initial_state_obj.name};")

        # Transition relation
        lines.append("  next(state) := case")

        for trans in sm.transitions:
            src = next(s for s in sm.states if s.id == trans.source)
            tgt = next(s for s in sm.states if s.id == trans.target)

            # Build condition
            condition = f"state = {src.name} & event = {trans.symbol}"
            if trans.guard:
                condition += f" & ({trans.guard})"

            lines.append(f"    {condition} : {tgt.name};")

        lines.append("    TRUE : state;")
        lines.append("  esac;")

        # Specifications (accepting states)
        if sm.accepting_states:
            lines.append("")
            lines.append("-- Final/Accepting states")
            for state_id in sm.accepting_states:
                state = next(s for s in sm.states if s.id == state_id)
                lines.append(f"DEFINE final_{state.name} := state = {state.name};")

        output_path.write_text("\n".join(lines))

    def validate_sequences(
        self, sm: StateMachine, sequences: list[list[str | int]]
    ) -> tuple[int, int]:
        """Validate sequences against state machine.

        Tests sequences to count how many are accepted vs rejected.

        Args:
            sm: State machine
            sequences: List of sequences to validate

        Returns:
            Tuple of (accepted_count, rejected_count)

        Example:
            >>> accepted, rejected = extractor.validate_sequences(sm, test_seqs)
            >>> print(f"Accepted: {accepted}/{len(test_seqs)}")
        """
        accepted = 0
        rejected = 0

        for seq in sequences:
            if sm.accepts(seq):
                accepted += 1
            else:
                rejected += 1

        return (accepted, rejected)

    def minimize_automaton(self, sm: StateMachine) -> StateMachine:
        """Minimize DFA using Hopcroft's algorithm.

        Args:
            sm: State machine to minimize

        Returns:
            Minimized state machine

        Example:
            >>> minimized = extractor.minimize_automaton(sm)
            >>> len(minimized.states) <= len(sm.states)
            True
        """
        return minimize_dfa(sm)


def _initialize_partitions(automaton: FiniteAutomaton) -> list[set[int]]:
    """Initialize partitions with accepting and non-accepting states.

    Args:
        automaton: DFA to partition

    Returns:
        Initial partition list
    """
    accepting = automaton.accepting_states
    non_accepting = {s.id for s in automaton.states if s.id not in accepting}

    partitions = []
    if accepting:
        partitions.append(accepting)
    if non_accepting:
        partitions.append(non_accepting)
    return partitions


def _find_target_partition(target: int, partitions: list[set[int]]) -> int | None:
    """Find which partition a state belongs to.

    Args:
        target: State ID to find
        partitions: Current partitions

    Returns:
        Partition index or None if not found
    """
    for i, p in enumerate(partitions):
        if target in p:
            return i
    return None


def _create_transition_signature(
    state_id: int,
    automaton: FiniteAutomaton,
    partitions: list[set[int]],
) -> tuple[tuple[str | int, int | None], ...]:
    """Create transition signature for partition refinement.

    Args:
        state_id: State to create signature for
        automaton: Automaton containing the state
        partitions: Current partitions

    Returns:
        Transition signature tuple
    """
    successors = automaton.get_successors(state_id)
    signature_list: list[tuple[str | int, int | None]] = []

    for symbol in sorted(automaton.alphabet, key=str):
        if symbol in successors:
            target = successors[symbol]
            target_partition = _find_target_partition(target, partitions)
            signature_list.append((symbol, target_partition))
        else:
            signature_list.append((symbol, None))

    return tuple(signature_list)


def _split_partition(
    partition: set[int],
    automaton: FiniteAutomaton,
    partitions: list[set[int]],
) -> list[set[int]]:
    """Split partition by grouping states with identical signatures.

    Args:
        partition: Partition to split
        automaton: Automaton containing states
        partitions: Current partitions for signature creation

    Returns:
        List of sub-partitions
    """
    if len(partition) <= 1:
        return [partition]

    groups: dict[tuple[tuple[str | int, int | None], ...], set[int]] = {}
    for state_id in partition:
        signature = _create_transition_signature(state_id, automaton, partitions)
        if signature not in groups:
            groups[signature] = set()
        groups[signature].add(state_id)

    return list(groups.values())


def _refine_partitions(
    automaton: FiniteAutomaton,
    partitions: list[set[int]],
) -> list[set[int]]:
    """Refine partitions until no more splits occur.

    Args:
        automaton: DFA to minimize
        partitions: Initial partitions

    Returns:
        Refined partitions
    """
    changed = True
    while changed:
        changed = False
        new_partitions = []

        for partition in partitions:
            splits = _split_partition(partition, automaton, partitions)
            if len(splits) > 1:
                changed = True
            new_partitions.extend(splits)

        partitions = new_partitions

    return partitions


def _build_state_mapping(partitions: list[set[int]]) -> dict[int, int]:
    """Map original state IDs to partition IDs.

    Args:
        partitions: Final partitions

    Returns:
        Mapping from old state ID to new state ID
    """
    state_to_partition = {}
    for i, partition in enumerate(partitions):
        for state_id in partition:
            state_to_partition[state_id] = i
    return state_to_partition


def _create_minimized_states(
    partitions: list[set[int]],
    automaton: FiniteAutomaton,
) -> list[State]:
    """Create new states for minimized automaton.

    Args:
        partitions: Final partitions
        automaton: Original automaton

    Returns:
        List of new states
    """
    new_states = []
    for i, partition in enumerate(partitions):
        is_accepting = any(sid in automaton.accepting_states for sid in partition)
        is_initial = automaton.initial_state in partition
        is_error = any(
            next(s for s in automaton.states if s.id == sid).is_error for sid in partition
        )

        new_state = State(
            id=i,
            name=f"q{i}",
            is_initial=is_initial,
            is_accepting=is_accepting,
            is_error=is_error,
        )
        new_states.append(new_state)

    return new_states


def _create_minimized_transitions(
    automaton: FiniteAutomaton,
    state_to_partition: dict[int, int],
) -> list[Transition]:
    """Create transitions for minimized automaton.

    Args:
        automaton: Original automaton
        state_to_partition: Mapping from old to new state IDs

    Returns:
        List of new transitions
    """
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
                    guard=trans.guard,
                    probability=trans.probability,
                    count=trans.count,
                )
            )

    return new_transitions


def minimize_dfa(automaton: FiniteAutomaton) -> FiniteAutomaton:
    """Minimize DFA using partition refinement.

    Uses Hopcroft's algorithm for DFA minimization.

    Args:
        automaton: DFA to minimize

    Returns:
        Minimized FiniteAutomaton

    Example:
        >>> minimized = minimize_dfa(original_dfa)
    """
    # Initialize partitions
    partitions = _initialize_partitions(automaton)

    # Refine partitions until stable
    partitions = _refine_partitions(automaton, partitions)

    # Build minimized automaton
    state_to_partition = _build_state_mapping(partitions)
    new_states = _create_minimized_states(partitions, automaton)
    new_transitions = _create_minimized_transitions(automaton, state_to_partition)

    # Find new initial state and accepting states
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

    Convenience function for DOT export.

    Args:
        automaton: Automaton to export

    Returns:
        DOT format string

    Example:
        >>> dot = to_dot(automaton)
        >>> Path("machine.dot").write_text(dot)
    """
    return automaton.to_dot()


def to_networkx(automaton: FiniteAutomaton) -> Any:
    """Export automaton to NetworkX graph.

    Convenience function for NetworkX export.

    Args:
        automaton: Automaton to export

    Returns:
        NetworkX DiGraph

    Example:
        >>> graph = to_networkx(automaton)
    """
    return automaton.to_networkx()


def infer_rpni(
    positive_traces: list[list[str | int]], negative_traces: list[list[str | int]] | None = None
) -> FiniteAutomaton:
    """Convenience function for RPNI inference.

    Top-level API for state machine inference.

    Args:
        positive_traces: List of accepted sequences
        negative_traces: List of rejected sequences (optional)

    Returns:
        Inferred FiniteAutomaton

    Example:
        >>> dfa = infer_rpni([["A", "B"], ["A", "C"]])
    """
    inferrer = StateMachineInferrer()
    return inferrer.infer_rpni(positive_traces, negative_traces)
