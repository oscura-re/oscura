# Active Learning for DFA Inference

This module implements **Angluin's L\* algorithm** for learning deterministic finite automata (DFAs) through active learning.

## Overview

L*is an active learning algorithm that learns the minimal DFA for a regular language by querying an oracle. Unlike passive learning (RPNI), which learns from a fixed dataset, L* can ask questions to refine its hypothesis.

### Key Features

- **Minimal DFA**: Guaranteed to produce the minimal DFA (fewest states)
- **Polynomial complexity**: O(|Q|²|Σ|) membership queries, O(|Q|) equivalence queries
- **No negative examples required**: Only needs access to oracle
- **Interactive learning**: Can learn from live systems, simulators, or models

## Algorithm

The L\* algorithm maintains an **observation table** with:

- **S**: Set of prefixes (representative strings for states)
- **E**: Set of suffixes (experiments to distinguish states)
- **T**: Table mapping (prefix + suffix) to acceptance

The algorithm iterates:

1. Fill observation table using membership queries
2. Make table **closed** (every extension has equivalent in S)
3. Make table **consistent** (equivalent rows have equivalent extensions)
4. Construct hypothesis DFA from table
5. Ask equivalence query
6. If counterexample: refine table and repeat
7. Return final hypothesis

## Usage

### Basic Example

```python
from oscura.inference.active_learning import LStarLearner, SimulatorTeacher

# Create oracle from captured traces
traces = [
    ["CONNECT", "ACK"],
    ["CONNECT", "ACK", "DATA", "ACK"],
    ["CONNECT", "ACK", "DISCONNECT"]
]

teacher = SimulatorTeacher(traces)

# Learn DFA
learner = LStarLearner(teacher, verbose=True)
dfa = learner.learn()

print(f"Learned DFA with {len(dfa.states)} states")
print(f"Membership queries: {learner.membership_queries}")
print(f"Equivalence queries: {learner.equivalence_queries}")
```

### Custom Oracle

```python
from oscura.inference.active_learning import Oracle, LStarLearner
from oscura.inference.state_machine import FiniteAutomaton

class MyOracle(Oracle):
    def membership_query(self, word: tuple[str, ...]) -> bool:
        # Implement your logic here
        return len(word) % 2 == 0

    def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[str, ...] | None:
        # Test hypothesis and return counterexample if wrong
        test_words = [(), ("a",), ("a", "a"), ("a", "a", "a")]
        for word in test_words:
            if hypothesis.accepts(list(word)) != self.membership_query(word):
                return word
        return None

    def get_alphabet(self) -> set[str]:
        return {"a", "b"}

oracle = MyOracle()
learner = LStarLearner(oracle)
dfa = learner.learn()
```

## Oracles (Teachers)

### SimulatorTeacher

Learn from captured protocol traces. Treats all prefixes of traces as valid.

```python
from oscura.inference.active_learning import SimulatorTeacher

traces = [["a", "b", "c"], ["a", "b", "d"]]
teacher = SimulatorTeacher(traces)
```

**Membership queries**: Returns True if word is a prefix of any trace
**Equivalence queries**: Tests hypothesis against all traces and their prefixes

### Future Oracles

- **InteractiveTeacher**: Query live device/system
- **ModelTeacher**: Query formal specification or model

## Observation Table

The observation table is the core data structure:

```python
from oscura.inference.active_learning import ObservationTable

table = ObservationTable(alphabet={"a", "b"})

# Check properties
table.is_closed()      # All extensions covered
table.is_consistent()  # Equivalent rows have equivalent extensions

# Convert to DFA
dfa = table.to_dfa()
```

## Performance

For small protocols (5-10 states):

- Membership queries: 50-200
- Equivalence queries: 3-10
- Time: < 1 second

The algorithm is polynomial in the number of states and alphabet size.

## Academic Reference

```
Angluin, D. (1987). Learning regular sets from queries and counterexamples.
Information and Computation, 75(2), 87-106.
```

## Comparison with RPNI

| Feature           | L\* (Active)            | RPNI (Passive)             |
| ----------------- | ----------------------- | -------------------------- | --- | --- | --- | --- | ----------- |
| Learning type     | Active (queries oracle) | Passive (fixed dataset)    |
| Minimal DFA       | Yes                     | No (may have extra states) |
| Negative examples | Not required            | Optional                   |
| Live learning     | Yes                     | No                         |
| Query complexity  | O(                      | Q                          | ²   | Σ   | )   | N/A | ## See Also |

- `oscura.inference.state_machine`: RPNI passive learning
- `examples/lstar_demo.py`: Complete usage examples
- `tests/unit/inference/test_lstar.py`: Test cases including academic examples
