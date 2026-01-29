"""Active learning for protocol inference.

This module implements Angluin's L* algorithm for learning deterministic
finite automata (DFAs) through active learning. Unlike passive learning
(RPNI), active learning interacts with an oracle to learn the minimal DFA.

Key advantages of L* over passive learning:
- Learns minimal DFA (fewest states)
- Polynomial query complexity
- No negative examples required
- Interactive learning from live systems

References:
    Angluin, D. (1987). Learning regular sets from queries and counterexamples.
    Information and Computation, 75(2), 87-106.

Example:
    >>> from oscura.inference.active_learning import LStarLearner, SimulatorTeacher
    >>> # Create oracle from captured traces
    >>> traces = [['A', 'B', 'C'], ['A', 'B', 'B', 'C']]
    >>> oracle = SimulatorTeacher(traces)
    >>> # Learn DFA
    >>> learner = LStarLearner(oracle)
    >>> dfa = learner.learn()
    >>> print(f"Learned DFA with {len(dfa.states)} states")
"""

from oscura.inference.active_learning.lstar import LStarLearner
from oscura.inference.active_learning.observation_table import ObservationTable
from oscura.inference.active_learning.oracle import Oracle
from oscura.inference.active_learning.teachers import SimulatorTeacher

__all__ = [
    "LStarLearner",
    "ObservationTable",
    "Oracle",
    "SimulatorTeacher",
]
