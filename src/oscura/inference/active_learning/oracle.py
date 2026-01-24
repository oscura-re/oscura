"""Oracle interface for L* active learning.

This module defines the oracle interface used in the L* algorithm.
The oracle answers membership and equivalence queries about the target DFA.

References:
    Angluin, D. (1987). Learning regular sets from queries and counterexamples.
    Information and Computation, 75(2), 87-106.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.inference.state_machine import FiniteAutomaton


class Oracle(ABC):
    """Oracle for answering membership and equivalence queries.

    An oracle provides access to the target language through two types of queries:

    1. Membership Query: "Is word w accepted by the target DFA?"
       - Returns True if w is in the target language, False otherwise
       - Must be consistent (same query always returns same answer)

    2. Equivalence Query: "Is hypothesis H equivalent to the target DFA?"
       - Returns None if H is equivalent to target
       - Returns a counterexample word if H differs from target
       - Counterexample is a word accepted by exactly one of H or target

    The oracle also provides the alphabet of the target language.
    """

    @abstractmethod
    def membership_query(self, word: tuple[str, ...]) -> bool:
        """Answer membership query: Is word accepted by target DFA?

        Args:
            word: Sequence of symbols to test

        Returns:
            True if word is accepted by target language, False otherwise
        """
        ...

    @abstractmethod
    def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[str, ...] | None:
        """Answer equivalence query: Is hypothesis equivalent to target?

        Args:
            hypothesis: Proposed DFA to test for equivalence

        Returns:
            None if hypothesis is equivalent to target, otherwise a
            counterexample word that is accepted by exactly one of
            hypothesis or target
        """
        ...

    @abstractmethod
    def get_alphabet(self) -> set[str]:
        """Get the alphabet of the target language.

        Returns:
            Set of symbols in the target alphabet
        """
        ...

    def get_query_counts(self) -> tuple[int, int]:
        """Get counts of membership and equivalence queries.

        This is useful for analyzing the efficiency of the learning algorithm.

        Returns:
            Tuple of (membership_queries, equivalence_queries)
        """
        return (0, 0)  # Default implementation, can be overridden
