"""Angluin's L* algorithm for DFA learning.

This module implements the L* algorithm for learning deterministic finite
automata through active learning. The algorithm interacts with an oracle
to learn the minimal DFA for the target language.

References:
    Angluin, D. (1987). Learning regular sets from queries and counterexamples.
    Information and Computation, 75(2), 87-106.

Algorithm Overview:
    1. Initialize observation table with S={ε}, E={ε}
    2. Fill table using membership queries
    3. While table not closed or not consistent:
        - If not closed: add row to S
        - If not consistent: add column to E
        - Refill table
    4. Construct hypothesis DFA from table
    5. Ask equivalence query
    6. If counterexample exists: process it and goto step 3
    7. Return hypothesis

Complexity:
    - O(|Q|²|Σ|) membership queries where |Q| is number of states, |Σ| is alphabet
    - O(|Q|) equivalence queries
    - Produces minimal DFA (fewest states)
"""

from oscura.inference.active_learning.observation_table import ObservationTable
from oscura.inference.active_learning.oracle import Oracle
from oscura.inference.state_machine import FiniteAutomaton


class LStarLearner:
    """Angluin's L* algorithm for learning DFAs.

    The L* algorithm learns the minimal DFA for a regular language by
    interacting with an oracle through membership and equivalence queries.

    Key properties:
    - Guaranteed convergence to correct minimal DFA
    - Polynomial query complexity
    - No negative examples required

    Attributes:
        oracle: Oracle for answering queries
        verbose: Print algorithm progress
        membership_queries: Count of membership queries made
        equivalence_queries: Count of equivalence queries made
    """

    def __init__(self, oracle: Oracle, verbose: bool = False):
        """Initialize L* learner.

        Args:
            oracle: Oracle for answering membership and equivalence queries
            verbose: Print algorithm progress if True
        """
        self.oracle = oracle
        self.verbose = verbose
        self.membership_queries = 0
        self.equivalence_queries = 0

    def learn(self, max_iterations: int = 1000) -> FiniteAutomaton:
        """Run L* algorithm to learn the target DFA.

        The algorithm proceeds as follows:
        1. Initialize observation table with S={ε}, E={ε}
        2. Fill table using membership queries
        3. Make table closed and consistent
        4. Construct hypothesis DFA
        5. Ask equivalence query
        6. If counterexample: update table and repeat from step 3
        7. Return final hypothesis

        Args:
            max_iterations: Maximum iterations to prevent infinite loops

        Returns:
            Learned minimal DFA

        Raises:
            ValueError: If algorithm doesn't converge within max_iterations
        """
        # Initialize observation table
        alphabet = self.oracle.get_alphabet()
        table = ObservationTable(alphabet=alphabet)

        if self.verbose:
            print(f"L* learning started with alphabet: {sorted(alphabet)}")

        # Fill initial table
        self._fill_table(table)

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Make table closed and consistent
            changed = True
            while changed:
                changed = False

                # Make closed
                if self._make_closed(table):
                    changed = True
                    self._fill_table(table)

                # Make consistent
                if self._make_consistent(table):
                    changed = True
                    self._fill_table(table)

            # Construct hypothesis
            hypothesis = table.to_dfa()

            if self.verbose:
                print(f"Iteration {iteration}: Hypothesis has {len(hypothesis.states)} states")

            # Equivalence query
            self.equivalence_queries += 1
            counterexample = self.oracle.equivalence_query(hypothesis)

            if counterexample is None:
                # Hypothesis is correct
                if self.verbose:
                    print(f"Learning complete after {iteration} iterations")
                    print(f"Membership queries: {self.membership_queries}")
                    print(f"Equivalence queries: {self.equivalence_queries}")
                return hypothesis

            # Process counterexample
            if self.verbose:
                print(f"Counterexample found: {counterexample}")

            self._process_counterexample(table, counterexample)
            self._fill_table(table)

        raise ValueError(f"L* did not converge after {max_iterations} iterations")

    def _fill_table(self, table: ObservationTable) -> None:
        """Fill table entries using membership queries.

        Fills all missing entries in the observation table by querying
        the oracle for membership of prefix+suffix combinations.

        For every s in S union SA and every e in E, ensures T[s·e] is filled.

        Args:
            table: Observation table to fill
        """
        # Compute SA = S union (S · Sigma)
        SA = set(table.S)
        for s in table.S:
            for a in table.alphabet:
                SA.add(s + (a,))

        # Fill all entries
        for s in SA:
            for e in table.E:
                word = s + e
                if word not in table.T:
                    self.membership_queries += 1
                    table.T[word] = self.oracle.membership_query(word)

    def _make_closed(self, table: ObservationTable) -> bool:
        """Make table closed.

        If table is not closed, adds one row from SA to S to move toward closure.

        A table is closed when every row in SA has an equivalent row in S.
        If not closed, we add the counterexample string to S.

        Args:
            table: Observation table to make closed

        Returns:
            True if table was modified, False if already closed
        """
        counterexample = table.find_closing_counterexample()

        if counterexample is None:
            return False

        # Add counterexample to S
        table.S.add(counterexample)

        if self.verbose:
            print(f"Added to S (closing): {counterexample}")

        return True

    def _make_consistent(self, table: ObservationTable) -> bool:
        """Make table consistent.

        If table is not consistent, adds a distinguishing suffix to E.

        A table is consistent when identical rows in S have identical
        one-step extensions. If inconsistent, we find the distinguishing
        suffix and add it to E.

        Args:
            table: Observation table to make consistent

        Returns:
            True if table was modified, False if already consistent
        """
        result = table.find_consistency_counterexample()

        if result is None:
            return False

        s1, s2, a = result

        # Find distinguishing suffix
        # row(s1·a) ≠ row(s2·a), so there exists e ∈ E where they differ
        for e in table.E:
            s1ae = s1 + (a,) + e
            s2ae = s2 + (a,) + e

            if table.T.get(s1ae, False) != table.T.get(s2ae, False):
                # Found distinguishing suffix: a·e
                distinguishing_suffix = (a,) + e
                table.E.add(distinguishing_suffix)

                if self.verbose:
                    print(f"Added to E (consistency): {distinguishing_suffix}")

                return True

        # Should never reach here if find_consistency_counterexample worked correctly
        return False

    def _process_counterexample(
        self, table: ObservationTable, counterexample: tuple[str, ...]
    ) -> None:
        """Process counterexample from equivalence query.

        Adds all prefixes of the counterexample and the counterexample
        itself to the observation table to refine the hypothesis.

        This uses the "add all suffixes" strategy which is simple and
        ensures progress.

        Args:
            table: Observation table to update
            counterexample: Counterexample word from equivalence query
        """
        # Add all suffixes of counterexample to E
        for i in range(len(counterexample) + 1):
            suffix = counterexample[i:]
            table.E.add(suffix)

        if self.verbose:
            print(f"Added {len(counterexample) + 1} suffixes to E from counterexample")
