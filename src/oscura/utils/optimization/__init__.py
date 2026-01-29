"""Parameter optimization and search algorithms.

This module provides grid search and randomized search for finding optimal
analysis parameters.
"""

from oscura.utils.optimization.search import (
    GridSearchCV,
    RandomizedSearchCV,
    ScoringFunction,
    SearchResult,
)

__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
    "ScoringFunction",
    "SearchResult",
]
