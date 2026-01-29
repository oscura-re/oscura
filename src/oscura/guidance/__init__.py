"""Oscura guidance module.

Provides guided analysis workflows and recommendations.
"""

from oscura.guidance.recommender import (
    AnalysisHistory,
    Recommendation,
    suggest_next_steps,
)
from oscura.guidance.wizard import (
    AnalysisWizard,
    WizardResult,
    WizardStep,
)

__all__ = [
    "AnalysisHistory",
    "AnalysisWizard",
    "Recommendation",
    "WizardResult",
    "WizardStep",
    "suggest_next_steps",
]
