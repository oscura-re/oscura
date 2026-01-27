"""Onboarding and help system for Oscura.

This package provides interactive tutorials, context-sensitive help,
and guided analysis features for new users.
"""

from oscura.cli.onboarding.help import (
    explain_result,
    get_example,
    get_help,
    suggest_commands,
)
from oscura.cli.onboarding.tutorials import (
    Tutorial,
    TutorialStep,
    get_tutorial,
    list_tutorials,
    run_tutorial,
)
from oscura.cli.onboarding.wizard import (
    AnalysisWizard,
    WizardStep,
    run_wizard,
)

__all__ = [
    "AnalysisWizard",
    "Tutorial",
    "TutorialStep",
    "WizardStep",
    "explain_result",
    "get_example",
    "get_help",
    "get_tutorial",
    "list_tutorials",
    "run_tutorial",
    "run_wizard",
    "suggest_commands",
]
