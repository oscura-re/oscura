"""Template system for reports."""

from oscura.reporting.templates.definition import (
    TemplateDefinition,
    list_builtin_templates,
    load_template,
    validate_template,
)

__all__ = [
    "TemplateDefinition",
    "list_builtin_templates",
    "load_template",
    "validate_template",
]
