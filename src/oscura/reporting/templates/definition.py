"""Template definition format for reports.

Defines YAML-based template format for report structure with Jinja2
templating for variables, conditionals, and loops.


References:
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class TemplateSection:
    """Template section definition.

    Attributes:
        title: Section title.
        content_type: Type of content (text, table, plot, jinja2).
        content: Content template string (may include Jinja2 variables).
        condition: Optional Jinja2 condition for inclusion.
        order: Display order.

    References:
        REPORT-007: Template Definition Format
    """

    title: str
    content_type: str = "text"
    content: str = ""
    condition: str | None = None
    order: int = 0


@dataclass
class TemplateDefinition:
    """Report template definition.

    Attributes:
        name: Template name.
        version: Template version.
        author: Template author.
        description: Template description.
        tags: Template tags.
        sections: Ordered list of sections.
        variables: Template variables with defaults.
        extends: Base template to extend (optional).

    References:
        REPORT-007: Template Definition Format
    """

    name: str
    version: str = "1.0"
    author: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    sections: list[TemplateSection] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    extends: str | None = None


def load_template(path: str | Path) -> TemplateDefinition:
    """Load template from YAML file.

    Args:
        path: Path to YAML template file.

    Returns:
        TemplateDefinition object.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If template file not found.

    Example:
        >>> template = load_template("templates/compliance.yaml")
        >>> print(template.name)

    References:
        REPORT-007: Template Definition Format
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for template loading. Install with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse template metadata
    template = TemplateDefinition(
        name=data.get("name", path.stem),
        version=data.get("version", "1.0"),
        author=data.get("author", ""),
        description=data.get("description", ""),
        tags=data.get("tags", []),
        extends=data.get("extends"),
    )

    # Parse sections
    for i, section_data in enumerate(data.get("sections", [])):
        section = TemplateSection(
            title=section_data.get("title", f"Section {i + 1}"),
            content_type=section_data.get("content_type", "text"),
            content=section_data.get("content", ""),
            condition=section_data.get("condition"),
            order=section_data.get("order", i),
        )
        template.sections.append(section)

    # Parse variables
    template.variables = data.get("variables", {})

    return template


def validate_template(template: TemplateDefinition) -> tuple[bool, list[str]]:
    """Validate template definition.

    Args:
        template: Template to validate.

    Returns:
        Tuple of (is_valid, list_of_errors).

    Example:
        >>> template = TemplateDefinition(name="test")
        >>> valid, errors = validate_template(template)
        >>> print(valid)
        True

    References:
        REPORT-007: Template Definition Format
    """
    errors = []

    # Check required fields
    if not template.name:
        errors.append("Template name is required")

    if not template.version:
        errors.append("Template version is required")

    # Check sections
    if not template.sections:
        errors.append("Template must have at least one section")

    for i, section in enumerate(template.sections):
        if not section.title:
            errors.append(f"Section {i} missing title")

        if section.content_type not in ["text", "table", "plot", "jinja2", "markdown"]:
            errors.append(f"Section {i} has invalid content_type: {section.content_type}")

    return len(errors) == 0, errors


def list_builtin_templates() -> list[str]:
    """List available built-in templates.

    Returns:
        List of built-in template names.

    Example:
        >>> templates = list_builtin_templates()
        >>> "compliance" in templates
        True

    References:
        REPORT-007: Template Definition Format
    """
    return [
        "default",
        "compliance",
        "characterization",
        "debug",
        "production",
        "comparison",
        "batch_summary",
    ]


__all__ = [
    "TemplateDefinition",
    "TemplateSection",
    "list_builtin_templates",
    "load_template",
    "validate_template",
]
