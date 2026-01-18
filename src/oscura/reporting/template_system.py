"""Template system for Oscura reports.

This module provides template loading, management, inheritance, and built-in
report templates.


Example:
    >>> from oscura.reporting.template_system import load_template, register_template
    >>> template = load_template("compliance")
    >>> # Create custom template extending compliance
    >>> custom = load_template("compliance")
    >>> register_template("my_compliance", custom)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TemplateSection:
    """A section definition in a template.

    Attributes:
        title: Section title.
        content_type: Type of content (text, table, figure, measurement).
        condition: Conditional expression for inclusion.
        template: Jinja2 template for content.
        subsections: Child sections.
        order: Section order (for sorting during inheritance merge).
        override: If True, replaces parent section with same title.

    References:
        REPORT-007: Template Definition Format
    """

    title: str
    content_type: str = "text"
    condition: str | None = None
    template: str = ""
    subsections: list[TemplateSection] = field(default_factory=list)
    order: int = 0
    override: bool = False


@dataclass
class ReportTemplate:
    """A report template definition.

    Attributes:
        name: Template name.
        version: Template version.
        description: Template description.
        author: Template author.
        extends: Parent template name for inheritance (REPORT-005).
        sections: Template sections.
        styles: Style definitions.
        metadata: Additional metadata.
        overrides: Section-specific overrides (REPORT-008).

    References:
        REPORT-005: Template Inheritance
        REPORT-007: Template Definition Format
        REPORT-008: Template Overrides
    """

    name: str
    version: str = "1.0"
    description: str = ""
    author: str = ""
    extends: str | None = None
    sections: list[TemplateSection] = field(default_factory=list)
    styles: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


# Built-in templates
BUILTIN_TEMPLATES: dict[str, ReportTemplate] = {
    "default": ReportTemplate(
        name="Default Report",
        version="1.0",
        description="Standard analysis report template",
        sections=[
            TemplateSection(
                title="Executive Summary",
                content_type="text",
                template="{{ summary }}",
                order=0,
            ),
            TemplateSection(
                title="Test Results",
                content_type="table",
                template="{{ results_table }}",
                order=10,
            ),
            TemplateSection(
                title="Methodology",
                content_type="text",
                condition="verbosity != 'executive'",
                order=20,
            ),
        ],
    ),
    "compliance": ReportTemplate(
        name="Compliance Report",
        version="1.0",
        description="Regulatory compliance testing report",
        extends="default",
        sections=[
            TemplateSection(
                title="Executive Summary",
                content_type="text",
                template="{{ compliance_summary }}",
                override=True,
                order=0,
            ),
            TemplateSection(
                title="Test Standards",
                content_type="text",
                template="Standards tested: {{ standards }}",
                order=5,
            ),
            TemplateSection(
                title="Violations",
                content_type="table",
                condition="has_violations",
                order=15,
            ),
            TemplateSection(
                title="Certificate",
                content_type="text",
                order=30,
            ),
        ],
    ),
    "characterization": ReportTemplate(
        name="Characterization Report",
        version="1.0",
        description="Device characterization report",
        sections=[
            TemplateSection(
                title="Summary",
                content_type="text",
                order=0,
            ),
            TemplateSection(
                title="Timing Parameters",
                content_type="table",
                order=10,
            ),
            TemplateSection(
                title="Signal Quality",
                content_type="table",
                order=20,
            ),
            TemplateSection(
                title="Margin Analysis",
                content_type="table",
                order=30,
            ),
            TemplateSection(
                title="Waveform Plots",
                content_type="figure",
                order=40,
            ),
        ],
    ),
    "debug": ReportTemplate(
        name="Debug Report",
        version="1.0",
        description="Detailed debug report with full data",
        sections=[
            TemplateSection(
                title="Summary",
                content_type="text",
                order=0,
            ),
            TemplateSection(
                title="Error Analysis",
                content_type="text",
                order=10,
            ),
            TemplateSection(
                title="Protocol Decode",
                content_type="table",
                order=20,
            ),
            TemplateSection(
                title="Timing Diagram",
                content_type="figure",
                order=30,
            ),
            TemplateSection(
                title="Raw Data",
                content_type="text",
                order=40,
            ),
            TemplateSection(
                title="Provenance",
                content_type="text",
                order=50,
            ),
        ],
    ),
    "production": ReportTemplate(
        name="Production Report",
        version="1.0",
        description="Production test report with pass/fail and yield",
        sections=[
            TemplateSection(
                title="Test Summary",
                content_type="text",
                template="Tested: {{ total }} | Passed: {{ passed }} | Failed: {{ failed }}",
                order=0,
            ),
            TemplateSection(
                title="Results",
                content_type="table",
                order=10,
            ),
            TemplateSection(
                title="Yield Analysis",
                content_type="table",
                order=20,
            ),
        ],
    ),
    "comparison": ReportTemplate(
        name="Comparison Report",
        version="1.0",
        description="Before/after comparison report",
        sections=[
            TemplateSection(
                title="Summary",
                content_type="text",
                order=0,
            ),
            TemplateSection(
                title="Differences",
                content_type="table",
                order=10,
            ),
            TemplateSection(
                title="Side-by-Side Comparison",
                content_type="figure",
                order=20,
            ),
        ],
    ),
}

# User-registered templates (REPORT-006)
_USER_TEMPLATES: dict[str, ReportTemplate] = {}


def register_template(
    name: str,
    template: ReportTemplate,
    *,
    overwrite: bool = False,
) -> None:
    """Register a user template.

    Allows users to define custom templates or extend built-in ones.

    Args:
        name: Template name for registration.
        template: Template definition.
        overwrite: If True, allows overwriting existing templates.

    Raises:
        ValueError: If name exists and overwrite=False.

    Example:
        >>> from oscura.reporting.template_system import (
        ...     register_template, ReportTemplate, TemplateSection
        ... )
        >>> my_template = ReportTemplate(
        ...     name="Custom Report",
        ...     sections=[TemplateSection(title="My Section")]
        ... )
        >>> register_template("custom", my_template)

    References:
        REPORT-006: User Template Registration
    """
    if name in _USER_TEMPLATES and not overwrite:
        raise ValueError(f"Template '{name}' already registered. Use overwrite=True to replace.")

    if name in BUILTIN_TEMPLATES and not overwrite:
        raise ValueError(f"Cannot overwrite built-in template '{name}'. Use overwrite=True.")

    _USER_TEMPLATES[name] = template


def unregister_template(name: str) -> bool:
    """Unregister a user template.

    Args:
        name: Template name.

    Returns:
        True if template was removed, False if not found.

    References:
        REPORT-006: User Template Registration
    """
    if name in _USER_TEMPLATES:
        del _USER_TEMPLATES[name]
        return True
    return False


def extend_template(
    base_name: str,
    *,
    name: str | None = None,
    description: str | None = None,
    add_sections: list[TemplateSection] | None = None,
    remove_sections: list[str] | None = None,
    section_overrides: dict[str, dict[str, Any]] | None = None,
    style_overrides: dict[str, Any] | None = None,
) -> ReportTemplate:
    """Create a new template by extending an existing one.

    Implements template inheritance with section merging and overrides.

    Args:
        base_name: Name of template to extend.
        name: Name for new template.
        description: Description for new template.
        add_sections: New sections to add.
        remove_sections: Section titles to remove.
        section_overrides: Dict of section title -> field overrides.
        style_overrides: Style fields to override.

    Returns:
        New ReportTemplate with inherited and modified sections.

    Example:
        >>> # Create custom compliance template
        >>> custom = extend_template(
        ...     "compliance",
        ...     name="FDA Compliance",
        ...     add_sections=[TemplateSection(title="FDA Requirements")],
        ...     section_overrides={
        ...         "Certificate": {"template": "FDA Certificate: {{ cert_id }}"}
        ...     }
        ... )

    References:
        REPORT-005: Template Inheritance
        REPORT-008: Template Overrides
    """
    # Load base template (resolving inheritance chain)
    base = load_template(base_name)

    # Deep copy to avoid modifying original
    new_template = copy.deepcopy(base)

    # Update metadata
    if name:
        new_template.name = name
    if description:
        new_template.description = description
    new_template.extends = base_name

    # Apply section removals
    if remove_sections:
        new_template.sections = [
            sec for sec in new_template.sections if sec.title not in remove_sections
        ]

    # Apply section overrides
    if section_overrides:
        for sec in new_template.sections:
            if sec.title in section_overrides:
                overrides = section_overrides[sec.title]
                for field_name, value in overrides.items():
                    if hasattr(sec, field_name):
                        setattr(sec, field_name, value)

    # Add new sections
    if add_sections:
        new_template.sections.extend(add_sections)

    # Sort sections by order
    new_template.sections.sort(key=lambda s: s.order)

    # Apply style overrides
    if style_overrides:
        new_template.styles.update(style_overrides)

    return new_template


def _resolve_inheritance(
    template: ReportTemplate, visited: set[str] | None = None
) -> ReportTemplate:
    """Resolve template inheritance chain.

    Args:
        template: Template to resolve.
        visited: Set of already visited template names (for cycle detection).

    Returns:
        Template with all inherited sections merged.

    Raises:
        ValueError: If circular inheritance detected.

    References:
        REPORT-005: Template Inheritance
    """
    if visited is None:
        visited = set()

    if template.name in visited:
        raise ValueError(f"Circular template inheritance detected: {template.name}")

    if not template.extends:
        return template

    visited.add(template.name)

    # Get parent template
    parent_name = template.extends
    if parent_name in _USER_TEMPLATES:
        parent = copy.deepcopy(_USER_TEMPLATES[parent_name])
    elif parent_name in BUILTIN_TEMPLATES:
        parent = copy.deepcopy(BUILTIN_TEMPLATES[parent_name])
    else:
        raise ValueError(f"Parent template not found: {parent_name}")

    # Recursively resolve parent inheritance
    parent = _resolve_inheritance(parent, visited)

    # Merge sections
    # Child sections with override=True replace parent sections with same title
    parent_sections = {sec.title: sec for sec in parent.sections}

    for child_sec in template.sections:
        if child_sec.override or child_sec.title in parent_sections:
            # Override or replace parent section
            parent_sections[child_sec.title] = child_sec
        else:
            # Add new section
            parent_sections[child_sec.title] = child_sec

    # Sort by order
    merged_sections = sorted(parent_sections.values(), key=lambda s: s.order)

    # Merge styles (child overrides parent)
    merged_styles = {**parent.styles, **template.styles}

    # Merge metadata
    merged_metadata = {**parent.metadata, **template.metadata}

    return ReportTemplate(
        name=template.name,
        version=template.version,
        description=template.description or parent.description,
        author=template.author or parent.author,
        extends=template.extends,
        sections=merged_sections,
        styles=merged_styles,
        metadata=merged_metadata,
        overrides=template.overrides,
    )


def load_template(name_or_path: str, *, resolve_inheritance: bool = True) -> ReportTemplate:
    """Load a report template.

    Args:
        name_or_path: Template name (builtin or registered) or path to YAML file.
        resolve_inheritance: If True, resolve template inheritance chain.

    Returns:
        ReportTemplate instance.

    Raises:
        ValueError: If template not found.

    Example:
        >>> template = load_template("compliance")
        >>> template = load_template("custom_template.yaml")

    References:
        REPORT-005: Template Inheritance
        REPORT-006: User Template Registration
    """
    template = None

    # Check user-registered templates first (REPORT-006)
    if name_or_path in _USER_TEMPLATES:
        template = copy.deepcopy(_USER_TEMPLATES[name_or_path])
    # Then check builtin templates
    elif name_or_path in BUILTIN_TEMPLATES:
        template = copy.deepcopy(BUILTIN_TEMPLATES[name_or_path])
    else:
        # Try loading from file
        path = Path(name_or_path)
        if path.exists():
            template = _load_template_file(path)
        else:
            # Try adding .yaml extension
            yaml_path = Path(f"{name_or_path}.yaml")
            if yaml_path.exists():
                template = _load_template_file(yaml_path)

    if template is None:
        raise ValueError(f"Template not found: {name_or_path}")

    # Resolve inheritance if requested (REPORT-005)
    if resolve_inheritance and template.extends:
        template = _resolve_inheritance(template)

    return template


def _load_template_file(path: Path) -> ReportTemplate:
    """Load template from YAML file.

    Args:
        path: Path to template YAML file.

    Returns:
        ReportTemplate instance loaded from file.

    References:
        REPORT-007: Template Definition Format
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    template_data = data.get("template", data)

    sections = []
    for idx, sec_data in enumerate(template_data.get("sections", [])):
        section = TemplateSection(
            title=sec_data.get("title", ""),
            content_type=sec_data.get("content_type", "text"),
            condition=sec_data.get("condition"),
            template=sec_data.get("template", sec_data.get("content", "")),
            order=sec_data.get("order", idx * 10),
            override=sec_data.get("override", False),
        )
        sections.append(section)

    return ReportTemplate(
        name=template_data.get("name", path.stem),
        version=template_data.get("version", "1.0"),
        description=template_data.get("description", ""),
        author=template_data.get("author", ""),
        extends=template_data.get("extends"),
        sections=sections,
        styles=template_data.get("styles", {}),
        metadata=template_data.get("metadata", {}),
        overrides=template_data.get("overrides", {}),
    )


def list_templates(*, include_user: bool = True) -> list[str]:
    """List available template names.

    Args:
        include_user: Include user-registered templates.

    Returns:
        List of template names.

    References:
        REPORT-006: User Template Registration
    """
    names = list(BUILTIN_TEMPLATES.keys())
    if include_user:
        names.extend(_USER_TEMPLATES.keys())
    return sorted(set(names))


def get_template_info(name: str) -> dict[str, Any]:
    """Get information about a template.

    Args:
        name: Template name.

    Returns:
        Dictionary with template info.

    Raises:
        ValueError: If template name unknown.

    References:
        REPORT-005: Template Inheritance
        REPORT-006: User Template Registration
    """
    if name in _USER_TEMPLATES:
        template = _USER_TEMPLATES[name]
        source = "user"
    elif name in BUILTIN_TEMPLATES:
        template = BUILTIN_TEMPLATES[name]
        source = "builtin"
    else:
        raise ValueError(f"Unknown template: {name}")

    return {
        "name": template.name,
        "version": template.version,
        "description": template.description,
        "author": template.author,
        "extends": template.extends,
        "num_sections": len(template.sections),
        "section_titles": [sec.title for sec in template.sections],
        "source": source,
    }


def save_template(template: ReportTemplate, path: str | Path) -> None:
    """Save template to YAML file.

    Args:
        template: Template to save.
        path: Output file path.

    References:
        REPORT-007: Template Definition Format
    """
    path = Path(path)

    data = {
        "template": {
            "name": template.name,
            "version": template.version,
            "description": template.description,
            "author": template.author,
            "extends": template.extends,
            "sections": [
                {
                    "title": sec.title,
                    "content_type": sec.content_type,
                    "condition": sec.condition,
                    "template": sec.template,
                    "order": sec.order,
                    "override": sec.override,
                }
                for sec in template.sections
            ],
            "styles": template.styles,
            "metadata": template.metadata,
            "overrides": template.overrides,
        }
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_template(
    name: str,
    sections: list[TemplateSection],
    *,
    extends: str | None = None,
    description: str = "",
    author: str = "",
    styles: dict[str, Any] | None = None,
) -> ReportTemplate:
    """Create a new template.

    Convenience function for creating templates programmatically.

    Args:
        name: Template name.
        sections: List of sections.
        extends: Parent template name for inheritance.
        description: Template description.
        author: Template author.
        styles: Style definitions.

    Returns:
        New ReportTemplate.

    Example:
        >>> template = create_template(
        ...     "quick_report",
        ...     sections=[
        ...         TemplateSection(title="Summary", template="{{ summary }}"),
        ...         TemplateSection(title="Results", content_type="table"),
        ...     ],
        ...     description="Quick summary report"
        ... )

    References:
        REPORT-007: Template Definition Format
    """
    return ReportTemplate(
        name=name,
        description=description,
        author=author,
        extends=extends,
        sections=sections,
        styles=styles or {},
    )


__all__ = [
    "BUILTIN_TEMPLATES",
    "ReportTemplate",
    "TemplateSection",
    "create_template",
    "extend_template",
    "get_template_info",
    "list_templates",
    "load_template",
    "register_template",
    "save_template",
    "unregister_template",
]
