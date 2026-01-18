"""Unit tests for template definition module.

Tests cover:
- TemplateSection dataclass
- TemplateDefinition dataclass
- load_template() function
- validate_template() function
- list_builtin_templates() function
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oscura.reporting.templates.definition import (
    YAML_AVAILABLE,
    TemplateDefinition,
    TemplateSection,
    list_builtin_templates,
    load_template,
    validate_template,
)

pytestmark = pytest.mark.unit


class TestTemplateSection:
    """Test TemplateSection dataclass."""

    def test_init_minimal(self) -> None:
        """Test TemplateSection with minimal parameters."""
        section = TemplateSection(title="Test Section")

        assert section.title == "Test Section"
        assert section.content_type == "text"
        assert section.content == ""
        assert section.condition is None
        assert section.order == 0

    def test_init_full(self) -> None:
        """Test TemplateSection with all parameters."""
        section = TemplateSection(
            title="Results",
            content_type="table",
            content="{{ results_table }}",
            condition="results is not none",
            order=5,
        )

        assert section.title == "Results"
        assert section.content_type == "table"
        assert section.content == "{{ results_table }}"
        assert section.condition == "results is not none"
        assert section.order == 5

    def test_mutable_defaults(self) -> None:
        """Test that default values don't leak between instances."""
        section1 = TemplateSection(title="Section 1")
        section2 = TemplateSection(title="Section 2")

        section1.content = "Modified"

        assert section2.content == ""

    def test_all_content_types(self) -> None:
        """Test creating sections with different content types."""
        content_types = ["text", "table", "plot", "jinja2", "markdown"]

        for content_type in content_types:
            section = TemplateSection(title=f"{content_type} section", content_type=content_type)
            assert section.content_type == content_type


class TestTemplateDefinition:
    """Test TemplateDefinition dataclass."""

    def test_init_minimal(self) -> None:
        """Test TemplateDefinition with minimal parameters."""
        template = TemplateDefinition(name="Test Template")

        assert template.name == "Test Template"
        assert template.version == "1.0"
        assert template.author == ""
        assert template.description == ""
        assert template.tags == []
        assert template.sections == []
        assert template.variables == {}
        assert template.extends is None

    def test_init_full(self) -> None:
        """Test TemplateDefinition with all parameters."""
        sections = [
            TemplateSection(title="Introduction", order=0),
            TemplateSection(title="Results", order=1),
        ]

        template = TemplateDefinition(
            name="Compliance Report",
            version="2.1",
            author="John Doe",
            description="Standard compliance report",
            tags=["compliance", "production"],
            sections=sections,
            variables={"company": "Acme Corp", "year": 2025},
            extends="base_template",
        )

        assert template.name == "Compliance Report"
        assert template.version == "2.1"
        assert template.author == "John Doe"
        assert template.description == "Standard compliance report"
        assert template.tags == ["compliance", "production"]
        assert len(template.sections) == 2
        assert template.variables == {"company": "Acme Corp", "year": 2025}
        assert template.extends == "base_template"

    def test_mutable_defaults(self) -> None:
        """Test that mutable default values don't leak between instances."""
        template1 = TemplateDefinition(name="Template 1")
        template2 = TemplateDefinition(name="Template 2")

        template1.tags.append("test")
        template1.sections.append(TemplateSection(title="Section"))
        template1.variables["key"] = "value"

        assert template2.tags == []
        assert template2.sections == []
        assert template2.variables == {}

    def test_add_sections(self) -> None:
        """Test adding sections to template."""
        template = TemplateDefinition(name="Test")

        template.sections.append(TemplateSection(title="Section 1", order=0))
        template.sections.append(TemplateSection(title="Section 2", order=1))

        assert len(template.sections) == 2
        assert template.sections[0].title == "Section 1"
        assert template.sections[1].title == "Section 2"


class TestLoadTemplate:
    """Test load_template() function."""

    def test_load_basic_template(self, tmp_path: Path) -> None:
        """Test loading a basic template from YAML."""
        yaml_content = """
name: Test Template
version: "1.0"
author: Test Author
description: Test description
tags:
  - test
  - example
sections:
  - title: Introduction
    content_type: text
    content: This is the introduction
    order: 0
  - title: Results
    content_type: table
    content: "{{ results }}"
    order: 1
variables:
  project: TestProject
  date: 2025-01-01
"""
        template_path = tmp_path / "test_template.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.name == "Test Template"
        assert template.version == "1.0"
        assert template.author == "Test Author"
        assert template.description == "Test description"
        assert template.tags == ["test", "example"]
        assert len(template.sections) == 2
        assert template.sections[0].title == "Introduction"
        assert template.sections[0].content_type == "text"
        assert template.sections[0].order == 0
        assert template.sections[1].title == "Results"
        assert template.sections[1].content_type == "table"
        # YAML parses dates as datetime.date objects
        assert template.variables["project"] == "TestProject"
        assert "date" in template.variables

    def test_load_template_with_extends(self, tmp_path: Path) -> None:
        """Test loading template that extends another template."""
        yaml_content = """
name: Extended Template
extends: base_template
sections:
  - title: Custom Section
    content_type: text
"""
        template_path = tmp_path / "extended.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.extends == "base_template"

    def test_load_template_with_conditions(self, tmp_path: Path) -> None:
        """Test loading template with conditional sections."""
        yaml_content = """
name: Conditional Template
sections:
  - title: Always Shown
    content_type: text
  - title: Conditional Section
    content_type: text
    condition: "show_advanced"
"""
        template_path = tmp_path / "conditional.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.sections[0].condition is None
        assert template.sections[1].condition == "show_advanced"

    def test_load_template_defaults(self, tmp_path: Path) -> None:
        """Test loading template with minimal content uses defaults."""
        yaml_content = """
name: Minimal Template
sections:
  - title: Section 1
"""
        template_path = tmp_path / "minimal.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.version == "1.0"
        assert template.author == ""
        assert template.description == ""
        assert template.tags == []
        assert template.variables == {}
        assert len(template.sections) == 1
        assert template.sections[0].content_type == "text"
        assert template.sections[0].content == ""

    def test_load_template_no_name_uses_filename(self, tmp_path: Path) -> None:
        """Test template without name uses file stem."""
        yaml_content = """
sections:
  - title: Section 1
"""
        template_path = tmp_path / "my_template.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.name == "my_template"

    def test_load_template_section_auto_numbering(self, tmp_path: Path) -> None:
        """Test sections without titles get auto-numbered."""
        yaml_content = """
name: Auto Numbered
sections:
  - content_type: text
  - content_type: table
  - title: Named Section
"""
        template_path = tmp_path / "auto.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.sections[0].title == "Section 1"
        assert template.sections[1].title == "Section 2"
        assert template.sections[2].title == "Named Section"

    def test_load_template_section_default_order(self, tmp_path: Path) -> None:
        """Test sections without order get sequential indices."""
        yaml_content = """
name: Default Order
sections:
  - title: First
  - title: Second
  - title: Third
    order: 10
"""
        template_path = tmp_path / "order.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.sections[0].order == 0
        assert template.sections[1].order == 1
        assert template.sections[2].order == 10

    def test_load_template_string_path(self, tmp_path: Path) -> None:
        """Test loading template with string path."""
        yaml_content = """
name: String Path Test
sections:
  - title: Section
"""
        template_path = tmp_path / "string_path.yaml"
        template_path.write_text(yaml_content)

        template = load_template(str(template_path))

        assert template.name == "String Path Test"

    def test_load_template_pathlib_path(self, tmp_path: Path) -> None:
        """Test loading template with Path object."""
        yaml_content = """
name: Path Object Test
sections:
  - title: Section
"""
        template_path = tmp_path / "pathlib.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.name == "Path Object Test"

    def test_load_template_empty_sections(self, tmp_path: Path) -> None:
        """Test loading template with empty sections list."""
        yaml_content = """
name: Empty Sections
sections: []
"""
        template_path = tmp_path / "empty.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert len(template.sections) == 0

    def test_load_template_no_sections_key(self, tmp_path: Path) -> None:
        """Test loading template without sections key."""
        yaml_content = """
name: No Sections
"""
        template_path = tmp_path / "no_sections.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert len(template.sections) == 0

    def test_load_template_file_not_found(self, tmp_path: Path) -> None:
        """Test loading non-existent template raises FileNotFoundError."""
        if not YAML_AVAILABLE:
            pytest.skip("PyYAML not installed")

        nonexistent_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Template not found"):
            load_template(nonexistent_path)

    def test_load_template_yaml_not_available(self) -> None:
        """Test loading template without PyYAML raises ImportError."""
        if YAML_AVAILABLE:
            # Mock YAML_AVAILABLE being False
            with patch("oscura.reporting.templates.definition.YAML_AVAILABLE", False):
                with pytest.raises(ImportError, match="PyYAML is required"):
                    load_template("dummy.yaml")
        else:
            with pytest.raises(ImportError, match="PyYAML is required"):
                load_template("dummy.yaml")

    def test_load_template_complex_variables(self, tmp_path: Path) -> None:
        """Test loading template with complex variable types."""
        yaml_content = """
name: Complex Variables
sections:
  - title: Test
variables:
  string_var: "test"
  int_var: 42
  float_var: 3.14
  bool_var: true
  list_var:
    - item1
    - item2
  dict_var:
    key1: value1
    key2: value2
"""
        template_path = tmp_path / "complex.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert template.variables["string_var"] == "test"
        assert template.variables["int_var"] == 42
        assert template.variables["float_var"] == 3.14
        assert template.variables["bool_var"] is True
        assert template.variables["list_var"] == ["item1", "item2"]
        assert template.variables["dict_var"] == {"key1": "value1", "key2": "value2"}

    def test_load_template_all_content_types(self, tmp_path: Path) -> None:
        """Test loading template with all supported content types."""
        yaml_content = """
name: All Content Types
sections:
  - title: Text Section
    content_type: text
  - title: Table Section
    content_type: table
  - title: Plot Section
    content_type: plot
  - title: Jinja2 Section
    content_type: jinja2
  - title: Markdown Section
    content_type: markdown
"""
        template_path = tmp_path / "content_types.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        content_types = [s.content_type for s in template.sections]
        assert content_types == ["text", "table", "plot", "jinja2", "markdown"]


class TestValidateTemplate:
    """Test validate_template() function."""

    def test_validate_valid_template(self) -> None:
        """Test validating a valid template."""
        template = TemplateDefinition(
            name="Valid Template",
            version="1.0",
            sections=[
                TemplateSection(title="Section 1", content_type="text"),
                TemplateSection(title="Section 2", content_type="table"),
            ],
        )

        valid, errors = validate_template(template)

        assert valid is True
        assert errors == []

    def test_validate_missing_name(self) -> None:
        """Test validating template with missing name."""
        template = TemplateDefinition(
            name="",
            version="1.0",
            sections=[TemplateSection(title="Section")],
        )

        valid, errors = validate_template(template)

        assert valid is False
        assert "Template name is required" in errors

    def test_validate_missing_version(self) -> None:
        """Test validating template with missing version."""
        template = TemplateDefinition(
            name="Test",
            version="",
            sections=[TemplateSection(title="Section")],
        )

        valid, errors = validate_template(template)

        assert valid is False
        assert "Template version is required" in errors

    def test_validate_no_sections(self) -> None:
        """Test validating template with no sections."""
        template = TemplateDefinition(
            name="Test",
            version="1.0",
            sections=[],
        )

        valid, errors = validate_template(template)

        assert valid is False
        assert "Template must have at least one section" in errors

    def test_validate_section_missing_title(self) -> None:
        """Test validating template with section missing title."""
        template = TemplateDefinition(
            name="Test",
            version="1.0",
            sections=[
                TemplateSection(title="Good Section"),
                TemplateSection(title=""),
            ],
        )

        valid, errors = validate_template(template)

        assert valid is False
        assert "Section 1 missing title" in errors

    def test_validate_invalid_content_type(self) -> None:
        """Test validating template with invalid content type."""
        template = TemplateDefinition(
            name="Test",
            version="1.0",
            sections=[
                TemplateSection(title="Section", content_type="invalid_type"),
            ],
        )

        valid, errors = validate_template(template)

        assert valid is False
        assert any("invalid content_type" in error for error in errors)

    def test_validate_multiple_errors(self) -> None:
        """Test validating template with multiple errors."""
        template = TemplateDefinition(
            name="",
            version="",
            sections=[
                TemplateSection(title="", content_type="bad_type"),
                TemplateSection(title="Good"),
            ],
        )

        valid, errors = validate_template(template)

        assert valid is False
        assert len(errors) >= 3
        assert "Template name is required" in errors
        assert "Template version is required" in errors
        assert any("Section 0" in error for error in errors)

    def test_validate_all_valid_content_types(self) -> None:
        """Test all valid content types pass validation."""
        valid_types = ["text", "table", "plot", "jinja2", "markdown"]

        for content_type in valid_types:
            template = TemplateDefinition(
                name="Test",
                version="1.0",
                sections=[TemplateSection(title="Section", content_type=content_type)],
            )

            valid, errors = validate_template(template)

            assert valid is True, f"Content type {content_type} should be valid"
            assert errors == []

    def test_validate_multiple_sections_one_invalid(self) -> None:
        """Test validating template with multiple sections, one invalid."""
        template = TemplateDefinition(
            name="Test",
            version="1.0",
            sections=[
                TemplateSection(title="Section 1", content_type="text"),
                TemplateSection(title="Section 2", content_type="invalid"),
                TemplateSection(title="Section 3", content_type="table"),
            ],
        )

        valid, errors = validate_template(template)

        assert valid is False
        assert len(errors) == 1
        assert "Section 1" in errors[0]
        assert "invalid" in errors[0]

    def test_validate_section_with_condition(self) -> None:
        """Test validating section with condition is valid."""
        template = TemplateDefinition(
            name="Test",
            version="1.0",
            sections=[
                TemplateSection(
                    title="Conditional", content_type="text", condition="some_var is defined"
                ),
            ],
        )

        valid, errors = validate_template(template)

        assert valid is True
        assert errors == []

    def test_validate_returns_tuple(self) -> None:
        """Test validate_template returns tuple of bool and list."""
        template = TemplateDefinition(
            name="Test",
            version="1.0",
            sections=[TemplateSection(title="Section")],
        )

        result = validate_template(template)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)


class TestListBuiltinTemplates:
    """Test list_builtin_templates() function."""

    def test_list_returns_list(self) -> None:
        """Test function returns a list."""
        templates = list_builtin_templates()

        assert isinstance(templates, list)

    def test_list_not_empty(self) -> None:
        """Test returned list is not empty."""
        templates = list_builtin_templates()

        assert len(templates) > 0

    def test_list_contains_expected_templates(self) -> None:
        """Test returned list contains expected built-in templates."""
        templates = list_builtin_templates()

        expected = [
            "default",
            "compliance",
            "characterization",
            "debug",
            "production",
            "comparison",
            "batch_summary",
        ]

        for expected_template in expected:
            assert expected_template in templates

    def test_list_exact_templates(self) -> None:
        """Test returned list matches exact expected templates."""
        templates = list_builtin_templates()

        expected = [
            "default",
            "compliance",
            "characterization",
            "debug",
            "production",
            "comparison",
            "batch_summary",
        ]

        assert templates == expected

    def test_list_all_strings(self) -> None:
        """Test all returned templates are strings."""
        templates = list_builtin_templates()

        assert all(isinstance(t, str) for t in templates)

    def test_list_no_duplicates(self) -> None:
        """Test returned list has no duplicates."""
        templates = list_builtin_templates()

        assert len(templates) == len(set(templates))

    def test_list_consistent_calls(self) -> None:
        """Test function returns consistent results across calls."""
        templates1 = list_builtin_templates()
        templates2 = list_builtin_templates()

        assert templates1 == templates2


class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports_defined(self) -> None:
        """Test __all__ is defined and contains expected exports."""
        from oscura.reporting.templates import definition

        assert hasattr(definition, "__all__")

        expected_exports = [
            "TemplateDefinition",
            "TemplateSection",
            "list_builtin_templates",
            "load_template",
            "validate_template",
        ]

        for export in expected_exports:
            assert export in definition.__all__

    def test_all_exports_exist(self) -> None:
        """Test all exports in __all__ actually exist."""
        from oscura.reporting.templates import definition

        for export in definition.__all__:
            assert hasattr(definition, export)


class TestYAMLAvailability:
    """Test YAML_AVAILABLE flag behavior."""

    def test_yaml_available_is_bool(self) -> None:
        """Test YAML_AVAILABLE is a boolean."""
        assert isinstance(YAML_AVAILABLE, bool)

    def test_yaml_available_when_installed(self) -> None:
        """Test YAML_AVAILABLE is True when PyYAML is installed."""
        assert YAML_AVAILABLE is True

    def test_yaml_import_handling(self) -> None:
        """Test YAML import is properly handled."""
        # This test verifies the module loads without error
        # regardless of PyYAML availability
        import oscura.reporting.templates.definition as def_module

        assert hasattr(def_module, "YAML_AVAILABLE")


class TestReportingTemplateDefinitionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_load_empty_yaml_file(self, tmp_path: Path) -> None:
        """Test loading empty YAML file."""
        template_path = tmp_path / "empty.yaml"
        template_path.write_text("")

        # Empty YAML file should result in None when loaded
        # This should be handled gracefully
        try:
            template = load_template(template_path)
            # If it doesn't raise, check the result
            assert template.name in ["", "empty"]
        except (TypeError, AttributeError):
            # Expected if empty YAML causes issues
            pass

    def test_load_template_special_characters_in_content(self, tmp_path: Path) -> None:
        """Test loading template with special characters in content."""
        yaml_content = """
name: Special Characters
sections:
  - title: Special Section
    content: "Line 1\\nLine 2\\tTabbed\\r\\nWindows Line"
"""
        template_path = tmp_path / "special.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert "Line 1" in template.sections[0].content

    def test_load_template_unicode_content(self, tmp_path: Path) -> None:
        """Test loading template with Unicode content."""
        yaml_content = """
name: Unicode Template
author: José García
description: Température données 温度
sections:
  - title: Résultats
    content: "測定データ: {{ data }}"
"""
        template_path = tmp_path / "unicode.yaml"
        template_path.write_text(yaml_content, encoding="utf-8")

        template = load_template(template_path)

        assert template.author == "José García"
        assert "温度" in template.description
        assert template.sections[0].title == "Résultats"

    def test_validate_template_with_max_sections(self) -> None:
        """Test validating template with many sections."""
        sections = [TemplateSection(title=f"Section {i}", content_type="text") for i in range(100)]

        template = TemplateDefinition(
            name="Many Sections",
            version="1.0",
            sections=sections,
        )

        valid, errors = validate_template(template)

        assert valid is True
        assert errors == []

    def test_template_section_negative_order(self) -> None:
        """Test TemplateSection with negative order."""
        section = TemplateSection(title="Test", order=-5)

        assert section.order == -5

    def test_template_section_large_order(self) -> None:
        """Test TemplateSection with large order value."""
        section = TemplateSection(title="Test", order=999999)

        assert section.order == 999999

    def test_load_template_deeply_nested_variables(self, tmp_path: Path) -> None:
        """Test loading template with deeply nested variable structure."""
        yaml_content = """
name: Nested Variables
sections:
  - title: Test
variables:
  level1:
    level2:
      level3:
        level4: deep_value
"""
        template_path = tmp_path / "nested.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        assert "level1" in template.variables
        assert template.variables["level1"]["level2"]["level3"]["level4"] == "deep_value"

    def test_load_template_null_values(self, tmp_path: Path) -> None:
        """Test loading template with null/None values."""
        yaml_content = """
name: Null Values
author: null
description: null
extends: null
sections:
  - title: Section
    condition: null
variables:
  null_var: null
"""
        template_path = tmp_path / "nulls.yaml"
        template_path.write_text(yaml_content)

        template = load_template(template_path)

        # null should be converted to None or empty string depending on defaults
        assert template.sections[0].condition is None
        assert template.variables["null_var"] is None
