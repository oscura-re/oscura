"""Tests for report template system."""

import tempfile
from pathlib import Path

import pytest

from oscura.reporting.template_system import (
    BUILTIN_TEMPLATES,
    ReportTemplate,
    TemplateSection,
    create_template,
    extend_template,
    get_template_info,
    list_templates,
    load_template,
    register_template,
    save_template,
    unregister_template,
)

pytestmark = pytest.mark.unit


class TestBuiltinTemplates:
    """Tests for built-in templates (REPORT-014)."""

    def test_builtin_templates_exist(self):
        """Test that expected built-in templates are defined."""
        expected_templates = [
            "default",
            "compliance",
            "characterization",
            "debug",
            "production",
            "comparison",
        ]

        for name in expected_templates:
            assert name in BUILTIN_TEMPLATES, f"Missing built-in template: {name}"

    def test_default_template_sections(self):
        """Test default template has expected sections."""
        template = BUILTIN_TEMPLATES["default"]
        section_titles = [sec.title for sec in template.sections]

        assert "Executive Summary" in section_titles
        assert "Test Results" in section_titles

    def test_compliance_template_extends_default(self):
        """Test compliance template extends default (REPORT-005)."""
        template = BUILTIN_TEMPLATES["compliance"]
        assert template.extends == "default"


class TestLoadTemplate:
    """Tests for template loading (REPORT-005, REPORT-006, REPORT-007)."""

    def test_load_builtin_template(self):
        """Test loading a built-in template."""
        template = load_template("default")
        assert template.name == "Default Report"
        assert len(template.sections) > 0

    def test_load_template_resolves_inheritance(self):
        """Test that loading resolves template inheritance (REPORT-005)."""
        template = load_template("compliance")

        # Should have sections from both default and compliance
        section_titles = [sec.title for sec in template.sections]

        # From default
        assert "Executive Summary" in section_titles
        assert "Test Results" in section_titles
        # From compliance
        assert "Test Standards" in section_titles
        assert "Certificate" in section_titles

    def test_load_template_not_found_raises(self):
        """Test loading non-existent template raises ValueError."""
        with pytest.raises(ValueError, match="Template not found"):
            load_template("nonexistent_template")

    def test_load_template_from_file(self):
        """Test loading template from YAML file (REPORT-007)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "custom.yaml"

            # Create a test template file
            template_content = """
template:
  name: "Custom Template"
  version: "1.0"
  description: "Test custom template"
  sections:
    - title: "Section 1"
      content_type: "text"
      template: "{{ content }}"
"""
            template_path.write_text(template_content)

            # Load and verify
            template = load_template(str(template_path))
            assert template.name == "Custom Template"
            assert len(template.sections) == 1
            assert template.sections[0].title == "Section 1"


class TestRegisterTemplate:
    """Tests for user template registration (REPORT-006)."""

    def setup_method(self):
        """Clean up registered templates before each test."""
        # Clean up any test templates
        for name in ["test_template", "test_override"]:
            unregister_template(name)

    def teardown_method(self):
        """Clean up registered templates after each test."""
        # Clean up any test templates
        for name in ["test_template", "test_override"]:
            unregister_template(name)

    def test_register_custom_template(self):
        """Test registering a custom template."""
        custom = ReportTemplate(
            name="Test Template",
            description="Test description",
            sections=[TemplateSection(title="Test Section")],
        )

        register_template("test_template", custom)

        # Should be loadable
        loaded = load_template("test_template")
        assert loaded.name == "Test Template"
        assert len(loaded.sections) == 1

    def test_register_duplicate_raises(self):
        """Test registering duplicate template raises without overwrite."""
        custom = ReportTemplate(name="Test", sections=[])
        register_template("test_template", custom)

        with pytest.raises(ValueError, match="already registered"):
            register_template("test_template", custom)

    def test_register_with_overwrite(self):
        """Test registering with overwrite=True replaces existing."""
        original = ReportTemplate(name="Original", sections=[])
        register_template("test_template", original)

        replacement = ReportTemplate(name="Replacement", sections=[])
        register_template("test_template", replacement, overwrite=True)

        loaded = load_template("test_template")
        assert loaded.name == "Replacement"

    def test_unregister_template(self):
        """Test unregistering a template."""
        custom = ReportTemplate(name="Test", sections=[])
        register_template("test_template", custom)

        result = unregister_template("test_template")
        assert result is True

        # Should no longer be loadable (falls back to file lookup)
        with pytest.raises(ValueError, match="Template not found"):
            load_template("test_template")

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering non-existent template returns False."""
        result = unregister_template("nonexistent_template_xyz")
        assert result is False


class TestExtendTemplate:
    """Tests for template extension (REPORT-005, REPORT-008)."""

    def test_extend_adds_sections(self):
        """Test extending template with new sections."""
        extended = extend_template(
            "default",
            name="Extended Default",
            add_sections=[
                TemplateSection(title="Custom Section", order=100),
            ],
        )

        section_titles = [sec.title for sec in extended.sections]
        assert "Custom Section" in section_titles
        assert "Executive Summary" in section_titles  # Inherited

    def test_extend_removes_sections(self):
        """Test extending template with section removal."""
        extended = extend_template(
            "default",
            name="Minimal Default",
            remove_sections=["Methodology"],
        )

        section_titles = [sec.title for sec in extended.sections]
        assert "Methodology" not in section_titles
        assert "Executive Summary" in section_titles  # Still there

    def test_extend_overrides_sections(self):
        """Test extending template with section overrides (REPORT-008)."""
        extended = extend_template(
            "default",
            name="Overridden Default",
            section_overrides={
                "Executive Summary": {
                    "template": "Custom: {{ custom_summary }}",
                },
            },
        )

        summary_section = next(sec for sec in extended.sections if sec.title == "Executive Summary")
        assert "custom_summary" in summary_section.template

    def test_extend_preserves_order(self):
        """Test that extended template sections are ordered correctly."""
        extended = extend_template(
            "default",
            add_sections=[
                TemplateSection(title="First", order=-1),  # Should be first
                TemplateSection(title="Last", order=1000),  # Should be last
            ],
        )

        assert extended.sections[0].title == "First"
        assert extended.sections[-1].title == "Last"


class TestListTemplates:
    """Tests for template listing."""

    def test_list_includes_builtins(self):
        """Test that list_templates includes built-in templates."""
        templates = list_templates()

        assert "default" in templates
        assert "compliance" in templates

    def test_list_includes_user_templates(self):
        """Test that list_templates includes user-registered templates."""
        custom = ReportTemplate(name="Test", sections=[])
        register_template("test_list_template", custom)

        try:
            templates = list_templates(include_user=True)
            assert "test_list_template" in templates
        finally:
            unregister_template("test_list_template")

    def test_list_excludes_user_when_requested(self):
        """Test list_templates can exclude user templates."""
        custom = ReportTemplate(name="Test", sections=[])
        register_template("test_list_exclude", custom)

        try:
            templates = list_templates(include_user=False)
            assert "test_list_exclude" not in templates
        finally:
            unregister_template("test_list_exclude")


class TestGetTemplateInfo:
    """Tests for template info retrieval."""

    def test_get_builtin_template_info(self):
        """Test getting info for built-in template."""
        info = get_template_info("default")

        assert info["name"] == "Default Report"
        assert info["source"] == "builtin"
        assert "num_sections" in info
        assert "section_titles" in info

    def test_get_template_info_shows_extends(self):
        """Test template info shows inheritance (REPORT-005)."""
        info = get_template_info("compliance")
        assert info["extends"] == "default"

    def test_get_unknown_template_raises(self):
        """Test getting info for unknown template raises."""
        with pytest.raises(ValueError, match="Unknown template"):
            get_template_info("unknown_xyz")


class TestSaveTemplate:
    """Tests for template saving (REPORT-007)."""

    def test_save_and_reload(self):
        """Test saving template to YAML and reloading."""
        template = create_template(
            name="Save Test",
            sections=[
                TemplateSection(title="Section A", order=0),
                TemplateSection(title="Section B", order=10),
            ],
            description="Test save functionality",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "saved_template.yaml"
            save_template(template, path)

            # Reload and verify
            reloaded = load_template(str(path))
            assert reloaded.name == "Save Test"
            assert len(reloaded.sections) == 2
            assert reloaded.description == "Test save functionality"


class TestCreateTemplate:
    """Tests for template creation helper."""

    def test_create_minimal_template(self):
        """Test creating a minimal template."""
        template = create_template(
            name="Minimal",
            sections=[TemplateSection(title="Content")],
        )

        assert template.name == "Minimal"
        assert len(template.sections) == 1

    def test_create_template_with_inheritance(self):
        """Test creating template that extends another (REPORT-005)."""
        template = create_template(
            name="Extended",
            sections=[TemplateSection(title="Extra")],
            extends="default",
        )

        assert template.extends == "default"


class TestTemplateSection:
    """Tests for TemplateSection dataclass."""

    def test_section_defaults(self):
        """Test section has correct defaults."""
        section = TemplateSection(title="Test")

        assert section.content_type == "text"
        assert section.condition is None
        assert section.template == ""
        assert section.order == 0
        assert section.override is False

    def test_section_override_flag(self):
        """Test section override flag for inheritance (REPORT-005)."""
        section = TemplateSection(
            title="Overriding Section",
            override=True,
        )

        assert section.override is True


class TestInheritanceResolution:
    """Tests for template inheritance chain resolution (REPORT-005)."""

    def setup_method(self):
        """Set up test templates."""
        # Create a chain: grandparent -> parent -> child
        grandparent = ReportTemplate(
            name="Grandparent",
            sections=[
                TemplateSection(title="GP Section", order=0),
            ],
        )
        register_template("test_grandparent", grandparent)

        parent = ReportTemplate(
            name="Parent",
            extends="test_grandparent",
            sections=[
                TemplateSection(title="Parent Section", order=10),
            ],
        )
        register_template("test_parent", parent)

    def teardown_method(self):
        """Clean up test templates."""
        unregister_template("test_child")
        unregister_template("test_parent")
        unregister_template("test_grandparent")

    def test_multi_level_inheritance(self):
        """Test resolving multi-level inheritance chain."""
        child = ReportTemplate(
            name="Child",
            extends="test_parent",
            sections=[
                TemplateSection(title="Child Section", order=20),
            ],
        )
        register_template("test_child", child)

        # Load and resolve
        loaded = load_template("test_child")

        # Should have sections from all three levels
        section_titles = [sec.title for sec in loaded.sections]
        assert "GP Section" in section_titles
        assert "Parent Section" in section_titles
        assert "Child Section" in section_titles

    def test_circular_inheritance_detected(self):
        """Test that circular inheritance is detected and raises."""
        # Create circular reference
        circular_a = ReportTemplate(name="A", extends="test_circular_b", sections=[])
        register_template("test_circular_a", circular_a)

        circular_b = ReportTemplate(name="B", extends="test_circular_a", sections=[])
        register_template("test_circular_b", circular_b)

        try:
            with pytest.raises(ValueError, match=r"Circular.*inheritance"):
                load_template("test_circular_a")
        finally:
            unregister_template("test_circular_a")
            unregister_template("test_circular_b")
