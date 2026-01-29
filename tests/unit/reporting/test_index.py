"""Tests for comprehensive analysis report index generation.

Tests verify the TemplateEngine and IndexGenerator correctly render
HTML and Markdown index files from analysis results.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oscura.reporting.config import AnalysisDomain, InputType
from oscura.reporting.index import IndexGenerator, TemplateEngine
from oscura.reporting.output import OutputManager

pytestmark = pytest.mark.unit


class TestTemplateEngine:
    """Tests for TemplateEngine class."""

    @pytest.fixture
    def engine(self):
        """Create template engine instance."""
        return TemplateEngine()

    def test_simple_variable_substitution(self, engine):
        """Test simple variable replacement."""
        template = "Hello {{name}}!"
        context = {"name": "World"}
        result = engine.render(template, context)
        assert result == "Hello World!"

    def test_multiple_variables(self, engine):
        """Test multiple variable replacements."""
        template = "{{greeting}} {{name}}, welcome to {{place}}."
        context = {"greeting": "Hello", "name": "User", "place": "Oscura"}
        result = engine.render(template, context)
        assert result == "Hello User, welcome to Oscura."

    def test_missing_variable_empty_string(self, engine):
        """Test missing variables render as empty string."""
        template = "Value: {{missing}}"
        context = {}
        result = engine.render(template, context)
        assert result == "Value: "

    def test_if_block_true(self, engine):
        """Test if block when condition is true."""
        template = "{{#if show}}visible{{/if}}"
        context = {"show": True}
        result = engine.render(template, context)
        assert result == "visible"

    def test_if_block_false(self, engine):
        """Test if block when condition is false."""
        template = "{{#if show}}visible{{/if}}"
        context = {"show": False}
        result = engine.render(template, context)
        assert result == ""

    def test_if_block_with_truthy_value(self, engine):
        """Test if block with truthy non-boolean values."""
        template = "{{#if items}}has items{{/if}}"
        context = {"items": [1, 2, 3]}
        result = engine.render(template, context)
        assert result == "has items"

    def test_if_block_with_empty_list(self, engine):
        """Test if block with empty list (falsy)."""
        template = "{{#if items}}has items{{/if}}"
        context = {"items": []}
        result = engine.render(template, context)
        assert result == ""

    def test_each_block_simple(self, engine):
        """Test each block with simple list."""
        template = "{{#each items}}{{this}} {{/each}}"
        context = {"items": ["a", "b", "c"]}
        result = engine.render(template, context)
        assert result == "a b c "

    def test_each_block_with_dict_items(self, engine):
        """Test each block with dictionary items."""
        template = "{{#each items}}{{name}}: {{value}} {{/each}}"
        context = {
            "items": [
                {"name": "x", "value": 1},
                {"name": "y", "value": 2},
            ]
        }
        result = engine.render(template, context)
        assert result == "x: 1 y: 2 "

    def test_each_block_empty_list(self, engine):
        """Test each block with empty list."""
        template = "{{#each items}}item{{/each}}"
        context = {"items": []}
        result = engine.render(template, context)
        assert result == ""

    def test_nested_property_access(self, engine):
        """Test accessing nested properties."""
        template = "{{data.nested.value}}"
        context = {"data": {"nested": {"value": 42}}}
        result = engine.render(template, context)
        assert result == "42"

    def test_combined_blocks(self, engine):
        """Test combining if and each blocks."""
        template = "{{#if show}}{{#each items}}{{this}}{{/each}}{{/if}}"
        context = {"show": True, "items": [1, 2, 3]}
        result = engine.render(template, context)
        assert result == "123"


class TestIndexGenerator:
    """Tests for IndexGenerator class."""

    @pytest.fixture
    def output_manager(self):
        """Create output manager with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(Path(tmpdir), "test", datetime(2026, 1, 1, 12, 0, 0))
            manager.create()
            yield manager

    @pytest.fixture
    def mock_result(self, output_manager):
        """Create mock AnalysisResult."""
        result = MagicMock()
        result.output_dir = output_manager.root
        result.input_file = "/data/test.wfm"
        result.input_type = InputType.WAVEFORM
        result.duration_seconds = 5.5
        result.total_analyses = 10
        result.successful_analyses = 9
        result.failed_analyses = 1
        result.domain_summaries = {
            AnalysisDomain.SPECTRAL: {
                "analyses_count": 5,
                "plots": [],
                "data_files": [],
            },
            AnalysisDomain.WAVEFORM: {
                "analyses_count": 5,
                "plots": [],
                "data_files": [],
            },
        }
        result.errors = []
        return result

    def test_generates_html_index(self, output_manager, mock_result):
        """Test HTML index generation."""
        generator = IndexGenerator(output_manager)

        # Need to create templates directory with templates
        templates_dir = Path(__file__).parents[3] / "src/oscura/reporting/templates"
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        paths = generator.generate(mock_result, ["html"])

        assert "html" in paths
        assert paths["html"].exists()
        assert paths["html"].suffix == ".html"

    def test_generates_markdown_index(self, output_manager, mock_result):
        """Test Markdown index generation."""
        generator = IndexGenerator(output_manager)

        templates_dir = Path(__file__).parents[3] / "src/oscura/reporting/templates"
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        paths = generator.generate(mock_result, ["md"])

        assert "md" in paths
        assert paths["md"].exists()
        assert paths["md"].suffix == ".md"

    def test_generates_both_formats(self, output_manager, mock_result):
        """Test generating both HTML and Markdown."""
        generator = IndexGenerator(output_manager)

        templates_dir = Path(__file__).parents[3] / "src/oscura/reporting/templates"
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        paths = generator.generate(mock_result, ["html", "md"])

        assert len(paths) == 2
        assert "html" in paths
        assert "md" in paths

    def test_format_duration(self, output_manager):
        """Test duration formatting."""
        generator = IndexGenerator(output_manager)

        assert generator._format_duration(30.5) == "30.5s"
        assert generator._format_duration(90) == "1m 30s"
        assert generator._format_duration(3720) == "1h 2m"

    def test_format_size(self, output_manager):
        """Test file size formatting."""
        generator = IndexGenerator(output_manager)

        # Test with None/missing file
        assert generator._format_size(None) == "N/A"
        assert generator._format_size("/nonexistent/file") == "N/A"


class TestIndexGeneratorContext:
    """Tests for IndexGenerator context building."""

    @pytest.fixture
    def output_manager(self):
        """Create output manager with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(Path(tmpdir), "test", datetime(2026, 1, 1, 12, 0, 0))
            manager.create()
            yield manager

    def test_builds_context_from_result(self, output_manager):
        """Test context building from AnalysisResult."""
        result = MagicMock()
        result.output_dir = output_manager.root
        result.input_file = "/data/signal.wfm"
        result.input_type = InputType.WAVEFORM
        result.duration_seconds = 10.0
        result.total_analyses = 5
        result.successful_analyses = 4
        result.failed_analyses = 1
        result.domain_summaries = {
            AnalysisDomain.SPECTRAL: {"analyses_count": 3},
        }
        result.errors = []

        generator = IndexGenerator(output_manager)
        context = generator._build_context(result)

        assert context["input_name"] == "/data/signal.wfm"
        assert context["input_type"] == "waveform"
        assert context["total_analyses"] == 5
        assert context["successful"] == 4
        assert context["failed"] == 1
        assert context["has_errors"] is False

    def test_includes_error_info(self, output_manager):
        """Test context includes error information."""
        result = MagicMock()
        result.output_dir = output_manager.root
        result.input_file = None
        result.input_type = InputType.WAVEFORM
        result.duration_seconds = 1.0
        result.total_analyses = 1
        result.successful_analyses = 0
        result.failed_analyses = 1
        result.domain_summaries = {}

        error = MagicMock()
        error.domain = AnalysisDomain.SPECTRAL
        error.function = "compute_fft"
        error.error_message = "Failed"
        result.errors = [error]

        generator = IndexGenerator(output_manager)
        context = generator._build_context(result)

        assert context["has_errors"] is True
        assert len(context["errors"]) == 1
        assert context["errors"][0]["domain"] == "spectral"
