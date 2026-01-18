"""Tests for plugin template generator (PLUG-008).

This module tests the plugin template generation functionality, ensuring
that generated plugins have all necessary files and correct structure.
"""

from pathlib import Path

import pytest

from oscura.extensibility.templates import (
    PluginTemplate,
    generate_plugin_template,
)

pytestmark = pytest.mark.unit


class TestPluginTemplateGeneration:
    """Tests for generate_plugin_template function."""

    def test_generate_decoder_plugin(self, tmp_path: Path) -> None:
        """Test generating a decoder plugin."""
        plugin_dir = tmp_path / "test_decoder"

        result = generate_plugin_template(
            name="test_decoder",
            plugin_type="decoder",
            output_dir=plugin_dir,
            author="Test Author",
            description="Test decoder plugin",
        )

        assert result == plugin_dir
        assert plugin_dir.exists()

        # Check directory structure
        assert (plugin_dir / "__init__.py").exists()
        assert (plugin_dir / "test_decoder.py").exists()
        assert (plugin_dir / "tests").is_dir()
        assert (plugin_dir / "tests" / "__init__.py").exists()
        assert (plugin_dir / "tests" / "test_test_decoder.py").exists()
        assert (plugin_dir / "README.md").exists()
        assert (plugin_dir / "pyproject.toml").exists()

    def test_generate_analyzer_plugin(self, tmp_path: Path) -> None:
        """Test generating an analyzer plugin."""
        plugin_dir = tmp_path / "custom_analyzer"

        result = generate_plugin_template(
            name="custom_analyzer",
            plugin_type="analyzer",
            output_dir=plugin_dir,
        )

        assert result == plugin_dir
        assert (plugin_dir / "custom_analyzer.py").exists()

        # Check analyzer-specific content
        content = (plugin_dir / "custom_analyzer.py").read_text()
        assert "class CustomAnalyzer:" in content
        assert "def analyze(" in content
        assert "sample_rate" in content

    def test_generate_loader_plugin(self, tmp_path: Path) -> None:
        """Test generating a loader plugin."""
        plugin_dir = tmp_path / "custom_loader"

        result = generate_plugin_template(
            name="custom_loader",
            plugin_type="loader",
            output_dir=plugin_dir,
        )

        assert result == plugin_dir
        assert (plugin_dir / "custom_loader.py").exists()

        # Check loader-specific content
        content = (plugin_dir / "custom_loader.py").read_text()
        assert "class CustomLoader:" in content
        assert "def load(" in content
        assert "def can_load(" in content

    def test_generate_exporter_plugin(self, tmp_path: Path) -> None:
        """Test generating an exporter plugin."""
        plugin_dir = tmp_path / "custom_exporter"

        result = generate_plugin_template(
            name="custom_exporter",
            plugin_type="exporter",
            output_dir=plugin_dir,
        )

        assert result == plugin_dir
        assert (plugin_dir / "custom_exporter.py").exists()

        # Check exporter-specific content
        content = (plugin_dir / "custom_exporter.py").read_text()
        assert "class CustomExporter:" in content
        assert "def export(" in content
        assert "def supports_format(" in content

    def test_invalid_plugin_type(self, tmp_path: Path) -> None:
        """Test that invalid plugin types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid plugin_type"):
            generate_plugin_template(
                name="bad_plugin",
                plugin_type="invalid",  # type: ignore
                output_dir=tmp_path / "bad",
            )

    def test_directory_already_exists(self, tmp_path: Path) -> None:
        """Test that existing directories raise FileExistsError."""
        plugin_dir = tmp_path / "existing"
        plugin_dir.mkdir()

        with pytest.raises(FileExistsError, match="already exists"):
            generate_plugin_template(
                name="test_plugin",
                plugin_type="decoder",
                output_dir=plugin_dir,
            )

    def test_default_description(self, tmp_path: Path) -> None:
        """Test auto-generated description when not provided."""
        plugin_dir = tmp_path / "test_plugin"

        generate_plugin_template(
            name="test_plugin",
            plugin_type="analyzer",
            output_dir=plugin_dir,
        )

        init_content = (plugin_dir / "__init__.py").read_text()
        assert "Custom analyzer plugin for Oscura" in init_content

    def test_custom_version(self, tmp_path: Path) -> None:
        """Test custom version in generated plugin."""
        plugin_dir = tmp_path / "versioned_plugin"

        generate_plugin_template(
            name="versioned_plugin",
            plugin_type="decoder",
            output_dir=plugin_dir,
            version="2.5.1",
        )

        init_content = (plugin_dir / "__init__.py").read_text()
        assert '__version__ = "2.5.1"' in init_content

        pyproject = (plugin_dir / "pyproject.toml").read_text()
        assert 'version = "2.5.1"' in pyproject


class TestGeneratedPluginStructure:
    """Tests for the structure of generated plugins."""

    def test_init_py_contains_metadata(self, tmp_path: Path) -> None:
        """Test that __init__.py contains required metadata."""
        plugin_dir = tmp_path / "metadata_test"

        generate_plugin_template(
            name="metadata_test",
            plugin_type="decoder",
            output_dir=plugin_dir,
            author="Jane Doe",
            description="Test metadata plugin",
        )

        init_content = (plugin_dir / "__init__.py").read_text()

        # Check metadata fields
        assert "Jane Doe" in init_content
        assert "Test metadata plugin" in init_content
        assert "metadata_test" in init_content
        assert "decoder" in init_content
        assert "__version__" in init_content
        assert "__all__" in init_content

    def test_main_module_has_docstring(self, tmp_path: Path) -> None:
        """Test that main module has proper docstring."""
        plugin_dir = tmp_path / "docstring_test"

        generate_plugin_template(
            name="docstring_test",
            plugin_type="analyzer",
            output_dir=plugin_dir,
        )

        module_content = (plugin_dir / "docstring_test.py").read_text()

        # Check for module-level docstring
        assert '"""' in module_content
        assert "PLUG-008" in module_content

    def test_pyproject_toml_has_entry_point(self, tmp_path: Path) -> None:
        """Test that pyproject.toml has correct entry point."""
        plugin_dir = tmp_path / "entrypoint_test"

        generate_plugin_template(
            name="entrypoint_test",
            plugin_type="decoder",
            output_dir=plugin_dir,
        )

        pyproject = (plugin_dir / "pyproject.toml").read_text()

        # Check entry point configuration
        assert "[project.entry-points" in pyproject
        assert "oscura.decoders" in pyproject
        assert "entrypoint_test" in pyproject
        assert "EntrypointTest" in pyproject  # PascalCase class name

    def test_readme_has_usage_instructions(self, tmp_path: Path) -> None:
        """Test that README has usage instructions."""
        plugin_dir = tmp_path / "readme_test"

        generate_plugin_template(
            name="readme_test",
            plugin_type="loader",
            output_dir=plugin_dir,
            author="Bob Smith",
        )

        readme = (plugin_dir / "README.md").read_text()

        # Check README sections
        assert "# ReadmeTest" in readme  # Title with PascalCase
        assert "## Installation" in readme
        assert "## Usage" in readme
        assert "## Development" in readme
        assert "pip install -e ." in readme
        assert "pytest tests/" in readme
        assert "Bob Smith" in readme

    def test_test_module_has_examples(self, tmp_path: Path) -> None:
        """Test that test module has example tests."""
        plugin_dir = tmp_path / "test_example"

        generate_plugin_template(
            name="test_example",
            plugin_type="decoder",
            output_dir=plugin_dir,
        )

        test_content = (plugin_dir / "tests" / "test_test_example.py").read_text()

        # Check for example tests
        assert "def test_test_example_initialization():" in test_content
        assert "def test_test_example_basic_functionality():" in test_content
        assert "def test_test_example_error_handling():" in test_content
        assert "@pytest.mark.parametrize" in test_content
        assert "import pytest" in test_content


class TestPluginTypeSpecifics:
    """Tests for plugin type-specific code generation."""

    def test_decoder_has_decode_method(self, tmp_path: Path) -> None:
        """Test decoder plugin has decode() method."""
        plugin_dir = tmp_path / "decoder"

        generate_plugin_template(
            name="decoder_plugin",
            plugin_type="decoder",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "decoder_plugin.py").read_text()
        assert "def decode(" in content
        assert "NDArray[np.uint8]" in content
        assert "list[dict[str, object]]" in content

    def test_analyzer_has_analyze_method(self, tmp_path: Path) -> None:
        """Test analyzer plugin has analyze() method."""
        plugin_dir = tmp_path / "analyzer"

        generate_plugin_template(
            name="analyzer_plugin",
            plugin_type="analyzer",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "analyzer_plugin.py").read_text()
        assert "def analyze(" in content
        assert "sample_rate" in content
        assert "dict[str, object]" in content

    def test_loader_has_load_and_can_load(self, tmp_path: Path) -> None:
        """Test loader plugin has load() and can_load() methods."""
        plugin_dir = tmp_path / "loader"

        generate_plugin_template(
            name="loader_plugin",
            plugin_type="loader",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "loader_plugin.py").read_text()
        assert "def load(" in content
        assert "def can_load(" in content
        assert "Path" in content
        assert "FileNotFoundError" in content

    def test_exporter_has_export_method(self, tmp_path: Path) -> None:
        """Test exporter plugin has export() method."""
        plugin_dir = tmp_path / "exporter"

        generate_plugin_template(
            name="exporter_plugin",
            plugin_type="exporter",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "exporter_plugin.py").read_text()
        assert "def export(" in content
        assert "dict[str, NDArray[np.float64]]" in content
        assert "output_path" in content


class TestPluginTemplate:
    """Tests for PluginTemplate dataclass."""

    def test_plugin_template_creation(self) -> None:
        """Test PluginTemplate can be created."""
        template = PluginTemplate(
            name="test_plugin",
            plugin_type="decoder",
            output_dir=Path("/tmp/test"),
            author="Test Author",
            description="Test description",
            version="1.0.0",
        )

        assert template.name == "test_plugin"
        assert template.plugin_type == "decoder"
        assert template.output_dir == Path("/tmp/test")
        assert template.author == "Test Author"
        assert template.description == "Test description"
        assert template.version == "1.0.0"

    def test_plugin_template_defaults(self) -> None:
        """Test PluginTemplate default values."""
        template = PluginTemplate(
            name="minimal",
            plugin_type="analyzer",
            output_dir=Path("/tmp/minimal"),
        )

        assert template.author == "Plugin Author"
        assert template.description == "Custom Oscura plugin"
        assert template.version == "0.1.0"


class TestClassNameGeneration:
    """Tests for class name generation from plugin names."""

    def test_snake_case_to_pascal_case(self, tmp_path: Path) -> None:
        """Test snake_case plugin name converts to PascalCase class."""
        plugin_dir = tmp_path / "my_custom_plugin"

        generate_plugin_template(
            name="my_custom_plugin",
            plugin_type="decoder",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "my_custom_plugin.py").read_text()
        assert "class MyCustomPlugin:" in content

    def test_single_word_name(self, tmp_path: Path) -> None:
        """Test single word plugin name."""
        plugin_dir = tmp_path / "flexray"

        generate_plugin_template(
            name="flexray",
            plugin_type="decoder",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "flexray.py").read_text()
        assert "class Flexray:" in content


class TestEntryPointGroups:
    """Tests for correct entry point group assignment."""

    @pytest.mark.parametrize(
        "plugin_type,expected_group",
        [
            ("decoder", "oscura.decoders"),
            ("analyzer", "oscura.analyzers"),
            ("loader", "oscura.loaders"),
            ("exporter", "oscura.exporters"),
        ],
    )
    def test_entry_point_group_mapping(
        self,
        plugin_type: str,
        expected_group: str,
        tmp_path: Path,
    ) -> None:
        """Test each plugin type gets correct entry point group."""
        plugin_dir = tmp_path / f"test_{plugin_type}"

        generate_plugin_template(
            name=f"test_{plugin_type}",
            plugin_type=plugin_type,  # type: ignore
            output_dir=plugin_dir,
        )

        pyproject = (plugin_dir / "pyproject.toml").read_text()
        assert expected_group in pyproject


class TestGeneratedCodeQuality:
    """Tests for quality of generated code."""

    def test_generated_code_has_type_hints(self, tmp_path: Path) -> None:
        """Test generated code includes type hints."""
        plugin_dir = tmp_path / "typed_plugin"

        generate_plugin_template(
            name="typed_plugin",
            plugin_type="decoder",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "typed_plugin.py").read_text()

        # Check for type hints
        assert "from __future__ import annotations" in content
        assert "NDArray" in content
        assert "->" in content  # Return type annotations

    def test_generated_code_has_error_handling(self, tmp_path: Path) -> None:
        """Test generated code includes error handling."""
        plugin_dir = tmp_path / "error_plugin"

        generate_plugin_template(
            name="error_plugin",
            plugin_type="decoder",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "error_plugin.py").read_text()

        # Check for error handling
        assert "ValueError" in content
        assert "if len(signal) == 0:" in content

    def test_generated_code_has_examples(self, tmp_path: Path) -> None:
        """Test generated code includes usage examples."""
        plugin_dir = tmp_path / "example_plugin"

        generate_plugin_template(
            name="example_plugin",
            plugin_type="analyzer",
            output_dir=plugin_dir,
        )

        content = (plugin_dir / "example_plugin.py").read_text()

        # Check for examples in docstrings
        assert "Example:" in content
        assert ">>>" in content


class TestExtensibilityTemplatesIntegration:
    """Integration tests for complete plugin generation."""

    def test_complete_plugin_structure(self, tmp_path: Path) -> None:
        """Test that complete plugin has all required components."""
        plugin_dir = tmp_path / "complete_plugin"

        generate_plugin_template(
            name="complete_plugin",
            plugin_type="decoder",
            output_dir=plugin_dir,
            author="Integration Tester",
            description="Complete integration test plugin",
            version="3.2.1",
        )

        # Verify all files exist
        assert (plugin_dir / "__init__.py").exists()
        assert (plugin_dir / "complete_plugin.py").exists()
        assert (plugin_dir / "tests" / "__init__.py").exists()
        assert (plugin_dir / "tests" / "test_complete_plugin.py").exists()
        assert (plugin_dir / "README.md").exists()
        assert (plugin_dir / "pyproject.toml").exists()

        # Verify __init__.py
        init_content = (plugin_dir / "__init__.py").read_text()
        assert "Integration Tester" in init_content
        assert "Complete integration test plugin" in init_content
        assert '__version__ = "3.2.1"' in init_content

        # Verify main module
        main_content = (plugin_dir / "complete_plugin.py").read_text()
        assert "class CompletePlugin:" in main_content
        assert "def decode(" in main_content

        # Verify test module
        test_content = (plugin_dir / "tests" / "test_complete_plugin.py").read_text()
        assert "def test_complete_plugin_initialization():" in test_content

        # Verify README
        readme = (plugin_dir / "README.md").read_text()
        assert "# CompletePlugin" in readme
        assert "Integration Tester" in readme

        # Verify pyproject.toml
        pyproject = (plugin_dir / "pyproject.toml").read_text()
        assert 'name = "complete_plugin"' in pyproject
        assert 'version = "3.2.1"' in pyproject
        assert "oscura.decoders" in pyproject

    def test_generated_code_syntax_valid(self, tmp_path: Path) -> None:
        """Test that generated code has valid Python syntax (PLUG-008 acceptance)."""
        import py_compile
        import tempfile

        for plugin_type in ["decoder", "analyzer", "loader", "exporter"]:
            plugin_dir = tmp_path / f"syntax_test_{plugin_type}"

            generate_plugin_template(
                name=f"syntax_test_{plugin_type}",
                plugin_type=plugin_type,  # type: ignore
                output_dir=plugin_dir,
                author="Syntax Tester",
                description=f"Syntax validation test for {plugin_type}",
            )

            # Verify all Python files compile successfully
            python_files = list(plugin_dir.rglob("*.py"))
            assert len(python_files) >= 4, (
                f"Should have at least 4 .py files, got {len(python_files)}"
            )

            for py_file in python_files:
                # Use py_compile to check syntax
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
                        py_compile.compile(str(py_file), cfile=tmp.name, doraise=True)
                except py_compile.PyCompileError as e:
                    pytest.fail(f"Syntax error in {py_file}: {e}")

    def test_generated_plugins_have_required_structure(self, tmp_path: Path) -> None:
        """Test all generated plugins match the required structure from PLUG-008."""
        expected_structure = [
            "__init__.py",
            "{name}.py",
            "tests/__init__.py",
            "tests/test_{name}.py",
            "README.md",
            "pyproject.toml",
        ]

        for plugin_type in ["decoder", "analyzer", "loader", "exporter"]:
            plugin_name = f"struct_test_{plugin_type}"
            plugin_dir = tmp_path / plugin_name

            generate_plugin_template(
                name=plugin_name,
                plugin_type=plugin_type,  # type: ignore
                output_dir=plugin_dir,
            )

            # Check each required file exists
            for file_pattern in expected_structure:
                file_path = file_pattern.format(name=plugin_name)
                full_path = plugin_dir / file_path
                assert full_path.exists(), (
                    f"Missing required file {file_path} in {plugin_type} plugin. "
                    f"Expected structure:\n  " + "\n  ".join(expected_structure)
                )
