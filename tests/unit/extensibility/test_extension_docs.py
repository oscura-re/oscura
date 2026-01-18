import pytest

"""Tests for extension documentation auto-generation.

Tests documentation extraction and generation for extensions.
"""

from oscura.extensibility.docs import (
    ClassDoc,
    ExtensionDocs,
    FunctionDoc,
    ModuleDoc,
    generate_decoder_docs,
)

pytestmark = pytest.mark.unit


class TestFunctionDoc:
    """Test FunctionDoc dataclass."""

    def test_initialization(self):
        """Test function doc initialization."""
        func_doc = FunctionDoc(name="test_func")
        assert func_doc.name == "test_func"
        assert func_doc.signature == ""
        assert func_doc.docstring == ""
        assert len(func_doc.parameters) == 0
        assert len(func_doc.examples) == 0

    def test_with_parameters(self):
        """Test function doc with parameters."""
        func_doc = FunctionDoc(
            name="process",
            parameters=[("data", "Input data"), ("rate", "Sample rate")],
        )
        assert len(func_doc.parameters) == 2
        assert func_doc.parameters[0] == ("data", "Input data")


class TestClassDoc:
    """Test ClassDoc dataclass."""

    def test_initialization(self):
        """Test class doc initialization."""
        class_doc = ClassDoc(name="MyClass")
        assert class_doc.name == "MyClass"
        assert class_doc.docstring == ""
        assert len(class_doc.methods) == 0
        assert len(class_doc.attributes) == 0
        assert len(class_doc.bases) == 0

    def test_with_methods(self):
        """Test class doc with methods."""
        method = FunctionDoc(name="process")
        class_doc = ClassDoc(name="MyClass", methods=[method])
        assert len(class_doc.methods) == 1
        assert class_doc.methods[0].name == "process"


class TestModuleDoc:
    """Test ModuleDoc dataclass."""

    def test_initialization(self):
        """Test module doc initialization."""
        module_doc = ModuleDoc(name="my_module")
        assert module_doc.name == "my_module"
        assert module_doc.docstring == ""
        assert len(module_doc.classes) == 0
        assert len(module_doc.functions) == 0


class TestExtensionDocs:
    """Test ExtensionDocs dataclass."""

    def test_initialization(self):
        """Test extension docs initialization."""
        docs = ExtensionDocs(name="my_extension")
        assert docs.name == "my_extension"
        assert docs.version == "0.1.0"
        assert docs.description == ""
        assert len(docs.modules) == 0
        assert docs.markdown == ""


class TestDecoderDocsGeneration:
    """Test decoder documentation generation."""

    def test_simple_decoder(self):
        """Test generating docs for simple decoder."""

        class SimpleDecoder:
            """Simple UART decoder.

            Example:
                >>> decoder = SimpleDecoder()
                >>> frames = decoder.decode(signal)
            """

            def decode(self, signal):
                """Decode signal.

                Args:
                    signal: Input signal

                Returns:
                    List of decoded frames
                """
                return []

        docs = generate_decoder_docs(SimpleDecoder)

        assert "SimpleDecoder" in docs
        assert "Simple UART decoder" in docs
        assert "decode" in docs

    def test_decoder_with_methods(self):
        """Test decoder with multiple methods."""

        class FullDecoder:
            """Full decoder implementation."""

            def decode(self, signal):
                """Decode signal."""
                return []

            def get_metadata(self):
                """Get metadata."""
                return {"name": "full"}

            def configure(self):
                """Configure decoder."""

        docs = generate_decoder_docs(FullDecoder)

        assert "decode" in docs
        assert "get_metadata" in docs
        assert "configure" in docs

    def test_decoder_without_examples(self):
        """Test decoder docs without examples."""

        class NoExampleDecoder:
            """Decoder without examples."""

            def decode(self, signal):
                """Decode signal."""
                return []

            def get_metadata(self):
                """Get metadata."""
                return {}

        docs = generate_decoder_docs(NoExampleDecoder, include_examples=False)

        assert "NoExampleDecoder" in docs
        # Should not fail even without examples


class TestMetadataExtraction:
    """Test plugin metadata extraction."""

    def test_extract_from_pyproject(self, tmp_path):
        """Test extracting metadata from pyproject.toml."""
        from oscura.extensibility.docs import extract_plugin_metadata

        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()

        pyproject = plugin_dir / "pyproject.toml"
        pyproject.write_text(
            """[project]
name = "test_plugin"
version = "1.2.3"
description = "Test plugin for Oscura"
authors = [
    {name = "John Doe"}
]
dependencies = [
    "oscura>=0.1.0",
    "numpy>=1.26.0"
]
"""
        )

        metadata = extract_plugin_metadata(plugin_dir)

        assert metadata["name"] == "test_plugin"
        assert metadata["version"] == "1.2.3"
        assert metadata["description"] == "Test plugin for Oscura"
        assert len(metadata["authors"]) == 1
        assert metadata["authors"][0]["name"] == "John Doe"
        assert "oscura>=0.1.0" in metadata["dependencies"]

    def test_extract_missing_metadata(self, tmp_path):
        """Test extracting metadata from empty directory."""
        from oscura.extensibility.docs import extract_plugin_metadata

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        metadata = extract_plugin_metadata(empty_dir)

        # Should return empty dict for missing files
        assert isinstance(metadata, dict)


class TestExtensionDocsGeneration:
    """Test full extension documentation generation."""

    def test_generate_for_simple_extension(self, tmp_path):
        """Test generating docs for simple extension."""
        from oscura.extensibility.docs import generate_extension_docs

        ext_dir = tmp_path / "simple_ext"
        ext_dir.mkdir()

        # Create pyproject.toml
        (ext_dir / "pyproject.toml").write_text(
            """[project]
name = "simple_ext"
version = "1.0.0"
description = "Simple extension"
"""
        )

        # Create module
        (ext_dir / "simple.py").write_text(
            '''"""Simple module."""
def process():
    """Process data."""
    return True
'''
        )

        docs = generate_extension_docs(ext_dir, output_format="markdown")

        assert docs.name == "simple_ext"
        assert docs.version == "1.0.0"
        assert docs.markdown != ""
        assert "simple_ext" in docs.markdown

    def test_generate_html_format(self, tmp_path):
        """Test generating HTML documentation."""
        from oscura.extensibility.docs import generate_extension_docs

        ext_dir = tmp_path / "html_ext"
        ext_dir.mkdir()

        (ext_dir / "pyproject.toml").write_text(
            """[project]
name = "html_ext"
version = "1.0.0"
description = "HTML test"
"""
        )

        docs = generate_extension_docs(ext_dir, output_format="html")

        assert docs.html != ""
        assert "<!DOCTYPE html>" in docs.html
        assert "html_ext" in docs.html

    def test_generate_with_class(self, tmp_path):
        """Test generating docs for extension with class."""
        from oscura.extensibility.docs import generate_extension_docs

        ext_dir = tmp_path / "class_ext"
        ext_dir.mkdir()

        (ext_dir / "pyproject.toml").write_text(
            """[project]
name = "class_ext"
version = "1.0.0"
description = "Extension with class"
"""
        )

        (ext_dir / "decoder.py").write_text(
            '''"""Decoder module."""
class MyDecoder:
    """Custom decoder.

    Attributes:
        sample_rate: Sample rate in Hz
    """
    def __init__(self):
        """Initialize decoder."""
        self.sample_rate = 1000000

    def decode(self, signal):
        """Decode signal.

        Args:
            signal: Input signal array

        Returns:
            List of decoded frames
        """
        return []
'''
        )

        docs = generate_extension_docs(ext_dir, output_format="markdown")

        assert "MyDecoder" in docs.markdown
        assert "decode" in docs.markdown
        assert len(docs.modules) > 0


class TestDocstringParsing:
    """Test Google-style docstring parsing."""

    def test_parse_args_section(self):
        """Test parsing Args section."""
        from oscura.extensibility.docs import generate_decoder_docs

        class TestClass:
            """Test class."""

            def method(self, arg1, arg2):
                """Test method.

                Args:
                    arg1: First argument
                    arg2: Second argument

                Returns:
                    Result value
                """

        docs = generate_decoder_docs(TestClass)
        # Args should be documented
        assert "arg1" in docs or "First argument" in docs

    def test_parse_examples_section(self):
        """Test parsing Examples section."""
        from oscura.extensibility.docs import generate_decoder_docs

        class TestClass:
            """Test class."""

            def method(self):
                """Test method.

                Example:
                    >>> obj = TestClass()
                    >>> obj.method()
                    True
                """

        docs = generate_decoder_docs(TestClass, include_examples=True)
        # Example should be included
        assert ">>>" in docs or "Example" in docs
