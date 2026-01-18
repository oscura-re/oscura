import pytest

"""Tests for extension validation system.

Tests validation of extensions, decoders, and hooks.
"""

from oscura.extensibility.validation import (
    ValidationResult,
    validate_decoder_interface,
    validate_hook_function,
)

pytestmark = pytest.mark.unit


class TestValidationResult:
    """Test ValidationResult class."""

    def test_initialization(self):
        """Test validation result initialization."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.info) == 0

    def test_add_error(self):
        """Test adding error invalidates result."""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].message == "Test error"
        assert result.errors[0].severity == "error"

    def test_add_warning(self):
        """Test adding warning doesn't invalidate result."""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].message == "Test warning"
        assert result.warnings[0].severity == "warning"

    def test_add_info(self):
        """Test adding info message."""
        result = ValidationResult()
        result.add_info("Test info")

        assert result.is_valid is True
        assert len(result.info) == 1
        assert result.info[0].message == "Test info"
        assert result.info[0].severity == "info"

    def test_all_issues(self):
        """Test getting all issues."""
        result = ValidationResult()
        result.add_error("Error")
        result.add_warning("Warning")
        result.add_info("Info")

        all_issues = result.all_issues
        assert len(all_issues) == 3
        assert all_issues[0].severity == "error"
        assert all_issues[1].severity == "warning"
        assert all_issues[2].severity == "info"


class TestDecoderInterfaceValidation:
    """Test decoder interface validation."""

    def test_valid_decoder(self):
        """Test validation of valid decoder class."""

        class ValidDecoder:
            """Valid decoder implementation."""

            def decode(self, trace):
                """Decode trace."""
                return []

            def get_metadata(self):
                """Get metadata."""
                return {"name": "valid"}

        result = validate_decoder_interface(ValidDecoder)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_decoder_missing_required_method(self):
        """Test decoder missing required method fails validation."""

        class InvalidDecoder:
            """Decoder missing required method."""

            def decode(self, trace):
                """Decode trace."""
                return []

            # Missing get_metadata

        result = validate_decoder_interface(InvalidDecoder)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "get_metadata" in result.errors[0].message

    def test_decoder_with_optional_methods(self):
        """Test decoder with optional methods validates successfully."""

        class DecoderWithOptional:
            """Decoder with optional methods."""

            def decode(self, trace):
                """Decode trace."""
                return []

            def get_metadata(self):
                """Get metadata."""
                return {}

            def configure(self):
                """Configure decoder."""

            def reset(self):
                """Reset decoder."""

        result = validate_decoder_interface(DecoderWithOptional)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_decoder_non_callable_method(self):
        """Test decoder with non-callable method fails validation."""

        class InvalidDecoder:
            """Decoder with non-callable attribute."""

            decode = "not a function"  # Not callable

            def get_metadata(self):
                """Get metadata."""
                return {}

        result = validate_decoder_interface(InvalidDecoder)
        assert result.is_valid is False
        assert any("not callable" in err.message for err in result.errors)

    def test_metadata_extraction(self):
        """Test metadata is extracted from decoder class."""

        class TestDecoder:
            """Test decoder."""

            def decode(self, trace):
                """Decode trace."""
                return []

            def get_metadata(self):
                """Get metadata."""
                return {}

        result = validate_decoder_interface(TestDecoder)
        assert "class_name" in result.metadata
        assert result.metadata["class_name"] == "TestDecoder"
        assert "required_methods" in result.metadata
        assert "decode" in result.metadata["required_methods"]


class TestHookFunctionValidation:
    """Test hook function validation."""

    def test_valid_hook(self):
        """Test validation of valid hook function."""

        def valid_hook(context):
            """Valid hook."""
            return context

        result = validate_hook_function(valid_hook)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_non_callable(self):
        """Test validation fails for non-callable."""
        result = validate_hook_function("not a function")
        assert result.is_valid is False
        assert any("callable" in err.message.lower() for err in result.errors)

    def test_hook_no_parameters(self):
        """Test validation fails for hook with no parameters."""

        def invalid_hook():
            """Hook with no parameters."""

        result = validate_hook_function(invalid_hook)
        assert result.is_valid is False
        assert any("parameter" in err.message.lower() for err in result.errors)

    def test_hook_without_docstring(self):
        """Test warning for hook without docstring."""

        def hook_no_doc(context):
            return context

        result = validate_hook_function(hook_no_doc)
        # Should pass but with warning
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("docstring" in warn.message.lower() for warn in result.warnings)

    def test_hook_with_docstring(self):
        """Test hook with docstring validates cleanly."""

        def hook_with_doc(context):
            """Hook with documentation."""
            return context

        result = validate_hook_function(hook_with_doc)
        assert result.is_valid is True
        # Should not have docstring warning
        assert not any("docstring" in warn.message.lower() for warn in result.warnings)

    def test_metadata_extraction(self):
        """Test metadata is extracted from hook function."""

        def test_hook(context, extra=None):
            """Test hook."""
            return context

        result = validate_hook_function(test_hook)
        assert "name" in result.metadata
        assert result.metadata["name"] == "test_hook"
        assert "params" in result.metadata
        assert "context" in result.metadata["params"]


class TestExtensionValidation:
    """Test full extension validation.

    Note: These tests require file I/O and are more integration-like.
    """

    def test_nonexistent_path(self, tmp_path):
        """Test validation of nonexistent path."""
        from oscura.extensibility.validation import validate_extension

        nonexistent = tmp_path / "does_not_exist"
        result = validate_extension(nonexistent)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "not exist" in result.errors[0].message

    def test_path_is_file(self, tmp_path):
        """Test validation fails when path is a file."""
        from oscura.extensibility.validation import validate_extension

        file_path = tmp_path / "test.py"
        file_path.write_text("# test file")

        result = validate_extension(file_path)

        assert result.is_valid is False
        assert any("not a directory" in err.message for err in result.errors)

    def test_empty_directory(self, tmp_path):
        """Test validation of empty extension directory."""
        from oscura.extensibility.validation import validate_extension

        empty_dir = tmp_path / "empty_plugin"
        empty_dir.mkdir()

        result = validate_extension(empty_dir, check_dependencies=False, check_security=False)

        # Should have warnings/errors about missing metadata
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_valid_extension_structure(self, tmp_path):
        """Test validation of properly structured extension."""
        from oscura.extensibility.validation import validate_extension

        ext_dir = tmp_path / "my_plugin"
        ext_dir.mkdir()

        # Create basic structure
        (ext_dir / "__init__.py").write_text('"""My plugin."""\n')
        (ext_dir / "plugin.py").write_text(
            """\"\"\"Plugin implementation.\"\"\"

class MyPlugin:
    \"\"\"Plugin class.\"\"\"

    def process(self):
        \"\"\"Process data.\"\"\"
        return {}
"""
        )

        # Create pyproject.toml
        (ext_dir / "pyproject.toml").write_text(
            """[project]
name = "my_plugin"
version = "1.0.0"
description = "Test plugin"
"""
        )

        result = validate_extension(ext_dir, check_dependencies=False, check_security=False)

        # Should validate structure successfully
        assert "name" in result.metadata
        assert result.metadata["name"] == "my_plugin"


class TestSecurityChecks:
    """Test security validation."""

    def test_security_check_dangerous_imports(self, tmp_path):
        """Test security check detects dangerous imports."""
        from oscura.extensibility.validation import validate_extension

        ext_dir = tmp_path / "unsafe_plugin"
        ext_dir.mkdir()

        # Create file with unsafe import
        (ext_dir / "unsafe.py").write_text(
            """\"\"\"Unsafe module.\"\"\"
import pickle

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
"""
        )

        (ext_dir / "pyproject.toml").write_text(
            """[project]
name = "unsafe_plugin"
version = "1.0.0"
description = "Unsafe plugin"
"""
        )

        result = validate_extension(ext_dir, check_security=True, check_dependencies=False)

        # Should have security warnings
        assert len(result.warnings) > 0
        assert any("unsafe" in warn.message.lower() for warn in result.warnings)
