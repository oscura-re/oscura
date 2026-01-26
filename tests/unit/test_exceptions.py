"""Comprehensive tests for oscura/exceptions.py (deprecated module).

This test module verifies the deprecated exception re-export module emits
proper deprecation warnings while maintaining backward compatibility.

Tests cover:
- Deprecation warning on import
- All exceptions properly re-exported
- Exception inheritance chain preserved
- Error messages and attributes work correctly
"""

import warnings

import pytest


class TestDeprecatedExceptionsModule:
    """Test suite for deprecated exceptions.py module.

    Validates that the deprecated module:
    - Emits DeprecationWarning on import
    - Re-exports all exceptions correctly
    - Maintains backward compatibility
    """

    def test_deprecation_warning_emitted(self) -> None:
        """Test that importing oscura.exceptions emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import triggers deprecation warning
            import oscura.exceptions  # noqa: F401

            # Should have deprecation warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            assert any("oscura.exceptions is deprecated" in str(warning.message) for warning in w)
            assert any("oscura.core.exceptions" in str(warning.message) for warning in w)

    def test_all_exceptions_exported(self) -> None:
        """Test that all expected exceptions are exported."""
        import oscura.exceptions as exc_module

        expected_exceptions = [
            "OscuraError",
            "LoaderError",
            "UnsupportedFormatError",
            "FormatError",
            "SampleRateError",
            "AnalysisError",
            "ExportError",
            "InsufficientDataError",
            "ValidationError",
            "ConfigurationError",
        ]

        for exc_name in expected_exceptions:
            assert hasattr(exc_module, exc_name), f"Missing exception: {exc_name}"
            exc_class = getattr(exc_module, exc_name)
            assert issubclass(exc_class, Exception)

    def test_exceptions_match_core_module(self) -> None:
        """Test that re-exported exceptions are identical to core.exceptions."""
        import oscura.exceptions as deprecated_exc
        from oscura.core import exceptions as core_exc

        # All exceptions should be the same objects
        assert deprecated_exc.OscuraError is core_exc.OscuraError
        assert deprecated_exc.LoaderError is core_exc.LoaderError
        assert deprecated_exc.AnalysisError is core_exc.AnalysisError
        assert deprecated_exc.ExportError is core_exc.ExportError
        assert deprecated_exc.ValidationError is core_exc.ValidationError
        assert deprecated_exc.ConfigurationError is core_exc.ConfigurationError

    def test_oscura_error_base_class(self) -> None:
        """Test OscuraError is proper base for all custom exceptions."""
        from oscura.exceptions import (
            AnalysisError,
            ConfigurationError,
            ExportError,
            LoaderError,
            OscuraError,
            ValidationError,
        )

        # All custom exceptions inherit from OscuraError
        assert issubclass(LoaderError, OscuraError)
        assert issubclass(AnalysisError, OscuraError)
        assert issubclass(ExportError, OscuraError)
        assert issubclass(ValidationError, OscuraError)
        assert issubclass(ConfigurationError, OscuraError)

        # OscuraError inherits from Exception
        assert issubclass(OscuraError, Exception)

    def test_exceptions_can_be_raised(self) -> None:
        """Test that re-exported exceptions can be raised normally."""
        from oscura.exceptions import LoaderError, OscuraError

        # Can raise and catch base exception
        with pytest.raises(OscuraError, match="test error"):
            raise OscuraError("test error")

        # Can raise and catch specific exception
        with pytest.raises(LoaderError, match="loader failed"):
            raise LoaderError("loader failed")

        # Can catch specific with base
        with pytest.raises(OscuraError):
            raise LoaderError("caught as base")

    def test_exceptions_preserve_messages(self) -> None:
        """Test exception messages are preserved correctly."""
        from oscura.exceptions import AnalysisError, ValidationError

        error_msg = "Custom error message with details"

        try:
            raise AnalysisError(error_msg)
        except AnalysisError as e:
            assert str(e) == error_msg
            assert e.args[0] == error_msg

        try:
            raise ValidationError(error_msg)
        except ValidationError as e:
            assert str(e) == error_msg

    def test_exceptions_with_multiple_args(self) -> None:
        """Test exceptions support multiple arguments."""
        from oscura.exceptions import LoaderError

        try:
            raise LoaderError("File not found", "path/to/file.vcd")
        except LoaderError as e:
            assert len(e.args) == 2
            assert e.args[0] == "File not found"
            assert e.args[1] == "path/to/file.vcd"

    def test_exception_inheritance_chain(self) -> None:
        """Test exception inheritance relationships."""
        from oscura.exceptions import (
            FormatError,
            LoaderError,
            UnsupportedFormatError,
        )

        # Check inheritance chain
        assert issubclass(UnsupportedFormatError, LoaderError)
        assert issubclass(FormatError, LoaderError)

        # Can catch with parent class
        with pytest.raises(LoaderError):
            raise UnsupportedFormatError("VCD not supported")

        with pytest.raises(LoaderError):
            raise FormatError("Invalid format")

    def test_sample_rate_error(self) -> None:
        """Test SampleRateError specific exception."""
        from oscura.exceptions import SampleRateError

        with pytest.raises(SampleRateError, match="Invalid sample rate"):
            raise SampleRateError("Invalid sample rate: must be positive")

    def test_insufficient_data_error(self) -> None:
        """Test InsufficientDataError specific exception."""
        from oscura.exceptions import InsufficientDataError

        with pytest.raises(InsufficientDataError, match="Not enough samples"):
            raise InsufficientDataError("Not enough samples for analysis")

    def test_configuration_error(self) -> None:
        """Test ConfigurationError specific exception."""
        from oscura.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Invalid config"):
            raise ConfigurationError("Invalid config: missing required field")

    def test_export_error(self) -> None:
        """Test ExportError specific exception."""
        from oscura.exceptions import ExportError

        with pytest.raises(ExportError, match="Export failed"):
            raise ExportError("Export failed: invalid format")

    def test_all_exports_defined(self) -> None:
        """Test __all__ contains all exported exceptions."""
        import oscura.exceptions as exc_module

        # __all__ should be defined
        assert hasattr(exc_module, "__all__")
        assert isinstance(exc_module.__all__, list)

        # All items in __all__ should be available
        for name in exc_module.__all__:
            assert hasattr(exc_module, name), f"{name} in __all__ but not exported"

    def test_backward_compatibility(self) -> None:
        """Test backward compatibility with old import style."""
        # Old-style import should still work (with deprecation warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            from oscura.exceptions import LoaderError as OldLoaderError

        # New-style import
        from oscura.core.exceptions import LoaderError as NewLoaderError

        # Should be the same class
        assert OldLoaderError is NewLoaderError

        # Can mix old and new style in exception handling
        try:
            raise OldLoaderError("test")
        except NewLoaderError:
            pass  # Should catch successfully
