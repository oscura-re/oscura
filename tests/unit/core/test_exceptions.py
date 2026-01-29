"""Comprehensive unit tests for Oscura exception hierarchy.

Requirements tested:
"""

import pytest

from oscura.core.exceptions import (
    DOCS_BASE_URL,
    AnalysisError,
    ConfigurationError,
    ExportError,
    FormatError,
    InsufficientDataError,
    LoaderError,
    OscuraError,
    SampleRateError,
    UnsupportedFormatError,
    ValidationError,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestOscuraError:
    """Test base OscuraError exception class."""

    def test_basic_message(self):
        """Test creating exception with basic message only."""
        error = OscuraError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.details is None
        assert error.fix_hint is None
        assert error.docs_path == "errors"

        # Check formatted message
        error_str = str(error)
        assert "Something went wrong" in error_str
        assert f"Docs: {DOCS_BASE_URL}/errors" in error_str

    def test_with_details(self):
        """Test creating exception with details."""
        error = OscuraError("Error occurred", details="Additional context")
        assert error.message == "Error occurred"
        assert error.details == "Additional context"

        error_str = str(error)
        assert "Error occurred" in error_str
        assert "Details: Additional context" in error_str

    def test_with_fix_hint(self):
        """Test creating exception with fix hint."""
        error = OscuraError("Error occurred", fix_hint="Try this fix")
        assert error.fix_hint == "Try this fix"

        error_str = str(error)
        assert "Fix: Try this fix" in error_str

    def test_with_custom_docs_path(self):
        """Test creating exception with custom docs path."""
        error = OscuraError("Error occurred", docs_path="custom/path")
        assert error.docs_path == "custom/path"

        error_str = str(error)
        assert f"Docs: {DOCS_BASE_URL}/custom/path" in error_str

    def test_all_parameters(self):
        """Test creating exception with all parameters."""
        error = OscuraError(
            "Main message",
            details="Extra details",
            fix_hint="How to fix",
            docs_path="custom/docs",
        )

        error_str = str(error)
        assert "Main message" in error_str
        assert "Details: Extra details" in error_str
        assert "Fix: How to fix" in error_str
        assert f"Docs: {DOCS_BASE_URL}/custom/docs" in error_str

    def test_inheritance(self):
        """Test that OscuraError inherits from Exception."""
        error = OscuraError("Test")
        assert isinstance(error, Exception)

    def test_message_format_order(self):
        """Test that message parts appear in correct order."""
        error = OscuraError(
            "Message",
            details="Details",
            fix_hint="Fix",
            docs_path="path",
        )

        error_str = str(error)
        lines = error_str.split("\n")

        assert lines[0] == "Message"
        assert lines[1] == "Details: Details"
        assert lines[2] == "Fix: Fix"
        assert lines[3] == f"Docs: {DOCS_BASE_URL}/path"


class TestLoaderError:
    """Test LoaderError exception class."""

    def test_basic_message(self):
        """Test creating LoaderError with basic message."""
        error = LoaderError("Failed to load file")
        assert error.message == "Failed to load file"
        assert error.file_path is None
        assert error.docs_path == "errors#loader"

    def test_with_file_path(self):
        """Test creating LoaderError with file path."""
        error = LoaderError("Failed to load file", file_path="/path/to/file.wfm")
        assert error.file_path == "/path/to/file.wfm"

        error_str = str(error)
        assert "File: /path/to/file.wfm" in error_str

    def test_with_file_path_and_details(self):
        """Test that file path is prepended to details."""
        error = LoaderError(
            "Failed to load file",
            file_path="/path/to/file.wfm",
            details="Corrupted header",
        )

        error_str = str(error)
        assert "File: /path/to/file.wfm. Corrupted header" in error_str

    def test_with_fix_hint(self):
        """Test creating LoaderError with fix hint."""
        error = LoaderError(
            "Failed to load file",
            file_path="/path/to/file.wfm",
            fix_hint="Check file permissions",
        )

        error_str = str(error)
        assert "Fix: Check file permissions" in error_str

    def test_inheritance(self):
        """Test that LoaderError inherits from OscuraError."""
        error = LoaderError("Test")
        assert isinstance(error, OscuraError)
        assert isinstance(error, Exception)


class TestUnsupportedFormatError:
    """Test UnsupportedFormatError exception class."""

    def test_basic_format(self):
        """Test creating error with unsupported format."""
        error = UnsupportedFormatError(".xyz")
        assert error.format_ext == ".xyz"
        assert error.supported_formats == []

        error_str = str(error)
        assert "Unsupported file format: .xyz" in error_str
        assert error.docs_path == "errors#unsupported-format"

    def test_with_supported_formats(self):
        """Test creating error with list of supported formats."""
        error = UnsupportedFormatError(".xyz", ["wfm", "csv", "npz"])
        assert error.format_ext == ".xyz"
        assert error.supported_formats == ["wfm", "csv", "npz"]

        error_str = str(error)
        assert "Unsupported file format: .xyz" in error_str
        assert "Supported formats: wfm, csv, npz" in error_str

    def test_with_file_path(self):
        """Test creating error with file path."""
        error = UnsupportedFormatError(
            ".xyz",
            ["wfm", "csv"],
            file_path="/path/to/file.xyz",
        )
        assert error.file_path == "/path/to/file.xyz"

        error_str = str(error)
        assert "File: /path/to/file.xyz" in error_str

    def test_default_fix_hint(self):
        """Test that default fix hint is provided."""
        error = UnsupportedFormatError(".xyz")

        error_str = str(error)
        assert "Convert the file to a supported format or use a custom loader." in error_str

    def test_inheritance(self):
        """Test that UnsupportedFormatError inherits from LoaderError."""
        error = UnsupportedFormatError(".xyz")
        assert isinstance(error, LoaderError)
        assert isinstance(error, OscuraError)
        assert isinstance(error, Exception)


class TestFormatError:
    """Test FormatError exception class."""

    def test_basic_message(self):
        """Test creating FormatError with basic message."""
        error = FormatError("Invalid file format")
        assert error.message == "Invalid file format"
        assert error.docs_path == "errors#format-error"

    def test_with_expected_and_got(self):
        """Test creating error with expected and got values."""
        error = FormatError(
            "Invalid header",
            expected="MAGIC",
            got="BADMG",
        )

        error_str = str(error)
        assert "Invalid header" in error_str
        assert "Expected: MAGIC. Got: BADMG" in error_str

    def test_with_expected_only(self):
        """Test creating error with expected value only."""
        error = FormatError("Missing field", expected="version number")

        error_str = str(error)
        assert "Expected: version number" in error_str

    def test_with_got_only(self):
        """Test creating error with got value only."""
        error = FormatError("Unexpected content", got="binary data")

        error_str = str(error)
        assert "Found: binary data" in error_str

    def test_with_custom_details(self):
        """Test that custom details override expected/got."""
        error = FormatError(
            "Invalid format",
            expected="MAGIC",
            got="BADMG",
            details="Custom details message",
        )

        error_str = str(error)
        assert "Custom details message" in error_str
        # Should NOT contain auto-generated expected/got
        assert "Expected: MAGIC" not in error_str

    def test_with_file_path(self):
        """Test creating error with file path."""
        error = FormatError(
            "Invalid format",
            file_path="/path/to/file.wfm",
            expected="WFM",
        )

        error_str = str(error)
        assert "File: /path/to/file.wfm" in error_str

    def test_default_fix_hint(self):
        """Test that default fix hint is provided."""
        error = FormatError("Invalid format")

        error_str = str(error)
        assert "Verify the file is not corrupted and matches the expected format." in error_str

    def test_custom_fix_hint(self):
        """Test creating error with custom fix hint."""
        error = FormatError(
            "Invalid format",
            fix_hint="Re-export the file from the original tool",
        )

        error_str = str(error)
        assert "Re-export the file from the original tool" in error_str

    def test_inheritance(self):
        """Test that FormatError inherits from LoaderError."""
        error = FormatError("Test")
        assert isinstance(error, LoaderError)
        assert isinstance(error, OscuraError)


class TestAnalysisError:
    """Test AnalysisError exception class."""

    def test_basic_message(self):
        """Test creating AnalysisError with basic message."""
        error = AnalysisError("Analysis failed")
        assert error.message == "Analysis failed"
        assert error.analysis_type is None
        assert error.docs_path == "errors#analysis"

    def test_with_analysis_type(self):
        """Test creating error with analysis type."""
        error = AnalysisError("Computation error", analysis_type="rise_time")
        assert error.analysis_type == "rise_time"

        error_str = str(error)
        assert "Analysis: rise_time" in error_str

    def test_with_analysis_type_and_details(self):
        """Test that analysis type is prepended to details."""
        error = AnalysisError(
            "Computation error",
            analysis_type="rise_time",
            details="No edges found",
        )

        error_str = str(error)
        assert "Analysis: rise_time. No edges found" in error_str

    def test_with_fix_hint(self):
        """Test creating error with fix hint."""
        error = AnalysisError(
            "Analysis failed",
            fix_hint="Increase sample rate",
        )

        error_str = str(error)
        assert "Fix: Increase sample rate" in error_str

    def test_inheritance(self):
        """Test that AnalysisError inherits from OscuraError."""
        error = AnalysisError("Test")
        assert isinstance(error, OscuraError)
        assert isinstance(error, Exception)


class TestInsufficientDataError:
    """Test InsufficientDataError exception class."""

    def test_basic_message(self):
        """Test creating error with basic message."""
        error = InsufficientDataError("Not enough data")
        assert error.message == "Not enough data"
        assert error.required is None
        assert error.available is None
        assert error.docs_path == "errors#insufficient-data"

    def test_with_required_and_available(self):
        """Test creating error with required and available counts."""
        error = InsufficientDataError(
            "Not enough samples",
            required=1000,
            available=500,
        )
        assert error.required == 1000
        assert error.available == 500

        error_str = str(error)
        assert "Required: 1000. Available: 500" in error_str

    def test_with_required_only(self):
        """Test creating error with required count only."""
        error = InsufficientDataError(
            "Not enough data",
            required=1000,
        )

        error_str = str(error)
        assert "Minimum required: 1000" in error_str

    def test_with_analysis_type(self):
        """Test creating error with analysis type."""
        error = InsufficientDataError(
            "Not enough edges",
            required=10,
            available=3,
            analysis_type="period_detection",
        )

        error_str = str(error)
        assert "Analysis: period_detection" in error_str
        assert "Required: 10. Available: 3" in error_str

    def test_default_fix_hint(self):
        """Test that default fix hint is provided."""
        error = InsufficientDataError("Not enough data")

        error_str = str(error)
        assert "Acquire more data or reduce analysis window." in error_str

    def test_inheritance(self):
        """Test that InsufficientDataError inherits from AnalysisError."""
        error = InsufficientDataError("Test")
        assert isinstance(error, AnalysisError)
        assert isinstance(error, OscuraError)


class TestSampleRateError:
    """Test SampleRateError exception class."""

    def test_basic_message(self):
        """Test creating error with basic message."""
        error = SampleRateError("Invalid sample rate")
        assert error.message == "Invalid sample rate"
        assert error.required_rate is None
        assert error.actual_rate is None
        assert error.docs_path == "errors#sample-rate"

    def test_with_required_and_actual(self):
        """Test creating error with required and actual rates."""
        error = SampleRateError(
            "Sample rate too low",
            required_rate=1e6,
            actual_rate=1e3,
        )
        assert error.required_rate == 1e6
        assert error.actual_rate == 1e3

        error_str = str(error)
        assert "Required: 1.00e+06 Hz. Got: 1.00e+03 Hz" in error_str

    def test_with_actual_only(self):
        """Test creating error with actual rate only."""
        error = SampleRateError(
            "Invalid sample rate",
            actual_rate=-100.0,
        )

        error_str = str(error)
        assert "Got: -1.00e+02 Hz" in error_str

    def test_default_fix_hint(self):
        """Test that default fix hint is provided."""
        error = SampleRateError("Invalid sample rate")

        error_str = str(error)
        assert "Ensure sample_rate is positive and sufficient for the analysis." in error_str

    def test_scientific_notation(self):
        """Test that rates are formatted in scientific notation."""
        error = SampleRateError(
            "Rate too low",
            required_rate=5e9,
            actual_rate=2.5e6,
        )

        error_str = str(error)
        assert "5.00e+09" in error_str
        assert "2.50e+06" in error_str

    def test_inheritance(self):
        """Test that SampleRateError inherits from AnalysisError."""
        error = SampleRateError("Test")
        assert isinstance(error, AnalysisError)
        assert isinstance(error, OscuraError)


class TestConfigurationError:
    """Test ConfigurationError exception class."""

    def test_basic_message(self):
        """Test creating error with basic message."""
        error = ConfigurationError("Invalid configuration")
        assert error.message == "Invalid configuration"
        assert error.config_key is None
        assert error.expected_type is None
        assert error.actual_value is None
        assert error.docs_path == "errors#configuration"

    def test_with_config_key(self):
        """Test creating error with config key."""
        error = ConfigurationError(
            "Invalid value",
            config_key="sample_rate",
        )
        assert error.config_key == "sample_rate"

        error_str = str(error)
        assert "Key: sample_rate" in error_str

    def test_with_expected_type(self):
        """Test creating error with expected type."""
        error = ConfigurationError(
            "Type mismatch",
            config_key="threshold",
            expected_type="float",
        )

        error_str = str(error)
        assert "Key: threshold" in error_str
        assert "Expected: float" in error_str

    def test_with_actual_value(self):
        """Test creating error with actual value."""
        error = ConfigurationError(
            "Invalid value",
            config_key="mode",
            actual_value="invalid_mode",
        )

        error_str = str(error)
        assert "Got: 'invalid_mode'" in error_str

    def test_all_config_parameters(self):
        """Test creating error with all configuration parameters."""
        error = ConfigurationError(
            "Type mismatch",
            config_key="threshold",
            expected_type="float",
            actual_value="string",
        )

        error_str = str(error)
        assert "Key: threshold" in error_str
        assert "Expected: float" in error_str
        assert "Got: 'string'" in error_str

    def test_with_custom_details(self):
        """Test that custom details override auto-generated details."""
        error = ConfigurationError(
            "Invalid config",
            config_key="key",
            details="Custom details message",
        )

        error_str = str(error)
        assert "Custom details message" in error_str
        # Should NOT contain auto-generated key
        assert "Key: key" not in error_str

    def test_default_fix_hint(self):
        """Test that default fix hint is provided."""
        error = ConfigurationError("Invalid configuration")

        error_str = str(error)
        assert "Check configuration file and ensure all values are valid." in error_str

    def test_custom_fix_hint(self):
        """Test creating error with custom fix hint."""
        error = ConfigurationError(
            "Invalid config",
            fix_hint="Use example config as template",
        )

        error_str = str(error)
        assert "Use example config as template" in error_str

    def test_with_none_actual_value(self):
        """Test that None actual_value is excluded from details."""
        error = ConfigurationError(
            "Missing value",
            config_key="required_field",
            actual_value=None,
        )

        error_str = str(error)
        # None values are excluded from details (intentional behavior)
        assert "Got:" not in error_str
        assert "Key: required_field" in error_str

    def test_inheritance(self):
        """Test that ConfigurationError inherits from OscuraError."""
        error = ConfigurationError("Test")
        assert isinstance(error, OscuraError)


class TestValidationError:
    """Test ValidationError exception class."""

    def test_basic_message(self):
        """Test creating error with basic message."""
        error = ValidationError("Validation failed")
        assert error.message == "Validation failed"
        assert error.field is None
        assert error.constraint is None
        assert error.value is None
        assert error.docs_path == "errors#validation"

    def test_with_field(self):
        """Test creating error with field name."""
        error = ValidationError("Invalid field", field="sample_rate")
        assert error.field == "sample_rate"

        error_str = str(error)
        assert "Field: sample_rate" in error_str

    def test_with_constraint(self):
        """Test creating error with constraint."""
        error = ValidationError(
            "Constraint violation",
            field="threshold",
            constraint=">= 0",
        )

        error_str = str(error)
        assert "Field: threshold" in error_str
        assert "Constraint: >= 0" in error_str

    def test_with_value(self):
        """Test creating error with value."""
        error = ValidationError(
            "Invalid value",
            field="mode",
            value="invalid",
        )

        error_str = str(error)
        assert "Field: mode" in error_str
        assert "Value: 'invalid'" in error_str

    def test_all_validation_parameters(self):
        """Test creating error with all validation parameters."""
        error = ValidationError(
            "Constraint violation",
            field="threshold",
            constraint=">= 0",
            value=-5,
        )

        error_str = str(error)
        assert "Field: threshold" in error_str
        assert "Constraint: >= 0" in error_str
        assert "Value: -5" in error_str

    def test_default_fix_hint(self):
        """Test that default fix hint is provided."""
        error = ValidationError("Validation failed")

        error_str = str(error)
        assert "Ensure input data meets all validation requirements." in error_str

    def test_with_none_value(self):
        """Test that None value is excluded from details."""
        error = ValidationError(
            "Missing required field",
            field="required_param",
            value=None,
        )

        error_str = str(error)
        # None values are excluded from details (intentional behavior)
        assert "Value:" not in error_str
        assert "Field: required_param" in error_str

    def test_inheritance(self):
        """Test that ValidationError inherits from OscuraError."""
        error = ValidationError("Test")
        assert isinstance(error, OscuraError)


class TestExportError:
    """Test ExportError exception class."""

    def test_basic_message(self):
        """Test creating error with basic message."""
        error = ExportError("Export failed")
        assert error.message == "Export failed"
        assert error.export_format is None
        assert error.output_path is None
        assert error.docs_path == "errors#export"

    def test_with_export_format(self):
        """Test creating error with export format."""
        error = ExportError("Export failed", export_format="csv")
        assert error.export_format == "csv"

        error_str = str(error)
        assert "Format: csv" in error_str

    def test_with_output_path(self):
        """Test creating error with output path."""
        error = ExportError(
            "Export failed",
            output_path="/path/to/output.csv",
        )
        assert error.output_path == "/path/to/output.csv"

        error_str = str(error)
        assert "Path: /path/to/output.csv" in error_str

    def test_with_format_and_path(self):
        """Test creating error with both format and path."""
        error = ExportError(
            "Export failed",
            export_format="hdf5",
            output_path="/data/output.h5",
        )

        error_str = str(error)
        assert "Format: hdf5" in error_str
        assert "Path: /data/output.h5" in error_str

    def test_with_additional_details(self):
        """Test creating error with additional details."""
        error = ExportError(
            "Export failed",
            export_format="csv",
            output_path="/output.csv",
            details="Disk full",
        )

        error_str = str(error)
        assert "Format: csv" in error_str
        assert "Path: /output.csv" in error_str
        assert "Disk full" in error_str

    def test_default_fix_hint(self):
        """Test that default fix hint is provided."""
        error = ExportError("Export failed")

        error_str = str(error)
        assert "Check output path is writable and data is valid for export." in error_str

    def test_inheritance(self):
        """Test that ExportError inherits from OscuraError."""
        error = ExportError("Test")
        assert isinstance(error, OscuraError)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""

    def test_raise_and_catch_oscuraerror(self):
        """Test raising and catching OscuraError."""
        with pytest.raises(OscuraError) as exc_info:
            raise OscuraError("Test error")

        assert str(exc_info.value).startswith("Test error")

    def test_raise_and_catch_loader_error(self):
        """Test raising and catching LoaderError."""
        with pytest.raises(LoaderError) as exc_info:
            raise LoaderError("Load failed", file_path="/test.wfm")

        assert "Load failed" in str(exc_info.value)
        assert "/test.wfm" in str(exc_info.value)

    def test_catch_loader_error_as_oscuraerror(self):
        """Test catching LoaderError as OscuraError."""
        with pytest.raises(OscuraError):
            raise LoaderError("Load failed")

    def test_catch_unsupported_format_as_loader_error(self):
        """Test catching UnsupportedFormatError as LoaderError."""
        with pytest.raises(LoaderError):
            raise UnsupportedFormatError(".xyz", ["wfm"])

    def test_catch_insufficient_data_as_analysis_error(self):
        """Test catching InsufficientDataError as AnalysisError."""
        with pytest.raises(AnalysisError):
            raise InsufficientDataError("Not enough data", required=100, available=50)

    def test_catch_sample_rate_as_analysis_error(self):
        """Test catching SampleRateError as AnalysisError."""
        with pytest.raises(AnalysisError):
            raise SampleRateError("Invalid rate", actual_rate=-1.0)


class TestModuleConstants:
    """Test module-level constants."""

    def test_docs_base_url_exists(self):
        """Test that DOCS_BASE_URL is defined and points to valid location."""
        # DOCS_BASE_URL should be a valid URL string
        assert isinstance(DOCS_BASE_URL, str)
        assert DOCS_BASE_URL.startswith("http")
        # Should point to GitHub repository docs directory
        assert "github.com" in DOCS_BASE_URL or "readthedocs" in DOCS_BASE_URL

    def test_docs_base_url_in_error_messages(self):
        """Test that DOCS_BASE_URL is used in error messages."""
        error = OscuraError("Test")
        assert DOCS_BASE_URL in str(error)


class TestCoreExceptionsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_message(self):
        """Test creating error with empty string message."""
        error = OscuraError("")
        assert error.message == ""
        error_str = str(error)
        # Should still have docs line even with empty message
        assert f"Docs: {DOCS_BASE_URL}/errors" in error_str

    def test_very_long_message(self):
        """Test creating error with very long message."""
        long_message = "A" * 1000
        error = OscuraError(long_message)
        assert error.message == long_message
        assert long_message in str(error)

    def test_special_characters_in_message(self):
        """Test creating error with special characters."""
        message = "Error: 'quoted' \"double\" \n newline \t tab"
        error = OscuraError(message)
        assert error.message == message

    def test_unicode_in_message(self):
        """Test creating error with unicode characters."""
        message = "Error: n a b g ni hao "
        error = OscuraError(message)
        assert error.message == message
        assert message in str(error)

    def test_zero_values(self):
        """Test creating errors with zero values."""
        error1 = InsufficientDataError("No data", required=0, available=0)
        assert error1.required == 0
        assert error1.available == 0

        error2 = SampleRateError("Zero rate", required_rate=0.0, actual_rate=0.0)
        assert error2.required_rate == 0.0
        assert error2.actual_rate == 0.0

    def test_negative_values(self):
        """Test creating errors with negative values."""
        error = SampleRateError("Negative rate", actual_rate=-1000.0)
        error_str = str(error)
        assert "-1.00e+03" in error_str

    def test_float_formatting_edge_cases(self):
        """Test scientific notation formatting with various magnitudes."""
        # Very small number
        error1 = SampleRateError("Test", actual_rate=1e-10)
        assert "1.00e-10" in str(error1)

        # Very large number
        error2 = SampleRateError("Test", actual_rate=1e20)
        assert "1.00e+20" in str(error2)

        # Normal number
        error3 = SampleRateError("Test", actual_rate=1234.56)
        assert "1.23e+03" in str(error3)

    def test_empty_list_supported_formats(self):
        """Test UnsupportedFormatError with empty supported formats list."""
        error = UnsupportedFormatError(".xyz", [])
        assert error.supported_formats == []
        error_str = str(error)
        # Should not have "Supported formats:" line
        assert "Supported formats:" not in error_str

    def test_single_format_in_list(self):
        """Test UnsupportedFormatError with single supported format."""
        error = UnsupportedFormatError(".xyz", ["wfm"])
        error_str = str(error)
        assert "Supported formats: wfm" in error_str

    def test_many_formats_in_list(self):
        """Test UnsupportedFormatError with many supported formats."""
        formats = ["wfm", "csv", "npz", "hdf5", "json", "xml", "dat"]
        error = UnsupportedFormatError(".xyz", formats)
        error_str = str(error)
        assert "Supported formats: wfm, csv, npz, hdf5, json, xml, dat" in error_str


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_file_not_found_scenario(self):
        """Test typical file not found error."""
        error = LoaderError(
            "File not found",
            file_path="/nonexistent/data.wfm",
            fix_hint="Check that the file path is correct and the file exists",
        )

        error_str = str(error)
        assert "File not found" in error_str
        assert "/nonexistent/data.wfm" in error_str
        assert "Check that the file path is correct" in error_str

    def test_corrupted_file_scenario(self):
        """Test typical corrupted file error."""
        error = FormatError(
            "Corrupted file header",
            file_path="/data/signal.wfm",
            expected="WFM magic bytes",
            got="Random data",
        )

        error_str = str(error)
        assert "Corrupted file header" in error_str
        assert "Expected: WFM magic bytes. Got: Random data" in error_str

    def test_insufficient_samples_scenario(self):
        """Test typical insufficient samples error."""
        error = InsufficientDataError(
            "Cannot compute FFT",
            required=256,
            available=100,
            analysis_type="frequency_analysis",
        )

        error_str = str(error)
        assert "Cannot compute FFT" in error_str
        assert "Required: 256. Available: 100" in error_str
        assert "Analysis: frequency_analysis" in error_str

    def test_invalid_config_scenario(self):
        """Test typical configuration error."""
        error = ConfigurationError(
            "Invalid threshold value",
            config_key="edge_threshold",
            expected_type="float between 0.0 and 1.0",
            actual_value=1.5,
        )

        error_str = str(error)
        assert "Invalid threshold value" in error_str
        assert "Key: edge_threshold" in error_str
        assert "Expected: float between 0.0 and 1.0" in error_str
        assert "Got: 1.5" in error_str

    def test_export_permission_denied_scenario(self):
        """Test typical export permission error."""
        error = ExportError(
            "Failed to write output file",
            export_format="csv",
            output_path="/root/protected/output.csv",
            details="Permission denied",
        )

        error_str = str(error)
        assert "Failed to write output file" in error_str
        assert "Format: csv" in error_str
        assert "Path: /root/protected/output.csv" in error_str
        assert "Permission denied" in error_str

    def test_unsupported_format_with_suggestion_scenario(self):
        """Test unsupported format with helpful suggestions."""
        error = UnsupportedFormatError(
            ".trc",
            ["wfm", "isf", "csv"],
            file_path="/data/scope_capture.trc",
        )

        error_str = str(error)
        assert "Unsupported file format: .trc" in error_str
        assert "Supported formats: wfm, isf, csv" in error_str
        assert "File: /data/scope_capture.trc" in error_str
        assert "Convert the file to a supported format" in error_str
