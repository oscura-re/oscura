"""Oscura exception hierarchy with helpful error messages.

This module provides custom exception classes that follow a consistent
template for error messages: WHAT, WHY, HOW TO FIX, DOCUMENTATION LINK.


Example:
    >>> try:
    ...     raise UnsupportedFormatError(".xyz", ["wfm", "csv", "npz"])
    ... except UnsupportedFormatError as e:
    ...     print(e)
    Unsupported file format: .xyz
    ...
"""

from typing import Any

# Documentation base URL
# Points to GitHub repository docs directory as primary documentation source.
# Update this when official documentation hosting is configured (e.g., ReadTheDocs).
#
# To verify/update this URL:
# 1. Check if https://oscura.readthedocs.io exists and is active
# 2. If not, use GitHub repository docs: https://github.com/lair-click-bats/oscura/tree/main/docs
# 3. Or local docs path for development: file:///path/to/docs/
DOCS_BASE_URL = "https://github.com/lair-click-bats/oscura/tree/main/docs"


class OscuraError(Exception):
    """Base exception for all Oscura errors.

    All Oscura exceptions inherit from this class, providing a
    consistent interface for error handling.

    Attributes:
        message: Brief description of the error.
        details: Additional context about the error.
        fix_hint: Suggestion for how to fix the error.
        docs_path: Path to relevant documentation.

    Example:
        >>> raise OscuraError("Something went wrong")
        OscuraError: Something went wrong
    """

    docs_path: str = "errors"

    def __init__(
        self,
        message: str,
        *,
        details: str | None = None,
        fix_hint: str | None = None,
        docs_path: str | None = None,
    ) -> None:
        """Initialize OscuraError.

        Args:
            message: Brief description of the error.
            details: Additional context about what caused the error.
            fix_hint: Suggestion for how to fix the error.
            docs_path: Path to relevant documentation (appended to base URL).
        """
        self.message = message
        self.details = details
        self.fix_hint = fix_hint
        if docs_path is not None:
            self.docs_path = docs_path
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with template.

        Returns:
            Formatted error message with WHAT, WHY, HOW TO FIX, DOCS.
        """
        parts = [self.message]

        if self.details:
            parts.append(f"Details: {self.details}")

        if self.fix_hint:
            parts.append(f"Fix: {self.fix_hint}")

        parts.append(f"Docs: {DOCS_BASE_URL}/{self.docs_path}")

        return "\n".join(parts)


class LoaderError(OscuraError):
    """Error loading trace data from file.

    Raised when a file cannot be read, parsed, or converted to
    a Oscura data structure.

    Attributes:
        file_path: Path to the file that failed to load.
    """

    docs_path: str = "errors#loader"

    def __init__(
        self,
        message: str,
        *,
        file_path: str | None = None,
        details: str | None = None,
        fix_hint: str | None = None,
    ) -> None:
        """Initialize LoaderError.

        Args:
            message: Brief description of the error.
            file_path: Path to the file that failed to load.
            details: Additional context about the error.
            fix_hint: Suggestion for how to fix the error.
        """
        self.file_path = file_path
        if file_path and not details:
            details = f"File: {file_path}"
        elif file_path and details:
            details = f"File: {file_path}. {details}"
        super().__init__(
            message,
            details=details,
            fix_hint=fix_hint,
            docs_path=self.docs_path,
        )


class UnsupportedFormatError(LoaderError):
    """File format not recognized or unsupported.

    Raised when attempting to load a file with an unsupported
    extension or format.

    Attributes:
        format_ext: The unsupported format extension.
        supported_formats: List of supported format extensions.
    """

    docs_path: str = "errors#unsupported-format"

    def __init__(
        self,
        format_ext: str,
        supported_formats: list[str] | None = None,
        *,
        file_path: str | None = None,
    ) -> None:
        """Initialize UnsupportedFormatError.

        Args:
            format_ext: The unsupported format extension (e.g., ".xyz").
            supported_formats: List of supported format extensions.
            file_path: Path to the file that failed to load.
        """
        self.format_ext = format_ext
        self.supported_formats = supported_formats or []

        message = f"Unsupported file format: {format_ext}"

        if self.supported_formats:
            formats_str = ", ".join(self.supported_formats)
            details = f"Supported formats: {formats_str}"
        else:
            details = None

        fix_hint = "Convert the file to a supported format or use a custom loader."

        super().__init__(
            message,
            file_path=file_path,
            details=details,
            fix_hint=fix_hint,
        )


class FormatError(LoaderError):
    """File format is invalid or corrupted.

    Raised when a file has the correct extension but invalid content.
    """

    docs_path: str = "errors#format-error"

    def __init__(
        self,
        message: str,
        *,
        file_path: str | None = None,
        expected: str | None = None,
        got: str | None = None,
        details: str | None = None,
        fix_hint: str | None = None,
    ) -> None:
        """Initialize FormatError.

        Args:
            message: Brief description of the error.
            file_path: Path to the file that failed to load.
            expected: What was expected in the file.
            got: What was actually found.
            details: Additional context about the error (overrides expected/got).
            fix_hint: Suggestion for how to fix the error.
        """
        # Build details from expected/got if not provided directly
        if details is None:
            if expected and got:
                details = f"Expected: {expected}. Got: {got}"
            elif expected:
                details = f"Expected: {expected}"
            elif got:
                details = f"Found: {got}"

        # Use default fix_hint if not provided
        if fix_hint is None:
            fix_hint = "Verify the file is not corrupted and matches the expected format."

        super().__init__(
            message,
            file_path=file_path,
            details=details,
            fix_hint=fix_hint,
        )


class AnalysisError(OscuraError):
    """Error during signal analysis.

    Raised when an analysis function encounters invalid data
    or cannot compute a result.
    """

    docs_path: str = "errors#analysis"

    def __init__(
        self,
        message: str,
        *,
        analysis_type: str | None = None,
        details: str | None = None,
        fix_hint: str | None = None,
    ) -> None:
        """Initialize AnalysisError.

        Args:
            message: Brief description of the error.
            analysis_type: Type of analysis that failed (e.g., "rise_time").
            details: Additional context about the error.
            fix_hint: Suggestion for how to fix the error.
        """
        self.analysis_type = analysis_type
        if analysis_type and not details:
            details = f"Analysis: {analysis_type}"
        elif analysis_type and details:
            details = f"Analysis: {analysis_type}. {details}"
        super().__init__(
            message,
            details=details,
            fix_hint=fix_hint,
            docs_path=self.docs_path,
        )


class InsufficientDataError(AnalysisError):
    """Not enough data points for the requested analysis.

    Raised when a signal is too short or lacks sufficient
    features (edges, periods, etc.) for analysis.

    Attributes:
        required: Minimum data points or features required.
        available: Actual data points or features available.
    """

    docs_path: str = "errors#insufficient-data"

    def __init__(
        self,
        message: str,
        *,
        required: int | None = None,
        available: int | None = None,
        analysis_type: str | None = None,
        fix_hint: str | None = None,
    ) -> None:
        """Initialize InsufficientDataError.

        Args:
            message: Brief description of the error.
            required: Minimum number of samples or features required.
            available: Actual number available.
            analysis_type: Type of analysis that failed.
            fix_hint: Optional custom fix suggestion (overrides default).
        """
        self.required = required
        self.available = available

        details = None
        if required is not None and available is not None:
            details = f"Required: {required}. Available: {available}"
        elif required is not None:
            details = f"Minimum required: {required}"

        # Use default fix hint if not provided
        if fix_hint is None:
            fix_hint = "Acquire more data or reduce analysis window."

        super().__init__(
            message,
            analysis_type=analysis_type,
            details=details,
            fix_hint=fix_hint,
        )


class SampleRateError(AnalysisError):
    """Invalid or missing sample rate.

    Raised when sample rate is invalid (zero, negative) or
    insufficient for the requested analysis.

    Attributes:
        required_rate: Minimum sample rate required.
        actual_rate: Actual sample rate provided.
    """

    docs_path: str = "errors#sample-rate"

    def __init__(
        self,
        message: str,
        *,
        required_rate: float | None = None,
        actual_rate: float | None = None,
    ) -> None:
        """Initialize SampleRateError.

        Args:
            message: Brief description of the error.
            required_rate: Minimum sample rate required in Hz.
            actual_rate: Actual sample rate in Hz.
        """
        self.required_rate = required_rate
        self.actual_rate = actual_rate

        details = None
        if required_rate is not None and actual_rate is not None:
            details = f"Required: {required_rate:.2e} Hz. Got: {actual_rate:.2e} Hz"
        elif actual_rate is not None:
            details = f"Got: {actual_rate:.2e} Hz"

        fix_hint = "Ensure sample_rate is positive and sufficient for the analysis."

        super().__init__(
            message,
            details=details,
            fix_hint=fix_hint,
        )


class ConfigurationError(OscuraError):
    """Invalid configuration parameters.

    Raised when configuration is invalid, missing required fields,
    or contains invalid values.

    Attributes:
        config_key: The configuration key that is invalid.
        expected_type: Expected type or format.
        actual_value: The invalid value that was provided.
    """

    docs_path: str = "errors#configuration"

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        expected_type: str | None = None,
        actual_value: Any = None,
        details: str | None = None,
        fix_hint: str | None = None,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Brief description of the error.
            config_key: The configuration key that is invalid.
            expected_type: Expected type or format.
            actual_value: The invalid value that was provided.
            details: Additional context about the error.
            fix_hint: Suggestion for how to fix the error.
        """
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value

        # Build details from parts if not provided directly
        if details is None:
            details_parts = []
            if config_key:
                details_parts.append(f"Key: {config_key}")
            if expected_type:
                details_parts.append(f"Expected: {expected_type}")
            if actual_value is not None:
                details_parts.append(f"Got: {actual_value!r}")
            details = ". ".join(details_parts) if details_parts else None

        if fix_hint is None:
            fix_hint = "Check configuration file and ensure all values are valid."

        super().__init__(
            message,
            details=details,
            fix_hint=fix_hint,
            docs_path=self.docs_path,
        )


class ValidationError(OscuraError):
    """Data validation failed.

    Raised when input data does not meet validation requirements.

    Attributes:
        field: The field that failed validation.
        constraint: The constraint that was violated.
        value: The value that failed validation.
    """

    docs_path: str = "errors#validation"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        constraint: str | None = None,
        value: Any = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Brief description of the error.
            field: The field that failed validation.
            constraint: The constraint that was violated.
            value: The value that failed validation.
        """
        self.field = field
        self.constraint = constraint
        self.value = value

        details_parts = []
        if field:
            details_parts.append(f"Field: {field}")
        if constraint:
            details_parts.append(f"Constraint: {constraint}")
        if value is not None:
            details_parts.append(f"Value: {value!r}")

        details = ". ".join(details_parts) if details_parts else None
        fix_hint = "Ensure input data meets all validation requirements."

        super().__init__(
            message,
            details=details,
            fix_hint=fix_hint,
            docs_path=self.docs_path,
        )


class ExportError(OscuraError):
    """Error exporting data.

    Raised when data cannot be exported to the requested format.

    Attributes:
        export_format: The format that failed.
        output_path: Path where export was attempted.
    """

    docs_path: str = "errors#export"

    def __init__(
        self,
        message: str,
        *,
        export_format: str | None = None,
        output_path: str | None = None,
        details: str | None = None,
    ) -> None:
        """Initialize ExportError.

        Args:
            message: Brief description of the error.
            export_format: The format that failed (e.g., "csv", "hdf5").
            output_path: Path where export was attempted.
            details: Additional context about the error.
        """
        self.export_format = export_format
        self.output_path = output_path

        details_parts = []
        if export_format:
            details_parts.append(f"Format: {export_format}")
        if output_path:
            details_parts.append(f"Path: {output_path}")
        if details:
            details_parts.append(details)

        combined_details = ". ".join(details_parts) if details_parts else None
        fix_hint = "Check output path is writable and data is valid for export."

        super().__init__(
            message,
            details=combined_details,
            fix_hint=fix_hint,
            docs_path=self.docs_path,
        )


class SecurityError(OscuraError):
    """Security validation failed.

    Raised when security checks fail, such as signature verification
    or file integrity validation.

    Attributes:
        file_path: Path to the file that failed security checks.
        check_type: Type of security check that failed.
    """

    docs_path: str = "errors#security"

    def __init__(
        self,
        message: str,
        *,
        file_path: str | None = None,
        check_type: str | None = None,
        details: str | None = None,
        fix_hint: str | None = None,
    ) -> None:
        """Initialize SecurityError.

        Args:
            message: Brief description of the error.
            file_path: Path to the file that failed security checks.
            check_type: Type of security check that failed.
            details: Additional context about the error.
            fix_hint: Suggestion for how to fix the error.
        """
        self.file_path = file_path
        self.check_type = check_type

        details_parts = []
        if file_path:
            details_parts.append(f"File: {file_path}")
        if check_type:
            details_parts.append(f"Check: {check_type}")
        if details:
            details_parts.append(details)

        combined_details = ". ".join(details_parts) if details_parts else None

        if fix_hint is None:
            fix_hint = "Only load files from trusted sources. File may be corrupted or tampered."

        super().__init__(
            message,
            details=combined_details,
            fix_hint=fix_hint,
            docs_path=self.docs_path,
        )


__all__ = [
    "DOCS_BASE_URL",
    "AnalysisError",
    "ConfigurationError",
    "ExportError",
    "FormatError",
    "InsufficientDataError",
    "LoaderError",
    "OscuraError",
    "SampleRateError",
    "SecurityError",
    "UnsupportedFormatError",
    "ValidationError",
]
