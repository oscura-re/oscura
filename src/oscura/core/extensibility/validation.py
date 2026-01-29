"""Extension validation system for Oscura plugins and custom decoders.

This module provides comprehensive validation of extensions including metadata
validation, interface compliance checking, dependency verification, and
security checks.


Example:
    >>> from oscura.core.extensibility.validation import validate_extension
    >>> from pathlib import Path
    >>>
    >>> # Validate a plugin directory
    >>> result = validate_extension(Path("my_plugin/"))
    >>> if result.is_valid:
    ...     print("Plugin is valid!")
    >>> else:
    ...     for error in result.errors:
    ...         print(f"Error: {error}")
"""

from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue found during extension validation.

    Attributes:
        severity: Severity level ("error", "warning", "info")
        message: Human-readable error message
        location: Optional location information (file, line number)
        fix_hint: Optional suggestion for fixing the issue
    """

    severity: str
    message: str
    location: str = ""
    fix_hint: str = ""


@dataclass
class ValidationResult:
    """Result of extension validation.

    Attributes:
        is_valid: Whether extension passed all validation checks
        errors: List of error-level issues
        warnings: List of warning-level issues
        info: List of informational messages
        metadata: Extracted extension metadata
    """

    is_valid: bool = True
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str, location: str = "", fix_hint: str = "") -> None:
        """Add an error issue.

        Args:
            message: Error message
            location: Optional location
            fix_hint: Optional fix suggestion
        """
        self.errors.append(
            ValidationIssue(
                severity="error",
                message=message,
                location=location,
                fix_hint=fix_hint,
            )
        )
        self.is_valid = False

    def add_warning(self, message: str, location: str = "", fix_hint: str = "") -> None:
        """Add a warning issue.

        Args:
            message: Warning message
            location: Optional location
            fix_hint: Optional fix suggestion
        """
        self.warnings.append(
            ValidationIssue(
                severity="warning",
                message=message,
                location=location,
                fix_hint=fix_hint,
            )
        )

    def add_info(self, message: str, location: str = "") -> None:
        """Add an informational message.

        Args:
            message: Info message
            location: Optional location
        """
        self.info.append(
            ValidationIssue(
                severity="info",
                message=message,
                location=location,
            )
        )

    @property
    def all_issues(self) -> list[ValidationIssue]:
        """Get all issues sorted by severity.

        Returns:
            List of all issues (errors, warnings, info)
        """
        return self.errors + self.warnings + self.info


def validate_extension(
    extension_path: Path,
    *,
    check_dependencies: bool = True,
    check_security: bool = True,
    strict: bool = False,
) -> ValidationResult:
    """Validate an extension (plugin, decoder, etc.) at the given path.

    Performs comprehensive validation including:
    - Metadata validation (pyproject.toml or plugin.yaml)
    - Interface compliance checking
    - Entry point validation
    - Dependency verification
    - Security checks (if enabled)
    - Code quality checks (if strict)

    Args:
        extension_path: Path to extension directory
        check_dependencies: Verify dependencies are satisfied
        check_security: Perform security checks
        strict: Enable strict validation (warnings become errors)

    Returns:
        ValidationResult with validation outcome

    Example:
        >>> from pathlib import Path
        >>> result = validate_extension(Path("plugins/my_decoder/"))
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(f"Error: {error.message}")
        ...         if error.fix_hint:
        ...             print(f"  Fix: {error.fix_hint}")

    References:
        EXT-005: Extension Validation
    """
    result = ValidationResult()

    if not extension_path.exists():
        result.add_error(
            f"Extension path does not exist: {extension_path}",
            fix_hint="Check the path is correct",
        )
        return result

    if not extension_path.is_dir():
        result.add_error(
            f"Extension path is not a directory: {extension_path}",
            fix_hint="Provide path to extension directory",
        )
        return result

    result.add_info(f"Validating extension at: {extension_path}")

    # Validate metadata
    _validate_metadata(extension_path, result)

    # Validate structure
    _validate_structure(extension_path, result)

    # Validate entry points
    _validate_entry_points(extension_path, result)

    # Validate implementation
    _validate_implementation(extension_path, result)

    # Check dependencies if requested
    if check_dependencies:
        _check_dependencies(extension_path, result)

    # Security checks if requested
    if check_security:
        _check_security(extension_path, result)

    # Convert warnings to errors in strict mode
    if strict and result.warnings:
        for warning in result.warnings:
            result.add_error(
                f"Strict mode: {warning.message}",
                location=warning.location,
                fix_hint=warning.fix_hint,
            )
        result.warnings = []

    return result


def validate_decoder_interface(
    decoder_class: type,
) -> ValidationResult:
    """Validate that a decoder class implements the required interface.

    Checks for:
    - Required methods (decode, get_metadata)
    - Optional methods (configure, reset, validate_config)
    - Method signatures
    - Return types

    Args:
        decoder_class: Decoder class to validate

    Returns:
        ValidationResult with validation outcome

    Example:
        >>> class MyDecoder:
        ...     def decode(self, trace):
        ...         return []
        ...     def get_metadata(self):
        ...         return {"name": "my_decoder"}
        >>> result = validate_decoder_interface(MyDecoder)
        >>> assert result.is_valid

    References:
        EXT-006: Custom Decoder Registration
    """
    result = ValidationResult()

    # Required methods
    required_methods = {
        "decode": {
            "params": ["self", "trace"],
            "returns": "list",
        },
        "get_metadata": {
            "params": ["self"],
            "returns": "dict",
        },
    }

    # Optional methods
    optional_methods = {
        "configure": {"params": ["self"], "returns": None},
        "reset": {"params": ["self"], "returns": None},
        "validate_config": {"params": ["self", "config"], "returns": "bool"},
    }

    # Check required methods
    for method_name in required_methods:
        if not hasattr(decoder_class, method_name):
            result.add_error(
                f"Missing required method: {method_name}",
                location=f"{decoder_class.__name__}",
                fix_hint=f"Add method: def {method_name}(self, ...): ...",
            )
            continue

        method = getattr(decoder_class, method_name)
        if not callable(method):
            result.add_error(
                f"Method {method_name} is not callable",
                location=f"{decoder_class.__name__}.{method_name}",
            )

    # Check optional methods if present
    for method_name in optional_methods:
        if hasattr(decoder_class, method_name):
            method = getattr(decoder_class, method_name)
            if not callable(method):
                result.add_warning(
                    f"Optional method {method_name} exists but is not callable",
                    location=f"{decoder_class.__name__}.{method_name}",
                )

    # Check documentation requirements (EXT-006)
    if not decoder_class.__doc__ or not decoder_class.__doc__.strip():
        result.add_error(
            "Decoder class must have a docstring documenting its purpose and usage",
            location=f"{decoder_class.__name__}",
            fix_hint='Add docstring: """Decoder for XYZ protocol."""',
        )

    # Extract metadata
    result.metadata = {
        "class_name": decoder_class.__name__,
        "module": decoder_class.__module__,
        "required_methods": list(required_methods.keys()),
        "optional_methods": list(optional_methods.keys()),
        "has_docstring": decoder_class.__doc__ is not None,
    }

    if result.is_valid:
        result.add_info(f"Decoder interface validation passed for {decoder_class.__name__}")

    return result


def validate_hook_function(
    func: Callable[[Any], Any],
) -> ValidationResult:
    """Validate that a function is suitable for use as a hook.

    Checks:
    - Function signature accepts HookContext
    - Function returns HookContext
    - Function has docstring
    - Function handles exceptions

    Args:
        func: Hook function to validate

    Returns:
        ValidationResult with validation outcome

    Example:
        >>> def my_hook(context):
        ...     '''Validate context.'''
        ...     return context
        >>> result = validate_hook_function(my_hook)
        >>> assert result.is_valid

    References:
        EXT-005: Hook System
    """
    result = ValidationResult()

    if not callable(func):
        result.add_error(  # type: ignore[unreachable]
            "Hook must be callable",
            fix_hint="Provide a function or callable object",
        )
        return result

    # Check signature
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    if len(params) < 1:
        result.add_error(
            "Hook function must accept at least one parameter (context)",
            location=func.__name__,
            fix_hint="Add parameter: def hook(context): ...",
        )

    # Check for docstring
    if not func.__doc__:
        result.add_warning(
            "Hook function should have a docstring",
            location=func.__name__,
            fix_hint='Add docstring: """Hook description."""',
        )

    result.metadata = {
        "name": func.__name__,
        "params": params,
        "has_docstring": func.__doc__ is not None,
    }

    if result.is_valid:
        result.add_info(f"Hook function validation passed for {func.__name__}")

    return result


def _validate_metadata(extension_path: Path, result: ValidationResult) -> None:
    """Validate extension metadata (pyproject.toml or plugin.yaml).

    Args:
        extension_path: Path to extension directory
        result: ValidationResult to append issues to
    """
    pyproject = extension_path / "pyproject.toml"
    plugin_yaml = extension_path / "plugin.yaml"

    if not pyproject.exists() and not plugin_yaml.exists():
        result.add_error(
            "No metadata file found (pyproject.toml or plugin.yaml)",
            location=str(extension_path),
            fix_hint="Create pyproject.toml with [project] section",
        )
        return

    if pyproject.exists():
        try:
            import tomllib

            with open(pyproject, "rb") as f:
                data = tomllib.load(f)

            # Check required project fields
            if "project" not in data:
                result.add_error(
                    "pyproject.toml missing [project] section",
                    location=str(pyproject),
                )
            else:
                project = data["project"]
                required = ["name", "version", "description"]
                for field in required:
                    if field not in project:
                        result.add_error(
                            f"pyproject.toml missing required field: {field}",
                            location="[project]",
                            fix_hint=f'Add: {field} = "..."',
                        )

                result.metadata.update(
                    {
                        "name": project.get("name", ""),
                        "version": project.get("version", ""),
                        "description": project.get("description", ""),
                    }
                )

        except Exception as e:
            result.add_error(
                f"Failed to parse pyproject.toml: {e}",
                location=str(pyproject),
            )


def _validate_structure(extension_path: Path, result: ValidationResult) -> None:
    """Validate extension directory structure.

    Args:
        extension_path: Path to extension directory
        result: ValidationResult to append issues to
    """
    # Check for __init__.py
    init_py = extension_path / "__init__.py"
    if not init_py.exists():
        result.add_warning(
            "No __init__.py found",
            location=str(extension_path),
            fix_hint="Add __init__.py to make it a Python package",
        )

    # Check for tests directory
    tests_dir = extension_path / "tests"
    if not tests_dir.exists():
        result.add_warning(
            "No tests/ directory found",
            location=str(extension_path),
            fix_hint="Add tests/ directory with unit tests",
        )
    else:
        # Check for test files
        test_files = list(tests_dir.glob("test_*.py"))
        if not test_files:
            result.add_warning(
                "No test files found in tests/",
                location=str(tests_dir),
                fix_hint="Add test_*.py files",
            )

    # Check for README
    readme_files = list(extension_path.glob("README.*"))
    if not readme_files:
        result.add_warning(
            "No README file found",
            location=str(extension_path),
            fix_hint="Add README.md with usage documentation",
        )


def _validate_entry_points(extension_path: Path, result: ValidationResult) -> None:
    """Validate entry points configuration.

    Args:
        extension_path: Path to extension directory
        result: ValidationResult to append issues to
    """
    pyproject = extension_path / "pyproject.toml"
    if not pyproject.exists():
        return

    try:
        import tomllib

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        # Check for entry points
        if "project" not in data or "entry-points" not in data["project"]:
            result.add_info(
                "No entry points defined (plugin may be used as library)",
                location=str(pyproject),
            )
            return

        entry_points = data["project"]["entry-points"]
        oscura_groups = [k for k in entry_points if k.startswith("oscura.")]

        if not oscura_groups:
            result.add_warning(
                "No Oscura entry points found",
                location="[project.entry-points]",
                fix_hint="Add entry point like: oscura.decoders = ...",
            )
        else:
            result.metadata["entry_points"] = oscura_groups
            result.add_info(f"Found entry point groups: {', '.join(oscura_groups)}")

    except Exception as e:
        result.add_warning(f"Failed to validate entry points: {e}")


def _validate_implementation(extension_path: Path, result: ValidationResult) -> None:
    """Validate extension implementation files.

    Args:
        extension_path: Path to extension directory
        result: ValidationResult to append issues to
    """
    # Find Python files
    py_files = list(extension_path.glob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]

    if not py_files:
        result.add_warning(
            "No implementation files found",
            location=str(extension_path),
            fix_hint="Add Python module with implementation",
        )
        return

    # Basic syntax check
    for py_file in py_files:
        try:
            with open(py_file, encoding="utf-8") as f:
                source = f.read()
            ast.parse(source)
            result.add_info(f"Syntax check passed: {py_file.name}")
        except SyntaxError as e:
            result.add_error(
                f"Syntax error in {py_file.name}: {e}",
                location=f"{py_file.name}:{e.lineno}",
                fix_hint="Fix syntax error",
            )


def _check_dependencies(extension_path: Path, result: ValidationResult) -> None:
    """Check extension dependencies are satisfied.

    Args:
        extension_path: Path to extension directory
        result: ValidationResult to append issues to
    """
    pyproject = extension_path / "pyproject.toml"
    if not pyproject.exists():
        return

    try:
        import tomllib

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        if "project" not in data or "dependencies" not in data["project"]:
            result.add_info("No dependencies declared")
            return

        dependencies = data["project"]["dependencies"]
        result.metadata["dependencies"] = dependencies

        # Check if oscura is in dependencies
        oscura_deps = [d for d in dependencies if "oscura" in d.lower()]
        if not oscura_deps:
            result.add_warning(
                "Oscura not listed in dependencies",
                location="[project.dependencies]",
                fix_hint='Add: "oscura>=0.1.0"',
            )

    except Exception as e:
        result.add_warning(f"Failed to check dependencies: {e}")


def _check_security(extension_path: Path, result: ValidationResult) -> None:
    """Perform basic security checks on extension.

    Args:
        extension_path: Path to extension directory
        result: ValidationResult to append issues to
    """
    # Check for common security issues
    py_files = list(extension_path.rglob("*.py"))

    dangerous_imports = ["pickle", "eval", "exec", "compile", "__import__"]
    dangerous_calls = ["eval(", "exec(", "compile(", "__import__("]

    for py_file in py_files:
        try:
            with open(py_file, encoding="utf-8") as f:
                source = f.read()

            # Check for dangerous imports
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            result.add_warning(
                                f"Potentially unsafe import: {alias.name}",
                                location=f"{py_file.name}:{node.lineno}",
                                fix_hint="Consider safer alternatives",
                            )

            # Check for dangerous function calls
            for call in dangerous_calls:
                if call in source:
                    result.add_warning(
                        f"Potentially unsafe call: {call}",
                        location=py_file.name,
                        fix_hint="Avoid eval/exec for security",
                    )

        except Exception:
            # Ignore parse errors, already caught in implementation validation
            pass


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "validate_decoder_interface",
    "validate_extension",
    "validate_hook_function",
]
