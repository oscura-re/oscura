"""Protocol grammar validator for detecting specification errors and inconsistencies.

This module provides comprehensive validation of protocol specifications to detect
grammar errors, field inconsistencies, dependency issues, and state machine problems.

Example:
    >>> from oscura.validation import ProtocolGrammarValidator
    >>> from oscura.sessions import ProtocolSpec, FieldHypothesis
    >>>
    >>> # Define protocol spec
    >>> spec = ProtocolSpec(
    ...     name="MyProtocol",
    ...     fields=[
    ...         FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
    ...         FieldHypothesis("length", 1, 1, "data", 0.90),
    ...         FieldHypothesis("payload", 2, 4, "data", 0.85),
    ...         FieldHypothesis("checksum", 6, 1, "checksum", 0.95, {"range": (0, 6)}),
    ...     ]
    ... )
    >>>
    >>> # Validate protocol specification
    >>> validator = ProtocolGrammarValidator()
    >>> report = validator.validate(spec)
    >>>
    >>> if report.has_errors():
    ...     print("Validation failed:")
    ...     for error in report.errors:
    ...         print(f"  {error.severity}: {error.message}")
    ... else:
    ...     print("Protocol specification is valid!")
    >>>
    >>> # Export report
    >>> report.export_json(Path("validation_report.json"))

References:
    V0.6.0_COMPLETE_COMPREHENSIVE_PLAN.md: Feature 36 (Conformance Testing)
    Formal Language Theory: Grammar validation and consistency checking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.sessions.blackbox import FieldHypothesis, ProtocolSpec


class ErrorSeverity(Enum):
    """Severity level for validation errors.

    Attributes:
        ERROR: Must fix - specification is invalid.
        WARNING: Should fix - potential issues or ambiguities.
        INFO: Advisory - suggestions for improvement.
    """

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ErrorType(Enum):
    """Type of validation error.

    Attributes:
        FIELD_OVERLAP: Fields occupy overlapping byte ranges.
        FIELD_GAP: Gap between consecutive fields.
        INVALID_OFFSET: Field offset is negative or invalid.
        INVALID_LENGTH: Field length is zero or negative.
        LENGTH_MISMATCH: Length field value doesn't match referenced field.
        CHECKSUM_RANGE: Checksum coverage range is invalid.
        DUPLICATE_FIELD: Multiple fields with same name.
        UNREACHABLE_STATE: State machine has unreachable states.
        MISSING_TRANSITION: State machine has incomplete transitions.
        AMBIGUOUS_GRAMMAR: Protocol grammar has ambiguities.
        INVALID_DEPENDENCY: Field dependency cannot be resolved.
        ALIGNMENT_WARNING: Field not aligned to expected boundary.
        ENUM_DUPLICATE: Enum has duplicate values.
        ENUM_GAP: Enum has missing values in range.
    """

    FIELD_OVERLAP = "FIELD_OVERLAP"
    FIELD_GAP = "FIELD_GAP"
    INVALID_OFFSET = "INVALID_OFFSET"
    INVALID_LENGTH = "INVALID_LENGTH"
    LENGTH_MISMATCH = "LENGTH_MISMATCH"
    CHECKSUM_RANGE = "CHECKSUM_RANGE"
    DUPLICATE_FIELD = "DUPLICATE_FIELD"
    UNREACHABLE_STATE = "UNREACHABLE_STATE"
    MISSING_TRANSITION = "MISSING_TRANSITION"
    AMBIGUOUS_GRAMMAR = "AMBIGUOUS_GRAMMAR"
    INVALID_DEPENDENCY = "INVALID_DEPENDENCY"
    ALIGNMENT_WARNING = "ALIGNMENT_WARNING"
    ENUM_DUPLICATE = "ENUM_DUPLICATE"
    ENUM_GAP = "ENUM_GAP"


@dataclass
class ValidationError:
    """Single validation error with context and suggestions.

    Attributes:
        error_type: Type of error encountered.
        severity: Severity level (ERROR/WARNING/INFO).
        field_name: Name of field causing error (if applicable).
        message: Human-readable error message.
        suggestion: Suggested fix for the error.
        line_number: Line number in specification (if applicable).
        context: Additional context information.

    Example:
        >>> error = ValidationError(
        ...     error_type=ErrorType.FIELD_OVERLAP,
        ...     severity=ErrorSeverity.ERROR,
        ...     field_name="checksum",
        ...     message="Field 'checksum' overlaps with 'payload' at byte 5",
        ...     suggestion="Move checksum field to byte 6 or reduce payload length"
        ... )
    """

    error_type: ErrorType
    severity: ErrorSeverity
    field_name: str | None
    message: str
    suggestion: str | None = None
    line_number: int | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Comprehensive validation report with errors, warnings, and info.

    Attributes:
        errors: List of ERROR severity issues (must fix).
        warnings: List of WARNING severity issues (should fix).
        info: List of INFO severity messages (advisory).
        protocol_name: Name of validated protocol.
        total_fields: Total number of fields validated.
        metadata: Additional validation metadata.

    Example:
        >>> report = ValidationReport(
        ...     errors=[],
        ...     warnings=[warning1, warning2],
        ...     info=[info1],
        ...     protocol_name="MyProtocol",
        ...     total_fields=4
        ... )
        >>> print(f"Valid: {not report.has_errors()}")
    """

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    info: list[ValidationError] = field(default_factory=list)
    protocol_name: str = ""
    total_fields: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Check if report contains any errors.

        Returns:
            True if errors exist, False otherwise.

        Example:
            >>> report = ValidationReport(errors=[error1])
            >>> report.has_errors()
            True
        """
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if report contains any warnings.

        Returns:
            True if warnings exist, False otherwise.

        Example:
            >>> report = ValidationReport(warnings=[warning1])
            >>> report.has_warnings()
            True
        """
        return len(self.warnings) > 0

    def all_issues(self) -> list[ValidationError]:
        """Get all issues (errors + warnings + info).

        Returns:
            Combined list of all validation issues.

        Example:
            >>> issues = report.all_issues()
            >>> print(f"Total issues: {len(issues)}")
        """
        return self.errors + self.warnings + self.info

    def export_json(self, output: Path) -> None:
        """Export validation report as JSON.

        Args:
            output: Output JSON file path.

        Example:
            >>> report.export_json(Path("validation.json"))
        """
        import json

        data = {
            "protocol_name": self.protocol_name,
            "total_fields": self.total_fields,
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.info),
                "is_valid": not self.has_errors(),
            },
            "errors": [
                {
                    "type": err.error_type.value,
                    "severity": err.severity.value,
                    "field": err.field_name,
                    "message": err.message,
                    "suggestion": err.suggestion,
                    "line": err.line_number,
                    "context": err.context,
                }
                for err in self.errors
            ],
            "warnings": [
                {
                    "type": warn.error_type.value,
                    "severity": warn.severity.value,
                    "field": warn.field_name,
                    "message": warn.message,
                    "suggestion": warn.suggestion,
                    "line": warn.line_number,
                    "context": warn.context,
                }
                for warn in self.warnings
            ],
            "info": [
                {
                    "type": i.error_type.value,
                    "severity": i.severity.value,
                    "field": i.field_name,
                    "message": i.message,
                    "suggestion": i.suggestion,
                    "line": i.line_number,
                    "context": i.context,
                }
                for i in self.info
            ],
            "metadata": self.metadata,
        }

        output.write_text(json.dumps(data, indent=2))

    def export_text(self, output: Path) -> None:
        """Export validation report as human-readable text.

        Args:
            output: Output text file path.

        Example:
            >>> report.export_text(Path("validation.txt"))
        """
        lines = [
            f"Validation Report: {self.protocol_name}",
            "=" * 80,
            f"Total Fields: {self.total_fields}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            f"Info: {len(self.info)}",
            f"Status: {'INVALID' if self.has_errors() else 'VALID'}",
            "",
        ]

        if self.errors:
            lines.extend(["", "ERRORS:", "-" * 80])
            for err in self.errors:
                lines.append(f"[{err.error_type.value}] {err.message}")
                if err.field_name:
                    lines.append(f"  Field: {err.field_name}")
                if err.suggestion:
                    lines.append(f"  Suggestion: {err.suggestion}")
                lines.append("")

        if self.warnings:
            lines.extend(["", "WARNINGS:", "-" * 80])
            for warn in self.warnings:
                lines.append(f"[{warn.error_type.value}] {warn.message}")
                if warn.field_name:
                    lines.append(f"  Field: {warn.field_name}")
                if warn.suggestion:
                    lines.append(f"  Suggestion: {warn.suggestion}")
                lines.append("")

        if self.info:
            lines.extend(["", "INFO:", "-" * 80])
            for i in self.info:
                lines.append(f"[{i.error_type.value}] {i.message}")
                if i.field_name:
                    lines.append(f"  Field: {i.field_name}")
                if i.suggestion:
                    lines.append(f"  Suggestion: {i.suggestion}")
                lines.append("")

        output.write_text("\n".join(lines))


class ProtocolGrammarValidator:
    """Validates protocol specifications for consistency and correctness.

    Performs comprehensive validation including:
    - Field definition checking (overlaps, gaps, valid ranges)
    - Length field consistency
    - Checksum coverage validation
    - Enum value checking
    - Conditional field dependencies
    - State machine completeness
    - Grammar ambiguity detection

    Example:
        >>> validator = ProtocolGrammarValidator()
        >>> report = validator.validate(protocol_spec)
        >>> if report.has_errors():
        ...     print("Invalid specification")
        >>> report.export_json(Path("report.json"))
    """

    def __init__(
        self,
        check_alignment: bool = True,
        check_gaps: bool = True,
        check_state_machine: bool = True,
    ) -> None:
        """Initialize grammar validator.

        Args:
            check_alignment: Enable byte alignment warnings (default: True).
            check_gaps: Enable gap detection between fields (default: True).
            check_state_machine: Enable state machine validation (default: True).

        Example:
            >>> validator = ProtocolGrammarValidator(check_gaps=False)
        """
        self.check_alignment = check_alignment
        self.check_gaps = check_gaps
        self.check_state_machine = check_state_machine

    def validate(self, spec: ProtocolSpec) -> ValidationReport:
        """Validate complete protocol specification.

        Args:
            spec: Protocol specification to validate.

        Returns:
            Validation report with errors, warnings, and info.

        Example:
            >>> report = validator.validate(spec)
            >>> print(f"Errors: {len(report.errors)}")
        """
        report = ValidationReport(
            protocol_name=spec.name,
            total_fields=len(spec.fields),
        )

        # Validate field definitions
        self._validate_field_definitions(spec, report)

        # Validate field dependencies
        self._validate_dependencies(spec, report)

        # Validate checksums
        self._validate_checksums(spec, report)

        # Validate enums (from evidence)
        self._validate_enums(spec, report)

        # Validate state machine
        if self.check_state_machine and spec.state_machine is not None:
            self._validate_state_machine(spec, report)

        # Add metadata
        report.metadata["validator_config"] = {
            "check_alignment": self.check_alignment,
            "check_gaps": self.check_gaps,
            "check_state_machine": self.check_state_machine,
        }

        return report

    def _check_duplicate_names(self, spec: ProtocolSpec, report: ValidationReport) -> None:
        """Check for duplicate field names.

        Args:
            spec: Protocol specification.
            report: Validation report to populate.
        """
        field_names = [f.name for f in spec.fields]
        seen_names: set[str] = set()
        for name in field_names:
            if name in seen_names:
                report.errors.append(
                    ValidationError(
                        error_type=ErrorType.DUPLICATE_FIELD,
                        severity=ErrorSeverity.ERROR,
                        field_name=name,
                        message=f"Duplicate field name: '{name}'",
                        suggestion="Rename one of the duplicate fields to be unique",
                    )
                )
            seen_names.add(name)

    def _check_field_basic_validity(
        self, field_def: FieldHypothesis, report: ValidationReport
    ) -> None:
        """Check basic field validity (offset and length).

        Args:
            field_def: Field definition to check.
            report: Validation report to populate.
        """
        if field_def.offset < 0:
            report.errors.append(
                ValidationError(
                    error_type=ErrorType.INVALID_OFFSET,
                    severity=ErrorSeverity.ERROR,
                    field_name=field_def.name,
                    message=f"Field '{field_def.name}' has invalid offset: {field_def.offset}",
                    suggestion="Offset must be >= 0",
                    context={"offset": field_def.offset},
                )
            )

        if field_def.length <= 0:
            report.errors.append(
                ValidationError(
                    error_type=ErrorType.INVALID_LENGTH,
                    severity=ErrorSeverity.ERROR,
                    field_name=field_def.name,
                    message=f"Field '{field_def.name}' has invalid length: {field_def.length}",
                    suggestion="Length must be > 0",
                    context={"length": field_def.length},
                )
            )

    def _check_field_overlap_and_gap(
        self, field_def: FieldHypothesis, next_field: FieldHypothesis, report: ValidationReport
    ) -> None:
        """Check for overlaps and gaps between consecutive fields.

        Args:
            field_def: Current field.
            next_field: Next field.
            report: Validation report to populate.
        """
        current_end = field_def.offset + field_def.length
        next_start = next_field.offset

        if current_end > next_start:
            report.errors.append(
                ValidationError(
                    error_type=ErrorType.FIELD_OVERLAP,
                    severity=ErrorSeverity.ERROR,
                    field_name=field_def.name,
                    message=(
                        f"Field '{field_def.name}' (bytes {field_def.offset}-"
                        f"{current_end - 1}) overlaps with '{next_field.name}' "
                        f"(starts at byte {next_start})"
                    ),
                    suggestion=(
                        f"Move '{next_field.name}' to byte {current_end} or "
                        f"reduce '{field_def.name}' length"
                    ),
                    context={
                        "current_field": field_def.name,
                        "current_range": (field_def.offset, current_end - 1),
                        "next_field": next_field.name,
                        "next_offset": next_start,
                    },
                )
            )
        elif self.check_gaps and current_end < next_start:
            gap_size = next_start - current_end
            report.warnings.append(
                ValidationError(
                    error_type=ErrorType.FIELD_GAP,
                    severity=ErrorSeverity.WARNING,
                    field_name=field_def.name,
                    message=(
                        f"Gap of {gap_size} byte(s) between '{field_def.name}' "
                        f"and '{next_field.name}'"
                    ),
                    suggestion=("Add padding field or adjust offsets to eliminate gap"),
                    context={
                        "gap_start": current_end,
                        "gap_end": next_start - 1,
                        "gap_size": gap_size,
                    },
                )
            )

    def _check_field_alignment(self, field_def: FieldHypothesis, report: ValidationReport) -> None:
        """Check field alignment.

        Args:
            field_def: Field definition.
            report: Validation report to populate.
        """
        if self.check_alignment and field_def.length in {2, 4, 8}:
            if field_def.offset % field_def.length != 0:
                report.info.append(
                    ValidationError(
                        error_type=ErrorType.ALIGNMENT_WARNING,
                        severity=ErrorSeverity.INFO,
                        field_name=field_def.name,
                        message=(
                            f"Field '{field_def.name}' ({field_def.length}-byte) is not "
                            f"aligned to {field_def.length}-byte boundary (offset: "
                            f"{field_def.offset})"
                        ),
                        suggestion=(
                            f"Consider aligning to offset "
                            f"{(field_def.offset // field_def.length + 1) * field_def.length}"
                        ),
                        context={
                            "offset": field_def.offset,
                            "alignment": field_def.length,
                        },
                    )
                )

    def _validate_field_definitions(self, spec: ProtocolSpec, report: ValidationReport) -> None:
        """Validate field definitions for overlaps, gaps, and invalid ranges.

        Args:
            spec: Protocol specification.
            report: Validation report to populate.
        """
        # Check for duplicate field names
        self._check_duplicate_names(spec, report)

        # Sort fields by offset for sequential validation
        sorted_fields = sorted(spec.fields, key=lambda f: f.offset)

        for i, field_def in enumerate(sorted_fields):
            # Check basic validity
            self._check_field_basic_validity(field_def, report)

            # Check overlap and gap with next field
            if i < len(sorted_fields) - 1:
                self._check_field_overlap_and_gap(field_def, sorted_fields[i + 1], report)

            # Check alignment
            self._check_field_alignment(field_def, report)

    def _validate_dependencies(self, spec: ProtocolSpec, report: ValidationReport) -> None:
        """Validate dependencies between fields (length fields, etc.).

        Args:
            spec: Protocol specification.
            report: Validation report to populate.
        """
        # Build field name -> field mapping
        field_map = {f.name: f for f in spec.fields}

        for field_def in spec.fields:
            # Check if field has dependency in evidence
            if "depends_on" in field_def.evidence:
                dep_name = field_def.evidence["depends_on"]
                if dep_name not in field_map:
                    report.errors.append(
                        ValidationError(
                            error_type=ErrorType.INVALID_DEPENDENCY,
                            severity=ErrorSeverity.ERROR,
                            field_name=field_def.name,
                            message=(
                                f"Field '{field_def.name}' depends on non-existent field "
                                f"'{dep_name}'"
                            ),
                            suggestion=f"Add field '{dep_name}' or remove dependency",
                            context={"dependency": dep_name},
                        )
                    )

            # Check length field references
            if "references" in field_def.evidence:
                ref_name = field_def.evidence["references"]
                if ref_name not in field_map:
                    report.errors.append(
                        ValidationError(
                            error_type=ErrorType.INVALID_DEPENDENCY,
                            severity=ErrorSeverity.ERROR,
                            field_name=field_def.name,
                            message=(
                                f"Length field '{field_def.name}' references non-existent "
                                f"field '{ref_name}'"
                            ),
                            suggestion=f"Add field '{ref_name}' or remove reference",
                            context={"reference": ref_name},
                        )
                    )

    def _validate_checksums(self, spec: ProtocolSpec, report: ValidationReport) -> None:
        """Validate checksum field coverage ranges.

        Args:
            spec: Protocol specification.
            report: Validation report to populate.
        """
        for field_def in spec.fields:
            if field_def.field_type == "checksum":
                # Check if range is specified in evidence
                if "range" in field_def.evidence:
                    range_tuple = field_def.evidence["range"]
                    if not isinstance(range_tuple, (list, tuple)) or len(range_tuple) != 2:
                        report.errors.append(
                            ValidationError(
                                error_type=ErrorType.CHECKSUM_RANGE,
                                severity=ErrorSeverity.ERROR,
                                field_name=field_def.name,
                                message=(
                                    f"Checksum '{field_def.name}' has invalid range format: "
                                    f"{range_tuple}"
                                ),
                                suggestion="Range must be (start_byte, end_byte) tuple",
                                context={"range": range_tuple},
                            )
                        )
                        continue

                    start_byte, end_byte = range_tuple

                    # Validate range bounds
                    if start_byte < 0 or end_byte < start_byte:
                        report.errors.append(
                            ValidationError(
                                error_type=ErrorType.CHECKSUM_RANGE,
                                severity=ErrorSeverity.ERROR,
                                field_name=field_def.name,
                                message=(
                                    f"Checksum '{field_def.name}' has invalid range: "
                                    f"({start_byte}, {end_byte})"
                                ),
                                suggestion="Range must be (start >= 0, end >= start)",
                                context={"range": (start_byte, end_byte)},
                            )
                        )

                    # Check if checksum covers itself (should not)
                    if start_byte <= field_def.offset < end_byte:
                        report.warnings.append(
                            ValidationError(
                                error_type=ErrorType.CHECKSUM_RANGE,
                                severity=ErrorSeverity.WARNING,
                                field_name=field_def.name,
                                message=(
                                    f"Checksum '{field_def.name}' at byte {field_def.offset} "
                                    f"is within its own coverage range ({start_byte}, {end_byte})"
                                ),
                                suggestion=("Checksum should typically not cover its own location"),
                                context={
                                    "checksum_offset": field_def.offset,
                                    "coverage_range": (start_byte, end_byte),
                                },
                            )
                        )

    def _check_enum_duplicates(
        self,
        field_def: FieldHypothesis,
        enum_values: dict[str, Any],
        report: ValidationReport,
    ) -> None:
        """Check for duplicate enum values.

        Args:
            field_def: Field definition
            enum_values: Enum values dict
            report: Validation report to populate
        """
        value_to_names: dict[Any, list[str]] = {}
        for name, value in enum_values.items():
            if value not in value_to_names:
                value_to_names[value] = []
            value_to_names[value].append(name)

        for value, names in value_to_names.items():
            if len(names) > 1:
                report.warnings.append(
                    ValidationError(
                        error_type=ErrorType.ENUM_DUPLICATE,
                        severity=ErrorSeverity.WARNING,
                        field_name=field_def.name,
                        message=(
                            f"Enum in '{field_def.name}' has duplicate value {value} "
                            f"for names: {', '.join(names)}"
                        ),
                        suggestion="Ensure each enum value is unique",
                        context={"value": value, "names": names},
                    )
                )

    def _check_enum_gaps(
        self,
        field_def: FieldHypothesis,
        enum_values: dict[str, Any],
        report: ValidationReport,
    ) -> None:
        """Check for gaps in sequential enums.

        Args:
            field_def: Field definition
            enum_values: Enum values dict
            report: Validation report to populate
        """
        if not all(isinstance(v, int) for v in enum_values.values()):
            return

        int_values = sorted([v for v in enum_values.values() if isinstance(v, int)])
        if not int_values:
            return

        min_val, max_val = int_values[0], int_values[-1]
        expected_range = set(range(min_val, max_val + 1))
        actual_set = set(int_values)
        missing = expected_range - actual_set

        if missing and len(missing) <= 5:  # Only report small gaps
            report.info.append(
                ValidationError(
                    error_type=ErrorType.ENUM_GAP,
                    severity=ErrorSeverity.INFO,
                    field_name=field_def.name,
                    message=(
                        f"Enum in '{field_def.name}' has gaps in value range: "
                        f"missing {sorted(missing)}"
                    ),
                    suggestion="Add missing enum values or verify range",
                    context={"missing_values": sorted(missing)},
                )
            )

    def _validate_enums(self, spec: ProtocolSpec, report: ValidationReport) -> None:
        """Validate enum values from field evidence.

        Args:
            spec: Protocol specification.
            report: Validation report to populate.
        """
        for field_def in spec.fields:
            if "enum_values" not in field_def.evidence:
                continue

            enum_values = field_def.evidence["enum_values"]
            if not isinstance(enum_values, dict):
                continue

            # Validate enum properties
            self._check_enum_duplicates(field_def, enum_values, report)
            self._check_enum_gaps(field_def, enum_values, report)

    def _validate_state_machine(self, spec: ProtocolSpec, report: ValidationReport) -> None:
        """Validate state machine completeness and reachability.

        Args:
            spec: Protocol specification.
            report: Validation report to populate.
        """
        sm = spec.state_machine
        if sm is None:
            return

        # Verify state machine structure
        if not self._check_state_machine_format(sm, report):
            return

        states = sm.states
        transitions = sm.transitions

        # Check state reachability
        self._check_unreachable_states(sm, states, transitions, report)

        # Check for dead-end states
        self._check_dead_end_states(sm, states, transitions, report)

    def _check_state_machine_format(self, sm: Any, report: ValidationReport) -> bool:
        """Check if state machine has required attributes.

        Args:
            sm: State machine object.
            report: Validation report to populate with errors.

        Returns:
            True if format is valid, False otherwise.
        """
        if not hasattr(sm, "states") or not hasattr(sm, "transitions"):
            report.warnings.append(
                ValidationError(
                    error_type=ErrorType.AMBIGUOUS_GRAMMAR,
                    severity=ErrorSeverity.WARNING,
                    field_name=None,
                    message="State machine format not recognized, skipping validation",
                    suggestion="Ensure state machine has 'states' and 'transitions' attributes",
                )
            )
            return False
        return True

    def _check_unreachable_states(
        self, sm: Any, states: Any, transitions: Any, report: ValidationReport
    ) -> None:
        """Find and report unreachable states.

        Args:
            sm: State machine with initial_state attribute.
            states: Collection of all states.
            transitions: Collection of transitions.
            report: Validation report to populate.
        """
        if not hasattr(sm, "initial_state"):
            return

        # Build reachability set using BFS
        reachable = self._find_reachable_states(sm.initial_state, transitions)

        # Report unreachable states
        unreachable = set(states) - reachable
        for state in unreachable:
            report.warnings.append(
                ValidationError(
                    error_type=ErrorType.UNREACHABLE_STATE,
                    severity=ErrorSeverity.WARNING,
                    field_name=None,
                    message=f"State '{state}' is unreachable from initial state",
                    suggestion="Add transition to this state or remove it",
                    context={"state": state, "initial_state": sm.initial_state},
                )
            )

    def _find_reachable_states(self, initial_state: Any, transitions: Any) -> set[Any]:
        """Find all states reachable from initial state using BFS.

        Args:
            initial_state: Starting state.
            transitions: Collection of transitions with source/target attributes.

        Returns:
            Set of reachable states.
        """
        reachable = {initial_state}
        queue = [initial_state]

        while queue:
            current = queue.pop(0)
            for trans in transitions:
                if hasattr(trans, "source") and trans.source == current:
                    target = trans.target if hasattr(trans, "target") else None
                    if target and target not in reachable:
                        reachable.add(target)
                        queue.append(target)

        return reachable

    def _check_dead_end_states(
        self, sm: Any, states: Any, transitions: Any, report: ValidationReport
    ) -> None:
        """Find and report states with no outgoing transitions.

        Args:
            sm: State machine with optional final_states attribute.
            states: Collection of all states.
            transitions: Collection of transitions.
            report: Validation report to populate.
        """
        # Identify states with outgoing transitions
        states_with_transitions = {
            trans.source for trans in transitions if hasattr(trans, "source")
        }

        # Get designated final states
        final_states = set()
        if hasattr(sm, "final_states"):
            final_states = set(sm.final_states)

        # Report dead ends (states without transitions and not marked as final)
        for state in states:
            if state not in states_with_transitions and state not in final_states:
                report.warnings.append(
                    ValidationError(
                        error_type=ErrorType.MISSING_TRANSITION,
                        severity=ErrorSeverity.WARNING,
                        field_name=None,
                        message=f"State '{state}' has no outgoing transitions (dead end)",
                        suggestion="Add transitions from this state or mark as final state",
                        context={"state": state},
                    )
                )
