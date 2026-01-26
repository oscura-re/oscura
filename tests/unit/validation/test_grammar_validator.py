"""Tests for protocol grammar validator.

Test Coverage:
- ValidationError dataclass
- ValidationReport with errors/warnings/info
- ProtocolGrammarValidator initialization
- Field definition validation (overlaps, gaps, invalid ranges)
- Field dependency validation
- Checksum range validation
- Enum validation (duplicates, gaps)
- State machine validation (reachability, completeness)
- JSON/text export
- Edge cases and error handling
"""

import pytest

from oscura.sessions import FieldHypothesis, ProtocolSpec
from oscura.validation.grammar_validator import (
    ErrorSeverity,
    ErrorType,
    ProtocolGrammarValidator,
    ValidationError,
    ValidationReport,
)


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = ValidationError(
            error_type=ErrorType.FIELD_OVERLAP,
            severity=ErrorSeverity.ERROR,
            field_name="checksum",
            message="Field overlaps with payload",
            suggestion="Move field to next byte",
        )

        assert error.error_type == ErrorType.FIELD_OVERLAP
        assert error.severity == ErrorSeverity.ERROR
        assert error.field_name == "checksum"
        assert error.message == "Field overlaps with payload"
        assert error.suggestion == "Move field to next byte"
        assert error.line_number is None
        assert error.context == {}

    def test_validation_error_with_context(self):
        """Test validation error with additional context."""
        error = ValidationError(
            error_type=ErrorType.INVALID_OFFSET,
            severity=ErrorSeverity.ERROR,
            field_name="data",
            message="Invalid offset",
            context={"offset": -5, "expected": 0},
        )

        assert error.context["offset"] == -5
        assert error.context["expected"] == 0

    def test_validation_error_with_line_number(self):
        """Test validation error with line number."""
        error = ValidationError(
            error_type=ErrorType.DUPLICATE_FIELD,
            severity=ErrorSeverity.ERROR,
            field_name="counter",
            message="Duplicate field name",
            line_number=42,
        )

        assert error.line_number == 42


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_empty_validation_report(self):
        """Test empty validation report."""
        report = ValidationReport()

        assert len(report.errors) == 0
        assert len(report.warnings) == 0
        assert len(report.info) == 0
        assert report.protocol_name == ""
        assert report.total_fields == 0
        assert not report.has_errors()
        assert not report.has_warnings()

    def test_validation_report_with_errors(self):
        """Test validation report with errors."""
        error = ValidationError(
            error_type=ErrorType.FIELD_OVERLAP,
            severity=ErrorSeverity.ERROR,
            field_name="field1",
            message="Overlap detected",
        )
        report = ValidationReport(errors=[error], protocol_name="TestProtocol", total_fields=4)

        assert report.has_errors()
        assert len(report.errors) == 1
        assert report.protocol_name == "TestProtocol"
        assert report.total_fields == 4

    def test_validation_report_with_warnings(self):
        """Test validation report with warnings only."""
        warning = ValidationError(
            error_type=ErrorType.FIELD_GAP,
            severity=ErrorSeverity.WARNING,
            field_name="field2",
            message="Gap detected",
        )
        report = ValidationReport(warnings=[warning])

        assert not report.has_errors()
        assert report.has_warnings()
        assert len(report.warnings) == 1

    def test_validation_report_all_issues(self):
        """Test getting all issues from report."""
        error = ValidationError(ErrorType.INVALID_OFFSET, ErrorSeverity.ERROR, "field1", "Error")
        warning = ValidationError(ErrorType.FIELD_GAP, ErrorSeverity.WARNING, "field2", "Warning")
        info = ValidationError(ErrorType.ALIGNMENT_WARNING, ErrorSeverity.INFO, "field3", "Info")

        report = ValidationReport(errors=[error], warnings=[warning], info=[info])
        all_issues = report.all_issues()

        assert len(all_issues) == 3
        assert error in all_issues
        assert warning in all_issues
        assert info in all_issues

    def test_export_json(self, tmp_path):
        """Test exporting validation report as JSON."""
        error = ValidationError(
            error_type=ErrorType.FIELD_OVERLAP,
            severity=ErrorSeverity.ERROR,
            field_name="field1",
            message="Overlap error",
            suggestion="Fix overlap",
            context={"offset": 5},
        )
        report = ValidationReport(errors=[error], protocol_name="TestProtocol", total_fields=3)

        output_file = tmp_path / "report.json"
        report.export_json(output_file)

        assert output_file.exists()
        import json

        data = json.loads(output_file.read_text())
        assert data["protocol_name"] == "TestProtocol"
        assert data["total_fields"] == 3
        assert data["summary"]["errors"] == 1
        assert data["summary"]["is_valid"] is False
        assert len(data["errors"]) == 1
        assert data["errors"][0]["type"] == "FIELD_OVERLAP"
        assert data["errors"][0]["severity"] == "ERROR"
        assert data["errors"][0]["message"] == "Overlap error"

    def test_export_text(self, tmp_path):
        """Test exporting validation report as text."""
        error = ValidationError(
            ErrorType.INVALID_LENGTH, ErrorSeverity.ERROR, "field1", "Invalid length"
        )
        warning = ValidationError(
            ErrorType.FIELD_GAP, ErrorSeverity.WARNING, "field2", "Gap detected"
        )
        report = ValidationReport(
            errors=[error],
            warnings=[warning],
            protocol_name="MyProtocol",
            total_fields=5,
        )

        output_file = tmp_path / "report.txt"
        report.export_text(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "MyProtocol" in content
        assert "Total Fields: 5" in content
        assert "Errors: 1" in content
        assert "Warnings: 1" in content
        assert "INVALID" in content
        assert "Invalid length" in content
        assert "Gap detected" in content


class TestProtocolGrammarValidator:
    """Tests for ProtocolGrammarValidator."""

    @pytest.fixture
    def valid_spec(self):
        """Create valid protocol specification for testing."""
        return ProtocolSpec(
            name="ValidProtocol",
            fields=[
                FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
                FieldHypothesis("cmd", 1, 1, "data", 0.85),
                FieldHypothesis("length", 2, 1, "data", 0.90),
                FieldHypothesis("payload", 3, 4, "data", 0.80),
                FieldHypothesis("checksum", 7, 1, "checksum", 0.95, {"range": (0, 7)}),
            ],
        )

    def test_validator_initialization(self):
        """Test validator initialization with default config."""
        validator = ProtocolGrammarValidator()

        assert validator.check_alignment is True
        assert validator.check_gaps is True
        assert validator.check_state_machine is True

    def test_validator_initialization_custom_config(self):
        """Test validator with custom configuration."""
        validator = ProtocolGrammarValidator(
            check_alignment=False, check_gaps=False, check_state_machine=False
        )

        assert validator.check_alignment is False
        assert validator.check_gaps is False
        assert validator.check_state_machine is False

    def test_validate_valid_spec(self, valid_spec):
        """Test validation of valid protocol specification."""
        validator = ProtocolGrammarValidator()
        report = validator.validate(valid_spec)

        assert not report.has_errors()
        assert report.protocol_name == "ValidProtocol"
        assert report.total_fields == 5

    def test_validate_field_overlap(self):
        """Test detection of overlapping fields."""
        spec = ProtocolSpec(
            name="OverlapProtocol",
            fields=[
                FieldHypothesis("field1", 0, 3, "data", 0.9),
                FieldHypothesis("field2", 2, 2, "data", 0.8),  # Overlaps at byte 2
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        assert any(e.error_type == ErrorType.FIELD_OVERLAP for e in report.errors)
        overlap_error = next(e for e in report.errors if e.error_type == ErrorType.FIELD_OVERLAP)
        assert "field1" in overlap_error.message
        assert "field2" in overlap_error.message

    def test_validate_field_gap(self):
        """Test detection of gaps between fields."""
        spec = ProtocolSpec(
            name="GapProtocol",
            fields=[
                FieldHypothesis("field1", 0, 2, "data", 0.9),
                FieldHypothesis("field2", 5, 2, "data", 0.8),  # Gap from byte 2 to 5
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_warnings()
        assert any(e.error_type == ErrorType.FIELD_GAP for e in report.warnings)
        gap_warning = next(w for w in report.warnings if w.error_type == ErrorType.FIELD_GAP)
        assert "3 byte(s)" in gap_warning.message

    def test_validate_no_gap_check(self):
        """Test disabling gap detection."""
        spec = ProtocolSpec(
            name="GapProtocol",
            fields=[
                FieldHypothesis("field1", 0, 2, "data", 0.9),
                FieldHypothesis("field2", 5, 2, "data", 0.8),
            ],
        )

        validator = ProtocolGrammarValidator(check_gaps=False)
        report = validator.validate(spec)

        assert not any(e.error_type == ErrorType.FIELD_GAP for e in report.warnings)

    def test_validate_invalid_offset(self):
        """Test detection of invalid negative offset."""
        spec = ProtocolSpec(
            name="InvalidOffsetProtocol",
            fields=[
                FieldHypothesis("bad_field", -5, 2, "data", 0.9),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        assert any(e.error_type == ErrorType.INVALID_OFFSET for e in report.errors)

    def test_validate_invalid_length(self):
        """Test detection of invalid zero/negative length."""
        spec = ProtocolSpec(
            name="InvalidLengthProtocol",
            fields=[
                FieldHypothesis("zero_length", 0, 0, "data", 0.9),
                FieldHypothesis("negative_length", 1, -3, "data", 0.8),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        invalid_length_errors = [
            e for e in report.errors if e.error_type == ErrorType.INVALID_LENGTH
        ]
        assert len(invalid_length_errors) == 2

    def test_validate_duplicate_field_names(self):
        """Test detection of duplicate field names."""
        spec = ProtocolSpec(
            name="DuplicateProtocol",
            fields=[
                FieldHypothesis("counter", 0, 1, "data", 0.9),
                FieldHypothesis("counter", 1, 1, "data", 0.8),  # Duplicate name
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        assert any(e.error_type == ErrorType.DUPLICATE_FIELD for e in report.errors)

    def test_validate_alignment_warning(self):
        """Test alignment warning for multi-byte fields."""
        spec = ProtocolSpec(
            name="AlignmentProtocol",
            fields=[
                FieldHypothesis("field1", 0, 1, "data", 0.9),
                FieldHypothesis("uint16", 1, 2, "data", 0.8),  # Not aligned to 2-byte boundary
                FieldHypothesis("uint32", 5, 4, "data", 0.7),  # Not aligned to 4-byte boundary
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert len(report.info) >= 2
        alignment_infos = [i for i in report.info if i.error_type == ErrorType.ALIGNMENT_WARNING]
        assert len(alignment_infos) == 2

    def test_validate_no_alignment_check(self):
        """Test disabling alignment warnings."""
        spec = ProtocolSpec(
            name="AlignmentProtocol",
            fields=[
                FieldHypothesis("uint16", 1, 2, "data", 0.8),
            ],
        )

        validator = ProtocolGrammarValidator(check_alignment=False)
        report = validator.validate(spec)

        assert not any(i.error_type == ErrorType.ALIGNMENT_WARNING for i in report.info)

    def test_validate_invalid_dependency(self):
        """Test detection of invalid field dependencies."""
        spec = ProtocolSpec(
            name="DependencyProtocol",
            fields=[
                FieldHypothesis("field1", 0, 2, "data", 0.9, {"depends_on": "nonexistent"}),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        assert any(e.error_type == ErrorType.INVALID_DEPENDENCY for e in report.errors)

    def test_validate_invalid_reference(self):
        """Test detection of invalid field references."""
        spec = ProtocolSpec(
            name="ReferenceProtocol",
            fields=[
                FieldHypothesis("length", 0, 1, "data", 0.9, {"references": "missing_field"}),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        assert any(e.error_type == ErrorType.INVALID_DEPENDENCY for e in report.errors)

    def test_validate_checksum_invalid_range_format(self):
        """Test detection of invalid checksum range format."""
        spec = ProtocolSpec(
            name="ChecksumProtocol",
            fields=[
                FieldHypothesis("checksum", 0, 1, "checksum", 0.95, {"range": "invalid"}),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        assert any(e.error_type == ErrorType.CHECKSUM_RANGE for e in report.errors)

    def test_validate_checksum_invalid_range_bounds(self):
        """Test detection of invalid checksum range bounds."""
        spec = ProtocolSpec(
            name="ChecksumProtocol",
            fields=[
                FieldHypothesis("checksum", 5, 1, "checksum", 0.95, {"range": (-1, 10)}),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_errors()
        checksum_errors = [e for e in report.errors if e.error_type == ErrorType.CHECKSUM_RANGE]
        assert len(checksum_errors) > 0

    def test_validate_checksum_covers_itself(self):
        """Test warning when checksum covers its own location."""
        spec = ProtocolSpec(
            name="ChecksumProtocol",
            fields=[
                FieldHypothesis("data", 0, 5, "data", 0.9),
                FieldHypothesis(
                    "checksum", 5, 1, "checksum", 0.95, {"range": (0, 10)}
                ),  # Covers itself
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_warnings()
        assert any(w.error_type == ErrorType.CHECKSUM_RANGE for w in report.warnings)

    def test_validate_enum_duplicate_values(self):
        """Test detection of duplicate enum values."""
        spec = ProtocolSpec(
            name="EnumProtocol",
            fields=[
                FieldHypothesis(
                    "command",
                    0,
                    1,
                    "data",
                    0.9,
                    {"enum_values": {"READ": 1, "WRITE": 2, "ERASE": 1}},  # Duplicate value 1
                ),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_warnings()
        assert any(w.error_type == ErrorType.ENUM_DUPLICATE for w in report.warnings)

    def test_validate_enum_gaps(self):
        """Test detection of gaps in enum values."""
        spec = ProtocolSpec(
            name="EnumProtocol",
            fields=[
                FieldHypothesis(
                    "status",
                    0,
                    1,
                    "data",
                    0.9,
                    {"enum_values": {"OK": 0, "ERROR": 5}},  # Gap from 1 to 4
                ),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert len(report.info) > 0
        enum_gap_infos = [i for i in report.info if i.error_type == ErrorType.ENUM_GAP]
        assert len(enum_gap_infos) == 1

    def test_validate_enum_no_gaps_non_sequential(self):
        """Test enum without gaps when values are intentionally sparse."""
        spec = ProtocolSpec(
            name="EnumProtocol",
            fields=[
                FieldHypothesis(
                    "flags",
                    0,
                    1,
                    "data",
                    0.9,
                    {"enum_values": {"FLAG_A": 0, "FLAG_B": 10, "FLAG_C": 20}},  # Large gaps
                ),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        # Should not report gaps for intentionally sparse enums (gap > 5)
        enum_gap_infos = [i for i in report.info if i.error_type == ErrorType.ENUM_GAP]
        assert len(enum_gap_infos) == 0

    def test_validate_state_machine_unreachable(self):
        """Test detection of unreachable states."""
        from dataclasses import dataclass

        @dataclass
        class Transition:
            source: str
            target: str

        @dataclass
        class StateMachine:
            states: list[str]
            transitions: list[Transition]
            initial_state: str
            final_states: list[str]

        sm = StateMachine(
            states=["INIT", "READY", "ACTIVE", "UNREACHABLE"],
            transitions=[
                Transition("INIT", "READY"),
                Transition("READY", "ACTIVE"),
                Transition("ACTIVE", "READY"),
            ],
            initial_state="INIT",
            final_states=["ACTIVE"],
        )

        spec = ProtocolSpec(name="SMProtocol", fields=[], state_machine=sm)

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_warnings()
        unreachable_warnings = [
            w for w in report.warnings if w.error_type == ErrorType.UNREACHABLE_STATE
        ]
        assert len(unreachable_warnings) == 1
        assert "UNREACHABLE" in unreachable_warnings[0].message

    def test_validate_state_machine_missing_transitions(self):
        """Test detection of states with no outgoing transitions."""
        from dataclasses import dataclass

        @dataclass
        class Transition:
            source: str
            target: str

        @dataclass
        class StateMachine:
            states: list[str]
            transitions: list[Transition]
            initial_state: str
            final_states: list[str]

        sm = StateMachine(
            states=["INIT", "READY", "DEAD_END"],
            transitions=[
                Transition("INIT", "READY"),
                Transition("READY", "DEAD_END"),
                # DEAD_END has no outgoing transitions and is not final
            ],
            initial_state="INIT",
            final_states=[],
        )

        spec = ProtocolSpec(name="SMProtocol", fields=[], state_machine=sm)

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert report.has_warnings()
        missing_trans_warnings = [
            w for w in report.warnings if w.error_type == ErrorType.MISSING_TRANSITION
        ]
        assert len(missing_trans_warnings) == 1
        assert "DEAD_END" in missing_trans_warnings[0].message

    def test_validate_state_machine_disabled(self):
        """Test disabling state machine validation."""
        from dataclasses import dataclass

        @dataclass
        class Transition:
            source: str
            target: str

        @dataclass
        class StateMachine:
            states: list[str]
            transitions: list[Transition]
            initial_state: str

        sm = StateMachine(states=["INIT", "UNREACHABLE"], transitions=[], initial_state="INIT")

        spec = ProtocolSpec(name="SMProtocol", fields=[], state_machine=sm)

        validator = ProtocolGrammarValidator(check_state_machine=False)
        report = validator.validate(spec)

        # Should not check state machine
        assert not any(w.error_type == ErrorType.UNREACHABLE_STATE for w in report.warnings)

    def test_validate_state_machine_invalid_format(self):
        """Test handling of invalid state machine format."""
        spec = ProtocolSpec(
            name="SMProtocol",
            fields=[],
            state_machine={"invalid": "format"},  # Not a proper state machine object
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        # Should skip validation with warning
        assert any(w.error_type == ErrorType.AMBIGUOUS_GRAMMAR for w in report.warnings)

    def test_validate_empty_spec(self):
        """Test validation of empty protocol specification."""
        spec = ProtocolSpec(name="EmptyProtocol", fields=[])

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        assert not report.has_errors()
        assert report.total_fields == 0

    def test_validate_metadata_in_report(self):
        """Test that validator config is included in metadata."""
        spec = ProtocolSpec(name="TestProtocol", fields=[])

        validator = ProtocolGrammarValidator(check_alignment=False, check_gaps=False)
        report = validator.validate(spec)

        assert "validator_config" in report.metadata
        assert report.metadata["validator_config"]["check_alignment"] is False
        assert report.metadata["validator_config"]["check_gaps"] is False

    def test_integration_full_validation(self):
        """Test complete validation workflow with multiple error types."""
        spec = ProtocolSpec(
            name="ComplexProtocol",
            fields=[
                FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
                FieldHypothesis("counter", 1, 2, "data", 0.9),  # Misaligned 2-byte field
                FieldHypothesis("payload", 5, 3, "data", 0.8),  # Gap from byte 3 to 5
                FieldHypothesis(
                    "checksum", 8, 1, "checksum", 0.95, {"range": (0, 9)}
                ),  # Covers itself
                FieldHypothesis(
                    "status",
                    9,
                    1,
                    "data",
                    0.85,
                    {"enum_values": {"OK": 0, "WARN": 1, "ERR": 1}},  # Duplicate enum
                ),
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        # Should have alignment info, gap warning, checksum warning, enum warning
        assert len(report.info) >= 1  # Alignment
        assert len(report.warnings) >= 3  # Gap, checksum, enum
        assert not report.has_errors()

    def test_export_integration(self, tmp_path):
        """Test JSON and text export integration."""
        spec = ProtocolSpec(
            name="ExportProtocol",
            fields=[
                FieldHypothesis("field1", 0, 2, "data", 0.9),
                FieldHypothesis("field2", 2, -1, "data", 0.8),  # Error: invalid length
            ],
        )

        validator = ProtocolGrammarValidator()
        report = validator.validate(spec)

        # Export JSON
        json_file = tmp_path / "validation.json"
        report.export_json(json_file)
        assert json_file.exists()

        # Export text
        text_file = tmp_path / "validation.txt"
        report.export_text(text_file)
        assert text_file.exists()

        text_content = text_file.read_text()
        assert "ExportProtocol" in text_content
        assert "INVALID" in text_content
