"""Tests for vintage logic CSV export functionality."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest

from oscura.analyzers.digital.vintage_result import (
    BOMEntry,
    ICIdentificationResult,
    VintageLogicAnalysisResult,
)
from oscura.exporters.vintage_logic_csv import (
    export_all_vintage_logic_csv,
    export_bom_csv,
    export_ic_identification_csv,
    export_timing_measurements_csv,
    export_voltage_levels_csv,
)

pytestmark = [pytest.mark.unit, pytest.mark.exporter]


@pytest.fixture
def sample_result():
    """Create sample VintageLogicAnalysisResult for testing."""
    return VintageLogicAnalysisResult(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        source_file="test.csv",
        analysis_duration=1.5,
        detected_family="TTL",
        family_confidence=0.95,
        voltage_levels={"VCC": 5.0, "VIL": 0.8, "VIH": 2.0, "VOL": 0.4, "VOH": 2.4},
        identified_ics=[
            ICIdentificationResult(
                ic_name="74LS00",
                confidence=0.9,
                family="TTL",
                timing_params={"t_pd": 10e-9, "t_su": 5e-9, "t_h": 2e-9},
                validation={"t_pd": {"passes": True, "measured": 9e-9, "spec": 10e-9}},
            ),
            ICIdentificationResult(
                ic_name="74LS74",
                confidence=0.85,
                family="TTL",
                timing_params={"t_pd": 25e-9, "t_w": 20e-9},
                validation={"t_pd": {"passes": False, "measured": 30e-9, "spec": 25e-9}},
            ),
        ],
        timing_measurements={
            "CLK→Q_t_pd": 10e-9,
            "DATA_t_su": 5e-9,
            "DATA_t_h": 2e-9,
            "PULSE_t_w": 20e-9,
        },
        timing_paths=None,
        decoded_protocols=None,
        open_collector_detected=False,
        asymmetry_ratio=0.5,
        modern_replacements=[],
        bom=[
            BOMEntry("74HCT00", "Quad NAND gate", 2, "IC", "Modern replacement"),
            BOMEntry("74HCT74", "Dual D flip-flop", 1, "IC", None),
            BOMEntry("0.1uF", "Decoupling capacitor", 10, "Capacitor", "Per IC"),
        ],
        warnings=[],
        confidence_scores={},
    )


def read_csv_file(path: Path) -> list[list[str]]:
    """Helper to read CSV file and return rows."""
    with path.open("r") as f:
        return list(csv.reader(f))


class TestExportTimingMeasurements:
    """Test export_timing_measurements_csv function."""

    def test_export_basic(self, sample_result, tmp_path):
        """Test basic export of timing measurements."""
        output_file = tmp_path / "timing.csv"
        export_timing_measurements_csv(sample_result, output_file)

        assert output_file.exists()
        rows = read_csv_file(output_file)

        assert rows[0] == ["parameter", "measured_value_ns", "measurement_type"]
        assert len(rows) == 5  # Header + 4 measurements

    def test_export_content(self, sample_result, tmp_path):
        """Test exported content is correct."""
        output_file = tmp_path / "timing.csv"
        export_timing_measurements_csv(sample_result, output_file)

        rows = read_csv_file(output_file)
        data_rows = rows[1:]

        # Check measurement types are correctly identified
        types = {row[0]: row[2] for row in data_rows}
        assert "CLK→Q_t_pd" in types
        assert types["CLK→Q_t_pd"] == "propagation_delay"
        assert types["DATA_t_su"] == "setup_time"
        assert types["DATA_t_h"] == "hold_time"
        assert types["PULSE_t_w"] == "pulse_width"

    def test_export_with_string_path(self, sample_result, tmp_path):
        """Test export accepts string path."""
        output_file = str(tmp_path / "timing.csv")
        export_timing_measurements_csv(sample_result, output_file)

        assert Path(output_file).exists()

    def test_export_empty_measurements(self, tmp_path):
        """Test export with no timing measurements."""
        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=0.0,
            detected_family="Unknown",
            family_confidence=0.0,
            voltage_levels={},
            timing_measurements={},  # Empty
            bom=[],
            identified_ics=[],
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=0.0,
            modern_replacements=[],
            warnings=[],
            confidence_scores={},
        )

        output_file = tmp_path / "empty.csv"
        export_timing_measurements_csv(result, output_file)

        rows = read_csv_file(output_file)
        assert len(rows) == 1  # Only header


class TestExportICIdentification:
    """Test export_ic_identification_csv function."""

    def test_export_basic(self, sample_result, tmp_path):
        """Test basic IC identification export."""
        output_file = tmp_path / "ic_id.csv"
        export_ic_identification_csv(sample_result, output_file)

        assert output_file.exists()
        rows = read_csv_file(output_file)

        expected_header = [
            "ic_name",
            "confidence",
            "family",
            "t_pd_ns",
            "t_su_ns",
            "t_h_ns",
            "t_w_ns",
            "validation_status",
        ]
        assert rows[0] == expected_header
        assert len(rows) == 3  # Header + 2 ICs

    def test_export_validation_status(self, sample_result, tmp_path):
        """Test validation status is correctly determined."""
        output_file = tmp_path / "ic_id.csv"
        export_ic_identification_csv(sample_result, output_file)

        rows = read_csv_file(output_file)

        # First IC passes validation
        assert rows[1][7] == "PASS"
        # Second IC fails validation
        assert rows[2][7] == "FAIL"

    def test_export_missing_timing_params(self, sample_result, tmp_path):
        """Test export handles missing timing parameters."""
        output_file = tmp_path / "ic_id.csv"
        export_ic_identification_csv(sample_result, output_file)

        rows = read_csv_file(output_file)

        # First IC has t_pd, t_su, t_h but not t_w
        assert rows[1][3] != ""  # t_pd present
        assert rows[1][6] == ""  # t_w missing

        # Second IC has t_pd, t_w but not t_su, t_h
        assert rows[2][3] != ""  # t_pd present
        assert rows[2][4] == ""  # t_su missing
        assert rows[2][5] == ""  # t_h missing


class TestExportBOM:
    """Test export_bom_csv function."""

    def test_export_basic(self, sample_result, tmp_path):
        """Test basic BOM export."""
        output_file = tmp_path / "bom.csv"
        export_bom_csv(sample_result, output_file)

        assert output_file.exists()
        rows = read_csv_file(output_file)

        assert rows[0] == ["part_number", "description", "quantity", "category", "notes"]
        assert len(rows) == 4  # Header + 3 entries

    def test_export_content(self, sample_result, tmp_path):
        """Test BOM content is correct."""
        output_file = tmp_path / "bom.csv"
        export_bom_csv(sample_result, output_file)

        rows = read_csv_file(output_file)

        # Check first entry
        assert rows[1][0] == "74HCT00"
        assert rows[1][1] == "Quad NAND gate"
        assert rows[1][2] == "2"
        assert rows[1][3] == "IC"
        assert rows[1][4] == "Modern replacement"

        # Check entry with no notes
        assert rows[2][4] == ""


class TestExportVoltageLevels:
    """Test export_voltage_levels_csv function."""

    def test_export_basic(self, sample_result, tmp_path):
        """Test basic voltage levels export."""
        output_file = tmp_path / "voltage.csv"
        export_voltage_levels_csv(sample_result, output_file)

        assert output_file.exists()
        rows = read_csv_file(output_file)

        assert rows[0] == ["parameter", "voltage_v", "logic_family"]
        assert len(rows) == 6  # Header + 5 voltage levels

    def test_export_content(self, sample_result, tmp_path):
        """Test voltage content is correct."""
        output_file = tmp_path / "voltage.csv"
        export_voltage_levels_csv(sample_result, output_file)

        rows = read_csv_file(output_file)

        # Check all rows have correct family
        for row in rows[1:]:
            assert row[2] == "TTL"

        # Check VCC value
        vcc_row = next(r for r in rows[1:] if r[0] == "VCC")
        assert vcc_row[1] == "5.000"


class TestExportAll:
    """Test export_all_vintage_logic_csv function."""

    def test_export_all_basic(self, sample_result, tmp_path):
        """Test exporting all data types."""
        paths = export_all_vintage_logic_csv(sample_result, tmp_path)

        assert len(paths) == 4
        assert "timing_measurements" in paths
        assert "ic_identification" in paths
        assert "bom" in paths
        assert "voltage_levels" in paths

        # All files should exist
        for path in paths.values():
            assert path.exists()

    def test_export_all_with_prefix(self, sample_result, tmp_path):
        """Test export with filename prefix."""
        paths = export_all_vintage_logic_csv(sample_result, tmp_path, prefix="analysis_")

        for key, path in paths.items():
            assert path.name.startswith("analysis_")
            assert key in path.name

    def test_export_all_creates_directory(self, sample_result, tmp_path):
        """Test export creates output directory if needed."""
        nested_dir = tmp_path / "level1" / "level2"
        paths = export_all_vintage_logic_csv(sample_result, nested_dir)

        assert nested_dir.exists()
        assert len(paths) > 0

    def test_export_all_skips_empty_data(self, tmp_path):
        """Test export skips data types with no content."""
        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=0.0,
            detected_family="Unknown",
            family_confidence=0.0,
            voltage_levels={"VCC": 5.0},  # Only this has data
            timing_measurements={},
            bom=[],
            identified_ics=[],
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=0.0,
            modern_replacements=[],
            warnings=[],
            confidence_scores={},
        )

        paths = export_all_vintage_logic_csv(result, tmp_path)

        # Should only export voltage_levels
        assert len(paths) == 1
        assert "voltage_levels" in paths
        assert "timing_measurements" not in paths
        assert "bom" not in paths
        assert "ic_identification" not in paths

    def test_export_all_returns_correct_paths(self, sample_result, tmp_path):
        """Test returned paths match actual files."""
        paths = export_all_vintage_logic_csv(sample_result, tmp_path, prefix="test_")

        for key, path in paths.items():
            assert path.parent == tmp_path
            assert path.name == f"test_{key}.csv"
            assert path.exists()


class TestIntegration:
    """Integration tests for CSV export."""

    def test_full_workflow(self, sample_result, tmp_path):
        """Test complete export workflow."""
        # Export all
        paths = export_all_vintage_logic_csv(sample_result, tmp_path)

        # Read and verify each file has content
        for path in paths.values():
            rows = read_csv_file(path)
            assert len(rows) > 1  # Header + data

    def test_round_trip_preserves_data(self, sample_result, tmp_path):
        """Test exported data can be read back correctly."""
        # Export timing measurements
        timing_file = tmp_path / "timing.csv"
        export_timing_measurements_csv(sample_result, timing_file)

        # Read back
        rows = read_csv_file(timing_file)
        data = {row[0]: float(row[1]) for row in rows[1:]}

        # Verify values (convert back from ns to seconds for comparison)
        assert abs(data["CLK→Q_t_pd"] - 10.0) < 0.01
        assert abs(data["DATA_t_su"] - 5.0) < 0.01
