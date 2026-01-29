"""JEDEC memory interface timing compliance tests.

This module tests compliance with JEDEC memory standards:
- JESD79-4: DDR4 SDRAM Standard
- JESD209-4: LPDDR4 SDRAM Standard

These tests validate timing parameter constraints and signal
integrity requirements for memory interface analysis.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.compliance


@pytest.mark.compliance
class TestDDR4TimingCompliance:
    """Test DDR4 timing parameter compliance per JESD79-4."""

    def test_tRCD_minimum(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tRCD (RAS to CAS Delay) meets minimum specification."""
        tRCD = jedec_ddr4_timing["tRCD"]
        assert tRCD["min"] is not None
        assert tRCD["min"] >= 10.0  # DDR4 typical minimum

    def test_tRP_minimum(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tRP (Row Precharge Time) meets minimum specification."""
        tRP = jedec_ddr4_timing["tRP"]
        assert tRP["min"] is not None
        assert tRP["min"] >= 10.0  # DDR4 typical minimum

    def test_tRAS_range(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tRAS (Row Active Time) is within valid range."""
        tRAS = jedec_ddr4_timing["tRAS"]
        assert tRAS["min"] is not None
        assert tRAS["max"] is not None
        assert tRAS["min"] < tRAS["max"]
        assert tRAS["max"] <= 70000  # Maximum per spec (70us)

    def test_tRC_relationship(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tRC >= tRAS + tRP per JEDEC specification."""
        tRC = jedec_ddr4_timing["tRC"]["min"]
        tRAS = jedec_ddr4_timing["tRAS"]["min"]
        tRP = jedec_ddr4_timing["tRP"]["min"]

        # tRC must be at least tRAS + tRP
        assert tRC is not None
        assert tRAS is not None
        assert tRP is not None
        assert tRC >= tRAS + tRP - 0.1  # Allow small tolerance

    def test_tRFC_density_scaling(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tRFC scales with memory density."""
        tRFC = jedec_ddr4_timing["tRFC"]["min"]
        assert tRFC is not None

        # JEDEC tRFC values (ns) by density:
        # 2Gb: 160ns, 4Gb: 260ns, 8Gb: 350ns, 16Gb: 550ns
        expected_tRFC_ranges = {
            "2Gb": (150, 170),
            "4Gb": (250, 270),
            "8Gb": (340, 360),
            "16Gb": (540, 560),
        }

        # Verify our fixture value is within a valid range
        valid_ranges = list(expected_tRFC_ranges.values())
        in_valid_range = any(low <= tRFC <= high for low, high in valid_ranges)
        assert in_valid_range, f"tRFC {tRFC} not in expected JEDEC ranges"

    def test_tWR_minimum(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tWR (Write Recovery Time) meets minimum."""
        tWR = jedec_ddr4_timing["tWR"]["min"]
        assert tWR is not None
        assert tWR >= 15.0  # DDR4 minimum

    def test_tWTR_short_vs_long(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tWTR_L >= tWTR_S per specification."""
        tWTR_S = jedec_ddr4_timing["tWTR_S"]["min"]
        tWTR_L = jedec_ddr4_timing["tWTR_L"]["min"]

        assert tWTR_S is not None
        assert tWTR_L is not None
        assert tWTR_L >= tWTR_S

    def test_tCCD_short_vs_long(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """tCCD_L >= tCCD_S per specification."""
        tCCD_S = jedec_ddr4_timing["tCCD_S"]["min"]
        tCCD_L = jedec_ddr4_timing["tCCD_L"]["min"]

        assert tCCD_S is not None
        assert tCCD_L is not None
        assert tCCD_L >= tCCD_S


@pytest.mark.compliance
class TestDDR4VoltageCompliance:
    """Test DDR4 voltage specification compliance."""

    def test_vdd_nominal(self, jedec_ddr4_voltage: dict[str, dict[str, float]]) -> None:
        """VDD nominal is 1.2V for DDR4."""
        vdd = jedec_ddr4_voltage["VDD"]
        assert vdd["nominal"] == 1.2

    def test_vdd_tolerance(self, jedec_ddr4_voltage: dict[str, dict[str, float]]) -> None:
        """VDD tolerance is +/- 5% per JEDEC."""
        vdd = jedec_ddr4_voltage["VDD"]
        tolerance = (vdd["max"] - vdd["min"]) / vdd["nominal"] / 2
        assert tolerance <= 0.06  # 5% + margin

    def test_vddq_matches_vdd(self, jedec_ddr4_voltage: dict[str, dict[str, float]]) -> None:
        """VDDQ should match VDD for DDR4."""
        vdd = jedec_ddr4_voltage["VDD"]
        vddq = jedec_ddr4_voltage["VDDQ"]

        assert vdd["nominal"] == vddq["nominal"]
        assert vdd["min"] == vddq["min"]
        assert vdd["max"] == vddq["max"]

    def test_vpp_nominal(self, jedec_ddr4_voltage: dict[str, dict[str, float]]) -> None:
        """VPP nominal is 2.5V for DDR4."""
        vpp = jedec_ddr4_voltage["VPP"]
        assert vpp["nominal"] == 2.5

    def test_input_threshold_levels(self, jedec_ddr4_voltage: dict[str, dict[str, float]]) -> None:
        """Input threshold levels are properly separated."""
        vih = jedec_ddr4_voltage["VIH_DC"]["min"]
        vil = jedec_ddr4_voltage["VIL_DC"]["max"]

        assert vih is not None
        assert vil is not None
        assert vih > vil  # Proper separation for noise margin

    def test_output_levels(self, jedec_ddr4_voltage: dict[str, dict[str, float]]) -> None:
        """Output voltage levels have proper separation."""
        voh = jedec_ddr4_voltage["VOH"]["min"]
        vol = jedec_ddr4_voltage["VOL"]["max"]

        assert voh is not None
        assert vol is not None
        assert voh > vol  # Output high must be greater than output low


@pytest.mark.compliance
class TestSDRAMCommandEncoding:
    """Test SDRAM command encoding compliance."""

    def test_nop_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """NOP command has correct encoding."""
        nop = jedec_sdram_commands["NOP"]
        assert nop["RAS"] == 1
        assert nop["CAS"] == 1
        assert nop["WE"] == 1
        assert nop["CS"] == 0  # CS must be active

    def test_activate_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """ACTIVATE command has correct encoding."""
        act = jedec_sdram_commands["ACTIVATE"]
        assert act["RAS"] == 0  # RAS active
        assert act["CAS"] == 1
        assert act["WE"] == 1

    def test_read_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """READ command has correct encoding."""
        read = jedec_sdram_commands["READ"]
        assert read["RAS"] == 1
        assert read["CAS"] == 0  # CAS active
        assert read["WE"] == 1

    def test_write_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """WRITE command has correct encoding."""
        write = jedec_sdram_commands["WRITE"]
        assert write["RAS"] == 1
        assert write["CAS"] == 0  # CAS active
        assert write["WE"] == 0  # WE active

    def test_precharge_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """PRECHARGE command has correct encoding."""
        pre = jedec_sdram_commands["PRECHARGE"]
        assert pre["RAS"] == 0  # RAS active
        assert pre["CAS"] == 1
        assert pre["WE"] == 0  # WE active

    def test_refresh_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """REFRESH command has correct encoding."""
        ref = jedec_sdram_commands["REFRESH"]
        assert ref["RAS"] == 0  # RAS active
        assert ref["CAS"] == 0  # CAS active
        assert ref["WE"] == 1

    def test_mrs_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """Mode Register Set command has correct encoding."""
        mrs = jedec_sdram_commands["MRS"]
        assert mrs["RAS"] == 0
        assert mrs["CAS"] == 0
        assert mrs["WE"] == 0

    def test_deselect_encoding(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """DESELECT (CS high) disables the chip."""
        desel = jedec_sdram_commands["DESELECT"]
        assert desel["CS"] == 1  # CS inactive

    def test_all_commands_unique(self, jedec_sdram_commands: dict[str, dict[str, int]]) -> None:
        """All command encodings are unique (except DESELECT)."""
        encodings = set()
        for cmd_name, cmd in jedec_sdram_commands.items():
            if cmd_name == "DESELECT":
                continue  # DESELECT is special (CS=1)
            encoding = (cmd["RAS"], cmd["CAS"], cmd["WE"])
            assert encoding not in encodings, f"Duplicate encoding for {cmd_name}"
            encodings.add(encoding)


@pytest.mark.compliance
class TestDDRClockGeneration:
    """Test DDR clock signal generation for compliance testing."""

    def test_clock_frequency_accuracy(self, generate_ddr_clock: callable) -> None:
        """Generated clock frequency matches specification."""
        signal, metadata = generate_ddr_clock(frequency_mhz=1200)

        assert metadata["frequency_mhz"] == 1200
        assert abs(metadata["period_ns"] - 0.833) < 0.01

    def test_clock_duty_cycle(self, generate_ddr_clock: callable) -> None:
        """Clock duty cycle is approximately 50%."""
        signal, _ = generate_ddr_clock(frequency_mhz=1200, duration_ns=100)

        # Count high and low samples
        high_samples = np.sum(signal > 0.6)  # Threshold at 0.6V
        total_samples = len(signal)

        duty_cycle = high_samples / total_samples
        assert 0.45 <= duty_cycle <= 0.55, f"Duty cycle {duty_cycle} not ~50%"

    def test_clock_voltage_levels(self, generate_ddr_clock: callable) -> None:
        """Clock voltage levels are within DDR4 spec."""
        signal, _ = generate_ddr_clock(frequency_mhz=1200)

        high_level = np.max(signal)
        low_level = np.min(signal)

        # DDR4 uses 1.2V signaling
        assert high_level <= 1.3  # Allow some margin
        assert low_level >= -0.1  # Allow some undershoot

    def test_clock_sample_count(self, generate_ddr_clock: callable) -> None:
        """Generated signal has expected sample count."""
        signal, metadata = generate_ddr_clock(
            frequency_mhz=1200, duration_ns=1000, sample_rate_ghz=10
        )

        expected_samples = int(1000e-9 * 10e9)  # duration * sample_rate
        assert len(signal) == expected_samples
        assert metadata["num_samples"] == expected_samples


@pytest.mark.compliance
class TestTimingValidation:
    """Test timing validation utilities."""

    def test_timing_parameter_bounds(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """All timing parameters have sensible bounds."""
        for param_name, bounds in jedec_ddr4_timing.items():
            min_val = bounds.get("min")
            max_val = bounds.get("max")

            # At least one bound should be defined
            assert min_val is not None or max_val is not None, f"{param_name} has no bounds"

            # If both defined, max > min
            if min_val is not None and max_val is not None:
                assert max_val >= min_val, f"{param_name}: max < min"

    def test_timing_parameter_units(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """All timing parameters are in nanoseconds (reasonable range)."""
        for param_name, bounds in jedec_ddr4_timing.items():
            min_val = bounds.get("min")
            if min_val is not None:
                # Timing values should be positive and reasonable
                assert min_val > 0, f"{param_name} min should be positive"
                assert min_val < 1_000_000, f"{param_name} min too large (not in ns?)"

    def test_timing_margin_calculation(
        self, jedec_ddr4_timing: dict[str, dict[str, float]]
    ) -> None:
        """Calculate setup/hold timing margins."""

        def calculate_margin(
            actual_ns: float, spec_min_ns: float, spec_max_ns: float | None = None
        ) -> dict[str, float]:
            """Calculate timing margins."""
            result = {"setup_margin": actual_ns - spec_min_ns}
            if spec_max_ns is not None:
                result["hold_margin"] = spec_max_ns - actual_ns
            return result

        # Example: tRCD with 15ns actual vs 13.75ns minimum
        tRCD_spec = jedec_ddr4_timing["tRCD"]
        actual_tRCD = 15.0
        margins = calculate_margin(actual_tRCD, tRCD_spec["min"])

        assert margins["setup_margin"] > 0, "Timing violation: below minimum"
        assert margins["setup_margin"] == pytest.approx(1.25, rel=0.01)


@pytest.mark.compliance
class TestCommandSequenceValidation:
    """Test DDR command sequence validation."""

    def test_activate_to_read_timing(
        self,
        jedec_ddr4_timing: dict[str, dict[str, float]],
        jedec_sdram_commands: dict[str, dict[str, int]],
    ) -> None:
        """ACTIVATE to READ requires tRCD delay."""
        tRCD_min = jedec_ddr4_timing["tRCD"]["min"]

        # Validate that tRCD is non-zero
        assert tRCD_min > 0

        # A sequence checker would verify:
        # ACTIVATE -> wait tRCD -> READ
        sequence = [
            (0, jedec_sdram_commands["ACTIVATE"]),
            (tRCD_min, jedec_sdram_commands["READ"]),
        ]
        assert len(sequence) == 2

    def test_precharge_to_activate_timing(
        self,
        jedec_ddr4_timing: dict[str, dict[str, float]],
        jedec_sdram_commands: dict[str, dict[str, int]],
    ) -> None:
        """PRECHARGE to ACTIVATE requires tRP delay."""
        tRP_min = jedec_ddr4_timing["tRP"]["min"]

        # Validate that tRP is non-zero
        assert tRP_min > 0

        # A sequence checker would verify:
        # PRECHARGE -> wait tRP -> ACTIVATE
        sequence = [
            (0, jedec_sdram_commands["PRECHARGE"]),
            (tRP_min, jedec_sdram_commands["ACTIVATE"]),
        ]
        assert len(sequence) == 2

    def test_refresh_interval(self, jedec_ddr4_timing: dict[str, dict[str, float]]) -> None:
        """Refresh must occur within tREFI interval."""
        # DDR4 tREFI is typically 7.8us (for 1x refresh)
        tREFI_typical_us = 7.8
        tRFC = jedec_ddr4_timing["tRFC"]["min"]

        # tRFC must be less than tREFI for refresh to complete
        assert tRFC < tREFI_typical_us * 1000, "tRFC exceeds refresh interval"
