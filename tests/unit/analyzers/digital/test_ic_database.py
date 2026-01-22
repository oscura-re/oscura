"""Tests for IC timing database and identification."""

import pytest

from oscura.analyzers.digital.ic_database import (
    IC_DATABASE,
    ICTiming,
    identify_ic,
    validate_ic_timing,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


class TestICDatabase:
    """Test IC timing database."""

    def test_database_exists(self):
        """Test that database is populated."""
        assert len(IC_DATABASE) > 0
        assert "74LS74" in IC_DATABASE
        assert "74LS00" in IC_DATABASE
        assert "74HC74" in IC_DATABASE

    def test_ic_timing_structure(self):
        """Test IC timing data structure."""
        ic = IC_DATABASE["74LS74"]
        assert isinstance(ic, ICTiming)
        assert ic.part_number == "74LS74"
        assert ic.description == "Dual D-type flip-flop"
        assert ic.family == "LS-TTL"
        assert ic.vcc_nom == 5.0
        assert isinstance(ic.timing, dict)
        assert "t_pd" in ic.timing
        assert "t_su" in ic.timing
        assert "t_h" in ic.timing

    def test_vintage_logic_families(self):
        """Test that vintage ICs are included."""
        # Check standard TTL
        assert "7400" in IC_DATABASE
        assert IC_DATABASE["7400"].family == "TTL"

        # Check 4000 series CMOS
        assert "4011" in IC_DATABASE
        assert IC_DATABASE["4011"].family == "CMOS_5V"

    def test_timing_values_are_reasonable(self):
        """Test that timing values are in reasonable ranges."""
        for ic_name, ic_spec in IC_DATABASE.items():
            for param, value in ic_spec.timing.items():
                # All timing values should be positive
                assert value > 0, f"{ic_name}.{param} should be positive"
                # Should be in nanosecond to microsecond range
                assert 1e-12 < value < 1e-3, f"{ic_name}.{param} out of range"


class TestIdentifyIC:
    """Test IC identification from measured timings."""

    def test_identify_exact_match(self):
        """Test identification with exact timing match."""
        # 74LS74 typical timings
        measured = {
            "t_pd": 25e-9,
            "t_su": 20e-9,
            "t_h": 5e-9,
        }

        ic_name, confidence = identify_ic(measured, tolerance=0.1)
        assert ic_name == "74LS74"
        assert confidence > 0.9

    def test_identify_with_tolerance(self):
        """Test identification with measurement variation."""
        # 74LS74 with 20% variation (at edge of 30% tolerance)
        measured = {
            "t_pd": 30e-9,  # 20% higher than typ
            "t_su": 24e-9,
            "t_h": 6e-9,
        }

        ic_name, confidence = identify_ic(measured, tolerance=0.3, min_confidence=0.3)
        assert ic_name == "74LS74"
        assert confidence > 0.3

    def test_identify_no_match(self):
        """Test identification when timings don't match."""
        # Arbitrary timings that don't match any IC
        measured = {
            "t_pd": 1e-6,  # Way too slow
            "t_su": 500e-9,
        }

        ic_name, confidence = identify_ic(measured, min_confidence=0.8)
        assert ic_name == "unknown"

    def test_identify_partial_params(self):
        """Test identification with partial parameters."""
        # Only propagation delay
        measured = {"t_pd": 25e-9}

        ic_name, confidence = identify_ic(measured, tolerance=0.2)
        # Should still identify, but with lower confidence
        assert confidence > 0

    def test_identify_74hc_vs_74ls(self):
        """Test distinguishing between HC and LS families."""
        # 74HC is faster than 74LS
        measured_hc = {"t_pd": 16e-9}
        measured_ls = {"t_pd": 25e-9}

        ic_hc, _ = identify_ic(measured_hc, tolerance=0.2)
        ic_ls, _ = identify_ic(measured_ls, tolerance=0.2)

        assert ic_hc == "74HC74"
        assert ic_ls == "74LS74"


class TestValidateICTiming:
    """Test IC timing validation."""

    def test_validate_passing(self):
        """Test validation with passing timings."""
        measured = {
            "t_pd": 25e-9,
            "t_su": 20e-9,
            "t_h": 5e-9,
        }

        results = validate_ic_timing("74LS74", measured, tolerance=0.3)

        assert "t_pd" in results
        assert results["t_pd"]["passes"] is True
        assert results["t_su"]["passes"] is True
        assert results["t_h"]["passes"] is True

    def test_validate_failing(self):
        """Test validation with failing timings."""
        measured = {
            "t_pd": 50e-9,  # Too slow (spec is 25ns typ)
            "t_su": 20e-9,
            "t_h": 5e-9,
        }

        results = validate_ic_timing("74LS74", measured, tolerance=0.3)

        assert results["t_pd"]["passes"] is False
        assert results["t_pd"]["error"] > 0.3

    def test_validate_unknown_param(self):
        """Test validation with unknown parameter."""
        measured = {
            "t_pd": 25e-9,
            "unknown_param": 100e-9,
        }

        results = validate_ic_timing("74LS74", measured)

        assert "t_pd" in results
        assert "unknown_param" in results
        assert results["unknown_param"]["passes"] is None

    def test_validate_nonexistent_ic(self):
        """Test validation with IC not in database."""
        with pytest.raises(KeyError):
            validate_ic_timing("NONEXISTENT", {"t_pd": 25e-9})

    def test_validate_error_calculation(self):
        """Test error percentage calculation."""
        measured = {"t_pd": 30e-9}  # Spec is 25ns
        results = validate_ic_timing("74LS74", measured, tolerance=0.5)

        error = results["t_pd"]["error"]
        # Error should be (30-25)/25 = 0.2 (20%)
        assert abs(error - 0.2) < 0.01


class TestICTimingValues:
    """Test specific IC timing values."""

    def test_74ls74_timing(self):
        """Test 74LS74 timing specifications."""
        ic = IC_DATABASE["74LS74"]
        assert ic.timing["t_pd"] == 25e-9
        assert ic.timing["t_su"] == 20e-9
        assert ic.timing["t_h"] == 5e-9
        assert ic.timing["t_w"] == 25e-9

    def test_74ls00_timing(self):
        """Test 74LS00 NAND gate timing."""
        ic = IC_DATABASE["74LS00"]
        assert ic.timing["t_pd"] == 10e-9
        assert ic.family == "LS-TTL"

    def test_74hc74_timing(self):
        """Test 74HC74 CMOS timing."""
        ic = IC_DATABASE["74HC74"]
        assert ic.timing["t_pd"] == 16e-9  # Faster than LS
        assert ic.family == "HC-CMOS"
        assert ic.vcc_range == (2.0, 6.0)  # Wide voltage range

    def test_4013_cmos_timing(self):
        """Test 4013 CMOS flip-flop."""
        ic = IC_DATABASE["4013"]
        assert ic.family == "CMOS_5V"
        # 4000 series is much slower than 74HC
        assert ic.timing["t_pd"] > 100e-9
        assert ic.vcc_range == (3.0, 18.0)  # Very wide voltage range
