"""Unit tests for threshold configuration.

Tests CFG-005, CFG-006, CFG-007, CFG-008
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oscura.config.thresholds import (
    LogicFamily,
    ThresholdProfile,
    ThresholdRegistry,
    get_threshold_registry,
    get_user_logic_families_dir,
    load_logic_family,
    load_user_logic_families,
)
from oscura.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


class TestLogicFamily:
    """Test LogicFamily dataclass."""

    def test_create_ttl_family(self) -> None:
        """Test creating TTL logic family."""
        ttl = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
            VCC=5.0,
        )

        assert ttl.name == "TTL"
        assert ttl.VIH == 2.0
        assert ttl.VIL == 0.8
        assert ttl.VOH == 2.4
        assert ttl.VOL == 0.4
        assert ttl.VCC == 5.0

    def test_create_cmos_family(self) -> None:
        """Test creating CMOS logic family."""
        cmos = LogicFamily(
            name="CMOS_3V3",
            VIH=2.0,
            VIL=0.7,
            VOH=2.4,
            VOL=0.4,
            VCC=3.3,
        )

        assert cmos.VCC == 3.3

    def test_noise_margin_calculation(self) -> None:
        """Test automatic noise margin calculation."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        # NM_high = VOH - VIH = 2.4 - 2.0 = 0.4
        assert family.noise_margin_high == pytest.approx(0.4)
        # NM_low = VIL - VOL = 0.8 - 0.4 = 0.4
        assert family.noise_margin_low == pytest.approx(0.4)

    def test_explicit_noise_margins(self) -> None:
        """Test providing explicit noise margins."""
        family = LogicFamily(
            name="Custom",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
            noise_margin_high=0.5,
            noise_margin_low=0.5,
        )

        assert family.noise_margin_high == 0.5
        assert family.noise_margin_low == 0.5

    def test_invalid_thresholds_vih_vil(self) -> None:
        """Test error when VIH <= VIL."""
        with pytest.raises(ConfigurationError, match="VIH.*must be > VIL"):
            LogicFamily(
                name="Invalid",
                VIH=0.8,
                VIL=2.0,
                VOH=2.4,
                VOL=0.4,
            )

    def test_invalid_thresholds_voh_vol(self) -> None:
        """Test error when VOH <= VOL."""
        with pytest.raises(ConfigurationError, match="VOH.*must be > VOL"):
            LogicFamily(
                name="Invalid",
                VIH=2.0,
                VIL=0.8,
                VOH=0.4,
                VOL=2.4,
            )

    def test_get_threshold_50_percent(self) -> None:
        """Test getting threshold at 50%."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        # 50% between VIL (0.8) and VIH (2.0) = 1.4
        threshold = family.get_threshold(50.0)
        assert threshold == 1.4

    def test_get_threshold_0_percent(self) -> None:
        """Test getting threshold at 0% (VIL)."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        threshold = family.get_threshold(0.0)
        assert threshold == 0.8

    def test_get_threshold_100_percent(self) -> None:
        """Test getting threshold at 100% (VIH)."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        threshold = family.get_threshold(100.0)
        assert threshold == 2.0

    def test_get_threshold_25_percent(self) -> None:
        """Test getting threshold at 25%."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        # 25% between 0.8 and 2.0 = 0.8 + 0.25 * (2.0 - 0.8) = 1.1
        threshold = family.get_threshold(25.0)
        assert threshold == 1.1

    def test_temperature_derating(self) -> None:
        """Test temperature derating."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
            VCC=5.0,
        )

        # Derate for 50C (25C delta from nominal)
        derated = family.with_temperature_derating(50.0, derating_factor=0.002)

        # Factor = 1.0 - (25 * 0.002) = 0.95
        assert derated.name == "TTL@50.0C"
        assert pytest.approx(2.0 * 0.95) == derated.VIH
        assert pytest.approx(0.8 * 0.95) == derated.VIL
        assert derated.VCC == 5.0  # VCC unchanged

    def test_temperature_derating_cold(self) -> None:
        """Test temperature derating for cold temperature."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        # Derate for 0C (-25C delta from nominal)
        derated = family.with_temperature_derating(0.0, derating_factor=0.002)

        # Factor = 1.0 - (-25 * 0.002) = 1.05
        assert pytest.approx(2.0 * 1.05) == derated.VIH

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        assert family.VCC == 5.0
        assert family.description == ""
        assert family.temperature_range == (0, 70)
        assert family.source == "builtin"


class TestThresholdProfile:
    """Test ThresholdProfile dataclass."""

    def test_create_profile(self) -> None:
        """Test creating threshold profile."""
        profile = ThresholdProfile(
            name="strict",
            base_family="TTL",
            overrides={"VIH": 2.2},
            tolerance=0.0,
        )

        assert profile.name == "strict"
        assert profile.base_family == "TTL"
        assert profile.overrides["VIH"] == 2.2
        assert profile.tolerance == 0.0

    def test_apply_to_family_with_overrides(self) -> None:
        """Test applying profile with threshold overrides."""
        base_family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        profile = ThresholdProfile(
            name="strict",
            base_family="TTL",
            overrides={"VIH": 2.5, "VIL": 0.7},
        )

        result = profile.apply_to(base_family)

        assert result.VIH == 2.5
        assert result.VIL == 0.7
        assert result.VOH == 2.4  # Unchanged
        assert result.VOL == 0.4  # Unchanged

    def test_apply_to_family_with_tolerance(self) -> None:
        """Test applying profile with tolerance."""
        base_family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        profile = ThresholdProfile(
            name="relaxed",
            base_family="TTL",
            tolerance=20.0,  # 20%
        )

        result = profile.apply_to(base_family)

        # VOH increased by 20%
        assert pytest.approx(2.4 * 1.2) == result.VOH
        # VOL decreased by 20% (divided by factor)
        assert pytest.approx(0.4 / 1.2) == result.VOL

    def test_apply_combines_overrides_and_tolerance(self) -> None:
        """Test applying profile with both overrides and tolerance."""
        base_family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        profile = ThresholdProfile(
            name="custom",
            base_family="TTL",
            overrides={"VIH": 2.5},
            tolerance=10.0,
        )

        result = profile.apply_to(base_family)

        # Override takes precedence for VIH
        assert result.VIH == 2.5
        # Tolerance applied to VOH
        assert pytest.approx(2.4 * 1.1) == result.VOH

    def test_profile_default_values(self) -> None:
        """Test default values for profile."""
        profile = ThresholdProfile(name="test")

        assert profile.base_family == "TTL"
        assert profile.overrides == {}
        assert profile.tolerance == 0.0
        assert profile.description == ""


class TestThresholdRegistry:
    """Test ThresholdRegistry singleton.

    Note: Uses reset_threshold_registry fixture to ensure clean state between tests,
    especially important when running in parallel with pytest-xdist.
    """

    def test_registry_singleton(self, reset_threshold_registry) -> None:
        """Test that registry is a singleton."""
        registry1 = ThresholdRegistry()
        registry2 = ThresholdRegistry()

        assert registry1 is registry2

    def test_builtin_families_loaded(self, reset_threshold_registry) -> None:
        """Test that builtin logic families are loaded."""
        registry = ThresholdRegistry()
        families = registry.list_families()

        assert "TTL" in families
        assert "CMOS_5V" in families
        assert "LVTTL_3V3" in families
        assert "LVCMOS_3V3" in families
        assert "ECL" in families

    def test_get_family_ttl(self, reset_threshold_registry) -> None:
        """Test getting TTL family."""
        registry = ThresholdRegistry()
        ttl = registry.get_family("TTL")

        assert ttl.name == "TTL"
        assert ttl.VIH == 2.0
        assert ttl.VIL == 0.8
        assert ttl.VCC == 5.0

    def test_get_family_cmos(self, reset_threshold_registry) -> None:
        """Test getting CMOS family."""
        registry = ThresholdRegistry()
        cmos = registry.get_family("CMOS_5V")

        assert cmos.name == "CMOS_5V"
        assert cmos.VCC == 5.0

    def test_get_family_case_insensitive(self, reset_threshold_registry) -> None:
        """Test getting family with case-insensitive match."""
        registry = ThresholdRegistry()

        ttl1 = registry.get_family("TTL")
        ttl2 = registry.get_family("ttl")

        assert ttl1.name == ttl2.name

    def test_get_family_not_found(self, reset_threshold_registry) -> None:
        """Test error when family not found."""
        registry = ThresholdRegistry()

        with pytest.raises(KeyError, match="Logic family.*not found"):
            registry.get_family("NONEXISTENT")

    def test_list_families(self, reset_threshold_registry) -> None:
        """Test listing all families."""
        registry = ThresholdRegistry()
        families = registry.list_families()

        assert isinstance(families, list)
        assert len(families) > 0
        assert all(isinstance(name, str) for name in families)

    def test_register_custom_family(self, reset_threshold_registry) -> None:
        """Test registering custom logic family."""
        registry = ThresholdRegistry()

        custom = LogicFamily(
            name="my_custom",
            VIH=2.5,
            VIL=1.0,
            VOH=3.0,
            VOL=0.5,
        )

        registry.register_family(custom)

        # Should be available with namespace
        result = registry.get_family("user.my_custom")
        assert result.VIH == 2.5

    def test_register_family_with_custom_namespace(self, reset_threshold_registry) -> None:
        """Test registering family with custom namespace."""
        registry = ThresholdRegistry()

        custom = LogicFamily(
            name="special",
            VIH=2.5,
            VIL=1.0,
            VOH=3.0,
            VOL=0.5,
        )

        registry.register_family(custom, namespace="custom")

        result = registry.get_family("custom.special")
        assert result.VIH == 2.5

    def test_set_threshold_override(self, reset_threshold_registry) -> None:
        """Test setting session threshold overrides."""
        registry = ThresholdRegistry()
        registry.reset_overrides()  # Ensure clean state

        registry.set_threshold_override(VIH=2.5, VIL=0.7)

        ttl = registry.get_family("TTL")
        assert ttl.VIH == 2.5
        assert ttl.VIL == 0.7
        assert ttl.source == "override"

    def test_set_threshold_override_invalid_key(self, reset_threshold_registry) -> None:
        """Test error with invalid threshold key."""
        registry = ThresholdRegistry()

        with pytest.raises(ValueError, match="Invalid threshold key"):
            registry.set_threshold_override(INVALID=1.0)

    def test_set_threshold_override_out_of_range(self, reset_threshold_registry) -> None:
        """Test error with out-of-range threshold value."""
        registry = ThresholdRegistry()

        with pytest.raises(ValueError, match="out of range"):
            registry.set_threshold_override(VIH=15.0)

    def test_reset_overrides(self, reset_threshold_registry) -> None:
        """Test resetting threshold overrides."""
        registry = ThresholdRegistry()

        registry.set_threshold_override(VIH=2.5)
        ttl1 = registry.get_family("TTL")
        assert ttl1.VIH == 2.5

        registry.reset_overrides()
        ttl2 = registry.get_family("TTL")
        assert ttl2.VIH == 2.0  # Back to default

    def test_get_profile(self, reset_threshold_registry) -> None:
        """Test getting builtin profile."""
        registry = ThresholdRegistry()

        strict = registry.get_profile("strict")
        assert strict.name == "strict"
        assert strict.tolerance == 0

        relaxed = registry.get_profile("relaxed")
        assert relaxed.tolerance == 20

    def test_get_profile_not_found(self, reset_threshold_registry) -> None:
        """Test error when profile not found."""
        registry = ThresholdRegistry()

        with pytest.raises(KeyError, match="Profile.*not found"):
            registry.get_profile("nonexistent")

    def test_apply_profile(self, reset_threshold_registry) -> None:
        """Test applying a threshold profile."""
        registry = ThresholdRegistry()

        result = registry.apply_profile("strict")

        assert "TTL" in result.name
        assert "strict" in result.name

    def test_save_profile(self, reset_threshold_registry) -> None:
        """Test saving current settings as profile."""
        registry = ThresholdRegistry()
        registry.reset_overrides()

        registry.set_threshold_override(VIH=2.5, VIL=0.7)
        registry.save_profile("my_profile")

        profile = registry.get_profile("my_profile")
        assert profile.name == "my_profile"
        assert profile.overrides["VIH"] == 2.5
        assert profile.overrides["VIL"] == 0.7

    def test_save_profile_to_file(self, reset_threshold_registry, tmp_path: Path) -> None:
        """Test saving profile to file."""
        registry = ThresholdRegistry()
        registry.reset_overrides()

        registry.set_threshold_override(VIH=2.5)
        profile_file = tmp_path / "profile.yaml"

        registry.save_profile("my_profile", path=profile_file)

        assert profile_file.exists()
        with open(profile_file) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "my_profile"
        assert data["overrides"]["VIH"] == 2.5


class TestLoadLogicFamily:
    """Test load_logic_family function."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Test loading logic family from YAML file."""
        family_file = tmp_path / "custom.yaml"
        family_file.write_text(
            """
name: Custom_TTL
VIH: 2.5
VIL: 1.0
VOH: 3.0
VOL: 0.5
VCC: 5.0
description: Custom TTL variant
"""
        )

        with patch("oscura.config.schema.validate_against_schema"):
            family = load_logic_family(family_file)

        assert family.name == "Custom_TTL"
        assert family.VIH == 2.5
        assert family.VIL == 1.0
        assert family.description == "Custom TTL variant"

    def test_load_with_temperature_range(self, tmp_path: Path) -> None:
        """Test loading family with temperature range."""
        family_file = tmp_path / "custom.yaml"
        family_file.write_text(
            """
name: Industrial
VIH: 2.0
VIL: 0.8
VOH: 2.4
VOL: 0.4
temperature_range:
  min: -40
  max: 85
"""
        )

        with patch("oscura.config.schema.validate_against_schema"):
            family = load_logic_family(family_file)

        assert family.temperature_range == (-40, 85)

    def test_load_with_noise_margins(self, tmp_path: Path) -> None:
        """Test loading family with explicit noise margins."""
        family_file = tmp_path / "custom.yaml"
        family_file.write_text(
            """
name: Custom
VIH: 2.0
VIL: 0.8
VOH: 2.4
VOL: 0.4
noise_margin_high: 0.5
noise_margin_low: 0.5
"""
        )

        with patch("oscura.config.schema.validate_against_schema"):
            family = load_logic_family(family_file)

        assert family.noise_margin_high == 0.5
        assert family.noise_margin_low == 0.5

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        """Test loading minimal logic family config."""
        family_file = tmp_path / "minimal.yaml"
        family_file.write_text(
            """
name: Minimal
VIH: 2.0
VIL: 0.8
VOH: 2.4
VOL: 0.4
"""
        )

        with patch("oscura.config.schema.validate_against_schema"):
            family = load_logic_family(family_file)

        assert family.name == "Minimal"
        assert family.VCC == 5.0  # Default value


class TestGetUserLogicFamiliesDir:
    """Test get_user_logic_families_dir function."""

    def test_returns_path(self) -> None:
        """Test that function returns a Path."""
        result = get_user_logic_families_dir()
        assert isinstance(result, Path)

    def test_creates_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that directory is created if it doesn't exist."""
        test_dir = tmp_path / "test_config"

        def mock_home() -> Path:
            return tmp_path

        monkeypatch.setattr(Path, "home", mock_home)
        monkeypatch.setenv("XDG_CONFIG_HOME", "")

        result = get_user_logic_families_dir()
        assert result.exists()
        assert result.is_dir()

    def test_xdg_config_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test using XDG_CONFIG_HOME environment variable."""
        xdg_config = tmp_path / "xdg_config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))

        result = get_user_logic_families_dir()
        assert str(xdg_config) in str(result)


@pytest.mark.xdist_group(name="user_logic_families_io")
class TestLoadUserLogicFamilies:
    """Test load_user_logic_families function.

    Note: Uses reset_threshold_registry fixture to ensure clean state between tests.
    Grouped with xdist_group to prevent filesystem race conditions when running
    in parallel - these tests mock get_user_logic_families_dir() which can conflict
    under extreme concurrency (18K+ tests).
    """

    def test_load_empty_directory(
        self, reset_threshold_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading from empty directory."""

        def mock_get_dir() -> Path:
            user_dir = tmp_path / "logic_families"
            user_dir.mkdir(parents=True, exist_ok=True)
            return user_dir

        monkeypatch.setattr("oscura.config.thresholds.get_user_logic_families_dir", mock_get_dir)

        families = load_user_logic_families()
        assert families == []

    def test_load_multiple_families(
        self, reset_threshold_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading multiple logic families."""
        user_dir = tmp_path / "logic_families"
        user_dir.mkdir(parents=True, exist_ok=True)

        # Create test family files
        (user_dir / "family1.yaml").write_text(
            """
name: Family1
VIH: 2.0
VIL: 0.8
VOH: 2.4
VOL: 0.4
"""
        )

        (user_dir / "family2.yaml").write_text(
            """
name: Family2
VIH: 1.5
VIL: 0.6
VOH: 1.8
VOL: 0.3
"""
        )

        def mock_get_dir() -> Path:
            return user_dir

        monkeypatch.setattr("oscura.config.thresholds.get_user_logic_families_dir", mock_get_dir)

        with patch("oscura.config.schema.validate_against_schema"):
            families = load_user_logic_families()

        assert len(families) == 2
        names = [f.name for f in families]
        assert "Family1" in names
        assert "Family2" in names

    def test_skip_invalid_files(
        self,
        reset_threshold_registry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that invalid files are skipped with warning."""
        user_dir = tmp_path / "logic_families"
        user_dir.mkdir(parents=True, exist_ok=True)

        # Create valid file
        (user_dir / "valid.yaml").write_text(
            """
name: Valid
VIH: 2.0
VIL: 0.8
VOH: 2.4
VOL: 0.4
"""
        )

        # Create invalid file
        (user_dir / "invalid.yaml").write_text("invalid: yaml: content:")

        def mock_get_dir() -> Path:
            return user_dir

        monkeypatch.setattr("oscura.config.thresholds.get_user_logic_families_dir", mock_get_dir)

        with patch("oscura.config.schema.validate_against_schema"):
            families = load_user_logic_families()

        # Should load valid file only
        assert len(families) == 1
        assert families[0].name == "Valid"


class TestGetThresholdRegistry:
    """Test get_threshold_registry function."""

    def test_returns_registry(self) -> None:
        """Test that function returns ThresholdRegistry."""
        registry = get_threshold_registry()
        assert isinstance(registry, ThresholdRegistry)

    def test_returns_singleton(self) -> None:
        """Test that function returns same instance."""
        registry1 = get_threshold_registry()
        registry2 = get_threshold_registry()
        assert registry1 is registry2


class TestConfigThresholdsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_logic_family_equal_thresholds(self) -> None:
        """Test error with equal VIH and VIL."""
        with pytest.raises(ConfigurationError):
            LogicFamily(
                name="Invalid",
                VIH=1.5,
                VIL=1.5,
                VOH=2.0,
                VOL=0.5,
            )

    def test_negative_voltage_thresholds(self) -> None:
        """Test logic family with negative voltages (ECL)."""
        ecl = LogicFamily(
            name="ECL",
            VIH=-0.9,
            VIL=-1.7,
            VOH=-0.9,
            VOL=-1.75,
            VCC=-5.2,
        )

        assert ecl.VIH == -0.9
        assert ecl.VCC == -5.2

    def test_very_small_voltage_thresholds(self) -> None:
        """Test logic family with very small voltages."""
        low_v = LogicFamily(
            name="LowVoltage",
            VIH=0.5,
            VIL=0.2,
            VOH=0.6,
            VOL=0.1,
            VCC=0.8,
        )

        assert low_v.VIH == 0.5
        assert low_v.get_threshold(50.0) == pytest.approx(0.35)

    def test_threshold_profile_zero_tolerance(self) -> None:
        """Test profile with zero tolerance."""
        profile = ThresholdProfile(name="zero", tolerance=0.0)
        base = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        result = profile.apply_to(base)
        # With zero tolerance, VOH and VOL should be unchanged
        assert result.VOH == 2.4
        assert result.VOL == 0.4

    def test_threshold_profile_large_tolerance(self) -> None:
        """Test profile with large tolerance."""
        profile = ThresholdProfile(name="large", tolerance=50.0)
        base = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        result = profile.apply_to(base)
        # VOH increased by 50%
        assert pytest.approx(2.4 * 1.5) == result.VOH
        # VOL decreased (divided by 1.5)
        assert pytest.approx(0.4 / 1.5) == result.VOL

    def test_registry_override_multiple_thresholds(self) -> None:
        """Test setting multiple threshold overrides."""
        registry = ThresholdRegistry()
        registry.reset_overrides()

        registry.set_threshold_override(VIH=2.5, VIL=0.7, VOH=3.0, VOL=0.3, VCC=5.5)

        ttl = registry.get_family("TTL")
        assert ttl.VIH == 2.5
        assert ttl.VIL == 0.7
        assert ttl.VOH == 3.0
        assert ttl.VOL == 0.3
        assert ttl.VCC == 5.5

    def test_temperature_derating_extreme_temperatures(self) -> None:
        """Test temperature derating at extreme temperatures."""
        family = LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
        )

        # Very hot
        hot = family.with_temperature_derating(125.0, derating_factor=0.002)
        # Delta = 100C, factor = 1 - 0.2 = 0.8
        assert pytest.approx(2.0 * 0.8) == hot.VIH

        # Very cold
        cold = family.with_temperature_derating(-55.0, derating_factor=0.002)
        # Delta = -80C, factor = 1 + 0.16 = 1.16
        assert pytest.approx(2.0 * 1.16) == cold.VIH
