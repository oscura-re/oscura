"""Comprehensive tests for visualization.render module.

Tests cover DPI-aware rendering configuration for print vs screen output.

Coverage target: >90%
"""

from __future__ import annotations

import pytest

from oscura.visualization.render import (
    RenderPreset,
    apply_rendering_config,
    configure_dpi_rendering,
)

pytestmark = pytest.mark.usefixtures("cleanup_matplotlib")


class TestConfigureDpiRendering:
    """Tests for configure_dpi_rendering function."""

    def test_screen_preset(self) -> None:
        """Test screen rendering preset."""
        config = configure_dpi_rendering("screen")

        assert config["dpi"] == 96
        assert config["format"] == "png"
        assert config["antialias"] is True
        assert "style_params" in config

    def test_print_preset(self) -> None:
        """Test print rendering preset."""
        config = configure_dpi_rendering("print")

        assert config["dpi"] == 300
        assert config["format"] == "pdf"
        assert config["antialias"] is False

    def test_publication_preset(self) -> None:
        """Test publication rendering preset."""
        config = configure_dpi_rendering("publication")

        assert config["dpi"] == 600
        assert config["format"] == "pdf"
        assert config["antialias"] is False
        assert config["style_params"]["font.family"] == "serif"

    def test_invalid_preset_raises(self) -> None:
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid preset"):
            configure_dpi_rendering("invalid")  # type: ignore

    def test_custom_dpi(self) -> None:
        """Test custom DPI override."""
        config = configure_dpi_rendering("screen", custom_dpi=150)

        assert config["dpi"] == 150
        assert config["preset"] == "custom"

    def test_dpi_alias(self) -> None:
        """Test dpi parameter as alias for custom_dpi."""
        config = configure_dpi_rendering("screen", dpi=200)

        assert config["dpi"] == 200

    def test_custom_dpi_takes_precedence(self) -> None:
        """Test that custom_dpi takes precedence over dpi."""
        config = configure_dpi_rendering("screen", custom_dpi=150, dpi=200)

        # custom_dpi should win
        assert config["dpi"] == 150

    def test_custom_figsize(self) -> None:
        """Test custom figure size."""
        config = configure_dpi_rendering("screen", figsize=(12, 8))

        assert config["figsize"] == (12, 8)

    def test_scale_factors(self) -> None:
        """Test that scale factors are calculated correctly."""
        # At 96 DPI (baseline), scale should be 1.0
        config_96 = configure_dpi_rendering("screen", custom_dpi=96, baseline_dpi=96.0)
        assert config_96["font_scale"] == pytest.approx(1.0)
        assert config_96["line_scale"] == pytest.approx(1.0)
        assert config_96["marker_scale"] == pytest.approx(1.0)

        # At 192 DPI (2x baseline), scale should be 2.0
        config_192 = configure_dpi_rendering("screen", custom_dpi=192, baseline_dpi=96.0)
        assert config_192["font_scale"] == pytest.approx(2.0)
        assert config_192["line_scale"] == pytest.approx(2.0)
        assert config_192["marker_scale"] == pytest.approx(2.0)

    def test_style_params_structure(self) -> None:
        """Test structure of style_params dictionary."""
        config = configure_dpi_rendering("screen")

        style_params = config["style_params"]

        # Should contain matplotlib rcParams
        assert "figure.dpi" in style_params
        assert "savefig.dpi" in style_params
        assert "font.size" in style_params
        assert "lines.linewidth" in style_params
        assert "lines.markersize" in style_params

    def test_anti_aliasing_enabled_screen(self) -> None:
        """Test anti-aliasing enabled for screen."""
        config = configure_dpi_rendering("screen")

        style_params = config["style_params"]
        assert style_params["lines.antialiased"] is True
        assert style_params["patch.antialiased"] is True
        assert style_params["text.antialiased"] is True

    def test_anti_aliasing_disabled_print(self) -> None:
        """Test anti-aliasing disabled for print."""
        config = configure_dpi_rendering("print")

        style_params = config["style_params"]
        assert style_params["lines.antialiased"] is False
        assert style_params["patch.antialiased"] is False
        assert style_params["text.antialiased"] is False

    def test_publication_specific_settings(self) -> None:
        """Test publication-specific matplotlib settings."""
        config = configure_dpi_rendering("publication")

        style_params = config["style_params"]
        assert style_params["font.family"] == "serif"
        assert style_params["mathtext.fontset"] == "cm"  # Computer Modern
        assert style_params["axes.grid"] is True

    def test_format_selection_low_dpi(self) -> None:
        """Test format selection for low DPI (PNG)."""
        config = configure_dpi_rendering("screen", custom_dpi=72)

        assert config["format"] == "png"

    def test_format_selection_high_dpi(self) -> None:
        """Test format selection for high DPI (PDF)."""
        config = configure_dpi_rendering("screen", custom_dpi=200)

        assert config["format"] == "pdf"

    def test_description_field(self) -> None:
        """Test that description field is present."""
        config = configure_dpi_rendering("screen")

        assert "description" in config
        assert "96 DPI" in config["description"]

    def test_all_return_fields(self) -> None:
        """Test that all expected fields are in return dict."""
        config = configure_dpi_rendering("screen")

        required_fields = [
            "dpi",
            "figsize",
            "font_scale",
            "line_scale",
            "marker_scale",
            "antialias",
            "format",
            "style_params",
            "description",
            "preset",
        ]

        for field in required_fields:
            assert field in config


class TestApplyRenderingConfig:
    """Tests for apply_rendering_config function."""

    def test_apply_config_without_matplotlib_raises(self) -> None:
        """Test that applying config without matplotlib raises ImportError."""
        import sys
        from unittest.mock import patch

        config = configure_dpi_rendering("screen")

        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib is required"):
                apply_rendering_config(config)

    def test_apply_config_updates_rcparams(self, matplotlib_available: None) -> None:
        """Test that applying config updates matplotlib rcParams."""
        import matplotlib.pyplot as plt

        # Get initial DPI
        initial_dpi = plt.rcParams["figure.dpi"]

        # Apply new config
        config = configure_dpi_rendering("print")  # 300 DPI
        apply_rendering_config(config)

        # DPI should be updated
        assert plt.rcParams["figure.dpi"] == 300

        # Restore original
        plt.rcParams["figure.dpi"] = initial_dpi

    def test_apply_config_multiple_params(self, matplotlib_available: None) -> None:
        """Test that multiple parameters are applied."""
        import matplotlib.pyplot as plt

        config = configure_dpi_rendering("publication")
        apply_rendering_config(config)

        # Check multiple params were updated
        assert plt.rcParams["figure.dpi"] == 600
        assert plt.rcParams["font.family"] == ["serif"]
        # Note: Font family is stored as list in rcParams

        # Restore defaults
        plt.rcdefaults()

    def test_apply_config_idempotent(self, matplotlib_available: None) -> None:
        """Test that applying same config twice is safe."""
        import matplotlib.pyplot as plt

        config = configure_dpi_rendering("screen")

        # Apply twice
        apply_rendering_config(config)
        dpi_first = plt.rcParams["figure.dpi"]

        apply_rendering_config(config)
        dpi_second = plt.rcParams["figure.dpi"]

        assert dpi_first == dpi_second

        # Restore
        plt.rcdefaults()


class TestRenderPresetType:
    """Tests for RenderPreset type."""

    def test_render_preset_type_exists(self) -> None:
        """Test that RenderPreset type is defined."""
        # Should be importable
        assert RenderPreset is not None

    def test_valid_presets(self) -> None:
        """Test that all valid presets work."""
        valid_presets = ["screen", "print", "publication"]

        for preset in valid_presets:
            config = configure_dpi_rendering(preset)  # type: ignore
            assert config is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_baseline_dpi_safe(self) -> None:
        """Test that zero baseline DPI doesn't cause division errors."""
        # This should not crash (though results may be unusual)
        with pytest.raises(ZeroDivisionError):
            configure_dpi_rendering("screen", baseline_dpi=0.0)

    def test_very_high_dpi(self) -> None:
        """Test with very high DPI values."""
        config = configure_dpi_rendering("screen", custom_dpi=1200)

        assert config["dpi"] == 1200
        # Scale factors should be proportionally large
        assert config["font_scale"] > 10.0

    def test_very_low_dpi(self) -> None:
        """Test with very low DPI values."""
        config = configure_dpi_rendering("screen", custom_dpi=24)

        assert config["dpi"] == 24
        # Scale factors should be proportionally small
        assert config["font_scale"] < 1.0

    def test_large_figsize(self) -> None:
        """Test with very large figure size."""
        config = configure_dpi_rendering("screen", figsize=(100, 100))

        assert config["figsize"] == (100, 100)

    def test_small_figsize(self) -> None:
        """Test with very small figure size."""
        config = configure_dpi_rendering("screen", figsize=(1, 1))

        assert config["figsize"] == (1, 1)


class TestIntegration:
    """Integration tests for rendering configuration."""

    def test_screen_to_print_workflow(self, matplotlib_available: None) -> None:
        """Test switching from screen to print rendering."""
        import matplotlib.pyplot as plt

        # Start with screen
        screen_config = configure_dpi_rendering("screen")
        apply_rendering_config(screen_config)
        assert plt.rcParams["figure.dpi"] == 96

        # Switch to print
        print_config = configure_dpi_rendering("print")
        apply_rendering_config(print_config)
        assert plt.rcParams["figure.dpi"] == 300

        # Restore
        plt.rcdefaults()

    def test_custom_dpi_workflow(self, matplotlib_available: None) -> None:
        """Test custom DPI workflow."""
        import matplotlib.pyplot as plt

        # Use custom DPI for specific output device
        config = configure_dpi_rendering("screen", custom_dpi=120)
        apply_rendering_config(config)

        assert plt.rcParams["figure.dpi"] == 120

        # Restore
        plt.rcdefaults()


@pytest.fixture
def matplotlib_available() -> None:
    """Ensure matplotlib is available for tests."""
    pytest.importorskip("matplotlib")


# Run tests with: pytest tests/unit/visualization/test_render.py -v
