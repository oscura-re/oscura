"""Tests for visualization style presets."""

import pytest

from oscura.visualization.styles import (
    PRESENTATION_PRESET,
    PRESETS,
    PRINT_PRESET,
    PUBLICATION_PRESET,
    SCREEN_PRESET,
    StylePreset,
    _preset_to_rcparams,
    apply_style_preset,
    create_custom_preset,
    list_presets,
    register_preset,
)

pytest.importorskip("matplotlib")

pytestmark = pytest.mark.usefixtures("cleanup_matplotlib")


class TestStylePreset:
    """Test StylePreset dataclass."""

    def test_default_creation(self) -> None:
        """Test creating preset with defaults."""
        preset = StylePreset(name="test")
        assert preset.name == "test"
        assert preset.dpi == 96
        assert preset.font_family == "sans-serif"
        assert preset.font_size == 10
        assert preset.line_width == 1.0
        assert preset.marker_size == 6.0
        assert preset.figure_facecolor == "white"
        assert preset.axes_facecolor == "white"
        assert preset.axes_edgecolor == "black"
        assert preset.grid_color == "#B0B0B0"
        assert preset.grid_alpha == 0.3
        assert preset.grid_linestyle == "-"
        assert preset.use_latex is False
        assert preset.tight_layout is True
        assert preset.rcparams == {}

    def test_custom_values(self) -> None:
        """Test creating preset with custom values."""
        preset = StylePreset(
            name="custom",
            dpi=300,
            font_family="serif",
            font_size=12,
            line_width=2.0,
            marker_size=8.0,
            use_latex=True,
        )
        assert preset.name == "custom"
        assert preset.dpi == 300
        assert preset.font_family == "serif"
        assert preset.font_size == 12
        assert preset.line_width == 2.0
        assert preset.marker_size == 8.0
        assert preset.use_latex is True

    def test_with_rcparams(self) -> None:
        """Test creating preset with additional rcparams."""
        extra_params = {"axes.linewidth": 2.0, "savefig.dpi": 600}
        preset = StylePreset(name="test", rcparams=extra_params)
        assert preset.rcparams == extra_params


class TestPredefinedPresets:
    """Test predefined preset configurations."""

    def test_publication_preset(self) -> None:
        """Test publication preset configuration."""
        assert PUBLICATION_PRESET.name == "publication"
        assert PUBLICATION_PRESET.dpi == 600
        assert PUBLICATION_PRESET.font_family == "serif"
        assert PUBLICATION_PRESET.line_width == 0.8
        assert PUBLICATION_PRESET.grid_linestyle == ":"
        assert "savefig.dpi" in PUBLICATION_PRESET.rcparams
        assert PUBLICATION_PRESET.rcparams["savefig.dpi"] == 600

    def test_presentation_preset(self) -> None:
        """Test presentation preset configuration."""
        assert PRESENTATION_PRESET.name == "presentation"
        assert PRESENTATION_PRESET.font_size == 18
        assert PRESENTATION_PRESET.line_width == 2.5
        assert PRESENTATION_PRESET.marker_size == 10.0

    def test_screen_preset(self) -> None:
        """Test screen preset configuration."""
        assert SCREEN_PRESET.name == "screen"
        assert SCREEN_PRESET.dpi == 96
        assert SCREEN_PRESET.font_family == "sans-serif"
        assert SCREEN_PRESET.line_width == 1.2

    def test_print_preset(self) -> None:
        """Test print preset configuration."""
        assert PRINT_PRESET.name == "print"
        assert PRINT_PRESET.dpi == 300
        assert PRINT_PRESET.font_family == "serif"
        assert "savefig.format" in PRINT_PRESET.rcparams
        assert PRINT_PRESET.rcparams["savefig.format"] == "pdf"

    def test_all_presets_in_registry(self) -> None:
        """Test that all presets are in registry."""
        assert "publication" in PRESETS
        assert "presentation" in PRESETS
        assert "screen" in PRESETS
        assert "print" in PRESETS

    def test_preset_objects_match(self) -> None:
        """Test that registry contains correct preset objects."""
        assert PRESETS["publication"] == PUBLICATION_PRESET
        assert PRESETS["presentation"] == PRESENTATION_PRESET
        assert PRESETS["screen"] == SCREEN_PRESET
        assert PRESETS["print"] == PRINT_PRESET


class TestPresetToRcparams:
    """Test _preset_to_rcparams conversion."""

    def test_basic_conversion(self) -> None:
        """Test basic preset to rcparams conversion."""
        preset = StylePreset(name="test", dpi=150, font_size=12)
        rc = _preset_to_rcparams(preset)
        assert rc["figure.dpi"] == 150
        assert rc["font.size"] == 12
        assert rc["font.family"] == "sans-serif"

    def test_with_latex(self) -> None:
        """Test conversion with LaTeX enabled."""
        preset = StylePreset(name="test", use_latex=True)
        rc = _preset_to_rcparams(preset)
        assert rc.get("text.usetex") is True

    def test_without_latex(self) -> None:
        """Test conversion without LaTeX."""
        preset = StylePreset(name="test", use_latex=False)
        rc = _preset_to_rcparams(preset)
        assert "text.usetex" not in rc

    def test_tight_layout(self) -> None:
        """Test tight layout setting."""
        preset = StylePreset(name="test", tight_layout=True)
        rc = _preset_to_rcparams(preset)
        assert rc["figure.autolayout"] is True

        preset2 = StylePreset(name="test2", tight_layout=False)
        rc2 = _preset_to_rcparams(preset2)
        assert rc2["figure.autolayout"] is False

    def test_additional_rcparams_merged(self) -> None:
        """Test that additional rcparams are merged."""
        extra = {"axes.linewidth": 2.5, "custom.param": "value"}
        preset = StylePreset(name="test", rcparams=extra)
        rc = _preset_to_rcparams(preset)
        assert rc["axes.linewidth"] == 2.5
        assert rc["custom.param"] == "value"

    def test_all_fields_converted(self) -> None:
        """Test that all preset fields are converted."""
        preset = StylePreset(
            name="test",
            dpi=200,
            font_family="monospace",
            font_size=14,
            line_width=2.0,
            marker_size=8.0,
            figure_facecolor="#EEEEEE",
            axes_facecolor="#DDDDDD",
            axes_edgecolor="#333333",
            grid_color="#AAAAAA",
            grid_alpha=0.5,
            grid_linestyle="--",
        )
        rc = _preset_to_rcparams(preset)
        assert rc["figure.dpi"] == 200
        assert rc["font.family"] == "monospace"
        assert rc["font.size"] == 14
        assert rc["lines.linewidth"] == 2.0
        assert rc["lines.markersize"] == 8.0
        assert rc["figure.facecolor"] == "#EEEEEE"
        assert rc["axes.facecolor"] == "#DDDDDD"
        assert rc["axes.edgecolor"] == "#333333"
        assert rc["grid.color"] == "#AAAAAA"
        assert rc["grid.alpha"] == 0.5
        assert rc["grid.linestyle"] == "--"


class TestApplyStylePreset:
    """Test apply_style_preset context manager."""

    def test_apply_by_name(self) -> None:
        """Test applying preset by name."""
        import matplotlib.pyplot as plt

        with apply_style_preset("screen"):
            assert plt.rcParams["figure.dpi"] == 96

    def test_apply_by_object(self) -> None:
        """Test applying preset by object."""
        import matplotlib.pyplot as plt

        custom = StylePreset(name="test", dpi=150)
        with apply_style_preset(custom):
            assert plt.rcParams["figure.dpi"] == 150

    def test_with_overrides(self) -> None:
        """Test applying preset with overrides."""
        import matplotlib.pyplot as plt

        overrides = {"font.size": 20}
        with apply_style_preset("screen", overrides=overrides):
            assert plt.rcParams["font.size"] == 20

    def test_unknown_preset_name_error(self) -> None:
        """Test error for unknown preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            with apply_style_preset("nonexistent"):
                pass

    def test_context_restoration(self) -> None:
        """Test that rcParams are restored after context."""
        import matplotlib.pyplot as plt

        original_dpi = plt.rcParams["figure.dpi"]
        with apply_style_preset("publication"):
            assert plt.rcParams["figure.dpi"] == 600
        # Should restore after context
        assert plt.rcParams["figure.dpi"] == original_dpi

    def test_nested_contexts(self) -> None:
        """Test nested style contexts."""
        import matplotlib.pyplot as plt

        with apply_style_preset("screen"):
            screen_dpi = plt.rcParams["figure.dpi"]
            assert screen_dpi == 96

            with apply_style_preset("publication"):
                assert plt.rcParams["figure.dpi"] == 600

            # Should restore to screen preset
            assert plt.rcParams["figure.dpi"] == screen_dpi


class TestCreateCustomPreset:
    """Test create_custom_preset function."""

    def test_inherit_from_base(self) -> None:
        """Test inheriting from base preset."""
        custom = create_custom_preset("my_style", base_preset="screen")
        assert custom.name == "my_style"
        # Should inherit screen values
        assert custom.dpi == 96
        assert custom.font_family == "sans-serif"

    def test_override_attributes(self) -> None:
        """Test overriding base preset attributes."""
        custom = create_custom_preset(
            "my_style",
            base_preset="publication",
            font_size=14,
            line_width=2.0,
        )
        assert custom.name == "my_style"
        assert custom.font_size == 14
        assert custom.line_width == 2.0
        # Should inherit other attributes
        assert custom.dpi == 600  # From publication preset

    def test_inherit_rcparams(self) -> None:
        """Test that rcparams are inherited."""
        custom = create_custom_preset("my_style", base_preset="publication")
        # Should inherit publication rcparams
        assert "savefig.dpi" in custom.rcparams

    def test_unknown_base_preset_error(self) -> None:
        """Test error for unknown base preset."""
        with pytest.raises(ValueError, match="Unknown base_preset"):
            create_custom_preset("my_style", base_preset="nonexistent")

    def test_multiple_overrides(self) -> None:
        """Test multiple attribute overrides."""
        custom = create_custom_preset(
            "my_style",
            base_preset="screen",
            dpi=200,
            font_family="monospace",
            font_size=12,
            line_width=1.5,
            marker_size=7.0,
            use_latex=True,
        )
        assert custom.dpi == 200
        assert custom.font_family == "monospace"
        assert custom.font_size == 12
        assert custom.line_width == 1.5
        assert custom.marker_size == 7.0
        assert custom.use_latex is True


class TestRegisterPreset:
    """Test register_preset function."""

    def test_register_new_preset(self) -> None:
        """Test registering a new preset."""
        custom = StylePreset(name="test_custom", dpi=250)
        register_preset(custom)
        assert "test_custom" in PRESETS
        assert PRESETS["test_custom"] == custom

    def test_register_overwrites_existing(self) -> None:
        """Test that registering overwrites existing preset."""
        custom1 = StylePreset(name="test_overwrite", dpi=100)
        register_preset(custom1)

        custom2 = StylePreset(name="test_overwrite", dpi=200)
        register_preset(custom2)

        assert PRESETS["test_overwrite"].dpi == 200

    def test_registered_preset_usable(self) -> None:
        """Test that registered preset can be used."""
        import matplotlib.pyplot as plt

        custom = StylePreset(name="test_usable", dpi=175)
        register_preset(custom)

        with apply_style_preset("test_usable"):
            assert plt.rcParams["figure.dpi"] == 175


class TestListPresets:
    """Test list_presets function."""

    def test_lists_all_presets(self) -> None:
        """Test that all predefined presets are listed."""
        presets = list_presets()
        assert "publication" in presets
        assert "presentation" in presets
        assert "screen" in presets
        assert "print" in presets

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert all(isinstance(p, str) for p in presets)

    def test_includes_registered_presets(self) -> None:
        """Test that registered presets appear in list."""
        custom = StylePreset(name="test_listed", dpi=150)
        register_preset(custom)
        presets = list_presets()
        assert "test_listed" in presets


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_rcparams_dict(self) -> None:
        """Test preset with empty rcparams."""
        preset = StylePreset(name="test", rcparams={})
        rc = _preset_to_rcparams(preset)
        assert isinstance(rc, dict)
        assert "figure.dpi" in rc

    def test_zero_values(self) -> None:
        """Test preset with zero values."""
        preset = StylePreset(
            name="test",
            dpi=0,
            font_size=0,
            line_width=0.0,
            grid_alpha=0.0,
        )
        rc = _preset_to_rcparams(preset)
        assert rc["figure.dpi"] == 0
        assert rc["font.size"] == 0
        assert rc["lines.linewidth"] == 0.0
        assert rc["grid.alpha"] == 0.0

    def test_very_high_dpi(self) -> None:
        """Test preset with very high DPI."""
        preset = StylePreset(name="test", dpi=2400)
        rc = _preset_to_rcparams(preset)
        assert rc["figure.dpi"] == 2400

    def test_special_characters_in_name(self) -> None:
        """Test preset name with special characters."""
        preset = StylePreset(name="test-preset_v1.0")
        assert preset.name == "test-preset_v1.0"

    def test_preset_immutability(self) -> None:
        """Test that applying preset doesn't modify original."""
        import matplotlib.pyplot as plt

        original_dpi = SCREEN_PRESET.dpi
        with apply_style_preset("screen", overrides={"figure.dpi": 300}):
            assert plt.rcParams["figure.dpi"] == 300
        # Original preset should be unchanged
        assert SCREEN_PRESET.dpi == original_dpi


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_publication_workflow(self) -> None:
        """Test typical publication workflow."""
        import matplotlib.pyplot as plt

        with apply_style_preset("publication"):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            # Should have publication settings
            assert plt.rcParams["figure.dpi"] == 600
            # font.family returns list
            assert "serif" in plt.rcParams["font.family"]
            plt.close(fig)

    def test_presentation_workflow(self) -> None:
        """Test typical presentation workflow."""
        import matplotlib.pyplot as plt

        with apply_style_preset("presentation"):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            # Should have large fonts for visibility
            assert plt.rcParams["font.size"] == 18
            plt.close(fig)

    def test_custom_preset_workflow(self) -> None:
        """Test custom preset creation and usage workflow."""
        import matplotlib.pyplot as plt

        # Create and register custom preset
        custom = create_custom_preset(
            "my_publication",
            base_preset="publication",
            font_size=11,
            line_width=1.0,
        )
        register_preset(custom)

        # Use custom preset
        with apply_style_preset("my_publication"):
            assert plt.rcParams["font.size"] == 11
            assert plt.rcParams["lines.linewidth"] == 1.0
            # Should still have publication DPI
            assert plt.rcParams["figure.dpi"] == 600
