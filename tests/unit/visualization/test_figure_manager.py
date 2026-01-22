"""Tests for figure manager functionality."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from oscura.visualization.figure_manager import FigureManager

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    return tmp_path / "test_figures"


@pytest.fixture
def sample_figure():
    """Create a simple matplotlib figure for testing."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1, 2], [0, 1, 0], label="test")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Test Figure")
    ax.legend()
    yield fig
    plt.close(fig)


class TestFigureManagerInit:
    """Test FigureManager initialization."""

    def test_init_creates_directory(self, temp_output_dir):
        """Test that initialization creates output directory."""
        manager = FigureManager(temp_output_dir)
        assert temp_output_dir.exists()
        assert temp_output_dir.is_dir()
        assert manager.output_dir == temp_output_dir

    def test_init_with_existing_directory(self, temp_output_dir):
        """Test initialization with existing directory."""
        temp_output_dir.mkdir(parents=True)
        manager = FigureManager(temp_output_dir)
        assert temp_output_dir.exists()
        assert manager.output_dir == temp_output_dir

    def test_init_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        path_str = str(tmp_path / "string_test")
        manager = FigureManager(path_str)
        assert Path(path_str).exists()
        assert manager.output_dir == Path(path_str)

    def test_init_creates_nested_directories(self, tmp_path):
        """Test creation of nested directory structure."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        manager = FigureManager(nested_dir)
        assert nested_dir.exists()
        assert manager.output_dir == nested_dir

    def test_saved_figures_initialized_empty(self, temp_output_dir):
        """Test that saved_figures dict is initialized empty."""
        manager = FigureManager(temp_output_dir)
        assert manager.saved_figures == {}
        assert isinstance(manager.saved_figures, dict)


class TestSaveFigure:
    """Test save_figure method."""

    def test_save_single_format(self, temp_output_dir, sample_figure):
        """Test saving figure in single format."""
        manager = FigureManager(temp_output_dir)
        paths = manager.save_figure(sample_figure, "test_fig", formats=["png"])

        assert "png" in paths
        assert paths["png"].exists()
        assert paths["png"] == temp_output_dir / "test_fig.png"
        assert paths["png"].suffix == ".png"

    def test_save_multiple_formats(self, temp_output_dir, sample_figure):
        """Test saving figure in multiple formats."""
        manager = FigureManager(temp_output_dir)
        paths = manager.save_figure(sample_figure, "multi_format", formats=["png", "svg", "pdf"])

        assert len(paths) == 3
        assert all(fmt in paths for fmt in ["png", "svg", "pdf"])
        assert all(path.exists() for path in paths.values())
        assert paths["png"] == temp_output_dir / "multi_format.png"
        assert paths["svg"] == temp_output_dir / "multi_format.svg"
        assert paths["pdf"] == temp_output_dir / "multi_format.pdf"

    def test_save_default_format(self, temp_output_dir, sample_figure):
        """Test that default format is png."""
        manager = FigureManager(temp_output_dir)
        paths = manager.save_figure(sample_figure, "default_fmt")

        assert len(paths) == 1
        assert "png" in paths
        assert paths["png"].exists()

    def test_save_with_custom_dpi(self, temp_output_dir, sample_figure):
        """Test saving with custom DPI."""
        manager = FigureManager(temp_output_dir)
        paths = manager.save_figure(sample_figure, "custom_dpi", dpi=150)

        assert paths["png"].exists()
        # File should exist, size may vary by DPI but file should be valid

    def test_save_updates_registry(self, temp_output_dir, sample_figure):
        """Test that save_figure updates the saved_figures registry."""
        manager = FigureManager(temp_output_dir)
        manager.save_figure(sample_figure, "registered", formats=["png", "svg"])

        assert "registered" in manager.saved_figures
        assert "png" in manager.saved_figures["registered"]
        assert "svg" in manager.saved_figures["registered"]

    def test_save_multiple_figures(self, temp_output_dir, sample_figure):
        """Test saving multiple different figures."""
        manager = FigureManager(temp_output_dir)

        paths1 = manager.save_figure(sample_figure, "fig1", formats=["png"])
        paths2 = manager.save_figure(sample_figure, "fig2", formats=["svg"])

        assert paths1["png"].exists()
        assert paths2["svg"].exists()
        assert len(manager.saved_figures) == 2
        assert "fig1" in manager.saved_figures
        assert "fig2" in manager.saved_figures

    def test_save_overwrites_existing(self, temp_output_dir, sample_figure):
        """Test that saving overwrites existing files."""
        manager = FigureManager(temp_output_dir)

        # Save once
        paths1 = manager.save_figure(sample_figure, "overwrite", formats=["png"])
        mtime1 = paths1["png"].stat().st_mtime

        # Save again with same name
        paths2 = manager.save_figure(sample_figure, "overwrite", formats=["png"])
        mtime2 = paths2["png"].stat().st_mtime

        assert paths2["png"].exists()
        assert mtime2 >= mtime1  # Second write should be same or later time


class TestEmbedAsBase64:
    """Test embed_as_base64 method."""

    def test_embed_png_format(self, temp_output_dir, sample_figure):
        """Test embedding figure as base64 PNG."""
        manager = FigureManager(temp_output_dir)
        base64_str = manager.embed_as_base64(sample_figure, format="png")

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(base64_str)
        assert decoded.startswith(b"\x89PNG")  # PNG magic number

    def test_embed_svg_format(self, temp_output_dir, sample_figure):
        """Test embedding figure as base64 SVG."""
        manager = FigureManager(temp_output_dir)
        base64_str = manager.embed_as_base64(sample_figure, format="svg")

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # SVG should decode to XML
        decoded = base64.b64decode(base64_str).decode("utf-8")
        assert "<?xml" in decoded or "<svg" in decoded

    def test_embed_default_format(self, temp_output_dir, sample_figure):
        """Test that default embed format is PNG."""
        manager = FigureManager(temp_output_dir)
        base64_str = manager.embed_as_base64(sample_figure)

        decoded = base64.b64decode(base64_str)
        assert decoded.startswith(b"\x89PNG")

    def test_embed_with_custom_dpi(self, temp_output_dir, sample_figure):
        """Test embedding with custom DPI."""
        manager = FigureManager(temp_output_dir)

        base64_low = manager.embed_as_base64(sample_figure, dpi=72)
        base64_high = manager.embed_as_base64(sample_figure, dpi=300)

        # Higher DPI should produce larger image (more bytes)
        assert len(base64_high) > len(base64_low)

    def test_embed_returns_string_without_prefix(self, temp_output_dir, sample_figure):
        """Test that embed returns base64 string without data URI prefix."""
        manager = FigureManager(temp_output_dir)
        base64_str = manager.embed_as_base64(sample_figure)

        # Should not contain data URI prefix
        assert not base64_str.startswith("data:image")
        # Should be pure base64
        assert base64_str.isalnum() or "=" in base64_str or "+" in base64_str


class TestGetSavedPath:
    """Test get_saved_path method."""

    def test_get_existing_path(self, temp_output_dir, sample_figure):
        """Test retrieving path to existing saved figure."""
        manager = FigureManager(temp_output_dir)
        manager.save_figure(sample_figure, "test_get", formats=["png", "svg"])

        png_path = manager.get_saved_path("test_get", "png")
        svg_path = manager.get_saved_path("test_get", "svg")

        assert png_path == temp_output_dir / "test_get.png"
        assert svg_path == temp_output_dir / "test_get.svg"
        assert png_path.exists()
        assert svg_path.exists()

    def test_get_nonexistent_figure(self, temp_output_dir):
        """Test retrieving path for nonexistent figure."""
        manager = FigureManager(temp_output_dir)
        path = manager.get_saved_path("nonexistent", "png")

        assert path is None

    def test_get_nonexistent_format(self, temp_output_dir, sample_figure):
        """Test retrieving nonexistent format for existing figure."""
        manager = FigureManager(temp_output_dir)
        manager.save_figure(sample_figure, "test_fmt", formats=["png"])

        png_path = manager.get_saved_path("test_fmt", "png")
        svg_path = manager.get_saved_path("test_fmt", "svg")

        assert png_path is not None
        assert svg_path is None


class TestListSavedFigures:
    """Test list_saved_figures method."""

    def test_list_empty(self, temp_output_dir):
        """Test listing when no figures saved."""
        manager = FigureManager(temp_output_dir)
        figures = manager.list_saved_figures()

        assert figures == []
        assert isinstance(figures, list)

    def test_list_single_figure(self, temp_output_dir, sample_figure):
        """Test listing with single saved figure."""
        manager = FigureManager(temp_output_dir)
        manager.save_figure(sample_figure, "single", formats=["png"])

        figures = manager.list_saved_figures()

        assert len(figures) == 1
        assert "single" in figures

    def test_list_multiple_figures(self, temp_output_dir, sample_figure):
        """Test listing multiple saved figures."""
        manager = FigureManager(temp_output_dir)
        manager.save_figure(sample_figure, "fig1", formats=["png"])
        manager.save_figure(sample_figure, "fig2", formats=["svg"])
        manager.save_figure(sample_figure, "fig3", formats=["pdf"])

        figures = manager.list_saved_figures()

        assert len(figures) == 3
        assert set(figures) == {"fig1", "fig2", "fig3"}

    def test_list_returns_copy(self, temp_output_dir, sample_figure):
        """Test that list returns names, not affecting internal state."""
        manager = FigureManager(temp_output_dir)
        manager.save_figure(sample_figure, "test", formats=["png"])

        figures1 = manager.list_saved_figures()
        figures2 = manager.list_saved_figures()

        # Should return consistent results
        assert figures1 == figures2
        # Modifying returned list shouldn't affect manager
        figures1.append("fake")
        assert "fake" not in manager.list_saved_figures()


class TestIntegration:
    """Integration tests for FigureManager."""

    def test_full_workflow(self, temp_output_dir, sample_figure):
        """Test complete workflow: init, save, embed, retrieve."""
        # Initialize
        manager = FigureManager(temp_output_dir)

        # Save figure
        paths = manager.save_figure(sample_figure, "workflow_test", formats=["png", "svg"])
        assert len(paths) == 2

        # Embed figure
        base64_img = manager.embed_as_base64(sample_figure)
        assert len(base64_img) > 0

        # Retrieve paths
        png_path = manager.get_saved_path("workflow_test", "png")
        assert png_path is not None
        assert png_path.exists()

        # List figures
        figures = manager.list_saved_figures()
        assert "workflow_test" in figures

    def test_multiple_managers_same_directory(self, temp_output_dir, sample_figure):
        """Test multiple managers can work with same directory."""
        manager1 = FigureManager(temp_output_dir)
        manager2 = FigureManager(temp_output_dir)

        manager1.save_figure(sample_figure, "from_m1", formats=["png"])
        manager2.save_figure(sample_figure, "from_m2", formats=["png"])

        # Both files should exist
        assert (temp_output_dir / "from_m1.png").exists()
        assert (temp_output_dir / "from_m2.png").exists()

        # But registries are separate
        assert "from_m1" in manager1.list_saved_figures()
        assert "from_m2" in manager2.list_saved_figures()
        assert "from_m2" not in manager1.list_saved_figures()
        assert "from_m1" not in manager2.list_saved_figures()
