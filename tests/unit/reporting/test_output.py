"""Tests for comprehensive analysis report output management.

Tests verify the OutputManager correctly creates directories
and saves files in various formats.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from oscura.reporting.config import AnalysisDomain
from oscura.reporting.output import OutputManager

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("cleanup_matplotlib")]


class TestOutputManagerInit:
    """Tests for OutputManager initialization."""

    def test_creates_timestamped_directory_name(self):
        """Test that output manager creates timestamped directory name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ts = datetime(2026, 1, 1, 12, 0, 0)
            manager = OutputManager(base, "test_data", ts)

            assert "20260101_120000" in str(manager.root)
            assert "test_data" in str(manager.root)
            assert "analysis" in str(manager.root)

    def test_timestamp_property(self):
        """Test timestamp property returns correct value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ts = datetime(2026, 6, 15, 14, 30, 45)
            manager = OutputManager(base, "signal", ts)

            assert manager.timestamp == ts
            assert manager.timestamp_str == "20260615_143045"

    def test_default_timestamp(self):
        """Test that timestamp defaults to now if not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "data")

            # Should have a recent timestamp
            now = datetime.now()
            diff = abs((manager.timestamp - now).total_seconds())
            assert diff < 2  # Within 2 seconds


class TestOutputManagerCreate:
    """Tests for OutputManager.create method."""

    def test_creates_root_directory(self):
        """Test that create method creates root directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            root = manager.create()

            assert root.exists()
            assert root.is_dir()

    def test_creates_standard_subdirectories(self):
        """Test that create method creates standard subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            root = manager.create()

            assert (root / "plots").exists()
            assert (root / "errors").exists()
            assert (root / "logs").exists()
            assert (root / "input").exists()

    def test_create_is_idempotent(self):
        """Test that create can be called multiple times safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")

            root1 = manager.create()
            root2 = manager.create()

            assert root1 == root2
            assert root1.exists()


class TestOutputManagerDomainDir:
    """Tests for OutputManager.create_domain_dir method."""

    def test_creates_domain_directory(self):
        """Test creating domain-specific directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            spectral_dir = manager.create_domain_dir(AnalysisDomain.SPECTRAL)

            assert spectral_dir.exists()
            assert spectral_dir.name == "spectral"

    def test_domain_dir_under_root(self):
        """Test domain directory is under root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            domain_dir = manager.create_domain_dir(AnalysisDomain.WAVEFORM)

            assert domain_dir.parent == manager.root


class TestOutputManagerSaveJson:
    """Tests for OutputManager.save_json method."""

    def test_saves_json_file(self):
        """Test saving data as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            data = {"key": "value", "number": 42}
            path = manager.save_json("results", data)

            assert path.exists()
            assert path.suffix == ".json"

            with path.open() as f:
                loaded = json.load(f)
            assert loaded == data

    def test_json_in_subdirectory(self):
        """Test saving JSON in subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            data = {"analysis": "spectral"}
            path = manager.save_json("fft_results", data, subdir="spectral")

            assert path.exists()
            assert "spectral" in str(path)

    def test_json_pretty_printed(self):
        """Test that JSON is pretty-printed with indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            data = {"a": 1, "b": 2}
            path = manager.save_json("data", data)

            content = path.read_text()
            assert "\n" in content  # Multi-line
            assert "  " in content  # Indented


class TestOutputManagerSaveYaml:
    """Tests for OutputManager.save_yaml method."""

    def test_saves_yaml_file(self):
        """Test saving data as YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            data = {"config": {"enabled": True, "value": 100}}
            path = manager.save_yaml("config", data)

            assert path.exists()
            assert path.suffix == ".yaml"

            with path.open() as f:
                loaded = yaml.safe_load(f)
            assert loaded == data

    def test_yaml_in_subdirectory(self):
        """Test saving YAML in subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            data = {"setting": "value"}
            path = manager.save_yaml("settings", data, subdir="config")

            assert path.exists()
            assert "config" in str(path)


class TestOutputManagerSaveText:
    """Tests for OutputManager.save_text method."""

    def test_saves_text_file(self):
        """Test saving text content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            content = "This is a summary report."
            path = manager.save_text("summary.txt", content)

            assert path.exists()
            assert path.read_text() == content

    def test_saves_html_file(self):
        """Test saving HTML content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            html = "<html><body><h1>Report</h1></body></html>"
            path = manager.save_text("index.html", html)

            assert path.exists()
            assert path.suffix == ".html"
            assert path.read_text() == html


class TestOutputManagerSavePlot:
    """Tests for OutputManager.save_plot method."""

    @pytest.fixture
    def simple_figure(self):
        """Create a simple matplotlib figure for testing."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        yield fig
        plt.close(fig)

    def test_saves_png_plot(self, simple_figure):
        """Test saving plot as PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            path = manager.save_plot(
                AnalysisDomain.SPECTRAL,
                "fft",
                simple_figure,
                format="png",
            )

            assert path.exists()
            assert path.suffix == ".png"
            assert "spectral_fft" in path.name

    def test_plot_in_plots_directory(self, simple_figure):
        """Test that plots are saved in plots/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = OutputManager(base, "test")
            manager.create()

            path = manager.save_plot(
                AnalysisDomain.WAVEFORM,
                "time_series",
                simple_figure,
            )

            assert path.parent.name == "plots"
