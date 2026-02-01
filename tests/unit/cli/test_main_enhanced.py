"""Comprehensive tests for enhanced CLI functionality.

Tests cover:
- New subcommands (analyze, export, visualize, benchmark, validate, config)
- Interactive mode
- Progress reporting
- Configuration file support
- Shell completion
- Batch processing improvements
- Output formatting enhancements
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.main import cli, format_output, load_config_file
from oscura.core.types import TraceMetadata, WaveformTrace


@pytest.fixture
def runner():
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_trace():
    """Create sample waveform trace."""
    data = np.sin(np.linspace(0, 2 * np.pi, 1000))
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def temp_waveform(tmp_path, sample_trace):
    """Create temporary waveform file."""
    wfm_file = tmp_path / "test.npy"
    np.save(wfm_file, sample_trace.data)
    return wfm_file


@pytest.fixture
def temp_config(tmp_path):
    """Create temporary config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
analysis:
  default_protocol: uart
  auto_detect_threshold: 0.7

export:
  default_format: json
  output_dir: oscura_output
""")
    return config_file


class TestCLIMain:
    """Test main CLI functionality."""

    def test_cli_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Oscura - Hardware Reverse Engineering Framework" in result.output

    def test_cli_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.9.0" in result.output

    def test_cli_verbose(self, runner):
        """Test verbose mode."""
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["-vv", "--help"])
        assert result.exit_code == 0

    def test_cli_quiet_mode(self, runner):
        """Test quiet mode."""
        result = runner.invoke(cli, ["-q", "--help"])
        assert result.exit_code == 0

    def test_cli_json_output_flag(self, runner):
        """Test JSON output mode flag."""
        result = runner.invoke(cli, ["--json", "--help"])
        assert result.exit_code == 0

    def test_cli_config_loading(self, runner, temp_config):
        """Test configuration file loading."""
        result = runner.invoke(cli, ["--config", str(temp_config), "--help"])
        assert result.exit_code == 0


class TestAnalyzeCommand:
    """Test analyze subcommand."""

    @patch("oscura.loaders.load")
    @patch("oscura.cli.analyze._characterize_signal")
    @patch("oscura.cli.analyze._detect_protocol")
    @patch("oscura.cli.analyze._decode_protocol")
    def test_analyze_basic(
        self,
        mock_decode,
        mock_detect,
        mock_char,
        mock_load,
        runner,
        temp_waveform,
        sample_trace,
    ):
        """Test basic analyze command."""
        mock_load.return_value = sample_trace
        mock_char.return_value = {"sample_rate": "1.0 MHz", "samples": 1000}
        mock_detect.return_value = {"protocol": "uart", "confidence": 0.9}
        mock_decode.return_value = {"packets_decoded": 10, "errors": 0}

        result = runner.invoke(cli, ["analyze", str(temp_waveform)])
        assert result.exit_code == 0

    @patch("oscura.loaders.load")
    @patch("oscura.cli.analyze._characterize_signal")
    @patch("oscura.cli.analyze._detect_protocol")
    @patch("oscura.cli.analyze._decode_protocol")
    def test_analyze_with_protocol_hint(
        self,
        mock_decode,
        mock_detect,
        mock_char,
        mock_load,
        runner,
        temp_waveform,
        sample_trace,
    ):
        """Test analyze with protocol hint."""
        mock_load.return_value = sample_trace
        mock_char.return_value = {"sample_rate": "1.0 MHz"}
        mock_decode.return_value = {"packets_decoded": 5}

        result = runner.invoke(cli, ["analyze", str(temp_waveform), "--protocol", "uart"])
        assert result.exit_code == 0
        mock_detect.assert_not_called()

    @patch("oscura.loaders.load")
    @patch("oscura.cli.analyze._characterize_signal")
    @patch("oscura.cli.analyze._detect_protocol")
    @patch("oscura.cli.analyze._decode_protocol")
    @patch("oscura.cli.analyze._export_results")
    def test_analyze_with_export(
        self,
        mock_export,
        mock_decode,
        mock_detect,
        mock_char,
        mock_load,
        runner,
        temp_waveform,
        sample_trace,
        tmp_path,
    ):
        """Test analyze with export directory."""
        mock_load.return_value = sample_trace
        mock_char.return_value = {}
        mock_detect.return_value = {"protocol": "uart"}
        mock_decode.return_value = {}

        export_dir = tmp_path / "output"
        result = runner.invoke(
            cli, ["analyze", str(temp_waveform), "--export-dir", str(export_dir)]
        )
        assert result.exit_code == 0
        mock_export.assert_called_once()


class TestExportCommand:
    """Test export subcommand."""

    def test_export_json(self, runner, tmp_path):
        """Test JSON export - now raises NotImplementedError."""
        session_file = tmp_path / "session.tks"
        session_file.touch()
        output_file = tmp_path / "output.json"

        result = runner.invoke(
            cli,
            ["export", "json", str(session_file), "--output", str(output_file)],
        )
        # Export has been redesigned and now raises NotImplementedError
        assert result.exit_code == 1
        assert "NotImplementedError" in result.output or "redesigned" in result.output

    def test_export_html(self, runner, tmp_path):
        """Test HTML export - now raises NotImplementedError."""
        session_file = tmp_path / "session.tks"
        session_file.touch()
        output_file = tmp_path / "output.html"

        result = runner.invoke(
            cli,
            ["export", "html", str(session_file), "--output", str(output_file)],
        )
        # Export has been redesigned and now raises NotImplementedError
        assert result.exit_code == 1
        assert "NotImplementedError" in result.output or "redesigned" in result.output


class TestVisualizeCommand:
    """Test visualize subcommand."""

    @patch("oscura.loaders.load")
    @patch("matplotlib.pyplot")
    def test_visualize_save_mode(
        self, mock_plt, mock_load, runner, temp_waveform, sample_trace, tmp_path
    ):
        """Test visualize with save option."""
        mock_load.return_value = sample_trace
        # Configure mock_plt.subplots() to return (fig, ax) tuple
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        output_file = tmp_path / "plot.png"

        result = runner.invoke(
            cli,
            ["visualize", str(temp_waveform), "--save", str(output_file)],
        )
        assert result.exit_code == 0
        mock_plt.savefig.assert_called_once()


class TestBenchmarkCommand:
    """Test benchmark subcommand."""

    def test_benchmark_all(self, runner):
        """Test benchmark all operations."""
        result = runner.invoke(cli, ["benchmark", "--operations", "all", "--iterations", "1"])
        assert result.exit_code == 0
        assert "Benchmark Results" in result.output

    def test_benchmark_specific_operation(self, runner):
        """Test benchmark specific operation."""
        result = runner.invoke(cli, ["benchmark", "--operations", "fft", "--iterations", "1"])
        assert result.exit_code == 0

    def test_benchmark_json_output(self, runner):
        """Test benchmark with JSON output."""
        result = runner.invoke(
            cli,
            ["benchmark", "--operations", "fft", "--iterations", "1", "--output", "json"],
        )
        assert result.exit_code == 0
        assert "{" in result.output


class TestValidateCommand:
    """Test validate subcommand."""

    def test_validate_spec(self, runner, tmp_path):
        """Test spec validation."""
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("""
name: TestProtocol
version: 1.0
description: Test protocol specification
fields:
  - name: header
    type: uint8
""")

        result = runner.invoke(cli, ["validate", str(spec_file)])
        assert result.exit_code == 0
        assert "Validation Results" in result.output

    def test_validate_missing_required_fields(self, runner, tmp_path):
        """Test validation failure on missing fields."""
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("""
description: Missing name and version
""")

        result = runner.invoke(cli, ["validate", str(spec_file)])
        assert result.exit_code == 1
        assert "Missing required field" in result.output

    @patch("oscura.loaders.load")
    def test_validate_with_test_data(self, mock_load, runner, tmp_path, sample_trace):
        """Test validation with test data."""
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("""
name: TestProtocol
version: 1.0
min_samples: 100
sample_rate_min: 1000
""")

        data_file = tmp_path / "data.npy"
        data_file.touch()

        mock_load.return_value = sample_trace

        result = runner.invoke(cli, ["validate", str(spec_file), "--test-data", str(data_file)])
        assert result.exit_code == 0


class TestConfigCommand:
    """Test config subcommand."""

    def test_config_show_no_file(self, runner, tmp_path, monkeypatch):
        """Test config show with no config file."""
        # Mock Path.home() to return tmp_path so config isn't found in user's home
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["config", "--show"])
            assert result.exit_code == 0
            assert "No configuration file found" in result.output

    def test_config_init(self, runner, tmp_path):
        """Test config initialization."""
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["config", "--init"])
            assert result.exit_code == 0
            assert "Initialized configuration" in result.output

    def test_config_path(self, runner):
        """Test config path display."""
        result = runner.invoke(cli, ["config", "--path"])
        assert result.exit_code == 0
        assert "Configuration file:" in result.output

    def test_config_set_value(self, runner, tmp_path):
        """Test setting config value."""
        with runner.isolated_filesystem():
            runner.invoke(cli, ["config", "--init"])
            result = runner.invoke(cli, ["config", "--set", "analysis.default_protocol=spi"])
            assert result.exit_code == 0
            assert "Updated configuration" in result.output


class TestProgressReporter:
    """Test progress reporting functionality."""

    def test_progress_reporter_basic(self):
        """Test basic progress reporter."""
        from oscura.cli.progress import ProgressReporter

        reporter = ProgressReporter(quiet=True, stages=3)
        reporter.start_stage("Stage 1")
        reporter.complete_stage()
        reporter.start_stage("Stage 2")
        reporter.complete_stage()
        reporter.finish()

    def test_progress_reporter_context_manager(self):
        """Test progress reporter as context manager."""
        from oscura.cli.progress import ProgressReporter

        with ProgressReporter(quiet=True, stages=2) as reporter:
            reporter.start_stage("Stage 1")
            reporter.complete_stage()


class TestOutputFormatting:
    """Test output formatting functions."""

    def test_format_output_json(self):
        """Test JSON output format."""
        data = {"key": "value", "number": 42}
        result = format_output(data, "json")
        assert "key" in result
        assert "value" in result
        assert "42" in result

    def test_format_output_csv(self):
        """Test CSV output format."""
        data = {"key": "value", "number": 42}
        result = format_output(data, "csv")
        assert "key,value" in result.split("\n")

    def test_format_output_html(self):
        """Test HTML output format."""
        data = {"key": "value"}
        result = format_output(data, "html")
        assert "<!DOCTYPE html>" in result
        assert "key" in result
        assert "value" in result

    def test_format_output_table(self):
        """Test table output format."""
        data = {"key": "value", "number": 42}
        result = format_output(data, "table")
        assert "key" in result
        assert "value" in result
        assert "42" in result


class TestConfigFileLoading:
    """Test configuration file loading."""

    def test_load_config_file_yaml(self, tmp_path):
        """Test loading YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
test:
  key: value
number: 42
""")

        config = load_config_file(config_file)
        assert config["test"]["key"] == "value"
        assert config["number"] == 42

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        config = load_config_file(Path("/nonexistent/config.yaml"))
        assert config == {}


class TestShellCompletion:
    """Test shell completion functionality."""

    def test_bash_completion_script(self):
        """Test bash completion script generation."""
        from oscura.cli.completion import get_completion_script

        script = get_completion_script("bash")
        assert "_oscura_completion" in script
        assert "complete -F _oscura_completion oscura" in script

    def test_zsh_completion_script(self):
        """Test zsh completion script generation."""
        from oscura.cli.completion import get_completion_script

        script = get_completion_script("zsh")
        assert "#compdef oscura" in script

    def test_fish_completion_script(self):
        """Test fish completion script generation."""
        from oscura.cli.completion import get_completion_script

        script = get_completion_script("fish")
        assert "complete -c oscura" in script

    def test_unsupported_shell(self):
        """Test error on unsupported shell."""
        from oscura.cli.completion import get_completion_script

        with pytest.raises(ValueError, match="Unsupported shell"):
            get_completion_script("invalid")


class TestBackwardCompatibility:
    """Test backward compatibility with existing CLI."""

    def test_characterize_still_works(self, runner):
        """Test that old characterize command still works."""
        result = runner.invoke(cli, ["characterize", "--help"])
        assert result.exit_code == 0

    def test_decode_still_works(self, runner):
        """Test that old decode command still works."""
        result = runner.invoke(cli, ["decode", "--help"])
        assert result.exit_code == 0

    def test_batch_still_works(self, runner):
        """Test that old batch command still works."""
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0

    def test_compare_still_works(self, runner):
        """Test that old compare command still works."""
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0

    def test_shell_still_works(self, runner):
        """Test that shell command still works."""
        result = runner.invoke(cli, ["shell", "--help"])
        assert result.exit_code == 0


class TestErrorHandling:
    """Test error handling in CLI commands."""

    def test_analyze_missing_file(self, runner):
        """Test analyze with missing file."""
        result = runner.invoke(cli, ["analyze", "/nonexistent/file.wfm"])
        assert result.exit_code != 0

    def test_export_invalid_format(self, runner, tmp_path):
        """Test export with invalid format."""
        session_file = tmp_path / "session.tks"
        session_file.touch()

        # Invalid format should be rejected by Click
        result = runner.invoke(
            cli,
            ["export", "invalid", str(session_file), "--output", "out.txt"],
        )
        assert result.exit_code != 0

    def test_config_invalid_set_format(self, runner):
        """Test config set with invalid format."""
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["config", "--set", "invalid_format"])
            assert result.exit_code != 0


# Performance and integration tests
class TestIntegration:
    """Integration tests for CLI workflows."""

    @patch("oscura.loaders.load")
    @patch("oscura.cli.analyze._characterize_signal")
    @patch("oscura.cli.analyze._detect_protocol")
    @patch("oscura.cli.analyze._decode_protocol")
    @patch("oscura.cli.analyze._export_results")
    def test_full_analysis_workflow(
        self,
        mock_export,
        mock_decode,
        mock_detect,
        mock_char,
        mock_load,
        runner,
        temp_waveform,
        sample_trace,
        tmp_path,
    ):
        """Test complete analysis workflow with all options."""
        mock_load.return_value = sample_trace
        mock_char.return_value = {"sample_rate": "1.0 MHz"}
        mock_detect.return_value = {"protocol": "uart"}
        mock_decode.return_value = {"packets_decoded": 10}

        export_dir = tmp_path / "export"
        session_file = tmp_path / "session.tks"

        result = runner.invoke(
            cli,
            [
                "analyze",
                str(temp_waveform),
                "--export-dir",
                str(export_dir),
                "--save-session",
                str(session_file),
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert export_dir.exists()
