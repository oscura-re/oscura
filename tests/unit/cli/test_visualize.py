"""Comprehensive unit tests for visualize.py CLI module.

This module provides extensive testing for the Oscura visualize command, including:
- Interactive waveform visualization
- IQ trace visualization
- Protocol overlay annotations
- Save to file functionality
- Matplotlib integration
- Edge detection and marking

Test Coverage:
- visualize() CLI command with all options
- _add_protocol_overlay() annotation function
- IQ trace handling
- Waveform plotting
- Save vs show modes
- Error handling
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.visualize import _add_protocol_overlay, visualize
from oscura.core.types import IQTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_metadata():
    """Create sample trace metadata."""
    return TraceMetadata(sample_rate=10e6, vertical_scale=1.0, vertical_offset=0.0)


@pytest.fixture
def sample_waveform_trace(sample_metadata):
    """Create sample waveform trace."""
    data = np.sin(2 * np.pi * 1000 * np.arange(1000) / 10e6)
    return WaveformTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def sample_iq_trace(sample_metadata):
    """Create sample IQ trace."""
    i_data = np.cos(2 * np.pi * 1000 * np.arange(1000) / 10e6)
    q_data = np.sin(2 * np.pi * 1000 * np.arange(1000) / 10e6)
    return IQTrace(i_data=i_data, q_data=q_data, metadata=sample_metadata)


# =============================================================================
# Test _add_protocol_overlay()
# =============================================================================


@pytest.mark.unit
def test_add_protocol_overlay_marks_edges(sample_waveform_trace):
    """Test that protocol overlay marks signal edges."""
    mock_ax = Mock()

    _add_protocol_overlay(mock_ax, sample_waveform_trace, "uart")

    # Should add vertical lines for edges
    assert mock_ax.axvline.called


@pytest.mark.unit
def test_add_protocol_overlay_limits_edges(sample_waveform_trace):
    """Test that protocol overlay limits edge count to 100."""
    mock_ax = Mock()

    _add_protocol_overlay(mock_ax, sample_waveform_trace, "uart")

    # Should not exceed 100 edge markers
    assert mock_ax.axvline.call_count <= 100


@pytest.mark.unit
def test_add_protocol_overlay_adds_protocol_label(sample_waveform_trace):
    """Test that protocol overlay adds protocol label."""
    mock_ax = Mock()

    _add_protocol_overlay(mock_ax, sample_waveform_trace, "spi")

    # Should add text annotation
    mock_ax.text.assert_called_once()
    # Check that protocol name is in the label
    call_args = str(mock_ax.text.call_args)
    assert "SPI" in call_args


@pytest.mark.unit
def test_add_protocol_overlay_uses_threshold(sample_metadata):
    """Test that overlay uses midpoint threshold for digital conversion."""
    data = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 0.0], dtype=np.float64)
    trace = WaveformTrace(data=data, metadata=sample_metadata)
    mock_ax = Mock()

    _add_protocol_overlay(mock_ax, trace, "uart")

    # Should convert to digital using threshold
    # With data [0, 0, 5, 5, 0, 0], threshold = 2.5
    # Digital should be [False, False, True, True, False, False]
    # Edges at indices 1 and 3 (transitions)
    assert mock_ax.axvline.called


@pytest.mark.unit
def test_add_protocol_overlay_handles_no_edges(sample_metadata):
    """Test overlay with constant signal (no edges)."""
    data = np.ones(100, dtype=np.float64)
    trace = WaveformTrace(data=data, metadata=sample_metadata)
    mock_ax = Mock()

    _add_protocol_overlay(mock_ax, trace, "uart")

    # Should handle gracefully (no edges to mark)
    # May or may not call axvline depending on implementation


@pytest.mark.unit
def test_add_protocol_overlay_time_axis_scaling(sample_waveform_trace):
    """Test that overlay uses millisecond time axis."""
    mock_ax = Mock()

    _add_protocol_overlay(mock_ax, sample_waveform_trace, "uart")

    # Check that axvline calls use ms time units
    if mock_ax.axvline.called:
        # Time values should be small (ms, not seconds)
        call_args = mock_ax.axvline.call_args_list[0]
        time_val = call_args[0][0]
        assert time_val < 1.0  # Should be in ms for short trace


# =============================================================================
# Test visualize() CLI command
# =============================================================================


@pytest.mark.unit
def test_visualize_command_basic(cli_runner, tmp_path):
    """Test basic visualize command execution."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show") as mock_show:
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                mock_metadata = Mock()
                mock_metadata.sample_rate = 1e6
                mock_trace = Mock(spec=WaveformTrace)
                mock_trace.data = np.array([1.0, 2.0, 3.0])
                mock_trace.metadata = mock_metadata
                mock_load.return_value = mock_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                mock_show.assert_called_once()


@pytest.mark.unit
def test_visualize_command_waveform_trace(cli_runner, tmp_path, sample_waveform_trace):
    """Test visualizing a waveform trace."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_waveform_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Should plot the data
                mock_ax.plot.assert_called()


@pytest.mark.unit
def test_visualize_command_iq_trace(cli_runner, tmp_path, sample_iq_trace):
    """Test visualizing an IQ trace."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_iq_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Should plot both I and Q components
                assert mock_ax.plot.call_count == 2
                # Should add legend
                mock_ax.legend.assert_called_once()


@pytest.mark.unit
def test_visualize_command_with_protocol(cli_runner, tmp_path, sample_waveform_trace):
    """Test visualize with protocol overlay."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                with patch("oscura.cli.visualize._add_protocol_overlay") as mock_overlay:
                    mock_fig = Mock()
                    mock_ax = Mock()
                    mock_subplots.return_value = (mock_fig, mock_ax)
                    mock_load.return_value = sample_waveform_trace

                    result = cli_runner.invoke(
                        visualize, [str(test_file), "--protocol", "uart"], obj={"verbose": 0}
                    )

                    assert result.exit_code == 0
                    mock_overlay.assert_called_once()
                    assert mock_overlay.call_args[0][2] == "uart"


@pytest.mark.unit
def test_visualize_command_no_protocol_overlay_for_iq(cli_runner, tmp_path, sample_iq_trace):
    """Test that protocol overlay is skipped for IQ traces."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                with patch("oscura.cli.visualize._add_protocol_overlay") as mock_overlay:
                    mock_fig = Mock()
                    mock_ax = Mock()
                    mock_subplots.return_value = (mock_fig, mock_ax)
                    mock_load.return_value = sample_iq_trace

                    result = cli_runner.invoke(
                        visualize, [str(test_file), "--protocol", "uart"], obj={"verbose": 0}
                    )

                    assert result.exit_code == 0
                    # Should not call overlay for IQ trace
                    mock_overlay.assert_not_called()


@pytest.mark.unit
def test_visualize_command_save_mode(cli_runner, tmp_path, sample_waveform_trace):
    """Test visualize with save to file."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()
    output_file = tmp_path / "plot.png"

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.savefig") as mock_savefig:
                with patch("matplotlib.pyplot.show") as mock_show:
                    mock_fig = Mock()
                    mock_ax = Mock()
                    mock_subplots.return_value = (mock_fig, mock_ax)
                    mock_load.return_value = sample_waveform_trace

                    result = cli_runner.invoke(
                        visualize, [str(test_file), "--save", str(output_file)], obj={"verbose": 0}
                    )

                    assert result.exit_code == 0
                    # Should save instead of show
                    mock_savefig.assert_called_once_with(
                        str(output_file), dpi=300, bbox_inches="tight"
                    )
                    mock_show.assert_not_called()
                    assert f"Saved to: {output_file}" in result.output


@pytest.mark.unit
def test_visualize_command_figure_size(cli_runner, tmp_path, sample_waveform_trace):
    """Test that visualize creates figure with correct size."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_waveform_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Check figsize argument
                mock_subplots.assert_called_once_with(figsize=(12, 6))


@pytest.mark.unit
def test_visualize_command_axis_labels(cli_runner, tmp_path, sample_waveform_trace):
    """Test that visualize sets axis labels."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_waveform_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Check axis labels
                mock_ax.set_xlabel.assert_called()
                mock_ax.set_ylabel.assert_called()
                mock_ax.set_title.assert_called()


@pytest.mark.unit
def test_visualize_command_grid(cli_runner, tmp_path, sample_waveform_trace):
    """Test that visualize enables grid."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_waveform_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Should enable grid
                mock_ax.grid.assert_called_once_with(True, alpha=0.3)


@pytest.mark.unit
def test_visualize_command_verbose_logging(cli_runner, tmp_path, sample_waveform_trace, caplog):
    """Test visualize with verbose logging."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_waveform_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 1})

                assert result.exit_code == 0


@pytest.mark.unit
def test_visualize_command_error_handling(cli_runner, tmp_path):
    """Test visualize error handling."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = Exception("Failed to load")

        result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

        assert result.exit_code == 1
        assert "Error: Failed to load" in result.output


@pytest.mark.unit
def test_visualize_command_error_with_verbose(cli_runner, tmp_path):
    """Test visualize error handling with verbose mode (should raise)."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = ValueError("Test error")

        result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 2})

        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


@pytest.mark.unit
def test_visualize_command_nonexistent_file(cli_runner):
    """Test visualize with nonexistent file."""
    result = cli_runner.invoke(visualize, ["/nonexistent/file.wfm"], obj={"verbose": 0})

    # Click should catch this with path validation
    assert result.exit_code != 0


@pytest.mark.unit
def test_visualize_command_includes_filename_in_title(cli_runner, tmp_path, sample_waveform_trace):
    """Test that visualize includes filename in plot title."""
    test_file = tmp_path / "my_signal.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_waveform_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Check title includes filename
                call_args = str(mock_ax.set_title.call_args)
                assert "my_signal.wfm" in call_args


@pytest.mark.unit
def test_visualize_command_time_axis_in_milliseconds(cli_runner, tmp_path, sample_waveform_trace):
    """Test that visualize plots time axis in milliseconds."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.show"):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                mock_load.return_value = sample_waveform_trace

                result = cli_runner.invoke(visualize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Check xlabel mentions ms
                call_args = str(mock_ax.set_xlabel.call_args)
                assert "ms" in call_args.lower()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_full_visualization_workflow(cli_runner, tmp_path, sample_waveform_trace):
    """Test complete visualization workflow."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()
    output_file = tmp_path / "plot.png"

    with patch("oscura.loaders.load") as mock_load:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            with patch("matplotlib.pyplot.savefig") as mock_savefig:
                with patch("oscura.cli.visualize._add_protocol_overlay") as mock_overlay:
                    mock_fig = Mock()
                    mock_ax = Mock()
                    mock_subplots.return_value = (mock_fig, mock_ax)
                    mock_load.return_value = sample_waveform_trace

                    result = cli_runner.invoke(
                        visualize,
                        [str(test_file), "--protocol", "uart", "--save", str(output_file)],
                        obj={"verbose": 0},
                    )

                    assert result.exit_code == 0
                    # Should load file
                    mock_load.assert_called_once()
                    # Should create plot
                    mock_subplots.assert_called_once()
                    # Should add protocol overlay
                    mock_overlay.assert_called_once()
                    # Should save to file
                    mock_savefig.assert_called_once()
