"""Comprehensive unit tests for benchmark.py CLI module.

This module provides extensive testing for the Oscura benchmark command, including:
- Performance benchmarking of core operations
- Load, decode, FFT, and measurement benchmarks
- Test data generation
- Output format handling
- Iteration control
- Error handling

Test Coverage:
- benchmark() CLI command with all options
- _generate_test_data() test data creation
- _benchmark_load() file loading performance
- _benchmark_decode() protocol decoding performance
- _benchmark_fft() FFT computation performance
- _benchmark_measurements() measurement performance
- _print_table() results formatting
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.benchmark import (
    _benchmark_decode,
    _benchmark_fft,
    _benchmark_load,
    _benchmark_measurements,
    _generate_test_data,
    _print_table,
    benchmark,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_test_data():
    """Create sample test data for benchmarking."""
    metadata = TraceMetadata(sample_rate=1e6)
    data = np.sin(2 * np.pi * 1000 * np.arange(10000) / 1e6)
    return WaveformTrace(data=data, metadata=metadata)


# =============================================================================
# Test _generate_test_data()
# =============================================================================


@pytest.mark.unit
def test_generate_test_data_basic():
    """Test basic test data generation."""
    trace = _generate_test_data()

    assert isinstance(trace, WaveformTrace)
    assert len(trace.data) == 100000  # 100k samples
    assert trace.metadata.sample_rate == 1e6


@pytest.mark.unit
def test_generate_test_data_contains_signals():
    """Test that generated data contains mixed sine waves."""
    trace = _generate_test_data()

    # Data should not be constant
    assert not np.all(trace.data == trace.data[0])
    # Should be roughly in range [-1.5, 1.5] (sum of two sine waves)
    assert np.abs(trace.data).max() < 2.0


# =============================================================================
# Test _benchmark_load()
# =============================================================================


@pytest.mark.unit
def test_benchmark_load_basic(sample_test_data):
    """Test basic load benchmarking."""
    with patch("oscura.loaders.load") as mock_load:
        with patch("numpy.save"):
            with patch("pathlib.Path.unlink"):
                mock_load.return_value = sample_test_data

                result = _benchmark_load(sample_test_data, iterations=5)

                assert "total_time" in result
                assert "avg_time" in result
                assert "throughput" in result
                assert "s" in result["total_time"]
                assert "ms" in result["avg_time"]
                assert "ops/sec" in result["throughput"]


@pytest.mark.unit
def test_benchmark_load_creates_temp_file(sample_test_data):
    """Test that load benchmark creates and cleans up temp file."""
    with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
        with patch("numpy.save"):
            with patch("oscura.loaders.load") as mock_load:
                with patch("pathlib.Path.unlink") as mock_unlink:
                    mock_temp = Mock()
                    mock_temp.name = "/tmp/test.npy"
                    mock_tempfile.return_value.__enter__.return_value = mock_temp
                    mock_load.return_value = sample_test_data

                    _benchmark_load(sample_test_data, iterations=1)

                    # Temp file should be cleaned up
                    mock_unlink.assert_called_once()


@pytest.mark.unit
def test_benchmark_load_measures_iterations(sample_test_data):
    """Test that load benchmark runs specified number of iterations."""
    with patch("oscura.loaders.load") as mock_load:
        with patch("numpy.save"):
            with patch("pathlib.Path.unlink"):
                mock_load.return_value = sample_test_data

                _benchmark_load(sample_test_data, iterations=10)

                assert mock_load.call_count == 10


# =============================================================================
# Test _benchmark_decode()
# =============================================================================


@pytest.mark.unit
def test_benchmark_decode_uart(sample_test_data):
    """Test UART decode benchmarking."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        result = _benchmark_decode(sample_test_data, "uart", iterations=5)

        assert "protocol" in result
        assert result["protocol"] == "uart"
        assert "total_time" in result
        assert "avg_time" in result
        assert "throughput" in result
        # Should run 5 iterations
        assert mock_decoder.decode.call_count == 5


@pytest.mark.unit
def test_benchmark_decode_converts_to_digital(sample_test_data):
    """Test that decode benchmark converts waveform to digital."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        _benchmark_decode(sample_test_data, "uart", iterations=1)

        # Should call decode with DigitalTrace
        call_args = mock_decoder.decode.call_args
        from oscura.core.types import DigitalTrace

        assert isinstance(call_args[0][0], DigitalTrace)


@pytest.mark.unit
def test_benchmark_decode_uses_threshold(sample_test_data):
    """Test that decode benchmark uses mean threshold for conversion."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        _benchmark_decode(sample_test_data, "uart", iterations=1)

        # Verify threshold was used (data > mean)
        call_args = mock_decoder.decode.call_args
        digital_trace = call_args[0][0]
        # With sine wave, roughly half should be True
        assert 0.3 < digital_trace.data.mean() < 0.7


# =============================================================================
# Test _benchmark_fft()
# =============================================================================


@pytest.mark.unit
def test_benchmark_fft_basic(sample_test_data):
    """Test basic FFT benchmarking."""
    with patch("oscura.analyzers.waveform.spectral.fft") as mock_fft:
        mock_fft.return_value = (np.array([0.0]), np.array([1.0]))

        result = _benchmark_fft(sample_test_data, iterations=5)

        assert "total_time" in result
        assert "avg_time" in result
        assert "throughput" in result
        assert mock_fft.call_count == 5


@pytest.mark.unit
def test_benchmark_fft_timing_format(sample_test_data):
    """Test FFT benchmark timing format."""
    with patch("oscura.analyzers.waveform.spectral.fft") as mock_fft:
        with patch("time.time") as mock_time:
            mock_fft.return_value = (np.array([0.0]), np.array([1.0]))
            # Simulate 0.5 seconds total
            mock_time.side_effect = [0.0, 0.5]

            result = _benchmark_fft(sample_test_data, iterations=10)

            # Check time formatting
            assert "0.500s" in result["total_time"]
            assert "50.00ms" in result["avg_time"]  # 0.5 / 10 * 1000
            assert "20.0 ops/sec" in result["throughput"]  # 10 / 0.5


# =============================================================================
# Test _benchmark_measurements()
# =============================================================================


@pytest.mark.unit
def test_benchmark_measurements_basic(sample_test_data):
    """Test basic measurements benchmarking."""
    with patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rt:
        with patch("oscura.analyzers.waveform.measurements.fall_time") as mock_ft:
            mock_rt.return_value = 1e-9
            mock_ft.return_value = 1e-9

            result = _benchmark_measurements(sample_test_data, iterations=5)

            assert "total_time" in result
            assert "avg_time" in result
            assert "throughput" in result
            # Should call both measurements for each iteration
            assert mock_rt.call_count == 5
            assert mock_ft.call_count == 5


@pytest.mark.unit
def test_benchmark_measurements_timing(sample_test_data):
    """Test measurements benchmark timing calculations."""
    with patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rt:
        with patch("oscura.analyzers.waveform.measurements.fall_time") as mock_ft:
            with patch("time.time") as mock_time:
                mock_rt.return_value = 1e-9
                mock_ft.return_value = 1e-9
                # Simulate 0.1 seconds
                mock_time.side_effect = [0.0, 0.1]

                result = _benchmark_measurements(sample_test_data, iterations=5)

                assert "0.100s" in result["total_time"]
                assert "20.00ms" in result["avg_time"]  # 0.1 / 5 * 1000


# =============================================================================
# Test _print_table()
# =============================================================================


@pytest.mark.unit
def test_print_table_basic():
    """Test basic table printing."""
    results = {
        "iterations": 10,
        "benchmarks": {
            "load": {"total_time": "1.0s", "avg_time": "100ms", "throughput": "10 ops/sec"},
            "decode": {
                "protocol": "uart",
                "total_time": "2.0s",
                "avg_time": "200ms",
                "throughput": "5 ops/sec",
            },
        },
    }

    with patch("click.echo") as mock_echo:
        _print_table(results)

        # Should print header, iterations, and each benchmark
        assert mock_echo.call_count > 5
        # Check for expected content
        calls = [str(call) for call in mock_echo.call_args_list]
        full_output = "".join(calls)
        assert "Benchmark Results" in full_output
        assert "Iterations: 10" in full_output
        assert "LOAD:" in full_output
        assert "DECODE:" in full_output


@pytest.mark.unit
def test_print_table_formats_benchmarks():
    """Test table printing formats each benchmark section."""
    results = {
        "iterations": 5,
        "benchmarks": {"fft": {"total_time": "0.5s", "avg_time": "100ms"}},
    }

    with patch("click.echo") as mock_echo:
        _print_table(results)

        calls = [str(call[0][0]) for call in mock_echo.call_args_list if call[0]]
        output = "\n".join(calls)

        assert "FFT:" in output
        assert "total_time: 0.5s" in output
        assert "avg_time: 100ms" in output


# =============================================================================
# Test benchmark() CLI command
# =============================================================================


@pytest.mark.unit
def test_benchmark_command_all_operations(cli_runner):
    """Test benchmark command with all operations."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_load") as mock_load:
            with patch("oscura.cli.benchmark._benchmark_decode") as mock_decode:
                with patch("oscura.cli.benchmark._benchmark_fft") as mock_fft:
                    with patch("oscura.cli.benchmark._benchmark_measurements") as mock_meas:
                        mock_gen.return_value = Mock()
                        mock_load.return_value = {"total_time": "1s"}
                        mock_decode.return_value = {"total_time": "2s"}
                        mock_fft.return_value = {"total_time": "0.5s"}
                        mock_meas.return_value = {"total_time": "0.3s"}

                        result = cli_runner.invoke(
                            benchmark, ["--operations", "all"], obj={"verbose": 0}
                        )

                        assert result.exit_code == 0
                        # All benchmarks should be run
                        mock_load.assert_called_once()
                        mock_decode.assert_called_once()
                        mock_fft.assert_called_once()
                        mock_meas.assert_called_once()


@pytest.mark.unit
def test_benchmark_command_specific_operation(cli_runner):
    """Test benchmark command with specific operation."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_load") as mock_load:
            with patch("oscura.cli.benchmark._benchmark_decode") as mock_decode:
                mock_gen.return_value = Mock()
                mock_load.return_value = {"total_time": "1s"}

                result = cli_runner.invoke(benchmark, ["--operations", "load"], obj={"verbose": 0})

                assert result.exit_code == 0
                # Only load should be run
                mock_load.assert_called_once()
                mock_decode.assert_not_called()


@pytest.mark.unit
def test_benchmark_command_custom_iterations(cli_runner):
    """Test benchmark with custom iteration count."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_fft") as mock_fft:
            mock_gen.return_value = Mock()
            mock_fft.return_value = {"total_time": "1s"}

            result = cli_runner.invoke(
                benchmark, ["--operations", "fft", "--iterations", "50"], obj={"verbose": 0}
            )

            assert result.exit_code == 0
            # Should use 50 iterations
            mock_fft.assert_called_once()
            assert mock_fft.call_args[0][1] == 50


@pytest.mark.unit
def test_benchmark_command_protocol_option(cli_runner):
    """Test benchmark decode with custom protocol."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_decode") as mock_decode:
            mock_gen.return_value = Mock()
            mock_decode.return_value = {"protocol": "spi"}

            result = cli_runner.invoke(
                benchmark, ["--operations", "decode", "--protocol", "spi"], obj={"verbose": 0}
            )

            assert result.exit_code == 0
            # Should pass protocol to decode benchmark
            mock_decode.assert_called_once()
            assert mock_decode.call_args[0][1] == "spi"


@pytest.mark.unit
def test_benchmark_command_json_output(cli_runner):
    """Test benchmark with JSON output format."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_load") as mock_load:
            mock_gen.return_value = Mock()
            mock_load.return_value = {"total_time": "1.0s"}

            result = cli_runner.invoke(
                benchmark, ["--operations", "load", "--output", "json"], obj={"verbose": 0}
            )

            assert result.exit_code == 0
            # Output should be JSON
            assert "{" in result.output
            assert "iterations" in result.output


@pytest.mark.unit
def test_benchmark_command_table_output(cli_runner):
    """Test benchmark with table output format (default)."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_load") as mock_load:
            with patch("oscura.cli.benchmark._print_table") as mock_print:
                mock_gen.return_value = Mock()
                mock_load.return_value = {"total_time": "1s"}

                result = cli_runner.invoke(
                    benchmark, ["--operations", "load", "--output", "table"], obj={"verbose": 0}
                )

                assert result.exit_code == 0
                mock_print.assert_called_once()


@pytest.mark.unit
def test_benchmark_command_verbose_logging(cli_runner, caplog):
    """Test benchmark with verbose logging."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_load") as mock_load:
            mock_gen.return_value = Mock()
            mock_load.return_value = {"total_time": "1s"}

            result = cli_runner.invoke(benchmark, ["--operations", "load"], obj={"verbose": 1})

            assert result.exit_code == 0


@pytest.mark.unit
def test_benchmark_command_error_handling(cli_runner):
    """Test benchmark error handling."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        mock_gen.side_effect = Exception("Failed to generate data")

        result = cli_runner.invoke(benchmark, ["--operations", "all"], obj={"verbose": 0})

        assert result.exit_code == 1
        assert "Error: Failed to generate data" in result.output


@pytest.mark.unit
def test_benchmark_command_error_with_verbose(cli_runner):
    """Test benchmark error handling with verbose mode (should raise)."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        mock_gen.side_effect = ValueError("Test error")

        result = cli_runner.invoke(benchmark, ["--operations", "all"], obj={"verbose": 2})

        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


@pytest.mark.unit
def test_benchmark_command_measurements_operation(cli_runner):
    """Test benchmark with measurements operation."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_measurements") as mock_meas:
            mock_gen.return_value = Mock()
            mock_meas.return_value = {"total_time": "0.5s"}

            result = cli_runner.invoke(
                benchmark, ["--operations", "measurements"], obj={"verbose": 0}
            )

            assert result.exit_code == 0
            mock_meas.assert_called_once()


@pytest.mark.unit
def test_benchmark_command_default_iterations():
    """Test that default iterations is 10."""
    from oscura.cli.benchmark import benchmark as bench_cmd

    # Check default value in command options
    for param in bench_cmd.params:
        if param.name == "iterations":
            assert param.default == 10


@pytest.mark.unit
def test_benchmark_all_includes_all_operations(cli_runner):
    """Test that 'all' operation runs all benchmark types."""
    with patch("oscura.cli.benchmark._generate_test_data") as mock_gen:
        with patch("oscura.cli.benchmark._benchmark_load") as mock_load:
            with patch("oscura.cli.benchmark._benchmark_decode") as mock_decode:
                with patch("oscura.cli.benchmark._benchmark_fft") as mock_fft:
                    with patch("oscura.cli.benchmark._benchmark_measurements") as mock_meas:
                        mock_gen.return_value = Mock()
                        mock_load.return_value = {}
                        mock_decode.return_value = {}
                        mock_fft.return_value = {}
                        mock_meas.return_value = {}

                        result = cli_runner.invoke(
                            benchmark, ["--operations", "all"], obj={"verbose": 0}
                        )

                        assert result.exit_code == 0
                        # Verify all operations were called
                        assert mock_load.called
                        assert mock_decode.called
                        assert mock_fft.called
                        assert mock_meas.called
