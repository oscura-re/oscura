"""Interactive REPL shell for Oscura exploration.

This module provides an interactive Python shell with Oscura auto-imports,
tab completion, and persistent history for exploratory data analysis.

  - Auto-imports Oscura modules
  - Tab completion for methods and attributes
  - Persistent command history
  - Customized prompt with context info

Example:
    $ oscura shell
    Oscura Shell v0.1.0
    Type 'help()' for Oscura help, 'exit()' to quit.

    In [1]: trace = load("capture.wfm")
    In [2]: rise_time(trace)
    Out[2]: 2.5e-9
    In [3]: freq, mag = fft(trace)

References:
    - Python readline module
    - IPython-style interaction patterns
"""

import atexit
import code
import contextlib
import readline
import rlcompleter
import sys
from pathlib import Path
from typing import Any

# History file location
HISTORY_FILE = Path.home() / ".oscura_history"
HISTORY_LENGTH = 1000


def get_oscura_namespace() -> dict[str, Any]:
    """Build namespace with Oscura auto-imports.

    Returns:
        Dictionary with all commonly-used Oscura functions and classes.
    """
    namespace: dict[str, Any] = {}

    _import_core_oscura(namespace)
    _import_protocols(namespace)
    _import_discovery(namespace)
    _import_common_utilities(namespace)

    return namespace


def _import_core_oscura(namespace: dict[str, Any]) -> None:
    """Import core Oscura functions and types."""
    try:
        import oscura as osc

        namespace["osc"] = osc
        imports = _get_oscura_imports()
        namespace.update(_build_namespace_dict(imports))
    except ImportError as e:
        print(f"Warning: Could not import Oscura: {e}")


def _get_oscura_imports() -> dict[str, Any]:
    """Get all Oscura imports as dictionary.

    Returns:
        Dictionary of imported symbols.
    """
    from oscura import (
        DigitalTrace,
        ProtocolPacket,
        TraceMetadata,
        WaveformTrace,
        add,
        amplitude,
        band_pass,
        band_stop,
        basic_stats,
        detect_edges,
        differentiate,
        divide,
        duty_cycle,
        enob,
        fall_time,
        fft,
        frequency,
        get_supported_formats,
        high_pass,
        histogram,
        integrate,
        load,
        low_pass,
        mean,
        measure,
        multiply,
        overshoot,
        percentiles,
        period,
        psd,
        pulse_width,
        rise_time,
        rms,
        sfdr,
        sinad,
        snr,
        spectrogram,
        subtract,
        thd,
        to_digital,
        undershoot,
    )

    return {
        "WaveformTrace": WaveformTrace,
        "DigitalTrace": DigitalTrace,
        "TraceMetadata": TraceMetadata,
        "ProtocolPacket": ProtocolPacket,
        "load": load,
        "get_supported_formats": get_supported_formats,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "frequency": frequency,
        "period": period,
        "amplitude": amplitude,
        "rms": rms,
        "mean": mean,
        "overshoot": overshoot,
        "undershoot": undershoot,
        "duty_cycle": duty_cycle,
        "pulse_width": pulse_width,
        "measure": measure,
        "fft": fft,
        "psd": psd,
        "thd": thd,
        "snr": snr,
        "sinad": sinad,
        "enob": enob,
        "sfdr": sfdr,
        "spectrogram": spectrogram,
        "to_digital": to_digital,
        "detect_edges": detect_edges,
        "low_pass": low_pass,
        "high_pass": high_pass,
        "band_pass": band_pass,
        "band_stop": band_stop,
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
        "differentiate": differentiate,
        "integrate": integrate,
        "basic_stats": basic_stats,
        "histogram": histogram,
        "percentiles": percentiles,
    }


def _build_namespace_dict(imports: dict[str, Any]) -> dict[str, Any]:
    """Build namespace dictionary from imports.

    Args:
        imports: Dictionary of imported symbols.

    Returns:
        Namespace dictionary.
    """
    return imports


def _import_protocols(namespace: dict[str, Any]) -> None:
    """Import protocol decoders."""
    try:
        from oscura.analyzers.protocols import (
            decode_can,
            decode_i2c,
            decode_spi,
            decode_uart,
        )

        namespace.update(
            {
                "decode_uart": decode_uart,
                "decode_spi": decode_spi,
                "decode_i2c": decode_i2c,
                "decode_can": decode_can,
            }
        )
    except ImportError:
        pass


def _import_discovery(namespace: dict[str, Any]) -> None:
    """Import discovery functions."""
    try:
        from oscura.discovery import (
            characterize_signal,
            decode_protocol,
            find_anomalies,
        )

        namespace.update(
            {
                "characterize_signal": characterize_signal,
                "find_anomalies": find_anomalies,
                "decode_protocol": decode_protocol,
            }
        )
    except ImportError:
        pass


def _import_common_utilities(namespace: dict[str, Any]) -> None:
    """Import common utilities like numpy and matplotlib."""
    try:
        import matplotlib.pyplot as plt

        namespace["plt"] = plt
    except ImportError:
        pass

    try:
        import numpy as np

        namespace["np"] = np
    except ImportError:
        pass


def setup_history() -> None:
    """Set up readline history with persistence."""
    # Enable tab completion
    readline.parse_and_bind("tab: complete")

    # Load history if exists
    if HISTORY_FILE.exists():
        with contextlib.suppress(Exception):
            readline.read_history_file(HISTORY_FILE)

    # Set history length
    readline.set_history_length(HISTORY_LENGTH)

    # Save history on exit
    atexit.register(lambda: readline.write_history_file(HISTORY_FILE))


def oscura_help() -> None:
    """Display Oscura help in the REPL."""
    help_text = """
Oscura Interactive Shell - Quick Reference
=============================================

Loading Data:
    trace = load("file.wfm")           # Auto-detect format
    trace = load("file.csv")           # CSV file
    formats = get_supported_formats()  # List supported formats

Waveform Measurements:
    rise_time(trace)                   # 10-90% rise time
    fall_time(trace)                   # 90-10% fall time
    frequency(trace)                   # Fundamental frequency
    amplitude(trace)                   # Peak-to-peak amplitude
    measure(trace)                     # All measurements

Spectral Analysis:
    freq, mag = fft(trace)             # FFT
    freq, pwr = psd(trace)             # Power Spectral Density
    thd(trace)                         # Total Harmonic Distortion
    snr(trace)                         # Signal-to-Noise Ratio

Digital Analysis:
    digital = to_digital(trace)        # Extract digital signal
    edges = detect_edges(trace)        # Find edges

Filtering:
    filtered = low_pass(trace, 1e6)    # Low-pass filter
    filtered = high_pass(trace, 1e3)   # High-pass filter

Protocol Decoding:
    packets = decode_uart(trace)       # UART decode
    packets = decode_spi(clk, mosi)    # SPI decode
    packets = decode_i2c(scl, sda)     # I2C decode

Discovery (Auto-Analysis):
    result = characterize_signal(trace)  # Auto-characterize
    anomalies = find_anomalies(trace)    # Find anomalies

For detailed help on any function:
    help(function_name)

Full documentation: https://github.com/lair-click-bats/oscura
"""
    print(help_text)


class OscuraConsole(code.InteractiveConsole):
    """Custom interactive console for Oscura.

    Provides IPython-style prompts and enhanced error handling.
    """

    def __init__(self, locals: dict[str, Any] | None = None) -> None:
        """Initialize the console with Oscura namespace."""
        super().__init__(locals=locals, filename="<oscura>")
        self.prompt_counter = 1

    def interact(self, banner: str | None = None, exitmsg: str | None = None) -> None:
        """Start the interactive session."""
        if banner is None:
            import oscura

            banner = f"""
Oscura Shell v{oscura.__version__}
Python {sys.version.split()[0]} on {sys.platform}
Type 'oscura_help()' for quick reference, 'exit()' to quit.

Auto-imported: tk (oscura), np (numpy), plt (matplotlib.pyplot)
Common functions: load, measure, fft, psd, thd, low_pass, high_pass
"""
        if exitmsg is None:
            exitmsg = "Goodbye!"

        super().interact(banner=banner, exitmsg=exitmsg)

    def raw_input(self, prompt: str = "") -> str:
        """Override prompt with counter."""
        custom_prompt = f"In [{self.prompt_counter}]: "
        result = super().raw_input(custom_prompt)
        self.prompt_counter += 1
        return result

    def showtraceback(self) -> None:
        """Show traceback with helpful hints."""
        super().showtraceback()
        # Could add context-sensitive hints here


def start_shell() -> None:
    """Start the Oscura interactive shell.

    This is the main entry point for the REPL, providing:
    - Auto-imported Oscura functions and modules
    - Tab completion
    - Persistent command history
    - Customized prompts
    """
    # Set up history
    setup_history()

    # Build namespace
    namespace = get_oscura_namespace()

    # Add help function
    namespace["oscura_help"] = oscura_help

    # Set up completer
    completer = rlcompleter.Completer(namespace)
    readline.set_completer(completer.complete)

    # Start console
    console = OscuraConsole(locals=namespace)
    console.interact()


if __name__ == "__main__":
    start_shell()
