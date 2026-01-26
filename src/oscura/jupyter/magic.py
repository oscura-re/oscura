"""IPython magic commands for Oscura.

This module provides IPython/Jupyter magic commands for convenient
trace analysis in notebooks.

  - %oscura load <file> - Load a trace file
  - %oscura measure - Run measurements on current trace
  - %%analyze - Multi-line analysis cell
  - Auto-display of results with rich HTML

Example:
    In [1]: %load_ext oscura

    In [2]: %oscura load capture.wfm
    Loaded: WaveformTrace with 10000 samples @ 1 GSa/s

    In [3]: %oscura measure rise_time fall_time
    rise_time: 2.5 ns
    fall_time: 2.8 ns

    In [4]: %%analyze
       ...: trace = load("capture.wfm")
       ...: print(f"THD: {thd(trace):.2f} dB")

References:
    - IPython Magic Commands documentation
    - Jupyter display architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell


# Store current trace for magic commands
_current_trace: Any = None
_current_file: str | None = None


def get_current_trace() -> Any:
    """Get the currently loaded trace from magic commands."""
    return _current_trace


def set_current_trace(trace: Any, filename: str | None = None) -> None:
    """Set the current trace for magic commands."""
    global _current_trace, _current_file
    _current_trace = trace
    _current_file = filename


try:
    from IPython.core.magic import (
        Magics,
        cell_magic,
        line_magic,
        magics_class,
    )
    from IPython.display import HTML, display  # noqa: F401

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

    class Magics:  # type: ignore[no-redef]
        """Fallback Magics class when IPython not available."""

    def magics_class(cls: Any) -> Any:
        """Dummy decorator when IPython not available."""
        return cls

    def line_magic(func):  # type: ignore[no-untyped-def]
        """Dummy decorator."""
        return func

    def cell_magic(func):  # type: ignore[no-untyped-def]
        """Dummy decorator."""
        return func


@magics_class
class OscuraMagics(Magics):
    """IPython magics for Oscura analysis.

    Provides convenient shortcuts for loading traces, running measurements,
    and displaying results in Jupyter notebooks.
    """

    @line_magic  # type: ignore[untyped-decorator]
    def oscura(self, line: str) -> Any:
        """Oscura line magic for quick operations.

        Usage:
            %oscura load <filename>    - Load a trace file
            %oscura measure [names...] - Run measurements
            %oscura info               - Show current trace info
            %oscura formats            - List supported formats
            %oscura help               - Show help

        Args:
            line: Magic command arguments

        Returns:
            Command result or None
        """
        import oscura as osc

        parts = line.strip().split()
        if not parts:
            return self._show_help()

        cmd = parts[0].lower()

        if cmd == "load":
            if len(parts) < 2:
                print("Usage: %oscura load <filename>")
                return None
            filename = " ".join(parts[1:])
            return self._load_trace(filename)

        elif cmd == "measure":
            measurements = parts[1:] if len(parts) > 1 else None
            return self._run_measurements(measurements)

        elif cmd == "info":
            return self._show_trace_info()

        elif cmd == "formats":
            formats = osc.get_supported_formats()
            print("Supported formats:")
            for fmt in formats:
                print(f"  - {fmt}")
            return formats

        elif cmd == "help":
            return self._show_help()

        else:
            print(f"Unknown command: {cmd}")
            return self._show_help()

    def _load_trace(self, filename: str) -> Any:
        """Load a trace file and store as current."""
        import oscura as osc

        try:
            trace = osc.load(filename)
            set_current_trace(trace, filename)

            # Display summary
            info = {
                "file": filename,
                "type": type(trace).__name__,
                "samples": len(trace.data) if hasattr(trace, "data") else "N/A",
                "sample_rate": f"{trace.metadata.sample_rate / 1e9:.3f} GSa/s"
                if hasattr(trace, "metadata")
                else "N/A",
            }

            print(f"Loaded: {info['type']} with {info['samples']} samples @ {info['sample_rate']}")
            return trace

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def _run_measurements(self, measurement_names: list[str] | None) -> dict[str, Any]:
        """Run measurements on current trace."""
        import oscura as osc

        trace = get_current_trace()
        if trace is None:
            print("No trace loaded. Use: %oscura load <filename>")
            return {}

        if measurement_names:
            # Run specific measurements
            results = {}
            for name in measurement_names:
                if hasattr(osc, name):
                    try:
                        func = getattr(osc, name)
                        results[name] = func(trace)
                    except Exception as e:
                        results[name] = f"Error: {e}"
                else:
                    results[name] = "Unknown measurement"
        else:
            # Run all measurements
            try:
                results = osc.measure(trace)
            except Exception as e:
                print(f"Error running measurements: {e}")
                return {}

        # Display results
        self._display_measurements(results)
        return results

    def _display_measurements(self, results: dict[str, Any]) -> None:
        """Display measurement results with formatting."""
        if IPYTHON_AVAILABLE:
            from oscura.jupyter.display import display_measurements

            display_measurements(results)
        else:
            for name, value in results.items():
                if isinstance(value, float):
                    print(f"{name}: {value:.6g}")
                else:
                    print(f"{name}: {value}")

    def _show_trace_info(self) -> dict[str, Any] | None:
        """Show information about current trace."""
        trace = get_current_trace()
        if trace is None:
            print("No trace loaded. Use: %oscura load <filename>")
            return None

        info = {
            "file": _current_file,
            "type": type(trace).__name__,
        }

        if hasattr(trace, "data"):
            info["samples"] = len(trace.data)  # type: ignore[assignment]

        if hasattr(trace, "metadata"):
            meta = trace.metadata
            if hasattr(meta, "sample_rate"):
                info["sample_rate"] = meta.sample_rate
            if hasattr(meta, "channel_name"):
                info["channel"] = meta.channel_name

        for key, value in info.items():
            print(f"{key}: {value}")

        return info

    def _show_help(self) -> None:
        """Show magic command help."""
        help_text = """
Oscura Magic Commands
=======================

%oscura load <filename>    Load a trace file
%oscura measure [names...] Run measurements on current trace
%oscura info               Show current trace info
%oscura formats            List supported file formats
%oscura help               Show this help

%%analyze                    Multi-line analysis cell magic

Available measurements: rise_time, fall_time, frequency, period,
    amplitude, rms, overshoot, undershoot, thd, snr, sinad

Example:
    %oscura load capture.wfm
    %oscura measure rise_time fall_time
"""
        print(help_text)

    @cell_magic  # type: ignore[untyped-decorator]
    def analyze(self, line: str, cell: str) -> Any:
        """Cell magic for multi-line analysis.

        Args:
            line: Magic command line arguments (unused).
            cell: Multi-line cell content to execute.

        Returns:
            Result from cell execution (if 'result' variable defined).

        Usage:
            %%analyze
            trace = load("capture.wfm")
            result = measure(trace)
            print(f"Rise time: {result['rise_time']}")

        All oscura functions are auto-imported in the cell namespace.
        """
        import oscura as osc

        # Build execution namespace with oscura imports
        namespace = {
            "osc": osc,
            "load": osc.load,
            "measure": osc.measure,
            "fft": osc.fft,
            "psd": osc.psd,
            "thd": osc.thd,
            "snr": osc.snr,
            "rise_time": osc.rise_time,
            "fall_time": osc.fall_time,
            "frequency": osc.frequency,
            "amplitude": osc.amplitude,
            "low_pass": osc.low_pass,
            "high_pass": osc.high_pass,
        }

        # Add current trace if available
        trace = get_current_trace()
        if trace is not None:
            namespace["trace"] = trace

        # Execute cell
        exec(cell, namespace)

        # Return any result variable if defined
        return namespace.get("result")


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Load Oscura IPython extension.

    Called when user runs: %load_ext oscura

    Args:
        ipython: The IPython shell instance
    """
    ipython.register_magics(OscuraMagics)
    print("Oscura magics loaded. Type '%oscura help' for usage.")


def unload_ipython_extension(ipython: InteractiveShell) -> None:
    """Unload Oscura IPython extension.

    Args:
        ipython: The IPython shell instance
    """
    # IPython handles magic cleanup automatically
