"""Context-sensitive help and command suggestions.

This module provides plain English help, command suggestions based on
context, and result explanations for non-expert users.


Example:
    >>> from oscura.cli.onboarding import get_help, suggest_commands
    >>> get_help("rise_time")
    >>> suggest_commands(trace)
"""

from __future__ import annotations

from typing import Any

# Plain English help database
HELP_DATABASE: dict[str, dict[str, str | list[str]]] = {
    "rise_time": {
        "summary": "Measures how quickly a signal transitions from low to high",
        "plain_english": """
Rise time tells you how fast your signal can switch from OFF to ON
(or from a low voltage to a high voltage). It's measured between
the 10% and 90% points of the transition by default.

In plain terms: A faster rise time means sharper edges on your signal,
which is important for high-speed digital circuits.

Typical values:
- Slow logic (old TTL): 10-50 nanoseconds
- Fast logic (modern CMOS): 0.1-2 nanoseconds
- High-speed serial (USB, PCIe): 50-200 picoseconds
""",
        "when_to_use": [
            "Characterizing digital buffer performance",
            "Checking if your driver is fast enough for your data rate",
            "Verifying signal integrity (slow rise = possible problems)",
        ],
        "related": ["fall_time", "slew_rate", "frequency"],
    },
    "fall_time": {
        "summary": "Measures how quickly a signal transitions from high to low",
        "plain_english": """
Fall time is the opposite of rise time - it measures how fast your
signal switches from ON to OFF (high voltage to low voltage).

It's measured between the 90% and 10% points of the transition.

Ideally, rise time and fall time should be similar. If they're very
different, it might indicate an issue with your circuit.
""",
        "when_to_use": [
            "Checking symmetry of your driver",
            "Verifying output stage performance",
            "Diagnosing asymmetric signal issues",
        ],
        "related": ["rise_time", "slew_rate", "duty_cycle"],
    },
    "frequency": {
        "summary": "Measures how many times per second your signal repeats",
        "plain_english": """
Frequency tells you how fast your signal is cycling. It's measured
in Hertz (Hz), which means 'cycles per second'.

Common scales:
- 1 kHz = 1,000 cycles/second (audio frequencies)
- 1 MHz = 1,000,000 cycles/second (radio, slow digital)
- 1 GHz = 1,000,000,000 cycles/second (fast digital, RF)

Oscura finds frequency by detecting repeated patterns in your signal.
""",
        "when_to_use": [
            "Verifying clock frequency",
            "Checking oscillator output",
            "Measuring PWM frequency",
        ],
        "related": ["period", "duty_cycle", "fft"],
    },
    "thd": {
        "summary": "Total Harmonic Distortion - measures signal purity",
        "plain_english": """
THD tells you how 'clean' your signal is. A perfect sine wave has
0% THD (or -infinity dB). Real signals have some distortion.

THD in dB (decibels):
- -60 dB or lower: Excellent (high-quality audio)
- -40 to -60 dB: Good (typical electronics)
- -20 to -40 dB: Fair (some distortion visible)
- Above -20 dB: Poor (significant distortion)

Note: THD is expressed as a negative number in dB.
More negative = less distortion = better signal.
""",
        "when_to_use": [
            "Testing audio amplifier quality",
            "Verifying oscillator purity",
            "Characterizing ADC/DAC performance",
        ],
        "related": ["snr", "sinad", "enob"],
    },
    "snr": {
        "summary": "Signal-to-Noise Ratio - measures how much signal vs noise",
        "plain_english": """
SNR tells you how much of your signal is actual signal versus noise.
Higher SNR = cleaner signal with less interference.

SNR in dB:
- 60+ dB: Excellent (barely any noise visible)
- 40-60 dB: Good (clean signal, some noise)
- 20-40 dB: Fair (visible noise)
- Below 20 dB: Poor (noisy signal)

In practical terms: Every 6 dB is roughly doubling the signal level
relative to noise.
""",
        "when_to_use": [
            "Evaluating measurement system quality",
            "Testing ADC performance",
            "Comparing different signal sources",
        ],
        "related": ["thd", "sinad", "enob"],
    },
    "fft": {
        "summary": "Fast Fourier Transform - shows frequency content of signal",
        "plain_english": """
FFT transforms your time-domain signal (voltage vs time) into the
frequency domain (power vs frequency). It's like an equalizer display
that shows what frequencies are present.

Returns two arrays:
- frequencies: The x-axis values in Hz
- magnitudes: The strength at each frequency (usually in dB)

Peaks in the FFT correspond to dominant frequencies in your signal.
A pure sine wave shows one peak. Square waves show peaks at odd
harmonics (1x, 3x, 5x, etc. of the fundamental).
""",
        "when_to_use": [
            "Finding the frequency of an unknown signal",
            "Looking for interference at specific frequencies",
            "Analyzing modulated signals",
        ],
        "related": ["psd", "thd", "snr", "spectrogram"],
    },
    "load": {
        "summary": "Load a trace file - Oscura's starting point",
        "plain_english": """
load() reads waveform data from a file. It auto-detects the format,
so you don't need to specify whether it's CSV, WFM, HDF5, etc.

Returns a WaveformTrace or DigitalTrace object containing:
- data: The actual voltage/value samples
- metadata: Sample rate, channel info, etc.
- time_vector: Time axis (computed from sample rate)

Supported formats: CSV, Tektronix WFM, Rigol WFM, NumPy NPZ,
HDF5, Sigrok sessions, VCD, TDMS, and more.
""",
        "when_to_use": [
            "Starting any Oscura analysis",
            "Loading oscilloscope captures",
            "Importing logic analyzer data",
        ],
        "related": ["get_supported_formats", "WaveformTrace", "DigitalTrace"],
    },
    "measure": {
        "summary": "Run all standard measurements on a trace",
        "plain_english": """
measure() is a convenience function that runs many common measurements
at once and returns them as a dictionary.

It's like clicking 'Auto-Measure' on an oscilloscope.

Measurements include:
- Timing: rise_time, fall_time, frequency, period, duty_cycle
- Amplitude: vpp, vmax, vmin, vmean, vrms
- Waveform quality: overshoot, undershoot

Results are returned in a dictionary for easy access.
""",
        "when_to_use": [
            "Quick signal characterization",
            "Getting an overview of signal properties",
            "When you're not sure which measurements you need",
        ],
        "related": ["rise_time", "frequency", "amplitude", "basic_stats"],
    },
}


def get_help(topic: str) -> str | None:
    """Get plain English help for a Oscura function or concept.

    Args:
        topic: Function name or concept to get help for

    Returns:
        Formatted help text or None if topic not found

    Example:
        >>> print(get_help("rise_time"))
    """
    topic = topic.lower().strip()

    if topic in HELP_DATABASE:
        entry = HELP_DATABASE[topic]
        output = []
        output.append(f"Help: {topic}")
        output.append("=" * 50)
        output.append(f"\n{entry['summary']}\n")
        output.append(entry["plain_english"])  # type: ignore[arg-type]

        if "when_to_use" in entry:
            output.append("\nWhen to use this:")
            for use in entry["when_to_use"]:
                output.append(f"  - {use}")

        if "related" in entry:
            output.append(f"\nRelated: {', '.join(entry['related'])}")

        return "\n".join(output)

    # Try to get docstring
    try:
        import oscura as osc

        if hasattr(osc, topic):
            func = getattr(osc, topic)
            if func.__doc__:
                return f"Help for {topic}:\n\n{func.__doc__}"
    except Exception:
        pass

    return None


def suggest_commands(trace: Any = None, context: str | None = None) -> list[dict[str, str]]:
    """Suggest next commands based on current context.

    Args:
        trace: Current trace object (if any)
        context: Description of what user is trying to do

    Returns:
        List of suggested commands with descriptions

    Example:
        >>> suggestions = suggest_commands(trace)
        >>> for s in suggestions:
        ...     print(f"{s['command']}: {s['description']}")
    """
    suggestions = []

    if trace is None:
        return _suggest_loading_commands()

    # Trace is loaded - suggest measurements
    suggestions.append(
        {
            "command": "measure(trace)",
            "description": "Run all standard measurements",
            "reason": "Quick overview of signal properties",
        }
    )

    # Add signal type-specific suggestions
    _add_signal_type_suggestions(suggestions, trace)

    # Always suggest filtering for noisy signals
    suggestions.append(
        {
            "command": "filtered = low_pass(trace, cutoff_hz)",
            "description": "Apply low-pass filter to remove noise",
            "reason": "Clean up high-frequency noise",
        }
    )

    # Add context-specific suggestions
    if context:
        _add_context_suggestions(suggestions, context)

    return suggestions


def _suggest_loading_commands() -> list[dict[str, str]]:
    """Get suggestions when no trace is loaded."""
    return [
        {
            "command": "trace = load('file.csv')",
            "description": "Load a trace file to get started",
            "reason": "No trace loaded yet",
        },
        {
            "command": "formats = get_supported_formats()",
            "description": "See what file formats are supported",
            "reason": "Helpful for knowing what files you can load",
        },
    ]


def _add_signal_type_suggestions(suggestions: list[dict[str, str]], trace: Any) -> None:
    """Add signal type-specific suggestions.

    Args:
        suggestions: List to append suggestions to.
        trace: Trace object to analyze.
    """
    if hasattr(trace, "data"):
        import numpy as np

        data = trace.data
        unique_levels = len(np.unique(np.round(data, 2)))

        if unique_levels < 5:
            # Likely digital
            suggestions.append(
                {
                    "command": "digital = to_digital(trace)",
                    "description": "Convert to digital trace",
                    "reason": "Signal appears to be digital (few voltage levels)",
                }
            )
            suggestions.append(
                {
                    "command": "characterize_signal(trace)",
                    "description": "Auto-detect signal type and protocol",
                    "reason": "May be a protocol like UART, SPI, I2C",
                }
            )
        else:
            # Likely analog
            suggestions.append(
                {
                    "command": "freq, mag = fft(trace)",
                    "description": "Compute frequency spectrum",
                    "reason": "See what frequencies are present",
                }
            )
            suggestions.append(
                {
                    "command": "thd(trace)",
                    "description": "Measure Total Harmonic Distortion",
                    "reason": "Check signal purity",
                }
            )


def _add_context_suggestions(suggestions: list[dict[str, str]], context: str) -> None:
    """Add context-specific suggestions based on user intent.

    Args:
        suggestions: List to prepend suggestions to.
        context: User context string.
    """
    context_lower = context.lower()

    if "uart" in context_lower or "serial" in context_lower:
        suggestions.insert(
            0,
            {
                "command": "packets = decode_uart(trace)",
                "description": "Decode UART serial data",
                "reason": "You mentioned UART/serial",
            },
        )
    elif "spi" in context_lower:
        suggestions.insert(
            0,
            {
                "command": "packets = decode_spi(clk_trace, data_trace)",
                "description": "Decode SPI bus",
                "reason": "You mentioned SPI",
            },
        )
    elif "i2c" in context_lower:
        suggestions.insert(
            0,
            {
                "command": "packets = decode_i2c(scl_trace, sda_trace)",
                "description": "Decode I2C bus",
                "reason": "You mentioned I2C",
            },
        )


def explain_result(
    value: Any,
    measurement: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Explain a measurement result in plain English.

    Args:
        value: The measurement value
        measurement: Name of the measurement (e.g., "rise_time")
        context: Additional context (e.g., signal type, expected values)

    Returns:
        Plain English explanation of the result

    Example:
        >>> print(explain_result(2.5e-9, "rise_time"))
        "Your rise time is 2.5 nanoseconds, which is..."
    """
    explanations = {
        "rise_time": lambda v: _explain_rise_time(v),
        "fall_time": lambda v: _explain_fall_time(v),
        "frequency": lambda v: _explain_frequency(v),
        "thd": lambda v: _explain_thd(v),
        "snr": lambda v: _explain_snr(v),
    }

    if measurement.lower() in explanations:
        return explanations[measurement.lower()](value)

    # Generic explanation
    return f"{measurement}: {value}"


def _explain_rise_time(value: float) -> str:
    """Explain rise time result."""
    if value < 1e-12:
        return f"Rise time: {value * 1e12:.2f} ps - Extremely fast! Sub-picosecond edge."
    elif value < 1e-9:
        return f"Rise time: {value * 1e12:.0f} ps - Very fast, typical of high-speed serial links."
    elif value < 10e-9:
        return f"Rise time: {value * 1e9:.2f} ns - Fast, suitable for most digital circuits."
    elif value < 100e-9:
        return f"Rise time: {value * 1e9:.1f} ns - Moderate, typical of standard logic."
    else:
        return f"Rise time: {value * 1e6:.2f} us - Slow, may limit data rate."


def _explain_fall_time(value: float) -> str:
    """Explain fall time result."""
    if value < 1e-9:
        return f"Fall time: {value * 1e12:.0f} ps - Very fast falling edge."
    elif value < 10e-9:
        return f"Fall time: {value * 1e9:.2f} ns - Fast, good for digital circuits."
    else:
        return f"Fall time: {value * 1e9:.1f} ns - Relatively slow falling edge."


def _explain_frequency(value: float) -> str:
    """Explain frequency result."""
    if value < 1e3:
        return f"Frequency: {value:.1f} Hz - Audio range or very slow signal."
    elif value < 1e6:
        return f"Frequency: {value / 1e3:.2f} kHz - Low frequency signal."
    elif value < 1e9:
        return f"Frequency: {value / 1e6:.2f} MHz - Radio/digital clock range."
    else:
        return f"Frequency: {value / 1e9:.3f} GHz - High-speed digital or RF."


def _explain_thd(value: float) -> str:
    """Explain THD result."""
    if value < -60:
        return f"THD: {value:.1f} dB - Excellent! Very low distortion (high-fidelity)."
    elif value < -40:
        return f"THD: {value:.1f} dB - Good, typical for quality electronics."
    elif value < -20:
        return f"THD: {value:.1f} dB - Fair, some distortion present."
    else:
        return f"THD: {value:.1f} dB - Poor, significant distortion visible."


def _explain_snr(value: float) -> str:
    """Explain SNR result."""
    if value > 60:
        return f"SNR: {value:.1f} dB - Excellent! Very clean signal."
    elif value > 40:
        return f"SNR: {value:.1f} dB - Good signal-to-noise ratio."
    elif value > 20:
        return f"SNR: {value:.1f} dB - Fair, some noise present."
    else:
        return f"SNR: {value:.1f} dB - Poor, noisy signal."


def get_example(function_name: str) -> str | None:
    """Get a code example for a function.

    Args:
        function_name: Name of the function

    Returns:
        Example code string or None
    """
    examples = {
        "load": """
# Load a trace file
import oscura as osc
trace = osc.load("capture.csv")
print(f"Loaded {len(trace.data)} samples")
""",
        "rise_time": """
# Measure rise time
import oscura as osc
trace = osc.load("signal.csv")
rt = osc.rise_time(trace)
print(f"Rise time: {rt*1e9:.2f} ns")
""",
        "fft": """
# Compute FFT spectrum
import oscura as osc
trace = osc.load("signal.csv")
freq, mag = osc.fft(trace)
print(f"Frequency resolution: {freq[1]:.2f} Hz")
""",
        "measure": """
# Run all measurements
import oscura as osc
trace = osc.load("signal.csv")
results = osc.measure(trace)
for name, value in results.items():
    print(f"{name}: {value}")
""",
    }

    return examples.get(function_name.lower())
