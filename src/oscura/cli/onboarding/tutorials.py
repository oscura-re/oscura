"""Interactive tutorial system for Oscura.

This module provides step-by-step interactive tutorials for new users,
covering common analysis workflows.

  - Interactive tutorial system
  - Step-by-step guidance
  - Code examples with explanations
  - Progress tracking

Example:
    >>> from oscura.cli.onboarding import run_tutorial
    >>> run_tutorial("getting_started")
    Welcome to Oscura!
    Step 1/5: Loading a trace file
    ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TutorialStep:
    """A single step in a tutorial.

    Attributes:
        title: Step title
        description: Detailed description with plain English explanation
        code: Example code to run
        expected_output: What the user should see
        hints: Optional hints if stuck
    """

    title: str
    description: str
    code: str
    expected_output: str = ""
    hints: list[str] = field(default_factory=list)
    validation_fn: Callable[..., bool] | None = None


@dataclass
class Tutorial:
    """An interactive tutorial.

    Attributes:
        id: Unique tutorial identifier
        title: Human-readable title
        description: Tutorial overview
        steps: List of tutorial steps
        difficulty: beginner, intermediate, or advanced
    """

    id: str
    title: str
    description: str
    steps: list[TutorialStep]
    difficulty: str = "beginner"


# Built-in tutorials
TUTORIALS: dict[str, Tutorial] = {}


def _register_getting_started() -> None:
    """Register the getting started tutorial."""
    steps = [
        _create_loading_step(),
        _create_measurements_step(),
        _create_spectral_step(),
        _create_protocol_step(),
        _create_discovery_step(),
    ]

    tutorial = Tutorial(
        id="getting_started",
        title="Getting Started with Oscura",
        description="""
Welcome to Oscura! This tutorial will teach you the basics of
signal analysis in 5 easy steps:

1. Loading trace files
2. Making basic measurements
3. Spectral analysis
4. Protocol decoding
5. Auto-discovery

No prior signal analysis experience required!
""",
        steps=steps,
        difficulty="beginner",
    )

    TUTORIALS[tutorial.id] = tutorial


def _create_loading_step() -> TutorialStep:
    """Create loading trace file tutorial step."""
    return TutorialStep(
        title="Loading a Trace File",
        description="""
Oscura can load waveform data from many file formats.
The simplest way is to use the load() function, which auto-detects the format.

Think of a trace like a recording of an electrical signal over time -
similar to how an audio file stores sound waves.
""",
        code="""
import oscura as osc

# Load a waveform file (replace with your file path)
trace = osc.load("signal.csv")

# See basic info
print(f"Loaded {len(trace.data)} samples")
print(f"Sample rate: {trace.metadata.sample_rate} Hz")
""",
        expected_output="Loaded 10000 samples\nSample rate: 1000000.0 Hz",
        hints=[
            "Try loading a CSV file with two columns: time and voltage",
            "Supported formats: .csv, .wfm, .npz, .hdf5, and more",
        ],
    )


def _create_measurements_step() -> TutorialStep:
    """Create measurements tutorial step."""
    return TutorialStep(
        title="Making Basic Measurements",
        description="""
Once you have a trace, you can measure things like:
- Rise time: How fast a signal goes from low to high
- Frequency: How many times per second the signal repeats
- Amplitude: The voltage difference between high and low

These are the same measurements an oscilloscope would show you!
""",
        code="""
import oscura as osc

trace = osc.load("signal.csv")

# Measure rise time (10% to 90% transition)
rt = osc.rise_time(trace)
print(f"Rise time: {rt*1e9:.2f} nanoseconds")

# Measure frequency
freq = osc.frequency(trace)
print(f"Frequency: {freq/1e6:.2f} MHz")

# Get all measurements at once
results = osc.measure(trace)
for name, value in results.items():
    print(f"{name}: {value}")
""",
        expected_output="Rise time: 2.50 nanoseconds\nFrequency: 10.00 MHz",
        hints=[
            "rise_time() measures 10%-90% transition by default",
            "Use measure() to get all measurements in one call",
        ],
    )


def _create_spectral_step() -> TutorialStep:
    """Create spectral analysis tutorial step."""
    return TutorialStep(
        title="Spectral Analysis (Frequency Domain)",
        description="""
Spectral analysis shows you what frequencies are present in your signal.
This is useful for:
- Finding the main frequency of a clock signal
- Detecting noise at specific frequencies
- Measuring signal quality (THD, SNR)

It's like looking at a music equalizer that shows bass, mid, and treble!
""",
        code="""
import oscura as osc

trace = osc.load("signal.csv")

# Compute FFT (Fast Fourier Transform)
freq, magnitude = osc.fft(trace)

# Find the dominant frequency
import numpy as np
peak_idx = np.argmax(magnitude)
print(f"Dominant frequency: {freq[peak_idx]/1e6:.2f} MHz")

# Measure signal quality
thd_value = osc.thd(trace)
snr_value = osc.snr(trace)
print(f"THD: {thd_value:.1f} dB")
print(f"SNR: {snr_value:.1f} dB")
""",
        expected_output="Dominant frequency: 10.00 MHz\nTHD: -45.2 dB\nSNR: 52.3 dB",
        hints=[
            "THD (Total Harmonic Distortion) should be negative in dB - more negative is better",
            "SNR (Signal-to-Noise Ratio) should be positive - higher is better",
        ],
    )


def _create_protocol_step() -> TutorialStep:
    """Create protocol decoding tutorial step."""
    return TutorialStep(
        title="Protocol Decoding",
        description="""
If your signal is a digital communication protocol like UART, SPI, or I2C,
Oscura can decode it to show you the actual data being transmitted.

Think of it like translating Morse code back into text!
""",
        code="""
import oscura as osc

# Load a UART signal
trace = osc.load("uart_signal.csv")

# Decode UART (auto-detects baud rate!)
from oscura.analyzers.protocols import decode_uart
packets = decode_uart(trace)

# Show decoded bytes
for pkt in packets[:5]:  # First 5 packets
    print(f"Time: {pkt.timestamp:.6f}s, Data: 0x{pkt.data:02X} ('{chr(pkt.data)}')")
""",
        expected_output="Time: 0.000001s, Data: 0x48 ('H')\nTime: 0.000086s, Data: 0x65 ('e')",
        hints=[
            "UART baud rate is auto-detected by default",
            "Supported protocols: UART, SPI, I2C, CAN, and many more",
        ],
    )


def _create_discovery_step() -> TutorialStep:
    """Create auto-discovery tutorial step."""
    return TutorialStep(
        title="Auto-Discovery for Beginners",
        description="""
Not sure what your signal is? Oscura can analyze it automatically!

The characterize_signal() function examines your trace and tells you:
- What type of signal it likely is
- Key parameters (voltage, frequency, etc.)
- Suggestions for further analysis

It's like having an expert look at your signal and give you hints!
""",
        code="""
import oscura as osc
from oscura.discovery import characterize_signal

trace = osc.load("mystery_signal.csv")

# Auto-characterize the signal
result = characterize_signal(trace)

print(f"Signal type: {result.signal_type}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Voltage range: {result.voltage_low:.2f}V to {result.voltage_high:.2f}V")

if result.confidence >= 0.8:
    print("High confidence - proceed with suggested analysis")
else:
    print("Consider alternatives:")
    for alt in result.alternatives:
        print(f"  - {alt.signal_type}: {alt.confidence:.1%}")
""",
        expected_output="Signal type: digital\nConfidence: 94.0%",
        hints=[
            "Confidence >= 80% means high confidence in the detection",
            "Low confidence? Check the alternatives for other possibilities",
        ],
    )


def _register_spectral_analysis() -> None:
    """Register the spectral analysis tutorial."""
    steps = [
        TutorialStep(
            title="Understanding FFT",
            description="""
The Fast Fourier Transform (FFT) converts a time-domain signal into
its frequency components. Think of it as breaking a chord into individual notes.
""",
            code="""
import oscura as osc
import numpy as np

trace = osc.load("signal.csv")
freq, mag = osc.fft(trace)

# Magnitude is in dB (decibels)
# 0 dB = full scale, -20 dB = 10x smaller, -40 dB = 100x smaller
print(f"Frequency range: 0 to {freq[-1]/1e6:.1f} MHz")
print(f"Peak magnitude: {np.max(mag):.1f} dB")
""",
            expected_output="Frequency range: 0 to 50.0 MHz\nPeak magnitude: -3.2 dB",
        ),
        TutorialStep(
            title="Power Spectral Density",
            description="""
PSD shows power distribution across frequencies. Unlike FFT magnitude,
PSD is normalized per Hz, making it easier to compare signals with
different durations or sample rates.
""",
            code="""
import oscura as osc

trace = osc.load("signal.csv")
freq, psd = osc.psd(trace)

# Find where most power is concentrated
import numpy as np
total_power = np.sum(psd)
cumsum = np.cumsum(psd) / total_power

# 90% of power is below this frequency
idx_90 = np.searchsorted(cumsum, 0.9)
print(f"90% of signal power below {freq[idx_90]/1e6:.1f} MHz")
""",
            expected_output="90% of signal power below 15.2 MHz",
        ),
    ]

    tutorial = Tutorial(
        id="spectral_analysis",
        title="Spectral Analysis Deep Dive",
        description="Learn advanced spectral analysis techniques.",
        steps=steps,
        difficulty="intermediate",
    )

    TUTORIALS[tutorial.id] = tutorial


# Register built-in tutorials
_register_getting_started()
_register_spectral_analysis()


def list_tutorials() -> list[dict[str, str]]:
    """List all available tutorials.

    Returns:
        List of tutorial info dictionaries with id, title, difficulty
    """
    return [
        {
            "id": t.id,
            "title": t.title,
            "difficulty": t.difficulty,
            "steps": len(t.steps),  # type: ignore[dict-item]
        }
        for t in TUTORIALS.values()
    ]


def get_tutorial(tutorial_id: str) -> Tutorial | None:
    """Get a tutorial by ID.

    Args:
        tutorial_id: Tutorial identifier

    Returns:
        Tutorial object or None if not found
    """
    return TUTORIALS.get(tutorial_id)


def run_tutorial(tutorial_id: str, interactive: bool = True) -> None:
    """Run an interactive tutorial.

    Args:
        tutorial_id: Tutorial to run (e.g., "getting_started")
        interactive: If True, pause between steps for user input

    Example:
        >>> run_tutorial("getting_started")
    """
    tutorial = get_tutorial(tutorial_id)
    if tutorial is None:
        print(f"Tutorial '{tutorial_id}' not found.")
        print("Available tutorials:")
        for t in list_tutorials():
            print(f"  - {t['id']}: {t['title']}")
        return

    print("=" * 60)
    print(f"Tutorial: {tutorial.title}")
    print(f"Difficulty: {tutorial.difficulty}")
    print("=" * 60)
    print(tutorial.description)
    print()

    for i, step in enumerate(tutorial.steps, 1):
        print(f"\n{'=' * 60}")
        print(f"Step {i}/{len(tutorial.steps)}: {step.title}")
        print("=" * 60)
        print(step.description)
        print("\nCode:")
        print("-" * 40)
        print(step.code)
        print("-" * 40)

        if step.expected_output:
            print(f"\nExpected output:\n{step.expected_output}")

        if step.hints:
            print("\nHints:")
            for hint in step.hints:
                print(f"  - {hint}")

        if interactive:
            input("\nPress Enter to continue...")

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("Next steps:")
    print("  - Try the examples with your own data")
    print("  - Run 'list_tutorials()' to see more tutorials")
    print("  - Use 'get_help(function_name)' for detailed help")
