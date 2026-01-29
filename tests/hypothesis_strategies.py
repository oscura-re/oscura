"""Custom Hypothesis strategies for Oscura testing.

This module provides reusable hypothesis strategies for property-based testing
across the Oscura codebase. These strategies generate realistic test data that
matches the structure and constraints of real-world signal analysis scenarios.

Usage:
    from tests.hypothesis_strategies import digital_signals, protocol_messages

    @given(signal=digital_signals())
    def test_signal_processing(signal):
        ...
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import strategies as st
from numpy.typing import NDArray

# =============================================================================
# Signal Generation Strategies
# =============================================================================


@st.composite
def digital_signals(
    draw: st.DrawFn,
    min_length: int = 100,
    max_length: int = 10000,
    levels: tuple[float, float] = (0.0, 3.3),
    noise_level: float = 0.0,
) -> NDArray[np.float64]:
    """Generate digital signal arrays with realistic properties.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum signal length
        max_length: Maximum signal length
        levels: (low, high) voltage levels
        noise_level: Standard deviation of gaussian noise to add

    Returns:
        Digital signal as numpy array
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    signal = draw(st.lists(st.sampled_from(levels), min_size=length, max_size=length))
    signal_array = np.array(signal, dtype=np.float64)

    if noise_level > 0:
        noise = draw(
            st.lists(
                st.floats(
                    min_value=-noise_level * 3,
                    max_value=noise_level * 3,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=length,
                max_size=length,
            )
        )
        signal_array += np.array(noise, dtype=np.float64)

    return signal_array


@st.composite
def analog_waveforms(
    draw: st.DrawFn,
    min_length: int = 100,
    max_length: int = 1000,  # Reduced to stay within Hypothesis entropy limits
    min_value: float = -5.0,
    max_value: float = 5.0,
) -> NDArray[np.float64]:
    """Generate analog waveform data.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum waveform length
        max_length: Maximum waveform length (max 5000 due to Hypothesis limits)
        min_value: Minimum voltage
        max_value: Maximum voltage

    Returns:
        Analog waveform as numpy array
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    waveform = draw(
        st.lists(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=length,
            max_size=length,
        )
    )
    return np.array(waveform, dtype=np.float64)


@st.composite
def edge_lists(draw: st.DrawFn, min_edges: int = 2, max_edges: int = 100) -> NDArray[np.float64]:
    """Generate lists of edge timestamps.

    Args:
        draw: Hypothesis draw function
        min_edges: Minimum number of edges
        max_edges: Maximum number of edges

    Returns:
        Sorted array of edge timestamps
    """
    num_edges = draw(st.integers(min_value=min_edges, max_value=max_edges))
    # Generate sorted timestamps
    edges = sorted(
        [
            draw(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False))
            for _ in range(num_edges)
        ]
    )
    return np.array(edges, dtype=np.float64)


# =============================================================================
# Protocol Message Strategies
# =============================================================================


@st.composite
def protocol_messages(
    draw: st.DrawFn,
    header_size: tuple[int, int] = (2, 4),
    payload_size: tuple[int, int] = (1, 64),
) -> bytes:
    """Generate protocol message bytes with headers and payloads.

    Args:
        draw: Hypothesis draw function
        header_size: (min, max) header size in bytes
        payload_size: (min, max) payload size in bytes

    Returns:
        Complete message as bytes
    """
    header_len = draw(st.integers(min_value=header_size[0], max_value=header_size[1]))
    payload_len = draw(st.integers(min_value=payload_size[0], max_value=payload_size[1]))

    header = draw(st.binary(min_size=header_len, max_size=header_len))
    payload = draw(st.binary(min_size=payload_len, max_size=payload_len))

    return header + payload


@st.composite
def message_streams(
    draw: st.DrawFn,
    min_messages: int = 10,
    max_messages: int = 100,
    header_size: int = 2,
) -> tuple[bytes, bytes]:
    """Generate streams of messages with consistent structure.

    Args:
        draw: Hypothesis draw function
        min_messages: Minimum number of messages
        max_messages: Maximum number of messages
        header_size: Fixed header size for all messages

    Returns:
        (concatenated_stream, header) tuple
    """
    num_messages = draw(st.integers(min_value=min_messages, max_value=max_messages))
    header = draw(st.binary(min_size=header_size, max_size=header_size))

    messages = []
    for _ in range(num_messages):
        payload_size = draw(st.integers(min_value=1, max_value=255))
        payload = draw(st.binary(min_size=payload_size, max_size=payload_size))
        messages.append(header + bytes([payload_size]) + payload)

    return b"".join(messages), header


@st.composite
def checksum_data(draw: st.DrawFn, min_size: int = 10, max_size: int = 1000) -> bytes:
    """Generate data suitable for checksum testing.

    Args:
        draw: Hypothesis draw function
        min_size: Minimum data size
        max_size: Maximum data size

    Returns:
        Random bytes for checksum computation
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    data = draw(st.binary(min_size=size, max_size=size))
    return data


# =============================================================================
# State Machine and Inference Strategies
# =============================================================================


@st.composite
def state_machine_samples(
    draw: st.DrawFn,
) -> tuple[int, list[int], list[int]]:
    """Generate valid state machine sample sequences.

    Args:
        draw: Hypothesis draw function

    Returns:
        (num_states, alphabet, sequence) tuple
    """
    num_states = draw(st.integers(min_value=2, max_value=10))
    alphabet_size = draw(st.integers(min_value=2, max_value=5))
    sequence_length = draw(st.integers(min_value=10, max_value=100))

    alphabet = list(range(alphabet_size))
    sequence = [draw(st.sampled_from(alphabet)) for _ in range(sequence_length)]

    return num_states, alphabet, sequence


@st.composite
def alignment_sequences(
    draw: st.DrawFn,
    min_length: int = 10,
    max_length: int = 100,
    alphabet_size: int = 4,
) -> tuple[list[int], list[int]]:
    """Generate two sequences for alignment testing.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        alphabet_size: Size of alphabet (e.g., 4 for DNA)

    Returns:
        (sequence1, sequence2) tuple
    """
    len1 = draw(st.integers(min_value=min_length, max_value=max_length))
    len2 = draw(st.integers(min_value=min_length, max_value=max_length))

    alphabet = list(range(alphabet_size))
    seq1 = [draw(st.sampled_from(alphabet)) for _ in range(len1)]
    seq2 = [draw(st.sampled_from(alphabet)) for _ in range(len2)]

    return seq1, seq2


@st.composite
def clustering_data(
    draw: st.DrawFn,
    min_points: int = 10,
    max_points: int = 100,
    dimensions: int = 2,
) -> NDArray[np.float64]:
    """Generate data points for clustering algorithms.

    Args:
        draw: Hypothesis draw function
        min_points: Minimum number of data points
        max_points: Maximum number of data points
        dimensions: Number of dimensions

    Returns:
        Data points as (n_points, dimensions) array
    """
    num_points = draw(st.integers(min_value=min_points, max_value=max_points))
    points = []

    for _ in range(num_points):
        point = [
            draw(
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            for _ in range(dimensions)
        ]
        points.append(point)

    return np.array(points, dtype=np.float64)


# =============================================================================
# Statistical Analysis Strategies
# =============================================================================


@st.composite
def entropy_data(
    draw: st.DrawFn,
    min_length: int = 100,
    max_length: int = 8192,  # Hypothesis max list size is 8192
    alphabet_size: int = 256,
) -> bytes:
    """Generate data for entropy calculation testing.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum data length
        max_length: Maximum data length (capped at 8192 for Hypothesis)
        alphabet_size: Number of unique symbols

    Returns:
        Data as bytes
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    data = draw(
        st.lists(
            st.integers(min_value=0, max_value=alphabet_size - 1),
            min_size=length,
            max_size=length,
        )
    )
    return bytes(data)


@st.composite
def ngram_sequences(
    draw: st.DrawFn,
    min_length: int = 20,
    max_length: int = 200,
    alphabet_size: int = 256,
) -> bytes:
    """Generate sequences for n-gram analysis.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        alphabet_size: Number of unique symbols

    Returns:
        Sequence as bytes
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    sequence = draw(
        st.lists(
            st.integers(min_value=0, max_value=alphabet_size - 1),
            min_size=length,
            max_size=length,
        )
    )
    return bytes(sequence)


@st.composite
def distribution_samples(
    draw: st.DrawFn,
    min_samples: int = 100,
    max_samples: int = 1000,
) -> NDArray[np.float64]:
    """Generate samples for distribution analysis.

    Args:
        draw: Hypothesis draw function
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples

    Returns:
        Sample array
    """
    num_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    samples = draw(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=num_samples,
            max_size=num_samples,
        )
    )
    return np.array(samples, dtype=np.float64)


# =============================================================================
# Pattern Recognition Strategies
# =============================================================================


@st.composite
def pattern_sequences(
    draw: st.DrawFn,
    min_length: int = 50,
    max_length: int = 500,
    alphabet_size: int = 256,
) -> bytes:
    """Generate sequences for pattern detection.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        alphabet_size: Size of alphabet

    Returns:
        Sequence as bytes
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    sequence = draw(
        st.lists(
            st.integers(min_value=0, max_value=alphabet_size - 1),
            min_size=length,
            max_size=length,
        )
    )
    return bytes(sequence)


@st.composite
def repetitive_sequences(
    draw: st.DrawFn,
    pattern_size: tuple[int, int] = (2, 10),
    repetitions: tuple[int, int] = (2, 20),
) -> bytes:
    """Generate sequences with repeated patterns.

    Args:
        draw: Hypothesis draw function
        pattern_size: (min, max) pattern size
        repetitions: (min, max) number of repetitions

    Returns:
        Sequence with repetitions
    """
    patt_size = draw(st.integers(min_value=pattern_size[0], max_value=pattern_size[1]))
    num_reps = draw(st.integers(min_value=repetitions[0], max_value=repetitions[1]))

    pattern = draw(st.binary(min_size=patt_size, max_size=patt_size))
    return pattern * num_reps


# =============================================================================
# Frequency and Spectral Strategies
# =============================================================================


@st.composite
def frequency_data(
    draw: st.DrawFn,
    min_freq: float = 1.0,
    max_freq: float = 1e9,
    min_bins: int = 64,
    max_bins: int = 4096,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate frequency domain data.

    Args:
        draw: Hypothesis draw function
        min_freq: Minimum frequency (Hz)
        max_freq: Maximum frequency (Hz)
        min_bins: Minimum number of frequency bins
        max_bins: Maximum number of frequency bins

    Returns:
        (frequencies, magnitudes) tuple
    """
    num_bins = draw(st.integers(min_value=min_bins, max_value=max_bins))
    frequencies = np.linspace(min_freq, max_freq, num_bins)
    magnitudes = draw(
        st.lists(
            st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=num_bins,
            max_size=num_bins,
        )
    )
    return frequencies, np.array(magnitudes, dtype=np.float64)


# =============================================================================
# Metadata and Configuration Strategies
# =============================================================================


@st.composite
def waveform_metadata(draw: st.DrawFn) -> dict[str, Any]:
    """Generate realistic waveform metadata.

    Args:
        draw: Hypothesis draw function

    Returns:
        Metadata dictionary
    """
    return {
        "sample_rate": draw(st.sampled_from([1e6, 10e6, 100e6, 1e9])),
        "channels": draw(st.integers(min_value=1, max_value=4)),
        "record_length": draw(st.integers(min_value=1000, max_value=100000)),
        "vertical_scale": draw(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
    }


@st.composite
def timing_parameters(draw: st.DrawFn) -> dict[str, float]:
    """Generate timing analysis parameters.

    Args:
        draw: Hypothesis draw function

    Returns:
        Timing parameters dictionary
    """
    return {
        "sample_rate": draw(
            st.floats(
                min_value=1e6,
                max_value=1e10,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        "threshold": draw(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
        ),
        "hysteresis": draw(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
    }


# =============================================================================
# Validation Data Strategies
# =============================================================================


@st.composite
def packet_data(
    draw: st.DrawFn,
    min_size: int = 10,
    max_size: int = 1500,
) -> bytes:
    """Generate packet data for validation testing.

    Args:
        draw: Hypothesis draw function
        min_size: Minimum packet size
        max_size: Maximum packet size (MTU)

    Returns:
        Packet as bytes
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(st.binary(min_size=size, max_size=size))


@st.composite
def framing_data(
    draw: st.DrawFn,
    frame_delimiter: bytes = b"\xaa\xaa",
) -> tuple[bytes, list[int]]:
    """Generate framed data with delimiters.

    Args:
        draw: Hypothesis draw function
        frame_delimiter: Frame boundary delimiter

    Returns:
        (framed_data, frame_boundaries) tuple
    """
    num_frames = draw(st.integers(min_value=1, max_value=20))
    frames = []
    boundaries = [0]

    for _ in range(num_frames):
        frame_size = draw(st.integers(min_value=5, max_value=100))
        frame_data = draw(st.binary(min_size=frame_size, max_size=frame_size))
        frames.append(frame_delimiter + frame_data)
        boundaries.append(boundaries[-1] + len(frame_delimiter) + len(frame_data))

    return b"".join(frames), boundaries[:-1]


# =============================================================================
# Digital Signal Quality Strategies
# =============================================================================


@st.composite
def noisy_digital_signal(
    draw: st.DrawFn,
    min_length: int = 100,
    max_length: int = 1000,
) -> NDArray[np.float64]:
    """Generate noisy digital signals for quality testing.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum signal length
        max_length: Maximum signal length

    Returns:
        Noisy digital signal
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    base_signal = draw(st.lists(st.sampled_from([0.0, 3.3]), min_size=length, max_size=length))

    # Add various noise types
    noise_std = draw(
        st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    noise = np.random.normal(0, noise_std, length)

    signal = np.array(base_signal, dtype=np.float64) + noise
    return signal


@st.composite
def clock_signals(
    draw: st.DrawFn,
    min_cycles: int = 10,
    max_cycles: int = 100,
) -> NDArray[np.float64]:
    """Generate clock-like signals for clock recovery testing.

    Args:
        draw: Hypothesis draw function
        min_cycles: Minimum number of clock cycles
        max_cycles: Maximum number of clock cycles

    Returns:
        Clock signal
    """
    num_cycles = draw(st.integers(min_value=min_cycles, max_value=max_cycles))
    samples_per_cycle = draw(st.integers(min_value=4, max_value=20))

    # Generate square wave
    signal = []
    for _ in range(num_cycles):
        signal.extend([0.0] * (samples_per_cycle // 2))
        signal.extend([3.3] * (samples_per_cycle // 2))

    return np.array(signal, dtype=np.float64)


# =============================================================================
# Jitter and Timing Strategies
# =============================================================================


@st.composite
def jitter_samples(
    draw: st.DrawFn,
    min_samples: int = 100,
    max_samples: int = 1000,
) -> NDArray[np.float64]:
    """Generate timing jitter samples.

    Args:
        draw: Hypothesis draw function
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples

    Returns:
        Jitter values in seconds
    """
    num_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    # Jitter typically in picoseconds to nanoseconds
    jitter = draw(
        st.lists(
            st.floats(
                min_value=-1e-9,
                max_value=1e-9,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=num_samples,
            max_size=num_samples,
        )
    )
    return np.array(jitter, dtype=np.float64)


# =============================================================================
# Power Analysis Strategies
# =============================================================================


@st.composite
def power_traces(
    draw: st.DrawFn,
    min_length: int = 100,
    max_length: int = 1000,  # Reduced to stay within Hypothesis limits
) -> NDArray[np.float64]:
    """Generate power consumption traces.

    Args:
        draw: Hypothesis draw function
        min_length: Minimum trace length
        max_length: Maximum trace length (max 1000 for Hypothesis efficiency)

    Returns:
        Power trace in watts
    """
    from hypothesis.extra.numpy import arrays

    length = draw(st.integers(min_value=min_length, max_value=max_length))
    # Use arrays strategy for better performance and larger size support
    power = draw(
        arrays(
            dtype=np.float64,
            shape=length,
            elements=st.floats(
                min_value=0.001,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return power
