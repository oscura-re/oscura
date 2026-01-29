"""Channel embedding and de-embedding functions.

This module provides time-domain de-embedding to remove fixture
effects and embedding to simulate channel effects.


Example:
    >>> from oscura.analyzers.signal_integrity.embedding import deembed
    >>> clean_trace = deembed(trace, s_params)

References:
    IEEE 370-2020: Standard for Electrical Characterization of PCBs
"""

import numpy as np

from oscura.analyzers.signal_integrity.sparams import (
    SParameterData,
    abcd_to_s,
    s_to_abcd,
)
from oscura.core.exceptions import AnalysisError
from oscura.core.types import WaveformTrace


def deembed(
    trace: WaveformTrace,
    s_params: SParameterData,
    *,
    method: str = "frequency_domain",
    regularization: float = 1e-6,
) -> WaveformTrace:
    """Remove fixture effects from waveform using S-parameters.

    Applies the inverse of the fixture transfer function in the
    frequency domain to recover the signal at the DUT reference plane.

    Args:
        trace: Input waveform trace.
        s_params: S-parameters of fixture to remove.
        method: De-embedding method ("frequency_domain" or "time_domain").
        regularization: Regularization for matrix inversion.

    Returns:
        De-embedded waveform trace.

    Raises:
        ValueError: If method is unknown.
        AnalysisError: If de-embedding fails.

    Example:
        >>> clean = deembed(measured_trace, fixture_sparams)
        >>> # clean now has fixture effects removed

    References:
        IEEE 370-2020 Section 7
    """
    if s_params.n_ports != 2:
        raise AnalysisError(
            f"De-embedding requires 2-port S-parameters, got {s_params.n_ports}-port"
        )

    if method == "frequency_domain":
        return _deembed_frequency_domain(trace, s_params, regularization)
    elif method == "time_domain":
        return _deembed_time_domain(trace, s_params)
    else:
        raise ValueError(f"Unknown method: {method}")


def _deembed_frequency_domain(
    trace: WaveformTrace,
    s_params: SParameterData,
    regularization: float,
) -> WaveformTrace:
    """De-embed using frequency domain approach."""
    data = trace.data
    sample_rate = trace.metadata.sample_rate
    n = len(data)

    # Compute FFT of input signal
    signal_fft = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Interpolate S21 to signal frequencies
    s21 = np.interp(
        frequencies,
        s_params.frequencies,
        s_params.s_matrix[:, 1, 0],
    )

    # Apply inverse transfer function with regularization
    # H_inv = 1 / S21, but regularized
    magnitude = np.abs(s21)
    magnitude_reg = np.maximum(magnitude, regularization)
    h_inv = np.conj(s21) / (magnitude_reg**2 + regularization)

    # Apply inverse filter
    deembedded_fft = signal_fft * h_inv

    # Inverse FFT
    deembedded_data = np.fft.irfft(deembedded_fft, n=n)

    return WaveformTrace(
        data=deembedded_data.astype(np.float64),
        metadata=trace.metadata,
    )


def _deembed_time_domain(
    trace: WaveformTrace,
    s_params: SParameterData,
) -> WaveformTrace:
    """De-embed using time domain impulse response."""
    # Convert S-parameters to impulse response
    s21 = s_params.s_matrix[:, 1, 0]
    frequencies = s_params.frequencies

    # Create symmetric frequency axis for IFFT
    n_freq = len(frequencies)
    2 * (n_freq - 1)

    # Pad with conjugate symmetric extension
    s21_symmetric = np.concatenate([s21, np.conj(s21[-2:0:-1])])

    # IFFT to get impulse response
    impulse_response = np.fft.ifft(s21_symmetric).real

    # Create inverse filter (approximate)
    # Use Wiener deconvolution approach
    data = trace.data
    n = len(data)

    # Pad impulse response
    ir_padded = np.zeros(n)
    ir_len = min(len(impulse_response), n)
    ir_padded[:ir_len] = impulse_response[:ir_len]

    # FFT-based deconvolution
    data_fft = np.fft.fft(data)
    ir_fft = np.fft.fft(ir_padded)

    # Wiener filter
    noise_power = 0.01  # Assumed noise level
    ir_power = np.abs(ir_fft) ** 2
    wiener = np.conj(ir_fft) / (ir_power + noise_power)

    deembedded_fft = data_fft * wiener
    deembedded_data = np.fft.ifft(deembedded_fft).real

    return WaveformTrace(
        data=deembedded_data.astype(np.float64),
        metadata=trace.metadata,
    )


def embed(
    trace: WaveformTrace,
    s_params: SParameterData,
) -> WaveformTrace:
    """Apply channel effects to waveform using S-parameters.

    Convolves the signal with the channel impulse response
    derived from S21 to simulate channel effects.

    Args:
        trace: Input (ideal) waveform trace.
        s_params: S-parameters of channel to apply.

    Returns:
        Waveform with channel effects applied.

    Raises:
        AnalysisError: If embedding fails.

    Example:
        >>> degraded = embed(ideal_trace, channel_sparams)
        >>> # degraded now has channel ISI/loss

    References:
        IEEE 370-2020 Section 7
    """
    if s_params.n_ports != 2:
        raise AnalysisError(f"Embedding requires 2-port S-parameters, got {s_params.n_ports}-port")

    data = trace.data
    sample_rate = trace.metadata.sample_rate
    n = len(data)

    # Compute FFT
    signal_fft = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Interpolate S21 to signal frequencies
    s21 = np.interp(
        frequencies,
        s_params.frequencies,
        s_params.s_matrix[:, 1, 0],
    )

    # Apply transfer function
    embedded_fft = signal_fft * s21

    # Inverse FFT
    embedded_data = np.fft.irfft(embedded_fft, n=n)

    return WaveformTrace(
        data=embedded_data.astype(np.float64),
        metadata=trace.metadata,
    )


def cascade_deembed(
    trace: WaveformTrace,
    fixtures: list[SParameterData],
    *,
    regularization: float = 1e-6,
) -> WaveformTrace:
    """Remove multiple fixture effects from waveform.

    Cascades multiple fixtures and removes their combined effect
    using ABCD matrix multiplication.

    Args:
        trace: Input waveform trace.
        fixtures: List of S-parameter fixtures to remove.
        regularization: Regularization for matrix inversion.

    Returns:
        De-embedded waveform trace.

    Example:
        >>> clean = cascade_deembed(trace, [fixture1, fixture2])

    References:
        IEEE 370-2020 Section 7.3
    """
    if len(fixtures) == 0:
        return trace

    if len(fixtures) == 1:
        return deembed(trace, fixtures[0], regularization=regularization)

    # Find common frequency points
    all_freqs = [f.frequencies for f in fixtures]
    min_freq = max(f.min() for f in all_freqs)
    max_freq = min(f.max() for f in all_freqs)
    n_freq = min(len(f) for f in all_freqs)

    common_freqs = np.linspace(min_freq, max_freq, n_freq)

    # Convert each fixture to ABCD and cascade
    abcd_cascade = np.zeros((n_freq, 2, 2), dtype=np.complex128)
    abcd_cascade[:, 0, 0] = 1
    abcd_cascade[:, 1, 1] = 1  # Identity matrix

    for fixture in fixtures:
        abcd = s_to_abcd(fixture)

        # Interpolate to common frequencies
        abcd_interp = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                abcd_interp[:, i, j] = np.interp(
                    common_freqs,
                    fixture.frequencies,
                    abcd[:, i, j],
                )

        # Matrix multiply for each frequency
        for f_idx in range(n_freq):
            abcd_cascade[f_idx] = abcd_cascade[f_idx] @ abcd_interp[f_idx]

    # Convert cascaded ABCD back to S-parameters
    s_cascade = abcd_to_s(abcd_cascade, z0=fixtures[0].z0)

    # Create combined S-parameter object
    combined = SParameterData(
        frequencies=common_freqs,
        s_matrix=s_cascade,
        n_ports=2,
        z0=fixtures[0].z0,
    )

    return deembed(trace, combined, regularization=regularization)


__all__ = [
    "cascade_deembed",
    "deembed",
    "embed",
]
