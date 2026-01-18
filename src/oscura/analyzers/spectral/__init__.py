"""Spectral analysis module.

This module re-exports spectral analysis functions from the waveform package
for convenient access.


Example:
    >>> from oscura.analyzers.spectral import fft, psd, thd, snr
    >>> freq, mag = fft(trace)
    >>> thd_db = thd(trace)
"""

from oscura.analyzers.waveform.spectral import (
    bartlett_psd,
    cwt,
    dwt,
    enob,
    fft,
    fft_chunked,
    hilbert_transform,
    idwt,
    mfcc,
    periodogram,
    psd,
    psd_chunked,
    sfdr,
    sinad,
    snr,
    spectrogram,
    spectrogram_chunked,
    thd,
)

__all__ = [
    "bartlett_psd",
    "cwt",
    "dwt",
    "enob",
    "fft",
    "fft_chunked",
    "hilbert_transform",
    "idwt",
    "mfcc",
    "periodogram",
    "psd",
    "psd_chunked",
    "sfdr",
    "sinad",
    "snr",
    "spectrogram",
    "spectrogram_chunked",
    "thd",
]
