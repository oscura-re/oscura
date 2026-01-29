"""Waveform measurements namespace.

This module provides a namespace for measurement functions to support:
    from oscura.analyzers import measurements
    measurements.rise_time(trace)

Re-exports waveform measurement functions.
"""

from oscura.analyzers.waveform.measurements import (
    amplitude,
    duty_cycle,
    fall_time,
    frequency,
    mean,
    measure,
    overshoot,
    period,
    preshoot,
    pulse_width,
    rise_time,
    rms,
    undershoot,
)

__all__ = [
    "amplitude",
    "duty_cycle",
    "fall_time",
    "frequency",
    "mean",
    "measure",
    "overshoot",
    "period",
    "preshoot",
    "pulse_width",
    "rise_time",
    "rms",
    "undershoot",
]
