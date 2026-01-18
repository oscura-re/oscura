"""Signal classification and measurement intelligence for Oscura.

This module provides intelligent signal type detection, quality assessment,
and measurement suitability checking to help users understand why they might
get NaN results and which measurements are appropriate for their signals.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('signal.wfm')
    >>> classification = osc.classify_signal(trace)
    >>> print(f"Signal type: {classification['type']}")
    >>> print(f"Characteristics: {classification['characteristics']}")
    >>> quality = osc.assess_signal_quality(trace)
    >>> print(f"SNR: {quality['snr']:.1f} dB")
    >>> suggestions = osc.suggest_measurements(trace)
    >>> print(f"Recommended measurements: {suggestions}")

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
    IEEE 1057-2017: Standard for Digitizing Waveform Recorders
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace
    from oscura.reporting.config import AnalysisDomain


def classify_signal(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    sample_rate: float = 1.0,
    *,
    digital_threshold_ratio: float = 0.8,
    dc_threshold_percent: float = 90.0,
    periodicity_threshold: float = 0.7,
) -> dict[str, Any]:
    """Classify signal type and characteristics.

    Automatically detects whether a signal is digital, analog, or mixed,
    identifies key characteristics like periodicity and noise, and estimates
    fundamental properties.

    Args:
        trace: Input waveform trace or numpy array to classify.
        sample_rate: Sample rate in Hz (only used if trace is ndarray).
        digital_threshold_ratio: Ratio of samples at two levels to consider digital (0-1).
        dc_threshold_percent: Percentage of DC component to classify as DC signal.
        periodicity_threshold: Correlation threshold for periodic detection (0-1).

    Returns:
        Dictionary containing:
        - signal_type: Signal type ("digital", "analog", "mixed", "dc")
        - is_digital: Boolean indicating if signal is digital
        - is_periodic: Boolean indicating if signal is periodic
        - characteristics: List of characteristics like "periodic", "noisy", "pulsed"
        - dc_component: True if significant DC offset present
        - frequency_estimate: Estimated fundamental frequency in Hz (or None)
        - dominant_frequency: Same as frequency_estimate (for compatibility)
        - snr_db: Estimated SNR in dB (or None)
        - confidence: Classification confidence (0.0-1.0)
        - noise_level: Estimated noise level in signal units
        - levels: For digital signals, dict with "low" and "high" levels

    Example:
        >>> trace = osc.load('square_wave.wfm')
        >>> info = osc.classify_signal(trace)
        >>> print(f"Type: {info['signal_type']}")
        Type: digital
        >>> print(f"Characteristics: {info['characteristics']}")
        Characteristics: ['periodic', 'clean']
        >>> print(f"Frequency: {info['frequency_estimate']:.3e} Hz")
        Frequency: 1.000e+06 Hz

    References:
        IEEE 181-2011: Digital waveform characterization
    """
    # Handle both WaveformTrace and ndarray inputs
    if isinstance(trace, np.ndarray):
        data = trace
        trace_sample_rate = sample_rate
    else:
        data = trace.data
        trace_sample_rate = trace.metadata.sample_rate

    n = len(data)

    if n < 10:
        return {
            "type": "unknown",
            "signal_type": "unknown",
            "is_digital": False,
            "is_periodic": False,
            "characteristics": ["insufficient_data"],
            "dc_component": False,
            "frequency_estimate": None,
            "dominant_frequency": None,
            "snr_db": None,
            "confidence": 0.0,
            "noise_level": 0.0,
            "levels": None,
        }

    # Calculate basic statistics
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    amplitude = max_val - min_val

    # Initialize result
    characteristics = []
    signal_type = "analog"
    confidence = 0.5

    # 1. Check for DC signal (very low variation)
    # Use coefficient of variation (CV) for DC detection
    cv = std_val / (abs(mean_val) + amplitude / 2 + 1e-12)
    if amplitude < 1e-9 or cv < 0.005:  # Less than 0.5% variation
        signal_type = "dc"
        characteristics.append("constant")
        confidence = 0.95
        return {
            "type": signal_type,
            "signal_type": signal_type,
            "is_digital": False,
            "is_periodic": False,
            "characteristics": characteristics,
            "dc_component": True,
            "frequency_estimate": None,
            "dominant_frequency": None,
            "snr_db": None,
            "confidence": confidence,
            "noise_level": std_val,
            "levels": None,
        }

    # 2. Check for digital signal (bimodal distribution)
    is_digital, digital_levels, digital_confidence = _detect_digital_signal(
        data, digital_threshold_ratio
    )

    if is_digital:
        signal_type = "digital"
        confidence = digital_confidence
        characteristics.append("digital_levels")

    # 3. Assess noise level
    noise_level = _estimate_noise_level(data)
    noise_ratio = noise_level / (amplitude + 1e-12)

    if noise_ratio < 0.05:
        characteristics.append("clean")
    elif noise_ratio < 0.15:
        characteristics.append("low_noise")
    elif noise_ratio < 0.30:
        characteristics.append("moderate_noise")
    else:
        characteristics.append("noisy")

    # 4. Check for periodicity
    is_periodic, period_estimate, periodicity_score = _detect_periodicity(
        data, trace_sample_rate, periodicity_threshold
    )

    # For digital signals, also try edge-based periodicity detection
    # This works better for signals with few periods
    if not is_periodic and is_digital:
        edge_periodic, edge_period, edge_confidence = _detect_edge_periodicity(
            data, trace_sample_rate, digital_levels
        )
        if edge_periodic:
            is_periodic = edge_periodic
            period_estimate = edge_period
            periodicity_score = edge_confidence

    # Also try FFT-based frequency detection
    # FFT is more reliable for undersampled signals where autocorrelation may detect harmonics
    if n >= 64:
        fft_periodic, fft_period, fft_confidence = _detect_periodicity_fft(data, trace_sample_rate)
        if fft_periodic:
            # If autocorrelation also found periodicity, compare results
            if is_periodic and period_estimate is not None:
                # If frequencies differ significantly (>20%), prefer the higher frequency
                # (lower frequencies are often harmonics or aliasing artifacts)
                auto_freq = 1.0 / period_estimate if period_estimate > 0 else 0
                fft_freq = 1.0 / fft_period if fft_period is not None and fft_period > 0 else 0
                freq_ratio = max(auto_freq, fft_freq) / (min(auto_freq, fft_freq) + 1e-12)

                if freq_ratio > 1.2:  # More than 20% difference
                    # Prefer higher frequency (more likely to be correct)
                    if fft_freq > auto_freq:
                        period_estimate = fft_period
                        periodicity_score = fft_confidence
            else:
                # Only FFT detected periodicity
                is_periodic = fft_periodic
                period_estimate = fft_period
                periodicity_score = fft_confidence

    if is_periodic:
        characteristics.append("periodic")
        frequency_estimate = (
            1.0 / period_estimate if period_estimate is not None and period_estimate > 0 else None
        )
        confidence = max(confidence, periodicity_score)
    else:
        characteristics.append("aperiodic")
        frequency_estimate = None

    # 5. Check for DC component
    dc_component = abs(mean_val) > (amplitude * dc_threshold_percent / 100.0)

    # 6. Detect pulsed/transient characteristics
    edge_count = _count_edges(data, digital_levels if is_digital else None)
    samples_per_edge = n / max(edge_count, 1)

    if edge_count > 2 and samples_per_edge > 100:
        characteristics.append("pulsed")
    elif edge_count < 3 and amplitude > std_val * 2:
        characteristics.append("transient")

    # 7. Check for mixed signal (both digital transitions and analog variation)
    if is_digital and digital_levels is not None:
        # Check if there's significant variation within digital levels
        low_region = data[data < (digital_levels["low"] + digital_levels["high"]) / 2]
        high_region = data[data >= (digital_levels["low"] + digital_levels["high"]) / 2]

        if len(low_region) > 0 and len(high_region) > 0:
            low_std = np.std(low_region)
            high_std = np.std(high_region)
            level_separation = digital_levels["high"] - digital_levels["low"]

            if low_std > level_separation * 0.1 or high_std > level_separation * 0.1:
                signal_type = "mixed"
                characteristics.append("analog_variation")

    # Calculate SNR estimate
    snr_db = None
    if amplitude > noise_level * 10:
        signal_power = amplitude**2 / 8  # Approximate for most waveforms
        noise_power = noise_level**2
        if noise_power > 1e-20:
            snr_db = 10 * np.log10(signal_power / noise_power)

    return {
        "type": signal_type,
        "signal_type": signal_type,
        "is_digital": is_digital,
        "is_periodic": is_periodic,
        "characteristics": characteristics,
        "dc_component": dc_component,
        "frequency_estimate": frequency_estimate,
        "dominant_frequency": frequency_estimate,
        "snr_db": float(snr_db) if snr_db is not None else None,
        "confidence": float(confidence),
        "noise_level": float(noise_level),
        "levels": digital_levels if is_digital else None,
    }


def assess_signal_quality(
    trace: WaveformTrace,
) -> dict[str, Any]:
    """Assess signal quality metrics.

    Analyzes signal quality including SNR, noise level, clipping, saturation,
    and other quality indicators that affect measurement accuracy.

    Args:
        trace: Input waveform trace to assess.

    Returns:
        Dictionary containing:
        - snr: Signal-to-noise ratio in dB (or None if not applicable)
        - noise_level: RMS noise level in signal units
        - clipping: True if signal shows clipping
        - saturation: True if signal appears saturated
        - warnings: List of quality warning strings
        - dynamic_range: Signal dynamic range in dB
        - crest_factor: Peak-to-RMS ratio

    Example:
        >>> trace = osc.load('noisy_sine.wfm')
        >>> quality = osc.assess_signal_quality(trace)
        >>> print(f"SNR: {quality['snr']:.1f} dB")
        SNR: 42.3 dB
        >>> if quality['warnings']:
        ...     print(f"Warnings: {quality['warnings']}")

    References:
        IEEE 1057-2017: ADC quality metrics
    """
    data = trace.data
    n = len(data)
    warnings = []

    if n < 10:
        warnings.append("Insufficient data for quality assessment")
        return {
            "snr": None,
            "noise_level": 0.0,
            "clipping": False,
            "saturation": False,
            "warnings": warnings,
            "dynamic_range": None,
            "crest_factor": None,
        }

    # Calculate statistics
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    mean_val = float(np.mean(data))
    rms_val = float(np.sqrt(np.mean(data**2)))
    amplitude = max_val - min_val

    # 1. Detect clipping (samples stuck at extremes)
    # Real clipping shows as CONSECUTIVE samples at extremes, not just many samples near extremes
    clipping = False
    if amplitude > 1e-9:
        tolerance = amplitude * 0.01  # 1% tolerance

        # Find consecutive runs at extremes
        at_min = data <= (min_val + tolerance)
        at_max = data >= (max_val - tolerance)

        # Check for long consecutive runs (clipping) vs brief peaks (natural waveform)
        # For analog signals like sine waves, peaks naturally have ~5-10% of samples near extremes
        # Real clipping typically shows >15-20% consecutive samples
        # For digital signals, even short runs at extremes can indicate clipping
        min_run_length = max(int(n * 0.15), 100)  # 15% of data or 100 samples minimum

        # Find maximum consecutive run lengths
        max_min_run = 0
        max_max_run = 0

        current_min_run = 0
        current_max_run = 0

        for i in range(n):
            if at_min[i]:
                current_min_run += 1
                max_min_run = max(max_min_run, current_min_run)
            else:
                current_min_run = 0

            if at_max[i]:
                current_max_run += 1
                max_max_run = max(max_max_run, current_max_run)
            else:
                current_max_run = 0

        # Clipping detected if we have long consecutive runs at extremes
        if max_min_run >= min_run_length:
            clipping = True
            warnings.append(
                f"Signal clipping detected at minimum ({max_min_run} consecutive samples)"
            )
        if max_max_run >= min_run_length:
            clipping = True
            warnings.append(
                f"Signal clipping detected at maximum ({max_max_run} consecutive samples)"
            )

    # 2. Detect saturation (signal stuck at one level)
    # For digital signals, 2 unique values is normal, not saturation
    saturation = False
    unique_values = len(np.unique(data))
    classification = classify_signal(trace)

    # Different thresholds for digital vs analog signals
    if classification["type"] == "digital":
        # Digital signals should have 2+ levels; saturation is when stuck at 1 level
        if unique_values < 2:
            saturation = True
            warnings.append(f"Signal saturation detected (only {unique_values} unique value)")
    else:
        # Analog signals should have many unique values
        if unique_values < max(10, n // 1000):
            saturation = True
            warnings.append(f"Signal saturation detected (only {unique_values} unique values)")

    # 3. Estimate noise level
    noise_level = _estimate_noise_level(data)

    # 4. Calculate SNR
    snr = None
    if amplitude > noise_level * 10:  # Only calculate if signal > noise
        # Remove DC and calculate signal power
        data_ac = data - mean_val
        signal_power = np.mean(data_ac**2)
        noise_power = noise_level**2

        if noise_power > 1e-20:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float("inf")

    # 5. Calculate dynamic range
    dynamic_range = None
    if min_val != 0 and max_val != 0 and max_val > 1e-20:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = max_val / (abs(min_val) + 1e-20)
            if ratio > 0 and np.isfinite(ratio):
                dynamic_range = 20 * np.log10(ratio)

    # 6. Calculate crest factor (peak-to-RMS)
    crest_factor = None
    if rms_val > 1e-12:
        crest_factor = max(abs(max_val), abs(min_val)) / rms_val

    # 7. Check for quantization issues
    if n > 100:
        # Estimate quantization step
        sorted_data = np.sort(data)
        diffs = np.diff(sorted_data)
        diffs = diffs[diffs > 1e-15]  # Remove near-zero differences

        if len(diffs) > 10:
            min_step = np.min(diffs)
            if amplitude / min_step < 256:
                warnings.append(
                    f"Low resolution detected ({int(amplitude / min_step)} levels), "
                    "may affect measurement accuracy"
                )

    # 8. Check sample rate adequacy
    classification = classify_signal(trace)
    if classification["frequency_estimate"] is not None:
        # Check if sample rate is at least 10x the detected frequency
        nyquist_rate = 2 * classification["frequency_estimate"]
        if trace.metadata.sample_rate < nyquist_rate * 5:
            warnings.append(
                f"Sample rate ({trace.metadata.sample_rate:.3e} Hz) may be "
                f"insufficient for signal frequency ({classification['frequency_estimate']:.3e} Hz). "
                "Recommend at least 10x oversampling"
            )

        # Additional check: if samples per period is very low, we might be undersampling
        # This catches cases where frequency detection may be wrong due to aliasing
        samples_per_period = trace.metadata.sample_rate / classification["frequency_estimate"]
        if samples_per_period < 10 and "sample rate" not in "".join(warnings).lower():
            warnings.append(
                f"Very low oversampling detected ({samples_per_period:.1f} samples per period). "
                f"Signal may be undersampled or frequency detection may be inaccurate. "
                "Recommend at least 10 samples per period"
            )

    return {
        "snr": float(snr) if snr is not None else None,
        "noise_level": float(noise_level),
        "clipping": bool(clipping),
        "saturation": bool(saturation),
        "warnings": warnings,
        "dynamic_range": float(dynamic_range) if dynamic_range is not None else None,
        "crest_factor": float(crest_factor) if crest_factor is not None else None,
    }


def check_measurement_suitability(
    trace: WaveformTrace,
    measurement_name: str,
) -> dict[str, Any]:
    """Check if a measurement is suitable for this signal.

    Analyzes signal characteristics to determine if a specific measurement
    will produce valid results, and provides warnings and suggestions.

    Args:
        trace: Input waveform trace.
        measurement_name: Name of measurement to check (e.g., "frequency", "rise_time").

    Returns:
        Dictionary containing:
        - suitable: True if measurement is appropriate for this signal
        - confidence: Confidence in suitability assessment (0.0-1.0)
        - warnings: List of warning strings
        - suggestions: List of suggestion strings
        - expected_result: "valid", "nan", or "unreliable"

    Example:
        >>> trace = osc.load('dc_signal.wfm')
        >>> check = osc.check_measurement_suitability(trace, "frequency")
        >>> if not check['suitable']:
        ...     print(f"Warning: {check['warnings']}")
        Warning: ['Frequency measurement not suitable for DC signal']

    References:
        IEEE 181-2011: Measurement applicability
    """
    classification = classify_signal(trace)
    quality = assess_signal_quality(trace)

    warnings = []
    suggestions = []
    suitable = True
    confidence = 0.8
    expected_result = "valid"

    signal_type = classification["type"]
    characteristics = classification["characteristics"]

    # Define measurement requirements
    frequency_measurements = ["frequency", "period"]
    edge_measurements = ["rise_time", "fall_time"]
    amplitude_measurements = ["amplitude", "overshoot", "undershoot", "preshoot"]
    duty_measurements = ["duty_cycle", "pulse_width"]
    _statistical_measurements = ["mean", "rms"]
    spectral_measurements = ["thd", "snr", "sinad", "enob", "sfdr", "fft", "psd"]

    # Check DC signals
    if signal_type == "dc":
        if measurement_name in frequency_measurements:
            suitable = False
            warnings.append(f"{measurement_name} measurement not suitable for DC signal")
            suggestions.append("Use 'mean' or 'rms' measurements for DC signals")
            expected_result = "nan"
        elif measurement_name in edge_measurements:
            suitable = False
            warnings.append(f"{measurement_name} requires signal transitions")
            suggestions.append("Signal appears to be DC with no edges")
            expected_result = "nan"
        elif measurement_name in duty_measurements:
            suitable = False
            warnings.append(f"{measurement_name} requires periodic signal")
            expected_result = "nan"

    # Check aperiodic signals
    if "aperiodic" in characteristics:
        if measurement_name in frequency_measurements + duty_measurements:
            suitable = False
            confidence = 0.6
            warnings.append(f"{measurement_name} requires periodic signal")
            suggestions.append("Signal does not appear periodic")
            expected_result = "nan"
        elif measurement_name in spectral_measurements:
            warnings.append("Spectral measurements on aperiodic signals may not show clear peaks")
            suggestions.append("Consider time-domain or statistical analysis")
            expected_result = "unreliable"

    # Check digital vs analog
    if signal_type == "digital":
        if measurement_name in amplitude_measurements and measurement_name != "amplitude":
            warnings.append(
                f"{measurement_name} designed for analog signals with overshoot/ringing"
            )
            suggestions.append("Digital signals may show zero overshoot/undershoot")
            expected_result = "unreliable"
            confidence = 0.5

    # Check for sufficient transitions
    if measurement_name in edge_measurements + duty_measurements:
        data = trace.data
        edge_count = _count_edges(data, classification.get("levels"))
        if edge_count < 2:
            suitable = False
            warnings.append(f"{measurement_name} requires at least 2 signal edges")
            suggestions.append(f"Signal has only {edge_count} detected edge(s)")
            expected_result = "nan"

    # Check signal quality impacts
    if quality["clipping"]:
        if measurement_name in edge_measurements + amplitude_measurements:
            warnings.append("Signal clipping detected, may affect measurement accuracy")
            # Don't override "nan" - if measurement is fundamentally unsuitable, keep it as "nan"
            if expected_result != "nan":
                expected_result = "unreliable"
            confidence = min(confidence, 0.6)

    if quality["saturation"]:
        warnings.append("Signal saturation detected, measurements may be unreliable")
        # Don't override "nan" - if measurement is fundamentally unsuitable, keep it as "nan"
        if expected_result != "nan":
            expected_result = "unreliable"
        confidence = min(confidence, 0.5)

    if quality["snr"] is not None and quality["snr"] < 20:
        if measurement_name in edge_measurements:
            warnings.append(
                f"Low SNR ({quality['snr']:.1f} dB) may affect edge timing measurements"
            )
            suggestions.append("Consider filtering signal to improve SNR")
            confidence = min(confidence, 0.7)

    # Check sample rate for timing measurements
    if measurement_name in edge_measurements + frequency_measurements:
        if classification["frequency_estimate"] is not None:
            nyquist_rate = 2 * classification["frequency_estimate"]
            if trace.metadata.sample_rate < nyquist_rate * 5:
                warnings.append("Sample rate may be too low for accurate timing measurements")
                suggestions.append(
                    f"Recommend sample rate > {nyquist_rate * 10:.3e} Hz (10x signal frequency)"
                )
                expected_result = "unreliable"
                confidence = min(confidence, 0.6)

    # Check data length
    n = len(trace.data)
    if measurement_name in spectral_measurements:
        if n < 256:
            warnings.append(f"Signal length ({n} samples) may be too short for spectral analysis")
            suggestions.append("Recommend at least 1024 samples for FFT-based measurements")
            expected_result = "unreliable"
            confidence = min(confidence, 0.5)

    if measurement_name in frequency_measurements:
        if classification["frequency_estimate"] is not None:
            min_samples = trace.metadata.sample_rate / classification["frequency_estimate"]
            # Require at least 0.5 periods for basic detection
            # Having 1+ complete periods is ideal, but FFT can work with less
            if n < min_samples * 0.5:
                warnings.append(
                    f"Signal length ({n} samples) captures < 0.5 periods, "
                    "frequency measurement may fail"
                )
                suggestions.append("Capture at least 2 periods for reliable frequency measurement")
                expected_result = "unreliable"
                confidence = min(confidence, 0.5)
            elif n < min_samples * 2:
                # Between 0.5 and 2 periods: usable but not ideal
                suggestions.append("Capture at least 10 periods for best accuracy")
                confidence = min(confidence, 0.75)

    return {
        "suitable": suitable,
        "confidence": float(confidence),
        "warnings": warnings,
        "suggestions": suggestions,
        "expected_result": expected_result,
    }


def suggest_measurements(
    trace: WaveformTrace,
    *,
    max_suggestions: int = 10,
) -> list[dict[str, Any]]:
    """Suggest appropriate measurements for a signal.

    Analyzes signal characteristics and recommends the most suitable
    measurements, ranked by relevance and reliability.

    Args:
        trace: Input waveform trace.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of dictionaries, each containing:
        - name: Measurement name
        - category: Measurement category (e.g., "timing", "amplitude", "spectral")
        - priority: Priority ranking (1=highest)
        - rationale: Why this measurement is recommended
        - confidence: Confidence in recommendation (0.0-1.0)

    Example:
        >>> trace = osc.load('square_wave.wfm')
        >>> suggestions = osc.suggest_measurements(trace)
        >>> for s in suggestions[:3]:
        ...     print(f"{s['name']}: {s['rationale']}")
        frequency: Periodic digital signal detected
        duty_cycle: Suitable for pulse analysis
        rise_time: Digital edges detected

    References:
        Best practices for waveform analysis
    """
    classification = classify_signal(trace)
    quality = assess_signal_quality(trace)

    signal_type = classification["type"]
    characteristics = classification["characteristics"]

    suggestions = []

    # Always suggest basic statistical measurements
    suggestions.append(
        {
            "name": "mean",
            "category": "statistical",
            "priority": 1,
            "rationale": "Basic DC level measurement, always applicable",
            "confidence": 1.0,
        }
    )

    suggestions.append(
        {
            "name": "rms",
            "category": "statistical",
            "priority": 2,
            "rationale": "RMS voltage measurement, useful for all signal types",
            "confidence": 1.0,
        }
    )

    # DC signals
    if signal_type == "dc":
        suggestions.append(
            {
                "name": "amplitude",
                "category": "amplitude",
                "priority": 3,
                "rationale": "Measure noise/variation level in DC signal",
                "confidence": 0.9,
            }
        )
        # Don't suggest frequency, edges, etc.
        return sorted(suggestions, key=lambda x: cast("int", x["priority"]))[:max_suggestions]

    # Amplitude measurements
    suggestions.append(
        {
            "name": "amplitude",
            "category": "amplitude",
            "priority": 3,
            "rationale": f"Peak-to-peak amplitude for {signal_type} signal",
            "confidence": 0.95,
        }
    )

    # Periodic signals
    if "periodic" in characteristics:
        suggestions.append(
            {
                "name": "frequency",
                "category": "timing",
                "priority": 4,
                "rationale": "Periodic signal detected, frequency measurement applicable",
                "confidence": classification["confidence"],
            }
        )

        suggestions.append(
            {
                "name": "period",
                "category": "timing",
                "priority": 5,
                "rationale": "Period measurement for periodic signal",
                "confidence": classification["confidence"],
            }
        )

    # Digital signals with edges
    if signal_type in ("digital", "mixed"):
        edge_count = _count_edges(trace.data, classification.get("levels"))

        if edge_count >= 2:
            suggestions.append(
                {
                    "name": "rise_time",
                    "category": "timing",
                    "priority": 6,
                    "rationale": f"Digital edges detected ({edge_count} edges)",
                    "confidence": 0.9 if quality["snr"] and quality["snr"] > 20 else 0.7,
                }
            )

            suggestions.append(
                {
                    "name": "fall_time",
                    "category": "timing",
                    "priority": 7,
                    "rationale": f"Digital edges detected ({edge_count} edges)",
                    "confidence": 0.9 if quality["snr"] and quality["snr"] > 20 else 0.7,
                }
            )

        if "periodic" in characteristics and edge_count >= 2:
            # Need at least 2 edges (1 complete cycle) for duty cycle
            suggestions.append(
                {
                    "name": "duty_cycle",
                    "category": "timing",
                    "priority": 8,
                    "rationale": "Periodic pulse train detected",
                    "confidence": 0.85 if edge_count >= 4 else 0.75,
                }
            )

            suggestions.append(
                {
                    "name": "pulse_width",
                    "category": "timing",
                    "priority": 9,
                    "rationale": "Pulse measurements suitable for periodic digital signal",
                    "confidence": 0.85 if edge_count >= 4 else 0.75,
                }
            )

    # Analog signals
    if signal_type in ("analog", "mixed"):
        if not quality["clipping"]:
            suggestions.append(
                {
                    "name": "overshoot",
                    "category": "amplitude",
                    "priority": 10,
                    "rationale": "Analog signal, overshoot measurement applicable",
                    "confidence": 0.8,
                }
            )

            suggestions.append(
                {
                    "name": "undershoot",
                    "category": "amplitude",
                    "priority": 11,
                    "rationale": "Analog signal, undershoot measurement applicable",
                    "confidence": 0.8,
                }
            )

    # Spectral measurements for clean, periodic signals
    if "periodic" in characteristics and "clean" in characteristics:
        if len(trace.data) >= 256:
            suggestions.append(
                {
                    "name": "thd",
                    "category": "spectral",
                    "priority": 12,
                    "rationale": "Clean periodic signal suitable for harmonic analysis",
                    "confidence": 0.85,
                }
            )

            suggestions.append(
                {
                    "name": "snr",
                    "category": "spectral",
                    "priority": 13,
                    "rationale": "Spectral SNR measurement for signal quality",
                    "confidence": 0.8,
                }
            )

    # Sort by priority and limit
    suggestions = sorted(suggestions, key=lambda x: cast("int", x["priority"]))
    return suggestions[:max_suggestions]


# =============================================================================
# Helper Functions
# =============================================================================


def _detect_digital_signal(
    data: NDArray[np.floating[Any]],
    threshold_ratio: float,
) -> tuple[bool, dict[str, float] | None, float]:
    """Detect if signal is digital based on bimodal distribution.

    Args:
        data: Signal data array.
        threshold_ratio: Ratio of samples at two levels to consider digital.

    Returns:
        Tuple of (is_digital, levels_dict, confidence).
    """
    # Use histogram to find peaks
    # Use more bins for better resolution on digital signals
    n_bins = min(100, len(np.unique(data)))
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks (local maxima or significant bins)
    peaks = []

    # Special case: if only 2 bins (perfect digital signal), both are peaks
    if len(hist) == 2:
        for i in range(len(hist)):
            if hist[i] > len(data) * 0.01:
                peaks.append((i, hist[i], bin_centers[i]))
    else:
        # Find local maxima in histogram
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                # Lower threshold for peak detection
                if hist[i] > len(data) * 0.01:  # At least 1% of samples
                    peaks.append((i, hist[i], bin_centers[i]))

    # If we have exactly 2 dominant peaks, likely digital
    if len(peaks) >= 2:
        # Sort by count
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

        # Take top 2 peaks
        peak1, peak2 = peaks[0], peaks[1]

        # Check if these two peaks account for most samples
        total_in_peaks = peak1[1] + peak2[1]
        ratio = total_in_peaks / len(data)

        # Also check that peaks are well separated
        peak_separation = abs(peak1[2] - peak2[2])
        data_range = np.ptp(data)

        # Peaks should be separated by at least 30% of data range
        if ratio >= threshold_ratio and peak_separation > data_range * 0.3:
            low_level = min(peak1[2], peak2[2])
            high_level = max(peak1[2], peak2[2])

            confidence = min(0.95, ratio)

            return True, {"low": float(low_level), "high": float(high_level)}, confidence

    return False, None, 0.0


def _estimate_noise_level(data: NDArray[np.floating[Any]]) -> float:
    """Estimate noise level using median absolute deviation.

    Args:
        data: Signal data array.

    Returns:
        Estimated RMS noise level.
    """
    if len(data) < 10:
        return 0.0

    # Use differencing to remove slow variations
    diffs = np.diff(data)

    # MAD (Median Absolute Deviation) is robust to outliers
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))

    # Convert MAD to RMS noise estimate
    # For Gaussian noise: sigma ≈ 1.4826 * MAD
    # Divide by sqrt(2) because diff amplifies noise by sqrt(2)
    noise_estimate = (1.4826 * mad) / np.sqrt(2)

    return float(noise_estimate)


def _detect_periodicity(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    threshold: float,
) -> tuple[bool, float | None, float]:
    """Detect if signal is periodic using autocorrelation.

    Args:
        data: Signal data array.
        sample_rate: Sampling rate in Hz.
        threshold: Correlation threshold for periodic detection.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    n = len(data)

    if n < 20:
        return False, None, 0.0

    # Remove DC for autocorrelation
    data_ac = data - np.mean(data)

    # Check if there's any variation
    if np.std(data_ac) < 1e-12:
        return False, None, 0.0

    # Compute autocorrelation for lags up to n-10 to detect signals with ~1 period
    # This allows finding periodicity even when we have just 1 period of data
    # Keep at least 10 samples of overlap for correlation
    max_lag = min(n - 10, 20000)  # Limit for performance

    autocorr = np.correlate(data_ac, data_ac, mode="full")
    autocorr = autocorr[n - 1 : n - 1 + max_lag]

    # Normalize
    if abs(autocorr[0]) > 1e-12:
        autocorr = autocorr / autocorr[0]
    else:
        return False, None, 0.0

    # Find peaks in autocorrelation (exclude lag=0 and very small lags)
    # Start searching from lag > n/100 to avoid noise
    min_lag = max(3, n // 100)
    peaks = []

    for i in range(min_lag, len(autocorr) - 2):
        # Use stronger peak detection
        if (
            autocorr[i] > autocorr[i - 1]
            and autocorr[i] > autocorr[i + 1]
            and autocorr[i] > autocorr[i - 2]
            and autocorr[i] > autocorr[i + 2]
        ):
            if autocorr[i] > threshold:
                peaks.append((i, autocorr[i]))

    if peaks:
        # Take first significant peak as period
        period_samples = peaks[0][0]
        confidence = float(peaks[0][1])

        period_seconds = period_samples / sample_rate

        return True, period_seconds, confidence

    return False, None, 0.0


def _count_edges(
    data: NDArray[np.floating[Any]],
    levels: dict[str, float] | None,
) -> int:
    """Count number of edges in signal.

    Args:
        data: Signal data array.
        levels: Optional digital levels dict with "low" and "high" keys.

    Returns:
        Number of edges detected.
    """
    if len(data) < 3:
        return 0

    if levels is not None:
        # Use provided levels
        threshold = (levels["low"] + levels["high"]) / 2
    else:
        # Use median as threshold
        threshold = float(np.median(data))

    # Find crossings
    above = data > threshold
    crossings = np.diff(above.astype(int))

    # Count non-zero crossings (both rising and falling)
    edge_count = np.sum(np.abs(crossings))

    return int(edge_count)


def _detect_periodicity_fft(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
) -> tuple[bool, float | None, float]:
    """Detect periodicity using FFT (frequency domain analysis).

    This method works well for signals with few periods where autocorrelation
    may fail. It finds the dominant frequency component in the signal.

    Args:
        data: Signal data array.
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    n = len(data)

    if n < 64:
        return False, None, 0.0

    # Remove DC component
    data_ac = data - np.mean(data)

    # Check if there's any variation
    if np.std(data_ac) < 1e-12:
        return False, None, 0.0

    # Compute FFT
    fft = np.fft.rfft(data_ac)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # Compute power spectrum
    power = np.abs(fft) ** 2

    # Skip DC component (index 0)
    if len(power) < 3:
        return False, None, 0.0

    power = power[1:]
    freqs = freqs[1:]

    # Find peak in power spectrum
    peak_idx = np.argmax(power)
    peak_power = power[peak_idx]
    peak_freq = freqs[peak_idx]

    # Check if peak is significant compared to total power
    total_power = np.sum(power)
    if total_power < 1e-20:
        return False, None, 0.0

    power_ratio = peak_power / total_power

    # For periodic signals, the dominant frequency should have significant power
    # Require at least 10% of total power in the peak
    if power_ratio < 0.1:
        return False, None, 0.0

    # Check that frequency is reasonable (not too low or too high)
    nyquist = sample_rate / 2
    if peak_freq < sample_rate / n or peak_freq > nyquist * 0.9:
        return False, None, 0.0

    # Estimate period
    period_seconds = 1.0 / peak_freq

    # Confidence based on how dominant the peak is
    # High power ratio -> high confidence
    confidence = min(0.95, 0.5 + power_ratio)

    return True, period_seconds, float(confidence)


def _detect_edge_periodicity(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    levels: dict[str, float] | None,
) -> tuple[bool, float | None, float]:
    """Detect periodicity in digital signals by analyzing edge spacing.

    This method works well for signals with few periods where autocorrelation
    may fail. It detects regular patterns in edge timing.

    Args:
        data: Signal data array.
        sample_rate: Sampling rate in Hz.
        levels: Digital levels dict with "low" and "high" keys.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    if len(data) < 10 or levels is None:
        return False, None, 0.0

    threshold = (levels["low"] + levels["high"]) / 2

    # Find edge positions
    above = data > threshold
    crossings = np.diff(above.astype(int))
    edge_positions = np.where(crossings != 0)[0]

    if len(edge_positions) < 2:
        # Need at least 2 edges (1 complete cycle) for detection
        return False, None, 0.0

    # Calculate intervals between edges
    intervals = np.diff(edge_positions)

    if len(intervals) < 1:
        return False, None, 0.0

    # For a periodic signal, intervals should form a repeating pattern
    # For a square wave: intervals alternate between high-time and low-time
    # Check if intervals show regular pattern

    # Calculate coefficient of variation of intervals
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)

    if mean_interval < 1:
        return False, None, 0.0

    cv = std_interval / mean_interval

    # Special case: exactly 1 interval (2 edges, half period of square wave)
    if len(intervals) == 1:
        # This represents half a period for a square wave
        period_samples = 2 * intervals[0]
        period_seconds = period_samples / sample_rate
        # Lower confidence since we only have half a period
        return True, period_seconds, 0.7

    # For highly periodic signals, CV should be low
    if cv > 0.3:
        # High variation - check if it's alternating pattern (square wave)
        if len(intervals) >= 4:
            # Check if odd and even intervals are each consistent
            odd_intervals = intervals[::2]
            even_intervals = intervals[1::2]

            odd_cv = np.std(odd_intervals) / (np.mean(odd_intervals) + 1e-12)
            even_cv = np.std(even_intervals) / (np.mean(even_intervals) + 1e-12)

            if odd_cv < 0.2 and even_cv < 0.2:
                # Alternating pattern detected (square wave)
                # Period is sum of two consecutive intervals
                period_samples = np.mean(odd_intervals) + np.mean(even_intervals)
                period_seconds = period_samples / sample_rate
                confidence = 1.0 - max(odd_cv, even_cv)
                return True, period_seconds, float(confidence)
        elif len(intervals) == 2:
            # Only 2 intervals - assume alternating pattern for square wave
            period_samples = intervals[0] + intervals[1]
            period_seconds = period_samples / sample_rate
            # Moderate confidence with only 2 intervals
            return True, period_seconds, 0.75

        return False, None, 0.0

    # Regular intervals detected
    # For square waves with 50% duty cycle, full period = 2 * interval
    # For other waveforms, check if all intervals are similar (uniform spacing)

    # Estimate period from intervals
    # If all intervals are similar, period might be 2*interval (square wave)
    # Check by seeing if we have roughly equal numbers of edges per inferred period
    period_samples = 2 * mean_interval  # Assume square wave initially
    num_periods = len(data) / period_samples

    # If we have at least 1 period, consider it periodic
    if num_periods >= 0.5:  # Allow detection with half a period
        period_seconds = period_samples / sample_rate
        confidence = 1.0 - min(cv / 0.3, 0.5)  # Scale confidence by CV
        return True, period_seconds, float(confidence)

    return False, None, 0.0


@dataclass
class AnalysisRecommendation:
    """Recommendation for an analysis to run.

    Attributes:
        domain: Analysis domain to run.
        priority: Priority ranking (1=highest).
        confidence: Expected confidence if run (0.0-1.0).
        reasoning: Human-readable explanation.
        estimated_runtime_ms: Estimated runtime in milliseconds.
        prerequisites_met: Whether all prerequisites are satisfied.
    """

    domain: AnalysisDomain
    priority: int  # 1=highest priority
    confidence: float  # Expected confidence if run
    reasoning: str
    estimated_runtime_ms: int = 100
    prerequisites_met: bool = True


def recommend_analyses(
    data: NDArray[np.floating[Any]],
    sample_rate: float = 1.0,
    *,
    time_budget_seconds: float | None = None,
    confidence_target: float = 0.7,
    exclude_domains: list[AnalysisDomain] | None = None,
) -> list[AnalysisRecommendation]:
    """Recommend which analyses to run based on signal characteristics.

    Uses signal classification, quality metrics, and heuristics to
    recommend the most valuable analyses for a given signal.

    Args:
        data: Input signal data.
        sample_rate: Sample rate in Hz.
        time_budget_seconds: Optional time budget (prioritizes faster analyses).
        confidence_target: Minimum expected confidence threshold.
        exclude_domains: Domains to exclude from recommendations.

    Returns:
        List of AnalysisRecommendation sorted by priority.

    Example:
        >>> import numpy as np
        >>> import oscura as osc
        >>> # Generate test signal
        >>> t = np.linspace(0, 1, 10000)
        >>> signal = np.sin(2 * np.pi * 100 * t)
        >>> recommendations = osc.recommend_analyses(signal, sample_rate=10000)
        >>> for rec in recommendations[:3]:
        ...     print(f"{rec.domain.value}: {rec.reasoning}")
        waveform: Basic waveform measurements are always applicable
        statistics: Statistical analysis provides foundational metrics
        spectral: Spectral analysis reveals frequency content - signal appears periodic
    """
    # Avoid circular import
    from oscura.reporting.config import AnalysisDomain

    recommendations = []
    exclude = set(exclude_domains or [])

    # Classify signal
    classification = classify_signal(data, sample_rate)
    _signal_type = classification.get("signal_type", "unknown")  # Reserved for future use
    is_digital = classification.get("is_digital", False)
    is_periodic = classification.get("is_periodic", False)
    _snr_db = classification.get("snr_db", 20)  # Reserved for future use
    dominant_freq = classification.get("dominant_frequency")

    # Always recommend these foundational domains
    if AnalysisDomain.WAVEFORM not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.WAVEFORM,
                priority=1,
                confidence=0.95,
                reasoning="Basic waveform measurements are always applicable",
                estimated_runtime_ms=50,
            )
        )

    if AnalysisDomain.STATISTICS not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.STATISTICS,
                priority=1,
                confidence=0.95,
                reasoning="Statistical analysis provides foundational metrics",
                estimated_runtime_ms=30,
            )
        )

    # Spectral analysis - good for most signals
    if AnalysisDomain.SPECTRAL not in exclude:
        spectral_conf = 0.85 if is_periodic else 0.70
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.SPECTRAL,
                priority=2 if is_periodic else 3,
                confidence=spectral_conf,
                reasoning="Spectral analysis reveals frequency content"
                + (" - signal appears periodic" if is_periodic else ""),
                estimated_runtime_ms=100,
            )
        )

    # Digital-specific analyses
    if is_digital:
        if AnalysisDomain.DIGITAL not in exclude:
            recommendations.append(
                AnalysisRecommendation(
                    domain=AnalysisDomain.DIGITAL,
                    priority=1,
                    confidence=0.90,
                    reasoning="Digital signal detected - edge and timing analysis recommended",
                    estimated_runtime_ms=80,
                )
            )

        if AnalysisDomain.TIMING not in exclude:
            recommendations.append(
                AnalysisRecommendation(
                    domain=AnalysisDomain.TIMING,
                    priority=2,
                    confidence=0.85,
                    reasoning="Timing analysis valuable for digital signals",
                    estimated_runtime_ms=60,
                )
            )

        if AnalysisDomain.PROTOCOLS not in exclude and dominant_freq:
            # Check if frequency matches common baud rates
            common_bauds = [9600, 19200, 38400, 57600, 115200]
            if any(abs(dominant_freq * 2 - b) / b < 0.1 for b in common_bauds):
                recommendations.append(
                    AnalysisRecommendation(
                        domain=AnalysisDomain.PROTOCOLS,
                        priority=3,
                        confidence=0.70,
                        reasoning=f"Frequency {dominant_freq:.0f} Hz suggests serial protocol",
                        estimated_runtime_ms=150,
                    )
                )

    # Periodic signal analyses
    if is_periodic:
        if AnalysisDomain.JITTER not in exclude and is_digital:
            recommendations.append(
                AnalysisRecommendation(
                    domain=AnalysisDomain.JITTER,
                    priority=3,
                    confidence=0.80,
                    reasoning="Periodic digital signal - jitter analysis applicable",
                    estimated_runtime_ms=120,
                )
            )

        if AnalysisDomain.EYE not in exclude and is_digital:
            recommendations.append(
                AnalysisRecommendation(
                    domain=AnalysisDomain.EYE,
                    priority=3,
                    confidence=0.75,
                    reasoning="Eye diagram analysis for signal integrity assessment",
                    estimated_runtime_ms=200,
                )
            )

    # Pattern analysis - good for complex signals
    if AnalysisDomain.PATTERNS not in exclude and len(data) > 1000:
        pattern_conf = 0.70 if is_periodic else 0.50
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.PATTERNS,
                priority=4,
                confidence=pattern_conf,
                reasoning="Pattern analysis can reveal repeating structures",
                estimated_runtime_ms=500,
            )
        )

    # Entropy analysis - useful for random/encrypted data
    if AnalysisDomain.ENTROPY not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.ENTROPY,
                priority=5,
                confidence=0.80,
                reasoning="Entropy analysis characterizes randomness and complexity",
                estimated_runtime_ms=100,
            )
        )

    # Apply confidence threshold filter
    recommendations = [r for r in recommendations if r.confidence >= confidence_target]

    # Apply time budget filter if specified
    if time_budget_seconds is not None:
        budget_ms = time_budget_seconds * 1000
        cumulative = 0
        filtered = []
        for rec in sorted(recommendations, key=lambda x: (x.priority, -x.confidence)):
            if cumulative + rec.estimated_runtime_ms <= budget_ms:
                filtered.append(rec)
                cumulative += rec.estimated_runtime_ms
        recommendations = filtered

    # Sort by priority, then by confidence
    recommendations.sort(key=lambda x: (x.priority, -x.confidence))

    return recommendations


def get_optimal_domain_order(
    recommendations: list[AnalysisRecommendation],
) -> list[AnalysisDomain]:
    """Get optimal order for running analyses.

    Considers dependencies and priorities to determine best order.

    Args:
        recommendations: List of analysis recommendations.

    Returns:
        Ordered list of domains to analyze.

    Example:
        >>> import numpy as np
        >>> import oscura as osc
        >>> # Generate test signal
        >>> t = np.linspace(0, 1, 10000)
        >>> signal = np.sin(2 * np.pi * 100 * t)
        >>> recommendations = osc.recommend_analyses(signal, sample_rate=10000)
        >>> order = osc.get_optimal_domain_order(recommendations)
        >>> print([d.value for d in order])
        ['waveform', 'statistics', 'spectral', 'patterns', 'entropy']
    """
    # Avoid circular import
    from oscura.reporting.config import AnalysisDomain

    # Define dependencies
    dependencies = {
        AnalysisDomain.JITTER: [AnalysisDomain.TIMING],
        AnalysisDomain.EYE: [AnalysisDomain.DIGITAL],
        AnalysisDomain.PROTOCOLS: [AnalysisDomain.DIGITAL],
        AnalysisDomain.INFERENCE: [AnalysisDomain.PATTERNS],
    }

    # Build order respecting dependencies
    ordered = []
    remaining = {r.domain for r in recommendations}

    while remaining:
        # Find domains with satisfied dependencies
        ready = []
        for domain in remaining:
            deps = dependencies.get(domain, [])
            if all(d not in remaining or d in ordered for d in deps):
                ready.append(domain)

        if not ready:
            # No ready domains - just add remaining (circular deps)
            ready = list(remaining)

        # Add highest priority ready domain
        for rec in sorted(recommendations, key=lambda x: (x.priority, -x.confidence)):
            if rec.domain in ready:
                ordered.append(rec.domain)
                remaining.discard(rec.domain)
                break

    return ordered


__all__ = [
    "AnalysisRecommendation",
    "assess_signal_quality",
    "check_measurement_suitability",
    "classify_signal",
    "get_optimal_domain_order",
    "recommend_analyses",
    "suggest_measurements",
]
