"""Power Analysis: IEEE 1459-2010 power quality measurements

Demonstrates:
- oscura.power.average_power() - Active power (P) in watts
- oscura.power.reactive_power() - Reactive power (Q) in VAR
- oscura.power.apparent_power() - Apparent power (S) in VA
- oscura.power.power_factor() - Power factor (PF = P/S)
- oscura.power.phase_angle() - Phase angle between voltage and current
- oscura.power.total_harmonic_distortion_power() - THD analysis
- Ripple analysis - DC-DC converter output quality
- Efficiency calculations - Power conversion efficiency

IEEE Standards: IEEE 1459-2010 (Power quality definitions)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/02_statistics.py

Uses AC and DC power waveforms to demonstrate power quality analysis.
Perfect for understanding power electronics and energy measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately
from oscura.analyzers.power.ac_power import (
    apparent_power,
    phase_angle,
    power_factor,
    reactive_power,
    total_harmonic_distortion_power,
)
from oscura.analyzers.power.basic import average_power
from oscura.core.types import TraceMetadata, WaveformTrace


class PowerAnalysisDemo(BaseDemo):
    """Comprehensive demonstration of power analysis techniques."""

    def __init__(self) -> None:
        """Initialize power analysis demonstration."""
        super().__init__(
            name="power_analysis",
            description="IEEE 1459-2010 power measurements: active, reactive, apparent power, PF",
            capabilities=[
                "oscura.power.average_power",
                "oscura.power.reactive_power",
                "oscura.power.apparent_power",
                "oscura.power.power_factor",
                "oscura.power.phase_angle",
                "oscura.power.total_harmonic_distortion_power",
            ],
            ieee_standards=[
                "IEEE 1459-2010",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate power waveforms for analysis.

        Creates:
        1. Resistive load: Unity power factor (in-phase V and I)
        2. Inductive load: Lagging power factor (current lags voltage)
        3. Capacitive load: Leading power factor (current leads voltage)
        4. Non-linear load: Harmonic distortion
        5. DC-DC converter: Ripple analysis

        Note: We use 10 kHz sampling (not 100 MHz) because power analysis
        at 50 Hz doesn't need high bandwidth. Even for harmonics up to the
        11th (550 Hz), 10 kHz provides adequate margin (20x Nyquist).
        """
        # Use appropriate sample rate for power line analysis
        # 10 kHz is plenty for 50 Hz fundamental with harmonics up to 11th order
        sample_rate = 10e3  # 10 kHz sampling - adequate for power analysis
        duration = 0.04  # 40 ms (2 complete cycles at 50 Hz)
        line_freq = 50.0  # 50 Hz AC line
        v_rms = 230.0  # 230 V RMS (European standard)
        v_peak = v_rms * np.sqrt(2)  # ~325 V peak

        # 1. Resistive load (unity power factor)
        # Current in phase with voltage
        v_resistive = generate_sine_wave(
            frequency=line_freq,
            amplitude=v_peak,
            duration=duration,
            sample_rate=sample_rate,
        )

        i_resistive = generate_sine_wave(
            frequency=line_freq,
            amplitude=10 * np.sqrt(2),  # 10 A RMS
            duration=duration,
            sample_rate=sample_rate,
            phase=0.0,  # In phase
        )

        # 2. Inductive load (lagging power factor)
        # Current lags voltage by 30 degrees
        v_inductive = generate_sine_wave(
            frequency=line_freq,
            amplitude=v_peak,
            duration=duration,
            sample_rate=sample_rate,
        )

        i_inductive = generate_sine_wave(
            frequency=line_freq,
            amplitude=10 * np.sqrt(2),  # 10 A RMS
            duration=duration,
            sample_rate=sample_rate,
            phase=-np.pi / 6,  # -30 degrees (lagging)
        )

        # 3. Capacitive load (leading power factor)
        # Current leads voltage by 30 degrees
        v_capacitive = generate_sine_wave(
            frequency=line_freq,
            amplitude=v_peak,
            duration=duration,
            sample_rate=sample_rate,
        )

        i_capacitive = generate_sine_wave(
            frequency=line_freq,
            amplitude=10 * np.sqrt(2),  # 10 A RMS
            duration=duration,
            sample_rate=sample_rate,
            phase=np.pi / 6,  # +30 degrees (leading)
        )

        # 4. Non-linear load (with harmonics)
        # Voltage is clean sine, current has 3rd and 5th harmonics
        v_nonlinear = generate_sine_wave(
            frequency=line_freq,
            amplitude=v_peak,
            duration=duration,
            sample_rate=sample_rate,
        )

        # Current: fundamental + 3rd harmonic (20%) + 5th harmonic (10%)
        i_fundamental = generate_sine_wave(
            frequency=line_freq,
            amplitude=10 * np.sqrt(2),
            duration=duration,
            sample_rate=sample_rate,
        )
        i_3rd = generate_sine_wave(
            frequency=3 * line_freq,
            amplitude=2 * np.sqrt(2),  # 20% of fundamental
            duration=duration,
            sample_rate=sample_rate,
        )
        i_5th = generate_sine_wave(
            frequency=5 * line_freq,
            amplitude=1 * np.sqrt(2),  # 10% of fundamental
            duration=duration,
            sample_rate=sample_rate,
        )

        i_nonlinear_data = i_fundamental.data + i_3rd.data + i_5th.data
        i_nonlinear = WaveformTrace(
            data=i_nonlinear_data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="i_nonlinear"),
        )

        # 5. DC-DC converter output (with ripple)
        # 12V output with 100 kHz switching ripple
        # Use 1 MHz for DC-DC to capture 100 kHz ripple
        dcdc_sample_rate = 1e6  # 1 MHz for ripple analysis
        dc_voltage = 12.0
        ripple_freq = 100e3  # 100 kHz
        ripple_amplitude = 0.05  # 50 mV peak ripple

        num_samples = int(1e-3 * dcdc_sample_rate)  # 1 ms
        t = np.arange(num_samples) / dcdc_sample_rate

        dc_output = dc_voltage + ripple_amplitude * np.sin(2 * np.pi * ripple_freq * t)

        v_dcdc = WaveformTrace(
            data=dc_output,
            metadata=TraceMetadata(sample_rate=dcdc_sample_rate, channel_name="v_dcdc"),
        )

        # DC-DC current (with ripple)
        dc_current = 5.0  # 5 A average
        i_dcdc_data = dc_current + 0.2 * np.sin(2 * np.pi * ripple_freq * t)

        i_dcdc = WaveformTrace(
            data=i_dcdc_data,
            metadata=TraceMetadata(sample_rate=dcdc_sample_rate, channel_name="i_dcdc"),
        )

        return {
            "v_resistive": v_resistive,
            "i_resistive": i_resistive,
            "v_inductive": v_inductive,
            "i_inductive": i_inductive,
            "v_capacitive": v_capacitive,
            "i_capacitive": i_capacitive,
            "v_nonlinear": v_nonlinear,
            "i_nonlinear": i_nonlinear,
            "v_dcdc": v_dcdc,
            "i_dcdc": i_dcdc,
            "v_rms_expected": v_rms,
            "i_rms_expected": 10.0,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive power analysis demonstration."""
        results = {}

        self.section("Oscura Power Analysis")
        self.info("Demonstrating IEEE 1459-2010 compliant power measurements")
        self.info("Using AC and DC power waveforms for quality analysis")

        # ========== PART 1: RESISTIVE LOAD (UNITY PF) ==========
        self.subsection("Part 1: Resistive Load (Unity Power Factor)")
        self.info("Pure resistive load: V and I in phase, PF = 1.0")

        v_res = data["v_resistive"]
        i_res = data["i_resistive"]

        p_res = average_power(voltage=v_res, current=i_res)
        q_res = reactive_power(v_res, i_res)
        s_res = apparent_power(v_res, i_res)
        pf_res = power_factor(v_res, i_res)
        phi_res = phase_angle(v_res, i_res)

        self.result("Active Power (P)", f"{p_res:.2f}", "W")
        self.result("Reactive Power (Q)", f"{q_res:.2f}", "VAR")
        self.result("Apparent Power (S)", f"{s_res:.2f}", "VA")
        self.result("Power Factor (PF)", f"{pf_res:.4f}")
        self.result("Phase Angle", f"{np.degrees(phi_res):.2f}", "degrees")

        results["p_resistive"] = p_res
        results["q_resistive"] = q_res
        results["s_resistive"] = s_res
        results["pf_resistive"] = pf_res

        # ========== PART 2: INDUCTIVE LOAD (LAGGING PF) ==========
        self.subsection("Part 2: Inductive Load (Lagging Power Factor)")
        self.info("Inductive load: Current lags voltage by 30 degrees, PF ~ 0.866")

        v_ind = data["v_inductive"]
        i_ind = data["i_inductive"]

        p_ind = average_power(voltage=v_ind, current=i_ind)
        q_ind = reactive_power(v_ind, i_ind)
        s_ind = apparent_power(v_ind, i_ind)
        pf_ind = power_factor(v_ind, i_ind)
        phi_ind = phase_angle(v_ind, i_ind)

        self.result("Active Power (P)", f"{p_ind:.2f}", "W")
        self.result("Reactive Power (Q)", f"{q_ind:.2f}", "VAR")
        self.result("Apparent Power (S)", f"{s_ind:.2f}", "VA")
        self.result("Power Factor (PF)", f"{pf_ind:.4f}")
        self.result("Phase Angle", f"{np.degrees(phi_ind):.2f}", "degrees")

        results["p_inductive"] = p_ind
        results["q_inductive"] = q_ind
        results["s_inductive"] = s_ind
        results["pf_inductive"] = pf_ind
        results["phi_inductive"] = phi_ind

        # ========== PART 3: CAPACITIVE LOAD (LEADING PF) ==========
        self.subsection("Part 3: Capacitive Load (Leading Power Factor)")
        self.info("Capacitive load: Current leads voltage by 30 degrees, PF ~ 0.866")

        v_cap = data["v_capacitive"]
        i_cap = data["i_capacitive"]

        p_cap = average_power(voltage=v_cap, current=i_cap)
        q_cap = reactive_power(v_cap, i_cap)
        s_cap = apparent_power(v_cap, i_cap)
        pf_cap = power_factor(v_cap, i_cap)
        phi_cap = phase_angle(v_cap, i_cap)

        self.result("Active Power (P)", f"{p_cap:.2f}", "W")
        self.result("Reactive Power (Q)", f"{q_cap:.2f}", "VAR")
        self.result("Apparent Power (S)", f"{s_cap:.2f}", "VA")
        self.result("Power Factor (PF)", f"{pf_cap:.4f}")
        self.result("Phase Angle", f"{np.degrees(phi_cap):.2f}", "degrees")

        results["p_capacitive"] = p_cap
        results["q_capacitive"] = q_cap
        results["pf_capacitive"] = pf_cap
        results["phi_capacitive"] = phi_cap

        # ========== PART 4: NON-LINEAR LOAD (WITH HARMONICS) ==========
        self.subsection("Part 4: Non-Linear Load (Harmonic Distortion)")
        self.info("Non-linear load: Current has 3rd (20%) and 5th (10%) harmonics")

        v_nl = data["v_nonlinear"]
        i_nl = data["i_nonlinear"]

        p_nl = average_power(voltage=v_nl, current=i_nl)
        s_nl = apparent_power(v_nl, i_nl)
        pf_nl = power_factor(v_nl, i_nl)

        # THD of current
        thd_i = total_harmonic_distortion_power(i_nl, fundamental_freq=50.0)

        self.result("Active Power (P)", f"{p_nl:.2f}", "W")
        self.result("Apparent Power (S)", f"{s_nl:.2f}", "VA")
        self.result("Power Factor (PF)", f"{pf_nl:.4f}")
        self.result("Current THD", f"{thd_i * 100:.2f}", "%")

        results["p_nonlinear"] = p_nl
        results["pf_nonlinear"] = pf_nl
        results["thd_current"] = thd_i

        # ========== PART 5: DC-DC CONVERTER (RIPPLE ANALYSIS) ==========
        self.subsection("Part 5: DC-DC Converter (Ripple Analysis)")
        self.info("12V DC-DC converter with 100 kHz switching ripple")

        v_dc = data["v_dcdc"]
        i_dc = data["i_dcdc"]

        # DC power
        p_dc = average_power(voltage=v_dc, current=i_dc)

        # Ripple analysis
        v_mean = np.mean(v_dc.data)
        v_ripple_pp = np.max(v_dc.data) - np.min(v_dc.data)
        v_ripple_percent = (v_ripple_pp / v_mean) * 100

        i_mean = np.mean(i_dc.data)
        i_ripple_pp = np.max(i_dc.data) - np.min(i_dc.data)

        self.result("Output Power", f"{p_dc:.2f}", "W")
        self.result("Output Voltage (mean)", f"{v_mean:.4f}", "V")
        self.result("Voltage Ripple (p-p)", f"{v_ripple_pp * 1e3:.2f}", "mV")
        self.result("Ripple Percentage", f"{v_ripple_percent:.3f}", "%")
        self.result("Output Current (mean)", f"{i_mean:.4f}", "A")
        self.result("Current Ripple (p-p)", f"{i_ripple_pp:.4f}", "A")

        # Efficiency estimate (assume input = 15V, 4.2A)
        v_in = 15.0
        i_in = 4.2
        p_in = v_in * i_in
        efficiency = (p_dc / p_in) * 100

        self.result("Estimated Efficiency", f"{efficiency:.2f}", "%")

        results["p_dcdc"] = p_dc
        results["v_mean"] = v_mean
        results["v_ripple_pp"] = v_ripple_pp
        results["v_ripple_percent"] = v_ripple_percent
        results["efficiency"] = efficiency

        # ========== POWER INTERPRETATION ==========
        self.subsection("Power Measurement Interpretation")

        self.info("\n[Active Power (P)]")
        self.info("  Real power consumed/delivered (watts)")
        self.info("  P = V_rms x I_rms x cos(phi)")

        self.info("\n[Reactive Power (Q)]")
        self.info("  Power oscillating between source and load (VAR)")
        self.info("  Q = V_rms x I_rms x sin(phi)")
        self.info("  Positive = inductive, Negative = capacitive")

        self.info("\n[Apparent Power (S)]")
        self.info("  Total power (VA): S = V_rms x I_rms")
        self.info("  S^2 = P^2 + Q^2")

        self.info("\n[Power Factor (PF)]")
        self.info("  Efficiency of power transfer: PF = P / S")
        self.info("  1.0 = ideal (all power is real)")
        self.info("  < 1.0 = reactive or distortion losses")

        self.info("\n[Expected values:]")
        self.info("  Resistive: PF = 1.0, phi = 0 degrees")
        self.info("  Inductive: PF = cos(30) = 0.866, phi = 30 degrees")
        self.info("  Capacitive: PF = cos(-30) = 0.866, phi = -30 degrees")

        self.success("All power measurements complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate power measurements."""
        self.info("Validating power measurements...")

        all_valid = True

        # Validate resistive load
        self.subsection("Resistive Load Validation")

        # Power factor should be ~1.0 for resistive load
        if not validate_approximately(
            results["pf_resistive"],
            1.0,
            tolerance=0.05,
            name="Resistive PF",
        ):
            all_valid = False

        # Reactive power should be near zero
        if abs(results["q_resistive"]) < 100:  # Less than 100 VAR
            self.success(f"Reactive power: {results['q_resistive']:.2f} VAR (near 0)")
        else:
            self.error(f"Reactive power too high: {results['q_resistive']:.2f} VAR")
            all_valid = False

        # P should equal S for resistive load
        if not validate_approximately(
            results["p_resistive"],
            results["s_resistive"],
            tolerance=0.05,
            name="P approx S for resistive",
        ):
            all_valid = False

        # Validate inductive load
        self.subsection("Inductive Load Validation")

        # Power factor should be ~0.866 (cos(30 degrees))
        if not validate_approximately(
            results["pf_inductive"],
            np.cos(np.pi / 6),
            tolerance=0.1,
            name="Inductive PF",
        ):
            all_valid = False

        # Phase angle should be ~30 degrees (positive for lagging)
        phi_deg = np.degrees(results["phi_inductive"])
        if 20 < phi_deg < 40:
            self.success(f"Phase angle: {phi_deg:.2f} degrees (near 30 degrees)")
        else:
            self.error(f"Phase angle out of range: {phi_deg:.2f} degrees (expected ~30 degrees)")
            all_valid = False

        # Reactive power should be positive (inductive)
        if results["q_inductive"] > 0:
            self.success(f"Reactive power positive (inductive): {results['q_inductive']:.2f} VAR")
        else:
            self.error(f"Reactive power should be positive: {results['q_inductive']:.2f} VAR")
            all_valid = False

        # Validate capacitive load
        self.subsection("Capacitive Load Validation")

        # Power factor should be ~0.866
        if not validate_approximately(
            results["pf_capacitive"],
            np.cos(np.pi / 6),
            tolerance=0.1,
            name="Capacitive PF",
        ):
            all_valid = False

        # Phase angle should be ~-30 degrees (negative for leading)
        phi_cap_deg = np.degrees(results["phi_capacitive"])
        if -40 < phi_cap_deg < -20:
            self.success(f"Phase angle: {phi_cap_deg:.2f} degrees (near -30 degrees)")
        else:
            self.error(
                f"Phase angle out of range: {phi_cap_deg:.2f} degrees (expected ~-30 degrees)"
            )
            all_valid = False

        # Reactive power should be negative (capacitive)
        if results["q_capacitive"] < 0:
            self.success(f"Reactive power negative (capacitive): {results['q_capacitive']:.2f} VAR")
        else:
            self.error(f"Reactive power should be negative: {results['q_capacitive']:.2f} VAR")
            all_valid = False

        # Validate non-linear load
        self.subsection("Non-Linear Load Validation")

        # THD should be approximately sqrt(0.2^2 + 0.1^2) = 0.2236 = 22.36%
        expected_thd = np.sqrt(0.2**2 + 0.1**2)
        if not validate_approximately(
            results["thd_current"],
            expected_thd,
            tolerance=0.2,
            name="Current THD",
        ):
            all_valid = False

        # Power factor should be less than unity due to harmonics
        if results["pf_nonlinear"] < 1.0:
            self.success(f"PF < 1.0 due to harmonics: {results['pf_nonlinear']:.4f}")
        else:
            self.warning(f"PF should be < 1.0: {results['pf_nonlinear']:.4f}")

        # Validate DC-DC converter
        self.subsection("DC-DC Converter Validation")

        # Output voltage should be ~12V
        if not validate_approximately(
            results["v_mean"],
            12.0,
            tolerance=0.05,
            name="DC output voltage",
        ):
            all_valid = False

        # Ripple should be low (<1%)
        if results["v_ripple_percent"] < 1.0:
            self.success(f"Ripple: {results['v_ripple_percent']:.3f}% < 1%")
        else:
            self.warning(f"Ripple high: {results['v_ripple_percent']:.3f}%")

        # Efficiency should be reasonable (>90%)
        if results["efficiency"] > 90:
            self.success(f"Efficiency: {results['efficiency']:.2f}% > 90%")
        else:
            self.warning(f"Efficiency low: {results['efficiency']:.2f}%")

        if all_valid:
            self.success("All power measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - Active Power: Real power consumed (watts)")
            self.info("  - Reactive Power: Oscillating power (VAR)")
            self.info("  - Apparent Power: Total power (VA)")
            self.info("  - Power Factor: P/S (efficiency of transfer)")
            self.info("  - THD: Harmonic distortion affects PF")
            self.info("\nNext steps:")
            self.info("  - Try 04_advanced_analysis/03_signal_integrity.py")
            self.info("  - Explore switching power supply analysis")
        else:
            self.error("Some power measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: PowerAnalysisDemo = PowerAnalysisDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
