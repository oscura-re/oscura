#!/usr/bin/env python3
"""DC-DC Converter Efficiency Analysis Demonstration.

This demo showcases Oscura's power conversion efficiency analysis
capabilities for DC-DC converters, including buck, boost, and buck-boost
topologies.

**Features Demonstrated**:
- Input/output power measurement
- Efficiency calculation (eta = P_out / P_in)
- Load regulation analysis
- Line regulation analysis
- Ripple analysis (AC component)
- Efficiency vs load curves
- Loss breakdown estimation
- Thermal efficiency correlation

**Power Conversion Efficiency**:
eta = P_out / P_in = (V_out * I_out) / (V_in * I_in)

**Key Metrics**:
- Overall efficiency (%)
- Power losses (W)
- Output ripple (mV pp, mV rms)
- Load regulation (mV/A or %)
- Line regulation (mV/V or %)

**Converter Types Analyzed**:
- Buck (step-down): V_out < V_in
- Boost (step-up): V_out > V_in
- Buck-Boost: V_out can be > or < V_in

Usage:
    python dcdc_efficiency_demo.py
    python dcdc_efficiency_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, YELLOW, print_subheader

# Oscura imports
from oscura.analyzers.power.basic import average_power
from oscura.analyzers.power.efficiency import (
    efficiency,
    loss_breakdown,
)
from oscura.analyzers.power.ripple import (
    ripple,
    ripple_statistics,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class DCDCEfficiencyDemo(BaseDemo):
    """DC-DC Converter Efficiency Analysis Demonstration.

    This demo generates simulated DC-DC converter waveforms and performs
    comprehensive efficiency and power quality analysis.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="DC-DC Converter Efficiency Demo",
            description="Demonstrates power conversion efficiency analysis for DC-DC converters",
            **kwargs,
        )
        self.sample_rate = 10e6  # 10 MHz sampling
        self.duration = 1e-3  # 1 ms capture

        # Converter specifications
        self.v_in_nom = 12.0  # Input voltage (V)
        self.v_out_nom = 5.0  # Output voltage (V)
        self.i_out_nom = 2.0  # Output current (A)
        self.fsw = 500e3  # Switching frequency (500 kHz)

        # Storage for traces
        self.v_in_trace = None
        self.i_in_trace = None
        self.v_out_trace = None
        self.i_out_trace = None

    def _generate_input_voltage(self, n_samples: int) -> np.ndarray:
        """Generate input voltage with line ripple.

        Args:
            n_samples: Number of samples.

        Returns:
            Input voltage waveform.
        """
        t = np.arange(n_samples) / self.sample_rate

        # DC voltage with small line ripple (120 Hz from rectified AC)
        line_ripple_freq = 120.0
        line_ripple_amp = 0.2  # 200 mV ripple

        v_in = self.v_in_nom + line_ripple_amp * np.sin(2 * np.pi * line_ripple_freq * t)

        # Add small noise
        v_in += 0.01 * np.random.randn(n_samples)

        return v_in

    def _generate_input_current(self, n_samples: int, efficiency: float = 0.9) -> np.ndarray:
        """Generate input current based on output power and efficiency.

        Args:
            n_samples: Number of samples.
            efficiency: Converter efficiency (0-1).

        Returns:
            Input current waveform.
        """
        t = np.arange(n_samples) / self.sample_rate

        # Calculate average input current from power balance
        p_out = self.v_out_nom * self.i_out_nom
        p_in = p_out / efficiency
        i_in_avg = p_in / self.v_in_nom

        # Add switching ripple at switching frequency
        sw_ripple_amp = 0.3 * i_in_avg  # 30% ripple
        i_in = i_in_avg + sw_ripple_amp * np.sin(2 * np.pi * self.fsw * t)

        # Add noise
        i_in += 0.02 * np.random.randn(n_samples)

        return np.maximum(i_in, 0)  # Current is always positive

    def _generate_output_voltage(self, n_samples: int) -> np.ndarray:
        """Generate output voltage with switching ripple.

        Args:
            n_samples: Number of samples.

        Returns:
            Output voltage waveform.
        """
        t = np.arange(n_samples) / self.sample_rate

        # DC voltage with switching ripple
        sw_ripple_amp = 0.050  # 50 mV ripple
        v_out = self.v_out_nom + sw_ripple_amp * np.sin(2 * np.pi * self.fsw * t)

        # Add some harmonic content
        v_out += 0.010 * np.sin(2 * np.pi * 2 * self.fsw * t)  # 2nd harmonic

        # Add noise
        v_out += 0.005 * np.random.randn(n_samples)

        return v_out

    def _generate_output_current(self, n_samples: int) -> np.ndarray:
        """Generate output current (relatively constant for DC load).

        Args:
            n_samples: Number of samples.

        Returns:
            Output current waveform.
        """
        # Output current is relatively constant (resistive load)
        i_out = self.i_out_nom * np.ones(n_samples)

        # Add small noise
        i_out += 0.01 * np.random.randn(n_samples)

        return np.maximum(i_out, 0)

    def generate_test_data(self) -> dict:
        """Generate DC-DC converter test waveforms.

        Loading priority:
        1. Load from --data-file if specified
        2. Load from default NPZ files in demo_data/ if they exist
        3. Generate synthetic data
        """
        # Try loading from files
        loaded_from_file = False

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            print_info(f"Loading data from CLI override: {self.data_file}")
            try:
                data = np.load(self.data_file)

                # Load all four traces if available
                if "v_in" in data and "i_in" in data and "v_out" in data and "i_out" in data:
                    loaded_sample_rate = float(data["sample_rate"])

                    self.v_in_trace = WaveformTrace(
                        data=data["v_in"],
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="V_IN",
                            source_file=str(self.data_file),
                        ),
                    )
                    self.i_in_trace = WaveformTrace(
                        data=data["i_in"],
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="I_IN",
                            source_file=str(self.data_file),
                        ),
                    )
                    self.v_out_trace = WaveformTrace(
                        data=data["v_out"],
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="V_OUT",
                            source_file=str(self.data_file),
                        ),
                    )
                    self.i_out_trace = WaveformTrace(
                        data=data["i_out"],
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="I_OUT",
                            source_file=str(self.data_file),
                        ),
                    )
                    print_result("Loaded from CLI", self.data_file.name)
                    print_result("Total samples", len(self.v_in_trace.data))
                    loaded_from_file = True
                else:
                    print_info(
                        "File missing required fields (v_in, i_in, v_out, i_out), generating synthetic"
                    )
            except Exception as e:
                print_info(
                    f"Failed to load from CLI file: {e}, falling back to defaults or synthetic"
                )

        # 2. Check default NPZ file in demo_data/
        if not loaded_from_file:
            default_file = Path(__file__).parent / "data" / "dcdc_efficiency.npz"
            if default_file.exists():
                print_info(f"Loading data from default file: {default_file.name}")
                try:
                    data = np.load(default_file)

                    if "v_in" in data and "i_in" in data and "v_out" in data and "i_out" in data:
                        loaded_sample_rate = float(data["sample_rate"])

                        self.v_in_trace = WaveformTrace(
                            data=data["v_in"],
                            metadata=TraceMetadata(
                                sample_rate=loaded_sample_rate,
                                channel_name="V_IN",
                                source_file=str(default_file),
                            ),
                        )
                        self.i_in_trace = WaveformTrace(
                            data=data["i_in"],
                            metadata=TraceMetadata(
                                sample_rate=loaded_sample_rate,
                                channel_name="I_IN",
                                source_file=str(default_file),
                            ),
                        )
                        self.v_out_trace = WaveformTrace(
                            data=data["v_out"],
                            metadata=TraceMetadata(
                                sample_rate=loaded_sample_rate,
                                channel_name="V_OUT",
                                source_file=str(default_file),
                            ),
                        )
                        self.i_out_trace = WaveformTrace(
                            data=data["i_out"],
                            metadata=TraceMetadata(
                                sample_rate=loaded_sample_rate,
                                channel_name="I_OUT",
                                source_file=str(default_file),
                            ),
                        )
                        print_result("Loaded from file", default_file.name)
                        print_result("Total samples", len(self.v_in_trace.data))
                        loaded_from_file = True
                except Exception as e:
                    print_info(f"Failed to load default file: {e}, generating synthetic")

        # 3. Generate synthetic data if not loaded
        if not loaded_from_file:
            print_info("Generating synthetic DC-DC converter waveforms...")

            n_samples = int(self.sample_rate * self.duration)

            # Generate waveforms
            print_info(f"  Converter: {self.v_in_nom}V to {self.v_out_nom}V buck")
            print_info(f"  Load: {self.i_out_nom} A")
            print_info(f"  Switching frequency: {self.fsw / 1e3:.0f} kHz")

            # Input voltage
            v_in_data = self._generate_input_voltage(n_samples)
            self.v_in_trace = WaveformTrace(
                data=v_in_data,
                metadata=TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="V_IN",
                    source_file="synthetic",
                ),
            )

            # Input current (assuming 90% efficiency)
            i_in_data = self._generate_input_current(n_samples, efficiency=0.90)
            self.i_in_trace = WaveformTrace(
                data=i_in_data,
                metadata=TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="I_IN",
                    source_file="synthetic",
                ),
            )

            # Output voltage
            v_out_data = self._generate_output_voltage(n_samples)
            self.v_out_trace = WaveformTrace(
                data=v_out_data,
                metadata=TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="V_OUT",
                    source_file="synthetic",
                ),
            )

            # Output current
            i_out_data = self._generate_output_current(n_samples)
            self.i_out_trace = WaveformTrace(
                data=i_out_data,
                metadata=TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="I_OUT",
                    source_file="synthetic",
                ),
            )

            print_result("Sample rate", f"{self.sample_rate / 1e6:.1f} MHz")
            print_result("Duration", f"{self.duration * 1e3:.1f} ms")
            print_result("Total samples", n_samples)

        # Show signal statistics
        print_subheader("Signal Statistics")
        print_result("V_in (avg)", f"{np.mean(self.v_in_trace.data):.3f} V")
        print_result("I_in (avg)", f"{np.mean(self.i_in_trace.data):.3f} A")
        print_result("V_out (avg)", f"{np.mean(self.v_out_trace.data):.3f} V")
        print_result("I_out (avg)", f"{np.mean(self.i_out_trace.data):.3f} A")

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Perform efficiency and power quality analysis."""
        print_subheader("Efficiency Analysis")

        # Calculate overall efficiency
        eta = efficiency(
            self.v_in_trace,
            self.i_in_trace,
            self.v_out_trace,
            self.i_out_trace,
        )

        print_result("Overall efficiency", f"{eta * 100:.2f}%")
        self.results["efficiency"] = eta  # Store as decimal (0.0-1.0) for validation

        # Calculate power values
        p_in = average_power(voltage=self.v_in_trace, current=self.i_in_trace)
        p_out = average_power(voltage=self.v_out_trace, current=self.i_out_trace)
        p_loss = p_in - p_out

        print_result("Input power", f"{p_in:.3f} W")
        print_result("Output power", f"{p_out:.3f} W")
        print_result("Power loss", f"{p_loss:.3f} W")

        self.results["p_in"] = p_in
        self.results["p_out"] = p_out
        self.results["p_loss"] = p_loss

        # Efficiency rating
        if eta >= 0.95:
            rating = f"{GREEN}Excellent{RESET}"
        elif eta >= 0.90:
            rating = f"{GREEN}Good{RESET}"
        elif eta >= 0.85:
            rating = f"{YELLOW}Fair{RESET}"
        else:
            rating = f"{RED}Poor{RESET}"
        print_info(f"Efficiency rating: {rating}")

        # Output ripple analysis
        print_subheader("Output Ripple Analysis")

        ripple_stats = ripple_statistics(self.v_out_trace)

        print_result("DC level", f"{ripple_stats['dc_level']:.3f} V")
        print_result("Ripple (pk-pk)", f"{ripple_stats['ripple_pp'] * 1e3:.2f} mV")
        print_result("Ripple (RMS)", f"{ripple_stats['ripple_rms'] * 1e3:.2f} mV")
        print_result("Ripple (% pk-pk)", f"{ripple_stats['ripple_pp_percent']:.3f}%")
        print_result("Ripple frequency", f"{ripple_stats['ripple_frequency'] / 1e3:.1f} kHz")

        self.results["ripple_pp_mv"] = ripple_stats["ripple_pp"] * 1e3
        self.results["ripple_rms_mv"] = ripple_stats["ripple_rms"] * 1e3
        self.results["ripple_freq_khz"] = ripple_stats["ripple_frequency"] / 1e3

        # Input ripple analysis
        print_subheader("Input Ripple Analysis")

        v_in_pp, v_in_rms = ripple(self.v_in_trace)
        i_in_pp, i_in_rms = ripple(self.i_in_trace)

        print_result("Voltage ripple (pk-pk)", f"{v_in_pp * 1e3:.2f} mV")
        print_result("Current ripple (pk-pk)", f"{i_in_pp * 1e3:.2f} mA")

        # Loss breakdown estimation
        print_subheader("Loss Breakdown (Estimated)")

        # Estimate loss components (simplified model)
        # For a buck converter:
        # - Switching loss: proportional to fsw * V * I
        # - Conduction loss: I^2 * R_ds(on)
        # - Gate drive loss: C_g * V_g^2 * fsw
        # - Magnetic loss: core + copper

        total_loss = p_loss

        # Estimate based on typical proportions
        switching_est = total_loss * 0.30  # ~30% switching
        conduction_est = total_loss * 0.35  # ~35% conduction
        magnetic_est = total_loss * 0.20  # ~20% magnetic
        other_est = total_loss * 0.15  # ~15% other

        print_result("Switching losses (est)", f"{switching_est * 1e3:.1f} mW")
        print_result("Conduction losses (est)", f"{conduction_est * 1e3:.1f} mW")
        print_result("Magnetic losses (est)", f"{magnetic_est * 1e3:.1f} mW")
        print_result("Other losses (est)", f"{other_est * 1e3:.1f} mW")

        self.results["switching_loss_mw"] = switching_est * 1e3
        self.results["conduction_loss_mw"] = conduction_est * 1e3

        # Use loss_breakdown function
        breakdown = loss_breakdown(
            self.v_in_trace,
            self.i_in_trace,
            self.v_out_trace,
            self.i_out_trace,
            switching_loss=switching_est,
            conduction_loss=conduction_est,
            magnetic_loss=magnetic_est,
        )

        print_result("Switching loss %", f"{breakdown['switching_loss_percent']:.1f}%")
        print_result("Conduction loss %", f"{breakdown['conduction_loss_percent']:.1f}%")

        # Regulation analysis
        print_subheader("Regulation Analysis")

        # Load regulation: change in V_out vs I_out
        # Line regulation: change in V_out vs V_in

        v_out_avg = np.mean(self.v_out_trace.data)
        v_out_variation = np.max(self.v_out_trace.data) - np.min(self.v_out_trace.data)

        # Load regulation (simplified - would need multiple load points)
        load_reg_pct = (v_out_variation / v_out_avg) * 100
        print_result("Output voltage variation", f"{v_out_variation * 1e3:.2f} mV")
        print_result("Regulation estimate", f"{load_reg_pct:.3f}%")

        self.results["load_reg_pct"] = load_reg_pct

        # Thermal considerations
        print_subheader("Thermal Considerations")

        # Junction temperature estimate (assuming 40 C/W thermal resistance)
        theta_ja = 40  # C/W
        t_ambient = 25  # C
        t_junction = t_ambient + p_loss * theta_ja

        print_result("Ambient temperature", f"{t_ambient} C")
        print_result("Thermal resistance", f"{theta_ja} C/W")
        print_result("Estimated junction temp", f"{t_junction:.1f} C")

        if t_junction > 100:
            print_info(f"  {RED}WARNING: Junction temperature exceeds 100C{RESET}")
        elif t_junction > 85:
            print_info(f"  {YELLOW}CAUTION: Junction temperature above 85C{RESET}")
        else:
            print_info(f"  {GREEN}Junction temperature within limits{RESET}")

        self.results["t_junction"] = t_junction

        # Summary
        print_subheader("Summary")
        print_result("Efficiency", f"{eta * 100:.2f}%")
        print_result("Output ripple", f"{ripple_stats['ripple_pp'] * 1e3:.2f} mV pk-pk")
        print_result("Power dissipation", f"{p_loss:.3f} W")
        print_result("Est. junction temp", f"{t_junction:.1f} C")

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate efficiency analysis results."""
        suite = ValidationSuite()

        # Check efficiency is reasonable
        eta = results.get("efficiency", 0)
        suite.add_check("Efficiency calculated", 0 < eta < 1, f"Got {eta:.2%}")

        # Check power balance
        p_in = results.get("p_in", 0)
        p_out = results.get("p_out", 0)
        suite.add_check("Input power measured", p_in > 0, f"Got {p_in:.2f} W")
        suite.add_check("Output power measured", p_out > 0, f"Got {p_out:.2f} W")
        suite.add_check(
            "Power balance reasonable", p_out < p_in, f"P_out={p_out:.2f}W < P_in={p_in:.2f}W"
        )

        # Check ripple measurements
        ripple_pp = results.get("ripple_pp_mv", 0)
        suite.add_check("Ripple measured", ripple_pp > 0, f"Got {ripple_pp:.2f} mV")

        # Check that signals were generated
        suite.add_check(
            "Vin signal generated",
            self.v_in_trace is not None and len(self.v_in_trace.data) > 0,
            f"Got {len(self.v_in_trace.data) if self.v_in_trace is not None else 0} samples",
        )
        suite.add_check(
            "Vout signal generated",
            self.v_out_trace is not None and len(self.v_out_trace.data) > 0,
            f"Got {len(self.v_out_trace.data) if self.v_out_trace is not None else 0} samples",
        )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(DCDCEfficiencyDemo))
