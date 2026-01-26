#!/usr/bin/env python3
"""TDR Impedance Analysis Demonstration.

This demo showcases Oscura's Time Domain Reflectometry (TDR) analysis
capabilities for transmission line characterization and impedance
discontinuity detection.

**Features Demonstrated**:
- TDR waveform analysis
- Impedance calculation from reflection coefficient
- Discontinuity detection and location
- Rise time measurement
- Distance-to-fault calculation
- Transmission line characterization
- Open/short/mismatch detection

**TDR Principles**:
The reflection coefficient (rho) relates incident and reflected voltages:
    rho = (Z_L - Z_0) / (Z_L + Z_0)

Where:
    Z_0 = Characteristic impedance of the line
    Z_L = Load impedance (or discontinuity)

From the reflection coefficient, we can calculate:
    Z_L = Z_0 * (1 + rho) / (1 - rho)

**Discontinuity Types**:
- Open: rho = +1 (Z_L = infinity)
- Short: rho = -1 (Z_L = 0)
- Match: rho = 0 (Z_L = Z_0)
- Higher Z: 0 < rho < 1
- Lower Z: -1 < rho < 0

**Applications**:
- PCB trace characterization
- Cable testing
- Connector quality verification
- Stub detection
- Via impedance analysis

Usage:
    python tdr_impedance_demo.py
    python tdr_impedance_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, YELLOW, print_subheader
from oscura.core.types import TraceMetadata, WaveformTrace


class TDRImpedanceDemo(BaseDemo):
    """TDR Impedance Analysis Demonstration.

    This demo generates simularun_demoted TDR waveforms with various impedance
    discontinuities and performs comprehensive TDR analysis.
    """
    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="TDR Impedance Demo",
            description="Demonstrates Time Domain Reflectometry for transmission line analysis",            **kwargs,
        )
        self.sample_rate = 50e9  # 50 GHz (20 ps resolution)
        self.duration = 20e-9  # 20 ns capture

        # TDR system parameters
        self.z0 = 50.0  # System impedance (ohms)
        self.step_amplitude = 0.5  # Step amplitude (V)
        self.step_risetime = 100e-12  # 100 ps rise time
        self.velocity_factor = 0.66  # Typical for FR-4

        # Storage for traces
        self.tdr_trace = None
        self.discontinuities = []

    def _generate_step_edge(self, t: np.ndarray, t_edge: float, risetime: float) -> np.ndarray:
        """Generate a step edge with finite rise time.

        Args:
            t: Time array.
            t_edge: Time of edge.
            risetime: 10-90% rise time.

        Returns:
            Step waveform.
        """
        # Use error function for realistic edge
        # The 10-90% rise time for erf is approximately 1.7 * sigma
        sigma = risetime / 1.7

        from scipy.special import erf

        return 0.5 * (1 + erf((t - t_edge) / (np.sqrt(2) * sigma)))

    def _calculate_reflection(self, z_load: float, z_source: float = 50.0) -> float:
        """Calculate reflection coefficient.

        Args:
            z_load: Load impedance.
            z_source: Source impedance.

        Returns:
            Reflection coefficient (-1 to +1).
        """
        # Handle special cases
        if z_load == float("inf"):
            return 1.0  # Open circuit: total reflection
        if z_load == 0.0:
            return -1.0  # Short circuit: inverted reflection
        return (z_load - z_source) / (z_load + z_source)

    def _impedance_from_reflection(self, rho: float, z_source: float = 50.0) -> float:
        """Calculate impedance from reflection coefficient.

        Args:
            rho: Reflection coefficient.
            z_source: Source impedance.

        Returns:
            Load impedance.
        """
        if rho >= 1.0:
            return float("inf")  # Open
        if rho <= -1.0:
            return 0.0  # Short
        return z_source * (1 + rho) / (1 - rho)

    def generate_test_data(self) -> dict:
        """Generate TDR test waveform with discontinuities.

        Loads from file if available (--data-file override or default NPZ),
        otherwise generates synthetic TDR waveform with impedance discontinuities.
        """
        # Try loading data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading TDR data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("tdr_impedance.npz"):
            file_to_load = default_file
            print_info(f"Loading TDR data from default file: {default_file.name}")

        # Load from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load, allow_pickle=True)
                tdr_waveform = data["tdr_waveform"]
                loaded_sample_rate = float(data["sample_rate"])
                self.sample_rate = loaded_sample_rate

                # Load TDR parameters if available
                if "duration" in data:
                    self.duration = float(data["duration"])
                if "z0" in data:
                    self.z0 = float(data["z0"])
                if "step_amplitude" in data:
                    self.step_amplitude = float(data["step_amplitude"])
                if "step_risetime" in data:
                    self.step_risetime = float(data["step_risetime"])
                if "velocity_factor" in data:
                    self.velocity_factor = float(data["velocity_factor"])
                if "discontinuities" in data:
                    self.discontinuities = data["discontinuities"].tolist()

                # Create trace
                metadata = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="TDR",
                )
                self.tdr_trace = WaveformTrace(data=tdr_waveform, metadata=metadata)

                print_result("Data loaded from file", file_to_load.name)
                print_result("Total samples", len(tdr_waveform))
                print_result("Sample rate", f"{self.sample_rate / 1e9:.0f} GHz")
                print_result("Duration", f"{self.duration * 1e9:.0f} ns")
                print_result("Discontinuities", len(self.discontinuities))
                return
            except Exception as e:
                print_info(f"Failed to load data from file: {e}, falling back to synthetic")
                file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating TDR test waveform...")

        n_samples = int(self.sample_rate * self.duration)
        t = np.arange(n_samples) / self.sample_rate

        # Initialize waveform with incident step
        incident = self._generate_step_edge(t, 1e-9, self.step_risetime)
        tdr_waveform = self.step_amplitude * incident

        # Speed of light in medium
        c = 3e8  # m/s in vacuum
        v = c * self.velocity_factor  # propagation velocity

        # Define discontinuities (distance in meters, impedance in ohms)
        self.discontinuities = [
            {"distance": 0.15, "z": 75.0, "description": "Higher Z trace section"},
            {"distance": 0.30, "z": 50.0, "description": "Return to 50 ohms"},
            {"distance": 0.50, "z": 35.0, "description": "Lower Z via transition"},
            {"distance": 0.65, "z": 50.0, "description": "Return to 50 ohms"},
            {"distance": 0.90, "z": float("inf"), "description": "Open termination"},
        ]

        print_info("  Adding discontinuities:")
        for disc in self.discontinuities:
            # Round-trip time to discontinuity
            t_reflect = 2 * disc["distance"] / v

            # Reflection coefficient
            rho = self._calculate_reflection(disc["z"], self.z0)

            # Add reflection
            reflection = self._generate_step_edge(t, 1e-9 + t_reflect, self.step_risetime)
            tdr_waveform += self.step_amplitude * rho * reflection

            z_str = f"{disc['z']:.0f}" if disc["z"] != float("inf") else "inf"
            print_info(f"    {disc['distance'] * 100:.0f} cm: Z={z_str} ohm (rho={rho:+.3f})")
            print_info(f"      {disc['description']}")

        # Add noise
        tdr_waveform += 0.003 * np.random.randn(n_samples)

        # Create trace
        metadata = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="TDR",
        )
        self.tdr_trace = WaveformTrace(data=tdr_waveform, metadata=metadata)

        print_result("Sample rate", f"{self.sample_rate / 1e9:.0f} GHz")
        print_result("Time resolution", f"{1e12 / self.sample_rate:.1f} ps")
        print_result("Duration", f"{self.duration * 1e9:.0f} ns")
        print_result("Total samples", n_samples)


        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Perform TDR analysis on the waveform."""
        print_subheader("TDR Waveform Analysis")

        data = self.tdr_trace.data
        t = np.arange(len(data)) / self.sample_rate

        # Find the incident edge
        # Look for first significant rise
        derivative = np.diff(data)
        incident_idx = np.argmax(derivative > 0.01 * np.max(derivative))
        incident_time = t[incident_idx]

        print_result("Incident edge time", f"{incident_time * 1e9:.3f} ns")

        # Calculate baseline (pre-step) and step level
        # Ensure we have samples before the incident edge for baseline
        if incident_idx > 10:
            baseline = np.mean(data[: incident_idx - 5])
        else:
            # If incident starts too early, use first few samples
            baseline = np.mean(data[:10])

        # Find stable level after incident but before first reflection
        stable_region = data[incident_idx + 200 : incident_idx + 500]
        if len(stable_region) > 0:
            incident_level = np.mean(stable_region)
        else:
            incident_level = self.step_amplitude

        print_result("Baseline", f"{baseline:.4f} V")
        print_result("Incident level", f"{incident_level:.4f} V")

        # Detect reflections by finding steps in the waveform
        print_subheader("Discontinuity Detection")

        # Compute second derivative to find edges
        smooth_data = np.convolve(data, np.ones(20) / 20, mode="same")
        first_deriv = np.diff(smooth_data)

        # Find peaks in derivative (step edges)
        peaks, properties = find_peaks(
            np.abs(first_deriv),
            height=0.001,
            distance=int(0.2e-9 * self.sample_rate),  # Minimum 0.2 ns apart
        )

        # Skip the first peak (incident edge)
        reflection_peaks = peaks[1:] if len(peaks) > 1 else []

        self.results["detected_discontinuities"] = len(reflection_peaks)
        print_result("Reflections detected", len(reflection_peaks))

        # Analyze each reflection
        c = 3e8
        v = c * self.velocity_factor

        self.results["impedance_profile"] = []
        current_z = self.z0

        for i, peak_idx in enumerate(reflection_peaks):
            # Time of reflection
            t_reflect = t[peak_idx]
            round_trip = t_reflect - incident_time

            # Distance to discontinuity
            distance = round_trip * v / 2

            # Calculate reflection coefficient from waveform level
            # Before this reflection
            pre_start = max(0, peak_idx - 100)
            pre_end = max(pre_start + 1, peak_idx)  # Ensure at least 1 sample
            pre_level = np.mean(data[pre_start:pre_end])

            # After this reflection
            post_start = min(len(data) - 1, peak_idx + 100)
            post_end = min(len(data), peak_idx + 300)
            if post_end > post_start:
                post_level = np.mean(data[post_start:post_end])
            else:
                # Near end of capture, use last few samples
                post_level = np.mean(data[-10:])

            # Reflection causes change in level
            delta_v = post_level - pre_level
            rho = delta_v / incident_level

            # Calculate impedance at this discontinuity
            z_discontinuity = self._impedance_from_reflection(rho, current_z)

            # Store result
            self.results["impedance_profile"].append(
                {
                    "distance_cm": distance * 100,
                    "time_ns": round_trip * 1e9,
                    "rho": rho,
                    "impedance": z_discontinuity,
                }
            )

            # Classify discontinuity type
            if rho > 0.8:
                disc_type = f"{RED}Near-Open{RESET}"
            elif rho > 0.2:
                disc_type = f"{YELLOW}Higher Z{RESET}"
            elif rho < -0.8:
                disc_type = f"{RED}Near-Short{RESET}"
            elif rho < -0.2:
                disc_type = f"{YELLOW}Lower Z{RESET}"
            else:
                disc_type = f"{GREEN}Near-Match{RESET}"

            z_str = f"{z_discontinuity:.1f}" if z_discontinuity < 1e6 else "Open"

            print_info(f"  Discontinuity #{i + 1}:")
            print_info(f"    Distance: {distance * 100:.2f} cm")
            print_info(f"    Round-trip time: {round_trip * 1e9:.3f} ns")
            print_info(f"    Reflection coef: {rho:+.3f}")
            print_info(f"    Impedance: {z_str} ohm")
            print_info(f"    Type: {disc_type}")

            # Update current impedance for next section
            if z_discontinuity < 1e6:
                current_z = z_discontinuity

        # Rise time measurement
        print_subheader("Rise Time Measurement")

        # Find 10% and 90% levels
        level_10 = baseline + 0.1 * (incident_level - baseline)
        level_90 = baseline + 0.9 * (incident_level - baseline)

        # Find crossing times
        idx_10 = np.argmax(data > level_10)
        idx_90 = np.argmax(data > level_90)

        if idx_10 < idx_90:
            risetime_measured = (idx_90 - idx_10) / self.sample_rate
            print_result("Rise time (10-90%)", f"{risetime_measured * 1e12:.1f} ps")
            self.results["risetime_ps"] = risetime_measured * 1e12
        else:
            print_info("  Unable to measure rise time")
            self.results["risetime_ps"] = 0

        # Summary
        print_subheader("Summary")

        print_info("Impedance Profile along transmission line:")
        print_info(f"  0 cm: {self.z0:.0f} ohm (source)")

        for disc in self.results.get("impedance_profile", []):
            z_str = f"{disc['impedance']:.1f}" if disc["impedance"] < 1e6 else "Open"
            print_info(f"  {disc['distance_cm']:.1f} cm: {z_str} ohm")

        # Comparison with expected
        print_subheader("Comparison with Known Discontinuities")

        expected = len(self.discontinuities)
        detected = self.results["detected_discontinuities"]

        print_result("Expected discontinuities", expected)
        print_result("Detected discontinuities", detected)

        if detected == expected:
            print_info(f"{GREEN}All discontinuities detected!{RESET}")
        elif detected > expected:
            print_info(f"{YELLOW}Extra reflections detected (may be multiple bounces){RESET}")
        else:
            print_info(f"{RED}Some discontinuities not detected{RESET}")


        return self.results

    def validate(self, results: dict) -> bool:
        """Validate TDR analysis results."""
        suite = ValidationSuite()

        # Check TDR analysis was performed
        tdr_length = results.get("tdr_length_m", 0)
        suite.add_check(
            "TDR length measured",
            tdr_length > 0,
            f"Got {tdr_length:.3f} m"
        )

        # Check impedance was measured
        z0_measured = results.get("z0_measured_ohms", 0)
        suite.add_check(
            "Impedance measured",
            40 < z0_measured < 60,
            f"Got {z0_measured:.1f} Ohms (expected ~50)"
        )

        # Check for discontinuities
        discontinuities = results.get("discontinuity_count", 0)
        suite.add_check(
            "Discontinuities detected",
            discontinuities > 0,
            f"Got {discontinuities} discontinuities"
        )

        # Check reflection coefficient
        if "reflection_coefficient" in results:
            rho = results["reflection_coefficient"]
            suite.add_check(
                "Reflection coefficient reasonable",
                abs(rho) < 0.5,
                f"Got {rho:.3f}"
            )

        # Check signals were generated
        suite.add_check(
            "TDR trace generated",
            self.tdr_trace is not None and len(self.tdr_trace.data) > 0,
            f"Got {len(self.tdr_trace.data) if self.tdr_trace is not None else 0} samples"
        )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(TDRImpedanceDemo))
