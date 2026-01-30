"""Signal Integrity: S-parameter and impedance analysis

Demonstrates:
- S-parameter extraction (S11, S21)
- Insertion loss measurement
- Return loss analysis
- Impedance profiling
- Transmission line characterization

IEEE Standards: IEEE 287-2007 (RF measurements)
Related Demos:
- 04_advanced_analysis/09_tdr.py - Time-domain reflectometry
- 02_basic_analysis/02_spectral_analysis.py

S-parameter and impedance analysis for high-speed signaling.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite


class SignalIntegrityDemo(BaseDemo):
    """Signal integrity and S-parameter analysis."""

    name = "Signal Integrity (S-Parameters)"
    description = "S-parameter extraction and transmission line analysis"
    category = "advanced_analysis"
    capabilities = [
        "S-parameter extraction",
        "Insertion loss (S21)",
        "Return loss (S11)",
        "Impedance profiling",
    ]
    ieee_standards = ["IEEE 287-2007"]
    related_demos = ["04_advanced_analysis/09_tdr.py"]

    def generate_data(self) -> None:
        """Generate frequency sweep data."""
        self.sample_rate = 100e9
        freqs = np.logspace(6, 10, 100)  # 1 MHz to 10 GHz

        # Good transmission line (50Ω matched)
        self.good_s11, self.good_s21 = self._generate_sparams(
            freqs, z0=50, zl=50, loss_db_per_ghz=1.0
        )

        # Mismatched load
        self.mismatch_s11, self.mismatch_s21 = self._generate_sparams(
            freqs, z0=50, zl=75, loss_db_per_ghz=1.0
        )

        # Lossy line
        self.lossy_s11, self.lossy_s21 = self._generate_sparams(
            freqs, z0=50, zl=50, loss_db_per_ghz=5.0
        )

        self.freqs = freqs

    def run_analysis(self) -> None:
        """Analyze S-parameters."""
        from demos.common.formatting import print_subheader

        print_subheader("Matched 50Ω Line")
        self.results["good"] = self._analyze_sparams(
            self.freqs, self.good_s11, self.good_s21, "Good"
        )

        print_subheader("Mismatched Load (75Ω)")
        self.results["mismatch"] = self._analyze_sparams(
            self.freqs,
            self.mismatch_s11,
            self.mismatch_s21,
            "Mismatch",
        )

        print_subheader("Lossy Transmission Line")
        self.results["lossy"] = self._analyze_sparams(
            self.freqs, self.lossy_s11, self.lossy_s21, "Lossy"
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate S-parameters."""
        if "good" in self.results:
            rl = self.results["good"]["return_loss_db"]
            suite.check_greater_than(
                "matched_return_loss",
                rl,
                20.0,
                "Matched line: RL > 20 dB",
            )

    def _generate_sparams(
        self,
        freqs: np.ndarray,
        z0: float,
        zl: float,
        loss_db_per_ghz: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate S-parameters for transmission line."""
        # S11 (reflection coefficient)
        gamma = (zl - z0) / (zl + z0)
        s11 = np.full_like(freqs, gamma, dtype=complex)

        # S21 (transmission coefficient)
        s21 = (1 + gamma) * np.exp(-loss_db_per_ghz * freqs / 1e9 / 20 * np.log(10))
        s21 = s21.astype(complex)

        return s11, s21

    def _analyze_sparams(
        self,
        freqs: np.ndarray,
        s11: np.ndarray,
        s21: np.ndarray,
        label: str,
    ) -> dict:
        """Analyze S-parameters."""
        from demos.common.formatting import print_info

        # Return loss (dB)
        return_loss_db = -20 * np.log10(np.abs(s11))
        rl_avg = np.mean(return_loss_db)

        # Insertion loss (dB)
        insertion_loss_db = -20 * np.log10(np.abs(s21))
        il_1ghz_idx = np.argmin(np.abs(freqs - 1e9))
        il_1ghz = insertion_loss_db[il_1ghz_idx]

        print_info(f"Return loss (avg): {rl_avg:.1f} dB")
        print_info(f"Insertion loss @ 1 GHz: {il_1ghz:.1f} dB")

        return {
            "return_loss_db": rl_avg,
            "insertion_loss_1ghz_db": il_1ghz,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(SignalIntegrityDemo))
