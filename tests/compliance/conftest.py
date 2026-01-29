"""Compliance test fixtures for format standards validation.

This module provides fixtures specific to compliance testing:
- VCD (Value Change Dump) format testing
- JEDEC memory timing compliance
- IEEE standard validation
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# =============================================================================
# VCD Compliance Fixtures
# =============================================================================


@pytest.fixture
def vcd_timescales() -> list[str]:
    """Valid IEEE 1364 VCD timescale values."""
    return [
        "1s",
        "100ms",
        "10ms",
        "1ms",
        "100us",
        "10us",
        "1us",
        "100ns",
        "10ns",
        "1ns",
        "100ps",
        "10ps",
        "1ps",
        "100fs",
        "10fs",
        "1fs",
    ]


@pytest.fixture
def vcd_variable_types() -> list[str]:
    """Valid VCD variable types per IEEE 1364."""
    return [
        "event",
        "integer",
        "parameter",
        "real",
        "realtime",
        "reg",
        "supply0",
        "supply1",
        "time",
        "tri",
        "triand",
        "trior",
        "trireg",
        "tri0",
        "tri1",
        "wand",
        "wire",
        "wor",
    ]


@pytest.fixture
def vcd_signal_values() -> dict[str, list[str]]:
    """Valid VCD signal values by type."""
    return {
        "scalar": ["0", "1", "x", "X", "z", "Z"],
        "vector": ["b0", "b1", "bx", "bz", "B0", "B1", "BX", "BZ"],
        "real": ["r0.0", "r1.5", "r-3.14", "R2.718"],
    }


@pytest.fixture
def minimal_vcd_content() -> str:
    """Minimal valid VCD file content."""
    return """$date
   Date text.
$end
$version
   VCD generator version.
$end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 8 " data [7:0] $end
$upscope $end
$enddefinitions $end
#0
$dumpvars
0!
b00000000 "
$end
#10
1!
#20
0!
b00001111 "
#30
1!
"""


@pytest.fixture
def vcd_with_hierarchy() -> str:
    """VCD content with nested scope hierarchy."""
    return """$timescale 1ns $end
$scope module top $end
$scope module cpu $end
$var wire 1 ! clk $end
$var reg 32 # pc [31:0] $end
$upscope $end
$scope module mem $end
$var wire 16 $ addr [15:0] $end
$var wire 8 % data [7:0] $end
$upscope $end
$upscope $end
$enddefinitions $end
#0
$dumpvars
0!
b00000000000000000000000000000000 #
b0000000000000000 $
b00000000 %
$end
#100
1!
b00000000000000000000000000000100 #
"""


@pytest.fixture
def vcd_with_real_values() -> str:
    """VCD content with real (floating point) values."""
    return """$timescale 1us $end
$scope module analog $end
$var real 1 ! voltage $end
$var real 1 " current $end
$upscope $end
$enddefinitions $end
#0
r0.0 !
r0.0 "
#100
r3.3 !
r0.001 "
#200
r1.8 !
r0.0005 "
"""


# =============================================================================
# JEDEC Compliance Fixtures
# =============================================================================


@pytest.fixture
def jedec_ddr4_timing() -> dict[str, dict[str, float]]:
    """JEDEC DDR4 timing parameters (in nanoseconds).

    Based on JEDEC JESD79-4 specification.
    Values represent common DDR4-2400 timing.
    """
    return {
        "tRCD": {"min": 13.75, "max": None},  # RAS to CAS Delay
        "tRP": {"min": 13.75, "max": None},  # Row Precharge Time
        "tRAS": {"min": 32.0, "max": 70000.0},  # Row Active Time
        "tRC": {"min": 45.75, "max": None},  # Row Cycle Time
        "tRFC": {"min": 350.0, "max": None},  # Refresh Cycle Time (8Gb)
        "tWR": {"min": 15.0, "max": None},  # Write Recovery Time
        "tRTP": {"min": 7.5, "max": None},  # Read to Precharge
        "tWTR_S": {"min": 2.5, "max": None},  # Write to Read (same bank)
        "tWTR_L": {"min": 7.5, "max": None},  # Write to Read (diff bank)
        "tCCD_S": {"min": 0.833, "max": None},  # CAS to CAS (short)
        "tCCD_L": {"min": 5.0, "max": None},  # CAS to CAS (long)
    }


@pytest.fixture
def jedec_ddr4_voltage() -> dict[str, dict[str, float]]:
    """JEDEC DDR4 voltage specifications."""
    return {
        "VDD": {"nominal": 1.2, "min": 1.14, "max": 1.26},
        "VDDQ": {"nominal": 1.2, "min": 1.14, "max": 1.26},
        "VPP": {"nominal": 2.5, "min": 2.375, "max": 2.75},
        "VIH_DC": {"min": 0.7, "max": None},  # Input High (DC)
        "VIL_DC": {"min": None, "max": 0.35},  # Input Low (DC)
        "VOH": {"min": 0.8, "max": None},  # Output High
        "VOL": {"min": None, "max": 0.2},  # Output Low
    }


@pytest.fixture
def jedec_sdram_commands() -> dict[str, dict[str, int]]:
    """JEDEC SDRAM command encoding.

    Command encoding based on RAS#, CAS#, WE# pins.
    """
    return {
        "DESELECT": {"RAS": 1, "CAS": 1, "WE": 1, "CS": 1},
        "NOP": {"RAS": 1, "CAS": 1, "WE": 1, "CS": 0},
        "ACTIVATE": {"RAS": 0, "CAS": 1, "WE": 1, "CS": 0},
        "READ": {"RAS": 1, "CAS": 0, "WE": 1, "CS": 0},
        "WRITE": {"RAS": 1, "CAS": 0, "WE": 0, "CS": 0},
        "PRECHARGE": {"RAS": 0, "CAS": 1, "WE": 0, "CS": 0},
        "REFRESH": {"RAS": 0, "CAS": 0, "WE": 1, "CS": 0},
        "MRS": {"RAS": 0, "CAS": 0, "WE": 0, "CS": 0},  # Mode Register Set
    }


# =============================================================================
# Signal Generation for Compliance Testing
# =============================================================================


@pytest.fixture
def generate_ddr_clock() -> callable:
    """Factory for generating DDR clock signals."""

    def _generate(
        frequency_mhz: float = 1200,
        duration_ns: float = 1000,
        sample_rate_ghz: float = 10,
        duty_cycle: float = 0.5,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Generate DDR clock signal.

        Args:
            frequency_mhz: Clock frequency in MHz.
            duration_ns: Signal duration in nanoseconds.
            sample_rate_ghz: Sample rate in GHz.

        Returns:
            Tuple of (signal, metadata).
        """
        sample_rate = sample_rate_ghz * 1e9
        frequency = frequency_mhz * 1e6
        duration = duration_ns * 1e-9

        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        period = 1 / frequency
        phase = (t % period) / period
        signal = np.where(phase < duty_cycle, 1.2, 0.0)

        # Add rise/fall time (simplified)
        # In real DDR, rise/fall times are ~0.5ns for DDR4

        metadata = {
            "frequency_mhz": frequency_mhz,
            "period_ns": 1000 / frequency_mhz,
            "sample_rate_ghz": sample_rate_ghz,
            "num_samples": num_samples,
        }

        return signal.astype(np.float64), metadata

    return _generate


@pytest.fixture
def generate_vcd_signal() -> callable:
    """Factory for generating VCD-compatible signal transitions."""

    def _generate(
        transitions: list[tuple[int, int]],
        end_time: int | None = None,
    ) -> list[tuple[int, str]]:
        """Generate VCD signal value changes.

        Args:
            transitions: List of (time, value) tuples.
            end_time: Optional end time for final state.

        Returns:
            List of (time, vcd_value) tuples.
        """
        result = []
        for time, value in transitions:
            if value == 0:
                result.append((time, "0"))
            elif value == 1:
                result.append((time, "1"))
            elif value == -1:  # Unknown
                result.append((time, "x"))
            elif value == -2:  # High-Z
                result.append((time, "z"))
            else:
                # Multi-bit value
                result.append((time, f"b{value:b}"))

        return result

    return _generate
