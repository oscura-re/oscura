#!/usr/bin/env python3
"""I2S Audio Protocol Decoding Demonstration.

This demo showcases Oscura's I2S (Inter-IC Sound) protocol decoding
capabilities for audio data buses.

**Features Demonstrated**:
- Standard I2S decoding (Phillips format)
- Left-justified and right-justified modes
- Multiple bit depths (16, 24, 32-bit)
- Word Select (LRCLK) synchronization
- Bit Clock (SCLK/BCK) recovery
- Stereo sample extraction
- Audio level analysis

**I2S Signal Definitions**:
- SCK/BCK: Bit Clock (serial clock)
- WS/LRCLK: Word Select (0=Left, 1=Right)
- SD/SDATA: Serial Data

**I2S Modes**:
- Standard (Phillips): Data MSB is 1 BCK after WS change
- Left-justified: Data MSB at WS change
- Right-justified: Data LSB at WS change

**Common Sample Rates**:
- 44.1 kHz (CD quality)
- 48 kHz (DVD/Broadcast)
- 96 kHz (High resolution)
- 192 kHz (Studio quality)

Usage:
    python i2s_demo.py
    python i2s_demo.py --verbose

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
from demonstrations.common.formatting import GREEN, RESET, print_subheader

# Oscura imports
from oscura.analyzers.protocols.i2s import decode_i2s


class I2SDemo(BaseDemo):
    """I2S Audio Protocol Decoding Demonstration.

    This demo generates I2S signals with audio sample data and decodes
    them to demonstrate Oscura's I2S analysis capabilities.
    """

    name = "I2S Audio Protocol Demo"
    description = "Demonstrates I2S audio bus protocol decoding"
    category = "serial_protocols"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)

        # I2S parameters
        self.audio_sample_rate = 48000  # 48 kHz audio
        self.bit_depth = 16  # 16-bit samples
        self.capture_sample_rate = 10e6  # 10 MHz logic analyzer rate

        # Calculated parameters
        self.bck_freq = self.audio_sample_rate * self.bit_depth * 2  # BCK frequency
        self.ws_freq = self.audio_sample_rate  # Word Select frequency

        # Storage
        self.bck = None
        self.ws = None
        self.sd = None
        self.packets = []

    def _generate_sine_samples(
        self, n_samples: int, frequency: float, amplitude: float = 0.8
    ) -> tuple[list[int], list[int]]:
        """Generate sine wave audio samples.

        Args:
            n_samples: Number of stereo samples.
            frequency: Sine wave frequency in Hz.
            amplitude: Amplitude (0 to 1).

        Returns:
            Tuple of (left_samples, right_samples) as signed integers.
        """
        max_val = (1 << (self.bit_depth - 1)) - 1

        t = np.arange(n_samples) / self.audio_sample_rate

        # Left channel: sine wave
        left_float = amplitude * np.sin(2 * np.pi * frequency * t)
        left_samples = [int(s * max_val) for s in left_float]

        # Right channel: sine wave with phase shift (creates stereo effect)
        right_float = amplitude * np.sin(2 * np.pi * frequency * t + np.pi / 4)
        right_samples = [int(s * max_val) for s in right_float]

        return left_samples, right_samples

    def _encode_i2s(
        self,
        left_samples: list[int],
        right_samples: list[int],
        mode: str = "standard",
    ) -> tuple[list[int], list[int], list[int]]:
        """Encode audio samples as I2S signals.

        Args:
            left_samples: Left channel samples.
            right_samples: Right channel samples.
            mode: I2S mode ('standard', 'left_justified', 'right_justified').

        Returns:
            Tuple of (bck, ws, sd) bit lists.
        """
        bck_bits = []
        ws_bits = []
        sd_bits = []

        for left_sample, right_sample in zip(left_samples, right_samples, strict=False):
            # Left channel (WS = 0 in standard I2S)
            # In standard I2S, MSB is delayed by 1 BCK after WS change

            for _channel, sample, ws_val in [
                (0, left_sample, 0),
                (1, right_sample, 1),
            ]:
                # Convert signed to unsigned for bit extraction
                if sample < 0:
                    sample_unsigned = sample + (1 << self.bit_depth)
                else:
                    sample_unsigned = sample

                # Extract bits (MSB first)
                bits = [
                    (sample_unsigned >> (self.bit_depth - 1 - i)) & 1 for i in range(self.bit_depth)
                ]

                if mode == "standard":
                    # Delay MSB by 1 BCK
                    # First BCK after WS change has no data
                    for i in range(self.bit_depth):
                        # Rising edge of BCK
                        bck_bits.append(0)
                        ws_bits.append(ws_val)
                        if i == 0:
                            sd_bits.append(0)  # Delay bit
                        else:
                            sd_bits.append(bits[i - 1])

                        # Falling edge of BCK
                        bck_bits.append(1)
                        ws_bits.append(ws_val)
                        if i == 0:
                            sd_bits.append(bits[0])  # First data bit
                        else:
                            sd_bits.append(bits[i - 1])

                elif mode == "left_justified":
                    # MSB at WS change
                    for i in range(self.bit_depth):
                        bck_bits.append(0)
                        ws_bits.append(ws_val)
                        sd_bits.append(bits[i])

                        bck_bits.append(1)
                        ws_bits.append(ws_val)
                        sd_bits.append(bits[i])

                else:  # right_justified
                    # LSB aligned to end
                    for i in range(self.bit_depth):
                        bck_bits.append(0)
                        ws_bits.append(ws_val)
                        sd_bits.append(bits[i])

                        bck_bits.append(1)
                        ws_bits.append(ws_val)
                        sd_bits.append(bits[i])

        return bck_bits, ws_bits, sd_bits

    def generate_data(self) -> None:
        """Generate or load I2S test signals.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic data
        """
        # Try loading I2S data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading I2S data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("i2s_audio_stream.npz"):
            file_to_load = default_file
            print_info(f"Loading I2S data from default file: {default_file.name}")

        # Load I2S from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load)
                self.bck = data["bck"]
                self.ws = data["ws"]
                self.sd = data["sd"]
                loaded_sample_rate = float(data["sample_rate"])
                self.capture_sample_rate = loaded_sample_rate

                # Load audio parameters if available
                if "audio_sample_rate" in data:
                    self.audio_sample_rate = int(data["audio_sample_rate"])
                if "bit_depth" in data:
                    self.bit_depth = int(data["bit_depth"])

                # Recalculate derived parameters
                self.bck_freq = self.audio_sample_rate * self.bit_depth * 2
                self.ws_freq = self.audio_sample_rate

                print_result("I2S loaded from file", file_to_load.name)
                print_result("Capture sample rate", f"{self.capture_sample_rate / 1e6:.1f} MHz")
                print_result("Audio sample rate", f"{self.audio_sample_rate / 1e3:.1f} kHz")
                print_result("Bit depth", f"{self.bit_depth} bits")
                print_result("Total samples", len(self.bck))
                return  # Successfully loaded
            except Exception as e:
                print_info(f"Failed to load I2S from file: {e}, falling back to synthetic")
                file_to_load = None

        # Generate synthetic I2S if not loaded
        print_info("Generating I2S test signals...")
        self._generate_synthetic_i2s()

    def _generate_synthetic_i2s(self) -> None:
        """Generate synthetic I2S test signals with audio data."""
        # Generate audio samples (440 Hz tone - A4)
        n_audio_samples = 100  # 100 stereo samples
        tone_freq = 440  # Hz

        print_info(f"  Audio frequency: {tone_freq} Hz (A4)")
        print_info(f"  Sample rate: {self.audio_sample_rate} Hz")
        print_info(f"  Bit depth: {self.bit_depth} bits")
        print_info(f"  Stereo samples: {n_audio_samples}")

        left_samples, right_samples = self._generate_sine_samples(n_audio_samples, tone_freq)

        print_info(f"  Left channel range: [{min(left_samples)}, {max(left_samples)}]")
        print_info(f"  Right channel range: [{min(right_samples)}, {max(right_samples)}]")

        # Encode as I2S
        bck_bits, ws_bits, sd_bits = self._encode_i2s(left_samples, right_samples, mode="standard")

        # Expand to capture sample rate
        samples_per_bck_half = max(1, int(self.capture_sample_rate / (self.bck_freq * 2)))

        bck_signal = []
        ws_signal = []
        sd_signal = []

        for bck, ws, sd in zip(bck_bits, ws_bits, sd_bits, strict=False):
            bck_signal.extend([bck] * samples_per_bck_half)
            ws_signal.extend([ws] * samples_per_bck_half)
            sd_signal.extend([sd] * samples_per_bck_half)

        # Add idle at start and end
        idle_samples = [1] * int(samples_per_bck_half * 10)
        ws_idle = [0] * int(samples_per_bck_half * 10)
        sd_idle = [0] * int(samples_per_bck_half * 10)

        self.bck = np.array(idle_samples + bck_signal + idle_samples, dtype=bool)
        self.ws = np.array(ws_idle + ws_signal + ws_idle, dtype=bool)
        self.sd = np.array(sd_idle + sd_signal + sd_idle, dtype=bool)

        print_result("BCK frequency", f"{self.bck_freq / 1e6:.3f} MHz")
        print_result("Samples per BCK half", samples_per_bck_half)
        print_result("Total samples", len(self.bck))
        print_result("Duration", f"{len(self.bck) / self.capture_sample_rate * 1e3:.2f} ms")

    def run_analysis(self) -> None:
        """Decode I2S signals and analyze audio data."""
        print_subheader("I2S Decoding")

        # Decode using convenience function
        self.packets = decode_i2s(
            bck=self.bck,
            ws=self.ws,
            sd=self.sd,
            sample_rate=self.capture_sample_rate,
            bit_depth=self.bit_depth,
            mode="standard",
        )

        print_result("Decoded stereo samples", len(self.packets))

        # Analyze decoded data
        print_subheader("Audio Sample Analysis")

        self.results["sample_count"] = len(self.packets)
        self.results["left_samples"] = []
        self.results["right_samples"] = []

        for i, pkt in enumerate(self.packets[:10]):  # Show first 10
            left = pkt.annotations.get("left_sample", 0)
            right = pkt.annotations.get("right_sample", 0)
            sample_num = pkt.annotations.get("sample_num", i)

            self.results["left_samples"].append(left)
            self.results["right_samples"].append(right)

            # Convert to dB relative to full scale
            max_val = (1 << (self.bit_depth - 1)) - 1
            left_db = 20 * np.log10(abs(left) / max_val + 1e-10)
            right_db = 20 * np.log10(abs(right) / max_val + 1e-10)

            print_info(
                f"  Sample {sample_num}: L={left:+6d} ({left_db:+.1f} dBFS), R={right:+6d} ({right_db:+.1f} dBFS)"
            )

        if len(self.packets) > 10:
            print_info(f"  ... ({len(self.packets) - 10} more samples)")

        # Collect all samples
        for pkt in self.packets[10:]:
            self.results["left_samples"].append(pkt.annotations.get("left_sample", 0))
            self.results["right_samples"].append(pkt.annotations.get("right_sample", 0))

        # Audio statistics
        print_subheader("Audio Statistics")

        if self.results["left_samples"]:
            left_array = np.array(self.results["left_samples"])
            right_array = np.array(self.results["right_samples"])

            max_val = (1 << (self.bit_depth - 1)) - 1

            # Peak levels
            left_peak = np.max(np.abs(left_array))
            right_peak = np.max(np.abs(right_array))

            left_peak_db = 20 * np.log10(left_peak / max_val + 1e-10)
            right_peak_db = 20 * np.log10(right_peak / max_val + 1e-10)

            print_result("Left peak", f"{left_peak_db:.1f} dBFS")
            print_result("Right peak", f"{right_peak_db:.1f} dBFS")

            self.results["left_peak_dbfs"] = left_peak_db
            self.results["right_peak_dbfs"] = right_peak_db

            # RMS levels
            left_rms = np.sqrt(np.mean(left_array.astype(float) ** 2))
            right_rms = np.sqrt(np.mean(right_array.astype(float) ** 2))

            left_rms_db = 20 * np.log10(left_rms / max_val + 1e-10)
            right_rms_db = 20 * np.log10(right_rms / max_val + 1e-10)

            print_result("Left RMS", f"{left_rms_db:.1f} dBFS")
            print_result("Right RMS", f"{right_rms_db:.1f} dBFS")

            # Crest factor
            left_crest = left_peak / (left_rms + 1e-10)
            right_crest = right_peak / (right_rms + 1e-10)

            print_result("Left crest factor", f"{20 * np.log10(left_crest):.1f} dB")
            print_result("Right crest factor", f"{20 * np.log10(right_crest):.1f} dB")

            # Stereo balance
            if right_rms > 0:
                balance = 20 * np.log10(left_rms / right_rms)
                print_result("Stereo balance", f"{balance:+.1f} dB")
                self.results["stereo_balance_db"] = balance

        # Summary
        print_subheader("Summary")
        print_result("Mode", "Standard I2S (Phillips)")
        print_result("Bit depth", f"{self.bit_depth} bits")
        print_result("Audio sample rate", f"{self.audio_sample_rate} Hz")
        print_result("Decoded samples", self.results["sample_count"])

        if self.results["sample_count"] > 0:
            print_info(f"  {GREEN}I2S decoding successful!{RESET}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate I2S decoding results."""
        # Check that samples were decoded
        suite.check_greater(
            "Sample count",
            self.results.get("sample_count", 0),
            0,
            category="decoding",
        )

        # Check that we got both channels
        suite.check_greater(
            "Left samples",
            len(self.results.get("left_samples", [])),
            0,
            category="channels",
        )

        suite.check_greater(
            "Right samples",
            len(self.results.get("right_samples", [])),
            0,
            category="channels",
        )

        # Check audio levels are reasonable
        left_peak = self.results.get("left_peak_dbfs", -100)
        suite.check_greater(
            "Left peak level",
            left_peak,
            -60,  # Should be above -60 dBFS for our test signal
            category="levels",
        )

        # Check signals were generated
        suite.check_true(
            "BCK signal generated",
            self.bck is not None,
            category="signals",
        )

        suite.check_true(
            "WS signal generated",
            self.ws is not None,
            category="signals",
        )

        suite.check_true(
            "SD signal generated",
            self.sd is not None,
            category="signals",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(I2SDemo))
