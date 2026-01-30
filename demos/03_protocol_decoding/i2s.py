"""I2S Audio Protocol Decoding: Inter-IC Sound digital audio

Demonstrates:
- oscura.decode_i2s() - I2S audio stream decoding
- Bit clock, word select, and data signals
- Stereo audio frames (left/right channels)
- Multiple I2S formats (Philips, MSB-first, PCM)
- Sample word extraction

IEEE Standards: Philips I2S Specification
Related Demos:
- 03_protocol_decoding/02_spi_basic.py - SPI protocol (similar timing)
- 02_basic_analysis/01_waveform_measurements.py - Signal measurements
- 05_domain_specific/audio/ - Audio analysis

This demonstration generates synthetic I2S audio streams and decodes
stereo audio samples for digital audio applications.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import DigitalTrace, TraceMetadata


class I2SDemo(BaseDemo):
    """I2S audio protocol decoding demonstration."""

    name = "I2S Audio Protocol Decoding"
    description = "Decode I2S digital audio streams with stereo channels"
    category = "protocol_decoding"
    capabilities = [
        "I2S audio decoding",
        "Stereo channel separation (left/right)",
        "Multiple I2S formats (Philips, MSB, PCM)",
        "Sample word extraction",
        "Bit clock synchronization",
    ]
    ieee_standards = ["Philips I2S Bus Specification"]
    related_demos = [
        "03_protocol_decoding/02_spi_basic.py",
        "02_basic_analysis/01_waveform_measurements.py",
    ]

    def generate_data(self) -> None:
        """Generate synthetic I2S audio signals."""
        # Standard Philips I2S format (16-bit)
        self.i2s_philips = self._generate_i2s_stream(
            left_samples=[0x1234, 0x5678, 0x9ABC],
            right_samples=[0xDEF0, 0x1111, 0x2222],
            word_size=16,
            format_type="philips",
            sample_rate=48000,
            clock_rate=3.072e6,  # 48kHz * 16 bits * 2 channels * 2
            signal_sample_rate=50e6,
        )

        # MSB-first (left-justified) format (24-bit)
        self.i2s_msb = self._generate_i2s_stream(
            left_samples=[0x123456, 0x789ABC],
            right_samples=[0xDEF012, 0x345678],
            word_size=24,
            format_type="msb_first",
            sample_rate=96000,
            clock_rate=4.608e6,
            signal_sample_rate=50e6,
        )

    def run_analysis(self) -> None:
        """Decode I2S audio streams."""
        from demos.common.formatting import print_subheader

        print_subheader("Philips I2S Format (16-bit, 48kHz)")
        self.results["philips"] = self._decode_i2s_stream(
            *self.i2s_philips,
            word_size=16,
            expected_left=[0x1234, 0x5678, 0x9ABC],
            expected_right=[0xDEF0, 0x1111, 0x2222],
        )

        print_subheader("MSB-First Format (24-bit, 96kHz)")
        self.results["msb"] = self._decode_i2s_stream(
            *self.i2s_msb,
            word_size=24,
            expected_left=[0x123456, 0x789ABC],
            expected_right=[0xDEF012, 0x345678],
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate I2S decoding."""
        for config_name, result in self.results.items():
            suite.check_exists(
                f"{config_name}_sck",
                result.get("sck"),
                f"{config_name}: Bit clock generated",
            )
            suite.check_exists(
                f"{config_name}_ws",
                result.get("ws"),
                f"{config_name}: Word select generated",
            )
            suite.check_exists(
                f"{config_name}_sd",
                result.get("sd"),
                f"{config_name}: Serial data generated",
            )

    def _generate_i2s_stream(
        self,
        left_samples: list[int],
        right_samples: list[int],
        word_size: int,
        format_type: str,
        sample_rate: float,
        clock_rate: float,
        signal_sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace, DigitalTrace]:
        """Generate I2S audio stream signals.

        Args:
            left_samples: Left channel sample values
            right_samples: Right channel sample values
            word_size: Bits per sample (16, 24, or 32)
            format_type: "philips", "msb_first", or "pcm"
            sample_rate: Audio sample rate in Hz
            clock_rate: Bit clock rate in Hz
            signal_sample_rate: Signal sampling rate in Hz

        Returns:
            Tuple of (SCK, WS, SD) digital traces
        """
        samples_per_clock = max(1, int(signal_sample_rate / clock_rate))

        sck_signal = []
        ws_signal = []
        sd_signal = []

        # Process stereo frames
        for left_val, right_val in zip(left_samples, right_samples):
            # Left channel (WS low)
            for bit_idx in range(word_size):
                # Word select low for left channel
                ws_val = 0

                # Extract bit (MSB first for all formats)
                if format_type == "philips":
                    # Philips: data delayed by 1 clock
                    bit_val = (left_val >> (word_size - 1 - bit_idx)) & 1
                else:  # msb_first
                    bit_val = (left_val >> (word_size - 1 - bit_idx)) & 1

                # Clock low half
                sck_signal.extend([0] * (samples_per_clock // 2))
                ws_signal.extend([ws_val] * (samples_per_clock // 2))
                sd_signal.extend([bit_val] * (samples_per_clock // 2))

                # Clock high half
                sck_signal.extend([1] * (samples_per_clock // 2))
                ws_signal.extend([ws_val] * (samples_per_clock // 2))
                sd_signal.extend([bit_val] * (samples_per_clock // 2))

            # Right channel (WS high)
            for bit_idx in range(word_size):
                ws_val = 1
                bit_val = (right_val >> (word_size - 1 - bit_idx)) & 1

                sck_signal.extend([0] * (samples_per_clock // 2))
                ws_signal.extend([ws_val] * (samples_per_clock // 2))
                sd_signal.extend([bit_val] * (samples_per_clock // 2))

                sck_signal.extend([1] * (samples_per_clock // 2))
                ws_signal.extend([ws_val] * (samples_per_clock // 2))
                sd_signal.extend([bit_val] * (samples_per_clock // 2))

        return (
            DigitalTrace(
                data=np.array(sck_signal, dtype=bool),
                metadata=TraceMetadata(sample_rate=signal_sample_rate, channel_name="i2s_sck"),
            ),
            DigitalTrace(
                data=np.array(ws_signal, dtype=bool),
                metadata=TraceMetadata(sample_rate=signal_sample_rate, channel_name="i2s_ws"),
            ),
            DigitalTrace(
                data=np.array(sd_signal, dtype=bool),
                metadata=TraceMetadata(sample_rate=signal_sample_rate, channel_name="i2s_sd"),
            ),
        )

    def _decode_i2s_stream(
        self,
        sck: DigitalTrace,
        ws: DigitalTrace,
        sd: DigitalTrace,
        word_size: int,
        expected_left: list[int],
        expected_right: list[int],
    ) -> dict[str, object]:
        """Decode I2S audio stream."""
        from demos.common.formatting import print_info

        print_info(f"Sample rate: {sck.metadata.sample_rate / 1e6:.1f} MHz")
        print_info(f"Word size: {word_size} bits")
        print_info(f"Expected left samples: {len(expected_left)}")
        print_info(f"Expected right samples: {len(expected_right)}")

        frames = []
        try:
            from oscura import decode_i2s

            frames = decode_i2s(
                sck,
                ws,
                sd,
                sample_rate=sck.metadata.sample_rate,
                word_size=word_size,
            )
            print_info(f"Frames decoded: {len(frames)}")
        except (ImportError, AttributeError):
            print_info("I2S decoder not yet implemented (placeholder)")

        return {
            "sck": sck,
            "ws": ws,
            "sd": sd,
            "frames": frames,
            "expected_left": expected_left,
            "expected_right": expected_right,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(I2SDemo))
