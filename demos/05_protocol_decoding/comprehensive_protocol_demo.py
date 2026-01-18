#!/usr/bin/env python3
"""Comprehensive Protocol Decoding Demo using BaseDemo Pattern.

This demo demonstrates Oscura's protocol decoding capabilities:
- UART with auto baud rate detection
- SPI multi-channel decoding
- I2C transaction decoding
- Auto protocol detection

Usage:
    python demos/05_protocol_decoding/comprehensive_protocol_demo.py
    python demos/05_protocol_decoding/comprehensive_protocol_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from demos.data_generation import SignalBuilder
from oscura.core.types import TraceMetadata, WaveformTrace


class ProtocolDecodingDemo(BaseDemo):
    """Protocol Decoding Demonstration.

    Demonstrates Oscura's comprehensive protocol decoding capabilities
    including auto-detection and multi-protocol support.
    """

    name = "Comprehensive Protocol Decoding"
    description = "Demonstrates protocol decoding (UART, SPI, I2C)"
    category = "protocol_decoding"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 10e6  # 10 MHz
        self.uart_trace = None
        self.spi_traces = {}
        self.i2c_traces = {}

    def generate_data(self) -> None:
        """Generate or load protocol signal data.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic data using SignalBuilder
        """
        import numpy as np

        # Try loading UART data from file
        uart_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            uart_file_to_load = self.data_file
            print_info(f"Loading UART data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("uart_rs232_communication.npz"):
            uart_file_to_load = default_file
            print_info(f"Loading UART data from default file: {default_file.name}")

        # Load UART from file if found
        if uart_file_to_load:
            try:
                data = np.load(uart_file_to_load)
                uart_data = data["rx"]
                loaded_sample_rate = float(data["sample_rate"])

                self.uart_trace = WaveformTrace(
                    data=uart_data,
                    metadata=TraceMetadata(
                        sample_rate=loaded_sample_rate,
                        channel_name="UART_RX",
                        source_file=str(uart_file_to_load),
                    ),
                )
                print_result("UART loaded from file", uart_file_to_load.name)
                print_result("UART samples", len(self.uart_trace.data))
            except Exception as e:
                print_info(f"Failed to load UART from file: {e}, falling back to synthetic")
                uart_file_to_load = None

        # Generate synthetic UART if not loaded
        if not uart_file_to_load:
            print_info("Generating synthetic protocol signals...")
            self._generate_uart()

        # Always generate SPI and I2C (or extend to support loading these too)
        self._generate_spi()
        self._generate_i2c()

    def _generate_uart(self) -> None:
        """Generate UART signal."""
        signal = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.01)
            .add_uart(baud_rate=115200, data=b"Hello Oscura!", amplitude=3.3)
            .build()
        )

        self.uart_trace = WaveformTrace(
            data=signal.data["uart"],
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="UART_TX",
                source_file="synthetic",
            ),
        )
        print_result("UART samples", len(self.uart_trace.data))

    def _generate_spi(self) -> None:
        """Generate SPI signals."""
        signal = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.001)
            .add_spi(
                clock_freq=1e6,
                mode=0,
                data_mosi=b"\x55\xaa\x12\x34",
                amplitude=3.3,
            )
            .build()
        )

        for ch_name in ["sck", "mosi", "miso", "cs"]:
            if ch_name in signal.data:
                self.spi_traces[ch_name] = WaveformTrace(
                    data=signal.data[ch_name],
                    metadata=TraceMetadata(
                        sample_rate=self.sample_rate,
                        channel_name=ch_name.upper(),
                        source_file="synthetic",
                    ),
                )
        print_result("SPI channels", len(self.spi_traces))

    def _generate_i2c(self) -> None:
        """Generate I2C signals."""
        signal = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.001)
            .add_i2c(
                clock_freq=100e3,
                address=0x50,
                data=b"\x00\x10\x20\x30",
                amplitude=3.3,
            )
            .build()
        )

        for ch_name in ["scl", "sda"]:
            if ch_name in signal.data:
                self.i2c_traces[ch_name] = WaveformTrace(
                    data=signal.data[ch_name],
                    metadata=TraceMetadata(
                        sample_rate=self.sample_rate,
                        channel_name=ch_name.upper(),
                        source_file="synthetic",
                    ),
                )
        print_result("I2C channels", len(self.i2c_traces))

    def run_analysis(self) -> None:
        """Execute protocol decoding."""
        # === Section 1: UART Decoding ===
        print_subheader("UART Decoding")
        self._decode_uart()

        # === Section 2: SPI Decoding ===
        print_subheader("SPI Decoding")
        self._decode_spi()

        # === Section 3: I2C Decoding ===
        print_subheader("I2C Decoding")
        self._decode_i2c()

        # === Section 4: Auto Protocol Detection ===
        print_subheader("Auto Protocol Detection")
        self._auto_detect()

    def _decode_uart(self) -> None:
        """Decode UART protocol."""
        # Try common baud rates
        common_rates = [9600, 19200, 38400, 57600, 115200, 230400]

        for baud_rate in common_rates:
            try:
                frames = osc.decode_uart(
                    self.uart_trace,
                    baudrate=baud_rate,
                    data_bits=8,
                    parity="none",
                    stop_bits=1,
                )
                frames_list = list(frames)

                if frames_list:
                    print_result("Detected baud rate", baud_rate, "bps")
                    print_result("Frames decoded", len(frames_list))

                    # Decode as text
                    text = ""
                    for frame in frames_list[:20]:
                        if hasattr(frame, "data") and frame.data:
                            text += chr(frame.data[0]) if 32 <= frame.data[0] < 127 else "."

                    if text:
                        print_info(f"Decoded text: '{text}'")

                    self.results["uart_baud_rate"] = baud_rate
                    self.results["uart_frames"] = len(frames_list)
                    return
            except Exception:
                continue

        print_info("No UART frames decoded")
        self.results["uart_frames"] = 0

    def _decode_spi(self) -> None:
        """Decode SPI protocol."""
        if len(self.spi_traces) < 4:
            print_info("Insufficient SPI channels for decoding")
            self.results["spi_frames"] = 0
            return

        try:
            frames = osc.decode_spi(
                self.spi_traces["sck"],
                self.spi_traces["mosi"],
                self.spi_traces["miso"],
                self.spi_traces["cs"],
                cpol=0,
                cpha=0,
            )
            frames_list = list(frames)

            print_result("SPI frames decoded", len(frames_list))

            # Show frame data
            for i, frame in enumerate(frames_list[:5]):
                if hasattr(frame, "mosi_data"):
                    print_info(f"  Frame {i}: MOSI=0x{frame.mosi_data.hex()}")

            self.results["spi_frames"] = len(frames_list)
        except Exception as e:
            print_info(f"SPI decoding: {e}")
            self.results["spi_frames"] = 0

    def _decode_i2c(self) -> None:
        """Decode I2C protocol."""
        if len(self.i2c_traces) < 2:
            print_info("Insufficient I2C channels for decoding")
            self.results["i2c_frames"] = 0
            return

        try:
            frames = osc.decode_i2c(
                self.i2c_traces["scl"],
                self.i2c_traces["sda"],
            )
            frames_list = list(frames)

            print_result("I2C frames decoded", len(frames_list))

            # Show frame data
            for i, frame in enumerate(frames_list[:5]):
                if hasattr(frame, "address"):
                    rw = "R" if getattr(frame, "read", False) else "W"
                    print_info(f"  Frame {i}: Addr=0x{frame.address:02X} ({rw})")

            self.results["i2c_frames"] = len(frames_list)
        except Exception as e:
            print_info(f"I2C decoding: {e}")
            self.results["i2c_frames"] = 0

    def _auto_detect(self) -> None:
        """Auto-detect protocol from signal."""
        try:
            protocol = osc.detect_protocol(self.uart_trace)
            print_result("Detected protocol", protocol if protocol else "Unknown")
            self.results["detected_protocol"] = protocol

            # Try auto-decode
            try:
                result = osc.auto_decode(self.uart_trace)
                if result:
                    print_result("Auto-decode protocol", result.protocol)
                    print_result("Auto-decode frames", len(result.frames))
                    self.results["auto_decode_protocol"] = result.protocol
                    self.results["auto_decode_frames"] = len(result.frames)
            except Exception:
                print_info("Auto-decode not available")
        except Exception as e:
            print_info(f"Protocol detection: {e}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate decoding results."""
        # UART decoding
        suite.check_greater(
            "UART frames decoded",
            self.results.get("uart_frames", 0),
            0,
            category="uart",
        )

        # Overall protocol support
        total_frames = (
            self.results.get("uart_frames", 0)
            + self.results.get("spi_frames", 0)
            + self.results.get("i2c_frames", 0)
        )
        suite.check_greater(
            "Total frames decoded",
            total_frames,
            0,
            category="overall",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(ProtocolDecodingDemo))
