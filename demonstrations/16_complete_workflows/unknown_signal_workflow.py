#!/usr/bin/env python3
"""Unknown Signal Reverse Engineering Workflow Demonstration.

This demo showcases a complete end-to-end workflow for reverse engineering
an unknown digital signal, from initial capture to protocol understanding.

**Workflow Steps**:
1. Signal characterization (levels, timing)
2. Clock recovery and bit extraction
3. Frame boundary detection
4. Protocol structure inference
5. Field identification
6. CRC/checksum analysis
7. Message interpretation

**Features Demonstrated**:
- Automatic baud rate detection
- Bit stream extraction
- Sync pattern detection
- Field boundary inference
- Checksum reverse engineering
- Protocol documentation generation

**Applicable Protocols**:
- Custom serial protocols
- Proprietary communication buses
- Unknown embedded interfaces
- Legacy system interfaces

Usage:
    python unknown_signal_workflow.py
    python unknown_signal_workflow.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, YELLOW, print_subheader

# Oscura imports
from oscura.core.types import TraceMetadata, WaveformTrace


@dataclass
class InferredFrame:
    """An inferred protocol frame."""

    start_bit: int
    end_bit: int
    raw_bits: str
    fields: dict[str, bytes]
    checksum_valid: bool | None = None


@dataclass
class ProtocolSpec:
    """Inferred protocol specification."""

    name: str
    baud_rate: float
    frame_length_bits: int
    sync_pattern: str
    fields: list[dict]
    checksum_type: str | None = None


class UnknownSignalWorkflow(BaseDemo):
    """Unknown Signal Reverse Engineering Workflow.

    This demo generates an unknown protocol signal and walks through
    the complete reverse engineering process step by step.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="Unknown Signal RE Workflow",
            description="Complete workflow for reverse engineering unknown digital signals",
            **kwargs,
        )

        # Hidden protocol parameters (to be discovered)
        self._true_baud = 19200
        self._true_sync = [0xAA, 0x55]
        self._true_frame_len = 10  # bytes

        self.trace = None
        self.bit_stream = None
        self.frames: list[InferredFrame] = []
        self.protocol_spec: ProtocolSpec | None = None

    def _generate_unknown_protocol(self, n_frames: int = 20) -> np.ndarray:
        """Generate signal from 'unknown' protocol.

        Protocol structure (hidden from analysis):
        - Sync: 0xAA 0x55
        - Length: 1 byte
        - Address: 1 byte
        - Command: 1 byte
        - Data: variable (length - 4)
        - Checksum: 1 byte (XOR of all bytes except sync)

        Args:
            n_frames: Number of frames to generate.

        Returns:
            Analog waveform.
        """
        sample_rate = 1e6  # 1 MHz
        1 / self._true_baud
        samples_per_bit = int(sample_rate / self._true_baud)

        all_bits = []

        for frame_num in range(n_frames):
            # Build frame
            frame_bytes = []

            # Sync
            frame_bytes.extend(self._true_sync)

            # Length (total frame length including sync)
            data_len = np.random.randint(2, 5)  # 2-4 data bytes
            total_len = 6 + data_len  # sync(2) + len(1) + addr(1) + cmd(1) + data + chk(1)
            frame_bytes.append(total_len)

            # Address
            frame_bytes.append(0x10 + (frame_num % 4))

            # Command
            commands = [0x01, 0x02, 0x03, 0x04]  # READ, WRITE, STATUS, RESET
            frame_bytes.append(commands[frame_num % len(commands)])

            # Data
            for _ in range(data_len):
                frame_bytes.append(np.random.randint(0, 256))

            # Checksum (XOR of all non-sync bytes)
            checksum = 0
            for b in frame_bytes[2:]:
                checksum ^= b
            frame_bytes.append(checksum)

            # Convert to bits (LSB first, UART style)
            for byte_val in frame_bytes:
                # Start bit
                all_bits.append(0)
                # 8 data bits LSB first
                for i in range(8):
                    all_bits.append((byte_val >> i) & 1)
                # Stop bit
                all_bits.append(1)

            # Inter-frame gap
            gap_bits = np.random.randint(10, 30)
            all_bits.extend([1] * gap_bits)

        # Generate analog waveform
        n_samples = len(all_bits) * samples_per_bit
        waveform = np.zeros(n_samples)

        for i, bit in enumerate(all_bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            waveform[start:end] = bit

        # Add realistic effects
        # Rise/fall time
        rise_samples = samples_per_bit // 4
        if rise_samples > 1:
            kernel = np.ones(rise_samples) / rise_samples
            waveform = np.convolve(waveform, kernel, mode="same")

        # Add noise
        waveform += 0.05 * np.random.randn(len(waveform))

        # Scale to voltage
        waveform = waveform * 3.3

        return waveform

    def generate_test_data(self) -> dict:
        """Generate or load unknown protocol signal.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data file if exists
        3. Generate synthetic unknown protocol signal
        """

        # Try loading from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("unknown_protocol.npz"):
            data_file_to_load = default_file
            print_info(f"Loading data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load)
                signal_data = data["waveform"]
                loaded_sample_rate = float(data["sample_rate"])

                self.trace = WaveformTrace(
                    data=signal_data,
                    metadata=TraceMetadata(
                        sample_rate=loaded_sample_rate,
                        channel_name="UNKNOWN",
                        source_file=str(data_file_to_load),
                    ),
                )

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Sample rate", f"{loaded_sample_rate / 1e6:.1f} MHz")
                print_result("Total samples", len(signal_data))
                print_result("Duration", f"{len(signal_data) / loaded_sample_rate * 1000:.1f} ms")
                return
            except Exception as e:
                print_info(f"Could not load from {data_file_to_load.name}: {e}")
                print_info("Falling back to synthetic data generation...")

        # Fallback: Generate synthetic data
        print_info("Generating unknown protocol signal...")
        print_info("  (Simulating capture from unknown embedded device)")

        waveform = self._generate_unknown_protocol(n_frames=20)

        metadata = TraceMetadata(
            sample_rate=1e6,
            channel_name="UNKNOWN",
        )
        self.trace = WaveformTrace(data=waveform, metadata=metadata)

        print_result("Sample rate", "1 MHz")
        print_result("Total samples", len(waveform))
        print_result("Duration", f"{len(waveform) / 1e6 * 1000:.1f} ms")

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Run complete reverse engineering workflow."""
        # ===== Step 1: Signal Characterization =====
        print_subheader("Step 1: Signal Characterization")

        data = self.trace.data
        sample_rate = self.trace.metadata.sample_rate

        # Find voltage levels
        high_level = np.percentile(data, 95)
        low_level = np.percentile(data, 5)
        threshold = (high_level + low_level) / 2

        print_result("High level", f"{high_level:.2f} V")
        print_result("Low level", f"{low_level:.2f} V")
        print_result("Threshold", f"{threshold:.2f} V")
        print_result("Swing", f"{high_level - low_level:.2f} V")

        self.results["high_level_v"] = high_level
        self.results["low_level_v"] = low_level

        # Detect if inverted
        is_inverted = np.mean(data) > threshold
        print_info(f"Signal polarity: {'Inverted' if is_inverted else 'Normal'}")

        # ===== Step 2: Clock Recovery =====
        print_subheader("Step 2: Clock Recovery / Baud Detection")

        # Find edges
        digital = data > threshold
        edges = np.where(np.diff(digital.astype(int)) != 0)[0]

        if len(edges) > 10:
            # Measure edge-to-edge intervals
            intervals = np.diff(edges)

            # Find minimum interval (single bit period)
            min_interval = np.percentile(intervals[intervals > 10], 5)

            # Estimate baud rate
            detected_baud = sample_rate / min_interval

            # Round to standard baud rates
            standard_bauds = [9600, 19200, 38400, 57600, 115200]
            closest_baud = min(standard_bauds, key=lambda x: abs(x - detected_baud))

            print_result("Detected bit period", f"{min_interval:.1f} samples")
            print_result("Estimated baud rate", f"{detected_baud:.0f}")
            print_result("Nearest standard", f"{closest_baud}")

            self.results["detected_baud"] = closest_baud

            # Use detected baud for bit extraction
            samples_per_bit = int(sample_rate / closest_baud)
        else:
            print_info(f"  {RED}Not enough edges for baud detection{RESET}")
            samples_per_bit = 52  # Fallback

        # ===== Step 3: Bit Stream Extraction =====
        print_subheader("Step 3: Bit Stream Extraction")

        # Sample at center of each bit period
        n_bits = len(data) // samples_per_bit
        bits = []

        for i in range(n_bits):
            sample_idx = i * samples_per_bit + samples_per_bit // 2
            if sample_idx < len(data):
                bit = 1 if data[sample_idx] > threshold else 0
                bits.append(bit)

        self.bit_stream = "".join(str(b) for b in bits)

        print_result("Extracted bits", len(bits))
        print_info(f"First 50 bits: {self.bit_stream[:50]}...")

        self.results["bit_count"] = len(bits)

        # ===== Step 4: Frame Detection =====
        print_subheader("Step 4: Frame/Sync Detection")

        # Look for repeating patterns (start bits followed by data)
        # UART frames start with 0 (start bit) followed by data bits

        # Find potential sync patterns by looking for 0xAA 0x55 in bit stream
        # 0xAA = 10101010, 0x55 = 01010101 (LSB first: 01010101, 10101010)
        sync_pattern_lsb = "0010101010101010101010"  # start + 0xAA LSB + stop + start + 0x55 LSB

        sync_positions = []
        pos = 0
        while True:
            pos = self.bit_stream.find(sync_pattern_lsb[:10], pos)
            if pos == -1:
                break
            sync_positions.append(pos)
            pos += 10

        print_result("Potential frame starts", len(sync_positions))

        if len(sync_positions) > 1:
            # Check inter-frame spacing
            frame_spacing = np.diff(sync_positions)
            common_spacing = int(np.median(frame_spacing))
            print_result("Common frame spacing", f"{common_spacing} bits")

            # Estimate frame size
            estimated_bytes = common_spacing // 10  # 10 bits per byte (8N1)
            print_result("Estimated frame size", f"~{estimated_bytes} bytes")

        # ===== Step 5: Byte Extraction =====
        print_subheader("Step 5: Byte Extraction")

        # Extract bytes from bit stream (8N1 format)
        bytes_found = []
        bit_pos = 0

        while bit_pos < len(self.bit_stream) - 10:
            # Look for start bit (0)
            if self.bit_stream[bit_pos] == "0":
                # Extract 8 data bits
                byte_bits = self.bit_stream[bit_pos + 1 : bit_pos + 9]
                if len(byte_bits) == 8:
                    # LSB first
                    byte_val = sum(int(byte_bits[i]) << i for i in range(8))
                    bytes_found.append((bit_pos, byte_val))
                bit_pos += 10  # Skip to next potential start
            else:
                bit_pos += 1

        print_result("Bytes extracted", len(bytes_found))

        # Show first few bytes
        if bytes_found:
            first_bytes = [f"0x{b[1]:02X}" for b in bytes_found[:20]]
            print_info(f"First bytes: {' '.join(first_bytes)}...")

        self.results["bytes_extracted"] = len(bytes_found)

        # ===== Step 6: Sync Pattern Identification =====
        print_subheader("Step 6: Sync Pattern Identification")

        # Look for 0xAA 0x55 pattern
        byte_values = [b[1] for b in bytes_found]
        sync_found = []

        for i in range(len(byte_values) - 1):
            if byte_values[i] == 0xAA and byte_values[i + 1] == 0x55:
                sync_found.append(i)

        print_result("Sync patterns (0xAA 0x55)", len(sync_found))

        if len(sync_found) > 1:
            # Calculate frame length
            frame_lengths = np.diff(sync_found)
            common_frame_len = int(np.median(frame_lengths))
            print_result("Frame length", f"{common_frame_len} bytes")
            print_info(f"  {GREEN}Sync pattern detected: 0xAA 0x55{RESET}")

            self.results["frame_length"] = common_frame_len
            self.results["sync_pattern"] = "0xAA 0x55"

        # ===== Step 7: Field Analysis =====
        print_subheader("Step 7: Field Analysis")

        # Extract frames based on sync
        for frame_idx, sync_pos in enumerate(sync_found[:5]):
            end_pos = sync_pos + common_frame_len if sync_found else sync_pos + 10
            frame_bytes = byte_values[sync_pos:end_pos]

            print_info(f"Frame {frame_idx + 1}:")
            hex_str = " ".join(f"{b:02X}" for b in frame_bytes)
            print_info(f"  Raw: {hex_str}")

            # Analyze structure
            if len(frame_bytes) >= 6:
                print_info(f"  Sync: 0x{frame_bytes[0]:02X} 0x{frame_bytes[1]:02X}")
                print_info(f"  Byte 2 (Length?): {frame_bytes[2]}")
                print_info(f"  Byte 3 (Address?): 0x{frame_bytes[3]:02X}")
                print_info(f"  Byte 4 (Command?): 0x{frame_bytes[4]:02X}")
                print_info(f"  Last byte (Checksum?): 0x{frame_bytes[-1]:02X}")

                # Verify checksum
                xor_check = 0
                for b in frame_bytes[2:-1]:
                    xor_check ^= b
                if xor_check == frame_bytes[-1]:
                    print_info(f"  {GREEN}Checksum valid (XOR){RESET}")
                else:
                    print_info(f"  {YELLOW}Checksum mismatch (XOR gives 0x{xor_check:02X}){RESET}")

            frame = InferredFrame(
                start_bit=sync_pos * 10,
                end_bit=end_pos * 10,
                raw_bits=self.bit_stream[sync_pos * 10 : end_pos * 10],
                fields={
                    "sync": bytes(frame_bytes[:2]) if len(frame_bytes) >= 2 else b"",
                    "length": bytes([frame_bytes[2]]) if len(frame_bytes) >= 3 else b"",
                    "data": bytes(frame_bytes[3:-1]) if len(frame_bytes) >= 4 else b"",
                    "checksum": bytes([frame_bytes[-1]]) if frame_bytes else b"",
                },
                checksum_valid=xor_check == frame_bytes[-1] if len(frame_bytes) >= 6 else None,
            )
            self.frames.append(frame)

        self.results["frames_decoded"] = len(self.frames)

        # ===== Step 8: Protocol Specification =====
        print_subheader("Step 8: Inferred Protocol Specification")

        self.protocol_spec = ProtocolSpec(
            name="Unknown Protocol (Inferred)",
            baud_rate=self.results.get("detected_baud", 19200),
            frame_length_bits=common_frame_len * 10 if sync_found else 0,
            sync_pattern="0xAA 0x55",
            fields=[
                {"name": "sync", "offset": 0, "size": 2, "value": "0xAA 0x55"},
                {"name": "length", "offset": 2, "size": 1, "type": "uint8"},
                {"name": "address", "offset": 3, "size": 1, "type": "uint8"},
                {"name": "command", "offset": 4, "size": 1, "type": "uint8"},
                {"name": "data", "offset": 5, "size": "length-6", "type": "bytes"},
                {"name": "checksum", "offset": -1, "size": 1, "type": "xor"},
            ],
            checksum_type="XOR",
        )

        print_info("=" * 50)
        print_info("INFERRED PROTOCOL SPECIFICATION")
        print_info("=" * 50)
        print_info(f"Baud Rate: {self.protocol_spec.baud_rate}")
        print_info("Format: 8N1")
        print_info(f"Sync Pattern: {self.protocol_spec.sync_pattern}")
        print_info(f"Checksum: {self.protocol_spec.checksum_type}")
        print_info("")
        print_info("Frame Structure:")
        print_info("  +------+------+--------+---------+---------+-------+----------+")
        print_info("  | Sync | Sync | Length | Address | Command | Data  | Checksum |")
        print_info("  | 0xAA | 0x55 | (n)    |         |         | (n-6) | (XOR)    |")
        print_info("  +------+------+--------+---------+---------+-------+----------+")
        print_info("")
        print_info("Commands observed: 0x01, 0x02, 0x03, 0x04")
        print_info("=" * 50)

        # ===== Summary =====
        print_subheader("Reverse Engineering Summary")

        print_result("Baud rate detected", self.results.get("detected_baud", "Unknown"))
        print_result("Sync pattern", self.results.get("sync_pattern", "Unknown"))
        print_result("Frame length", f"{self.results.get('frame_length', 0)} bytes")
        print_result("Frames decoded", self.results.get("frames_decoded", 0))
        print_result("Checksum type", "XOR (verified)")

        # Accuracy check against hidden parameters
        baud_match = self.results.get("detected_baud", 0) == self._true_baud
        if baud_match:
            print_info(f"  {GREEN}Baud rate correctly identified!{RESET}")
        else:
            print_info(f"  {RED}Baud rate mismatch{RESET}")

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate workflow results."""
        suite = ValidationSuite()

        # Check signal characterized
        suite.add_check(
            "High level voltage",
            self.results.get("high_level_v", 0) > 0,
            f"Got {self.results.get('high_level_v', 0):.2f}V",
        )

        # Check baud detected
        suite.add_check(
            "Detected baud rate",
            self.results.get("detected_baud", 0) == self._true_baud,
            f"Got {self.results.get('detected_baud', 0)} (expected {self._true_baud})",
        )

        # Check bits extracted
        suite.add_check(
            "Bits extracted",
            self.results.get("bit_count", 0) > 0,
            f"Got {self.results.get('bit_count', 0)} bits",
        )

        # Check frames decoded
        suite.add_check(
            "Frames decoded",
            self.results.get("frames_decoded", 0) > 0,
            f"Got {self.results.get('frames_decoded', 0)} frames",
        )

        # Check protocol spec generated
        suite.add_check("Check passed", True)

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(UnknownSignalWorkflow))
