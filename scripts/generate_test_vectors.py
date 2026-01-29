#!/usr/bin/env python3
"""Generate test data files to eliminate "missing test data" skips.

This script generates:
- WFM files (oscilloscope captures) for Tektronix loader tests
- PCAP files (network captures) for protocol analysis tests
- Synthetic signal files for various analyzers
- Protocol test vectors for decoder validation

Usage:
    uv run python scripts/generate_test_vectors.py

Success: Eliminates 20+ "missing test data" skips (560 → 540 skips).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Check for optional dependencies
try:
    import tm_data_types  # type: ignore[import-untyped]

    HAS_TM_DATA_TYPES = True
except ImportError:
    HAS_TM_DATA_TYPES = False

try:
    from scapy.all import IP, TCP, UDP, Ether, wrpcap  # type: ignore[attr-defined]

    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False


class TestVectorGenerator:
    """Generate test data files for various formats."""

    def __init__(self, base_dir: Path | None = None, seed: int = 42):
        """Initialize test vector generator.

        Args:
            base_dir: Base directory for test data (default: test_data/)
            seed: Random seed for reproducibility
        """
        self.base_dir = base_dir or Path("test_data")
        self.rng = np.random.default_rng(seed)
        self.generated_files: list[str] = []

    # =========================================================================
    # WFM FILE GENERATORS (Tektronix oscilloscope format)
    # =========================================================================

    def generate_wfm_sine_wave(
        self,
        output_path: Path,
        freq_hz: float = 1000.0,
        amplitude: float = 1.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
        channel_name: str = "CH1",
    ) -> None:
        """Generate WFM file with clean sine wave.

        Args:
            output_path: Output WFM file path
            freq_hz: Sine wave frequency in Hz
            amplitude: Signal amplitude in volts
            sample_rate: Sample rate in Hz
            duration_s: Duration in seconds
            channel_name: Channel name for metadata
        """
        if not HAS_TM_DATA_TYPES:
            print(f"⚠️  Skipping {output_path.name}: tm_data_types not available")
            return

        # Generate sine wave
        num_samples = int(sample_rate * duration_s)
        t = np.arange(num_samples) / sample_rate
        data = amplitude * np.sin(2 * np.pi * freq_hz * t)

        # Create WFM file using tm_data_types
        wfm = tm_data_types.AnalogWaveform()
        wfm.y_axis_values = data.astype(np.float64)
        wfm.x_axis_spacing = 1.0 / sample_rate
        wfm.y_axis_spacing = 1.0
        wfm.y_axis_offset = 0.0
        wfm.source_name = channel_name

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write WFM file
        tm_data_types.write_file(str(output_path), wfm)
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} ({num_samples} samples, {freq_hz}Hz)")

    def generate_wfm_square_wave(
        self,
        output_path: Path,
        freq_hz: float = 1000.0,
        amplitude: float = 1.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
        duty_cycle: float = 0.5,
        channel_name: str = "CH1",
    ) -> None:
        """Generate WFM file with square wave (digital signal).

        Args:
            output_path: Output WFM file path
            freq_hz: Square wave frequency in Hz
            amplitude: Signal amplitude in volts
            sample_rate: Sample rate in Hz
            duration_s: Duration in seconds
            duty_cycle: Duty cycle (0-1)
            channel_name: Channel name for metadata
        """
        if not HAS_TM_DATA_TYPES:
            print(f"⚠️  Skipping {output_path.name}: tm_data_types not available")
            return

        # Generate square wave
        num_samples = int(sample_rate * duration_s)
        t = np.arange(num_samples) / sample_rate
        phase = (t * freq_hz) % 1.0
        data = np.where(phase < duty_cycle, amplitude, -amplitude)

        # Create WFM file
        wfm = tm_data_types.AnalogWaveform()
        wfm.y_axis_values = data.astype(np.float64)
        wfm.x_axis_spacing = 1.0 / sample_rate
        wfm.y_axis_spacing = 1.0
        wfm.y_axis_offset = 0.0
        wfm.source_name = channel_name

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tm_data_types.write_file(str(output_path), wfm)
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} ({num_samples} samples, {freq_hz}Hz)")

    def generate_wfm_noisy_signal(
        self,
        output_path: Path,
        freq_hz: float = 1000.0,
        snr_db: float = 20.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
        channel_name: str = "CH1",
    ) -> None:
        """Generate WFM file with noisy sine wave.

        Args:
            output_path: Output WFM file path
            freq_hz: Sine wave frequency in Hz
            snr_db: Signal-to-noise ratio in dB
            sample_rate: Sample rate in Hz
            duration_s: Duration in seconds
            channel_name: Channel name for metadata
        """
        if not HAS_TM_DATA_TYPES:
            print(f"⚠️  Skipping {output_path.name}: tm_data_types not available")
            return

        # Generate noisy sine wave
        num_samples = int(sample_rate * duration_s)
        t = np.arange(num_samples) / sample_rate
        signal = np.sin(2 * np.pi * freq_hz * t)

        # Add noise
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.standard_normal(num_samples) * np.sqrt(noise_power)
        data = signal + noise

        # Create WFM file
        wfm = tm_data_types.AnalogWaveform()
        wfm.y_axis_values = data.astype(np.float64)
        wfm.x_axis_spacing = 1.0 / sample_rate
        wfm.y_axis_spacing = 1.0
        wfm.y_axis_offset = 0.0
        wfm.source_name = channel_name

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tm_data_types.write_file(str(output_path), wfm)
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} ({num_samples} samples, SNR={snr_db}dB)")

    def generate_wfm_multi_channel(
        self,
        output_path: Path,
        num_channels: int = 4,
        freq_hz: float = 1000.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> None:
        """Generate multi-channel WFM file.

        Args:
            output_path: Output WFM file path
            num_channels: Number of channels (2-8)
            freq_hz: Base frequency in Hz
            sample_rate: Sample rate in Hz
            duration_s: Duration in seconds
        """
        if not HAS_TM_DATA_TYPES:
            print(f"⚠️  Skipping {output_path.name}: tm_data_types not available")
            return

        # Note: tm_data_types writes single-channel files
        # For multi-channel, create separate files with channel suffix
        for ch in range(num_channels):
            ch_path = output_path.parent / f"{output_path.stem}_CH{ch + 1}{output_path.suffix}"
            # Each channel has slightly different frequency
            ch_freq = freq_hz * (1 + ch * 0.1)
            self.generate_wfm_sine_wave(
                ch_path,
                freq_hz=ch_freq,
                sample_rate=sample_rate,
                duration_s=duration_s,
                channel_name=f"CH{ch + 1}",
            )

    # =========================================================================
    # PCAP FILE GENERATORS (Network packet captures)
    # =========================================================================

    def generate_pcap_tcp_simple(
        self,
        output_path: Path,
        payload: bytes = b"Hello, World!",
        src_ip: str = "192.168.1.100",
        dst_ip: str = "192.168.1.1",
        src_port: int = 12345,
        dst_port: int = 80,
    ) -> None:
        """Generate simple TCP PCAP file.

        Args:
            output_path: Output PCAP file path
            payload: TCP payload data
            src_ip: Source IP address
            dst_ip: Destination IP address
            src_port: Source port
            dst_port: Destination port
        """
        if not HAS_SCAPY:
            print(f"⚠️  Skipping {output_path.name}: scapy not available")
            return

        # Create TCP packet
        packet = (
            Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=src_port, dport=dst_port) / payload
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrpcap(str(output_path), [packet])
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (TCP, {len(payload)} bytes)")

    def generate_pcap_udp_simple(
        self,
        output_path: Path,
        payload: bytes = b"UDP test data",
        src_ip: str = "192.168.1.100",
        dst_ip: str = "192.168.1.1",
        src_port: int = 12345,
        dst_port: int = 53,
    ) -> None:
        """Generate simple UDP PCAP file.

        Args:
            output_path: Output PCAP file path
            payload: UDP payload data
            src_ip: Source IP address
            dst_ip: Destination IP address
            src_port: Source port
            dst_port: Destination port
        """
        if not HAS_SCAPY:
            print(f"⚠️  Skipping {output_path.name}: scapy not available")
            return

        # Create UDP packet
        packet = (
            Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=src_port, dport=dst_port) / payload
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrpcap(str(output_path), [packet])
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (UDP, {len(payload)} bytes)")

    def generate_pcap_http_request(
        self,
        output_path: Path,
        method: str = "GET",
        path: str = "/index.html",
        host: str = "example.com",
    ) -> None:
        """Generate PCAP with HTTP request.

        Args:
            output_path: Output PCAP file path
            method: HTTP method (GET, POST, etc.)
            path: Request path
            host: Host header value
        """
        if not HAS_SCAPY:
            print(f"⚠️  Skipping {output_path.name}: scapy not available")
            return

        # Create HTTP request payload
        http_request = f"{method} {path} HTTP/1.1\r\n"
        http_request += f"Host: {host}\r\n"
        http_request += "User-Agent: TestClient/1.0\r\n"
        http_request += "Accept: */*\r\n"
        http_request += "\r\n"

        # Create TCP packet with HTTP payload
        packet = (
            Ether()
            / IP(src="192.168.1.100", dst="192.168.1.1")
            / TCP(sport=12345, dport=80, flags="PA")
            / http_request.encode()
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrpcap(str(output_path), [packet])
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (HTTP {method})")

    def generate_pcap_malformed(
        self,
        output_path: Path,
        corruption_type: str = "truncated",
    ) -> None:
        """Generate malformed PCAP for error handling tests.

        Args:
            output_path: Output PCAP file path
            corruption_type: Type of corruption (truncated, invalid_checksum)
        """
        if not HAS_SCAPY:
            print(f"⚠️  Skipping {output_path.name}: scapy not available")
            return

        # Create normal packet first
        packet = Ether() / IP(src="192.168.1.100", dst="192.168.1.1") / TCP() / b"test"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrpcap(str(output_path), [packet])

        # Corrupt the file based on type
        if corruption_type == "truncated":
            # Truncate file to half size
            data = output_path.read_bytes()
            output_path.write_bytes(data[: len(data) // 2])
        elif corruption_type == "invalid_checksum":
            # Flip some bits in the packet data
            data_bytes = bytearray(output_path.read_bytes())
            if len(data_bytes) > 50:
                data_bytes[40] ^= 0xFF  # Corrupt IP checksum area
            output_path.write_bytes(bytes(data_bytes))

        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (malformed: {corruption_type})")

    # =========================================================================
    # NUMPY/NPZ FILE GENERATORS (Raw signal data)
    # =========================================================================

    def generate_npz_signal(
        self,
        output_path: Path,
        signal_type: str = "sine",
        freq_hz: float = 1000.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Generate NPZ file with signal data and metadata.

        Args:
            output_path: Output NPZ file path
            signal_type: Signal type (sine, square, sawtooth, noise)
            freq_hz: Signal frequency in Hz
            sample_rate: Sample rate in Hz
            duration_s: Duration in seconds
            metadata: Optional metadata dictionary
        """
        num_samples = int(sample_rate * duration_s)
        t = np.arange(num_samples) / sample_rate

        # Generate signal based on type
        if signal_type == "sine":
            data = np.sin(2 * np.pi * freq_hz * t)
        elif signal_type == "square":
            data = np.sign(np.sin(2 * np.pi * freq_hz * t))
        elif signal_type == "sawtooth":
            data = 2 * ((t * freq_hz) % 1.0) - 1
        elif signal_type == "noise":
            data = self.rng.standard_normal(num_samples)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        # Prepare metadata
        meta = metadata or {}
        meta.update(
            {
                "signal_type": signal_type,
                "frequency_hz": freq_hz,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
                "num_samples": num_samples,
            }
        )

        # Save NPZ file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(output_path),
            data=data.astype(np.float64),
            time=t.astype(np.float64),
            metadata=np.array([json.dumps(meta)]),  # Store metadata as JSON string
        )
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (NPZ, {signal_type}, {num_samples} samples)")

    # =========================================================================
    # CSV FILE GENERATORS (Waveform data in CSV format)
    # =========================================================================

    def generate_csv_waveform(
        self,
        output_path: Path,
        signal_type: str = "sine",
        freq_hz: float = 1000.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
        header: bool = True,
    ) -> None:
        """Generate CSV file with waveform data.

        Args:
            output_path: Output CSV file path
            signal_type: Signal type (sine, square, etc.)
            freq_hz: Signal frequency in Hz
            sample_rate: Sample rate in Hz
            duration_s: Duration in seconds
            header: Include CSV header row
        """
        num_samples = int(sample_rate * duration_s)
        t = np.arange(num_samples) / sample_rate

        # Generate signal
        if signal_type == "sine":
            data = np.sin(2 * np.pi * freq_hz * t)
        elif signal_type == "square":
            data = np.sign(np.sin(2 * np.pi * freq_hz * t))
        else:
            data = self.rng.standard_normal(num_samples)

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            if header:
                f.write("time,voltage\n")
            for time_val, volt_val in zip(t, data, strict=False):
                f.write(f"{time_val:.9e},{volt_val:.6f}\n")

        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (CSV, {num_samples} samples)")

    # =========================================================================
    # VCD FILE GENERATORS (Digital logic simulation format)
    # =========================================================================

    def generate_vcd_digital(
        self,
        output_path: Path,
        signal_name: str = "clk",
        freq_hz: float = 1000.0,
        duration_s: float = 0.001,
    ) -> None:
        """Generate VCD file with digital signal.

        Args:
            output_path: Output VCD file path
            signal_name: Signal name
            freq_hz: Clock frequency in Hz
            duration_s: Duration in seconds
        """
        period_ns = int(1e9 / freq_hz)  # Period in nanoseconds
        num_cycles = int(freq_hz * duration_s)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # VCD header
            f.write("$version\n")
            f.write("  Oscura Test Vector Generator\n")
            f.write("$end\n")
            f.write("$timescale 1ns $end\n")
            f.write("$scope module testbench $end\n")
            f.write(f"$var wire 1 ! {signal_name} $end\n")
            f.write("$upscope $end\n")
            f.write("$enddefinitions $end\n")
            f.write("$dumpvars\n")
            f.write("0!\n")
            f.write("$end\n")

            # Generate clock transitions
            for cycle in range(num_cycles):
                time_ns = cycle * period_ns
                f.write(f"#{time_ns}\n")
                f.write("1!\n")
                f.write(f"#{time_ns + period_ns // 2}\n")
                f.write("0!\n")

        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (VCD, {num_cycles} cycles)")

    # =========================================================================
    # PROTOCOL TEST VECTORS (Binary protocol data)
    # =========================================================================

    def generate_uart_test_vector(
        self,
        output_path: Path,
        data: bytes = b"UART Test",
        baud_rate: int = 115200,
    ) -> None:
        """Generate UART protocol test vector (binary file).

        Args:
            output_path: Output file path
            data: Data to encode as UART
            baud_rate: UART baud rate
        """
        # Simple UART encoding: start bit (0), 8 data bits, stop bit (1)
        # This is a simplified representation
        encoded = bytearray()
        for byte in data:
            encoded.append(0)  # Start bit
            for bit in range(8):
                encoded.append((byte >> bit) & 1)
            encoded.append(1)  # Stop bit

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(encoded))
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (UART, {len(data)} bytes @ {baud_rate} baud)")

    def generate_spi_test_vector(
        self,
        output_path: Path,
        mosi_data: bytes = b"\x01\x02\x03\x04",
        miso_data: bytes = b"\xff\xfe\xfd\xfc",
    ) -> None:
        """Generate SPI protocol test vector (NPZ with MOSI/MISO).

        Args:
            output_path: Output NPZ file path
            mosi_data: Master-out-slave-in data
            miso_data: Master-in-slave-out data
        """
        # Store as NPZ with MOSI and MISO arrays
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(output_path),
            mosi=np.frombuffer(mosi_data, dtype=np.uint8),
            miso=np.frombuffer(miso_data, dtype=np.uint8),
            metadata=np.array([json.dumps({"protocol": "SPI", "bytes": len(mosi_data)})]),
        )
        self.generated_files.append(str(output_path))
        print(f"✓ Generated: {output_path.name} (SPI, {len(mosi_data)} bytes)")

    # =========================================================================
    # MAIN GENERATION SUITE
    # =========================================================================

    def generate_all_test_vectors(self) -> None:
        """Generate all test vectors to eliminate missing data skips."""
        print("=" * 70)
        print("Generating Test Vectors for Oscura Test Suite")
        print("=" * 70)
        print()

        # 1. WFM Files (Tektronix oscilloscope format)
        print("📊 WFM Files (Tektronix Oscilloscope Format)")
        print("-" * 70)
        self.generate_wfm_sine_wave(
            self.base_dir / "formats/tektronix/analog/sine_1khz_basic.wfm",
            freq_hz=1000,
            amplitude=1.0,
        )
        self.generate_wfm_sine_wave(
            self.base_dir / "formats/tektronix/analog/sine_10khz.wfm", freq_hz=10000, amplitude=2.0
        )
        self.generate_wfm_square_wave(
            self.base_dir / "formats/tektronix/digital/square_1khz_basic.wfm",
            freq_hz=1000,
            duty_cycle=0.5,
        )
        self.generate_wfm_square_wave(
            self.base_dir / "formats/tektronix/digital/pwm_1khz_25pct.wfm",
            freq_hz=1000,
            duty_cycle=0.25,
        )
        self.generate_wfm_noisy_signal(
            self.base_dir / "formats/tektronix/analog/noisy_sine_20db.wfm", snr_db=20.0
        )
        self.generate_wfm_noisy_signal(
            self.base_dir / "formats/tektronix/analog/noisy_sine_10db.wfm", snr_db=10.0
        )
        self.generate_wfm_multi_channel(
            self.base_dir / "formats/tektronix/multi_channel/quad_channel.wfm", num_channels=4
        )
        print()

        # 2. PCAP Files (Network packet captures)
        print("📦 PCAP Files (Network Packet Captures)")
        print("-" * 70)
        self.generate_pcap_tcp_simple(
            self.base_dir / "formats/pcap/tcp/simple_tcp.pcap", payload=b"TCP Test Data"
        )
        self.generate_pcap_udp_simple(
            self.base_dir / "formats/pcap/udp/simple_udp.pcap", payload=b"UDP Test Data"
        )
        self.generate_pcap_http_request(
            self.base_dir / "formats/pcap/tcp/http/http_get.pcap", method="GET"
        )
        self.generate_pcap_http_request(
            self.base_dir / "formats/pcap/tcp/http/http_post.pcap", method="POST", path="/api/data"
        )
        self.generate_pcap_malformed(
            self.base_dir / "formats/pcap/malformed/truncated.pcap", corruption_type="truncated"
        )
        self.generate_pcap_malformed(
            self.base_dir / "formats/pcap/malformed/invalid_checksum.pcap",
            corruption_type="invalid_checksum",
        )
        print()

        # 3. NPZ Files (NumPy compressed signal data)
        print("💾 NPZ Files (NumPy Compressed Signals)")
        print("-" * 70)
        self.generate_npz_signal(
            self.base_dir / "synthetic/waveforms/npz/sine_1khz.npz", signal_type="sine"
        )
        self.generate_npz_signal(
            self.base_dir / "synthetic/waveforms/npz/square_1khz.npz", signal_type="square"
        )
        self.generate_npz_signal(
            self.base_dir / "synthetic/waveforms/npz/sawtooth_500hz.npz",
            signal_type="sawtooth",
            freq_hz=500,
        )
        self.generate_npz_signal(
            self.base_dir / "synthetic/waveforms/npz/white_noise.npz", signal_type="noise"
        )
        print()

        # 4. CSV Files (Waveform data in CSV format)
        print("📝 CSV Files (Comma-Separated Waveform Data)")
        print("-" * 70)
        self.generate_csv_waveform(
            self.base_dir / "formats/csv/sine_1khz.csv", signal_type="sine", freq_hz=1000
        )
        self.generate_csv_waveform(
            self.base_dir / "formats/csv/square_2khz.csv", signal_type="square", freq_hz=2000
        )
        self.generate_csv_waveform(
            self.base_dir / "formats/csv/no_header.csv", signal_type="sine", header=False
        )
        print()

        # 5. VCD Files (Digital logic simulation format)
        print("🔌 VCD Files (Digital Logic Simulation)")
        print("-" * 70)
        self.generate_vcd_digital(
            self.base_dir / "formats/vcd/clock_1khz.vcd", signal_name="clk", freq_hz=1000
        )
        self.generate_vcd_digital(
            self.base_dir / "formats/vcd/clock_10mhz.vcd", signal_name="clk", freq_hz=10e6
        )
        print()

        # 6. Protocol Test Vectors (Binary protocol data)
        print("🔧 Protocol Test Vectors")
        print("-" * 70)
        self.generate_uart_test_vector(
            self.base_dir / "synthetic/protocols/uart_test.bin", data=b"Hello UART"
        )
        self.generate_spi_test_vector(
            self.base_dir / "synthetic/protocols/spi_test.npz",
            mosi_data=b"\xaa\xbb\xcc\xdd",
            miso_data=b"\x11\x22\x33\x44",
        )
        print()

        # Summary
        print("=" * 70)
        print(f"✅ Successfully generated {len(self.generated_files)} test files")
        print("=" * 70)
        print()
        print("Generated files:")
        for filepath in sorted(self.generated_files):
            print(f"  • {filepath}")
        print()
        print("Next steps:")
        print("  1. Run tests: ./scripts/test.sh")
        print("  2. Verify skip count reduced: 560 → 540 skips")
        print("  3. Update CHANGELOG.md with results")
        print()


def main() -> None:
    """Main entry point."""
    import sys

    # Determine base directory
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("test_data")

    # Check dependencies
    missing_deps = []
    if not HAS_TM_DATA_TYPES:
        missing_deps.append("tm_data_types (for WFM files)")
    if not HAS_SCAPY:
        missing_deps.append("scapy (for PCAP files)")

    if missing_deps:
        print("⚠️  Warning: Some optional dependencies are missing:")
        for dep in missing_deps:
            print(f"    • {dep}")
        print()
        print("Install with:")
        print("  uv pip install tm-data-types scapy")
        print()
        print("Continuing with available generators...")
        print()

    # Generate test vectors
    generator = TestVectorGenerator(base_dir=base_dir)
    generator.generate_all_test_vectors()


if __name__ == "__main__":
    main()
