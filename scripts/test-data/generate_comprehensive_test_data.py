#!/usr/bin/env python3
"""Generate COMPREHENSIVE test data for robust Analysis Report validation.

This script generates test data covering ALL variations and edge cases that
real-world captures may exhibit, including:

- Clean and noisy signals
- Anomalies, glitches, and corruption
- Edge cases (empty, NaN, extreme values)
- All supported file formats
- All 14 analysis domains
- Protocol errors and recovery
- Various sample rates and sizes

Usage:
    uv run python scripts/generate_comprehensive_test_data.py [output_dir]

This creates a complete validation suite for the analyze() API.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TestCase:
    """Metadata for a test case."""

    file: str
    category: str
    input_type: str
    description: str
    expected_domains: list[str]
    edge_cases: list[str] = field(default_factory=list)
    expected_issues: list[str] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)


class ComprehensiveTestDataGenerator:
    """Generate comprehensive test data for all edge cases."""

    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
        self.test_cases: list[TestCase] = []

    # =========================================================================
    # WAVEFORM GENERATORS - All variations
    # =========================================================================

    def generate_waveform_clean_sine(
        self,
        freq_hz: float = 1000.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
        amplitude: float = 1.0,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Clean sine wave - ideal case."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = amplitude * np.sin(2 * np.pi * freq_hz * t)
        return signal, {
            "type": "clean_sine",
            "frequency_hz": freq_hz,
            "amplitude": amplitude,
            "snr_db": float("inf"),
        }

    def generate_waveform_noisy(
        self,
        freq_hz: float = 1000.0,
        snr_db: float = 20.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Noisy signal with specified SNR."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = np.sin(2 * np.pi * freq_hz * t)
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.standard_normal(len(signal)) * np.sqrt(noise_power)
        return (signal + noise).astype(np.float64), {
            "type": "noisy",
            "frequency_hz": freq_hz,
            "snr_db": snr_db,
        }

    def generate_waveform_very_noisy(
        self, sample_rate: float = 1e6, duration_s: float = 0.01
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Extremely noisy - SNR < 0dB, signal barely visible."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = 0.1 * np.sin(2 * np.pi * 1000 * t)
        noise = self.rng.standard_normal(len(signal))
        return (signal + noise).astype(np.float64), {
            "type": "very_noisy",
            "snr_db": -10.0,
            "expected_issues": ["frequency_detection_unreliable"],
        }

    def generate_waveform_dc(
        self, level: float = 2.5, sample_rate: float = 1e6, num_samples: int = 10000
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Constant DC level - no frequency content."""
        signal = np.full(num_samples, level, dtype=np.float64)
        return signal, {
            "type": "dc",
            "dc_level": level,
            "expected_issues": ["no_frequency", "no_edges", "zero_std"],
        }

    def generate_waveform_dc_with_noise(
        self,
        level: float = 2.5,
        noise_amplitude: float = 0.01,
        sample_rate: float = 1e6,
        num_samples: int = 10000,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """DC with small noise - tests edge detection threshold."""
        signal = np.full(num_samples, level) + noise_amplitude * self.rng.standard_normal(
            num_samples
        )
        return signal.astype(np.float64), {
            "type": "dc_noisy",
            "dc_level": level,
            "noise_amplitude": noise_amplitude,
        }

    def generate_waveform_clipped(
        self,
        freq_hz: float = 1000.0,
        clip_level: float = 0.5,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Clipped/saturated signal - flat tops and bottoms."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = np.sin(2 * np.pi * freq_hz * t)
        signal = np.clip(signal, -clip_level, clip_level)
        return signal.astype(np.float64), {
            "type": "clipped",
            "clip_level": clip_level,
            "expected_issues": ["thd_high", "harmonics_present"],
        }

    def generate_waveform_with_glitches(
        self,
        freq_hz: float = 1000.0,
        glitch_count: int = 5,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Signal with random glitches/spikes."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = np.sin(2 * np.pi * freq_hz * t)

        # Add random glitches
        glitch_positions = self.rng.integers(0, len(signal), glitch_count)
        glitch_amplitudes = self.rng.uniform(2.0, 5.0, glitch_count) * self.rng.choice(
            [-1, 1], glitch_count
        )
        for pos, amp in zip(glitch_positions, glitch_amplitudes, strict=False):
            signal[pos] = amp

        return signal.astype(np.float64), {
            "type": "glitchy",
            "glitch_count": glitch_count,
            "glitch_positions": glitch_positions.tolist(),
        }

    def generate_waveform_with_dropout(
        self,
        freq_hz: float = 1000.0,
        dropout_fraction: float = 0.1,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Signal with missing data (NaN values)."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = np.sin(2 * np.pi * freq_hz * t)

        # Add dropouts as NaN
        dropout_mask = self.rng.random(len(signal)) < dropout_fraction
        signal[dropout_mask] = np.nan

        return signal.astype(np.float64), {
            "type": "dropout",
            "dropout_fraction": dropout_fraction,
            "nan_count": int(np.sum(dropout_mask)),
        }

    def generate_waveform_with_offset_drift(
        self,
        freq_hz: float = 1000.0,
        drift_rate: float = 0.1,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Signal with slow DC drift."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = np.sin(2 * np.pi * freq_hz * t)
        drift = drift_rate * t / duration_s
        return (signal + drift).astype(np.float64), {
            "type": "drifting",
            "drift_rate": drift_rate,
        }

    def generate_waveform_frequency_sweep(
        self,
        f_start: float = 100.0,
        f_end: float = 10000.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Chirp/frequency sweep signal."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        # Linear chirp
        phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * duration_s) * t**2)
        signal = np.sin(phase)
        return signal.astype(np.float64), {
            "type": "chirp",
            "f_start": f_start,
            "f_end": f_end,
            "expected_issues": ["variable_frequency"],
        }

    def generate_waveform_burst(
        self,
        freq_hz: float = 1000.0,
        burst_duty: float = 0.3,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Burst signal - intermittent oscillation."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = np.sin(2 * np.pi * freq_hz * t)

        # Create burst envelope
        burst_period = duration_s / 5
        envelope = ((t % burst_period) / burst_period) < burst_duty
        signal = signal * envelope

        return signal.astype(np.float64), {
            "type": "burst",
            "burst_duty": burst_duty,
        }

    def generate_waveform_multi_frequency(
        self,
        frequencies: list[float] | None = None,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Multi-tone signal for THD/intermodulation testing."""
        if frequencies is None:
            frequencies = [1000.0, 2000.0, 3000.0, 5000.0]

        t = np.arange(0, duration_s, 1 / sample_rate)
        signal = np.zeros_like(t)

        amplitudes = [1.0, 0.5, 0.25, 0.1]
        for freq, amp in zip(frequencies, amplitudes[: len(frequencies)], strict=False):
            signal += amp * np.sin(2 * np.pi * freq * t)

        return signal.astype(np.float64), {
            "type": "multi_tone",
            "frequencies": frequencies,
        }

    def generate_waveform_pwm(
        self,
        pwm_freq: float = 10000.0,
        duty_cycle: float = 0.5,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """PWM signal for digital extraction."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        period = 1 / pwm_freq
        phase = (t % period) / period
        signal = (phase < duty_cycle).astype(np.float64)
        return signal, {
            "type": "pwm",
            "frequency_hz": pwm_freq,
            "duty_cycle": duty_cycle,
        }

    def generate_waveform_ringing(
        self,
        freq_hz: float = 1000.0,
        ring_freq: float = 50000.0,
        ring_decay: float = 0.1,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Square wave with ringing on edges."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        period = 1 / freq_hz

        signal = np.zeros_like(t)
        for i, sample_t in enumerate(t):
            # Base square wave
            phase = (sample_t % period) / period
            base = 1.0 if phase < 0.5 else 0.0

            # Add ringing at transitions
            time_since_edge = min(phase, abs(phase - 0.5)) * period
            if time_since_edge < ring_decay:
                ringing = (
                    0.3
                    * np.exp(-time_since_edge / ring_decay * 10)
                    * np.sin(2 * np.pi * ring_freq * time_since_edge)
                )
                base += ringing

            signal[i] = base

        return signal.astype(np.float64), {
            "type": "ringing",
            "base_freq_hz": freq_hz,
            "ring_freq_hz": ring_freq,
        }

    def generate_waveform_overshoot(
        self,
        freq_hz: float = 1000.0,
        overshoot_pct: float = 20.0,
        sample_rate: float = 1e6,
        duration_s: float = 0.01,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Square wave with overshoot on rising edge."""
        t = np.arange(0, duration_s, 1 / sample_rate)
        period = 1 / freq_hz
        rise_time = period * 0.05

        signal = np.zeros_like(t)
        for i, sample_t in enumerate(t):
            phase = sample_t % period
            if phase < rise_time:
                # Rising with overshoot
                progress = phase / rise_time
                overshoot = overshoot_pct / 100 * np.sin(np.pi * progress)
                signal[i] = progress + overshoot
            elif phase < period / 2:
                signal[i] = 1.0
            elif phase < period / 2 + rise_time:
                # Falling
                progress = (phase - period / 2) / rise_time
                signal[i] = 1.0 - progress
            else:
                signal[i] = 0.0

        return signal.astype(np.float64), {
            "type": "overshoot",
            "overshoot_pct": overshoot_pct,
        }

    # =========================================================================
    # EDGE CASE GENERATORS
    # =========================================================================

    def generate_empty_waveform(self) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Empty array - 0 samples."""
        return np.array([], dtype=np.float64), {
            "type": "empty",
            "expected_issues": ["no_data", "all_analyses_skip"],
        }

    def generate_single_sample(
        self, value: float = 1.0
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Single sample - minimum possible."""
        return np.array([value], dtype=np.float64), {
            "type": "single_sample",
            "expected_issues": ["insufficient_samples"],
        }

    def generate_two_samples(
        self, values: tuple[float, float] = (0.0, 1.0)
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Two samples - edge case for period calculation."""
        return np.array(values, dtype=np.float64), {
            "type": "two_samples",
            "expected_issues": ["one_edge_only"],
        }

    def generate_three_samples(
        self, values: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Three samples - minimum for frequency."""
        return np.array(values, dtype=np.float64), {
            "type": "three_samples",
        }

    def generate_all_nan(self, size: int = 1000) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """All NaN values."""
        return np.full(size, np.nan, dtype=np.float64), {
            "type": "all_nan",
            "expected_issues": ["no_valid_data"],
        }

    def generate_all_inf(self, size: int = 1000) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """All infinite values."""
        signal = np.full(size, np.inf, dtype=np.float64)
        signal[::2] = -np.inf
        return signal, {
            "type": "all_inf",
            "expected_issues": ["infinite_values"],
        }

    def generate_mixed_nan_inf(
        self, size: int = 1000
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Mix of valid, NaN, and Inf values."""
        signal = np.sin(np.linspace(0, 10 * np.pi, size))
        signal[::10] = np.nan
        signal[::17] = np.inf
        signal[::23] = -np.inf
        return signal.astype(np.float64), {
            "type": "mixed_nan_inf",
            "nan_count": size // 10,
        }

    def generate_extreme_small(
        self, size: int = 1000
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Extremely small values near machine epsilon."""
        signal = np.sin(np.linspace(0, 10 * np.pi, size)) * 1e-300
        return signal.astype(np.float64), {
            "type": "extreme_small",
            "amplitude": 1e-300,
        }

    def generate_extreme_large(
        self, size: int = 1000
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Extremely large values."""
        signal = np.sin(np.linspace(0, 10 * np.pi, size)) * 1e100
        return signal.astype(np.float64), {
            "type": "extreme_large",
            "amplitude": 1e100,
        }

    # =========================================================================
    # DIGITAL SIGNAL GENERATORS
    # =========================================================================

    def generate_vcd_clean_uart(
        self,
        baud_rate: int = 115200,
        message: str = "Hello, World!\r\n",
        sample_rate: float = 10e6,
    ) -> tuple[str, dict[str, Any]]:
        """Clean UART VCD."""
        return self._generate_uart_vcd(baud_rate, message, sample_rate, error_rate=0.0)

    def generate_vcd_noisy_uart(
        self,
        baud_rate: int = 115200,
        message: str = "Test",
        sample_rate: float = 10e6,
        error_rate: float = 0.05,
    ) -> tuple[str, dict[str, Any]]:
        """UART with bit errors."""
        return self._generate_uart_vcd(baud_rate, message, sample_rate, error_rate)

    def generate_vcd_framing_errors(
        self,
        baud_rate: int = 115200,
        sample_rate: float = 10e6,
    ) -> tuple[str, dict[str, Any]]:
        """UART with framing errors (wrong stop bits)."""
        # Generate malformed UART with missing stop bits
        samples_per_bit = int(sample_rate / baud_rate)
        transitions = [(0, 1)]  # Start high

        time = 0
        # Send a byte with missing stop bit
        byte_val = 0x55

        # Start bit
        transitions.append((time, 0))
        time += samples_per_bit

        # Data bits
        for bit in range(8):
            transitions.append((time, (byte_val >> bit) & 1))
            time += samples_per_bit

        # Missing stop bit - go low immediately for next start
        transitions.append((time, 0))
        time += samples_per_bit

        vcd = self._transitions_to_vcd(transitions, sample_rate)

        return vcd, {
            "type": "framing_error",
            "baud_rate": baud_rate,
            "expected_issues": ["framing_error", "incomplete_frame"],
        }

    def _generate_uart_vcd(
        self,
        baud_rate: int,
        message: str,
        sample_rate: float,
        error_rate: float = 0.0,
    ) -> tuple[str, dict[str, Any]]:
        """Generate UART VCD with optional errors."""
        samples_per_bit = int(sample_rate / baud_rate)
        transitions = [(0, 1)]  # Start high (idle)

        time = 100  # Small gap at start

        for char in message:
            byte_val = ord(char)

            # Start bit (low)
            transitions.append((time, 0))
            time += samples_per_bit

            # 8 data bits LSB first
            for bit in range(8):
                bit_val = (byte_val >> bit) & 1

                # Inject errors
                if self.rng.random() < error_rate:
                    bit_val = 1 - bit_val

                transitions.append((time, bit_val))
                time += samples_per_bit

            # Stop bit (high)
            transitions.append((time, 1))
            time += samples_per_bit

            # Inter-byte gap
            time += samples_per_bit

        return self._transitions_to_vcd(transitions, sample_rate), {
            "type": "uart",
            "baud_rate": baud_rate,
            "message": message,
            "error_rate": error_rate,
        }

    def _transitions_to_vcd(self, transitions: list[tuple[int, int]], sample_rate: float) -> str:
        """Convert transitions to VCD format."""
        timescale_ns = 1e9 / sample_rate
        lines = [
            "$timescale 1ns $end",
            "$scope module signals $end",
            "$var wire 1 ! data $end",
            "$upscope $end",
            "$enddefinitions $end",
            "#0",
            "1!",  # Initial state high
        ]

        for time, value in transitions:
            lines.append(f"#{int(time * timescale_ns)}")
            lines.append(f"{value}!")

        return "\n".join(lines)

    def generate_vcd_spi(
        self,
        clock_freq: float = 1e6,
        data: bytes = b"\xaa\x55\x00\xff",
        sample_rate: float = 100e6,
    ) -> tuple[str, dict[str, Any]]:
        """SPI clock + data VCD."""
        samples_per_bit = int(sample_rate / clock_freq)
        clk_transitions = [(0, 0)]
        mosi_transitions = [(0, 0)]

        time = 100

        for byte_val in data:
            for bit in range(8):
                bit_val = (byte_val >> (7 - bit)) & 1

                # MOSI changes on falling edge
                mosi_transitions.append((time, bit_val))

                # Clock rising edge
                clk_transitions.append((time + samples_per_bit // 4, 1))

                # Clock falling edge
                clk_transitions.append((time + 3 * samples_per_bit // 4, 0))

                time += samples_per_bit

        # Generate multi-signal VCD
        timescale_ns = 1e9 / sample_rate
        lines = [
            "$timescale 1ns $end",
            "$scope module spi $end",
            "$var wire 1 C clk $end",
            "$var wire 1 D mosi $end",
            "$upscope $end",
            "$enddefinitions $end",
            "#0",
            "0C",
            "0D",
        ]

        # Merge transitions by time
        all_transitions = []
        for t, v in clk_transitions:
            all_transitions.append((t, "C", v))
        for t, v in mosi_transitions:
            all_transitions.append((t, "D", v))
        all_transitions.sort(key=lambda x: x[0])

        for t, sig, v in all_transitions:
            lines.append(f"#{int(t * timescale_ns)}")
            lines.append(f"{v}{sig}")

        return "\n".join(lines), {
            "type": "spi",
            "clock_freq": clock_freq,
            "data_hex": data.hex(),
        }

    # =========================================================================
    # BINARY DATA GENERATORS
    # =========================================================================

    def generate_binary_structured_clean(
        self,
        packet_size: int = 256,
        num_packets: int = 100,
    ) -> tuple[bytes, dict[str, Any]]:
        """Clean structured binary packets."""
        return self._generate_binary_packets(
            packet_size, num_packets, corruption_rate=0.0, gap_rate=0.0
        )

    def generate_binary_corrupted(
        self,
        packet_size: int = 256,
        num_packets: int = 100,
        corruption_rate: float = 0.05,
    ) -> tuple[bytes, dict[str, Any]]:
        """Binary packets with byte corruption."""
        return self._generate_binary_packets(
            packet_size, num_packets, corruption_rate=corruption_rate, gap_rate=0.0
        )

    def generate_binary_with_gaps(
        self,
        packet_size: int = 256,
        num_packets: int = 100,
        gap_rate: float = 0.1,
    ) -> tuple[bytes, dict[str, Any]]:
        """Binary packets with sequence gaps (missing packets)."""
        return self._generate_binary_packets(
            packet_size, num_packets, corruption_rate=0.0, gap_rate=gap_rate
        )

    def generate_binary_bad_checksums(
        self,
        packet_size: int = 256,
        num_packets: int = 100,
        bad_checksum_rate: float = 0.2,
    ) -> tuple[bytes, dict[str, Any]]:
        """Binary packets with invalid checksums."""
        packets, truth = self._generate_binary_packets(
            packet_size, num_packets, corruption_rate=0.0, gap_rate=0.0
        )

        # Corrupt checksums
        data = bytearray(packets)
        bad_checksums = 0
        for i in range(num_packets):
            if self.rng.random() < bad_checksum_rate:
                # Corrupt the last 2 bytes (CRC)
                offset = (i + 1) * packet_size - 2
                data[offset] ^= 0xFF
                bad_checksums += 1

        truth["bad_checksum_count"] = bad_checksums
        truth["expected_issues"] = ["checksum_failures"]

        return bytes(data), truth

    def _generate_binary_packets(
        self,
        packet_size: int,
        num_packets: int,
        corruption_rate: float,
        gap_rate: float,
    ) -> tuple[bytes, dict[str, Any]]:
        """Generate binary packets with optional issues."""
        sync_pattern = b"\xaa\x55\xaa\x55"
        packets = bytearray()
        sequence = 0
        gaps = []

        for i in range(num_packets):
            # Skip some packets to create gaps
            if self.rng.random() < gap_rate and i > 0:
                gaps.append(sequence)
                sequence += 1
                continue

            packet = bytearray()

            # Sync pattern
            packet.extend(sync_pattern)

            # Sequence number (2 bytes)
            packet.extend(struct.pack("<H", sequence))
            sequence += 1

            # Timestamp (4 bytes)
            packet.extend(struct.pack("<I", i * 1000))

            # Length field
            payload_len = packet_size - len(packet) - 4 - 2  # -4 header, -2 CRC
            packet.extend(struct.pack("<H", payload_len))

            # Payload
            for j in range(payload_len):
                packet.append(j & 0xFF)

            # CRC-16
            crc = self._crc16(packet)
            packet.extend(struct.pack("<H", crc))

            # Apply corruption
            if corruption_rate > 0:
                for j in range(len(packet)):
                    if self.rng.random() < corruption_rate:
                        packet[j] ^= self.rng.integers(1, 256)

            packets.extend(packet)

        return bytes(packets), {
            "type": "structured_packets",
            "packet_size": packet_size,
            "num_packets": num_packets - len(gaps),
            "corruption_rate": corruption_rate,
            "sequence_gaps": gaps,
        }

    def _crc16(self, data: bytes) -> int:
        """Calculate CRC-16 CCITT."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
            crc &= 0xFFFF
        return crc

    def generate_binary_random(self, size: int = 65536) -> tuple[bytes, dict[str, Any]]:
        """High entropy random data."""
        return bytes(self.rng.integers(0, 256, size, dtype=np.uint8)), {
            "type": "random",
            "size": size,
            "expected_entropy": 7.99,
            "classification": "random",
        }

    def generate_binary_low_entropy(self, size: int = 65536) -> tuple[bytes, dict[str, Any]]:
        """Low entropy repeating pattern."""
        pattern = b"\x00\x01\x02\x03"
        data = (pattern * (size // len(pattern) + 1))[:size]
        return data, {
            "type": "low_entropy",
            "size": size,
            "expected_entropy": 2.0,
            "classification": "structured",
        }

    def generate_binary_text(self, size: int = 65536) -> tuple[bytes, dict[str, Any]]:
        """English text (medium entropy)."""
        words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "and",
            "runs",
        ]
        text = ""
        while len(text) < size:
            text += " ".join(self.rng.choice(words, 10)) + ". "
        return text[:size].encode("utf-8"), {
            "type": "text",
            "size": size,
            "expected_entropy": 4.5,
            "classification": "text",
        }

    def generate_binary_compressed(self, size: int = 65536) -> tuple[bytes, dict[str, Any]]:
        """Compressed data (high entropy, structured)."""
        original = bytes(range(256)) * (size // 256)
        compressed = zlib.compress(original, level=9)
        # Pad to size
        if len(compressed) < size:
            compressed = compressed + bytes(size - len(compressed))
        return compressed[:size], {
            "type": "compressed",
            "size": size,
            "expected_entropy": 7.8,
            "classification": "compressed",
        }

    def generate_binary_empty(self) -> tuple[bytes, dict[str, Any]]:
        """Empty binary data."""
        return b"", {
            "type": "empty",
            "expected_issues": ["no_data"],
        }

    def generate_binary_single_byte(self) -> tuple[bytes, dict[str, Any]]:
        """Single byte."""
        return b"\x42", {
            "type": "single_byte",
            "expected_issues": ["insufficient_data"],
        }

    def generate_binary_all_zeros(self, size: int = 65536) -> tuple[bytes, dict[str, Any]]:
        """All zeros (minimum entropy)."""
        return bytes(size), {
            "type": "all_zeros",
            "size": size,
            "expected_entropy": 0.0,
            "classification": "constant",
        }

    def generate_binary_all_ones(self, size: int = 65536) -> tuple[bytes, dict[str, Any]]:
        """All 0xFF bytes."""
        return bytes([0xFF] * size), {
            "type": "all_ones",
            "size": size,
            "expected_entropy": 0.0,
            "classification": "constant",
        }

    # =========================================================================
    # FILE SAVING UTILITIES
    # =========================================================================

    def save_npz(
        self,
        path: Path,
        data: NDArray[np.float64],
        sample_rate: float = 1e6,
        **kwargs: Any,
    ) -> None:
        """Save as NPZ with metadata."""
        np.savez(
            path,
            data=data,
            sample_rate=np.array([sample_rate]),
            **{k: np.array([v]) if np.isscalar(v) else v for k, v in kwargs.items()},
        )

    def save_csv(
        self,
        path: Path,
        data: NDArray[np.float64],
        sample_rate: float = 1e6,
        header: bool = True,
    ) -> None:
        """Save as CSV."""
        t = np.arange(len(data)) / sample_rate
        if header:
            np.savetxt(
                path,
                np.column_stack([t, data]),
                delimiter=",",
                header="time,voltage",
                comments="",
            )
        else:
            np.savetxt(path, np.column_stack([t, data]), delimiter=",")

    def save_binary(self, path: Path, data: bytes) -> None:
        """Save binary data."""
        path.write_bytes(data)

    def save_vcd(self, path: Path, vcd_content: str) -> None:
        """Save VCD content."""
        path.write_text(vcd_content)

    def save_test_case(self, path: Path, test_case: TestCase) -> None:
        """Save test case metadata as JSON."""
        with path.open("w") as f:
            json.dump(asdict(test_case), f, indent=2, default=str)


def generate_comprehensive_test_suite(output_dir: Path) -> dict[str, Any]:
    """Generate the complete test suite."""
    gen = ComprehensiveTestDataGenerator(seed=42)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_files": 0,
        "categories": {},
    }

    # =========================================================================
    # 1. WAVEFORM TESTS
    # =========================================================================
    print("Generating waveform test data...")
    waveform_dir = output_dir / "waveform"
    waveform_dir.mkdir(exist_ok=True)
    waveform_count = 0

    # Clean signals
    for freq in [100, 1000, 10000, 100000]:
        signal, truth = gen.generate_waveform_clean_sine(freq_hz=float(freq))
        path = waveform_dir / f"clean_sine_{freq}hz.npz"
        gen.save_npz(path, signal)
        gen.save_test_case(
            path.with_suffix(".json"),
            TestCase(
                file=path.name,
                category="waveform",
                input_type="waveform",
                description=f"Clean {freq}Hz sine wave",
                expected_domains=["waveform", "spectral", "statistics"],
                ground_truth=truth,
            ),
        )
        waveform_count += 1

    # Noisy signals at various SNR
    for snr in [60, 40, 20, 10, 3, 0]:
        signal, truth = gen.generate_waveform_noisy(snr_db=float(snr))
        path = waveform_dir / f"noisy_{snr}db.npz"
        gen.save_npz(path, signal)
        gen.save_test_case(
            path.with_suffix(".json"),
            TestCase(
                file=path.name,
                category="waveform",
                input_type="waveform",
                description=f"Sine wave with {snr}dB SNR",
                expected_domains=["waveform", "spectral", "statistics"],
                edge_cases=["noisy"] if snr < 20 else [],
                ground_truth=truth,
            ),
        )
        waveform_count += 1

    # Very noisy (SNR < 0)
    signal, truth = gen.generate_waveform_very_noisy()
    path = waveform_dir / "very_noisy_negative_snr.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Signal buried in noise (SNR < 0dB)",
            expected_domains=["statistics"],
            edge_cases=["very_noisy", "frequency_unreliable"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # DC signals
    for level in [0.0, 1.0, 2.5, -5.0]:
        signal, truth = gen.generate_waveform_dc(level=level)
        path = waveform_dir / f"dc_{level}v.npz"
        gen.save_npz(path, signal)
        gen.save_test_case(
            path.with_suffix(".json"),
            TestCase(
                file=path.name,
                category="waveform",
                input_type="waveform",
                description=f"DC level at {level}V",
                expected_domains=["statistics"],
                edge_cases=["dc", "no_frequency", "no_edges"],
                expected_issues=["frequency_nan", "period_nan"],
                ground_truth=truth,
            ),
        )
        waveform_count += 1

    # Clipped
    signal, truth = gen.generate_waveform_clipped()
    path = waveform_dir / "clipped.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Clipped/saturated sine wave",
            expected_domains=["waveform", "spectral"],
            edge_cases=["clipped", "high_thd"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Glitchy
    signal, truth = gen.generate_waveform_with_glitches()
    path = waveform_dir / "glitchy.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Sine wave with random glitches",
            expected_domains=["waveform", "spectral", "statistics"],
            edge_cases=["glitches", "outliers"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Dropout (NaN values)
    signal, truth = gen.generate_waveform_with_dropout()
    path = waveform_dir / "dropout.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Signal with missing data (NaN)",
            expected_domains=["statistics"],
            edge_cases=["nan_values", "missing_data"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Drifting
    signal, truth = gen.generate_waveform_with_offset_drift()
    path = waveform_dir / "drifting.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Sine wave with DC drift",
            expected_domains=["waveform", "spectral", "statistics"],
            edge_cases=["dc_drift"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Chirp
    signal, truth = gen.generate_waveform_frequency_sweep()
    path = waveform_dir / "chirp.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Frequency sweep/chirp",
            expected_domains=["spectral"],
            edge_cases=["variable_frequency"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Burst
    signal, truth = gen.generate_waveform_burst()
    path = waveform_dir / "burst.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Burst signal (intermittent)",
            expected_domains=["waveform", "patterns"],
            edge_cases=["burst", "intermittent"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Multi-tone
    signal, truth = gen.generate_waveform_multi_frequency()
    path = waveform_dir / "multi_tone.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Multi-frequency signal for THD",
            expected_domains=["spectral"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # PWM
    signal, truth = gen.generate_waveform_pwm()
    path = waveform_dir / "pwm.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="PWM signal",
            expected_domains=["waveform", "digital", "timing"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Ringing
    signal, truth = gen.generate_waveform_ringing()
    path = waveform_dir / "ringing.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Square wave with ringing",
            expected_domains=["waveform", "spectral"],
            edge_cases=["ringing", "overshoot"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    # Overshoot
    signal, truth = gen.generate_waveform_overshoot()
    path = waveform_dir / "overshoot.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="waveform",
            input_type="waveform",
            description="Square wave with 20% overshoot",
            expected_domains=["waveform"],
            edge_cases=["overshoot"],
            ground_truth=truth,
        ),
    )
    waveform_count += 1

    summary["categories"]["waveform"] = waveform_count

    # =========================================================================
    # 2. EDGE CASE TESTS
    # =========================================================================
    print("Generating edge case test data...")
    edge_dir = output_dir / "edge_cases"
    edge_dir.mkdir(exist_ok=True)
    edge_count = 0

    # Empty
    signal, truth = gen.generate_empty_waveform()
    path = edge_dir / "empty.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="Empty array (0 samples)",
            expected_domains=[],
            edge_cases=["empty", "no_data"],
            expected_issues=["all_analyses_skip"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # Single sample
    signal, truth = gen.generate_single_sample()
    path = edge_dir / "single_sample.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="Single sample",
            expected_domains=["statistics"],
            edge_cases=["single_sample"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # Two samples
    signal, truth = gen.generate_two_samples()
    path = edge_dir / "two_samples.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="Two samples",
            expected_domains=["statistics"],
            edge_cases=["two_samples", "one_edge"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # Three samples
    signal, truth = gen.generate_three_samples()
    path = edge_dir / "three_samples.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="Three samples (minimum for frequency)",
            expected_domains=["waveform", "statistics"],
            edge_cases=["minimum_samples"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # All NaN
    signal, truth = gen.generate_all_nan()
    path = edge_dir / "all_nan.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="All NaN values",
            expected_domains=[],
            edge_cases=["all_nan"],
            expected_issues=["no_valid_data"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # All Inf
    signal, truth = gen.generate_all_inf()
    path = edge_dir / "all_inf.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="All infinite values",
            expected_domains=[],
            edge_cases=["all_inf"],
            expected_issues=["infinite_values"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # Mixed NaN/Inf
    signal, truth = gen.generate_mixed_nan_inf()
    path = edge_dir / "mixed_nan_inf.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="Mix of valid, NaN, and Inf values",
            expected_domains=["statistics"],
            edge_cases=["mixed_nan_inf"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # Extreme small
    signal, truth = gen.generate_extreme_small()
    path = edge_dir / "extreme_small.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="Extremely small values (1e-300)",
            expected_domains=["statistics"],
            edge_cases=["extreme_small", "near_zero"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    # Extreme large
    signal, truth = gen.generate_extreme_large()
    path = edge_dir / "extreme_large.npz"
    gen.save_npz(path, signal)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="edge_case",
            input_type="waveform",
            description="Extremely large values (1e100)",
            expected_domains=["statistics"],
            edge_cases=["extreme_large"],
            ground_truth=truth,
        ),
    )
    edge_count += 1

    summary["categories"]["edge_cases"] = edge_count

    # =========================================================================
    # 3. DIGITAL TESTS
    # =========================================================================
    print("Generating digital test data...")
    digital_dir = output_dir / "digital"
    digital_dir.mkdir(exist_ok=True)
    digital_count = 0

    # Clean UART
    for baud in [9600, 115200, 921600]:
        vcd, truth = gen.generate_vcd_clean_uart(baud_rate=baud)
        path = digital_dir / f"uart_clean_{baud}.vcd"
        gen.save_vcd(path, vcd)
        gen.save_test_case(
            path.with_suffix(".json"),
            TestCase(
                file=path.name,
                category="digital",
                input_type="digital",
                description=f"Clean UART at {baud} baud",
                expected_domains=["digital", "protocols", "timing"],
                ground_truth=truth,
            ),
        )
        digital_count += 1

    # Noisy UART
    vcd, truth = gen.generate_vcd_noisy_uart(error_rate=0.05)
    path = digital_dir / "uart_noisy_5pct.vcd"
    gen.save_vcd(path, vcd)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="digital",
            input_type="digital",
            description="UART with 5% bit errors",
            expected_domains=["digital", "protocols"],
            edge_cases=["bit_errors"],
            ground_truth=truth,
        ),
    )
    digital_count += 1

    # Framing errors
    vcd, truth = gen.generate_vcd_framing_errors()
    path = digital_dir / "uart_framing_error.vcd"
    gen.save_vcd(path, vcd)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="digital",
            input_type="digital",
            description="UART with framing errors",
            expected_domains=["digital"],
            edge_cases=["framing_error"],
            expected_issues=["incomplete_frame"],
            ground_truth=truth,
        ),
    )
    digital_count += 1

    # SPI
    vcd, truth = gen.generate_vcd_spi()
    path = digital_dir / "spi_1mhz.vcd"
    gen.save_vcd(path, vcd)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="digital",
            input_type="digital",
            description="SPI at 1MHz",
            expected_domains=["digital", "protocols"],
            ground_truth=truth,
        ),
    )
    digital_count += 1

    summary["categories"]["digital"] = digital_count

    # =========================================================================
    # 4. BINARY TESTS
    # =========================================================================
    print("Generating binary test data...")
    binary_dir = output_dir / "binary"
    binary_dir.mkdir(exist_ok=True)
    binary_count = 0

    # Clean packets
    data, truth = gen.generate_binary_structured_clean()
    path = binary_dir / "packets_clean.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Clean structured packets",
            expected_domains=["entropy", "patterns", "inference", "packet"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    # Corrupted packets
    for rate in [0.01, 0.05, 0.10]:
        data, truth = gen.generate_binary_corrupted(corruption_rate=rate)
        path = binary_dir / f"packets_corrupted_{int(rate * 100)}pct.bin"
        gen.save_binary(path, data)
        gen.save_test_case(
            path.with_suffix(".json"),
            TestCase(
                file=path.name,
                category="binary",
                input_type="binary",
                description=f"Packets with {int(rate * 100)}% byte corruption",
                expected_domains=["entropy", "patterns"],
                edge_cases=["corruption"],
                ground_truth=truth,
            ),
        )
        binary_count += 1

    # Sequence gaps
    data, truth = gen.generate_binary_with_gaps(gap_rate=0.1)
    path = binary_dir / "packets_with_gaps.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Packets with 10% sequence gaps",
            expected_domains=["entropy", "packet"],
            edge_cases=["sequence_gaps"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    # Bad checksums
    data, truth = gen.generate_binary_bad_checksums()
    path = binary_dir / "packets_bad_checksums.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Packets with 20% invalid checksums",
            expected_domains=["entropy", "statistics"],
            edge_cases=["checksum_failures"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    # Entropy variants
    data, truth = gen.generate_binary_random()
    path = binary_dir / "random_high_entropy.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Random data (high entropy ~8.0)",
            expected_domains=["entropy", "statistics"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    data, truth = gen.generate_binary_low_entropy()
    path = binary_dir / "pattern_low_entropy.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Repeating pattern (low entropy ~2.0)",
            expected_domains=["entropy", "patterns"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    data, truth = gen.generate_binary_text()
    path = binary_dir / "english_text.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="English text (medium entropy ~4.5)",
            expected_domains=["entropy", "statistics"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    data, truth = gen.generate_binary_compressed()
    path = binary_dir / "compressed.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Compressed data (high entropy, structured)",
            expected_domains=["entropy"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    # Edge cases
    data, truth = gen.generate_binary_empty()
    path = binary_dir / "empty.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Empty binary file",
            expected_domains=[],
            edge_cases=["empty"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    data, truth = gen.generate_binary_single_byte()
    path = binary_dir / "single_byte.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="Single byte",
            expected_domains=["statistics"],
            edge_cases=["single_byte"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    data, truth = gen.generate_binary_all_zeros()
    path = binary_dir / "all_zeros.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="All zeros (entropy = 0)",
            expected_domains=["entropy", "statistics"],
            edge_cases=["constant", "zero_entropy"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    data, truth = gen.generate_binary_all_ones()
    path = binary_dir / "all_ones.bin"
    gen.save_binary(path, data)
    gen.save_test_case(
        path.with_suffix(".json"),
        TestCase(
            file=path.name,
            category="binary",
            input_type="binary",
            description="All 0xFF bytes (entropy = 0)",
            expected_domains=["entropy", "statistics"],
            edge_cases=["constant", "zero_entropy"],
            ground_truth=truth,
        ),
    )
    binary_count += 1

    summary["categories"]["binary"] = binary_count

    # =========================================================================
    # SUMMARY
    # =========================================================================
    summary["total_files"] = sum(summary["categories"].values())

    # Save summary
    summary_path = output_dir / "test_suite_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Comprehensive Test Data Generation Complete")
    print(f"{'=' * 60}")
    print(f"Total files: {summary['total_files']}")
    for cat, count in summary["categories"].items():
        print(f"  {cat}: {count} files")
    print(f"\nOutput directory: {output_dir}")

    return summary


def main() -> None:
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("test_data/comprehensive_validation")

    generate_comprehensive_test_suite(output_dir)


if __name__ == "__main__":
    main()
