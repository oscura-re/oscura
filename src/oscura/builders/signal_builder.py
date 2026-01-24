"""Fluent SignalBuilder for composable signal generation.

This module provides the main SignalBuilder class that allows fluent
composition of signals for test data generation, demos, and protocol testing.

Example:
    >>> from oscura import SignalBuilder
    >>> trace = (SignalBuilder()
    ...     .sample_rate(10e6)
    ...     .duration(0.01)
    ...     .add_sine(frequency=1000, amplitude=1.0)
    ...     .add_noise(snr_db=40)
    ...     .build())

The builder supports:
- Analog waveforms (sine, square, triangle, chirp, multitone)
- Protocol signals (UART, SPI, I2C, CAN)
- Noise and impairments (gaussian, pink, jitter, quantization)
- Multi-channel signals (use build_channels() to get all channels)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import signal as scipy_signal

from oscura.core.types import TraceMetadata, WaveformTrace


class SignalBuilder:
    """Fluent builder for composable signal generation.

    This class provides a chainable API for building complex test signals
    by combining basic waveforms, protocol signals, noise, and impairments.

    Example:
        >>> # Simple sine wave with noise
        >>> trace = (SignalBuilder()
        ...     .sample_rate(1e6)
        ...     .duration(0.01)
        ...     .add_sine(frequency=1000, amplitude=1.0)
        ...     .add_noise(snr_db=40)
        ...     .build())
        >>>
        >>> # UART signal with realistic characteristics
        >>> uart_trace = (SignalBuilder()
        ...     .sample_rate(10e6)
        ...     .add_uart(baud_rate=115200, data=b"Hello Oscura!", config="8N1")
        ...     .add_noise(snr_db=30)
        ...     .build())
    """

    def __init__(self, sample_rate: float = 1e6, duration: float = 0.01):
        """Initialize builder with default parameters.

        Args:
            sample_rate: Sample rate in Hz (default 1 MHz).
            duration: Signal duration in seconds (default 10 ms).
        """
        self._sample_rate = sample_rate
        self._duration = duration
        self._channels: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        self._description = ""
        self._parameters: dict[str, Any] = {}

    # ========== Configuration Methods ==========

    def sample_rate(self, rate: float) -> SignalBuilder:
        """Set sample rate in Hz.

        Args:
            rate: Sample rate in Hz.

        Returns:
            Self for chaining.
        """
        self._sample_rate = rate
        return self

    def duration(self, seconds: float) -> SignalBuilder:
        """Set signal duration.

        Args:
            seconds: Duration in seconds.

        Returns:
            Self for chaining.
        """
        self._duration = seconds
        return self

    def description(self, desc: str) -> SignalBuilder:
        """Set signal description.

        Args:
            desc: Human-readable description.

        Returns:
            Self for chaining.
        """
        self._description = desc
        return self

    # ========== Analog Signal Methods ==========

    def add_sine(
        self,
        frequency: float = 1e3,
        amplitude: float = 1.0,
        phase: float = 0.0,
        dc_offset: float = 0.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add sinusoidal component.

        Args:
            frequency: Signal frequency in Hz.
            amplitude: Signal amplitude.
            phase: Phase offset in radians.
            dc_offset: DC offset.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset

        self._add_to_channel(channel, signal)
        self._parameters[f"{channel}_sine_freq"] = frequency
        return self

    def add_harmonics(
        self,
        fundamental: float = 1e3,
        thd_percent: float = 1.0,
        harmonics: list[tuple[int, float]] | None = None,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add harmonic distortion.

        Args:
            fundamental: Fundamental frequency in Hz.
            thd_percent: Total harmonic distortion percentage (if harmonics not specified).
            harmonics: List of (harmonic_number, relative_amplitude) tuples.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()

        if harmonics is None:
            # Generate typical harmonic profile
            thd_linear = thd_percent / 100
            harmonics = [
                (2, thd_linear * 0.7),  # 2nd harmonic
                (3, thd_linear * 0.5),  # 3rd harmonic
                (4, thd_linear * 0.3),  # 4th harmonic
                (5, thd_linear * 0.2),  # 5th harmonic
            ]

        signal = np.zeros_like(t)
        for harm_num, rel_amp in harmonics:
            signal += rel_amp * np.sin(2 * np.pi * harm_num * fundamental * t)

        self._add_to_channel(channel, signal)
        return self

    def add_square(
        self,
        frequency: float = 1e3,
        amplitude: float = 1.0,
        duty_cycle: float = 0.5,
        rise_time: float | None = None,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add square wave with optional edge rate.

        Args:
            frequency: Signal frequency in Hz.
            amplitude: Signal amplitude.
            duty_cycle: Duty cycle 0-1 (default 0.5 = 50%).
            rise_time: Rise time in seconds (None for ideal edges).
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()
        signal = amplitude * scipy_signal.square(2 * np.pi * frequency * t, duty=duty_cycle)

        # Apply finite rise time if specified
        if rise_time is not None and rise_time > 0:
            tau = rise_time / 2.2  # 10-90% rise time
            alpha = 1 / (tau * self._sample_rate + 1)
            filtered = np.zeros_like(signal)
            filtered[0] = signal[0]
            for i in range(1, len(signal)):
                filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i - 1]
            signal = filtered

        self._add_to_channel(channel, signal)
        self._parameters[f"{channel}_square_freq"] = frequency
        return self

    def add_triangle(
        self,
        frequency: float = 1e3,
        amplitude: float = 1.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add triangle wave.

        Args:
            frequency: Signal frequency in Hz.
            amplitude: Signal amplitude.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()
        signal = amplitude * scipy_signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
        self._add_to_channel(channel, signal)
        return self

    def add_sawtooth(
        self,
        frequency: float = 1e3,
        amplitude: float = 1.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add sawtooth wave.

        Args:
            frequency: Signal frequency in Hz.
            amplitude: Signal amplitude.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()
        signal = amplitude * scipy_signal.sawtooth(2 * np.pi * frequency * t)
        self._add_to_channel(channel, signal)
        return self

    def add_chirp(
        self,
        f0: float = 1e3,
        f1: float = 10e3,
        method: Literal["linear", "quadratic", "logarithmic"] = "linear",
        amplitude: float = 1.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add chirp (frequency sweep) signal.

        Args:
            f0: Starting frequency in Hz.
            f1: Ending frequency in Hz.
            method: Sweep type.
            amplitude: Signal amplitude.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()
        signal = amplitude * scipy_signal.chirp(t, f0, self._duration, f1, method=method)
        self._add_to_channel(channel, signal)
        return self

    def add_multitone(
        self,
        frequencies: list[float],
        amplitudes: list[float] | None = None,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add multi-tone signal.

        Args:
            frequencies: List of frequencies in Hz.
            amplitudes: List of amplitudes (default: all 1.0).
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()
        if amplitudes is None:
            amplitudes = [1.0] * len(frequencies)

        signal = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes, strict=False):
            signal += amp * np.sin(2 * np.pi * freq * t)

        self._add_to_channel(channel, signal)
        return self

    def add_dc(
        self,
        level: float = 1.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add DC level.

        Args:
            level: DC voltage level.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        n_samples = self._get_num_samples()
        signal = np.full(n_samples, level)
        self._add_to_channel(channel, signal)
        return self

    def add_pulse(
        self,
        width: float = 1e-6,
        amplitude: float = 1.0,
        delay: float = 0.0,
        rise_time: float = 0.0,
        fall_time: float = 0.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add single pulse.

        Args:
            width: Pulse width in seconds.
            amplitude: Pulse amplitude.
            delay: Delay before pulse in seconds.
            rise_time: Rise time in seconds.
            fall_time: Fall time in seconds.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        t = self._get_time()
        signal = np.zeros_like(t)

        start_idx = int(delay * self._sample_rate)
        width_samples = int(width * self._sample_rate)
        end_idx = min(start_idx + width_samples, len(signal))

        if start_idx < len(signal):
            signal[start_idx:end_idx] = amplitude

        # Apply rise/fall times with simple filtering
        if rise_time > 0 or fall_time > 0:
            tau = max(rise_time, fall_time) / 2.2
            if tau > 0:
                alpha = 1 / (tau * self._sample_rate + 1)
                filtered = np.zeros_like(signal)
                filtered[0] = signal[0]
                for i in range(1, len(signal)):
                    filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i - 1]
                signal = filtered

        self._add_to_channel(channel, signal)
        return self

    # ========== Noise Methods ==========

    def add_noise(
        self,
        snr_db: float = 40.0,
        noise_type: Literal["gaussian", "white", "pink"] = "gaussian",
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add noise at specified SNR.

        Args:
            snr_db: Target signal-to-noise ratio in dB.
            noise_type: Type of noise.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        if channel not in self._channels:
            raise ValueError(f"Channel '{channel}' does not exist. Add a signal first.")

        signal = self._channels[channel]
        signal_power = np.mean(signal**2)

        if signal_power == 0:
            signal_power = 1.0

        noise_power = signal_power / (10 ** (snr_db / 10))
        n_samples = len(signal)

        if noise_type in ["gaussian", "white"]:
            noise = np.sqrt(noise_power) * np.random.randn(n_samples)
        elif noise_type == "pink":
            white = np.random.randn(n_samples)
            fft_white = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n_samples)
            freqs[0] = 1  # Avoid division by zero
            pink_filter = 1 / np.sqrt(freqs)
            fft_pink = fft_white * pink_filter
            noise = np.fft.irfft(fft_pink, n=n_samples)
            noise = noise * np.sqrt(noise_power / np.mean(noise**2))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        self._channels[channel] = signal + noise
        self._parameters[f"{channel}_snr_db"] = snr_db
        return self

    def add_white_noise(
        self,
        amplitude: float = 1.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add white noise signal.

        Args:
            amplitude: Noise amplitude (standard deviation).
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        n_samples = self._get_num_samples()
        noise = amplitude * np.random.randn(n_samples)
        self._add_to_channel(channel, noise)
        return self

    # ========== Protocol Signal Methods ==========

    def add_uart(
        self,
        baud_rate: int = 115200,
        data: bytes = b"Hello",
        config: str = "8N1",
        amplitude: float = 3.3,
        idle_high: bool = True,
        channel: str = "uart",
    ) -> SignalBuilder:
        """Add UART transmission signal.

        Args:
            baud_rate: UART baud rate.
            data: Data bytes to transmit.
            config: Configuration string "XYZ" where X=data bits, Y=parity, Z=stop bits.
            amplitude: Logic high voltage level.
            idle_high: If True, idle state is high (standard UART).
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        # Parse config
        data_bits = int(config[0])
        parity = config[1].upper()  # N, O, E
        stop_bits = int(config[2])

        samples_per_bit = int(self._sample_rate / baud_rate)
        bits: list[int] = []

        # Add initial idle
        idle_level = 1 if idle_high else 0
        bits.extend([idle_level] * (samples_per_bit * 2))

        for byte_val in data:
            # Start bit (opposite of idle)
            bits.extend([1 - idle_level] * samples_per_bit)

            # Data bits (LSB first)
            ones_count = 0
            for i in range(data_bits):
                bit = (byte_val >> i) & 1
                ones_count += bit
                bits.extend([bit] * samples_per_bit)

            # Parity bit
            if parity == "O":  # Odd parity
                parity_bit = (ones_count + 1) % 2
                bits.extend([parity_bit] * samples_per_bit)
            elif parity == "E":  # Even parity
                parity_bit = ones_count % 2
                bits.extend([parity_bit] * samples_per_bit)
            # N = no parity

            # Stop bits
            bits.extend([idle_level] * (samples_per_bit * stop_bits))

            # Inter-frame gap
            bits.extend([idle_level] * samples_per_bit)

        # Final idle
        bits.extend([idle_level] * (samples_per_bit * 2))

        signal = np.array(bits, dtype=np.float64) * amplitude
        self._add_to_channel(channel, signal)
        self._parameters["uart_baud_rate"] = baud_rate
        self._parameters["uart_data"] = data.hex()
        return self

    def add_spi(
        self,
        clock_freq: float = 1e6,
        mode: int = 0,
        data_mosi: bytes = b"\x00",
        data_miso: bytes | None = None,
        amplitude: float = 3.3,
        channels: tuple[str, str, str, str] = ("sck", "mosi", "miso", "cs"),
    ) -> SignalBuilder:
        """Add SPI transaction signals.

        Args:
            clock_freq: SPI clock frequency in Hz.
            mode: SPI mode (0-3, combines CPOL and CPHA).
            data_mosi: MOSI data bytes.
            data_miso: MISO data bytes (default: same as MOSI).
            amplitude: Logic high voltage level.
            channels: Tuple of channel names (SCK, MOSI, MISO, CS).

        Returns:
            Self for chaining.
        """
        if data_miso is None:
            data_miso = data_mosi

        cpol = (mode >> 1) & 1
        cpha = mode & 1

        samples_per_half_clock = int(self._sample_rate / clock_freq / 2)
        total_bits = len(data_mosi) * 8
        total_samples = samples_per_half_clock * 2 * total_bits + samples_per_half_clock * 4

        # Initialize signals
        sck = np.full(total_samples, amplitude if cpol else 0.0)
        mosi = np.zeros(total_samples)
        miso = np.zeros(total_samples)
        cs = np.full(total_samples, amplitude)  # Active low

        idx = samples_per_half_clock  # Start after idle
        cs[idx:] = 0.0  # Activate CS

        for byte_idx in range(len(data_mosi)):
            mosi_byte = data_mosi[byte_idx]
            miso_byte = data_miso[byte_idx] if byte_idx < len(data_miso) else 0

            for bit_idx in range(8):
                mosi_bit = (mosi_byte >> (7 - bit_idx)) & 1
                miso_bit = (miso_byte >> (7 - bit_idx)) & 1

                if cpha == 0:
                    mosi[idx : idx + samples_per_half_clock * 2] = amplitude if mosi_bit else 0.0
                    miso[idx : idx + samples_per_half_clock * 2] = amplitude if miso_bit else 0.0

                # Clock edge
                if cpol == 0:
                    sck[idx + samples_per_half_clock : idx + samples_per_half_clock * 2] = amplitude
                else:
                    sck[idx + samples_per_half_clock : idx + samples_per_half_clock * 2] = 0.0

                if cpha == 1:
                    mosi[idx + samples_per_half_clock : idx + samples_per_half_clock * 2] = (
                        amplitude if mosi_bit else 0.0
                    )
                    miso[idx + samples_per_half_clock : idx + samples_per_half_clock * 2] = (
                        amplitude if miso_bit else 0.0
                    )

                idx += samples_per_half_clock * 2

        cs[idx:] = amplitude  # Deactivate CS

        self._channels[channels[0]] = sck
        self._channels[channels[1]] = mosi
        self._channels[channels[2]] = miso
        self._channels[channels[3]] = cs

        self._parameters["spi_clock_freq"] = clock_freq
        self._parameters["spi_mode"] = mode
        return self

    def add_i2c(
        self,
        clock_freq: float = 100e3,
        address: int = 0x50,
        data: bytes = b"\x00",
        read: bool = False,
        amplitude: float = 3.3,
        channels: tuple[str, str] = ("scl", "sda"),
    ) -> SignalBuilder:
        """Add I2C transaction signals.

        Args:
            clock_freq: I2C clock frequency in Hz.
            address: 7-bit I2C address.
            data: Data bytes to transmit.
            read: True for read, False for write.
            amplitude: Logic high voltage level.
            channels: Tuple of channel names (SCL, SDA).

        Returns:
            Self for chaining.
        """
        samples_per_bit = int(self._sample_rate / clock_freq)
        half_bit = samples_per_bit // 2

        # Calculate total samples
        total_bits = 1 + 8 + 1 + len(data) * 9 + 1
        total_samples = samples_per_bit * total_bits + samples_per_bit * 2

        scl = np.full(total_samples, amplitude)
        sda = np.full(total_samples, amplitude)

        idx = samples_per_bit  # Start after idle

        # START: SDA falls while SCL high
        sda[idx : idx + half_bit] = 0.0
        idx += half_bit

        # Address + R/W bit
        addr_byte = (address << 1) | (1 if read else 0)

        for bit_idx in range(8):
            bit = (addr_byte >> (7 - bit_idx)) & 1
            scl[idx : idx + half_bit] = 0.0
            sda[idx : idx + samples_per_bit] = amplitude if bit else 0.0
            idx += half_bit
            scl[idx : idx + half_bit] = amplitude
            idx += half_bit

        # ACK bit
        scl[idx : idx + half_bit] = 0.0
        sda[idx : idx + samples_per_bit] = 0.0  # ACK (low)
        idx += half_bit
        scl[idx : idx + half_bit] = amplitude
        idx += half_bit

        # Data bytes
        for byte_val in data:
            for bit_idx in range(8):
                bit = (byte_val >> (7 - bit_idx)) & 1
                scl[idx : idx + half_bit] = 0.0
                sda[idx : idx + samples_per_bit] = amplitude if bit else 0.0
                idx += half_bit
                scl[idx : idx + half_bit] = amplitude
                idx += half_bit

            # ACK
            scl[idx : idx + half_bit] = 0.0
            sda[idx : idx + samples_per_bit] = 0.0
            idx += half_bit
            scl[idx : idx + half_bit] = amplitude
            idx += half_bit

        # STOP: SDA rises while SCL high
        scl[idx : idx + half_bit] = 0.0
        sda[idx : idx + half_bit] = 0.0
        idx += half_bit
        scl[idx:] = amplitude
        sda[idx:] = amplitude

        self._channels[channels[0]] = scl[:idx]
        self._channels[channels[1]] = sda[:idx]

        self._parameters["i2c_clock_freq"] = clock_freq
        self._parameters["i2c_address"] = address
        return self

    def add_can(
        self,
        bitrate: int = 500000,
        arbitration_id: int = 0x100,
        data: bytes = b"\x00",
        extended: bool = False,
        amplitude: float = 2.5,
        channel: str = "can",
    ) -> SignalBuilder:
        """Add CAN message signal.

        Args:
            bitrate: CAN bit rate.
            arbitration_id: Message arbitration ID.
            data: Data bytes (max 8).
            extended: True for extended (29-bit) ID.
            amplitude: Logic high voltage level.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        samples_per_bit = int(self._sample_rate / bitrate)
        bits: list[int] = []

        # Start of frame (dominant = 0)
        bits.append(0)

        # Arbitration ID
        id_bits = 29 if extended else 11
        for i in range(id_bits - 1, -1, -1):
            bits.append((arbitration_id >> i) & 1)

        # RTR (0 for data frame)
        bits.append(0)

        # IDE (0 for standard, 1 for extended)
        if not extended:
            bits.append(0)

        # Reserved bit
        bits.append(0)

        # DLC (data length code)
        dlc = min(len(data), 8)
        for i in range(3, -1, -1):
            bits.append((dlc >> i) & 1)

        # Data bytes
        for byte_val in data[:8]:
            for i in range(7, -1, -1):
                bits.append((byte_val >> i) & 1)

        # CRC (simplified placeholder)
        for _ in range(15):
            bits.append(0)

        # CRC delimiter
        bits.append(1)

        # ACK slot and delimiter
        bits.append(0)  # ACK
        bits.append(1)  # ACK delimiter

        # End of frame (7 recessive bits)
        bits.extend([1] * 7)

        # Inter-frame space
        bits.extend([1] * 3)

        # Convert to signal
        signal_bits: list[float] = []
        for bit in bits:
            level = amplitude if bit else 0.0
            signal_bits.extend([level] * samples_per_bit)

        signal = np.array(signal_bits, dtype=np.float64)
        self._add_to_channel(channel, signal)
        self._parameters["can_bitrate"] = bitrate
        self._parameters["can_id"] = arbitration_id
        return self

    def add_digital_pattern(
        self,
        pattern: str = "01010101",
        bit_rate: float = 1e6,
        amplitude: float = 3.3,
        channel: str = "digital",
    ) -> SignalBuilder:
        """Add digital bit pattern.

        Args:
            pattern: Binary pattern string (e.g., "01010101").
            bit_rate: Bit rate in bps.
            amplitude: Logic high voltage level.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        samples_per_bit = int(self._sample_rate / bit_rate)
        bits: list[float] = []

        for bit_char in pattern:
            level = amplitude if bit_char == "1" else 0.0
            bits.extend([level] * samples_per_bit)

        signal = np.array(bits, dtype=np.float64)
        self._add_to_channel(channel, signal)
        return self

    def add_clock(
        self,
        frequency: float = 1e6,
        duty_cycle: float = 0.5,
        amplitude: float = 3.3,
        channel: str = "clk",
    ) -> SignalBuilder:
        """Add clock signal.

        Args:
            frequency: Clock frequency in Hz.
            duty_cycle: Duty cycle (0-1).
            amplitude: Logic high voltage level.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        return self.add_square(
            frequency=frequency,
            amplitude=amplitude,
            duty_cycle=duty_cycle,
            channel=channel,
        )

    # ========== Impairment Methods ==========

    def add_jitter(
        self,
        rj_rms: float = 0.0,
        dj_pp: float = 0.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Add jitter to digital signal.

        Args:
            rj_rms: Random jitter RMS in seconds.
            dj_pp: Deterministic jitter peak-to-peak in seconds.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        if channel not in self._channels:
            raise ValueError(f"Channel '{channel}' does not exist.")

        if rj_rms == 0 and dj_pp == 0:
            return self

        signal = self._channels[channel]
        threshold = (np.max(signal) + np.min(signal)) / 2
        edges = np.where(np.diff((signal > threshold).astype(int)))[0]

        if len(edges) == 0:
            return self

        t_original = np.arange(len(signal)) / self._sample_rate
        t_jittered = t_original.copy()

        for edge_idx in edges:
            jitter = 0.0
            if rj_rms > 0:
                jitter += np.random.randn() * rj_rms
            if dj_pp > 0:
                jitter += (dj_pp / 2) * np.sin(2 * np.pi * edge_idx / max(len(edges), 1))

            edge_region = slice(max(0, edge_idx - 5), min(len(signal), edge_idx + 6))
            t_jittered[edge_region] += jitter

        self._channels[channel] = np.interp(t_original, t_jittered, signal)
        return self

    def add_quantization(
        self,
        bits: int = 8,
        full_scale: float = 2.0,
        channel: str = "ch1",
    ) -> SignalBuilder:
        """Apply ADC quantization effects.

        Args:
            bits: Number of ADC bits.
            full_scale: Full scale range.
            channel: Channel name.

        Returns:
            Self for chaining.
        """
        if channel not in self._channels:
            raise ValueError(f"Channel '{channel}' does not exist.")

        signal = self._channels[channel]
        levels = 2**bits
        lsb = full_scale / levels

        # Quantize
        quantized = np.round(signal / lsb) * lsb
        # Clip to full scale
        quantized = np.clip(quantized, -full_scale / 2, full_scale / 2 - lsb)

        self._channels[channel] = quantized
        self._parameters[f"{channel}_adc_bits"] = bits
        return self

    # ========== Build Methods ==========

    def build(self, channel: str | None = None) -> WaveformTrace:
        """Build and return signal as WaveformTrace.

        This is the primary build method that returns a WaveformTrace ready
        for analysis with Oscura. For multi-channel signals, specify which
        channel to return, or use build_channels() to get all channels.

        Args:
            channel: Channel name to build. If None, uses first channel.
                For single-channel signals, this parameter is optional.

        Returns:
            WaveformTrace ready for Oscura analysis.

        Raises:
            ValueError: If no signals have been added or channel doesn't exist.

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6).add_sine(1000)
            >>> trace = builder.build()
            >>> print(f"Generated {len(trace.data)} samples")

            >>> # Multi-channel
            >>> builder = SignalBuilder(sample_rate=1e6)
            >>> builder.add_sine(1000, channel="sig")
            >>> builder.add_square(500, channel="clk")
            >>> sig_trace = builder.build(channel="sig")
            >>> clk_trace = builder.build(channel="clk")
        """
        if not self._channels:
            raise ValueError("No signals added. Call add_* methods before build().")

        # Ensure all channels have same length (pad if necessary)
        max_len = max(len(s) for s in self._channels.values())
        for name, signal in self._channels.items():
            if len(signal) < max_len:
                self._channels[name] = np.pad(signal, (0, max_len - len(signal)), mode="edge")

        # Determine which channel to return
        if channel is None:
            # Use first channel
            channel = next(iter(self._channels.keys()))
        elif channel not in self._channels:
            available = list(self._channels.keys())
            raise ValueError(f"Channel '{channel}' not found. Available: {available}")

        # Get channel data
        data = self._channels[channel]

        # Build TraceMetadata
        trace_metadata = TraceMetadata(
            sample_rate=self._sample_rate,
            channel_name=channel,
        )

        return WaveformTrace(data=data, metadata=trace_metadata)

    def build_channels(self) -> dict[str, WaveformTrace]:
        """Build and return all channels as dictionary of WaveformTraces.

        Use this method when you need all channels from a multi-channel
        signal generation.

        Returns:
            Dictionary mapping channel names to WaveformTrace objects.

        Raises:
            ValueError: If no signals have been added.

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6)
            >>> builder.add_sine(1000, channel="sig")
            >>> builder.add_square(500, channel="clk")
            >>> channels = builder.build_channels()
            >>> print(f"Generated {len(channels)} channels")
            >>> sig_trace = channels["sig"]
            >>> clk_trace = channels["clk"]
        """
        if not self._channels:
            raise ValueError("No signals added. Call add_* methods before build().")

        # Ensure all channels have same length (pad if necessary)
        max_len = max(len(s) for s in self._channels.values())
        for name, signal in self._channels.items():
            if len(signal) < max_len:
                self._channels[name] = np.pad(signal, (0, max_len - len(signal)), mode="edge")

        # Build WaveformTrace for each channel
        traces: dict[str, WaveformTrace] = {}
        for channel_name, data in self._channels.items():
            trace_metadata = TraceMetadata(
                sample_rate=self._sample_rate,
                channel_name=channel_name,
            )
            traces[channel_name] = WaveformTrace(data=data, metadata=trace_metadata)

        return traces

    def save_npz(self, path: Path | str, channel: str | None = None) -> WaveformTrace:
        """Build and save signal to NPZ format.

        Args:
            path: Output file path.
            channel: Channel to save (for multi-channel signals).

        Returns:
            WaveformTrace that was saved.

        Example:
            >>> builder = SignalBuilder().sample_rate(1e6).add_sine(1000)
            >>> trace = builder.save_npz("signal.npz")
        """
        trace = self.build(channel=channel)

        # Save as NPZ using numpy
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            data=trace.data,
            sample_rate=trace.metadata.sample_rate,
            channel_name=trace.metadata.channel_name or "ch1",
        )

        return trace

    # ========== Internal Methods ==========

    def _get_time(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Get time array based on current settings."""
        n_samples = self._get_num_samples()
        return np.arange(n_samples) / self._sample_rate

    def _get_num_samples(self) -> int:
        """Get number of samples based on current settings."""
        return int(self._sample_rate * self._duration)

    def _add_to_channel(self, channel: str, signal: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        """Add signal to channel, summing if channel already exists."""
        if channel in self._channels:
            current = self._channels[channel]
            if len(signal) > len(current):
                current = np.pad(current, (0, len(signal) - len(current)))
            elif len(signal) < len(current):
                signal = np.pad(signal, (0, len(current) - len(signal)))
            self._channels[channel] = current + signal
        else:
            self._channels[channel] = signal
