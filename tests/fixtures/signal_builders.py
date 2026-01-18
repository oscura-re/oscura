"""Standardized signal generation for all tests.

Replaces 446+ copies of inline signal generators across unit tests.
This is the SINGLE SOURCE OF TRUTH for test signal generation.

Usage:
    from tests.fixtures.signal_builders import SignalBuilder

    # Direct usage
    signal = SignalBuilder.sine_wave(frequency=1e3, sample_rate=1e6)

    # Via pytest fixture
    def test_fft(standard_signals):
        signal = standard_signals["sine_1khz"]
        result = analyze_fft(signal)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal as scipy_signal


class SignalBuilder:
    """Factory for common test signals.

    All methods are static and return numpy arrays containing signal data.
    Methods include sensible defaults for common test scenarios.
    """

    @staticmethod
    def sine_wave(
        frequency: float = 1e3,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        amplitude: float = 1.0,
        phase: float = 0.0,
        noise_snr_db: float | None = None,
    ) -> np.ndarray:
        """Generate sine wave for frequency domain tests.

        Args:
            frequency: Signal frequency in Hz (default 1 kHz)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitude: Signal amplitude (default 1.0)
            phase: Phase offset in radians (default 0.0)
            noise_snr_db: Optional SNR in dB for added noise

        Returns:
            Generated sine wave signal as numpy array
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)

        if noise_snr_db is not None:
            noise_power = amplitude**2 / (2 * 10 ** (noise_snr_db / 10))
            signal += np.sqrt(noise_power) * np.random.randn(len(t))

        return signal

    @staticmethod
    def square_wave(
        frequency: float = 1e3,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        amplitude: float = 1.0,
        duty_cycle: float = 0.5,
    ) -> np.ndarray:
        """Generate square wave for edge detection tests.

        Args:
            frequency: Signal frequency in Hz (default 1 kHz)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitude: Signal amplitude (default 1.0)
            duty_cycle: Duty cycle 0-1 (default 0.5 = 50%)

        Returns:
            Generated square wave signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        return amplitude * scipy_signal.square(2 * np.pi * frequency * t, duty=duty_cycle)

    @staticmethod
    def triangle_wave(
        frequency: float = 1e3,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate triangle wave for slew rate tests.

        Args:
            frequency: Signal frequency in Hz (default 1 kHz)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitude: Signal amplitude (default 1.0)

        Returns:
            Generated triangle wave signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        return amplitude * scipy_signal.sawtooth(2 * np.pi * frequency * t, width=0.5)

    @staticmethod
    def sawtooth_wave(
        frequency: float = 1e3,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate sawtooth wave.

        Args:
            frequency: Signal frequency in Hz (default 1 kHz)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitude: Signal amplitude (default 1.0)

        Returns:
            Generated sawtooth wave signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        return amplitude * scipy_signal.sawtooth(2 * np.pi * frequency * t)

    @staticmethod
    def multitone(
        frequencies: list[float],
        sample_rate: float = 1e6,
        duration: float = 0.01,
        amplitudes: list[float] | None = None,
    ) -> np.ndarray:
        """Generate multitone signal for spectral analysis tests.

        Args:
            frequencies: List of frequencies in Hz
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitudes: Optional list of amplitudes (default: all 1.0)

        Returns:
            Generated multitone signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        if amplitudes is None:
            amplitudes = [1.0] * len(frequencies)

        signal = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes, strict=False):
            signal += amp * np.sin(2 * np.pi * freq * t)

        return signal

    @staticmethod
    def chirp(
        f0: float = 1e3,
        f1: float = 10e3,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        method: Literal["linear", "quadratic", "logarithmic"] = "linear",
    ) -> np.ndarray:
        """Generate chirp signal for frequency response tests.

        Args:
            f0: Starting frequency in Hz (default 1 kHz)
            f1: Ending frequency in Hz (default 10 kHz)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            method: Chirp method - linear, quadratic, or logarithmic

        Returns:
            Generated chirp signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        return scipy_signal.chirp(t, f0, duration, f1, method=method)

    @staticmethod
    def white_noise(
        sample_rate: float = 1e6, duration: float = 0.01, amplitude: float = 1.0
    ) -> np.ndarray:
        """Generate white noise for SNR tests.

        Args:
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitude: Noise amplitude (default 1.0)

        Returns:
            Generated white noise signal
        """
        num_samples = int(duration * sample_rate)
        return amplitude * np.random.randn(num_samples)

    @staticmethod
    def noisy_sine(
        frequency: float = 1e3,
        snr_db: float = 20,
        sample_rate: float = 1e6,
        duration: float = 0.01,
    ) -> np.ndarray:
        """Generate sine wave with calibrated noise level.

        Args:
            frequency: Signal frequency in Hz (default 1 kHz)
            snr_db: Signal-to-noise ratio in dB (default 20 dB)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)

        Returns:
            Generated noisy sine wave
        """
        return SignalBuilder.sine_wave(frequency, sample_rate, duration, noise_snr_db=snr_db)

    @staticmethod
    def pulse_train(
        frequency: float = 1e3,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        pulse_width: float = 0.1,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate pulse train for timing tests.

        Args:
            frequency: Pulse frequency in Hz (default 1 kHz)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            pulse_width: Pulse width as fraction of period (default 0.1 = 10%)
            amplitude: Pulse amplitude (default 1.0)

        Returns:
            Generated pulse train signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        period = 1.0 / frequency
        signal = np.zeros_like(t)

        for i, time in enumerate(t):
            phase = (time % period) / period
            if phase < pulse_width:
                signal[i] = amplitude

        return signal

    @staticmethod
    def digital_pattern(
        pattern: str = "01010101",
        sample_rate: float = 1e6,
        bit_rate: float = 1e6,
        amplitude: float = 3.3,
    ) -> np.ndarray:
        """Generate digital bit pattern for protocol tests.

        Args:
            pattern: Binary pattern string (default "01010101")
            sample_rate: Sampling rate in Hz (default 1 MHz)
            bit_rate: Bit rate in bps (default 1 Mbps)
            amplitude: Logic high level in volts (default 3.3V)

        Returns:
            Generated digital signal
        """
        samples_per_bit = int(sample_rate / bit_rate)
        signal = []

        for bit in pattern:
            level = amplitude if bit == "1" else 0.0
            signal.extend([level] * samples_per_bit)

        return np.array(signal)

    @staticmethod
    def uart_frame(
        data: bytes | int,
        baudrate: int = 115200,
        sample_rate: float = 10e6,
        amplitude: float = 3.3,
        data_bits: int = 8,
        parity: Literal["none", "odd", "even", "mark", "space"] = "none",
        stop_bits: float = 1,
        idle_level: int = 1,
        inject_parity_error: bool = False,
        inject_framing_error: bool = False,
    ) -> np.ndarray:
        """Generate UART frame with configurable parameters.

        Args:
            data: Byte value (0-255) or bytes to transmit
            baudrate: UART baud rate (default 115200)
            sample_rate: Sampling rate in Hz (default 10 MHz)
            amplitude: Logic high level in volts (default 3.3V)
            data_bits: Number of data bits (5-9, default 8)
            parity: Parity mode (default "none")
            stop_bits: Number of stop bits (1, 1.5, 2, default 1)
            idle_level: Idle line level (0 or 1, default 1)
            inject_parity_error: Force incorrect parity bit
            inject_framing_error: Force incorrect stop bit

        Returns:
            Generated UART frame signal with idle before and after
        """
        samples_per_bit = int(sample_rate / baudrate)

        # Convert single int to bytes
        if isinstance(data, int):
            data = bytes([data])

        # Build complete signal with idle periods
        signal = []
        idle_samples = samples_per_bit * 2  # 2 bit periods of idle
        idle_value = amplitude if idle_level else 0.0

        # Initial idle
        signal.extend([idle_value] * idle_samples)

        # Generate frame for each byte
        for data_byte in data:
            frame = []

            # Start bit (opposite of idle)
            frame.append(0 if idle_level else 1)

            # Data bits (LSB first by default)
            data_value = data_byte
            for i in range(data_bits):
                frame.append((data_value >> i) & 1)

            # Parity bit
            if parity != "none":
                ones_count = sum((data_value >> i) & 1 for i in range(data_bits))

                if parity == "odd":
                    parity_bit = (ones_count + 1) % 2
                elif parity == "even":
                    parity_bit = ones_count % 2
                elif parity == "mark":
                    parity_bit = 1
                else:  # space
                    parity_bit = 0

                # Inject parity error if requested
                if inject_parity_error:
                    parity_bit = 1 - parity_bit

                frame.append(parity_bit)

            # Stop bits
            stop_bit_value = idle_level
            if inject_framing_error:
                stop_bit_value = 1 - idle_level

            # Handle fractional stop bits
            if stop_bits == 1:
                frame.append(stop_bit_value)
            elif stop_bits == 1.5:
                frame.extend([stop_bit_value] * 2)  # Will use 1.5x samples_per_bit
            elif stop_bits == 2:
                frame.extend([stop_bit_value, stop_bit_value])

            # Convert bits to voltage levels
            for bit_idx, bit in enumerate(frame):
                level = amplitude if bit else 0.0

                # Handle 1.5 stop bits
                if stop_bits == 1.5 and bit_idx == len(frame) - 1:
                    signal.extend([level] * int(samples_per_bit * 1.5))
                else:
                    signal.extend([level] * samples_per_bit)

        # Final idle
        signal.extend([idle_value] * idle_samples)

        return np.array(signal)

    @staticmethod
    def spi_transaction(
        mosi_data: bytes,
        miso_data: bytes | None = None,
        clock_rate: int = 1_000_000,
        sample_rate: float = 10e6,
        cpol: int = 0,
        cpha: int = 0,
        amplitude: float = 3.3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate SPI transaction signals (CLK, MOSI, MISO, CS).

        Args:
            mosi_data: Data to transmit on MOSI
            miso_data: Data to transmit on MISO (default: same as MOSI)
            clock_rate: SPI clock rate in Hz (default 1 MHz)
            sample_rate: Sampling rate in Hz (default 10 MHz)
            cpol: Clock polarity (0 or 1, default 0)
            cpha: Clock phase (0 or 1, default 0)
            amplitude: Logic high level in volts (default 3.3V)

        Returns:
            Tuple of (CLK, MOSI, MISO, CS) signals as numpy arrays
        """
        if miso_data is None:
            miso_data = mosi_data

        samples_per_bit = int(sample_rate / clock_rate / 2)  # Half clock period

        # Calculate total samples
        total_bits = len(mosi_data) * 8
        total_samples = samples_per_bit * 2 * total_bits + samples_per_bit * 4  # Add idle

        # Initialize arrays
        clk = np.full(total_samples, amplitude if cpol else 0.0)
        mosi = np.zeros(total_samples)
        miso = np.zeros(total_samples)
        cs = np.full(total_samples, amplitude)  # Active low

        idx = samples_per_bit  # Start after idle period
        cs[idx:] = 0.0  # Activate CS

        for byte_idx in range(len(mosi_data)):
            mosi_byte = mosi_data[byte_idx]
            miso_byte = miso_data[byte_idx] if byte_idx < len(miso_data) else 0

            for bit_idx in range(8):
                # Extract bit (MSB first)
                mosi_bit = (mosi_byte >> (7 - bit_idx)) & 1
                miso_bit = (miso_byte >> (7 - bit_idx)) & 1

                # Set data based on CPHA
                if cpha == 0:
                    # Data set before clock edge
                    mosi[idx : idx + samples_per_bit * 2] = amplitude if mosi_bit else 0.0
                    miso[idx : idx + samples_per_bit * 2] = amplitude if miso_bit else 0.0

                # Generate clock pulse
                if cpol == 0:
                    clk[idx + samples_per_bit : idx + samples_per_bit * 2] = amplitude
                else:
                    clk[idx + samples_per_bit : idx + samples_per_bit * 2] = 0.0

                if cpha == 1:
                    # Data set after first edge
                    mosi[idx + samples_per_bit : idx + samples_per_bit * 2] = (
                        amplitude if mosi_bit else 0.0
                    )
                    miso[idx + samples_per_bit : idx + samples_per_bit * 2] = (
                        amplitude if miso_bit else 0.0
                    )

                idx += samples_per_bit * 2

        # Deactivate CS at end
        cs[idx:] = amplitude

        return clk, mosi, miso, cs

    @staticmethod
    def i2c_transaction(
        address: int,
        data: bytes,
        clock_rate: int = 100_000,
        sample_rate: float = 10e6,
        amplitude: float = 3.3,
        inject_nack: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate I2C transaction signals (SCL, SDA).

        Args:
            address: 7-bit I2C address
            data: Data bytes to transmit
            clock_rate: I2C clock rate in Hz (default 100 kHz)
            sample_rate: Sampling rate in Hz (default 10 MHz)
            amplitude: Logic high level in volts (default 3.3V)
            inject_nack: Force NACK instead of ACK

        Returns:
            Tuple of (SCL, SDA) signals as numpy arrays
        """
        samples_per_bit = int(sample_rate / clock_rate)
        half_bit = samples_per_bit // 2

        # Calculate total samples (start + addr + ack + N*data + N*ack + stop)
        total_bits = 1 + 8 + 1 + len(data) * 9 + 1  # Conservative estimate
        total_samples = samples_per_bit * total_bits + samples_per_bit * 2

        # Initialize with idle (both high)
        scl = np.full(total_samples, amplitude)
        sda = np.full(total_samples, amplitude)

        idx = samples_per_bit  # Start after idle

        # START condition: SDA falls while SCL high
        sda[idx : idx + half_bit] = 0.0
        idx += half_bit

        # Address + R/W bit (LSB = 0 for write)
        addr_byte = (address << 1) | 0  # Write mode

        for bit_idx in range(8):
            bit = (addr_byte >> (7 - bit_idx)) & 1

            # Clock low
            scl[idx : idx + half_bit] = 0.0
            # Set data
            sda[idx : idx + samples_per_bit] = amplitude if bit else 0.0

            idx += half_bit

            # Clock high
            scl[idx : idx + half_bit] = amplitude
            idx += half_bit

        # ACK/NACK bit
        scl[idx : idx + half_bit] = 0.0
        if inject_nack:
            sda[idx : idx + samples_per_bit] = amplitude  # NACK (high)
        else:
            sda[idx : idx + samples_per_bit] = 0.0  # ACK (low)
        idx += half_bit
        scl[idx : idx + half_bit] = amplitude
        idx += half_bit

        # Data bytes
        for byte_val in data:
            for bit_idx in range(8):
                bit = (byte_val >> (7 - bit_idx)) & 1

                # Clock low
                scl[idx : idx + half_bit] = 0.0
                sda[idx : idx + samples_per_bit] = amplitude if bit else 0.0
                idx += half_bit

                # Clock high
                scl[idx : idx + half_bit] = amplitude
                idx += half_bit

            # ACK bit
            scl[idx : idx + half_bit] = 0.0
            sda[idx : idx + samples_per_bit] = 0.0  # ACK
            idx += half_bit
            scl[idx : idx + half_bit] = amplitude
            idx += half_bit

        # STOP condition: SDA rises while SCL high
        scl[idx : idx + half_bit] = 0.0
        sda[idx : idx + half_bit] = 0.0
        idx += half_bit
        scl[idx:] = amplitude
        sda[idx:] = amplitude

        return scl[:idx], sda[:idx]

    @staticmethod
    def dc_offset(
        offset: float = 1.0, sample_rate: float = 1e6, duration: float = 0.01
    ) -> np.ndarray:
        """Generate DC offset signal for offset tests.

        Args:
            offset: DC offset value (default 1.0)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)

        Returns:
            Generated DC signal
        """
        num_samples = int(duration * sample_rate)
        return np.full(num_samples, offset)

    @staticmethod
    def step_response(
        step_time: float = 0.005,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate step response signal for transient tests.

        Args:
            step_time: Time of step transition in seconds (default 5 ms)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitude: Step amplitude (default 1.0)

        Returns:
            Generated step response signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        return np.where(t >= step_time, amplitude, 0.0)

    @staticmethod
    def exponential_decay(
        time_constant: float = 0.001,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate exponential decay signal for RC circuit tests.

        Args:
            time_constant: Time constant tau in seconds (default 1 ms)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)
            amplitude: Initial amplitude (default 1.0)

        Returns:
            Generated exponential decay signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        return amplitude * np.exp(-t / time_constant)

    @staticmethod
    def am_modulated(
        carrier_freq: float = 10e3,
        modulation_freq: float = 1e3,
        modulation_index: float = 0.5,
        sample_rate: float = 1e6,
        duration: float = 0.01,
    ) -> np.ndarray:
        """Generate AM modulated signal for modulation tests.

        Args:
            carrier_freq: Carrier frequency in Hz (default 10 kHz)
            modulation_freq: Modulation frequency in Hz (default 1 kHz)
            modulation_index: Modulation depth 0-1 (default 0.5)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)

        Returns:
            Generated AM signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulation = 1 + modulation_index * np.sin(2 * np.pi * modulation_freq * t)
        return carrier * modulation

    @staticmethod
    def fm_modulated(
        carrier_freq: float = 10e3,
        modulation_freq: float = 1e3,
        frequency_deviation: float = 2e3,
        sample_rate: float = 1e6,
        duration: float = 0.01,
    ) -> np.ndarray:
        """Generate FM modulated signal for modulation tests.

        Args:
            carrier_freq: Carrier frequency in Hz (default 10 kHz)
            modulation_freq: Modulation frequency in Hz (default 1 kHz)
            frequency_deviation: Peak frequency deviation in Hz (default 2 kHz)
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Duration in seconds (default 10 ms)

        Returns:
            Generated FM signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        # FM: f(t) = A*sin(2*pi*fc*t + (fd/fm)*sin(2*pi*fm*t))
        modulation_phase = (frequency_deviation / modulation_freq) * np.sin(
            2 * np.pi * modulation_freq * t
        )
        return np.sin(2 * np.pi * carrier_freq * t + modulation_phase)
