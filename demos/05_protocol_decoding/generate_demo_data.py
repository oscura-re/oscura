#!/usr/bin/env python3
"""Generate optimal demo data for Protocol Decoding demo.

Creates four protocol capture files showcasing bit-accurate decoding:
1. uart_rs232_communication.wfm - 115200 baud, 8N1, ASCII text + binary + errors
2. spi_flash_transaction.wfm - SPI Mode 0, 10 MHz, flash read command
3. i2c_sensor_bus.wfm - 400 kHz I2C, multi-device (temp sensor, EEPROM, RTC)
4. can_automotive_bus.wfm - CAN 2.0B @ 500 kbps, automotive messages

Usage:
    python generate_demo_data.py [--force]

Author: Oscura Development Team
Date: 2026-01-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{BLUE}INFO:{RESET} {msg}")


def generate_uart_rs232_communication(output_file: Path) -> None:
    """Generate UART RS-232 communication WFM file.

    Protocol: RS-232 UART @ 115200 baud
    Content: ASCII text transmission + binary data + framing errors
    Characteristics: 8N1, occasional parity errors, break conditions

    Sample rate: 10 MS/s
    Duration: 50 ms
    """
    print_info("Generating uart_rs232_communication.wfm...")

    baud_rate = 115200
    sample_rate = 10e6  # 10 MS/s
    duration = 50e-3  # 50 ms
    num_samples = int(sample_rate * duration)
    samples_per_bit = int(sample_rate / baud_rate)

    # Message content: ASCII text + binary data
    messages = [
        b"Hello Oscura!",  # ASCII greeting
        b"UART @ 115200 baud",  # Parameter info
        b"\x00\x01\x02\x03\x04\x05",  # Binary data
        b"ERROR",  # Message with intentional framing error after
    ]

    uart_bits = []

    # Idle high for initial period
    uart_bits.extend([1] * (samples_per_bit * 100))

    for msg_idx, message in enumerate(messages):
        for _byte_idx, byte_val in enumerate(message):
            # START bit (0)
            uart_bits.extend([0] * samples_per_bit)

            # Data bits (LSB first, 8 bits)
            for i in range(8):
                bit = (byte_val >> i) & 1
                uart_bits.extend([bit] * samples_per_bit)

            # STOP bit (1)
            uart_bits.extend([1] * samples_per_bit)

            # Inter-byte gap
            uart_bits.extend([1] * (samples_per_bit * 2))

        # After ERROR message, create a framing error (break condition)
        if msg_idx == 3:
            # Break condition: hold line low for longer than one frame
            uart_bits.extend([0] * (samples_per_bit * 15))

        # Inter-message gap
        uart_bits.extend([1] * (samples_per_bit * 20))

    # Pad to full duration
    if len(uart_bits) < num_samples:
        uart_bits.extend([1] * (num_samples - len(uart_bits)))

    # Convert to numpy array
    uart_signal = np.array(uart_bits[:num_samples], dtype=np.uint8)

    # Add realistic noise (small amplitude to not corrupt digital levels)
    noise = 0.05 * np.random.randn(num_samples)
    uart_analog = uart_signal.astype(float) * 3.3 + noise  # 3.3V logic levels

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        rx=uart_analog,
        sample_rate=sample_rate,
        duration=duration,
        baud_rate=baud_rate,
        channel_names=["RX_UART"],
        metadata={
            "protocol": "UART",
            "config": "115200 8N1",
            "content": "ASCII text + binary + framing errors",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated uart_rs232_communication.npz ({size_mb:.2f} MB)")


def generate_spi_flash_transaction(output_file: Path) -> None:
    """Generate SPI flash transaction WFM file.

    Protocol: SPI Mode 0 (CPOL=0, CPHA=0) @ 10 MHz
    Channels: CLK, MOSI, MISO, CS
    Transaction: Flash read command (0x03) + 24-bit address + data

    Sample rate: 100 MS/s
    Duration: 10 ms
    """
    print_info("Generating spi_flash_transaction.wfm...")

    spi_clock_freq = 10e6  # 10 MHz
    sample_rate = 100e6  # 100 MS/s
    duration = 10e-3  # 10 ms
    num_samples = int(sample_rate * duration)
    samples_per_bit = int(sample_rate / (2 * spi_clock_freq))  # Half clock period

    # Initialize signals (idle state)
    clk = np.ones(num_samples, dtype=np.uint8)  # Idle low (Mode 0)
    clk[:] = 0
    mosi = np.zeros(num_samples, dtype=np.uint8)
    miso = np.zeros(num_samples, dtype=np.uint8)
    cs = np.ones(num_samples, dtype=np.uint8)  # Idle high (active low)

    sample_idx = 1000  # Start after some idle time

    # Transaction 1: Read flash command (0x03) + address (0x001000) + 16 bytes data
    cs[sample_idx : sample_idx + 10] = 0  # CS assert
    sample_idx += 100

    # Command byte: 0x03 (Read Data)
    command = 0x03
    for bit_idx in range(8):
        bit = (command >> (7 - bit_idx)) & 1  # MSB first

        # Clock low, set MOSI
        mosi[sample_idx : sample_idx + samples_per_bit] = bit
        sample_idx += samples_per_bit

        # Clock high, data valid
        clk[sample_idx : sample_idx + samples_per_bit] = 1
        sample_idx += samples_per_bit

    # Address: 0x001000 (24 bits)
    address = 0x001000
    for bit_idx in range(24):
        bit = (address >> (23 - bit_idx)) & 1  # MSB first

        clk[sample_idx : sample_idx + samples_per_bit] = 0
        mosi[sample_idx : sample_idx + samples_per_bit] = bit
        sample_idx += samples_per_bit

        clk[sample_idx : sample_idx + samples_per_bit] = 1
        sample_idx += samples_per_bit

    # Data bytes from flash (MISO): simulated flash memory content
    flash_data = bytes.fromhex("48656C6C6F2046 6C617368204D656D6F7279")  # "Hello Flash Memory"
    for byte_val in flash_data:
        for bit_idx in range(8):
            bit_miso = (byte_val >> (7 - bit_idx)) & 1  # MSB first

            clk[sample_idx : sample_idx + samples_per_bit] = 0
            miso[sample_idx : sample_idx + samples_per_bit] = bit_miso
            sample_idx += samples_per_bit

            clk[sample_idx : sample_idx + samples_per_bit] = 1
            sample_idx += samples_per_bit

    # CS deassert
    cs[sample_idx : sample_idx + 100] = 1
    sample_idx += 500

    # Transaction 2: Write Enable command (0x06)
    cs[sample_idx : sample_idx + 10] = 0
    sample_idx += 100

    command = 0x06
    for bit_idx in range(8):
        bit = (command >> (7 - bit_idx)) & 1

        clk[sample_idx : sample_idx + samples_per_bit] = 0
        mosi[sample_idx : sample_idx + samples_per_bit] = bit
        sample_idx += samples_per_bit

        clk[sample_idx : sample_idx + samples_per_bit] = 1
        sample_idx += samples_per_bit

    cs[sample_idx : sample_idx + 100] = 1

    # Add realistic noise
    noise_amp = 0.05
    clk_analog = clk.astype(float) * 3.3 + noise_amp * np.random.randn(num_samples)
    mosi_analog = mosi.astype(float) * 3.3 + noise_amp * np.random.randn(num_samples)
    miso_analog = miso.astype(float) * 3.3 + noise_amp * np.random.randn(num_samples)
    cs_analog = cs.astype(float) * 3.3 + noise_amp * np.random.randn(num_samples)

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        clk=clk_analog,
        mosi=mosi_analog,
        miso=miso_analog,
        cs=cs_analog,
        sample_rate=sample_rate,
        duration=duration,
        spi_clock_freq=spi_clock_freq,
        channel_names=["CLK", "MOSI", "MISO", "CS"],
        metadata={
            "protocol": "SPI",
            "mode": "Mode 0 (CPOL=0, CPHA=0)",
            "clock": "10 MHz",
            "transaction": "Flash read (0x03) + Write enable (0x06)",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated spi_flash_transaction.npz ({size_mb:.2f} MB)")


def generate_i2c_sensor_bus(output_file: Path) -> None:
    """Generate I2C sensor bus WFM file.

    Protocol: I2C @ 400 kHz (fast mode)
    Channels: SCL, SDA
    Devices: Temperature sensor (0x48), EEPROM (0x50), RTC (0x68)
    Transactions: Reads, writes, repeated starts, NACK

    Sample rate: 20 MS/s
    Duration: 25 ms
    """
    print_info("Generating i2c_sensor_bus.wfm...")

    i2c_clock_freq = 400e3  # 400 kHz
    sample_rate = 20e6  # 20 MS/s
    duration = 25e-3  # 25 ms
    num_samples = int(sample_rate * duration)
    samples_per_bit = int(sample_rate / i2c_clock_freq)

    # Initialize signals (idle high)
    scl = np.ones(num_samples, dtype=np.uint8)
    sda = np.ones(num_samples, dtype=np.uint8)

    sample_idx = 1000

    def i2c_start(idx: int) -> int:
        """Generate I2C START condition."""
        # SDA falls while SCL is high
        sda[idx : idx + samples_per_bit // 2] = 0
        idx += samples_per_bit // 2
        scl[idx : idx + samples_per_bit // 2] = 0
        idx += samples_per_bit // 2
        return idx

    def i2c_stop(idx: int) -> int:
        """Generate I2C STOP condition."""
        # SDA rises while SCL is high
        scl[idx : idx + samples_per_bit // 2] = 1
        idx += samples_per_bit // 2
        sda[idx : idx + samples_per_bit // 2] = 1
        idx += samples_per_bit // 2
        return idx

    def i2c_write_byte(idx: int, byte_val: int, ack: bool = True) -> int:
        """Write one I2C byte with ACK/NACK."""
        for bit_idx in range(8):
            bit = (byte_val >> (7 - bit_idx)) & 1  # MSB first

            # Data valid while SCL low
            scl[idx : idx + samples_per_bit // 2] = 0
            sda[idx : idx + samples_per_bit // 2] = bit
            idx += samples_per_bit // 2

            # Clock high, data sampled
            scl[idx : idx + samples_per_bit // 2] = 1
            idx += samples_per_bit // 2

        # ACK/NACK bit
        scl[idx : idx + samples_per_bit // 2] = 0
        sda[idx : idx + samples_per_bit // 2] = 0 if ack else 1
        idx += samples_per_bit // 2

        scl[idx : idx + samples_per_bit // 2] = 1
        idx += samples_per_bit // 2

        return idx

    # Transaction 1: Read temperature from sensor (address 0x48)
    sample_idx = i2c_start(sample_idx)
    sample_idx = i2c_write_byte(sample_idx, 0x48 << 1 | 1, ack=True)  # Read address
    sample_idx = i2c_write_byte(sample_idx, 0x19, ack=True)  # Temp MSB (25°C)
    sample_idx = i2c_write_byte(sample_idx, 0x80, ack=False)  # Temp LSB, NACK
    sample_idx = i2c_stop(sample_idx)
    sample_idx += samples_per_bit * 10  # Inter-transaction gap

    # Transaction 2: Write to EEPROM (address 0x50)
    sample_idx = i2c_start(sample_idx)
    sample_idx = i2c_write_byte(sample_idx, 0x50 << 1 | 0, ack=True)  # Write address
    sample_idx = i2c_write_byte(sample_idx, 0x00, ack=True)  # Memory address
    sample_idx = i2c_write_byte(sample_idx, 0xDE, ack=True)  # Data byte 1
    sample_idx = i2c_write_byte(sample_idx, 0xAD, ack=True)  # Data byte 2
    sample_idx = i2c_write_byte(sample_idx, 0xBE, ack=True)  # Data byte 3
    sample_idx = i2c_write_byte(sample_idx, 0xEF, ack=True)  # Data byte 4
    sample_idx = i2c_stop(sample_idx)
    sample_idx += samples_per_bit * 10

    # Transaction 3: Read RTC time (address 0x68)
    sample_idx = i2c_start(sample_idx)
    sample_idx = i2c_write_byte(sample_idx, 0x68 << 1 | 0, ack=True)  # Write address
    sample_idx = i2c_write_byte(sample_idx, 0x00, ack=True)  # Register address
    sample_idx = i2c_start(sample_idx)  # Repeated START
    sample_idx = i2c_write_byte(sample_idx, 0x68 << 1 | 1, ack=True)  # Read address
    sample_idx = i2c_write_byte(sample_idx, 0x45, ack=True)  # Seconds (45)
    sample_idx = i2c_write_byte(sample_idx, 0x23, ack=True)  # Minutes (23)
    sample_idx = i2c_write_byte(sample_idx, 0x14, ack=False)  # Hours (14), NACK
    sample_idx = i2c_stop(sample_idx)

    # Add realistic noise
    noise_amp = 0.05
    scl_analog = scl.astype(float) * 3.3 + noise_amp * np.random.randn(num_samples)
    sda_analog = sda.astype(float) * 3.3 + noise_amp * np.random.randn(num_samples)

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        scl=scl_analog,
        sda=sda_analog,
        sample_rate=sample_rate,
        duration=duration,
        i2c_clock_freq=i2c_clock_freq,
        channel_names=["SCL", "SDA"],
        metadata={
            "protocol": "I2C",
            "speed": "400 kHz (fast mode)",
            "devices": "Temp sensor (0x48), EEPROM (0x50), RTC (0x68)",
            "transactions": "Read temp, write EEPROM, read RTC",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated i2c_sensor_bus.npz ({size_mb:.2f} MB)")


def generate_can_automotive_bus(output_file: Path) -> None:
    """Generate CAN automotive bus WFM file.

    Protocol: CAN 2.0B @ 500 kbps
    Messages: Standard IDs (11-bit) and extended IDs (29-bit)
    Content: Engine RPM, vehicle speed, diagnostic messages

    Sample rate: 10 MS/s
    Duration: 50 ms
    """
    print_info("Generating can_automotive_bus.wfm...")

    can_bitrate = 500e3  # 500 kbps
    sample_rate = 10e6  # 10 MS/s
    duration = 50e-3  # 50 ms
    num_samples = int(sample_rate * duration)
    samples_per_bit = int(sample_rate / can_bitrate)

    # Initialize signal (idle high/recessive)
    can_bus = np.ones(num_samples, dtype=np.uint8)

    sample_idx = 1000

    def stuff_bits(bits: list[int]) -> list[int]:
        """Apply bit stuffing (insert opposite bit after 5 consecutive same bits)."""
        stuffed = []
        count = 0
        last_bit = -1

        for bit in bits:
            if bit == last_bit:
                count += 1
            else:
                count = 1
                last_bit = bit

            stuffed.append(bit)

            if count == 5:
                # Insert opposite bit
                stuffed.append(1 - bit)
                count = 1
                last_bit = 1 - bit

        return stuffed

    def can_write_bits(idx: int, bits: list[int]) -> int:
        """Write CAN bits with bit stuffing."""
        stuffed_bits = stuff_bits(bits)
        for bit in stuffed_bits:
            can_bus[idx : idx + samples_per_bit] = bit
            idx += samples_per_bit
        return idx

    def crc15(data_bits: list[int]) -> int:
        """Calculate CAN CRC-15."""
        crc = 0
        for bit in data_bits:
            crc_next = bit ^ (crc >> 14)
            crc = ((crc << 1) & 0x7FFF) | crc_next
            if crc_next:
                crc ^= 0x4599
        return crc & 0x7FFF

    # Message 1: Engine RPM (ID=0x100, 2 bytes: 2000 RPM)
    msg_id = 0x100
    data = [0x07, 0xD0]  # 2000 RPM

    # SOF (Start of Frame) - dominant
    can_bus[sample_idx : sample_idx + samples_per_bit] = 0
    sample_idx += samples_per_bit

    # Arbitration field (11-bit ID + RTR)
    arb_bits = []
    for i in range(11):
        arb_bits.append((msg_id >> (10 - i)) & 1)
    arb_bits.append(0)  # RTR (0 = data frame)

    # Control field (IDE + r0 + DLC)
    ctrl_bits = [0, 0] + [0, 0, 1, 0]  # IDE=0, r0=0, DLC=2

    # Data field
    data_bits = []
    for byte_val in data:
        for i in range(8):
            data_bits.append((byte_val >> (7 - i)) & 1)

    # Calculate CRC
    all_bits = arb_bits + ctrl_bits + data_bits
    crc_val = crc15(all_bits)
    crc_bits = [(crc_val >> (14 - i)) & 1 for i in range(15)]

    # Write frame fields
    sample_idx = can_write_bits(sample_idx, arb_bits + ctrl_bits + data_bits + crc_bits)

    # CRC delimiter + ACK + ACK delimiter + EOF
    delim_bits = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # All recessive
    for bit in delim_bits:
        can_bus[sample_idx : sample_idx + samples_per_bit] = bit
        sample_idx += samples_per_bit

    # Inter-frame space
    sample_idx += samples_per_bit * 3

    # Message 2: Vehicle speed (ID=0x200, 2 bytes: 65 km/h)
    msg_id = 0x200
    data = [0x00, 0x41]  # 65 km/h

    can_bus[sample_idx : sample_idx + samples_per_bit] = 0  # SOF
    sample_idx += samples_per_bit

    arb_bits = [(msg_id >> (10 - i)) & 1 for i in range(11)] + [0]
    ctrl_bits = [0, 0, 0, 0, 1, 0]
    data_bits = []
    for byte_val in data:
        for i in range(8):
            data_bits.append((byte_val >> (7 - i)) & 1)

    crc_val = crc15(arb_bits + ctrl_bits + data_bits)
    crc_bits = [(crc_val >> (14 - i)) & 1 for i in range(15)]

    sample_idx = can_write_bits(sample_idx, arb_bits + ctrl_bits + data_bits + crc_bits)

    delim_bits = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    for bit in delim_bits:
        can_bus[sample_idx : sample_idx + samples_per_bit] = bit
        sample_idx += samples_per_bit

    # Add realistic noise
    noise_amp = 0.05
    can_analog = can_bus.astype(float) * 2.5 + noise_amp * np.random.randn(num_samples)

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        can=can_analog,
        sample_rate=sample_rate,
        duration=duration,
        can_bitrate=can_bitrate,
        channel_names=["CAN_BUS"],
        metadata={
            "protocol": "CAN 2.0B",
            "bitrate": "500 kbps",
            "messages": "Engine RPM (ID=0x100), Vehicle speed (ID=0x200)",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated can_automotive_bus.npz ({size_mb:.2f} MB)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demo data for protocol decoding")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Create demo_data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "demo_data"
    data_dir.mkdir(exist_ok=True)

    # Define output files
    files_to_generate = [
        ("uart_rs232_communication.wfm", generate_uart_rs232_communication),
        ("spi_flash_transaction.wfm", generate_spi_flash_transaction),
        ("i2c_sensor_bus.wfm", generate_i2c_sensor_bus),
        ("can_automotive_bus.wfm", generate_can_automotive_bus),
    ]

    print(f"\n{BOLD}{BLUE}Generating Protocol Decoding Demo Data{RESET}")
    print("=" * 80)

    for filename, generator_func in files_to_generate:
        output_file = data_dir / filename

        if output_file.with_suffix(".npz").exists() and not args.force:
            print_info(f"Skipping {filename} (already exists, use --force to overwrite)")
            continue

        generator_func(output_file)

    print(f"\n{GREEN}{BOLD}✓ Demo data generation complete!{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
