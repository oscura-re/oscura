"""Hardware Abstraction Layer (HAL) detection and analysis.

    - RE-HAL-001: Register Access Pattern Detection
    - RE-HAL-002: Peripheral Driver Identification
    - RE-HAL-003: HAL Framework Recognition

This module provides tools for identifying hardware abstraction layer patterns
in firmware binaries and protocol traffic, detecting register access patterns,
peripheral drivers, and HAL framework signatures.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class RegisterAccess:
    """Register access pattern.

    Implements RE-HAL-001: Register access pattern representation.

    Attributes:
        address: Register address (hex).
        access_type: Type of access (read/write/rmw).
        bit_mask: Bit mask for read-modify-write operations.
        frequency: Number of times this access occurred.
        offset_from_base: Offset from peripheral base address.

    Example:
        >>> reg = RegisterAccess(
        ...     address=0x40021000,
        ...     access_type="rmw",
        ...     bit_mask=0x00000001,
        ...     frequency=5
        ... )
        >>> reg.access_type
        'rmw'
    """

    address: int
    access_type: Literal["read", "write", "rmw"]
    bit_mask: int | None = None
    frequency: int = 1
    offset_from_base: int | None = None


@dataclass
class Peripheral:
    """Peripheral driver information.

    Implements RE-HAL-002: Peripheral driver representation.

    Attributes:
        peripheral_type: Type of peripheral (GPIO, UART, SPI, etc.).
        base_address: Base address of peripheral registers.
        registers: Dictionary of register offsets to access patterns.
        access_patterns: List of detected access patterns.
        initialization_sequence: Ordered list of register accesses during init.

    Example:
        >>> periph = Peripheral(
        ...     peripheral_type="UART",
        ...     base_address=0x40011000,
        ...     registers={0x00: "Control Register"}
        ... )
        >>> periph.peripheral_type
        'UART'
    """

    peripheral_type: str
    base_address: int
    registers: dict[int, str] = field(default_factory=dict)
    access_patterns: list[RegisterAccess] = field(default_factory=list)
    initialization_sequence: list[int] = field(default_factory=list)


@dataclass
class HALAnalysisResult:
    """Result of HAL detection analysis.

    Implements RE-HAL-003: HAL analysis result.

    Attributes:
        detected_hal: Name of detected HAL framework (or "Unknown").
        peripherals: List of identified peripherals.
        register_map: Complete register address map.
        initialization_sequence: Ordered initialization operations.
        confidence: Detection confidence (0.0-1.0).
        framework_signatures: List of detected framework signatures.

    Example:
        >>> result = HALAnalysisResult(
        ...     detected_hal="STM32 HAL",
        ...     peripherals=[],
        ...     register_map={},
        ...     initialization_sequence=[],
        ...     confidence=0.95
        ... )
        >>> result.detected_hal
        'STM32 HAL'
    """

    detected_hal: str
    peripherals: list[Peripheral]
    register_map: dict[int, str]
    initialization_sequence: list[dict[str, Any]]
    confidence: float = 0.0
    framework_signatures: list[str] = field(default_factory=list)


# Known HAL framework signatures
HAL_SIGNATURES: dict[str, dict[str, Any]] = {
    "STM32 HAL": {
        "patterns": [
            b"HAL_Init",
            b"HAL_Delay",
            b"HAL_GPIO_",
            b"HAL_UART_",
            b"HAL_SPI_",
            b"HAL_I2C_",
        ],
        "base_addresses": {
            0x40000000: "APB1 Peripheral",
            0x40010000: "APB2 Peripheral",
            0x40020000: "AHB1 Peripheral",
            0x50000000: "AHB2 Peripheral",
        },
    },
    "Nordic SDK": {
        "patterns": [
            b"nrf_",
            b"NRF_",
            b"nrfx_",
            b"NRFX_",
            b"app_uart_",
            b"ble_",
        ],
        "base_addresses": {
            0x40000000: "APB Peripheral",
            0x50000000: "AHB Peripheral",
        },
    },
    "ESP-IDF": {
        "patterns": [
            b"esp_",
            b"ESP_",
            b"gpio_",
            b"uart_",
            b"spi_",
            b"i2c_",
        ],
        "base_addresses": {
            0x3FF00000: "DPORT Peripheral",
            0x3FF40000: "UART Peripheral",
            0x3FF44000: "SPI Peripheral",
        },
    },
    "Arduino": {
        "patterns": [
            b"digitalWrite",
            b"digitalRead",
            b"pinMode",
            b"analogRead",
            b"analogWrite",
            b"Serial.begin",
        ],
        "base_addresses": {},
    },
    "CMSIS": {
        "patterns": [
            b"CMSIS",
            b"__NVIC_",
            b"SysTick_",
            b"SCB->",
            b"NVIC->",
        ],
        "base_addresses": {
            0xE000E000: "System Control Space",
            0xE000E010: "SysTick",
            0xE000E100: "NVIC",
        },
    },
}

# Common MCU peripheral base addresses (ARM Cortex-M)
ARM_PERIPHERAL_BASES: dict[int, str] = {
    # STM32F4 family
    0x40000000: "TIM2",
    0x40000400: "TIM3",
    0x40000800: "TIM4",
    0x40000C00: "TIM5",
    0x40001000: "TIM6",
    0x40001400: "TIM7",
    0x40002800: "RTC",
    0x40002C00: "WWDG",
    0x40003000: "IWDG",
    0x40003800: "SPI2",
    0x40003C00: "SPI3",
    0x40004400: "USART2",
    0x40004800: "USART3",
    0x40004C00: "UART4",
    0x40005000: "UART5",
    0x40005400: "I2C1",
    0x40005800: "I2C2",
    0x40005C00: "I2C3",
    0x40007000: "PWR",
    0x40007400: "DAC",
    0x40010000: "TIM1",
    0x40011000: "USART1",
    0x40011400: "USART6",
    0x40012C00: "ADC1",
    0x40013000: "SPI1",
    0x40014000: "SYSCFG",
    0x40020000: "GPIOA",
    0x40020400: "GPIOB",
    0x40020800: "GPIOC",
    0x40020C00: "GPIOD",
    0x40021000: "GPIOE",
    0x40021400: "GPIOF",
    0x40021800: "GPIOG",
    0x40021C00: "GPIOH",
    0x40023000: "CRC",
    0x40023800: "RCC",
    0x40023C00: "FLASH",
    0x40026000: "DMA1",
    0x40026400: "DMA2",
}

# AVR peripheral base addresses
AVR_PERIPHERAL_BASES: dict[int, str] = {
    0x0020: "PORTA",
    0x0023: "PORTB",
    0x0026: "PORTC",
    0x0029: "PORTD",
    0x00B8: "USART0",
    0x00C0: "USART1",
    0x004C: "SPI",
    0x00B0: "TWI (I2C)",
    0x0080: "ADC",
}

# PIC peripheral base addresses
PIC_PERIPHERAL_BASES: dict[int, str] = {
    0x0005: "PORTA",
    0x0006: "PORTB",
    0x0007: "PORTC",
    0x0011: "USART",
    0x0015: "SPI",
    0x001E: "ADC",
}


class HALDetector:
    """Hardware Abstraction Layer detector.

    Identifies HAL patterns, peripheral drivers, and register access
    patterns in firmware binaries and protocol traffic.

    Attributes:
        register_accesses: Tracked register access patterns.
        peripherals: Detected peripheral drivers.
        hal_framework: Detected HAL framework name.

    Example:
        >>> detector = HALDetector()
        >>> # Analyze firmware binary
        >>> binary = b"\\x00\\x10\\x02\\x40HAL_Init..."
        >>> result = detector.analyze_firmware(binary)
        >>> result.detected_hal
        'STM32 HAL'
        >>> len(result.peripherals) > 0
        True
    """

    def __init__(self) -> None:
        """Initialize HAL detector."""
        self.register_accesses: list[RegisterAccess] = []
        self.peripherals: list[Peripheral] = []
        self.hal_framework: str = "Unknown"
        self._address_frequency: Counter[int] = Counter()

    def analyze_firmware(
        self,
        binary_data: bytes,
        *,
        detect_peripherals: bool = True,
        detect_framework: bool = True,
    ) -> HALAnalysisResult:
        """Analyze firmware binary for HAL patterns.

        Args:
            binary_data: Firmware binary data.
            detect_peripherals: Enable peripheral detection.
            detect_framework: Enable HAL framework detection.

        Returns:
            HAL analysis result with detected patterns.

        Raises:
            ValueError: If binary_data is empty.

        Example:
            >>> detector = HALDetector()
            >>> binary = b"HAL_Init\\x00\\x00\\x10\\x02\\x40"
            >>> result = detector.analyze_firmware(binary)
            >>> result.confidence > 0.0
            True
        """
        if not binary_data:
            raise ValueError("Binary data cannot be empty")

        # Reset state
        self.register_accesses = []
        self.peripherals = []
        self.hal_framework = "Unknown"
        self._address_frequency = Counter()

        # Detect HAL framework
        framework_confidence = 0.0
        if detect_framework:
            self.hal_framework, framework_confidence = self._detect_hal_framework(binary_data)

        # Extract register access patterns
        self._extract_register_accesses(binary_data)

        # Detect peripherals
        if detect_peripherals:
            self._detect_peripherals()

        # Build register map
        register_map = self._build_register_map()

        # Extract initialization sequence
        init_sequence = self._extract_initialization_sequence()

        # Calculate overall confidence
        confidence = self._calculate_confidence(framework_confidence)

        return HALAnalysisResult(
            detected_hal=self.hal_framework,
            peripherals=self.peripherals,
            register_map=register_map,
            initialization_sequence=init_sequence,
            confidence=confidence,
            framework_signatures=self._get_detected_signatures(binary_data),
        )

    def _detect_hal_framework(self, binary_data: bytes) -> tuple[str, float]:
        """Detect HAL framework from binary signatures.

        Args:
            binary_data: Firmware binary data.

        Returns:
            Tuple of (framework_name, confidence).
        """
        scores: dict[str, float] = {}

        for framework, info in HAL_SIGNATURES.items():
            matches = 0
            patterns = info["patterns"]

            for pattern in patterns:
                if pattern in binary_data:
                    matches += 1

            if len(patterns) > 0:
                score = matches / len(patterns)
                scores[framework] = score

        if not scores:
            return "Unknown", 0.0

        best_framework = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_framework]

        return best_framework, confidence

    def _extract_register_accesses(self, binary_data: bytes) -> None:
        """Extract register access patterns from binary.

        Args:
            binary_data: Firmware binary data.
        """
        # Convert to numpy array for easier processing
        data_array = np.frombuffer(binary_data, dtype=np.uint8)

        # Look for 32-bit addresses (little-endian)
        if len(data_array) < 4:
            return

        # Scan for potential memory-mapped register addresses
        for i in range(len(data_array) - 3):
            # Extract potential 32-bit address
            addr = int.from_bytes(data_array[i : i + 4].tobytes(), byteorder="little", signed=False)

            # Check if address looks like a peripheral register
            if self._is_peripheral_address(addr):
                self._address_frequency[addr] += 1

                # Try to determine access type from surrounding bytes
                access_type = self._infer_access_type(data_array, i)

                self.register_accesses.append(
                    RegisterAccess(address=addr, access_type=access_type, frequency=1)
                )

    def _is_peripheral_address(self, addr: int) -> bool:
        """Check if address is likely a peripheral register.

        Args:
            addr: Memory address.

        Returns:
            True if address is in peripheral memory range.
        """
        # ARM Cortex-M peripheral ranges
        if 0x40000000 <= addr < 0x60000000:  # Peripheral range
            return True
        if 0xE0000000 <= addr < 0xE0100000:  # System control
            return True

        # AVR peripheral range
        if 0x0020 <= addr < 0x0100:
            return True

        # PIC peripheral range
        return 0x0000 <= addr < 0x0030

    def _infer_access_type(
        self, data_array: np.ndarray[tuple[int], np.dtype[np.uint8]], index: int
    ) -> Literal["read", "write", "rmw"]:
        """Infer register access type from context.

        Args:
            data_array: Binary data as numpy array.
            index: Index of address in array.

        Returns:
            Access type (read, write, or rmw).
        """
        # Simple heuristic: look at nearby opcodes (ARM Thumb)
        if index > 0:
            prev_byte = int(data_array[index - 1])

            # Common ARM Thumb load/store opcodes
            if prev_byte in (0x68, 0x78, 0x88):  # LDR variants
                return "read"
            if prev_byte in (0x60, 0x70, 0x80):  # STR variants
                return "write"

        # Check for bit manipulation patterns (read-modify-write)
        if index + 8 < len(data_array):
            # Look for ORR, AND, BIC patterns
            window = data_array[index : index + 8]
            if any(b in (0x43, 0x40, 0x44) for b in window):  # ORR, AND, BIC opcodes
                return "rmw"

        return "write"

    def _detect_peripherals(self) -> None:
        """Detect peripheral drivers from register access patterns."""
        # Group accesses by base address
        base_groups: dict[int, list[RegisterAccess]] = defaultdict(list)

        for access in self.register_accesses:
            # Try to find matching peripheral base
            base = self._find_peripheral_base(access.address)
            if base is not None:
                access.offset_from_base = access.address - base
                base_groups[base].append(access)

        # Create peripheral objects
        for base, accesses in base_groups.items():
            periph_type = self._identify_peripheral_type(base, accesses)

            peripheral = Peripheral(
                peripheral_type=periph_type,
                base_address=base,
                access_patterns=accesses,
            )

            # Extract register names
            peripheral.registers = self._extract_register_names(base)

            self.peripherals.append(peripheral)

    def _find_peripheral_base(self, address: int) -> int | None:
        """Find peripheral base address for given register address.

        Args:
            address: Register address.

        Returns:
            Base address or None.
        """
        # Check ARM peripherals
        for base in ARM_PERIPHERAL_BASES:
            if base <= address < base + 0x400:  # Typical peripheral size
                return base

        # Check AVR peripherals
        for base in AVR_PERIPHERAL_BASES:
            if base <= address < base + 0x10:
                return base

        # Check PIC peripherals
        for base in PIC_PERIPHERAL_BASES:
            if base <= address < base + 0x10:
                return base

        return None

    def _identify_peripheral_type(self, base_address: int, accesses: list[RegisterAccess]) -> str:
        """Identify peripheral type from base address and access patterns.

        Args:
            base_address: Peripheral base address.
            accesses: List of register accesses.

        Returns:
            Peripheral type name.
        """
        # Check known base addresses
        if base_address in ARM_PERIPHERAL_BASES:
            return ARM_PERIPHERAL_BASES[base_address]

        if base_address in AVR_PERIPHERAL_BASES:
            return AVR_PERIPHERAL_BASES[base_address]

        if base_address in PIC_PERIPHERAL_BASES:
            return PIC_PERIPHERAL_BASES[base_address]

        # Heuristic: identify by access patterns
        if len(accesses) > 10 and any(a.access_type == "rmw" for a in accesses):
            return "GPIO"

        return "Unknown Peripheral"

    def _extract_register_names(self, base_address: int) -> dict[int, str]:
        """Extract register names for peripheral.

        Args:
            base_address: Peripheral base address.

        Returns:
            Dictionary of offset to register name.
        """
        registers: dict[int, str] = {}

        # GPIO registers (common pattern)
        if "GPIO" in self._identify_peripheral_type(base_address, []):
            registers = {
                0x00: "MODER (Mode Register)",
                0x04: "OTYPER (Output Type)",
                0x08: "OSPEEDR (Output Speed)",
                0x0C: "PUPDR (Pull-up/Pull-down)",
                0x10: "IDR (Input Data)",
                0x14: "ODR (Output Data)",
                0x18: "BSRR (Bit Set/Reset)",
            }
        # UART registers
        elif "UART" in self._identify_peripheral_type(base_address, []):
            registers = {
                0x00: "CR1 (Control 1)",
                0x04: "CR2 (Control 2)",
                0x08: "CR3 (Control 3)",
                0x0C: "BRR (Baud Rate)",
                0x1C: "RDR (Receive Data)",
                0x28: "TDR (Transmit Data)",
            }

        return registers

    def _build_register_map(self) -> dict[int, str]:
        """Build complete register address map.

        Returns:
            Dictionary of address to description.
        """
        register_map: dict[int, str] = {}

        for peripheral in self.peripherals:
            for offset, name in peripheral.registers.items():
                addr = peripheral.base_address + offset
                register_map[addr] = f"{peripheral.peripheral_type}.{name}"

        return register_map

    def _extract_initialization_sequence(self) -> list[dict[str, Any]]:
        """Extract initialization sequence from register accesses.

        Returns:
            List of initialization operations.
        """
        init_ops: list[dict[str, Any]] = []

        # Group accesses by peripheral
        for peripheral in self.peripherals:
            # Clock configuration often comes first
            if "RCC" in peripheral.peripheral_type or "Clock" in peripheral.peripheral_type:
                init_ops.append(
                    {
                        "step": "clock_config",
                        "peripheral": peripheral.peripheral_type,
                        "base_address": hex(peripheral.base_address),
                    }
                )

            # GPIO configuration
            elif "GPIO" in peripheral.peripheral_type:
                init_ops.append(
                    {
                        "step": "gpio_init",
                        "peripheral": peripheral.peripheral_type,
                        "base_address": hex(peripheral.base_address),
                        "registers": list(peripheral.registers.keys()),
                    }
                )

            # Peripheral initialization
            elif peripheral.peripheral_type in ("UART", "SPI", "I2C", "ADC"):
                init_ops.append(
                    {
                        "step": "peripheral_init",
                        "peripheral": peripheral.peripheral_type,
                        "base_address": hex(peripheral.base_address),
                    }
                )

        return init_ops

    def _calculate_confidence(self, framework_confidence: float) -> float:
        """Calculate overall detection confidence.

        Args:
            framework_confidence: HAL framework detection confidence.

        Returns:
            Overall confidence score (0.0-1.0).
        """
        # Weight different factors
        weights = {
            "framework": 0.4,
            "peripherals": 0.3,
            "registers": 0.3,
        }

        # Framework confidence
        conf = framework_confidence * weights["framework"]

        # Peripheral detection confidence
        if len(self.peripherals) > 0:
            conf += min(len(self.peripherals) / 5.0, 1.0) * weights["peripherals"]

        # Register access confidence
        if len(self.register_accesses) > 0:
            conf += min(len(self.register_accesses) / 20.0, 1.0) * weights["registers"]

        return min(conf, 1.0)

    def _get_detected_signatures(self, binary_data: bytes) -> list[str]:
        """Get list of detected HAL framework signatures.

        Args:
            binary_data: Firmware binary data.

        Returns:
            List of detected signature strings.
        """
        signatures: list[str] = []

        if self.hal_framework in HAL_SIGNATURES:
            patterns = HAL_SIGNATURES[self.hal_framework]["patterns"]

            for pattern in patterns:
                if pattern in binary_data:
                    signatures.append(pattern.decode("utf-8", errors="ignore"))

        return signatures

    def export_to_json(self, result: HALAnalysisResult, *, indent: int = 2) -> str:
        """Export HAL analysis result to JSON.

        Args:
            result: HAL analysis result.
            indent: JSON indentation level.

        Returns:
            JSON string representation.

        Example:
            >>> detector = HALDetector()
            >>> result = HALAnalysisResult(
            ...     detected_hal="STM32 HAL",
            ...     peripherals=[],
            ...     register_map={},
            ...     initialization_sequence=[],
            ...     confidence=0.8
            ... )
            >>> json_str = detector.export_to_json(result)
            >>> "STM32 HAL" in json_str
            True
        """
        data = {
            "hal_framework": result.detected_hal,
            "confidence": result.confidence,
            "framework_signatures": result.framework_signatures,
            "peripherals": [
                {
                    "type": p.peripheral_type,
                    "base_address": hex(p.base_address),
                    "registers": {hex(offset): name for offset, name in p.registers.items()},
                    "access_count": len(p.access_patterns),
                }
                for p in result.peripherals
            ],
            "register_map": {hex(addr): desc for addr, desc in result.register_map.items()},
            "initialization_sequence": result.initialization_sequence,
        }

        return json.dumps(data, indent=indent)


__all__ = [
    "HALAnalysisResult",
    "HALDetector",
    "Peripheral",
    "RegisterAccess",
]
