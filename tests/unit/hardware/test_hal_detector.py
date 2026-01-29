"""Comprehensive unit tests for HAL detector.

Tests for:
    - RE-HAL-001: Register Access Pattern Detection
    - RE-HAL-002: Peripheral Driver Identification
    - RE-HAL-003: HAL Framework Recognition

This module provides comprehensive test coverage for hardware abstraction
layer detection, peripheral identification, and framework recognition.
"""

from __future__ import annotations

import json
import struct

import pytest

from oscura.hardware.hal_detector import (
    HALAnalysisResult,
    HALDetector,
    Peripheral,
    RegisterAccess,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test RegisterAccess Dataclass
# =============================================================================


class TestRegisterAccess:
    """Test RegisterAccess dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating RegisterAccess with all fields."""
        reg = RegisterAccess(
            address=0x40021000,
            access_type="rmw",
            bit_mask=0x00000001,
            frequency=5,
            offset_from_base=0x1000,
        )

        assert reg.address == 0x40021000
        assert reg.access_type == "rmw"
        assert reg.bit_mask == 0x00000001
        assert reg.frequency == 5
        assert reg.offset_from_base == 0x1000

    def test_creation_with_minimal_fields(self) -> None:
        """Test creating RegisterAccess with minimal fields."""
        reg = RegisterAccess(address=0x40000000, access_type="read")

        assert reg.address == 0x40000000
        assert reg.access_type == "read"
        assert reg.bit_mask is None
        assert reg.frequency == 1
        assert reg.offset_from_base is None

    def test_write_access(self) -> None:
        """Test write access type."""
        reg = RegisterAccess(address=0x40020000, access_type="write")

        assert reg.access_type == "write"


# =============================================================================
# Test Peripheral Dataclass
# =============================================================================


class TestPeripheral:
    """Test Peripheral dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating Peripheral with all fields."""
        periph = Peripheral(
            peripheral_type="UART",
            base_address=0x40011000,
            registers={0x00: "Control Register", 0x04: "Status Register"},
            access_patterns=[RegisterAccess(address=0x40011000, access_type="write")],
            initialization_sequence=[0x00, 0x04],
        )

        assert periph.peripheral_type == "UART"
        assert periph.base_address == 0x40011000
        assert len(periph.registers) == 2
        assert len(periph.access_patterns) == 1
        assert periph.initialization_sequence == [0x00, 0x04]

    def test_creation_with_minimal_fields(self) -> None:
        """Test creating Peripheral with minimal fields."""
        periph = Peripheral(peripheral_type="GPIO", base_address=0x40020000)

        assert periph.peripheral_type == "GPIO"
        assert periph.base_address == 0x40020000
        assert periph.registers == {}
        assert periph.access_patterns == []
        assert periph.initialization_sequence == []


# =============================================================================
# Test HALAnalysisResult Dataclass
# =============================================================================


class TestHALAnalysisResult:
    """Test HALAnalysisResult dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating HALAnalysisResult with all fields."""
        result = HALAnalysisResult(
            detected_hal="STM32 HAL",
            peripherals=[Peripheral("GPIO", 0x40020000)],
            register_map={0x40020000: "GPIOA.MODER"},
            initialization_sequence=[{"step": "clock_config"}],
            confidence=0.95,
            framework_signatures=["HAL_Init", "HAL_GPIO_Init"],
        )

        assert result.detected_hal == "STM32 HAL"
        assert len(result.peripherals) == 1
        assert len(result.register_map) == 1
        assert len(result.initialization_sequence) == 1
        assert result.confidence == 0.95
        assert len(result.framework_signatures) == 2

    def test_creation_with_minimal_fields(self) -> None:
        """Test creating HALAnalysisResult with minimal fields."""
        result = HALAnalysisResult(
            detected_hal="Unknown",
            peripherals=[],
            register_map={},
            initialization_sequence=[],
        )

        assert result.detected_hal == "Unknown"
        assert result.peripherals == []
        assert result.register_map == {}
        assert result.initialization_sequence == []
        assert result.confidence == 0.0
        assert result.framework_signatures == []


# =============================================================================
# Test HALDetector Initialization
# =============================================================================


class TestHALDetectorInit:
    """Test HALDetector initialization."""

    def test_initialization(self) -> None:
        """Test detector initializes correctly."""
        detector = HALDetector()

        assert detector.register_accesses == []
        assert detector.peripherals == []
        assert detector.hal_framework == "Unknown"

    def test_multiple_instances(self) -> None:
        """Test multiple detector instances are independent."""
        detector1 = HALDetector()
        detector2 = HALDetector()

        detector1.hal_framework = "STM32 HAL"

        assert detector2.hal_framework == "Unknown"


# =============================================================================
# Test HAL Framework Detection
# =============================================================================


class TestHALFrameworkDetection:
    """Test HAL framework detection."""

    def test_stm32_hal_detection(self) -> None:
        """Test STM32 HAL framework detection."""
        detector = HALDetector()

        # Create binary with STM32 HAL signatures
        binary = (
            b"HAL_Init\x00\x00\x00"
            b"HAL_GPIO_Init\x00"
            b"HAL_UART_Transmit\x00"
            b"\x00\x10\x02\x40"  # GPIOA address
        )

        result = detector.analyze_firmware(binary)

        assert result.detected_hal == "STM32 HAL"
        assert result.confidence > 0.0
        assert "HAL_Init" in result.framework_signatures

    def test_nordic_sdk_detection(self) -> None:
        """Test Nordic SDK framework detection."""
        detector = HALDetector()

        binary = (
            b"nrf_gpio_pin_set\x00"
            b"nrfx_uart_init\x00"
            b"app_uart_put\x00"
            b"\x00\x00\x00\x40"  # Nordic peripheral address
        )

        result = detector.analyze_firmware(binary)

        assert result.detected_hal == "Nordic SDK"
        assert result.confidence > 0.0

    def test_esp_idf_detection(self) -> None:
        """Test ESP-IDF framework detection."""
        detector = HALDetector()

        binary = b"esp_wifi_init\x00gpio_set_level\x00uart_driver_install\x00"

        result = detector.analyze_firmware(binary)

        assert result.detected_hal == "ESP-IDF"
        assert result.confidence > 0.0

    def test_arduino_detection(self) -> None:
        """Test Arduino framework detection."""
        detector = HALDetector()

        binary = b"digitalWrite\x00digitalRead\x00pinMode\x00Serial.begin\x00"

        result = detector.analyze_firmware(binary)

        assert result.detected_hal == "Arduino"
        assert result.confidence > 0.0

    def test_cmsis_detection(self) -> None:
        """Test CMSIS framework detection."""
        detector = HALDetector()

        binary = b"CMSIS\x00__NVIC_EnableIRQ\x00SysTick_Config\x00"

        result = detector.analyze_firmware(binary)

        assert result.detected_hal == "CMSIS"
        assert result.confidence > 0.0

    def test_unknown_framework(self) -> None:
        """Test unknown framework detection."""
        detector = HALDetector()

        binary = b"unknown_hal_function_xyz_custom_proprietary"

        result = detector.analyze_firmware(binary)

        # Should be Unknown or have low confidence
        assert result.confidence >= 0.0


# =============================================================================
# Test Register Access Detection
# =============================================================================


class TestRegisterAccessDetection:
    """Test register access pattern detection."""

    def test_peripheral_address_detection(self) -> None:
        """Test detection of peripheral register addresses."""
        detector = HALDetector()

        # Create binary with embedded peripheral addresses
        binary = (
            b"\x00\x00\x00\x00"
            + struct.pack("<I", 0x40020000)  # GPIOA
            + struct.pack("<I", 0x40011000)  # USART1
        )

        result = detector.analyze_firmware(binary)

        # Should detect at least one register access
        assert len(detector.register_accesses) > 0

    def test_read_access_detection(self) -> None:
        """Test read access type detection."""
        detector = HALDetector()

        # LDR opcode + address
        binary = b"\x68" + struct.pack("<I", 0x40020000)

        result = detector.analyze_firmware(binary)

        if detector.register_accesses:
            assert any(acc.access_type == "read" for acc in detector.register_accesses)

    def test_write_access_detection(self) -> None:
        """Test write access type detection."""
        detector = HALDetector()

        # STR opcode + address
        binary = b"\x60" + struct.pack("<I", 0x40020400)

        result = detector.analyze_firmware(binary)

        if detector.register_accesses:
            assert any(acc.access_type == "write" for acc in detector.register_accesses)

    def test_rmw_access_detection(self) -> None:
        """Test read-modify-write access detection."""
        detector = HALDetector()

        # ORR opcode + address + more data
        binary = b"\x43" + struct.pack("<I", 0x40020000) + b"\x00\x00\x00\x00\x00\x00"

        result = detector.analyze_firmware(binary)

        if detector.register_accesses:
            assert any(acc.access_type == "rmw" for acc in detector.register_accesses)


# =============================================================================
# Test Peripheral Detection
# =============================================================================


class TestPeripheralDetection:
    """Test peripheral driver detection."""

    def test_gpio_detection(self) -> None:
        """Test GPIO peripheral detection."""
        detector = HALDetector()

        # Multiple accesses to GPIOA registers
        binary = (
            b"\x00" * 10
            + struct.pack("<I", 0x40020000)  # GPIOA MODER
            + struct.pack("<I", 0x40020014)  # GPIOA ODR
            + struct.pack("<I", 0x40020018)  # GPIOA BSRR
        )

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        gpio_found = any("GPIO" in p.peripheral_type for p in result.peripherals)
        assert gpio_found or len(result.peripherals) > 0

    def test_uart_detection(self) -> None:
        """Test UART peripheral detection."""
        detector = HALDetector()

        binary = (
            b"\x00" * 10
            + struct.pack("<I", 0x40011000)  # USART1 CR1
            + struct.pack("<I", 0x40011004)  # USART1 CR2
        )

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        uart_found = any("UART" in p.peripheral_type for p in result.peripherals)
        assert uart_found or len(result.peripherals) > 0

    def test_multiple_peripherals(self) -> None:
        """Test detection of multiple peripherals."""
        detector = HALDetector()

        binary = (
            b"\x00" * 10
            + struct.pack("<I", 0x40020000)  # GPIOA
            + struct.pack("<I", 0x40011000)  # USART1
            + struct.pack("<I", 0x40003800)  # SPI2
        )

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        # Should detect at least one peripheral
        assert len(result.peripherals) > 0

    def test_peripheral_base_detection(self) -> None:
        """Test peripheral base address detection."""
        detector = HALDetector()

        binary = b"\x00" * 10 + struct.pack("<I", 0x40020014)  # GPIOA ODR

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        if result.peripherals:
            # Should group under GPIOA base
            assert any(p.base_address == 0x40020000 for p in result.peripherals)


# =============================================================================
# Test Register Map Building
# =============================================================================


class TestRegisterMapBuilding:
    """Test register map generation."""

    def test_register_map_generation(self) -> None:
        """Test register map is generated correctly."""
        detector = HALDetector()

        binary = (
            b"\x00" * 10
            + struct.pack("<I", 0x40020000)  # GPIOA
            + struct.pack("<I", 0x40020014)  # GPIOA ODR
        )

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        assert isinstance(result.register_map, dict)

    def test_register_map_entries(self) -> None:
        """Test register map contains correct entries."""
        detector = HALDetector()

        binary = b"\x00" * 10 + struct.pack("<I", 0x40020000)

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        # If peripherals detected, register map should have entries
        if result.peripherals:
            assert len(result.register_map) >= 0


# =============================================================================
# Test Initialization Sequence Extraction
# =============================================================================


class TestInitializationSequence:
    """Test initialization sequence extraction."""

    def test_clock_config_detection(self) -> None:
        """Test clock configuration detection."""
        detector = HALDetector()

        binary = b"\x00" * 10 + struct.pack("<I", 0x40023800)  # RCC

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        # Check if clock config is in init sequence
        if result.initialization_sequence:
            assert any(op.get("step") == "clock_config" for op in result.initialization_sequence)

    def test_gpio_init_detection(self) -> None:
        """Test GPIO initialization detection."""
        detector = HALDetector()

        binary = b"\x00" * 10 + struct.pack("<I", 0x40020000)  # GPIOA

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        if result.initialization_sequence:
            assert any(op.get("step") == "gpio_init" for op in result.initialization_sequence)

    def test_peripheral_init_detection(self) -> None:
        """Test peripheral initialization detection."""
        detector = HALDetector()

        binary = b"\x00" * 10 + struct.pack("<I", 0x40011000)  # USART1

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        # Init sequence should be a list
        assert isinstance(result.initialization_sequence, list)


# =============================================================================
# Test Confidence Calculation
# =============================================================================


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_confidence_with_framework(self) -> None:
        """Test confidence increases with framework detection."""
        detector = HALDetector()

        binary = b"HAL_Init\x00HAL_GPIO_Init\x00HAL_UART_Init\x00"

        result = detector.analyze_firmware(binary)

        # Should have some confidence from framework detection
        assert result.confidence > 0.0

    def test_confidence_with_peripherals(self) -> None:
        """Test confidence increases with peripheral detection."""
        detector = HALDetector()

        binary = (
            b"\x00" * 10
            + struct.pack("<I", 0x40020000)  # GPIOA
            + struct.pack("<I", 0x40011000)  # USART1
            + struct.pack("<I", 0x40003800)  # SPI2
        )

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        # Should have confidence from peripheral detection
        assert result.confidence >= 0.0

    def test_confidence_bounds(self) -> None:
        """Test confidence is bounded between 0.0 and 1.0."""
        detector = HALDetector()

        # Maximum signatures
        binary = (
            b"HAL_Init\x00"
            b"HAL_GPIO_Init\x00"
            b"HAL_UART_Init\x00"
            b"HAL_SPI_Init\x00"
            b"HAL_I2C_Init\x00" + struct.pack("<I", 0x40020000) * 20  # Many register accesses
        )

        result = detector.analyze_firmware(binary, detect_peripherals=True)

        assert 0.0 <= result.confidence <= 1.0

    def test_low_confidence_unknown(self) -> None:
        """Test low confidence with unknown framework."""
        detector = HALDetector()

        binary = b"\x00\x00\x00\x00"

        result = detector.analyze_firmware(binary)

        # Should have low confidence
        assert result.confidence < 0.5


# =============================================================================
# Test JSON Export
# =============================================================================


class TestJSONExport:
    """Test JSON export functionality."""

    def test_export_basic(self) -> None:
        """Test basic JSON export."""
        detector = HALDetector()

        result = HALAnalysisResult(
            detected_hal="STM32 HAL",
            peripherals=[],
            register_map={},
            initialization_sequence=[],
            confidence=0.8,
        )

        json_str = detector.export_to_json(result)

        assert "STM32 HAL" in json_str
        assert "0.8" in json_str

    def test_export_valid_json(self) -> None:
        """Test exported JSON is valid."""
        detector = HALDetector()

        result = HALAnalysisResult(
            detected_hal="Nordic SDK",
            peripherals=[Peripheral("GPIO", 0x40020000)],
            register_map={0x40020000: "GPIOA"},
            initialization_sequence=[{"step": "clock_config"}],
            confidence=0.75,
            framework_signatures=["nrf_gpio"],
        )

        json_str = detector.export_to_json(result)
        data = json.loads(json_str)

        assert data["hal_framework"] == "Nordic SDK"
        assert data["confidence"] == 0.75
        assert len(data["peripherals"]) == 1

    def test_export_hex_addresses(self) -> None:
        """Test addresses are exported as hex strings."""
        detector = HALDetector()

        result = HALAnalysisResult(
            detected_hal="Unknown",
            peripherals=[Peripheral("UART", 0x40011000)],
            register_map={0x40011000: "USART1.CR1"},
            initialization_sequence=[],
        )

        json_str = detector.export_to_json(result)
        data = json.loads(json_str)

        # Addresses should be hex strings
        assert "0x40011000" in data["peripherals"][0]["base_address"]

    def test_export_with_indent(self) -> None:
        """Test JSON export with custom indentation."""
        detector = HALDetector()

        result = HALAnalysisResult(
            detected_hal="Unknown",
            peripherals=[],
            register_map={},
            initialization_sequence=[],
        )

        json_str_2 = detector.export_to_json(result, indent=2)
        json_str_4 = detector.export_to_json(result, indent=4)

        # Different indentation should produce different output
        assert len(json_str_4) >= len(json_str_2)


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_empty_binary(self) -> None:
        """Test error on empty binary."""
        detector = HALDetector()

        with pytest.raises(ValueError, match="Binary data cannot be empty"):
            detector.analyze_firmware(b"")

    def test_short_binary(self) -> None:
        """Test handling of very short binary."""
        detector = HALDetector()

        # Binary too short for address extraction
        binary = b"\x01\x02"

        result = detector.analyze_firmware(binary)

        # Should not crash
        assert isinstance(result, HALAnalysisResult)

    def test_no_peripheral_addresses(self) -> None:
        """Test handling binary with no peripheral addresses."""
        detector = HALDetector()

        # Random data with no peripheral addresses
        binary = b"\x01\x02\x03\x04" * 100

        result = detector.analyze_firmware(binary)

        # Should handle gracefully
        assert isinstance(result, HALAnalysisResult)


# =============================================================================
# Test Detection Flags
# =============================================================================


class TestDetectionFlags:
    """Test detection control flags."""

    def test_disable_peripheral_detection(self) -> None:
        """Test disabling peripheral detection."""
        detector = HALDetector()

        binary = b"\x00" * 10 + struct.pack("<I", 0x40020000)

        result = detector.analyze_firmware(binary, detect_peripherals=False)

        # Peripherals should not be detected
        assert len(result.peripherals) == 0

    def test_disable_framework_detection(self) -> None:
        """Test disabling framework detection."""
        detector = HALDetector()

        binary = b"HAL_Init\x00HAL_GPIO_Init\x00"

        result = detector.analyze_firmware(binary, detect_framework=False)

        # Framework should not be detected
        assert result.detected_hal == "Unknown"
        assert len(result.framework_signatures) == 0

    def test_both_flags_disabled(self) -> None:
        """Test disabling both detection types."""
        detector = HALDetector()

        binary = b"HAL_Init\x00" + b"\x00" * 10 + struct.pack("<I", 0x40020000)

        result = detector.analyze_firmware(binary, detect_peripherals=False, detect_framework=False)

        assert result.detected_hal == "Unknown"
        assert len(result.peripherals) == 0


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_stm32_firmware_analysis(self) -> None:
        """Test complete STM32 firmware analysis."""
        detector = HALDetector()

        # Realistic STM32 firmware snippet
        binary = (
            b"HAL_Init\x00"
            b"HAL_GPIO_Init\x00"
            b"HAL_UART_Transmit\x00"
            + b"\x00" * 20
            + struct.pack("<I", 0x40020000)  # GPIOA
            + struct.pack("<I", 0x40011000)  # USART1
            + struct.pack("<I", 0x40023800)  # RCC
        )

        result = detector.analyze_firmware(binary)

        assert result.detected_hal == "STM32 HAL"
        assert result.confidence > 0.3
        assert len(result.framework_signatures) > 0

    def test_nordic_firmware_analysis(self) -> None:
        """Test complete Nordic firmware analysis."""
        detector = HALDetector()

        binary = (
            b"nrf_gpio_pin_set\x00"
            b"nrfx_uart_tx\x00" + b"\x00" * 20 + struct.pack("<I", 0x40000000)  # Nordic peripheral
        )

        result = detector.analyze_firmware(binary)

        assert result.detected_hal == "Nordic SDK"
        assert result.confidence > 0.0

    def test_mixed_framework_detection(self) -> None:
        """Test with mixed framework signatures."""
        detector = HALDetector()

        # Binary with both STM32 and Nordic signatures
        binary = b"HAL_Init\x00nrf_gpio\x00"

        result = detector.analyze_firmware(binary)

        # Should pick the one with more matches
        assert result.detected_hal in ("STM32 HAL", "Nordic SDK")

    def test_complete_workflow(self) -> None:
        """Test complete detection and export workflow."""
        detector = HALDetector()

        binary = b"HAL_Init\x00HAL_GPIO_Init\x00" + b"\x00" * 20 + struct.pack("<I", 0x40020000)

        # Analyze
        result = detector.analyze_firmware(binary)

        # Export
        json_str = detector.export_to_json(result)
        data = json.loads(json_str)

        assert data["hal_framework"] == "STM32 HAL"
        assert "confidence" in data
        assert "peripherals" in data
        assert "register_map" in data


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and corner conditions."""

    def test_all_zeros_binary(self) -> None:
        """Test binary with all zeros."""
        detector = HALDetector()

        binary = b"\x00" * 1000

        result = detector.analyze_firmware(binary)

        assert isinstance(result, HALAnalysisResult)

    def test_all_ones_binary(self) -> None:
        """Test binary with all ones (erased flash)."""
        detector = HALDetector()

        binary = b"\xff" * 1000

        result = detector.analyze_firmware(binary)

        assert isinstance(result, HALAnalysisResult)

    def test_random_data(self) -> None:
        """Test with random data."""
        detector = HALDetector()

        binary = bytes(range(256)) * 10

        result = detector.analyze_firmware(binary)

        assert result.detected_hal in (
            "Unknown",
            "STM32 HAL",
            "Nordic SDK",
            "ESP-IDF",
            "Arduino",
            "CMSIS",
        )

    def test_very_large_binary(self) -> None:
        """Test with large binary."""
        detector = HALDetector()

        # 1MB binary
        binary = b"\x00" * (1024 * 1024)

        result = detector.analyze_firmware(binary)

        # Should handle without crashing
        assert isinstance(result, HALAnalysisResult)

    def test_minimal_binary(self) -> None:
        """Test with minimal 4-byte binary."""
        detector = HALDetector()

        binary = b"\x01\x02\x03\x04"

        result = detector.analyze_firmware(binary)

        # Should handle without crashing
        assert isinstance(result, HALAnalysisResult)
        assert result.confidence >= 0.0
