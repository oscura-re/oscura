"""Tests for firmware pattern recognition."""

from __future__ import annotations

import json
import struct

import pytest

from oscura.hardware.firmware.pattern_recognition import (
    Architecture,
    CompilerSignature,
    FirmwareAnalysisResult,
    FirmwarePatternRecognizer,
    Function,
    InterruptVector,
    StringTable,
)


class TestFunction:
    """Tests for Function dataclass."""

    def test_function_creation(self) -> None:
        """Test Function dataclass creation."""
        func = Function(
            address=0x08000100,
            size=64,
            name="reset_handler",
            confidence=0.85,
            architecture=Architecture.ARM_THUMB,
        )

        assert func.address == 0x08000100
        assert func.size == 64
        assert func.name == "reset_handler"
        assert func.confidence == 0.85
        assert func.architecture == Architecture.ARM_THUMB
        assert isinstance(func.metadata, dict)

    def test_function_defaults(self) -> None:
        """Test Function with default values."""
        func = Function(address=0x1000)

        assert func.address == 0x1000
        assert func.size == 0
        assert func.name is None
        assert func.confidence == 0.0
        assert func.architecture == Architecture.UNKNOWN
        assert func.metadata == {}

    def test_function_metadata(self) -> None:
        """Test Function metadata storage."""
        func = Function(
            address=0x2000,
            metadata={"prologue": "push_lr", "calls": 3},
        )

        assert func.metadata["prologue"] == "push_lr"
        assert func.metadata["calls"] == 3


class TestStringTable:
    """Tests for StringTable dataclass."""

    def test_string_table_creation(self) -> None:
        """Test StringTable creation."""
        st = StringTable(
            address=0x10000,
            size=256,
            strings=["Hello", "World", "Test"],
            encoding="utf-8",
        )

        assert st.address == 0x10000
        assert st.size == 256
        assert st.strings == ["Hello", "World", "Test"]
        assert st.encoding == "utf-8"

    def test_string_table_empty(self) -> None:
        """Test empty StringTable."""
        st = StringTable(address=0x5000, size=0, strings=[])

        assert st.address == 0x5000
        assert st.size == 0
        assert st.strings == []
        assert st.encoding == "utf-8"


class TestInterruptVector:
    """Tests for InterruptVector dataclass."""

    def test_interrupt_vector_creation(self) -> None:
        """Test InterruptVector creation."""
        vec = InterruptVector(
            index=1,
            address=0x08000200,
            name="Reset_Handler",
        )

        assert vec.index == 1
        assert vec.address == 0x08000200
        assert vec.name == "Reset_Handler"

    def test_interrupt_vector_unnamed(self) -> None:
        """Test unnamed InterruptVector."""
        vec = InterruptVector(index=16, address=0x08001000)

        assert vec.index == 16
        assert vec.address == 0x08001000
        assert vec.name is None


class TestFirmwareAnalysisResult:
    """Tests for FirmwareAnalysisResult dataclass."""

    def test_result_creation(self) -> None:
        """Test FirmwareAnalysisResult creation."""
        result = FirmwareAnalysisResult(
            detected_architecture=Architecture.ARM_THUMB,
            functions=[Function(address=0x100)],
            string_tables=[StringTable(address=0x1000, size=100, strings=["test"])],
            interrupt_vectors=[InterruptVector(index=1, address=0x200)],
            compiler_signature=CompilerSignature.GCC,
            base_address=0x08000000,
            firmware_size=4096,
        )

        assert result.detected_architecture == Architecture.ARM_THUMB
        assert len(result.functions) == 1
        assert len(result.string_tables) == 1
        assert len(result.interrupt_vectors) == 1
        assert result.compiler_signature == CompilerSignature.GCC
        assert result.base_address == 0x08000000
        assert result.firmware_size == 4096

    def test_result_to_dict(self) -> None:
        """Test FirmwareAnalysisResult.to_dict()."""
        result = FirmwareAnalysisResult(
            detected_architecture=Architecture.ARM_THUMB,
            functions=[
                Function(
                    address=0x100,
                    size=64,
                    name="test_func",
                    confidence=0.8,
                    architecture=Architecture.ARM_THUMB,
                )
            ],
            string_tables=[StringTable(address=0x1000, size=50, strings=["hello"])],
            interrupt_vectors=[InterruptVector(index=1, address=0x200, name="Reset")],
            base_address=0x08000000,
            firmware_size=2048,
        )

        d = result.to_dict()

        assert d["detected_architecture"] == "arm_thumb"
        assert d["base_address"] == "0x8000000"
        assert d["firmware_size"] == 2048
        assert len(d["functions"]) == 1
        assert d["functions"][0]["address"] == "0x100"
        assert d["functions"][0]["name"] == "test_func"
        assert d["functions"][0]["confidence"] == 0.8
        assert len(d["string_tables"]) == 1
        assert len(d["interrupt_vectors"]) == 1

    def test_result_to_json(self) -> None:
        """Test FirmwareAnalysisResult.to_json()."""
        result = FirmwareAnalysisResult(
            detected_architecture=Architecture.X86,
            functions=[],
            string_tables=[],
            interrupt_vectors=[],
            base_address=0,
            firmware_size=1024,
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["detected_architecture"] == "x86"
        assert parsed["firmware_size"] == 1024
        assert isinstance(parsed["functions"], list)


class TestFirmwarePatternRecognizer:
    """Tests for FirmwarePatternRecognizer."""

    def test_recognizer_creation(self) -> None:
        """Test FirmwarePatternRecognizer instantiation."""
        recognizer = FirmwarePatternRecognizer()
        assert recognizer is not None

    def test_analyze_empty_data(self) -> None:
        """Test analyze with empty firmware data."""
        recognizer = FirmwarePatternRecognizer()

        with pytest.raises(ValueError, match="cannot be empty"):
            recognizer.analyze(b"")

    def test_analyze_minimal_data(self) -> None:
        """Test analyze with minimal firmware data."""
        recognizer = FirmwarePatternRecognizer()
        data = b"\x00" * 32

        result = recognizer.analyze(data, base_address=0)

        assert isinstance(result, FirmwareAnalysisResult)
        assert result.firmware_size == 32
        assert result.base_address == 0

    def test_analyze_with_list_input(self) -> None:
        """Test analyze with list of integers."""
        recognizer = FirmwarePatternRecognizer()
        data = [0x00] * 64

        result = recognizer.analyze(data, base_address=0x1000)

        assert result.firmware_size == 64
        assert result.base_address == 0x1000

    def test_analyze_with_architecture_hint(self) -> None:
        """Test analyze with architecture hint."""
        recognizer = FirmwarePatternRecognizer()
        data = b"\x00" * 128

        result = recognizer.analyze(data, base_address=0, architecture_hint=Architecture.X86)

        assert result.detected_architecture == Architecture.X86


class TestArchitectureDetection:
    """Tests for architecture detection."""

    def test_detect_arm_thumb(self) -> None:
        """Test ARM Thumb architecture detection."""
        recognizer = FirmwarePatternRecognizer()

        # Create firmware with ARM Thumb patterns
        # Vector table with Thumb bit set + prologue patterns
        data = bytearray(1024)

        # Vector table (Thumb bit set on addresses)
        struct.pack_into("<I", data, 0, 0x20001000)  # Stack pointer
        struct.pack_into("<I", data, 4, 0x08000101)  # Reset handler (Thumb bit)
        struct.pack_into("<I", data, 8, 0x08000201)  # NMI handler (Thumb bit)
        struct.pack_into("<I", data, 12, 0x08000301)  # HardFault (Thumb bit)

        # Add Thumb prologue patterns
        data[0x100:0x102] = b"\xb5\x00"  # PUSH {lr}
        data[0x200:0x202] = b"\xb5\x10"  # PUSH {r4, lr}
        data[0x300:0x302] = b"\xb5\x30"  # PUSH {r4, r5, lr}

        # Add Thumb epilogue patterns
        data[0x150:0x152] = b"\xbd\x00"  # POP {pc}
        data[0x250:0x252] = b"\x70\x47"  # BX lr

        result = recognizer.analyze(bytes(data), base_address=0x08000000)

        assert result.detected_architecture == Architecture.ARM_THUMB

    def test_detect_x86(self) -> None:
        """Test x86 architecture detection."""
        recognizer = FirmwarePatternRecognizer()

        # Create firmware with x86 patterns
        data = bytearray(512)

        # Add x86 prologue patterns
        data[0x00:0x03] = b"\x55\x89\xe5"  # PUSH ebp; MOV ebp, esp
        data[0x20:0x23] = b"\x55\x89\xe5"  # Multiple occurrences
        data[0x40:0x43] = b"\x55\x8b\xec"  # Alternate prologue

        # Add x86 epilogue patterns
        data[0x10:0x12] = b"\xc9\xc3"  # LEAVE; RET
        data[0x30:0x32] = b"\x5d\xc3"  # POP ebp; RET

        result = recognizer.analyze(bytes(data), base_address=0)

        assert result.detected_architecture == Architecture.X86

    def test_detect_unknown_architecture(self) -> None:
        """Test unknown architecture detection."""
        recognizer = FirmwarePatternRecognizer()
        data = b"\x00" * 256  # All zeros, no patterns

        result = recognizer.analyze(data, base_address=0)

        assert result.detected_architecture == Architecture.UNKNOWN


class TestFunctionDetection:
    """Tests for function boundary detection."""

    def test_detect_arm_thumb_functions(self) -> None:
        """Test ARM Thumb function detection."""
        recognizer = FirmwarePatternRecognizer()

        data = bytearray(512)
        # Function 1: prologue at 0x100, epilogue at 0x150
        data[0x100:0x102] = b"\xb5\x00"  # PUSH {lr}
        data[0x150:0x152] = b"\xbd\x00"  # POP {pc}

        # Function 2: prologue at 0x200, epilogue at 0x280
        data[0x200:0x202] = b"\xb5\x10"  # PUSH {r4, lr}
        data[0x280:0x282] = b"\x70\x47"  # BX lr

        result = recognizer.analyze(
            bytes(data), base_address=0x1000, architecture_hint=Architecture.ARM_THUMB
        )

        assert len(result.functions) >= 2
        # Check first function
        func1 = next(f for f in result.functions if f.address == 0x1100)
        assert func1.size > 0
        assert func1.architecture == Architecture.ARM_THUMB
        assert func1.confidence > 0.5

    def test_detect_x86_functions(self) -> None:
        """Test x86 function detection."""
        recognizer = FirmwarePatternRecognizer()

        data = bytearray(256)
        data[0x10:0x13] = b"\x55\x89\xe5"  # PUSH ebp; MOV ebp, esp
        data[0x80:0x83] = b"\x55\x8b\xec"  # Alternate prologue

        result = recognizer.analyze(
            bytes(data), base_address=0x400000, architecture_hint=Architecture.X86
        )

        assert len(result.functions) >= 2
        func = result.functions[0]
        assert func.architecture == Architecture.X86

    def test_function_confidence_scoring(self) -> None:
        """Test function confidence scoring."""
        recognizer = FirmwarePatternRecognizer()

        data = bytearray(256)
        # Prologue with matching epilogue (higher confidence)
        data[0x00:0x02] = b"\xb5\x00"  # PUSH {lr}
        data[0x40:0x42] = b"\xbd\x00"  # POP {pc}

        # Prologue without epilogue (lower confidence)
        data[0x80:0x82] = b"\xb5\x10"  # PUSH {r4, lr}

        result = recognizer.analyze(
            bytes(data), base_address=0, architecture_hint=Architecture.ARM_THUMB
        )

        # Function with epilogue should have higher confidence
        funcs_with_epilogue = [f for f in result.functions if f.size > 0]
        funcs_without = [f for f in result.functions if f.size == 0]

        if funcs_with_epilogue and funcs_without:
            assert funcs_with_epilogue[0].confidence > funcs_without[0].confidence


class TestStringDetection:
    """Tests for string table detection."""

    def test_detect_string_tables(self) -> None:
        """Test string table detection."""
        recognizer = FirmwarePatternRecognizer()

        # Create firmware with string data
        data = bytearray(512)
        strings = [b"Hello\x00", b"World\x00", b"Test\x00", b"String\x00"]
        offset = 0x100
        for s in strings:
            data[offset : offset + len(s)] = s
            offset += len(s)

        result = recognizer.analyze(bytes(data), base_address=0x8000)

        assert len(result.string_tables) > 0
        st = result.string_tables[0]
        assert len(st.strings) >= 3  # Minimum for table
        assert "Hello" in st.strings or "World" in st.strings

    def test_string_minimum_length(self) -> None:
        """Test string minimum length filtering."""
        recognizer = FirmwarePatternRecognizer()

        data = bytearray(256)
        # Short strings (< 4 chars) should be filtered
        data[0x10:0x13] = b"ab\x00"
        data[0x20:0x24] = b"xyz\x00"

        # Long strings should be detected
        data[0x30:0x3A] = b"LongStr\x00"
        data[0x40:0x4A] = b"Another\x00"
        data[0x50:0x5A] = b"Testing\x00"

        result = recognizer.analyze(bytes(data), base_address=0)

        # Should have string table with long strings only
        if result.string_tables:
            for st in result.string_tables:
                for s in st.strings:
                    assert len(s) >= 4

    def test_string_encoding(self) -> None:
        """Test string encoding detection."""
        recognizer = FirmwarePatternRecognizer()

        data = bytearray(256)
        # UTF-8 strings
        strings = [b"Test1\x00", b"Test2\x00", b"Test3\x00", b"Test4\x00"]
        offset = 0x50
        for s in strings:
            data[offset : offset + len(s)] = s
            offset += len(s)

        result = recognizer.analyze(bytes(data), base_address=0)

        if result.string_tables:
            assert result.string_tables[0].encoding == "utf-8"


class TestInterruptVectorDetection:
    """Tests for interrupt vector detection."""

    def test_detect_cortex_m_vectors(self) -> None:
        """Test Cortex-M interrupt vector detection."""
        recognizer = FirmwarePatternRecognizer()

        # Create Cortex-M vector table
        data = bytearray(256)
        struct.pack_into("<I", data, 0, 0x20001000)  # Initial stack pointer
        struct.pack_into("<I", data, 4, 0x08000101)  # Reset handler
        struct.pack_into("<I", data, 8, 0x08000201)  # NMI handler
        struct.pack_into("<I", data, 12, 0x08000301)  # HardFault handler
        struct.pack_into("<I", data, 16, 0x08000401)  # MemManage handler

        result = recognizer.analyze(
            bytes(data), base_address=0x08000000, architecture_hint=Architecture.ARM_THUMB
        )

        assert len(result.interrupt_vectors) > 0
        # Check stack pointer
        sp_vec = next((v for v in result.interrupt_vectors if v.index == 0), None)
        assert sp_vec is not None
        assert sp_vec.address == 0x20001000
        assert sp_vec.name == "Initial_Stack_Pointer"

        # Check reset handler
        reset_vec = next((v for v in result.interrupt_vectors if v.index == 1), None)
        assert reset_vec is not None
        assert reset_vec.address == 0x08000100  # Thumb bit stripped
        assert reset_vec.name == "Reset_Handler"

    def test_vector_thumb_bit_stripping(self) -> None:
        """Test Thumb bit is stripped from handler addresses."""
        recognizer = FirmwarePatternRecognizer()

        data = bytearray(64)
        struct.pack_into("<I", data, 0, 0x20001000)
        struct.pack_into("<I", data, 4, 0x08000101)  # Thumb bit set

        result = recognizer.analyze(
            bytes(data), base_address=0, architecture_hint=Architecture.ARM_THUMB
        )

        reset_vec = next((v for v in result.interrupt_vectors if v.index == 1), None)
        if reset_vec:
            assert reset_vec.address == 0x08000100  # Bit stripped

    def test_invalid_vectors_filtered(self) -> None:
        """Test invalid vector addresses are filtered."""
        recognizer = FirmwarePatternRecognizer()

        data = bytearray(64)
        struct.pack_into("<I", data, 0, 0x20001000)
        struct.pack_into("<I", data, 4, 0x00000000)  # Invalid (zero)
        struct.pack_into("<I", data, 8, 0xFFFFFFFF)  # Invalid (all ones)
        struct.pack_into("<I", data, 12, 0x08000301)  # Valid

        result = recognizer.analyze(
            bytes(data), base_address=0, architecture_hint=Architecture.ARM_THUMB
        )

        # Should only have stack pointer and valid handler
        valid_handlers = [v for v in result.interrupt_vectors if v.index > 0]
        for vec in valid_handlers:
            assert vec.address not in (0, 0xFFFFFFFF)


class TestCompilerDetection:
    """Tests for compiler signature detection."""

    def test_detect_gcc(self) -> None:
        """Test GCC compiler detection."""
        recognizer = FirmwarePatternRecognizer()

        data = b"\x00" * 256 + b"GCC: (GNU Tools for ARM) 9.2.1\x00" + b"\x00" * 256

        result = recognizer.analyze(data, base_address=0)

        assert result.compiler_signature == CompilerSignature.GCC

    def test_detect_iar(self) -> None:
        """Test IAR compiler detection."""
        recognizer = FirmwarePatternRecognizer()

        data = b"\x00" * 128 + b"IAR ANSI C/C++ Compiler\x00" + b"\x00" * 128

        result = recognizer.analyze(data, base_address=0)

        assert result.compiler_signature == CompilerSignature.IAR

    def test_detect_keil(self) -> None:
        """Test Keil compiler detection."""
        recognizer = FirmwarePatternRecognizer()

        data = b"\x00" * 100 + b"ARMCC Version 5.06\x00" + b"\x00" * 100

        result = recognizer.analyze(data, base_address=0)

        assert result.compiler_signature == CompilerSignature.KEIL

    def test_detect_llvm(self) -> None:
        """Test LLVM compiler detection."""
        recognizer = FirmwarePatternRecognizer()

        data = b"\x00" * 150 + b"clang version 10.0.0\x00" + b"\x00" * 150

        result = recognizer.analyze(data, base_address=0)

        assert result.compiler_signature == CompilerSignature.LLVM

    def test_detect_unknown_compiler(self) -> None:
        """Test unknown compiler detection."""
        recognizer = FirmwarePatternRecognizer()

        data = b"\x00" * 256  # No compiler strings

        result = recognizer.analyze(data, base_address=0)

        assert result.compiler_signature == CompilerSignature.UNKNOWN


class TestRegionClassification:
    """Tests for code/data region classification."""

    def test_classify_code_regions(self) -> None:
        """Test code region classification using entropy."""
        recognizer = FirmwarePatternRecognizer()

        # Create firmware with distinct regions
        data = bytearray(512)

        # Code region (medium entropy ~5-6 bits/byte)
        # Mix of instructions and immediate values
        for i in range(128):
            data[i] = (i * 17 + 42) % 256

        # Data region (low entropy - mostly zeros)
        for i in range(256, 384):
            data[i] = 0x00

        result = recognizer.analyze(bytes(data), base_address=0x1000)

        # Should detect both code and data regions
        assert len(result.code_regions) > 0 or len(result.data_regions) > 0

    def test_entropy_calculation(self) -> None:
        """Test entropy calculation for different data types."""
        recognizer = FirmwarePatternRecognizer()

        # Low entropy (all zeros)
        low_entropy_data = b"\x00" * 64
        low_entropy = recognizer._calculate_entropy(low_entropy_data)
        assert low_entropy < 1.0

        # High entropy (random-like)
        high_entropy_data = bytes(range(256))
        high_entropy = recognizer._calculate_entropy(high_entropy_data)
        assert high_entropy > 6.0

        # Medium entropy (mixed)
        # Pattern with varied bytes for actual medium entropy
        medium_entropy_data = bytes(i % 16 for i in range(256))
        medium_entropy = recognizer._calculate_entropy(medium_entropy_data)
        assert 3.0 < medium_entropy < 6.0

    def test_empty_data_entropy(self) -> None:
        """Test entropy calculation with empty data."""
        recognizer = FirmwarePatternRecognizer()

        entropy = recognizer._calculate_entropy(b"")

        assert entropy == 0.0


class TestIntegrationScenarios:
    """Integration tests for complete firmware analysis."""

    def test_complete_arm_firmware_analysis(self) -> None:
        """Test complete ARM firmware analysis workflow."""
        recognizer = FirmwarePatternRecognizer()

        # Create realistic ARM Cortex-M firmware
        firmware = bytearray(2048)

        # Vector table
        struct.pack_into("<I", firmware, 0, 0x20002000)  # Stack
        struct.pack_into("<I", firmware, 4, 0x08000101)  # Reset
        struct.pack_into("<I", firmware, 8, 0x08000201)  # NMI
        struct.pack_into("<I", firmware, 12, 0x08000301)  # HardFault

        # Functions with prologues/epilogues
        firmware[0x100:0x102] = b"\xb5\x00"  # PUSH {lr}
        firmware[0x140:0x142] = b"\xbd\x00"  # POP {pc}

        firmware[0x200:0x202] = b"\xb5\x10"  # PUSH {r4, lr}
        firmware[0x250:0x252] = b"\x70\x47"  # BX lr

        # String data
        strings = b"FirmwareV1\x00Reset\x00Error\x00Ready\x00"
        firmware[0x800 : 0x800 + len(strings)] = strings

        # Compiler signature
        compiler_str = b"GCC: (GNU) 9.3.1\x00"
        firmware[0x900 : 0x900 + len(compiler_str)] = compiler_str

        # Analyze
        result = recognizer.analyze(bytes(firmware), base_address=0x08000000)

        # Verify results
        assert result.detected_architecture in (
            Architecture.ARM_THUMB,
            Architecture.ARM_ARM,
        )
        assert result.firmware_size >= 2048  # May include trailing data
        assert result.base_address == 0x08000000
        assert len(result.functions) >= 2
        assert len(result.interrupt_vectors) >= 2
        assert len(result.string_tables) >= 1
        assert result.compiler_signature == CompilerSignature.GCC

        # Verify metadata
        assert result.metadata["function_count"] >= 2
        assert result.metadata["string_count"] >= 4
        assert result.metadata["vector_count"] >= 2

    def test_x86_firmware_analysis(self) -> None:
        """Test x86 firmware analysis."""
        recognizer = FirmwarePatternRecognizer()

        firmware = bytearray(1024)

        # x86 functions
        firmware[0x100:0x103] = b"\x55\x89\xe5"  # PUSH ebp; MOV ebp, esp
        firmware[0x140:0x142] = b"\xc9\xc3"  # LEAVE; RET

        firmware[0x200:0x203] = b"\x55\x8b\xec"
        firmware[0x250:0x252] = b"\x5d\xc3"  # POP ebp; RET

        # Strings
        strings = b"BIOS v1.0\x00Copyright\x00Vendor\x00"
        firmware[0x500 : 0x500 + len(strings)] = strings

        result = recognizer.analyze(bytes(firmware), base_address=0xF0000)

        assert result.detected_architecture == Architecture.X86
        assert len(result.functions) >= 2
        assert result.base_address == 0xF0000

    def test_json_export_complete(self) -> None:
        """Test complete JSON export."""
        recognizer = FirmwarePatternRecognizer()

        firmware = bytearray(512)
        firmware[0x00:0x02] = b"\xb5\x00"
        firmware[0x40:0x42] = b"\xbd\x00"

        result = recognizer.analyze(bytes(firmware), base_address=0x10000)

        json_str = result.to_json(indent=2)
        parsed = json.loads(json_str)

        # Verify JSON structure
        assert "detected_architecture" in parsed
        assert "functions" in parsed
        assert "string_tables" in parsed
        assert "interrupt_vectors" in parsed
        assert "compiler_signature" in parsed
        assert "base_address" in parsed
        assert "firmware_size" in parsed
        assert "code_regions" in parsed
        assert "data_regions" in parsed
        assert "metadata" in parsed

        # Verify it's valid JSON
        assert isinstance(parsed["functions"], list)
        assert isinstance(parsed["metadata"], dict)

    def test_edge_case_tiny_firmware(self) -> None:
        """Test edge case with very small firmware."""
        recognizer = FirmwarePatternRecognizer()

        tiny_firmware = b"\x00\x01\x02\x03"

        result = recognizer.analyze(tiny_firmware, base_address=0)

        assert result.firmware_size == 4
        assert result.detected_architecture == Architecture.UNKNOWN
        assert len(result.functions) == 0
        assert len(result.interrupt_vectors) == 0

    def test_edge_case_all_zeros(self) -> None:
        """Test edge case with all-zero firmware."""
        recognizer = FirmwarePatternRecognizer()

        zeros = b"\x00" * 1024

        result = recognizer.analyze(zeros, base_address=0)

        assert result.firmware_size == 1024
        assert len(result.functions) == 0
        # Low entropy should classify as data
        if result.data_regions:
            assert sum(size for _, size in result.data_regions) > 0

    def test_mixed_architecture_data(self) -> None:
        """Test firmware with mixed architecture patterns."""
        recognizer = FirmwarePatternRecognizer()

        mixed = bytearray(512)
        # ARM patterns
        mixed[0x00:0x02] = b"\xb5\x00"
        mixed[0x20:0x22] = b"\xbd\x00"
        # x86 patterns
        mixed[0x100:0x103] = b"\x55\x89\xe5"
        mixed[0x140:0x142] = b"\xc9\xc3"

        result = recognizer.analyze(bytes(mixed), base_address=0)

        # Should detect one architecture (whichever has stronger signal)
        assert result.detected_architecture != Architecture.UNKNOWN
        # Should find functions from detected architecture
        assert len(result.functions) > 0
