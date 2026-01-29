"""Firmware pattern recognition for binary analysis.

This module provides comprehensive firmware analysis capabilities including:
- Function boundary detection using architecture-specific patterns
- Architecture fingerprinting (ARM Thumb/ARM, x86, MIPS)
- String and data region identification
- Interrupt vector table detection
- Compiler signature detection

Example:
    >>> from oscura.hardware.firmware.pattern_recognition import FirmwarePatternRecognizer
    >>> with open("firmware.bin", "rb") as f:
    ...     data = f.read()
    >>> recognizer = FirmwarePatternRecognizer()
    >>> result = recognizer.analyze(data, base_address=0x08000000)
    >>> print(f"Architecture: {result.detected_architecture}")
    >>> print(f"Functions: {len(result.functions)}")
    >>> for func in result.functions[:5]:
    ...     print(f"  {func.address:08X}: {func.name or 'unknown'} ({func.confidence:.2f})")

References:
    ARM Architecture Reference Manual (ARMv7-M)
    Intel 64 and IA-32 Architectures Software Developer's Manual
    MIPS Architecture For Programmers
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class Architecture(Enum):
    """Detected CPU architecture."""

    UNKNOWN = "unknown"
    ARM_THUMB = "arm_thumb"
    ARM_ARM = "arm_arm"
    X86 = "x86"
    X86_64 = "x86_64"
    MIPS = "mips"
    MIPS64 = "mips64"


class CompilerSignature(Enum):
    """Detected compiler toolchain."""

    UNKNOWN = "unknown"
    GCC = "gcc"
    IAR = "iar"
    KEIL = "keil"
    LLVM = "llvm"
    MSVC = "msvc"


@dataclass
class Function:
    """Detected function in firmware.

    Attributes:
        address: Function start address (absolute or offset from base)
        size: Function size in bytes (0 if unknown)
        name: Function name if identifiable (e.g., "reset_handler")
        confidence: Confidence score 0.0-1.0
        architecture: Detected architecture for this function
        metadata: Additional function information
    """

    address: int
    size: int = 0
    name: str | None = None
    confidence: float = 0.0
    architecture: Architecture = Architecture.UNKNOWN
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class StringTable:
    """Detected string table or string region.

    Attributes:
        address: Start address of string table
        size: Size in bytes
        strings: List of decoded strings
        encoding: String encoding (utf-8, ascii, utf-16le)
    """

    address: int
    size: int
    strings: list[str]
    encoding: str = "utf-8"


@dataclass
class InterruptVector:
    """Detected interrupt vector table entry.

    Attributes:
        index: Vector index (0 = reset/stack pointer)
        address: Handler address (or stack pointer for index 0)
        name: Vector name if known (e.g., "SysTick_Handler")
    """

    index: int
    address: int
    name: str | None = None


@dataclass
class FirmwareAnalysisResult:
    """Complete firmware analysis result.

    Attributes:
        detected_architecture: Most likely CPU architecture
        functions: List of detected functions
        string_tables: List of detected string tables
        interrupt_vectors: List of detected interrupt vectors
        compiler_signature: Detected compiler toolchain
        base_address: Base address used for analysis
        firmware_size: Total firmware size in bytes
        code_regions: List of (start, size) tuples for code regions
        data_regions: List of (start, size) tuples for data regions
        metadata: Additional analysis metadata
    """

    detected_architecture: Architecture
    functions: list[Function]
    string_tables: list[StringTable]
    interrupt_vectors: list[InterruptVector]
    compiler_signature: CompilerSignature = CompilerSignature.UNKNOWN
    base_address: int = 0
    firmware_size: int = 0
    code_regions: list[tuple[int, int]] = field(default_factory=list)
    data_regions: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Export analysis result to dictionary.

        Returns:
            Dictionary representation suitable for JSON export
        """
        return {
            "detected_architecture": self.detected_architecture.value,
            "base_address": hex(self.base_address),
            "firmware_size": self.firmware_size,
            "compiler_signature": self.compiler_signature.value,
            "functions": [
                {
                    "address": hex(f.address),
                    "size": f.size,
                    "name": f.name,
                    "confidence": f.confidence,
                    "architecture": f.architecture.value,
                    "metadata": f.metadata,
                }
                for f in self.functions
            ],
            "string_tables": [
                {
                    "address": hex(s.address),
                    "size": s.size,
                    "encoding": s.encoding,
                    "strings": s.strings,
                }
                for s in self.string_tables
            ],
            "interrupt_vectors": [
                {"index": v.index, "address": hex(v.address), "name": v.name}
                for v in self.interrupt_vectors
            ],
            "code_regions": [[hex(addr), size] for addr, size in self.code_regions],
            "data_regions": [[hex(addr), size] for addr, size in self.data_regions],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export analysis result to JSON.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)


class FirmwarePatternRecognizer:
    """Firmware pattern recognition and analysis.

    This class provides comprehensive firmware analysis including function
    detection, architecture fingerprinting, and data region identification.

    Example:
        >>> recognizer = FirmwarePatternRecognizer()
        >>> result = recognizer.analyze(firmware_data, base_address=0x08000000)
        >>> print(result.to_json())
    """

    # ARM Thumb function prologue patterns (16-bit instructions)
    ARM_THUMB_PROLOGUES: ClassVar[list[bytes]] = [
        b"\xb5\x00",  # PUSH {lr} (0xb500)
        b"\xb5\x10",  # PUSH {r4, lr}
        b"\xb5\x30",  # PUSH {r4, r5, lr}
        b"\xb5\x70",  # PUSH {r4, r5, r6, lr}
        b"\xb5\xf0",  # PUSH {r4, r5, r6, r7, lr}
        b"\xb5\x80",  # PUSH {r7, lr}
    ]

    # ARM Thumb function epilogue patterns
    ARM_THUMB_EPILOGUES: ClassVar[list[bytes]] = [
        b"\xbd\x00",  # POP {pc} (0xbd00)
        b"\xbd\x10",  # POP {r4, pc}
        b"\xbd\x30",  # POP {r4, r5, pc}
        b"\xbd\x70",  # POP {r4, r5, r6, pc}
        b"\xbd\xf0",  # POP {r4, r5, r6, r7, pc}
        b"\xbd\x80",  # POP {r7, pc}
        b"\x70\x47",  # BX lr (0x4770)
    ]

    # x86 function prologue patterns
    X86_PROLOGUES: ClassVar[list[bytes]] = [
        b"\x55\x89\xe5",  # PUSH ebp; MOV ebp, esp
        b"\x55\x8b\xec",  # PUSH ebp; MOV ebp, esp (alternate)
        b"\x48\x89\x5c\x24",  # x64: MOV [rsp+X], rbx
        b"\x48\x83\xec",  # x64: SUB rsp, X
    ]

    # x86 function epilogue patterns
    X86_EPILOGUES: ClassVar[list[bytes]] = [
        b"\xc9\xc3",  # LEAVE; RET
        b"\x5d\xc3",  # POP ebp; RET
        b"\xc3",  # RET (simple)
        b"\x48\x83\xc4",  # x64: ADD rsp, X
    ]

    # ARM Cortex-M vector table standard handlers
    CORTEX_M_VECTORS: ClassVar[dict[int, str]] = {
        1: "Reset_Handler",
        2: "NMI_Handler",
        3: "HardFault_Handler",
        4: "MemManage_Handler",
        5: "BusFault_Handler",
        6: "UsageFault_Handler",
        11: "SVC_Handler",
        12: "DebugMon_Handler",
        14: "PendSV_Handler",
        15: "SysTick_Handler",
    }

    def analyze(
        self,
        firmware_data: bytes | Sequence[int],
        base_address: int = 0,
        architecture_hint: Architecture | None = None,
    ) -> FirmwareAnalysisResult:
        """Analyze firmware binary for patterns and structures.

        Args:
            firmware_data: Raw firmware binary data
            base_address: Base address for firmware (default 0)
            architecture_hint: Optional architecture hint to guide analysis

        Returns:
            Complete firmware analysis result

        Raises:
            ValueError: If firmware_data is empty or invalid
        """
        if not firmware_data:
            raise ValueError("Firmware data cannot be empty")

        # Convert to bytes if needed
        data: bytes
        if isinstance(firmware_data, bytes):
            data = firmware_data
        else:
            data = bytes(firmware_data)

        # Detect architecture
        if architecture_hint is None:
            detected_arch = self._detect_architecture(data, base_address)
        else:
            detected_arch = architecture_hint

        # Detect functions
        functions = self._detect_functions(data, base_address, detected_arch)

        # Detect string tables
        string_tables = self._detect_string_tables(data, base_address)

        # Detect interrupt vectors
        interrupt_vectors = self._detect_interrupt_vectors(data, base_address, detected_arch)

        # Detect compiler signature
        compiler_sig = self._detect_compiler(data)

        # Classify code vs data regions
        code_regions, data_regions = self._classify_regions(data, base_address, detected_arch)

        return FirmwareAnalysisResult(
            detected_architecture=detected_arch,
            functions=functions,
            string_tables=string_tables,
            interrupt_vectors=interrupt_vectors,
            compiler_signature=compiler_sig,
            base_address=base_address,
            firmware_size=len(data),
            code_regions=code_regions,
            data_regions=data_regions,
            metadata={
                "function_count": len(functions),
                "string_count": sum(len(st.strings) for st in string_tables),
                "vector_count": len(interrupt_vectors),
            },
        )

    def _detect_architecture(self, firmware_data: bytes, base_address: int) -> Architecture:
        """Detect CPU architecture from binary patterns.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware

        Returns:
            Detected architecture
        """
        # Check for ARM Thumb (16-bit instruction alignment)
        thumb_score = self._score_arm_thumb(firmware_data)

        # Check for ARM (32-bit instruction alignment)
        arm_score = self._score_arm_arm(firmware_data)

        # Check for x86
        x86_score = self._score_x86(firmware_data)

        # Check for MIPS
        mips_score = self._score_mips(firmware_data)

        # Determine architecture based on scores
        scores = {
            Architecture.ARM_THUMB: thumb_score,
            Architecture.ARM_ARM: arm_score,
            Architecture.X86: x86_score,
            Architecture.MIPS: mips_score,
        }

        max_arch = max(scores, key=lambda a: scores[a])
        if scores[max_arch] > 0.3:  # Minimum confidence threshold
            return max_arch

        return Architecture.UNKNOWN

    def _score_arm_thumb(self, firmware_data: bytes) -> float:
        """Score likelihood of ARM Thumb architecture.

        Args:
            firmware_data: Raw firmware binary

        Returns:
            Confidence score 0.0-1.0
        """
        if len(firmware_data) < 32:
            return 0.0

        score = 0.0
        score += self._score_thumb_prologues(firmware_data)
        score += self._score_thumb_epilogues(firmware_data)
        score += self._score_thumb_vector_table(firmware_data)

        return min(score, 1.0)

    def _score_thumb_prologues(self, firmware_data: bytes) -> float:
        """Score Thumb prologue patterns."""
        prologue_count = sum(firmware_data.count(pattern) for pattern in self.ARM_THUMB_PROLOGUES)
        return min(prologue_count / 10.0, 0.4) if prologue_count > 0 else 0.0

    def _score_thumb_epilogues(self, firmware_data: bytes) -> float:
        """Score Thumb epilogue patterns."""
        epilogue_count = sum(firmware_data.count(pattern) for pattern in self.ARM_THUMB_EPILOGUES)
        return min(epilogue_count / 10.0, 0.4) if epilogue_count > 0 else 0.0

    def _score_thumb_vector_table(self, firmware_data: bytes) -> float:
        """Score Thumb LSB bit in vector table."""
        if len(firmware_data) < 64:
            return 0.0

        try:
            thumb_bit_count = 0
            valid_addresses = 0
            for i in range(1, min(16, len(firmware_data) // 4)):
                word = struct.unpack("<I", firmware_data[i * 4 : i * 4 + 4])[0]
                if word != 0 and word != 0xFFFFFFFF:
                    valid_addresses += 1
                    if word & 1:
                        thumb_bit_count += 1

            if valid_addresses >= 3 and thumb_bit_count >= valid_addresses // 2:
                return 0.3
        except struct.error:
            pass

        return 0.0

    def _score_arm_arm(self, firmware_data: bytes) -> float:
        """Score likelihood of ARM (32-bit) architecture.

        Args:
            firmware_data: Raw firmware binary

        Returns:
            Confidence score 0.0-1.0
        """
        if len(firmware_data) < 32:
            return 0.0

        score = 0.0

        # ARM instructions are 32-bit aligned and often have condition codes
        # Check for common ARM instruction patterns (simplified)
        arm_patterns = [
            b"\x1e\xff\x2f\xe1",  # BX lr (0xe12fff1e)
            b"\x00\x00\xa0\xe3",  # MOV r0, #0
        ]

        for pattern in arm_patterns:
            count = firmware_data.count(pattern)
            if count > 0:
                score += min(count / 50.0, 0.3)

        return min(score, 1.0)

    def _score_x86(self, firmware_data: bytes) -> float:
        """Score likelihood of x86 architecture.

        Args:
            firmware_data: Raw firmware binary

        Returns:
            Confidence score 0.0-1.0
        """
        if len(firmware_data) < 32:
            return 0.0

        score = 0.0

        # Check for x86 prologue patterns
        prologue_count = 0
        for pattern in self.X86_PROLOGUES:
            prologue_count += firmware_data.count(pattern)
        if prologue_count > 0:
            score += min(prologue_count / 5.0, 0.5)

        # Check for x86 epilogue patterns
        epilogue_count = 0
        for pattern in self.X86_EPILOGUES:
            epilogue_count += firmware_data.count(pattern)
        if epilogue_count > 0:
            score += min(epilogue_count / 5.0, 0.5)

        return min(score, 1.0)

    def _score_mips(self, firmware_data: bytes) -> float:
        """Score likelihood of MIPS architecture.

        Args:
            firmware_data: Raw firmware binary

        Returns:
            Confidence score 0.0-1.0
        """
        if len(firmware_data) < 32:
            return 0.0

        # MIPS instructions are 32-bit aligned
        # Check for common MIPS patterns (simplified)
        score = 0.0

        # JR ra (return) - 0x03e00008
        jr_ra_count = firmware_data.count(b"\x03\xe0\x00\x08")
        if jr_ra_count > 0:
            score += min(jr_ra_count / 50.0, 0.5)

        return min(score, 1.0)

    def _detect_functions(
        self, firmware_data: bytes, base_address: int, architecture: Architecture
    ) -> list[Function]:
        """Detect function boundaries using pattern matching.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware
            architecture: Detected architecture

        Returns:
            List of detected functions
        """
        functions: list[Function] = []

        if architecture == Architecture.ARM_THUMB:
            functions = self._detect_arm_thumb_functions(firmware_data, base_address)
        elif architecture == Architecture.X86:
            functions = self._detect_x86_functions(firmware_data, base_address)
        elif architecture == Architecture.ARM_ARM:
            functions = self._detect_arm_functions(firmware_data, base_address)

        return functions

    def _detect_arm_thumb_functions(
        self, firmware_data: bytes, base_address: int
    ) -> list[Function]:
        """Detect ARM Thumb function boundaries.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware

        Returns:
            List of detected functions
        """
        functions: list[Function] = []

        # Find all prologue patterns
        prologues: list[tuple[int, bytes]] = []
        for pattern in self.ARM_THUMB_PROLOGUES:
            offset = 0
            while True:
                idx = firmware_data.find(pattern, offset)
                if idx == -1:
                    break
                # ARM Thumb requires 2-byte alignment
                if idx % 2 == 0:
                    prologues.append((idx, pattern))
                offset = idx + 1

        # Find all epilogue patterns
        epilogues: set[int] = set()
        for pattern in self.ARM_THUMB_EPILOGUES:
            offset = 0
            while True:
                idx = firmware_data.find(pattern, offset)
                if idx == -1:
                    break
                if idx % 2 == 0:
                    epilogues.add(idx)
                offset = idx + 1

        # Sort prologues by address
        prologues.sort(key=lambda x: x[0])

        # Match prologues with epilogues
        for i, (prologue_addr, pattern) in enumerate(prologues):
            # Find next epilogue after this prologue
            size = 0
            confidence = 0.5  # Base confidence for pattern match

            # Find closest epilogue
            matching_epilogues = [e for e in epilogues if e > prologue_addr]
            if matching_epilogues:
                epilogue_addr = min(matching_epilogues)
                size = epilogue_addr - prologue_addr + 2
                confidence = 0.7  # Higher confidence with epilogue match

            # Check if next prologue is too close (avoid false positives)
            if i + 1 < len(prologues):
                next_prologue = prologues[i + 1][0]
                if size == 0 or next_prologue < prologue_addr + size:
                    size = next_prologue - prologue_addr

            # Create function entry
            func = Function(
                address=base_address + prologue_addr,
                size=size,
                confidence=confidence,
                architecture=Architecture.ARM_THUMB,
                metadata={"prologue_pattern": pattern.hex()},
            )
            functions.append(func)

        return functions

    def _detect_x86_functions(self, firmware_data: bytes, base_address: int) -> list[Function]:
        """Detect x86 function boundaries.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware

        Returns:
            List of detected functions
        """
        functions: list[Function] = []

        # Find all prologue patterns
        prologues: list[tuple[int, bytes]] = []
        for pattern in self.X86_PROLOGUES:
            offset = 0
            while True:
                idx = firmware_data.find(pattern, offset)
                if idx == -1:
                    break
                prologues.append((idx, pattern))
                offset = idx + 1

        prologues.sort(key=lambda x: x[0])

        for prologue_addr, pattern in prologues:
            func = Function(
                address=base_address + prologue_addr,
                size=0,
                confidence=0.6,
                architecture=Architecture.X86,
                metadata={"prologue_pattern": pattern.hex()},
            )
            functions.append(func)

        return functions

    def _detect_arm_functions(self, firmware_data: bytes, base_address: int) -> list[Function]:
        """Detect ARM (32-bit) function boundaries.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware

        Returns:
            List of detected functions
        """
        functions: list[Function] = []

        # ARM BX lr return pattern
        bx_lr = b"\x1e\xff\x2f\xe1"
        offset = 0
        while True:
            idx = firmware_data.find(bx_lr, offset)
            if idx == -1:
                break
            if idx % 4 == 0:  # ARM requires 4-byte alignment
                # Assume function starts ~32 bytes before return
                func_start = max(0, idx - 32)
                func = Function(
                    address=base_address + func_start,
                    size=idx - func_start + 4,
                    confidence=0.5,
                    architecture=Architecture.ARM_ARM,
                )
                functions.append(func)
            offset = idx + 1

        return functions

    def _detect_string_tables(self, firmware_data: bytes, base_address: int) -> list[StringTable]:
        """Detect string tables and string regions.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware

        Returns:
            List of detected string tables
        """
        string_tables: list[StringTable] = []
        strings: list[str] = []
        current_start = -1
        current_string = bytearray()

        for i, byte in enumerate(firmware_data):
            if 0x20 <= byte <= 0x7E:  # Printable ASCII
                if current_start == -1:
                    current_start = i
                current_string.append(byte)
            elif byte == 0:  # Null terminator
                if len(current_string) >= 4:  # Minimum string length
                    try:
                        decoded = current_string.decode("utf-8")
                        strings.append(decoded)
                    except UnicodeDecodeError:
                        pass
                current_string.clear()
            else:
                # Non-string data
                if len(strings) >= 3:  # Minimum strings per table
                    string_tables.append(
                        StringTable(
                            address=base_address + current_start,
                            size=i - current_start,
                            strings=strings.copy(),
                            encoding="utf-8",
                        )
                    )
                strings.clear()
                current_start = -1
                current_string.clear()

        # Final table
        if len(strings) >= 3:
            string_tables.append(
                StringTable(
                    address=base_address + current_start,
                    size=len(firmware_data) - current_start,
                    strings=strings,
                    encoding="utf-8",
                )
            )

        return string_tables

    def _detect_interrupt_vectors(
        self, firmware_data: bytes, base_address: int, architecture: Architecture
    ) -> list[InterruptVector]:
        """Detect interrupt vector table.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware
            architecture: Detected architecture

        Returns:
            List of detected interrupt vectors
        """
        vectors: list[InterruptVector] = []

        # ARM Cortex-M vector table is at offset 0
        if architecture in (Architecture.ARM_THUMB, Architecture.ARM_ARM):
            if len(firmware_data) >= 64:  # Minimum vector table size
                try:
                    # First word is initial stack pointer
                    stack_ptr = struct.unpack("<I", firmware_data[0:4])[0]
                    vectors.append(
                        InterruptVector(index=0, address=stack_ptr, name="Initial_Stack_Pointer")
                    )

                    # Next words are exception/interrupt handlers
                    for i in range(1, min(16, len(firmware_data) // 4)):
                        handler_addr = struct.unpack("<I", firmware_data[i * 4 : i * 4 + 4])[0]
                        # Validate address (should be in firmware range, Thumb bit set)
                        if handler_addr != 0 and handler_addr != 0xFFFFFFFF:
                            name = self.CORTEX_M_VECTORS.get(i)
                            vectors.append(
                                InterruptVector(index=i, address=handler_addr & ~1, name=name)
                            )
                except struct.error:
                    pass

        return vectors

    def _detect_compiler(self, firmware_data: bytes) -> CompilerSignature:
        """Detect compiler toolchain from binary signatures.

        Args:
            firmware_data: Raw firmware binary

        Returns:
            Detected compiler signature
        """
        # Check for compiler-specific strings
        if b"GCC:" in firmware_data or b"gcc version" in firmware_data:
            return CompilerSignature.GCC
        if b"IAR ANSI C" in firmware_data or b"ICCARM" in firmware_data:
            return CompilerSignature.IAR
        if b"Keil" in firmware_data or b"ARMCC" in firmware_data:
            return CompilerSignature.KEIL
        if b"clang version" in firmware_data or b"LLVM" in firmware_data:
            return CompilerSignature.LLVM
        if b"Microsoft" in firmware_data and b"Visual C++" in firmware_data:
            return CompilerSignature.MSVC

        return CompilerSignature.UNKNOWN

    def _classify_regions(
        self, firmware_data: bytes, base_address: int, architecture: Architecture
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Classify memory regions as code or data using entropy analysis.

        Args:
            firmware_data: Raw firmware binary
            base_address: Base address for firmware
            architecture: Detected architecture

        Returns:
            Tuple of (code_regions, data_regions) as (address, size) pairs
        """
        code_regions: list[tuple[int, int]] = []
        data_regions: list[tuple[int, int]] = []

        if len(firmware_data) < 64:
            return code_regions, data_regions

        # Analyze in 64-byte chunks
        chunk_size = 64
        entropies: list[float] = []

        for i in range(0, len(firmware_data), chunk_size):
            chunk = firmware_data[i : i + chunk_size]
            if len(chunk) < chunk_size:
                break

            # Calculate Shannon entropy
            entropy = self._calculate_entropy(chunk)
            entropies.append(entropy)

        if not entropies:
            return code_regions, data_regions

        # Code typically has medium entropy (4-7 bits/byte)
        # Data/constants have low entropy (<4)
        # Crypto/compressed have high entropy (>7)

        current_type: str | None = None
        region_start = 0

        for i, entropy in enumerate(entropies):
            addr = i * chunk_size

            if 4.0 <= entropy <= 7.0:
                region_type = "code"
            else:
                region_type = "data"

            if current_type is None:
                current_type = region_type
                region_start = addr
            elif current_type != region_type:
                # Region boundary
                size = addr - region_start
                if current_type == "code":
                    code_regions.append((base_address + region_start, size))
                else:
                    data_regions.append((base_address + region_start, size))
                current_type = region_type
                region_start = addr

        # Final region
        if current_type:
            size = len(firmware_data) - region_start
            if current_type == "code":
                code_regions.append((base_address + region_start, size))
            else:
                data_regions.append((base_address + region_start, size))

        return code_regions, data_regions

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte sequence.

        Args:
            data: Byte sequence

        Returns:
            Entropy in bits per byte (0.0-8.0)
        """
        if not data:
            return 0.0

        # Count byte frequencies
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / len(data)

        # Calculate Shannon entropy
        # Avoid log(0) by filtering zero probabilities
        nonzero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))

        return float(entropy)
