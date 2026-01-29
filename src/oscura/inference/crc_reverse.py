"""CRC polynomial reverse engineering.

This module implements CRC parameter recovery from message-CRC pairs using
the XOR differential technique (Greg Ewing method). It can recover CRC
polynomials, initialization values, and other parameters without prior
knowledge of the CRC algorithm.

References:
    - Greg Ewing's CRC Reverse Engineering Essay:
      https://www.csse.canterbury.ac.nz/greg.ewing/essays/CRC-Reverse-Engineering.html
    - CRC RevEng: https://reveng.sourceforge.io/
    - CRC Beagle: https://github.com/colinoflynn/crcbeagle

Example:
    >>> # Capture some messages with CRC-16-CCITT
    >>> messages = [
    ...     (b"Hello", b"\x1a\x2b"),
    ...     (b"World", b"\x3c\x4d"),
    ... ]
    >>> reverser = CRCReverser()
    >>> params = reverser.reverse(messages)
    >>> print(f"Polynomial: 0x{params.polynomial:04x}")
    Polynomial: 0x1021
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CRCParameters", "CRCReverser", "verify_crc"]


@dataclass
class CRCParameters:
    """Recovered CRC parameters.

    Attributes:
        polynomial: CRC polynomial (without leading 1 bit).
        width: CRC width in bits (8, 16, 32, 64).
        init: Initial CRC register value.
        xor_out: Final XOR value applied to CRC.
        reflect_in: Whether input bytes are bit-reflected.
        reflect_out: Whether output CRC is bit-reflected.
        confidence: Confidence score (0.0-1.0) based on validation.
        test_pass_rate: Percentage of test messages that validate correctly.
        algorithm_name: Name of matching standard algorithm (if identified).
    """

    polynomial: int
    width: int
    init: int
    xor_out: int
    reflect_in: bool
    reflect_out: bool
    confidence: float
    test_pass_rate: float = 1.0
    algorithm_name: str | None = None

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"CRCParameters(polynomial=0x{self.polynomial:0{self.width // 4}x}, "
            f"width={self.width}, init=0x{self.init:0{self.width // 4}x}, "
            f"xor_out=0x{self.xor_out:0{self.width // 4}x}, "
            f"reflect_in={self.reflect_in}, reflect_out={self.reflect_out}, "
            f"confidence={self.confidence:.2f})"
        )


# Standard CRC algorithms for identification
STANDARD_CRCS = {
    "CRC-8": {
        "width": 8,
        "poly": 0x07,
        "init": 0x00,
        "xor_out": 0x00,
        "refin": False,
        "refout": False,
    },
    "CRC-8-CCITT": {
        "width": 8,
        "poly": 0x07,
        "init": 0x00,
        "xor_out": 0x00,
        "refin": False,
        "refout": False,
    },
    "CRC-8-MAXIM": {
        "width": 8,
        "poly": 0x31,
        "init": 0x00,
        "xor_out": 0x00,
        "refin": True,
        "refout": True,
    },
    "CRC-16-CCITT": {
        "width": 16,
        "poly": 0x1021,
        "init": 0xFFFF,
        "xor_out": 0x0000,
        "refin": False,
        "refout": False,
    },
    "CRC-16-IBM": {
        "width": 16,
        "poly": 0x8005,
        "init": 0x0000,
        "xor_out": 0x0000,
        "refin": True,
        "refout": True,
    },
    "CRC-16-XMODEM": {
        "width": 16,
        "poly": 0x1021,
        "init": 0x0000,
        "xor_out": 0x0000,
        "refin": False,
        "refout": False,
    },
    "CRC-16-MODBUS": {
        "width": 16,
        "poly": 0x8005,
        "init": 0xFFFF,
        "xor_out": 0x0000,
        "refin": True,
        "refout": True,
    },
    "CRC-32": {
        "width": 32,
        "poly": 0x04C11DB7,
        "init": 0xFFFFFFFF,
        "xor_out": 0xFFFFFFFF,
        "refin": True,
        "refout": True,
    },
    "CRC-32-BZIP2": {
        "width": 32,
        "poly": 0x04C11DB7,
        "init": 0xFFFFFFFF,
        "xor_out": 0xFFFFFFFF,
        "refin": False,
        "refout": False,
    },
}


class CRCReverser:
    """Reverse engineer CRC parameters from message-CRC pairs.

    This class implements the XOR differential technique to recover CRC
    polynomials and parameters from observed message-CRC pairs without
    prior knowledge of the algorithm.

    The algorithm works by:
    1. XORing message pairs to eliminate init/xor_out effects
    2. Brute-forcing polynomial candidates
    3. Determining reflect_in/reflect_out flags
    4. Recovering init and xor_out values
    5. Validating against all messages

    Minimum Requirements:
        - At least 4 message-CRC pairs (more is better)
        - Messages should have varying content
        - All pairs must use the same CRC algorithm
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize CRC reverser.

        Args:
            verbose: Enable verbose logging during analysis.
        """
        self.verbose = verbose

    def reverse(
        self,
        messages: list[tuple[bytes, bytes]],
        width: int | None = None,
    ) -> CRCParameters | None:
        """Reverse engineer CRC parameters from message-CRC pairs.

        Args:
            messages: List of (data, crc) tuples. Minimum 4 pairs required.
            width: CRC width in bits (8, 16, 32). Auto-detect if None.

        Returns:
            Recovered CRC parameters, or None if recovery failed.

        Raises:
            ValueError: If fewer than 4 message pairs provided.

        Example:
            >>> messages = [
            ...     (b"test1", b"\\x12\\x34"),
            ...     (b"test2", b"\\x56\\x78"),
            ...     (b"test3", b"\\x9a\\xbc"),
            ...     (b"test4", b"\\xde\\xf0"),
            ... ]
            >>> reverser = CRCReverser()
            >>> params = reverser.reverse(messages)
        """
        if len(messages) < 4:
            raise ValueError(f"Need at least 4 message pairs, got {len(messages)}")

        # Step 1: Detect CRC width if not provided
        if width is None:
            width = self._detect_width(messages)
            if self.verbose:
                print(f"Detected CRC width: {width} bits")

        # Step 2: Find polynomial using XOR differential
        polynomial = self._find_polynomial(messages, width)
        if polynomial is None:
            return None

        if self.verbose:
            print(f"Found polynomial: 0x{polynomial:0{width // 4}x}")

        # Step 3: Determine reflect_in and reflect_out
        reflect_in, reflect_out = self._find_reflect_flags(messages, polynomial, width)
        if self.verbose:
            print(f"Reflect flags: refin={reflect_in}, refout={reflect_out}")

        # Step 4: Recover init and xor_out
        init, xor_out = self._find_init_xorout(messages, polynomial, width, reflect_in, reflect_out)
        if self.verbose:
            print(f"Init: 0x{init:0{width // 4}x}, XorOut: 0x{xor_out:0{width // 4}x}")

        # Step 5: Validate against all messages
        test_pass_rate = self._validate(
            messages, polynomial, width, init, xor_out, reflect_in, reflect_out
        )
        confidence = test_pass_rate

        # Try to identify standard algorithm
        algorithm_name = self._identify_standard(
            polynomial, width, init, xor_out, reflect_in, reflect_out
        )

        return CRCParameters(
            polynomial=polynomial,
            width=width,
            init=init,
            xor_out=xor_out,
            reflect_in=reflect_in,
            reflect_out=reflect_out,
            confidence=confidence,
            test_pass_rate=test_pass_rate,
            algorithm_name=algorithm_name,
        )

    def _detect_width(self, messages: list[tuple[bytes, bytes]]) -> int:
        """Detect CRC width from CRC field size.

        Args:
            messages: List of (data, crc) tuples.

        Returns:
            Detected width in bits.
        """
        crc_bytes = len(messages[0][1])
        return crc_bytes * 8

    def _generate_differentials(
        self, messages: list[tuple[bytes, bytes]]
    ) -> list[tuple[bytes, bytes]]:
        """Generate XOR differentials from message pairs.

        Args:
            messages: List of (data, crc) tuples

        Returns:
            List of (data_xor, crc_xor) tuples
        """
        differentials = []
        for i in range(len(messages)):
            for j in range(i + 1, len(messages)):
                data1, crc1 = messages[i]
                data2, crc2 = messages[j]

                if len(data1) == len(data2):
                    data_xor = self._xor_bytes(data1, data2)
                    crc_xor = self._xor_bytes(crc1, crc2)
                    differentials.append((data_xor, crc_xor))

        return differentials

    def _test_poly_all_reflections(
        self,
        differentials: list[tuple[bytes, bytes]],
        poly: int,
        width: int,
    ) -> bool:
        """Test polynomial with all reflection combinations.

        Args:
            differentials: XOR differentials
            poly: Polynomial to test
            width: CRC width

        Returns:
            True if polynomial matches with any reflection combination
        """
        for refin in [False, True]:
            for refout in [False, True]:
                if self._test_polynomial(differentials, poly, width, refin, refout):
                    return True
        return False

    def _try_differential_search(
        self,
        differentials: list[tuple[bytes, bytes]],
        width: int,
    ) -> int | None:
        """Try polynomial search using differentials.

        Args:
            differentials: XOR differentials
            width: CRC width

        Returns:
            Polynomial or None
        """
        common_polys = self._get_common_polynomials(width)

        # Try common polynomials first
        for poly in common_polys:
            if self._test_poly_all_reflections(differentials, poly, width):
                return poly

        # For wide CRCs, only try common ones
        if width >= 32:
            return None

        # Brute-force for smaller widths
        max_poly = (1 << width) - 1
        for poly in range(1, max_poly + 1):
            if poly not in common_polys:
                if self._test_poly_all_reflections(differentials, poly, width):
                    return poly

        return None

    def _try_direct_matching(self, messages: list[tuple[bytes, bytes]], width: int) -> int | None:
        """Try direct matching with common polynomials and parameters.

        Args:
            messages: List of (data, crc) tuples
            width: CRC width

        Returns:
            Polynomial or None
        """
        common_polys = self._get_common_polynomials(width)
        max_val = (1 << width) - 1
        common_inits = [i for i in [0x0000, 0xFFFF, 0xFFFFFFFF] if i <= max_val]
        common_xorouts = [x for x in [0x0000, 0xFFFF, 0xFFFFFFFF] if x <= max_val]

        for poly in common_polys:
            for refin in [False, True]:
                for refout in [False, True]:
                    for init in common_inits:
                        for xor_out in common_xorouts:
                            matches = sum(
                                1
                                for data, crc_bytes in messages
                                if self._calculate_crc(
                                    data, poly, width, init, xor_out, refin, refout
                                )
                                == int.from_bytes(crc_bytes, "big")
                            )
                            if matches == len(messages):
                                return poly

        return None

    def _find_polynomial(
        self,
        messages: list[tuple[bytes, bytes]],
        width: int,
    ) -> int | None:
        """Find CRC polynomial using XOR differential technique.

        Args:
            messages: List of (data, crc) tuples.
            width: CRC width in bits.

        Returns:
            Polynomial (without leading 1), or None if not found.
        """
        differentials = self._generate_differentials(messages)

        if len(differentials) >= 2:
            result = self._try_differential_search(differentials, width)
            if result is not None:
                return result

        return self._try_direct_matching(messages, width)

    def _get_common_polynomials(self, width: int) -> list[int]:
        """Get list of common CRC polynomials for a given width.

        Args:
            width: CRC width in bits.

        Returns:
            List of common polynomial values.
        """
        common = {
            8: [0x07, 0x31, 0x9B, 0xD5],
            16: [0x1021, 0x8005, 0x8BB7, 0xA001, 0xC867],
            32: [0x04C11DB7, 0x1EDC6F41, 0x741B8CD7, 0x814141AB],
        }
        return common.get(width, [])

    def _test_polynomial(
        self,
        differentials: list[tuple[bytes, bytes]],
        poly: int,
        width: int,
        refin: bool,
        refout: bool,
    ) -> bool:
        """Test if polynomial matches all differentials.

        Args:
            differentials: List of (data_xor, crc_xor) tuples.
            poly: Polynomial to test.
            width: CRC width in bits.
            refin: Reflect input flag.
            refout: Reflect output flag.

        Returns:
            True if polynomial matches all differentials.
        """
        for data_xor, crc_xor in differentials:
            # Calculate CRC of XORed data with this polynomial
            # Using init=0 and xor_out=0 for differential
            calc_crc = self._calculate_crc(data_xor, poly, width, 0, 0, refin, refout)
            expected_crc = int.from_bytes(crc_xor, "big")

            if calc_crc != expected_crc:
                return False

        return True

    def _find_reflect_flags(
        self,
        messages: list[tuple[bytes, bytes]],
        poly: int,
        width: int,
    ) -> tuple[bool, bool]:
        """Determine reflect_in and reflect_out flags.

        Args:
            messages: List of (data, crc) tuples.
            poly: CRC polynomial.
            width: CRC width in bits.

        Returns:
            Tuple of (reflect_in, reflect_out).
        """
        # Try all reflection combinations with common init/xor_out values
        # and see which combination produces the best results
        max_val = (1 << width) - 1
        common_inits = [0x0000, 0xFFFF, 0xFFFFFFFF]
        common_xorouts = [0x0000, 0xFFFF, 0xFFFFFFFF]

        best_match = (False, False)
        best_score = 0

        for refin in [False, True]:
            for refout in [False, True]:
                # Try common init/xor_out combinations
                for init in common_inits:
                    if init > max_val:
                        continue
                    for xor_out in common_xorouts:
                        if xor_out > max_val:
                            continue

                        score = sum(
                            1
                            for data, crc_bytes in messages
                            if self._calculate_crc(data, poly, width, init, xor_out, refin, refout)
                            == int.from_bytes(crc_bytes, "big")
                        )

                        if score > best_score:
                            best_score = score
                            best_match = (refin, refout)

        return best_match

    def _find_init_xorout(
        self,
        messages: list[tuple[bytes, bytes]],
        poly: int,
        width: int,
        refin: bool,
        refout: bool,
    ) -> tuple[int, int]:
        """Find init and xor_out values.

        Args:
            messages: List of (data, crc) tuples.
            poly: CRC polynomial.
            width: CRC width in bits.
            refin: Reflect input flag.
            refout: Reflect output flag.

        Returns:
            Tuple of (init, xor_out).
        """
        max_val = (1 << width) - 1

        # Common init values to try first
        common_inits = [0x0000, 0xFFFF, 0xFFFFFFFF]
        common_xorouts = [0x0000, 0xFFFF, 0xFFFFFFFF]

        # Try common combinations first
        for init in common_inits:
            if init > max_val:
                continue
            for xor_out in common_xorouts:
                if xor_out > max_val:
                    continue

                matches = sum(
                    1
                    for data, crc_bytes in messages
                    if self._calculate_crc(data, poly, width, init, xor_out, refin, refout)
                    == int.from_bytes(crc_bytes, "big")
                )

                if matches == len(messages):
                    return init, xor_out

        # If common values don't work, brute-force (expensive for 32-bit)
        if width <= 16:
            for init in range(max_val + 1):
                for xor_out in range(max_val + 1):
                    matches = sum(
                        1
                        for data, crc_bytes in messages
                        if self._calculate_crc(data, poly, width, init, xor_out, refin, refout)
                        == int.from_bytes(crc_bytes, "big")
                    )

                    if matches == len(messages):
                        return init, xor_out

        # Default to 0 if not found
        return 0, 0

    def _validate(
        self,
        messages: list[tuple[bytes, bytes]],
        poly: int,
        width: int,
        init: int,
        xor_out: int,
        refin: bool,
        refout: bool,
    ) -> float:
        """Validate CRC parameters against all messages.

        Args:
            messages: List of (data, crc) tuples.
            poly: CRC polynomial.
            width: CRC width in bits.
            init: Initial value.
            xor_out: Final XOR value.
            refin: Reflect input flag.
            refout: Reflect output flag.

        Returns:
            Pass rate (0.0 to 1.0).
        """
        matches = 0
        for data, crc_bytes in messages:
            calc_crc = self._calculate_crc(data, poly, width, init, xor_out, refin, refout)
            expected_crc = int.from_bytes(crc_bytes, "big")
            if calc_crc == expected_crc:
                matches += 1

        return matches / len(messages)

    def _identify_standard(
        self,
        poly: int,
        width: int,
        init: int,
        xor_out: int,
        refin: bool,
        refout: bool,
    ) -> str | None:
        """Identify if parameters match a standard CRC algorithm.

        Args:
            poly: CRC polynomial.
            width: CRC width in bits.
            init: Initial value.
            xor_out: Final XOR value.
            refin: Reflect input flag.
            refout: Reflect output flag.

        Returns:
            Name of standard algorithm, or None if not recognized.
        """
        for name, params in STANDARD_CRCS.items():
            if (
                params["width"] == width
                and params["poly"] == poly
                and params["init"] == init
                and params["xor_out"] == xor_out
                and params["refin"] == refin
                and params["refout"] == refout
            ):
                return name
        return None

    def _calculate_crc(
        self,
        data: bytes,
        poly: int,
        width: int,
        init: int,
        xor_out: int,
        refin: bool,
        refout: bool,
    ) -> int:
        """Calculate CRC with given parameters.

        Args:
            data: Input data.
            poly: CRC polynomial.
            width: CRC width in bits.
            init: Initial value.
            xor_out: Final XOR value.
            refin: Reflect input bytes.
            refout: Reflect output CRC.

        Returns:
            Calculated CRC value.
        """
        crc = init
        mask = (1 << width) - 1

        for byte in data:
            if refin:
                byte = self._reflect_byte(byte)

            crc ^= byte << (width - 8)

            for _ in range(8):
                if crc & (1 << (width - 1)):
                    crc = (crc << 1) ^ poly
                else:
                    crc = crc << 1

            crc &= mask

        if refout:
            crc = self._reflect(crc, width)

        return crc ^ xor_out

    def _reflect_byte(self, byte: int) -> int:
        """Reflect bits in a byte.

        Args:
            byte: Input byte (0-255).

        Returns:
            Bit-reflected byte.
        """
        result = 0
        for i in range(8):
            if byte & (1 << i):
                result |= 1 << (7 - i)
        return result

    def _reflect(self, value: int, width: int) -> int:
        """Reflect bits in a value.

        Args:
            value: Input value.
            width: Number of bits to reflect.

        Returns:
            Bit-reflected value.
        """
        result = 0
        for i in range(width):
            if value & (1 << i):
                result |= 1 << (width - 1 - i)
        return result

    def _xor_bytes(self, b1: bytes, b2: bytes) -> bytes:
        """XOR two byte sequences.

        Args:
            b1: First byte sequence.
            b2: Second byte sequence.

        Returns:
            XORed result (length = max(len(b1), len(b2))).
        """
        # Pad shorter sequence with zeros
        max_len = max(len(b1), len(b2))
        b1_padded = b1.ljust(max_len, b"\x00")
        b2_padded = b2.ljust(max_len, b"\x00")

        return bytes(a ^ b for a, b in zip(b1_padded, b2_padded, strict=True))


def verify_crc(
    data: bytes,
    crc: bytes,
    params: CRCParameters,
) -> bool:
    """Verify if CRC is correct for given data.

    Args:
        data: Input data.
        crc: Expected CRC value.
        params: CRC parameters.

    Returns:
        True if CRC is correct.

    Example:
        >>> params = CRCParameters(
        ...     polynomial=0x1021,
        ...     width=16,
        ...     init=0xFFFF,
        ...     xor_out=0x0000,
        ...     reflect_in=False,
        ...     reflect_out=False,
        ...     confidence=1.0,
        ... )
        >>> verify_crc(b"Hello", b"\\x1a\\x2b", params)
        True
    """
    reverser = CRCReverser()
    calc_crc = reverser._calculate_crc(
        data,
        params.polynomial,
        params.width,
        params.init,
        params.xor_out,
        params.reflect_in,
        params.reflect_out,
    )
    expected_crc = int.from_bytes(crc, "big")
    return calc_crc == expected_crc
