"""LoRaWAN protocol decoder with MAC layer parsing and payload decryption.

This module provides comprehensive LoRaWAN MAC frame decoding including:
- MAC header (MHDR) parsing
- Frame control (FCtrl) field parsing
- MAC command parsing from FOpts field
- Payload decryption using AES-128 CTR mode
- Message Integrity Code (MIC) verification

References:
    LoRaWAN Specification 1.0.3: https://lora-alliance.org/resource_hub/lorawan-specification-v1-0-3/
    Section 4 - MAC Message Formats
    Section 4.3 - MAC Frame Payload Encryption
    Section 4.4 - Message Integrity Code (MIC)

Example:
    >>> from oscura.iot.lorawan import LoRaWANDecoder, LoRaWANKeys
    >>> keys = LoRaWANKeys(
    ...     app_skey=bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C"),
    ...     nwk_skey=bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C"),
    ... )
    >>> decoder = LoRaWANDecoder(keys=keys)
    >>> frame_bytes = bytes.fromhex("40...")
    >>> frame = decoder.decode_frame(frame_bytes, timestamp=0.0)
    >>> print(f"MType: {frame.mtype}, DevAddr: 0x{frame.dev_addr:08X}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from oscura.iot.lorawan.mac_commands import parse_mac_commands


@dataclass
class LoRaWANKeys:
    """LoRaWAN encryption keys.

    Attributes:
        app_skey: Application session key (16 bytes) for encrypting application data.
        nwk_skey: Network session key (16 bytes) for MIC calculation.
        app_key: Application key (16 bytes) for Join-accept decryption.

    Example:
        >>> keys = LoRaWANKeys(
        ...     app_skey=bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C"),
        ...     nwk_skey=bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C"),
        ... )
    """

    app_skey: bytes | None = None
    nwk_skey: bytes | None = None
    app_key: bytes | None = None

    def __post_init__(self) -> None:
        """Validate key lengths."""
        if self.app_skey is not None and len(self.app_skey) != 16:
            msg = f"AppSKey must be 16 bytes, got {len(self.app_skey)}"
            raise ValueError(msg)
        if self.nwk_skey is not None and len(self.nwk_skey) != 16:
            msg = f"NwkSKey must be 16 bytes, got {len(self.nwk_skey)}"
            raise ValueError(msg)
        if self.app_key is not None and len(self.app_key) != 16:
            msg = f"AppKey must be 16 bytes, got {len(self.app_key)}"
            raise ValueError(msg)


@dataclass
class LoRaWANFrame:
    """LoRaWAN MAC frame representation.

    Attributes:
        timestamp: Frame timestamp in seconds.
        mtype: Message type string (e.g., "Unconfirmed Data Up").
        dev_addr: Device address (4 bytes, optional).
        fctrl: Frame control flags dictionary.
        fcnt: Frame counter.
        fopts: Frame options (MAC commands).
        fport: Application port number.
        frm_payload: Encrypted or plaintext payload.
        mic: Message Integrity Code (32-bit).
        decrypted_payload: Decrypted payload if keys available.
        parsed_mac_commands: Parsed MAC commands from FOpts.
        mic_valid: Whether MIC verification passed (None if not checked).
        errors: List of parsing/validation errors.

    Example:
        >>> frame = LoRaWANFrame(
        ...     timestamp=0.0,
        ...     mtype="Unconfirmed Data Up",
        ...     dev_addr=0x01020304,
        ...     fcnt=1,
        ... )
    """

    timestamp: float
    mtype: str
    dev_addr: int | None = None
    fctrl: dict[str, bool | int] | None = None
    fcnt: int | None = None
    fopts: bytes = b""
    fport: int | None = None
    frm_payload: bytes = b""
    mic: int | None = None
    decrypted_payload: bytes | None = None
    parsed_mac_commands: list[dict[str, Any]] = field(default_factory=list)
    mic_valid: bool | None = None
    errors: list[str] = field(default_factory=list)


class LoRaWANDecoder:
    """LoRaWAN protocol decoder with payload decryption.

    Supports all LoRaWAN message types and provides MAC command parsing
    and optional payload decryption when session keys are provided.

    Attributes:
        MTYPES: Message type lookup table.
        MAJOR_VERSIONS: LoRaWAN version lookup table.

    Example:
        >>> decoder = LoRaWANDecoder()
        >>> frame = decoder.decode_frame(raw_bytes, timestamp=0.0)
        >>> print(f"MType: {frame.mtype}")
    """

    # Message types (MType field in MHDR)
    MTYPES: ClassVar[dict[int, str]] = {
        0x00: "Join-request",
        0x01: "Join-accept",
        0x02: "Unconfirmed Data Up",
        0x03: "Unconfirmed Data Down",
        0x04: "Confirmed Data Up",
        0x05: "Confirmed Data Down",
        0x06: "RFU",
        0x07: "Proprietary",
    }

    # Major version
    MAJOR_VERSIONS: ClassVar[dict[int, str]] = {
        0x00: "LoRaWAN R1",
    }

    def __init__(self, keys: LoRaWANKeys | None = None) -> None:
        """Initialize LoRaWAN decoder with optional encryption keys.

        Args:
            keys: LoRaWAN encryption keys for payload decryption and MIC verification.

        Example:
            >>> keys = LoRaWANKeys(app_skey=bytes(16), nwk_skey=bytes(16))
            >>> decoder = LoRaWANDecoder(keys=keys)
        """
        self.keys = keys or LoRaWANKeys()
        self.frames: list[LoRaWANFrame] = []

    def set_keys(self, keys: LoRaWANKeys) -> None:
        """Set encryption keys for payload decryption.

        Args:
            keys: LoRaWAN encryption keys.

        Example:
            >>> decoder.set_keys(LoRaWANKeys(app_skey=bytes(16)))
        """
        self.keys = keys

    def decode_frame(self, data: bytes, timestamp: float = 0.0) -> LoRaWANFrame:
        """Decode LoRaWAN MAC frame.

        Frame Format:
            MHDR (1 byte) | MACPayload (variable) | MIC (4 bytes)

        Data Frame MACPayload:
            FHDR | FPort (optional) | FRMPayload (optional)

        FHDR Format:
            DevAddr (4 bytes) | FCtrl (1 byte) | FCnt (2 bytes) | FOpts (0-15 bytes)

        Args:
            data: Raw frame bytes.
            timestamp: Frame timestamp in seconds.

        Returns:
            Decoded LoRaWAN frame.

        Raises:
            ValueError: If frame is too short or malformed.

        Example:
            >>> frame = decoder.decode_frame(bytes.fromhex("40..."), timestamp=1.0)
            >>> print(f"DevAddr: 0x{frame.dev_addr:08X}")
        """
        errors: list[str] = []

        # Minimum frame: MHDR (1) + MIC (4) = 5 bytes
        if len(data) < 5:
            msg = f"Frame too short: {len(data)} bytes (minimum 5)"
            raise ValueError(msg)

        # Parse frame components
        mtype, mac_payload, mic = self._extract_frame_components(data)

        # Route to specific decoder based on message type
        return self._route_frame_decoder(mtype, mac_payload, mic, timestamp, data, errors)

    def _extract_frame_components(self, data: bytes) -> tuple[str, bytes, int]:
        """Extract MHDR, MACPayload, and MIC from frame.

        Args:
            data: Raw frame bytes.

        Returns:
            Tuple of (mtype, mac_payload, mic).
        """
        # Parse MHDR (MAC Header)
        mhdr = data[0]
        mtype_val, rfu, major = self._parse_mhdr(mhdr)
        mtype = self.MTYPES.get(mtype_val, f"Unknown_0x{mtype_val:02X}")

        # Extract MIC (last 4 bytes)
        mic = int.from_bytes(data[-4:], "little")

        # Extract MACPayload (between MHDR and MIC)
        mac_payload = data[1:-4]

        return mtype, mac_payload, mic

    def _route_frame_decoder(
        self,
        mtype: str,
        mac_payload: bytes,
        mic: int,
        timestamp: float,
        full_frame: bytes,
        errors: list[str],
    ) -> LoRaWANFrame:
        """Route frame to appropriate decoder based on message type.

        Args:
            mtype: Message type string.
            mac_payload: MACPayload bytes.
            mic: Message Integrity Code.
            timestamp: Frame timestamp.
            full_frame: Complete frame including MHDR and MIC.
            errors: Error list.

        Returns:
            Decoded LoRaWAN frame.
        """
        # Data frames (uplink/downlink)
        if mtype in (
            "Unconfirmed Data Up",
            "Unconfirmed Data Down",
            "Confirmed Data Up",
            "Confirmed Data Down",
        ):
            return self._decode_data_frame(mtype, mac_payload, mic, timestamp, full_frame, errors)

        # Join frames
        if mtype == "Join-request":
            return self._decode_join_request(mac_payload, mic, timestamp, errors)
        if mtype == "Join-accept":
            return self._decode_join_accept(mac_payload, mic, timestamp, errors)

        # Unknown or proprietary frame
        frame = LoRaWANFrame(
            timestamp=timestamp,
            mtype=mtype,
            frm_payload=mac_payload,
            mic=mic,
            errors=errors,
        )
        return frame

    def _parse_mhdr(self, mhdr: int) -> tuple[int, int, int]:
        """Parse MAC header (MHDR) into MType, RFU, Major.

        MHDR format (1 byte):
            Bits 7-5: MType (message type)
            Bits 4-2: RFU (reserved for future use)
            Bits 1-0: Major (LoRaWAN version)

        Args:
            mhdr: MHDR byte value.

        Returns:
            Tuple of (mtype, rfu, major).

        Example:
            >>> mtype, rfu, major = decoder._parse_mhdr(0x40)
            >>> mtype
            2
        """
        mtype = (mhdr >> 5) & 0x07
        rfu = (mhdr >> 2) & 0x07
        major = mhdr & 0x03
        return mtype, rfu, major

    def _parse_fctrl(self, fctrl: int, direction: Literal["up", "down"]) -> dict[str, bool | int]:
        """Parse frame control byte.

        FCtrl format (1 byte):
            Uplink:
                Bit 7: ADR (Adaptive Data Rate)
                Bit 6: ADRACKReq
                Bit 5: ACK
                Bit 4: ClassB
                Bits 3-0: FOptsLen
            Downlink:
                Bit 7: ADR
                Bit 6: RFU
                Bit 5: ACK
                Bit 4: FPending
                Bits 3-0: FOptsLen

        Args:
            fctrl: FCtrl byte value.
            direction: "up" for uplink, "down" for downlink.

        Returns:
            Dictionary of frame control flags.

        Example:
            >>> flags = decoder._parse_fctrl(0x80, "up")
            >>> flags["adr"]
            True
        """
        result: dict[str, bool | int] = {
            "adr": bool(fctrl & 0x80),
            "ack": bool(fctrl & 0x20),
            "fopts_len": fctrl & 0x0F,
        }

        if direction == "up":
            result["adr_ack_req"] = bool(fctrl & 0x40)
            result["class_b"] = bool(fctrl & 0x10)
        else:  # downlink
            result["fpending"] = bool(fctrl & 0x10)

        return result

    def _decode_data_frame(
        self,
        mtype: str,
        mac_payload: bytes,
        mic: int,
        timestamp: float,
        full_frame: bytes,
        errors: list[str],
    ) -> LoRaWANFrame:
        """Decode data frame (unconfirmed or confirmed).

        Args:
            mtype: Message type string.
            mac_payload: MACPayload bytes (FHDR | FPort | FRMPayload).
            mic: Message Integrity Code.
            timestamp: Frame timestamp.
            full_frame: Complete frame including MHDR and MIC.
            errors: Error list to append to.

        Returns:
            Decoded LoRaWAN frame.

        Example:
            >>> decoder = LoRaWANDecoder()
            >>> frame = decoder._decode_data_frame(
            ...     "Unconfirmed Data Up", b"\\x01\\x02\\x03\\x04...", 0x12345678, 0.0, b"...", []
            ... )
        """
        if len(mac_payload) < 7:  # Minimum FHDR length
            errors.append("MACPayload too short for data frame")
            return LoRaWANFrame(
                timestamp=timestamp,
                mtype=mtype,
                mic=mic,
                errors=errors,
            )

        direction: Literal["up", "down"] = "up" if "Up" in mtype else "down"

        # Parse all frame components
        frame_data = self._parse_data_frame_components(
            mac_payload, mtype, full_frame, mic, direction, errors
        )

        # Create and store frame
        frame = LoRaWANFrame(
            timestamp=timestamp,
            mtype=mtype,
            dev_addr=frame_data["dev_addr"],
            fctrl=frame_data["fctrl"],
            fcnt=frame_data["fcnt"],
            fopts=frame_data["fopts"],
            fport=frame_data["fport"],
            frm_payload=frame_data["frm_payload"],
            mic=mic,
            decrypted_payload=frame_data["decrypted_payload"],
            parsed_mac_commands=frame_data["parsed_mac_commands"],
            mic_valid=frame_data["mic_valid"],
            errors=errors,
        )

        self.frames.append(frame)
        return frame

    def _parse_data_frame_components(
        self,
        mac_payload: bytes,
        mtype: str,
        full_frame: bytes,
        mic: int,
        direction: Literal["up", "down"],
        errors: list[str],
    ) -> dict[str, Any]:
        """Parse all components of data frame.

        Args:
            mac_payload: MACPayload bytes.
            mtype: Message type string.
            full_frame: Complete frame.
            mic: Message Integrity Code.
            direction: Frame direction.
            errors: Error list.

        Returns:
            Dictionary of parsed frame components.
        """
        # Parse frame header
        dev_addr, fctrl, fcnt, fopts = self._parse_fhdr(mac_payload, mtype, errors)

        # Extract port and payload
        fport, frm_payload = self._extract_port_and_payload(mac_payload, fopts)

        # Parse MAC commands
        parsed_mac_commands = self._parse_fopts_mac_commands(fopts, direction, errors)

        # Decrypt payload
        decrypted_payload = self._decrypt_frm_payload(
            fport, frm_payload, dev_addr, fcnt, direction, errors
        )

        # Verify MIC
        mic_valid = self._verify_frame_mic(full_frame, mic, dev_addr, fcnt, direction, errors)

        return {
            "dev_addr": dev_addr,
            "fctrl": fctrl,
            "fcnt": fcnt,
            "fopts": fopts,
            "fport": fport,
            "frm_payload": frm_payload,
            "decrypted_payload": decrypted_payload,
            "parsed_mac_commands": parsed_mac_commands,
            "mic_valid": mic_valid,
        }

    def _parse_fhdr(
        self,
        mac_payload: bytes,
        mtype: str,
        errors: list[str],
    ) -> tuple[int, dict[str, bool | int], int, bytes]:
        """Parse frame header (FHDR) fields.

        Args:
            mac_payload: MACPayload bytes.
            mtype: Message type string.
            errors: Error list to append to.

        Returns:
            Tuple of (DevAddr, FCtrl, FCnt, FOpts).
        """
        dev_addr = int.from_bytes(mac_payload[0:4], "little")
        fctrl_byte = mac_payload[4]
        fcnt = int.from_bytes(mac_payload[5:7], "little")

        direction: Literal["up", "down"] = "up" if "Up" in mtype else "down"
        fctrl = self._parse_fctrl(fctrl_byte, direction)

        fopts_len = fctrl["fopts_len"]
        if fopts_len > 15:
            errors.append(f"Invalid FOpts length: {fopts_len}")
            fopts_len = 0

        fopts = mac_payload[7 : 7 + fopts_len] if fopts_len > 0 else b""
        return dev_addr, fctrl, fcnt, fopts

    def _extract_port_and_payload(
        self, mac_payload: bytes, fopts: bytes
    ) -> tuple[int | None, bytes]:
        """Extract FPort and FRMPayload from MAC payload.

        Args:
            mac_payload: MACPayload bytes.
            fopts: FOpts bytes (for calculating offset).

        Returns:
            Tuple of (FPort, FRMPayload). FPort is None if not present.
        """
        offset = 7 + len(fopts)
        fport = None
        frm_payload = b""

        if offset < len(mac_payload):
            fport = mac_payload[offset]
            frm_payload = mac_payload[offset + 1 :] if offset + 1 < len(mac_payload) else b""

        return fport, frm_payload

    def _parse_fopts_mac_commands(
        self,
        fopts: bytes,
        direction: Literal["up", "down"],
        errors: list[str],
    ) -> list[dict[str, Any]]:
        """Parse MAC commands from FOpts field.

        Args:
            fopts: FOpts bytes.
            direction: Message direction ("up" or "down").
            errors: Error list to append to.

        Returns:
            List of parsed MAC command dictionaries.
        """
        if not fopts:
            return []

        try:
            return parse_mac_commands(fopts, direction)
        except Exception as exc:
            errors.append(f"Failed to parse MAC commands: {exc}")
            return []

    def _decrypt_frm_payload(
        self,
        fport: int | None,
        frm_payload: bytes,
        dev_addr: int,
        fcnt: int,
        direction: Literal["up", "down"],
        errors: list[str],
    ) -> bytes | None:
        """Decrypt FRMPayload using AES-128 CTR mode.

        Args:
            fport: FPort value.
            frm_payload: Encrypted FRMPayload.
            dev_addr: Device address.
            fcnt: Frame counter.
            direction: Message direction.
            errors: Error list to append to.

        Returns:
            Decrypted payload bytes, or None if decryption not performed.
        """
        if fport is None or not frm_payload or not self.keys.app_skey:
            return None

        try:
            from oscura.iot.lorawan.crypto import decrypt_payload

            # Use AppSKey for FPort != 0, NwkSKey for FPort == 0
            key = self.keys.nwk_skey if fport == 0 else self.keys.app_skey
            if key:
                return decrypt_payload(frm_payload, key, dev_addr, fcnt, direction)
        except ImportError:
            errors.append("PyCryptodome not available for decryption")
        except Exception as exc:
            errors.append(f"Decryption failed: {exc}")

        return None

    def _verify_frame_mic(
        self,
        full_frame: bytes,
        mic: int,
        dev_addr: int,
        fcnt: int,
        direction: Literal["up", "down"],
        errors: list[str],
    ) -> bool | None:
        """Verify Message Integrity Code (MIC).

        Args:
            full_frame: Complete frame including MHDR and MIC.
            mic: Received MIC value.
            dev_addr: Device address.
            fcnt: Frame counter.
            direction: Message direction.
            errors: Error list to append to.

        Returns:
            True if MIC valid, False if invalid, None if not checked.
        """
        if not self.keys.nwk_skey:
            return None

        try:
            from oscura.iot.lorawan.crypto import verify_mic

            # MIC is computed over MHDR | FHDR | FPort | FRMPayload
            mic_data = full_frame[:-4]
            mic_valid = verify_mic(mic_data, mic, self.keys.nwk_skey, dev_addr, fcnt, direction)
            if not mic_valid:
                errors.append("MIC verification failed")
            return mic_valid
        except ImportError:
            return None  # Crypto not available
        except Exception as exc:
            errors.append(f"MIC verification error: {exc}")
            return None

    def _decode_join_request(
        self,
        mac_payload: bytes,
        mic: int,
        timestamp: float,
        errors: list[str],
    ) -> LoRaWANFrame:
        """Decode Join-request frame.

        Join-request format:
            AppEUI (8 bytes) | DevEUI (8 bytes) | DevNonce (2 bytes)

        Args:
            mac_payload: Join-request payload.
            mic: Message Integrity Code.
            timestamp: Frame timestamp.
            errors: Error list.

        Returns:
            Decoded frame.
        """
        if len(mac_payload) < 18:
            errors.append(f"Join-request too short: {len(mac_payload)} bytes")

        frame = LoRaWANFrame(
            timestamp=timestamp,
            mtype="Join-request",
            frm_payload=mac_payload,
            mic=mic,
            errors=errors,
        )

        self.frames.append(frame)
        return frame

    def _decode_join_accept(
        self,
        mac_payload: bytes,
        mic: int,
        timestamp: float,
        errors: list[str],
    ) -> LoRaWANFrame:
        """Decode Join-accept frame.

        Join-accept is encrypted with AppKey and includes:
            AppNonce (3 bytes) | NetID (3 bytes) | DevAddr (4 bytes) |
            DLSettings (1 byte) | RxDelay (1 byte) | CFList (optional, 16 bytes)

        Args:
            mac_payload: Encrypted Join-accept payload.
            mic: Message Integrity Code.
            timestamp: Frame timestamp.
            errors: Error list.

        Returns:
            Decoded frame.
        """
        # Join-accept decryption requires AppKey
        frame = LoRaWANFrame(
            timestamp=timestamp,
            mtype="Join-accept",
            frm_payload=mac_payload,
            mic=mic,
            errors=errors,
        )

        self.frames.append(frame)
        return frame

    def export_json(self) -> list[dict[str, Any]]:
        """Export decoded frames as JSON-serializable list.

        Returns:
            List of frame dictionaries.

        Example:
            >>> frames_json = decoder.export_json()
            >>> import json
            >>> print(json.dumps(frames_json, indent=2))
        """
        result = []
        for frame in self.frames:
            frame_dict: dict[str, Any] = {
                "timestamp": frame.timestamp,
                "mtype": frame.mtype,
            }

            if frame.dev_addr is not None:
                frame_dict["dev_addr"] = f"0x{frame.dev_addr:08X}"

            if frame.fctrl:
                frame_dict["fctrl"] = frame.fctrl

            if frame.fcnt is not None:
                frame_dict["fcnt"] = frame.fcnt

            if frame.fopts:
                frame_dict["fopts"] = frame.fopts.hex()

            if frame.fport is not None:
                frame_dict["fport"] = frame.fport

            if frame.frm_payload:
                frame_dict["frm_payload"] = frame.frm_payload.hex()

            if frame.decrypted_payload:
                frame_dict["decrypted_payload"] = frame.decrypted_payload.hex()

            if frame.mic is not None:
                frame_dict["mic"] = f"0x{frame.mic:08X}"

            if frame.mic_valid is not None:
                frame_dict["mic_valid"] = frame.mic_valid

            if frame.parsed_mac_commands:
                frame_dict["mac_commands"] = frame.parsed_mac_commands

            if frame.errors:
                frame_dict["errors"] = frame.errors

            result.append(frame_dict)

        return result

    def export_csv_rows(self) -> list[dict[str, str]]:
        """Export decoded frames as CSV rows.

        Returns:
            List of dictionaries suitable for CSV export.

        Example:
            >>> import csv
            >>> rows = decoder.export_csv_rows()
            >>> with open("frames.csv", "w") as f:
            ...     writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            ...     writer.writeheader()
            ...     writer.writerows(rows)
        """
        rows = []
        for frame in self.frames:
            row = {
                "timestamp": str(frame.timestamp),
                "mtype": frame.mtype,
                "dev_addr": f"0x{frame.dev_addr:08X}" if frame.dev_addr else "",
                "fcnt": str(frame.fcnt) if frame.fcnt is not None else "",
                "fport": str(frame.fport) if frame.fport is not None else "",
                "payload_hex": frame.frm_payload.hex(),
                "decrypted_hex": frame.decrypted_payload.hex() if frame.decrypted_payload else "",
                "mic": f"0x{frame.mic:08X}" if frame.mic is not None else "",
                "mic_valid": str(frame.mic_valid) if frame.mic_valid is not None else "",
                "errors": "; ".join(frame.errors),
            }
            rows.append(row)

        return rows


def decode_lorawan_frame(
    data: bytes,
    timestamp: float = 0.0,
    keys: LoRaWANKeys | None = None,
) -> LoRaWANFrame:
    """Convenience function to decode a single LoRaWAN frame.

    Args:
        data: Raw frame bytes.
        timestamp: Frame timestamp in seconds.
        keys: Optional encryption keys for decryption and MIC verification.

    Returns:
        Decoded LoRaWAN frame.

    Example:
        >>> frame = decode_lorawan_frame(bytes.fromhex("40..."))
        >>> print(f"MType: {frame.mtype}")
    """
    decoder = LoRaWANDecoder(keys=keys)
    return decoder.decode_frame(data, timestamp=timestamp)


__all__ = [
    "LoRaWANDecoder",
    "LoRaWANFrame",
    "LoRaWANKeys",
    "decode_lorawan_frame",
]
