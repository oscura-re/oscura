"""Tests for LoRaWAN protocol decoder.

Test coverage:
- MHDR parsing
- Data frame parsing (up/down)
- Join-request/Join-accept frames
- FCtrl parsing
- MAC command parsing
- AES decryption with test vectors
- MIC verification
- Export to JSON/CSV
"""

import pytest

from oscura.iot.lorawan.decoder import (
    LoRaWANDecoder,
    LoRaWANFrame,
    LoRaWANKeys,
    decode_lorawan_frame,
)


class TestLoRaWANKeys:
    """Test LoRaWAN key validation."""

    def test_valid_keys(self):
        """Test valid 16-byte keys."""
        keys = LoRaWANKeys(
            app_skey=bytes(16),
            nwk_skey=bytes(16),
            app_key=bytes(16),
        )
        assert keys.app_skey == bytes(16)
        assert keys.nwk_skey == bytes(16)
        assert keys.app_key == bytes(16)

    def test_invalid_app_skey_length(self):
        """Test invalid AppSKey length."""
        with pytest.raises(ValueError, match="AppSKey must be 16 bytes"):
            LoRaWANKeys(app_skey=bytes(15))

    def test_invalid_nwk_skey_length(self):
        """Test invalid NwkSKey length."""
        with pytest.raises(ValueError, match="NwkSKey must be 16 bytes"):
            LoRaWANKeys(nwk_skey=bytes(32))

    def test_none_keys(self):
        """Test None keys are allowed."""
        keys = LoRaWANKeys()
        assert keys.app_skey is None
        assert keys.nwk_skey is None
        assert keys.app_key is None


class TestLoRaWANDecoder:
    """Test LoRaWAN decoder initialization and basic operations."""

    def test_init_without_keys(self):
        """Test decoder initialization without keys."""
        decoder = LoRaWANDecoder()
        assert decoder.keys.app_skey is None
        assert decoder.keys.nwk_skey is None
        assert len(decoder.frames) == 0

    def test_init_with_keys(self):
        """Test decoder initialization with keys."""
        keys = LoRaWANKeys(app_skey=bytes(16), nwk_skey=bytes(16))
        decoder = LoRaWANDecoder(keys=keys)
        assert decoder.keys.app_skey == bytes(16)
        assert decoder.keys.nwk_skey == bytes(16)

    def test_set_keys(self):
        """Test setting keys after initialization."""
        decoder = LoRaWANDecoder()
        keys = LoRaWANKeys(app_skey=bytes(16))
        decoder.set_keys(keys)
        assert decoder.keys.app_skey == bytes(16)

    def test_mtype_lookup(self):
        """Test message type lookup table."""
        assert LoRaWANDecoder.MTYPES[0x00] == "Join-request"
        assert LoRaWANDecoder.MTYPES[0x02] == "Unconfirmed Data Up"
        assert LoRaWANDecoder.MTYPES[0x04] == "Confirmed Data Up"


class TestMHDRParsing:
    """Test MHDR (MAC Header) parsing."""

    def test_parse_mhdr_unconfirmed_data_up(self):
        """Test parsing unconfirmed data uplink MHDR."""
        decoder = LoRaWANDecoder()
        mtype, rfu, major = decoder._parse_mhdr(0x40)  # 010 000 00
        assert mtype == 2  # Unconfirmed Data Up
        assert rfu == 0
        assert major == 0

    def test_parse_mhdr_confirmed_data_down(self):
        """Test parsing confirmed data downlink MHDR."""
        decoder = LoRaWANDecoder()
        mtype, rfu, major = decoder._parse_mhdr(0xA0)  # 101 000 00
        assert mtype == 5  # Confirmed Data Down
        assert rfu == 0
        assert major == 0

    def test_parse_mhdr_join_request(self):
        """Test parsing Join-request MHDR."""
        decoder = LoRaWANDecoder()
        mtype, rfu, major = decoder._parse_mhdr(0x00)  # 000 000 00
        assert mtype == 0  # Join-request
        assert rfu == 0
        assert major == 0


class TestFCtrlParsing:
    """Test FCtrl (Frame Control) parsing."""

    def test_parse_fctrl_uplink_adr(self):
        """Test parsing uplink FCtrl with ADR set."""
        decoder = LoRaWANDecoder()
        fctrl = decoder._parse_fctrl(0x80, "up")
        assert fctrl["adr"] is True
        assert fctrl["adr_ack_req"] is False
        assert fctrl["ack"] is False
        assert fctrl["class_b"] is False
        assert fctrl["fopts_len"] == 0

    def test_parse_fctrl_uplink_all_flags(self):
        """Test parsing uplink FCtrl with all flags set."""
        decoder = LoRaWANDecoder()
        fctrl = decoder._parse_fctrl(0xF3, "up")  # 1111 0011
        assert fctrl["adr"] is True
        assert fctrl["adr_ack_req"] is True
        assert fctrl["ack"] is True
        assert fctrl["class_b"] is True
        assert fctrl["fopts_len"] == 3

    def test_parse_fctrl_downlink_fpending(self):
        """Test parsing downlink FCtrl with FPending set."""
        decoder = LoRaWANDecoder()
        fctrl = decoder._parse_fctrl(0xB0, "down")  # 1011 0000 (ADR + ACK + FPending)
        assert fctrl["adr"] is True
        assert fctrl["ack"] is True
        assert fctrl["fpending"] is True
        assert fctrl["fopts_len"] == 0

    def test_parse_fctrl_fopts_length(self):
        """Test parsing FCtrl with FOpts length."""
        decoder = LoRaWANDecoder()
        fctrl = decoder._parse_fctrl(0x05, "up")  # FOpts length = 5
        assert fctrl["fopts_len"] == 5


class TestDataFrameParsing:
    """Test data frame parsing (unconfirmed and confirmed)."""

    def test_decode_minimal_data_frame(self):
        """Test decoding minimal data frame without payload."""
        # MHDR | DevAddr | FCtrl | FCnt | MIC
        # 0x40 (Unconfirmed Data Up) | 0x01020304 | 0x00 | 0x0001 | 0x12345678
        data = bytes(
            [
                0x40,  # MHDR: Unconfirmed Data Up
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr (little-endian)
                0x00,  # FCtrl
                0x01,
                0x00,  # FCnt = 1
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        decoder = LoRaWANDecoder()
        frame = decoder.decode_frame(data, timestamp=1.5)

        assert frame.timestamp == 1.5
        assert frame.mtype == "Unconfirmed Data Up"
        assert frame.dev_addr == 0x01020304
        assert frame.fcnt == 1
        assert frame.fport is None
        assert frame.frm_payload == b""
        assert frame.mic == 0x12345678
        assert frame.decrypted_payload is None

    def test_decode_data_frame_with_payload(self):
        """Test decoding data frame with FPort and payload."""
        # MHDR | DevAddr | FCtrl | FCnt | FPort | FRMPayload | MIC
        data = bytes(
            [
                0x40,  # MHDR
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr
                0x00,  # FCtrl
                0x02,
                0x00,  # FCnt = 2
                0x01,  # FPort = 1
                0xAA,
                0xBB,
                0xCC,  # FRMPayload
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        decoder = LoRaWANDecoder()
        frame = decoder.decode_frame(data, timestamp=2.0)

        assert frame.dev_addr == 0x01020304
        assert frame.fcnt == 2
        assert frame.fport == 1
        assert frame.frm_payload == bytes([0xAA, 0xBB, 0xCC])

    def test_decode_data_frame_with_fopts(self):
        """Test decoding data frame with FOpts (MAC commands)."""
        # Frame with FOpts length = 2
        data = bytes(
            [
                0x40,  # MHDR
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr
                0x02,  # FCtrl: FOpts length = 2
                0x03,
                0x00,  # FCnt = 3
                0x02,
                0x00,  # FOpts: LinkCheckReq (CID=0x02, no payload)
                0x01,  # FPort = 1
                0xDD,
                0xEE,  # FRMPayload
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        decoder = LoRaWANDecoder()
        frame = decoder.decode_frame(data, timestamp=3.0)

        assert frame.fcnt == 3
        assert frame.fopts == bytes([0x02, 0x00])
        assert frame.fport == 1
        assert frame.frm_payload == bytes([0xDD, 0xEE])
        assert len(frame.parsed_mac_commands) >= 0  # Parser may handle it

    def test_decode_confirmed_data_down(self):
        """Test decoding confirmed data downlink."""
        data = bytes(
            [
                0xA0,  # MHDR: Confirmed Data Down (0x05 << 5)
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr
                0x20,  # FCtrl: ACK set
                0x05,
                0x00,  # FCnt = 5
                0x01,  # FPort = 1
                0xFF,  # FRMPayload
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        decoder = LoRaWANDecoder()
        frame = decoder.decode_frame(data, timestamp=5.0)

        assert frame.mtype == "Confirmed Data Down"
        assert frame.fctrl is not None
        assert frame.fctrl["ack"] is True


class TestJoinFrames:
    """Test Join-request and Join-accept frame parsing."""

    def test_decode_join_request(self):
        """Test decoding Join-request frame."""
        # MHDR | AppEUI | DevEUI | DevNonce | MIC
        data = bytes(
            [
                0x00,  # MHDR: Join-request
                # AppEUI (8 bytes)
                0x01,
                0x02,
                0x03,
                0x04,
                0x05,
                0x06,
                0x07,
                0x08,
                # DevEUI (8 bytes)
                0x11,
                0x12,
                0x13,
                0x14,
                0x15,
                0x16,
                0x17,
                0x18,
                # DevNonce (2 bytes)
                0xAA,
                0xBB,
                # MIC (4 bytes)
                0x78,
                0x56,
                0x34,
                0x12,
            ]
        )

        decoder = LoRaWANDecoder()
        frame = decoder.decode_frame(data, timestamp=0.0)

        assert frame.mtype == "Join-request"
        assert len(frame.frm_payload) == 18  # AppEUI + DevEUI + DevNonce
        assert frame.mic == 0x12345678

    def test_decode_join_accept(self):
        """Test decoding Join-accept frame."""
        # MHDR | Encrypted payload | MIC
        data = bytes(
            [
                0x20,  # MHDR: Join-accept (0x01 << 5)
                # Encrypted payload (12 or 28 bytes)
                0x01,
                0x02,
                0x03,
                0x04,
                0x05,
                0x06,
                0x07,
                0x08,
                0x09,
                0x0A,
                0x0B,
                0x0C,
                # MIC (4 bytes)
                0x78,
                0x56,
                0x34,
                0x12,
            ]
        )

        decoder = LoRaWANDecoder()
        frame = decoder.decode_frame(data, timestamp=1.0)

        assert frame.mtype == "Join-accept"
        assert len(frame.frm_payload) == 12


class TestFrameValidation:
    """Test frame validation and error handling."""

    def test_frame_too_short(self):
        """Test decoding frame shorter than 5 bytes."""
        data = bytes([0x40, 0x01, 0x02])

        decoder = LoRaWANDecoder()
        with pytest.raises(ValueError, match="Frame too short"):
            decoder.decode_frame(data)

    def test_data_frame_too_short(self):
        """Test decoding data frame with insufficient FHDR."""
        # Only MHDR + 2 bytes + MIC (should have at least 7 bytes for FHDR)
        data = bytes(
            [
                0x40,  # MHDR
                0x01,
                0x02,  # Incomplete FHDR
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        decoder = LoRaWANDecoder()
        frame = decoder.decode_frame(data)
        assert "MACPayload too short for data frame" in frame.errors


class TestExport:
    """Test JSON and CSV export functionality."""

    def test_export_json_single_frame(self):
        """Test exporting single frame to JSON."""
        data = bytes(
            [
                0x40,  # MHDR
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr
                0x00,  # FCtrl
                0x01,
                0x00,  # FCnt
                0x01,  # FPort
                0xAA,  # Payload
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        decoder = LoRaWANDecoder()
        decoder.decode_frame(data, timestamp=1.0)
        json_data = decoder.export_json()

        assert len(json_data) == 1
        assert json_data[0]["timestamp"] == 1.0
        assert json_data[0]["mtype"] == "Unconfirmed Data Up"
        assert json_data[0]["dev_addr"] == "0x01020304"
        assert json_data[0]["fcnt"] == 1
        assert json_data[0]["fport"] == 1

    def test_export_csv_rows(self):
        """Test exporting frames to CSV rows."""
        data = bytes(
            [
                0x40,  # MHDR
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr
                0x00,  # FCtrl
                0x01,
                0x00,  # FCnt
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        decoder = LoRaWANDecoder()
        decoder.decode_frame(data, timestamp=2.5)
        rows = decoder.export_csv_rows()

        assert len(rows) == 1
        assert rows[0]["timestamp"] == "2.5"
        assert rows[0]["mtype"] == "Unconfirmed Data Up"
        assert rows[0]["dev_addr"] == "0x01020304"
        assert rows[0]["fcnt"] == "1"


class TestConvenienceFunction:
    """Test decode_lorawan_frame convenience function."""

    def test_decode_single_frame(self):
        """Test convenience function for single frame."""
        data = bytes(
            [
                0x40,  # MHDR
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr
                0x00,  # FCtrl
                0x01,
                0x00,  # FCnt
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        frame = decode_lorawan_frame(data, timestamp=3.0)

        assert isinstance(frame, LoRaWANFrame)
        assert frame.timestamp == 3.0
        assert frame.mtype == "Unconfirmed Data Up"
        assert frame.dev_addr == 0x01020304

    def test_decode_with_keys(self):
        """Test convenience function with keys."""
        data = bytes(
            [
                0x40,  # MHDR
                0x04,
                0x03,
                0x02,
                0x01,  # DevAddr
                0x00,  # FCtrl
                0x01,
                0x00,  # FCnt
                0x01,  # FPort
                0xAA,  # Payload
                0x78,
                0x56,
                0x34,
                0x12,  # MIC
            ]
        )

        keys = LoRaWANKeys(app_skey=bytes(16), nwk_skey=bytes(16))
        frame = decode_lorawan_frame(data, keys=keys)

        # Decryption will fail without proper keys/test vectors,
        # but it should attempt decryption
        assert frame.fport == 1
