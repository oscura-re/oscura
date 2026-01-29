"""Tests for LoRaWAN MAC command parsing.

Test coverage:
- MAC command identification
- LinkCheck command parsing
- LinkADR command parsing
- DevStatus command parsing
- Multi-command parsing from FOpts
- Direction-specific parsing
"""

from oscura.iot.lorawan.mac_commands import (
    MAC_COMMANDS,
    get_mac_command_name,
    parse_mac_command,
    parse_mac_commands,
)


class TestMACCommandLookup:
    """Test MAC command lookup and naming."""

    def test_mac_commands_table(self):
        """Test MAC commands table contains expected entries."""
        assert 0x02 in MAC_COMMANDS
        assert 0x03 in MAC_COMMANDS
        assert 0x06 in MAC_COMMANDS

    def test_get_mac_command_name_uplink(self):
        """Test getting MAC command name for uplink."""
        name = get_mac_command_name(0x02, "up")
        # Should contain "Req" or be "LinkCheckReq"
        assert "LinkCheck" in name

    def test_get_mac_command_name_downlink(self):
        """Test getting MAC command name for downlink."""
        name = get_mac_command_name(0x02, "down")
        # Should contain "Ans" or be "LinkCheckAns"
        assert "LinkCheck" in name

    def test_get_mac_command_name_unknown(self):
        """Test getting name for unknown command."""
        name = get_mac_command_name(0xFF, "up")
        assert "Unknown" in name or "0xFF" in name.upper()


class TestParseMACCommand:
    """Test individual MAC command parsing."""

    def test_parse_linkcheck_req(self):
        """Test parsing LinkCheckReq (uplink, no payload)."""
        result = parse_mac_command(0x02, b"", "up")

        assert result["cid"] == 0x02
        assert result["direction"] == "up"
        assert "LinkCheck" in result["name"]

    def test_parse_linkcheck_ans(self):
        """Test parsing LinkCheckAns (downlink, 2 bytes)."""
        # margin=5, gw_count=3
        payload = bytes([0x05, 0x03])
        result = parse_mac_command(0x02, payload, "down")

        assert result["cid"] == 0x02
        assert result["direction"] == "down"
        assert result.get("margin") == 5
        assert result.get("gw_count") == 3

    def test_parse_linkadr_req(self):
        """Test parsing LinkADRReq (downlink, 4 bytes)."""
        # data_rate_tx_power, ch_mask (2 bytes), redundancy
        payload = bytes([0x50, 0xFF, 0x00, 0x07])
        result = parse_mac_command(0x03, payload, "down")

        assert result["cid"] == 0x03
        assert result["direction"] == "down"
        assert "data_rate_tx_power" in result

    def test_parse_linkadr_ans(self):
        """Test parsing LinkADRAns (uplink, 1 byte)."""
        # status byte: power_ack | data_rate_ack | channel_mask_ack
        status = 0x07  # All bits set
        payload = bytes([status])
        result = parse_mac_command(0x03, payload, "up")

        assert result["cid"] == 0x03
        assert result.get("power_ack") is True
        assert result.get("data_rate_ack") is True
        assert result.get("channel_mask_ack") is True

    def test_parse_devstatus_req(self):
        """Test parsing DevStatusReq (downlink, no payload)."""
        result = parse_mac_command(0x06, b"", "down")

        assert result["cid"] == 0x06
        assert result["direction"] == "down"

    def test_parse_devstatus_ans(self):
        """Test parsing DevStatusAns (uplink, 2 bytes)."""
        # battery=255 (external power), margin=-5 (signed 6-bit)
        battery = 255
        margin = 0x3B  # -5 in 6-bit signed format (59 -> -5 when interpreted)
        payload = bytes([battery, margin])
        result = parse_mac_command(0x06, payload, "up")

        assert result["cid"] == 0x06
        assert result.get("battery") == 255
        assert "margin" in result

    def test_parse_dutycycle_req(self):
        """Test parsing DutyCycleReq (downlink, 1 byte)."""
        max_duty_cycle = 0x0F
        payload = bytes([max_duty_cycle])
        result = parse_mac_command(0x04, payload, "down")

        assert result["cid"] == 0x04
        assert result.get("max_duty_cycle") == 0x0F

    def test_parse_newchannel_req(self):
        """Test parsing NewChannelReq (downlink, 5 bytes)."""
        # ch_index, freq (3 bytes), dr_range
        payload = bytes([0x03, 0x00, 0x4C, 0x86, 0x50])
        result = parse_mac_command(0x07, payload, "down")

        assert result["cid"] == 0x07
        assert "ch_index" in result
        assert "freq" in result
        assert "dr_range" in result

    def test_parse_newchannel_ans(self):
        """Test parsing NewChannelAns (uplink, 1 byte)."""
        status = 0x03  # Both data_rate_range_ok and channel_freq_ok
        payload = bytes([status])
        result = parse_mac_command(0x07, payload, "up")

        assert result["cid"] == 0x07
        assert result.get("data_rate_range_ok") is True
        assert result.get("channel_freq_ok") is True

    def test_parse_unknown_command(self):
        """Test parsing unknown MAC command."""
        result = parse_mac_command(0xFF, bytes([0x01, 0x02]), "up")

        assert result["cid"] == 0xFF
        assert result["direction"] == "up"
        assert result["payload"] == "0102"


class TestParseMACCommands:
    """Test parsing multiple MAC commands from FOpts."""

    def test_parse_single_command(self):
        """Test parsing single MAC command."""
        # LinkCheckReq (CID=0x02, no payload)
        fopts = bytes([0x02])
        commands = parse_mac_commands(fopts, "up")

        assert len(commands) == 1
        assert commands[0]["cid"] == 0x02

    def test_parse_multiple_commands(self):
        """Test parsing multiple MAC commands."""
        # LinkCheckReq (0x02) + DutyCycleAns (0x04, no payload in uplink)
        fopts = bytes([0x02, 0x04])
        commands = parse_mac_commands(fopts, "up")

        assert len(commands) >= 1
        assert commands[0]["cid"] == 0x02

    def test_parse_command_with_payload(self):
        """Test parsing command with payload."""
        # LinkCheckAns (CID=0x02, 2-byte payload: margin, gw_count)
        fopts = bytes([0x02, 0x07, 0x02])  # margin=7, gw_count=2
        commands = parse_mac_commands(fopts, "down")

        assert len(commands) == 1
        assert commands[0]["cid"] == 0x02
        assert commands[0].get("margin") == 7
        assert commands[0].get("gw_count") == 2

    def test_parse_empty_fopts(self):
        """Test parsing empty FOpts."""
        commands = parse_mac_commands(b"", "up")
        assert len(commands) == 0

    def test_parse_incomplete_command(self):
        """Test parsing incomplete command (truncated payload)."""
        # LinkADRReq needs 4 bytes payload, but only provide 2
        # Parser will read what's available then continue to next CID
        fopts = bytes([0x03, 0x01, 0x02])
        commands = parse_mac_commands(fopts, "down")

        # Parser extracts truncated LinkADR with 1 byte, then treats 0x02 as next command
        assert len(commands) == 2
        assert commands[0]["cid"] == 0x03  # LinkADR (incomplete)
        assert commands[1]["cid"] == 0x02  # LinkCheck

    def test_parse_mixed_commands(self):
        """Test parsing mix of commands with and without payloads."""
        # LinkCheckReq (no payload) + LinkCheckAns (2-byte payload)
        # This is unusual but tests robustness
        fopts = bytes([0x02, 0x02, 0x05, 0x03])
        commands = parse_mac_commands(fopts, "up")

        # Should parse at least the first command
        assert len(commands) >= 1


class TestDirectionSpecificParsing:
    """Test direction-specific command parsing."""

    def test_linkcheck_uplink_vs_downlink(self):
        """Test LinkCheck command differs by direction."""
        # Uplink: LinkCheckReq (no payload)
        req = parse_mac_command(0x02, b"", "up")
        assert req["direction"] == "up"

        # Downlink: LinkCheckAns (2-byte payload)
        ans = parse_mac_command(0x02, bytes([0x05, 0x02]), "down")
        assert ans["direction"] == "down"
        assert "margin" in ans

    def test_linkadr_payload_length_differs(self):
        """Test LinkADR has different payloads for uplink/downlink."""
        # Downlink: 4 bytes
        req_payload = bytes([0x50, 0xFF, 0x00, 0x07])
        req = parse_mac_command(0x03, req_payload, "down")
        assert req["direction"] == "down"

        # Uplink: 1 byte
        ans_payload = bytes([0x07])
        ans = parse_mac_command(0x03, ans_payload, "up")
        assert ans["direction"] == "up"

    def test_devstatus_direction_affects_fields(self):
        """Test DevStatus has different fields for uplink/downlink."""
        # Downlink: DevStatusReq (no payload)
        req = parse_mac_command(0x06, b"", "down")
        assert req["direction"] == "down"

        # Uplink: DevStatusAns (2 bytes: battery, margin)
        ans = parse_mac_command(0x06, bytes([0xFF, 0x1F]), "up")
        assert ans["direction"] == "up"
        assert "battery" in ans
        assert "margin" in ans
