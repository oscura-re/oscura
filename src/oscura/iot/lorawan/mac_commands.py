"""LoRaWAN MAC command parsers.

This module provides parsers for LoRaWAN MAC commands transmitted in the
FOpts field or as port 0 payload as defined in LoRaWAN Specification 1.0.3.

References:
    LoRaWAN Specification 1.0.3 Section 5 - MAC Commands
"""

from __future__ import annotations

from typing import Any

# MAC Command identifiers (CID)
# Note: Same CID is used for Req(uplink) and Ans(downlink) - direction determines meaning
MAC_COMMANDS: dict[int, str] = {
    # Class A commands
    0x02: "LinkCheck",  # LinkCheckReq (up) / LinkCheckAns (down)
    0x03: "LinkADR",  # LinkADRReq (down) / LinkADRAns (up)
    0x04: "DutyCycle",  # DutyCycleReq (down) / DutyCycleAns (up)
    0x05: "RXParamSetup",  # RXParamSetupReq (down) / RXParamSetupAns (up)
    0x06: "DevStatus",  # DevStatusReq (down) / DevStatusAns (up)
    0x07: "NewChannel",  # NewChannelReq (down) / NewChannelAns (up)
    0x08: "RXTimingSetup",  # RXTimingSetupReq (down) / RXTimingSetupAns (up)
    0x09: "TxParamSetup",  # TxParamSetupReq (down) / TxParamSetupAns (up) - LoRaWAN 1.0.2+
    0x0A: "DlChannel",  # DlChannelReq (down) / DlChannelAns (up) - LoRaWAN 1.0.2+
    # Class B commands
    0x10: "PingSlotInfo",  # PingSlotInfoReq (up) / PingSlotInfoAns (down)
    0x11: "PingSlotChannel",  # PingSlotChannelReq (down) / PingSlotChannelAns (up)
    0x12: "BeaconTiming",  # BeaconTimingReq (up) / BeaconTimingAns (down) - deprecated in 1.0.3
    0x13: "BeaconFreq",  # BeaconFreqReq (down) / BeaconFreqAns (up)
}

# MAC command lengths (payload bytes after CID)
# Format: {cid: (uplink_length, downlink_length)}
MAC_COMMAND_LENGTHS: dict[int, tuple[int, int]] = {
    0x02: (0, 2),  # LinkCheck: Req=0, Ans=2 (margin, gw_cnt)
    0x03: (4, 1),  # LinkADR: Req=4, Ans=1 (status)
    0x04: (1, 1),  # DutyCycle: Req=1, Ans=0
    0x05: (4, 1),  # RXParamSetup: Req=4, Ans=1 (status)
    0x06: (0, 2),  # DevStatus: Req=0, Ans=2 (battery, margin)
    0x07: (5, 1),  # NewChannel: Req=5, Ans=1 (status)
    0x08: (1, 0),  # RXTimingSetup: Req=1, Ans=0
    0x09: (1, 0),  # TxParamSetup: Req=1, Ans=0 (LoRaWAN 1.0.2+)
    0x0A: (4, 1),  # DlChannel: Req=4, Ans=1 (LoRaWAN 1.0.2+)
    0x10: (1, 1),  # PingSlotInfo: Req=1, Ans=0
    0x11: (4, 1),  # PingSlotChannel: Req=4, Ans=1
    0x12: (0, 3),  # BeaconTiming: Req=0, Ans=3 (deprecated)
    0x13: (3, 1),  # BeaconFreq: Req=3, Ans=1
}


def parse_mac_command(
    cid: int,
    payload: bytes,
    direction: str,
) -> dict[str, Any]:
    """Parse a LoRaWAN MAC command.

    Args:
        cid: Command identifier (CID).
        payload: Command payload bytes (after CID).
        direction: "up" for uplink, "down" for downlink.

    Returns:
        Dictionary with parsed command fields.

    Example:
        >>> payload = bytes([0x05, 0x03])  # LinkCheckAns: margin=5, gw_count=3
        >>> result = parse_mac_command(0x02, payload, "down")
        >>> result["margin"]
        5
    """
    cmd_name = get_mac_command_name(cid, direction)

    result: dict[str, Any] = {
        "cid": cid,
        "name": cmd_name,
        "direction": direction,
        "payload": payload.hex(),
    }

    # Dispatch to command-specific parser
    parsers = {
        0x02: _parse_link_check,
        0x03: _parse_link_adr,
        0x04: _parse_duty_cycle,
        0x05: _parse_rx_param_setup,
        0x06: _parse_dev_status,
        0x07: _parse_new_channel,
        0x08: _parse_rx_timing_setup,
        0x09: _parse_tx_param_setup,
        0x0A: _parse_dl_channel,
    }

    parser = parsers.get(cid)
    if parser:
        parser(payload, direction, result)

    return result


def _parse_link_check(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse LinkCheck command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 2:  # LinkCheckAns
        result["margin"] = payload[0]
        result["gw_count"] = payload[1]
    # LinkCheckReq has no payload


def _parse_link_adr(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse LinkADR command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 4:  # LinkADRReq
        result["data_rate_tx_power"] = payload[0]
        result["ch_mask"] = int.from_bytes(payload[1:3], "little")
        result["redundancy"] = payload[3]
    elif direction == "up" and len(payload) >= 1:  # LinkADRAns
        status = payload[0]
        result["power_ack"] = bool(status & 0x04)
        result["data_rate_ack"] = bool(status & 0x02)
        result["channel_mask_ack"] = bool(status & 0x01)


def _parse_duty_cycle(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse DutyCycle command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 1:  # DutyCycleReq
        result["max_duty_cycle"] = payload[0]
    # DutyCycleAns has no payload


def _parse_rx_param_setup(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse RXParamSetup command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 4:  # RXParamSetupReq
        result["dl_settings"] = payload[0]
        result["frequency"] = int.from_bytes(payload[1:4], "little")
    elif direction == "up" and len(payload) >= 1:  # RXParamSetupAns
        status = payload[0]
        result["rx1_dr_offset_ack"] = bool(status & 0x04)
        result["rx2_data_rate_ack"] = bool(status & 0x02)
        result["channel_ack"] = bool(status & 0x01)


def _parse_dev_status(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse DevStatus command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "up" and len(payload) >= 2:  # DevStatusAns
        result["battery"] = payload[0]
        margin = payload[1] & 0x3F
        # Convert to signed 6-bit value
        if margin & 0x20:
            margin = margin - 64
        result["margin"] = margin
    # DevStatusReq has no payload


def _parse_new_channel(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse NewChannel command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 5:  # NewChannelReq
        result["ch_index"] = payload[0]
        result["freq"] = int.from_bytes(payload[1:4], "little")
        result["dr_range"] = payload[4]
    elif direction == "up" and len(payload) >= 1:  # NewChannelAns
        status = payload[0]
        result["data_rate_range_ok"] = bool(status & 0x02)
        result["channel_freq_ok"] = bool(status & 0x01)


def _parse_rx_timing_setup(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse RXTimingSetup command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 1:  # RXTimingSetupReq
        result["settings"] = payload[0] & 0x0F
    # RXTimingSetupAns has no payload


def _parse_tx_param_setup(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse TxParamSetup command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 1:  # TxParamSetupReq
        result["eirp_dwell_time"] = payload[0]
    # TxParamSetupAns has no payload


def _parse_dl_channel(payload: bytes, direction: str, result: dict[str, Any]) -> None:
    """Parse DlChannel command.

    Args:
        payload: Command payload bytes.
        direction: "up" or "down".
        result: Result dictionary to update.
    """
    if direction == "down" and len(payload) >= 4:  # DlChannelReq
        result["ch_index"] = payload[0]
        result["freq"] = int.from_bytes(payload[1:4], "little")
    elif direction == "up" and len(payload) >= 1:  # DlChannelAns
        status = payload[0]
        result["uplink_freq_exists"] = bool(status & 0x02)
        result["channel_freq_ok"] = bool(status & 0x01)


def parse_mac_commands(
    fopts: bytes,
    direction: str,
) -> list[dict[str, Any]]:
    """Parse multiple MAC commands from FOpts field.

    Args:
        fopts: FOpts field bytes containing MAC commands.
        direction: "up" for uplink, "down" for downlink.

    Returns:
        List of parsed MAC command dictionaries.

    Example:
        >>> fopts = bytes([0x02, 0x05, 0x03])  # LinkCheckAns
        >>> commands = parse_mac_commands(fopts, "down")
        >>> len(commands)
        1
    """
    commands = []
    idx = 0

    while idx < len(fopts):
        if idx >= len(fopts):
            break

        cid = fopts[idx]
        idx += 1

        # Determine payload length
        length = 0
        if cid in MAC_COMMAND_LENGTHS:
            up_len, down_len = MAC_COMMAND_LENGTHS[cid]
            length = up_len if direction == "up" else down_len

        # Extract payload
        if idx + length <= len(fopts):
            payload = fopts[idx : idx + length]
            idx += length
        else:
            # Insufficient bytes for complete command
            payload = fopts[idx:]
            idx = len(fopts)

        # Parse command
        cmd = parse_mac_command(cid, payload, direction)
        commands.append(cmd)

    return commands


def get_mac_command_name(cid: int, direction: str) -> str:
    """Get MAC command name from CID and direction.

    Args:
        cid: Command identifier.
        direction: "up" for uplink, "down" for downlink.

    Returns:
        Command name string.

    Example:
        >>> get_mac_command_name(0x02, "down")
        'LinkCheckAns'
    """
    base_name = MAC_COMMANDS.get(cid, f"Unknown_0x{cid:02X}")

    # Unknown commands don't get Req/Ans suffix
    if "Unknown" in base_name:
        return base_name

    # Add direction-specific suffix
    # Special cases: commands that originated from different directions
    # Most uplink commands use "Req" (request), downlink use "Req" (command/request)
    # Responses use "Ans" (answer)
    if cid in {0x02, 0x10, 0x12}:  # LinkCheck, PingSlotInfo, BeaconTiming
        # These are initiated by end device (uplink Req, downlink Ans)
        if direction == "up":
            return f"{base_name}Req"
        else:
            return f"{base_name}Ans"
    else:
        # Most commands are initiated by network server (downlink Req, uplink Ans)
        if direction == "down":
            return f"{base_name}Req"
        else:
            return f"{base_name}Ans"


__all__ = [
    "MAC_COMMANDS",
    "MAC_COMMAND_LENGTHS",
    "get_mac_command_name",
    "parse_mac_command",
    "parse_mac_commands",
]
