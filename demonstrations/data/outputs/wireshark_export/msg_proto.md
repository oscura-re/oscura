# msg_proto Protocol

**Version**: 1.0
**Description**: Variable-length message protocol
**Endianness**: little

## Settings

- **transport**: udp
- **port**: 9999

## Fields

| Field | Type | Size | Description |
|-------|------|------|-------------|
| header | uint8 | N/A | Frame header |
| msg_type | uint8 | N/A | Message type |
| payload_length | uint16 | N/A | Payload length |
| payload | bytes | payload_length | Message payload |
| crc8 | uint8 | N/A | CRC-8 checksum |

### msg_type Values

| Value | Name |
|-------|------|
| 0x00 | Heartbeat |
| 0x01 | Request |
| 0x02 | Response |
| 0xFF | Error |
