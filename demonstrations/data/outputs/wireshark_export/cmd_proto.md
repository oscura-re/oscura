# cmd_proto Protocol

**Version**: 1.0
**Description**: Command and response protocol
**Endianness**: big

## Settings

- **transport**: tcp
- **port**: 8000

## Fields

| Field | Type | Size | Description |
|-------|------|------|-------------|
| sync | bytes | 2 | Sync pattern |
| cmd_code | uint8 | N/A | Command code |
| sequence | uint16 | N/A | Sequence number |
| param_count | uint8 | N/A | Number of parameters |
| params | bytes | param_count | Parameters |
| crc16 | uint16 | N/A | CRC-16 checksum |

### cmd_code Values

| Value | Name |
|-------|------|
| 0x01 | READ_STATUS |
| 0x02 | WRITE_CONFIG |
| 0x03 | RESET |
| 0x04 | GET_VERSION |
