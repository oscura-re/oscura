# sensor_proto Protocol

**Version**: 1.0
**Description**: Simple sensor data protocol
**Endianness**: big

## Settings

- **transport**: udp
- **port**: 12345

## Fields

| Field | Type | Size | Description |
|-------|------|------|-------------|
| magic | uint16 | N/A | Protocol magic number |
| version | uint8 | N/A | Protocol version |
| sensor_type | uint8 | N/A | Sensor type |
| sensor_id | uint16 | N/A | Sensor device ID |
| timestamp | uint32 | N/A | Unix timestamp |
| value | float32 | N/A | Sensor reading |
| checksum | uint8 | N/A | XOR checksum |

### sensor_type Values

| Value | Name |
|-------|------|
| 0x01 | Temperature |
| 0x02 | Humidity |
| 0x03 | Pressure |
| 0x04 | Light |
