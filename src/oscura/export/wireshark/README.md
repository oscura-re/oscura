# Wireshark Lua Dissector Export

This module generates Wireshark Lua dissectors from Oscura protocol definitions, enabling seamless integration with Wireshark for interactive protocol analysis.

## Features

- **Complete Protocol Support**: Handles all Oscura field types (uint8, uint16, uint32, int8, etc.)
- **Variable-Length Fields**: Supports fields with dynamic lengths based on other fields
- **Endianness Support**: Generates correct big-endian and little-endian field readers
- **Enum Values**: Creates value_string tables for human-readable enum display
- **Transport Registration**: Automatic TCP/UDP port registration
- **Syntax Validation**: Optional Lua syntax validation using `luac`
- **Error Handling**: Generates robust dissectors with malformed packet detection

## Quick Start

```python
from pathlib import Path
from oscura.export.wireshark import WiresharkDissectorGenerator
from oscura.inference.protocol_dsl import ProtocolDefinition, FieldDefinition

# Define your protocol
protocol = ProtocolDefinition(
    name="myproto",
    description="My Custom Protocol",
    settings={"transport": "tcp", "port": 8000},
    fields=[
        FieldDefinition(name="type", field_type="uint8", size=1),
        FieldDefinition(name="length", field_type="uint16", size=2),
        FieldDefinition(name="data", field_type="bytes", size="length"),
    ],
)

# Generate dissector
generator = WiresharkDissectorGenerator()
generator.generate(protocol, Path("myproto.lua"))
```

## Installation in Wireshark

Copy the generated `.lua` file to your Wireshark plugins directory:

**Linux:**

```bash
mkdir -p ~/.local/lib/wireshark/plugins/
cp myproto.lua ~/.local/lib/wireshark/plugins/
```

**macOS:**

```bash
mkdir -p ~/.config/wireshark/plugins/
cp myproto.lua ~/.config/wireshark/plugins/
```

**Windows:**

```powershell
Copy-Item myproto.lua $env:APPDATA\Wireshark\plugins\
```

Then reload Lua plugins in Wireshark: **Analyze > Reload Lua Plugins** (Ctrl+Shift+L)

## Supported Field Types

| Oscura Type | Wireshark Type      | Size     | Notes                     |
| ----------- | ------------------- | -------- | ------------------------- |
| `uint8`     | `ProtoField.uint8`  | 1 byte   | Unsigned 8-bit integer    |
| `uint16`    | `ProtoField.uint16` | 2 bytes  | Unsigned 16-bit integer   |
| `uint32`    | `ProtoField.uint32` | 4 bytes  | Unsigned 32-bit integer   |
| `uint64`    | `ProtoField.uint64` | 8 bytes  | Unsigned 64-bit integer   |
| `int8`      | `ProtoField.int8`   | 1 byte   | Signed 8-bit integer      |
| `int16`     | `ProtoField.int16`  | 2 bytes  | Signed 16-bit integer     |
| `int32`     | `ProtoField.int32`  | 4 bytes  | Signed 32-bit integer     |
| `int64`     | `ProtoField.int64`  | 8 bytes  | Signed 64-bit integer     |
| `float32`   | `ProtoField.float`  | 4 bytes  | IEEE 754 single precision |
| `float64`   | `ProtoField.double` | 8 bytes  | IEEE 754 double precision |
| `bool`      | `ProtoField.bool`   | 1 byte   | Boolean value             |
| `bytes`     | `ProtoField.bytes`  | Variable | Raw byte array            |
| `string`    | `ProtoField.string` | Variable | Text string               |

## Display Bases

The generator automatically selects appropriate display bases:

- **Unsigned integers** (uint\*): Displayed in hexadecimal (`base.HEX`)
- **Signed integers** (int\*): Displayed in decimal (`base.DEC`)
- **Floating point**: No base (`base.NONE`)
- **Strings/bytes**: No base (`base.NONE`)

You can override the display base using the `display_base` parameter.

## Variable-Length Fields

For fields with variable lengths, reference another field's value:

```python
fields=[
    FieldDefinition(name="length", field_type="uint16", size=2),
    FieldDefinition(name="payload", field_type="bytes", size="length"),
]
```

The generator automatically:

1. Reads the `length` field value
2. Uses it to determine `payload` size
3. Validates that enough data is available

## Enum Fields

Create human-readable enum displays:

```python
FieldDefinition(
    name="msg_type",
    field_type="uint8",
    size=1,
    enum={
        0x01: "REQUEST",
        0x02: "RESPONSE",
        0x03: "ERROR",
    }
)
```

Wireshark will display "REQUEST" instead of "0x01" in the packet tree.

## Transport Registration

### TCP Registration

```python
settings={"transport": "tcp", "port": 8000}
```

### UDP Registration

```python
settings={"transport": "udp", "port": 5000}
```

The dissector automatically registers on the specified port. Packets on that port will be decoded automatically.

## Endianness

### Protocol-Level

```python
protocol = ProtocolDefinition(
    name="myproto",
    endian="little",  # All fields default to little-endian
    ...
)
```

### Field-Level

```python
FieldDefinition(
    name="value",
    field_type="uint32",
    size=4,
    endian="little",  # Override protocol endianness
)
```

## Conditional Fields

Fields with conditions are documented in comments (evaluation not yet implemented):

```python
FieldDefinition(
    name="optional_field",
    field_type="uint32",
    size=4,
    condition="flags == 1",  # Only present when flags field equals 1
)
```

## Syntax Validation

Enable automatic Lua syntax validation:

```python
generator = WiresharkDissectorGenerator(validate=True)
generator.generate(protocol, Path("myproto.lua"))
```

Requires `luac` (Lua compiler) to be installed:

- **Linux**: `sudo apt install lua5.3` or `sudo yum install lua`
- **macOS**: `brew install lua`
- **Windows**: Download from [lua.org](https://www.lua.org/download.html)

## Examples

See `examples/05_export/wireshark_dissector_example.py` for complete examples:

- Simple protocol with enum fields
- Modbus-like protocol
- Custom protocol with multiple field types

Run the example:

```bash
python examples/05_export/wireshark_dissector_example.py
```

## API Reference

### WiresharkDissectorGenerator

Main class for generating dissectors.

**Methods:**

- `generate(protocol, output_path)`: Generate dissector to file
- `generate_to_string(protocol)`: Generate dissector as string

### Helper Functions

- `get_protofield_type(field_type, display_base)`: Map field type to ProtoField
- `get_field_size(field_type)`: Get fixed size for field type
- `is_variable_length(field_type)`: Check if field is variable length
- `get_lua_reader_function(field_type, endian)`: Get Lua buffer reader function
- `validate_lua_syntax(lua_code)`: Validate Lua code syntax
- `check_luac_available()`: Check if luac is available

## Limitations

- **Conditional fields**: Comment generated but evaluation not implemented
- **Nested structures**: Not yet supported
- **Arrays**: Not yet supported
- **Heuristic dissectors**: Pattern-based registration not implemented
- **TCP reassembly**: Not yet supported for stream protocols

## Troubleshooting

### Dissector not loading in Wireshark

1. Check Wireshark's Lua console: **Tools > Lua > Evaluate**
2. Verify file is in correct plugins directory
3. Check for syntax errors: `luac -p myproto.lua`
4. Ensure Lua support is enabled in Wireshark build

### Protocol not decoding automatically

1. Verify port registration matches capture
2. Try manual decode: Right-click packet > Decode As > select protocol
3. Check minimum length requirements

### Malformed packet errors

1. Verify field sizes match actual protocol
2. Check variable-length field references are correct
3. Ensure endianness is set correctly

## References

- [Wireshark Lua API](https://wiki.wireshark.org/LuaAPI)
- [Writing Lua Dissectors](https://wiki.wireshark.org/lua/dissectors)
- [ProtoField Reference](https://wiki.wireshark.org/LuaAPI/Proto)
- [DissectorTable Reference](https://wiki.wireshark.org/LuaAPI/DissectorTable)
