# Ground Truth JSON Schema Documentation

This document describes the JSON schema for ground truth files used in Oscura validation.

## Overview

Ground truth files provide known-correct answers for validating Oscura's analysis capabilities.
Each test data file should have a corresponding `*_ground_truth.json` or `.json` file.

## Schema Types

### 1. Waveform Ground Truth

Used for analog and digital waveform files (.npz, .wfm).

```json
{
  "file": "waveform_sine_1khz.npz",
  "input_type": "waveform",
  "signal_type": "sine|square|pwm|noise|mixed",
  "expected_domains": ["waveform", "spectral", "statistics"],
  "expected_metrics": {
    "frequency_hz": 1000.0,
    "amplitude_vpp": 2.0,
    "dc_offset": 0.0,
    "sample_rate": 1000000,
    "num_samples": 10000,
    "snr_db": 60.0,
    "thd_percent": 0.01
  },
  "tolerance": {
    "frequency_percent": 1.0,
    "amplitude_percent": 5.0
  }
}
```

### 2. Binary Ground Truth

Used for binary packet and protocol data (.bin).

```json
{
  "file": "structured_packets.bin",
  "input_type": "binary",
  "structure": {
    "packet_size": 256,
    "num_packets": 100,
    "sync_pattern": "aa55aa55",
    "has_sequence": true,
    "has_timestamp": true,
    "has_checksum": true,
    "checksum_type": "crc16",
    "checksum_offset": -2
  },
  "expected_domains": ["entropy", "patterns", "inference"],
  "expected_metrics": {
    "entropy": 5.5,
    "classification": "structured"
  }
}
```

### 3. Digital Signal Ground Truth

Used for VCD and logic analyzer captures (.vcd, .sr).

```json
{
  "file": "digital_uart_115200.vcd",
  "input_type": "digital",
  "protocol": "uart",
  "expected_domains": ["digital", "protocols", "timing"],
  "protocol_params": {
    "baud_rate": 115200,
    "data_bits": 8,
    "stop_bits": 1,
    "parity": "none"
  },
  "decoded_data": {
    "message": "Hello, World!",
    "hex": "48656c6c6f2c20576f726c6421"
  }
}
```

### 4. Message/Packet Ground Truth

Used for protocol messages and packet captures (.bin, .pcap).

```json
{
  "file": "protocol_messages_64b.bin",
  "input_type": "binary",
  "message_format": {
    "header_size": 8,
    "payload_size": 52,
    "footer_size": 4,
    "field_boundaries": [0, 2, 4, 8, 60, 64]
  },
  "expected_domains": ["inference", "packet"],
  "expected_fields": [
    { "name": "sync", "offset": 0, "size": 2, "type": "magic" },
    { "name": "seq", "offset": 2, "size": 2, "type": "counter" },
    { "name": "timestamp", "offset": 4, "size": 4, "type": "timestamp" },
    { "name": "payload", "offset": 8, "size": 52, "type": "data" },
    { "name": "crc", "offset": 60, "size": 4, "type": "checksum" }
  ]
}
```

### 5. PCAP Ground Truth

Used for network captures (.pcap).

```json
{
  "file": "http_session.pcap",
  "input_type": "pcap",
  "packet_count": 15,
  "protocols": ["ethernet", "ip", "tcp", "http"],
  "expected_domains": ["packet", "protocols"],
  "streams": [
    {
      "src_ip": "192.168.1.100",
      "dst_ip": "93.184.216.34",
      "src_port": 12345,
      "dst_port": 80,
      "protocol": "http",
      "requests": 1,
      "responses": 1
    }
  ]
}
```

## Common Fields

All ground truth files should include:

|Field|Type|Required|Description|
|---|---|---|---|
|`file`|string|Yes|Name of the test data file|
|`input_type`|string|Yes|One of: waveform, binary, digital, pcap|
|`expected_domains`|array|Yes|Analysis domains that should produce results|
|`description`|string|No|Human-readable description|

## Validation

Ground truth files are validated against actual analysis results:

```python
from oscura.reporting import analyze
import json

result = analyze("test_file.npz")
truth = json.load(open("test_file.json"))

# Check expected domains are present
for domain in truth["expected_domains"]:
    assert domain in result.domain_summaries

# Check metrics within tolerance
for metric, expected in truth["expected_metrics"].items():
    actual = result.get_metric(metric)
    tolerance = truth.get("tolerance", {}).get(f"{metric}_percent", 5.0) / 100
    assert abs(actual - expected) / expected < tolerance
```

## File Naming Convention

Ground truth files should be named to match their test data file:

- `test_file.npz` -> `test_file.json` or `test_file_ground_truth.json`
- `test_file.bin` -> `test_file.json`
- `test_file.vcd` -> `test_file.json`
- `test_file.pcap` -> `test_file.json`

## Adding New Ground Truth

1. Create test data file in appropriate directory
2. Create JSON ground truth file with same base name
3. Include all required fields
4. Validate by loading the file and checking expected metrics:

   ```python
   from oscura.reporting import analyze
   import json

   result = analyze("test_data/path/to/file")
   truth = json.load(open("test_data/path/to/file.json"))

   # Verify expected domains are present
   for domain in truth["expected_domains"]:
       assert domain in result.domain_summaries
   ```
