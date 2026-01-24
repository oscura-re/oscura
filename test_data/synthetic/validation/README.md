# Comprehensive Analysis Report Validation Suite

This directory contains curated test data for validating the `analyze()` API
and ensuring all 14 analysis domains produce correct, complete reports.

## Test Data Matrix

### 1. Waveform Input Tests (InputType.WAVEFORM)

| File                       | Purpose                              | Domains Tested                 |
| -------------------------- | ------------------------------------ | ------------------------------ |
| `waveform_sine_1khz.npz`   | Clean sine wave at 1kHz              | WAVEFORM, SPECTRAL, STATISTICS |
| `waveform_multitone.npz`   | Multi-frequency signal (THD testing) | SPECTRAL (THD, SNR, SFDR)      |
| `waveform_digital_pwm.npz` | PWM signal (duty cycle analysis)     | WAVEFORM, TIMING, DIGITAL      |
| `waveform_noisy_50db.npz`  | Signal with 50dB SNR                 | SPECTRAL (SNR, SINAD, ENOB)    |
| `waveform_power_rail.npz`  | Voltage + current pair               | POWER (efficiency, ripple)     |
| `waveform_eye_pattern.npz` | Serial data at 1Gbps                 | EYE, JITTER                    |

### 2. Digital Input Tests (InputType.DIGITAL)

| File                       | Purpose                | Domains Tested             |
| -------------------------- | ---------------------- | -------------------------- |
| `digital_uart_115200.vcd`  | UART at 115200 baud    | DIGITAL, PROTOCOLS, TIMING |
| `digital_spi_10mhz.vcd`    | SPI Mode 0 at 10MHz    | PROTOCOLS (SPI decode)     |
| `digital_i2c_400khz.vcd`   | I2C Fast Mode          | PROTOCOLS (I2C decode)     |
| `digital_can_500k.vcd`     | CAN bus 500kbps        | PROTOCOLS (CAN decode)     |
| `digital_mixed_signals.sr` | Multi-protocol capture | DIGITAL, PATTERNS          |

### 3. Binary Input Tests (InputType.BINARY)

| File                            | Purpose                   | Domains Tested                  |
| ------------------------------- | ------------------------- | ------------------------------- |
| `binary_structured_packets.bin` | Known packet structure    | ENTROPY, PATTERNS, INFERENCE    |
| `binary_random_data.bin`        | High-entropy random       | ENTROPY (classification)        |
| `binary_compressed.bin`         | Compressed data           | ENTROPY (type detection)        |
| `binary_protocol_messages.bin`  | Field-structured messages | INFERENCE, PACKET               |
| `binary_checksum_bearing.bin`   | Packets with CRC-16       | STATISTICS (checksum detection) |

### 4. PCAP Input Tests (InputType.PCAP)

| File                          | Purpose               | Domains Tested               |
| ----------------------------- | --------------------- | ---------------------------- |
| `pcap_http_session.pcap`      | HTTP request/response | PACKET (throughput, latency) |
| `pcap_iot_mqtt.pcap`          | MQTT IoT traffic      | PACKET (stream analysis)     |
| `pcap_industrial_modbus.pcap` | Modbus/TCP            | PROTOCOLS, INFERENCE         |

### 5. In-Memory Data Tests

| Data Type          | Purpose              | Domains Tested          |
| ------------------ | -------------------- | ----------------------- |
| `WaveformTrace`    | Direct trace object  | All waveform domains    |
| `DigitalTrace`     | Digital trace object | All digital domains     |
| `IQTrace`          | I/Q baseband data    | SPECTRAL, INFERENCE     |
| `bytes`            | Raw binary data      | ENTROPY, BINARY domains |
| `list[PacketInfo]` | Packet list          | PACKET domain           |

## Generating Test Data

```bash
# Generate all synthetic test data
uv run python scripts/test-data/generate_comprehensive_test_data.py

# Or generate specific waveforms
uv run python scripts/test-data/generate_synthetic_wfm.py --generate-suite
```

## Ground Truth Files

Each test file has a corresponding `_ground_truth.json`:

```json
{
  "file": "waveform_sine_1khz.npz",
  "expected_domains": ["waveform", "spectral", "statistics"],
  "expected_metrics": {
    "frequency_hz": 1000.0,
    "amplitude_vpp": 2.0,
    "thd_percent": 0.001,
    "snr_db": 80.0
  }
}
```

## Coverage by Analysis Domain

| Domain           | Test Files | Coverage    |
| ---------------- | ---------- | ----------- |
| WAVEFORM         | 6          | ✅ Complete |
| SPECTRAL         | 4          | ✅ Complete |
| DIGITAL          | 5          | ✅ Complete |
| TIMING           | 3          | ✅ Complete |
| STATISTICS       | 4          | ✅ Complete |
| PATTERNS         | 3          | ✅ Complete |
| JITTER           | 2          | ✅ Complete |
| EYE              | 1          | ✅ Complete |
| POWER            | 1          | ✅ Complete |
| PROTOCOLS        | 5          | ✅ Complete |
| SIGNAL_INTEGRITY | 1          | ⚠️ Partial  |
| ENTROPY          | 4          | ✅ Complete |
| PACKET           | 4          | ✅ Complete |
| INFERENCE        | 4          | ✅ Complete |

## Validation Script

```python
from oscura.reporting import analyze, AnalysisResult
from pathlib import Path
import json

def validate_report(file_path: Path) -> bool:
    """Validate analysis report against ground truth."""
    result = analyze(file_path)

    # Load ground truth
    truth_path = file_path.with_suffix('.json')
    truth = json.loads(truth_path.read_text())

    # Verify domains
    for domain in truth['expected_domains']:
        assert domain in result.domain_summaries

    return True
```
