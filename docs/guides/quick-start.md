# Quick Start Guide

Get started with Oscura in 5 minutes.

## What is Oscura?

Oscura analyzes captured signals from oscilloscopes, logic analyzers, and data acquisition systems. It helps you:

- **Understand signals** - Load waveforms and extract measurements
- **Decode protocols** - Identify UART, SPI, I2C, CAN, and 16+ other protocols
- **Reverse engineer** - Discover unknown protocols and signal patterns
- **Validate compliance** - Check against IEEE and EMC standards

## Installation

```bash
pip install oscura
```

## Your First Analysis (30 seconds)

```python
import oscura as osc

# Load a waveform file
trace = osc.load("capture.wfm")

# Analyze it
measurements = osc.analyze(trace)
print(f"Frequency: {measurements['frequency']} Hz")
print(f"Amplitude: {measurements['amplitude']} V")

# Spectral analysis
metrics = osc.quick_spectral(trace)
print(f"THD: {metrics.thd_db:.2f} dB")
print(f"SNR: {metrics.snr_db:.2f} dB")
```

That's it! You've loaded, analyzed, and characterized a signal.

## Next Steps by Use Case

### 🔌 **Protocol Analysis**

Want to decode UART, SPI, I2C, or CAN signals?

→ **[Serial Protocols Demo](https://github.com/oscura-re/oscura/tree/main/demonstrations/04_serial_protocols)**

### 🚗 **Automotive Diagnostics**

Working with CAN bus, OBD-II, or J1939?

→ **[Automotive Demo](https://github.com/oscura-re/oscura/tree/main/demonstrations/09_automotive)**

### 📊 **Spectral Analysis**

Need FFT, THD, SNR, or IEEE 1241 compliance?

→ **[Spectral Compliance Demo](https://github.com/oscura-re/oscura/tree/main/demonstrations/12_spectral_compliance)**

### ⚡ **Power Analysis**

Analyzing power quality, ripple, or efficiency?

→ **[Power Analysis Demo](https://github.com/oscura-re/oscura/tree/main/demonstrations/14_power_analysis)**

### 🔍 **Reverse Engineering**

Unknown signal? Need to discover the protocol?

→ **[Signal RE Demo](https://github.com/oscura-re/oscura/tree/main/demonstrations/17_signal_reverse_engineering)**

## Learning Path

1. **Start here**: Try the example above with your own data
2. **Explore demos**: Find the demo matching your use case
3. **Read API docs**: Deep dive into specific functions
4. **Check workflows**: See [Common Workflows](workflows.md) for patterns

## Common Questions

**Q: What file formats are supported?**
→ See [Loader API](../api/loader.md) - Tektronix WFM, VCD, PCAP, CSV, HDF5, and more

**Q: How do I decode a specific protocol?**
→ See [Protocol Decoders](https://github.com/oscura-re/oscura/tree/main/demonstrations/05_protocol_decoding)

**Q: Can I analyze large files?**
→ See [Custom DAQ Demo](https://github.com/oscura-re/oscura/tree/main/demonstrations/03_custom_daq) for streaming

**Q: What measurements are available?**
→ See [Analysis API](../api/analysis.md) for complete list

## Help & Resources

- **All Demos**: [Browse 19 comprehensive demos](https://github.com/oscura-re/oscura/tree/main/demos)
- **API Reference**: [Complete function documentation](../api/index.md)
- **Error Codes**: [Troubleshooting guide](../error-codes.md)
- **Contributing**: [How to contribute](https://github.com/oscura-re/oscura/blob/main/CONTRIBUTING.md)
