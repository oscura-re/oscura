# API Reference

> **Version**: 0.1.0 | **Last Updated**: 2026-01-08

Complete API documentation for Oscura.

## Quick Links

|Category|Documentation|
|---|---|
|Loading Data|[loader.md](loader.md)|
|Analysis|[analysis.md](analysis.md)|
|Pipelines|[pipelines.md](pipelines.md)|
|Component Analysis|[component-analysis.md](component-analysis.md)|
|Comparison & Limits|[comparison-and-limits.md](comparison-and-limits.md)|
|EMC Compliance|[emc-compliance.md](emc-compliance.md)|
|Session Management|[session-management.md](session-management.md)|
|Export|[export.md](export.md)|
|Reporting|[reporting.md](reporting.md)|
|Visualization|[visualization.md](visualization.md)|
|**Expert API**|**[expert-api.md](expert-api.md)**|

## API Overview

### Loading Data

```python
import oscura as osc

# Load waveform (auto-detect format)
trace = osc.load("capture.wfm")

# Load with options
trace = osc.load("capture.wfm", lazy=True)

# Load all channels
channels = osc.load_all_channels("multi_channel.wfm")

# Check supported formats
formats = osc.get_supported_formats()
```

**Full documentation**: [loader.md](loader.md)

### Measurements

```python
import oscura as osc

# Time-domain
freq = osc.frequency(trace)
period = osc.period(trace)
rise_time = osc.rise_time(trace)
fall_time = osc.fall_time(trace)
duty_cycle = osc.duty_cycle(trace)

# Amplitude
amplitude = osc.amplitude(trace)
peak_to_peak = osc.peak_to_peak(trace)
rms = osc.rms(trace)

# Edges
edges = osc.find_edges(trace, threshold=0.5)
```

**Full documentation**: [analysis.md](analysis.md)

### Protocol Decoding

```python
import oscura as osc
from oscura.analyzers.protocols import (
    UARTDecoder,
    SPIDecoder,
    I2CDecoder,
    CANDecoder,
)

# UART (using convenience function)
messages = osc.decode_uart(trace, baud_rate=115200)

# Or use decoder class directly
decoder = UARTDecoder(baud_rate=115200)
messages = decoder.decode(trace)

# SPI (multi-channel)
decoder = SPIDecoder(clock=ch_clk, mosi=ch_mosi, miso=ch_miso, cs=ch_cs)
transactions = decoder.decode()

# I2C
decoder = I2CDecoder(sda=ch_sda, scl=ch_scl)
transactions = decoder.decode()
```

**Full documentation**: [analysis.md](analysis.md#protocol-decoding)

### Spectral Analysis

```python
import oscura as osc

# FFT and PSD
spectrum = osc.fft(trace, window="hanning")
psd = osc.psd(trace, window="hanning")

# Quality metrics
thd = osc.thd(trace, fundamental_freq=1e6)
snr = osc.snr(trace, signal_freq=1e6)
sinad = osc.sinad(trace, signal_freq=1e6)
sfdr = osc.sfdr(trace)
enob = osc.enob(trace, signal_freq=1e6)
```

**Full documentation**: [analysis.md](analysis.md#spectral-analysis)

### Pipelines & Composition

```python
import oscura as osc
from functools import partial

# Create analysis pipeline
pipeline = osc.Pipeline([
    ('filter', osc.LowPassFilter(cutoff=1e6)),
    ('normalize', osc.Normalize(method='peak')),
    ('fft', osc.FFT(nfft=8192))
])

# Transform trace
result = pipeline.transform(trace)

# Access intermediate results
filtered = pipeline.get_intermediate('filter')
spectrum = pipeline.get_intermediate('fft', 'spectrum')

# Functional composition
result = osc.pipe(
    trace,
    partial(osc.low_pass, cutoff=1e6),
    partial(osc.normalize, method='peak'),
    partial(osc.fft, nfft=8192)
)

# Custom transformers
class CustomTransformer(osc.TraceTransformer):
    def transform(self, trace):
        # Custom processing
        return trace
```

**Full documentation**: [pipelines.md](pipelines.md)

### Component Analysis

```python
import oscura as osc

# TDR impedance profiling
z0, profile = osc.extract_impedance(tdr_trace, z0_source=50.0)
discontinuities = osc.discontinuity_analysis(tdr_trace)

# Capacitance and inductance
C = osc.measure_capacitance(voltage, current, method="charge")
L = osc.measure_inductance(voltage, current, method="slope")

# Parasitic extraction
params = osc.extract_parasitics(voltage, current, model="series_RLC")

# Transmission line parameters
z0 = osc.characteristic_impedance(tdr_trace)
delay = osc.propagation_delay(tdr_trace)
vf = osc.velocity_factor(tdr_trace, line_length=0.1)
```

**Full documentation**: [component-analysis.md](component-analysis.md)

### Comparison & Limit Testing

```python
import oscura as osc
from oscura.comparison import (
    compare_traces,
    create_golden,
    compare_to_golden,
    create_limit_spec,
    check_limits,
    eye_mask,
    mask_test,
)

# Compare waveforms
result = compare_traces(measured, reference, tolerance=0.01)

# Golden reference testing
golden = create_golden(reference, tolerance_pct=5)
test_result = compare_to_golden(measured, golden)

# Limit testing
spec = create_limit_spec(upper=3.3, lower=2.7)
limit_result = check_limits(trace, spec)

# Eye diagram mask testing
mask = eye_mask(eye_width=0.5, eye_height=0.4)
mask_result = mask_test(eye_trace, mask)
```

**Full documentation**: [comparison-and-limits.md](comparison-and-limits.md)

### EMC Compliance Testing

```python
import oscura as osc
from oscura.compliance import (
    load_limit_mask,
    check_compliance,
    create_custom_mask,
    generate_compliance_report,
)

# Test against FCC/CE/MIL standards
mask = load_limit_mask("FCC_Part15_ClassB")
result = check_compliance(trace, mask, detector="quasi-peak")

# Create custom automotive EMC mask (CISPR 25)
cispr25 = create_custom_mask(
    name="CISPR_25_ClassB",
    frequencies=[150e3, 30e6, 108e6, 1000e6],
    limits=[74, 54, 44, 44],
    unit="dBuV",
    description="CISPR 25 Class B radiated emissions"
)

# Generate compliance report
generate_compliance_report(
    result,
    "emc_report.html",
    title="EMC Compliance Test",
    dut_info={"Model": "XYZ-100", "Serial": "12345"}
)
```

**Full documentation**: [emc-compliance.md](emc-compliance.md)

### Session Management & Audit Trail

```python
import oscura as osc

# Create and manage analysis sessions
session = osc.Session(name="Power Supply Analysis")
trace = session.load_trace("capture.wfm")
session.annotate("Voltage spike", time=1.5e-6)
session.record_measurement("rise_time", 2.3e-9, unit="s")
session.save("analysis.tks")

# Resume saved session
session = osc.load_session("analysis.tks")
print(session.summary())

# Audit trail for compliance
audit = osc.AuditTrail(secret_key=b"your-secret-key")
audit.record_action("load_trace", {"file": "data.wfm"})
assert audit.verify_integrity()
audit.export_audit_log("audit.json", format="json")
```

**Full documentation**: [session-management.md](session-management.md)

### Report Generation

```python
from oscura.reporting import (
    generate_report,
    save_pdf_report,
    save_html_report,
    ReportConfig,
)

# Generate and save
report = generate_report(trace, title="Analysis Report")
save_pdf_report(report, "report.pdf")
save_html_report(report, "report.html")
```

**Full documentation**: [reporting.md](reporting.md)

### Data Export

```python
import oscura as osc

# Export to various formats
osc.export_csv(trace, "data.csv")
osc.export_hdf5(trace, "data.h5", compression="gzip")
osc.export_npz(trace, "data.npz")
osc.export_json(trace, "data.json")
osc.export_mat(trace, "data.mat")
osc.export_pwl(trace, "data.pwl")  # For SPICE
```

**Full documentation**: [export.md](export.md)

## Module Reference

### Core Modules

|Module|Description|
|---|---|
|`oscura`|Main package with convenience functions|
|`oscura.core`|Core data types (WaveformTrace, DigitalTrace, TraceMetadata)|
|`oscura.core.exceptions`|Exception hierarchy|
|`oscura.core.config`|Configuration management|

### Loaders

|Module|Description|
|---|---|
|`oscura.loaders`|File format loaders|
|`oscura.loaders.tektronix`|Tektronix WFM loader|
|`oscura.loaders.rigol`|Rigol WFM loader|
|`oscura.loaders.sigrok`|Sigrok SR loader|
|`oscura.loaders.csv`|CSV loader|
|`oscura.loaders.hdf5`|HDF5 loader|
|`oscura.loaders.configurable`|Schema-driven packet loader|

### Analyzers

|Module|Description|
|---|---|
|`oscura.analyzers.waveform`|Waveform measurements|
|`oscura.analyzers.digital`|Digital signal analysis|
|`oscura.analyzers.spectral`|FFT, PSD, spectral metrics|
|`oscura.analyzers.jitter`|Jitter measurements|
|`oscura.analyzers.eye`|Eye diagram analysis|
|`oscura.analyzers.statistical`|Statistical analysis|
|`oscura.analyzers.patterns`|Pattern detection|

### Protocol Decoders

|Module|Protocol|
|---|---|
|`oscura.analyzers.protocols.uart`|UART/RS-232|
|`oscura.analyzers.protocols.spi`|SPI|
|`oscura.analyzers.protocols.i2c`|I2C|
|`oscura.analyzers.protocols.can`|CAN/CAN-FD|
|`oscura.analyzers.protocols.lin`|LIN|
|`oscura.analyzers.protocols.flexray`|FlexRay|
|`oscura.analyzers.protocols.onewire`|1-Wire|
|`oscura.analyzers.protocols.jtag`|JTAG|
|`oscura.analyzers.protocols.swd`|SWD|
|`oscura.analyzers.protocols.i2s`|I2S|
|`oscura.analyzers.protocols.usb`|USB|
|`oscura.analyzers.protocols.hdlc`|HDLC|
|`oscura.analyzers.protocols.manchester`|Manchester encoding|

### Inference

|Module|Description|
|---|---|
|`oscura.inference`|Protocol inference|
|`oscura.inference.message_format`|Message structure detection|
|`oscura.inference.state_machine`|State machine inference|
|`oscura.inference.alignment`|Sequence alignment|

### Comparison & Testing

|Module|Description|
|---|---|
|`oscura.comparison`|Waveform comparison|
|`oscura.comparison.compare`|Trace comparison functions|
|`oscura.comparison.golden`|Golden reference testing|
|`oscura.comparison.limits`|Specification limit testing|
|`oscura.comparison.mask`|Mask-based pass/fail testing|

### EMC Compliance

|Module|Description|
|---|---|
|`oscura.compliance`|EMC/EMI compliance testing|
|`oscura.compliance.masks`|Regulatory limit masks (FCC, CISPR, MIL-STD)|
|`oscura.compliance.testing`|Compliance test execution and result analysis|
|`oscura.compliance.reporting`|Compliance report generation (HTML, PDF, JSON)|
|`oscura.compliance.advanced`|Advanced detectors and interpolation methods|

### Session Management & Audit

|Module|Description|
|---|---|
|`oscura.session`|Session management and annotations|
|`oscura.core.audit`|Audit trail with HMAC verification|

### Export & Reporting

|Module|Description|
|---|---|
|`oscura.exporters`|Data exporters|
|`oscura.reporting`|Report generation|
|`oscura.visualization`|Plotting utilities|

## Accessing Documentation

### Python Help

```python
import oscura as osc

# Get help on any function
help(osc.load)
help(osc.frequency)

# Module documentation
help(osc.analyzers.spectral)
```

### Docstrings

All public functions include comprehensive docstrings with:

- Parameter descriptions
- Return value documentation
- Usage examples
- IEEE standard references (where applicable)

Example:

```python
def measure_rise_time(
    trace: WaveformTrace,
    low: float = 0.1,
    high: float = 0.9,
) -> float:
    """Calculate rise time per IEEE 181-2011 Section 5.2.

    Parameters
    ----------
    trace : WaveformTrace
        Input waveform trace.
    low : float, optional
        Low reference level (0-1). Default 10%.
    high : float, optional
        High reference level (0-1). Default 90%.

    Returns
    -------
    float
        Rise time in seconds. NaN if measurement not applicable.

    Examples
    --------
    >>> trace = osc.load("capture.wfm")
    >>> rise = osc.measure_rise_time(trace)
    >>> print(f"Rise time: {rise*1e9:.2f} ns")

    References
    ----------
    IEEE 181-2011 Section 5.2 "Rise Time and Fall Time"
    """
```

## Exception Handling

```python
from oscura import LoaderError, DecodeError, MeasurementError

try:
    trace = osc.load("file.wfm")
except LoaderError as e:
    print(f"Load failed: {e}")
    print(f"Fix hint: {e.fix_hint}")

try:
    freq = osc.frequency(trace)
except MeasurementError as e:
    print(f"Measurement failed: {e}")
```

See [Error Codes](../error-codes.md) for complete error reference.

## Type Hints

Oscura is fully type-annotated for IDE support:

```python
from oscura import WaveformTrace, DigitalTrace, TraceMetadata
from oscura.analyzers.spectral import Spectrum, PowerSpectralDensity

def analyze_signal(trace: WaveformTrace) -> dict[str, float]:
    ...
```

## See Also

- [Demos](https://github.com/lair-click-bats/oscura/tree/main/demos) - Working code examples
- [CLI Reference](../cli.md) - Command-line tools
- [CLI Reference](../cli.md) - Command-line interface
