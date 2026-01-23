# Oscura API Patterns

**Version**: 0.5.0 (Phase 0)
**Status**: Foundation Architecture
**Last Updated**: 2026-01-20

This document provides a decision framework for choosing the right API pattern in Oscura. Use this guide when adding new features or refactoring existing code.

---

## Quick Reference

| Use Case                | Pattern                 | Example                          |
| ----------------------- | ----------------------- | -------------------------------- |
| One-shot measurement    | Function                | `rise_time(trace)`               |
| Data transformation     | Function → Trace        | `low_pass(trace, cutoff=1e6)`    |
| Protocol decoding       | Decoder class           | `UARTDecoder().decode(trace)`    |
| Data acquisition        | Source subclass         | `FileSource("capture.wfm")`      |
| Interactive exploration | Session subclass        | `CANSession()`                   |
| Automated workflow      | Workflow function (DAG) | `reverse_engineer_signal(trace)` |
| Sequential composition  | Pipeline                | `low_pass(1e6) \| decimate(4)`   |
| One-call convenience    | Convenience function    | `quick_spectral(trace)`          |

---

## Pattern 1: Pure Functions

**When to use**:

- One-shot measurements (rise time, frequency, amplitude)
- Stateless transformations (filtering, resampling, differentiation)
- Calculations that don't require configuration state

**Signature**: `function(trace, params) -> result`

**Example**:

```python
def rise_time(trace: WaveformTrace, low: float = 0.1, high: float = 0.9) -> float:
    """Calculate 10-90% rise time.

    Args:
        trace: Input waveform trace.
        low: Lower threshold (default: 0.1 = 10%).
        high: Upper threshold (default: 0.9 = 90%).

    Returns:
        Rise time in seconds.
    """
    # Pure function - no side effects, no state
    ...

# Usage
time = rise_time(trace)
time = rise_time(trace, low=0.2, high=0.8)  # Custom thresholds
```

**Characteristics**:

- ✅ Stateless (same inputs → same outputs)
- ✅ No side effects (doesn't modify inputs, no I/O)
- ✅ Easily composable
- ✅ Easy to test
- ✅ Thread-safe by default

**When NOT to use**:

- Multi-step workflows requiring state
- Operations needing previous results
- Interactive exploration

---

## Pattern 2: Source Protocol

**When to use**:

- Data acquisition from ANY source (files, hardware, synthetic)
- Need polymorphic acquisition (file or hardware interchangeable)
- Streaming or chunked reading

**Interface**:

```python
class Source(Protocol):
    def read(self) -> Trace: ...
    def stream(self, chunk_size: int) -> Iterator[Trace]: ...
    def close(self) -> None: ...
    def __enter__(self) -> Source: ...
    def __exit__(self, *args) -> None: ...
```

**Example**:

```python
# File acquisition
source = FileSource("capture.wfm")
trace = source.read()

# Hardware acquisition (Phase 2)
source = HardwareSource.socketcan("can0", bitrate=500000)
for chunk in source.stream(chunk_size=1000):
    process(chunk)

# Synthetic acquisition
source = SyntheticSource(SignalBuilder().sine(1000))
trace = source.read()

# Polymorphic usage
def analyze_from_source(source: Source):
    """Works with ANY source."""
    return analyze(source.read())
```

**Implementation checklist**:

- [ ] Implement `read()` for one-shot acquisition
- [ ] Implement `stream()` for chunked acquisition
- [ ] Implement context manager (`__enter__`, `__exit__`)
- [ ] Implement `close()` for resource cleanup
- [ ] Document which Trace type is returned

**When NOT to use**:

- Simple one-time file loading (use `load()` convenience function)
- Operations that don't acquire new data

---

## Pattern 3: AnalysisSession Subclass

**When to use**:

- Interactive multi-step analysis
- Need to manage multiple recordings/captures
- Comparison and differential analysis
- Domain-specific analysis requiring state
- Export to domain-specific formats (DBC, Wireshark, etc.)

**Base class**:

```python
class AnalysisSession(ABC):
    def add_recording(self, name: str, source: Source): ...
    def get_recording(self, name: str) -> Trace: ...
    def compare(self, name1: str, name2: str) -> ComparisonResult: ...
    def export_results(self, format: str, path: str): ...

    @abstractmethod
    def analyze(self) -> Any:
        """Domain-specific analysis."""
        ...
```

**Example**:

```python
class CANSession(AnalysisSession):
    """Interactive CAN bus analysis."""

    def analyze(self) -> dict:
        """Discover CAN signals."""
        return self.discover_signals()

    def discover_signals(self) -> dict:
        """CAN-specific signal discovery."""
        main = self.get_recording("main")
        # ... CAN signal extraction ...
        return signals

    def export_dbc(self, path: str):
        """Export to DBC format (CAN-specific)."""
        signals = self.discover_signals()
        # ... DBC generation ...

# Usage
session = CANSession(name="Vehicle Debug")
session.add_recording("baseline", FileSource("idle.blf"))
session.add_recording("active", FileSource("running.blf"))

# Compare recordings
diff = session.compare("baseline", "active")
print(f"Similarity: {diff.similarity_score:.2%}")

# Domain-specific analysis
signals = session.discover_signals()

# Domain-specific export
session.export_dbc("output.dbc")
```

**Implementation checklist**:

- [ ] Inherit from `AnalysisSession`
- [ ] Implement `analyze()` with domain logic
- [ ] Override `export_results()` for custom formats (optional)
- [ ] Override `compare()` for domain-specific comparison (optional)
- [ ] Add domain-specific methods (e.g., `discover_signals()`)

**When NOT to use**:

- One-shot analysis (use pure function or workflow)
- Fully automated pipelines (use workflow)
- Simple transformations (use pipeline)

---

## Pattern 4: Workflow Function

**When to use**:

- Automated end-to-end analysis pipeline
- Repeatable process that doesn't need interaction
- Complex multi-step analysis with pre-defined steps
- Wrapping session-based analysis for function call simplicity

**Signature**: `workflow(inputs, config) -> results`

**Example**:

```python
def reverse_engineer_signal(
    trace: Trace,
    stimulus_recordings: list[tuple[str, Source]],
    config: REConfig | None = None,
) -> ReverseEngineeringResult:
    """Automated protocol reverse engineering.

    Internally creates BlackBoxSession, runs analysis, returns results.

    Args:
        trace: Main signal to reverse engineer.
        stimulus_recordings: List of (name, source) for differential analysis.
        config: Optional configuration.

    Returns:
        ReverseEngineeringResult with protocol spec, state machine, etc.
    """
    # Workflow wraps session internally
    session = BlackBoxSession()
    session.add_recording("main", SyntheticSource(...))  # Wrap trace

    for name, source in stimulus_recordings:
        session.add_recording(name, source)

    # Run analysis
    protocol_spec = session.generate_protocol_spec()
    state_machine = session.infer_state_machine()

    return ReverseEngineeringResult(
        protocol_spec=protocol_spec,
        state_machine=state_machine,
    )

# Usage - one function call
result = reverse_engineer_signal(
    trace=FileSource("unknown_protocol.wfm").read(),
    stimulus_recordings=[
        ("baseline", FileSource("idle.wfm")),
        ("button_press", FileSource("pressed.wfm")),
    ],
)

print(result.protocol_spec)
```

**Characteristics**:

- ✅ Simple interface (one function call)
- ✅ Encapsulates complexity
- ✅ Reproducible
- ✅ Internally uses sessions and pipelines
- ❌ Less flexible than sessions

**When NOT to use**:

- Interactive exploration (use session)
- Need to inspect intermediate results
- Workflow steps need customization

---

## Pattern 5: Pipeline Composition

**When to use**:

- Sequential data transformations
- Need to compose multiple operations
- Reusable transformation chains

**Operators**: `|`, `>>`, `compose()`, `pipe()`

**Example**:

```python
# Pipeline for signal conditioning
conditioning_pipeline = (
    low_pass(cutoff=1e6)
    | decimate(factor=4)
    | normalize(method="zscore")
)

# Apply to trace
conditioned = conditioning_pipeline(trace)

# Pipeline for protocol analysis
protocol_analysis = (
    to_digital(threshold=1.2)
    | decode_uart(baud=115200)
    | extract_frames()
    | validate_checksums()
)

frames = protocol_analysis(trace)

# Reusable pipelines
can_decoder = to_digital(1.5) | decode_can(bitrate=500000)
trace1_frames = can_decoder(trace1)
trace2_frames = can_decoder(trace2)
```

**Multi-type composition**:

```python
# Pipeline handles type transitions
analysis = (
    low_pass(1e6)           # Trace → Trace
    | to_digital(1.2)       # Trace → DigitalTrace
    | decode_uart(115200)   # DigitalTrace → List[Packet]
    | extract_frames()      # List[Packet] → List[Frame]
)
```

**When NOT to use**:

- Non-linear workflows (use DAG)
- Need to inspect intermediate results (use explicit steps)
- Complex stateful operations (use session)

---

## Pattern 6: Convenience Functions

**When to use**:

- Common 80% use cases
- Simplify complex multi-step workflows
- Provide sensible defaults

**Pattern**: Wrapper around more complex APIs

**Example**:

```python
def quick_spectral(
    trace: WaveformTrace,
    fundamental: float | None = None,
) -> SpectralMetrics:
    """One-call spectral analysis with sensible defaults.

    Internally: FFT + THD + SNR + SINAD + ENOB calculation.

    Args:
        trace: Input waveform.
        fundamental: Fundamental frequency (Hz). If None, auto-detect.

    Returns:
        SpectralMetrics with all common measurements.
    """
    # Wrapper around multiple calls
    freq, mag = fft(trace, window="blackman-harris")
    thd_value = thd(trace, fundamental=fundamental)
    snr_value = snr(trace, fundamental=fundamental)
    # ...

    return SpectralMetrics(
        thd_db=thd_value,
        snr_db=snr_value,
        # ...
    )

# Usage - one line
metrics = quick_spectral(trace, fundamental=1000)
print(f"THD: {metrics.thd_db:.1f} dB")
```

**Guidelines**:

- Provide sensible defaults
- Return structured result (dataclass)
- Document what it does internally
- Link to low-level functions for customization

**When NOT to use**:

- User needs fine control
- Defaults aren't appropriate for use case

---

## Pattern 7: Decoder Classes

**When to use**:

- Protocol decoding with configuration state
- Stateful parsing (e.g., CRC accumulation)
- Need to configure decoder once, decode many traces

**Pattern**:

```python
class ProtocolDecoder(ABC):
    @abstractmethod
    def decode(self, trace: DigitalTrace) -> list[Packet]:
        """Decode protocol from trace."""
        ...

class UARTDecoder(ProtocolDecoder):
    def __init__(self, baud_rate: int, parity: str = "none"):
        self.baud_rate = baud_rate
        self.parity = parity

    def decode(self, trace: DigitalTrace) -> list[Packet]:
        """Decode UART frames."""
        # Use self.baud_rate, self.parity for decoding
        ...

# Usage
decoder = UARTDecoder(baud_rate=115200, parity="even")
frames = decoder.decode(trace1)
more_frames = decoder.decode(trace2)  # Reuse configuration
```

**When NOT to use**:

- Decoder has no configuration (use pure function)
- One-time decoding (use convenience function like `decode_uart()`)

---

## Decision Tree

```
Need to acquire data?
├─ Yes → Use Source (FileSource, HardwareSource, SyntheticSource)
└─ No → Continue

Interactive multi-step analysis?
├─ Yes → Use AnalysisSession subclass
└─ No → Continue

Automated end-to-end workflow?
├─ Yes → Use Workflow function
└─ No → Continue

Sequential transformations?
├─ Yes → Use Pipeline
└─ No → Continue

One-shot measurement or transformation?
├─ Yes → Use Pure Function
└─ No → Continue

Protocol decoding with state?
├─ Yes → Use Decoder class
└─ No → Use Pure Function
```

---

## Common Scenarios

### Scenario 1: Load file and analyze

**Simple (80% case)**:

```python
# Use convenience wrapper
trace = load("capture.wfm")
metrics = quick_spectral(trace, fundamental=1000)
```

**Flexible (20% case)**:

```python
# Use Source for explicit control
source = FileSource("capture.wfm")
trace = source.read()
# Custom analysis with primitives
freq, mag = fft(trace, nfft=8192, window="hann")
thd_value = thd(trace, fundamental=1000)
```

### Scenario 2: CAN bus signal discovery

**Interactive exploration**:

```python
# Use Session for multi-step interactive work
session = CANSession(name="ECU Debug")
session.add_recording("baseline", FileSource("idle.blf"))
session.add_recording("throttle", FileSource("throttle.blf"))

# Explore interactively
diff = session.compare("baseline", "throttle")
print(f"Changed signals: {diff.changed_bytes}")

signals = session.discover_signals()
print(f"Found {len(signals)} signals")

# Export results
session.export_dbc("ecu.dbc")
```

**Automated workflow**:

```python
# Use Workflow for automation
result = discover_can_signals(
    main=FileSource("capture.blf"),
    comparisons=[
        ("baseline", FileSource("idle.blf")),
        ("active", FileSource("running.blf")),
    ],
)

# One-shot, get structured result
print(result.signals)
result.export_dbc("output.dbc")
```

### Scenario 3: Signal conditioning pipeline

**Reusable pipeline**:

```python
# Define once, use many times
conditioning = (
    low_pass(cutoff=1e6)
    | high_pass(cutoff=100)
    | decimate(factor=4)
    | normalize(method="zscore")
)

# Apply to multiple traces
conditioned1 = conditioning(trace1)
conditioned2 = conditioning(trace2)
```

### Scenario 4: Protocol reverse engineering

**Interactive (explore as you go)**:

```python
session = BlackBoxSession()
session.add_recording("baseline", FileSource("idle.wfm"))
session.add_recording("stim1", FileSource("button1.wfm"))
session.add_recording("stim2", FileSource("button2.wfm"))

# Explore differences
diff1 = session.compare("baseline", "stim1")
diff2 = session.compare("baseline", "stim2")

# Generate protocol spec
spec = session.generate_protocol_spec()
print(spec.fields)

# Refine and iterate
session.add_recording("stim3", FileSource("button3.wfm"))
refined_spec = session.generate_protocol_spec()
```

**Automated (batch processing)**:

```python
# Workflow for reproducible analysis
result = reverse_engineer_signal(
    trace=FileSource("unknown.wfm").read(),
    stimulus_recordings=[
        ("idle", FileSource("idle.wfm")),
        ("button1", FileSource("btn1.wfm")),
        ("button2", FileSource("btn2.wfm")),
    ],
)

print(result.protocol_spec)
result.export_wireshark_dissector("protocol.lua")
```

---

## Layer Reference

### Layer 1: Foundation (Low-Level API)

**Purpose**: Building blocks

**Patterns**:

- Pure functions for measurements
- Source implementations
- Data types (WaveformTrace, DigitalTrace, IQTrace)
- Protocol decoders

**Examples**:

- `rise_time()`, `frequency()`, `thd()`
- `FileSource`, `HardwareSource`, `SyntheticSource`
- `UARTDecoder`, `CANDecoder`

### Layer 2: Composition (Mid-Level API)

**Purpose**: Combining Layer 1 pieces

**Patterns**:

- Pipeline composition
- DAG for complex workflows
- Streaming processors

**Examples**:

- `low_pass(1e6) | decimate(4)`
- `compose(f, g, h)`
- `StreamingAnalyzer`

### Layer 3: User-Facing (High-Level API)

**Purpose**: Convenience and domain-specific workflows

**Patterns**:

- AnalysisSession subclasses
- Workflow functions
- Convenience wrappers

**Examples**:

- `CANSession`, `SerialSession`, `BlackBoxSession`
- `reverse_engineer_signal()`, `debug_protocol()`
- `quick_spectral()`, `auto_decode()`

---

## References

- **Design Principles**: `docs/architecture/design-principles.md`
- **Phase 0 Implementation**: Strategic enhancement plan
- **Testing Guide**: `docs/testing/test-suite-guide.md`

---

## Changelog

- **2026-01-20**: Initial API patterns guide (Phase 0)
  - Pattern catalog with examples
  - Decision tree
  - Layer reference
  - Common scenarios
