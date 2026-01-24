# Oscura Design Principles

**Version**: 0.5.1 (Phase 0)
**Status**: Foundation Architecture
**Last Updated**: 2026-01-20

This document defines the core design principles that guide Oscura's architecture. All new features and refactoring must align with these principles.

---

## Executive Summary

Oscura implements a **3-layer architecture** designed for:

- **Consistency**: Unified patterns across all features
- **Composability**: Mix and match components naturally
- **Progressive Disclosure**: Simple for common tasks, powerful for complex ones

The foundation (Phase 0) establishes these principles through:

1. Unified acquisition via `Source` protocol
2. Unified data types (`WaveformTrace`, `DigitalTrace`, `IQTrace`)
3. Unified session pattern via `AnalysisSession` hierarchy
4. Composable pipelines for data transformation

---

## Core Principles

### 1. Unified Interfaces Over Ad-Hoc Solutions

**Principle**: Use protocol-based interfaces to enable polymorphism and composition.

**Rationale**: Multiple implementations of the same interface can be used interchangeably, reducing cognitive load and enabling powerful abstractions.

**Examples**:

```python
# GOOD: Unified Source protocol
from oscura.acquisition import FileSource, HardwareSource, SyntheticSource

def analyze_from_source(source: Source):
    """Works with ANY source - file, hardware, or synthetic."""
    trace = source.read()
    return analyze(trace)

# Works polymorphically
analyze_from_source(FileSource("capture.wfm"))
analyze_from_source(HardwareSource.socketcan("can0"))
analyze_from_source(SyntheticSource(builder))

# BAD: Ad-hoc function-based approach
def analyze_from_file(path: str): ...
def analyze_from_hardware(device: str): ...
def analyze_from_synthetic(builder: SignalBuilder): ...
# Each function has different interface - no polymorphism
```

**Implementation**:

- `Source` protocol for acquisition (`FileSource`, `HardwareSource`, `SyntheticSource`)
- `AnalysisSession` hierarchy for sessions (`GenericSession`, `CANSession`, etc.)
- `TraceTransformer` for pipeline stages

### 2. Single Responsibility Per Layer

**Principle**: Each layer has ONE clear purpose. Don't mix concerns.

**3-Layer Architecture**:

```
Layer 3 (High-Level API - User-Facing)
├── AnalysisSession subclasses (CANSession, SerialSession, BlackBoxSession)
├── Workflows (pre-built DAGs with function wrappers)
└── One-call convenience functions (quick_spectral, auto_decode)

Layer 2 (Mid-Level API - Composition)
├── Pipeline (linear composition: trace | filter | analyze)
├── DAG (non-linear multi-output analysis)
└── Streaming (real-time processing)

Layer 1 (Low-Level API - Foundation)
├── Unified data model (WaveformTrace, DigitalTrace, IQTrace)
├── Source protocol (FileSource, HardwareSource, SyntheticSource)
├── Pure processing primitives (stateless transformations & measurements)
└── Protocol decoders (consistent interface)
```

**Guidelines**:

- **Layer 1**: Stateless, composable primitives. No workflows or sessions here.
- **Layer 2**: Composition logic only. No domain-specific knowledge.
- **Layer 3**: User-facing convenience. Can use Layers 1 & 2, but not vice versa.

**Example**:

```python
# GOOD: Clear layer separation
# Layer 1: Pure primitive
def low_pass(trace: WaveformTrace, cutoff: float) -> WaveformTrace:
    """Stateless filter operation."""
    ...

# Layer 2: Composition
pipeline = low_pass(1e6) | decimate(4) | to_digital()

# Layer 3: User-facing session
class CANSession(AnalysisSession):
    def analyze(self):
        # Uses Layer 1 & 2, adds CAN-specific logic
        ...

# BAD: Mixed concerns
def analyze_can_with_filtering_and_export(file, cutoff, output):
    # Mixes filtering, analysis, and export - violates SRP
    ...
```

### 3. Explicit Over Implicit

**Principle**: Make behavior obvious from the code. Avoid hidden state and magic.

**Examples**:

```python
# GOOD: Explicit channel selection
trace = builder.build(channel="sig")  # Clear which channel
channels = builder.build_channels()   # Clear: getting all channels

# BAD: Implicit behavior
trace = builder.build()  # Which channel? First? All? Unclear.

# GOOD: Explicit source type
source = FileSource("capture.wfm")
trace = source.read()

# BAD: Hidden file I/O
trace = some_function()  # Where does data come from?
```

**Guidelines**:

- Require explicit parameters instead of defaulting to "magic" behavior
- Make resource acquisition explicit (Source protocol, context managers)
- Avoid global state and hidden caching

### 4. Composability Over Monoliths

**Principle**: Build small, focused components that compose naturally.

**Example**:

```python
# GOOD: Composable building blocks
analysis = (
    low_pass(cutoff=1e6)
    | to_digital(threshold=1.2)
    | decode_uart(baud=115200)
    | extract_frames()
)

# Each stage is independent and reusable

# BAD: Monolithic function
def analyze_uart_from_file_with_filtering(
    file, cutoff, threshold, baud, export_format
):
    # Can't reuse parts, hard to test, inflexible
    ...
```

**Benefits**:

- **Reusability**: Each component works standalone
- **Testability**: Test each piece independently
- **Flexibility**: Mix and match as needed
- **Maintainability**: Easy to understand and modify

### 5. Type Safety Where It Matters

**Principle**: Use strong typing for correctness, but don't over-constrain.

**Type Unification**:

```python
# Unified types across APIs
sig = SignalBuilder().build()     # Returns WaveformTrace
trace = load("file.wfm")          # Returns WaveformTrace
# Both are WaveformTrace - fully composable!

# Multi-channel signals
channels = SignalBuilder().add_sine(1000, channel="sig").add_square(500, channel="clk").build_channels()
# Returns dict[str, WaveformTrace]
```

**Guidelines**:

- Use union types (`Trace = WaveformTrace | DigitalTrace | IQTrace`)
- Protocol-based interfaces for flexibility (Source, AnalysisSession)
- Avoid premature type constraints that limit composition

### 6. Progressive Disclosure

**Principle**: Simple tasks should be simple. Complex tasks should be possible.

**Example**:

```python
# Level 1: One-liner for common task
metrics = quick_spectral(trace, fundamental=1000)

# Level 2: More control with pipeline
result = trace | low_pass(1e6) | fft() | thd()

# Level 3: Full customization with session
session = CANSession()
session.add_recording("baseline", FileSource("idle.blf"))
session.add_recording("active", FileSource("running.blf"))
signals = session.discover_signals()
session.export_dbc("output.dbc")

# Level 4: Expert control with primitives
filtered = low_pass(trace, cutoff=1e6, order=8, filter_type="butter")
spectrum = fft(filtered, nfft=8192, window="blackman-harris")
...
```

**Guidelines**:

- Provide convenience functions for 80% use cases
- Allow access to low-level primitives for power users
- Document progression from simple to complex

### 7. Fail Fast and Loudly

**Principle**: Detect errors early. Give actionable error messages.

**Examples**:

```python
# GOOD: Immediate validation with helpful message
def add_recording(self, name: str, source: Source):
    if name in self.recordings:
        raise ValueError(
            f"Recording '{name}' already exists. "
            f"Available: {list(self.recordings.keys())}"
        )

# BAD: Silent failure or cryptic error
def add_recording(self, name, source):
    self.recordings[name] = source  # Silently overwrites existing!

# GOOD: Type checking at construction
@dataclass
class TraceMetadata:
    sample_rate: float

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )

# BAD: Defer error until later use
metadata = TraceMetadata(sample_rate=-1)  # No error
trace.time_vector  # Crashes later with confusing error
```

**Guidelines**:

- Validate inputs immediately
- Provide context in error messages
- Suggest fixes when possible

### 8. Clean Architecture and Type Safety

**Principle**: Maintain clean APIs with strong types. Remove deprecated code.

**SignalBuilder Example**:

```python
# New unified API
trace = SignalBuilder().add_sine(1000).build()
# Returns: WaveformTrace

channels = SignalBuilder().add_uart(115200).add_spi(1e6).build_channels()
# Returns: dict[str, WaveformTrace]
```

**Guidelines**:

- Remove deprecated code after migration period
- Maintain single source of truth for each capability
- Use strong types (WaveformTrace, DigitalTrace, etc.)
- Deprecate before removing (announce in CHANGELOG.md)
- Update documentation when APIs change

---

## Design Patterns

### Pattern 1: Source Protocol for Acquisition

**When to use**: Any data acquisition (files, hardware, synthetic).

**Example**:

```python
class FileSource:
    def read(self) -> Trace: ...
    def stream(self, chunk_size: int) -> Iterator[Trace]: ...
    def close(self) -> None: ...
    def __enter__(self) -> Source: ...
    def __exit__(self, *args) -> None: ...
```

### Pattern 2: AnalysisSession for Interactive Work

**When to use**: Multi-step interactive analysis requiring state management.

**Example**:

```python
class CANSession(AnalysisSession):
    def analyze(self) -> dict:
        # Domain-specific analysis
        return self.discover_signals()

    def discover_signals(self):
        # CAN-specific logic
        ...
```

### Pattern 3: Workflow for Automation

**When to use**: Repeatable automated analysis.

**Example**:

```python
def reverse_engineer_signal(
    trace: Trace,
    stimulus_recordings: list[tuple[str, Source]],
) -> ReverseEngineeringResult:
    # Pre-built DAG wrapping session internally
    session = BlackBoxSession()
    # ... automation logic ...
    return result
```

### Pattern 4: Pipeline for Composition

**When to use**: Sequential data transformations.

**Example**:

```python
analysis = (
    low_pass(1e6)
    | decimate(4)
    | to_digital(1.2)
    | decode_uart(115200)
)
result = analysis(trace)
```

---

## Anti-Patterns to Avoid

### 1. ❌ Mixing Abstraction Layers

```python
# BAD: Session calls workflow calls session
class CANSession(AnalysisSession):
    def analyze(self):
        return reverse_engineer_signal(self.recordings)  # Wrong layer!

# GOOD: Sessions use primitives, workflows use sessions
class CANSession(AnalysisSession):
    def analyze(self):
        # Use Layer 1 primitives directly
        return discover_can_signals(self.get_recording("main"))
```

### 2. ❌ Hidden State and Side Effects

```python
# BAD: Mutates global state
_global_cache = {}
def analyze(trace):
    _global_cache[id(trace)] = result  # Side effect!

# GOOD: Explicit state management
class AnalysisSession:
    def __init__(self):
        self.recordings = {}  # Explicit instance state
```

### 3. ❌ Monolithic Functions

```python
# BAD: Does everything
def analyze_can_from_file_with_export(file, filters, output_format):
    # 500 lines of loading, filtering, analysis, export
    ...

# GOOD: Composable pieces
trace = FileSource(file).read()
filtered = apply_filters(trace, filters)
results = analyze_can(filtered)
export(results, output_format)
```

---

## References

- **Phase 0 Implementation Plan**: Strategic enhancement architecture document
- **API Patterns**: `docs/architecture/api-patterns.md`
- **Testing Strategy**: `docs/testing/test-suite-guide.md`
- **Contributing Guide**: `CONTRIBUTING.md`

---

## Changelog

- **2026-01-20**: Initial Phase 0 design principles
  - 3-layer architecture defined
  - 8 core principles established
  - Pattern catalog created
