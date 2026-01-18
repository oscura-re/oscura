# Session Management & Audit Trail API

> **Version**: 0.1.0
> **Last Updated**: 2026-01-17

## Overview

Oscura provides comprehensive session management and audit trail capabilities for tracking, saving, and resuming analysis work. Sessions capture traces, annotations, measurements, and operation history, while audit trails provide tamper-evident logging for compliance requirements.

## Quick Start

```python
import oscura as osc

# Create and work with a session
session = osc.Session(name="Power Supply Analysis")
trace = session.load_trace("capture.wfm")
session.annotate("Voltage spike", time=1.5e-6)
session.record_measurement("rise_time", 2.3e-9, unit="s")
session.save("analysis.tks")

# Resume later
session = osc.load_session("analysis.tks")
print(session.summary())

# Enable audit trail for compliance
audit = osc.AuditTrail(secret_key=b"your-secret-key")
audit.record_action("load_trace", {"file": "data.wfm"})
assert audit.verify_integrity()
audit.export_audit_log("audit.json", format="json")
```

## Session Management

### `Session`

Analysis session container that manages traces, annotations, measurements, and operation history.

**Class Definition:**

```python
@dataclass
class Session:
    """Analysis session container.

    Attributes:
        name: Session name
        traces: Dictionary of loaded traces (name -> trace)
        annotation_layers: Annotation layers
        measurements: Recorded measurements
        history: Operation history
        metadata: Session metadata
        created_at: Creation timestamp
        modified_at: Last modification timestamp
    """
```

**Example - Basic Session Usage:**

```python
import oscura as osc

# Create a new session
session = osc.Session(name="Debug Session")

# Load traces
trace1 = session.load_trace("oscilloscope.wfm", name="CH1")
trace2 = session.load_trace("logic_analyzer.sr", name="DIGITAL")

# List loaded traces
print(session.list_traces())  # ['CH1', 'DIGITAL']

# Get a specific trace
trace = session.get_trace("CH1")

# Add in-memory trace
import numpy as np
data = np.sin(np.linspace(0, 2*np.pi, 1000))
my_trace = osc.WaveformTrace(data=data, sample_rate=1e6)
session.add_trace("SYNTHETIC", my_trace)
```

**Example - Complete Analysis Workflow:**

```python
import oscura as osc

# Create session
session = osc.Session(name="Power Rail Analysis")

# Load and analyze
trace = session.load_trace("power_supply.wfm", name="5V_RAIL")

# Record measurements
freq = osc.frequency(trace)
session.record_measurement(
    "frequency",
    freq,
    unit="Hz",
    trace_name="5V_RAIL"
)

ripple = osc.amplitude(trace)
session.record_measurement(
    "ripple_vpp",
    ripple,
    unit="V",
    trace_name="5V_RAIL",
    acceptable=ripple < 0.1
)

# Add annotations
session.annotate(
    "Transient response",
    time_range=(1e-3, 2e-3),
    layer="analysis"
)

session.annotate(
    "Maximum ripple",
    time=1.5e-3,
    layer="measurements"
)

# Save session
session.save("power_analysis.tks", compress=True)

# Get summary
print(session.summary())
```

### Session Methods

#### `load_trace()`

Load a trace file into the session.

```python
def load_trace(
    path: str | Path,
    name: str | None = None,
    **load_kwargs: Any,
) -> Any:
    """Load a trace into the session.

    Args:
        path: Path to trace file.
        name: Name for trace in session (default: filename).
        **load_kwargs: Additional arguments for load().

    Returns:
        Loaded trace.
    """
```

**Example:**

```python
session = osc.Session()

# Load with auto-generated name
trace1 = session.load_trace("capture.wfm")

# Load with custom name
trace2 = session.load_trace("data.csv", name="SENSOR_DATA")

# Load with loader options
trace3 = session.load_trace("large_file.wfm", lazy=True)
```

#### `add_trace()`

Add a programmatically created trace to the session.

```python
def add_trace(
    name: str,
    trace: Any,
) -> None:
    """Add an in-memory trace to the session.

    Args:
        name: Name for the trace in the session.
        trace: Trace object (WaveformTrace, DigitalTrace, etc.).
    """
```

**Example:**

```python
import numpy as np
import oscura as osc

session = osc.Session()

# Create synthetic waveform
t = np.linspace(0, 1e-3, 10000)
data = 3.3 * np.sin(2 * np.pi * 1000 * t)
trace = osc.WaveformTrace(data=data, sample_rate=10e6)

# Add to session
session.add_trace("SINE_1KHZ", trace)
```

#### `annotate()`

Add an annotation to the session.

```python
def annotate(
    text: str,
    *,
    time: float | None = None,
    time_range: tuple[float, float] | None = None,
    layer: str = "default",
    **kwargs: Any,
) -> None:
    """Add annotation to session.

    Args:
        text: Annotation text.
        time: Time point for annotation.
        time_range: Time range for annotation.
        layer: Annotation layer name.
        **kwargs: Additional annotation parameters (color, style, etc.).
    """
```

**Example:**

```python
session = osc.Session()
session.load_trace("debug.wfm")

# Point annotation
session.annotate("Glitch detected", time=1.5e-6)

# Range annotation
session.annotate(
    "Data packet",
    time_range=(2e-6, 5e-6),
    layer="protocol"
)

# Custom styling
session.annotate(
    "Critical error",
    time=10e-6,
    layer="errors",
    color="#FF0000",
    style="dashed"
)
```

#### `record_measurement()`

Record a measurement result.

```python
def record_measurement(
    name: str,
    value: Any,
    unit: str = "",
    trace_name: str | None = None,
    **metadata: Any,
) -> None:
    """Record a measurement result.

    Args:
        name: Measurement name (e.g., 'rise_time').
        value: Measurement value.
        unit: Unit of measurement.
        trace_name: Associated trace name.
        **metadata: Additional metadata.
    """
```

**Example:**

```python
import oscura as osc

session = osc.Session()
trace = session.load_trace("signal.wfm", name="CH1")

# Record measurements
rise = osc.rise_time(trace)
session.record_measurement(
    "rise_time",
    rise,
    unit="s",
    trace_name="CH1",
    standard="IEEE 181-2011"
)

freq = osc.frequency(trace)
session.record_measurement(
    "frequency",
    freq,
    unit="Hz",
    trace_name="CH1",
    confidence=0.95
)

# Get all measurements
measurements = session.get_measurements()
print(measurements)
```

#### `save()`

Save session to file.

```python
def save(
    path: str | Path | None = None,
    *,
    include_traces: bool = True,
    compress: bool = True,
) -> Path:
    """Save session to file.

    Args:
        path: Output path (default: use existing or generate).
        include_traces: Include trace data in session file.
        compress: Compress session file with gzip.

    Returns:
        Path to saved file.
    """
```

**Example:**

```python
session = osc.Session(name="Analysis")
session.load_trace("capture.wfm")
session.annotate("Start", time=0)

# Save with all defaults
session.save("analysis.tks")

# Save without trace data (smaller file)
session.save("analysis_meta.tks", include_traces=False)

# Save uncompressed
session.save("analysis_raw.tks", compress=False)
```

#### `get_annotations()`

Retrieve annotations with optional filtering.

```python
def get_annotations(
    layer: str | None = None,
    time_range: tuple[float, float] | None = None,
) -> list[Annotation]:
    """Get annotations.

    Args:
        layer: Filter by layer name (None for all layers).
        time_range: Filter by time range.

    Returns:
        List of annotations.
    """
```

**Example:**

```python
session = osc.Session()
session.load_trace("signal.wfm")

# Add annotations
session.annotate("Event 1", time=1e-6, layer="events")
session.annotate("Event 2", time=2e-6, layer="events")
session.annotate("Error", time=1.5e-6, layer="errors")

# Get all annotations
all_annotations = session.get_annotations()

# Get annotations from specific layer
events = session.get_annotations(layer="events")

# Get annotations in time range
window = session.get_annotations(time_range=(1e-6, 2e-6))
```

### `load_session()`

Load a saved session from file.

```python
def load_session(path: str | Path) -> Session:
    """Load session from file.

    Args:
        path: Path to session file (.tks).

    Returns:
        Loaded Session object.
    """
```

**Example:**

```python
import oscura as osc

# Load saved session
session = osc.load_session("debug_session.tks")

# Resume analysis
print(session.summary())
print(f"Traces: {session.list_traces()}")

# Access saved data
measurements = session.get_measurements()
annotations = session.get_annotations()
history = session.history.entries
```

## Annotations

### `Annotation`

Single annotation marking a point or region of interest in a trace.

**Class Definition:**

```python
@dataclass
class Annotation:
    """Single annotation on a trace.

    Attributes:
        text: Annotation text/label
        time: Time point (for point annotations)
        time_range: (start, end) time range
        amplitude: Amplitude value (for horizontal lines)
        amplitude_range: (min, max) amplitude range
        annotation_type: Type of annotation
        color: Display color (hex or name)
        style: Line style ('solid', 'dashed', 'dotted')
        visible: Whether annotation is visible
        created_at: Creation timestamp
        metadata: Additional metadata
    """
```

**Example:**

```python
from oscura import Annotation, AnnotationType

# Point annotation
ann1 = Annotation(
    text="Trigger point",
    time=1.5e-6,
    color="#00FF00"
)

# Range annotation
ann2 = Annotation(
    text="Data burst",
    time_range=(2e-6, 5e-6),
    annotation_type=AnnotationType.RANGE,
    color="#FF6B6B",
    style="dashed"
)

# Horizontal reference line
ann3 = Annotation(
    text="Threshold",
    amplitude=3.3,
    annotation_type=AnnotationType.HORIZONTAL,
    color="#0000FF"
)

# Region annotation (time + amplitude)
ann4 = Annotation(
    text="Operating range",
    time_range=(0, 1e-3),
    amplitude_range=(2.5, 3.5),
    annotation_type=AnnotationType.REGION
)
```

### `AnnotationType`

Enumeration of annotation types.

```python
class AnnotationType(Enum):
    """Types of annotations."""

    POINT = "point"              # Single time point
    RANGE = "range"              # Time range
    VERTICAL = "vertical"        # Vertical line
    HORIZONTAL = "horizontal"    # Horizontal line
    REGION = "region"            # 2D region (time + amplitude)
    TEXT = "text"                # Free-floating text
```

### `AnnotationLayer`

Collection of related annotations organized in a named layer.

**Class Definition:**

```python
@dataclass
class AnnotationLayer:
    """Collection of related annotations.

    Attributes:
        name: Layer name
        annotations: List of annotations
        visible: Whether layer is visible
        locked: Whether layer is locked (read-only)
        color: Default color for new annotations
        description: Layer description
    """
```

**Example - Layer Management:**

```python
from oscura import AnnotationLayer, Annotation

# Create layer
events = AnnotationLayer(
    name="Protocol Events",
    color="#00FF00",
    description="Communication protocol events"
)

# Add annotations
events.add(text="START", time=0)
events.add(text="DATA", time_range=(1e-6, 5e-6))
events.add(text="STOP", time=10e-6)

# Find annotations
at_time = events.find_at_time(1e-6, tolerance=100e-9)
in_range = events.find_in_range(0, 5e-6)

# Lock layer to prevent modifications
events.locked = True

# Clear all annotations
events.locked = False
events.clear()
```

**Example - Multiple Layers:**

```python
import oscura as osc

session = osc.Session()
session.load_trace("debug.wfm")

# Add annotations to different layers
session.annotate("Start bit", time=0, layer="uart")
session.annotate("Data byte", time_range=(1e-6, 9e-6), layer="uart")
session.annotate("Parity error", time=8e-6, layer="errors")
session.annotate("Maximum voltage", time=5e-6, layer="measurements")

# Access layer directly
uart_layer = session.annotation_layers["uart"]
uart_layer.color = "#0000FF"
uart_layer.description = "UART protocol decode"

# Query by layer
uart_annotations = session.get_annotations(layer="uart")
error_annotations = session.get_annotations(layer="errors")
```

## Operation History

### `HistoryEntry`

Single entry recording an operation performed during analysis.

**Class Definition:**

```python
@dataclass
class HistoryEntry:
    """Single history entry recording an operation.

    Attributes:
        operation: Operation name (function/method called)
        parameters: Input parameters
        result: Operation result (summary)
        timestamp: When operation was performed
        duration_ms: Operation duration in milliseconds
        success: Whether operation succeeded
        error_message: Error message if failed
        metadata: Additional metadata
    """
```

### `OperationHistory`

History tracking and replay system for analysis operations.

**Class Definition:**

```python
@dataclass
class OperationHistory:
    """History of analysis operations.

    Attributes:
        entries: List of history entries
        max_entries: Maximum entries to keep (0 = unlimited)
        auto_record: Whether to automatically record operations
    """
```

**Example - Tracking Operations:**

```python
from oscura.session import OperationHistory

# Create history
history = OperationHistory(max_entries=100)

# Record operations
history.record(
    "load_trace",
    parameters={"path": "capture.wfm"},
    result="Loaded successfully",
    duration_ms=45.3
)

history.record(
    "measure_frequency",
    parameters={"trace": "CH1"},
    result="1000000.0 Hz",
    duration_ms=2.1
)

history.record(
    "apply_filter",
    parameters={"type": "lowpass", "cutoff": 1e6},
    success=False,
    error_message="Cutoff frequency too high"
)

# Query history
all_ops = history.entries
successful = history.find(success_only=True)
measurements = history.find(operation="measure_frequency")

# Get summary statistics
stats = history.summary()
print(f"Total operations: {stats['total_operations']}")
print(f"Success rate: {stats['successful']}/{stats['total_operations']}")
print(f"Total time: {stats['total_duration_ms']:.1f} ms")
```

**Example - Script Generation:**

```python
import oscura as osc

session = osc.Session()
session.load_trace("signal.wfm")
osc.frequency(session.get_trace("signal"))
osc.rise_time(session.get_trace("signal"))

# Export history as Python script
script = session.history.to_script()
print(script)
```

Output:

```python
#!/usr/bin/env python3
"""Oscura analysis script.

Generated: 2026-01-08T10:30:00
"""

import oscura as osc

# 10:30:01 - load_trace
osc.load_trace(path="signal.wfm")

# 10:30:02 - measure_frequency
osc.measure_frequency()

# 10:30:03 - measure_rise_time
osc.measure_rise_time()
```

## Audit Trail

### `AuditTrail`

Tamper-evident audit trail with HMAC chain verification for compliance.

**Class Definition:**

```python
class AuditTrail:
    """Tamper-evident audit trail with HMAC chain verification.

    Maintains a chain of audit entries where each entry is cryptographically
    linked to the previous entry using HMAC signatures. This allows detection
    of any tampering or modification of the audit log.
    """
```

**Example - Basic Audit Trail:**

```python
import oscura as osc

# Create audit trail with secret key
# WARNING: Use secure key management in production!
audit = osc.AuditTrail(secret_key=b"your-secret-key")

# Record actions
audit.record_action(
    "load_trace",
    {"file": "oscilloscope.wfm", "size_mb": 150},
    user="alice"
)

audit.record_action(
    "compute_fft",
    {"samples": 1000000, "window": "hann"},
    user="alice"
)

audit.record_action(
    "measure_thd",
    {"fundamental_freq": 1000.0, "thd_db": -65.3},
    user="alice"
)

# Verify integrity
is_valid = audit.verify_integrity()
print(f"Audit trail valid: {is_valid}")

# Export audit log
audit.export_audit_log("audit.json", format="json")
audit.export_audit_log("audit.csv", format="csv")
```

### `AuditEntry`

Single audit trail entry with HMAC signature.

**Class Definition:**

```python
@dataclass
class AuditEntry:
    """Single audit trail entry with HMAC signature.

    Attributes:
        timestamp: ISO 8601 timestamp (UTC) of the action
        action: Action identifier (e.g., "load_trace")
        details: Additional details about the action
        user: Username who performed the action
        host: Hostname where action was performed
        previous_hash: HMAC of the previous entry
        hmac: HMAC signature of this entry
    """
```

### Audit Trail Methods

#### `record_action()`

Record an auditable action.

```python
def record_action(
    action: str,
    details: dict[str, Any],
    user: str | None = None,
) -> AuditEntry:
    """Record an auditable action.

    Args:
        action: Action identifier.
        details: Dictionary of action details.
        user: Username (defaults to current user).

    Returns:
        Created AuditEntry.
    """
```

**Example:**

```python
audit = osc.AuditTrail(secret_key=b"key")

# Record with automatic user detection
entry1 = audit.record_action(
    "load_trace",
    {"file": "data.wfm", "size_mb": 100}
)

# Record with explicit user
entry2 = audit.record_action(
    "export_results",
    {"format": "csv", "rows": 1000},
    user="bob"
)

print(f"Action: {entry2.action}")
print(f"User: {entry2.user}")
print(f"Host: {entry2.host}")
print(f"Time: {entry2.timestamp}")
```

#### `verify_integrity()`

Verify HMAC chain integrity.

```python
def verify_integrity() -> bool:
    """Verify HMAC chain integrity.

    Returns:
        True if audit trail is intact and untampered.
    """
```

**Example:**

```python
audit = osc.AuditTrail(secret_key=b"key")
audit.record_action("action1", {"value": 100})
audit.record_action("action2", {"value": 200})

# Verify integrity
assert audit.verify_integrity()  # Should pass

# Detect tampering
audit._entries[0].details["value"] = 999
assert not audit.verify_integrity()  # Should fail
```

#### `get_entries()`

Query audit entries with filtering.

```python
def get_entries(
    since: datetime | None = None,
    action_type: str | None = None,
) -> list[AuditEntry]:
    """Query audit entries with optional filtering.

    Args:
        since: Return only entries after this datetime (UTC).
        action_type: Return only entries with this action type.

    Returns:
        List of matching AuditEntry objects.
    """
```

**Example:**

```python
from datetime import datetime, UTC, timedelta

audit = osc.AuditTrail(secret_key=b"key")

# Record various actions
audit.record_action("load", {"file": "a.wfm"})
audit.record_action("analyze", {"type": "fft"})
audit.record_action("load", {"file": "b.wfm"})
audit.record_action("export", {"format": "csv"})

# Query by action type
loads = audit.get_entries(action_type="load")
print(f"Load operations: {len(loads)}")

# Query by time
one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
recent = audit.get_entries(since=one_hour_ago)
print(f"Recent entries: {len(recent)}")
```

#### `export_audit_log()`

Export audit trail to file.

```python
def export_audit_log(
    path: str,
    format: Literal["json", "csv"] = "json",
) -> None:
    """Export audit trail to file.

    Args:
        path: Path to export file.
        format: Export format (json or csv).
    """
```

**Example:**

```python
audit = osc.AuditTrail(secret_key=b"key")
audit.record_action("test1", {"value": 1})
audit.record_action("test2", {"value": 2})

# Export as JSON (human-readable, structured)
audit.export_audit_log("audit.json", format="json")

# Export as CSV (spreadsheet-friendly)
audit.export_audit_log("audit.csv", format="csv")
```

### Global Audit Trail

#### `get_global_audit_trail()`

Get or create the global audit trail singleton.

```python
def get_global_audit_trail(
    secret_key: bytes | None = None
) -> AuditTrail:
    """Get or create the global audit trail.

    Args:
        secret_key: Secret key (only used on first call).

    Returns:
        Global AuditTrail instance.
    """
```

#### `record_audit()`

Record to the global audit trail.

```python
def record_audit(
    action: str,
    details: dict[str, Any],
    user: str | None = None,
) -> AuditEntry:
    """Record an action to the global audit trail.

    Args:
        action: Action identifier.
        details: Action details.
        user: Username (defaults to current user).

    Returns:
        Created AuditEntry.
    """
```

**Example - Global Audit Trail:**

```python
import oscura as osc

# Use global singleton (convenient for simple cases)
osc.record_audit("start_analysis", {"project": "power_supply"})
osc.record_audit("load_trace", {"file": "capture.wfm"})
osc.record_audit("compute_measurements", {"count": 10})

# Access global trail
audit = osc.get_global_audit_trail()
entries = audit.get_entries()
print(f"Total audit entries: {len(entries)}")

# Verify and export
assert audit.verify_integrity()
audit.export_audit_log("global_audit.json")
```

## Complete Examples

### Example 1: Complete Analysis Session

```python
import oscura as osc
import numpy as np

# Create session with metadata
session = osc.Session(name="Motor Controller Debug")
session.metadata["project"] = "MC-2024"
session.metadata["engineer"] = "Alice"

# Load multiple traces
session.load_trace("pwm_signal.wfm", name="PWM")
session.load_trace("current_sense.wfm", name="CURRENT")
session.load_trace("can_bus.sr", name="CAN")

# Perform measurements
pwm = session.get_trace("PWM")
freq = osc.frequency(pwm)
duty = osc.duty_cycle(pwm)

session.record_measurement("pwm_frequency", freq, unit="Hz", trace_name="PWM")
session.record_measurement("duty_cycle", duty, unit="%", trace_name="PWM")

# Add structured annotations
session.annotate("Motor start", time=0, layer="events")
session.annotate("Acceleration phase", time_range=(0, 0.1), layer="phases")
session.annotate("Steady state", time_range=(0.1, 0.5), layer="phases")
session.annotate("Overshoot detected", time=0.05, layer="issues", color="#FF0000")

# Save session
path = session.save("motor_debug.tks")
print(f"Session saved to {path}")

# Print summary
print("\n" + session.summary())
```

### Example 2: Session with Audit Trail

```python
import oscura as osc

# Create session with audit trail
session = osc.Session(name="Compliance Test")
audit = osc.AuditTrail(secret_key=b"compliance-secret-key")

# Record all operations
audit.record_action("create_session", {"name": session.name})

# Load and analyze
trace = session.load_trace("test_signal.wfm")
audit.record_action("load_trace", {"file": "test_signal.wfm"})

# Measurements with audit
freq = osc.frequency(trace)
session.record_measurement("frequency", freq, "Hz")
audit.record_action(
    "measure_frequency",
    {"result": freq, "unit": "Hz", "standard": "IEEE 181"}
)

thd = osc.thd(trace)
session.record_measurement("thd", thd, "%")
audit.record_action(
    "measure_thd",
    {"result": thd, "unit": "%", "harmonics": 10}
)

# Save both
session.save("compliance_session.tks")
audit.record_action("save_session", {"file": "compliance_session.tks"})

audit.export_audit_log("compliance_audit.json")
audit.record_action("export_audit", {"file": "compliance_audit.json"})

# Verify audit integrity
if audit.verify_integrity():
    print("Audit trail verified - no tampering detected")
else:
    print("WARNING: Audit trail integrity check failed!")
```

### Example 3: Resuming Saved Session

```python
import oscura as osc

# Load previous session
session = osc.load_session("motor_debug.tks")

print(f"Loaded session: {session.name}")
print(f"Created: {session.created_at}")
print(f"Modified: {session.modified_at}")

# Access saved data
print(f"\nTraces: {session.list_traces()}")

measurements = session.get_measurements()
for name, data in measurements.items():
    print(f"  {name}: {data['value']} {data['unit']}")

# View annotations by layer
for layer_name, layer in session.annotation_layers.items():
    print(f"\nLayer '{layer_name}':")
    for ann in layer.annotations:
        print(f"  - {ann.text} @ {ann.time}")

# Review operation history
print(f"\nOperation history ({len(session.history.entries)} entries):")
for entry in session.history.entries[:5]:
    status = "✓" if entry.success else "✗"
    print(f"  {status} {entry.operation} ({entry.duration_ms:.1f}ms)")

# Continue analysis
trace = session.get_trace("PWM")
rise = osc.rise_time(trace)
session.record_measurement("rise_time", rise, "s", trace_name="PWM")

# Save updated session
session.save()
```

### Example 4: Multi-Layer Annotations

```python
import oscura as osc

session = osc.Session(name="Protocol Analysis")
trace = session.load_trace("uart_capture.wfm")

# Protocol decode layer
session.annotate("START", time=0, layer="protocol", color="#00FF00")
session.annotate("DATA: 0x41", time_range=(1e-6, 9e-6), layer="protocol")
session.annotate("PARITY", time=9e-6, layer="protocol")
session.annotate("STOP", time=10e-6, layer="protocol")

# Timing analysis layer
session.annotate("Bit 0", time_range=(1e-6, 2e-6), layer="timing")
session.annotate("Bit 1", time_range=(2e-6, 3e-6), layer="timing")

# Issues layer
session.annotate(
    "Timing violation",
    time=5e-6,
    layer="issues",
    color="#FF0000",
    style="dashed"
)

# Access layers
protocol_layer = session.annotation_layers["protocol"]
protocol_layer.description = "UART protocol decode at 115200 baud"
protocol_layer.color = "#00FF00"

issues_layer = session.annotation_layers["issues"]
issues_layer.locked = False  # Allow modifications

# Query annotations
protocol_events = session.get_annotations(layer="protocol")
issues = session.get_annotations(layer="issues")

print(f"Protocol events: {len(protocol_events)}")
print(f"Issues found: {len(issues)}")

# Find annotations in specific time window
window_annotations = session.get_annotations(
    time_range=(0, 5e-6)
)
print(f"Annotations in first 5μs: {len(window_annotations)}")

session.save("uart_analysis.tks")
```

### Example 5: Compliance Audit Trail

```python
import oscura as osc
from datetime import datetime, UTC

# Initialize audit trail with production settings
# In production: load secret_key from environment or secrets manager
audit = osc.AuditTrail(
    secret_key=b"production-secret-key",
    hash_algorithm="sha256"
)

# Record compliance operations
audit.record_action(
    "calibration_check",
    {
        "device": "Oscilloscope-001",
        "calibration_date": "2026-01-01",
        "status": "valid"
    },
    user="calibration_lab"
)

audit.record_action(
    "load_test_data",
    {
        "file": "compliance_test_001.wfm",
        "sha256": "abc123...",
        "size_bytes": 1048576
    },
    user="test_engineer"
)

audit.record_action(
    "run_measurement",
    {
        "type": "THD",
        "result": -65.3,
        "unit": "dB",
        "standard": "IEC 61000-4-7",
        "pass": True
    },
    user="test_engineer"
)

audit.record_action(
    "generate_report",
    {
        "format": "PDF",
        "pages": 15,
        "includes_raw_data": True
    },
    user="test_engineer"
)

audit.record_action(
    "review_results",
    {
        "reviewer": "senior_engineer",
        "status": "approved",
        "comments": "All measurements within specification"
    },
    user="senior_engineer"
)

# Verify integrity before export
if not audit.verify_integrity():
    raise RuntimeError("Audit trail integrity check failed!")

# Export for archival
audit.export_audit_log("compliance_audit_2026-01-08.json", format="json")
audit.export_audit_log("compliance_audit_2026-01-08.csv", format="csv")

# Generate compliance report
entries = audit.get_entries()
print("COMPLIANCE AUDIT REPORT")
print("=" * 60)
print(f"Total actions: {len(entries)}")
print(f"Integrity: VERIFIED")
print(f"\nAudit chain:")
for i, entry in enumerate(entries, 1):
    print(f"{i}. {entry.timestamp} - {entry.action} by {entry.user}")
    print(f"   HMAC: {entry.hmac[:32]}...")
```

## Best Practices

### Session Management

1. **Use Descriptive Names**: Give sessions clear, descriptive names

   ```python
   session = osc.Session(name="2026-01-08_Power_Supply_Debug")
   ```

2. **Add Metadata**: Store project context in metadata

   ```python
   session.metadata["project"] = "PSU-REV-B"
   session.metadata["engineer"] = "Alice"
   session.metadata["dut_serial"] = "PSU-001234"
   ```

3. **Organize with Layers**: Use annotation layers for organization

   ```python
   session.annotate("...", layer="protocol")
   session.annotate("...", layer="measurements")
   session.annotate("...", layer="issues")
   ```

4. **Save Regularly**: Save sessions periodically during analysis

   ```python
   session.save()  # Quick save
   ```

5. **Record Measurements**: Capture all important results

   ```python
   session.record_measurement(
       name="thd",
       value=thd_value,
       unit="dB",
       trace_name="SIGNAL",
       standard="IEC 61000-4-7",
       pass_criteria=thd_value < -60
   )
   ```

### Audit Trail

1. **Secure Key Management**: Never hardcode secret keys

   ```python
   import os
   secret_key = os.environ.get("AUDIT_SECRET_KEY").encode()
   audit = osc.AuditTrail(secret_key=secret_key)
   ```

2. **Record All Actions**: Be comprehensive in what you audit

   ```python
   audit.record_action("action", {
       "parameter": value,
       "context": "information",
       "result": outcome
   })
   ```

3. **Verify Regularly**: Check integrity before critical operations

   ```python
   assert audit.verify_integrity(), "Audit tampering detected!"
   ```

4. **Export for Archival**: Save audit logs for long-term storage

   ```python
   from datetime import datetime
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   audit.export_audit_log(f"audit_{timestamp}.json")
   ```

5. **Include Context**: Add relevant details to audit entries

   ```python
   audit.record_action(
       "measurement",
       {
           "type": "THD",
           "standard": "IEC 61000-4-7",
           "equipment": "OSC-001",
           "calibration_date": "2026-01-01",
           "result": result_value
       }
   )
   ```

## See Also

- [Analysis API](analysis.md) - Measurement functions
- [Loader API](loader.md) - Loading trace data
- [Export API](export.md) - Exporting results
- [Reporting API](reporting.md) - Report generation
