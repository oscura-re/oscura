# Session Management & Audit Trail API

> **Version**: 0.8.0
> **Last Updated**: 2026-01-25

## Overview

Oscura provides comprehensive session management and audit trail capabilities for tracking, saving, and resuming analysis work. Sessions capture traces, annotations, measurements, and operation history, while audit trails provide tamper-evident logging for compliance requirements.

## Quick Start

```python
import oscura as osc
from oscura.sessions import BlackBoxSession, GenericSession

# Create and work with a session (use BlackBoxSession for protocol RE)
session = BlackBoxSession(name="Power Supply Analysis")

# Or use GenericSession for general waveform analysis
session = GenericSession(name="Power Supply Analysis")
session.add_recording("capture", source)  # Add data source
results = session.analyze()  # Run analysis
print(results["summary"])

# Enable audit trail for compliance
audit = osc.AuditTrail(secret_key=b"your-secret-key")
audit.record_action("load_trace", {"file": "data.wfm"})
assert audit.verify_integrity()
audit.export_audit_log("audit.json", format="json")
```

> **Note**: The legacy `osc.Session()` and `osc.load_session()` APIs have been removed.
> Use `BlackBoxSession`, `GenericSession`, or `AnalysisSession` from `oscura.sessions` instead.

## Session Management

### Session Classes

Oscura provides three session classes for different use cases:

| Class | Use Case |
|-------|----------|
| `BlackBoxSession` | Protocol reverse engineering, differential analysis |
| `GenericSession` | General waveform analysis workflows |
| `AnalysisSession` | Abstract base class for custom sessions |

**Import Pattern:**

```python
from oscura.sessions import BlackBoxSession, GenericSession, AnalysisSession
```

### `BlackBoxSession`

Specialized session for unknown protocol reverse engineering with differential analysis.

**Example - Protocol Reverse Engineering:**

```python
from oscura.sessions import BlackBoxSession

# Create session for protocol analysis
session = BlackBoxSession(name="IoT Device Protocol Analysis")

# Add captures (recordings) for analysis
session.add_recording("baseline", source)
session.add_recording("button_press", source)

# Compare captures to find differences
diff = session.compare("baseline", "button_press")
print(f"Changed bytes: {diff.changed_bytes}")
print(f"Similarity: {diff.similarity_score:.4f}")

# Run comprehensive analysis
results = session.analyze()
print(f"Fields detected: {len(results['field_hypotheses'])}")

# Generate protocol specification
spec = session.generate_protocol_spec()

# Export results
session.export_results("dissector", "protocol.lua")  # Wireshark dissector
session.export_results("spec", "protocol.json")       # JSON specification
```

### `GenericSession`

General-purpose session for waveform analysis workflows.

**Example - General Waveform Analysis:**

```python
import oscura as osc
from oscura.sessions import GenericSession

# Create session
session = GenericSession(name="Power Rail Analysis")

# Add recordings
session.add_recording("5V_RAIL", source)

# Run analysis
results = session.analyze()
print(results["summary"])

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

## Session API Reference

> **Note**: The following methods document the legacy `Session` API for historical reference.
> New code should use `BlackBoxSession` or `GenericSession` from `oscura.sessions`.
> See the Quick Start section for current usage patterns.

### Recording Management

Sessions manage recordings (data sources) using `add_recording()` and `list_recordings()`:

```python
from oscura.sessions import BlackBoxSession

session = BlackBoxSession(name="Analysis")

# Add recordings
session.add_recording("baseline", source)
session.add_recording("stimulus", source)

# List recordings
recordings = session.list_recordings()
print(recordings)  # ['baseline', 'stimulus']
```

### Differential Analysis (BlackBoxSession)

```python
from oscura.sessions import BlackBoxSession

session = BlackBoxSession(name="Protocol RE")
session.add_recording("baseline", source)
session.add_recording("button_press", source)

# Compare two recordings
diff = session.compare("baseline", "button_press")
print(f"Changed bytes: {diff.changed_bytes}")
print(f"Similarity: {diff.similarity_score:.4f}")
print(f"Changed regions: {diff.changed_regions}")
```

### Export Results

```python
from oscura.sessions import BlackBoxSession

session = BlackBoxSession(name="Protocol RE")
# ... add recordings and analyze ...

# Export to various formats
session.export_results("report", "analysis.md")      # Markdown report
session.export_results("dissector", "proto.lua")    # Wireshark dissector
session.export_results("spec", "proto.json")        # JSON specification
session.export_results("csv", "fields.csv")         # CSV format
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

> **Deprecated**: The legacy `osc.Session()` and `osc.load_session()` APIs have been removed.
> Use the new session classes from `oscura.sessions` instead.

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

> **Note**: The annotation layer example below shows conceptual usage.
> For current API usage, see BlackBoxSession and GenericSession documentation.

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
from oscura.sessions import GenericSession

session = GenericSession(name="Signal Analysis")
trace = osc.load("signal.wfm")
session.add_recording("signal", trace)

# Perform measurements
freq = osc.frequency(trace)
rise = osc.rise_time(trace)
print(f"Frequency: {freq} Hz, Rise time: {rise} s")
```

> **Note**: Script generation from operation history is a planned feature.
> For now, use GenericSession.analyze() to get analysis results programmatically.

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
from oscura.sessions import GenericSession

# Create session with metadata
session = GenericSession(name="Motor Controller Debug")
session.metadata["project"] = "MC-2024"
session.metadata["engineer"] = "Alice"

# Load traces and add to session
pwm = osc.load("pwm_signal.wfm")
current = osc.load("current_sense.wfm")
can_trace = osc.load("can_bus.sr")

session.add_recording("PWM", pwm)
session.add_recording("CURRENT", current)
session.add_recording("CAN", can_trace)

# Perform measurements
freq = osc.frequency(pwm)
duty = osc.duty_cycle(pwm)
print(f"PWM Frequency: {freq} Hz, Duty Cycle: {duty}%")

# Run analysis on all recordings
results = session.analyze()
print(f"Analyzed {results['num_recordings']} recordings")

# Export results
session.export_results("report", "motor_debug.md")
session.export_results("json", "motor_debug.json")
```

### Example 2: Session with Audit Trail

```python
import oscura as osc
from oscura.sessions import GenericSession

# Create session with audit trail
session = GenericSession(name="Compliance Test")
audit = osc.AuditTrail(secret_key=b"compliance-secret-key")

# Record all operations
audit.record_action("create_session", {"name": session.name})

# Load and add to session
trace = osc.load("test_signal.wfm")
session.add_recording("test_signal", trace)
audit.record_action("load_trace", {"file": "test_signal.wfm"})

# Measurements with audit
freq = osc.frequency(trace)
audit.record_action(
    "measure_frequency",
    {"result": freq, "unit": "Hz", "standard": "IEEE 181"}
)

thd = osc.thd(trace)
audit.record_action(
    "measure_thd",
    {"result": thd, "unit": "%", "harmonics": 10}
)

# Export results and audit log
session.export_results("json", "compliance_session.json")
audit.record_action("export_session", {"file": "compliance_session.json"})

audit.export_audit_log("compliance_audit.json")
audit.record_action("export_audit", {"file": "compliance_audit.json"})

# Verify audit integrity
if audit.verify_integrity():
    print("Audit trail verified - no tampering detected")
else:
    print("WARNING: Audit trail integrity check failed!")
```

### Example 3: Multi-Recording Workflow

```python
import oscura as osc
from oscura.sessions import BlackBoxSession

# Create session for protocol reverse engineering
session = BlackBoxSession(name="IoT Device Analysis")

# Add multiple captures
baseline = osc.load("baseline.wfm")
button_press = osc.load("button_press.wfm")
temp_change = osc.load("temp_change.wfm")

session.add_recording("baseline", baseline)
session.add_recording("button_press", button_press)
session.add_recording("temp_change", temp_change)

print(f"Session: {session.name}")
print(f"Created: {session.created_at}")
print(f"Recordings: {session.list_recordings()}")

# Compare recordings (differential analysis)
diff = session.compare("baseline", "button_press")
print(f"\nBaseline vs Button Press:")
print(f"  Changed bytes: {diff.changed_bytes}")
print(f"  Similarity: {diff.similarity_score:.4f}")

# Run comprehensive analysis
results = session.analyze()
print(f"\nAnalysis Results:")
print(f"  Fields detected: {len(results.get('field_hypotheses', []))}")

# Generate protocol specification
spec = session.generate_protocol_spec()
print(f"  Protocol spec: {spec.name}")

# Export in multiple formats
session.export_results("report", "analysis.md")
session.export_results("dissector", "protocol.lua")
session.export_results("spec", "protocol.json")
```

### Example 4: Protocol Decoding with Sessions

```python
import oscura as osc
from oscura.sessions import GenericSession

# Create session for UART analysis
session = GenericSession(name="UART Protocol Analysis")

# Load UART capture
trace = osc.load("uart_capture.wfm")
session.add_recording("uart", trace)

# Decode UART protocol
messages = osc.decode_uart(trace, baud_rate=115200)
print(f"Decoded {len(messages)} UART frames")

for i, msg in enumerate(messages[:5]):
    print(f"  Frame {i}: {msg}")

# Run session analysis for statistics
results = session.analyze()
print(f"\nSession Analysis:")
print(f"  Mean amplitude: {results['summary']['uart']['mean']:.4f}")
print(f"  RMS: {results['summary']['uart']['rms']:.4f}")

# Export results
session.export_results("report", "uart_analysis.md")
session.export_results("csv", "uart_data.csv")
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
   from oscura.sessions import GenericSession, BlackBoxSession

   session = GenericSession(name="2026-01-08_Power_Supply_Debug")
   # Or for protocol RE:
   session = BlackBoxSession(name="IoT_Protocol_RE")
   ```

2. **Add Metadata**: Store project context in metadata

   ```python
   session.metadata["project"] = "PSU-REV-B"
   session.metadata["engineer"] = "Alice"
   session.metadata["dut_serial"] = "PSU-001234"
   ```

3. **Use Meaningful Recording Names**: Name recordings descriptively

   ```python
   session.add_recording("baseline_idle", baseline_trace)
   session.add_recording("button_press", stimulus_trace)
   session.add_recording("temp_25C", temp_trace)
   ```

4. **Export Results Regularly**: Save analysis results periodically

   ```python
   session.export_results("report", "analysis.md")
   session.export_results("json", "results.json")
   ```

5. **Use Compare for Differential Analysis**: Find changes between captures

   ```python
   diff = session.compare("baseline", "stimulus")
   print(f"Changed: {diff.changed_bytes} bytes")
   print(f"Similarity: {diff.similarity_score:.4f}")
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
