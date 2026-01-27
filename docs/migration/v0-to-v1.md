# Migration Guide: v0.x to v1.0

**Version**: 0.6.0 → 1.0.0 (Future)
**Last Updated**: 2026-01-25

This guide helps you migrate from Oscura v0.x to v1.0 when released. While v1.0 is not yet available, this document tracks breaking changes and provides migration paths for each change introduced during the 0.x series.

---

## BREAKING CHANGES - Clean Architecture

**IMPORTANT**: Oscura v1.0 introduces a clean, unified architecture with **NO backward compatibility** for deprecated APIs. This is a clean break to establish a solid foundation.

### What Was Removed

**CANSession Legacy Methods**:

- `CANSession.from_log()` - REMOVED
- `CANSession.from_messages()` - REMOVED

**SignalBuilder Legacy Methods**:

- `SignalBuilder.build_as_generated_signal()` - REMOVED
- `GeneratedSignal` type - REMOVED

**Session File Format**:

- Legacy session files (pre-v0.3.0) - NO LONGER SUPPORTED
- Old session files must be re-saved with new format

### New Required Patterns

**CANSession** (Unified Source Protocol):

```python
# OLD (REMOVED):
session = CANSession.from_log("file.blf")
session = CANSession.from_messages(message_list)

# NEW (REQUIRED):
from oscura.acquisition import FileSource
from oscura.automotive.can import CANSession

session = CANSession()
session.add_recording("baseline", FileSource("file.blf"))
```

**SignalBuilder** (Unified Type):

```python
# OLD (REMOVED):
signal = builder.build_as_generated_signal()
trace = signal.to_trace()

# NEW (REQUIRED):
trace = builder.build()  # Returns WaveformTrace directly
```

**Migration Timeline**:

- **v0.3.0**: Deprecation warnings issued
- **v1.0.0**: Old APIs completely removed (BREAKING)

---

## Overview

Oscura follows **semantic versioning**:

- **0.x versions**: Breaking changes allowed with deprecation warnings
- **1.0+ versions**: Backward compatibility maintained, major version bump for breaking changes

This guide covers all breaking changes from v0.1.0 through v0.3.0 and planned changes for v1.0.

---

## Breaking Changes by Version

### v0.3.0 Changes

#### 1. SignalBuilder Type Unification (BREAKING in v1.0)

**Change**: `SignalBuilder.build()` now returns `WaveformTrace` instead of `GeneratedSignal`.

**Before (v0.2.x)**:

```python
from oscura import SignalBuilder

builder = SignalBuilder(sample_rate=1e6, duration=0.01)
signal = builder.add_sine(frequency=1000).build()
# signal is GeneratedSignal, not compatible with WaveformTrace

# Had to convert manually
trace = signal.to_trace()
```

**After (v1.0.0)**:

```python
from oscura import SignalBuilder

builder = SignalBuilder(sample_rate=1e6, duration=0.01)
trace = builder.add_sine(frequency=1000).build()
# trace is WaveformTrace - directly composable!
```

**Migration**:

- Remove `.to_trace()` calls after `.build()`
- Old code: `builder.build().to_trace()` → New code: `builder.build()`
- For multi-channel signals, use `.build_channels()` instead of `.build()`

**Status**:

- **v0.3.0**: `build_as_generated_signal()` deprecated (issues warning)
- **v1.0.0**: `build_as_generated_signal()` and `GeneratedSignal` REMOVED (BREAKING)

---

#### 2. Unified Acquisition via Source Protocol

**Change**: New `Source` protocol for unified acquisition pattern.

**Before (v0.2.x)**:

```python
from oscura import load

# Only file-based loading
trace = load("capture.wfm")
```

**After (v0.3.0)**:

```python
from oscura.acquisition import FileSource, SyntheticSource
from oscura import SignalBuilder

# File-based
source = FileSource("capture.wfm")
trace = source.read()

# Synthetic (new capability)
builder = SignalBuilder().add_sine(frequency=1000)
source = SyntheticSource(builder)
trace = source.read()

# Polymorphic usage
def analyze_from_source(source: Source):
    trace = source.read()
    return analyze(trace)
```

**Migration**:

- Old `load()` function still works (backward compatible)
- New code should use `FileSource` for consistency
- Benefits: Enables hardware sources (Phase 2), streaming, unified interface

**Backward Compatibility**:

```python
# Still works, no changes needed
from oscura import load
trace = load("capture.wfm")
```

**Timeline**: `load()` will remain for convenience, but `FileSource` is preferred.

---

#### 3. Session Management Hierarchy

**Change**: New `AnalysisSession` base class for domain-specific sessions.

**Before (v0.2.x)**:

```python
# No unified session pattern - ad-hoc approaches
```

**After (v0.3.0)**:

```python
from oscura.sessions import BlackBoxSession, GenericSession
from oscura.acquisition import FileSource

# Domain-specific sessions
session = BlackBoxSession(name="IoT Protocol Analysis")
session.add_recording("baseline", FileSource("idle.bin"))
session.add_recording("active", FileSource("running.bin"))

# Comparison
diff = session.compare("baseline", "active")
print(f"Changed bytes: {diff.changed_bytes}")

# Analysis
spec = session.generate_protocol_spec()
```

**Migration**:

- No breaking changes - this is a new capability
- Existing analysis code continues to work
- Consider migrating to sessions for interactive workflows

**Timeline**: Sessions are stable and recommended for new code.

---

### Planned v1.0 Changes

These changes are planned for v1.0 but not yet implemented:

#### 1. Hardware Acquisition Sources

**Status**: Placeholder in v0.3.0, full implementation planned for v1.0.

**Future API**:

```python
from oscura.acquisition import HardwareSource

# SocketCAN (Linux CAN interface)
can = HardwareSource.socketcan("can0", bitrate=500000)
trace = can.read()

# Saleae Logic Analyzer
logic = HardwareSource.saleae(device_id="ABC123")
for chunk in logic.stream(duration=60):
    analyze(chunk)

# PyVISA Oscilloscope
scope = HardwareSource.visa("USB0::0x0699::0x0401::INSTR")
scope.configure(channels=[1, 2], timebase=1e-6)
trace = scope.read()
```

**Migration Path**:

- Code using `Source` protocol will work unchanged
- Hardware sources will be drop-in replacements for `FileSource`

---

#### 2. CANSession Refactor (BREAKING in v1.0)

**Change**: CANSession now extends AnalysisSession and uses unified Source protocol.

**Before (v0.2.x)**:

```python
from oscura.automotive.can import CANSession

# Legacy class methods (REMOVED in v1.0)
session = CANSession.from_log("idle.blf")
session = CANSession.from_messages(message_list)
```

**After (v1.0.0)**:

```python
from oscura.automotive.can import CANSession
from oscura.acquisition import FileSource

# NEW REQUIRED PATTERN
session = CANSession(name="Vehicle Analysis")
session.add_recording("baseline", FileSource("idle.blf"))

# Analyze
analysis = session.analyze()
print(f"Messages: {analysis['inventory']['total_messages']}")

# Compare recordings
session.add_recording("active", FileSource("active.blf"))
diff = session.compare("baseline", "active")

# Export DBC
session.export_dbc("output.dbc")
```

**Migration Path**:

1. Replace `CANSession.from_log(path)` with:

   ```python
   session = CANSession()
   session.add_recording("default", FileSource(path))
   ```

2. Replace `CANSession.from_messages(msgs)` with:

   ```python
   # Save messages to file first, then load via FileSource
   # OR use direct message addition (if supported in your version)
   ```

**Status**:

- **v0.3.0**: `from_log()` and `from_messages()` deprecated (issues warning)
- **v1.0.0**: Legacy methods REMOVED (BREAKING)

---

## Migration Checklist

Use this checklist to migrate your codebase from v0.2.x to v0.3.0:

### Phase 1: Immediate Changes (v0.3.0)

- [ ] **SignalBuilder**: Remove `.to_trace()` calls after `.build()`

  ```python
  # Before: builder.build().to_trace()
  # After:  builder.build()
  ```

- [ ] **Multi-channel signals**: Use `.build_channels()`

  ```python
  # Before: trace = builder.build()  # Which channel?
  # After:  channels = builder.build_channels()
  ```

- [ ] **Update imports** if using deprecated functions

  ```python
  # Check for deprecation warnings in logs
  ```

### Phase 2: Recommended Updates (v0.3.0)

- [ ] **Adopt Source protocol** for new code

  ```python
  from oscura.acquisition import FileSource
  source = FileSource("capture.wfm")
  trace = source.read()
  ```

- [ ] **Use sessions** for interactive analysis

  ```python
  from oscura.sessions import BlackBoxSession
  session = BlackBoxSession()
  session.add_recording("test", FileSource("data.bin"))
  ```

### Phase 3: Future-Proofing (v1.0 prep)

- [ ] **Remove deprecated API usage**:
  - Replace `build_as_generated_signal()` with `build()`
  - Remove `GeneratedSignal` type annotations

- [ ] **Prepare for hardware sources**:
  - Use `Source` protocol in type hints
  - Design code to work with any acquisition source

- [ ] **Consider session-based workflows**:
  - Evaluate if interactive sessions fit your use case
  - Migrate complex analysis to session pattern

---

## Common Migration Patterns

### Pattern 1: Signal Generation

**Before**:

```python
from oscura import SignalBuilder

builder = SignalBuilder(sample_rate=1e6, duration=0.01)
signal = builder.add_sine(frequency=1000).build()
trace = signal.to_trace()

# Use trace
analyze(trace)
```

**After**:

```python
from oscura import SignalBuilder

builder = SignalBuilder(sample_rate=1e6, duration=0.01)
trace = builder.add_sine(frequency=1000).build()

# Use trace directly
analyze(trace)
```

---

### Pattern 2: File Loading with Processing

**Before**:

```python
from oscura import load

trace = load("capture.wfm")
analyze(trace)
```

**After** (Option 1 - Backward Compatible):

```python
from oscura import load

trace = load("capture.wfm")
analyze(trace)
```

**After** (Option 2 - New Pattern):

```python
from oscura.acquisition import FileSource

with FileSource("capture.wfm") as source:
    trace = source.read()
    analyze(trace)
```

---

### Pattern 3: Multi-Channel Analysis

**Before**:

```python
from oscura import SignalBuilder

builder = SignalBuilder(sample_rate=1e6, duration=0.01)
builder.add_sine(frequency=1000, channel="ch1")
builder.add_sine(frequency=2000, channel="ch2")

# Unclear which channel
trace = builder.build()
```

**After**:

```python
from oscura import SignalBuilder

builder = SignalBuilder(sample_rate=1e6, duration=0.01)
builder.add_sine(frequency=1000, channel="ch1")
builder.add_sine(frequency=2000, channel="ch2")

# Explicit: get all channels
channels = builder.build_channels()
ch1 = channels["ch1"]
ch2 = channels["ch2"]
```

---

### Pattern 4: Interactive Analysis

**Before**:

```python
# Ad-hoc analysis scripts
trace1 = load("baseline.bin")
trace2 = load("test.bin")

# Manual comparison
diff = compare_traces(trace1, trace2)
```

**After**:

```python
from oscura.sessions import BlackBoxSession
from oscura.acquisition import FileSource

session = BlackBoxSession(name="Device Analysis")
session.add_recording("baseline", FileSource("baseline.bin"))
session.add_recording("test", FileSource("test.bin"))

# Built-in comparison
diff = session.compare("baseline", "test")
print(f"Changed bytes: {diff.changed_bytes}")

# Generate protocol spec
spec = session.generate_protocol_spec()
```

---

## Troubleshooting

### Issue: "AttributeError: 'WaveformTrace' object has no attribute 'to_trace'"

**Cause**: Code trying to call `.to_trace()` on a `WaveformTrace` (already the target type).

**Solution**: Remove `.to_trace()` call.

```python
# Before: trace = builder.build().to_trace()
# After:  trace = builder.build()
```

---

### Issue: "DeprecationWarning: build_as_generated_signal() is deprecated"

**Cause**: Using deprecated method instead of new `.build()`.

**Solution**: Replace with `.build()`:

```python
# Before: signal = builder.build_as_generated_signal()
# After:  trace = builder.build()
```

---

### Issue: "TypeError: Expected Source, got str"

**Cause**: Passing filename string where `Source` object expected.

**Solution**: Wrap filename in `FileSource`:

```python
from oscura.acquisition import FileSource

# Before: session.add_recording("test", "file.bin")
# After:  session.add_recording("test", FileSource("file.bin"))
```

---

### Issue: "Cannot access channel - use build_channels()"

**Cause**: Trying to build multi-channel signal with `.build()`.

**Solution**: Use `.build_channels()` for multi-channel signals:

```python
# Before: trace = builder.build()  # Ambiguous
# After:  channels = builder.build_channels()
```

---

## Testing Your Migration

### 1. Run Tests with Warnings

Enable deprecation warnings to find issues:

```bash
python -W default -m pytest tests/
```

### 2. Check for Deprecated Imports

```bash
grep -r "build_as_generated_signal" src/
grep -r "GeneratedSignal" src/
```

### 3. Validate Type Hints

```bash
./scripts/check.sh  # Runs mypy type checking
```

### 4. Review Logs

Check application logs for deprecation warnings:

```python
import warnings
warnings.simplefilter("default")  # Show all deprecation warnings
```

---

## Getting Help

If you encounter migration issues:

1. **Check CHANGELOG.md**: Detailed change descriptions
2. **Review demos**: Updated examples for each feature
3. **Read API docs**: Complete API reference documentation
4. **GitHub Issues**: Report migration problems
5. **Discussions**: Ask questions in community forum

---

## Timeline

| Version | Release    | Key Changes                                   | Migration Required              |
| ------- | ---------- | --------------------------------------------- | ------------------------------- |
| v0.3.0  | 2026-01-20 | Type unification, Source protocol, Sessions   | Optional (deprecation warnings) |
| v0.4.0  | TBD        | Hardware sources (Phase 2)                    | No                              |
| v0.5.0  | TBD        | Remove deprecated APIs                        | Yes (remove deprecated code)    |
| v1.0.0  | TBD        | Stable API, backward compatibility guaranteed | No                              |

---

## Summary

**Key Takeaways**:

1. **v0.3.0 changes are backward compatible** - deprecation warnings guide migration
2. **SignalBuilder type unification** - biggest change, but simple fix (remove `.to_trace()`)
3. **New capabilities** - Source protocol and Sessions are additions, not replacements
4. **Gradual migration** - update at your own pace, deprecated APIs work until v0.5.0
5. **Future-proof** - Code using new patterns will work unchanged in v1.0+

**Migration Priority**:

- **High**: Remove `.to_trace()` calls (simple, prevents future breaks)
- **Medium**: Adopt Source protocol for new code (better architecture)
- **Low**: Migrate to sessions (only if beneficial for your workflow)

---

**Next Steps**: See individual guides for specific features:

- [BlackBox Analysis Guide](../guides/blackbox-analysis.md)
- [Hardware Acquisition Guide](../guides/hardware-acquisition.md)
- [Session Management Guide](../api/session-management.md)
