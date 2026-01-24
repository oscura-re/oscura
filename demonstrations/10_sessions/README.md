# Analysis Sessions and Interactive Workflows

**Manage multi-recording analysis workflows with persistent sessions.**

This section contains 5 demonstrations designed to teach you how to use Oscura's session system for interactive analysis, multi-recording workflows, and persistent analysis state. Perfect for interactive reverse engineering, comparative analysis, and long-running investigation workflows.

---

## Prerequisites

See [main demonstrations README](../README.md#installation) for installation instructions.

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_analysis_session.py → 02_can_session.py → 03_blackbox_session.py
        ↓                       ↓                       ↓
   Core session API       Domain-specific         Reverse engineering
   Multi-recording        CAN bus analysis        Differential analysis
   Comparison ops         Pattern discovery       Field inference

        ↓                       ↓
04_session_persistence.py → 05_interactive_analysis.py
        ↓                       ↓
   Save/load state         Interactive workflows
   Metadata tracking       Hypothesis testing
   Audit trails            Collaborative analysis
```

### Estimated Time

| Demo                    | Time       | Difficulty                | Topics                                     |
| ----------------------- | ---------- | ------------------------- | ------------------------------------------ |
| 01_analysis_session     | 15 min     | Intermediate              | Core sessions, multi-recording, comparison |
| 02_can_session          | 15 min     | Intermediate              | CAN-specific analysis, pattern discovery   |
| 03_blackbox_session     | 20 min     | Advanced                  | Protocol RE, differential analysis         |
| 04_session_persistence  | 10 min     | Intermediate              | State tracking, metadata, annotations      |
| 05_interactive_analysis | 15 min     | Intermediate              | Interactive workflows, hypothesis testing  |
| **Total**               | **75 min** | **Intermediate-Advanced** | **Session mastery**                        |

---

## Demonstrations

### Demo 01: Analysis Session

**File**: `01_analysis_session.py`

**What it teaches**:

- AnalysisSession base class usage and patterns
- GenericSession for non-domain-specific analysis
- Recording management (add, get, list recordings)
- Comparison operations between recordings
- Multi-recording analysis workflows
- Session lifecycle management

**What you'll do**:

1. Create a GenericSession for multi-recording analysis
2. Add multiple recordings to the session
3. List and query recordings by name/metadata
4. Compare recordings (differential analysis)
5. Access recording data and metadata
6. Close session and cleanup resources

**Key concepts**:

- `AnalysisSession` - Base class for all session types
- `GenericSession` - General-purpose session
- Recording management - Add, get, list, remove recordings
- Comparison operations - Differential analysis between recordings
- Session state - Track analysis progress

**Use cases**:

- **Before/after testing** - Compare signal before and after modification
- **Multi-device testing** - Analyze signals from multiple devices
- **Time-series analysis** - Track signal changes over time
- **Regression testing** - Compare current vs baseline captures

**Why this matters**: Sessions provide a unified interface for managing multiple related recordings, enabling sophisticated comparative analysis and interactive workflows.

---

### Demo 02: CAN Session

**File**: `02_can_session.py`

**What it teaches**:

- Domain-specific sessions extend AnalysisSession
- CAN bus analysis workflows (concept demonstration)
- Message inventory generation patterns
- Pattern discovery (pairs, sequences, correlations)
- Stimulus-response analysis concepts
- Integration with domain-specific tools

**What you'll do**:

1. Understand domain-specific session patterns (using GenericSession as reference)
2. Explore CAN-specific analysis concepts
3. Learn message inventory and pattern discovery
4. Understand stimulus-response correlation
5. See how domain sessions extend core capabilities

**Domain-specific features** (CAN example):

- **Message inventory** - Catalog all CAN message IDs
- **Pattern discovery** - Find recurring message pairs/sequences
- **Stimulus-response** - Correlate input events with CAN messages
- **Time-based analysis** - Message timing and periodicity
- **Integration** - Links to automotive diagnostic tools

**IEEE Standards**: ISO 11898 (CAN), ISO 15765-2 (ISO-TP)

**Why this matters**: Domain-specific sessions package specialized analysis workflows (automotive, RF, power electronics) into reusable, validated patterns. This demo shows the pattern using CAN as an example.

**Note**: Full CANSession implementation is in `oscura.automotive.can.session`. This demo illustrates the pattern with GenericSession.

---

### Demo 03: BlackBox Session

**File**: `03_blackbox_session.py`

**What it teaches**:

- BlackBoxSession for unknown protocol reverse engineering
- Differential analysis workflow
- Field hypothesis generation
- State machine inference fundamentals
- Protocol specification export
- Byte-level comparison and analysis

**What you'll do**:

1. Create a BlackBoxSession for protocol analysis
2. Add recordings with different protocol states/commands
3. Perform differential analysis to identify variable fields
4. Generate field hypotheses (address, length, checksum, etc.)
5. Infer state machine from recording sequences
6. Export discovered protocol specification

**Reverse engineering workflow**:

```
Capture data → Differential analysis → Field inference → Validation
     ↓                ↓                      ↓              ↓
  Multiple      Compare bytes          Hypothesize      Test on
  recordings    Find patterns          field types      new data
```

**Analysis techniques**:

- **Byte-level diff** - Find changing bytes between recordings
- **Field inference** - Identify addresses, lengths, checksums, data
- **Pattern matching** - Recognize common protocol structures
- **State tracking** - Build state machine from sequences
- **Validation** - Test hypotheses on new captures

**Why this matters**: BlackBoxSession is specialized for the most common reverse engineering task - understanding unknown protocols through systematic differential analysis.

---

### Demo 04: Session Persistence

**File**: `04_session_persistence.py`

**What it teaches**:

- Session metadata and annotations
- Session state tracking (created_at, modified_at)
- Session history and audit trail concepts
- Metadata management best practices
- Documentation patterns for reproducibility

**What you'll do**:

1. Create session with comprehensive metadata
2. Add annotations describing analysis steps
3. Track session state changes (timestamps, modifications)
4. Query session history and metadata
5. Document analysis workflow for collaboration
6. Understand reproducibility best practices

**Metadata tracked**:

- **Session info** - Name, description, created/modified timestamps
- **Recordings** - Source files, capture conditions, device info
- **Analysis steps** - Operations performed, hypotheses tested
- **Annotations** - User notes, findings, decisions
- **State** - Current analysis state, progress markers

**Audit trail**:

```
Session: Protocol RE - Widget XYZ
Created: 2026-01-20 09:15:00
Modified: 2026-01-20 14:30:00

Recordings:
- baseline.bin (2026-01-20 09:20:00)
- test_cmd_01.bin (2026-01-20 10:15:00)
- test_cmd_02.bin (2026-01-20 11:30:00)

Analysis:
- 09:25 - Loaded baseline recording
- 10:20 - Differential analysis: bytes 2-3 vary
- 11:35 - Hypothesis: bytes 2-3 = command ID
- 14:30 - Validated hypothesis on 5 new captures
```

**Why this matters**: Persistent session metadata enables reproducibility, collaboration, and audit trails for regulated industries. Essential for documentation and knowledge transfer.

**Note**: This demo focuses on metadata and state management available in current AnalysisSession. Full serialization to `.tks` files with HMAC integrity is a future enhancement.

---

### Demo 05: Interactive Analysis

**File**: `05_interactive_analysis.py`

**What it teaches**:

- Interactive session workflows
- Annotation and note-taking patterns
- History tracking and replay concepts
- Collaborative analysis patterns
- Session comparison and merging concepts
- Hypothesis testing workflows

**What you'll do**:

1. Create interactive session for exploratory analysis
2. Add recordings incrementally as analysis progresses
3. Annotate findings and hypotheses in real-time
4. Track analysis history (what was tried, what worked)
5. Compare different analysis approaches
6. Document conclusions for team collaboration

**Interactive workflow**:

```
1. Start session with initial recording
2. Perform measurements, note observations
3. Add hypothesis as annotation
4. Load additional recording to test hypothesis
5. Compare results, update hypothesis
6. Document findings
7. Repeat until protocol understood
```

**Collaboration patterns**:

- **Shared sessions** - Multiple analysts work on same investigation
- **Annotation exchange** - Share findings and hypotheses
- **History tracking** - See what others have tried
- **Session forking** - Branch analysis for different approaches
- **Merging insights** - Combine findings from parallel investigations

**Why this matters**: Interactive analysis is how real reverse engineering happens - iterative hypothesis testing, incremental understanding, collaborative investigation. Sessions provide structure for this exploratory process.

---

## Running the Demonstrations

See [main demonstrations README](../README.md#running-demonstrations) for all execution options.

**Category-specific tip:** Start with the first demonstration (e.g., `01_analysis_session.py`) before exploring advanced examples.

---

## What You'll Learn

After completing this section, you will understand:

### Core Session Concepts

- **Session lifecycle** - Create, use, close sessions
- **Recording management** - Add, get, list, remove recordings
- **Comparison operations** - Differential analysis between recordings
- **Session types** - Generic, domain-specific, specialized sessions
- **State management** - Track analysis progress

### Multi-Recording Analysis

- **Before/after testing** - Compare signal changes
- **Multi-device analysis** - Compare signals from different sources
- **Time-series tracking** - Analyze signal evolution
- **Regression testing** - Current vs baseline comparison

### Domain-Specific Sessions

- **CANSession pattern** - CAN bus analysis workflows
- **BlackBoxSession** - Protocol reverse engineering
- **Custom sessions** - Build your own domain sessions
- **Integration** - Link to external tools

### Interactive Workflows

- **Hypothesis testing** - Iterative analysis approach
- **Annotation patterns** - Document findings
- **History tracking** - Record analysis steps
- **Collaboration** - Team investigation workflows

### Persistence and Reproducibility

- **Metadata tracking** - Comprehensive session information
- **State persistence** - Save/restore analysis state
- **Audit trails** - Who, what, when documentation
- **Reproducibility** - Recreate analysis workflows

---

## Common Issues and Solutions

### "Cannot add recording to session"

**Solution**: Ensure recording has unique name and valid data:

```python
from oscura.sessions import GenericSession

session = GenericSession(name="my_session")

# Correct: unique name, valid source
session.add_recording("baseline", source=trace_source)

# Error: duplicate name
# session.add_recording("baseline", source=trace_source)  # Raises error

# Correct: different name
session.add_recording("test_01", source=trace_source2)
```

### Session metadata not preserved

**Solution**: Set metadata when creating session and update as needed:

```python
session = GenericSession(
    name="protocol_analysis",
    description="Reverse engineering Widget XYZ protocol"
)

# Update metadata
session.metadata["analyst"] = "Alice"
session.metadata["date"] = "2026-01-23"
session.metadata["hypothesis"] = "Bytes 2-3 are command ID"
```

### Cannot compare recordings

**Solution**: Recordings must have compatible data types:

```python
# Both must be same trace type
recording1 = session.get_recording("baseline")  # WaveformTrace
recording2 = session.get_recording("test")      # WaveformTrace

# Compare (works - same type)
diff = compare_traces(recording1.read(), recording2.read())

# Error if different types
# recording3 has DigitalTrace - can't compare directly to WaveformTrace
```

### Session state not persisting

**Solution**: Current implementation tracks state in memory. For persistence, manually save metadata:

```python
import json
from datetime import datetime

# Save session metadata
metadata = {
    "name": session.name,
    "created_at": datetime.now().isoformat(),
    "recordings": [r.name for r in session.list_recordings()],
    "annotations": session.metadata.get("annotations", [])
}

with open("session_state.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Restore later
with open("session_state.json", "r") as f:
    saved_state = json.load(f)
```

**Note**: Full `.tks` serialization with HMAC is a future enhancement.

---

## Next Steps: Where to Go After Sessions

### If You Want to...

| Goal                           | Next Demo                                        | Path                            |
| ------------------------------ | ------------------------------------------------ | ------------------------------- |
| Process sessions in batch      | `09_batch_processing/01_parallel_batch.py`       | Batch → Apply to sessions       |
| Build domain-specific session  | `08_extensibility/04_plugin_development.py`      | Extensibility → Custom sessions |
| Add quality checks to sessions | `12_quality_tools/02_quality_scoring.py`         | Quality → Session validation    |
| Export session results         | `15_export_visualization/`                       | Export → Session reports        |
| Complete RE workflow           | `16_complete_workflows/01_protocol_discovery.py` | Workflows → Full examples       |

### Recommended Learning Sequence

1. **Complete Sessions** (this section)
   - Master session API
   - Learn multi-recording workflows
   - Understand persistence patterns

2. **Apply to Batch Processing** (09_batch_processing/)
   - Process multiple sessions in parallel
   - Aggregate results across sessions
   - Track batch session progress

3. **Add Quality Tools** (12_quality_tools/)
   - Quality scoring within sessions
   - Automated session validation
   - Quality-based recommendations

4. **Complete Workflows** (16_complete_workflows/)
   - Full reverse engineering workflows
   - Production analysis pipelines
   - Real-world examples

5. **Production Deployment** (11_integration/)
   - CI/CD with sessions
   - Automated regression testing
   - Team collaboration workflows

---

## Session Design Patterns

**Before/After**: Load baseline + modified version, compare, document findings in metadata

**Multi-Device**: Create session, add recordings from each device, analyze and store metrics

**Interactive RE**: Start with baseline, add observations, test hypotheses, document iteratively with timestamps

## Session Best Practices

**Creation DO**: Descriptive names, comprehensive metadata, document purpose, timestamp creation

**Creation DON'T**: Generic names, skip metadata, use for single recordings

**Recording Management DO**: Descriptive names, include capture metadata, keep related recordings, remove unused

**Recording Management DON'T**: Unrelated recordings, numeric-only names, forget to close

**Metadata/Annotations DO**: Timestamp all, document hypotheses, include analyst info, record steps

**Metadata/Annotations DON'T**: Skip documentation, use ambiguous terms, forget timestamps

**Collaboration DO**: Shared format, document assumptions, provide context, version state

**Collaboration DON'T**: Assume prior knowledge, skip explanations, overwrite annotations

---

## Resources

### In This Repository

- **`src/oscura/sessions/`** - Session implementation
- **`src/oscura/automotive/can/session.py`** - CANSession example
- **`tests/unit/test_sessions.py`** - Session test examples

### Session Types

- **GenericSession** - General-purpose analysis
- **BlackBoxSession** - Protocol reverse engineering
- **CANSession** - CAN bus analysis (in `oscura.automotive.can`)

### Related Topics

- **Batch processing** - Process multiple sessions
- **Quality tools** - Validate session results
- **Export** - Generate session reports

---

## Summary

The Analysis Sessions section covers:

| Demo                    | Focus                   | Outcome                   |
| ----------------------- | ----------------------- | ------------------------- |
| 01_analysis_session     | Core session API        | Multi-recording workflows |
| 02_can_session          | Domain-specific pattern | CAN analysis concepts     |
| 03_blackbox_session     | Protocol RE             | Differential analysis     |
| 04_session_persistence  | State tracking          | Metadata and audit trails |
| 05_interactive_analysis | Interactive workflows   | Hypothesis testing        |

After completing these 75-minute demonstrations, you'll be able to:

- Manage multi-recording analysis workflows
- Perform comparative analysis between recordings
- Use domain-specific sessions for specialized analysis
- Apply systematic reverse engineering methods
- Track analysis state and metadata
- Build interactive, collaborative workflows

**Ready to start?** Run this to understand sessions:

```bash
python demonstrations/10_sessions/01_analysis_session.py
```

Happy analyzing!
