# Architecture Overview

Oscura is designed as a modular, extensible framework for hardware reverse engineering with clear separation of concerns.

## Design Principles

1. **Modularity:** Independent components with well-defined interfaces
2. **Extensibility:** Plugin architecture for custom protocols and analyzers
3. **Performance:** NumPy/SciPy for computational efficiency
4. **Usability:** High-level workflows + low-level API access
5. **Standards Compliance:** IEEE-compliant measurements and calculations

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Applications                        │
│   (Scripts, Jupyter Notebooks, CLI, Web Interface)          │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                    High-Level Workflows                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ complete_re  │  │ power_analysis│  │signal_integrity     │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                     Session Management                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ BlackBoxSession  CANSession  │  │ UDSSession   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                    Analysis & Inference                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Differential │  │ CRC Recovery │  │ State Machine│      │
│  │ Analysis     │  │              │  │ Inference    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Signal       │  │ Pattern      │  │ Entropy      │      │
│  │ Classification   Detection    │  │ Analysis     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                   Protocol Decoders                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ UART         │  │ CAN/CAN-FD   │  │ SPI          │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ I2C          │  │ LIN          │  │ FlexRay      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ JTAG         │  │ SWD          │  │ Manchester   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                   Analyzers & Measurements                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Waveform     │  │ Spectral     │  │ Statistics   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Eye Diagram  │  │ Jitter       │  │ Power        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                    File Format Loaders                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ WFM (Tek)    │  │ BLF (Vector) │  │ VCD (Verilog)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ WAV (Audio)  │  │ CSV          │  │ HDF5         │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│              Core Data Structures & Utilities                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Waveform     │  │ Message      │  │ Signal       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. Loaders (`src/oscura/loaders/`)

**Purpose:** Load waveform data from various file formats

**Key Components:**

- `load()` - Auto-detect format and load
- Format-specific loaders: WFM, BLF, VCD, WAV, CSV, HDF5
- Streaming loaders for large files
- Memory-mapped file I/O

**Data Flow:**

```
File → Format Detection → Parser → Waveform Object → Analyzers
```

**Example:**

```python
from oscura.loaders import load_waveform

# Auto-detect and load any supported format
waveform = load("capture.wfm")
# Returns: Waveform object with .data, .sample_rate, .channels
```

---

### 2. Analyzers (`src/oscura/analyzers/`)

**Purpose:** Signal processing and measurement extraction

**Submodules:**

- `waveform/` - Rise/fall time, amplitude, duty cycle
- `spectral/` - FFT, PSD, harmonics
- `statistics/` - Mean, RMS, peak-to-peak
- `eye/` - Eye diagram generation and quality metrics
- `jitter/` - TIE, period jitter, cycle-to-cycle
- `power/` - Power analysis, CPA, DPA
- `protocols/` - Protocol-specific decoders

**Architecture:**

```python
class Analyzer(ABC):
    """Base class for all analyzers."""

    @abstractmethod
    def analyze(self, data: np.ndarray, **params) -> dict[str, Any]:
        """Run analysis and return results."""
        pass

    @abstractmethod
    def get_metadata(self) -> AnalyzerMetadata:
        """Return analyzer metadata."""
        pass
```

**Example:**

```python
from oscura.analyzers.waveform import rise_time

# IEEE 181-2011 compliant measurement
result = rise_time(
    waveform=waveform,
    low_threshold=0.1,   # 10%
    high_threshold=0.9   # 90%
)

print(f"Rise time: {result.value:.2f} ns")
print(f"Uncertainty: ±{result.uncertainty:.2f} ns")
```

---

### 3. Protocol Decoders (`src/oscura/analyzers/protocols/`)

**Purpose:** Decode digital communication protocols

**Base Class:**

```python
class ProtocolDecoder(ABC):
    """Base class for protocol decoders."""

    @abstractmethod
    def decode(self, waveform: Waveform) -> list[Message]:
        """Decode waveform into messages."""
        pass

    @abstractmethod
    def encode(self, messages: list[Message]) -> Waveform:
        """Encode messages into waveform."""
        pass

    @classmethod
    @abstractmethod
    def auto_detect(cls, waveform: Waveform) -> ProtocolParameters | None:
        """Auto-detect protocol parameters."""
        pass
```

**Implemented Protocols:**

- Serial: UART, SPI, I2C, 1-Wire
- Automotive: CAN, CAN-FD, LIN, FlexRay
- Debug: JTAG, SWD
- Industrial: Modbus RTU, PROFIBUS
- Encoding: Manchester, HDLC
- Others: USB, I2S

**Example:**

```python
from oscura.analyzers.protocols import UARTDecoder

decoder = UARTDecoder(baud_rate=115200)
messages = decoder.decode(waveform)

# Auto-detection
params = UARTDecoder.auto_detect(waveform)
decoder = UARTDecoder(**params)
```

---

### 4. Sessions (`src/oscura/session/`)

**Purpose:** High-level analysis sessions with state management

**Session Types:**

**BlackBoxSession:**

```python
# Unknown protocol reverse engineering
session = BlackBoxSession()
session.add_recording("idle", "idle.bin")
session.add_recording("active", "active.bin")

diff = session.compare("idle", "active")
spec = session.generate_protocol_spec()
session.export_results("dissector", "proto.lua")
```

**CANSession:**

```python
# CAN bus analysis
session = CANSession(bitrate=500000)
session.load("vehicle.blf")

signals = session.extract_signals()
session.export_dbc("vehicle.dbc")
```

**Architecture:**

```python
class Session(ABC):
    """Base session class."""

    def __init__(self, name: str):
        self.name = name
        self.recordings: dict[str, Recording] = {}
        self.hypotheses: list[Hypothesis] = []
        self.results: dict[str, Any] = {}

    @abstractmethod
    def analyze(self) -> AnalysisResults:
        """Run session-specific analysis."""
        pass

    def export_results(self, format: str, output: str):
        """Export results in specified format."""
        pass
```

---

### 5. Inference (`src/oscura/inference/`)

**Purpose:** Automated protocol structure inference

**Components:**

**CRC Recovery:**

```python
from oscura.inference.crc_reverse import CRCReverser

reverser = CRCReverser()
crc_params = reverser.find_crc(message_checksum_pairs)

# Returns: polynomial, init, xor_out, reflected
```

**Differential Analysis:**

```python
from oscura.utils.comparison import compare_traces

diff = compare_traces(
    baseline=idle_messages,
    compare=active_messages
)

# Identifies: changed fields, static headers, patterns
```

**State Machine Inference:**

```python
from oscura.inference.state_machine import infer_state_machine

fsm = infer_state_machine(
    message_sequences=sequences,
    algorithm="rpni"  # or "edsm", "bluesfringe"
)

# Returns: states, transitions, accepting states
```

---

### 6. Export (`src/oscura/export/`)

**Purpose:** Generate protocol artifacts

**Exporters:**

- **Wireshark:** Lua dissectors
- **Scapy:** Python packet definitions
- **Kaitai Struct:** Multi-language parsers
- **DBC/LDF/FIBEX:** Automotive formats
- **C/C++:** Header files
- **Markdown/HTML:** Documentation

**Example:**

```python
from oscura.export import WiresharkExporter

exporter = WiresharkExporter(protocol_spec)
exporter.generate(
    output="protocol.lua",
    validate=True  # Check Lua syntax
)
```

---

### 7. Workflows (`src/oscura/workflows/`)

**Purpose:** Complete end-to-end workflows

**Available Workflows:**

```python
# Complete RE workflow
from oscura.workflows import complete_re

result = complete_re(
    captures={"idle": "idle.bin", "active": "active.bin"},
    auto_crc=True,
    detect_crypto=True,
    generate_dissector=True,
    export_dir="output/"
)

# Power analysis workflow
from oscura.workflows import power_analysis

result = power_analysis(
    traces="traces.npy",
    plaintexts="plaintexts.npy",
    algorithm="AES-128"
)

# Signal integrity workflow
from oscura.workflows import signal_integrity_analysis

result = signal_integrity_analysis(
    waveform=waveform,
    specification="USB 2.0"
)
```

---

## Data Structures

### Waveform

```python
@dataclass
class Waveform:
    """Core waveform data structure."""

    data: np.ndarray              # Shape: (channels, samples)
    sample_rate: float            # Hz
    start_time: float             # Seconds
    channel_names: list[str]
    metadata: dict[str, Any]

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data[0]) / self.sample_rate

    @property
    def channels(self) -> int:
        """Number of channels."""
        return len(self.data)
```

### Message

```python
@dataclass
class Message:
    """Decoded protocol message."""

    timestamp: float              # Seconds
    data: bytes                   # Payload
    metadata: dict[str, Any]      # Protocol-specific

    # Optional fields
    checksum: int | None = None
    checksum_valid: bool | None = None
    sequence: int | None = None
    has_error: bool = False
```

### ProtocolSpec

```python
@dataclass
class ProtocolSpec:
    """Inferred protocol specification."""

    name: str
    message_length: int
    fields: list[FieldSpec]
    header_pattern: bytes | None
    checksum_algorithm: str | None

@dataclass
class FieldSpec:
    """Individual field specification."""

    name: str
    offset: int
    size: int
    inferred_type: str  # "data", "counter", "checksum", "enum"
    possible_values: set[int] | None
    description: str | None
```

---

## Extension Points

### Custom Protocol Decoder

```python
from oscura.analyzers.protocols import ProtocolDecoder

class MyProtocol(ProtocolDecoder):
    """Custom protocol decoder."""

    def __init__(self, **params):
        super().__init__(name="my_proto")
        self.params = params

    def decode(self, waveform: Waveform) -> list[Message]:
        # Implement decoding logic
        messages = []
        # ... extract messages from waveform
        return messages

    def encode(self, messages: list[Message]) -> Waveform:
        # Implement encoding logic
        # ... convert messages to waveform
        return waveform

    @classmethod
    def auto_detect(cls, waveform: Waveform):
        # Implement parameter detection
        # ... analyze waveform to find protocol params
        return params if detected else None

# Register decoder
from oscura.plugins import register_decoder
register_decoder("my_proto", MyProtocol)
```

### Custom Analyzer

```python
from oscura.analyzers import Analyzer

class CustomAnalyzer(Analyzer):
    """Custom signal analyzer."""

    def analyze(self, data: np.ndarray, **params) -> dict[str, Any]:
        # Implement analysis algorithm
        result = {
            "metric1": calculate_metric1(data),
            "metric2": calculate_metric2(data),
        }
        return result

    def get_metadata(self) -> AnalyzerMetadata:
        return AnalyzerMetadata(
            name="custom_analyzer",
            version="1.0.0",
            description="Custom analysis algorithm"
        )

# Register analyzer
from oscura.plugins import register_analyzer
register_analyzer("custom", CustomAnalyzer)
```

---

## Performance Considerations

### Memory Management

**Large Files:**

```python
# Use streaming for files >1GB
from oscura.loaders import load_waveform_streaming

stream = load_waveform_streaming("large.wfm", chunk_size=1_000_000)
for chunk in stream:
    analyze(chunk)
```

**Memory-Mapped I/O:**

```python
# Efficient access to large files
from oscura.io import MemoryMappedWaveform

waveform = MemoryMappedWaveform("huge_file.bin")
# Data accessed on-demand, not loaded into RAM
```

### Parallel Processing

**Batch Analysis:**

```python
# NOTE: Use workflows or manual iteration in v0.6
# from oscura.workflows import batch_analyze

results = batch_analyze(
    files=file_list,
    analysis_func=my_analysis,
    parallel=True,
    num_workers=8  # Use 8 CPU cores
)
```

**GPU Acceleration (Experimental):**

```python
from oscura.performance import enable_gpu

enable_gpu()  # Use CUDA for FFT, correlation, CPA
```

---

## Testing Strategy

### Unit Tests

- Located in `tests/unit/`
- Test individual functions and classes
- Use pytest fixtures from `tests/conftest.py`
- Synthetic test data only

### Integration Tests

- Located in `tests/integration/`
- Test complete workflows
- Use realistic (but synthetic) captures
- Test error handling and edge cases

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(0, 255), min_size=8, max_size=8))
def test_crc_reversal_finds_valid_crc(message: list[int]):
    """Property: CRC reverser should find valid CRC for any message."""
    # ... test implementation
```

---

## See Also

- [Contributing Guide](../contributing.md)
- [API Reference](../api/)
- [Design Principles](design-principles.md)
- [Plugin Development](custom-protocols.md)
