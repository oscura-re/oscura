# Oscura Architecture

## Overview

Oscura follows a modular architecture with clear separation of concerns:

### Core Modules

- **`core/`** - Signal data structures and fundamental operations
- **`loaders/`** - File format parsers (WFM, VCD, PCAP, etc.)
- **`analyzers/`** - Signal analysis algorithms
  - Digital signal analysis
  - Spectral analysis
  - Protocol decoders
  - Power analysis
  - Jitter analysis
- **`inference/`** - Protocol reverse engineering
- **`reporting/`** - Export and report generation
- **`workflows/`** - High-level analysis pipelines

### Design Principles

1. **Composability** - Small, focused modules that work together
2. **Standards Compliance** - IEEE 181, 1241, 1459, 2414
3. **Extensibility** - Plugin system for custom analyzers
4. **Performance** - Optimized for large signals with streaming support

### Data Flow

```
Load Signal → Analyze → Process → Export
     ↓          ↓         ↓         ↓
  Loaders → Analyzers → Filters → Reporters
```

## Module Dependencies

_(Detailed dependency graphs will be added here)_

## Package Structure

See the [API Reference](../../api/index.md) for detailed documentation of each module.
