# Advanced API: Expert-Level Patterns

Expert-level API patterns for sophisticated signal processing workflows. These demonstrations show advanced composition, optimization, and parallel processing techniques.

## Demonstrations

### 01. Pipeline API (`01_pipeline_api.py`)

**Multi-stage processing with pipe() and compose()**

Demonstrates functional composition for elegant multi-stage processing:

- `pipe()` - Left-to-right function composition
- `compose()` - Right-to-left function composition
- Reusable processing chains
- Performance comparison

**Use when**: Building complex analysis workflows, creating reusable processing functions

### 02. DSL Syntax (`02_dsl_syntax.py`)

**Domain-Specific Language for readable signal processing**

Create expressive, readable code with fluent APIs:

- Method chaining patterns
- Query-like syntax for signal operations
- SignalBuilder for declarative signal construction
- Self-documenting code patterns

**Use when**: Building user-facing APIs, improving code readability

### 03. Operators (`03_operators.py`)

**Natural mathematical expressions with operator overloading**

Enable intuitive signal mathematics:

- Arithmetic operators (+, -, \*, /)
- Comparison operators for thresholding
- Unary operators (negation, absolute value)
- Practical application: differential signaling

**Use when**: Complex signal math, differential signaling, natural expressions

### 04. Composition (`04_composition.py`)

**Higher-order functions and decorator patterns**

Advanced functional programming techniques:

- Function composition and currying
- Higher-order functions for flexibility
- Decorator patterns (timing, caching, validation)
- Measurement suites with decorators

**Use when**: Building extensible systems, adding cross-cutting concerns

### 05. Optimization (`05_optimization.py`)

**Performance optimization techniques and caching**

Maximize processing performance:

- FFT result caching (2-10x speedup)
- Lazy evaluation patterns
- Memory-efficient processing
- Batch processing optimization

**Use when**: Processing large signals, real-time requirements, memory constraints

### 06. Streaming API (`06_streaming_api.py`)

**Real-time and large-file processing**

Process unlimited data with streaming:

- Generator-based chunk processing
- Online algorithms (Welford's for statistics)
- StreamingAnalyzer for complex analysis
- Backpressure and flow control
- Low-latency real-time processing

**Use when**: Large files don't fit in memory, real-time processing, continuous data

### 07. Parallel Processing (`07_parallel_processing.py`)

**Multi-core signal analysis**

Utilize multiple CPU cores for performance:

- Process pool vs thread pool patterns
- Scaling analysis and efficiency
- Progress tracking for batch operations
- Performance comparison and best practices

**Use when**: Batch processing many signals, CPU-bound workloads, multi-core systems

## Quick Start

Run any demonstration:

```bash
uv run python demonstrations/07_advanced_api/01_pipeline_api.py
uv run python demonstrations/07_advanced_api/05_optimization.py
```

## Key Concepts

### Functional Composition

```python
# pipe(): left-to-right (intuitive)
result = pipe(signal, high_pass_filter, low_pass_filter, rms)

# compose(): right-to-left (mathematical)
processor = compose(rms, low_pass_filter, high_pass_filter)
result = processor(signal)
```

### Performance Optimization

```python
# Enable FFT caching
configure_fft_cache(max_size=10, enabled=True)

# Cache provides 2-10x speedup for repeated spectral analysis
thd1 = thd(signal, fundamental=1000)  # Computes FFT
thd2 = thd(signal, fundamental=1000)  # Reuses cached FFT
```

### Streaming Processing

```python
# Process large files in chunks
analyzer = StreamingAnalyzer(chunk_size=1000)

for chunk in load_trace_chunks(filename, chunk_size=1000):
    analyzer.process_chunk(chunk)

results = analyzer.get_results()
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

# Process batch in parallel
with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    results = executor.map(process_signal, signal_batch)
```

## Performance Guide

| Technique           | Speedup  | Use Case                             |
| ------------------- | -------- | ------------------------------------ |
| FFT Caching         | 2-10x    | Repeated spectral analysis           |
| Parallel Processing | 2-8x     | Batch processing (scales with cores) |
| Streaming           | Memory   | Large files, real-time               |
| Lazy Evaluation     | Variable | Conditional processing               |

## Best Practices

1. **Use `pipe()` for readability** - Left-to-right is more intuitive
2. **Enable FFT caching** - Significant speedup for spectral analysis
3. **Stream large files** - Don't load everything into memory
4. **Parallelize batch operations** - Utilize multiple cores for CPU-bound tasks
5. **Profile before optimizing** - Measure to find real bottlenecks

## Learning Path

1. Start with **Pipeline API** to learn functional composition
2. Try **DSL Syntax** to see fluent API patterns
3. Explore **Optimization** for performance techniques
4. Learn **Streaming API** for large-file processing
5. Use **Parallel Processing** for batch workflows
6. Apply **Composition** patterns for extensibility

## Integration with Other Demos

- **Basic Analysis (02\_\*)** - Shows what to analyze; Advanced API shows how to organize it
- **Protocol Decoding (03\_\*)** - Can be streamlined with pipelines
- **Domain Specific (05\_\*)** - Advanced patterns improve domain-specific workflows
- **Extensibility (08\_\*)** - Composition enables plugin architectures

## Performance Considerations

### Memory Usage

- **Streaming**: Constant memory regardless of file size
- **Batch**: O(n) memory for n signals
- **Parallel**: O(n × workers) memory

### CPU Utilization

- **Sequential**: Single core
- **Thread pool**: Limited by GIL for CPU-bound tasks
- **Process pool**: Full multi-core utilization

### Latency

- **Streaming**: ~0.1-1ms per chunk (real-time capable)
- **Batch**: Variable depending on signal size
- **Parallel**: Overhead ~10-50ms for process pool startup

## Troubleshooting

**Q: Pipeline seems slower than sequential code?**
A: Pipelines have minimal overhead. Check if operations are too simple (<1ms each). Pipeline benefits are code organization, not performance.

**Q: Parallel processing not faster?**
A: Ensure tasks are CPU-bound and substantial (>10ms each). Process pool has startup overhead. Use thread pool for I/O-bound tasks only.

**Q: FFT cache not helping?**
A: Verify cache is enabled and signals are identical. Check cache stats with `get_fft_cache_stats()`.

**Q: Streaming slower than batch?**
A: Chunk size may be too small. Try larger chunks (1000-10000 samples). Streaming trades speed for memory efficiency.

## See Also

- [Function Composition (Oscura docs)](../../../docs/api/composition.md)
- [Performance Optimization Guide](../../../docs/guides/performance.md)
- [Streaming API Reference](../../../docs/api/streaming.md)
- [Parallel Processing Examples](../../../examples/parallel/)
