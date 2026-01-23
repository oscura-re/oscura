# Batch Processing and Parallel Execution

**Process multiple signal files efficiently with parallel execution and result aggregation.**

This section contains 4 demonstrations designed to teach you how to process large datasets efficiently using parallel execution, track progress across batch operations, aggregate results for statistical analysis, and optimize performance with the right parallelization strategy. Perfect for production workflows, regression testing, and large-scale signal analysis.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Python 3.12+** - Required for Oscura
- **Understanding of basic measurements** - Complete `02_basic_analysis/01_waveform_measurements.py`
- **Multiple CPU cores** - For observing parallel speedup (optional)
- **Familiarity with Python concurrency** - Understanding of threads vs processes helps

Check your environment:

```bash
# Check CPU cores (for parallelism)
python -c "import os; print(f'CPU cores: {os.cpu_count()}')"

# Verify Oscura works
python demonstrations/00_getting_started/00_hello_world.py
```

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_parallel_batch.py → 02_result_aggregation.py → 03_progress_tracking.py → 04_optimization.py
        ↓                       ↓                           ↓                        ↓
  Thread/Process pools    Statistical analysis         Real-time monitoring    Performance optimization
  Multi-file processing   Outlier detection            ETA calculation         Serial vs parallel vs GPU
  Error handling          Comparative analysis         Resume on failure       Best practice guidelines
```

### Estimated Time

| Demo                  | Time       | Difficulty                | Topics                                     |
| --------------------- | ---------- | ------------------------- | ------------------------------------------ |
| 01_parallel_batch     | 15 min     | Intermediate              | Thread/process pools, parallel execution   |
| 02_result_aggregation | 10 min     | Intermediate              | Statistics, outliers, comparison           |
| 03_progress_tracking  | 10 min     | Intermediate              | Progress monitoring, ETA, resumption       |
| 04_optimization       | 15 min     | Advanced                  | Performance benchmarking, GPU acceleration |
| **Total**             | **50 min** | **Intermediate-Advanced** | **Production batch workflows**             |

---

## Demonstrations

### Demo 01: Parallel Batch Processing

**File**: `01_parallel_batch.py`

**What it teaches**:

- Multi-file parallel processing with ThreadPoolExecutor
- Process pool vs thread pool comparison
- Progress tracking across parallel workers
- Result aggregation from concurrent tasks
- Error handling in parallel contexts
- Performance optimization strategies

**What you'll do**:

1. Generate multiple synthetic signal files
2. Process files in parallel using thread pool
3. Process files in parallel using process pool
4. Compare thread pool vs process pool performance
5. Handle errors gracefully in parallel workers
6. Aggregate results from all workers

**Key concepts**:

- `ThreadPoolExecutor` - Thread-based parallelism (I/O bound)
- `ProcessPoolExecutor` - Process-based parallelism (CPU bound)
- `as_completed()` - Process results as they finish
- Worker coordination and result collection
- Error propagation and handling

**Performance comparison**:

| Workload Type   | Best Choice         | Reason                       |
| --------------- | ------------------- | ---------------------------- |
| File I/O heavy  | ThreadPoolExecutor  | GIL not held during I/O      |
| CPU computation | ProcessPoolExecutor | True parallelism, bypass GIL |
| Mixed workload  | ProcessPoolExecutor | Better overall performance   |

**Why this matters**: Production workflows often involve processing hundreds or thousands of signal captures. Parallel execution can reduce processing time from hours to minutes.

---

### Demo 02: Result Aggregation

**File**: `02_result_aggregation.py`

**What it teaches**:

- Statistical aggregation across batch results
- Summary report generation
- Outlier detection in measurement data
- Comparative analysis between batches
- Data quality metrics calculation
- Statistical validation

**What you'll do**:

1. Generate batch processing results (multiple measurements)
2. Calculate statistical summaries (mean, std dev, min, max)
3. Detect outliers using statistical methods (IQR, Z-score)
4. Compare results across different batches
5. Generate comprehensive summary reports
6. Identify data quality issues automatically

**Statistical methods**:

- **Descriptive statistics** - Mean, median, std dev, quartiles
- **Outlier detection** - IQR method, Z-score method
- **Comparative analysis** - Batch-to-batch comparison
- **Quality metrics** - Success rate, error rate, consistency

**Output format**:

```
Batch Summary Report
====================
Total measurements: 1000
Mean amplitude: 2.0015 V ± 0.0234 V
Outliers detected: 12 (1.2%)
Quality score: 94.3%
```

**Why this matters**: Aggregated results reveal trends, anomalies, and quality issues that individual measurements miss. Essential for regression testing, production validation, and quality assurance.

---

### Demo 03: Progress Tracking

**File**: `03_progress_tracking.py`

**What it teaches**:

- Real-time progress monitoring for batch operations
- ETA (Estimated Time to Arrival) calculation
- Throughput tracking (files/second)
- Error recovery and resumption from failures
- Progress persistence across sessions
- Visual progress indicators

**What you'll do**:

1. Implement a progress tracker for batch operations
2. Calculate and display ETA dynamically
3. Track throughput (items processed per second)
4. Handle failures with graceful recovery
5. Persist progress state for resumption
6. Display progress with percentage and time remaining

**Progress metrics**:

- **Percentage complete** - Items processed / total items
- **ETA** - Estimated time remaining based on current throughput
- **Throughput** - Items per second processing rate
- **Success rate** - Percentage of successful operations
- **Failure tracking** - Failed items for retry

**Example output**:

```
Processing: [████████████░░░░░░░░] 62% complete
ETA: 3m 24s | Throughput: 8.4 files/sec | Success: 618/620
```

**Why this matters**: Long-running batch operations need monitoring to estimate completion time, detect stalls, and enable recovery from failures. Critical for production environments.

---

### Demo 04: Batch Processing Optimization

**File**: `04_optimization.py`

**What it teaches**:

- Performance comparison: serial vs parallel vs GPU
- Choosing optimal parallelization strategy
- GPU acceleration with automatic CPU fallback
- Performance benchmarking methodology
- Worker count optimization
- Optimization decision guidelines

**What you'll do**:

1. Generate test signal dataset (50 files)
2. Benchmark serial processing (baseline)
3. Benchmark parallel thread-based processing
4. Benchmark parallel process-based processing
5. Benchmark GPU-accelerated processing (if available)
6. Compare performance across all methods
7. Learn when to use each strategy

**Performance strategies**:

| Method         | Best For                | Speedup | Memory   | Overhead      |
| -------------- | ----------------------- | ------- | -------- | ------------- |
| Serial         | <10 files, simple ops   | 1.0x    | Low      | None          |
| Threads        | I/O-bound, file loading | 2-4x    | Shared   | Low           |
| Processes      | CPU-bound, FFT-heavy    | 1.5-3x  | Isolated | Medium        |
| GPU            | Large FFT, many signals | 5-50x   | GPU VRAM | High transfer |
| Advanced Batch | Production workflows    | 2-4x    | Moderate | Low           |

**Optimization guidelines**:

- **Serial**: Small datasets (<10 files), simple operations
- **Thread pool**: I/O-bound operations (file loading, network), moderate CPU work
- **Process pool**: CPU-bound operations (FFT, filtering, complex analysis), bypasses Python GIL
- **GPU**: FFT-heavy workloads, large arrays, convolution (requires CuPy + NVIDIA GPU)
- **AdvancedBatchProcessor**: Production use with checkpointing, timeout, error isolation

**Example output**:

```
Method              Time (s)  Speedup  Throughput (files/s)
-----------------------------------------------------------------
Serial                0.58     1.00x      86.3
Threads               0.17     3.33x     287.5
Processes             0.29     1.98x     170.6
GPU                   0.62     0.93x      80.6
Advanced Batch        0.23     2.55x     219.6
```

**Key considerations**:

- **Threads**: Low overhead, shares memory, limited by GIL for CPU work
- **Processes**: Higher overhead, isolated memory, true parallelism
- **GPU**: Best for large-scale FFT, requires data transfer overhead consideration
- **Batch**: Adds checkpointing, resume capability, timeout enforcement, error isolation

**Why this matters**: Choosing the wrong parallelization strategy can make performance worse instead of better. This demo teaches you how to benchmark, compare, and select the optimal strategy for your workload characteristics and available hardware.

---

## How to Run the Demos

### Option 1: Run Individual Demo

Run a single demo to learn a specific concept:

```bash
# From the project root
python demonstrations/09_batch_processing/01_parallel_batch.py

# Or from the demo directory
cd demonstrations/09_batch_processing
python 01_parallel_batch.py
```

Expected output: Timing comparisons, progress indicators, aggregated results.

### Option 2: Run All Batch Processing Demos

Run all three demos in sequence:

```bash
# From the project root
python demonstrations/09_batch_processing/01_parallel_batch.py && \
python demonstrations/09_batch_processing/02_result_aggregation.py && \
python demonstrations/09_batch_processing/03_progress_tracking.py
```

### Option 3: Validate All Demonstrations

Validate all demonstrations in the project:

```bash
# From the project root
python demonstrations/validate_all.py
```

This runs all demonstrations including batch processing and reports coverage.

---

## What You'll Learn

After completing this section, you will understand:

### Parallel Processing

- **Thread pools** - When and how to use ThreadPoolExecutor
- **Process pools** - When and how to use ProcessPoolExecutor
- **Worker coordination** - Managing concurrent tasks
- **Error handling** - Graceful failure in parallel contexts
- **Performance optimization** - Choosing thread vs process parallelism

### Result Aggregation

- **Statistical summaries** - Mean, median, std dev, quartiles
- **Outlier detection** - IQR and Z-score methods
- **Quality metrics** - Success rate, consistency, reliability
- **Comparative analysis** - Batch-to-batch comparison
- **Report generation** - Comprehensive summary reports

### Progress Monitoring

- **Real-time tracking** - Display progress during execution
- **ETA calculation** - Estimate time remaining
- **Throughput metrics** - Items per second processing rate
- **Error recovery** - Resume from failures
- **State persistence** - Save progress across sessions

### Production Workflows

- **Scalability patterns** - Handle hundreds of files efficiently
- **Monitoring strategies** - Track long-running operations
- **Quality assurance** - Automated validation and reporting
- **Error handling** - Graceful degradation and recovery
- **Performance tuning** - Optimize throughput and resource usage

---

## Common Issues and Solutions

### "ProcessPoolExecutor hangs on Windows"

**Solution**: Windows requires `if __name__ == "__main__":` protection:

```python
from concurrent.futures import ProcessPoolExecutor

def process_file(filename):
    # Your processing logic
    return result

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, filenames))
```

This prevents infinite process spawning on Windows.

### Performance worse with parallelism

**Solution**: Parallelism overhead can exceed benefits for small workloads. Guidelines:

- **Thread pool**: Use when I/O operations dominate (file reading, network)
- **Process pool**: Use when CPU computation dominates (FFT, filtering)
- **Serial**: Use when processing time < 1 second per file

```python
# Rule of thumb: estimate speedup
num_files = 1000
time_per_file = 0.5  # seconds
total_serial = num_files * time_per_file  # 500 seconds

# Parallel with 4 workers
total_parallel = total_serial / 4 + overhead  # ~125 seconds + overhead

# Worth it if overhead < 125 seconds
```

### Memory issues with large batches

**Solution**: Process in chunks to control memory usage:

```python
def process_in_chunks(files, chunk_size=100):
    results = []
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        with ProcessPoolExecutor(max_workers=4) as executor:
            chunk_results = list(executor.map(process_file, chunk))
        results.extend(chunk_results)
    return results
```

### Progress not updating in real-time

**Solution**: Use `as_completed()` for real-time progress:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_file, f): f for f in files}

    for future in as_completed(futures):
        result = future.result()
        update_progress()  # Update after each completion
```

### ETA calculation unstable

**Solution**: Use moving average for stable ETA:

```python
class ProgressTracker:
    def __init__(self):
        self.recent_times = []  # Last N processing times
        self.window_size = 10

    def update(self, processing_time):
        self.recent_times.append(processing_time)
        if len(self.recent_times) > self.window_size:
            self.recent_times.pop(0)

        # Use average of recent times for ETA
        avg_time = sum(self.recent_times) / len(self.recent_times)
        remaining_items = total - completed
        eta = avg_time * remaining_items
```

---

## Next Steps: Where to Go After Batch Processing

### If You Want to...

| Goal                               | Next Demo                                   | Path                           |
| ---------------------------------- | ------------------------------------------- | ------------------------------ |
| Build interactive batch analysis   | `10_sessions/01_analysis_session.py`        | Sessions → Multi-recording     |
| Apply custom measurements in batch | `08_extensibility/02_custom_measurement.py` | Extensibility → Custom metrics |
| Batch process with quality checks  | `12_quality_tools/02_quality_scoring.py`    | Quality → Automated validation |
| Export batch results               | `15_export_visualization/`                  | Export → Report generation     |
| Production deployment              | `11_integration/`                           | Integration → CI/CD pipelines  |

### Recommended Learning Sequence

1. **Complete Batch Processing** (this section)
   - Master parallel execution
   - Learn result aggregation
   - Implement progress tracking

2. **Integrate with Sessions** (10_sessions/)
   - Process multiple recordings in sessions
   - Compare batch results interactively
   - Persistent batch analysis state

3. **Add Quality Tools** (12_quality_tools/)
   - Automated quality scoring in batch
   - Quality-based filtering
   - Anomaly detection at scale

4. **Export and Visualize** (15_export_visualization/)
   - Generate batch reports
   - Visualize aggregated results
   - Create dashboards

5. **Production Deployment** (11_integration/)
   - CI/CD integration for batch processing
   - Automated regression testing
   - Production monitoring

---

## Performance Guidelines

### Choosing Thread vs Process Pool

**Use ThreadPoolExecutor when**:

- File I/O is the bottleneck (loading files)
- Network operations dominate
- Lightweight computations
- Shared memory needed

**Use ProcessPoolExecutor when**:

- Heavy CPU computation (FFT, filtering)
- NumPy operations dominate
- Need true parallelism
- Memory isolation required

### Worker Count Optimization

```python
import os

# Rule of thumb for worker count
cpu_bound_workers = os.cpu_count()  # One per core
io_bound_workers = os.cpu_count() * 2  # More workers OK

# Example
if computation_heavy:
    max_workers = os.cpu_count()
else:
    max_workers = os.cpu_count() * 2
```

### Batch Size Considerations

| Batch Size        | Pros                                  | Cons                       |
| ----------------- | ------------------------------------- | -------------------------- |
| Small (10-100)    | Low memory, frequent progress updates | Higher overhead            |
| Medium (100-1000) | Balanced performance                  | Moderate memory            |
| Large (1000+)     | Lower overhead                        | High memory, slow progress |

---

## Real-World Use Cases

### Regression Testing

Process nightly signal captures to detect changes:

```python
# Process today's captures
results_today = process_batch(todays_files)

# Compare to baseline
results_baseline = load_baseline()

# Detect regressions
regressions = compare_batches(results_today, results_baseline)
if regressions:
    alert_team(regressions)
```

### Production Validation

Validate manufactured units at scale:

```python
# Process 1000 unit test captures
results = process_batch_parallel(unit_test_files, max_workers=8)

# Aggregate quality metrics
summary = aggregate_results(results)

# Pass/fail decisions
passed = [r for r in results if r.quality_score > 95]
failed = [r for r in results if r.quality_score <= 95]

# Generate report
generate_production_report(summary, passed, failed)
```

### Dataset Analysis

Analyze large signal datasets:

```python
# Process 10,000 signal captures
with progress_tracker(total=10000) as tracker:
    results = []
    for chunk in chunked(files, chunk_size=100):
        chunk_results = process_batch(chunk, max_workers=4)
        results.extend(chunk_results)
        tracker.update(len(chunk))

# Statistical analysis
stats = calculate_statistics(results)
outliers = detect_outliers(results)
```

---

## Best Practices

### Error Handling

**DO**:

- Catch exceptions in worker functions
- Log errors with context (filename, timestamp)
- Continue processing after errors
- Report failed items at end

**DON'T**:

- Let one error stop entire batch
- Silently ignore failures
- Skip error logging

### Progress Tracking

**DO**:

- Update progress after each item
- Calculate ETA from recent samples
- Display both percentage and ETA
- Show throughput metrics

**DON'T**:

- Update too frequently (performance impact)
- Calculate ETA from first sample only
- Block on progress updates

### Result Aggregation

**DO**:

- Store raw results for later analysis
- Calculate multiple statistical measures
- Include metadata (timestamps, versions)
- Generate human-readable reports

**DON'T**:

- Aggregate prematurely (lose details)
- Skip outlier detection
- Ignore data quality metrics

---

## Resources

### In This Repository

- **`examples/batch_processing/`** - Complete batch processing examples
- **`scripts/batch_analysis.py`** - Production batch scripts
- **`tests/integration/test_batch.py`** - Batch processing tests

### Python Concurrency

- **[concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)** - Thread/process pools
- **[multiprocessing](https://docs.python.org/3/library/multiprocessing.html)** - Process-based parallelism
- **[threading](https://docs.python.org/3/library/threading.html)** - Thread-based concurrency

### Statistical Analysis

- **[NumPy statistics](https://numpy.org/doc/stable/reference/routines.statistics.html)** - Statistical functions
- **[SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html)** - Advanced statistics

---

## Summary

The Batch Processing section covers:

| Demo                  | Focus                | Outcome                       |
| --------------------- | -------------------- | ----------------------------- |
| 01_parallel_batch     | Parallel execution   | Process files efficiently     |
| 02_result_aggregation | Statistical analysis | Aggregate and compare results |
| 03_progress_tracking  | Monitoring           | Track progress with ETA       |

After completing these 35-minute demonstrations, you'll be able to:

- Process multiple files in parallel efficiently
- Choose appropriate parallelism strategy (thread vs process)
- Aggregate results with statistical analysis
- Detect outliers and quality issues automatically
- Monitor long-running operations with progress tracking
- Resume batch operations after failures

**Ready to start?** Run this to understand parallel processing:

```bash
python demonstrations/09_batch_processing/01_parallel_batch.py
```

Happy batch processing!
