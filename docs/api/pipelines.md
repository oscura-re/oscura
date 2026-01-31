# Pipeline API Reference

> **Version**: 0.8.0
> **Last Updated**: 2026-01-25

## Overview

Oscura's pipeline architecture provides a powerful framework for building composable, reusable analysis workflows. Inspired by scikit-learn's Pipeline design, it enables declarative signal processing through:

- **Pipeline** - Chain multiple operations into sequential workflows
- **TraceTransformer** - Base class for creating custom transformations
- **Functional composition** - compose(), pipe(), and curry() for functional programming
- **Composable mixin** - Enable >> operator for fluent pipelines

## Quick Start

```python
import oscura as osc

# Load trace
trace = osc.load("signal.wfm")

# Create a simple pipeline
pipeline = osc.Pipeline([
    ('filter', osc.LowPassFilter(cutoff=1e6)),
    ('normalize', osc.Normalize(method='peak')),
    ('resample', osc.Resample(rate=1e9))
])

# Transform trace through pipeline
result = pipeline.transform(trace)

# Or use functional composition
from functools import partial
result = osc.pipe(
    trace,
    partial(osc.low_pass, cutoff=1e6),
    partial(osc.normalize, method='peak'),
    partial(osc.resample, rate=1e9)
)
```

## Pipeline Class

### `Pipeline`

Chain multiple trace transformers into a sequential processing pipeline.

```python
class Pipeline(TraceTransformer):
    def __init__(self, steps: Sequence[tuple[str, TraceTransformer]])
    def fit(self, trace: WaveformTrace) -> Pipeline
    def transform(self, trace: WaveformTrace) -> WaveformTrace
    def fit_transform(self, trace: WaveformTrace) -> WaveformTrace
    def get_intermediate(self, step_name: str, key: str | None = None) -> Any
```

**Parameters:**

| Parameter | Type                                     | Description                                        |
| --------- | ---------------------------------------- | -------------------------------------------------- |
| `steps`   | `Sequence[tuple[str, TraceTransformer]]` | List of (name, transformer) tuples defining stages |

**Attributes:**

| Attribute     | Type                                 | Description                              |
| ------------- | ------------------------------------ | ---------------------------------------- |
| `steps`       | `list[tuple[str, TraceTransformer]]` | Pipeline stages as list of tuples        |
| `named_steps` | `dict[str, TraceTransformer]`        | Dictionary mapping names to transformers |

**Returns:** Transformed WaveformTrace after passing through all stages.

**Example:**

```python
import oscura as osc

# Create analysis pipeline
pipeline = osc.Pipeline([
    ('filter', osc.LowPassFilter(cutoff=1e6)),
    ('normalize', osc.Normalize()),
    ('fft', osc.FFT(nfft=8192))
])

# Transform trace
result = pipeline.transform(trace)

# Access steps
filter_step = pipeline['filter']
filter_step = pipeline[0]  # By index
print(f"Pipeline has {len(pipeline)} steps")
```

### Creating Custom Transformers

Build custom transformers by inheriting from `TraceTransformer`:

```python
import oscura as osc
import numpy as np

class AmplitudeScaler(osc.TraceTransformer):
    """Scale waveform amplitude by a fixed factor."""

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def transform(self, trace):
        scaled_data = trace.data * self.scale_factor
        return osc.WaveformTrace(
            data=scaled_data,
            metadata=trace.metadata
        )

# Use in pipeline
pipeline = osc.Pipeline([
    ('scale', AmplitudeScaler(scale_factor=2.0)),
    ('normalize', osc.Normalize())
])

result = pipeline.transform(trace)
```

### Stateful Transformers

Create transformers that learn parameters using the fit/transform pattern:

```python
import oscura as osc
import numpy as np

class AdaptiveNormalizer(osc.TraceTransformer):
    """Normalize using statistics learned from reference trace."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, trace):
        """Learn normalization parameters from reference."""
        self.mean_ = trace.data.mean()
        self.std_ = trace.data.std()
        return self

    def transform(self, trace):
        """Apply learned normalization."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Must call fit() first")

        normalized = (trace.data - self.mean_) / self.std_
        return osc.WaveformTrace(
            data=normalized,
            metadata=trace.metadata
        )

# Fit on reference, transform test traces
normalizer = AdaptiveNormalizer()
normalizer.fit(reference_trace)

results = [normalizer.transform(t) for t in test_traces]
```

### Pipeline Methods

#### `fit()`

Fit all transformers in the pipeline sequentially.

```python
def fit(self, trace: WaveformTrace) -> Pipeline
```

**Parameters:**

| Parameter | Type            | Description               |
| --------- | --------------- | ------------------------- |
| `trace`   | `WaveformTrace` | Reference trace to fit to |

**Returns:** Self for method chaining.

**Example:**

```python
# Create pipeline with stateful transformers
pipeline = osc.Pipeline([
    ('adaptive_filter', AdaptiveFilter()),
    ('normalizer', AdaptiveNormalizer())
])

# Fit on reference trace
pipeline.fit(reference_trace)

# Transform multiple traces with fitted parameters
results = [pipeline.transform(t) for t in test_traces]
```

#### `transform()`

Transform a trace through all pipeline stages.

```python
def transform(self, trace: WaveformTrace) -> WaveformTrace
```

**Parameters:**

| Parameter | Type            | Description |
| --------- | --------------- | ----------- |
| `trace`   | `WaveformTrace` | Input trace |

**Returns:** Transformed WaveformTrace.

#### `fit_transform()`

Fit pipeline to trace and immediately transform it.

```python
def fit_transform(self, trace: WaveformTrace) -> WaveformTrace
```

Convenience method equivalent to `pipeline.fit(trace).transform(trace)`.

#### `get_intermediate()`

Retrieve cached intermediate results from pipeline stages.

```python
def get_intermediate(
    self,
    step_name: str,
    key: str | None = None
) -> Any
```

**Parameters:**

| Parameter   | Type         | Description                                   |
| ----------- | ------------ | --------------------------------------------- |
| `step_name` | `str`        | Name of the pipeline step                     |
| `key`       | `str \|None` | Optional key for transformer-internal results |

**Returns:** WaveformTrace output from that stage (if key=None), or specific intermediate result.

**Raises:** `KeyError` if step not found or transform() not yet called.

**Example:**

```python
# Create pipeline with multiple stages
pipeline = osc.Pipeline([
    ('filter', osc.LowPassFilter(cutoff=1e6)),
    ('fft', osc.FFT(nfft=8192)),
    ('normalize', osc.Normalize())
])

# Transform trace
result = pipeline.transform(trace)

# Get trace output from filter stage
filtered = pipeline.get_intermediate('filter')

# Get FFT-specific intermediates
spectrum = pipeline.get_intermediate('fft', 'spectrum')
frequencies = pipeline.get_intermediate('fft', 'frequencies')
power = pipeline.get_intermediate('fft', 'power')
```

#### `has_intermediate()`

Check if intermediate result is available.

```python
def has_intermediate(
    self,
    step_name: str,
    key: str | None = None
) -> bool
```

**Example:**

```python
if pipeline.has_intermediate('fft', 'spectrum'):
    spectrum = pipeline.get_intermediate('fft', 'spectrum')
```

#### `list_intermediates()`

List available intermediate results.

```python
def list_intermediates(
    self,
    step_name: str | None = None
) -> list[str] | dict[str, list[str]]
```

**Parameters:**

| Parameter   | Type         | Description                                         |
| ----------- | ------------ | --------------------------------------------------- |
| `step_name` | `str \|None` | If specified, list intermediates for that step only |

**Returns:** List of keys for a step, or dict mapping all steps to their intermediates.

**Example:**

```python
# List all intermediates
all_intermediates = pipeline.list_intermediates()
print(all_intermediates)
# {'filter': ['transfer_function', 'impulse_response'],
#  'fft': ['spectrum', 'frequencies', 'power', 'phase']}

# List intermediates for specific step
fft_intermediates = pipeline.list_intermediates('fft')
print(fft_intermediates)
# ['spectrum', 'frequencies', 'power', 'phase']
```

#### Parameter Management

Get and set pipeline parameters using sklearn-style syntax:

```python
# Get all parameters
params = pipeline.get_params()
print(params['filter__cutoff'])  # Access nested parameter

# Set parameters
pipeline.set_params(
    filter__cutoff=2e6,
    normalize__method='zscore'
)

# Clone pipeline
pipeline_copy = pipeline.clone()
```

## TraceTransformer Base Class

### `TraceTransformer`

Abstract base class for all trace transformations.

```python
from abc import ABC, abstractmethod

class TraceTransformer(ABC):
    @abstractmethod
    def transform(self, trace: WaveformTrace) -> WaveformTrace

    def fit(self, trace: WaveformTrace) -> TraceTransformer
    def fit_transform(self, trace: WaveformTrace) -> WaveformTrace
    def get_params(self, deep: bool = True) -> dict[str, Any]
    def set_params(self, **params: Any) -> TraceTransformer
    def clone(self) -> TraceTransformer
```

All custom transformers must inherit from this class and implement `transform()`.

**Required Methods:**

- `transform(trace)` - Transform the input trace (must override)

**Optional Methods:**

- `fit(trace)` - Learn parameters from reference trace
- `fit_transform(trace)` - Fit then transform
- `get_params()` - Get transformer parameters
- `set_params(**params)` - Set transformer parameters
- `clone()` - Create a copy of the transformer

**Example:**

```python
import oscura as osc
import numpy as np

class NoiseReducer(osc.TraceTransformer):
    """Reduce noise using moving average."""

    def __init__(self, window_size=5):
        self.window_size = window_size

    def transform(self, trace):
        kernel = np.ones(self.window_size) / self.window_size
        smoothed = np.convolve(trace.data, kernel, mode='same')

        return osc.WaveformTrace(
            data=smoothed,
            metadata=trace.metadata
        )

# Use standalone
reducer = NoiseReducer(window_size=10)
result = reducer.transform(trace)

# Or in pipeline
pipeline = osc.Pipeline([
    ('noise_reduce', NoiseReducer(window_size=10)),
    ('filter', osc.LowPassFilter(cutoff=1e6))
])
```

### Intermediate Results

Transformers can cache intermediate results for inspection:

```python
class FFTTransformer(osc.TraceTransformer):
    """FFT transformer that caches intermediate results."""

    def transform(self, trace):
        # Clear previous intermediates
        self._clear_intermediates()

        # Compute FFT
        spectrum = np.fft.fft(trace.data)
        frequencies = np.fft.fftfreq(len(trace.data),
                                      1.0/trace.metadata.sample_rate)
        power = np.abs(spectrum) ** 2
        phase = np.angle(spectrum)

        # Cache intermediates
        self._cache_intermediate('spectrum', spectrum)
        self._cache_intermediate('frequencies', frequencies)
        self._cache_intermediate('power', power)
        self._cache_intermediate('phase', phase)

        # Return transformed trace
        magnitude = np.abs(spectrum)
        return osc.WaveformTrace(
            data=magnitude,
            metadata=trace.metadata
        )

# Use transformer
fft = FFTTransformer()
result = fft.transform(trace)

# Access intermediates
if fft.has_intermediate_result('power'):
    power = fft.get_intermediate_result('power')

# List available intermediates
print(fft.list_intermediate_results())
# ['spectrum', 'frequencies', 'power', 'phase']
```

## Functional Composition

### `compose()`

Compose functions right-to-left (mathematical composition).

```python
def compose(*funcs: TraceFunc) -> TraceFunc
```

Creates a single function that applies functions in reverse order: `compose(f, g, h)(x)` equals `f(g(h(x)))`.

**Parameters:**

| Parameter | Type                                       | Description                           |
| --------- | ------------------------------------------ | ------------------------------------- |
| `*funcs`  | `Callable[[WaveformTrace], WaveformTrace]` | Functions to compose in reverse order |

**Returns:** Composite function.

**Example:**

```python
import oscura as osc
from functools import partial

# Create composed analysis function
analyze_signal = osc.compose(
    osc.extract_thd,                              # Apply last
    partial(osc.fft, nfft=8192, window='hann'),   # Apply second
    partial(osc.normalize, method='peak'),        # Apply first (innermost)
)

# Apply to trace: normalize -> fft -> extract_thd
thd = analyze_signal(trace)

# Equivalent to:
# thd = osc.extract_thd(
#     osc.fft(
#         osc.normalize(trace, method='peak'),
#         nfft=8192, window='hann'
#     )
# )
```

### `pipe()`

Apply functions left-to-right (pipeline order).

```python
def pipe(data: WaveformTrace, *funcs: TraceFunc) -> WaveformTrace
```

Applies functions sequentially: `pipe(x, f, g, h)` equals `h(g(f(x)))`.

**Parameters:**

| Parameter | Type                                       | Description                 |
| --------- | ------------------------------------------ | --------------------------- |
| `data`    | `WaveformTrace`                            | Initial trace to process    |
| `*funcs`  | `Callable[[WaveformTrace], WaveformTrace]` | Functions to apply in order |

**Returns:** Transformed WaveformTrace.

**Example:**

```python
import oscura as osc
from functools import partial

# Apply operations left-to-right (more intuitive)
result = osc.pipe(
    trace,
    partial(osc.low_pass, cutoff=1e6),        # Apply first
    partial(osc.resample, rate=1e9),          # Apply second
    partial(osc.normalize, method='zscore'),  # Apply third
    partial(osc.fft, nfft=8192)               # Apply last
)

# Equivalent to Pipeline
pipeline = osc.Pipeline([
    ('filter', osc.LowPassFilter(cutoff=1e6)),
    ('resample', osc.Resample(rate=1e9)),
    ('normalize', osc.Normalize(method='zscore')),
    ('fft', osc.FFT(nfft=8192))
])
result = pipeline.transform(trace)
```

### `curry()`

Curry a function for easier composition.

```python
def curry(func: Callable[..., WaveformTrace]) -> Callable[..., TraceFunc]
```

Transforms a multi-argument function into a series of single-argument functions.

**Parameters:**

| Parameter | Type                           | Description       |
| --------- | ------------------------------ | ----------------- |
| `func`    | `Callable[..., WaveformTrace]` | Function to curry |

**Returns:** Curried version of the function.

**Example:**

```python
import oscura as osc
import numpy as np

@osc.curry
def scale_and_offset(trace, scale, offset):
    """Scale and offset trace data."""
    return osc.WaveformTrace(
        data=trace.data * scale + offset,
        metadata=trace.metadata
    )

# Create specialized functions
double_and_shift = scale_and_offset(scale=2.0, offset=1.0)
result = double_and_shift(trace)

# Use in composition
result = osc.pipe(
    trace,
    scale_and_offset(scale=2.0, offset=0.0),
    scale_and_offset(scale=0.5, offset=1.0)
)
```

### `make_composable()`

Decorator to make a function support partial application.

```python
def make_composable(func: Callable[..., WaveformTrace]) -> Callable[..., TraceFunc]
```

**Example:**

```python
import oscura as osc
import numpy as np

@osc.make_composable
def moving_average(trace, window_size=5):
    """Apply moving average filter."""
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(trace.data, kernel, mode='same')
    return osc.WaveformTrace(
        data=smoothed,
        metadata=trace.metadata
    )

# Use with partial application
smooth_5 = moving_average(window_size=5)
smooth_10 = moving_average(window_size=10)

# Use in pipeline
result = osc.pipe(
    trace,
    moving_average(window_size=5),
    moving_average(window_size=3)
)
```

### `Composable` Mixin

Mixin class to enable `>>` operator for function composition.

```python
class Composable:
    def __rshift__(self, func: Callable[[Any], Any]) -> Any
```

**Example:**

```python
# If WaveformTrace inherits from Composable
result = (
    trace
    >> low_pass(cutoff=1e6)
    >> normalize()
    >> fft(nfft=8192)
)

# Equivalent to:
result = fft(normalize(low_pass(trace, cutoff=1e6)), nfft=8192)
```

## Complete Examples

### Building a Signal Analysis Pipeline

```python
import oscura as osc
import numpy as np

# Define custom transformers
class OutlierClipper(osc.TraceTransformer):
    """Clip outliers beyond N standard deviations."""

    def __init__(self, n_sigma=3.0):
        self.n_sigma = n_sigma

    def transform(self, trace):
        mean = trace.data.mean()
        std = trace.data.std()
        lower = mean - self.n_sigma * std
        upper = mean + self.n_sigma * std
        clipped = np.clip(trace.data, lower, upper)

        return osc.WaveformTrace(
            data=clipped,
            metadata=trace.metadata
        )

class TrendRemover(osc.TraceTransformer):
    """Remove linear trend from signal."""

    def __init__(self):
        self.slope_ = None
        self.intercept_ = None

    def fit(self, trace):
        x = np.arange(len(trace.data))
        coeffs = np.polyfit(x, trace.data, 1)
        self.slope_ = coeffs[0]
        self.intercept_ = coeffs[1]
        return self

    def transform(self, trace):
        if self.slope_ is None:
            trend = trace.data.mean()
        else:
            x = np.arange(len(trace.data))
            trend = self.slope_ * x + self.intercept_

        detrended = trace.data - trend
        return osc.WaveformTrace(
            data=detrended,
            metadata=trace.metadata
        )

# Build comprehensive analysis pipeline
pipeline = osc.Pipeline([
    ('clip_outliers', OutlierClipper(n_sigma=3.0)),
    ('remove_trend', TrendRemover()),
    ('filter', osc.LowPassFilter(cutoff=1e6)),
    ('normalize', osc.Normalize(method='zscore'))
])

# Fit and transform
pipeline.fit(reference_trace)
result = pipeline.transform(test_trace)

# Access intermediate results
after_clipping = pipeline.get_intermediate('clip_outliers')
after_detrend = pipeline.get_intermediate('remove_trend')
filtered = pipeline.get_intermediate('filter')

print(f"Original std: {test_trace.data.std():.3f}")
print(f"After clipping std: {after_clipping.data.std():.3f}")
print(f"After detrend std: {after_detrend.data.std():.3f}")
print(f"Final std: {result.data.std():.3f}")
```

### Functional Programming Style

```python
import oscura as osc
from functools import partial

# Define transformations as functions
def remove_dc(trace):
    """Remove DC component."""
    centered = trace.data - trace.data.mean()
    return osc.WaveformTrace(data=centered, metadata=trace.metadata)

def scale_to_range(trace, min_val=-1.0, max_val=1.0):
    """Scale trace to specified range."""
    data_min = trace.data.min()
    data_max = trace.data.max()
    scaled = (trace.data - data_min) / (data_max - data_min)
    scaled = scaled * (max_val - min_val) + min_val
    return osc.WaveformTrace(data=scaled, metadata=trace.metadata)

# Compose operations
preprocess = osc.compose(
    partial(scale_to_range, min_val=-1.0, max_val=1.0),
    remove_dc,
    partial(osc.low_pass, cutoff=1e6)
)

# Apply to trace
result = preprocess(trace)

# Or use pipe for left-to-right flow
result = osc.pipe(
    trace,
    partial(osc.low_pass, cutoff=1e6),
    remove_dc,
    partial(scale_to_range, min_val=-1.0, max_val=1.0)
)
```

### Creating Reusable Analysis Components

```python
import oscura as osc
from functools import partial

# Create reusable analysis pipeline factory
def create_digital_analysis_pipeline(
    threshold=0.5,
    filter_cutoff=1e6,
    sample_rate=1e9
):
    """Factory for digital signal analysis pipeline."""
    return osc.Pipeline([
        ('filter', osc.LowPassFilter(cutoff=filter_cutoff)),
        ('resample', osc.Resample(rate=sample_rate)),
        ('threshold', osc.Threshold(level=threshold)),
        ('edges', osc.EdgeDetector())
    ])

# Create reusable spectral analysis function
def spectral_analysis(nfft=8192, window='hann'):
    """Create spectral analysis function."""
    return osc.compose(
        osc.extract_harmonics,
        partial(osc.fft, nfft=nfft, window=window),
        partial(osc.normalize, method='peak')
    )

# Use reusable components
digital_pipeline = create_digital_analysis_pipeline(
    threshold=0.5,
    filter_cutoff=5e6
)
result = digital_pipeline.transform(trace)

# Apply spectral analysis
analyze = spectral_analysis(nfft=16384, window='hamming')
harmonics = analyze(trace)
```

### Combining Pipeline and Functional Approaches

```python
import oscura as osc
from functools import partial

# Create base pipeline
base_pipeline = osc.Pipeline([
    ('filter', osc.BandPassFilter(low=1e5, high=1e7)),
    ('normalize', osc.Normalize(method='peak'))
])

# Define post-processing functions
def compute_envelope(trace):
    """Compute signal envelope."""
    analytic = scipy.signal.hilbert(trace.data)
    envelope = np.abs(analytic)
    return osc.WaveformTrace(data=envelope, metadata=trace.metadata)

def extract_peaks(trace, threshold=0.5):
    """Extract peak locations."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(trace.data, height=threshold)
    return peaks

# Combine pipeline with functional composition
def analyze_signal(trace):
    """Complete signal analysis workflow."""
    # Apply base pipeline
    processed = base_pipeline.transform(trace)

    # Get filtered intermediate
    filtered = base_pipeline.get_intermediate('filter')

    # Apply post-processing
    envelope = compute_envelope(processed)
    peaks = extract_peaks(envelope, threshold=0.5)

    return {
        'processed': processed,
        'filtered': filtered,
        'envelope': envelope,
        'peaks': peaks,
        'num_peaks': len(peaks)
    }

# Run analysis
results = analyze_signal(trace)
print(f"Found {results['num_peaks']} peaks")
```

## Pipeline Serialization

Save and load pipelines for reuse:

```python
import joblib
import oscura as osc

# Create and fit pipeline
pipeline = osc.Pipeline([
    ('filter', osc.LowPassFilter(cutoff=1e6)),
    ('normalize', osc.Normalize())
])
pipeline.fit(reference_trace)

# Save pipeline
joblib.dump(pipeline, 'analysis_pipeline.pkl')

# Load pipeline
loaded_pipeline = joblib.load('analysis_pipeline.pkl')
result = loaded_pipeline.transform(test_trace)
```

## Best Practices

### When to Use Pipeline vs Functional Composition

**Use Pipeline when:**

- You need stateful transformers (fit/transform pattern)
- You want to inspect intermediate results
- You need parameter tuning (get_params/set_params)
- You want to serialize the workflow
- Building complex, multi-stage analysis workflows

**Use functional composition when:**

- You have simple, stateless transformations
- You want concise, readable code
- You're working with pure functions
- You need quick prototyping

### Pipeline Design Tips

1. **Keep transformers focused**: Each transformer should do one thing well
2. **Name steps clearly**: Use descriptive names for pipeline steps
3. **Cache intermediates**: Store useful intermediate results for inspection
4. **Document parameters**: Include docstrings with parameter descriptions
5. **Handle edge cases**: Validate inputs and handle NaN/inf values
6. **Make transformers stateless when possible**: Easier to reason about and test

### Performance Considerations

```python
# Avoid: Creating pipeline in loop
for trace in traces:
    pipeline = osc.Pipeline([...])  # Recreated each time
    result = pipeline.transform(trace)

# Better: Reuse pipeline
pipeline = osc.Pipeline([...])
results = [pipeline.transform(t) for t in traces]

# Best: Use fit once for all traces
pipeline.fit(reference_trace)
results = [pipeline.transform(t) for t in traces]
```

## See Also

- [Analysis API](analysis.md) - Built-in analysis functions
- [Loader API](loader.md) - Data loading
- [Export API](export.md) - Save results
- [Advanced Examples](../../demos/15_complete_workflows/) - Complete workflow demonstrations
