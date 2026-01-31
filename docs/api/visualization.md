# Oscura Visualization API

> **Version**: 0.8.0 | **Last Updated**: 2026-01-25

The Oscura Visualization API provides comprehensive plotting functions for waveform and spectral analysis. All visualization functions are accessible from the top-level API as `osc.plot_*()`.

## Overview

Oscura provides a complete suite of visualization functions organized by category:

### Time-Domain Visualization

1. **`osc.plot_waveform()`** - Time-domain waveform plotting
2. **`osc.plot_multi_channel()`** - Multi-channel stacked plots
3. **`osc.plot_xy()`** - X-Y (Lissajous) plots

### Frequency-Domain Visualization

1. **`osc.plot_spectrum()`** - Frequency-domain spectrum plotting
2. **`osc.plot_fft()`** - FFT magnitude spectrum plotting
3. **`osc.plot_psd()`** - Power spectral density plots
4. **`osc.plot_spectrogram()`** - Time-frequency spectrograms
5. **`osc.plot_waterfall()`** - 3D waterfall plots

### Digital Signal Visualization

1. **`osc.plot_timing()`** - Digital timing diagrams
2. **`osc.plot_logic_analyzer()`** - Logic analyzer style plots

### Signal Integrity & Analysis

1. **`osc.plot_eye()`** - Eye diagrams for signal integrity
2. **`osc.plot_bathtub()`** - Bathtub curves for BER analysis
3. **`osc.plot_bode()`** - Bode magnitude and phase plots
4. **`osc.plot_phase()`** - Phase relationship plots
5. **`osc.plot_histogram()`** - Amplitude distribution histograms

All functions support:

- Automatic unit selection (time/frequency)
- Custom styling (colors, labels, titles, sizes)
- High-resolution file export (PNG, PDF, SVG)
- GUI and non-GUI (server) environments
- Publication-quality output
- Interactive and non-interactive modes

## Quick Start

```python
import oscura as osc

# Load a trace
trace = osc.load("signal.wfm")

# Plot time-domain waveform
osc.plot_waveform(trace)

# Plot frequency spectrum
osc.plot_spectrum(trace)

# Plot FFT with custom settings
osc.plot_fft(trace, freq_unit="MHz", log_scale=True)
```

## API Reference

### osc.plot_waveform()

Plot time-domain waveform with automatic time unit selection.

**Signature:**

```python
osc.plot_waveform(
    trace,
    ax=None,
    time_unit="auto",
    show_grid=True,
    color="C0",
    label=None,
    show_measurements=None,
    title=None,
    xlabel="Time",
    ylabel="Amplitude",
    show=True,
    save_path=None,
    figsize=(10, 6)
)
```

**Parameters:**

- `trace` (WaveformTrace): Waveform trace to plot
- `ax` (Axes, optional): Matplotlib axes object. If None, creates new figure
- `time_unit` (str): Time unit - "s", "ms", "us", "ns", or "auto" (default: "auto")
- `show_grid` (bool): Display grid lines (default: True)
- `color` (str): Line color (default: "C0")
- `label` (str, optional): Legend label
- `show_measurements` (dict, optional): Dictionary of measurements to annotate
- `title` (str, optional): Plot title
- `xlabel` (str): X-axis label prefix (default: "Time")
- `ylabel` (str): Y-axis label (default: "Amplitude")
- `show` (bool): Call plt.show() to display (default: True)
- `save_path` (str, optional): Path to save figure
- `figsize` (tuple): Figure size in inches (width, height) (default: (10, 6))

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Basic usage
osc.plot_waveform(trace)

# Custom styling
fig = osc.plot_waveform(
    trace,
    title="Captured Signal",
    time_unit="us",
    color="blue",
    ylabel="Voltage (V)",
    show=False,
    save_path="waveform.png"
)
```

---

### osc.plot_spectrum()

Plot frequency-domain magnitude spectrum with automatic frequency unit selection.

**Signature:**

```python
osc.plot_spectrum(
    trace,
    ax=None,
    freq_unit="auto",
    db_ref=None,
    show_grid=True,
    color="C0",
    title=None,
    window="hann",
    xscale="log",
    show=True,
    save_path=None,
    figsize=(10, 6),
    xlim=None,
    ylim=None,
    fft_result=None,
    log_scale=True
)
```

**Parameters:**

- `trace` (WaveformTrace): Waveform trace to analyze
- `ax` (Axes, optional): Matplotlib axes object
- `freq_unit` (str): Frequency unit - "Hz", "kHz", "MHz", "GHz", or "auto" (default: "auto")
- `db_ref` (float, optional): Reference for dB scaling
- `show_grid` (bool): Display grid lines (default: True)
- `color` (str): Line color (default: "C0")
- `title` (str, optional): Plot title
- `window` (str): Window function for FFT (default: "hann")
- `xscale` (str): X-axis scale - "linear" or "log" (deprecated, use log_scale)
- `show` (bool): Call plt.show() to display (default: True)
- `save_path` (str, optional): Path to save figure
- `figsize` (tuple): Figure size in inches (default: (10, 6))
- `xlim` (tuple, optional): X-axis limits (min, max) in selected units
- `ylim` (tuple, optional): Y-axis limits (min, max) in dB
- `fft_result` (tuple, optional): Pre-computed FFT (frequencies, magnitudes)
- `log_scale` (bool): Use logarithmic frequency axis (default: True)

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Basic usage
osc.plot_spectrum(trace)

# With custom settings
osc.plot_spectrum(
    trace,
    freq_unit="MHz",
    log_scale=True,
    xlim=(1, 100),
    ylim=(-80, 0),
    save_path="spectrum.png"
)

# Using pre-computed FFT
freq, mag = osc.fft(trace)
osc.plot_spectrum(trace, fft_result=(freq, mag))
```

---

### osc.plot_fft()

Plot FFT magnitude spectrum with automatic frequency unit selection. This is a convenience function that combines FFT computation and visualization.

**Signature:**

```python
osc.plot_fft(
    trace,
    ax=None,
    show=True,
    save_path=None,
    title=None,
    xlabel="Frequency",
    ylabel="Magnitude (dB)",
    figsize=(10, 6),
    freq_unit="auto",
    log_scale=True,
    show_grid=True,
    color="C0",
    window="hann",
    xlim=None,
    ylim=None
)
```

**Parameters:**

- `trace` (WaveformTrace): Waveform trace to analyze
- `ax` (Axes, optional): Matplotlib axes object
- `show` (bool): Call plt.show() to display (default: True)
- `save_path` (str, optional): Path to save figure
- `title` (str, optional): Plot title (default: "FFT Magnitude Spectrum")
- `xlabel` (str): X-axis label prefix (default: "Frequency")
- `ylabel` (str): Y-axis label (default: "Magnitude (dB)")
- `figsize` (tuple): Figure size in inches (default: (10, 6))
- `freq_unit` (str): Frequency unit - "Hz", "kHz", "MHz", "GHz", or "auto" (default: "auto")
- `log_scale` (bool): Use logarithmic frequency axis (default: True)
- `show_grid` (bool): Display grid lines (default: True)
- `color` (str): Line color (default: "C0")
- `window` (str): Window function for FFT (default: "hann")
- `xlim` (tuple, optional): X-axis limits (min, max)
- `ylim` (tuple, optional): Y-axis limits (min, max)

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Basic usage
osc.plot_fft(trace)

# Custom FFT plot
fig = osc.plot_fft(
    trace,
    title="Signal FFT Analysis",
    freq_unit="MHz",
    log_scale=True,
    xlim=(0.1, 100),
    ylim=(-100, 0),
    figsize=(12, 6),
    show=False,
    save_path="fft.png"
)
```

---

### osc.plot_multi_channel()

Plot multiple channels in stacked subplots with synchronized time axes.

**Signature:**

```python
osc.plot_multi_channel(
    traces,
    names=None,
    shared_x=True,
    colors=None,
    time_unit="auto",
    show_grid=True,
    figsize=None,
    title=None
)
```

**Parameters:**

- `traces` (list[WaveformTrace]): List of traces to plot
- `names` (list[str], optional): Channel names for labels. If None, uses CH1, CH2, etc.
- `shared_x` (bool): Share x-axis across subplots (default: True)
- `colors` (list[str], optional): List of colors for each trace
- `time_unit` (str): Time unit - "s", "ms", "us", "ns", or "auto" (default: "auto")
- `show_grid` (bool): Display grid lines (default: True)
- `figsize` (tuple, optional): Figure size in inches (width, height)
- `title` (str, optional): Overall figure title

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

# Load multiple channels
ch1 = osc.load("channel1.wfm")
ch2 = osc.load("channel2.wfm")
ch3 = osc.load("channel3.wfm")

# Plot all channels
fig = osc.plot_multi_channel(
    [ch1, ch2, ch3],
    names=["CLK", "DATA", "CS"],
    title="SPI Bus Signals"
)

# Custom colors
fig = osc.plot_multi_channel(
    [ch1, ch2],
    names=["Input", "Output"],
    colors=["blue", "red"],
    figsize=(12, 8)
)
```

**Use Cases:**

- Multi-channel oscilloscope data visualization
- Bus signal analysis (I2C, SPI, UART)
- Comparing multiple related signals
- Trigger and data signal correlation

---

### osc.plot_xy()

Create X-Y (Lissajous) plots showing phase relationships between two signals.

**Signature:**

```python
osc.plot_xy(
    x_trace,
    y_trace,
    ax=None,
    color="C0",
    marker="",
    alpha=0.7,
    title=None
)
```

**Parameters:**

- `x_trace` (WaveformTrace or ndarray): X-axis waveform
- `y_trace` (WaveformTrace or ndarray): Y-axis waveform
- `ax` (Axes, optional): Matplotlib axes object
- `color` (str): Line/marker color (default: "C0")
- `marker` (str): Marker style (default: "")
- `alpha` (float): Transparency (0.0 to 1.0) (default: 0.7)
- `title` (str, optional): Plot title

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

# Load two phase-shifted signals
ch1 = osc.load("signal_x.wfm")
ch2 = osc.load("signal_y.wfm")

# Create Lissajous figure
fig = osc.plot_xy(ch1, ch2, title="Phase Relationship")

# With markers
fig = osc.plot_xy(
    ch1, ch2,
    marker="o",
    alpha=0.5,
    color="purple"
)
```

**Use Cases:**

- Visualizing phase relationships
- Checking signal orthogonality
- Analyzing quadrature signals (I/Q)
- Detecting phase drift

---

### osc.plot_psd()

Plot Power Spectral Density showing power distribution across frequencies.

**Signature:**

```python
osc.plot_psd(
    trace,
    ax=None,
    freq_unit="auto",
    show_grid=True,
    color="C0",
    title=None,
    window="hann",
    xscale="log"
)
```

**Parameters:**

- `trace` (WaveformTrace): Waveform trace to analyze
- `ax` (Axes, optional): Matplotlib axes object
- `freq_unit` (str): Frequency unit - "Hz", "kHz", "MHz", "GHz", or "auto" (default: "auto")
- `show_grid` (bool): Display grid lines (default: True)
- `color` (str): Line color (default: "C0")
- `title` (str, optional): Plot title
- `window` (str): Window function for FFT (default: "hann")
- `xscale` (str): X-axis scale - "linear" or "log" (default: "log")

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

trace = osc.load("noisy_signal.wfm")

# Plot PSD
fig = osc.plot_psd(trace)

# With linear frequency axis
fig = osc.plot_psd(
    trace,
    freq_unit="MHz",
    xscale="linear",
    title="Power Spectral Density"
)
```

**Use Cases:**

- Noise characterization
- Signal power analysis
- Interference detection
- Frequency content analysis

---

### osc.plot_spectrogram()

Create time-frequency spectrogram showing how spectrum evolves over time.

**Signature:**

```python
osc.plot_spectrogram(
    trace,
    ax=None,
    time_unit="auto",
    freq_unit="auto",
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
    window="hann",
    nperseg=None,
    nfft=None,
    overlap=None
)
```

**Parameters:**

- `trace` (WaveformTrace): Waveform trace to analyze
- `ax` (Axes, optional): Matplotlib axes object
- `time_unit` (str): Time unit - "s", "ms", "us", "ns", or "auto" (default: "auto")
- `freq_unit` (str): Frequency unit - "Hz", "kHz", "MHz", "GHz", or "auto" (default: "auto")
- `cmap` (str): Colormap name (default: "viridis")
- `vmin` (float, optional): Minimum dB value for color scaling
- `vmax` (float, optional): Maximum dB value for color scaling
- `title` (str, optional): Plot title
- `window` (str): Window function (default: "hann")
- `nperseg` (int, optional): Segment length for STFT
- `nfft` (int, optional): FFT length (overrides nperseg)
- `overlap` (float, optional): Overlap fraction (0.0 to 1.0, default: 0.5)

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

trace = osc.load("chirp_signal.wfm")

# Basic spectrogram
fig = osc.plot_spectrogram(trace)

# With custom settings
fig = osc.plot_spectrogram(
    trace,
    nperseg=1024,
    overlap=0.75,
    cmap="hot",
    title="Chirp Signal Spectrogram"
)
```

**Use Cases:**

- Time-varying frequency analysis
- Chirp signal analysis
- Transient detection
- Frequency hopping visualization
- FM/AM signal analysis

---

### osc.plot_waterfall()

Create 3D waterfall plot showing spectrum evolution with depth visualization.

**Signature:**

```python
osc.plot_waterfall(
    data,
    time_axis=None,
    freq_axis=None,
    sample_rate=1.0,
    nperseg=256,
    noverlap=None,
    cmap="viridis",
    ax=None
)
```

**Parameters:**

- `data` (ndarray): Input signal (1D) or pre-computed spectrogram (2D)
- `time_axis` (ndarray, optional): Time axis for signal
- `freq_axis` (ndarray, optional): Frequency axis (if pre-computed)
- `sample_rate` (float): Sample rate in Hz (default: 1.0)
- `nperseg` (int): Segment length for FFT (default: 256)
- `noverlap` (int, optional): Overlap between segments
- `cmap` (str): Colormap for amplitude coloring (default: "viridis")
- `ax` (3D Axes, optional): Existing 3D axes to plot on

**Returns:**

- `tuple`: (Figure, Axes)

**Example:**

```python
import oscura as osc

trace = osc.load("fm_signal.wfm")

# Create waterfall plot
fig, ax = osc.plot_waterfall(
    trace.data,
    sample_rate=trace.metadata.sample_rate
)

# With custom parameters
fig, ax = osc.plot_waterfall(
    trace.data,
    sample_rate=1e6,
    nperseg=512,
    cmap="hot"
)
```

**Use Cases:**

- Spectrum monitoring over time
- Radio frequency analysis
- EMI/EMC testing visualization
- Frequency stability analysis

---

### osc.plot_timing()

Create digital timing diagrams with protocol decode overlay support.

**Signature:**

```python
osc.plot_timing(
    traces,
    names=None,
    annotations=None,
    time_unit="auto",
    show_grid=True,
    figsize=None,
    title=None,
    time_range=None,
    threshold="auto"
)
```

**Parameters:**

- `traces` (list[WaveformTrace]): List of traces to plot (analog or digital)
- `names` (list[str], optional): Channel names for labels
- `annotations` (list[list[Annotation]], optional): Protocol annotations per channel
- `time_unit` (str): Time unit - "s", "ms", "us", "ns", or "auto" (default: "auto")
- `show_grid` (bool): Show vertical grid lines (default: True)
- `figsize` (tuple, optional): Figure size in inches
- `title` (str, optional): Overall figure title
- `time_range` (tuple, optional): (start, end) time range in seconds
- `threshold` (float or str): Threshold for analog-to-digital conversion (default: "auto")

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

# Load digital signals
clk = osc.load("clock.wfm")
data = osc.load("data.wfm")
cs = osc.load("chip_select.wfm")

# Create timing diagram
fig = osc.plot_timing(
    [clk, data, cs],
    names=["CLK", "DATA", "CS"],
    title="SPI Communication"
)

# With time range zoom
fig = osc.plot_timing(
    [clk, data],
    names=["Clock", "Data"],
    time_range=(0, 1e-6),  # First microsecond
    time_unit="ns"
)
```

**Use Cases:**

- Digital protocol debugging
- Timing relationship visualization
- Setup and hold time analysis
- Protocol decode verification

---

### osc.plot_logic_analyzer()

Create logic analyzer style display with bus grouping support.

**Signature:**

```python
osc.plot_logic_analyzer(
    traces,
    names=None,
    bus_groups=None,
    time_unit="auto",
    show_grid=True,
    figsize=None,
    title=None
)
```

**Parameters:**

- `traces` (list[DigitalTrace]): List of digital traces
- `names` (list[str], optional): Channel names
- `bus_groups` (dict, optional): Dictionary mapping bus names to channel indices
  - Example: `{"DATA": [0, 1, 2, 3], "ADDR": [4, 5, 6, 7]}`
- `time_unit` (str): Time unit (default: "auto")
- `show_grid` (bool): Show vertical grid lines (default: True)
- `figsize` (tuple, optional): Figure size
- `title` (str, optional): Plot title

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

# Load 8 data lines
traces = [osc.load(f"d{i}.wfm") for i in range(8)]

# Plot with bus grouping
fig = osc.plot_logic_analyzer(
    traces,
    names=[f"D{i}" for i in range(8)],
    bus_groups={"DATA": [0, 1, 2, 3, 4, 5, 6, 7]}
)

# Multiple buses
fig = osc.plot_logic_analyzer(
    traces,
    bus_groups={
        "DATA": [0, 1, 2, 3],
        "ADDR": [4, 5, 6, 7]
    }
)
```

**Use Cases:**

- Multi-bit bus visualization
- Parallel data analysis
- Address/data bus monitoring
- Digital system debugging

---

### osc.plot_eye()

Create eye diagram for signal integrity analysis with clock recovery.

**Signature:**

```python
osc.plot_eye(
    trace,
    bit_rate=None,
    clock_recovery="edge",
    samples_per_bit=None,
    ui_count=2,
    ax=None,
    cmap="hot",
    alpha=0.3,
    show_measurements=True,
    title=None,
    colorbar=False
)
```

**Parameters:**

- `trace` (WaveformTrace): Input waveform (serial data signal)
- `bit_rate` (float, optional): Bit rate in bits/second. Auto-recovered if None
- `clock_recovery` (str): Method for clock recovery - "fft" or "edge" (default: "edge")
- `samples_per_bit` (int, optional): Samples per bit period (auto-calculated if None)
- `ui_count` (int): Number of Unit Intervals to display (default: 2)
- `ax` (Axes, optional): Matplotlib axes object
- `cmap` (str): Colormap for density visualization (default: "hot")
- `alpha` (float): Transparency for overlaid traces (default: 0.3)
- `show_measurements` (bool): Annotate eye opening measurements (default: True)
- `title` (str, optional): Plot title
- `colorbar` (bool): Show colorbar for density plot (default: False)

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

trace = osc.load("serial_data.wfm")

# With known bit rate
fig = osc.plot_eye(trace, bit_rate=1e9)  # 1 Gbps

# Auto-recover clock
fig = osc.plot_eye(
    trace,
    clock_recovery="fft",
    show_measurements=True,
    cmap="viridis"
)

# High-speed SerDes analysis
fig = osc.plot_eye(
    trace,
    bit_rate=10e9,  # 10 Gbps
    ui_count=1,
    colorbar=True
)
```

**Use Cases:**

- High-speed serial link analysis
- Signal integrity verification
- Jitter analysis
- BER estimation
- Compliance testing (PCIe, USB, SATA, etc.)

---

### osc.plot_bathtub()

Create bathtub curve showing BER vs. sampling position for timing margin analysis.

**Signature:**

```python
osc.plot_bathtub(
    trace,
    bit_rate=None,
    ber_target=1e-12,
    ax=None,
    title=None
)
```

**Parameters:**

- `trace` (WaveformTrace): Input waveform trace
- `bit_rate` (float, optional): Bit rate in bits/second
- `ber_target` (float): Target bit error rate for margin calculation (default: 1e-12)
- `ax` (Axes, optional): Matplotlib axes object
- `title` (str, optional): Plot title

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc

trace = osc.load("serdes_output.wfm")

# Create bathtub curve
fig = osc.plot_bathtub(
    trace,
    bit_rate=1e9,
    ber_target=1e-12
)

# Different BER target
fig = osc.plot_bathtub(
    trace,
    bit_rate=10e9,
    ber_target=1e-15,
    title="10G SerDes Bathtub Curve"
)
```

**Use Cases:**

- Determining optimal sampling point
- Timing margin analysis
- BER extrapolation
- Link budget analysis
- Compliance margin verification

---

### osc.plot_bode()

Create Bode plot with magnitude and phase response.

**Signature:**

```python
osc.plot_bode(
    frequencies,
    magnitude,
    phase=None,
    magnitude_db=True,
    phase_degrees=True,
    show_margins=False,
    fig=None
)
```

**Parameters:**

- `frequencies` (ndarray): Frequency array in Hz
- `magnitude` (ndarray or complex ndarray): Magnitude array or complex transfer function H(s)
- `phase` (ndarray, optional): Phase array in radians (ignored if magnitude is complex)
- `magnitude_db` (bool): If True, magnitude is already in dB (default: True)
- `phase_degrees` (bool): Convert phase to degrees (default: True)
- `show_margins` (bool): Annotate gain and phase margins (default: False)
- `fig` (Figure, optional): Existing figure to plot on

**Returns:**

- `Figure`: Matplotlib Figure object

**Example:**

```python
import oscura as osc
import numpy as np

# Define frequency range
freqs = np.logspace(1, 6, 1000)  # 10 Hz to 1 MHz

# Complex transfer function (low-pass filter)
H = 1 / (1 + 1j * freqs / 10000)

# Create Bode plot
fig = osc.plot_bode(freqs, H)

# With separate magnitude and phase
mag_db = 20 * np.log10(np.abs(H))
phase_rad = np.angle(H)
fig = osc.plot_bode(
    freqs,
    mag_db,
    phase_rad,
    magnitude_db=True,
    show_margins=True
)
```

**Use Cases:**

- Filter frequency response analysis
- Control system stability analysis
- Amplifier characterization
- Feedback loop analysis
- Gain and phase margin determination

---

### osc.plot_phase()

Create phase plot (X-Y plot) showing phase relationships between signals.

**Signature:**

```python
osc.plot_phase(
    trace1,
    trace2=None,
    delay=1,
    delay_samples=None,
    ax=None
)
```

**Parameters:**

- `trace1` (WaveformTrace or ndarray): Signal for X-axis
- `trace2` (WaveformTrace or ndarray, optional): Signal for Y-axis. If None, uses delayed trace1
- `delay` (int): Sample delay for self-phase plot when trace2=None (default: 1)
- `delay_samples` (int, optional): Alias for delay parameter
- `ax` (Axes, optional): Existing axes to plot on

**Returns:**

- `tuple`: (Figure, Axes)

**Example:**

```python
import oscura as osc

# Load two signals
sig_x = osc.load("signal_x.wfm")
sig_y = osc.load("signal_y.wfm")

# Phase relationship plot
fig, ax = osc.plot_phase(sig_x, sig_y)

# Self-phase plot (attractor)
fig, ax = osc.plot_phase(sig_x, delay_samples=10)

# Quadrature signals
i_signal = osc.load("i_channel.wfm")
q_signal = osc.load("q_channel.wfm")
fig, ax = osc.plot_phase(i_signal, q_signal)
```

**Use Cases:**

- Phase relationship visualization
- Lissajous figures
- I/Q signal analysis
- Chaos analysis (strange attractors)
- Feedback system analysis

---

### osc.plot_histogram()

Create histogram of signal amplitude distribution with optional statistics.

**Signature:**

```python
osc.plot_histogram(
    trace,
    bins="auto",
    density=True,
    show_stats=True,
    show_kde=False,
    ax=None
)
```

**Parameters:**

- `trace` (WaveformTrace or ndarray): Input trace or array
- `bins` (int, str, or ndarray): Number of bins or binning strategy (default: "auto")
- `density` (bool): Normalize to probability density (default: True)
- `show_stats` (bool): Show mean and standard deviation lines (default: True)
- `show_kde` (bool): Overlay kernel density estimate (default: False)
- `ax` (Axes, optional): Existing axes to plot on

**Returns:**

- `tuple`: (Figure, Axes, stats_dict)

**Example:**

```python
import oscura as osc

trace = osc.load("noisy_signal.wfm")

# Basic histogram
fig, ax, stats = osc.plot_histogram(trace)
print(f"Mean: {stats['mean']:.3f} V")
print(f"Std Dev: {stats['std']:.3f} V")

# With KDE overlay
fig, ax, stats = osc.plot_histogram(
    trace,
    bins=50,
    show_kde=True,
    show_stats=True
)

# Custom binning
fig, ax, stats = osc.plot_histogram(
    trace,
    bins=np.linspace(-1, 1, 100),
    density=False
)
```

**Use Cases:**

- Noise distribution analysis
- Signal quality assessment
- Detecting multi-level signals
- Statistical characterization
- Gaussian/non-Gaussian noise identification

---

## Style Presets

Oscura provides comprehensive style presets for different output contexts. Style presets control appearance, resolution, fonts, and layout for publication-quality output.

### Available Presets

#### Publication Preset

Optimized for academic papers and technical publications:

```python
import oscura as osc
from oscura.visualization import apply_style_preset

with apply_style_preset("publication"):
    trace = osc.load("signal.wfm")
    osc.plot_waveform(trace, save_path="figure.pdf")
```

**Features:**

- High DPI (300)
- LaTeX-compatible fonts
- Optimized for PDF/SVG export
- Clean, professional appearance
- Suitable for IEEE, ACM, and other academic journals

#### Presentation Preset

Optimized for slides and presentations:

```python
with apply_style_preset("presentation"):
    osc.plot_fft(trace, save_path="slide.png")
```

**Features:**

- Larger fonts for visibility
- High contrast colors
- Bold lines
- Optimized for projectors
- 16:9 aspect ratio friendly

#### Screen Preset

Optimized for interactive viewing and desktop displays:

```python
with apply_style_preset("screen"):
    osc.plot_spectrum(trace)  # Interactive display
```

**Features:**

- Medium DPI (100)
- Comfortable font sizes
- Balanced colors
- Quick rendering
- Default for interactive work

#### Print Preset

Optimized for printed output on paper:

```python
with apply_style_preset("print"):
    osc.plot_eye(trace, save_path="printout.png")
```

**Features:**

- High DPI (300)
- Grayscale-friendly colors
- Clear line styles
- Optimized for black & white printing

### Using Style Presets

**Context Manager (Recommended):**

```python
import oscura as osc
from oscura.visualization import apply_style_preset

# Temporary style application
with apply_style_preset("publication"):
    fig1 = osc.plot_waveform(trace1)
    fig2 = osc.plot_fft(trace2)
# Reverts to previous style after block

# Multiple plots with same style
with apply_style_preset("presentation"):
    for i, trace in enumerate(traces):
        osc.plot_waveform(trace, save_path=f"slide_{i}.png")
```

**Custom Overrides:**

```python
# Apply preset with custom modifications
with apply_style_preset("publication", overrides={"font.size": 14}):
    osc.plot_spectrum(trace)
```

### Creating Custom Presets

Create your own presets based on existing ones:

```python
from oscura.visualization import create_custom_preset, register_preset

# Create custom preset
custom = create_custom_preset(
    "my_style",
    base_preset="publication",
    font_size=12,
    line_width=2.0,
    dpi=600
)

# Register for future use
register_preset(custom)

# Use it
with apply_style_preset("my_style"):
    osc.plot_waveform(trace)
```

### Listing Available Presets

```python
from oscura.visualization import list_presets

presets = list_presets()
print("Available presets:", presets)
# Output: ['publication', 'presentation', 'screen', 'print', ...]
```

---

## Accessibility Features

Oscura provides comprehensive accessibility features to ensure visualizations are usable by all users, including those with color vision deficiencies.

### Colorblind-Safe Palettes

All Oscura visualizations use colorblind-safe color palettes by default:

```python
import oscura as osc
from oscura.visualization import get_colorblind_palette

# Get colorblind-safe colormap
cmap = get_colorblind_palette("viridis")

# Use in plots
osc.plot_spectrogram(trace, cmap=cmap)
```

**Available Colorblind-Safe Colormaps:**

- `"viridis"` - Perceptually uniform, colorblind-safe (default)
- `"plasma"` - High contrast, warm colors
- `"cividis"` - Optimized specifically for colorblind users
- `"inferno"` - High contrast, warm-to-cool

### Multi-Line Distinguishability

For multi-line plots, Oscura combines colors with line styles for maximum distinguishability:

```python
from oscura.visualization import get_multi_line_styles

# Get distinct styles for 4 lines
styles = get_multi_line_styles(4)

# Apply to plots
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i, (color, linestyle) in enumerate(styles):
    ax.plot(data[i], color=color, linestyle=linestyle)
```

**Features:**

- Combines colorblind-safe colors with varied line styles
- Ensures plots are readable in grayscale
- Automatic cycling through patterns
- WCAG 2.1 compliant contrast ratios

### Pre-Defined Line Styles

```python
from oscura.visualization import LINE_STYLES

print(LINE_STYLES)
# ['solid', 'dashed', 'dotted', 'dashdot']
```

### Qualitative Color Palette

For categorical data:

```python
from oscura.visualization import COLORBLIND_SAFE_QUALITATIVE

# Use first 5 colors
colors = COLORBLIND_SAFE_QUALITATIVE[:5]

# Apply to bar chart or categorical plot
for i, color in enumerate(colors):
    plt.bar(i, values[i], color=color)
```

### Alt-Text Generation

Generate descriptive text for accessibility tools:

```python
from oscura.visualization import generate_alt_text

trace = osc.load("signal.wfm")
fig = osc.plot_waveform(trace)

# Generate alt text
alt_text = generate_alt_text(fig, trace)
print(alt_text)
# "Time-domain waveform plot showing signal with frequency 1.5 MHz,
#  amplitude range -0.5V to 0.5V, duration 10 microseconds"
```

### Pass/Fail Formatting

Format test results with symbols in addition to colors:

```python
from oscura.visualization import format_pass_fail

result = format_pass_fail(True)  # "✓ PASS"
result = format_pass_fail(False)  # "✗ FAIL"

# Use in annotations
ax.text(0.5, 0.5, format_pass_fail(test_passed))
```

### Accessibility Checklist

When creating visualizations:

1. ✓ Use default colorblind-safe palettes
2. ✓ Combine colors with line styles for multi-line plots
3. ✓ Ensure text has sufficient contrast (WCAG AA: 4.5:1 minimum)
4. ✓ Provide alt-text for screen readers
5. ✓ Use symbols in addition to colors for critical information
6. ✓ Test plots in grayscale

**Example - Fully Accessible Plot:**

```python
import oscura as osc
from oscura.visualization import (
    apply_style_preset,
    get_multi_line_styles,
    generate_alt_text
)

# Load data
traces = [osc.load(f"ch{i}.wfm") for i in range(4)]

# Apply accessible style
with apply_style_preset("publication"):
    # Get accessible line styles
    styles = get_multi_line_styles(4)

    # Create plot with distinct styles
    fig, ax = plt.subplots(figsize=(10, 6))
    for trace, (color, linestyle), name in zip(traces, styles, names):
        ax.plot(
            trace.time_vector,
            trace.data,
            color=color,
            linestyle=linestyle,
            label=name,
            linewidth=2
        )

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")

    # Generate alt text
    alt_text = "Multi-channel waveform showing 4 synchronized signals..."

    # Save with high DPI
    fig.savefig("accessible_plot.png", dpi=300)
```

---

## Advanced Usage

### Combining Multiple Visualization Functions

Create comprehensive analysis plots combining multiple visualization types:

```python
import oscura as osc
import matplotlib.pyplot as plt

trace = osc.load("signal.wfm")

# Create multi-panel figure
fig = plt.figure(figsize=(12, 10))

# Time domain
ax1 = plt.subplot(2, 2, 1)
osc.plot_waveform(trace, ax=ax1, show=False)

# Frequency domain
ax2 = plt.subplot(2, 2, 2)
osc.plot_spectrum(trace, ax=ax2, show=False)

# Histogram
ax3 = plt.subplot(2, 2, 3)
fig_hist, ax3, stats = osc.plot_histogram(trace, ax=ax3, show=False)

# Eye diagram (if serial data)
ax4 = plt.subplot(2, 2, 4)
fig_eye = osc.plot_eye(trace, ax=ax4, show=False)

plt.tight_layout()
plt.savefig("comprehensive_analysis.png", dpi=300)
```

### Non-GUI Mode (Server Environments)

For headless servers or automated pipelines, set matplotlib to use the 'Agg' backend:

```python
import matplotlib
matplotlib.use('Agg')  # Must be called before importing oscura

import oscura as osc

trace = osc.load("signal.wfm")

# Generate plots without display
osc.plot_waveform(trace, show=False, save_path="waveform.png")
osc.plot_fft(trace, show=False, save_path="fft.png")
osc.plot_eye(trace, show=False, save_path="eye.png")
osc.plot_spectrogram(trace, show=False, save_path="spectrogram.png")
```

### Publication-Quality Plots

Generate high-resolution plots suitable for publications:

```python
import oscura as osc

trace = osc.load("signal.wfm")

# High-quality FFT plot
fig = osc.plot_fft(
    trace,
    figsize=(8, 5),      # Appropriate size for publication
    show=False,
    save_path="fft.pdf"   # Vector format for scaling
)
```

The plots are saved at 300 DPI by default, suitable for most publications.

### Multiple Trace Overlay

To overlay multiple traces on the same plot:

```python
import matplotlib.pyplot as plt
import oscura as osc

trace1 = osc.load("signal1.wfm")
trace2 = osc.load("signal2.wfm")

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot waveforms
osc.plot_waveform(trace1, ax=ax1, color='blue', label='Signal 1', show=False)
osc.plot_waveform(trace2, ax=ax1, color='red', label='Signal 2', show=False)
ax1.legend()

# Plot spectra
osc.plot_spectrum(trace1, ax=ax2, color='blue', show=False)
osc.plot_spectrum(trace2, ax=ax2, color='red', show=False)

plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
```

### Custom Axis Limits

Control frequency and amplitude ranges:

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Focus on specific frequency range
osc.plot_spectrum(
    trace,
    xlim=(1e6, 10e6),    # 1-10 MHz
    ylim=(-80, 0),       # -80 to 0 dB
    freq_unit="MHz",
    log_scale=False      # Linear scale for narrow range
)
```

### Pre-computed FFT Results

For efficiency when creating multiple plots from the same data:

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Compute FFT once
freq, mag = osc.fft(trace)

# Create multiple plots without recomputing
osc.plot_spectrum(trace, fft_result=(freq, mag),
                 title="Full Spectrum", show=False, save_path="full.png")

osc.plot_spectrum(trace, fft_result=(freq, mag),
                 xlim=(1, 10), freq_unit="MHz",
                 title="Zoomed Spectrum", show=False, save_path="zoom.png")
```

## Integration with Oscura Workflow

### Complete Analysis Workflow

```python
import oscura as osc

# Load signal
trace = osc.load("signal.wfm")

# Perform measurements
freq = osc.frequency(trace)
rise = osc.rise_time(trace)
thd_val = osc.thd(trace)

print(f"Frequency: {freq:.2f} Hz")
print(f"Rise time: {rise:.2e} s")
print(f"THD: {thd_val:.1f} dB")

# Visualize
osc.plot_waveform(trace, title=f"Signal @ {freq/1e6:.2f} MHz")
osc.plot_fft(trace, title=f"FFT (THD: {thd_val:.1f} dB)")
```

### Filtering and Visualization

```python
import oscura as osc

# Load noisy signal
trace = osc.load("noisy_signal.wfm")

# Apply low-pass filter
filtered = osc.low_pass(trace, cutoff=10e6)

# Compare original and filtered
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

osc.plot_waveform(trace, ax=ax1, label="Original", color="gray", show=False)
osc.plot_waveform(filtered, ax=ax1, label="Filtered", color="blue", show=False)
ax1.legend()
ax1.set_title("Time Domain Comparison")

osc.plot_spectrum(trace, ax=ax2, color="gray", show=False)
osc.plot_spectrum(filtered, ax=ax2, color="blue", show=False)
ax2.set_title("Frequency Domain Comparison")

plt.tight_layout()
plt.savefig("filtering_comparison.png", dpi=300)
```

## Supported File Formats

When saving plots, the file format is automatically detected from the extension:

- **PNG** (`.png`) - Raster format, 300 DPI
- **PDF** (`.pdf`) - Vector format, scalable
- **SVG** (`.svg`) - Vector format, web-friendly
- **JPEG** (`.jpg`, `.jpeg`) - Compressed raster
- **TIFF** (`.tiff`, `.tif`) - High-quality raster

Example:

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Save in different formats
osc.plot_waveform(trace, show=False, save_path="waveform.png")
osc.plot_waveform(trace, show=False, save_path="waveform.pdf")
osc.plot_waveform(trace, show=False, save_path="waveform.svg")
```

## Window Functions

Both `plot_spectrum()` and `plot_fft()` support various window functions via the `window` parameter:

- `"hann"` (default) - Good general-purpose window
- `"hamming"` - Similar to Hann, slightly different sidelobe characteristics
- `"blackman"` - Excellent sidelobe rejection
- `"bartlett"` - Triangular window
- `"rectangular"` - No windowing (spectral leakage)
- `"kaiser"` - Adjustable sidelobe characteristics

Example:

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Compare different windows
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, window in zip(axes.flat, ["hann", "blackman", "hamming", "kaiser"]):
    osc.plot_fft(trace, ax=ax, window=window,
                title=f"{window.capitalize()} Window", show=False)

plt.tight_layout()
plt.savefig("window_comparison.png", dpi=300)
```

## Matplotlib Backend Configuration

Oscura automatically handles matplotlib backends, but you can configure them manually:

### GUI Environments

```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc.

import oscura as osc
trace = osc.load("signal.wfm")
osc.plot_waveform(trace)  # Opens interactive window
```

### Non-GUI Environments (Servers, Docker, etc.)

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import oscura as osc
trace = osc.load("signal.wfm")
osc.plot_waveform(trace, show=False, save_path="output.png")
```

### Jupyter Notebooks

```python
%matplotlib inline  # Or 'notebook' for interactive

import oscura as osc
trace = osc.load("signal.wfm")
osc.plot_waveform(trace)  # Displays inline
```

## Troubleshooting

### Plot Not Displaying

If plots don't display:

1. Check matplotlib backend: `import matplotlib; print(matplotlib.get_backend())`
2. Ensure `show=True` (default)
3. Try calling `plt.show()` manually
4. Check if running in non-GUI environment

### Memory Issues with Large Files

For very large traces:

```python
import oscura as osc

# Use decimation for display
from oscura.visualization.optimization import decimate_for_display

trace = osc.load("large_file.wfm")
decimated_data = decimate_for_display(trace.data, max_points=10000)

# Create temporary trace for visualization
import numpy as np
viz_trace = osc.WaveformTrace(
    data=decimated_data,
    metadata=trace.metadata
)

osc.plot_waveform(viz_trace)
```

### Font/Text Issues

If fonts appear incorrect:

```python
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

import oscura as osc
# ... rest of code
```

## See Also

- [Loader API](loader.md) - Loading waveform data
- [Analysis API](analysis.md) - Signal analysis functions
- [Component Analysis API](component-analysis.md) - TDR, capacitance, inductance measurements
- [Export API](export.md) - Exporting data
- [Reporting API](reporting.md) - Report generation with charts
- [matplotlib Documentation](https://matplotlib.org/)
