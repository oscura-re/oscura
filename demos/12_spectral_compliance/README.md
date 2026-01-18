# Spectral and Compliance Demos

Demonstrates **comprehensive spectral analysis** and **IEEE 1241-2010 compliance validation** using Oscura's advanced frequency-domain analysis capabilities.

---

## Files in This Demo

1. **`comprehensive_spectral_demo.py`** ⭐ **COMPREHENSIVE**
   - FFT with proper windowing
   - Power Spectral Density (Welch, Bartlett, Periodogram)
   - THD, SNR, SINAD, ENOB, SFDR measurements
   - IEEE 1241-2010 compliance validation
   - Harmonic analysis
   - Spectrogram generation

---

## Quick Start

### 1. Audio Signal Analysis

```bash
# Using synthetic audio signal
python demos/06_spectral_compliance/comprehensive_spectral_demo.py --type audio

# Analyze audio file
python demos/06_spectral_compliance/comprehensive_spectral_demo.py \
    --file audio_recording.wav \
    --type audio \
    --window hann
```

### 2. ADC Characterization

```bash
# Characterize ADC performance (IEEE 1241-2010)
python demos/06_spectral_compliance/comprehensive_spectral_demo.py \
    --file adc_capture.wfm \
    --type adc \
    --all-analysis
```

### 3. Power Quality Analysis

```bash
# Analyze power line harmonics
python demos/06_spectral_compliance/comprehensive_spectral_demo.py \
    --file power_60hz.wfm \
    --type power \
    --verbose
```

---

## Analysis Capabilities

### ✅ FFT Analysis

**Methods**:

- Single FFT with configurable windowing
- Coherent sampling optimization
- Sub-sample frequency resolution
- Cache performance monitoring

**Windows**: Hann, Hamming, Blackman, Kaiser, Rectangular

### ✅ Power Spectral Density

**Methods**:

- **Welch**: Overlapped segment averaging (default)
- **Periodogram**: Single-segment estimate
- **Bartlett**: Non-overlapping segment averaging

**Scaling**: Density (V²/Hz) or Spectrum (V²)

### ✅ Harmonic Distortion Analysis

**Metrics**:

- **THD** (Total Harmonic Distortion): Ratio of harmonic power to fundamental
- Configurable number of harmonics (default: 10)
- dB or percentage output

**Applications**:

- Audio amplifier characterization
- Power supply ripple analysis
- ADC linearity testing

### ✅ Signal Quality Metrics (IEEE 1241-2010)

**SNR** (Signal-to-Noise Ratio):

- Excludes fundamental and harmonics
- Measures noise floor
- Quality grading: >90 dB (Excellent), >72 dB (Good), >54 dB (Fair)

**SINAD** (Signal-to-Noise and Distortion):

- Combined noise + distortion measurement
- Used for ENOB calculation

**ENOB** (Effective Number of Bits):

- ENOB = (SINAD - 1.76) / 6.02
- ADC resolution characterization

**SFDR** (Spurious-Free Dynamic Range):

- Ratio of fundamental to largest spur
- Quality grading: >80 dBc (Excellent), >60 dBc (Good)

### ✅ Time-Frequency Analysis

**Spectrogram**:

- Short-Time Fourier Transform (STFT)
- Configurable time/frequency resolution
- Ideal for non-stationary signals

---

## IEEE 1241-2010 Compliance

The demo validates compliance with **IEEE Standard 1241-2010** for ADC testing:

### Validation Checks

1. **Oversampling Ratio**: Sample rate should be ≥5× fundamental frequency
2. **Coherent Sampling**: Integer number of cycles in capture window
3. **Dynamic Range**: SFDR should be >60 dBc for quality measurements
4. **Window Function**: Proper windowing to minimize spectral leakage

### Compliance Report

Generates automatic PASS/FAIL report with:

- Nyquist frequency analysis
- Coherent sampling verification
- Dynamic range assessment
- Violation summary

---

## Python API Usage

```python
import oscura as osc
from oscura.analyzers.waveform.spectral import (
    fft, psd, thd, snr, sinad, enob, sfdr, spectrogram
)

# Load signal
signal = osc.load("audio.wav")

# FFT with Hann window
freq, mag_db = fft(signal, window="hann", detrend="mean")

# Power Spectral Density (Welch's method)
freq_psd, psd_db = psd(signal, window="hann", nperseg=1024, noverlap=512)

# Harmonic distortion
thd_db = thd(signal, n_harmonics=10, return_db=True)
thd_pct = thd(signal, n_harmonics=10, return_db=False)
print(f"THD: {thd_db:.1f} dB ({thd_pct:.3f}%)")

# Signal quality metrics
snr_db = snr(signal, n_harmonics=10)
sinad_db = sinad(signal)
enob_bits = enob(signal)
sfdr_db = sfdr(signal)

print(f"SNR: {snr_db:.1f} dB")
print(f"SINAD: {sinad_db:.1f} dB")
print(f"ENOB: {enob_bits:.2f} bits")
print(f"SFDR: {sfdr_db:.1f} dBc")

# Spectrogram for time-frequency analysis
t, f, Sxx = spectrogram(signal, window="hann", nperseg=256)
```

---

## Use Cases

### Audio Engineering

- Amplifier THD+N characterization
- Speaker distortion analysis
- Microphone frequency response
- Audio codec quality testing

### ADC Characterization

- Effective resolution (ENOB)
- Linearity testing (THD, SFDR)
- Noise floor measurement (SNR)
- IEEE 1241-2010 compliance

### Power Electronics

- Power supply ripple analysis
- Harmonic distortion in AC systems
- Switching noise characterization
- Power quality monitoring (IEEE 1459)

### RF/Communications

- Spectral mask compliance
- Spurious emission detection
- Carrier-to-noise ratio
- Baseband signal analysis

---

## Window Functions

**Hann** (default):

- Good frequency resolution
- Moderate sidelobe suppression
- Best for general-purpose analysis

**Hamming**:

- Better sidelobe suppression than Hann
- Slightly worse frequency resolution

**Blackman**:

- Excellent sidelobe suppression
- Wider main lobe (lower frequency resolution)

**Kaiser**:

- Adjustable sidelobe/resolution tradeoff
- Best for customized applications

**Rectangular**:

- Best frequency resolution
- Poor sidelobe suppression (spectral leakage)
- Only for perfectly coherent signals

---

## Output Interpretation

### THD Values

- **<0.01%**: Excellent (Hi-Fi audio, precision instrumentation)
- **<0.1%**: Good (Professional audio equipment)
- **<1%**: Fair (Consumer audio)
- **>1%**: Poor (High distortion, investigate)

### SNR Values

- **>90 dB**: Excellent (16+ bit ADC)
- **>72 dB**: Good (12-16 bit ADC)
- **>54 dB**: Fair (8-12 bit ADC)
- **<54 dB**: Poor (<8 bit equivalent)

### ENOB Values

- **>12 bits**: High resolution ADC
- **8-12 bits**: Medium resolution
- **6-8 bits**: Low resolution
- **<6 bits**: Very low resolution

### SFDR Values

- **>80 dBc**: Excellent dynamic range
- **>60 dBc**: Good (suitable for most applications)
- **>40 dBc**: Fair (may have spurious issues)
- **<40 dBc**: Poor (investigate spurs)

---

## Common Issues

### Issue: "Insufficient oversampling"

**Solution**: Increase sample rate to at least 5× fundamental frequency

### Issue: "Non-coherent sampling"

**Solution**: Adjust capture time or use prime frequencies for coherent sampling

### Issue: "Could not compute THD/SNR/SINAD"

**Solution**: Check signal level, ensure fundamental is present, reduce noise

### Issue: "Low SFDR detected"

**Solution**: Investigate spurious components, check for aliasing or harmonics

---

## Performance Notes

- FFT results are cached for repeated analysis
- Use `clear_fft_cache()` to free memory
- Welch's method trades frequency resolution for variance reduction
- Coherent sampling eliminates spectral leakage

---

## Related Documentation

- **Main demos**: `demos/README.md`
- **Mixed-signal**: `demos/07_mixed_signal/`
- **Examples**: `examples/03_spectral_analysis/`
- **IEEE 1241-2010**: Standard for ADC testing terminology

---

**Last Updated**: 2026-01-15
**Status**: Production-ready
