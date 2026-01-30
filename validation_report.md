# Mathematical Validation Report

**Total Tests**: 18
**Passed**: 13
**Failed**: 5
**Pass Rate**: 72.2%

## Validation Results

### Amplitude: basic/sine_1khz.wfm - Vpp (V)

- **Status**: ✓ PASS
- **Formula**: `Vpp = max(signal) - min(signal) = 2 × amplitude`
- **Expected**: 2.000000e+00
- **Actual**: 1.960000e+00
- **Error**: 2.000%
- **Notes**: 1 kHz sine, amplitude 1.0V → Vpp = 2.0V

### Amplitude: basic/square_5khz.wfm - Vpp (V)

- **Status**: ✓ PASS
- **Formula**: `Vpp = max(signal) - min(signal) = 2 × amplitude`
- **Expected**: 4.000000e+00
- **Actual**: 3.869790e+00
- **Error**: 3.255%
- **Notes**: 5 kHz square, amplitude 2.0V → Vpp = 4.0V

### Amplitude: basic/triangle_2khz.wfm - Vpp (V)

- **Status**: ✓ PASS
- **Formula**: `Vpp = max(signal) - min(signal) = 2 × amplitude`
- **Expected**: 3.000000e+00
- **Actual**: 2.880473e+00
- **Error**: 3.984%
- **Notes**: 2 kHz triangle, amplitude 1.5V → Vpp = 3.0V

### Frequency Detection: basic/sine_1khz.wfm - Frequency (Hz)

- **Status**: ✗ FAIL
- **Formula**: `f_peak = argmax(|FFT(signal)|)`
- **Expected**: 1.000000e+03
- **Actual**: nan
- **Error**: nan%
- **Notes**: Signal: basic/sine_1khz.wfm

### Frequency Detection: basic/square_5khz.wfm - Frequency (Hz)

- **Status**: ✓ PASS
- **Formula**: `f_peak = argmax(|FFT(signal)|)`
- **Expected**: 5.000000e+03
- **Actual**: 5.000000e+03
- **Error**: 0.000%
- **Notes**: Signal: basic/square_5khz.wfm

### Frequency Detection: basic/triangle_2khz.wfm - Frequency (Hz)

- **Status**: ✗ FAIL
- **Formula**: `f_peak = argmax(|FFT(signal)|)`
- **Expected**: 2.000000e+03
- **Actual**: nan
- **Error**: nan%
- **Notes**: Signal: basic/triangle_2khz.wfm

### Frequency Detection: frequencies/audio_freq_440hz.wfm - Frequency (Hz)

- **Status**: ✓ PASS
- **Formula**: `f_peak = argmax(|FFT(signal)|)`
- **Expected**: 4.400000e+02
- **Actual**: 4.399996e+02
- **Error**: 0.000%
- **Notes**: Signal: frequencies/audio_freq_440hz.wfm

### Frequency Detection: edge_cases/high_frequency_100khz.wfm - Frequency (Hz)

- **Status**: ✓ PASS
- **Formula**: `f_peak = argmax(|FFT(signal)|)`
- **Expected**: 1.000000e+05
- **Actual**: 1.000000e+05
- **Error**: 0.000%
- **Notes**: Signal: edge_cases/high_frequency_100khz.wfm

### RMS: Sine Wave (AC-coupled) - RMS_AC (V)

- **Status**: ✓ PASS
- **Formula**: `RMS_sine = A/√2 = 1.000000/√2 = 0.707107`
- **Expected**: 7.071068e-01
- **Actual**: 7.070931e-01
- **Error**: 0.002%
- **Notes**: 1 kHz sine, measured Vpp=2.000V, DC removed for AC RMS

### RMS: Square Wave (AC-coupled) - RMS_AC (V)

- **Status**: ✗ FAIL
- **Formula**: `RMS_square ≈ A = 2.359628`
- **Expected**: 2.359628e+00
- **Actual**: 1.979613e+00
- **Error**: 16.105%
- **Notes**: 5 kHz square, measured Vpp=4.719V (Fourier series)

### RMS: DC Signal - RMS (V)

- **Status**: ✓ PASS
- **Formula**: `RMS_dc = DC_level = 2.5 V`
- **Expected**: 2.500000e+00
- **Actual**: 2.500000e+00
- **Error**: 0.000%
- **Notes**: Constant DC signal at 2.5 V

### Duty Cycle: advanced/pulse_train_10pct.wfm - Duty Cycle (ratio)

- **Status**: ✗ FAIL
- **Formula**: `duty_cycle = 0.1 (from generator config)`
- **Expected**: 1.000000e-01
- **Actual**: nan
- **Error**: nan%
- **Notes**: 10% duty cycle pulse train

### Duty Cycle: advanced/pulse_train_90pct.wfm - Duty Cycle (ratio)

- **Status**: ✗ FAIL
- **Formula**: `duty_cycle = 0.9 (from generator config)`
- **Expected**: 9.000000e-01
- **Actual**: nan
- **Error**: nan%
- **Notes**: 90% duty cycle pulse train

### SNR: edge_cases/sine_with_noise_snr30.wfm - SNR (dB)

- **Status**: ✓ PASS
- **Formula**: `SNR_dB = 10 * log10(signal_power / noise_power)`
- **Expected**: 3.000000e+01
- **Actual**: 2.925859e+01
- **Error**: 2.471%
- **Notes**: 1 kHz sine + noise, SNR = 30 dB

### Statistics: Mean - Mean (V)

- **Status**: ✓ PASS
- **Formula**: `mean = sum(x) / N`
- **Expected**: -1.000015e+00
- **Actual**: -1.000015e+00
- **Error**: 0.000%
- **Notes**: 1 kHz sine wave, compared to numpy.mean

### Statistics: Standard Deviation - Std Dev (V)

- **Status**: ✓ PASS
- **Formula**: `std = √(sum((x - mean)²) / N)`
- **Expected**: 7.070931e-01
- **Actual**: 7.070931e-01
- **Error**: 0.000%
- **Notes**: 1 kHz sine wave, compared to numpy.std

### Statistics: DC Mean - Mean (V)

- **Status**: ✓ PASS
- **Formula**: `mean_dc = DC_level = 2.5 V`
- **Expected**: 2.500000e+00
- **Actual**: 2.500000e+00
- **Error**: 0.000%
- **Notes**: Constant DC signal - mean should equal DC level

### Statistics: DC Std Dev - Std Dev (V)

- **Status**: ✓ PASS
- **Formula**: `std_dc ≈ 0 (constant signal)`
- **Expected**: 0.000000e+00
- **Actual**: 0.000000e+00
- **Error**: 0.000%
- **Notes**: DC signal - std dev should be ~0
