"""Multi-Channel Waveform Loading

Demonstrates loading and handling multi-channel waveform data:
- load_all_channels() for simultaneous channel loading
- Channel selection and indexing
- Multi-trace synchronized analysis
- Channel alignment and timing validation
- Cross-channel correlation and relationships

IEEE Standards: IEEE 181-2011 (Waveform and Vector Measurements)
Related Demos:
- 01_data_loading/01_oscilloscopes.py
- 01_data_loading/02_logic_analyzers.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration shows:
1. How to use load_all_channels() for efficient multi-channel loading
2. How to select and access individual channels by name or index
3. How to perform synchronized analysis across multiple channels
4. How to validate channel alignment and timing synchronization
5. How to compute cross-channel relationships (correlation, phase)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    format_table,
    generate_sine_wave,
    generate_square_wave,
    validate_approximately,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class MultiChannelDemo(BaseDemo):
    """Demonstrate multi-channel waveform loading and analysis."""

    def __init__(self) -> None:
        """Initialize multi-channel demonstration."""
        super().__init__(
            name="multi_channel",
            description="Load and analyze multi-channel waveform data",
            capabilities=[
                "oscura.loaders.load_all_channels",
                "Channel selection by name/index",
                "Synchronized multi-channel analysis",
                "Channel alignment validation",
                "Cross-channel correlation",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "01_data_loading/01_oscilloscopes.py",
                "01_data_loading/02_logic_analyzers.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_multichan_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic multi-channel waveform data."""
        self.info("Creating synthetic multi-channel captures...")

        # 4-channel oscilloscope capture
        osc_4ch = self._create_4channel_oscilloscope()
        self.info("  ✓ 4-channel oscilloscope (synchronized, 1 MHz)")

        # 8-channel DAQ system
        daq_8ch = self._create_8channel_daq()
        self.info("  ✓ 8-channel DAQ (multi-sensor, 100 kHz)")

        # Mixed analog/digital capture
        mixed_signal = self._create_mixed_signal_capture()
        self.info("  ✓ Mixed-signal capture (2 analog + 4 digital)")

        # Phase-shifted channels
        phase_shifted = self._create_phase_shifted_channels()
        self.info("  ✓ Phase-shifted channels (3-phase power)")

        return {
            "osc_4ch": osc_4ch,
            "daq_8ch": daq_8ch,
            "mixed_signal": mixed_signal,
            "phase_shifted": phase_shifted,
        }

    def _create_4channel_oscilloscope(self) -> dict[str, WaveformTrace]:
        """Create synthetic 4-channel oscilloscope capture."""
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1 ms

        channels = {}

        # CH1: 1 kHz sine wave
        ch1 = generate_sine_wave(
            frequency=1e3, amplitude=1.0, duration=duration, sample_rate=sample_rate
        )
        ch1.metadata.channel_name = "CH1"
        ch1.metadata.vertical_scale = 0.5  # 0.5 V/div
        channels["ch1"] = ch1

        # CH2: 2 kHz sine wave (2x frequency)
        ch2 = generate_sine_wave(
            frequency=2e3, amplitude=0.8, duration=duration, sample_rate=sample_rate
        )
        ch2.metadata.channel_name = "CH2"
        ch2.metadata.vertical_scale = 0.5
        channels["ch2"] = ch2

        # CH3: 1 kHz square wave (same frequency as CH1)
        ch3 = generate_square_wave(
            frequency=1e3, amplitude=1.2, duration=duration, sample_rate=sample_rate
        )
        ch3.metadata.channel_name = "CH3"
        ch3.metadata.vertical_scale = 0.5
        channels["ch3"] = ch3

        # CH4: 500 Hz sine wave
        ch4 = generate_sine_wave(
            frequency=500, amplitude=1.5, duration=duration, sample_rate=sample_rate
        )
        ch4.metadata.channel_name = "CH4"
        ch4.metadata.vertical_scale = 1.0  # 1.0 V/div
        channels["ch4"] = ch4

        return channels

    def _create_8channel_daq(self) -> dict[str, WaveformTrace]:
        """Create synthetic 8-channel DAQ system."""
        sample_rate = 100e3  # 100 kHz
        duration = 0.01  # 10 ms

        channels = {}

        # Simulate 8 sensors with different characteristics
        sensor_configs = [
            ("Temperature_1", 5.0, 25.0),  # 5 Hz, 25°C baseline
            ("Temperature_2", 5.0, 27.0),  # 5 Hz, 27°C baseline
            ("Pressure_1", 10.0, 101.3),  # 10 Hz, atmospheric pressure
            ("Pressure_2", 10.0, 102.1),  # 10 Hz, slightly elevated
            ("Vibration_X", 1000.0, 0.0),  # 1 kHz, no DC offset
            ("Vibration_Y", 1000.0, 0.0),  # 1 kHz, no DC offset
            ("Voltage_1", 60.0, 120.0),  # 60 Hz AC, 120V RMS baseline
            ("Voltage_2", 60.0, 115.0),  # 60 Hz AC, 115V RMS baseline
        ]

        for i, (name, freq, offset) in enumerate(sensor_configs):
            signal = generate_sine_wave(
                frequency=freq,
                amplitude=0.5 * (i + 1) / 8,  # Varying amplitudes
                duration=duration,
                sample_rate=sample_rate,
                offset=offset,
            )
            signal.metadata.channel_name = name
            channels[f"ch{i + 1}"] = signal

        return channels

    def _create_mixed_signal_capture(self) -> dict[str, Any]:
        """Create mixed analog/digital capture."""
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1 ms

        # Analog channels
        analog = {}

        # Analog CH1: 10 kHz sine
        a1 = generate_sine_wave(
            frequency=10e3, amplitude=2.0, duration=duration, sample_rate=sample_rate
        )
        a1.metadata.channel_name = "A1"
        analog["a1"] = a1

        # Analog CH2: 20 kHz sine
        a2 = generate_sine_wave(
            frequency=20e3, amplitude=1.5, duration=duration, sample_rate=sample_rate
        )
        a2.metadata.channel_name = "A2"
        analog["a2"] = a2

        # Digital channels (simulated as 0/1 waveforms)
        digital = {}
        num_samples = int(sample_rate * duration)
        t = np.arange(num_samples) / sample_rate

        # Digital D1: 1 kHz clock
        d1_data = (np.sin(2 * np.pi * 1e3 * t) > 0).astype(np.float64)
        d1 = WaveformTrace(
            data=d1_data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="D1"),
        )
        digital["d1"] = d1

        # Digital D2: 2 kHz clock
        d2_data = (np.sin(2 * np.pi * 2e3 * t) > 0).astype(np.float64)
        d2 = WaveformTrace(
            data=d2_data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="D2"),
        )
        digital["d2"] = d2

        # Digital D3: Data signal (slower toggle)
        d3_data = (np.sin(2 * np.pi * 100 * t) > 0).astype(np.float64)
        d3 = WaveformTrace(
            data=d3_data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="D3"),
        )
        digital["d3"] = d3

        # Digital D4: Chip select (active for middle portion)
        d4_data = np.zeros(num_samples)
        d4_data[num_samples // 3 : 2 * num_samples // 3] = 1.0
        d4 = WaveformTrace(
            data=d4_data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="D4"),
        )
        digital["d4"] = d4

        return {
            "analog": analog,
            "digital": digital,
            "sample_rate": sample_rate,
        }

    def _create_phase_shifted_channels(self) -> dict[str, WaveformTrace]:
        """Create 3-phase power signals with 120° phase shifts."""
        sample_rate = 10e3  # 10 kHz (sufficient for 60 Hz)
        duration = 0.1  # 100 ms (6 cycles of 60 Hz)
        frequency = 60.0  # 60 Hz AC power

        channels = {}

        # Phase A: 0°
        phase_a = generate_sine_wave(
            frequency=frequency,
            amplitude=170.0,  # 120V RMS = 170V peak
            duration=duration,
            sample_rate=sample_rate,
            phase=0.0,
        )
        phase_a.metadata.channel_name = "Phase_A"
        channels["phase_a"] = phase_a

        # Phase B: 120° (2π/3 radians)
        phase_b = generate_sine_wave(
            frequency=frequency,
            amplitude=170.0,
            duration=duration,
            sample_rate=sample_rate,
            phase=2 * np.pi / 3,
        )
        phase_b.metadata.channel_name = "Phase_B"
        channels["phase_b"] = phase_b

        # Phase C: 240° (4π/3 radians)
        phase_c = generate_sine_wave(
            frequency=frequency,
            amplitude=170.0,
            duration=duration,
            sample_rate=sample_rate,
            phase=4 * np.pi / 3,
        )
        phase_c.metadata.channel_name = "Phase_C"
        channels["phase_c"] = phase_c

        return channels

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the multi-channel demonstration."""
        results = {}

        self.section("Multi-Channel Waveform Loading")
        self.info("Modern oscilloscopes and DAQ systems capture multiple channels simultaneously.")
        self.info("Key concepts:")
        self.info("  • Synchronized sampling: All channels share the same timebase")
        self.info("  • Channel indexing: Access by name ('ch1') or number (0)")
        self.info("  • Cross-channel analysis: Correlation, phase, timing relationships")
        self.info("  • Mixed-signal: Analog + digital channels in one capture")
        self.info("")

        # 4-channel oscilloscope
        self.section("1. Four-Channel Oscilloscope Capture")
        results["osc_4ch"] = self._analyze_4channel_oscilloscope(data["osc_4ch"])

        # 8-channel DAQ
        self.section("2. Eight-Channel DAQ System")
        results["daq_8ch"] = self._analyze_8channel_daq(data["daq_8ch"])

        # Mixed-signal
        self.section("3. Mixed Analog/Digital Capture")
        results["mixed_signal"] = self._analyze_mixed_signal(data["mixed_signal"])

        # Phase analysis
        self.section("4. Three-Phase Power Analysis")
        results["phase_shifted"] = self._analyze_phase_shifted(data["phase_shifted"])

        # Cross-channel operations
        self.section("5. Cross-Channel Analysis")
        self._demonstrate_cross_channel_analysis(data["osc_4ch"])

        # Best practices
        self.section("Multi-Channel Best Practices")
        self._show_best_practices()

        return results

    def _analyze_4channel_oscilloscope(self, channels: dict[str, WaveformTrace]) -> dict[str, Any]:
        """Analyze 4-channel oscilloscope capture."""
        self.subsection("Channel Overview")

        self.result("Total Channels", len(channels))
        self.result(
            "Sample Rate",
            f"{next(iter(channels.values())).metadata.sample_rate / 1e6:.1f}",
            "MHz",
        )
        self.info("")

        # Build channel table
        channel_info = []
        for name, trace in channels.items():
            num_samples = len(trace.data)
            duration = num_samples / trace.metadata.sample_rate
            v_rms = float(np.sqrt(np.mean(trace.data**2)))
            v_peak = float(np.max(np.abs(trace.data)))

            # Estimate frequency
            fft = np.fft.rfft(trace.data)
            freqs = np.fft.rfftfreq(num_samples, 1 / trace.metadata.sample_rate)
            peak_idx = np.argmax(np.abs(fft[1:])) + 1
            est_freq = freqs[peak_idx]

            channel_info.append(
                [
                    trace.metadata.channel_name or name,
                    f"{num_samples}",
                    f"{duration * 1e3:.2f} ms",
                    f"{v_rms:.3f} V",
                    f"{v_peak:.3f} V",
                    f"{est_freq / 1e3:.2f} kHz",
                ]
            )

        headers = ["Channel", "Samples", "Duration", "RMS", "Peak", "Frequency"]
        self.info(format_table(channel_info, headers))
        self.info("")

        return {
            "num_channels": len(channels),
            "channel_names": [ch.metadata.channel_name for ch in channels.values()],
        }

    def _analyze_8channel_daq(self, channels: dict[str, WaveformTrace]) -> dict[str, Any]:
        """Analyze 8-channel DAQ system."""
        self.subsection("Multi-Sensor DAQ Overview")

        self.result("Total Channels", len(channels))
        self.result(
            "Sample Rate",
            f"{next(iter(channels.values())).metadata.sample_rate / 1e3:.1f}",
            "kHz",
        )
        self.info("")

        # Group channels by sensor type
        sensor_groups = {
            "Temperature": [],
            "Pressure": [],
            "Vibration": [],
            "Voltage": [],
        }

        for name, trace in channels.items():
            sensor_type = trace.metadata.channel_name.split("_")[0]
            if sensor_type in sensor_groups:
                sensor_groups[sensor_type].append((name, trace))

        # Analyze each sensor group
        self.subsection("Sensor Group Analysis")
        for sensor_type, group_channels in sensor_groups.items():
            if not group_channels:
                continue

            self.info(f"\n{sensor_type} Sensors ({len(group_channels)} channels):")
            for _name, trace in group_channels:
                mean = float(np.mean(trace.data))
                std = float(np.std(trace.data))
                self.info(f"  {trace.metadata.channel_name}: mean={mean:.2f}, std={std:.3f}")

        self.info("")

        return {
            "num_channels": len(channels),
            "sensor_groups": {k: len(v) for k, v in sensor_groups.items()},
        }

    def _analyze_mixed_signal(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze mixed analog/digital capture."""
        analog = data["analog"]
        digital = data["digital"]

        self.subsection("Mixed-Signal Capture Overview")
        self.result("Analog Channels", len(analog))
        self.result("Digital Channels", len(digital))
        self.result("Sample Rate", f"{data['sample_rate'] / 1e6:.1f}", "MHz")
        self.info("")

        # Analyze analog channels
        self.subsection("Analog Channels")
        for trace in analog.values():
            v_rms = float(np.sqrt(np.mean(trace.data**2)))
            self.result(f"  {trace.metadata.channel_name}", f"RMS={v_rms:.3f} V")
        self.info("")

        # Analyze digital channels
        self.subsection("Digital Channels")
        for trace in digital.values():
            # Count transitions
            transitions = np.sum(np.abs(np.diff(trace.data)) > 0.5)
            duty_cycle = float(np.mean(trace.data)) * 100

            self.result(
                f"  {trace.metadata.channel_name}",
                f"Transitions={transitions}, Duty={duty_cycle:.1f}%",
            )
        self.info("")

        return {
            "num_analog": len(analog),
            "num_digital": len(digital),
            "sample_rate": data["sample_rate"],
        }

    def _analyze_phase_shifted(self, channels: dict[str, WaveformTrace]) -> dict[str, Any]:
        """Analyze 3-phase power signals."""
        self.subsection("Three-Phase Power Analysis")

        # Extract phases
        phase_a = channels["phase_a"]
        phase_b = channels["phase_b"]
        phase_c = channels["phase_c"]

        # Compute RMS values
        rms_a = float(np.sqrt(np.mean(phase_a.data**2)))
        rms_b = float(np.sqrt(np.mean(phase_b.data**2)))
        rms_c = float(np.sqrt(np.mean(phase_c.data**2)))

        self.result("Phase A RMS", f"{rms_a:.1f}", "V")
        self.result("Phase B RMS", f"{rms_b:.1f}", "V")
        self.result("Phase C RMS", f"{rms_c:.1f}", "V")
        self.info("")

        # Compute phase relationships
        self.subsection("Phase Relationships")

        # Cross-correlation to find phase shifts
        corr_ab = np.correlate(phase_a.data, phase_b.data, mode="same")
        corr_ac = np.correlate(phase_a.data, phase_c.data, mode="same")
        corr_bc = np.correlate(phase_b.data, phase_c.data, mode="same")

        # Find peak correlation lag
        center = len(corr_ab) // 2
        lag_ab = np.argmax(corr_ab) - center
        lag_ac = np.argmax(corr_ac) - center
        lag_bc = np.argmax(corr_bc) - center

        # Convert lag to phase (degrees)
        sample_rate = phase_a.metadata.sample_rate
        samples_per_cycle = sample_rate / 60.0  # 60 Hz
        phase_ab = (lag_ab / samples_per_cycle) * 360
        phase_ac = (lag_ac / samples_per_cycle) * 360
        phase_bc = (lag_bc / samples_per_cycle) * 360

        self.result("Phase A-B", f"{phase_ab:.1f}", "degrees")
        self.result("Phase A-C", f"{phase_ac:.1f}", "degrees")
        self.result("Phase B-C", f"{phase_bc:.1f}", "degrees")
        self.info("")

        # Verify balanced 3-phase (should be ~120° apart)
        expected_phase = 120.0
        phase_error_ab = abs(abs(phase_ab) - expected_phase)
        phase_error_bc = abs(abs(phase_bc) - expected_phase)

        if phase_error_ab < 5.0 and phase_error_bc < 5.0:
            self.success("✓ Balanced 3-phase system detected")
        else:
            self.warning("⚠ Phase imbalance detected")

        self.info("")

        return {
            "rms_values": [rms_a, rms_b, rms_c],
            "phase_shifts": [phase_ab, phase_ac, phase_bc],
        }

    def _demonstrate_cross_channel_analysis(self, channels: dict[str, WaveformTrace]) -> None:
        """Demonstrate cross-channel correlation and analysis."""
        self.subsection("Cross-Channel Correlation")

        # Use CH1 and CH3 (same frequency, different waveforms)
        ch1 = channels["ch1"]
        ch3 = channels["ch3"]

        # Compute cross-correlation
        correlation = np.correlate(ch1.data, ch3.data, mode="same")
        max_corr = float(np.max(correlation))
        normalized_corr = max_corr / (len(ch1.data) * np.std(ch1.data) * np.std(ch3.data))

        self.result("CH1 ↔ CH3 Correlation", f"{normalized_corr:.3f}")
        self.info("")

        # Compute coherence (frequency-domain correlation)
        self.subsection("Coherence Analysis")

        fft1 = np.fft.rfft(ch1.data)
        fft3 = np.fft.rfft(ch3.data)

        # Cross-spectral density
        cross_spec = fft1 * np.conj(fft3)

        # Power spectral densities
        psd1 = np.abs(fft1) ** 2
        psd3 = np.abs(fft3) ** 2

        # Coherence
        coherence = np.abs(cross_spec) ** 2 / (psd1 * psd3 + 1e-10)

        # Find peak coherence
        freqs = np.fft.rfftfreq(len(ch1.data), 1 / ch1.metadata.sample_rate)
        peak_coh_idx = np.argmax(coherence[1:100]) + 1  # First 100 points
        peak_coh_freq = freqs[peak_coh_idx]
        peak_coh_value = float(coherence[peak_coh_idx])

        self.result("Peak Coherence Frequency", f"{peak_coh_freq / 1e3:.2f}", "kHz")
        self.result("Peak Coherence Value", f"{peak_coh_value:.3f}")
        self.info("")

        self.info("Coherence interpretation:")
        self.info("  • 0.0: Completely uncorrelated")
        self.info("  • 0.5: Moderately correlated")
        self.info("  • 1.0: Perfectly correlated")
        self.info("")

    def _show_best_practices(self) -> None:
        """Show best practices for multi-channel analysis."""
        self.info("""
Best practices for multi-channel waveform analysis:

1. LOADING MULTI-CHANNEL FILES
   ```python
   from oscura.loaders import load_all_channels

   # Load all channels at once (more efficient)
   channels = load_all_channels("capture.wfm")

   # Access by name
   ch1 = channels["ch1"]
   ch2 = channels["ch2"]

   # Or iterate over all channels
   for name, trace in channels.items():
       print(f"{name}: {len(trace.data)} samples")
   ```

2. CHANNEL SYNCHRONIZATION
   ✓ Verify all channels have same sample rate
   ✓ Check all channels have same number of samples
   ✓ Validate trigger alignment (look for common edge)
   ✓ Correct for skew if channels have different delays

3. CROSS-CHANNEL ANALYSIS
   • Correlation: Measure similarity between channels
   • Coherence: Frequency-domain correlation
   • Phase: Timing relationship (use cross-correlation)
   • Transfer function: Output/input relationship (H(f) = Y(f)/X(f))

4. NAMING CONVENTIONS
   ✓ Use descriptive names: "Phase_A" not "CH1"
   ✓ Include units in metadata: "Temperature_C"
   ✓ Group related channels: "Accel_X", "Accel_Y", "Accel_Z"
   ✓ Document channel assignments

5. MEMORY MANAGEMENT
   • For N channels with M samples each: Memory = N x M x 8 bytes
   • 4 channels x 10M samples = 320 MB
   • Consider chunked loading for very long captures
   • Use channel selection to load only needed channels

6. VALIDATION CHECKS
   ```python
   # Verify synchronization
   sample_rates = [ch.metadata.sample_rate for ch in channels.values()]
   assert len(set(sample_rates)) == 1, "Sample rate mismatch"

   # Verify length
   lengths = [len(ch.data) for ch in channels.values()]
   assert len(set(lengths)) == 1, "Channel length mismatch"

   # Check for clipping
   for name, ch in channels.items():
       if np.any(np.abs(ch.data) > 0.99 * ch.metadata.vertical_scale * 8):
           print(f"Warning: {name} may be clipping")
   ```

7. MIXED ANALOG/DIGITAL
   • Digital channels: Use threshold (0.5) to convert to boolean
   • Timing analysis: Count transitions, measure pulse widths
   • Protocol decoding: Use digital channels with protocol decoders
   • Correlation: Compare analog triggers with digital events
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate multi-channel loading results."""
        self.info("Validating multi-channel operations...")

        all_valid = True

        # Validate 4-channel oscilloscope
        if "osc_4ch" in results:
            osc = results["osc_4ch"]
            if osc["num_channels"] != 4:
                self.error(f"Expected 4 channels, got {osc['num_channels']}")
                all_valid = False

            expected_names = ["CH1", "CH2", "CH3", "CH4"]
            if osc["channel_names"] != expected_names:
                self.error(f"Channel name mismatch: {osc['channel_names']}")
                all_valid = False

        # Validate 8-channel DAQ
        if "daq_8ch" in results:
            daq = results["daq_8ch"]
            if daq["num_channels"] != 8:
                self.error(f"Expected 8 channels, got {daq['num_channels']}")
                all_valid = False

            total_sensors = sum(daq["sensor_groups"].values())
            if total_sensors != 8:
                self.error(f"Sensor group count mismatch: {total_sensors}")
                all_valid = False

        # Validate mixed-signal
        if "mixed_signal" in results:
            mixed = results["mixed_signal"]
            if mixed["num_analog"] != 2:
                self.error(f"Expected 2 analog channels, got {mixed['num_analog']}")
                all_valid = False

            if mixed["num_digital"] != 4:
                self.error(f"Expected 4 digital channels, got {mixed['num_digital']}")
                all_valid = False

        # Validate phase-shifted signals
        if "phase_shifted" in results:
            phase = results["phase_shifted"]

            # Check RMS values are approximately equal (balanced system)
            rms_values = phase["rms_values"]
            rms_mean = np.mean(rms_values)
            for i, rms in enumerate(rms_values):
                if not validate_approximately(rms, rms_mean, tolerance=0.05, name=f"Phase {i} RMS"):
                    all_valid = False

            # Check phase shifts are approximately 120° apart
            phase_shifts = phase["phase_shifts"]
            # A-B should be ~120°
            if not validate_approximately(
                abs(phase_shifts[0]), 120.0, tolerance=5.0, name="Phase A-B shift"
            ):
                all_valid = False

        if all_valid:
            self.success("All multi-channel validations passed!")
            self.info("""
Next steps for multi-channel analysis:

1. LOAD ALL CHANNELS
   from oscura.loaders import load_all_channels
   channels = load_all_channels("capture.wfm")

2. SYNCHRONIZED ANALYSIS
   # Verify synchronization
   assert all(len(ch.data) == len(channels['ch1'].data)
              for ch in channels.values())

   # Process all channels
   for name, trace in channels.items():
       analyze(trace)

3. CROSS-CHANNEL CORRELATION
   corr = np.correlate(ch1.data, ch2.data, mode='same')
   max_corr_idx = np.argmax(corr)
   time_delay = (max_corr_idx - len(ch1.data)//2) / sample_rate

4. PHASE ANALYSIS
   fft1 = np.fft.rfft(ch1.data)
   fft2 = np.fft.rfft(ch2.data)
   phase_diff = np.angle(fft2 / fft1)  # Radians
            """)
        else:
            self.error("Some multi-channel validations failed!")

        return all_valid


if __name__ == "__main__":
    demo = MultiChannelDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
