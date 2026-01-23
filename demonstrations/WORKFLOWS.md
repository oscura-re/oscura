# Oscura Demonstrations: Complete Workflows

End-to-end workflows for common hardware reverse engineering and signal analysis tasks. Each workflow combines multiple demonstrations to solve a real-world problem.

## Workflow 1: Reverse Engineer Unknown Serial Protocol

**Goal:** Identify protocol type, baud rate, frame format, and data encoding from raw captured signals.

**Time Budget:** 10-15 minutes

**Steps:**

1. **Load captured waveform data**
   - Demonstration: `01_data_loading/01_oscilloscopes.py`
   - Load oscilloscope captures (Tektronix, Rigol, LeCroy formats)
   - Extract metadata (sample rate, coupling, vertical scale)

2. **Perform initial signal characterization**
   - Demonstration: `14_exploratory/01_unknown_signals.py`
   - Detect analog vs. digital signals
   - Identify if signal is periodic or aperiodic
   - Measure basic amplitude and frequency

3. **Classify signal and identify protocol family**
   - Demonstration: `06_reverse_engineering/08_data_classification.py`
   - Use machine learning to classify signal type
   - Predict protocol family (UART, SPI, I2C, CAN, LIN, etc.)

4. **Detect protocol parameters automatically**
   - Demonstration: `03_protocol_decoding/06_auto_detection.py`
   - Auto-detect baud rate (UART), clock frequency (SPI), or bit rate
   - Measure timing parameters
   - Validate detection against known protocols

5. **Validate by attempting to decode**
   - Demonstration: `03_protocol_decoding/01_serial_comprehensive.py`
   - Try UART decoding with detected parameters
   - Try SPI or I2C if appropriate
   - Check if decoded data makes sense

6. **Extract and analyze message patterns**
   - Demonstration: `06_reverse_engineering/05_pattern_discovery.py`
   - Identify message boundaries and frames
   - Find repeated patterns and command structures
   - Extract payload data

7. **Infer message format and fields**
   - Demonstration: `06_reverse_engineering/04_field_inference.py`
   - Detect field boundaries in messages
   - Classify field types (address, data, checksum, etc.)
   - Build protocol specification

**Expected Output:**
- Protocol type and parameters (baud rate, bit rate, clock frequency)
- Message format specification with field sizes
- Example decoded messages
- Confidence scores for each analysis step

**Success Criteria:**
- Successfully decode at least 5 consecutive messages
- Field boundaries aligned with observed protocol structure
- Detected parameters match data rate characteristics
- No checksum errors in decoded frames

## Workflow 2: Reverse Engineer Automotive Diagnostic Bus

**Goal:** Capture vehicle diagnostics, decode protocol, identify fault codes, and generate diagnostic report.

**Time Budget:** 20-30 minutes

**Steps:**

1. **Capture CAN/LIN/FlexRay bus traffic**
   - Demonstration: `01_data_loading/03_automotive_formats.py`
   - Load vehicle bus capture in CANDUMP or DBC format
   - Extract metadata (CAN ID, DLC, baudrate)
   - Validate frame structure

2. **Decode automotive bus protocol**
   - Demonstration: `03_protocol_decoding/02_automotive_protocols.py`
   - Decode CAN frames (11-bit and 29-bit IDs)
   - Identify LIN frames if applicable
   - Parse FlexRay if multi-protocol vehicle

3. **Decode OBD-II/UDS protocol layer**
   - Demonstration: `05_domain_specific/01_automotive_diagnostics.py`
   - Decode OBD-II Mode 01-09 responses
   - Decode UDS (ISO 14229) services
   - Extract PID values and live data

4. **Extract and classify diagnostic trouble codes**
   - Demonstration: `16_complete_workflows/02_automotive_diagnostics.py`
   - Identify DTCs from diagnostic messages
   - Classify severity (Critical/Major/Minor)
   - Look up fault descriptions in DTC database

5. **Analyze fault patterns and correlation**
   - Demonstration: `06_reverse_engineering/07_entropy_analysis.py`
   - Identify if faults are persistent or intermittent
   - Correlate DTCs with sensor data
   - Detect cascading failures

6. **Generate diagnostic report**
   - Demonstration: `15_export_visualization/04_report_generation.py`
   - Create formatted diagnostic report
   - Include live data, DTCs, and analysis
   - Export for mechanic or engineer review

**Expected Output:**
- List of active and historical DTCs
- Live sensor data (RPM, temperature, fuel pressure, etc.)
- Fault correlation analysis
- Formatted diagnostic report in PDF/HTML

**Success Criteria:**
- All DTCs decoded without CRC errors
- Sensor values in realistic ranges for vehicle
- Report includes both raw and interpreted data
- Timing information for fault occurrence

## Workflow 3: Validate Signal Integrity and Compliance

**Goal:** Measure signal quality, validate IEEE compliance, and identify signal integrity issues.

**Time Budget:** 15-20 minutes

**Steps:**

1. **Load high-speed digital or analog signal**
   - Demonstration: `01_data_loading/01_oscilloscopes.py`
   - Load captured waveforms from mixed-signal scope
   - Ensure adequate sampling rate (>10x signal frequency)
   - Extract clean capture windows

2. **Measure basic waveform parameters**
   - Demonstration: `02_basic_analysis/01_waveform_measurements.py`
   - Measure amplitude, frequency, and duty cycle
   - Measure rise/fall times
   - Measure overshoot and undershoot

3. **Analyze signal integrity metrics**
   - Demonstration: `04_advanced_analysis/03_signal_integrity.py`
   - Measure transition quality (rise/fall times)
   - Quantify ringing and overshoot/undershoot
   - Calculate TDR impedance profile
   - Measure skew and jitter

4. **Generate eye diagram analysis**
   - Demonstration: `04_advanced_analysis/04_eye_diagrams.py`
   - Build multi-sweep eye diagram
   - Measure eye height, width, opening
   - Calculate Q-factor
   - Identify margin for safe operation

5. **Perform IEEE 181 compliance validation**
   - Demonstration: `19_standards_compliance/01_ieee_181.py`
   - Use standardized measurement definitions
   - Report measurements with IEEE terminology
   - Validate against IEEE 181-2011 standard
   - Include uncertainty budgets

6. **Generate signal integrity report**
   - Demonstration: `15_export_visualization/04_report_generation.py`
   - Create compliance checklist
   - Include margin analysis
   - Recommend design improvements if needed
   - Export detailed measurements

**Expected Output:**
- Signal integrity report with all key metrics
- Eye diagram showing operation margin
- IEEE 181 measurement validation
- Recommendation list for design improvements

**Success Criteria:**
- All measurements compliant with IEEE 181-2011
- Eye opening >50% of unit interval
- Rise/fall times within specification
- Overshoot <10% of nominal level

## Workflow 4: Debug Power Supply Ripple and Transients

**Goal:** Measure power supply output quality, quantify ripple, analyze transients, and identify root causes.

**Time Budget:** 20-25 minutes

**Steps:**

1. **Capture power supply output waveforms**
   - Demonstration: `01_data_loading/01_oscilloscopes.py`
   - Load voltage and current waveforms
   - Ensure proper probe grounding (minimize noise)
   - Multi-channel capture with time alignment

2. **Measure DC output level and ripple**
   - Demonstration: `02_basic_analysis/02_statistical_measurements.py`
   - Calculate mean voltage (DC level)
   - Measure ripple amplitude (peak-to-peak)
   - Calculate ripple percentage
   - Measure DC offset

3. **Perform spectral analysis on ripple**
   - Demonstration: `02_basic_analysis/03_spectral_analysis.py`
   - FFT analysis of ripple content
   - Identify dominant ripple frequency (switching frequency)
   - Measure harmonic content at 2x, 3x switching frequency
   - Calculate total harmonic distortion (THD)

4. **Analyze power quality metrics**
   - Demonstration: `04_advanced_analysis/02_power_analysis.py`
   - Measure active and reactive power
   - Calculate power factor
   - Measure efficiency if load current available
   - Analyze load-dependent transients

5. **Detect and characterize transients**
   - Demonstration: `04_advanced_analysis/09_digital_timing.py`
   - Trigger on load step changes
   - Measure load-step response time
   - Quantify transient voltage overshoot
   - Measure settling time to steady state

6. **Export waveform data and analysis**
   - Demonstration: `15_export_visualization/02_wavedrom_timing.py`
   - Export ripple waveform for detailed analysis
   - Generate timing diagrams
   - Export power quality measurements
   - Create diagnostic report

**Expected Output:**
- Power supply output voltage and current waveforms
- Ripple spectrum showing frequency components
- Transient response measurements
- Power quality report
- Recommendations for ripple reduction

**Success Criteria:**
- Ripple <2% of nominal output voltage
- Transient response within spec (typically <100 us)
- Harmonic content decays properly
- Efficiency >85% at rated load

## Workflow 5: Characterize Logic Family and Voltage Levels

**Goal:** Identify logic family (TTL, CMOS, LVDS, etc.), measure voltage levels, and validate compatibility.

**Time Budget:** 15-20 minutes

**Steps:**

1. **Capture logic signals from device pins**
   - Demonstration: `01_data_loading/01_oscilloscopes.py`
   - Multi-channel digital capture
   - Capture multiple state transitions
   - Ensure adequate sample rate (>100 MHz for logic)

2. **Analyze voltage levels and thresholds**
   - Demonstration: `02_basic_analysis/02_statistical_measurements.py`
   - Measure high and low voltage levels
   - Calculate noise margins
   - Identify threshold voltages
   - Detect level shifters

3. **Characterize logic transition timing**
   - Demonstration: `04_advanced_analysis/03_signal_integrity.py`
   - Measure rise and fall times
   - Calculate propagation delay
   - Identify setup/hold time violations
   - Detect timing skew

4. **Classify logic family automatically**
   - Demonstration: `05_domain_specific/03_vintage_logic.py`
   - Use voltage levels to identify family
   - Detect if TTL (0-5V), CMOS (0-3.3V), LVDS, or other
   - Identify if mixed-voltage design
   - Cross-check with timing measurements

5. **Validate logic compatibility**
   - Demonstration: `06_reverse_engineering/08_data_classification.py`
   - Check if outputs are compatible with inputs
   - Verify noise margins are adequate
   - Identify potential interface issues
   - Predict reliability concerns

6. **Generate compatibility report**
   - Demonstration: `15_export_visualization/04_report_generation.py`
   - List detected logic families per signal
   - Report voltage levels and margins
   - Identify interface risks
   - Recommend solutions

**Expected Output:**
- Identified logic families for all signals
- Voltage level measurements
- Noise margin calculations
- Compatibility assessment
- Risk mitigation recommendations

**Success Criteria:**
- All logic families correctly identified
- Noise margins >300 mV (typical)
- No setup/hold violations detected
- Compatible signal levels across interfaces

## Workflow 6: Analyze Clock Recovery and Jitter Characteristics

**Goal:** Extract clock information from data, measure jitter, characterize PLL behavior, and validate timing.

**Time Budget:** 25-30 minutes

**Steps:**

1. **Capture clock and data waveforms**
   - Demonstration: `01_data_loading/01_oscilloscopes.py`
   - Multi-channel capture with clock and data
   - High sample rate (>10x data rate)
   - Long capture for statistical analysis (1000+ cycles)

2. **Measure clock frequency and stability**
   - Demonstration: `02_basic_analysis/01_waveform_measurements.py`
   - Measure clock frequency
   - Measure period jitter (cycle-to-cycle)
   - Measure frequency stability over time
   - Detect frequency drift or modulation

3. **Perform detailed jitter analysis**
   - Demonstration: `04_advanced_analysis/01_jitter_analysis.py`
   - Separate jitter into periodic and random components
   - Measure phase jitter (RMS and peak)
   - Calculate timing jitter (ps RMS)
   - Identify jitter sources (deterministic vs. random)

4. **Extract clock from data signal**
   - Demonstration: `14_exploratory/03_signal_recovery.py`
   - Use pattern recognition to extract embedded clock
   - Detect clock recovery circuit behavior
   - Analyze data eye relative to recovered clock
   - Measure clock-data timing

5. **Characterize PLL or oscillator behavior**
   - Demonstration: `04_advanced_analysis/01_jitter_analysis.py`
   - Measure lock-in time
   - Measure jitter tracking bandwidth
   - Analyze loop filter response
   - Detect any instabilities

6. **Generate timing compliance report**
   - Demonstration: `19_standards_compliance/01_ieee_181.py`
   - Compare measurements to IEEE 181-2011
   - Validate against timing specifications
   - Report jitter in standard units (ps, UI, ppm)
   - Identify timing violations

**Expected Output:**
- Clock frequency and period measurements
- Jitter analysis (periodic and random components)
- PLL lock-time and stability measurements
- Phase margin and timing violations
- Compliance report against specifications

**Success Criteria:**
- Phase jitter <100 ps (typical)
- Period jitter <0.5% of clock period
- PLL lock achieved within 100 cycles
- Clock-data timing within specifications

## Workflow 7: Memory Interface Characterization and Validation

**Goal:** Analyze address, data, and control signals, validate timing, and verify protocol compliance.

**Time Budget:** 25-35 minutes

**Steps:**

1. **Capture multi-channel memory bus signals**
   - Demonstration: `01_data_loading/07_multi_channel.py`
   - Capture address, data, control (CS, WE, OE, etc.)
   - High sample rate for timing accuracy
   - Synchronize all channels to common clock
   - Multiple read/write cycles

2. **Decode memory protocol (DDR/SPI Flash/Parallel)**
   - Demonstration: `03_protocol_decoding/04_parallel_bus.py`
   - Decode parallel memory bus
   - Extract address and data values
   - Identify transaction type (read/write/refresh)
   - Validate command sequences

3. **Measure timing parameters**
   - Demonstration: `04_advanced_analysis/09_digital_timing.py`
   - Measure setup and hold times
   - Measure access time (address to data valid)
   - Measure output delay
   - Measure cycle time
   - Detect timing violations

4. **Analyze signal integrity on memory bus**
   - Demonstration: `04_advanced_analysis/03_signal_integrity.py`
   - Measure address/data line crosstalk
   - Measure line reflections (TDR)
   - Analyze data valid window
   - Quantify noise on bus

5. **Characterize memory device behavior**
   - Demonstration: `05_domain_specific/03_vintage_logic.py`
   - Identify memory type (SRAM, DRAM, Flash)
   - Measure access time and speed grade
   - Detect refresh cycles (if DRAM)
   - Verify device specifications

6. **Generate memory validation report**
   - Demonstration: `15_export_visualization/04_report_generation.py`
   - Timing diagram for each transaction type
   - All measured timing parameters
   - Comparison to device datasheet
   - Identified violations or margin issues
   - Reliability recommendations

**Expected Output:**
- Decoded memory transactions
- Timing measurements for all critical paths
- Signal integrity analysis
- Memory device identification
- Protocol compliance report
- Margin analysis and recommendations

**Success Criteria:**
- All memory transactions decoded correctly
- Setup/hold time margins >200 ps
- Data valid window sufficient for sampling
- No timing violations detected
- Protocol matches device datasheet

## Workflow 8: Wireless Protocol Analysis and Demodulation

**Goal:** Capture wireless signal, demodulate, decode protocol, and validate link quality.

**Time Budget:** 30-40 minutes

**Steps:**

1. **Capture wireless signal (RF or baseband)**
   - Demonstration: `01_data_loading/01_oscilloscopes.py`
   - Capture RF or demodulated signal
   - Ensure adequate sample rate (>2x bandwidth)
   - Capture multiple frames/packets
   - Maintain SNR >20 dB

2. **Characterize modulation and signal**
   - Demonstration: `14_exploratory/01_unknown_signals.py`
   - Detect modulation type (AM, FM, FSK, PSK, QAM)
   - Measure carrier frequency (if modulated)
   - Measure bandwidth
   - Estimate SNR

3. **Demodulate wireless signal**
   - Demonstration: `17_signal_generation/03_impairment_simulation.py`
   - Apply appropriate demodulation (FM, FSK, PSK, etc.)
   - Generate complex baseband signal
   - Perform phase recovery if needed
   - Extract bit stream

4. **Decode wireless protocol layer**
   - Demonstration: `03_protocol_decoding/01_serial_comprehensive.py`
   - Identify packet/frame boundaries
   - Decode header and payload
   - Extract addresses and control information
   - Verify checksums

5. **Analyze protocol and messages**
   - Demonstration: `06_reverse_engineering/05_pattern_discovery.py`
   - Identify repeated message patterns
   - Extract command sequences
   - Detect handshake or negotiation
   - Classify message types

6. **Assess wireless link quality**
   - Demonstration: `04_advanced_analysis/03_signal_integrity.py`
   - Measure SNR and EVM (Error Vector Magnitude)
   - Estimate bit error rate
   - Analyze phase noise
   - Measure symbol timing accuracy

7. **Export protocol and signals**
   - Demonstration: `15_export_visualization/02_wavedrom_timing.py`
   - Export demodulated signal
   - Export extracted bit stream
   - Create timing diagram of protocol
   - Generate analysis report

**Expected Output:**
- Demodulated signal and bit stream
- Decoded wireless protocol packets
- Modulation type and parameters
- Link quality measurements (SNR, EVM, BER estimate)
- Protocol specification
- Signal analysis report

**Success Criteria:**
- All packets decoded without errors
- SNR >20 dB for reliable operation
- EVM <10% for QAM-based systems
- Protocol structure matches observed traffic
- Link quality meets specification

## Running Demonstrations from Workflows

To execute a demonstration:

```bash
cd /path/to/oscura
python3 demonstrations/PATH/TO/DEMONSTRATION.py
```

For example, to run the hello world demonstration:

```bash
python3 demonstrations/00_getting_started/00_hello_world.py
```

To run all demonstrations in a category:

```bash
python3 demonstrations/generate_all_data.py
```

To validate all demonstrations:

```bash
python3 demonstrations/validate_all.py
```

## Tips for Success

**Protocol Reverse Engineering:**
- Capture at least 10-20 message exchanges for pattern analysis
- Use triggering to isolate specific operations
- Compare multiple capture files to confirm patterns
- Validate inferred fields against device behavior

**Automotive Diagnostics:**
- Capture at engine start, idle, and load conditions
- Extract both OBD-II Mode 01 (live data) and Mode 03 (DTCs)
- Cross-reference DTC codes with sensor data
- Include timestamp information for fault correlation

**Signal Integrity:**
- Use differential probing for balanced signals
- Minimize probe parasitic effects with de-embedding
- Capture steady-state and transient conditions
- Use multiple timebase scales for complete picture

**Wireless Analysis:**
- Maintain >20 dB SNR for reliable demodulation
- Capture multiple frames to detect protocol errors
- Validate demodulation quality with constellation diagram
- Use known reference signals for calibration

## Integration with Existing Tools

Oscura demonstrations integrate with:

- **Wireshark**: Export protocol dissectors for packet analysis
- **Jupyter**: Interactive analysis in notebooks
- **LTspice**: Compare measurements to simulation
- **GitLab CI**: Automated analysis in pipelines
- **Python**: Extend demonstrations with custom analysis

## Standards Compliance

Demonstrations validate against:

- **IEEE 181-2011**: Waveform and vector measurements
- **IEEE 1241-2010**: Analog-to-digital converter testing
- **IEEE 1459-2010**: Power quality definitions
- **IEEE 2414-2020**: Software-defined metrology
- **SAE J1979**: OBD-II standard
- **ISO 14229**: UDS diagnostic standard

## Next Steps After Workflows

After completing a workflow:

1. **Extend Analysis**: Use custom measurements and plugins
2. **Automate Testing**: Create production test scripts
3. **Export Results**: Generate reports and dissectors
4. **Scale Up**: Process multiple captures in batch mode
5. **Integrate**: Connect to external tools and dashboards

For more information, see the main demonstrations README and individual demonstration files.
