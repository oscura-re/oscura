# Domain-Specific Applications

**Apply Oscura to real-world industry domains: automotive diagnostics, EMC compliance, vintage logic, and side-channel analysis.**

This section contains 4 demonstrations covering specialized applications across automotive (OBD-II, J1939, UDS), electromagnetic compatibility (CISPR, MIL-STD), vintage computing (TTL, CMOS, ECL identification), and security (power analysis attacks). Perfect for domain experts and industry practitioners.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Finish `00_getting_started/` first
- **Completed Basic Analysis** - Finish `02_basic_analysis/` REQUIRED
- **Completed Advanced Analysis** - Finish `04_advanced_analysis/` recommended
- **Python 3.12+** - Oscura requires Python 3.12 or higher
- **Oscura installed** - Install with `pip install oscura` or `uv add oscura`
- **Domain knowledge** - Understanding of target industry domain helpful
- **Industry standards** - Familiarity with relevant standards (SAE, ISO, CISPR)

---

## Demonstrations

| Demo      | File                           | Time       | Difficulty                 | Topics                                             |
| --------- | ------------------------------ | ---------- | -------------------------- | -------------------------------------------------- |
| **01**    | `01_automotive_diagnostics.py` | 25 min     | Advanced                   | OBD-II (54+ PIDs), J1939 (154+ PGNs), UDS, DTCs    |
| **02**    | `02_emc_compliance.py`         | 20 min     | Advanced                   | CISPR 16/32, MIL-STD-461G, emissions testing       |
| **03**    | `03_vintage_logic.py`          | 20 min     | Intermediate               | TTL, CMOS, ECL family detection, IC identification |
| **04**    | `04_side_channel.py`           | 25 min     | Expert                     | DPA, CPA, timing attacks, AES key extraction       |
| **Total** |                                | **90 min** | **Intermediate to Expert** | **Industry applications**                          |

---

## Learning Path

These demonstrations can be completed **in any order** based on your domain of interest:

```
01_automotive_diagnostics.py → 02_emc_compliance.py
        ↓                              ↓
  Automotive industry          Compliance testing
  OBD-II, J1939, UDS          CISPR, IEC, MIL-STD
  DTC database (210+ codes)   Conducted/radiated emissions
        ↓                              ↓
03_vintage_logic.py → 04_side_channel.py
        ↓                    ↓
  Vintage computing      Security research
  TTL, CMOS, ECL        DPA, CPA attacks
  IC identification     Cryptographic key extraction
```

### Recommended Time

**Automotive engineers** (25 min): Demo 01
**EMC test engineers** (20 min): Demo 02
**Vintage computing enthusiasts** (20 min): Demo 03
**Security researchers** (25 min): Demo 04
**Complete domain coverage** (90 min): All demos

---

## Key Concepts

### What You'll Learn

**Automotive Diagnostics** (Demo 01):

- OBD-II Mode 01-09 diagnostics (54+ PIDs)
- J1939 heavy-duty vehicle diagnostics (154+ PGNs)
- UDS (Unified Diagnostic Services) per ISO 14229
- Diagnostic Trouble Code (DTC) database (210+ codes)
- Live data monitoring (RPM, speed, temperature, etc.)
- Freeze frame data extraction
- Emissions readiness status

**EMC Compliance** (Demo 02):

- Conducted emissions measurement per CISPR 16
- Radiated emissions measurement
- CISPR 32 Class A/B limit masks
- MIL-STD-461G military EMC requirements
- Quasi-peak and average detection
- Frequency domain compliance analysis
- Automated pass/fail determination

**Vintage Logic Analysis** (Demo 03):

- TTL family detection (74xx, 74LSxx, 74Sxx, 74ASxx, 74ALSxx)
- CMOS family detection (4000 series, 74HCxx, 74HCTxx)
- ECL family detection (10K, 10H series)
- RTL (Resistor-Transistor Logic) identification
- DTL (Diode-Transistor Logic) identification
- Voltage level analysis for family classification
- IC identification from timing parameters
- Modern replacement part recommendations

**Side-Channel Analysis** (Demo 04):

- Differential Power Analysis (DPA)
- Correlation Power Analysis (CPA)
- Timing attack detection and exploitation
- Statistical leakage detection (Welch's t-test)
- AES cryptographic key extraction
- Template attacks with pre-characterization
- ChipWhisperer format support
- Countermeasure effectiveness evaluation

---

## Running Demonstrations

### Option 1: Run Individual Demo

Run a single demo to focus on your domain:

```bash
# From the project root
python demonstrations/05_domain_specific/01_automotive_diagnostics.py

# Or from the demo directory
cd demonstrations/05_domain_specific
python 01_automotive_diagnostics.py
```

Expected output: Domain-specific analysis with industry-standard validation.

### Option 2: Run All Domain-Specific Demos

Run all four demos in sequence:

```bash
# From the project root
for demo in demonstrations/05_domain_specific/*.py; do
    python "$demo"
done
```

### Option 3: Validate All Demonstrations

Validate all demonstrations in the project:

```bash
# From the project root
python demonstrations/validate_all.py
```

This runs all demonstrations and reports coverage, validation status, and failures.

---

## What You'll Learn

### Industry Standards

**Automotive Standards**:

- SAE J1979 (OBD-II diagnostic specifications)
- SAE J1939 (Heavy-duty vehicle network)
- ISO 14229 (Unified Diagnostic Services)
- ISO 15031 (Communication vehicle to external equipment)
- ISO 11898-1 (CAN protocol)

**EMC Standards**:

- CISPR 16-1-1 (Measurement apparatus and methods)
- CISPR 32 (Multimedia equipment emissions)
- IEC 61000-4-x (Immunity testing methods)
- MIL-STD-461G (Military EMC requirements)
- FCC Part 15 (Radio frequency devices)

**Logic Family Standards**:

- TTL Data Books (Texas Instruments, Fairchild)
- CMOS Data Books (RCA, Motorola)
- ECL Data Books (Motorola, On Semiconductor)
- Vintage IC cross-reference guides

**Security Standards**:

- ISO/IEC 17825 (Testing methods for side-channel attacks)
- NIST SP 800-57 (Key management recommendations)
- FIPS 140-2/140-3 (Cryptographic module validation)

### Domain-Specific Techniques

**Automotive Diagnostics**:

- CAN bus traffic analysis
- PID (Parameter ID) decoding and interpretation
- DTC code lookup and description
- Service ID recognition (0x01-0x09 for OBD-II)
- PGN (Parameter Group Number) extraction
- Multi-frame message reassembly
- Checksum validation

**EMC Compliance Testing**:

- Quasi-peak detector simulation
- Average detector implementation
- Frequency sweep analysis
- Limit line comparison
- Margin calculation
- Pre-compliance testing
- Troubleshooting emissions failures

**Vintage Logic Analysis**:

- Voltage threshold detection
- Propagation delay measurement
- Logic family voltage level comparison
- Timing parameter extraction
- IC pinout inference
- Part number cross-referencing
- Modern equivalent identification

**Side-Channel Analysis**:

- Power trace acquisition and preprocessing
- Differential trace computation
- Correlation coefficient calculation
- Hypothesis testing for key bytes
- Statistical significance testing
- Key rank analysis
- Attack success validation

---

## Common Issues and Solutions

### "OBD-II PID not recognized"

**Solution**: PID database may not include all manufacturer-specific PIDs:

1. Check if PID is standard (0x00-0x4E) or manufacturer-specific (0x80+)
2. Consult vehicle-specific documentation for proprietary PIDs
3. Use raw value extraction if PID formula unknown
4. Verify service mode (0x01 for live data, 0x02 for freeze frame)

### "EMC measurements don't match spectrum analyzer"

**Solution**: Detector type and measurement bandwidth matter:

1. Verify detector type (quasi-peak vs average vs peak)
2. Check resolution bandwidth (RBW) settings
3. Ensure frequency sweep is complete
4. Validate calibration and antenna factors
5. Compare measurement methods (conducted vs radiated)

### "Logic family detection shows mixed results"

**Solution**: Voltage levels may be non-standard:

1. Check for degraded components (capacitors, power supply)
2. Verify measurement at IC pins (not on PCB traces)
3. Look for voltage level translators in circuit
4. Consider custom/modified logic levels
5. Measure multiple transitions for statistics

### "Side-channel attack fails to extract key"

**Solution**: Insufficient traces or noise:

1. Increase number of power traces (1000+ recommended)
2. Apply signal averaging to reduce noise
3. Verify correct alignment of traces
4. Check for countermeasures (masking, shuffling)
5. Validate Hamming weight model assumptions

---

## Next Steps: Where to Go After Domain-Specific

### If You Want to...

| Goal                     | Next Demo                                        | Path                   |
| ------------------------ | ------------------------------------------------ | ---------------------- |
| Build complete workflows | `16_complete_workflows/01_protocol_discovery.py` | Domain → Complete RE   |
| Learn extensibility      | `08_extensibility/01_plugin_basics.py`           | Domain → Custom tools  |
| Batch processing         | `09_batch_processing/01_multi_file.py`           | Domain → Automation    |
| Export and visualization | `15_export_visualization/01_reports.py`          | Domain → Documentation |

### Recommended Learning Sequence

1. **Master Domain-Specific Applications** (this section)
   - Apply to your industry domain
   - Understand relevant standards
   - Use industry-standard workflows

2. **Explore Complete Workflows** (16_complete_workflows/)
   - End-to-end reverse engineering
   - Multi-stage analysis pipelines
   - Production-ready solutions

3. **Learn Extensibility** (08_extensibility/)
   - Create custom domain-specific plugins
   - Extend measurement registry
   - Build reusable components

4. **Implement Batch Processing** (09_batch_processing/)
   - Automate domain workflows
   - Process multiple captures
   - Generate batch reports

---

## Tips for Learning

### Understand Your Domain Standards

Each demonstration references industry standards:

```python
# Automotive: SAE J1979 defines OBD-II PIDs
pid_0x0c = 0x0C  # Engine RPM
formula = "((A * 256) + B) / 4"  # Per SAE J1979

# EMC: CISPR 32 defines emission limits
cispr32_class_b_limit = {
    0.15: (66, 56),  # MHz: (quasi-peak dBμV, average dBμV)
    0.50: (56, 46),
}

# Logic: TTL has specific voltage thresholds
ttl_vil_max = 0.8  # V - Maximum LOW input
ttl_vih_min = 2.0  # V - Minimum HIGH input
```

Read the standards for authoritative specifications.

### Start with Known Reference Data

Validate techniques with known-good data:

```python
# Automotive: Test with known OBD-II responses
known_rpm_response = [0x41, 0x0C, 0x0F, 0xA0]  # 1000 RPM
rpm = decode_obd2_pid(known_rpm_response, pid=0x0C)
assert rpm == 1000

# EMC: Test with known signal vs limits
test_signal = generate_emissions_signature(freq=150e3, level=60)
result = check_cispr32_compliance(test_signal, class_type='B')
assert result.passes
```

### Combine Domain Knowledge with Analysis

Real-world applications require both:

```python
# Automotive diagnostic workflow
# 1. Decode CAN traffic
can_frames = decode_can(trace, bitrate=500000)

# 2. Extract OBD-II responses
obd_responses = [f for f in can_frames if f.identifier == 0x7E8]

# 3. Decode PIDs with domain knowledge
for response in obd_responses:
    pid = response.data[1]
    value = decode_obd2_pid(response.data, pid)
    print(f"PID 0x{pid:02X}: {value}")
```

### Visualize Domain-Specific Results

Domain experts expect specific visualizations:

```python
import matplotlib.pyplot as plt

# EMC: Plot spectrum vs limits
freqs, levels = measure_conducted_emissions(trace)
limits = get_cispr32_limits(class_type='B')

plt.plot(freqs / 1e6, levels, label='Measured')
plt.plot(limit_freqs / 1e6, limit_levels, 'r--', label='CISPR 32 Class B')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Level (dBμV)')
plt.legend()
plt.grid(True)

# Side-channel: Plot correlation traces
for key_hypothesis in range(256):
    correlation = compute_cpa_correlation(traces, key_hypothesis)
    plt.plot(correlation, alpha=0.1, color='gray')
plt.plot(max_correlation, 'r', linewidth=2, label='Correct key')
plt.xlabel('Sample')
plt.ylabel('Correlation')
```

### Reference Industry Tools

Compare with professional domain tools:

- **Automotive**: CANalyzer, CANoe, Vehicle Spy
- **EMC**: R&S EMI Test Receiver, Keysight MXE EMI
- **Vintage Logic**: Logic analyzers with family detection
- **Side-Channel**: ChipWhisperer, Riscure Inspector

---

## Understanding the Framework

### Automotive Diagnostics API

```python
from oscura import decode_can
from oscura.automotive.diagnostics import decode_obd2, decode_j1939, decode_uds

# Decode CAN bus traffic
can_frames = decode_can(trace, bitrate=500000)

# OBD-II decoding
obd_responses = [f for f in can_frames if f.identifier == 0x7E8]
for response in obd_responses:
    pid_data = decode_obd2(response, mode=0x01)
    print(f"PID: {pid_data['pid_name']} = {pid_data['value']} {pid_data['unit']}")

# J1939 decoding
j1939_messages = [f for f in can_frames if f.identifier & 0x1FFFFFFF > 0]
for msg in j1939_messages:
    pgn_data = decode_j1939(msg)
    print(f"PGN: {pgn_data['pgn_name']} = {pgn_data['parameters']}")

# UDS decoding
uds_responses = [f for f in can_frames if f.data[0] in [0x50, 0x7F]]
for response in uds_responses:
    uds_data = decode_uds(response)
    print(f"Service: {uds_data['service_name']} - {uds_data['response']}")
```

### EMC Compliance API

```python
from oscura.analyzers.emc import (
    measure_conducted_emissions,
    measure_radiated_emissions,
    check_cispr32_compliance,
    check_mil_std_461g_compliance
)

# Measure conducted emissions
freqs, levels = measure_conducted_emissions(trace, rbw=9000)

# Check CISPR 32 compliance
result = check_cispr32_compliance(
    frequencies=freqs,
    levels=levels,
    class_type='B',  # Class B (residential)
    detector='quasi-peak'
)

if result.passes:
    print(f"PASS - Margin: {result.margin_db:.1f} dB")
else:
    print(f"FAIL - {len(result.violations)} violations")
    for v in result.violations:
        print(f"  {v.frequency/1e6:.2f} MHz: {v.level:.1f} dBμV (limit: {v.limit:.1f})")
```

### Vintage Logic API

```python
from oscura.analyzers.vintage import (
    detect_logic_family,
    identify_ic,
    suggest_replacement
)

# Detect logic family from voltage levels
family = detect_logic_family(trace, threshold_method='automatic')
print(f"Logic family: {family.name}")
print(f"VIL: {family.vil_max:.2f} V, VIH: {family.vih_min:.2f} V")

# Identify specific IC from timing
ic_info = identify_ic(trace, family=family)
print(f"Likely IC: {ic_info.part_number}")
print(f"Function: {ic_info.description}")

# Suggest modern replacement
replacement = suggest_replacement(ic_info.part_number)
print(f"Modern equivalent: {replacement.part_number}")
print(f"Manufacturer: {replacement.manufacturer}")
```

### Side-Channel Analysis API

```python
from oscura.analyzers.side_channel import (
    dpa_attack,
    cpa_attack,
    timing_attack,
    t_test_leakage
)

# Differential Power Analysis
key_candidates = dpa_attack(
    power_traces=traces,
    plaintexts=plaintexts,
    ciphertexts=ciphertexts,
    target_byte=0
)
print(f"Most likely key byte: 0x{key_candidates[0]:02X}")

# Correlation Power Analysis
correlation_results = cpa_attack(
    power_traces=traces,
    plaintexts=plaintexts,
    target_byte=0,
    model='hamming_weight'
)

# Statistical leakage detection
leakage = t_test_leakage(
    traces_set1=traces_fixed,
    traces_set2=traces_random
)
if leakage.significant:
    print(f"Leakage detected at sample {leakage.max_sample}")
```

---

## Resources

### In This Repository

- **`src/oscura/automotive/`** - Automotive diagnostics implementations
- **`src/oscura/analyzers/emc/`** - EMC compliance testing
- **`src/oscura/analyzers/vintage/`** - Vintage logic analysis
- **`src/oscura/analyzers/side_channel/`** - Side-channel attack implementations

### External Resources

**Automotive**:

- **[SAE International](https://www.sae.org/)** - J1979, J1939 standards
- **[ISO Standards](https://www.iso.org/)** - ISO 14229, 15031, 11898
- **[ASAM](https://www.asam.net/)** - MDF/MF4 specifications

**EMC**:

- **[CISPR](https://www.iec.ch/emc/emc_prod/prod_main.htm)** - Emission standards
- **[FCC](https://www.fcc.gov/)** - Part 15 regulations
- **[MIL-STD](https://quicksearch.dla.mil/)** - Military standards

**Vintage Logic**:

- **[All About Circuits](https://www.allaboutcircuits.com/)** - Logic family tutorials
- **[Vintage IC Data](https://www.datasheetarchive.com/)** - Historical datasheets

**Side-Channel**:

- **[ChipWhisperer](https://www.newae.com/)** - Open-source SCA platform
- **[NIST Cryptography](https://csrc.nist.gov/)** - Cryptographic standards
- **[Riscure](https://www.riscure.com/)** - Side-channel research

### Professional Certifications

- **ASE Certification** (Automotive Service Excellence)
- **NARTE EMC Certification** (EMC engineering)
- **CISSP** (Security - for side-channel work)

### Getting Help

1. Consult industry standards for authoritative guidance
2. Compare with professional domain tools
3. Review demo docstrings for methodology
4. Test with known reference data first
5. Engage with domain-specific communities

---

## Summary

The Domain-Specific Applications section covers:

| Demo                      | Focus             | Outcome                                     |
| ------------------------- | ----------------- | ------------------------------------------- |
| 01_automotive_diagnostics | Automotive        | OBD-II, J1939, UDS, 210+ DTCs               |
| 02_emc_compliance         | EMC testing       | CISPR 16/32, MIL-STD-461G compliance        |
| 03_vintage_logic          | Vintage computing | TTL, CMOS, ECL detection, IC identification |
| 04_side_channel           | Security          | DPA, CPA, timing attacks, key extraction    |

After completing these four 90-minute demonstrations, you'll understand:

- How to apply Oscura to real-world industry domains
- Industry-standard compliance testing and validation
- Domain-specific analysis workflows
- Integration with professional tools
- Standards-compliant measurement techniques

**Ready to start?** Choose your domain:

```bash
# Automotive engineers
python demonstrations/05_domain_specific/01_automotive_diagnostics.py

# EMC test engineers
python demonstrations/05_domain_specific/02_emc_compliance.py

# Vintage computing enthusiasts
python demonstrations/05_domain_specific/03_vintage_logic.py

# Security researchers
python demonstrations/05_domain_specific/04_side_channel.py
```

Happy domain exploring!
