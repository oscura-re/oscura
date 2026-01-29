# Demos Migration Status: Categories 03-05

**Migration Date**: 2026-01-29
**Status**: In Progress - Core Demos Completed

## Overview

This document tracks the consolidation migration of protocol decoding, advanced analysis, and domain-specific demonstrations from multiple source directories into the unified `demos/` structure.

## Category 03: Protocol Decoding (Target: 12 demos)

### Completed ✅

1. **01_uart_basic.py** - Comprehensive UART decoding
   - Sources: demonstrations/03_protocol_decoding/01_serial_comprehensive.py (UART portion)
   - Features: Multiple baudrates (9600, 115200), parity modes, data bit configs
   - Validation: ValidationSuite with expected data matching

2. **02_spi_basic.py** - Comprehensive SPI decoding
   - Sources: demonstrations/03_protocol_decoding/01_serial_comprehensive.py (SPI portion)
   - Features: All 4 SPI modes (CPOL/CPHA combinations), full-duplex
   - Validation: Master/slave data verification

3. **03_i2c_basic.py** - Comprehensive I2C decoding
   - Sources: demonstrations/03_protocol_decoding/01_serial_comprehensive.py (I2C portion)
   - Features: Standard/Fast modes, multi-byte transactions, START/STOP
   - Validation: Address and data packet verification

4. **04_can_basic.py** - CAN 2.0 protocol decoding
   - Sources: demonstrations/03_protocol_decoding/02_automotive_protocols.py (CAN portion)
   - Features: Standard/extended IDs, variable bitrates (500k, 1M)
   - Validation: Frame structure and CRC

### Pending 🔲

5. **05_can_fd.py** - CAN-FD with dual bitrate
   - Sources: demonstrations/03_protocol_decoding/02_automotive_protocols.py (CAN-FD portion)
   - Plan: Dual bitrate (500k nominal, 2M data), extended payloads (up to 64 bytes)

6. **06_lin.py** - LIN bus protocol
   - Sources: demonstrations/03_protocol_decoding/02_automotive_protocols.py (LIN portion)
   - Sources: demonstrations/05_domain_specific/automotive_lin.py
   - Plan: Merge automotive-specific LIN features

7. **07_flexray.py** - FlexRay deterministic protocol
   - Sources: demonstrations/03_protocol_decoding/02_automotive_protocols.py (FlexRay portion)
   - Sources: demonstrations/05_domain_specific/automotive_flexray.py
   - Plan: Merge slot-based scheduling examples

8. **08_jtag.py** - JTAG debug protocol (IEEE 1149.1)
   - Sources: demonstrations/03_protocol_decoding/jtag.py, demonstrations/03_protocol_decoding/03_debug_protocols.py
   - Plan: TAP state machine, boundary scan

9. **09_swd.py** - SWD debug protocol
   - Sources: demonstrations/03_protocol_decoding/swd.py, demonstrations/03_protocol_decoding/03_debug_protocols.py
   - Plan: ARM CoreSight debugging

10. **10_i2s.py** - I2S audio protocol
    - Sources: demonstrations/03_protocol_decoding/i2s.py
    - Plan: Multiple sample rates (44.1k, 48k, 96k), bit depths

11. **11_usb.py** - USB protocol decoding
    - Sources: demonstrations/03_protocol_decoding/usb.py
    - Note: SKIP_VALIDATION marker present - PID validation issues
    - Plan: USB Low-Speed, NRZI decoding, packet types

12. **12_comprehensive_protocols.py** - Multi-protocol analysis
    - Sources: demonstrations/03_protocol_decoding/protocol_comprehensive.py
    - Plan: Unified demo showing all protocols working together

## Category 04: Advanced Analysis (Target: 12 demos)

### Completed ✅

(None yet)

### Pending 🔲

1. **01_jitter_analysis.py** - TIE, jitter histogram
   - Sources: demos/13_jitter_analysis/, demonstrations/04_advanced_analysis/

2. **02_jitter_decomposition.py** - RJ/DJ separation
   - Sources: Jitter analysis sources

3. **03_bathtub_curves.py** - BER analysis
   - Sources: Jitter analysis sources

4. **04_eye_diagrams.py** - Eye diagram generation
   - Sources: Multiple eye diagram demos to merge

5. **05_eye_metrics.py** - Eye height/width measurements
   - Sources: Eye diagram sources

6. **06_power_analysis.py** - DC/AC power measurements
   - Sources: demos/14_power_analysis/, demonstrations/04_advanced_analysis/

7. **07_efficiency.py** - Power efficiency analysis
   - Sources: Power analysis sources

8. **08_signal_integrity.py** - S-parameters, impedance
   - Sources: demos/15_signal_integrity/, demonstrations/04_advanced_analysis/

9. **09_tdr.py** - Time-domain reflectometry
   - Sources: Signal integrity sources

10. **10_correlation.py** - Cross-correlation analysis
    - Sources: demonstrations/04_advanced_analysis/

11. **11_statistics_advanced.py** - Advanced statistical methods
    - Sources: demonstrations/04_advanced_analysis/

12. **12_comprehensive_analysis.py** - Multi-analyzer workflow
    - Sources: demonstrations/04_advanced_analysis/

## Category 05: Domain Specific (Target: 8 demos)

### Completed ✅

(None yet)

### Pending 🔲

1. **01_automotive_diagnostics.py** - CAN diagnostics, UDS
   - Sources: demos/08_automotive_protocols/, demos/09_automotive/

2. **02_automotive_comprehensive.py** - Complete automotive workflow
   - Sources: Automotive sources

3. **03_emc_compliance.py** - EMI/EMC testing
   - Sources: demos/16_emc_compliance/, demonstrations/05_domain_specific/

4. **04_emc_comprehensive.py** - Complete EMC workflow
   - Sources: EMC sources

5. **05_side_channel_basics.py** - Power/timing analysis
   - Sources: examples/side_channel_analysis_demo.py
   - Plan: Convert to BaseDemo format

6. **06_side_channel_dpa.py** - Differential Power Analysis
   - Sources: Side-channel sources

7. **07_timing_ieee181.py** - IEEE 181 timing compliance
   - Sources: demonstrations/19_standards_compliance/

8. **08_vintage_logic.py** - Vintage hardware protocols
   - Sources: demonstrations/05_domain_specific/

## Source Directory Mapping

### demonstrations/03_protocol_decoding/
- 01_serial_comprehensive.py → Split into 01_uart, 02_spi, 03_i2c, (onewire portion to be added)
- 02_automotive_protocols.py → Split into 04_can, 05_can_fd, 06_lin, 07_flexray
- 03_debug_protocols.py → Content to 08_jtag, 09_swd
- jtag.py, swd.py, i2s.py, usb.py → Direct conversions with BaseDemo
- protocol_comprehensive.py → 12_comprehensive_protocols.py

### demos/04_serial_protocols/
- manchester_demo.py, i2s_demo.py, jtag_demo.py, onewire_demo.py, swd_demo.py, usb_demo.py
- Plan: Merge validation logic into consolidated demos

### demos/05_protocol_decoding/
- comprehensive_protocol_demo.py → Merge into 12_comprehensive_protocols.py

### demonstrations/04_advanced_analysis/
- Multiple files → Category 04 demos

### demos/13_jitter_analysis/, demos/14_power_analysis/, demos/15_signal_integrity/
- Content → Category 04 demos

### demonstrations/05_domain_specific/, demos/08_automotive_protocols/, demos/09_automotive/
- Content → Category 05 demos

### demos/16_emc_compliance/
- Content → Category 05 EMC demos

### examples/side_channel_analysis_demo.py
- Convert to Category 05 demo with BaseDemo

## Migration Strategy

### Completed Work
1. ✅ Created directory structure (03_protocol_decoding/, 04_advanced_analysis/, 05_domain_specific/)
2. ✅ Created __init__.py files for all three categories
3. ✅ Migrated 4 core protocol demos (UART, SPI, I2C, CAN basic)
4. ✅ All migrated demos use demos.common infrastructure
5. ✅ All migrated demos include ValidationSuite
6. ✅ All demos syntax-validated (py_compile)

### Remaining Work
1. 🔲 Complete remaining 8 protocol demos (CAN-FD, LIN, FlexRay, JTAG, SWD, I2S, USB, comprehensive)
2. 🔲 Create all 12 advanced analysis demos
3. 🔲 Create all 8 domain-specific demos
4. 🔲 Add comprehensive metadata (capabilities, IEEE standards) to all demos
5. 🔲 Test all demos execute successfully
6. 🔲 Update demos/README.md with new structure

## Validation Checklist

For each migrated demo:
- [x] Uses demos.common.BaseDemo
- [x] Includes ValidationSuite
- [x] Self-contained data generation
- [x] Comprehensive docstring with IEEE standards
- [x] Related demos metadata
- [x] Capabilities list
- [x] Syntax validated (py_compile)
- [ ] Execution tested
- [ ] All source capabilities preserved

## Notes

- **UART, SPI, I2C, CAN**: Core protocol demos completed with comprehensive configurations
- **BaseDemo Pattern**: All demos follow demonstrations/ BaseDemo pattern with metadata
- **ValidationSuite**: All demos use ValidationSuite for consistent validation
- **Imports**: All demos use `demos.common` imports (verified working)
- **Synthetic Data**: All demos generate self-contained test data
- **Token Efficiency**: Created core demos first, remaining can follow same patterns

## Next Steps

1. Complete remaining protocol demos (5-12) using same pattern as 1-4
2. Begin advanced analysis demos (jitter, power, signal integrity)
3. Create domain-specific demos (automotive, EMC, side-channel)
4. Full execution testing of all demos
5. Update master README and documentation

---

**Completion**: 4/32 demos (12.5%)
**Category 03**: 4/12 complete (33%)
**Category 04**: 0/12 complete (0%)
**Category 05**: 0/8 complete (0%)
