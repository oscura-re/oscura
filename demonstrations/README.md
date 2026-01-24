# Oscura Demonstrations

**Complete capability showcase for the Oscura hardware reverse engineering framework.**

This directory contains **100% comprehensive demonstrations** of all Oscura capabilities, organized for progressive learning and complete API verification.

---

## Quick Start

```bash
# 1. Generate test data (first time only)
python demonstrations/generate_all_data.py

# 2. Run your first demo
python demonstrations/00_getting_started/00_hello_world.py

# 3. Validate all demonstrations
python demonstrations/validate_all.py

# 4. Generate capability coverage report
python demonstrations/capability_index.py --format markdown
```

---

## Organization

Demonstrations are organized in **progressive learning order** from beginner to expert:

| Section                     | Focus       | Demos | Description                                         |
| --------------------------- | ----------- | ----- | --------------------------------------------------- |
| **00_getting_started**      | Basics      | 3     | Installation validation, core types, format support |
| **01_data_loading**         | I/O         | 7     | All 21+ file format loaders                         |
| **02_basic_analysis**       | Core        | 6     | Measurements, statistics, spectral, filtering       |
| **03_protocol_decoding**    | Protocols   | 6     | All 17+ protocol decoders                           |
| **04_advanced_analysis**    | Advanced    | 6     | Jitter, power, signal integrity, patterns           |
| **05_domain_specific**      | Industry    | 4     | Automotive, EMC, vintage logic, side-channel        |
| **06_reverse_engineering**  | RE          | 6     | Protocol inference, CRC recovery, state machines    |
| **07_advanced_api**         | Expert API  | 7     | Pipelines, DSL, operators, optimization             |
| **08_extensibility**        | Plugins     | 6     | Custom measurements, algorithms, plugins            |
| **09_batch_processing**     | Batch       | 3     | Parallel processing, aggregation                    |
| **10_sessions**             | Sessions    | 5     | Interactive analysis, annotations, history          |
| **11_integration**          | Integration | 5     | CLI, Jupyter, LLM, configuration                    |
| **12_quality_tools**        | Quality     | 4     | Ensemble methods, scoring, warnings                 |
| **13_guidance**             | Guidance    | 3     | Recommendations, wizards, onboarding                |
| **14_exploratory**          | Exploration | 4     | Unknown signals, fuzzy matching, recovery           |
| **15_export_visualization** | Export      | 5     | All formats, WaveDrom, Wireshark, reports           |
| **16_complete_workflows**   | Workflows   | 6     | End-to-end reverse engineering pipelines            |
| **17_signal_generation**    | Generation  | 3     | SignalBuilder, synthetic signals, protocols         |
| **18_comparison_testing**   | Testing     | 4     | Golden reference, limits, masks                     |
| **19_standards_compliance** | Standards   | 4     | IEEE 181, 1241, 1459, 2414 validation               |

**Total:** ~100 demonstrations covering 813 API symbols across 45 modules

---

## Finding What You Need

### By Task

| I want to...              | Go to                                           |
| ------------------------- | ----------------------------------------------- |
| Load oscilloscope files   | 01_data_loading/01_oscilloscopes.py             |
| Decode UART/SPI/I2C       | 03_protocol_decoding/01_serial_comprehensive.py |
| Analyze power supply      | 04_advanced_analysis/02_power_analysis.py       |
| Reverse engineer protocol | 06_reverse_engineering/01_unknown_protocol.py   |
| Create custom measurement | 08_extensibility/02_custom_measurement.py       |
| Build signal generator    | 17_signal_generation/01_signal_builder.py       |
| Check EMC compliance      | 05_domain_specific/02_emc_compliance.py         |

### By Capability

Run the capability index to find demonstrations by API function:

```bash
python demonstrations/capability_index.py | grep "your_function"
```

---

## Demonstration Format

Every demonstration follows a **standard template**:

```python
"""Title: One-line description

Demonstrates:
- Capability 1 (oscura.function)
- Capability 2 (oscura.class)

IEEE Standards: [if applicable]
Related Demos: [cross-references]
"""

from demonstrations.common import BaseDemo
from oscura import ...

class MyDemo(BaseDemo):
    def __init__(self):
        super().__init__(
            name="my_demo",
            description="Demonstrates X",
            capabilities=["oscura.func1", "oscura.func2"],
        )

    def generate_test_data(self) -> dict:
        # Self-contained synthetic data
        return {"signal": ...}

    def run_demonstration(self, data: dict) -> dict:
        # Demonstration logic
        self.section("Part 1")
        results = {}
        # ... code ...
        return results

    def validate(self, results: dict) -> bool:
        # Validation
        return True

if __name__ == "__main__":
    demo = MyDemo()
    success = demo.execute()
    exit(0 if success else 1)
```

**All demonstrations:**

- ✅ Execute without external files (self-contained)
- ✅ Print "DEMONSTRATION PASSED" on success
- ✅ Include timing and validation
- ✅ Cross-reference related demonstrations
- ✅ Document IEEE standards (if applicable)

---

## Validation

Comprehensive validation ensures quality:

```bash
# Validate all demonstrations
python demonstrations/validate_all.py

# Validate specific section
python demonstrations/validate_all.py --section 00_getting_started

# Verbose output
python demonstrations/validate_all.py --verbose

# Fast mode (skip slow demos)
python demonstrations/validate_all.py --fast
```

**Quality Metrics:**

- 100% pass rate required
- < 60 seconds per demo (unless documented)
- Zero external dependencies beyond oscura
- Complete validation coverage

---

## Coverage Tracking

Generate comprehensive coverage reports:

```bash
# Full capability index
python demonstrations/capability_index.py

# Markdown report
python demonstrations/capability_index.py --format markdown --output INDEX.md

# Show only gaps
python demonstrations/capability_index.py --gaps-only
```

**Coverage Goals:**

- ✅ 100% module coverage (45/45 modules)
- ✅ 100% API symbol coverage (813/813 symbols)
- ✅ All IEEE standards validated
- ✅ All protocols demonstrated

---

## Data Generation

Test data is synthetic and reproducible:

```bash
# Generate all test data
python demonstrations/generate_all_data.py

# Generate for specific section
python demonstrations/generate_all_data.py --section 01_data_loading

# Quick generation (minimal data)
python demonstrations/generate_all_data.py --quick
```

**Why not in git?**

- All data is 100% synthetic and reproducible
- Reduces repository size significantly
- Faster clones for all users
- No large binary files in version control

---

## Contributing

### Adding a New Demonstration

1. Choose appropriate section (or create new numbered section)
2. Use the template from `demonstrations/common/base_demo.py`
3. Ensure self-contained (no external files)
4. Add validation that prints "DEMONSTRATION PASSED"
5. Update section README.md
6. Run validation: `python demonstrations/validate_all.py --section <section>`
7. Update capability index
8. Create PR

### Quality Checklist

- [ ] Follows BaseDemo template
- [ ] Self-contained test data
- [ ] Prints "DEMONSTRATION PASSED"
- [ ] < 60 seconds execution (or documented)
- [ ] Type hints on all functions
- [ ] Google-style docstrings
- [ ] Cross-references related demos
- [ ] Documents IEEE standards (if applicable)
- [ ] Added to section README
- [ ] Passes `validate_all.py`
- [ ] Coverage index updated

---

## IEEE Standards Compliance

Demonstrations validate against industry standards:

| Standard                | Section                 | Coverage |
| ----------------------- | ----------------------- | -------- |
| IEEE 181-2011 (Pulse)   | 19_standards_compliance | Full     |
| IEEE 1241-2010 (ADC)    | 19_standards_compliance | Full     |
| IEEE 1459-2010 (Power)  | 19_standards_compliance | Full     |
| IEEE 2414-2020 (Jitter) | 19_standards_compliance | Full     |
| ISO 11898 (CAN)         | 05_domain_specific      | Full     |
| ISO 14229 (UDS)         | 05_domain_specific      | Full     |
| CISPR 32 (EMC)          | 05_domain_specific      | Full     |
| FCC Part 15             | 05_domain_specific      | Full     |

---

## Performance

Typical execution times:

- Single demo: < 60 seconds
- Full validation: < 10 minutes
- Data generation: < 5 minutes
- Coverage analysis: < 10 seconds

**Optimization tips:**

- Use `--fast` for quick validation
- Generate data once, reuse multiple times
- Run specific sections during development

---

## Support

- **Issues:** Found a bug in a demo? [Report it](https://github.com/oscura-re/oscura/issues)
- **Questions:** Need help? [Discussions](https://github.com/oscura-re/oscura/discussions)
- **Contribute:** Add your use case as a demonstration!

---

**Last Updated:** 2026-01-22
**Status:** Production Ready
**Coverage:** 100% of Oscura capabilities
