# Advanced Inference

> Bayesian inference, Active Learning (L\*), Protocol DSL, sequence alignment

**Oscura Version**: 0.3.0 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers advanced machine learning and inference techniques for protocol analysis. Oscura provides Bayesian inference, active learning for automaton discovery, and a protocol definition DSL.

### Key Capabilities

- Bayesian inference with uncertainty quantification
- Active learning (L\* algorithm)
- Protocol definition DSL
- Global/local sequence alignment
- Multiple sequence alignment
- Stream reassembly (TCP/UDP)
- Signal quality assessment

---

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 14`)

### 30-Second Example

```python
from oscura.inference.state_machine import infer_rpni, minimize_dfa

positive_traces = [
    ["INIT", "READ", "ACK", "DONE"],
    ["INIT", "WRITE", "ACK", "DONE"],
]
negative_traces = [["INIT", "DONE"], ["READ", "INIT"]]

automaton = infer_rpni(positive_traces, negative_traces)
minimized = minimize_dfa(automaton)
dot_output = automaton.to_dot()

print(f"States: {len(automaton.states)}")
print(f"Transitions: {len(automaton.transitions)}")
```

---

## Demo Scripts

| Script                  | Purpose                      | Complexity   |
| ----------------------- | ---------------------------- | ------------ |
| `bayesian_inference.py` | Uncertainty quantification   | Advanced     |
| `active_learning.py`    | L\* algorithm                | Advanced     |
| `protocol_dsl.py`       | Protocol definition language | Intermediate |
| `sequence_alignment.py` | Global/local/multiple        | Intermediate |
| `stream_reassembly.py`  | TCP/UDP reconstruction       | Intermediate |
| `signal_quality.py`     | Automated signal assessment  | Basic        |

## Related Demos

- [11_protocol_inference](../11_protocol_inference/) - Basic inference techniques
- [15_complete_workflows](../15_complete_workflows/) - Full RE workflows
- [03_serial_protocols](../03_serial_protocols/) - Known protocol decoding

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
