# Intelligent Guidance and Recommendations

**Context-aware analysis guidance system for optimized workflows.**

This section contains 3 demonstrations showing how to use Oscura's intelligent guidance capabilities to automatically detect signal types, recommend analysis methods, and optimize workflow parameters. Perfect for users who want automated assistance in choosing the right analysis approach.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Run `demonstrations/00_getting_started/` first
- **Understanding of Basic Analysis** - Familiarity with `demonstrations/02_basic_analysis/`
- **Signal Types Knowledge** - Know the difference between analog/digital signals
- **Analysis Methods** - Understanding of common measurements (amplitude, frequency, FFT)

Check your setup:

```bash
python demonstrations/00_getting_started/00_hello_world.py
# Should show: ✓ All measurements validated!
```

---

## Demonstrations

| Demo                     | Time       | Difficulty       | Focus                              |
| ------------------------ | ---------- | ---------------- | ---------------------------------- |
| 01_smart_recommendations | 10 min     | Intermediate     | Context-aware analysis suggestions |
| 02_analysis_wizards      | 15 min     | Intermediate     | Interactive workflow assistants    |
| 03_onboarding_helpers    | 10 min     | Beginner         | First-time user guidance           |
| **Total**                | **35 min** | **Intermediate** | **Intelligent guidance**           |

---

## Learning Path

Complete these demonstrations in order for the best learning experience:

```
01_smart_recommendations.py    02_analysis_wizards.py      03_onboarding_helpers.py
         ↓                             ↓                            ↓
Auto signal detection          Interactive wizards         New user guidance
Analysis suggestions          Step-by-step workflows       Getting started help
```

### Estimated Time: 35 minutes

---

## Key Concepts

This section teaches:

1. **Automatic Signal Detection** - How Oscura identifies signal characteristics
2. **Context-Aware Recommendations** - Analysis method suggestions based on signal type
3. **Workflow Optimization** - Automatic parameter tuning for best results
4. **Interactive Wizards** - Step-by-step guidance for complex analyses
5. **Onboarding Assistance** - Help for first-time users and new features

---

## Running Demonstrations

### Option 1: Run Individual Demo

```bash
# From the project root
python demonstrations/13_guidance/01_smart_recommendations.py

# Or from the demo directory
cd demonstrations/13_guidance
python 01_smart_recommendations.py
```

### Option 2: Run All Guidance Demos

```bash
# From the project root
python demonstrations/13_guidance/01_smart_recommendations.py && \
python demonstrations/13_guidance/02_analysis_wizards.py && \
python demonstrations/13_guidance/03_onboarding_helpers.py
```

### Option 3: Validate All Demonstrations

```bash
# From the project root
python demonstrations/validate_all.py
```

---

## What You'll Learn

### Demo 01: Smart Recommendations

**File**: `01_smart_recommendations.py`

**Demonstrates**:

- Automatic signal type detection (analog/digital/mixed)
- Context-aware analysis method recommendations
- Optimal workflow suggestions based on signal characteristics
- Automatic parameter tuning for measurements

**What you'll do**:

1. Analyze signals with automatic type detection
2. Get recommended analysis methods for each signal type
3. See suggested workflows for common reverse engineering tasks
4. Learn how Oscura selects optimal parameters

**Capabilities**:

- `oscura.guidance.signal_detection` - Automatic signal classification
- `oscura.guidance.analysis_recommendations` - Context-aware suggestions
- `oscura.guidance.workflow_suggestions` - Workflow optimization
- `oscura.guidance.parameter_tuning` - Automatic parameter selection

**Related Demos**:

- `12_quality_tools/04_recommendations.py` - Quality-driven recommendations
- `14_exploratory/01_unknown_signals.py` - Unknown signal characterization

---

### Demo 02: Analysis Wizards

**File**: `02_analysis_wizards.py`

**Demonstrates**:

- Interactive step-by-step analysis workflows
- Guided parameter selection with validation
- Multi-stage analysis pipelines
- Progress tracking and intermediate results

**What you'll do**:

1. Use interactive wizards for common analysis tasks
2. Follow guided workflows with real-time feedback
3. See how complex analyses break down into steps
4. Learn best practices through guided execution

**Capabilities**:

- Interactive workflow guidance
- Real-time validation and feedback
- Multi-stage analysis pipelines
- Progress tracking and checkpoints

**Related Demos**:

- `16_complete_workflows/01_protocol_discovery.py` - Complete workflow examples
- `02_basic_analysis/01_waveform_measurements.py` - Basic measurements

---

### Demo 03: Onboarding Helpers

**File**: `03_onboarding_helpers.py`

**Demonstrates**:

- First-time user guidance
- Feature discovery assistance
- Interactive tutorials and help
- Common task quick-starts

**What you'll do**:

1. Experience the new user onboarding flow
2. Discover available features and capabilities
3. Get quick-start templates for common tasks
4. Learn how to find help for specific features

**Capabilities**:

- First-time user onboarding
- Feature discovery
- Interactive help system
- Quick-start templates

**Related Demos**:

- `00_getting_started/00_hello_world.py` - First demonstration
- `13_guidance/01_smart_recommendations.py` - Automated guidance

---

## Troubleshooting

### "No recommendations available"

**Solution**: The signal may not have enough distinguishing characteristics. Try:

```python
# Ensure signal has enough samples
trace = generate_sine_wave(num_samples=1000)  # Minimum 1000 samples

# Ensure signal has variation
trace.data.std() > 0.01  # Should have non-zero standard deviation
```

### "Analysis wizard stuck at step"

**Solution**: Check that prerequisite steps completed successfully:

```python
# Verify each step's output before proceeding
result = wizard.step_1()
assert result.status == "success"
```

### "Guidance system not providing suggestions"

**Solution**: Ensure signal metadata is complete:

```python
metadata = TraceMetadata(
    sample_rate=100000.0,  # Required
    vertical_scale=1.0,    # Required
    channel_name="CH1",    # Helpful for multi-channel
)
```

---

## Next Steps

### If You Want to...

| Goal                                  | Next Demo                                        | Path                     |
| ------------------------------------- | ------------------------------------------------ | ------------------------ |
| Explore unknown signals automatically | `14_exploratory/01_unknown_signals.py`           | Exploratory analysis     |
| Build complete analysis workflows     | `16_complete_workflows/01_protocol_discovery.py` | Complete workflows       |
| Learn quality assurance practices     | `12_quality_tools/01_measurement_validation.py`  | Quality tools            |
| Export and visualize results          | `15_export_visualization/01_export_formats.py`   | Export and visualization |

### Recommended Next Sections

1. **Exploratory Analysis** (14_exploratory/)
   - Unknown signal characterization
   - Fuzzy pattern matching
   - Signal recovery techniques

2. **Export and Visualization** (15_export_visualization/)
   - Multiple export formats
   - Visualization galleries
   - Report generation

3. **Complete Workflows** (16_complete_workflows/)
   - End-to-end reverse engineering examples
   - Production-ready workflows
   - Real-world case studies

---

## Understanding Intelligent Guidance

### How Signal Detection Works

Oscura's guidance system analyzes signal characteristics to determine:

1. **Signal Type** - Analog, digital, or mixed-signal
2. **Frequency Content** - Dominant frequencies and harmonics
3. **Modulation** - AM, FM, or unmodulated
4. **Protocol Likelihood** - Probability of containing encoded data

### Recommendation Algorithm

The recommendation engine considers:

- **Signal characteristics** - Type, frequency, amplitude
- **Analysis goals** - What you're trying to discover
- **Performance** - Computational efficiency
- **Accuracy** - Measurement precision requirements

### Workflow Optimization

Intelligent workflows:

1. **Detect signal type** automatically
2. **Recommend methods** based on characteristics
3. **Tune parameters** for optimal results
4. **Validate outputs** at each stage
5. **Suggest next steps** based on findings

---

## Best Practices

### Using Smart Recommendations

1. **Provide Complete Metadata** - More information enables better recommendations
2. **Start with Auto-Detection** - Let the system characterize first
3. **Review Suggestions** - Understand why methods are recommended
4. **Iterate** - Refine analysis based on initial results

### Working with Wizards

1. **Follow Step Order** - Wizards are optimized for sequential execution
2. **Validate Each Step** - Check intermediate results before proceeding
3. **Save Checkpoints** - Save state at key milestones
4. **Learn the Pattern** - Understand the workflow for future manual use

### Onboarding Tips

1. **Run Getting Started First** - Foundation is critical
2. **Explore Features** - Use discovery tools to learn capabilities
3. **Try Examples** - Quick-start templates teach patterns
4. **Ask for Help** - Built-in help system is comprehensive

---

## Tips for Success

### Maximize Recommendation Quality

Provide rich signal metadata:

```python
metadata = TraceMetadata(
    sample_rate=100000.0,
    vertical_scale=1.0,
    channel_name="UART_TX",      # Descriptive names help
    source_file="device_comms",  # Context aids recommendations
)
```

### Use Wizards for Learning

Even if you know the analysis, use wizards to learn best practices:

```python
# Wizard shows optimal parameter selection
wizard = AnalysisWizard(trace)
recommendations = wizard.suggest_parameters()
# Learn what values experts would choose
```

### Leverage Onboarding for New Features

When Oscura adds new capabilities, use onboarding helpers to discover them:

```python
# Discover newly added features
onboarding.show_new_features(since_version="0.5.0")
```

---

## Summary

The Intelligent Guidance section covers:

| Demo                     | Focus                          | Outcome                |
| ------------------------ | ------------------------------ | ---------------------- |
| 01_smart_recommendations | Auto-detection and suggestions | Context-aware analysis |
| 02_analysis_wizards      | Interactive workflows          | Guided execution       |
| 03_onboarding_helpers    | First-time user guidance       | Quick start assistance |

After completing these 35-minute demonstrations, you'll understand:

- How to use automatic signal detection for analysis selection
- How to leverage context-aware recommendations
- How to work with interactive analysis wizards
- How to get help as a new user or with new features
- How to optimize workflows based on signal characteristics

**Ready to start?** Run this to see intelligent recommendations:

```bash
python demonstrations/13_guidance/01_smart_recommendations.py
```

Happy analyzing with automated guidance!
