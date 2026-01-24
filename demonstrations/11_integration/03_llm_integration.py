"""LLM Integration: AI-friendly output formats and analysis

Demonstrates:
- Structured JSON output for LLM consumption
- Natural language result summaries
- Question-answering over analysis results
- Claude/ChatGPT integration patterns
- Semantic search over signal data

IEEE Standards: N/A
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 04_advanced_analysis/02_spectral_analysis.py
- 06_reverse_engineering/01_unknown_protocol.py

This demonstrates how to format Oscura outputs for Large Language
Models, enabling AI-assisted signal analysis and interpretation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_complex_signal,
    generate_sine_wave,
    validate_exists,
)


class LLMIntegrationDemo(BaseDemo):
    """Demonstrates LLM-friendly output formats."""

    def __init__(self):
        """Initialize LLM integration demonstration."""
        super().__init__(
            name="llm_integration",
            description="AI-friendly output formats and integration patterns",
            capabilities=[
                "oscura.analyzers.waveform.measurements",
                "oscura.analyzers.waveform.spectral",
                "oscura.inference.detect_protocol",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals for LLM integration."""
        # Create test signal
        signal = generate_complex_signal(
            fundamentals=[1000, 3000],
            amplitudes=[1.0, 0.3],
            duration=0.01,
            sample_rate=100e3,
        )

        # Create UART-like digital signal for protocol detection
        uart_signal = generate_sine_wave(
            frequency=9600.0,  # 9600 baud
            amplitude=3.3,  # 3.3V logic
            duration=0.01,
            sample_rate=100e3,
        )

        return {
            "signal": signal,
            "uart_signal": uart_signal,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Demonstrate LLM integration patterns."""

        from oscura.analyzers.waveform.measurements import (
            amplitude,
            fall_time,
            frequency,
            rise_time,
            rms,
        )
        from oscura.analyzers.waveform.spectral import thd

        signal = data["signal"]
        _uart_signal = data["uart_signal"]  # Available for protocol demonstrations

        self.section("1. Structured JSON Output")
        self.info("Format analysis results for LLM consumption")

        # Collect comprehensive measurements
        measurements = {
            "frequency": frequency(signal),
            "amplitude": amplitude(signal),
            "rms": rms(signal),
            "rise_time": rise_time(signal),
            "fall_time": fall_time(signal),
            "thd": thd(signal),
        }

        # Build structured output
        structured_output = {
            "analysis_type": "waveform_characterization",
            "signal_info": {
                "samples": len(signal.data),
                "sample_rate": signal.metadata.sample_rate,
                "duration": len(signal.data) / signal.metadata.sample_rate,
                "channel": getattr(signal.metadata, "channel_name", "CH1"),
            },
            "measurements": {
                "time_domain": {
                    "frequency_hz": measurements["frequency"],
                    "amplitude_v": measurements["amplitude"],
                    "rms_v": measurements["rms"],
                    "rise_time_s": measurements["rise_time"],
                    "fall_time_s": measurements["fall_time"],
                },
                "frequency_domain": {
                    "thd_db": measurements["thd"],
                },
            },
            "metadata": {
                "analyzer": "oscura",
                "version": "0.5.0",
                "timestamp": "2026-01-22T12:00:00Z",
            },
        }

        # Display JSON
        self.subsection("Structured JSON")
        json_str = json.dumps(structured_output, indent=2)
        print(json_str)

        self.section("2. Natural Language Summaries")
        self.info("Generate human-readable summaries for LLM context")

        def generate_summary(measurements: dict, signal_info: dict) -> str:
            """Generate natural language summary.

            Args:
                measurements: Measurement results
                signal_info: Signal metadata

            Returns:
                Natural language summary string
            """
            freq = measurements["frequency"]
            amp = measurements["amplitude"]
            rms_val = measurements["rms"]
            rt = measurements["rise_time"]
            ft = measurements["fall_time"]
            thd_val = measurements["thd"]

            duration = signal_info["duration"]
            sample_rate = signal_info["sample_rate"]

            summary = f"""
Signal Analysis Summary
=======================

This is a {freq:.1f} Hz signal captured at {sample_rate / 1e6:.1f} MHz sampling rate
over {duration * 1e3:.2f} ms duration.

Amplitude Characteristics:
- Peak-to-peak amplitude: {amp:.3f} V
- RMS voltage: {rms_val:.3f} V
- Form factor: {rms_val / (amp / 2):.3f}

Timing Characteristics:
- Rise time: {rt * 1e9:.2f} ns
- Fall time: {ft * 1e9:.2f} ns
- Edge asymmetry: {abs(rt - ft) / max(rt, ft) * 100:.1f}%

Spectral Quality:
- Total Harmonic Distortion: {thd_val:.1f} dB
- Signal quality: {"Excellent" if thd_val < -40 else "Good" if thd_val < -30 else "Fair"}

Interpretation:
This signal shows {"clean" if thd_val < -40 else "moderate"} spectral characteristics
with {"fast" if rt < 10e-9 else "moderate"} edge transitions suitable for
{"high-speed" if freq > 1e6 else "standard"} digital communications.
            """.strip()

            return summary

        summary = generate_summary(measurements, structured_output["signal_info"])
        print(summary)

        self.section("3. Question-Answering Format")
        self.info("Structure data for Q&A with LLMs")

        # Create Q&A pairs
        qa_pairs = [
            {
                "question": "What is the fundamental frequency of the signal?",
                "answer": f"{measurements['frequency']:.1f} Hz",
                "context": "time_domain_analysis",
            },
            {
                "question": "Is the signal suitable for high-speed communication?",
                "answer": "Yes" if measurements["rise_time"] < 10e-9 else "No",
                "reasoning": f"Rise time is {measurements['rise_time'] * 1e9:.2f} ns, "
                f"which is {'faster' if measurements['rise_time'] < 10e-9 else 'slower'} "
                f"than the 10 ns threshold for high-speed signals.",
            },
            {
                "question": "What is the signal quality based on THD?",
                "answer": (
                    "Excellent"
                    if measurements["thd"] < -40
                    else "Good"
                    if measurements["thd"] < -30
                    else "Fair"
                ),
                "context": f"THD = {measurements['thd']:.1f} dB",
            },
            {
                "question": "Are the rising and falling edges symmetric?",
                "answer": "Yes"
                if abs(measurements["rise_time"] - measurements["fall_time"])
                / max(measurements["rise_time"], measurements["fall_time"])
                < 0.1
                else "No",
                "context": f"Rise: {measurements['rise_time'] * 1e9:.2f} ns, "
                f"Fall: {measurements['fall_time'] * 1e9:.2f} ns",
            },
        ]

        self.subsection("Q&A Pairs")
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\nQ{i}: {qa['question']}")
            print(f"A{i}: {qa['answer']}")
            if "reasoning" in qa:
                print(f"Reasoning: {qa['reasoning']}")
            if "context" in qa:
                print(f"Context: {qa['context']}")

        self.section("4. LLM Prompt Templates")
        self.info("Ready-to-use prompts for AI analysis")

        # Analysis prompt
        analysis_prompt = f"""
You are a signal analysis expert. Analyze the following signal data:

{json.dumps(structured_output, indent=2)}

Please provide:
1. Overall assessment of signal quality
2. Potential applications based on characteristics
3. Any anomalies or areas of concern
4. Recommendations for further analysis
        """.strip()

        self.subsection("Analysis Prompt")
        print(analysis_prompt[:300] + "...")

        # Comparison prompt
        comparison_prompt = """
Compare these two signals and identify key differences:

Signal A:
{signal_a}

Signal B:
{signal_b}

Focus on:
- Frequency differences
- Amplitude variations
- Timing characteristics
- Potential causes of differences
        """.strip()

        self.subsection("Comparison Prompt Template")
        print(comparison_prompt[:200] + "...")

        # Troubleshooting prompt
        troubleshooting_prompt = """
A signal shows the following characteristics:
{signal_data}

Expected characteristics:
{expected_data}

What might be causing the differences? Suggest troubleshooting steps.
        """.strip()

        self.subsection("Troubleshooting Prompt Template")
        print(troubleshooting_prompt)

        self.section("5. Protocol Detection for LLMs")
        self.info("Format protocol detection results for AI interpretation")

        # Simulate protocol detection
        protocol_result = {
            "detected_protocol": "UART",
            "confidence": 0.85,
            "parameters": {
                "baud_rate": 9600,
                "data_bits": 8,
                "parity": "none",
                "stop_bits": 1,
            },
            "frames_detected": 10,
            "sample_frames": [
                {"timestamp": 0.001, "data": "0x48", "ascii": "H"},
                {"timestamp": 0.002, "data": "0x65", "ascii": "e"},
                {"timestamp": 0.003, "data": "0x6C", "ascii": "l"},
            ],
            "interpretation": "ASCII text transmission: 'Hel...'",
        }

        self.subsection("Protocol Detection Output")
        print(json.dumps(protocol_result, indent=2))

        # Natural language protocol summary
        protocol_summary = f"""
Protocol Detection Results:

Detected Protocol: {protocol_result["detected_protocol"]} ({protocol_result["confidence"] * 100:.0f}% confidence)

Configuration:
- Baud rate: {protocol_result["parameters"]["baud_rate"]} bps
- Format: {protocol_result["parameters"]["data_bits"]} data bits, {protocol_result["parameters"]["parity"]} parity, {protocol_result["parameters"]["stop_bits"]} stop bit

Decoded Data:
The signal contains {protocol_result["frames_detected"]} UART frames transmitting ASCII text.
Sample: "{protocol_result["interpretation"]}"
        """.strip()

        self.subsection("Natural Language Protocol Summary")
        print(protocol_summary)

        self.section("6. Semantic Search Context")
        self.info("Structure data for embedding-based search")

        # Create searchable documents
        search_documents = [
            {
                "id": "measurement_001",
                "type": "waveform_measurement",
                "content": f"Signal with {measurements['frequency']:.1f} Hz frequency and {measurements['amplitude']:.3f} V amplitude",
                "metadata": {
                    "frequency": measurements["frequency"],
                    "amplitude": measurements["amplitude"],
                    "timestamp": "2026-01-22T12:00:00Z",
                },
            },
            {
                "id": "measurement_002",
                "type": "timing_analysis",
                "content": f"Rise time {measurements['rise_time'] * 1e9:.2f} ns, fall time {measurements['fall_time'] * 1e9:.2f} ns",
                "metadata": {
                    "rise_time": measurements["rise_time"],
                    "fall_time": measurements["fall_time"],
                },
            },
            {
                "id": "protocol_001",
                "type": "protocol_detection",
                "content": f"UART protocol detected at {protocol_result['parameters']['baud_rate']} baud with ASCII text",
                "metadata": protocol_result,
            },
        ]

        self.subsection("Search Documents")
        for doc in search_documents:
            print(f"\nID: {doc['id']}")
            print(f"Type: {doc['type']}")
            print(f"Content: {doc['content']}")

        self.section("7. Claude/ChatGPT Integration Pattern")
        self.info("Complete integration example")

        integration_code = '''
import anthropic
import oscura as osc

# Load and analyze signal
trace = osc.load("capture.wfm")
measurements = osc.measure(trace)

# Format for LLM
analysis_data = {
    "signal_info": {
        "samples": len(trace.data),
        "sample_rate": trace.metadata.sample_rate,
    },
    "measurements": measurements,
}

# Send to Claude
client = anthropic.Anthropic(api_key="your-api-key")
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"""
Analyze this signal:
{json.dumps(analysis_data, indent=2)}

What protocol might this be? What issues do you see?
        """
    }]
)

print(message.content)
        '''.strip()

        print(integration_code)

        self.success("LLM integration patterns demonstrated!")

        return {
            "structured_output": structured_output,
            "summary": summary,
            "qa_pairs": qa_pairs,
            "protocol_result": protocol_result,
            "search_documents": search_documents,
        }

    def validate(self, results: dict) -> bool:
        """Validate LLM integration results."""
        self.info("Validating LLM integration...")

        # Check structured output
        if not validate_exists(results.get("structured_output"), "structured_output"):
            return False

        output = results["structured_output"]
        required_keys = ["analysis_type", "signal_info", "measurements", "metadata"]
        for key in required_keys:
            if key not in output:
                self.error(f"Missing key in structured output: {key}")
                return False

        # Check summary
        if not validate_exists(results.get("summary"), "summary"):
            return False

        # Check Q&A pairs
        if not validate_exists(results.get("qa_pairs"), "qa_pairs"):
            return False

        if len(results["qa_pairs"]) < 4:
            self.error("Expected at least 4 Q&A pairs")
            return False

        # Check protocol result
        if not validate_exists(results.get("protocol_result"), "protocol_result"):
            return False

        # Check search documents
        if not validate_exists(results.get("search_documents"), "search_documents"):
            return False

        self.success("All LLM integration tests passed!")
        self.info("\nNext steps:")
        self.info("  - Integrate with Claude or ChatGPT")
        self.info("  - Build semantic search over signal data")
        self.info("  - Create AI-assisted analysis workflows")
        self.info("  - See 04_configuration_files.py for config-driven analysis")

        return True


if __name__ == "__main__":
    demo = LLMIntegrationDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
