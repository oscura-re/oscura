"""WaveDrom Timing Diagrams: Digital Signal Visualization

Demonstrates:
- WaveDrom timing diagram generation
- Digital signal visualization
- Timing annotation and labels
- Protocol-specific diagrams
- Export to SVG/PNG formats

This demonstration shows how to generate WaveDrom timing diagrams
from digital signals for documentation and presentation purposes.

Note: WaveDrom is a JavaScript-based timing diagram tool. This demo
generates WaveDrom JSON that can be rendered using wavedrom-cli or
the WaveDrom web editor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import DigitalTrace, TraceMetadata


class WaveDromTimingDemo(BaseDemo):
    """Demonstrate WaveDrom timing diagram generation."""

    def __init__(self) -> None:
        """Initialize WaveDrom timing demonstration."""
        super().__init__(
            name="wavedrom_timing",
            description="Generate WaveDrom timing diagrams from digital signals",
            capabilities=[
                "visualization.wavedrom",
                "visualization.timing_diagrams",
                "export.wavedrom_json",
            ],
            related_demos=[
                "03_protocol_decoding/01_serial_comprehensive.py",
                "15_export_visualization/04_report_generation.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate digital signals for timing diagrams.

        Returns:
            Dictionary containing digital traces
        """
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1 ms
        num_samples = int(duration * sample_rate)

        # Clock signal
        t = np.arange(num_samples) / sample_rate
        clock_freq = 100e3  # 100 kHz
        clock = ((t * clock_freq) % 1.0) < 0.5

        # Data signal (some pattern)
        data = np.zeros(num_samples, dtype=bool)
        data[100:200] = True
        data[300:350] = True
        data[500:700] = True

        # Enable signal
        enable = np.zeros(num_samples, dtype=bool)
        enable[150:750] = True

        clock_trace = DigitalTrace(
            data=clock,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="CLK"),
        )

        data_trace = DigitalTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="DATA"),
        )

        enable_trace = DigitalTrace(
            data=enable,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="EN"),
        )

        return {
            "clock": clock_trace,
            "data": data_trace,
            "enable": enable_trace,
            "sample_rate": sample_rate,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the WaveDrom timing demonstration."""
        results: dict[str, Any] = {}
        output_dir = self.get_output_dir()

        self.section("WaveDrom Timing Diagram Demonstration")
        self.info("Generate timing diagrams from digital signals")

        clock = data["clock"]
        data_signal = data["data"]
        enable = data["enable"]

        # Part 1: Basic WaveDrom diagram
        self.subsection("Part 1: Basic WaveDrom Diagram")
        self.info("Generate simple clock waveform in WaveDrom format.")

        # Create simplified representation (downsample for clarity)
        downsample = 100
        clock_simple = clock.data[::downsample][:20]

        wavedrom_basic = {
            "signal": [
                {
                    "name": "CLK",
                    "wave": self._trace_to_wave(clock_simple),
                }
            ],
            "config": {"hscale": 2},
        }

        basic_path = output_dir / "wavedrom_basic.json"
        with open(basic_path, "w") as f:
            json.dump(wavedrom_basic, f, indent=2)

        self.result("Basic diagram saved", str(basic_path))
        self.info(f"\nWaveDrom wave string: {wavedrom_basic['signal'][0]['wave']}")

        results["basic_path"] = str(basic_path)

        # Part 2: Multi-signal diagram
        self.subsection("Part 2: Multi-Signal Timing Diagram")
        self.info("Create diagram with multiple signals and annotations.")

        data_simple = data_signal.data[::downsample][:20]
        enable_simple = enable.data[::downsample][:20]

        wavedrom_multi = {
            "signal": [
                {
                    "name": "CLK",
                    "wave": self._trace_to_wave(clock_simple),
                },
                {
                    "name": "EN",
                    "wave": self._trace_to_wave(enable_simple),
                },
                {
                    "name": "DATA",
                    "wave": self._trace_to_wave(data_simple),
                    "data": ["D0", "D1", "D2", "D3"],
                },
            ],
            "config": {"hscale": 2},
            "head": {"text": "Example Timing Diagram", "tick": 0},
        }

        multi_path = output_dir / "wavedrom_multi.json"
        with open(multi_path, "w") as f:
            json.dump(wavedrom_multi, f, indent=2)

        self.result("Multi-signal diagram saved", str(multi_path))
        results["multi_path"] = str(multi_path)

        # Part 3: Protocol-specific diagram
        self.subsection("Part 3: Protocol-Specific Diagram (SPI)")
        self.info("Generate timing diagram with protocol annotations.")

        spi_diagram = {
            "signal": [
                {"name": "CLK", "wave": "p.......", "period": 2},
                {"name": "MOSI", "wave": "x345x", "data": ["A", "B", "C"]},
                {"name": "MISO", "wave": "x456x", "data": ["Q", "R", "S"]},
                {"name": "CS", "wave": "10...1"},
            ],
            "config": {"hscale": 2},
            "head": {"text": "SPI Transaction", "tick": 0},
        }

        spi_path = output_dir / "wavedrom_spi.json"
        with open(spi_path, "w") as f:
            json.dump(spi_diagram, f, indent=2)

        self.result("SPI diagram saved", str(spi_path))
        results["spi_path"] = str(spi_path)

        # Part 4: Usage instructions
        self.subsection("Part 4: Rendering WaveDrom Diagrams")
        self.info("How to render WaveDrom JSON to images.")

        instructions = """
WaveDrom Rendering Options:

1. WaveDrom Editor (Web):
   - Visit: https://wavedrom.com/editor.html
   - Paste JSON content
   - Export as SVG/PNG

2. wavedrom-cli (Command Line):
   - Install: npm install -g wavedrom-cli
   - Render: wavedrom-cli -i diagram.json -s diagram.svg

3. Python Integration (wavedrom package):
   - Install: pip install wavedrom
   - Use in Python scripts for automation

4. Markdown Documentation:
   - Many Markdown renderers support WaveDrom
   - Embed in documentation with code blocks
        """

        usage_path = output_dir / "wavedrom_usage.txt"
        with open(usage_path, "w") as f:
            f.write(instructions)

        self.info("\nUsage instructions:")
        for line in instructions.split("\n")[:10]:
            self.info(f"  {line}")
        self.info("  ...")

        self.result("Usage instructions", str(usage_path))
        results["usage_path"] = str(usage_path)

        # Part 5: Summary
        self.subsection("Part 5: WaveDrom Format Summary")
        self.info("Common WaveDrom wave characters:")

        wave_chars = {
            "p": "Clock (positive edge)",
            "n": "Clock (negative edge)",
            "0": "Logic low",
            "1": "Logic high",
            "x": "Don't care / Unknown",
            "=": "Data value",
            ".": "Continue previous state",
            "|": "Gap in signal",
        }

        self.info("\nWave character reference:")
        for char, description in wave_chars.items():
            self.info(f"  '{char}' : {description}")

        results["wave_chars"] = wave_chars

        self.success("WaveDrom timing diagram demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating WaveDrom diagrams...")

        # Validate files exist
        required_files = ["basic_path", "multi_path", "spi_path", "usage_path"]
        for file_key in required_files:
            if file_key not in results:
                self.error(f"Missing {file_key}")
                return False

            file_path = Path(results[file_key])
            if not file_path.exists():
                self.error(f"File not found: {file_path}")
                return False

        # Validate JSON structure
        basic_path = Path(results["basic_path"])
        with open(basic_path) as f:
            wavedrom_data = json.load(f)

        if "signal" not in wavedrom_data:
            self.error("Invalid WaveDrom structure: missing 'signal'")
            return False

        self.success("All WaveDrom validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - WaveDrom: Standard format for timing diagrams")
        self.info("  - Simple JSON structure: signal name + wave string")
        self.info("  - Multiple rendering options: web, CLI, Python")
        self.info("  - Ideal for documentation and presentations")

        return True

    def _trace_to_wave(self, digital_data: np.ndarray) -> str:
        """Convert digital trace to WaveDrom wave string.

        Args:
            digital_data: Boolean array

        Returns:
            WaveDrom wave string
        """
        wave = []
        prev_state = None

        for value in digital_data:
            current_state = "1" if value else "0"

            if current_state == prev_state:
                wave.append(".")
            else:
                wave.append(current_state)

            prev_state = current_state

        return "".join(wave)


if __name__ == "__main__":
    demo: WaveDromTimingDemo = WaveDromTimingDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
