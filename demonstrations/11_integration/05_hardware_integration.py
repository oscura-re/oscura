"""Hardware Integration: Real-time data acquisition patterns

Demonstrates:
- Hardware abstraction layer patterns
- Real-time data acquisition simulation
- Hardware configuration management
- Error handling for hardware failures
- Mock hardware sources for testing

IEEE Standards: N/A
Related Demos:
- 01_data_loading/01_loading_waveforms.py
- 02_basic_analysis/01_waveform_measurements.py
- 04_configuration_files.py

This demonstrates patterns for integrating Oscura with hardware sources
like oscilloscopes, logic analyzers, and network interfaces. Uses mock
hardware for demonstration since actual hardware may not be available.

Note: This demo uses mock/simulated hardware since actual hardware
(SocketCAN, Saleae, oscilloscopes) may not be present. The patterns
shown are production-ready and can be adapted to real hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_sine_wave,
    validate_exists,
)


class HardwareIntegrationDemo(BaseDemo):
    """Demonstrates hardware integration patterns."""

    def __init__(self):
        """Initialize hardware integration demonstration."""
        super().__init__(
            name="hardware_integration",
            description="Real-time hardware data acquisition patterns",
            capabilities=[
                "oscura.loaders.load",
                "oscura.core.WaveformTrace",
                "oscura.core.Metadata",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate simulated hardware data."""
        # Simulate oscilloscope capture
        osc_trace = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
        )

        return {"osc_trace": osc_trace}

    def run_demonstration(self, data: dict) -> dict:
        """Demonstrate hardware integration patterns."""
        import numpy as np

        from oscura.core import TraceMetadata, WaveformTrace

        _osc_trace = data["osc_trace"]  # Available for hardware demos

        self.section("1. Hardware Abstraction Layer")
        self.info("Define abstract interface for hardware sources")

        # Base hardware source interface
        class HardwareSource:
            """Abstract base class for hardware data sources.

            Defines the interface all hardware sources must implement.
            """

            def connect(self) -> bool:
                """Connect to hardware device.

                Returns:
                    True if connection successful
                """
                raise NotImplementedError

            def disconnect(self) -> None:
                """Disconnect from hardware device."""
                raise NotImplementedError

            def configure(self, config: dict) -> None:
                """Configure hardware parameters.

                Args:
                    config: Configuration dictionary
                """
                raise NotImplementedError

            def acquire(self, duration: float) -> WaveformTrace:
                """Acquire data from hardware.

                Args:
                    duration: Acquisition duration in seconds

                Returns:
                    WaveformTrace with acquired data
                """
                raise NotImplementedError

            def get_status(self) -> dict:
                """Get hardware status.

                Returns:
                    Status dictionary
                """
                raise NotImplementedError

        self.info("HardwareSource abstract interface defined")

        self.section("2. Mock Oscilloscope Implementation")
        self.info("Simulate oscilloscope for demonstration")

        class MockOscilloscope(HardwareSource):
            """Mock oscilloscope for testing.

            Simulates a real oscilloscope without requiring hardware.
            """

            def __init__(self, device_id: str = "MOCK-OSC-001"):
                """Initialize mock oscilloscope.

                Args:
                    device_id: Device identifier
                """
                self.device_id = device_id
                self.connected = False
                self.sample_rate = 1e6  # 1 MSa/s
                self.channel_count = 4
                self.vertical_scale = 1.0  # V/div
                self.timebase = 1e-3  # s/div

            def connect(self) -> bool:
                """Connect to mock oscilloscope."""
                time.sleep(0.1)  # Simulate connection delay
                self.connected = True
                return True

            def disconnect(self) -> None:
                """Disconnect from mock oscilloscope."""
                self.connected = False

            def configure(self, config: dict) -> None:
                """Configure oscilloscope parameters."""
                if not self.connected:
                    raise RuntimeError("Oscilloscope not connected")

                self.sample_rate = config.get("sample_rate", self.sample_rate)
                self.vertical_scale = config.get("vertical_scale", self.vertical_scale)
                self.timebase = config.get("timebase", self.timebase)

            def acquire(self, duration: float) -> WaveformTrace:
                """Acquire simulated data."""
                if not self.connected:
                    raise RuntimeError("Oscilloscope not connected")

                # Generate simulated data
                num_samples = int(duration * self.sample_rate)
                t = np.linspace(0, duration, num_samples, endpoint=False)

                # Simulate 1 kHz sine wave with noise
                data = np.sin(2 * np.pi * 1000 * t) * self.vertical_scale
                data += np.random.normal(0, 0.01, num_samples)

                metadata = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="CH1",
                    source_file=f"{self.device_id}:CH1",
                )

                return WaveformTrace(data=data, metadata=metadata)

            def get_status(self) -> dict:
                """Get oscilloscope status."""
                return {
                    "device_id": self.device_id,
                    "connected": self.connected,
                    "sample_rate": self.sample_rate,
                    "channels": self.channel_count,
                    "vertical_scale": self.vertical_scale,
                    "timebase": self.timebase,
                }

        # Demonstrate oscilloscope usage
        self.subsection("Using Mock Oscilloscope")
        osc = MockOscilloscope()

        # Connect
        self.info("Connecting to oscilloscope...")
        connected = osc.connect()
        self.success(f"Connected: {connected}")

        # Configure
        self.info("Configuring acquisition parameters...")
        config = {"sample_rate": 100e3, "vertical_scale": 1.0, "timebase": 1e-3}
        osc.configure(config)
        self.success("Configuration applied")

        # Get status
        status = osc.get_status()
        for key, value in status.items():
            self.result(key, value)

        # Acquire data
        self.info("Acquiring data...")
        acquired_trace = osc.acquire(duration=0.01)
        self.result("Samples acquired", len(acquired_trace.data))

        # Disconnect
        osc.disconnect()
        self.info("Disconnected")

        self.section("3. Mock Logic Analyzer Implementation")
        self.info("Simulate logic analyzer (e.g., Saleae)")

        class MockLogicAnalyzer(HardwareSource):
            """Mock logic analyzer for testing."""

            def __init__(self, device_id: str = "MOCK-LA-001"):
                """Initialize mock logic analyzer.

                Args:
                    device_id: Device identifier
                """
                self.device_id = device_id
                self.connected = False
                self.sample_rate = 10e6  # 10 MSa/s
                self.digital_channels = 8
                self.voltage_threshold = 1.65  # V

            def connect(self) -> bool:
                """Connect to mock logic analyzer."""
                time.sleep(0.1)
                self.connected = True
                return True

            def disconnect(self) -> None:
                """Disconnect from mock logic analyzer."""
                self.connected = False

            def configure(self, config: dict) -> None:
                """Configure logic analyzer parameters."""
                if not self.connected:
                    raise RuntimeError("Logic analyzer not connected")

                self.sample_rate = config.get("sample_rate", self.sample_rate)
                self.voltage_threshold = config.get("voltage_threshold", self.voltage_threshold)

            def acquire(self, duration: float) -> WaveformTrace:
                """Acquire simulated digital data."""
                if not self.connected:
                    raise RuntimeError("Logic analyzer not connected")

                # Generate simulated UART-like digital signal
                num_samples = int(duration * self.sample_rate)
                data = np.random.choice([0.0, 3.3], size=num_samples)

                metadata = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="D0",
                    source_file=f"{self.device_id}:D0",
                )

                return WaveformTrace(data=data, metadata=metadata)

            def get_status(self) -> dict:
                """Get logic analyzer status."""
                return {
                    "device_id": self.device_id,
                    "connected": self.connected,
                    "sample_rate": self.sample_rate,
                    "digital_channels": self.digital_channels,
                    "voltage_threshold": self.voltage_threshold,
                }

        # Demonstrate logic analyzer
        self.subsection("Using Mock Logic Analyzer")
        la = MockLogicAnalyzer()
        la.connect()
        la.configure({"sample_rate": 10e6, "voltage_threshold": 1.65})
        la_status = la.get_status()
        for key, value in la_status.items():
            self.result(key, value)

        la_trace = la.acquire(duration=0.001)
        self.result("Digital samples acquired", len(la_trace.data))
        la.disconnect()

        self.section("4. Hardware Factory Pattern")
        self.info("Create hardware sources from configuration")

        class HardwareFactory:
            """Factory for creating hardware sources."""

            @staticmethod
            def create(hw_type: str, config: dict) -> HardwareSource:
                """Create hardware source from type and config.

                Args:
                    hw_type: Hardware type (oscilloscope, logic_analyzer, etc.)
                    config: Hardware configuration

                Returns:
                    HardwareSource instance
                """
                if hw_type == "oscilloscope":
                    return MockOscilloscope(device_id=config.get("device_id", "OSC-001"))
                elif hw_type == "logic_analyzer":
                    return MockLogicAnalyzer(device_id=config.get("device_id", "LA-001"))
                else:
                    raise ValueError(f"Unknown hardware type: {hw_type}")

        # Use factory
        self.subsection("Using Hardware Factory")
        hw_config = {"device_id": "TEST-OSC-001"}
        hardware = HardwareFactory.create("oscilloscope", hw_config)
        self.info(f"Created: {type(hardware).__name__}")

        self.section("5. Error Handling for Hardware")
        self.info("Robust error handling for hardware failures")

        def safe_acquire(hardware: HardwareSource, duration: float) -> WaveformTrace | None:
            """Safely acquire data with error handling.

            Args:
                hardware: Hardware source
                duration: Acquisition duration

            Returns:
                WaveformTrace or None on error
            """
            max_retries = 3
            retry_delay = 0.5

            for attempt in range(max_retries):
                try:
                    # Try to connect if not connected
                    status = hardware.get_status()
                    if not status.get("connected", False):
                        hardware.connect()

                    # Acquire data
                    trace = hardware.acquire(duration)
                    return trace

                except RuntimeError as e:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"  Failed after {max_retries} attempts")
                        return None

                except Exception as e:
                    print(f"  Unexpected error: {e}")
                    return None

            return None

        # Demonstrate error handling
        self.subsection("Safe Acquisition with Retry")
        test_hw = MockOscilloscope()
        result = safe_acquire(test_hw, duration=0.01)
        if result:
            self.success(f"Successfully acquired {len(result.data)} samples")
        else:
            self.error("Acquisition failed")

        self.section("6. Real-Time Streaming Pattern")
        self.info("Continuous data acquisition and processing")

        class StreamingAcquisition:
            """Real-time streaming data acquisition."""

            def __init__(self, hardware: HardwareSource, chunk_duration: float = 0.1):
                """Initialize streaming acquisition.

                Args:
                    hardware: Hardware source
                    chunk_duration: Duration of each chunk in seconds
                """
                self.hardware = hardware
                self.chunk_duration = chunk_duration
                self.streaming = False

            def start_stream(self, callback) -> None:
                """Start streaming acquisition.

                Args:
                    callback: Function to call with each chunk
                """
                if not self.hardware.get_status().get("connected", False):
                    self.hardware.connect()

                self.streaming = True
                chunk_count = 0

                while self.streaming and chunk_count < 5:  # Limit for demo
                    trace = self.hardware.acquire(self.chunk_duration)
                    callback(trace, chunk_count)
                    chunk_count += 1

            def stop_stream(self) -> None:
                """Stop streaming acquisition."""
                self.streaming = False

        # Demonstrate streaming
        self.subsection("Streaming Acquisition")

        def process_chunk(trace: WaveformTrace, chunk_id: int) -> None:
            """Process streaming chunk.

            Args:
                trace: Acquired trace chunk
                chunk_id: Chunk identifier
            """
            mean_val = float(np.mean(trace.data))
            print(f"  Chunk {chunk_id}: {len(trace.data)} samples, mean={mean_val:.4f} V")

        stream_hw = MockOscilloscope()
        streamer = StreamingAcquisition(stream_hw, chunk_duration=0.01)
        self.info("Starting stream (5 chunks)...")
        streamer.start_stream(process_chunk)
        streamer.stop_stream()
        stream_hw.disconnect()

        self.section("7. Hardware Configuration Management")
        self.info("Manage hardware settings persistently")

        hardware_config = {
            "oscilloscope": {
                "device_id": "OSC-001",
                "interface": "USB",
                "sample_rate": 100e3,
                "channels": [
                    {"id": "CH1", "enabled": True, "vertical_scale": 1.0, "coupling": "DC"},
                    {"id": "CH2", "enabled": False, "vertical_scale": 2.0, "coupling": "AC"},
                ],
                "trigger": {
                    "source": "CH1",
                    "level": 0.5,
                    "edge": "rising",
                },
            },
            "logic_analyzer": {
                "device_id": "LA-001",
                "interface": "USB",
                "sample_rate": 10e6,
                "channels": {
                    "D0-D7": {"enabled": True, "voltage_threshold": 1.65},
                    "D8-D15": {"enabled": False, "voltage_threshold": 1.65},
                },
            },
        }

        self.subsection("Hardware Configuration")
        import json

        print(json.dumps(hardware_config, indent=2))

        # Save configuration
        output_dir = self.get_output_dir()
        config_path = output_dir / "hardware_config.json"
        config_path.write_text(json.dumps(hardware_config, indent=2))
        self.info(f"Configuration saved to {config_path}")

        self.section("8. Best Practices")
        self.info("Hardware integration recommendations")

        best_practices = [
            ("Use abstraction", "Define common interface for all hardware"),
            ("Handle errors", "Implement retry logic and graceful degradation"),
            ("Mock for testing", "Create mock hardware for CI/CD"),
            ("Connection pooling", "Reuse connections when possible"),
            ("Configuration files", "Store hardware settings externally"),
            ("Status monitoring", "Check hardware status before operations"),
            ("Cleanup resources", "Always disconnect in finally blocks"),
            ("Async operations", "Use async/await for non-blocking acquisition"),
        ]

        for practice, description in best_practices:
            self.result(practice, description)

        self.success("Hardware integration patterns demonstrated!")

        return {
            "oscilloscope": osc,
            "logic_analyzer": la,
            "acquired_trace": acquired_trace,
            "la_trace": la_trace,
            "hardware_config": hardware_config,
            "config_path": str(config_path),
        }

    def validate(self, results: dict) -> bool:
        """Validate hardware integration results."""
        self.info("Validating hardware integration...")

        # Check oscilloscope was created
        if not validate_exists(results.get("oscilloscope"), "oscilloscope"):
            return False

        if type(results.get("oscilloscope")).__name__ != "MockOscilloscope":
            return False

        # Check logic analyzer was created
        if not validate_exists(results.get("logic_analyzer"), "logic_analyzer"):
            return False

        if type(results.get("logic_analyzer")).__name__ != "MockLogicAnalyzer":
            return False

        # Check traces were acquired
        if not validate_exists(results.get("acquired_trace"), "acquired_trace"):
            return False

        if type(results.get("acquired_trace")).__name__ != "WaveformTrace":
            return False

        if not validate_exists(results.get("la_trace"), "la_trace"):
            return False

        # Check hardware config
        if not validate_exists(results.get("hardware_config"), "hardware_config"):
            return False

        config = results["hardware_config"]
        if "oscilloscope" not in config or "logic_analyzer" not in config:
            self.error("Hardware config missing expected sections")
            return False

        # Check config file was saved
        if not validate_exists(results.get("config_path"), "config_path"):
            return False

        self.success("All hardware integration tests passed!")
        self.info("\nNext steps:")
        self.info("  - Adapt patterns for real hardware (SocketCAN, Saleae, PyVISA)")
        self.info("  - Implement hardware-specific drivers")
        self.info("  - Add real-time streaming analysis")
        self.info("  - See demonstrations/README.md for complete overview")

        return True


if __name__ == "__main__":
    demo = HardwareIntegrationDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
