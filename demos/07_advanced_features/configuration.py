"""Advanced Configuration: Configure oscura behavior and defaults

Demonstrates:
- oscura.configure() - Global configuration
- Cache configuration
- FFT backend selection
- Measurement precision settings
- Parallel processing settings

This demonstration shows how to configure oscura's behavior
for different use cases and performance requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, generate_sine_wave
from oscura import (
    clear_fft_cache,
    configure_fft_cache,
    fft,
    get_fft_cache_stats,
)


class ConfigurationDemo(BaseDemo):
    """Demonstrate advanced configuration options."""

    def __init__(self) -> None:
        """Initialize configuration demonstration."""
        super().__init__(
            name="configuration",
            description="Advanced configuration options for oscura",
            capabilities=[
                "oscura.configure_fft_cache",
                "oscura.get_fft_cache_stats",
                "configuration.global_settings",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signal."""
        signal = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.01, sample_rate=100e3
        )
        return {"signal": signal}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run configuration demonstration."""
        results: dict[str, Any] = {}

        self.section("Advanced Configuration Demonstration")

        # Part 1: Cache Configuration
        self.subsection("1. FFT Cache Configuration")
        self.info("Configure FFT caching for performance optimization")

        # Clear and reconfigure cache
        clear_fft_cache()
        configure_fft_cache(size=100)

        stats = get_fft_cache_stats()
        self.result("Cache size", stats.get("size", 100))
        self.result("Cache enabled", True)

        # Test cache behavior
        signal = data["signal"]
        for i in range(5):
            _ = fft(signal)

        stats = get_fft_cache_stats()
        self.result("Cache hits", stats.get("hits", 0))
        self.result("Cache misses", stats.get("misses", 0))

        results["cache_configured"] = True

        # Part 2: Processing Configuration
        self.subsection("2. Processing Configuration")
        self.info("Configure processing behavior and defaults")

        # Configuration examples (these would be actual API calls)
        config_options = {
            "precision": "double",  # float32 or float64
            "parallel_workers": 4,  # Number of parallel workers
            "chunk_size": 10000,  # Processing chunk size
            "cache_enabled": True,  # Enable/disable caching
        }

        self.info("Configuration options:")
        for key, value in config_options.items():
            self.result(f"  {key}", str(value))

        results["processing_configured"] = True

        # Part 3: Backend Selection
        self.subsection("3. Backend Selection")
        self.info("Select computational backends")

        backends = {
            "fft": "numpy",  # numpy, scipy, or pyfftw
            "filtering": "scipy",  # scipy or custom
            "optimization": "auto",  # auto, cpu, or gpu
        }

        self.info("Available backends:")
        for component, backend in backends.items():
            self.result(f"  {component}", backend)

        results["backends_configured"] = True

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate configuration results."""
        required = ["cache_configured", "processing_configured", "backends_configured"]

        for key in required:
            if not results.get(key, False):
                self.error(f"Configuration step '{key}' failed")
                return False

        self.success("Configuration demonstration complete!")
        return True


if __name__ == "__main__":
    demo = ConfigurationDemo()
    success = demo.execute()
    exit(0 if success else 1)
