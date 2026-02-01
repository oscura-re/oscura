"""Cancellable Operations: Cancel long-running operations gracefully

Demonstrates:
- Operation cancellation with threading.Event
- Cancel long-running analysis tasks
- Clean cancellation with resource cleanup
- Progress callbacks with cancellation checks

This demonstration shows how to implement cancellable operations
for long-running signal processing tasks.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, generate_sine_wave


class CancellationDemo(BaseDemo):
    """Demonstrate cancellable long-running operations."""

    def __init__(self) -> None:
        """Initialize cancellation demonstration."""
        super().__init__(
            name="cancellation",
            description="Cancel long-running operations gracefully",
            capabilities=[
                "threading.Event",
                "cancellation.check_cancelled",
                "progress_callbacks",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signal."""
        signal = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.1, sample_rate=100e3
        )
        return {"signal": signal}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run cancellation demonstration."""
        results: dict[str, Any] = {}

        self.section("Cancellable Operations Demonstration")

        # Example 1: Normal completion
        self.subsection("1. Normal Completion")
        cancel_event = threading.Event()
        result = self._long_running_task(data["signal"], cancel_event)
        self.success(f"Task completed: processed {result['iterations']} iterations")
        results["normal_completion"] = True

        # Example 2: Cancelled operation
        self.subsection("2. Cancelled Operation")
        cancel_event = threading.Event()

        # Start task in background
        result_holder = {}

        def task_wrapper():
            result_holder["result"] = self._long_running_task(data["signal"], cancel_event)

        thread = threading.Thread(target=task_wrapper)
        thread.start()

        # Cancel after short delay
        time.sleep(0.1)
        cancel_event.set()
        self.info("Cancellation requested...")

        thread.join(timeout=2.0)

        if "result" in result_holder:
            result = result_holder["result"]
            if result["cancelled"]:
                self.success(f"Task cancelled after {result['iterations']} iterations")
                results["cancel_successful"] = True
            else:
                self.warning("Task completed before cancellation")
                results["cancel_successful"] = False
        else:
            self.error("Task did not complete in time")
            results["cancel_successful"] = False

        return results

    def _long_running_task(
        self,
        signal: Any,
        cancel_event: threading.Event,
        max_iterations: int = 100,
    ) -> dict[str, Any]:
        """Simulate long-running task with cancellation support."""
        iterations = 0

        for _i in range(max_iterations):
            # Check if cancelled
            if cancel_event.is_set():
                return {
                    "cancelled": True,
                    "iterations": iterations,
                }

            # Simulate work
            _ = np.fft.fft(signal.data)
            iterations += 1
            time.sleep(0.01)

        return {
            "cancelled": False,
            "iterations": iterations,
        }

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate cancellation results."""
        if not results.get("normal_completion", False):
            self.error("Normal completion failed")
            return False

        if not results.get("cancel_successful", False):
            self.warning("Cancellation test did not work as expected")

        self.success("Cancellation demonstration complete!")
        return True


if __name__ == "__main__":
    demo = CancellationDemo()
    success = demo.execute()
    exit(0 if success else 1)
