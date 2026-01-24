"""Progress Tracking: Real-time Monitoring of Batch Operations

Demonstrates:
- Real-time progress monitoring
- ETA (Estimated Time to Arrival) calculation
- Throughput tracking (files/second)
- Error recovery and resumption
- Progress persistence across sessions

This demonstration shows how to implement comprehensive progress tracking
for long-running batch operations with ETA calculation, throughput metrics,
and the ability to resume from failures.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class ProgressTracker:
    """Progress tracker for batch operations."""

    def __init__(self, total_items: int):
        """Initialize progress tracker.

        Args:
            total_items: Total number of items to process
        """
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.item_times: list[float] = []

    def update(self, success: bool = True, processing_time: float | None = None) -> None:
        """Update progress after processing an item.

        Args:
            success: Whether item processed successfully
            processing_time: Optional processing time for this item
        """
        if success:
            self.completed_items += 1
        else:
            self.failed_items += 1

        if processing_time is not None:
            self.item_times.append(processing_time)

    def get_progress_percent(self) -> float:
        """Get progress percentage.

        Returns:
            Progress as percentage (0-100)
        """
        total_processed = self.completed_items + self.failed_items
        return 100.0 * total_processed / self.total_items if self.total_items > 0 else 0.0

    def get_eta_seconds(self) -> float | None:
        """Estimate time remaining in seconds.

        Returns:
            Estimated seconds remaining, or None if insufficient data
        """
        if not self.item_times:
            return None

        total_processed = self.completed_items + self.failed_items
        remaining = self.total_items - total_processed

        if remaining <= 0:
            return 0.0

        # Use recent average for ETA
        recent_times = self.item_times[-10:] if len(self.item_times) > 10 else self.item_times
        avg_time = np.mean(recent_times)

        return float(avg_time * remaining)

    def get_throughput(self) -> float:
        """Get current throughput in items per second.

        Returns:
            Items processed per second
        """
        elapsed = time.time() - self.start_time
        total_processed = self.completed_items + self.failed_items

        return total_processed / elapsed if elapsed > 0 else 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get progress summary.

        Returns:
            Dictionary with progress metrics
        """
        elapsed = time.time() - self.start_time
        eta = self.get_eta_seconds()

        return {
            "total_items": self.total_items,
            "completed": self.completed_items,
            "failed": self.failed_items,
            "progress_percent": self.get_progress_percent(),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "throughput": self.get_throughput(),
        }


class ProgressTrackingDemo(BaseDemo):
    """Demonstrate progress tracking for batch operations."""

    def __init__(self) -> None:
        """Initialize progress tracking demonstration."""
        super().__init__(
            name="progress_tracking",
            description="Real-time progress monitoring with ETA and throughput tracking",
            capabilities=[
                "batch.progress_tracking",
                "batch.eta_calculation",
                "batch.throughput_monitoring",
                "batch.error_recovery",
            ],
            related_demos=[
                "09_batch_processing/01_parallel_batch.py",
                "09_batch_processing/02_result_aggregation.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test data for progress tracking.

        Returns:
            Dictionary containing:
            - num_files: Number of files to process
            - file_sizes: Simulated file sizes
            - failure_indices: Indices that will fail
        """
        num_files = 30
        file_sizes = np.random.randint(100, 1000, size=num_files)  # KB
        failure_indices = [7, 15, 22]  # Simulate some failures

        return {
            "num_files": num_files,
            "file_sizes": file_sizes.tolist(),
            "failure_indices": failure_indices,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the progress tracking demonstration."""
        results: dict[str, Any] = {}

        self.section("Progress Tracking Demonstration")
        self.info("Real-time monitoring of batch processing operations")

        num_files = data["num_files"]
        file_sizes = data["file_sizes"]
        failure_indices = data["failure_indices"]

        # Part 1: Basic progress tracking
        self.subsection("Part 1: Basic Progress Tracking")
        self.info("Track progress with simple percentage updates.")

        tracker = ProgressTracker(total_items=num_files)
        progress_snapshots = []

        self.info("\nProcessing files:")
        for i in range(num_files):
            # Simulate processing time based on file size
            processing_time = file_sizes[i] / 10000.0  # ~0.01-0.1 seconds

            # Simulate work
            time.sleep(processing_time)

            # Check for failures
            success = i not in failure_indices

            # Update tracker
            tracker.update(success=success, processing_time=processing_time)

            # Report progress every 5 files
            if (i + 1) % 5 == 0:
                summary = tracker.get_summary()
                progress_snapshots.append(summary)

                self.info(
                    f"  Progress: {i + 1}/{num_files} "
                    f"({summary['progress_percent']:.1f}%) - "
                    f"Throughput: {summary['throughput']:.2f} files/sec"
                )

        final_summary = tracker.get_summary()
        results["basic_tracking"] = {
            "summary": final_summary,
            "snapshots": progress_snapshots,
        }

        self.result("Total processed", final_summary["completed"] + final_summary["failed"])
        self.result("Successful", final_summary["completed"])
        self.result("Failed", final_summary["failed"])
        self.result("Elapsed time", f"{final_summary['elapsed_seconds']:.2f}", "seconds")
        self.result("Throughput", f"{final_summary['throughput']:.2f}", "files/sec")

        # Part 2: ETA calculation
        self.subsection("Part 2: ETA Calculation")
        self.info("Estimate time to completion based on recent processing rates.")

        tracker2 = ProgressTracker(total_items=num_files)
        eta_history = []

        self.info("\nProcessing with ETA:")
        for i in range(num_files):
            processing_time = file_sizes[i] / 10000.0
            time.sleep(processing_time)

            success = i not in failure_indices
            tracker2.update(success=success, processing_time=processing_time)

            # Calculate ETA every 5 files
            if (i + 1) % 5 == 0:
                eta = tracker2.get_eta_seconds()
                eta_history.append({"completed": i + 1, "eta_seconds": eta})

                if eta is not None:
                    eta_str = self._format_eta(eta)
                    self.info(f"  Progress: {i + 1}/{num_files} - ETA: {eta_str}")
                else:
                    self.info(f"  Progress: {i + 1}/{num_files} - ETA: Calculating...")

        results["eta_tracking"] = eta_history

        # Part 3: Throughput monitoring
        self.subsection("Part 3: Throughput Monitoring")
        self.info("Monitor processing throughput over time.")

        throughput_history = []
        for snapshot in progress_snapshots:
            throughput_history.append(
                {
                    "completed": snapshot["completed"],
                    "throughput": snapshot["throughput"],
                    "elapsed": snapshot["elapsed_seconds"],
                }
            )

        self.info("\nThroughput over time:")
        for entry in throughput_history[::2]:  # Show every other entry
            self.result(
                f"  After {entry['completed']} files",
                f"{entry['throughput']:.2f}",
                "files/sec",
            )

        results["throughput_history"] = throughput_history

        # Part 4: Error recovery and resumption
        self.subsection("Part 4: Error Recovery and Resumption")
        self.info("Demonstrate resuming from failed processing with checkpoint.")

        # Simulate checkpoint after partial processing
        checkpoint_at = 15
        tracker3 = ProgressTracker(total_items=num_files)

        # First pass: process up to checkpoint
        self.info(f"\nFirst pass (0 to {checkpoint_at}):")
        for i in range(checkpoint_at):
            processing_time = file_sizes[i] / 10000.0
            time.sleep(processing_time)
            success = i not in failure_indices
            tracker3.update(success=success, processing_time=processing_time)

        checkpoint_summary = tracker3.get_summary()
        self.result("Checkpoint at", f"{checkpoint_at}/{num_files}")
        self.result("Completed", checkpoint_summary["completed"])
        self.result("Failed", checkpoint_summary["failed"])

        # Save checkpoint
        checkpoint_data = {
            "last_processed": checkpoint_at - 1,
            "completed": checkpoint_summary["completed"],
            "failed": checkpoint_summary["failed"],
        }

        # Simulate resumption from checkpoint
        self.info(f"\nResuming from checkpoint (file {checkpoint_at}):")
        for i in range(checkpoint_at, num_files):
            processing_time = file_sizes[i] / 10000.0
            time.sleep(processing_time)
            success = i not in failure_indices
            tracker3.update(success=success, processing_time=processing_time)

        resumed_summary = tracker3.get_summary()
        self.result("Total completed", resumed_summary["completed"])
        self.result("Total failed", resumed_summary["failed"])
        self.result("Total time", f"{resumed_summary['elapsed_seconds']:.2f}", "seconds")

        results["checkpoint"] = checkpoint_data
        results["resumed_summary"] = resumed_summary

        # Part 5: Progress persistence
        self.subsection("Part 5: Progress Persistence")
        self.info("Save progress state to disk for cross-session resumption.")

        # Save progress to file
        output_dir = self.get_output_dir()
        progress_file = output_dir / "progress_state.txt"

        with open(progress_file, "w") as f:
            f.write("BATCH PROCESSING PROGRESS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total items: {resumed_summary['total_items']}\n")
            f.write(f"Completed: {resumed_summary['completed']}\n")
            f.write(f"Failed: {resumed_summary['failed']}\n")
            f.write(f"Progress: {resumed_summary['progress_percent']:.1f}%\n")
            f.write(f"Throughput: {resumed_summary['throughput']:.2f} files/sec\n")
            f.write(f"Elapsed: {resumed_summary['elapsed_seconds']:.2f} seconds\n")

        self.success(f"Progress state saved to: {progress_file}")
        results["progress_file"] = str(progress_file)

        self.success("Progress tracking demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating progress tracking...")

        # Validate basic tracking
        if "basic_tracking" not in results:
            self.error("Missing basic tracking results")
            return False

        basic = results["basic_tracking"]["summary"]
        if basic["completed"] + basic["failed"] != basic["total_items"]:
            self.error("Total processed doesn't match total items")
            return False

        # Validate ETA tracking
        if "eta_tracking" not in results:
            self.error("Missing ETA tracking results")
            return False

        if len(results["eta_tracking"]) == 0:
            self.error("No ETA history recorded")
            return False

        # Validate throughput history
        if "throughput_history" not in results:
            self.error("Missing throughput history")
            return False

        # Validate checkpoint and resumption
        if "checkpoint" not in results or "resumed_summary" not in results:
            self.error("Missing checkpoint/resumption results")
            return False

        # Validate progress persistence
        if "progress_file" not in results:
            self.error("Missing progress file")
            return False

        progress_file = Path(results["progress_file"])
        if not progress_file.exists():
            self.error(f"Progress file not found: {progress_file}")
            return False

        self.success("All progress tracking validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - Use ProgressTracker to monitor batch operations")
        self.info("  - Calculate ETA from recent processing times (rolling average)")
        self.info("  - Track throughput to identify performance bottlenecks")
        self.info("  - Implement checkpoints for resumable processing")
        self.info("  - Persist progress state for cross-session recovery")

        return True

    def _format_eta(self, seconds: float) -> str:
        """Format ETA in human-readable format.

        Args:
            seconds: Seconds remaining

        Returns:
            Formatted string (e.g., "2m 30s")
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


if __name__ == "__main__":
    demo: ProgressTrackingDemo = ProgressTrackingDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
