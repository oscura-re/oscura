"""Performance and scalability tests for search modules.

Tests SRCH-001, SRCH-002, SRCH-003: Performance characteristics


This test suite validates that search operations scale appropriately
and complete within reasonable time bounds for various dataset sizes.

NOTE: Performance tests with hardcoded time assertions are environment-dependent
and unreliable in CI environments. They are skipped in CI and should be run
on dedicated performance testing infrastructure.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from oscura.search import extract_context, find_anomalies, find_pattern

# Skip performance tests in CI environments (timing is unreliable)
pytestmark = [
    pytest.mark.unit,
    pytest.mark.performance,
    pytest.mark.slow,  # All performance tests are slow (>1s)
    pytest.mark.skipif(
        os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
        reason="Performance tests unreliable in CI - use dedicated benchmark runners",
    ),
]


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.requirement("SRCH-001")
class TestPatternSearchPerformance:
    """Test pattern search performance and scalability."""

    def test_large_trace_pattern_search(self) -> None:
        """Test pattern search scales to large traces (1MB)."""
        # Create 1 million sample trace
        large_trace = np.random.randint(0, 256, 1_000_000, dtype=np.uint8)

        # Plant patterns at known locations
        pattern_locations = [10000, 500000, 900000]
        for loc in pattern_locations:
            large_trace[loc] = 0xAA

        # Search should complete in reasonable time
        start_time = time.time()
        matches = find_pattern(large_trace, 0xAA)
        elapsed = time.time() - start_time

        # Should complete in under 10 seconds (realistic threshold with margin)
        assert elapsed < 10.0, f"Pattern search took {elapsed:.2f}s (expected <10s)"
        # Should find at least the planted patterns
        assert len(matches) >= 3

    def test_many_matches_performance(self) -> None:
        """Test performance when pattern appears frequently."""
        # Create trace with pattern every 100 bytes (10k matches)
        trace = np.zeros(1_000_000, dtype=np.uint8)
        for i in range(0, 1_000_000, 100):
            trace[i] = 0xFF

        start_time = time.time()
        matches = find_pattern(trace, 0xFF)
        elapsed = time.time() - start_time

        # Should handle many matches efficiently (10k matches in large array)
        assert elapsed < 10.0, f"Many matches search took {elapsed:.2f}s (expected <10s)"
        assert len(matches) == 10000

    def test_no_match_worst_case(self) -> None:
        """Test worst-case performance when no matches exist."""
        # Large trace with no matches
        trace = np.zeros(5_000_000, dtype=np.uint8)

        start_time = time.time()
        matches = find_pattern(trace, 0xFF)
        elapsed = time.time() - start_time

        # Should still complete reasonably even with no matches (5M samples worst case)
        assert elapsed < 40.0, f"Worst-case no-match search took {elapsed:.2f}s (expected <40s)"
        assert len(matches) == 0

    def test_multi_byte_pattern_performance(self) -> None:
        """Test performance with multi-byte patterns."""
        # Large trace
        trace = np.random.randint(0, 256, 2_000_000, dtype=np.uint8)

        # Plant 4-byte pattern
        pattern = np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8)
        trace[1000000:1000004] = pattern

        start_time = time.time()
        matches = find_pattern(trace, pattern)
        elapsed = time.time() - start_time

        # Multi-byte pattern search on 2M samples
        assert elapsed < 15.0, f"Multi-byte pattern search took {elapsed:.2f}s (expected <15s)"
        assert len(matches) >= 1

    def test_masked_search_performance(self) -> None:
        """Test performance of wildcard pattern matching."""
        # Large trace
        trace = np.random.randint(0, 256, 1_000_000, dtype=np.uint8)

        # Search with mask (upper nibble only)
        start_time = time.time()
        matches = find_pattern(trace, 0xA0, mask=0xF0)
        elapsed = time.time() - start_time

        # Masked search on 1M samples (more complex than plain search)
        assert elapsed < 10.0, f"Masked pattern search took {elapsed:.2f}s (expected <10s)"
        # Should find approximately 1/16 of samples (upper nibble = 0xA)
        assert 50000 <= len(matches) <= 80000

    def test_analog_conversion_performance(self) -> None:
        """Test performance of analog to digital conversion."""
        # Large analog trace
        analog = np.random.randn(1_000_000)

        start_time = time.time()
        matches = find_pattern(analog, 0xAA, threshold=0.0)
        elapsed = time.time() - start_time

        # Conversion and search should complete quickly
        assert elapsed < 4.0


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.requirement("SRCH-002")
class TestAnomalyDetectionPerformance:
    """Test anomaly detection performance and scalability."""

    def test_large_trace_glitch_detection(self) -> None:
        """Test glitch detection on large traces."""
        # Create 5 million sample trace
        large_trace = np.random.randn(5_000_000) * 0.1

        # Add glitches
        for i in range(10):
            large_trace[500000 * i] = 5.0

        start_time = time.time()
        anomalies = find_anomalies(large_trace, anomaly_type="glitch", threshold=1.0)
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 10.0
        assert len(anomalies) >= 10

    def test_timing_violation_performance(self) -> None:
        """Test timing violation detection performance."""
        # Large trace with pulses
        trace = np.zeros(2_000_000, dtype=np.float64)

        # Add pulses of varying widths
        for i in range(0, 2_000_000, 10000):
            width = (i // 10000) % 20 + 1
            trace[i : i + width] = 1.0

        start_time = time.time()
        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=1e6,
            min_width=5e-6,
            max_width=15e-6,
        )
        elapsed = time.time() - start_time

        # Should handle edge detection efficiently
        assert elapsed < 8.0
        assert len(violations) >= 50

    def test_context_extraction_performance(self) -> None:
        """Test that context extraction doesn't significantly impact performance."""
        trace = np.random.randn(1_000_000)

        # Add glitches
        for i in range(100):
            trace[10000 * i] = 3.0

        # Without context
        start_time = time.time()
        anomalies_no_context = find_anomalies(
            trace, anomaly_type="glitch", threshold=2.0, context_samples=0
        )
        time_no_context = time.time() - start_time

        # With large context
        start_time = time.time()
        anomalies_with_context = find_anomalies(
            trace, anomaly_type="glitch", threshold=2.0, context_samples=200
        )
        time_with_context = time.time() - start_time

        # Context extraction should not more than double execution time
        assert time_with_context < time_no_context * 3
        assert len(anomalies_no_context) == len(anomalies_with_context)

    def test_many_anomalies_performance(self) -> None:
        """Test performance when many anomalies are present."""
        # Create trace with frequent anomalies
        trace = np.random.randn(500_000)
        trace[::100] = 5.0  # Anomaly every 100 samples (5000 total)

        start_time = time.time()
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=2.0)
        elapsed = time.time() - start_time

        # Should handle many anomalies
        assert elapsed < 5.0
        assert len(anomalies) >= 4000

    def test_clean_signal_performance(self) -> None:
        """Test best-case performance with clean signal."""
        # Large clean signal (sine wave)
        t = np.linspace(0, 1, 2_000_000)
        trace = np.sin(2 * np.pi * 100 * t)

        start_time = time.time()
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=1.5)
        elapsed = time.time() - start_time

        # Clean signal should process quickly
        assert elapsed < 3.0
        assert len(anomalies) == 0


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.requirement("SRCH-003")
class TestContextExtractionPerformance:
    """Test context extraction performance and scalability."""

    def test_large_trace_single_extraction(self) -> None:
        """Test extraction from very large trace."""
        # 5 million sample trace
        large_trace = np.random.randn(5_000_000)

        start_time = time.time()
        context = extract_context(large_trace, 2_500_000, before=1000, after=1000)
        elapsed = time.time() - start_time

        # Should be near-instant (slicing operation)
        assert elapsed < 0.2
        assert context["length"] == 2001

    def test_batch_extraction_performance(self) -> None:
        """Test batch extraction efficiency."""
        trace = np.random.randn(2_000_000)

        # Extract contexts at 1000 locations
        indices = list(range(5000, 1_990_000, 2000))

        start_time = time.time()
        contexts = extract_context(trace, indices, before=100, after=100)
        elapsed = time.time() - start_time

        # Batch extraction should be efficient
        assert elapsed < 2.0
        assert len(contexts) == len(indices)

    def test_time_reference_overhead(self) -> None:
        """Test overhead of time reference calculations."""
        trace = np.random.randn(5_000_000)
        indices = list(range(1000, 5_000_000 - 1000, 5000))

        # Without time reference
        start_time = time.time()
        contexts_no_time = extract_context(trace, indices, before=50, after=50)
        time_no_ref = time.time() - start_time

        # With time reference
        start_time = time.time()
        contexts_with_time = extract_context(trace, indices, before=50, after=50, sample_rate=1e6)
        time_with_ref = time.time() - start_time

        # Time reference calculation should be minimal overhead
        assert time_with_ref < time_no_ref * 2
        assert len(contexts_no_time) == len(contexts_with_time)

    def test_large_window_extraction(self) -> None:
        """Test extraction with very large windows."""
        trace = np.random.randn(1_000_000)

        # Extract with 100k sample window
        start_time = time.time()
        context = extract_context(trace, 500_000, before=50_000, after=50_000)
        elapsed = time.time() - start_time

        # Should still be fast (just array slicing)
        assert elapsed < 0.2
        assert context["length"] == 100_001

    def test_metadata_calculation_overhead(self) -> None:
        """Test overhead of metadata calculation."""
        trace = np.random.randn(2_000_000)
        indices = list(range(10000, 1_990_000, 10000))

        # Without metadata
        start_time = time.time()
        contexts_no_meta = extract_context(
            trace, indices, before=50, after=50, include_metadata=False
        )
        time_no_meta = time.time() - start_time

        # With metadata
        start_time = time.time()
        contexts_with_meta = extract_context(
            trace, indices, before=50, after=50, include_metadata=True
        )
        time_with_meta = time.time() - start_time

        # Metadata should add minimal overhead
        assert time_with_meta < time_no_meta * 2
        assert len(contexts_no_meta) == len(contexts_with_meta)


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.stress
class TestStressScenarios:
    """Stress tests for extreme scenarios."""

    def test_maximum_pattern_density(self) -> None:
        """Test pattern search with maximum match density."""
        # Every byte matches
        trace = np.full(500_000, 0xAA, dtype=np.uint8)

        start_time = time.time()
        matches = find_pattern(trace, 0xAA)
        elapsed = time.time() - start_time

        # Should handle maximum density
        assert elapsed < 5.0
        assert len(matches) == 500_000

    def test_continuous_anomalies(self) -> None:
        """Test anomaly detection with continuous anomalies."""
        # Entire trace is anomalous
        trace = np.random.uniform(5, 10, 200_000)

        start_time = time.time()
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=1.0)
        elapsed = time.time() - start_time

        # Should handle worst case
        assert elapsed < 10.0

    def test_boundary_heavy_extraction(self) -> None:
        """Test extraction with many boundary cases."""
        trace = np.random.randn(100_000)

        # Extract at boundaries
        indices = list(range(0, 100, 1)) + list(range(99_900, 100_000, 1))

        start_time = time.time()
        contexts = extract_context(trace, indices, before=200, after=200)
        elapsed = time.time() - start_time

        # Should handle boundary cases efficiently
        assert elapsed < 1.0
        assert len(contexts) == len(indices)

    def test_alternating_pattern_search(self) -> None:
        """Test search in alternating pattern (worst case for some algorithms)."""
        # Create alternating pattern
        trace = np.tile(np.array([0xAA, 0x55], dtype=np.uint8), 500_000)

        start_time = time.time()
        matches = find_pattern(trace, 0xAA)
        elapsed = time.time() - start_time

        # Should handle alternating pattern
        assert elapsed < 3.0
        assert len(matches) == 500_000

    def test_high_frequency_anomaly_detection(self) -> None:
        """Test anomaly detection with high-frequency content."""
        # High frequency signal with anomalies
        t = np.linspace(0, 1, 1_000_000)
        trace = np.sin(2 * np.pi * 100000 * t)  # 100 kHz sine
        trace[::10000] = 5.0  # Periodic anomalies

        start_time = time.time()
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=2.0, sample_rate=1e6)
        elapsed = time.time() - start_time

        # Should handle high frequency content
        assert elapsed < 8.0
        assert len(anomalies) >= 50

    def test_memory_efficiency_large_batch(self) -> None:
        """Test memory efficiency with large batch extraction."""
        trace = np.random.randn(2_000_000)

        # Extract 500 contexts
        indices = list(range(10000, 1_990_000, 4000))

        start_time = time.time()
        contexts = extract_context(trace, indices, before=100, after=100)
        elapsed = time.time() - start_time

        # Should complete efficiently
        assert elapsed < 2.0
        assert len(contexts) == len(indices)

        # Verify contexts are independent copies
        contexts[0]["data"][0] = 999999.0
        assert trace[indices[0] - 100] != 999999.0


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.scalability
class TestScalabilityCharacteristics:
    """Test that algorithms scale linearly with input size."""

    @pytest.mark.parametrize("size", [10_000, 100_000, 1_000_000])
    def test_pattern_search_linear_scaling(self, size: int) -> None:
        """Test that pattern search scales linearly."""
        trace = np.random.randint(0, 256, size, dtype=np.uint8)

        start_time = time.time()
        matches = find_pattern(trace, 0xAA)
        elapsed = time.time() - start_time

        # Should scale roughly linearly
        # 1M samples should take less than 1 second
        expected_time = (size / 1_000_000) * 1.0
        assert elapsed < expected_time * 3  # Allow 3x margin

    @pytest.mark.parametrize("size", [10_000, 100_000, 1_000_000])
    def test_anomaly_detection_linear_scaling(self, size: int) -> None:
        """Test that anomaly detection scales linearly."""
        trace = np.random.randn(size)

        start_time = time.time()
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=3.0)
        elapsed = time.time() - start_time

        # Should scale roughly linearly
        expected_time = (size / 1_000_000) * 2.0
        assert elapsed < expected_time * 3

    @pytest.mark.parametrize("count", [10, 100, 1000])
    def test_batch_extraction_linear_scaling(self, count: int) -> None:
        """Test that batch extraction scales linearly with count."""
        trace = np.random.randn(2_000_000)
        indices = list(range(10000, 1_990_000, 2_000_000 // count))[:count]

        start_time = time.time()
        contexts = extract_context(trace, indices, before=50, after=50)
        elapsed = time.time() - start_time

        # Should scale linearly with number of extractions
        expected_time = (count / 1000) * 0.5
        assert elapsed < expected_time * 3
        assert len(contexts) == count


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.optimization
class TestOptimizationOpportunities:
    """Test to identify potential optimization opportunities."""

    def test_min_spacing_optimization(self) -> None:
        """Test that min_spacing parameter improves performance."""
        # Trace with many adjacent matches
        trace = np.full(1_000_000, 0xAA, dtype=np.uint8)

        # Without spacing (finds all)
        start_time = time.time()
        matches_all = find_pattern(trace, 0xAA, min_spacing=1)
        time_all = time.time() - start_time

        # With large spacing (finds fewer)
        start_time = time.time()
        matches_spaced = find_pattern(trace, 0xAA, min_spacing=1000)
        time_spaced = time.time() - start_time

        # Spaced search should be faster
        assert time_spaced <= time_all * 1.2  # Allow small margin
        assert len(matches_spaced) < len(matches_all)

    def test_threshold_vs_digital_performance(self) -> None:
        """Compare analog conversion vs pre-converted digital."""
        # Create analog trace
        analog = np.random.randn(500_000)

        # Convert to digital first
        digital = (analog >= 0.0).astype(np.uint8)
        digital_packed = np.packbits(digital, bitorder="big")

        # Search in analog (with conversion)
        start_time = time.time()
        matches_analog = find_pattern(analog, 0xAA, threshold=0.0)
        time_analog = time.time() - start_time

        # Search in pre-converted digital
        start_time = time.time()
        matches_digital = find_pattern(digital_packed, 0xAA)
        time_digital = time.time() - start_time

        # Pre-converted should be faster
        assert time_digital <= time_analog

    def test_context_window_size_impact(self) -> None:
        """Test impact of context window size on performance."""
        trace = np.random.randn(1_000_000)
        indices = list(range(10000, 990_000, 10000))

        # Small window
        start_time = time.time()
        contexts_small = extract_context(trace, indices, before=10, after=10)
        time_small = time.time() - start_time

        # Large window
        start_time = time.time()
        contexts_large = extract_context(trace, indices, before=1000, after=1000)
        time_large = time.time() - start_time

        # Both should be similar (slicing is O(1))
        assert time_large < time_small * 3
        assert len(contexts_small) == len(contexts_large)
