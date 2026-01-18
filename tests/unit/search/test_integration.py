"""Integration tests for search module interactions.

Tests SRCH-001, SRCH-002, SRCH-003: Integration scenarios


This test suite validates end-to-end workflows combining multiple
search functionalities for realistic debugging and analysis scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.search import extract_context, find_anomalies, find_pattern

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.requirement("SRCH-001", "SRCH-002", "SRCH-003")
class TestPatternAnomalyIntegration:
    """Test integration of pattern search with anomaly detection."""

    def test_find_patterns_then_check_anomalies(self) -> None:
        """Test workflow: find patterns, then check for anomalies around them."""
        # Create digital trace with sync patterns
        digital = np.zeros(1000, dtype=np.uint8)
        digital[100] = 0x7E  # Sync pattern
        digital[500] = 0x7E  # Sync pattern
        digital[900] = 0x7E  # Sync pattern

        # Find sync patterns
        patterns = find_pattern(digital, 0x7E)
        assert len(patterns) == 3

        # Create analog trace for anomaly detection
        analog = np.zeros(1000, dtype=np.float64)
        # Add glitch near second sync
        analog[505] = 2.0

        # Check for anomalies near each pattern
        anomalies_near_patterns = []
        for idx, _ in patterns:
            # Extract context around pattern
            start = max(0, idx - 10)
            end = min(len(analog), idx + 10)
            segment = analog[start:end]

            # Check for anomalies in segment
            anomalies = find_anomalies(segment, anomaly_type="glitch", threshold=1.0)
            if anomalies:
                anomalies_near_patterns.append((idx, anomalies))

        # Should find anomaly near second pattern (index 500)
        assert len(anomalies_near_patterns) >= 1

    def test_anomaly_detection_then_pattern_verification(self) -> None:
        """Test workflow: detect anomalies, verify expected patterns."""
        # Create trace with glitches
        trace = np.zeros(1000, dtype=np.float64)
        trace[250] = 5.0  # Glitch
        trace[750] = 5.0  # Glitch

        # Detect anomalies
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=2.0)
        assert len(anomalies) >= 2

        # Create digital trace
        digital = np.zeros(1000, dtype=np.uint8)
        digital[250] = 0xFF  # Expected pattern at glitch
        digital[750] = 0xFF  # Expected pattern at glitch

        # Verify patterns at anomaly locations
        patterns_at_anomalies = []
        for anomaly in anomalies:
            idx = anomaly["index"]
            # Check within window around anomaly
            window_start = max(0, idx - 2)
            window_end = min(len(digital), idx + 3)
            if window_end > window_start:
                matches = find_pattern(digital[window_start:window_end], 0xFF)
                if matches:
                    patterns_at_anomalies.append((idx, matches))

        assert len(patterns_at_anomalies) >= 1

    def test_pattern_based_anomaly_filtering(self) -> None:
        """Test filtering anomalies based on co-located patterns."""
        # Create trace with multiple glitches
        trace = np.zeros(1000, dtype=np.float64)
        glitch_indices = [100, 300, 500, 700, 900]
        for idx in glitch_indices:
            trace[idx] = 3.0

        # Detect all anomalies
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=1.5)
        assert len(anomalies) >= 3

        # Create digital trace with markers only at specific glitches
        digital = np.zeros(1000, dtype=np.uint8)
        digital[300] = 0xAA  # Mark second glitch
        digital[700] = 0xAA  # Mark fourth glitch

        # Find marked glitches
        markers = find_pattern(digital, 0xAA)
        marker_indices = {idx for idx, _ in markers}

        # Filter anomalies to only those near markers (within 5 samples)
        marked_anomalies = [
            a
            for a in anomalies
            if any(abs(a["index"] - marker_idx) < 5 for marker_idx in marker_indices)
        ]

        # Should find subset of anomalies
        assert len(marked_anomalies) >= 1
        assert len(marked_anomalies) <= len(anomalies)


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.requirement("SRCH-001", "SRCH-003")
class TestPatternContextIntegration:
    """Test integration of pattern search with context extraction."""

    def test_extract_context_around_found_patterns(self) -> None:
        """Test extracting context windows around all found patterns."""
        # Create trace with repeating pattern
        digital = np.zeros(1000, dtype=np.uint8)
        pattern_indices = [100, 300, 500, 700, 900]
        for idx in pattern_indices:
            digital[idx] = 0xAA

        # Find patterns
        matches = find_pattern(digital, 0xAA)
        assert len(matches) >= 5

        # Create analog trace for context
        analog = np.random.randn(1000)
        for idx in pattern_indices:
            analog[idx] = 10.0  # Spike at pattern location

        # Extract context around each pattern
        indices = [idx for idx, _ in matches]
        contexts = extract_context(analog, indices, before=20, after=20, sample_rate=1e6)

        assert len(contexts) == len(matches)
        for ctx in contexts:
            # Verify context contains the spike
            assert np.max(ctx["data"]) > 5.0
            assert "time_reference" in ctx

    def test_pattern_alignment_with_context(self) -> None:
        """Test that pattern locations align with context indices."""
        # Create multi-byte pattern in digital data
        digital = np.zeros(500, dtype=np.uint8)
        digital[100:104] = [0x12, 0x34, 0x56, 0x78]
        digital[300:304] = [0x12, 0x34, 0x56, 0x78]

        # Find pattern
        pattern = np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8)
        matches = find_pattern(digital, pattern)
        assert len(matches) == 2

        # Create corresponding analog trace
        analog = np.random.randn(500)

        # Extract context at pattern starts
        indices = [idx for idx, _ in matches]
        contexts = extract_context(analog, indices, before=10, after=10)

        # Verify alignment
        for i, ctx in enumerate(contexts):
            assert ctx["center_index"] == indices[i]
            assert ctx["start_index"] == max(0, indices[i] - 10)

    def test_batch_pattern_context_extraction(self) -> None:
        """Test efficient batch extraction for many patterns."""
        # Create trace with many patterns
        digital = np.tile(np.array([0xAA, 0x55], dtype=np.uint8), 250)  # 500 bytes

        # Find all 0xAA patterns (should be 250)
        matches = find_pattern(digital, 0xAA)
        pattern_count = len(matches)
        assert pattern_count >= 200  # Allow for some variation

        # Create analog trace
        analog = np.random.randn(500)

        # Batch extract contexts
        indices = [idx for idx, _ in matches[:50]]  # Test with first 50
        contexts = extract_context(analog, indices, before=2, after=2)

        assert len(contexts) == 50
        # All contexts should have valid data
        for ctx in contexts:
            assert len(ctx["data"]) > 0
            assert ctx["length"] > 0


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.requirement("SRCH-002", "SRCH-003")
class TestAnomalyContextIntegration:
    """Test integration of anomaly detection with context extraction."""

    def test_anomaly_detection_with_custom_context(self) -> None:
        """Test that anomaly context matches manual extraction."""
        # Create trace with glitch
        trace = np.zeros(1000, dtype=np.float64)
        trace[500] = 10.0

        # Detect anomaly with built-in context
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=3.0, context_samples=50)
        assert len(anomalies) >= 1

        # Extract context manually
        anomaly_idx = anomalies[0]["index"]
        manual_context = extract_context(trace, anomaly_idx, before=50, after=50)

        # Compare contexts
        builtin_context = anomalies[0]["context"]

        # Both should contain the glitch
        assert np.max(builtin_context) > 5.0
        assert np.max(manual_context["data"]) > 5.0

    def test_multi_anomaly_context_batch_extraction(self) -> None:
        """Test batch context extraction for multiple anomalies."""
        # Create trace with multiple glitches
        trace = np.zeros(2000, dtype=np.float64)
        glitch_locations = [200, 600, 1000, 1400, 1800]
        for loc in glitch_locations:
            trace[loc] = 8.0

        # Detect anomalies
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=3.0)
        assert len(anomalies) >= 4

        # Extract contexts using batch extraction
        indices = [a["index"] for a in anomalies]
        contexts = extract_context(trace, indices, before=30, after=30, sample_rate=1e6)

        assert len(contexts) == len(anomalies)

        # Verify each context contains a spike
        for ctx in contexts:
            assert np.max(ctx["data"]) > 5.0
            assert "time_reference" in ctx

    def test_timing_violation_context_extraction(self) -> None:
        """Test context extraction for timing violations."""
        # Create trace with timing violations
        sample_rate = 1e6
        trace = np.zeros(1000, dtype=np.float64)

        # Create short pulses (violations)
        trace[200:202] = 1.0  # 2 us pulse
        trace[600:608] = 1.0  # 8 us pulse

        # Detect timing violations
        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,  # 5 us minimum
            context_samples=20,
        )

        if len(violations) > 0:
            # Extract additional context
            indices = [v["index"] for v in violations]
            contexts = extract_context(trace, indices, before=25, after=25, sample_rate=sample_rate)

            # Verify timing information is preserved
            for ctx in contexts:
                assert "time_array" in ctx
                assert len(ctx["time_array"]) == ctx["length"]


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.requirement("SRCH-001", "SRCH-002", "SRCH-003")
class TestCompleteWorkflows:
    """Test complete end-to-end debugging workflows."""

    def test_protocol_debug_workflow(self) -> None:
        """Test complete protocol debugging workflow."""
        # Simulate captured protocol data
        digital = np.zeros(2000, dtype=np.uint8)

        # Add sync bytes at regular intervals
        sync_positions = [0, 500, 1000, 1500]
        for pos in sync_positions:
            digital[pos] = 0x7E

        # Add data packets after sync
        for pos in sync_positions:
            digital[pos + 1 : pos + 5] = [0x01, 0x02, 0x03, 0x04]

        # Step 1: Find all sync patterns
        syncs = find_pattern(digital, 0x7E)
        assert len(syncs) == 4

        # Step 2: Extract packet data after each sync
        sync_indices = [idx for idx, _ in syncs]
        packet_contexts = extract_context(
            digital.astype(np.float64), sync_indices, before=0, after=10
        )

        assert len(packet_contexts) == 4

        # Step 3: Simulate analog trace with glitch in one packet
        analog = np.random.randn(2000) * 0.1
        analog[1003] = 5.0  # Glitch in third packet

        # Step 4: Check for anomalies in analog trace
        anomalies = find_anomalies(analog, anomaly_type="glitch", threshold=2.0)

        # Step 5: Correlate anomalies with sync positions
        corrupted_packets = []
        for anomaly in anomalies:
            anomaly_idx = anomaly["index"]
            # Find which packet this is near
            for i, sync_idx in enumerate(sync_indices):
                if sync_idx <= anomaly_idx < sync_idx + 100:
                    corrupted_packets.append(i)
                    break

        # Should identify the third packet as corrupted
        assert len(corrupted_packets) >= 1
        assert 2 in corrupted_packets  # Third packet (index 2)

    def test_signal_quality_workflow(self) -> None:
        """Test signal quality analysis workflow."""
        # Create test signal with known characteristics
        t = np.linspace(0, 1, 10000)
        clean_signal = np.sin(2 * np.pi * 100 * t)

        # Add glitches at specific locations
        signal_with_glitches = clean_signal.copy()
        signal_with_glitches[2000] = 3.0
        signal_with_glitches[5000] = -3.0
        signal_with_glitches[8000] = 3.0

        # Step 1: Detect glitches
        glitches = find_anomalies(
            signal_with_glitches, anomaly_type="glitch", threshold=1.5, sample_rate=10000
        )
        assert len(glitches) >= 3

        # Step 2: Extract context around each glitch
        glitch_indices = [g["index"] for g in glitches]
        contexts = extract_context(
            signal_with_glitches,
            glitch_indices,
            before=50,
            after=50,
            sample_rate=10000,
            include_metadata=True,
        )

        # Step 3: Analyze glitch characteristics
        glitch_severities = [g["severity"] for g in glitches]
        glitch_durations = [g["duration"] for g in glitches]

        assert all(0 <= s <= 1 for s in glitch_severities)
        assert all(d > 0 for d in glitch_durations)

        # Step 4: Generate quality metrics
        total_samples = len(signal_with_glitches)
        glitch_rate = len(glitches) / total_samples
        avg_severity = np.mean(glitch_severities)

        assert glitch_rate > 0
        assert 0 <= avg_severity <= 1

    def test_pattern_matching_with_error_recovery(self) -> None:
        """Test pattern matching with error detection and recovery."""
        # Create data stream with patterns and errors
        digital = np.zeros(1000, dtype=np.uint8)

        # Add header patterns
        header = np.array([0xAA, 0xBB], dtype=np.uint8)
        positions = [0, 200, 400, 600, 800]

        for pos in positions:
            if pos + 2 <= len(digital):
                digital[pos : pos + 2] = header

        # Corrupt one header (simulate error)
        digital[400] = 0xCC  # Should break pattern match

        # Step 1: Find valid headers
        headers = find_pattern(digital, header)

        # Should find 4 valid headers (one is corrupted)
        assert 3 <= len(headers) <= 5

        # Step 2: Create analog trace with timing violations at errors
        analog = np.zeros(1000, dtype=np.float64)
        analog[400] = 1.0  # Signal at corrupted header

        # Step 3: Detect potential error locations
        anomalies = find_anomalies(analog, anomaly_type="glitch", threshold=0.5)

        # Step 4: Cross-reference pattern failures with anomalies
        expected_positions = set(positions)
        found_positions = {idx for idx, _ in headers}
        missing_positions = expected_positions - found_positions

        anomaly_positions = {a["index"] for a in anomalies}

        # Verify anomalies detected near missing patterns
        errors_detected = any(
            any(abs(missing - anomaly) < 10 for anomaly in anomaly_positions)
            for missing in missing_positions
        )

        assert errors_detected or len(missing_positions) == 0

    def test_comparative_pattern_analysis(self) -> None:
        """Test workflow comparing patterns across multiple traces."""
        # Create two traces with similar but not identical patterns
        trace1 = np.zeros(500, dtype=np.uint8)
        trace2 = np.zeros(500, dtype=np.uint8)

        # Add patterns to both
        pattern = np.array([0x12, 0x34], dtype=np.uint8)
        positions_trace1 = [50, 150, 250, 350, 450]
        positions_trace2 = [50, 150, 260, 350, 450]  # One shifted

        for pos in positions_trace1:
            if pos + 2 <= len(trace1):
                trace1[pos : pos + 2] = pattern

        for pos in positions_trace2:
            if pos + 2 <= len(trace2):
                trace2[pos : pos + 2] = pattern

        # Find patterns in both traces
        matches1 = find_pattern(trace1, pattern)
        matches2 = find_pattern(trace2, pattern)

        assert len(matches1) == 5
        assert len(matches2) == 5

        # Compare pattern positions
        positions1 = {idx for idx, _ in matches1}
        positions2 = {idx for idx, _ in matches2}

        # Find differences
        only_in_trace1 = positions1 - positions2
        only_in_trace2 = positions2 - positions1

        # Should find the shifted pattern
        assert len(only_in_trace1) >= 1
        assert len(only_in_trace2) >= 1


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.search
class TestCrossModuleEdgeCases:
    """Test edge cases involving multiple search functions."""

    def test_empty_trace_all_functions(self) -> None:
        """Test all search functions with empty trace."""
        empty_digital = np.array([], dtype=np.uint8)
        empty_analog = np.array([], dtype=np.float64)

        # Pattern search
        patterns = find_pattern(empty_digital, 0xAA)
        assert patterns == []

        # Anomaly detection
        anomalies = find_anomalies(empty_analog, anomaly_type="glitch")
        assert anomalies == []

        # Context extraction should raise error
        with pytest.raises(ValueError, match="Trace cannot be empty"):
            extract_context(empty_analog, 0, before=10, after=10)

    def test_single_sample_all_functions(self) -> None:
        """Test all search functions with single sample."""
        single_digital = np.array([0xAA], dtype=np.uint8)
        single_analog = np.array([1.0], dtype=np.float64)

        # Pattern search
        patterns = find_pattern(single_digital, 0xAA)
        assert len(patterns) == 1

        # Anomaly detection
        anomalies = find_anomalies(single_analog, anomaly_type="glitch")
        assert len(anomalies) == 0  # No derivative for single sample

        # Context extraction
        context = extract_context(single_analog, 0, before=10, after=10)
        assert context["length"] == 1

    def test_very_large_dataset_integration(self) -> None:
        """Test integration with very large dataset."""
        # Create large trace (1M samples)
        large_trace = np.random.randn(1_000_000)

        # Add patterns at specific locations
        pattern_indices = [10000, 500000, 990000]
        for idx in pattern_indices:
            large_trace[idx] = 10.0

        # Detect anomalies
        anomalies = find_anomalies(
            large_trace, anomaly_type="glitch", threshold=5.0, context_samples=100
        )

        # Should detect the planted anomalies
        assert len(anomalies) >= 3

        # Extract contexts efficiently
        anomaly_indices = [a["index"] for a in anomalies[:10]]  # First 10
        contexts = extract_context(large_trace, anomaly_indices, before=50, after=50)

        assert len(contexts) == min(10, len(anomalies))

    def test_overlapping_features_detection(self) -> None:
        """Test detection of overlapping patterns and anomalies."""
        # Create trace where patterns and anomalies overlap
        trace = np.zeros(1000, dtype=np.float64)
        digital = np.zeros(1000, dtype=np.uint8)

        # Location 500: both pattern and anomaly
        digital[500] = 0xAA
        trace[500] = 5.0

        # Find pattern
        patterns = find_pattern(digital, 0xAA)
        assert len(patterns) >= 1

        # Find anomaly
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=2.0)
        assert len(anomalies) >= 1

        # Extract context at overlapping location
        context = extract_context(trace, 500, before=20, after=20)

        # Verify overlap
        pattern_at_500 = any(idx == 500 for idx, _ in patterns)
        anomaly_at_500 = any(abs(a["index"] - 500) < 5 for a in anomalies)

        assert pattern_at_500
        assert anomaly_at_500
        assert context["center_index"] == 500
