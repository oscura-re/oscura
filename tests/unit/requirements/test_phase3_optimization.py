"""Tests for Phase 3 Optimization & Specialized features.

Requirements tested:
- MEM-014, 020, 023, 025, 028, 030-033: Advanced Memory Management
- COMP-005, 006, 007: Advanced Compliance
- FUZZY-004, 005: Advanced Fuzzy Matching
- RPT-002, 003: Report Templates and Batch Reports
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestAdvancedMemoryManagement:
    """Tests for MEM-014, 020, 023, 025, 028, 030-033."""

    def test_quality_mode_config(self) -> None:
        """Test quality mode configuration - MEM-014."""
        from oscura.utils.memory_advanced import (
            QualityMode,
            QualityModeConfig,
            get_quality_config,
            get_quality_mode,
            set_quality_mode,
        )

        # Test preview mode
        config = QualityModeConfig.for_mode(QualityMode.PREVIEW)
        assert config.downsample_factor == 8
        assert config.use_approximations is True

        # Test high quality mode
        config = QualityModeConfig.for_mode(QualityMode.HIGH_QUALITY)
        assert config.downsample_factor == 1
        assert config.use_approximations is False

        # Test balanced mode
        config = QualityModeConfig.for_mode(QualityMode.BALANCED)
        assert config.enable_caching is True

        # Test global quality mode
        set_quality_mode("preview")
        assert get_quality_mode() == QualityMode.PREVIEW
        config = get_quality_config()
        assert config.mode == QualityMode.PREVIEW

    def test_gc_controller(self) -> None:
        """Test garbage collection controller - MEM-020."""
        from oscura.utils.memory_advanced import GCController, gc_aggressive

        controller = GCController()
        assert controller.aggressive is False

        # Enable aggressive mode
        controller.aggressive = True
        assert controller.aggressive is True

        # Test collection
        collected = controller.collect()
        assert isinstance(collected, int)

        # Test stats
        stats = controller.get_stats()
        assert "collection_count" in stats
        assert stats["collection_count"] >= 1

        # Test global function
        gc_aggressive(True)

    def test_wsl_swap_checker(self) -> None:
        """Test WSL swap awareness - MEM-023."""
        from oscura.utils.memory_advanced import WSLSwapChecker, get_wsl_memory_limits

        checker = WSLSwapChecker()
        # is_wsl property should work regardless of environment
        assert isinstance(checker.is_wsl, bool)

        # get_safe_memory should return positive value
        safe_mem = checker.get_safe_memory()
        assert safe_mem > 0

        # Test convenience function
        limits = get_wsl_memory_limits()
        assert "is_wsl" in limits
        assert "safe_memory" in limits

    def test_memory_logger(self) -> None:
        """Test memory usage logging - MEM-025."""
        from oscura.utils.memory_advanced import MemoryLogger

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            log_path = f.name

        try:
            logger = MemoryLogger(log_file=log_path, format="csv")
            logger.enable()

            # Log some operations
            logger.log_operation("test_op_1", duration=0.1)
            logger.log_operation("test_op_2", duration=0.2)

            # Get summary
            summary = logger.get_summary()
            assert summary["entries_logged"] == 2
            assert summary["enabled"] is True

            # Disable and flush
            logger.disable()

            # Check file was created
            assert Path(log_path).exists()
        finally:
            Path(log_path).unlink(missing_ok=True)

    def test_adaptive_measurement_selector(self) -> None:
        """Test adaptive measurement selection - MEM-028."""
        from oscura.utils.memory_advanced import (
            AdaptiveMeasurementSelector,
            adaptive_measurements,
        )

        # Small file - all measurements enabled
        small_selector = AdaptiveMeasurementSelector(file_size_samples=1000)
        assert small_selector.is_enabled("eye_diagram") is True
        assert small_selector.is_enabled("spectrogram") is True

        # Large file - some measurements disabled
        large_selector = AdaptiveMeasurementSelector(file_size_samples=int(2e9))
        assert large_selector.is_enabled("eye_diagram") is False

        # Get recommendations
        recommendations = large_selector.get_recommendations()
        assert "eye_diagram" in recommendations

        # Test with enable_all
        all_selector = AdaptiveMeasurementSelector(file_size_samples=int(2e9), enable_all=True)
        assert all_selector.is_enabled("eye_diagram") is True

        # Test convenience function
        selector = adaptive_measurements(1000)
        assert isinstance(selector, AdaptiveMeasurementSelector)

    def test_cache_invalidation_strategy(self) -> None:
        """Test cache invalidation - MEM-030."""
        from oscura.utils.memory_advanced import CacheInvalidationStrategy

        cache = CacheInvalidationStrategy(max_size=100, default_ttl=3600)

        # Set and get
        cache.set("key1", "value1")
        value, hit = cache.get("key1")
        assert value == "value1"
        assert hit is True

        # Miss
        value, hit = cache.get("nonexistent")
        assert value is None
        assert hit is False

        # Invalidation by key
        assert cache.invalidate("key1") is True
        value, hit = cache.get("key1")
        assert hit is False

        # Source data validation
        data = np.array([1, 2, 3])
        cache.set("key2", "result", source_data=data)
        value, hit = cache.get("key2", source_data=data)
        assert hit is True

        # Changed source data
        new_data = np.array([4, 5, 6])
        value, hit = cache.get("key2", source_data=new_data)
        assert hit is False

        # Stats
        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats

    def test_disk_cache(self) -> None:
        """Test persistent disk cache - MEM-031."""
        from oscura.utils.memory_advanced import DiskCache

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = DiskCache(cache_dir=cache_dir, max_memory_mb=10, max_disk_mb=100, ttl_hours=1.0)

            # Set and get
            cache.set("key1", {"data": [1, 2, 3]})
            value, hit = cache.get("key1")
            assert hit is True
            assert value["data"] == [1, 2, 3]

            # Store numpy array
            arr = np.zeros(1000)
            cache.set("array_key", arr)
            value, hit = cache.get("array_key")
            assert hit is True
            assert len(value) == 1000

            # Clear
            cache.clear()

    def test_backpressure_controller(self) -> None:
        """Test streaming backpressure - MEM-032."""
        from oscura.utils.memory_advanced import BackpressureController

        controller = BackpressureController(buffer_size=10, drop_oldest=True)

        # Initial state
        assert controller.is_paused is False
        assert controller.buffer_usage == 0.0

        # Push data
        for i in range(5):
            assert controller.push(f"data_{i}") is True

        assert controller.buffer_usage == 0.5

        # Pop data
        data = controller.pop()
        assert data == "data_0"

        # Pop all
        all_data = controller.pop_all()
        assert len(all_data) == 4

        # Test overflow (drop oldest)
        for i in range(15):
            controller.push(f"overflow_{i}")

        assert controller.dropped_count > 0

        # Stats
        stats = controller.get_stats()
        assert "buffer_size" in stats
        assert "dropped_count" in stats

    def test_multi_channel_memory_manager(self) -> None:
        """Test multi-channel memory management - MEM-033."""
        from oscura.utils.memory_advanced import MultiChannelMemoryManager

        manager = MultiChannelMemoryManager(max_memory_mb=1024, bytes_per_sample=8)

        # Estimate memory
        estimate = manager.estimate_channel_memory(samples_per_channel=1000000, num_channels=8)
        assert estimate == 1000000 * 8 * 8

        # Check if can load all
        assert manager.can_load_all(samples_per_channel=1000, num_channels=4) is True

        # Get batches for large data
        batches = manager.get_channel_batches(
            samples_per_channel=100000000, channel_indices=list(range(32))
        )
        assert len(batches) > 1

        # Suggest subset
        suggestion = manager.suggest_subset(samples_per_channel=1000000000, total_channels=64)
        assert suggestion["can_load_all"] is False
        assert "recommendation" in suggestion


class TestAdvancedCompliance:
    """Tests for COMP-005, 006, 007."""

    def test_limit_interpolation(self) -> None:
        """Test limit interpolation - COMP-005."""
        from oscura.compliance.advanced import (
            InterpolationMethod,
            LimitInterpolator,
            interpolate_limit,
        )
        from oscura.compliance.masks import load_limit_mask

        mask = load_limit_mask("FCC_Part15_ClassB")
        interp = LimitInterpolator(mask, method=InterpolationMethod.LOG_LINEAR)

        # Interpolate at defined point
        limit = interp.interpolate(np.array([100e6]))
        assert len(limit) == 1

        # Interpolate at edge
        freq_min, _freq_max = mask.frequency_range
        limit_at_edge = interp.interpolate(np.array([freq_min]))
        assert not np.isnan(limit_at_edge[0])

        # Get with metadata
        _limit_value, metadata = interp.get_limit_at(100e6)
        assert "method" in metadata
        assert metadata["method"] == "log-linear"

        # Test error on out-of-range
        no_extrap = LimitInterpolator(mask, extrapolate=False)
        with pytest.raises(ValueError, match="outside mask range"):
            no_extrap.interpolate(np.array([1.5e9]))

        # Test negative frequency error
        with pytest.raises(ValueError, match="positive"):
            interp.interpolate(np.array([-100e6]))

        # Convenience function
        limit = interpolate_limit(mask, 100e6)
        assert len(limit) == 1

    def test_compliance_test_runner(self) -> None:
        """Test compliance test execution - COMP-006."""
        from oscura.compliance.advanced import (
            ComplianceTestRunner,
            ComplianceTestSuite,
            run_compliance_suite,
        )

        # Create test spectrum
        frequencies = np.linspace(30e6, 1e9, 1000)
        # Levels below limit (should pass)
        levels = np.ones(1000) * 20  # 20 dBuV/m

        # Test with runner
        runner = ComplianceTestRunner()
        runner.add_mask("FCC_Part15_ClassB")
        result = runner.run(frequencies, levels)

        assert result.overall_passed is True
        assert len(result.results) == 1
        assert result.summary["total_tests"] == 1

        # Test failing spectrum
        high_levels = np.ones(1000) * 80  # Way above limit
        fail_result = runner.run(frequencies, high_levels)
        assert fail_result.overall_passed is False

        # Test suite
        suite_result = run_compliance_suite(frequencies, levels, suite="residential")
        assert suite_result.overall_passed is True

        # Pre-built suites
        residential = ComplianceTestSuite.residential()
        commercial = ComplianceTestSuite.commercial()
        military = ComplianceTestSuite.military()

        assert residential is not None
        assert commercial is not None
        assert military is not None

    def test_quasi_peak_detector(self) -> None:
        """Test quasi-peak detection - COMP-007."""
        from oscura.compliance.advanced import QPDetectorBand, QuasiPeakDetector

        detector = QuasiPeakDetector()

        # Get band for frequency
        band = detector.get_band(100e6)
        assert band == QPDetectorBand.BAND_C

        band = detector.get_band(1e6)
        assert band == QPDetectorBand.BAND_B

        # Get params
        params = detector.get_params(100e6)
        assert params is not None
        assert params.bandwidth == 120000  # 120 kHz for Band C

        # Apply QP to peak levels
        peak_levels = np.array([50.0, 55.0, 60.0])
        frequencies = np.array([10e6, 100e6, 500e6])
        qp_levels = detector.apply(peak_levels, frequencies)

        # QP should be <= peak
        assert np.all(qp_levels <= peak_levels)

        # Compare peak and QP
        comparison = detector.compare_peak_qp(peak_levels, frequencies)
        assert "difference_db" in comparison
        assert comparison["max_difference_db"] >= 0

        # Get bandwidth for frequency
        bw = detector.get_bandwidth(100e6)
        assert bw == 120000

        # Validate bandwidth
        with pytest.raises(ValueError, match="positive"):
            detector.validate_bandwidth(0)


class TestAdvancedFuzzyMatching:
    """Tests for FUZZY-004, 005."""

    def test_variant_characterization(self) -> None:
        """Test binary pattern variant characterization - FUZZY-004."""
        from oscura.exploratory.fuzzy_advanced import (
            VariationType,
            characterize_variants,
        )

        # Create patterns with some variation
        patterns = [
            b"\x12\x34\x56\x78",
            b"\x12\x35\x56\x78",  # One byte different
            b"\x12\x34\x56\x78",
            b"\x12\x34\x57\x78",  # Different byte
            b"\x12\x34\x56\x78",
        ]

        result = characterize_variants(patterns)

        assert result.pattern_count == 5
        assert result.min_length == 4
        assert len(result.positions) == 4

        # First byte should be constant
        assert result.positions[0].variation_type == VariationType.CONSTANT
        assert result.positions[0].consensus_byte == 0x12

        # Check variable positions
        assert len(result.variable_positions) > 0

        # Check consensus
        assert result.consensus[0] == 0x12

        # Empty patterns
        empty_result = characterize_variants([])
        assert empty_result.pattern_count == 0

    def test_sequence_alignment(self) -> None:
        """Test multiple sequence alignment - FUZZY-005."""
        from oscura.exploratory.fuzzy_advanced import (
            align_sequences,
            align_two_sequences,
        )

        # Test pairwise alignment
        seq1 = b"\x12\x34\x56\x78"
        seq2 = b"\x12\x34\x78"  # Missing 0x56

        aligned1, aligned2, score = align_two_sequences(seq1, seq2)

        # Should have gaps
        assert len(aligned1) == len(aligned2)
        assert score > 0

        # Test local alignment
        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="local")
        assert score >= 0

        # Test MSA
        sequences = [
            b"\x12\x34\x56",
            b"\x12\x56",
            b"\x12\x34\x78",
        ]

        result = align_sequences(sequences)

        assert len(result.sequences) == 3
        assert len(result.conservation_scores) > 0

        # Conservation scores should be valid
        for score in result.conservation_scores:
            assert 0 <= score <= 1

        # Conserved regions
        # At least some regions should exist
        assert isinstance(result.conserved_regions, list)

        # Single sequence
        single_result = align_sequences([b"\x12\x34\x56"])
        assert len(single_result.sequences) == 1

        # Empty input
        empty_result = align_sequences([])
        assert len(empty_result.sequences) == 0

    def test_conservation_scores(self) -> None:
        """Test conservation score computation - FUZZY-005."""
        from oscura.exploratory.fuzzy_advanced import (
            compute_conservation_scores,
        )

        # Identical sequences = perfect conservation
        identical = [b"\x12\x34\x56", b"\x12\x34\x56", b"\x12\x34\x56"]
        scores = compute_conservation_scores(identical)
        assert all(s == 1.0 for s in scores)

        # All different = low conservation
        different = [b"\x12", b"\x34", b"\x56"]
        scores = compute_conservation_scores(different)
        # Each position has 3 different values among 3 sequences
        # Conservation = 1/3 = 0.333...
        for score in scores:
            assert score <= 0.34


class TestDAQGapDetection:
    """Tests for PKT-008."""

    def test_detect_gaps_by_timestamps(self) -> None:
        """Test gap detection using timestamps - PKT-008."""
        from oscura.analyzers.packet.daq import detect_gaps_by_timestamps

        # Create timestamps with a gap
        # Normal 1us intervals with a 5us gap in the middle
        timestamps = np.array([0.0, 1e-6, 2e-6, 7e-6, 8e-6, 9e-6])

        result = detect_gaps_by_timestamps(timestamps, tolerance=0.5)

        assert result.total_gaps >= 1
        assert result.total_missing_samples > 0
        assert result.acquisition_efficiency < 1.0

        # Check gap details
        assert len(result.gaps) >= 1
        gap = result.gaps[0]
        assert gap.gap_type == "timestamp"
        assert gap.missing_samples >= 1

        # No gaps case
        regular = np.arange(0, 10e-6, 1e-6)
        no_gap_result = detect_gaps_by_timestamps(regular, tolerance=0.5)
        assert no_gap_result.total_gaps == 0

        # Too few samples
        short_result = detect_gaps_by_timestamps(np.array([0.0]))
        assert short_result.total_gaps == 0

    def test_detect_gaps_by_samples(self) -> None:
        """Test gap detection using sample discontinuities - PKT-008."""
        from oscura.analyzers.packet.daq import detect_gaps_by_samples

        # Create data with a discontinuity (big jump in value)
        data = np.concatenate(
            [
                np.linspace(0, 1, 100),  # Smooth ramp
                np.linspace(10, 11, 100),  # Sudden jump to 10
            ]
        )

        result = detect_gaps_by_samples(
            data,
            sample_rate=1e6,
            check_discontinuities=True,
        )

        assert result.total_gaps >= 1 or len(result.discontinuities) >= 1

        # Smooth data should have no gaps
        smooth = np.linspace(0, 1, 1000)
        smooth_result = detect_gaps_by_samples(
            smooth,
            sample_rate=1e6,
            check_discontinuities=True,
        )
        # Very smooth data should have few or no detected gaps
        assert smooth_result.total_gaps == 0 or smooth_result.acquisition_efficiency > 0.9

    def test_dag_gap_dataclasses(self) -> None:
        """Test DAQ gap dataclasses - PKT-008."""
        from oscura.analyzers.packet.daq import DAQGap, DAQGapAnalysis

        gap = DAQGap(
            start_index=100,
            end_index=105,
            start_time=1e-3,
            end_time=1.5e-3,
            duration=0.5e-3,
            expected_samples=5,
            missing_samples=4,
            gap_type="timestamp",
        )

        assert gap.duration == 0.5e-3
        assert gap.missing_samples == 4

        analysis = DAQGapAnalysis(
            gaps=[gap],
            total_gaps=1,
            total_missing_samples=4,
            total_gap_duration=0.5e-3,
            acquisition_efficiency=0.96,
            sample_rate=1e6,
            discontinuities=[100],
        )

        assert analysis.total_gaps == 1
        assert analysis.acquisition_efficiency == 0.96


class TestReportTemplates:
    """Tests for RPT-002, 003."""

    def test_report_templates(self) -> None:
        """Test report templates - RPT-002."""
        from oscura.reporting.template_system import (
            list_templates,
            load_template,
            save_template,
        )

        # List templates
        templates = list_templates()
        assert "default" in templates
        assert "compliance" in templates
        assert "characterization" in templates
        assert "debug" in templates
        assert "production" in templates

        # Load builtin template
        template = load_template("compliance")
        assert template.name == "Compliance Report"
        assert len(template.sections) > 0

        # Load production template
        prod = load_template("production")
        assert "yield" in prod.name.lower() or "production" in prod.name.lower()

        # Save and reload custom template
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            custom_path = f.name

        try:
            save_template(template, custom_path)
            reloaded = load_template(custom_path)
            assert reloaded.name == template.name
        finally:
            Path(custom_path).unlink(missing_ok=True)

    def test_batch_report(self) -> None:
        """Test batch report generation - RPT-003."""
        from oscura.reporting.batch import (
            BatchReportResult,
            aggregate_batch_measurements,
            generate_batch_report,
        )

        # Create mock batch results
        batch_results = [
            {
                "dut_id": "DUT-001",
                "measurements": {
                    "rise_time": {"value": 1.5e-9, "unit": "s", "passed": True},
                    "fall_time": {"value": 1.6e-9, "unit": "s", "passed": True},
                },
                "pass_count": 2,
                "total_count": 2,
            },
            {
                "dut_id": "DUT-002",
                "measurements": {
                    "rise_time": {"value": 2.0e-9, "unit": "s", "passed": True},
                    "fall_time": {"value": 1.8e-9, "unit": "s", "passed": False},
                },
                "pass_count": 1,
                "total_count": 2,
            },
            {
                "dut_id": "DUT-003",
                "measurements": {
                    "rise_time": {"value": 1.4e-9, "unit": "s", "passed": True},
                    "fall_time": {"value": 1.5e-9, "unit": "s", "passed": True},
                },
                "pass_count": 2,
                "total_count": 2,
            },
        ]

        # Generate batch report
        report = generate_batch_report(batch_results)
        assert report.config.title == "Batch Test Summary Report"
        assert len(report.sections) > 0

        # Check sections exist
        section_titles = [s.title for s in report.sections]
        assert any("Summary" in t for t in section_titles)

        # Aggregate measurements
        aggregated = aggregate_batch_measurements(batch_results)
        assert "rise_time" in aggregated
        assert len(aggregated["rise_time"]) == 3

        # BatchReportResult
        result = BatchReportResult()
        result.total_duts = 3
        result.passed_duts = 2
        result.failed_duts = 1

        assert result.dut_yield == pytest.approx(66.67, rel=0.1)
