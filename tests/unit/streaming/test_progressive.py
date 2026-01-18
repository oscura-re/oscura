"""Comprehensive tests for progressive streaming analysis module.

Tests cover:
- StreamingConfig validation
- StreamingProgress data class
- ProgressiveAnalyzer incremental analysis
- Confidence calculation and monotonic growth
- Callback subscription and notification
- Early stopping functionality
- Reset and finalization
- Integration with quality scoring

Requirements tested:
"""

import numpy as np
import pytest

from oscura.streaming.progressive import (
    ProgressiveAnalyzer,
    StreamingConfig,
    StreamingProgress,
    create_progressive_analyzer,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.streaming
class TestStreamingProgress:
    """Test StreamingProgress dataclass."""

    def test_basic_initialization(self):
        """Test basic progress initialization."""
        progress = StreamingProgress(
            samples_processed=1000,
            total_samples=10000,
            confidence=0.75,
            preliminary_results={"mean": 0.5},
        )

        assert progress.samples_processed == 1000
        assert progress.total_samples == 10000
        assert progress.confidence == 0.75
        assert progress.preliminary_results == {"mean": 0.5}
        assert not progress.is_complete
        assert not progress.can_stop_early
        assert progress.message == ""

    def test_progress_percent_with_known_total(self):
        """Test progress percentage calculation with known total."""
        progress = StreamingProgress(
            samples_processed=2500,
            total_samples=10000,
            confidence=0.5,
            preliminary_results={},
        )

        assert progress.progress_percent == 25.0

    def test_progress_percent_with_unknown_total(self):
        """Test progress percentage with unknown total."""
        progress = StreamingProgress(
            samples_processed=1000,
            total_samples=None,
            confidence=0.5,
            preliminary_results={},
        )

        assert progress.progress_percent is None

    def test_complete_progress(self):
        """Test completed progress."""
        progress = StreamingProgress(
            samples_processed=10000,
            total_samples=10000,
            confidence=0.95,
            preliminary_results={},
            is_complete=True,
            message="Analysis complete",
        )

        assert progress.is_complete
        assert progress.progress_percent == 100.0


@pytest.mark.unit
@pytest.mark.streaming
class TestStreamingConfig:
    """Test StreamingConfig dataclass."""

    def test_default_initialization(self):
        """Test config with default parameters."""
        config = StreamingConfig()

        assert config.chunk_size == 1024
        assert config.overlap == 0.25
        assert config.min_samples_for_result == 100
        assert config.early_stop_confidence == 0.9
        assert config.max_buffer_size == 100000
        assert config.update_interval_samples == 512

    def test_custom_initialization(self):
        """Test config with custom parameters."""
        config = StreamingConfig(
            chunk_size=512,
            overlap=0.5,
            min_samples_for_result=200,
            early_stop_confidence=0.85,
            max_buffer_size=50000,
            update_interval_samples=256,
        )

        assert config.chunk_size == 512
        assert config.overlap == 0.5
        assert config.min_samples_for_result == 200
        assert config.early_stop_confidence == 0.85
        assert config.max_buffer_size == 50000
        assert config.update_interval_samples == 256


@pytest.mark.unit
@pytest.mark.streaming
class TestProgressiveAnalyzer:
    """Test ProgressiveAnalyzer streaming analysis."""

    def test_initialization(self):
        """Test basic analyzer initialization."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        assert analyzer.sample_rate == 1000.0
        assert analyzer._samples_processed == 0
        assert analyzer._confidence == 0.0
        assert len(analyzer._buffer) == 0
        assert len(analyzer._callbacks) == 0

    def test_initialization_with_config(self):
        """Test analyzer initialization with custom config."""
        config = StreamingConfig(chunk_size=512, early_stop_confidence=0.85)
        analyzer = ProgressiveAnalyzer(sample_rate=2000.0, config=config)

        assert analyzer.sample_rate == 2000.0
        assert analyzer.config.chunk_size == 512
        assert analyzer.config.early_stop_confidence == 0.85

    def test_process_single_chunk(self):
        """Test processing a single chunk."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)
        chunk = np.random.randn(100)

        progress = analyzer.process_chunk(chunk)

        assert progress.samples_processed == 100
        assert not progress.is_complete
        assert progress.confidence >= 0.0
        assert progress.confidence <= 1.0

    def test_process_multiple_chunks(self):
        """Test processing multiple chunks."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        progress1 = analyzer.process_chunk(np.random.randn(100))
        progress2 = analyzer.process_chunk(np.random.randn(200))
        progress3 = analyzer.process_chunk(np.random.randn(150))

        assert progress1.samples_processed == 100
        assert progress2.samples_processed == 300
        assert progress3.samples_processed == 450

    def test_confidence_monotonic_increase(self):
        """Test that confidence increases or stays constant."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)
        confidences = []

        for _ in range(10):
            chunk = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))
            progress = analyzer.process_chunk(chunk)
            confidences.append(progress.confidence)

        # Confidence should be monotonically non-decreasing
        for i in range(len(confidences) - 1):
            assert confidences[i + 1] >= confidences[i], (
                f"Confidence decreased: {confidences[i]} -> {confidences[i + 1]}"
            )

    def test_preliminary_results_generation(self):
        """Test generation of preliminary results."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        # Process enough data for results
        for _ in range(3):
            chunk = np.random.randn(100)
            progress = analyzer.process_chunk(chunk)

        assert "mean" in progress.preliminary_results
        assert "std" in progress.preliminary_results
        assert "min" in progress.preliminary_results
        assert "max" in progress.preliminary_results
        assert "amplitude" in progress.preliminary_results
        assert "sample_count" in progress.preliminary_results

    def test_frequency_estimation(self):
        """Test frequency estimation from periodic signal."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        # Generate 10 Hz sine wave
        for i in range(10):
            t = np.linspace(i, i + 1, 1000)
            chunk = np.sin(2 * np.pi * 10 * t)
            progress = analyzer.process_chunk(chunk)

        # Should detect frequency around 10 Hz
        if "frequency_estimate" in progress.preliminary_results:
            freq = progress.preliminary_results["frequency_estimate"]
            assert 5 < freq < 50  # Reasonable range

    def test_subscribe_callback(self):
        """Test subscribing to progress updates."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)
        updates = []

        def callback(progress):
            updates.append(progress)

        analyzer.subscribe(callback)

        # Process data to trigger callbacks
        for _ in range(5):
            analyzer.process_chunk(np.random.randn(600))

        assert len(updates) > 0
        assert all(isinstance(u, StreamingProgress) for u in updates)

    def test_multiple_subscribers(self):
        """Test multiple subscribers receive updates."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)
        updates1 = []
        updates2 = []

        analyzer.subscribe(lambda p: updates1.append(p))
        analyzer.subscribe(lambda p: updates2.append(p))

        for _ in range(3):
            analyzer.process_chunk(np.random.randn(600))

        assert len(updates1) > 0
        assert len(updates2) > 0
        assert len(updates1) == len(updates2)

    def test_callback_exception_handling(self):
        """Test that callback exceptions don't break processing."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        def bad_callback(progress):
            raise ValueError("Intentional error")

        analyzer.subscribe(bad_callback)

        # Should not raise despite callback error
        progress = analyzer.process_chunk(np.random.randn(600))
        assert progress.samples_processed == 600

    def test_early_stopping_detection(self):
        """Test early stopping flag when confidence threshold met."""
        config = StreamingConfig(early_stop_confidence=0.7)
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0, config=config)

        can_stop = False
        for i in range(100):
            chunk = np.sin(2 * np.pi * 10 * np.linspace(i, i + 1, 1000))
            progress = analyzer.process_chunk(chunk)

            if progress.can_stop_early:
                can_stop = True
                assert progress.confidence >= 0.7
                break

        # Should eventually reach stopping threshold
        assert can_stop or progress.confidence >= 0.7

    def test_finalize(self):
        """Test finalization of analysis."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        # Process some data
        for _ in range(5):
            analyzer.process_chunk(np.random.randn(500))

        final = analyzer.finalize()

        assert final.is_complete
        assert final.samples_processed == 2500
        assert final.total_samples == 2500
        assert final.progress_percent == 100.0
        assert not final.can_stop_early

    def test_finalize_includes_quality_assessment(self):
        """Test finalize includes data quality assessment."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        for _ in range(3):
            analyzer.process_chunk(np.random.randn(500))

        final = analyzer.finalize()

        assert "data_quality" in final.preliminary_results
        dq = final.preliminary_results["data_quality"]
        assert "snr_db" in dq
        assert "sample_count" in dq
        assert "completeness" in dq

    def test_finalize_includes_fft_frequency(self):
        """Test finalize includes FFT-based frequency detection."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        # Generate periodic signal
        for i in range(5):
            t = np.linspace(i, i + 1, 1000)
            chunk = np.sin(2 * np.pi * 25 * t)
            analyzer.process_chunk(chunk)

        final = analyzer.finalize()

        assert "frequency_final" in final.preliminary_results

    def test_reset(self):
        """Test reset clears analyzer state."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        # Process some data
        for _ in range(3):
            analyzer.process_chunk(np.random.randn(100))

        assert analyzer._samples_processed > 0
        assert analyzer._confidence > 0

        analyzer.reset()

        assert analyzer._samples_processed == 0
        assert analyzer._confidence == 0.0
        assert len(analyzer._buffer) == 0
        assert len(analyzer._current_results) == 0

    def test_reset_preserves_callbacks(self):
        """Test reset preserves callback subscriptions."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)
        updates = []

        analyzer.subscribe(lambda p: updates.append(p))
        analyzer.process_chunk(np.random.randn(600))

        initial_count = len(updates)
        analyzer.reset()

        analyzer.process_chunk(np.random.randn(600))

        # Should still receive callbacks after reset
        assert len(updates) > initial_count

    def test_buffer_size_limit(self):
        """Test buffer respects max size limit."""
        config = StreamingConfig(max_buffer_size=500)
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0, config=config)

        # Process more data than buffer can hold
        for _ in range(10):
            analyzer.process_chunk(np.random.randn(200))

        # Buffer should be limited
        assert len(analyzer._buffer) <= 500

    def test_running_statistics_accuracy(self):
        """Test running statistics match batch calculations."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        all_data = []
        for _ in range(5):
            chunk = np.random.randn(200)
            all_data.extend(chunk)
            analyzer.process_chunk(chunk)

        progress = analyzer.process_chunk(np.array([]))

        # Compare with batch calculation
        all_data_array = np.array(all_data)
        expected_mean = np.mean(all_data_array)
        expected_std = np.std(all_data_array)

        actual_mean = progress.preliminary_results["mean"]
        actual_std = progress.preliminary_results["std"]

        np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(actual_std, expected_std, rtol=1e-10)

    def test_status_messages(self):
        """Test status message generation."""
        config = StreamingConfig(min_samples_for_result=100)
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0, config=config)

        # Early stage - collecting data
        progress1 = analyzer.process_chunk(np.random.randn(50))
        assert "Collecting data" in progress1.message

        # After minimum samples
        progress2 = analyzer.process_chunk(np.random.randn(100))
        assert "confidence" in progress2.message.lower()


@pytest.mark.unit
@pytest.mark.streaming
class TestProgressiveAnalyzerFactory:
    """Test factory function for creating analyzers."""

    def test_create_progressive_analyzer(self):
        """Test factory function creates properly configured analyzer."""
        analyzer = create_progressive_analyzer(
            sample_rate=2000.0,
            chunk_size=512,
            early_stop_confidence=0.85,
        )

        assert analyzer.sample_rate == 2000.0
        assert analyzer.config.chunk_size == 512
        assert analyzer.config.early_stop_confidence == 0.85

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        analyzer = create_progressive_analyzer()

        assert analyzer.sample_rate == 1.0
        assert analyzer.config.chunk_size == 1024
        assert analyzer.config.early_stop_confidence == 0.9


@pytest.mark.unit
@pytest.mark.streaming
class TestProgressiveAnalyzerIntegration:
    """Integration tests for progressive analyzer."""

    def test_full_workflow(self):
        """Test complete workflow from start to finish."""
        analyzer = create_progressive_analyzer(
            sample_rate=1000.0,
            chunk_size=512,
            early_stop_confidence=0.85,
        )

        updates = []
        analyzer.subscribe(lambda p: updates.append(p))

        # Generate and process streaming data
        for i in range(10):
            t = np.linspace(i, i + 1, 512)
            signal = 2.0 * np.sin(2 * np.pi * 20 * t)
            noise = 0.3 * np.random.randn(512)
            chunk = signal + noise

            progress = analyzer.process_chunk(chunk)

            if progress.can_stop_early:
                break

        final = analyzer.finalize()

        # Verify complete workflow
        assert final.is_complete
        assert final.samples_processed > 0
        assert len(updates) > 0
        assert "data_quality" in final.preliminary_results

    def test_periodic_signal_analysis(self):
        """Test analysis of known periodic signal."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        # Generate 50 Hz sine wave
        for i in range(20):
            t = np.linspace(i, i + 1, 1000)
            chunk = np.sin(2 * np.pi * 50 * t)
            analyzer.process_chunk(chunk)

        final = analyzer.finalize()

        # Should have high confidence for clean signal
        assert final.confidence > 0.7

        # Check amplitude detection
        if "amplitude" in final.preliminary_results:
            amp = final.preliminary_results["amplitude"]
            assert 1.8 < amp < 2.2  # Sine wave amplitude ~2

    def test_noisy_signal_analysis(self):
        """Test analysis handles noisy signals."""
        analyzer = ProgressiveAnalyzer(sample_rate=1000.0)

        # Generate noisy signal
        for _ in range(10):
            chunk = np.random.randn(1000)
            analyzer.process_chunk(chunk)

        final = analyzer.finalize()

        # Should complete without errors
        assert final.is_complete
        assert "data_quality" in final.preliminary_results

        # Mean should be close to zero
        if "mean" in final.preliminary_results:
            assert abs(final.preliminary_results["mean"]) < 0.5
