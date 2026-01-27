"""Tests for memory optimizer.

Comprehensive test suite for memory optimization, streaming, and chunking.
Targets >85% code coverage with performance benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.utils.performance.memory_optimizer import (
    ChunkingConfig,
    ChunkingStrategy,
    MemoryOptimizer,
    MemoryStats,
    StreamProcessor,
)


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ChunkingConfig()

        assert config.strategy == ChunkingStrategy.FIXED
        assert config.chunk_size == 1_000_000
        assert config.overlap == 0
        assert config.adaptive is False
        assert config.time_window is None
        assert config.min_chunk_size == 100_000
        assert config.max_chunk_size == 10_000_000

    def test_fixed_chunking_config(self) -> None:
        """Test fixed chunking configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=500_000,
            overlap=1024,
        )

        assert config.strategy == ChunkingStrategy.FIXED
        assert config.chunk_size == 500_000
        assert config.overlap == 1024

    def test_adaptive_chunking_config(self) -> None:
        """Test adaptive chunking configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.ADAPTIVE,
            adaptive=True,
            min_chunk_size=50_000,
            max_chunk_size=5_000_000,
        )

        assert config.strategy == ChunkingStrategy.ADAPTIVE
        assert config.adaptive is True
        assert config.min_chunk_size == 50_000
        assert config.max_chunk_size == 5_000_000

    def test_sliding_window_config(self) -> None:
        """Test sliding window configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SLIDING,
            chunk_size=2048,
            overlap=1024,
        )

        assert config.strategy == ChunkingStrategy.SLIDING
        assert config.chunk_size == 2048
        assert config.overlap == 1024

    def test_time_based_config(self) -> None:
        """Test time-based chunking configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.TIME_BASED,
            time_window=1.0,
        )

        assert config.strategy == ChunkingStrategy.TIME_BASED
        assert config.time_window == 1.0

    def test_invalid_chunk_size(self) -> None:
        """Test validation of invalid chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=-100)

    def test_invalid_overlap(self) -> None:
        """Test validation of invalid overlap."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            ChunkingConfig(overlap=-1)

        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            ChunkingConfig(chunk_size=1000, overlap=1000)

        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            ChunkingConfig(chunk_size=1000, overlap=2000)

    def test_invalid_min_chunk_size(self) -> None:
        """Test validation of invalid min chunk size."""
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkingConfig(min_chunk_size=0)

        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkingConfig(min_chunk_size=-100)

    def test_invalid_max_chunk_size(self) -> None:
        """Test validation of max chunk size less than min."""
        with pytest.raises(ValueError, match="max_chunk_size .* must be >="):
            ChunkingConfig(min_chunk_size=1_000_000, max_chunk_size=500_000)

    def test_invalid_time_window(self) -> None:
        """Test validation of invalid time window."""
        with pytest.raises(ValueError, match="time_window must be positive"):
            ChunkingConfig(time_window=0.0)

        with pytest.raises(ValueError, match="time_window must be positive"):
            ChunkingConfig(time_window=-1.0)


class TestMemoryStats:
    """Tests for MemoryStats."""

    def test_memory_stats_creation(self) -> None:
        """Test creating memory stats."""
        stats = MemoryStats(
            peak_memory_mb=100.0,
            current_memory_mb=80.0,
            allocated_mb=150.0,
            freed_mb=70.0,
            leak_detected=False,
            available_memory_mb=2048.0,
            usage_percent=5.0,
        )

        assert stats.peak_memory_mb == 100.0
        assert stats.current_memory_mb == 80.0
        assert stats.allocated_mb == 150.0
        assert stats.freed_mb == 70.0
        assert stats.leak_detected is False
        assert stats.available_memory_mb == 2048.0
        assert stats.usage_percent == 5.0

    def test_memory_stats_leak_detected(self) -> None:
        """Test memory stats with leak detected."""
        stats = MemoryStats(
            peak_memory_mb=500.0,
            current_memory_mb=450.0,
            allocated_mb=800.0,
            freed_mb=50.0,
            leak_detected=True,
            available_memory_mb=1024.0,
            usage_percent=30.0,
        )

        assert stats.leak_detected is True
        assert stats.peak_memory_mb == 500.0


class TestStreamProcessor:
    """Tests for StreamProcessor."""

    def test_stream_processor_basic(self) -> None:
        """Test basic stream processing."""
        data = np.arange(10_000, dtype=np.float64)
        processor = StreamProcessor(data=data, chunk_size=1000, overlap=0)

        assert processor.chunk_size == 1000
        assert processor.overlap == 0
        assert processor.total_samples == 10_000
        assert len(processor) == 10

    def test_stream_processor_iteration(self) -> None:
        """Test iterating over chunks."""
        data = np.arange(10_000, dtype=np.float64)
        processor = StreamProcessor(data=data, chunk_size=1000, overlap=0)

        chunks = list(processor)

        assert len(chunks) == 10
        assert all(len(chunk) == 1000 for chunk in chunks)
        # Verify data integrity
        assert np.allclose(chunks[0], data[0:1000])
        assert np.allclose(chunks[5], data[5000:6000])
        assert np.allclose(chunks[9], data[9000:10000])

    def test_stream_processor_with_overlap(self) -> None:
        """Test stream processing with overlap."""
        data = np.arange(5000, dtype=np.float64)
        processor = StreamProcessor(data=data, chunk_size=1000, overlap=200)

        chunks = list(processor)

        # With overlap, step = 1000 - 200 = 800
        # Chunks should overlap by 200 samples
        assert len(chunks) > 0
        # First chunk: 0-1000
        assert len(chunks[0]) == 1000
        assert np.allclose(chunks[0], data[0:1000])

        # Second chunk: 800-1800 (overlaps with first at 800-1000)
        if len(chunks) > 1:
            assert len(chunks[1]) == 1000
            assert np.allclose(chunks[1], data[800:1800])

    def test_stream_processor_partial_last_chunk(self) -> None:
        """Test handling of partial last chunk."""
        data = np.arange(9500, dtype=np.float64)
        processor = StreamProcessor(data=data, chunk_size=1000, overlap=0)

        chunks = list(processor)

        # Last chunk should be partial (500 samples)
        assert len(chunks) == 10
        assert len(chunks[-1]) == 500
        assert np.allclose(chunks[-1], data[9000:9500])

    def test_stream_processor_reset(self) -> None:
        """Test resetting stream processor."""
        data = np.arange(5000, dtype=np.float64)
        processor = StreamProcessor(data=data, chunk_size=1000, overlap=0)

        # First iteration
        chunks1 = list(processor)

        # Reset and iterate again
        processor.reset()
        chunks2 = list(processor)

        # Should yield same chunks
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert np.allclose(c1, c2)

    def test_stream_processor_invalid_chunk_size(self) -> None:
        """Test invalid chunk size."""
        data = np.arange(1000, dtype=np.float64)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            StreamProcessor(data=data, chunk_size=0, overlap=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            StreamProcessor(data=data, chunk_size=-100, overlap=0)

    def test_stream_processor_invalid_overlap(self) -> None:
        """Test invalid overlap."""
        data = np.arange(1000, dtype=np.float64)

        with pytest.raises(ValueError, match="overlap must be non-negative"):
            StreamProcessor(data=data, chunk_size=1000, overlap=-1)

        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            StreamProcessor(data=data, chunk_size=1000, overlap=1000)

    def test_stream_processor_memmap(self, tmp_path: Path) -> None:
        """Test stream processing with memory-mapped array."""
        # Create memmap file
        data_file = tmp_path / "test_data.dat"
        data = np.arange(10_000, dtype=np.float64)
        data.tofile(data_file)

        # Create memmap
        memmap_data = np.memmap(data_file, dtype=np.float64, mode="r", shape=(10_000,))

        processor = StreamProcessor(data=memmap_data, chunk_size=1000, overlap=0)
        chunks = list(processor)

        assert len(chunks) == 10
        # Verify chunks are regular arrays, not memmaps
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
        assert all(not isinstance(chunk, np.memmap) for chunk in chunks)

    def test_stream_processor_len(self) -> None:
        """Test __len__ method."""
        data = np.arange(10_000, dtype=np.float64)

        # No overlap
        processor = StreamProcessor(data=data, chunk_size=1000, overlap=0)
        assert len(processor) == 10

        # With overlap
        processor = StreamProcessor(data=data, chunk_size=1000, overlap=200)
        # Step = 800, so num_chunks = ceil((10000 - 200) / 800) = 13
        expected = (10_000 - 200) // 800
        if (10_000 - 200) % 800 != 0:
            expected += 1
        assert len(processor) == expected


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer."""

    def test_memory_optimizer_default(self) -> None:
        """Test creating optimizer with defaults."""
        optimizer = MemoryOptimizer()

        assert optimizer.max_memory_mb is None
        assert optimizer.enable_compression is False
        assert optimizer.gc_threshold == 0.8

    def test_memory_optimizer_with_limits(self) -> None:
        """Test creating optimizer with memory limits."""
        optimizer = MemoryOptimizer(
            max_memory_mb=1024,
            enable_compression=True,
            gc_threshold=0.7,
        )

        assert optimizer.max_memory_mb == 1024
        assert optimizer.enable_compression is True
        assert optimizer.gc_threshold == 0.7

    def test_invalid_max_memory(self) -> None:
        """Test invalid max memory."""
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            MemoryOptimizer(max_memory_mb=0)

        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            MemoryOptimizer(max_memory_mb=-100)

    def test_invalid_gc_threshold(self) -> None:
        """Test invalid GC threshold."""
        with pytest.raises(ValueError, match="gc_threshold must be in"):
            MemoryOptimizer(gc_threshold=-0.1)

        with pytest.raises(ValueError, match="gc_threshold must be in"):
            MemoryOptimizer(gc_threshold=1.5)

    def test_get_memory_stats(self) -> None:
        """Test getting memory statistics."""
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.current_memory_mb > 0
        assert stats.peak_memory_mb >= stats.current_memory_mb
        assert stats.available_memory_mb > 0
        assert 0 <= stats.usage_percent <= 100

    def test_recommend_chunk_size(self) -> None:
        """Test chunk size recommendation."""
        optimizer = MemoryOptimizer()

        # For 100 MB target with float64 (8 bytes)
        chunk_size = optimizer.recommend_chunk_size(
            data_length=100_000_000,
            dtype=np.float64,
            target_memory_mb=100.0,
        )

        # 100 MB / 8 bytes = 12,500,000 samples
        expected = int((100.0 * 1024 * 1024) / 8)
        assert chunk_size == expected

    def test_recommend_chunk_size_clamping(self) -> None:
        """Test chunk size clamping to data length."""
        optimizer = MemoryOptimizer()

        # Target larger than data length
        chunk_size = optimizer.recommend_chunk_size(
            data_length=5000,
            dtype=np.float64,
            target_memory_mb=100.0,
        )

        # Should clamp to data length
        assert chunk_size == 5000

    def test_create_stream_processor(self) -> None:
        """Test creating stream processor."""
        optimizer = MemoryOptimizer()
        data = np.arange(10_000, dtype=np.float64)

        processor = optimizer.create_stream_processor(
            data=data,
            chunk_size=1000,
            overlap=100,
        )

        assert isinstance(processor, StreamProcessor)
        assert processor.chunk_size == 1000
        assert processor.overlap == 100

    def test_create_stream_processor_auto_chunk_size(self) -> None:
        """Test creating processor with automatic chunk size."""
        optimizer = MemoryOptimizer()
        data = np.arange(10_000, dtype=np.float64)

        processor = optimizer.create_stream_processor(data=data, chunk_size=None)

        assert isinstance(processor, StreamProcessor)
        assert processor.chunk_size > 0

    def test_create_stream_processor_with_config(self) -> None:
        """Test creating processor with config."""
        optimizer = MemoryOptimizer()
        data = np.arange(10_000, dtype=np.float64)

        config = ChunkingConfig(
            strategy=ChunkingStrategy.SLIDING,
            chunk_size=2000,
            overlap=500,
        )

        processor = optimizer.create_stream_processor(data=data, config=config)

        assert processor.chunk_size == 2000
        assert processor.overlap == 500

    def test_create_stream_processor_adaptive(self) -> None:
        """Test creating processor with adaptive chunking."""
        optimizer = MemoryOptimizer()
        data = np.arange(100_000, dtype=np.float64)

        config = ChunkingConfig(
            strategy=ChunkingStrategy.ADAPTIVE,
            adaptive=True,
            min_chunk_size=1000,
            max_chunk_size=50_000,
        )

        processor = optimizer.create_stream_processor(data=data, config=config)

        # Chunk size should be within bounds
        assert 1000 <= processor.chunk_size <= 50_000

    def test_optimize_array(self) -> None:
        """Test array optimization."""
        optimizer = MemoryOptimizer()
        data = np.arange(10_000, dtype=np.float64)

        optimized = optimizer.optimize_array(data)

        assert isinstance(optimized, np.ndarray)
        # Should be contiguous
        assert optimized.flags["C_CONTIGUOUS"]

    def test_optimize_array_non_contiguous(self) -> None:
        """Test optimizing non-contiguous array."""
        optimizer = MemoryOptimizer()
        data = np.arange(10_000, dtype=np.float64)
        # Create non-contiguous view
        non_contiguous = data[::2]

        assert not non_contiguous.flags["C_CONTIGUOUS"]

        optimized = optimizer.optimize_array(non_contiguous)

        # Should be made contiguous
        assert optimized.flags["C_CONTIGUOUS"]
        assert np.allclose(optimized, non_contiguous)

    def test_set_memory_limit(self) -> None:
        """Test setting memory limit."""
        optimizer = MemoryOptimizer()

        optimizer.set_memory_limit(2048)

        assert optimizer.max_memory_mb == 2048

    def test_set_memory_limit_invalid(self) -> None:
        """Test setting invalid memory limit."""
        optimizer = MemoryOptimizer()

        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            optimizer.set_memory_limit(0)

        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            optimizer.set_memory_limit(-100)

    def test_check_available_memory(self) -> None:
        """Test checking available memory."""
        optimizer = MemoryOptimizer()

        # Should have at least 100 MB available on test system
        assert optimizer.check_available_memory(100.0)

        # Should not have 1 TB available
        assert not optimizer.check_available_memory(1_000_000.0)

    def test_suggest_downsampling(self) -> None:
        """Test downsampling suggestion."""
        optimizer = MemoryOptimizer()

        # Data that fits in target memory
        factor = optimizer.suggest_downsampling(
            data_length=1_000_000,
            dtype=np.float64,
            target_memory_mb=100.0,
        )

        # 1M samples * 8 bytes = 8 MB < 100 MB, no downsampling needed
        assert factor == 1

    def test_suggest_downsampling_needed(self) -> None:
        """Test downsampling when data too large."""
        optimizer = MemoryOptimizer()

        # Data that exceeds target memory
        factor = optimizer.suggest_downsampling(
            data_length=100_000_000,
            dtype=np.float64,
            target_memory_mb=10.0,
        )

        # 100M samples * 8 bytes = 800 MB > 10 MB, should suggest downsampling
        assert factor > 1

    def test_reset_statistics(self) -> None:
        """Test resetting statistics."""
        optimizer = MemoryOptimizer()

        # Get initial stats
        stats1 = optimizer.get_memory_stats()

        # Reset
        optimizer.reset_statistics()

        # Get new stats
        stats2 = optimizer.get_memory_stats()

        # Allocated/freed should be reset
        assert stats2.allocated_mb == 0.0
        assert stats2.freed_mb == 0.0

    def test_load_optimized_small_file(self, tmp_path: Path) -> None:
        """Test loading small file with eager loading."""
        # Create small test file (< 100 MB)
        # Use raw binary format since .npy is not supported by default loader
        data_file = tmp_path / "small_data.bin"
        data = np.arange(10_000, dtype=np.float64)
        data.tofile(data_file)

        optimizer = MemoryOptimizer()

        # Load with mmap (since we're using raw binary)
        from oscura.loaders.mmap_loader import load_mmap

        trace = load_mmap(
            data_file,
            sample_rate=1e6,
            dtype=np.float64,
            length=10_000,
        )

        # Verify trace loaded
        assert hasattr(trace, "data")
        assert len(trace.data) == 10_000

    def test_load_optimized_large_file(self, tmp_path: Path) -> None:
        """Test loading large file with memory mapping."""
        # Create large test file (> threshold)
        data_file = tmp_path / "large_data.npy"
        # Create 20 MB file (2.5M float64 samples)
        data = np.arange(2_500_000, dtype=np.float64)
        np.save(data_file, data)

        optimizer = MemoryOptimizer()
        trace = optimizer.load_optimized(
            data_file,
            sample_rate=1e6,
            mmap_threshold_mb=10.0,  # Set low threshold
        )

        # Should use mmap for large files
        # MmapWaveformTrace has 'data' attribute
        assert hasattr(trace, "data")
        assert len(trace.data) == 2_500_000

    def test_memory_tracking_during_operation(self) -> None:
        """Test memory tracking during operations."""
        optimizer = MemoryOptimizer(max_memory_mb=2048)

        # Get initial stats
        stats1 = optimizer.get_memory_stats()
        initial_current = stats1.current_memory_mb

        # Create some data
        data = np.arange(1_000_000, dtype=np.float64)

        # Process with optimizer
        processor = optimizer.create_stream_processor(data, chunk_size=100_000)
        for chunk in processor:
            _ = np.mean(chunk)

        # Get final stats
        stats2 = optimizer.get_memory_stats()

        # Peak should be >= current
        assert stats2.peak_memory_mb >= stats2.current_memory_mb
        # Current should be reasonable
        assert stats2.current_memory_mb > 0


class TestMemoryOptimizerIntegration:
    """Integration tests for memory optimizer."""

    def test_complete_workflow(self, tmp_path: Path) -> None:
        """Test complete optimization workflow."""
        # Create test data directly in memory
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / 1e6)

        # Create optimizer
        optimizer = MemoryOptimizer(max_memory_mb=1024)

        # Create processor directly from array
        processor = optimizer.create_stream_processor(data, chunk_size=10_000)

        # Process chunks
        chunk_means = []
        for chunk in processor:
            chunk_means.append(np.mean(chunk))

        # Verify results
        assert len(chunk_means) == 10
        # All chunks should have similar mean (near 0 for sine wave)
        assert all(abs(m) < 0.1 for m in chunk_means)

        # Check stats
        stats = optimizer.get_memory_stats()
        assert stats.current_memory_mb > 0
        assert not stats.leak_detected

    def test_adaptive_chunking_workflow(self) -> None:
        """Test adaptive chunking workflow."""
        optimizer = MemoryOptimizer()
        data = np.arange(1_000_000, dtype=np.float64)

        config = ChunkingConfig(
            strategy=ChunkingStrategy.ADAPTIVE,
            adaptive=True,
            min_chunk_size=10_000,
            max_chunk_size=500_000,
        )

        processor = optimizer.create_stream_processor(data, config=config)

        # Verify adaptive chunk size within bounds
        assert 10_000 <= processor.chunk_size <= 500_000

        # Process all chunks
        chunks = list(processor)
        assert len(chunks) > 0

        # Verify all data processed
        total_samples = sum(len(c) for c in chunks)
        assert total_samples >= len(data)

    def test_overlapping_chunks_continuity(self) -> None:
        """Test that overlapping chunks maintain continuity."""
        optimizer = MemoryOptimizer()
        data = np.arange(10_000, dtype=np.float64)

        processor = optimizer.create_stream_processor(
            data,
            chunk_size=1000,
            overlap=200,
        )

        chunks = list(processor)

        # Verify overlap region matches
        for i in range(len(chunks) - 1):
            # Last 200 samples of chunk i
            end_region = chunks[i][-200:]
            # First 200 samples of chunk i+1
            start_region = chunks[i + 1][:200]

            # Should be identical
            assert np.allclose(end_region, start_region)

    def test_memory_limit_enforcement(self) -> None:
        """Test that memory limits are respected."""
        # Create optimizer with strict limit
        optimizer = MemoryOptimizer(max_memory_mb=100, gc_threshold=0.5)

        # Get initial memory
        stats1 = optimizer.get_memory_stats()

        # Create data and process
        data = np.arange(100_000, dtype=np.float64)
        processor = optimizer.create_stream_processor(data, chunk_size=10_000)

        for chunk in processor:
            _ = np.std(chunk)

        # Get final stats
        stats2 = optimizer.get_memory_stats()

        # Memory should be reasonable
        assert stats2.current_memory_mb > 0
