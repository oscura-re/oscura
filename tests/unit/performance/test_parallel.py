"""Tests for parallel processing - simplified version without nested functions."""

import pytest

from oscura.utils.performance.parallel import ParallelConfig, ParallelProcessor


# Module-level test functions (required for pickling)
def _square(x: int) -> int:
    return x * x


class TestParallelConfig:
    """Basic configuration tests."""

    def test_default_configuration(self) -> None:
        """Test ParallelConfig initialization with default values."""
        config = ParallelConfig()
        assert config.num_workers is None
        assert config.strategy == "auto"

    def test_invalid_num_workers(self) -> None:
        """Test ParallelConfig raises ValueError for invalid num_workers."""
        with pytest.raises(ValueError):
            ParallelConfig(num_workers=0)


class TestParallelProcessor:
    """Basic processor tests."""

    def test_initialization(self) -> None:
        """Test ParallelProcessor initializes with valid config."""
        processor = ParallelProcessor()
        assert processor.config is not None

    def test_map_sequential(self) -> None:
        """Test ParallelProcessor.map() with sequential processing."""
        processor = ParallelProcessor()
        items = list(range(5))
        result = processor.map(_square, items)

        assert len(result.results) == 5
        assert result.results == [0, 1, 4, 9, 16]

    def test_map_process(self) -> None:
        """Test ParallelProcessor.map() with multiprocess strategy."""
        config = ParallelConfig(num_workers=2, strategy="process")
        processor = ParallelProcessor(config)

        items = list(range(20))
        result = processor.map(_square, items)

        assert len(result.results) == 20
        assert result.results == [x * x for x in items]
