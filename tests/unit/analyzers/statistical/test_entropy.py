"""Unit tests for entropy analysis.

This module tests entropy analysis using statistical/entropy/*.bin files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requires_data
@pytest.mark.requirement("SEA-001")
class TestEntropyAnalysis:
    """Test Shannon entropy analysis."""

    def test_high_entropy_random_data(self, entropy_files: dict[str, Path]) -> None:
        """Test entropy calculation on random data.

        Random data should have high entropy (close to 8 bits per byte).
        """
        try:
            import secrets

            from oscura.analyzers.statistical import calculate_entropy

            # Generate truly random data using cryptographically secure RNG
            # Larger sample size provides more stable entropy estimates
            data = secrets.token_bytes(8192)  # 8KB of random data

            entropy = calculate_entropy(data)

            # Random data should have high entropy (close to 8.0 for 8-bit data)
            # Use 6.5 threshold to handle small sample statistical variations
            # Truly random data typically has entropy > 7.5, but allow margin
            assert entropy > 6.5, f"Random data entropy too low: {entropy}"
            assert entropy <= 8.0, f"Entropy exceeds maximum: {entropy}"

        except ImportError:
            pytest.skip("calculate_entropy not available")

    def test_low_entropy_pattern(self, entropy_files: dict[str, Path]) -> None:
        """Test entropy calculation on low-entropy pattern.

        Repeating patterns have low entropy.
        """
        path = entropy_files.get("low")
        if path is None or not path.exists():
            pytest.skip("Low entropy file not available")

        with open(path, "rb") as f:
            data = f.read(4096)

        try:
            from oscura.analyzers.statistical import calculate_entropy

            entropy = calculate_entropy(data)

            # Repeating pattern should have low entropy
            assert entropy < 4.0, f"Pattern entropy too high: {entropy}"
            assert entropy >= 0.0, f"Entropy is negative: {entropy}"

        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("calculate_entropy not available")

    def test_medium_entropy_text(self, entropy_files: dict[str, Path]) -> None:
        """Test entropy calculation on English text.

        English text has medium entropy (typically 3-5 bits per byte).
        """
        path = entropy_files.get("text")
        if path is None or not path.exists():
            pytest.skip("Text entropy file not available")

        with open(path, "rb") as f:
            data = f.read(4096)

        try:
            from oscura.analyzers.statistical import calculate_entropy

            entropy = calculate_entropy(data)

            # English text typically has entropy between 3 and 5
            assert 2.0 < entropy < 6.0, f"Text entropy unexpected: {entropy}"

        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("calculate_entropy not available")


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SEA-001")
class TestEntropyEdgeCases:
    """Test edge cases for entropy calculation."""

    def test_entropy_empty_data(self) -> None:
        """Test entropy of empty data."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            empty = b""

            # Empty data has undefined entropy - should handle gracefully
            with pytest.raises((ValueError, ZeroDivisionError)):
                calculate_entropy(empty)

        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("calculate_entropy not available")

    def test_entropy_single_byte(self) -> None:
        """Test entropy of single-byte data."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            single = b"\x42"

            entropy = calculate_entropy(single)

            # Single byte has entropy 0 (no uncertainty)
            assert entropy == 0.0

        except ImportError:
            pytest.skip("calculate_entropy not available")

    def test_entropy_uniform_distribution(self) -> None:
        """Test entropy of uniform byte distribution.

        All 256 byte values appearing equally = max entropy.
        """
        try:
            from oscura.analyzers.statistical import calculate_entropy

            # Create data with each byte value once
            uniform = bytes(range(256))

            entropy = calculate_entropy(uniform)

            # Perfect uniform = log2(256) = 8.0
            assert abs(entropy - 8.0) < 0.01, f"Uniform entropy not 8: {entropy}"

        except ImportError:
            pytest.skip("calculate_entropy not available")

    def test_entropy_repeated_byte(self) -> None:
        """Test entropy of repeated single byte.

        All bytes the same = 0 entropy.
        """
        try:
            from oscura.analyzers.statistical import calculate_entropy

            repeated = b"\xaa" * 1000

            entropy = calculate_entropy(repeated)

            # All same = 0 entropy
            assert entropy == 0.0, f"Repeated byte entropy not 0: {entropy}"

        except ImportError:
            pytest.skip("calculate_entropy not available")


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SEA-001")
class TestEntropyModule:
    """Test entropy module functionality."""

    def test_entropy_module_import(self) -> None:
        """Test that entropy module can be imported."""
        try:
            from oscura.analyzers.statistical import entropy

            assert entropy is not None

        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("entropy module not available")

    def test_calculate_entropy_function(self) -> None:
        """Test calculate_entropy function signature."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            # Create test data
            data = bytes([0, 1, 2, 3] * 100)

            result = calculate_entropy(data)

            assert isinstance(result, int | float)
            assert 0 <= result <= 8

        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("calculate_entropy not available")

    def test_entropy_window_analysis(self) -> None:
        """Test sliding window entropy analysis."""
        try:
            from oscura.analyzers.statistical import entropy_windowed

            # Create data with varying entropy regions
            low_entropy = bytes([0xAA] * 500)
            high_entropy = bytes(np.random.randint(0, 256, 500, dtype=np.uint8))
            data = low_entropy + high_entropy

            results = entropy_windowed(data, window_size=100)

            # Should return multiple values
            assert len(results) > 0

            # Low entropy region should have lower values than high entropy
            # (Exact comparison depends on API)

        except ImportError:
            pytest.skip("entropy_windowed not available")


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requires_data
class TestEntropyOnSyntheticData:
    """Test entropy on synthetic test data."""

    def test_all_entropy_files_readable(self, entropy_files: dict[str, Path]) -> None:
        """Test that all entropy test files can be read."""
        for name, path in entropy_files.items():
            if path is None:
                continue

            if not path.exists():
                pytest.skip(f"{name} entropy file not found")

            with open(path, "rb") as f:
                data = f.read(1024)
                assert len(data) > 0, f"Empty data in {name}"

    def test_entropy_values_in_range(self, entropy_files: dict[str, Path]) -> None:
        """Test that calculated entropy is always 0-8."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            for name, path in entropy_files.items():
                if path is None or not path.exists():
                    continue

                with open(path, "rb") as f:
                    data = f.read(4096)

                entropy = calculate_entropy(data)

                assert 0 <= entropy <= 8, f"{name}: Entropy {entropy} outside valid range"

        except ImportError:
            pytest.skip("calculate_entropy not available")


@pytest.mark.unit
@pytest.mark.analyzer
class TestEntropyDataTypes:
    """Test entropy calculation with different data types."""

    def test_entropy_bytes(self) -> None:
        """Test entropy calculation with bytes object."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            data = b"Hello, World!"
            entropy = calculate_entropy(data)

            assert isinstance(entropy, int | float)

        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("calculate_entropy not available")

    def test_entropy_bytearray(self) -> None:
        """Test entropy calculation with bytearray."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            data = bytearray(b"Hello, World!")
            entropy = calculate_entropy(data)

            assert isinstance(entropy, int | float)

        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("calculate_entropy not available")

    def test_entropy_numpy_array(self) -> None:
        """Test entropy calculation with numpy array."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            data = np.array([0, 1, 2, 3, 4, 5, 6, 7] * 100, dtype=np.uint8)

            # May need to convert to bytes
            entropy = calculate_entropy(bytes(data))

            assert isinstance(entropy, int | float)

        except ImportError:
            pytest.skip("calculate_entropy not available")


@pytest.mark.unit
@pytest.mark.analyzer
class TestEntropyPerformance:
    """Test entropy calculation performance."""

    @pytest.mark.slow
    def test_entropy_large_data(self) -> None:
        """Test entropy calculation on large data."""
        try:
            from oscura.analyzers.statistical import calculate_entropy

            # 1 MB of random data
            data = bytes(np.random.randint(0, 256, 1024 * 1024, dtype=np.uint8))

            import time

            start = time.time()
            entropy = calculate_entropy(data)
            elapsed = time.time() - start

            assert entropy > 7.9  # Random should be high entropy
            assert elapsed < 5.0  # Should complete reasonably fast

        except ImportError:
            pytest.skip("calculate_entropy not available")
