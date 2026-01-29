"""Tests for pre-flight memory checking module.

Requirements tested:
"""

from unittest.mock import patch

import numpy as np
import pytest

from oscura.core.memory_check import (
    _AUTO_CHECK_OPERATIONS,
    _reset_auto_check_operations,
    auto_check_memory,
    check_operation_memory,
    get_auto_check_operations,
    is_force_memory,
    register_auto_check_operation,
    set_force_memory,
    with_memory_check,
)
from oscura.utils.memory import MemoryCheck, MemoryCheckError

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestSetForceMemory:
    """Test set_force_memory function."""

    def teardown_method(self):
        """Reset force memory flag after each test."""
        set_force_memory(False)

    def test_enable_force_memory(self):
        """Test enabling force memory bypass."""
        set_force_memory(True)
        assert is_force_memory() is True

    def test_disable_force_memory(self):
        """Test disabling force memory bypass."""
        set_force_memory(True)
        set_force_memory(False)
        assert is_force_memory() is False

    def test_default_state(self):
        """Test default state is False."""
        set_force_memory(False)
        assert is_force_memory() is False

    def test_multiple_toggles(self):
        """Test multiple enable/disable cycles."""
        set_force_memory(True)
        assert is_force_memory() is True
        set_force_memory(False)
        assert is_force_memory() is False
        set_force_memory(True)
        assert is_force_memory() is True


class TestIsForceMemory:
    """Test is_force_memory function."""

    def teardown_method(self):
        """Reset force memory flag after each test."""
        set_force_memory(False)

    def test_returns_true_when_enabled(self):
        """Test returns True when force memory is enabled."""
        set_force_memory(True)
        assert is_force_memory() is True

    def test_returns_false_when_disabled(self):
        """Test returns False when force memory is disabled."""
        set_force_memory(False)
        assert is_force_memory() is False


class TestCheckOperationMemory:
    """Test check_operation_memory function."""

    def teardown_method(self):
        """Reset force memory flag after each test."""
        set_force_memory(False)

    def test_bypass_when_forced(self):
        """Test memory check bypassed when force_memory is enabled."""
        set_force_memory(True)
        result = check_operation_memory("fft", samples=1e9)

        assert isinstance(result, MemoryCheck)
        assert result.sufficient is True
        assert result.available == 0
        assert result.required == 0
        assert "bypassed" in result.recommendation.lower()

    @patch("oscura.core.memory_check.check_memory_available")
    def test_normal_check_when_not_forced(self, mock_check):
        """Test normal memory check when force_memory is disabled."""
        mock_check.return_value = MemoryCheck(
            sufficient=True,
            available=8000000000,
            required=1000000000,
            recommendation="Memory sufficient for operation.",
        )

        result = check_operation_memory("fft", samples=1e6, nfft=8192)

        assert result.sufficient is True
        assert result.available == 8000000000
        assert result.required == 1000000000
        mock_check.assert_called_once_with("fft", 1e6, nfft=8192)

    @patch("oscura.core.memory_check.check_memory_available")
    def test_insufficient_memory(self, mock_check):
        """Test when insufficient memory is available."""
        mock_check.return_value = MemoryCheck(
            sufficient=False,
            available=1000000000,
            required=8000000000,
            recommendation="Use chunked processing or downsample by 8x.",
        )

        result = check_operation_memory("spectrogram", samples=1e9, nperseg=4096)

        assert result.sufficient is False
        assert result.available == 1000000000
        assert result.required == 8000000000

    @patch("oscura.core.memory_check.check_memory_available")
    def test_kwargs_passed_through(self, mock_check):
        """Test that kwargs are passed through to check_memory_available."""
        mock_check.return_value = MemoryCheck(
            sufficient=True,
            available=8000000000,
            required=1000000000,
            recommendation="Memory sufficient for operation.",
        )

        check_operation_memory("spectrogram", samples=1e6, nperseg=2048, noverlap=1024, nfft=4096)

        mock_check.assert_called_once_with(
            "spectrogram", 1e6, nperseg=2048, noverlap=1024, nfft=4096
        )

    @patch("oscura.core.memory_check.check_memory_available")
    def test_none_samples(self, mock_check):
        """Test with None samples parameter."""
        mock_check.return_value = MemoryCheck(
            sufficient=True,
            available=8000000000,
            required=1000000000,
            recommendation="Memory sufficient for operation.",
        )

        check_operation_memory("fft", samples=None)

        mock_check.assert_called_once_with("fft", None)


class TestAutoCheckMemory:
    """Test auto_check_memory function."""

    def teardown_method(self):
        """Reset force memory flag after each test."""
        set_force_memory(False)

    def test_skip_unsupported_operation(self):
        """Test that unsupported operations are skipped."""
        # Should not raise even with impossible parameters
        auto_check_memory("unsupported_op", samples=1e20)

    def test_bypass_when_forced(self):
        """Test that checks are bypassed when force_memory is enabled."""
        set_force_memory(True)
        # Should not raise even with impossible parameters
        auto_check_memory("fft", samples=1e20)

    @patch("oscura.core.memory_check.require_memory")
    def test_calls_require_memory_for_supported_ops(self, mock_require):
        """Test that require_memory is called for supported operations."""
        auto_check_memory("fft", samples=1e6, nfft=8192)

        mock_require.assert_called_once_with("fft", 1e6, nfft=8192)

    @patch("oscura.core.memory_check.require_memory")
    def test_raises_on_insufficient_memory(self, mock_require):
        """Test that MemoryCheckError is raised on insufficient memory."""
        mock_require.side_effect = MemoryCheckError(
            "Insufficient memory for fft",
            required=8000000000,
            available=1000000000,
            recommendation="Use chunked processing.",
        )

        with pytest.raises(MemoryCheckError) as exc_info:
            auto_check_memory("fft", samples=1e9)

        assert "Insufficient memory for fft" in str(exc_info.value)
        assert exc_info.value.required == 8000000000
        assert exc_info.value.available == 1000000000

    @patch("oscura.core.memory_check.require_memory")
    def test_auto_check_operations_set(self, mock_require):
        """Test all operations in AUTO_CHECK_OPERATIONS trigger checks."""
        for operation in ["fft", "psd", "spectrogram", "correlate"]:
            mock_require.reset_mock()
            auto_check_memory(operation, samples=1e6)
            mock_require.assert_called_once()


class TestWithMemoryCheck:
    """Test with_memory_check decorator."""

    def teardown_method(self):
        """Reset force memory flag after each test."""
        set_force_memory(False)

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name, doc, and module."""

        @with_memory_check
        def test_func():
            """Test docstring."""

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test docstring."

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_with_operation_attribute(self, mock_auto_check):
        """Test decorator uses function's operation attribute."""

        def my_transform(signal, samples=None, **kwargs):
            return signal

        # Set operation BEFORE decoration
        my_transform.operation = "fft"
        decorated = with_memory_check(my_transform)

        signal = np.array([1, 2, 3, 4])
        decorated(signal, samples=100, nfft=256)

        # samples is passed as both positional arg and in kwargs
        mock_auto_check.assert_called_once_with("fft", 100, samples=100, nfft=256)

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_without_operation_attribute(self, mock_auto_check):
        """Test decorator uses function name when no operation attribute."""

        @with_memory_check
        def fft(signal, samples=None, **kwargs):
            return signal

        signal = np.array([1, 2, 3, 4])
        fft(signal, samples=100)

        # samples is passed as both positional arg and in kwargs
        mock_auto_check.assert_called_once_with("fft", 100, samples=100)

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_infers_samples_from_array(self, mock_auto_check):
        """Test decorator infers samples from numpy array argument."""

        @with_memory_check
        def fft(signal, **kwargs):
            return signal

        signal = np.array([1, 2, 3, 4, 5])
        fft(signal, nfft=256)

        # Should infer samples=5 from signal length
        mock_auto_check.assert_called_once_with("fft", 5, nfft=256)

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_samples_kwarg_overrides_inference(self, mock_auto_check):
        """Test explicit samples kwarg overrides array inference."""

        @with_memory_check
        def fft(signal, samples=None, **kwargs):
            return signal

        signal = np.array([1, 2, 3, 4, 5])
        result = fft(signal, samples=100, nfft=256)

        # Explicit samples=100 should override inferred samples=5
        # samples is passed as both positional arg and in kwargs
        mock_auto_check.assert_called_once_with("fft", 100, samples=100, nfft=256)
        assert result is signal

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_with_unsupported_operation(self, mock_auto_check):
        """Test decorator with unsupported operation doesn't check."""

        @with_memory_check
        def custom_op(signal, samples=None, **kwargs):
            return signal

        custom_op.operation = "custom_unsupported"

        signal = np.array([1, 2, 3, 4])
        result = custom_op(signal, samples=100)

        # Should not call auto_check_memory since operation not in AUTO_CHECK_OPERATIONS
        mock_auto_check.assert_not_called()
        assert result is signal

    def test_decorator_with_force_memory(self):
        """Test decorator respects force_memory flag."""
        set_force_memory(True)

        @with_memory_check
        def fft(signal, samples=None, **kwargs):
            return signal

        signal = np.array([1, 2, 3, 4])
        # Should not raise even with impossible samples
        result = fft(signal, samples=1e20)
        assert result is signal

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_with_none_samples_no_array(self, mock_auto_check):
        """Test decorator when samples is None and no array arg."""

        @with_memory_check
        def fft(samples=None, **kwargs):
            pass

        fft(nfft=256)

        mock_auto_check.assert_called_once_with("fft", None, nfft=256)

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_passes_return_value(self, mock_auto_check):
        """Test decorator returns function's return value."""

        @with_memory_check
        def fft(signal, samples=None, **kwargs):
            return signal * 2

        signal = np.array([1, 2, 3, 4])
        result = fft(signal, samples=100)

        np.testing.assert_array_equal(result, signal * 2)

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_with_non_numpy_arg(self, mock_auto_check):
        """Test decorator with non-numpy first argument."""

        @with_memory_check
        def fft(data, samples=None, **kwargs):
            return data

        # First arg is not numpy array - use fft (in AUTO_CHECK_OPERATIONS)
        result = fft([1, 2, 3], samples=100)

        # samples is passed as both positional arg and in kwargs
        mock_auto_check.assert_called_once_with("fft", 100, samples=100)
        assert result == [1, 2, 3]


class TestRegisterAutoCheckOperation:
    """Test register_auto_check_operation function."""

    def test_register_new_operation(self):
        """Test registering a new operation."""
        initial_ops = get_auto_check_operations()

        register_auto_check_operation("custom_transform")

        updated_ops = get_auto_check_operations()
        assert "custom_transform" in updated_ops
        assert len(updated_ops) == len(initial_ops) + 1

    def test_register_existing_operation(self):
        """Test registering an operation that already exists."""
        initial_ops = get_auto_check_operations()

        register_auto_check_operation("fft")  # Already exists

        updated_ops = get_auto_check_operations()
        assert len(updated_ops) == len(initial_ops)  # No change in size

    def test_registered_operation_triggers_check(self):
        """Test that registered operation triggers auto check."""
        register_auto_check_operation("my_custom_op")

        with patch("oscura.core.memory_check.require_memory") as mock_require:
            auto_check_memory("my_custom_op", samples=1e6)
            mock_require.assert_called_once()


class TestGetAutoCheckOperations:
    """Test get_auto_check_operations function."""

    def test_returns_set(self):
        """Test that function returns a set."""
        result = get_auto_check_operations()
        assert isinstance(result, set)

    def test_contains_expected_operations(self):
        """Test that result contains expected operations."""
        result = get_auto_check_operations()

        expected_ops = {"fft", "psd", "spectrogram", "eye_diagram", "correlate"}
        assert expected_ops.issubset(result)

    def test_returns_copy(self):
        """Test that function returns a copy, not the original set."""
        result1 = get_auto_check_operations()
        result2 = get_auto_check_operations()

        # Modify result1
        result1.add("test_operation_xyz")

        # result2 should not be affected
        assert "test_operation_xyz" not in result2

    def test_modification_does_not_affect_original(self):
        """Test that modifying returned set doesn't affect the original."""
        result = get_auto_check_operations()
        initial_size = len(result)

        result.add("temporary_op")

        # Get fresh copy and verify original is unchanged
        fresh_result = get_auto_check_operations()
        assert len(fresh_result) == initial_size
        assert "temporary_op" not in fresh_result


class TestMemoryCheckIntegration:
    """Integration tests for memory check module."""

    def teardown_method(self):
        """Reset force memory flag after each test."""
        set_force_memory(False)

    @patch("oscura.core.memory_check.check_memory_available")
    def test_end_to_end_sufficient_memory(self, mock_check):
        """Test complete workflow with sufficient memory."""
        mock_check.return_value = MemoryCheck(
            sufficient=True,
            available=8000000000,
            required=1000000000,
            recommendation="Memory sufficient for operation.",
        )

        # Check memory
        check = check_operation_memory("fft", samples=1e6, nfft=8192)
        assert check.sufficient is True

        # Auto check should not raise
        auto_check_memory("fft", samples=1e6, nfft=8192)

    @patch("oscura.core.memory_check.require_memory")
    def test_end_to_end_insufficient_memory(self, mock_require):
        """Test complete workflow with insufficient memory."""
        mock_require.side_effect = MemoryCheckError(
            "Insufficient memory for fft",
            required=8000000000,
            available=1000000000,
            recommendation="Use chunked processing.",
        )

        # Auto check should raise
        with pytest.raises(MemoryCheckError):
            auto_check_memory("fft", samples=1e9, nfft=8192)

    @patch("oscura.core.memory_check.check_memory_available")
    def test_force_memory_overrides_all_checks(self, mock_check):
        """Test that force_memory bypasses all checks."""
        # Set up mock to return insufficient memory
        mock_check.return_value = MemoryCheck(
            sufficient=False,
            available=1000000000,
            required=8000000000,
            recommendation="Use chunked processing.",
        )

        set_force_memory(True)

        # Check should show sufficient even though underlying check would fail
        check = check_operation_memory("fft", samples=1e9)
        assert check.sufficient is True

        # Auto check should not raise
        auto_check_memory("fft", samples=1e9)

    def test_decorator_integration_with_real_function(self):
        """Test decorator integration with a real function."""

        @with_memory_check
        def compute_fft(signal, samples=None, nfft=None):
            """Compute FFT of signal."""
            return np.fft.fft(signal, n=nfft)

        compute_fft.operation = "fft"

        # With force_memory, should work even with large parameters
        set_force_memory(True)
        signal = np.array([1, 2, 3, 4])
        result = compute_fft(signal, samples=1e9, nfft=8)
        assert len(result) == 8


class TestModuleConstants:
    """Test module-level constants and default values."""

    def test_auto_check_operations_constant(self):
        """Test AUTO_CHECK_OPERATIONS contains expected values."""
        expected_ops = {
            "fft",
            "psd",
            "spectrogram",
            "eye_diagram",
            "correlate",
            "filter",
            "stft",
            "cwt",
            "dwt",
        }
        assert expected_ops.issubset(_AUTO_CHECK_OPERATIONS)

    def test_force_memory_default(self):
        """Test default value of _force_memory is False."""
        # Reset to ensure clean state
        set_force_memory(False)
        assert is_force_memory() is False


class TestCoreMemoryCheckEdgeCases:
    """Test edge cases and error conditions."""

    def teardown_method(self):
        """Reset global state after each test."""
        set_force_memory(False)
        # Reset auto-check operations to prevent test pollution
        _reset_auto_check_operations()

    @patch("oscura.core.memory_check.check_memory_available")
    def test_zero_samples(self, mock_check):
        """Test with zero samples."""
        mock_check.return_value = MemoryCheck(
            sufficient=True,
            available=8000000000,
            required=0,
            recommendation="Memory sufficient for operation.",
        )

        result = check_operation_memory("fft", samples=0)
        assert result.sufficient is True
        mock_check.assert_called_once_with("fft", 0)

    @patch("oscura.core.memory_check.check_memory_available")
    def test_very_large_samples(self, mock_check):
        """Test with very large number of samples."""
        mock_check.return_value = MemoryCheck(
            sufficient=False,
            available=8000000000,
            required=80000000000,
            recommendation="Data too large for available memory.",
        )

        result = check_operation_memory("fft", samples=1e12)
        assert result.sufficient is False

    def test_empty_operation_name(self):
        """Test with empty operation name."""
        with patch("oscura.core.memory_check.require_memory") as mock_require:
            auto_check_memory("", samples=1e6)
            # Empty string not in AUTO_CHECK_OPERATIONS, so should skip
            mock_require.assert_not_called()

    def test_register_empty_operation_name(self):
        """Test registering empty operation name."""
        register_auto_check_operation("")
        ops = get_auto_check_operations()
        assert "" in ops

    @patch("oscura.core.memory_check.auto_check_memory")
    def test_decorator_with_exception_in_function(self, mock_auto_check):
        """Test decorator when wrapped function raises exception."""

        @with_memory_check
        def fft(signal, samples=None):
            raise ValueError("Test error")

        signal = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError, match="Test error"):
            fft(signal, samples=100)

        # Memory check should have been called before exception (fft is in AUTO_CHECK_OPERATIONS)
        mock_auto_check.assert_called_once()

    def test_multiple_threads_force_memory(self):
        """Test that force_memory is a global flag (not thread-safe by design)."""
        # This is a documentation test - the module uses a global flag
        set_force_memory(True)
        assert is_force_memory() is True

        set_force_memory(False)
        assert is_force_memory() is False


class TestMemoryCheckExports:
    """Test that all expected symbols are exported."""

    def test_all_exports_importable(self):
        """Test that all items in __all__ can be imported."""
        from oscura.core.memory_check import __all__

        expected_exports = [
            "MemoryCheck",
            "MemoryCheckError",
            "auto_check_memory",
            "check_operation_memory",
            "get_auto_check_operations",
            "is_force_memory",
            "register_auto_check_operation",
            "unregister_auto_check_operation",
            "set_force_memory",
            "with_memory_check",
        ]

        assert sorted(__all__) == sorted(expected_exports)

    def test_memory_check_re_exported(self):
        """Test that MemoryCheck is re-exported from utils.memory."""
        from oscura.core.memory_check import MemoryCheck as CheckFromCore
        from oscura.utils.memory import MemoryCheck as CheckFromUtils

        assert CheckFromCore is CheckFromUtils

    def test_memory_check_error_re_exported(self):
        """Test that MemoryCheckError is re-exported from utils.memory."""
        from oscura.core.memory_check import MemoryCheckError as ErrorFromCore
        from oscura.utils.memory import MemoryCheckError as ErrorFromUtils

        assert ErrorFromCore is ErrorFromUtils
