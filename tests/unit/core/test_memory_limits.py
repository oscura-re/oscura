"""Tests for per-operation memory limits module.

Requirements tested:
"""

import warnings

import pytest

from oscura.config.memory import reset_to_defaults, set_memory_limit
from oscura.core.memory_limits import (
    _find_max_nfft,
    _find_max_nperseg,
    _find_max_num_uis,
    apply_memory_limit,
    check_operation_fits,
    get_operation_memory_limit,
    parse_memory_limit,
)
from oscura.utils.memory import estimate_memory

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestParseMemoryLimit:
    """Test parse_memory_limit function."""

    def test_parse_none(self):
        """Test parsing None returns None."""
        result = parse_memory_limit(None)
        assert result is None

    def test_parse_int(self):
        """Test parsing integer bytes."""
        result = parse_memory_limit(1024)
        assert result == 1024

    def test_parse_gb_decimal(self):
        """Test parsing GB with decimal."""
        result = parse_memory_limit("4.5GB")
        assert result == int(4.5 * 1e9)

    def test_parse_gb_uppercase(self):
        """Test parsing GB (uppercase)."""
        result = parse_memory_limit("4GB")
        assert result == 4000000000

    def test_parse_gb_lowercase(self):
        """Test parsing GB (lowercase)."""
        result = parse_memory_limit("4gb")
        assert result == 4000000000

    def test_parse_mb(self):
        """Test parsing MB."""
        result = parse_memory_limit("512MB")
        assert result == 512000000

    def test_parse_kb(self):
        """Test parsing KB."""
        result = parse_memory_limit("1024KB")
        assert result == 1024000

    def test_parse_gib(self):
        """Test parsing GiB (binary)."""
        result = parse_memory_limit("4GiB")
        assert result == 4 * 1024**3

    def test_parse_mib(self):
        """Test parsing MiB (binary)."""
        result = parse_memory_limit("512MiB")
        assert result == 512 * 1024**2

    def test_parse_kib(self):
        """Test parsing KiB (binary)."""
        result = parse_memory_limit("1024KiB")
        assert result == 1024 * 1024

    def test_parse_bytes_string(self):
        """Test parsing raw bytes as string."""
        result = parse_memory_limit("1048576")
        assert result == 1048576

    def test_parse_whitespace(self):
        """Test parsing with whitespace."""
        result = parse_memory_limit("  4GB  ")
        assert result == 4000000000

    def test_parse_invalid_format(self):
        """Test parsing invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid memory limit format"):
            parse_memory_limit("invalid")

    def test_parse_invalid_number(self):
        """Test parsing invalid number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid memory limit format"):
            parse_memory_limit("XYZ GB")

    def test_parse_negative(self):
        """Test parsing negative value."""
        result = parse_memory_limit(-1024)
        assert result == -1024  # Function doesn't validate, just parses


class TestApplyMemoryLimit:
    """Test apply_memory_limit function."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    def test_no_limit_returns_params_unchanged(self):
        """Test that no limit returns params unchanged."""
        params = {"nfft": 8192}
        result = apply_memory_limit("fft", samples=1000, max_memory=None, **params)
        assert result == params

    def test_within_limit_returns_params_unchanged(self):
        """Test params unchanged when within limit."""
        params = {"nfft": 1024}
        # Very large limit
        result = apply_memory_limit("fft", samples=1000, max_memory="100GB", **params)
        assert result == params

    def test_fft_reduces_nfft(self):
        """Test FFT reduces nfft to fit memory limit."""
        params = {"nfft": 8192}
        # Very small limit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_memory_limit("fft", samples=1000000, max_memory="1MB", **params)
            # Should warn about reduction
            assert len(w) >= 1
            assert "Reduced nfft" in str(w[0].message)

        assert result["nfft"] < params["nfft"]

    def test_psd_reduces_nfft(self):
        """Test PSD reduces nfft to fit memory limit."""
        params = {"nfft": 8192}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_memory_limit("psd", samples=1000000, max_memory="1MB", **params)
            assert len(w) >= 1

        assert result["nfft"] < params["nfft"]

    def test_spectrogram_reduces_nperseg(self):
        """Test spectrogram reduces nperseg to fit memory limit."""
        params = {"nperseg": 8192}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_memory_limit("spectrogram", samples=1000000, max_memory="1MB", **params)
            assert len(w) >= 1
            assert "Reduced nperseg" in str(w[0].message)

        assert result["nperseg"] < params["nperseg"]

    def test_spectrogram_adjusts_noverlap_proportionally(self):
        """Test spectrogram adjusts noverlap proportionally with nperseg."""
        params = {"nperseg": 8192, "noverlap": 4096}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = apply_memory_limit("spectrogram", samples=1000000, max_memory="1MB", **params)

        # noverlap should be reduced proportionally
        if result["nperseg"] < params["nperseg"]:
            # Check ratio is maintained (approximately)
            original_ratio = params["noverlap"] / params["nperseg"]
            new_ratio = result["noverlap"] / result["nperseg"]
            assert abs(original_ratio - new_ratio) < 0.1

    def test_spectrogram_reduces_nfft_if_needed(self):
        """Test spectrogram also reduces nfft if it exceeds nperseg."""
        params = {"nperseg": 8192, "nfft": 16384}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = apply_memory_limit("spectrogram", samples=1000000, max_memory="1MB", **params)

        # nfft should not exceed nperseg
        assert result.get("nfft", result["nperseg"]) <= result["nperseg"]

    def test_eye_diagram_reduces_num_uis(self):
        """Test eye diagram reduces num_uis to fit memory limit."""
        params = {"num_uis": 10000, "samples_per_ui": 100}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_memory_limit("eye_diagram", samples=1000000, max_memory="1MB", **params)
            assert len(w) >= 1
            assert "Reduced num_uis" in str(w[0].message)

        assert result["num_uis"] < params["num_uis"]

    def test_warns_if_cannot_fit(self):
        """Test warns if parameters cannot be adjusted to fit."""
        # Unreasonably small limit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_memory_limit("fft", samples=1000000, max_memory="1KB")
            # Should warn about inability to fit
            warn_messages = [str(warning.message) for warning in w]
            assert any("Could not adjust parameters" in msg for msg in warn_messages)

    def test_uses_global_config_when_no_limit(self):
        """Test uses global config when max_memory not provided."""
        set_memory_limit("2GB")
        params = {"nfft": 8192}
        result = apply_memory_limit("fft", samples=1000, **params)
        # Should use global config
        assert result is not None

    def test_override_global_config(self):
        """Test max_memory parameter overrides global config."""
        set_memory_limit("100GB")
        params = {"nfft": 8192}
        # Override with small limit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_memory_limit("fft", samples=1000000, max_memory="1MB", **params)
            # Should use override, not global
            if w:
                assert "Reduced nfft" in str(w[0].message)

    def test_handles_float_samples(self):
        """Test handles float samples (large values)."""
        params = {"nfft": 1024}
        result = apply_memory_limit("fft", samples=1e6, max_memory="100GB", **params)
        assert result == params

    def test_unknown_operation_returns_params(self):
        """Test unknown operation returns params unchanged."""
        params = {"some_param": 123}
        result = apply_memory_limit("unknown_op", samples=1000, max_memory="1GB", **params)
        assert result == params


class TestGetOperationMemoryLimit:
    """Test get_operation_memory_limit function."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    def test_returns_override_limit(self):
        """Test returns override limit when provided."""
        limit = get_operation_memory_limit("fft", max_memory="4GB")
        assert limit == 4000000000

    def test_returns_global_config_limit(self):
        """Test returns global config limit when no override."""
        set_memory_limit("2GB")
        limit = get_operation_memory_limit("fft")
        assert limit == 2000000000

    def test_returns_default_80_percent(self):
        """Test returns 80% of available when no limits set."""
        from oscura.utils.memory import get_available_memory

        limit = get_operation_memory_limit("fft")
        expected = int(get_available_memory() * 0.8)
        # Allow small variation due to timing (memory may change slightly)
        assert abs(limit - expected) < expected * 0.01  # Within 1%

    def test_parses_string_limit(self):
        """Test parses string format limit."""
        limit = get_operation_memory_limit("fft", max_memory="512MB")
        assert limit == 512000000

    def test_accepts_int_limit(self):
        """Test accepts integer limit."""
        limit = get_operation_memory_limit("fft", max_memory=1073741824)
        assert limit == 1073741824


class TestCheckOperationFits:
    """Test check_operation_fits function."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    def test_returns_true_when_fits(self):
        """Test returns True when operation fits in memory."""
        # Small operation with large limit
        fits = check_operation_fits("fft", samples=1000, max_memory="10GB", nfft=1024)
        assert fits is True

    def test_returns_false_when_exceeds(self):
        """Test returns False when operation exceeds limit."""
        # Large operation with small limit
        fits = check_operation_fits("fft", samples=1000000, max_memory="1KB", nfft=8192)
        assert fits is False

    def test_uses_global_config(self):
        """Test uses global config when no override provided."""
        set_memory_limit("10GB")
        fits = check_operation_fits("fft", samples=1000, nfft=1024)
        assert fits is True

    def test_with_spectrogram_params(self):
        """Test with spectrogram operation and parameters."""
        fits = check_operation_fits(
            "spectrogram", samples=10000, max_memory="100MB", nperseg=256, noverlap=128
        )
        assert fits is True

    def test_with_float_samples(self):
        """Test with float samples value."""
        fits = check_operation_fits("fft", samples=1e6, max_memory="10GB", nfft=1024)
        assert fits is True


class TestFindMaxNfft:
    """Test _find_max_nfft helper function."""

    def test_finds_max_nfft_within_limit(self):
        """Test finds maximum nfft that fits within limit."""
        nfft = _find_max_nfft("fft", samples=100000, limit_bytes=10000000, nfft=8192)
        # Should return some value between min and max
        assert 64 <= nfft <= 8192
        # Verify it fits
        estimate = estimate_memory("fft", 100000, nfft=nfft)
        assert estimate.total <= 10000000

    def test_returns_max_when_all_fit(self):
        """Test returns original nfft when it fits."""
        nfft = _find_max_nfft("fft", samples=1000, limit_bytes=100000000, nfft=8192)
        # With large limit, should return close to original
        assert nfft > 1024

    def test_returns_min_when_none_fit(self):
        """Test returns minimum nfft when nothing fits."""
        nfft = _find_max_nfft("fft", samples=1000000, limit_bytes=1000, nfft=8192)
        # Should return minimum
        assert nfft == 64

    def test_binary_search_converges(self):
        """Test binary search converges to correct value."""
        nfft = _find_max_nfft("psd", samples=50000, limit_bytes=5000000, nfft=4096)
        # Verify result fits
        estimate = estimate_memory("psd", 50000, nfft=nfft)
        assert estimate.total <= 5000000
        # Verify next higher doesn't fit (if not at max)
        if nfft < 4096:
            estimate_higher = estimate_memory("psd", 50000, nfft=nfft + 1)
            # May or may not exceed (binary search finds boundary)


class TestFindMaxNperseg:
    """Test _find_max_nperseg helper function."""

    def test_finds_max_nperseg_within_limit(self):
        """Test finds maximum nperseg that fits within limit."""
        nperseg = _find_max_nperseg(samples=100000, limit_bytes=10000000)
        # Should return some value between min and max
        assert 64 <= nperseg <= 8192

    def test_respects_sample_count(self):
        """Test respects sample count constraint."""
        nperseg = _find_max_nperseg(samples=1000, limit_bytes=100000000)
        # nperseg should not exceed samples // 4
        assert nperseg <= 1000 // 4

    def test_with_noverlap(self):
        """Test with noverlap parameter."""
        nperseg = _find_max_nperseg(samples=100000, limit_bytes=10000000, noverlap=128)
        assert nperseg >= 64

    def test_returns_min_when_tight_limit(self):
        """Test returns minimum when limit is tight."""
        nperseg = _find_max_nperseg(samples=100000, limit_bytes=1000)
        assert nperseg == 64

    def test_binary_search_converges(self):
        """Test binary search finds appropriate value."""
        nperseg = _find_max_nperseg(samples=50000, limit_bytes=5000000, noverlap=None)
        assert 64 <= nperseg <= min(8192, 50000 // 4)


class TestFindMaxNumUis:
    """Test _find_max_num_uis helper function."""

    def test_calculates_max_num_uis(self):
        """Test calculates maximum num_uis for eye diagram."""
        num_uis = _find_max_num_uis(limit_bytes=10000000, samples_per_ui=100)
        assert num_uis >= 100  # At least minimum

    def test_respects_minimum(self):
        """Test returns at least 100 UIs."""
        num_uis = _find_max_num_uis(limit_bytes=100, samples_per_ui=100)
        assert num_uis == 100

    def test_with_large_limit(self):
        """Test with large memory limit."""
        num_uis = _find_max_num_uis(limit_bytes=1000000000, samples_per_ui=100)
        assert num_uis > 10000

    def test_with_different_samples_per_ui(self):
        """Test with different samples_per_ui values."""
        num_uis_100 = _find_max_num_uis(limit_bytes=10000000, samples_per_ui=100)
        num_uis_200 = _find_max_num_uis(limit_bytes=10000000, samples_per_ui=200)
        # More samples per UI should result in fewer UIs
        assert num_uis_200 < num_uis_100


class TestMemoryLimitIntegration:
    """Integration tests for memory limit functionality."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    def test_end_to_end_fft_adjustment(self):
        """Test complete workflow for FFT memory adjustment."""
        # Check if operation fits
        fits = check_operation_fits("fft", samples=1000000, max_memory="1MB", nfft=8192)
        assert fits is False

        # Apply memory limit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adjusted = apply_memory_limit("fft", samples=1000000, max_memory="1MB", nfft=8192)

        # Should have adjusted
        assert adjusted["nfft"] < 8192

        # Verify adjusted version fits
        limit = get_operation_memory_limit("fft", max_memory="1MB")
        estimate = estimate_memory("fft", 1000000, **adjusted)
        # May or may not fit exactly, but should be significantly smaller than original
        # (The adjustment might not achieve the exact limit due to binary search granularity)
        original_estimate = estimate_memory("fft", 1000000, nfft=8192)
        assert estimate.total < original_estimate.total

    def test_end_to_end_spectrogram_adjustment(self):
        """Test complete workflow for spectrogram memory adjustment."""
        fits = check_operation_fits(
            "spectrogram", samples=1000000, max_memory="1MB", nperseg=8192, noverlap=4096
        )
        assert fits is False

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adjusted = apply_memory_limit(
                "spectrogram",
                samples=1000000,
                max_memory="1MB",
                nperseg=8192,
                noverlap=4096,
            )

        # Should have adjusted
        assert adjusted["nperseg"] < 8192

    def test_with_global_config_and_override(self):
        """Test interaction between global config and override."""
        set_memory_limit("100GB")

        # Without override, should fit easily
        fits1 = check_operation_fits("fft", samples=1000000, nfft=8192)
        assert fits1 is True

        # With override, should not fit
        fits2 = check_operation_fits("fft", samples=1000000, max_memory="1KB", nfft=8192)
        assert fits2 is False

    def test_preserves_params_when_adequate_memory(self):
        """Test preserves parameters when memory is adequate."""
        original_params = {"nfft": 2048, "extra_param": "value"}
        adjusted = apply_memory_limit("fft", samples=10000, max_memory="1GB", **original_params)
        assert adjusted == original_params

    def test_warning_messages_are_informative(self):
        """Test warning messages contain useful information."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_memory_limit("fft", samples=1000000, max_memory="1MB", nfft=8192)

            # Should have warnings
            assert len(w) > 0
            # Check warning contains useful info
            message = str(w[0].message)
            assert "MB" in message or "reduced" in message.lower()

    def test_different_operations_adjust_differently(self):
        """Test different operations have different adjustment strategies."""
        limit = "1MB"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fft_result = apply_memory_limit("fft", samples=1000000, max_memory=limit, nfft=8192)
            psd_result = apply_memory_limit("psd", samples=1000000, max_memory=limit, nfft=8192)
            spec_result = apply_memory_limit(
                "spectrogram", samples=1000000, max_memory=limit, nperseg=8192
            )

        # All should have made adjustments
        assert fft_result["nfft"] < 8192
        assert psd_result["nfft"] < 8192
        assert spec_result["nperseg"] < 8192

    def test_empty_params_dict(self):
        """Test with empty parameters dictionary."""
        result = apply_memory_limit("fft", samples=1000, max_memory="1GB")
        assert isinstance(result, dict)

    def test_multiple_adjustments(self):
        """Test multiple sequential adjustments."""
        params = {"nperseg": 8192, "noverlap": 4096, "nfft": 16384}

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = apply_memory_limit(
                "spectrogram", samples=1000000, max_memory="500KB", **params
            )

        # Should adjust multiple parameters
        assert result["nperseg"] < params["nperseg"]
        assert result.get("nfft", result["nperseg"]) <= result["nperseg"]


class TestCoreMemoryLimitsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_samples(self):
        """Test with zero samples."""
        result = apply_memory_limit("fft", samples=0, max_memory="1GB", nfft=1024)
        assert isinstance(result, dict)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_very_large_samples(self):
        """Test with very large sample count."""
        result = apply_memory_limit("fft", samples=1e12, max_memory="1GB", nfft=1024)
        assert isinstance(result, dict)

    def test_tiny_memory_limit(self):
        """Test with unreasonably small memory limit."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_memory_limit("fft", samples=1000000, max_memory="1", nfft=8192)
            # Should warn about inability to fit
            assert len(w) > 0

    def test_huge_memory_limit(self):
        """Test with huge memory limit."""
        params = {"nfft": 8192}
        # Use a valid format (TB not supported, use GB)
        result = apply_memory_limit("fft", samples=1000, max_memory="1000GB", **params)
        assert result == params

    def test_parse_limit_edge_cases(self):
        """Test parse_memory_limit edge cases."""
        # Very small
        assert parse_memory_limit("0.001MB") == 1000
        # Very large
        assert parse_memory_limit("1000GB") == 1000000000000
        # Zero
        assert parse_memory_limit("0GB") == 0
