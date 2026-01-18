"""Property-based tests for template matching."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.pattern, pytest.mark.hypothesis]


class TestTemplateMatchingProperties:
    """Property-based tests for template matching."""

    @given(
        template_length=st.integers(min_value=10, max_value=50),
        signal_length=st.integers(min_value=100, max_value=1000),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_template_found_in_signal(
        self, template_length: int, signal_length: int, seed: int
    ) -> None:
        """Property: Template is found when embedded in signal using normalized correlation."""
        # Need enough space for prefix + template + suffix
        prefix_len = 10
        min_required = prefix_len + template_length + 10  # Add margin for suffix
        assume(signal_length >= min_required)

        rng = np.random.default_rng(seed)

        # Use a distinctive template with a clear pattern
        # Sine wave with multiple periods ensures uniqueness
        template = np.sin(np.linspace(0, 4 * np.pi, template_length))

        # Embed template in signal with random noise
        suffix_len = signal_length - prefix_len - template_length
        assert suffix_len >= 0, "Suffix length must be non-negative"

        # Create signal with noise
        signal = np.concatenate(
            [
                0.1 * rng.standard_normal(prefix_len),
                template,
                0.1 * rng.standard_normal(suffix_len),
            ]
        )

        # Normalized cross-correlation to find template
        # Normalize both template and signal sections
        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)

        # Compute correlation for each valid position
        correlations = []
        for i in range(len(signal) - len(template) + 1):
            signal_section = signal[i : i + len(template)]
            signal_norm = (signal_section - np.mean(signal_section)) / (
                np.std(signal_section) + 1e-10
            )
            corr = np.sum(template_norm * signal_norm) / len(template)
            correlations.append(corr)

        max_corr_idx = np.argmax(correlations)

        # Should find template at expected position (within small tolerance)
        assert abs(max_corr_idx - prefix_len) <= 1

    @given(template_length=st.integers(min_value=5, max_value=50))
    @settings(max_examples=30, deadline=None)
    def test_template_matches_itself_perfectly(self, template_length: int) -> None:
        """Property: Template matches itself with correlation = 1."""
        rng = np.random.default_rng(42)
        template = rng.random(template_length)

        # Normalize
        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)

        # Cross-correlation with itself
        correlation = np.correlate(template_norm, template_norm, mode="valid")

        # Should be maximum (1.0 when normalized)
        max_corr = np.max(np.abs(correlation))
        assert max_corr >= template_length * 0.9  # Close to perfect match
