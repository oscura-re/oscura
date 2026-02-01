"""Tests for ReverseEngineerSignal."""

from __future__ import annotations

import numpy as np
import pytest

import oscura as osc
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


class TestReverseEngineerSignal:
    """Tests for the reverse_engineer_signal workflow."""

    def test_basic_workflow(self) -> None:
        """Test basic reverse engineering workflow."""
        # Generate UART-like signal
        sample_rate = 1e6
        baud_rate = 19200
        samples_per_bit = int(sample_rate / baud_rate)

        # Create simple pattern: idle, start, data, stop
        bits = [1] * 100  # Idle
        for _ in range(5):  # 5 bytes
            bits.append(0)  # Start bit
            for i in range(8):  # Data bits
                bits.append((0xAA >> i) & 1)
            bits.append(1)  # Stop bit
            bits.extend([1] * 10)  # Gap

        # Expand to samples
        signal_data = []
        for bit in bits:
            signal_data.extend([bit * 3.3] * samples_per_bit)

        signal_data = np.array(signal_data) + 0.05 * np.random.randn(len(signal_data))

        trace = WaveformTrace(
            data=signal_data, metadata=TraceMetadata(sample_rate=sample_rate, channel="test")
        )

        result = osc.workflows.reverse_engineer_signal(trace)

        # Should detect something
        assert result.baud_rate > 0
        assert len(result.bit_stream) > 100
        assert result.confidence > 0
