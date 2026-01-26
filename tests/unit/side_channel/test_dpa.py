"""Tests for DPA (Differential Power Analysis) module.

Test coverage:
- Hamming weight calculation
- Hamming distance calculation
- AES S-box output
- Hypothetical power calculation
- DPA attack
- CPA attack
- Template attack
- Visualization
- Export functionality
- Edge cases and error handling
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oscura.side_channel.dpa import AES_SBOX, DPAAnalyzer, DPAResult, PowerTrace


class TestPowerTrace:
    """Tests for PowerTrace dataclass."""

    def test_power_trace_creation(self) -> None:
        """Test basic PowerTrace creation."""
        timestamp = np.linspace(0, 1e-6, 1000)
        power = np.random.randn(1000)
        plaintext = bytes(range(16))

        trace = PowerTrace(
            timestamp=timestamp,
            power=power,
            plaintext=plaintext,
        )

        assert len(trace.timestamp) == 1000
        assert len(trace.power) == 1000
        assert trace.plaintext == plaintext
        assert trace.ciphertext is None
        assert trace.metadata == {}

    def test_power_trace_with_metadata(self) -> None:
        """Test PowerTrace with metadata."""
        trace = PowerTrace(
            timestamp=np.arange(100),
            power=np.random.randn(100),
            plaintext=bytes(16),
            ciphertext=bytes(16),
            metadata={"device": "STM32", "temperature": 25.0},
        )

        assert trace.metadata["device"] == "STM32"
        assert trace.metadata["temperature"] == 25.0


class TestDPAAnalyzer:
    """Tests for DPAAnalyzer class."""

    def test_analyzer_initialization_cpa(self) -> None:
        """Test CPA analyzer initialization."""
        analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")

        assert analyzer.attack_type == "cpa"
        assert analyzer.leakage_model == "hamming_weight"
        assert analyzer.templates == {}

    def test_analyzer_initialization_dpa(self) -> None:
        """Test DPA analyzer initialization."""
        analyzer = DPAAnalyzer(attack_type="dpa", leakage_model="hamming_distance")

        assert analyzer.attack_type == "dpa"
        assert analyzer.leakage_model == "hamming_distance"

    def test_analyzer_invalid_attack_type(self) -> None:
        """Test invalid attack type raises error."""
        with pytest.raises(ValueError, match="Invalid attack_type"):
            DPAAnalyzer(attack_type="invalid")

    def test_analyzer_invalid_leakage_model(self) -> None:
        """Test invalid leakage model raises error."""
        with pytest.raises(ValueError, match="Invalid leakage_model"):
            DPAAnalyzer(attack_type="cpa", leakage_model="invalid")

    def test_hamming_weight_zero(self) -> None:
        """Test Hamming weight of zero."""
        analyzer = DPAAnalyzer()
        assert analyzer._hamming_weight(0x00) == 0

    def test_hamming_weight_all_ones(self) -> None:
        """Test Hamming weight of all ones."""
        analyzer = DPAAnalyzer()
        assert analyzer._hamming_weight(0xFF) == 8

    def test_hamming_weight_mixed(self) -> None:
        """Test Hamming weight of mixed bits."""
        analyzer = DPAAnalyzer()
        assert analyzer._hamming_weight(0x0F) == 4  # 0b00001111
        assert analyzer._hamming_weight(0xAA) == 4  # 0b10101010
        assert analyzer._hamming_weight(0x55) == 4  # 0b01010101

    def test_hamming_distance_identical(self) -> None:
        """Test Hamming distance between identical values."""
        analyzer = DPAAnalyzer()
        assert analyzer._hamming_distance(0x42, 0x42) == 0

    def test_hamming_distance_opposite(self) -> None:
        """Test Hamming distance between opposite values."""
        analyzer = DPAAnalyzer()
        assert analyzer._hamming_distance(0x00, 0xFF) == 8

    def test_hamming_distance_mixed(self) -> None:
        """Test Hamming distance for various values."""
        analyzer = DPAAnalyzer()
        assert analyzer._hamming_distance(0x0F, 0x0E) == 1  # differ in 1 bit
        assert analyzer._hamming_distance(0xAA, 0x55) == 8  # all bits differ

    def test_aes_sbox_output(self) -> None:
        """Test AES S-box output calculation."""
        analyzer = DPAAnalyzer()

        # Test known S-box values
        assert analyzer._aes_sbox_output(0x00, 0x00) == AES_SBOX[0x00]
        assert analyzer._aes_sbox_output(0xFF, 0x00) == AES_SBOX[0xFF]
        assert analyzer._aes_sbox_output(0x00, 0xFF) == AES_SBOX[0xFF]

    def test_aes_sbox_output_xor(self) -> None:
        """Test AES S-box XOR operation."""
        analyzer = DPAAnalyzer()

        # plaintext ^ key should be used as S-box index
        plaintext = 0x42
        key = 0x17
        expected_index = plaintext ^ key
        assert analyzer._aes_sbox_output(plaintext, key) == AES_SBOX[expected_index]

    def test_calculate_hypothetical_power_hamming_weight(self) -> None:
        """Test hypothetical power with Hamming weight model."""
        analyzer = DPAAnalyzer(leakage_model="hamming_weight")

        plaintexts = [bytes([0x00] * 16), bytes([0xFF] * 16)]
        key_guess = 0x42
        target_byte = 0

        hyp_power = analyzer._calculate_hypothetical_power(plaintexts, key_guess, target_byte)

        assert len(hyp_power) == 2
        # All values should be Hamming weights (0-8)
        assert all(0 <= p <= 8 for p in hyp_power)

    def test_calculate_hypothetical_power_hamming_distance(self) -> None:
        """Test hypothetical power with Hamming distance model."""
        analyzer = DPAAnalyzer(leakage_model="hamming_distance")

        plaintexts = [bytes([0x00] * 16), bytes([0xFF] * 16)]
        key_guess = 0x42
        target_byte = 0

        hyp_power = analyzer._calculate_hypothetical_power(plaintexts, key_guess, target_byte)

        assert len(hyp_power) == 2
        # All values should be Hamming distances (0-8)
        assert all(0 <= p <= 8 for p in hyp_power)

    def test_calculate_hypothetical_power_identity(self) -> None:
        """Test hypothetical power with identity model."""
        analyzer = DPAAnalyzer(leakage_model="identity")

        plaintexts = [bytes([0x00] * 16), bytes([0xFF] * 16)]
        key_guess = 0x42
        target_byte = 0

        hyp_power = analyzer._calculate_hypothetical_power(plaintexts, key_guess, target_byte)

        assert len(hyp_power) == 2
        # All values should be S-box outputs (0-255)
        assert all(0 <= p <= 255 for p in hyp_power)

    def test_calculate_hypothetical_power_none_plaintext(self) -> None:
        """Test hypothetical power with None plaintexts."""
        analyzer = DPAAnalyzer()

        plaintexts = [None, bytes([0x42] * 16)]
        hyp_power = analyzer._calculate_hypothetical_power(plaintexts, 0x00, 0)

        assert len(hyp_power) == 2
        assert hyp_power[0] == 0.0  # None plaintext -> 0 power


class TestDPAAttack:
    """Tests for DPA attack method."""

    def generate_synthetic_traces(
        self,
        num_traces: int,
        num_samples: int,
        true_key_byte: int,
        target_byte: int = 0,
        noise_level: float = 1.0,
    ) -> list[PowerTrace]:
        """Generate synthetic power traces with leakage.

        Args:
            num_traces: Number of traces to generate.
            num_samples: Number of samples per trace.
            true_key_byte: Actual key byte value.
            target_byte: Target byte position.
            noise_level: Noise standard deviation.

        Returns:
            List of PowerTrace objects.
        """
        traces = []
        rng = np.random.RandomState(42)  # Deterministic

        for _ in range(num_traces):
            # Random plaintext
            plaintext = bytes(rng.randint(0, 256, 16))

            # Generate power trace with leakage at specific point
            power = rng.randn(num_samples) * noise_level

            # Add leakage at sample 100 (Hamming weight of S-box output)
            intermediate = AES_SBOX[plaintext[target_byte] ^ true_key_byte]
            hw = bin(intermediate).count("1")
            power[100] += hw * 2.0  # Strong leakage signal

            traces.append(
                PowerTrace(
                    timestamp=np.arange(num_samples),
                    power=power,
                    plaintext=plaintext,
                )
            )

        return traces

    def test_dpa_attack_success(self) -> None:
        """Test successful DPA attack."""
        true_key = 0x42
        traces = self.generate_synthetic_traces(
            num_traces=200,
            num_samples=500,
            true_key_byte=true_key,
            noise_level=0.5,
        )

        analyzer = DPAAnalyzer(attack_type="dpa")
        result = analyzer.dpa_attack(traces, target_byte=0)

        # Should recover correct key (confidence may vary with DPA)
        assert result.recovered_key[0] == true_key
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.key_ranks) == 256

    def test_dpa_attack_no_traces(self) -> None:
        """Test DPA attack with no traces."""
        analyzer = DPAAnalyzer(attack_type="dpa")

        with pytest.raises(ValueError, match="No traces provided"):
            analyzer.dpa_attack([], target_byte=0)

    def test_dpa_attack_no_plaintexts(self) -> None:
        """Test DPA attack with no plaintexts."""
        analyzer = DPAAnalyzer(attack_type="dpa")

        traces = [
            PowerTrace(timestamp=np.arange(100), power=np.random.randn(100)) for _ in range(10)
        ]

        with pytest.raises(ValueError, match="No plaintexts in traces"):
            analyzer.dpa_attack(traces, target_byte=0)


class TestCPAAttack:
    """Tests for CPA (Correlation Power Analysis) attack."""

    def generate_synthetic_traces(
        self,
        num_traces: int,
        num_samples: int,
        true_key_byte: int,
        target_byte: int = 0,
        noise_level: float = 1.0,
    ) -> list[PowerTrace]:
        """Generate synthetic power traces with Hamming weight leakage."""
        traces = []
        rng = np.random.RandomState(42)

        for _ in range(num_traces):
            plaintext = bytes(rng.randint(0, 256, 16))
            power = rng.randn(num_samples) * noise_level

            # Add Hamming weight leakage at sample 100
            intermediate = AES_SBOX[plaintext[target_byte] ^ true_key_byte]
            hw = bin(intermediate).count("1")
            power[100] += hw * 3.0  # Strong correlation

            traces.append(
                PowerTrace(
                    timestamp=np.arange(num_samples),
                    power=power,
                    plaintext=plaintext,
                )
            )

        return traces

    def test_cpa_attack_success(self) -> None:
        """Test successful CPA attack."""
        true_key = 0x2A
        traces = self.generate_synthetic_traces(
            num_traces=200,
            num_samples=500,
            true_key_byte=true_key,
            noise_level=0.3,
        )

        analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")
        result = analyzer.cpa_attack(traces, target_byte=0)

        # Should recover correct key
        assert result.recovered_key[0] == true_key
        assert result.confidence > 0.5
        assert result.correlation_traces is not None
        assert result.correlation_traces.shape == (256, 500)

    def test_cpa_attack_high_noise(self) -> None:
        """Test CPA attack with high noise."""
        true_key = 0x7F
        traces = self.generate_synthetic_traces(
            num_traces=50,  # Few traces
            num_samples=200,
            true_key_byte=true_key,
            noise_level=5.0,  # High noise
        )

        analyzer = DPAAnalyzer(attack_type="cpa")
        result = analyzer.cpa_attack(traces, target_byte=0)

        # May not recover correct key, but should not crash
        assert len(result.recovered_key) == 1
        assert 0 <= result.confidence <= 1.0

    def test_cpa_attack_no_traces(self) -> None:
        """Test CPA attack with no traces."""
        analyzer = DPAAnalyzer(attack_type="cpa")

        with pytest.raises(ValueError, match="No traces provided"):
            analyzer.cpa_attack([], target_byte=0)

    def test_cpa_attack_constant_power(self) -> None:
        """Test CPA attack with constant power traces."""
        rng = np.random.RandomState(42)
        traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=np.ones(100) * 5.0 + rng.randn(100) * 0.01,  # Nearly constant
                plaintext=bytes(range(16)),
            )
            for _ in range(50)
        ]

        analyzer = DPAAnalyzer(attack_type="cpa")
        # Suppress runtime warnings for near-constant signals
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = analyzer.cpa_attack(traces, target_byte=0)

        # Should handle gracefully (all correlations near 0)
        assert len(result.recovered_key) == 1
        assert result.correlation_traces is not None


class TestTemplateAttack:
    """Tests for template attack."""

    def generate_profiling_traces(
        self,
        num_traces: int,
        num_samples: int,
        known_key_byte: int = 0x00,
    ) -> list[PowerTrace]:
        """Generate profiling traces with known key."""
        traces = []
        rng = np.random.RandomState(42)

        for _ in range(num_traces):
            plaintext = bytes(rng.randint(0, 256, 16))
            power = rng.randn(num_samples) * 0.5

            # Add leakage
            intermediate = AES_SBOX[plaintext[0] ^ known_key_byte]
            hw = bin(intermediate).count("1")
            power[50] += hw * 2.0

            traces.append(
                PowerTrace(
                    timestamp=np.arange(num_samples),
                    power=power,
                    plaintext=plaintext,
                )
            )

        return traces

    def test_template_attack_success(self) -> None:
        """Test template attack with profiling."""
        profiling_traces = self.generate_profiling_traces(
            num_traces=100,
            num_samples=200,
            known_key_byte=0x00,
        )

        attack_traces = self.generate_profiling_traces(
            num_traces=20,
            num_samples=200,
            known_key_byte=0x00,  # Same key for simplicity
        )

        analyzer = DPAAnalyzer(attack_type="template")
        # Suppress warnings for template attack calculations
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = analyzer.template_attack(
                profiling_traces=profiling_traces,
                attack_traces=attack_traces,
                target_byte=0,
            )

        assert len(result.recovered_key) == 1
        assert 0 <= result.confidence <= 1.0

    def test_template_attack_no_profiling_traces(self) -> None:
        """Test template attack without profiling traces."""
        analyzer = DPAAnalyzer(attack_type="template")

        attack_traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=np.random.randn(100),
                plaintext=bytes(range(16)),
            )
        ]

        with pytest.raises(ValueError, match="Both profiling and attack traces required"):
            analyzer.template_attack(
                profiling_traces=[],
                attack_traces=attack_traces,
                target_byte=0,
            )

    def test_template_attack_no_attack_traces(self) -> None:
        """Test template attack without attack traces."""
        analyzer = DPAAnalyzer(attack_type="template")

        profiling_traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=np.random.randn(100),
                plaintext=bytes(range(16)),
            )
        ]

        with pytest.raises(ValueError, match="Both profiling and attack traces required"):
            analyzer.template_attack(
                profiling_traces=profiling_traces,
                attack_traces=[],
                target_byte=0,
            )


class TestPerformAttack:
    """Tests for perform_attack dispatch method."""

    def test_perform_attack_dpa(self) -> None:
        """Test perform_attack dispatches to DPA."""
        traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=np.random.randn(100),
                plaintext=bytes(range(16)),
            )
            for _ in range(50)
        ]

        analyzer = DPAAnalyzer(attack_type="dpa")
        result = analyzer.perform_attack(traces, target_byte=0)

        assert len(result.recovered_key) == 1

    def test_perform_attack_cpa(self) -> None:
        """Test perform_attack dispatches to CPA."""
        rng = np.random.RandomState(42)
        traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=rng.randn(100) * 2.0 + rng.rand() * 0.1,  # Add variation
                plaintext=bytes(rng.randint(0, 256, 16)),
            )
            for _ in range(50)
        ]

        analyzer = DPAAnalyzer(attack_type="cpa")
        result = analyzer.perform_attack(traces, target_byte=0)

        assert len(result.recovered_key) == 1
        assert result.correlation_traces is not None

    def test_perform_attack_template_raises(self) -> None:
        """Test perform_attack raises for template attack."""
        traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=np.random.randn(100),
                plaintext=bytes(range(16)),
            )
        ]

        analyzer = DPAAnalyzer(attack_type="template")

        with pytest.raises(ValueError, match="Template attack requires"):
            analyzer.perform_attack(traces, target_byte=0)

    def test_perform_attack_no_traces(self) -> None:
        """Test perform_attack with no traces."""
        analyzer = DPAAnalyzer(attack_type="cpa")

        with pytest.raises(ValueError, match="No traces provided"):
            analyzer.perform_attack([], target_byte=0)

    def test_perform_attack_invalid_target_byte(self) -> None:
        """Test perform_attack with invalid target byte."""
        traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=np.random.randn(100),
                plaintext=bytes(range(16)),
            )
        ]

        analyzer = DPAAnalyzer(attack_type="cpa")

        with pytest.raises(ValueError, match="Invalid target_byte"):
            analyzer.perform_attack(traces, target_byte=16)

        with pytest.raises(ValueError, match="Invalid target_byte"):
            analyzer.perform_attack(traces, target_byte=-1)

    def test_perform_attack_unsupported_algorithm(self) -> None:
        """Test perform_attack with unsupported algorithm."""
        traces = [
            PowerTrace(
                timestamp=np.arange(100),
                power=np.random.randn(100),
                plaintext=bytes(range(16)),
            )
        ]

        analyzer = DPAAnalyzer(attack_type="cpa")

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            analyzer.perform_attack(traces, target_byte=0, algorithm="des")


class TestVisualization:
    """Tests for visualization functions."""

    def test_visualize_attack_success(self, tmp_path: Path) -> None:
        """Test attack visualization."""
        # Create dummy result with correlation traces
        correlation_traces = np.random.rand(256, 500)
        correlation_traces[42, :] = 0.9  # Make key 0x42 stand out

        result = DPAResult(
            recovered_key=bytes([0x42]),
            key_ranks=np.max(correlation_traces, axis=1),
            correlation_traces=correlation_traces,
            confidence=0.85,
            successful=True,
        )

        output_path = tmp_path / "attack_plot.png"

        analyzer = DPAAnalyzer(attack_type="cpa")
        analyzer.visualize_attack(result, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_visualize_attack_no_correlation_traces(self, tmp_path: Path) -> None:
        """Test visualization without correlation traces."""
        result = DPAResult(
            recovered_key=bytes([0x42]),
            key_ranks=np.random.rand(256),
            correlation_traces=None,
            confidence=0.5,
            successful=False,
        )

        output_path = tmp_path / "plot.png"

        analyzer = DPAAnalyzer(attack_type="cpa")

        with pytest.raises(ValueError, match="Visualization requires correlation_traces"):
            analyzer.visualize_attack(result, output_path)


class TestExport:
    """Tests for export functionality."""

    def test_export_results_cpa(self, tmp_path: Path) -> None:
        """Test exporting CPA results."""
        correlation_traces = np.random.rand(256, 500)
        result = DPAResult(
            recovered_key=bytes([0x2A]),
            key_ranks=np.max(correlation_traces, axis=1),
            correlation_traces=correlation_traces,
            confidence=0.92,
            successful=True,
        )

        output_path = tmp_path / "results.json"

        analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")
        analyzer.export_results(result, output_path)

        assert output_path.exists()

        # Verify JSON content
        with open(output_path) as f:
            data = json.load(f)

        assert data["recovered_key"] == "2a"
        assert data["confidence"] == 0.92
        assert data["successful"] is True
        assert data["attack_type"] == "cpa"
        assert data["leakage_model"] == "hamming_weight"
        assert len(data["key_ranks"]) == 256
        assert "max_correlations" in data

    def test_export_results_dpa(self, tmp_path: Path) -> None:
        """Test exporting DPA results."""
        result = DPAResult(
            recovered_key=bytes([0x7F]),
            key_ranks=np.random.rand(256),
            correlation_traces=None,
            confidence=0.75,
            successful=True,
        )

        output_path = tmp_path / "dpa_results.json"

        analyzer = DPAAnalyzer(attack_type="dpa")
        analyzer.export_results(result, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["recovered_key"] == "7f"
        assert data["attack_type"] == "dpa"
        assert "max_correlations" not in data  # No correlation traces
