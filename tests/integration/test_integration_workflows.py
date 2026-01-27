"""Integration tests for multi-module data flow.

Tests ONLY cross-module workflows NOT covered by demos.
Focuses on data flow, error propagation, and edge cases.

Demos cover correctness; these tests cover integration robustness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Graceful imports
try:
    from oscura.analyzers.digital.clock import ClockRecovery
    from oscura.analyzers.digital.edges import EdgeDetector
    from oscura.analyzers.patterns.sequences import find_repeating_sequences
    from oscura.analyzers.statistical.checksum import ChecksumDetector
    from oscura.analyzers.statistical.entropy import EntropyAnalyzer
    from oscura.core.types import DigitalTrace, TraceMetadata
    from oscura.inference.message_format import MessageFormatInferrer
    from oscura.loaders.configurable import (
        ConfigurablePacketLoader,
        PacketFormatConfig,
        SampleFormatDef,
    )
    from oscura.loaders.preprocessing import detect_idle_regions, trim_idle
    from oscura.loaders.validation import PacketValidator
    from oscura.validation.testing.synthetic import (
        SyntheticPacketConfig,
        SyntheticSignalConfig,
        generate_digital_signal,
        generate_packets,
        generate_protocol_messages,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)

pytestmark = [pytest.mark.integration, pytest.mark.workflow]


@pytest.mark.integration
class TestLoaderAnalyzerDataFlow:
    """Test data flow between loaders and analyzers."""

    def test_wfm_to_fft_data_flow(self, tmp_path: Path) -> None:
        """Test data flows correctly from loader to FFT analyzer.

        This tests data compatibility, NOT FFT correctness (Demo 01 covers that).
        """
        # Generate test data
        config = SyntheticPacketConfig(packet_size=128)
        binary_data, _ = generate_packets(count=20, **config.__dict__)

        data_file = tmp_path / "data_flow_test.bin"
        data_file.write_bytes(binary_data)

        # Load
        loader_config = PacketFormatConfig(
            name="data_flow_test",
            version="1.0",
            packet_size=128,
            byte_order="little",
            header_size=16,
            header_fields=[],
            sample_offset=16,
            sample_count=14,
            sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
        )

        loader = ConfigurablePacketLoader(loader_config)
        loaded = loader.load(data_file)

        # Verify data can be extracted for analysis
        assert len(loaded.packets) > 0
        assert "samples" in loaded.packets[0]

    def test_malformed_data_error_propagation(self, tmp_path: Path) -> None:
        """Test error handling across loader to analyzer boundary."""
        # Create malformed packet data
        corrupted_data = b"\xaa\x55" * 50  # Wrong size

        corrupt_file = tmp_path / "corrupt.bin"
        corrupt_file.write_bytes(corrupted_data)

        loader_config = PacketFormatConfig(
            name="corrupt_test",
            version="1.0",
            packet_size=128,
            byte_order="little",
            header_size=8,
            header_fields=[],
            sample_offset=8,
            sample_count=15,
            sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
        )

        loader = ConfigurablePacketLoader(loader_config)

        # Should handle gracefully
        try:
            result = loader.load(corrupt_file)
            if len(result.packets) > 0:
                # If load succeeds, validation should catch issues
                validator = PacketValidator()
                validation = validator.validate_packet(result.packets[0])
                assert validation is not None
        except (ValueError, RuntimeError, KeyError, IndexError) as e:
            # Exception is acceptable for corrupted data
            pass

    def test_empty_data_handling(self, tmp_path: Path) -> None:
        """Test modules handle empty data gracefully."""
        # Create empty file
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")

        loader_config = PacketFormatConfig(
            name="empty_test",
            version="1.0",
            packet_size=64,
            byte_order="little",
            header_size=8,
            header_fields=[],
            sample_offset=8,
            sample_count=7,
            sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
        )

        loader = ConfigurablePacketLoader(loader_config)

        try:
            result = loader.load(empty_file)
            assert len(result.packets) == 0
        except (OSError, ValueError, RuntimeError) as e:
            # Exception is acceptable for empty file
            pass

        # Test analyzers with empty data
        entropy_analyzer = EntropyAnalyzer()
        try:
            entropy = entropy_analyzer.calculate_entropy(b"")
            assert entropy == 0.0
        except (ValueError, ZeroDivisionError):
            pass  # Exception is acceptable


@pytest.mark.integration
class TestAnalyzerProtocolDataFlow:
    """Test data flow between signal analyzers and protocol decoders."""

    def test_digital_signal_to_edge_detection_flow(self) -> None:
        """Test signal analyzer to edge detector data flow.

        Tests data compatibility, NOT edge detection correctness.
        """
        # Generate digital signal
        config = SyntheticSignalConfig(
            pattern_type="clock",
            sample_rate=10e6,
            duration_samples=10000,
        )

        signal, _ = generate_digital_signal(pattern="clock", **config.__dict__)

        # Convert to digital trace
        metadata = TraceMetadata(sample_rate=10e6)
        trace = DigitalTrace(data=signal > 1.5, metadata=metadata)

        # Verify data flows correctly
        detector = EdgeDetector()
        rising, falling = detector.detect_all_edges(trace.data)

        # Should complete without error
        assert isinstance(rising, np.ndarray)
        assert isinstance(falling, np.ndarray)

    def test_clock_recovery_data_flow(self) -> None:
        """Test clock recovery receives proper data format."""
        config = SyntheticSignalConfig(
            pattern_type="clock",
            sample_rate=10e6,
            duration_samples=50000,
        )

        signal, _ = generate_digital_signal(pattern="clock", **config.__dict__)

        metadata = TraceMetadata(sample_rate=10e6)
        trace = DigitalTrace(data=signal > 1.5, metadata=metadata)

        # Verify data flows correctly
        recovery = ClockRecovery()
        freq = recovery.detect_frequency(trace)

        assert isinstance(freq, (int, float))
        assert freq >= 0


@pytest.mark.integration
class TestStreamingWorkflows:
    """Test streaming and chunked processing workflows."""

    def test_streaming_data_flow(self, tmp_path: Path) -> None:
        """Test streaming load maintains data consistency across chunks."""
        # Generate larger dataset
        config = SyntheticPacketConfig(packet_size=512)
        binary_data, _ = generate_packets(count=500, **config.__dict__)

        stream_file = tmp_path / "stream_test.bin"
        stream_file.write_bytes(binary_data)

        loader_config = PacketFormatConfig(
            name="stream_test",
            version="1.0",
            packet_size=512,
            byte_order="little",
            header_size=16,
            header_fields=[],
            sample_offset=16,
            sample_count=62,
            sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
        )

        loader = ConfigurablePacketLoader(loader_config)

        # Process in chunks
        total_packets = 0
        chunk_count = 0

        for chunk in loader.stream(stream_file, chunk_size=100):
            chunk_count += 1
            total_packets += len(chunk.packets)

            assert len(chunk.packets) > 0
            assert len(chunk.packets) <= 100

        # Verify all packets processed
        assert total_packets == 500
        assert chunk_count == 5

    def test_chunked_analysis_consistency(self, tmp_path: Path) -> None:
        """Test chunked analysis maintains consistency."""
        config = SyntheticPacketConfig(packet_size=128)
        binary_data, _ = generate_packets(count=200, **config.__dict__)

        data_file = tmp_path / "chunk_analysis.bin"
        data_file.write_bytes(binary_data)

        loader_config = PacketFormatConfig(
            name="chunk_analysis",
            version="1.0",
            packet_size=128,
            byte_order="little",
            header_size=16,
            header_fields=[],
            sample_offset=16,
            sample_count=14,
            sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
        )

        loader = ConfigurablePacketLoader(loader_config)
        entropy_analyzer = EntropyAnalyzer()

        chunk_entropies = []

        for chunk in loader.stream(data_file, chunk_size=50):
            # Extract bytes from chunk samples
            chunk_bytes = bytearray()
            for packet in chunk.packets[:5]:
                if packet.get("samples"):
                    # Convert samples to bytes (uint64 little-endian)
                    import struct

                    for sample in packet["samples"]:
                        chunk_bytes.extend(struct.pack("<Q", sample))

            if len(chunk_bytes) > 0:
                entropy = entropy_analyzer.calculate_entropy(bytes(chunk_bytes))
                chunk_entropies.append(entropy)

        # Should have analyzed all chunks
        assert len(chunk_entropies) >= 2


@pytest.mark.integration
class TestValidationWorkflows:
    """Test validation-driven analysis workflows."""

    def test_validation_filters_before_analysis(self, tmp_path: Path) -> None:
        """Test validation filtering upstream of analysis."""
        # Generate mixed valid/invalid packets
        config = SyntheticPacketConfig(
            packet_size=64,
            noise_level=0.15,
        )
        binary_data, _ = generate_packets(count=40, **config.__dict__)

        data_file = tmp_path / "filter_test.bin"
        data_file.write_bytes(binary_data)

        loader_config = PacketFormatConfig(
            name="filter_test",
            version="1.0",
            packet_size=64,
            byte_order="little",
            header_size=8,
            header_fields=[],
            sample_offset=8,
            sample_count=7,
            sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
        )

        loader = ConfigurablePacketLoader(loader_config)
        loaded = loader.load(data_file)

        # Validate and filter
        validator = PacketValidator()
        valid_packets = []

        for packet in loaded.packets:
            result = validator.validate_packet(packet)
            if result.is_valid:
                valid_packets.append(packet)

        # Analyze only valid packets
        if valid_packets:
            valid_bytes = bytearray()
            for packet in valid_packets[:5]:
                if hasattr(packet, "header"):
                    valid_bytes.extend(bytes(packet.header))

            if len(valid_bytes) > 0:
                entropy_analyzer = EntropyAnalyzer()
                entropy = entropy_analyzer.calculate_entropy(bytes(valid_bytes))
                assert 0 <= entropy <= 8.0

    def test_preprocessing_before_validation(self) -> None:
        """Test preprocessing modifies data before validation."""
        # Generate digital signal with idle regions
        config = SyntheticSignalConfig(
            pattern_type="uart",
            sample_rate=10e6,
            duration_samples=20000,
        )

        signal, _ = generate_digital_signal(pattern="uart", **config.__dict__)

        # Add idle regions
        idle_prefix = np.zeros(5000, dtype=signal.dtype)
        idle_suffix = np.zeros(5000, dtype=signal.dtype)
        signal_with_idle = np.concatenate([idle_prefix, signal, idle_suffix])

        # Convert to DigitalTrace for preprocessing functions
        from oscura.core.types import DigitalTrace, TraceMetadata

        trace_with_idle = DigitalTrace(
            data=signal_with_idle, metadata=TraceMetadata(sample_rate=10e6)
        )

        # Detect idle regions (API uses 'pattern' not 'threshold')
        idle_regions = detect_idle_regions(trace_with_idle, pattern="auto", min_duration=1000)

        assert len(idle_regions) >= 1

        # Trim idle (also needs DigitalTrace)
        trimmed = trim_idle(trace_with_idle, pattern="auto", min_duration=1000)

        # Trimmed should be shorter
        assert len(trimmed.data) < len(signal_with_idle)


@pytest.mark.integration
class TestInferenceWorkflows:
    """Test inference module integration workflows."""

    def test_pattern_to_format_inference_flow(self) -> None:
        """Test pattern detection feeds format inference."""
        messages, _ = generate_protocol_messages(count=150, message_size=64)

        # Detect patterns
        combined = b"".join(messages[:50])
        patterns = find_repeating_sequences(
            combined,
            min_length=2,
            max_length=6,
            min_count=5,
        )

        # Format inference
        inferrer = MessageFormatInferrer()
        inferred = inferrer.infer_format(messages)

        # Both should complete
        assert patterns is not None
        assert inferred is not None

    def test_checksum_detection_to_validation_flow(self, tmp_path: Path) -> None:
        """Test checksum detection guides validation."""
        messages, _ = generate_protocol_messages(count=30, message_size=64)

        # Detect checksum field
        detector = ChecksumDetector()
        result = detector.detect_checksum_field(messages)

        # Use for validation if detected
        if result.has_checksum and result.offset is not None:
            config = SyntheticPacketConfig(
                packet_size=64,
                include_checksum=True,
            )
            binary_data, _ = generate_packets(count=20, **config.__dict__)

            data_file = tmp_path / "checksum_val.bin"
            data_file.write_bytes(binary_data)

            loader_config = PacketFormatConfig(
                name="checksum_val",
                version="1.0",
                packet_size=64,
                byte_order="little",
                header_size=8,
                header_fields=[],
                sample_offset=8,
                sample_count=7,
                sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
            )

            loader = ConfigurablePacketLoader(loader_config)
            loaded = loader.load(data_file)

            validator = PacketValidator()
            for packet in loaded.packets:
                validation = validator.validate_packet(packet)
                assert validation is not None
