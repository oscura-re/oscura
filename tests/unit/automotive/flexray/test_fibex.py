"""Tests for FIBEX import/export functionality.

Tests cover:
- FIBEX XML export
- FIBEX XML import
- Cluster configuration
- Signal definitions
- Frame definitions
- Round-trip (export then import)
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from oscura.automotive.flexray import (
    FlexRayAnalyzer,
    FlexRaySignal,
)
from oscura.automotive.flexray.crc import calculate_frame_crc, calculate_header_crc
from oscura.automotive.flexray.fibex import FIBEXExporter, FIBEXImporter


class TestFIBEXExporter:
    """Tests for FIBEX exporter."""

    def test_export_basic_fibex(self, tmp_path: Path) -> None:
        """Test exporting basic FIBEX file."""
        analyzer = FlexRayAnalyzer()
        exporter = FIBEXExporter(analyzer)

        output_file = tmp_path / "test_network.xml"
        exporter.export(output_file)

        assert output_file.exists()

        # Parse and verify structure
        tree = ET.parse(output_file)  # noqa: S314
        root = tree.getroot()

        # Handle namespace in tag
        assert root.tag.endswith("FIBEX")
        assert root.get("VERSION") == "4.0.0"

    def test_export_with_frames(self, tmp_path: Path) -> None:
        """Test exporting FIBEX with frame definitions."""
        analyzer = FlexRayAnalyzer()

        # Add some frames
        for frame_id in [100, 200, 300]:
            header_crc = calculate_header_crc(0, 0, 0, 0, 0, frame_id, 5)
            header_int = (
                (0 << 39)
                | (0 << 38)
                | (0 << 37)
                | (0 << 36)
                | (0 << 35)
                | (frame_id << 24)
                | (5 << 17)
                | (header_crc << 6)
                | (0 << 0)
            )
            header_bytes = header_int.to_bytes(5, "big")
            payload = bytes(range(10))
            frame_crc = calculate_frame_crc(header_bytes, payload)
            crc_bytes = frame_crc.to_bytes(3, "big")
            frame_data = header_bytes + payload + crc_bytes

            analyzer.parse_frame(frame_data)

        exporter = FIBEXExporter(analyzer)
        output_file = tmp_path / "frames_network.xml"
        exporter.export(output_file)

        # Parse and verify frames
        tree = ET.parse(output_file)  # noqa: S314
        root = tree.getroot()

        frames = root.findall(".//{*}FRAME")
        assert len(frames) == 3

        # Check frame IDs
        slot_ids = []
        for frame in frames:
            slot_id_elem = frame.find("{*}SLOT-ID")
            if slot_id_elem is not None and slot_id_elem.text:
                slot_ids.append(int(slot_id_elem.text))

        assert sorted(slot_ids) == [100, 200, 300]

    def test_export_with_signals(self, tmp_path: Path) -> None:
        """Test exporting FIBEX with signal definitions."""
        analyzer = FlexRayAnalyzer()

        # Add frame
        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)
        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (100 << 24)
            | (5 << 17)
            | (header_crc << 6)
            | (0 << 0)
        )
        header_bytes = header_int.to_bytes(5, "big")
        payload = bytes(range(10))
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")
        frame_data = header_bytes + payload + crc_bytes
        analyzer.parse_frame(frame_data)

        # Add signals
        signal1 = FlexRaySignal(
            name="EngineSpeed",
            frame_id=100,
            start_bit=0,
            bit_length=16,
            factor=0.25,
            unit="rpm",
        )
        signal2 = FlexRaySignal(
            name="VehicleSpeed",
            frame_id=100,
            start_bit=16,
            bit_length=16,
            factor=0.01,
            unit="km/h",
        )
        analyzer.add_signal(signal1)
        analyzer.add_signal(signal2)

        exporter = FIBEXExporter(analyzer)
        output_file = tmp_path / "signals_network.xml"
        exporter.export(output_file)

        # Parse and verify signals
        tree = ET.parse(output_file)  # noqa: S314
        root = tree.getroot()

        signals = root.findall(".//{*}SIGNAL")
        assert len(signals) == 2

        # Check signal names
        signal_names = []
        for signal in signals:
            name_elem = signal.find("{*}SHORT-NAME")
            if name_elem is not None and name_elem.text:
                signal_names.append(name_elem.text)

        assert sorted(signal_names) == ["EngineSpeed", "VehicleSpeed"]

    def test_export_cluster_configuration(self, tmp_path: Path) -> None:
        """Test exporting cluster configuration."""
        config = {
            "static_slot_count": 150,
            "dynamic_slot_count": 75,
            "cycle_length": 6000,
        }
        analyzer = FlexRayAnalyzer(cluster_config=config)

        exporter = FIBEXExporter(analyzer)
        output_file = tmp_path / "cluster_config.xml"
        exporter.export(output_file)

        # Parse and verify cluster params
        tree = ET.parse(output_file)  # noqa: S314
        root = tree.getroot()

        cluster_params = root.find(".//{*}CLUSTER-PARAMS")
        assert cluster_params is not None

        static_slots = cluster_params.find("{*}NUMBER-OF-STATIC-SLOTS")
        assert static_slots is not None
        assert static_slots.text == "150"

        dynamic_slots = cluster_params.find("{*}NUMBER-OF-DYNAMIC-SLOTS")
        assert dynamic_slots is not None
        assert dynamic_slots.text == "75"

    def test_export_multi_channel(self, tmp_path: Path) -> None:
        """Test exporting frames from multiple channels."""
        analyzer = FlexRayAnalyzer()

        # Add frames on different channels
        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 0)
        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (100 << 24)
            | (0 << 17)
            | (header_crc << 6)
            | (0 << 0)
        )
        header_bytes = header_int.to_bytes(5, "big")
        payload = b""
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")
        frame_data = header_bytes + payload + crc_bytes

        analyzer.parse_frame(frame_data, channel="A")
        analyzer.parse_frame(frame_data, channel="B")

        exporter = FIBEXExporter(analyzer)
        output_file = tmp_path / "multi_channel.xml"
        exporter.export(output_file)

        # Parse and verify channels
        tree = ET.parse(output_file)  # noqa: S314
        root = tree.getroot()

        channels = root.findall(".//{*}CHANNEL")
        assert len(channels) == 2


class TestFIBEXImporter:
    """Tests for FIBEX importer."""

    def test_import_nonexistent_file(self) -> None:
        """Test importing nonexistent FIBEX file."""
        importer = FIBEXImporter()

        with pytest.raises(FileNotFoundError):
            importer.load(Path("/nonexistent/file.xml"))

    def test_import_basic_fibex(self, tmp_path: Path) -> None:
        """Test importing basic FIBEX file."""
        # Create minimal FIBEX
        root = ET.Element("FIBEX")
        root.set("xmlns", "http://www.asam.net/xml/fbx")
        root.set("VERSION", "4.0.0")

        project = ET.SubElement(root, "PROJECT")
        project.set("ID", "Test")

        tree = ET.ElementTree(root)
        fibex_file = tmp_path / "basic.xml"
        tree.write(fibex_file, encoding="utf-8", xml_declaration=True)

        # Import
        importer = FIBEXImporter()
        cluster_config, signals = importer.load(fibex_file)

        assert isinstance(cluster_config, dict)
        assert isinstance(signals, list)

    def test_import_cluster_configuration(self, tmp_path: Path) -> None:
        """Test importing cluster configuration."""
        # Create FIBEX with cluster params
        root = ET.Element("FIBEX")
        project = ET.SubElement(root, "PROJECT")
        clusters = ET.SubElement(project, "CLUSTERS")
        cluster = ET.SubElement(clusters, "CLUSTER")

        cluster_params = ET.SubElement(cluster, "CLUSTER-PARAMS")

        static_slots = ET.SubElement(cluster_params, "NUMBER-OF-STATIC-SLOTS")
        static_slots.text = "150"

        dynamic_slots = ET.SubElement(cluster_params, "NUMBER-OF-DYNAMIC-SLOTS")
        dynamic_slots.text = "75"

        cycle_length = ET.SubElement(cluster_params, "CYCLE-LENGTH-IN-MACROTICKS")
        cycle_length.text = "6000"

        tree = ET.ElementTree(root)
        fibex_file = tmp_path / "cluster.xml"
        tree.write(fibex_file, encoding="utf-8", xml_declaration=True)

        # Import
        importer = FIBEXImporter()
        cluster_config, _ = importer.load(fibex_file)

        assert cluster_config["static_slot_count"] == 150
        assert cluster_config["dynamic_slot_count"] == 75
        assert cluster_config["cycle_length"] == 6000

    def test_import_signals(self, tmp_path: Path) -> None:
        """Test importing signal definitions."""
        # Create FIBEX with signals
        root = ET.Element("FIBEX")
        project = ET.SubElement(root, "PROJECT")

        # Add frame
        clusters = ET.SubElement(project, "CLUSTERS")
        cluster = ET.SubElement(clusters, "CLUSTER")
        frames_elem = ET.SubElement(cluster, "FRAMES")
        frame = ET.SubElement(frames_elem, "FRAME")
        frame.set("ID", "Frame_100")

        slot_id = ET.SubElement(frame, "SLOT-ID")
        slot_id.text = "100"

        signals_in_frame = ET.SubElement(frame, "SIGNALS")
        signal_ref = ET.SubElement(signals_in_frame, "SIGNAL-REF")
        signal_ref.set("ID-REF", "EngineSpeed")

        # Add signal definition
        signals_elem = ET.SubElement(project, "SIGNALS")
        signal = ET.SubElement(signals_elem, "SIGNAL")
        signal.set("ID", "EngineSpeed")

        name = ET.SubElement(signal, "SHORT-NAME")
        name.text = "EngineSpeed"

        bit_position = ET.SubElement(signal, "BIT-POSITION")
        bit_position.text = "0"

        bit_length = ET.SubElement(signal, "BIT-LENGTH")
        bit_length.text = "16"

        byte_order = ET.SubElement(signal, "BYTE-ORDER")
        byte_order.text = "BIG-ENDIAN"

        coding = ET.SubElement(signal, "CODING")
        factor = ET.SubElement(coding, "FACTOR")
        factor.text = "0.25"

        offset = ET.SubElement(coding, "OFFSET")
        offset.text = "0"

        unit = ET.SubElement(coding, "UNIT")
        unit.text = "rpm"

        tree = ET.ElementTree(root)
        fibex_file = tmp_path / "signals.xml"
        tree.write(fibex_file, encoding="utf-8", xml_declaration=True)

        # Import
        importer = FIBEXImporter()
        _, signals = importer.load(fibex_file)

        assert len(signals) == 1
        assert signals[0].name == "EngineSpeed"
        assert signals[0].frame_id == 100
        assert signals[0].start_bit == 0
        assert signals[0].bit_length == 16
        assert signals[0].factor == 0.25
        assert signals[0].unit == "rpm"


class TestFIBEXRoundTrip:
    """Tests for FIBEX round-trip (export then import)."""

    def test_round_trip_cluster_config(self, tmp_path: Path) -> None:
        """Test round-trip of cluster configuration."""
        # Export
        config = {
            "static_slot_count": 150,
            "dynamic_slot_count": 75,
            "cycle_length": 6000,
        }
        analyzer = FlexRayAnalyzer(cluster_config=config)

        exporter = FIBEXExporter(analyzer)
        fibex_file = tmp_path / "roundtrip.xml"
        exporter.export(fibex_file)

        # Import
        importer = FIBEXImporter()
        imported_config, _ = importer.load(fibex_file)

        # Verify
        assert imported_config["static_slot_count"] == config["static_slot_count"]
        assert imported_config["dynamic_slot_count"] == config["dynamic_slot_count"]
        assert imported_config["cycle_length"] == config["cycle_length"]

    def test_round_trip_signals(self, tmp_path: Path) -> None:
        """Test round-trip of signal definitions."""
        # Create analyzer with signals
        analyzer = FlexRayAnalyzer()

        # Add frame
        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 0)
        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (100 << 24)
            | (0 << 17)
            | (header_crc << 6)
            | (0 << 0)
        )
        header_bytes = header_int.to_bytes(5, "big")
        payload = b""
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")
        frame_data = header_bytes + payload + crc_bytes
        analyzer.parse_frame(frame_data)

        # Add signals
        signal1 = FlexRaySignal(
            name="EngineSpeed",
            frame_id=100,
            start_bit=0,
            bit_length=16,
            factor=0.25,
            unit="rpm",
        )
        signal2 = FlexRaySignal(
            name="VehicleSpeed",
            frame_id=100,
            start_bit=16,
            bit_length=16,
            factor=0.01,
            unit="km/h",
        )
        analyzer.add_signal(signal1)
        analyzer.add_signal(signal2)

        # Export
        exporter = FIBEXExporter(analyzer)
        fibex_file = tmp_path / "signals_roundtrip.xml"
        exporter.export(fibex_file)

        # Import
        importer = FIBEXImporter()
        _, imported_signals = importer.load(fibex_file)

        # Verify
        assert len(imported_signals) == 2

        # Find signals by name
        engine_speed = next(s for s in imported_signals if s.name == "EngineSpeed")
        vehicle_speed = next(s for s in imported_signals if s.name == "VehicleSpeed")

        assert engine_speed.frame_id == 100
        assert engine_speed.start_bit == 0
        assert engine_speed.bit_length == 16
        assert engine_speed.factor == 0.25
        assert engine_speed.unit == "rpm"

        assert vehicle_speed.frame_id == 100
        assert vehicle_speed.start_bit == 16
        assert vehicle_speed.bit_length == 16
        assert vehicle_speed.factor == 0.01
        assert vehicle_speed.unit == "km/h"
