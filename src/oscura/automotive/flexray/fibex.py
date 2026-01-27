"""FIBEX (FlexRay Interface Bus Exchange) format support.

This module implements FIBEX XML import and export for FlexRay network
configuration, frame definitions, and signal definitions.

References:
    ASAM FIBEX 4.0.0 Specification
    FlexRay Communications System Protocol Specification Version 3.0.1

Example:
    >>> from oscura.automotive.flexray import FIBEXExporter, FlexRayAnalyzer
    >>> analyzer = FlexRayAnalyzer()
    >>> # ... parse frames and add signals ...
    >>> exporter = FIBEXExporter(analyzer)
    >>> exporter.export(Path("network.xml"))
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.automotive.flexray.analyzer import FlexRayAnalyzer, FlexRaySignal


class FIBEXExporter:
    """FIBEX XML exporter for FlexRay networks.

    Exports FlexRay network configuration as FIBEX 4.0 XML format.

    Attributes:
        analyzer: FlexRay analyzer with frames and signals.

    Example:
        >>> exporter = FIBEXExporter(analyzer)
        >>> exporter.export(Path("flexray_network.xml"))
    """

    FIBEX_NAMESPACE = "http://www.asam.net/xml/fbx"
    FIBEX_VERSION = "4.0.0"

    def __init__(self, analyzer: FlexRayAnalyzer) -> None:
        """Initialize FIBEX exporter.

        Args:
            analyzer: FlexRay analyzer containing parsed frames and signals.
        """
        self.analyzer = analyzer

    def export(self, output_path: Path) -> None:
        """Export FIBEX XML file.

        Creates a FIBEX XML file containing:
        - Project information
        - Cluster configuration
        - Frame definitions
        - Signal definitions
        - Coding information

        Args:
            output_path: Output file path for FIBEX XML.

        Example:
            >>> exporter.export(Path("output/flexray_network.xml"))
        """
        # Create root element
        root = ET.Element("FIBEX")
        root.set("xmlns", self.FIBEX_NAMESPACE)
        root.set("VERSION", self.FIBEX_VERSION)

        # Add project
        project = ET.SubElement(root, "PROJECT")
        project.set("ID", "FlexRay_Network")

        project_name = ET.SubElement(project, "SHORT-NAME")
        project_name.text = "FlexRay Network"

        # Add cluster
        clusters = ET.SubElement(project, "CLUSTERS")
        cluster = ET.SubElement(clusters, "CLUSTER")
        cluster.set("ID", "FlexRay_Cluster_1")

        cluster_name = ET.SubElement(cluster, "SHORT-NAME")
        cluster_name.text = "FlexRay_Cluster"

        # Add cluster parameters
        self._add_cluster_parameters(cluster)

        # Add channels
        self._add_channels(cluster)

        # Add frames
        self._add_frames(cluster)

        # Add signals
        self._add_signals(project)

        # Write XML with pretty formatting
        self._indent(root)
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def _add_cluster_parameters(self, cluster: ET.Element) -> None:
        """Add cluster parameters to FIBEX.

        Args:
            cluster: Cluster XML element.
        """
        params = ET.SubElement(cluster, "CLUSTER-PARAMS")

        # Get configuration
        config = self.analyzer.cluster_config

        # Speed
        speed = ET.SubElement(params, "SPEED")
        speed.text = "10000"  # 10 Mbps

        # Static slots
        static_slots = ET.SubElement(params, "NUMBER-OF-STATIC-SLOTS")
        static_slots.text = str(config.get("static_slot_count", 100))

        # Dynamic slots
        dynamic_slots = ET.SubElement(params, "NUMBER-OF-DYNAMIC-SLOTS")
        dynamic_slots.text = str(config.get("dynamic_slot_count", 50))

        # Cycle length
        cycle_length = ET.SubElement(params, "CYCLE-LENGTH-IN-MACROTICKS")
        cycle_length.text = str(config.get("cycle_length", 5000))

    def _add_channels(self, cluster: ET.Element) -> None:
        """Add channel definitions to FIBEX.

        Args:
            cluster: Cluster XML element.
        """
        channels = ET.SubElement(cluster, "CHANNELS")

        # Get unique channels from frames
        unique_channels = {frame.channel for frame in self.analyzer.frames}

        for channel_id in sorted(unique_channels):
            channel = ET.SubElement(channels, "CHANNEL")
            channel.set("ID", f"Channel_{channel_id}")

            channel_name = ET.SubElement(channel, "SHORT-NAME")
            channel_name.text = f"Channel_{channel_id}"

    def _add_frames(self, cluster: ET.Element) -> None:
        """Add frame definitions to FIBEX.

        Args:
            cluster: Cluster XML element.
        """
        frames_elem = ET.SubElement(cluster, "FRAMES")

        # Get unique frame IDs
        unique_frame_ids = sorted({frame.header.frame_id for frame in self.analyzer.frames})

        for frame_id in unique_frame_ids:
            # Get representative frame for this ID
            frame_example = next(f for f in self.analyzer.frames if f.header.frame_id == frame_id)

            frame_elem = ET.SubElement(frames_elem, "FRAME")
            frame_elem.set("ID", f"Frame_{frame_id}")

            # Frame name
            frame_name = ET.SubElement(frame_elem, "SHORT-NAME")
            frame_name.text = f"Frame_{frame_id}"

            # Slot ID
            slot_id_elem = ET.SubElement(frame_elem, "SLOT-ID")
            slot_id_elem.text = str(frame_id)

            # Frame length
            frame_length = ET.SubElement(frame_elem, "FRAME-LENGTH")
            frame_length.text = str(frame_example.header.payload_length)

            # Segment type
            segment_type = ET.SubElement(frame_elem, "SEGMENT-TYPE")
            segment_type.text = frame_example.segment_type.upper()

            # Add signals in this frame
            frame_signals = [s for s in self.analyzer.signals if s.frame_id == frame_id]
            if frame_signals:
                signals_elem = ET.SubElement(frame_elem, "SIGNALS")
                for signal in frame_signals:
                    signal_ref = ET.SubElement(signals_elem, "SIGNAL-REF")
                    signal_ref.set("ID-REF", signal.name)

    def _add_signals(self, project: ET.Element) -> None:
        """Add signal definitions to FIBEX.

        Args:
            project: Project XML element.
        """
        if not self.analyzer.signals:
            return

        signals_elem = ET.SubElement(project, "SIGNALS")

        for signal in self.analyzer.signals:
            signal_elem = ET.SubElement(signals_elem, "SIGNAL")
            signal_elem.set("ID", signal.name)

            # Signal name
            signal_name = ET.SubElement(signal_elem, "SHORT-NAME")
            signal_name.text = signal.name

            # Bit position
            bit_position = ET.SubElement(signal_elem, "BIT-POSITION")
            bit_position.text = str(signal.start_bit)

            # Bit length
            bit_length = ET.SubElement(signal_elem, "BIT-LENGTH")
            bit_length.text = str(signal.bit_length)

            # Byte order
            byte_order = ET.SubElement(signal_elem, "BYTE-ORDER")
            byte_order.text = "BIG-ENDIAN" if signal.byte_order == "big_endian" else "LITTLE-ENDIAN"

            # Coding
            coding = ET.SubElement(signal_elem, "CODING")

            # Factor
            factor_elem = ET.SubElement(coding, "FACTOR")
            factor_elem.text = str(signal.factor)

            # Offset
            offset_elem = ET.SubElement(coding, "OFFSET")
            offset_elem.text = str(signal.offset)

            # Unit
            if signal.unit:
                unit_elem = ET.SubElement(coding, "UNIT")
                unit_elem.text = signal.unit

    def _indent(self, elem: ET.Element, level: int = 0) -> None:
        """Add pretty-printing indentation to XML tree.

        Args:
            elem: XML element to indent.
            level: Current indentation level.
        """
        indent = "  "
        i = "\n" + level * indent
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + indent
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if elem and (not elem[-1].tail or not elem[-1].tail.strip()):
                elem[-1].tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class FIBEXImporter:
    """FIBEX XML importer for FlexRay networks.

    Imports FlexRay network configuration from FIBEX 4.0 XML format.

    Example:
        >>> importer = FIBEXImporter()
        >>> cluster_config, signals = importer.load(Path("network.xml"))
        >>> analyzer = FlexRayAnalyzer(cluster_config=cluster_config)
        >>> for signal in signals:
        ...     analyzer.add_signal(signal)
    """

    def __init__(self) -> None:
        """Initialize FIBEX importer."""

    def load(self, fibex_path: Path) -> tuple[dict[str, Any], list[FlexRaySignal]]:
        """Load FIBEX XML file.

        Args:
            fibex_path: Path to FIBEX XML file.

        Returns:
            Tuple of (cluster_config, signals) where:
            - cluster_config: Dictionary with cluster parameters
            - signals: List of FlexRaySignal definitions

        Raises:
            FileNotFoundError: If FIBEX file not found.
            ValueError: If FIBEX file is invalid.

        Example:
            >>> config, signals = importer.load(Path("flexray_network.xml"))
            >>> print(f"Loaded {len(signals)} signals")
        """
        # Import here to avoid circular dependency

        if not fibex_path.exists():
            raise FileNotFoundError(f"FIBEX file not found: {fibex_path}")

        tree = ET.parse(fibex_path)
        root = tree.getroot()

        # Extract cluster configuration
        cluster_config = self._parse_cluster_config(root)

        # Extract signal definitions
        signals = self._parse_signals(root)

        return cluster_config, signals

    def _parse_cluster_config(self, root: ET.Element) -> dict[str, Any]:
        """Parse cluster configuration from FIBEX.

        Args:
            root: FIBEX root element.

        Returns:
            Cluster configuration dictionary.
        """
        config: dict[str, Any] = {}

        # Find cluster parameters
        cluster_params = root.find(".//{*}CLUSTER-PARAMS")
        if cluster_params is not None:
            # Static slots
            static_slots = cluster_params.find("{*}NUMBER-OF-STATIC-SLOTS")
            if static_slots is not None and static_slots.text:
                config["static_slot_count"] = int(static_slots.text)

            # Dynamic slots
            dynamic_slots = cluster_params.find("{*}NUMBER-OF-DYNAMIC-SLOTS")
            if dynamic_slots is not None and dynamic_slots.text:
                config["dynamic_slot_count"] = int(dynamic_slots.text)

            # Cycle length
            cycle_length = cluster_params.find("{*}CYCLE-LENGTH-IN-MACROTICKS")
            if cycle_length is not None and cycle_length.text:
                config["cycle_length"] = int(cycle_length.text)

        return config

    def _parse_signals(self, root: ET.Element) -> list[FlexRaySignal]:
        """Parse signal definitions from FIBEX.

        Args:
            root: FIBEX root element.

        Returns:
            List of FlexRay signal definitions.
        """
        # Import here to avoid circular dependency
        from oscura.automotive.flexray.analyzer import FlexRaySignal

        signals: list[FlexRaySignal] = []

        # Find all signal elements
        for signal_elem in root.findall(".//{*}SIGNAL"):
            signal_id = signal_elem.get("ID", "")

            # Signal name
            name_elem = signal_elem.find("{*}SHORT-NAME")
            name = name_elem.text if name_elem is not None and name_elem.text else signal_id

            # Bit position
            bit_pos_elem = signal_elem.find("{*}BIT-POSITION")
            start_bit = (
                int(bit_pos_elem.text) if bit_pos_elem is not None and bit_pos_elem.text else 0
            )

            # Bit length
            bit_len_elem = signal_elem.find("{*}BIT-LENGTH")
            bit_length = (
                int(bit_len_elem.text) if bit_len_elem is not None and bit_len_elem.text else 8
            )

            # Byte order
            byte_order_elem = signal_elem.find("{*}BYTE-ORDER")
            byte_order = "big_endian"
            if byte_order_elem is not None and byte_order_elem.text:
                byte_order = (
                    "big_endian" if byte_order_elem.text == "BIG-ENDIAN" else "little_endian"
                )

            # Coding
            coding_elem = signal_elem.find("{*}CODING")
            factor = 1.0
            offset = 0.0
            unit = ""

            if coding_elem is not None:
                factor_elem = coding_elem.find("{*}FACTOR")
                if factor_elem is not None and factor_elem.text:
                    factor = float(factor_elem.text)

                offset_elem = coding_elem.find("{*}OFFSET")
                if offset_elem is not None and offset_elem.text:
                    offset = float(offset_elem.text)

                unit_elem = coding_elem.find("{*}UNIT")
                if unit_elem is not None and unit_elem.text:
                    unit = unit_elem.text

            # Find frame ID (need to search for frame containing this signal)
            frame_id = self._find_frame_id_for_signal(root, signal_id)

            signal = FlexRaySignal(
                name=name,
                frame_id=frame_id,
                start_bit=start_bit,
                bit_length=bit_length,
                byte_order=byte_order,
                factor=factor,
                offset=offset,
                unit=unit,
            )

            signals.append(signal)

        return signals

    def _find_frame_id_for_signal(self, root: ET.Element, signal_id: str) -> int:
        """Find frame ID containing a signal.

        Args:
            root: FIBEX root element.
            signal_id: Signal ID to find.

        Returns:
            Frame ID (slot ID) containing the signal, or 0 if not found.
        """
        # Find frame containing this signal reference
        for frame_elem in root.findall(".//{*}FRAME"):
            signal_refs = frame_elem.findall(".//{*}SIGNAL-REF")
            for sig_ref in signal_refs:
                if sig_ref.get("ID-REF") == signal_id:
                    # Found frame, get slot ID
                    slot_id_elem = frame_elem.find("{*}SLOT-ID")
                    if slot_id_elem is not None and slot_id_elem.text:
                        return int(slot_id_elem.text)

        return 0  # Default if not found


__all__ = [
    "FIBEXExporter",
    "FIBEXImporter",
]
