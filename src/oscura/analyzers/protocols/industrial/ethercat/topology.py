"""EtherCAT topology discovery utilities.

This module provides utilities for discovering and analyzing EtherCAT network
topology, including ring/line detection and slave ordering.

Example:
    >>> from oscura.analyzers.protocols.industrial.ethercat.topology import TopologyAnalyzer
    >>> from oscura.analyzers.protocols.industrial.ethercat.analyzer import EtherCATAnalyzer
    >>> analyzer = EtherCATAnalyzer()
    >>> # Parse frames...
    >>> topology = TopologyAnalyzer(analyzer)
    >>> network_type = topology.detect_network_type()
    >>> print(f"Network: {network_type}")

References:
    ETG.1000 Section 5: Topology and Addressing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.analyzers.protocols.industrial.ethercat.analyzer import (
        EtherCATAnalyzer,
    )


@dataclass
class TopologyInfo:
    """EtherCAT network topology information.

    Attributes:
        network_type: Network topology type ("line", "ring", "star", "unknown").
        slave_count: Number of discovered slaves.
        slave_addresses: List of slave addresses in order.
        open_ports: List of detected open ports (for line topology).
    """

    network_type: str
    slave_count: int
    slave_addresses: list[int]
    open_ports: list[int]


class TopologyAnalyzer:
    """EtherCAT topology analyzer.

    Analyzes EtherCAT frames to determine network topology and slave ordering.

    Attributes:
        analyzer: EtherCAT analyzer instance.

    Example:
        >>> from oscura.analyzers.protocols.industrial.ethercat.analyzer import EtherCATAnalyzer
        >>> analyzer = EtherCATAnalyzer()
        >>> # Parse some frames...
        >>> topology = TopologyAnalyzer(analyzer)
        >>> info = topology.analyze()
        >>> print(f"Topology: {info.network_type}, Slaves: {info.slave_count}")
    """

    def __init__(self, analyzer: EtherCATAnalyzer) -> None:
        """Initialize topology analyzer.

        Args:
            analyzer: EtherCAT analyzer with parsed frames.
        """
        self.analyzer = analyzer

    def detect_network_type(self) -> str:
        """Detect network topology type.

        Analyzes port descriptor registers and working counters to determine
        whether the network is configured as a line or ring topology.

        Returns:
            Network type: "line", "ring", "star", or "unknown".

        Example:
            >>> topology = TopologyAnalyzer(analyzer)
            >>> network_type = topology.detect_network_type()
            >>> assert network_type in ["line", "ring", "star", "unknown"]
        """
        # Analyze frames for topology indicators
        open_ports = self._detect_open_ports()

        if len(open_ports) == 0:
            # All ports connected - likely ring topology
            return "ring"
        elif len(open_ports) == 2:
            # Two open ports - line topology (first and last slave)
            return "line"
        elif len(open_ports) > 2:
            # Multiple open ports - star or complex topology
            return "star"
        else:
            return "unknown"

    def _detect_open_ports(self) -> list[int]:
        """Detect open (unconnected) ports in the network.

        Returns:
            List of slave addresses with open ports.
        """
        open_ports: list[int] = []

        # Port descriptor register is at address 0x0007
        # Open ports show as not connected in the port descriptor
        for frame in self.analyzer.frames:
            for datagram in frame.datagrams:
                if datagram.ado == 0x0007:  # Port descriptor register
                    if len(datagram.data) >= 1:
                        port_desc = datagram.data[0]
                        # Bits indicate port types: 0=not implemented, 1=not configured,
                        # 2=EBUS, 3=MII
                        # Open ports show as 0 or 1
                        if port_desc & 0x03 in {0, 1}:  # Port 0 or 1 open
                            open_ports.append(datagram.adp)

        return open_ports

    def analyze(self) -> TopologyInfo:
        """Analyze complete topology information.

        Returns:
            Complete topology information.

        Example:
            >>> topology = TopologyAnalyzer(analyzer)
            >>> info = topology.analyze()
            >>> print(f"Found {info.slave_count} slaves in {info.network_type} topology")
        """
        network_type = self.detect_network_type()
        slave_addresses = self.analyzer.discover_topology()
        open_ports = self._detect_open_ports()

        return TopologyInfo(
            network_type=network_type,
            slave_count=len(slave_addresses),
            slave_addresses=slave_addresses,
            open_ports=open_ports,
        )

    def get_slave_order(self) -> list[int]:
        """Get slaves in physical connection order.

        For line topology, returns slaves ordered from master to end of chain.
        For ring topology, returns slaves in ring order.

        Returns:
            Ordered list of slave addresses.

        Example:
            >>> topology = TopologyAnalyzer(analyzer)
            >>> order = topology.get_slave_order()
            >>> print(f"Slave chain: {order}")
        """
        # Use auto-increment addressing order as physical order
        # Slaves are enumerated in the order they appear in the segment
        slave_addresses = self.analyzer.discover_topology()
        return slave_addresses


__all__ = ["TopologyAnalyzer", "TopologyInfo"]
