"""Configurable multi-bit parallel bus decoding.

This module provides configurable bus decoding for parallel digital signals,
supporting various bit orderings, active-low signaling, and clock-based
or interval-based sampling strategies.


Example:
    >>> import numpy as np
    >>> from oscura.analyzers.digital.bus import BusConfig, BusDecoder
    >>> # Define 8-bit bus configuration
    >>> config = BusConfig(name="data_bus", width=8, bit_order='lsb_first')
    >>> config.bits = [{'channel': i, 'bit': i, 'name': f'D{i}'} for i in range(8)]
    >>> # Create decoder
    >>> decoder = BusDecoder(config, sample_rate=100e6)
    >>> # Decode bus values from bit traces
    >>> bit_traces = {i: np.random.randint(0, 2, 1000, dtype=np.uint8) for i in range(8)}
    >>> transactions = decoder.decode_bus(bit_traces)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class BusConfig:
    """Configuration for parallel bus decoding.

    Attributes:
        name: Descriptive name for the bus.
        width: Number of bits in the bus.
        bit_order: Bit ordering, 'lsb_first' or 'msb_first'.
        active_low: Whether signals are active-low (inverted).
        bits: List of bit definitions with channel mapping.

    Example:
        >>> config = BusConfig(name="addr_bus", width=12, bit_order='lsb_first')
        >>> config.bits = [{'channel': i, 'bit': i} for i in range(12)]
    """

    name: str
    width: int  # Number of bits
    bit_order: Literal["lsb_first", "msb_first"] = "lsb_first"
    active_low: bool = False
    bits: list[dict[str, Any]] = field(
        default_factory=list
    )  # [{channel: 0, bit: 0, name: 'D0'}, ...]

    def __post_init__(self) -> None:
        """Validate bus configuration."""
        if self.width <= 0:
            raise ValueError(f"Bus width must be positive, got {self.width}")
        if self.bit_order not in ["lsb_first", "msb_first"]:
            raise ValueError(f"Invalid bit_order: {self.bit_order}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> BusConfig:
        """Load bus configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            BusConfig instance loaded from file.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If file does not exist.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML loading. Install with: pip install pyyaml"
            ) from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BusConfig:
        """Create bus configuration from dictionary.

        Args:
            config: Dictionary with bus configuration parameters.

        Returns:
            BusConfig instance created from dictionary.

        Example:
            >>> config_dict = {
            ...     'name': 'data_bus',
            ...     'width': 8,
            ...     'bit_order': 'lsb_first',
            ...     'bits': [{'channel': i, 'bit': i} for i in range(8)]
            ... }
            >>> config = BusConfig.from_dict(config_dict)
        """
        return cls(
            name=config.get("name", "bus"),
            width=config["width"],
            bit_order=config.get("bit_order", "lsb_first"),
            active_low=config.get("active_low", False),
            bits=config.get("bits", []),
        )


@dataclass
class ParallelBusConfig:
    """Configuration for parallel bus decoding with simplified interface.

    This is a convenience class for tests that provides a simpler interface
    than the full BusConfig.

    Attributes:
        data_width: Number of data bits in the bus.
        bit_order: Bit ordering, 'lsb_first' or 'msb_first'.
        has_clock: Whether the bus uses a clock signal.
        address_width: Optional number of address bits.
        active_low: Whether signals are active-low.

    Example:
        >>> config = ParallelBusConfig(data_width=8, bit_order='lsb_first')
    """

    data_width: int
    bit_order: Literal["lsb_first", "msb_first"] = "lsb_first"
    has_clock: bool = False
    address_width: int | None = None
    active_low: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.data_width <= 0:
            raise ValueError(f"data_width must be positive, got {self.data_width}")
        if self.address_width is not None and self.address_width <= 0:
            raise ValueError(f"address_width must be positive, got {self.address_width}")

    def to_bus_config(self, name: str = "parallel_bus") -> BusConfig:
        """Convert to BusConfig.

        Args:
            name: Name for the bus configuration.

        Returns:
            BusConfig instance.
        """
        return BusConfig(
            name=name,
            width=self.data_width,
            bit_order=self.bit_order,
            active_low=self.active_low,
            bits=[{"channel": i, "bit": i} for i in range(self.data_width)],
        )


@dataclass
class BusTransaction:
    """A decoded bus transaction.

    Attributes:
        timestamp: Time in seconds when transaction occurred.
        sample_index: Sample index in the original traces.
        value: Decoded bus value as integer.
        raw_bits: Individual bit values (after active-low inversion if applicable).
        transaction_type: Optional transaction type label.
        address: Optional address field if this is an address bus.
        data: Optional data field if this is a data bus.
    """

    timestamp: float  # Time in seconds
    sample_index: int
    value: int  # Decoded bus value
    raw_bits: list[int]  # Individual bit values
    transaction_type: str = ""  # 'read', 'write', etc.
    address: int | None = None  # If address bus present
    data: int | None = None  # If data value


class BusDecoder:
    """Decode multi-bit parallel buses from individual bit traces.

    Supports configurable bit ordering, active-low signaling, and various
    sampling strategies (clock-based or interval-based).

    Attributes:
        config: Bus configuration specifying width, ordering, etc.
        sample_rate: Sample rate of input traces in Hz.

    Example:
        >>> config = BusConfig(name="data", width=8, bit_order='lsb_first')
        >>> decoder = BusDecoder(config, sample_rate=100e6)
        >>> bit_traces = {i: trace_data for i in range(8)}
        >>> transactions = decoder.decode_bus(bit_traces)
    """

    def __init__(
        self,
        config: BusConfig | ParallelBusConfig,
        sample_rate: float = 1.0,
    ):
        """Initialize decoder with configuration.

        Args:
            config: Bus configuration (BusConfig or ParallelBusConfig).
            sample_rate: Sample rate of input traces in Hz.

        Raises:
            ValueError: If sample rate is invalid.
        """
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")

        # Handle ParallelBusConfig
        self._parallel_config: ParallelBusConfig | None
        if isinstance(config, ParallelBusConfig):
            self._parallel_config = config
            self.config = config.to_bus_config()
        else:
            self._parallel_config = None
            self.config = config

        self.sample_rate = sample_rate
        self._time_base = 1.0 / sample_rate

    def decode_bus(
        self,
        bit_traces: dict[int, NDArray[np.uint8]],  # channel_index -> trace data
        clock_trace: NDArray[np.uint8] | None = None,
        clock_edge: Literal["rising", "falling"] = "rising",
    ) -> list[BusTransaction]:
        """Decode bus values from individual bit traces.

        Args:
            bit_traces: Dictionary mapping channel index to trace data (boolean or 0/1).
            clock_trace: Optional clock signal for synchronous sampling.
            clock_edge: Which clock edge to sample on ('rising' or 'falling').

        Returns:
            List of BusTransaction objects with decoded values.

        Raises:
            ValueError: If bit traces don't match configuration.

        Example:
            >>> bit_traces = {0: np.array([0,1,1,0]), 1: np.array([1,1,0,0])}
            >>> transactions = decoder.decode_bus(bit_traces)
        """
        if not bit_traces:
            raise ValueError("bit_traces cannot be empty")

        # Use clock-based or interval-based sampling
        if clock_trace is not None:
            return self.sample_at_clock(bit_traces, clock_trace, clock_edge)
        else:
            # Sample every point (could be optimized with interval sampling)
            _trace_length = len(next(iter(bit_traces.values())))
            return self.sample_at_intervals(bit_traces, interval_samples=1)

    def decode_parallel(
        self,
        channels: list[NDArray[np.uint8]],
    ) -> list[int]:
        """Decode parallel bus values from channel list.

        Simplified interface for parallel bus decoding without clock.

        Args:
            channels: List of channel data arrays, indexed by bit position.

        Returns:
            List of decoded integer values (one per sample).

        Example:
            >>> channels = [ch0, ch1, ch2, ch3]  # 4-bit bus
            >>> values = decoder.decode_parallel(channels)
        """
        if not channels:
            return []

        trace_length = len(channels[0])
        width = len(channels)
        bit_order = self.config.bit_order

        values = []
        for sample_idx in range(trace_length):
            value = 0
            for bit_idx in range(width):
                bit_val = int(bool(channels[bit_idx][sample_idx]))
                if self.config.active_low:
                    bit_val = 1 - bit_val

                if bit_order == "lsb_first":
                    if bit_val:
                        value |= 1 << bit_idx
                else:  # msb_first
                    if bit_val:
                        value |= 1 << (width - 1 - bit_idx)

            values.append(value)

        return values

    def decode_with_clock(
        self,
        channels: list[NDArray[np.uint8]],
        clock: NDArray[np.uint8],
        edge: Literal["rising", "falling"] = "rising",
    ) -> list[int]:
        """Decode parallel bus values at clock edges.

        Args:
            channels: List of channel data arrays, indexed by bit position.
            clock: Clock signal trace (boolean or 0/1).
            edge: Which edge to sample on ('rising' or 'falling').

        Returns:
            List of decoded integer values (one per clock edge).

        Example:
            >>> values = decoder.decode_with_clock(channels, clock, 'rising')
        """
        if not channels:
            return []

        # Convert clock to boolean
        clock_bool = np.asarray(clock, dtype=bool)

        # Find edges
        if edge == "rising":
            edges = np.where(np.diff(clock_bool.astype(int)) > 0)[0] + 1
        else:
            edges = np.where(np.diff(clock_bool.astype(int)) < 0)[0] + 1

        width = len(channels)
        bit_order = self.config.bit_order

        values = []
        for edge_idx in edges:
            value = 0
            for bit_idx in range(width):
                if edge_idx < len(channels[bit_idx]):
                    bit_val = int(bool(channels[bit_idx][edge_idx]))
                else:
                    bit_val = 0

                if self.config.active_low:
                    bit_val = 1 - bit_val

                if bit_order == "lsb_first":
                    if bit_val:
                        value |= 1 << bit_idx
                else:  # msb_first
                    if bit_val:
                        value |= 1 << (width - 1 - bit_idx)

            values.append(value)

        return values

    def decode_transactions(
        self,
        address_channels: list[NDArray[np.uint8]],
        data_channels: list[NDArray[np.uint8]],
        clock: NDArray[np.uint8],
        edge: Literal["rising", "falling"] = "rising",
    ) -> list[dict[str, int]]:
        """Decode bus transactions with address and data.

        Args:
            address_channels: List of address channel data arrays.
            data_channels: List of data channel data arrays.
            clock: Clock signal trace.
            edge: Which clock edge to sample on.

        Returns:
            List of transaction dictionaries with 'address' and 'data' keys.

        Example:
            >>> transactions = decoder.decode_transactions(
            ...     address_channels=addr_ch,
            ...     data_channels=data_ch,
            ...     clock=clk
            ... )
        """
        # Convert clock to boolean
        clock_bool = np.asarray(clock, dtype=bool)

        # Find edges
        if edge == "rising":
            edges = np.where(np.diff(clock_bool.astype(int)) > 0)[0] + 1
        else:
            edges = np.where(np.diff(clock_bool.astype(int)) < 0)[0] + 1

        addr_width = len(address_channels)
        data_width = len(data_channels)
        bit_order = self.config.bit_order

        transactions = []
        for edge_idx in edges:
            # Decode address
            address = 0
            for bit_idx in range(addr_width):
                if edge_idx < len(address_channels[bit_idx]):
                    bit_val = int(bool(address_channels[bit_idx][edge_idx]))
                else:
                    bit_val = 0

                if bit_order == "lsb_first":
                    if bit_val:
                        address |= 1 << bit_idx
                else:
                    if bit_val:
                        address |= 1 << (addr_width - 1 - bit_idx)

            # Decode data
            data = 0
            for bit_idx in range(data_width):
                if edge_idx < len(data_channels[bit_idx]):
                    bit_val = int(bool(data_channels[bit_idx][edge_idx]))
                else:
                    bit_val = 0

                if bit_order == "lsb_first":
                    if bit_val:
                        data |= 1 << bit_idx
                else:
                    if bit_val:
                        data |= 1 << (data_width - 1 - bit_idx)

            transactions.append(
                {
                    "address": address,
                    "data": data,
                    "sample_index": int(edge_idx),
                }
            )

        return transactions

    def sample_at_clock(
        self,
        bit_traces: dict[int, NDArray[np.uint8]],
        clock_trace: NDArray[np.uint8],
        edge: Literal["rising", "falling"] = "rising",
    ) -> list[BusTransaction]:
        """Sample bus at clock edges.

        Args:
            bit_traces: Dictionary mapping channel index to trace data.
            clock_trace: Clock signal trace (boolean or 0/1).
            edge: Which edge to sample on ('rising' or 'falling').

        Returns:
            List of BusTransaction objects sampled at clock edges.

        Example:
            >>> clock = np.array([0,1,0,1,0,1], dtype=bool)
            >>> transactions = decoder.sample_at_clock(bit_traces, clock, 'rising')
        """
        # Convert clock to boolean
        clock_bool = np.asarray(clock_trace, dtype=bool)

        # Find edges
        if edge == "rising":
            # Rising edge: 0->1 transition
            edges = np.where(np.diff(clock_bool.astype(int)) > 0)[0] + 1
        else:
            # Falling edge: 1->0 transition
            edges = np.where(np.diff(clock_bool.astype(int)) < 0)[0] + 1

        transactions = []

        for edge_idx in edges:
            # Sample all bits at this edge
            bit_values = []
            for bit_def in self.config.bits:
                channel = bit_def.get("channel", bit_def.get("bit", 0))
                if channel in bit_traces:
                    trace = bit_traces[channel]
                    if edge_idx < len(trace):
                        bit_val = int(bool(trace[edge_idx]))
                        bit_values.append(bit_val)
                    else:
                        bit_values.append(0)
                else:
                    bit_values.append(0)

            # Apply active-low inversion if needed
            if self.config.active_low:
                bit_values = self._apply_active_low(bit_values)

            # Reconstruct bus value
            value = self._reconstruct_value(bit_values)

            # Create transaction
            transaction = BusTransaction(
                timestamp=edge_idx * self._time_base,
                sample_index=int(edge_idx),
                value=value,
                raw_bits=bit_values,
            )
            transactions.append(transaction)

        return transactions

    def sample_at_intervals(
        self, bit_traces: dict[int, NDArray[np.uint8]], interval_samples: int
    ) -> list[BusTransaction]:
        """Sample bus at regular intervals.

        Args:
            bit_traces: Dictionary mapping channel index to trace data.
            interval_samples: Number of samples between each bus sample.

        Returns:
            List of BusTransaction objects sampled at intervals.

        Raises:
            ValueError: If interval_samples is not positive.

        Example:
            >>> transactions = decoder.sample_at_intervals(bit_traces, interval_samples=10)
        """
        if interval_samples <= 0:
            raise ValueError(f"interval_samples must be positive, got {interval_samples}")

        # Determine trace length
        trace_length = len(next(iter(bit_traces.values())))

        transactions = []

        for sample_idx in range(0, trace_length, interval_samples):
            # Sample all bits at this index
            bit_values = []
            for bit_def in self.config.bits:
                channel = bit_def.get("channel", bit_def.get("bit", 0))
                if channel in bit_traces:
                    trace = bit_traces[channel]
                    if sample_idx < len(trace):
                        bit_val = int(bool(trace[sample_idx]))
                        bit_values.append(bit_val)
                    else:
                        bit_values.append(0)
                else:
                    bit_values.append(0)

            # Apply active-low inversion if needed
            if self.config.active_low:
                bit_values = self._apply_active_low(bit_values)

            # Reconstruct bus value
            value = self._reconstruct_value(bit_values)

            # Create transaction
            transaction = BusTransaction(
                timestamp=sample_idx * self._time_base,
                sample_index=sample_idx,
                value=value,
                raw_bits=bit_values,
            )
            transactions.append(transaction)

        return transactions

    def _reconstruct_value(self, bit_values: list[int]) -> int:
        """Reconstruct bus value from individual bits.

        Args:
            bit_values: List of bit values (0 or 1) in config order.

        Returns:
            Integer value reconstructed from bits.
        """
        if not bit_values:
            return 0

        value = 0

        if self.config.bit_order == "lsb_first":
            # LSB is first in list, MSB is last
            for i, bit_val in enumerate(bit_values):
                if bit_val:
                    value |= 1 << i
        else:  # msb_first
            # MSB is first in list, LSB is last
            n_bits = len(bit_values)
            for i, bit_val in enumerate(bit_values):
                if bit_val:
                    value |= 1 << (n_bits - 1 - i)

        return value

    def _apply_active_low(self, bit_values: list[int]) -> list[int]:
        """Apply active-low inversion if configured.

        Args:
            bit_values: List of bit values (0 or 1).

        Returns:
            Inverted bit values if active_low is True, otherwise unchanged.
        """
        if self.config.active_low:
            return [1 - bit for bit in bit_values]
        return bit_values


# Convenience functions


def decode_bus(
    bit_traces: dict[int, NDArray[np.uint8]],
    config: BusConfig | str | Path,
    sample_rate: float,
    clock_trace: NDArray[np.uint8] | None = None,
    clock_edge: Literal["rising", "falling"] = "rising",
) -> list[BusTransaction]:
    """Decode bus from bit traces.

    Convenience function for quick bus decoding without creating a decoder instance.

    Args:
        bit_traces: Dictionary mapping channel index to trace data.
        config: BusConfig instance or path to YAML config file.
        sample_rate: Sample rate of traces in Hz.
        clock_trace: Optional clock signal for synchronous sampling.
        clock_edge: Which clock edge to sample on.

    Returns:
        List of BusTransaction objects.

    Example:
        >>> transactions = decode_bus(bit_traces, 'bus_config.yaml', 100e6)
    """
    if isinstance(config, str | Path):
        config = BusConfig.from_yaml(config)

    decoder = BusDecoder(config, sample_rate)
    return decoder.decode_bus(bit_traces, clock_trace, clock_edge)


def sample_at_clock(
    bit_traces: dict[int, NDArray[np.uint8]],
    clock_trace: NDArray[np.uint8],
    config: BusConfig,
    sample_rate: float,
    edge: Literal["rising", "falling"] = "rising",
) -> list[BusTransaction]:
    """Sample bus at clock edges.

    Convenience function for clock-based bus sampling.

    Args:
        bit_traces: Dictionary mapping channel index to trace data.
        clock_trace: Clock signal trace.
        config: Bus configuration.
        sample_rate: Sample rate of traces in Hz.
        edge: Which clock edge to sample on.

    Returns:
        List of BusTransaction objects.

    Example:
        >>> transactions = sample_at_clock(bit_traces, clock, config, 100e6, 'rising')
    """
    decoder = BusDecoder(config, sample_rate)
    return decoder.sample_at_clock(bit_traces, clock_trace, edge)


__all__ = [
    "BusConfig",
    "BusDecoder",
    "BusTransaction",
    "ParallelBusConfig",
    "decode_bus",
    "sample_at_clock",
]
