"""S-Parameter handling and analysis.

This module provides S-parameter calculations including return loss and
insertion loss for signal integrity analysis.

For loading Touchstone files, use:
    >>> from oscura.loaders import load_touchstone

Example:
    >>> from oscura.loaders import load_touchstone
    >>> from oscura.analyzers.signal_integrity.sparams import return_loss
    >>> s_params = load_touchstone("cable.s2p")
    >>> rl = return_loss(s_params, frequency=1e9)

References:
    Touchstone 2.0 File Format Specification
    IEEE 370-2020: Standard for Electrical Characterization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SParameterData:
    """S-parameter data from Touchstone file.

    Attributes:
        frequencies: Frequency points in Hz.
        s_matrix: Complex S-parameter matrix (n_freq x n_ports x n_ports).
        n_ports: Number of ports.
        z0: Reference impedance in Ohms.
        format: Original format ("db", "ma", "ri").
        source_file: Path to source file.
        comments: Comments from file header.
    """

    frequencies: NDArray[np.float64]
    s_matrix: NDArray[np.complex128]
    n_ports: int
    z0: float = 50.0
    format: str = "ri"
    source_file: str | None = None
    comments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate S-parameter data."""
        if len(self.frequencies) == 0:
            raise ValueError("frequencies cannot be empty")

        expected_shape = (len(self.frequencies), self.n_ports, self.n_ports)
        if self.s_matrix.shape != expected_shape:
            raise ValueError(
                f"s_matrix shape {self.s_matrix.shape} does not match expected {expected_shape}"
            )

    def get_s(
        self,
        i: int,
        j: int,
        frequency: float | None = None,
    ) -> complex | NDArray[np.complex128]:
        """Get S-parameter Sij.

        Args:
            i: Output port (1-indexed).
            j: Input port (1-indexed).
            frequency: Frequency for interpolation (None = all).

        Returns:
            Complex S-parameter value(s).
        """
        # Convert to 0-indexed
        i_idx = i - 1
        j_idx = j - 1

        if frequency is None:
            return self.s_matrix[:, i_idx, j_idx]

        # Interpolate
        return np.interp(
            frequency,
            self.frequencies,
            self.s_matrix[:, i_idx, j_idx],
        )


def return_loss(
    s_params: SParameterData,
    frequency: float | None = None,
    *,
    port: int = 1,
) -> float | NDArray[np.float64]:
    """Calculate return loss from S-parameters.

    Return loss = -20 * log10(|S11|)

    Args:
        s_params: S-parameter data.
        frequency: Frequency in Hz (None = all frequencies).
        port: Port number (1-indexed).

    Returns:
        Return loss in dB.

    Example:
        >>> rl = return_loss(s_params, frequency=1e9)
        >>> print(f"Return loss: {rl:.1f} dB")

    References:
        IEEE 370-2020 Section 5.2
    """
    s11 = s_params.get_s(port, port, frequency)
    magnitude = np.abs(s11)

    # Avoid log(0)
    magnitude = np.maximum(magnitude, 1e-10)

    rl = -20 * np.log10(magnitude)

    if isinstance(rl, np.ndarray):
        return rl
    return float(rl)


def insertion_loss(
    s_params: SParameterData,
    frequency: float | None = None,
    *,
    input_port: int = 1,
    output_port: int = 2,
) -> float | NDArray[np.float64]:
    """Calculate insertion loss from S-parameters.

    Insertion loss = -20 * log10(|S21|)

    Args:
        s_params: S-parameter data.
        frequency: Frequency in Hz (None = all frequencies).
        input_port: Input port number (1-indexed).
        output_port: Output port number (1-indexed).

    Returns:
        Insertion loss in dB.

    Example:
        >>> il = insertion_loss(s_params, frequency=1e9)
        >>> print(f"Insertion loss: {il:.2f} dB")

    References:
        IEEE 370-2020 Section 5.3
    """
    s21 = s_params.get_s(output_port, input_port, frequency)
    magnitude = np.abs(s21)

    # Avoid log(0)
    magnitude = np.maximum(magnitude, 1e-10)

    il = -20 * np.log10(magnitude)

    if isinstance(il, np.ndarray):
        return il
    return float(il)


def s_to_abcd(
    s_params: SParameterData,
    frequency_idx: int | None = None,
) -> NDArray[np.complex128]:
    """Convert S-parameters to ABCD (chain) parameters.

    Used for cascading networks.

    Args:
        s_params: S-parameter data (2-port only).
        frequency_idx: Index for specific frequency (None = all).

    Returns:
        ABCD matrix (2x2) or array of matrices.

    Raises:
        ValueError: If s_params is not a 2-port network.

    Example:
        >>> abcd = s_to_abcd(s_params)

    References:
        Pozar, "Microwave Engineering", Chapter 4
    """
    if s_params.n_ports != 2:
        raise ValueError("ABCD conversion only supported for 2-port networks")

    z0 = s_params.z0

    if frequency_idx is not None:
        s = s_params.s_matrix[frequency_idx]
        return _s_to_abcd_single(s, z0)

    # Convert all frequencies
    n_freq = len(s_params.frequencies)
    abcd = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    for i in range(n_freq):
        abcd[i] = _s_to_abcd_single(s_params.s_matrix[i], z0)

    return abcd


def _s_to_abcd_single(s: NDArray[np.complex128], z0: float) -> NDArray[np.complex128]:
    """Convert single frequency S-matrix to ABCD."""
    s11, s12, s21, s22 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]

    denominator = 2 * s21

    if abs(denominator) < 1e-12:
        # Singular - return identity-ish
        return np.array([[1, 0], [0, 1]], dtype=np.complex128)

    A = ((1 + s11) * (1 - s22) + s12 * s21) / denominator
    B = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / denominator
    C = ((1 - s11) * (1 - s22) - s12 * s21) / (z0 * denominator)
    D = ((1 - s11) * (1 + s22) + s12 * s21) / denominator

    return np.array([[A, B], [C, D]], dtype=np.complex128)


def abcd_to_s(
    abcd: NDArray[np.complex128],
    z0: float = 50.0,
) -> NDArray[np.complex128]:
    """Convert ABCD parameters to S-parameters.

    Args:
        abcd: ABCD matrix (2x2) or array of matrices.
        z0: Reference impedance.

    Returns:
        S-parameter matrix.

    References:
        Pozar, "Microwave Engineering", Chapter 4
    """
    if abcd.ndim == 2:
        return _abcd_to_s_single(abcd, z0)

    # Handle array of matrices
    n_freq = abcd.shape[0]
    s = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    for i in range(n_freq):
        s[i] = _abcd_to_s_single(abcd[i], z0)

    return s


def _abcd_to_s_single(abcd: NDArray[np.complex128], z0: float) -> NDArray[np.complex128]:
    """Convert single ABCD matrix to S-parameters."""
    A, B, C, D = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]

    denominator = A + B / z0 + C * z0 + D

    if abs(denominator) < 1e-12:
        return np.zeros((2, 2), dtype=np.complex128)

    S11 = (A + B / z0 - C * z0 - D) / denominator
    S12 = 2 * (A * D - B * C) / denominator
    S21 = 2 / denominator
    S22 = (-A + B / z0 - C * z0 + D) / denominator

    return np.array([[S11, S12], [S21, S22]], dtype=np.complex128)


# load_touchstone has been moved to oscura.loaders module


__all__ = [
    "SParameterData",
    "abcd_to_s",
    "insertion_loss",
    "return_loss",
    "s_to_abcd",
]
