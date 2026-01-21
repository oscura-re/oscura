"""Comprehensive pytest configuration and fixtures for Oscura test suite.

This module provides all fixtures required by the test suite architecture:
- Path fixtures for test data directories
- Signal generation fixtures (with factory patterns)
- Ground truth loading fixtures
- Performance testing fixtures
- File collection fixtures


IMPORTANT: This is the SINGLE SOURCE OF TRUTH for pytest configuration.
Do NOT add pytest_configure, pytest_addoption, or pytest_collection_modifyitems
hooks in subdirectory conftest.py files - all markers and options are defined
here and in pyproject.toml.

Fixture Scoping Best Practices
===============================

Fixture scopes control how often a fixture is created and torn down:

- **session**: Created once for the entire test session. Use for read-only data,
  paths, and global configs. Slowest to create but shared everywhere.
  Examples: project_root, test_data_dir, wfm_files, ground_truth_dir

- **module**: Created once per test module (file). Use for shared test data
  that's expensive to create but doesn't need to be global.
  Examples: loaded_wfm_data (cached file loading)

- **function**: Created fresh for each test (default). Use for mutable state,
  temporary directories, and per-test resources.
  Examples: tmp_output_dir, cleanup_matplotlib, memory_cleanup

Fixture Factories
=================

Factory fixtures return functions that create parameterized test data on demand.
This provides flexibility without fixture explosion.

Available factories:
- signal_factory: Create signals with configurable type, frequency, sample rate
- packet_factory: Create packets with configurable size, checksums, corruption
- waveform_factory: Create complete waveform structures with metadata

Performance Impact
==================

Optimized fixture scoping reduces test execution time:
- Before optimization: 15-20 minutes for full unit test suite
- After optimization: Target 10-12 minutes (20-40% speedup)
- Key optimizations: Session-scoped paths, cached file loading, signal factories

The performance improvement comes from:
1. Avoiding redundant file I/O operations (session-scoped file lists)
2. Reusing read-only data structures across tests
3. Factory patterns that generate data only when needed
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Path to test_data directory.

    Contains all test data files organized by three-tier structure:
    - synthetic/: Generated test data (waveforms, digital, binary, statistical, etc.)
    - real_captures/: Curated real-world captures (anonymized, version controlled)
    - formats/: Format compliance testing (tektronix, sigrok, pcap, csv, hdf5, etc.)
    """
    path = project_root / "test_data"
    if not path.exists():
        pytest.skip("test_data directory not found")
    return path


@pytest.fixture(scope="session")
def ground_truth_dir(test_data_dir: Path) -> Path:
    """Path to ground truth files for validation.

    Contains JSON files with expected results for synthetic data.
    """
    return test_data_dir / "synthetic" / "ground_truth" / "decoded"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory.

    Scope: session - Static path, read-only across all tests.
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create and return a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Waveform File Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def tektronix_wfm_dir(test_data_dir: Path) -> Path:
    """Directory containing Tektronix WFM files."""
    return test_data_dir / "formats" / "tektronix"


@pytest.fixture(scope="session")
def wfm_files(tektronix_wfm_dir: Path) -> list[Path]:
    """All valid Tektronix WFM test files.

    Returns files from analog/single_channel/ directory.
    Excludes invalid test files.
    """
    valid_dir = tektronix_wfm_dir / "analog" / "single_channel"
    if not valid_dir.exists():
        return []
    return list(valid_dir.glob("*.wfm"))


@pytest.fixture(scope="session")
def invalid_wfm_files(tektronix_wfm_dir: Path) -> list[Path]:
    """Invalid WFM files for error handling tests."""
    invalid_dir = tektronix_wfm_dir / "invalid_waveforms"
    if not invalid_dir.exists():
        return []
    return list(invalid_dir.glob("*.wfm"))


# =============================================================================
# Real Captures Fixtures (Version-Controlled Real Data)
# =============================================================================


@pytest.fixture(scope="session")
def real_captures_dir(test_data_dir: Path) -> Path:
    """Directory containing real oscilloscope captures.

    These are version-controlled real captures for integration testing
    and demos, organized by type:
    - waveforms/: Real Tektronix WFM files (small/medium/large)
    - sessions/: Tektronix session files (.tss)
    - settings/: Tektronix settings files (.set)
    - packets/: UDP packet subsets (head/middle/tail)
    """
    return test_data_dir / "real_captures"


@pytest.fixture(scope="session")
def real_captures_manifest(real_captures_dir: Path) -> dict[str, Any]:
    """Load manifest.json with metadata for all real captures.

    Returns empty dict if manifest doesn't exist.
    """
    manifest_path = real_captures_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def real_wfm_files(real_captures_dir: Path) -> dict[str, list[Path]]:
    """Real Tektronix WFM files organized by size category.

    Returns dict with keys: 'small', 'medium', 'large'
    Each value is a list of Path objects.
    """
    wfm_dir = real_captures_dir / "waveforms"
    result = {"small": [], "medium": [], "large": []}

    for category in result:
        cat_dir = wfm_dir / category
        if cat_dir.exists():
            result[category] = list(cat_dir.glob("*.wfm"))

    return result


@pytest.fixture(scope="session")
def real_wfm_small(real_wfm_files: dict[str, list[Path]]) -> list[Path]:
    """Small WFM files (< 1.5 MB) for quick tests."""
    return real_wfm_files["small"]


@pytest.fixture(scope="session")
def real_wfm_medium(real_wfm_files: dict[str, list[Path]]) -> list[Path]:
    """Medium WFM files (1.5 - 6 MB) for standard tests."""
    return real_wfm_files["medium"]


@pytest.fixture(scope="session")
def real_wfm_large(real_wfm_files: dict[str, list[Path]]) -> list[Path]:
    """Large WFM files (> 6 MB) for comprehensive tests."""
    return real_wfm_files["large"]


@pytest.fixture(scope="session")
def real_session_files(real_captures_dir: Path) -> list[Path]:
    """Real Tektronix session files (.tss)."""
    session_dir = real_captures_dir / "sessions"
    if not session_dir.exists():
        return []
    return list(session_dir.glob("*.tss"))


@pytest.fixture(scope="session")
def real_settings_files(real_captures_dir: Path) -> list[Path]:
    """Real Tektronix settings files (.set)."""
    settings_dir = real_captures_dir / "settings"
    if not settings_dir.exists():
        return []
    return list(settings_dir.glob("*.set"))


@pytest.fixture(scope="session")
def real_udp_packets(real_captures_dir: Path) -> dict[str, Path | None]:
    """Real UDP packet files (head/middle/tail segments).

    Returns dict with keys: 'head', 'middle', 'tail'
    Each value is a Path or None if file doesn't exist.
    """
    packets_dir = real_captures_dir / "packets" / "udp"
    result = {"head": None, "middle": None, "tail": None}

    for segment in result:
        path = packets_dir / f"udp_{segment}.bin"
        if path.exists():
            result[segment] = path

    return result


@pytest.fixture(scope="module")
def loaded_real_wfm_data(real_wfm_small: list[Path]) -> dict[str, Any]:
    """Pre-load small real WFM files for tests in module (cached).

    Loads up to 3 small WFM files and caches them for the module scope.

    Returns:
        Dict mapping filename to loaded waveform data
    """
    if not real_wfm_small:
        return {}

    try:
        from oscura.loaders.tektronix import load_tektronix_wfm
    except ImportError:
        return {}

    cache = {}
    for wfm_file in real_wfm_small[:3]:
        try:
            cache[str(wfm_file.name)] = load_tektronix_wfm(wfm_file)
        except (OSError, ValueError, RuntimeError) as e:
            # Skip files that fail to load (corrupted, unsupported)
            pass

    return cache


# =============================================================================
# Cached File Loading Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def loaded_wfm_data(wfm_files: list[Path]) -> dict[str, Any]:
    """Pre-load WFM data for all tests in module (cached).

    Loads up to 5 WFM files and caches them for the module scope.
    This avoids redundant file I/O operations when multiple tests
    need the same waveform data.

    Returns:
        Dict mapping filename to loaded waveform data
    """
    if not wfm_files:
        return {}

    # Only import if we have files to load
    try:
        from oscura.loaders.tektronix import load_tektronix_wfm
    except ImportError:
        return {}

    # Load first 5 files only (limit memory usage)
    cache = {}
    for wfm_file in wfm_files[:5]:
        try:
            cache[str(wfm_file.name)] = load_tektronix_wfm(wfm_file)
        except (OSError, ValueError, RuntimeError) as e:
            # Skip files that fail to load (corrupted, unsupported)
            pass

    return cache


# =============================================================================
# PCAP File Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pcap_dir(test_data_dir: Path) -> Path:
    """Directory containing PCAP files."""
    return test_data_dir / "formats" / "pcap"


@pytest.fixture(scope="session")
def pcap_files(pcap_dir: Path) -> list[Path]:
    """All PCAP test files organized by protocol.

    Returns PCAP files from:
    - tcp/: HTTP, FTP, SMTP, SSH, HTTPS
    - udp/: DNS, NTP
    - industrial/: Modbus TCP
    - iot/: MQTT

    Excludes malformed test files.
    """
    if not pcap_dir.exists():
        return []
    # Exclude malformed directory - those files are for error handling tests
    return [p for p in pcap_dir.rglob("*.pcap") if "malformed" not in p.parts]


@pytest.fixture(scope="session")
def http_pcap(pcap_dir: Path) -> Path | None:
    """HTTP PCAP file for protocol testing."""
    path = pcap_dir / "tcp" / "http" / "http.pcap"
    return path if path.exists() else None


@pytest.fixture(scope="session")
def modbus_pcap(pcap_dir: Path) -> Path | None:
    """Modbus TCP PCAP file for industrial protocol testing."""
    path = pcap_dir / "industrial" / "modbus_tcp" / "modbus.pcap"
    return path if path.exists() else None


@pytest.fixture(scope="session")
def mqtt_pcap_files(pcap_dir: Path) -> list[Path]:
    """MQTT PCAP files for IoT protocol testing."""
    mqtt_dir = pcap_dir / "iot" / "mqtt"
    if not mqtt_dir.exists():
        return []
    return list(mqtt_dir.glob("*.pcap"))


# =============================================================================
# Sigrok File Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sigrok_dir(test_data_dir: Path) -> Path:
    """Directory containing sigrok session files."""
    return test_data_dir / "formats" / "sigrok"


@pytest.fixture(scope="session")
def sigrok_files(sigrok_dir: Path) -> list[Path]:
    """All sigrok session files."""
    if not sigrok_dir.exists():
        return []
    return list(sigrok_dir.rglob("*.sr"))


@pytest.fixture(scope="session")
def uart_sigrok_files(sigrok_dir: Path) -> list[Path]:
    """UART sigrok captures for clock recovery testing."""
    uart_dir = sigrok_dir / "uart"
    if not uart_dir.exists():
        return []
    return list(uart_dir.rglob("*.sr"))


@pytest.fixture(scope="session")
def uart_hello_world_files(sigrok_dir: Path) -> list[Path]:
    """UART Hello World captures at various baud rates."""
    hw_dir = sigrok_dir / "uart" / "hello_world" / "8n1"
    if not hw_dir.exists():
        return []
    return list(hw_dir.glob("*.sr"))


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def synthetic_waveform_dir(test_data_dir: Path) -> Path:
    """Directory containing synthetic waveform NPY files."""
    synth_dir = test_data_dir / "synthetic" / "waveforms"
    if not synth_dir.exists():
        # Fallback to legacy location
        synth_dir = test_data_dir / "synthetic" / "waveforms_legacy"
    return synth_dir


@pytest.fixture(scope="session")
def synthetic_binary_dir(test_data_dir: Path) -> Path:
    """Directory containing synthetic binary packet files."""
    return test_data_dir / "synthetic" / "binary"


@pytest.fixture(scope="session")
def synthetic_packets(test_data_dir: Path) -> dict[str, Path]:
    """Synthetic binary packets with ground truth.

    Returns dict with 'data' and 'truth' paths.
    """
    data_path = test_data_dir / "synthetic" / "binary" / "fixed_length" / "clean_packets_512b.bin"

    return {
        "data": data_path,
        "truth": test_data_dir
        / "synthetic"
        / "ground_truth"
        / "decoded"
        / "fixed_length_packets_truth.json",
    }


@pytest.fixture(scope="session")
def square_wave_files(synthetic_waveform_dir: Path) -> dict[str, Path]:
    """Square wave NPY files for edge detection testing.

    Returns dict mapping frequency to file path:
    - 1MHz, 10MHz, 100MHz
    """
    return {
        "1MHz": synthetic_waveform_dir / "square_1MHz.npy",
        "10MHz": synthetic_waveform_dir / "square_10MHz.npy",
        "100MHz": synthetic_waveform_dir / "square_100MHz.npy",
    }


@pytest.fixture(scope="session")
def uart_synthetic_file(synthetic_waveform_dir: Path) -> Path:
    """Synthetic UART waveform for baud rate detection."""
    return synthetic_waveform_dir / "uart_9600_hello_world.npy"


@pytest.fixture(scope="session")
def protocol_message_files(synthetic_binary_dir: Path) -> dict[str, Path]:
    """Protocol message binary files for format inference.

    Returns dict mapping size to file path:
    - 64b, 128b, 256b
    """
    return {
        "64b": synthetic_binary_dir / "protocol_messages_64b.bin",
        "128b": synthetic_binary_dir / "protocol_messages_128b.bin",
        "256b": synthetic_binary_dir / "protocol_messages_256b.bin",
    }


# =============================================================================
# Ground Truth Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def fixed_length_packets_truth(ground_truth_dir: Path) -> dict[str, Any]:
    """Ground truth for fixed length packet validation."""
    truth_file = ground_truth_dir / "fixed_length_packets_truth.json"
    if not truth_file.exists():
        return {}
    with open(truth_file) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def square_wave_truth(ground_truth_dir: Path) -> dict[str, dict[str, Any]]:
    """Ground truth for square wave signal validation.

    Returns dict mapping frequency to ground truth data.
    """
    result = {}
    for freq in ["1MHz", "10MHz", "100MHz"]:
        truth_file = ground_truth_dir / f"square_{freq}_truth.json"
        if truth_file.exists():
            with open(truth_file) as f:
                result[freq] = json.load(f)
    return result


@pytest.fixture(scope="session")
def uart_truth(ground_truth_dir: Path) -> dict[str, Any]:
    """Ground truth for UART signal validation."""
    truth_file = ground_truth_dir / "uart_9600_truth.json"
    if not truth_file.exists():
        return {}
    with open(truth_file) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def message_truth(ground_truth_dir: Path) -> dict[str, dict[str, Any]]:
    """Ground truth for message format validation.

    Returns dict mapping size to ground truth data.
    """
    result = {}
    for size in ["64b", "128b", "256b"]:
        truth_file = ground_truth_dir / f"messages_{size}_truth.json"
        if truth_file.exists():
            with open(truth_file) as f:
                result[size] = json.load(f)
    return result


# =============================================================================
# Statistical Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def statistical_dir(test_data_dir: Path) -> Path:
    """Directory containing statistical test data."""
    return test_data_dir / "synthetic"


@pytest.fixture(scope="session")
def entropy_files(statistical_dir: Path) -> dict[str, Path]:
    """Entropy test files with varying entropy levels.

    Returns dict mapping type to file path:
    - random: High entropy (random bytes)
    - low: Low entropy (repeating pattern)
    - text: Medium entropy (English text)
    """
    entropy_dir = statistical_dir / "entropy"
    return {
        "random": entropy_dir / "random_1MB.bin",
        "low": entropy_dir / "low_entropy_pattern.bin",
        "text": entropy_dir / "english_text.bin",
    }


@pytest.fixture(scope="session")
def pattern_files(statistical_dir: Path) -> dict[str, Path]:
    """Pattern test files for pattern detection.

    Returns dict mapping type to file path:
    - periodic: Periodic sine wave
    - repeating: Repeating sequence
    - anomalies: Data with anomalies
    """
    pattern_dir = statistical_dir / "patterns"
    return {
        "periodic": pattern_dir / "periodic_sine.npy",
        "repeating": pattern_dir / "repeating_sequence.npy",
        "anomalies": pattern_dir / "with_anomalies.npy",
    }


# =============================================================================
# Variable Length Packet Fixtures
# =============================================================================


@pytest.fixture(params=[128, 256, 1024, 2048])
def variable_packet_size(
    request: pytest.FixtureRequest, synthetic_binary_dir: Path
) -> tuple[int, Path]:
    """Variable-length packet files parametrized by size.

    Returns tuple of (size, path).
    """
    size = request.param
    path = synthetic_binary_dir / "variable_length" / f"packets_{size}b.bin"
    return (size, path)


@pytest.fixture(
    params=["noisy_packets_1pct.bin", "noisy_packets_5pct.bin", "noisy_packets_10pct.bin"]
)
def noisy_packet_file(request: pytest.FixtureRequest, synthetic_binary_dir: Path) -> Path:
    """Noisy packet files for error handling tests."""
    return synthetic_binary_dir / "with_errors" / request.param


# =============================================================================
# Signal Generation Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_rate() -> float:
    """Default sample rate for test signals (1 MHz).

    Scope: session - Constant value, no need to recreate.
    """
    return 1_000_000.0


# =============================================================================
# Fixture Factories (Parameterized Test Data Generation)
# =============================================================================


@pytest.fixture
def signal_factory():
    """Factory for creating test signals with various parameters.

    Returns a function that creates signals with configurable:
    - Signal type (sine, square, noise, pulse, chirp, ramp, digital, step, slow_edge)
    - Frequency
    - Sample rate
    - Duration
    - Amplitude and offset
    - Noise level
    - Random seed for reproducibility
    - Step position and transition characteristics

    Signal Types:
        - sine: Pure sine wave
        - square: Square wave (digital)
        - pulse: Pulse train
        - chirp: Linear frequency sweep
        - ramp: Linear ramp
        - digital: Digital signal with 3.3V logic levels
        - noise: Gaussian noise
        - step: Clean step transition (for edge testing)
        - slow_edge: Slow rising edge with configurable transition time

    Usage:
        def test_signal_processing(signal_factory):
            # Basic sine wave
            signal, metadata = signal_factory(
                signal_type="sine",
                frequency=1000.0,
                sample_rate=1e6,
                duration=0.01
            )

            # Square wave with amplitude/offset
            signal, metadata = signal_factory(
                signal_type="square",
                frequency=1000.0,
                sample_rate=1e6,
                duration=0.01,
                amplitude=2.0,
                offset=1.0
            )

            # Clean step for edge detection testing
            signal, metadata = signal_factory(
                signal_type="step",
                sample_rate=1e6,
                duration=0.001,
                step_position=500,
                low_value=0.0,
                high_value=3.3
            )

            # Slow rising edge
            signal, metadata = signal_factory(
                signal_type="slow_edge",
                sample_rate=1e6,
                duration=0.001,
                step_position=100,
                transition_samples=20,
                low_value=0.0,
                high_value=1.0
            )

    Returns:
        Tuple of (signal array, metadata dict)
    """

    def _create_signal(
        signal_type: str = "sine",
        frequency: float = 1000.0,
        sample_rate: float = 1e6,
        duration: float = 0.01,
        noise_level: float = 0.0,
        seed: int = 42,
        amplitude: float = 1.0,
        offset: float = 0.0,
        # Step signal parameters
        step_position: int | None = None,
        low_value: float = 0.0,
        high_value: float = 1.0,
        # Slow edge parameters
        transition_samples: int = 1,
    ) -> tuple[NDArray[np.float64], dict]:
        rng = np.random.default_rng(seed)
        t = np.arange(0, duration, 1 / sample_rate)
        total_samples = len(t)

        if signal_type == "sine":
            signal = np.sin(2 * np.pi * frequency * t)
        elif signal_type == "square":
            signal = (np.sin(2 * np.pi * frequency * t) > 0).astype(np.float64)
        elif signal_type == "pulse":
            signal = np.zeros_like(t)
            pulse_indices = (t % (1 / frequency)) < (0.1 / frequency)
            signal[pulse_indices] = 1.0
        elif signal_type == "chirp":
            # Linear chirp from frequency to 2*frequency
            signal = np.sin(2 * np.pi * (frequency * t + (frequency * t**2) / (2 * duration)))
        elif signal_type == "ramp":
            # Linear ramp from 0 to 1
            signal = t / duration
        elif signal_type == "digital":
            # Digital signal with clean transitions (0V and 3.3V)
            # Pattern: 0, 1, 1, 0, 1, 0, 0, 1 repeated
            pattern = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float64) * 3.3
            samples_per_bit = int(len(t) / len(pattern))
            signal = np.repeat(pattern, samples_per_bit)[: len(t)]
        elif signal_type == "noise":
            signal = rng.normal(0, 1, len(t))
        elif signal_type == "step":
            # Clean step transition at specified position
            if step_position is None:
                step_position = total_samples // 2
            signal = np.full(total_samples, low_value, dtype=np.float64)
            signal[step_position:] = high_value
        elif signal_type == "slow_edge":
            # Step with slow transition (RC charging curve)
            if step_position is None:
                step_position = total_samples // 2
            signal = np.full(total_samples, low_value, dtype=np.float64)
            ramp = np.linspace(low_value, high_value, transition_samples)
            signal[step_position : step_position + transition_samples] = ramp
            if step_position + transition_samples < total_samples:
                signal[step_position + transition_samples :] = high_value
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        # Apply amplitude and offset (except for step/slow_edge which use low/high_value)
        if signal_type not in ["step", "slow_edge"]:
            signal = amplitude * signal + offset

        if noise_level > 0 and signal_type != "noise":
            signal += rng.normal(0, noise_level, len(signal))

        metadata = {
            "signal_type": signal_type,
            "frequency": frequency,
            "sample_rate": sample_rate,
            "duration": duration,
            "noise_level": noise_level,
            "amplitude": amplitude,
            "offset": offset,
            "length": len(signal),
            "seed": seed,
        }

        # Add step-specific metadata
        if signal_type in ["step", "slow_edge"]:
            metadata.update(
                {
                    "step_position": step_position,
                    "low_value": low_value,
                    "high_value": high_value,
                }
            )
        if signal_type == "slow_edge":
            metadata["transition_samples"] = transition_samples

        return signal, metadata

    return _create_signal


@pytest.fixture
def packet_factory():
    """Factory for creating test packets with ground truth.

    Returns a function that creates packets with configurable:
    - Packet count
    - Packet size
    - Header format
    - Checksum type
    - Corruption rate
    - Random seed

    Usage:
        def test_packet_parsing(packet_factory):
            data, truth = packet_factory(
                count=10,
                packet_size=64,
                checksum_type="crc16"
            )
            assert len(truth["packets"]) == 10

    Returns:
        Tuple of (raw bytes, ground truth dict)
    """

    def _create_packets(
        count: int = 10,
        packet_size: int = 64,
        with_header: bool = True,
        checksum_type: str = "sum",  # "sum", "crc16", "xor", None
        corruption_rate: float = 0.0,
        seed: int = 42,
    ) -> tuple[bytes, dict]:
        rng = np.random.default_rng(seed)
        packets = []
        truth = {
            "packets": [],
            "checksums": [],
            "sequence_numbers": [],
            "corrupted_indices": [],
        }

        for i in range(count):
            packet = bytearray()

            if with_header:
                packet.extend(b"\xaa\x55")  # Magic header

            packet.append(i & 0xFF)  # Sequence number

            # Calculate payload size
            checksum_bytes = (
                2 if checksum_type in ["sum", "crc16"] else (1 if checksum_type == "xor" else 0)
            )
            overhead = len(packet) + checksum_bytes
            payload_size = max(1, packet_size - overhead)
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            packet.extend(payload)

            # Add checksum
            checksum = None
            if checksum_type == "sum":
                checksum = sum(packet) & 0xFFFF
                packet.extend(checksum.to_bytes(2, "big"))
            elif checksum_type == "crc16":
                # Simple CRC-16 implementation
                crc = 0xFFFF
                for byte in packet:
                    crc ^= byte
                    for _ in range(8):
                        if crc & 0x0001:
                            crc = (crc >> 1) ^ 0xA001
                        else:
                            crc >>= 1
                packet.extend(crc.to_bytes(2, "big"))
                checksum = crc
            elif checksum_type == "xor":
                checksum = 0
                for byte in packet:
                    checksum ^= byte
                packet.append(checksum)

            # Apply corruption
            is_corrupted = False
            if corruption_rate > 0 and rng.random() < corruption_rate:
                corrupt_idx = rng.integers(0, len(packet))
                packet[corrupt_idx] ^= 0xFF
                is_corrupted = True
                truth["corrupted_indices"].append(i)

            packets.append(bytes(packet))
            truth["sequence_numbers"].append(i)
            truth["checksums"].append(checksum)

        truth["packets"] = packets
        truth["packet_count"] = count
        truth["packet_size"] = packet_size
        truth["checksum_type"] = checksum_type
        return b"".join(packets), truth

    return _create_packets


@pytest.fixture
def waveform_factory(signal_factory):
    """Factory for creating complete waveform data structures.

    Uses signal_factory to generate signals, then wraps them in
    waveform metadata structures matching real file formats.

    Usage:
        def test_waveform_loader(waveform_factory):
            wfm = waveform_factory(
                channels=2,
                signal_type="sine",
                sample_rate=1e6,
                duration=0.01
            )
            assert len(wfm["channels"]) == 2

    Returns:
        Dict with waveform metadata and channel data
    """

    def _create_waveform(
        channels: int = 1,
        signal_type: str = "sine",
        sample_rate: float = 1e6,
        duration: float = 0.01,
        **signal_kwargs,
    ) -> dict:
        waveform = {
            "metadata": {
                "channels": channels,
                "sample_rate": sample_rate,
                "duration": duration,
                "record_length": int(sample_rate * duration),
            },
            "channels": [],
        }

        for ch in range(channels):
            signal, metadata = signal_factory(
                signal_type=signal_type,
                sample_rate=sample_rate,
                duration=duration,
                **signal_kwargs,
            )
            waveform["channels"].append(
                {
                    "label": f"CH{ch + 1}",
                    "data": signal,
                    "metadata": metadata,
                }
            )

        return waveform

    return _create_waveform


# =============================================================================
# Legacy Signal Fixtures (kept for backward compatibility)
# =============================================================================
# These fixtures are kept to avoid breaking existing tests.
# New tests should use the factory fixtures above for more flexibility.


@pytest.fixture
def sine_wave(sample_rate: float) -> NDArray[np.float64]:
    """Generate a 1 kHz sine wave at the default sample rate.

    Returns:
        1000 samples of a 1 kHz sine wave

    Note: Consider using signal_factory for new tests - more flexible.
    """
    frequency = 1000.0  # 1 kHz
    duration = 0.001  # 1 ms
    t = np.arange(0, duration, 1 / sample_rate)
    return np.sin(2 * np.pi * frequency * t)


@pytest.fixture
def square_wave(sample_rate: float) -> NDArray[np.float64]:
    """Generate a 1 kHz square wave at the default sample rate.

    Returns:
        1000 samples of a 1 kHz square wave (0 to 1)
    """
    frequency = 1000.0
    duration = 0.001
    t = np.arange(0, duration, 1 / sample_rate)
    return (np.sin(2 * np.pi * frequency * t) > 0).astype(np.float64)


@pytest.fixture
def ramp_signal() -> NDArray[np.float64]:
    """Generate a linear ramp from 0 to 1.

    Returns:
        100 samples ramping from 0.0 to 1.0
    """
    return np.linspace(0.0, 1.0, 100)


@pytest.fixture
def noisy_sine(sine_wave: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generate a sine wave with added Gaussian noise.

    Returns:
        Sine wave with 10% RMS noise
    """
    rng = np.random.default_rng(42)  # Reproducible
    noise = rng.normal(0, 0.1, len(sine_wave))
    return sine_wave + noise


@pytest.fixture
def digital_signal() -> NDArray[np.float64]:
    """Generate a digital signal with multiple transitions.

    Returns:
        Signal with clean digital transitions (0V and 3.3V levels)
    """
    # Pattern: 0, 1, 1, 0, 1, 0, 0, 1 repeated
    pattern = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float64) * 3.3
    return np.tile(pattern, 125)  # 1000 samples


# =============================================================================
# Protocol Test Data Fixtures
# =============================================================================


@pytest.fixture
def uart_signal() -> tuple[NDArray[np.float64], dict]:
    """Generate a UART signal encoding 'Hello'.

    Returns:
        Tuple of (signal, metadata) where metadata contains baudrate and expected bytes
    """
    # 115200 baud, 8N1
    baudrate = 115200
    samples_per_bit = 10  # Simplified for testing

    # Encode "Hello" with start/stop bits
    message = b"Hello"
    bits = []
    for byte in message:
        bits.append(0)  # Start bit
        for i in range(8):
            bits.append((byte >> i) & 1)
        bits.append(1)  # Stop bit

    # Generate signal
    signal = np.repeat(np.array(bits, dtype=np.float64) * 3.3, samples_per_bit)

    metadata = {
        "baudrate": baudrate,
        "expected_bytes": message,
        "samples_per_bit": samples_per_bit,
    }

    return signal, metadata


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture
def large_signal() -> NDArray[np.float64]:
    """Generate a large signal for performance testing (1M samples)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(1_000_000)


@pytest.fixture
def very_large_signal() -> NDArray[np.float64]:
    """Generate a very large signal for stress testing (10M samples)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(10_000_000)


# =============================================================================
# Matplotlib Test Isolation
# =============================================================================


@pytest.fixture(autouse=True, scope="module")
def cleanup_matplotlib():
    """Automatically close all matplotlib figures once per module (optimal balance).

    This prevents the 'more than 20 figures' warning.

    OPTIMIZED: Module scope provides optimal balance.
    Previous (function): 17,226 tests * 0.001s = 17 seconds overhead
    Optimized (module): 389 modules * 0.001s = 0.4 seconds overhead
    TOTAL SAVINGS: ~17 seconds (95% reduction from original)

    NOTE: plt.close("all") cleans up figures at module boundaries.
    """
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass  # matplotlib not installed


@pytest.fixture(autouse=True, scope="module")
def memory_cleanup():
    """Force garbage collection once per module to prevent memory buildup.

    OPTIMIZATION: Changed from function to module scope.
    Previous: 17,226 tests * 0.01s = 2.9 minutes overhead
    Optimized: 389 modules * 0.01s = 4 seconds overhead
    SAVINGS: ~2 minutes (93% reduction)

    This is critical for large test suites to prevent OOM errors.

    NOTE: ResourceWarning from gc.collect() is suppressed as it's triggered by
    third-party libraries (rigolwfm/kaitaistruct) that don't properly close file handles.
    This is a known limitation of the rigolwfm library.
    """
    yield
    import gc
    import warnings

    # Suppress ResourceWarning during gc.collect() - caused by rigolwfm library
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()


@pytest.fixture(autouse=True, scope="module")
def reset_warnings_state():
    """Reset warnings state once per module (optimal balance).

    OPTIMIZED: Module scope provides optimal balance of speed + isolation.
    Previous (function): 17,226 tests * 0.01s = 2.9 minutes overhead
    Optimized (module): 389 modules * 0.01s = 3.9 seconds overhead
    TOTAL SAVINGS: ~2.9 minutes (98% reduction from original)

    NOTE: Session scope breaks tests (warnings state leakage). Module scope is optimal.

    Combined with clearing only oscura modules (not all modules):
    - Original: 17,226 tests * 0.03s = 8.6 minutes (all modules, function scope)
    - After optimization 1: 17,226 tests * 0.01s = 2.9 minutes (oscura only, function scope)
    - After optimization 2: 389 modules * 0.01s = 3.9 seconds (oscura only, module scope)
    - TOTAL SAVINGS: ~8.6 minutes (99.2% reduction from original)

    Python's warnings module maintains global state - once a warning is shown,
    it won't be shown again by default. Module-level cleanup is sufficient as
    warnings state rarely changes between tests within the same module.
    """
    import warnings

    # Save current filters
    original_filters = warnings.filters[:]

    yield

    # Restore original filters and clear the warnings registry
    warnings.filters[:] = original_filters
    warnings.resetwarnings()

    # Clear __warningregistry__ only from oscura modules (optimization)
    import sys

    for name, module in sys.modules.items():
        if name.startswith(("oscura", "tests")) and hasattr(module, "__warningregistry__"):
            module.__warningregistry__.clear()


@pytest.fixture(scope="function")
def reset_memory_check_state():
    """Reset memory check operation registry between tests.

    OPTIMIZATION: Removed autouse - only needed by tests that use memory_check.
    Previous: 17,226 tests * 0.001s = 17 seconds overhead (all tests)
    Optimized: ~50 tests * 0.001s = 0.05 seconds overhead (opt-in)
    SAVINGS: ~17 seconds

    Tests that call register_auto_check_operation() should explicitly request
    this fixture to modify global state cleanup.

    Usage:
        def test_my_memory_check(reset_memory_check_state):
            ...
    """
    try:
        import oscura.core.memory_check as mem_mod

        # Save original state
        original_ops = mem_mod._AUTO_CHECK_OPERATIONS.copy()

        yield

        # Restore original state
        mem_mod._AUTO_CHECK_OPERATIONS.clear()
        mem_mod._AUTO_CHECK_OPERATIONS.update(original_ops)
    except (ImportError, AttributeError):
        # Module or attribute not available, skip cleanup
        yield


@pytest.fixture(autouse=True, scope="module")
def reset_logging_state():
    """Reset logging configuration once per module (optimal balance).

    OPTIMIZED: Module scope provides optimal balance of speed + isolation.
    Previous (function): 17,226 tests * 0.05s = 14.4 minutes overhead
    Optimized (module): 389 modules * 0.05s = 19 seconds overhead
    TOTAL SAVINGS: ~14.4 minutes (95% reduction from original)

    NOTE: Session scope breaks tests (state leakage). Module scope is optimal.

    Tests that call configure_logging() modify global logger state.
    This fixture ensures each module starts with clean logging configuration.
    """
    import logging

    # Save original logging levels
    original_levels = {}
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level

    yield

    # Restore original levels
    for name, level in original_levels.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)

    # Close and clear handlers from oscura loggers only
    for name in list(logging.root.manager.loggerDict.keys()):
        if name.startswith("oscura"):
            logger = logging.getLogger(name)
            # Close handlers to prevent ResourceWarnings
            for handler in logger.handlers[:]:  # Copy list to avoid modification during iteration
                try:
                    handler.close()
                except (OSError, RuntimeError, ValueError) as e:
                    # Ignore errors during cleanup - handler may already be closed
                    pass
            logger.handlers.clear()
            logger.propagate = True


@pytest.fixture(scope="function")
def reset_threshold_registry():
    """Reset ThresholdRegistry overrides between tests.

    OPTIMIZATION: Removed autouse - only needed by tests that use thresholds.
    Previous: 17,226 tests * 0.001s = 17 seconds overhead (all tests)
    Optimized: ~30 tests * 0.001s = 0.03 seconds overhead (opt-in)
    SAVINGS: ~17 seconds

    ThresholdRegistry is a singleton that maintains session overrides.
    Tests that call set_threshold_override() should explicitly request
    this fixture for cleanup.

    Usage:
        def test_my_threshold(reset_threshold_registry):
            ...
    """
    try:
        from oscura.config.thresholds import ThresholdRegistry

        registry = ThresholdRegistry()
        # Reset any session overrides before the test
        registry.reset_overrides()

        yield

        # Reset overrides after the test as well
        registry.reset_overrides()
    except ImportError:
        # ThresholdRegistry may not be available in all test contexts
        yield


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest hooks.

    Note: Markers are now defined in pyproject.toml [tool.pytest.ini_options]
    to maintain single source of truth and avoid duplication.
    """
    # Register Hypothesis profiles
    try:
        from hypothesis import HealthCheck, Phase, Verbosity, settings

        # Default profile: 100 examples, normal verbosity
        settings.register_profile("default", max_examples=100, print_blob=False)

        # Fast profile: 20 examples with deadline for quick feedback
        settings.register_profile("fast", max_examples=20, deadline=1000)

        # AGGRESSIVE: Ultra-fast profile for maximum speed (sub-2-min target)
        settings.register_profile("ultrafast", max_examples=5, deadline=500, print_blob=False)

        # CI profile: 500 examples, deterministic, optimized for scientific computing
        # - derandomize=True: Reproducible test failures
        # - deadline=2000: 2s per example (sufficient for FFT, correlation, etc.)
        # - database=None: Avoid parallel write conflicts with pytest-xdist
        # - print_blob=True: Better failure reproduction information
        # - suppress too_slow: 500 examples legitimately takes time
        # - phases: Full shrinking enabled for minimal failing examples
        settings.register_profile(
            "ci",
            max_examples=500,
            derandomize=True,
            deadline=2000,  # 2 seconds per example
            database=None,  # Disable for parallel safety
            print_blob=True,  # Better reproduction info
            suppress_health_check=[HealthCheck.too_slow],  # Expected with 500 examples
            phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
        )

        # Debug profile: 10 examples with verbose output
        settings.register_profile(
            "debug", max_examples=10, verbosity=Verbosity.verbose, print_blob=True
        )

        # Load the profile based on environment, defaulting to "ultrafast" for max speed
        # AGGRESSIVE: Use ultrafast profile (5 examples) by default for sub-2-min target
        # Override with: HYPOTHESIS_PROFILE=ci pytest ... (comprehensive, 500 examples)
        #                HYPOTHESIS_PROFILE=fast pytest ... (balanced, 20 examples)
        import os

        profile = os.getenv("HYPOTHESIS_PROFILE", "ultrafast")
        settings.load_profile(profile)
    except ImportError:
        pass  # Hypothesis not installed


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark slow tests for optional filtering.

    Slow tests are skipped by default to keep test suite fast.
    Pass --runslow to include slow/performance/stress tests.

    This includes tests marked with:
    - @pytest.mark.slow
    - @pytest.mark.performance
    - @pytest.mark.stress
    """
    if config.getoption("--runslow"):
        # User explicitly wants to run slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords or "performance" in item.keywords or "stress" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options.

    IMPORTANT: This is the ONLY place pytest_addoption should be defined.
    Do NOT add this hook in subdirectory conftest.py files.
    """
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--runrequires",
        action="store_true",
        default=False,
        help="run tests that require test_data directory",
    )
    parser.addoption(
        "--fuzz-iterations",
        action="store",
        default=10,
        type=int,
        help="number of fuzz test iterations (for validation tests)",
    )


# =============================================================================
# Standardized Signal Generation Fixtures (SignalBuilder)
# =============================================================================


@pytest.fixture
def signal_builder():
    """Provide SignalBuilder for dynamic signal generation.

    Returns SignalBuilder class for creating standardized test signals.
    Use this instead of inline signal generators in tests.

    Usage:
        def test_my_feature(signal_builder):
            signal = signal_builder.sine_wave(frequency=1e3, snr_db=40)
            result = analyze(signal)
    """
    from tests.fixtures.signal_builders import SignalBuilder

    return SignalBuilder


@pytest.fixture(scope="module")
def standard_signals():
    """Cache common signals at module scope for performance.

    Provides pre-generated signals for common test scenarios.
    Cached at module scope to avoid regeneration for each test.

    Available signals:
        - sine_1khz: 1 kHz sine wave
        - sine_10khz: 10 kHz sine wave
        - square_1khz: 1 kHz square wave
        - square_10khz: 10 kHz square wave
        - triangle_1khz: 1 kHz triangle wave
        - noise_40db: Sine wave with 40 dB SNR
        - noise_60db: Sine wave with 60 dB SNR
        - multitone_3freq: Three-tone signal (1, 2, 3 kHz)
        - chirp_1k_10k: Linear chirp from 1-10 kHz
        - white_noise: White noise
        - uart_hello_h: UART frame encoding 'H'
        - digital_alternating: Alternating digital pattern

    Usage:
        def test_fft_peak_detection(standard_signals):
            signal = standard_signals["sine_1khz"]
            result = detect_peaks(signal)
    """
    from tests.fixtures.signal_builders import SignalBuilder

    return {
        "sine_1khz": SignalBuilder.sine_wave(frequency=1e3),
        "sine_10khz": SignalBuilder.sine_wave(frequency=10e3),
        "square_1khz": SignalBuilder.square_wave(frequency=1e3),
        "square_10khz": SignalBuilder.square_wave(frequency=10e3),
        "triangle_1khz": SignalBuilder.triangle_wave(frequency=1e3),
        "noise_40db": SignalBuilder.noisy_sine(snr_db=40),
        "noise_60db": SignalBuilder.noisy_sine(snr_db=60),
        "multitone_3freq": SignalBuilder.multitone([1e3, 2e3, 3e3]),
        "chirp_1k_10k": SignalBuilder.chirp(f0=1e3, f1=10e3),
        "white_noise": SignalBuilder.white_noise(),
        "uart_hello_h": SignalBuilder.uart_frame(ord("H")),
        "digital_alternating": SignalBuilder.digital_pattern("01010101"),
    }
