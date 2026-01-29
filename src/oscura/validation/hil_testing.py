"""Hardware-in-Loop (HIL) testing framework for real hardware validation.

This module provides comprehensive HIL testing capabilities for validating protocol
implementations against real hardware using various interface types.

Supports multiple hardware interfaces:
- Serial ports (UART, RS-232, RS-485) via pyserial
- SocketCAN for automotive/embedded CAN testing via python-can
- USB devices via pyusb
- SPI/I2C via spidev/smbus (Linux only)
- GPIO control via RPi.GPIO or gpiod
- Optional oscilloscope integration via PyVISA

Example:
    >>> from oscura.validation.hil_testing import HILTester, HILConfig
    >>> config = HILConfig(
    ...     interface="serial",
    ...     port="/dev/ttyUSB0",
    ...     baud_rate=115200,
    ...     reset_gpio=17
    ... )
    >>> tester = HILTester(config)
    >>> test_cases = [
    ...     {"name": "ping", "send": b"\\x01\\x02", "expect": b"\\x03\\x04", "timeout": 0.5}
    ... ]
    >>> report = tester.run_tests(test_cases)
    >>> print(f"Passed: {report.passed}/{report.total}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

try:
    import can  # type: ignore[import-untyped]
except ImportError:
    can = None  # type: ignore[assignment]

try:
    import usb  # type: ignore[import-not-found]
    import usb.core  # type: ignore[import-not-found]
except ImportError:
    # Create module structure for test patching even when pyusb unavailable
    import types

    usb = types.ModuleType("usb")  # type: ignore[assignment]
    usb.core = None  # type: ignore[attr-defined]

try:
    import spidev  # type: ignore[import-not-found]
except ImportError:
    spidev = None  # type: ignore[assignment]

try:
    from smbus2 import SMBus  # type: ignore[import-not-found]
except ImportError:
    SMBus = None  # type: ignore[assignment]

try:
    import RPi.GPIO as GPIO  # type: ignore[import-untyped]
except ImportError:
    try:
        import gpiod  # type: ignore[import-not-found]

        GPIO = None  # type: ignore[assignment]
    except ImportError:
        GPIO = None  # type: ignore[assignment]
        gpiod = None  # type: ignore[assignment]

try:
    from scapy.all import IP, UDP, Packet, wrpcap  # type: ignore[attr-defined]
except ImportError:
    IP = None  # type: ignore[assignment]
    UDP = None  # type: ignore[assignment]
    Packet = None  # type: ignore[assignment,misc]
    wrpcap = None  # type: ignore[assignment]

try:
    import serial  # type: ignore[import-untyped]
except ImportError:
    serial = None  # type: ignore[assignment]

from oscura.utils.serial import connect_serial_port


class CANBusProtocol(Protocol):
    """Protocol for python-can Bus interface."""

    def send(self, msg: Any) -> None:
        """Send a CAN message."""
        ...

    def recv(self, timeout: float | None = None) -> Any:
        """Receive a CAN message."""
        ...


class InterfaceType(str, Enum):
    """Supported hardware interface types."""

    SERIAL = "serial"
    SOCKETCAN = "socketcan"
    USB = "usb"
    SPI = "spi"
    I2C = "i2c"


class TestStatus(str, Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class HILConfig:
    """Configuration for Hardware-in-Loop testing.

    Attributes:
        interface: Interface type (serial, socketcan, usb, spi, i2c).
        port: Port identifier (e.g., "/dev/ttyUSB0", "can0", spi device number).
        baud_rate: Baud rate for serial interface (default: 115200).
        timeout: Default timeout in seconds for responses (default: 1.0).
        reset_gpio: GPIO pin number for device reset (optional).
        power_gpio: GPIO pin number for device power control (optional).
        reset_duration: Reset pulse duration in seconds (default: 0.1).
        setup_delay: Delay after setup before testing in seconds (default: 0.5).
        teardown_delay: Delay before teardown in seconds (default: 0.1).
        dry_run: Enable dry-run mode without real hardware (default: False).
        validate_timing: Enable timing validation (default: True).
        capture_pcap: Enable PCAP capture of traffic (default: False).
        pcap_file: Output PCAP file path (default: "hil_capture.pcap").
        oscilloscope_address: VISA address for oscilloscope (optional).
        usb_vendor_id: USB vendor ID (for USB interface).
        usb_product_id: USB product ID (for USB interface).
        spi_bus: SPI bus number (default: 0).
        spi_device: SPI device number (default: 0).
        spi_speed_hz: SPI clock speed in Hz (default: 1000000).
        i2c_bus: I2C bus number (default: 1).
        i2c_address: I2C device address (default: 0x50).
    """

    interface: InterfaceType | str
    port: str | int
    baud_rate: int = 115200
    timeout: float = 1.0
    reset_gpio: int | None = None
    power_gpio: int | None = None
    reset_duration: float = 0.1
    setup_delay: float = 0.5
    teardown_delay: float = 0.1
    dry_run: bool = False
    validate_timing: bool = True
    capture_pcap: bool = False
    pcap_file: str = "hil_capture.pcap"
    oscilloscope_address: str | None = None
    usb_vendor_id: int | None = None
    usb_product_id: int | None = None
    spi_bus: int = 0
    spi_device: int = 0
    spi_speed_hz: int = 1000000
    i2c_bus: int = 1
    i2c_address: int = 0x50

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert string to enum if needed
        if isinstance(self.interface, str):
            try:
                self.interface = InterfaceType(self.interface)
            except ValueError as e:
                raise ValueError(
                    f"Invalid interface: {self.interface}. "
                    f"Must be one of: {', '.join(t.value for t in InterfaceType)}"
                ) from e

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.baud_rate <= 0:
            raise ValueError(f"baud_rate must be positive, got {self.baud_rate}")
        if self.reset_duration < 0:
            raise ValueError(f"reset_duration must be non-negative, got {self.reset_duration}")
        if self.setup_delay < 0:
            raise ValueError(f"setup_delay must be non-negative, got {self.setup_delay}")


@dataclass
class HILTestResult:
    """Result from a single HIL test case.

    Attributes:
        test_name: Name of the test case.
        status: Test execution status (passed, failed, error, timeout, skipped).
        sent_data: Data sent to hardware (as hex string).
        received_data: Data received from hardware (as hex string, None if timeout).
        expected_data: Expected response data (as hex string, None if not specified).
        latency: Response latency in seconds (None if timeout).
        error: Error message if status is ERROR (None otherwise).
        timestamp: Test execution timestamp.
        timing_valid: Whether timing was within tolerance (None if not validated).
        bit_errors: Number of bit errors detected (0 if perfect match).
    """

    test_name: str
    status: TestStatus
    sent_data: str
    received_data: str | None
    expected_data: str | None
    latency: float | None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    timing_valid: bool | None = None
    bit_errors: int = 0

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED


@dataclass
class HILTestReport:
    """Comprehensive report from HIL test execution.

    Attributes:
        test_results: List of individual test results.
        total: Total number of tests executed.
        passed: Number of tests that passed.
        failed: Number of tests that failed.
        errors: Number of tests with errors.
        timeouts: Number of tests that timed out.
        skipped: Number of tests that were skipped.
        hardware_info: Hardware configuration information.
        timing_statistics: Timing statistics (min/max/avg latency in seconds).
        start_time: Test suite start timestamp.
        end_time: Test suite end timestamp.
        duration: Total execution duration in seconds.
    """

    test_results: list[HILTestResult]
    total: int
    passed: int
    failed: int
    errors: int
    timeouts: int
    skipped: int
    hardware_info: dict[str, Any]
    timing_statistics: dict[str, float]
    start_time: float
    end_time: float
    duration: float

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate.

        Returns:
            Success rate as fraction (0.0-1.0).
        """
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export report to dictionary for JSON serialization.

        Returns:
            Dictionary with complete test report data.
        """
        return {
            "test_results": [
                {
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "sent_data": r.sent_data,
                    "received_data": r.received_data,
                    "expected_data": r.expected_data,
                    "latency": r.latency,
                    "error": r.error,
                    "timestamp": r.timestamp,
                    "timing_valid": r.timing_valid,
                    "bit_errors": r.bit_errors,
                }
                for r in self.test_results
            ],
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "timeouts": self.timeouts,
                "skipped": self.skipped,
                "success_rate": self.success_rate,
            },
            "hardware_info": self.hardware_info,
            "timing_statistics": self.timing_statistics,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }


class HILTester:
    """Hardware-in-Loop testing framework.

    Automates hardware validation testing by sending test vectors to real hardware
    and validating responses against expected behavior. Supports multiple interface
    types with automatic setup/teardown.

    Example:
        >>> config = HILConfig(interface="serial", port="/dev/ttyUSB0")
        >>> tester = HILTester(config)
        >>> tester.setup()
        >>> test = {"name": "echo", "send": b"\\x01", "expect": b"\\x01"}
        >>> result = tester.run_test(test)
        >>> tester.teardown()
    """

    def __init__(self, config: HILConfig) -> None:
        """Initialize HIL tester with configuration.

        Args:
            config: HIL testing configuration.
        """
        self.config = config
        self._connection: Any = None
        self._gpio_controller: Any = None
        self._is_setup = False
        self._pcap_packets: list[tuple[float, bytes, bytes | None]] = []

    def __enter__(self) -> HILTester:
        """Context manager entry - setup hardware."""
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - teardown hardware."""
        self.teardown()

    def setup(self) -> None:
        """Setup hardware connection and initialize device.

        Performs:
        1. GPIO initialization (if configured)
        2. Power on device (if power GPIO configured)
        3. Reset device (if reset GPIO configured)
        4. Initialize interface connection
        5. Wait for device stability

        Raises:
            ImportError: If required library is not installed.
            OSError: If hardware connection fails.
            RuntimeError: If already setup.
        """
        if self._is_setup:
            raise RuntimeError("Already setup. Call teardown() first.")

        # Initialize GPIO if needed
        if self.config.reset_gpio is not None or self.config.power_gpio is not None:
            self._setup_gpio()

        # Power on device if configured
        if self.config.power_gpio is not None:
            self._power_on()

        # Reset device if configured
        if self.config.reset_gpio is not None:
            self._reset_device()

        # Connect to hardware interface
        if not self.config.dry_run:
            self._connect()

        # Wait for device to stabilize
        if self.config.setup_delay > 0:
            time.sleep(self.config.setup_delay)

        self._is_setup = True

    def teardown(self) -> None:
        """Teardown hardware connection and cleanup resources.

        Performs:
        1. Wait for teardown delay
        2. Close interface connection
        3. Power off device (if power GPIO configured)
        4. Cleanup GPIO resources
        5. Export PCAP if capture enabled
        """
        if not self._is_setup:
            return

        # Wait before teardown
        if self.config.teardown_delay > 0:
            time.sleep(self.config.teardown_delay)

        # Close connection
        if self._connection is not None:
            try:
                if hasattr(self._connection, "close"):
                    self._connection.close()
                elif hasattr(self._connection, "shutdown"):
                    self._connection.shutdown()
            except Exception:
                pass  # Best effort cleanup
            finally:
                self._connection = None

        # Power off device if configured
        if self.config.power_gpio is not None and self._gpio_controller is not None:
            self._power_off()

        # Cleanup GPIO
        if self._gpio_controller is not None:
            try:
                if hasattr(self._gpio_controller, "cleanup"):
                    self._gpio_controller.cleanup()
            except Exception:
                pass
            finally:
                self._gpio_controller = None

        # Export PCAP if enabled
        if self.config.capture_pcap and self._pcap_packets:
            self._export_pcap()

        self._is_setup = False

    def run_test(
        self,
        test_case: dict[str, Any],
    ) -> HILTestResult:
        """Execute a single test case.

        Args:
            test_case: Test case dictionary with keys:
                - name (str): Test case name
                - send (bytes): Data to send to hardware
                - expect (bytes, optional): Expected response data
                - timeout (float, optional): Override default timeout
                - max_latency (float, optional): Maximum acceptable latency
                - min_latency (float, optional): Minimum acceptable latency
                - skip (bool, optional): Skip this test

        Returns:
            Test result with status, timing, and validation info.

        Raises:
            RuntimeError: If not setup.

        Example:
            >>> tester = HILTester(HILConfig("serial", "/dev/ttyUSB0"))
            >>> tester.setup()
            >>> result = tester.run_test({
            ...     "name": "echo_test",
            ...     "send": b"\\x01\\x02",
            ...     "expect": b"\\x01\\x02",
            ...     "timeout": 0.5
            ... })
            >>> tester.teardown()
        """
        if not self._is_setup:
            raise RuntimeError("Not setup. Call setup() first.")

        test_params = self._extract_test_parameters(test_case)
        if test_params["skip"]:
            return self._create_skipped_result(test_params)

        try:
            return self._execute_test_case(test_params)
        except Exception as e:
            return self._create_error_result(test_params, e)

    def _extract_test_parameters(self, test_case: dict[str, Any]) -> dict[str, Any]:
        """Extract test parameters from test case dictionary."""
        return {
            "test_name": test_case.get("name", "unnamed_test"),
            "send_data": test_case.get("send", b""),
            "expect_data": test_case.get("expect"),
            "timeout": test_case.get("timeout", self.config.timeout),
            "max_latency": test_case.get("max_latency"),
            "min_latency": test_case.get("min_latency"),
            "skip": test_case.get("skip", False),
        }

    def _create_skipped_result(self, test_params: dict[str, Any]) -> HILTestResult:
        """Create result for skipped test."""
        return HILTestResult(
            test_name=test_params["test_name"],
            status=TestStatus.SKIPPED,
            sent_data=test_params["send_data"].hex(),
            received_data=None,
            expected_data=test_params["expect_data"].hex() if test_params["expect_data"] else None,
            latency=None,
        )

    def _execute_test_case(self, test_params: dict[str, Any]) -> HILTestResult:
        """Execute test case and return result."""
        start_time = time.time()
        response = self._send_receive(test_params["send_data"], test_params["timeout"])
        end_time = time.time()
        latency = end_time - start_time

        if self.config.capture_pcap:
            self._pcap_packets.append((start_time, test_params["send_data"], response))

        if response is None:
            return self._create_timeout_result(test_params)

        status, bit_errors = self._evaluate_response(response, test_params["expect_data"])
        timing_valid = self._validate_timing(
            latency, test_params["max_latency"], test_params["min_latency"], status
        )

        if timing_valid is False and status == TestStatus.PASSED:
            status = TestStatus.FAILED

        return HILTestResult(
            test_name=test_params["test_name"],
            status=status,
            sent_data=test_params["send_data"].hex(),
            received_data=response.hex(),
            expected_data=test_params["expect_data"].hex() if test_params["expect_data"] else None,
            latency=latency,
            timing_valid=timing_valid,
            bit_errors=bit_errors,
        )

    def _create_timeout_result(self, test_params: dict[str, Any]) -> HILTestResult:
        """Create result for timed-out test."""
        return HILTestResult(
            test_name=test_params["test_name"],
            status=TestStatus.TIMEOUT,
            sent_data=test_params["send_data"].hex(),
            received_data=None,
            expected_data=test_params["expect_data"].hex() if test_params["expect_data"] else None,
            latency=None,
            timing_valid=None,
            bit_errors=0,
        )

    def _evaluate_response(
        self, response: bytes, expect_data: bytes | None
    ) -> tuple[TestStatus, int]:
        """Evaluate response against expected data."""
        if expect_data is None:
            return TestStatus.PASSED, 0
        if response == expect_data:
            return TestStatus.PASSED, 0
        return TestStatus.FAILED, self._count_bit_errors(response, expect_data)

    def _validate_timing(
        self,
        latency: float,
        max_latency: float | None,
        min_latency: float | None,
        status: TestStatus,
    ) -> bool | None:
        """Validate timing constraints."""
        if not self.config.validate_timing or (not max_latency and not min_latency):
            return None
        timing_valid = True
        if max_latency and latency > max_latency:
            timing_valid = False
        if min_latency and latency < min_latency:
            timing_valid = False
        return timing_valid

    def _create_error_result(self, test_params: dict[str, Any], error: Exception) -> HILTestResult:
        """Create result for test with error."""
        return HILTestResult(
            test_name=test_params["test_name"],
            status=TestStatus.ERROR,
            sent_data=test_params["send_data"].hex(),
            received_data=None,
            expected_data=test_params["expect_data"].hex() if test_params["expect_data"] else None,
            latency=None,
            error=f"{type(error).__name__}: {error}",
        )

    def run_tests(self, test_cases: list[dict[str, Any]]) -> HILTestReport:
        """Execute a suite of test cases.

        Args:
            test_cases: List of test case dictionaries (see run_test for format).

        Returns:
            Comprehensive test report with statistics and all results.

        Example:
            >>> config = HILConfig(interface="serial", port="/dev/ttyUSB0")
            >>> tester = HILTester(config)
            >>> tests = [
            ...     {"name": "test1", "send": b"\\x01", "expect": b"\\x02"},
            ...     {"name": "test2", "send": b"\\x03", "expect": b"\\x04"},
            ... ]
            >>> with tester:
            ...     report = tester.run_tests(tests)
            >>> print(f"Success rate: {report.success_rate:.1%}")
        """
        start_time = time.time()
        results: list[HILTestResult] = []
        latencies: list[float] = []

        for test_case in test_cases:
            result = self.run_test(test_case)
            results.append(result)
            if result.latency is not None:
                latencies.append(result.latency)

        end_time = time.time()

        # Calculate statistics
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        timeouts = sum(1 for r in results if r.status == TestStatus.TIMEOUT)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)

        timing_stats = {}
        if latencies:
            timing_stats = {
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "avg_latency": sum(latencies) / len(latencies),
                "total_samples": len(latencies),
            }

        # Interface is always InterfaceType after __post_init__
        interface_value = (
            self.config.interface.value
            if isinstance(self.config.interface, InterfaceType)
            else self.config.interface
        )

        hardware_info = {
            "interface": interface_value,
            "port": str(self.config.port),
            "baud_rate": self.config.baud_rate,
            "timeout": self.config.timeout,
            "dry_run": self.config.dry_run,
        }

        return HILTestReport(
            test_results=results,
            total=len(results),
            passed=passed,
            failed=failed,
            errors=errors,
            timeouts=timeouts,
            skipped=skipped,
            hardware_info=hardware_info,
            timing_statistics=timing_stats,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )

    def _setup_gpio(self) -> None:
        """Initialize GPIO controller.

        Tries to use gpiod first (modern Linux), falls back to RPi.GPIO.

        Raises:
            ImportError: If no GPIO library is available.
        """
        if "gpiod" in globals() and gpiod is not None:
            # Use libgpiod for modern Linux systems
            self._gpio_controller = gpiod
        elif GPIO is not None:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._gpio_controller = GPIO

            # Setup pins as outputs
            if self.config.reset_gpio is not None:
                GPIO.setup(self.config.reset_gpio, GPIO.OUT, initial=GPIO.HIGH)
            if self.config.power_gpio is not None:
                GPIO.setup(self.config.power_gpio, GPIO.OUT, initial=GPIO.LOW)
        else:
            raise ImportError(
                "No GPIO library available. Install gpiod or RPi.GPIO: "
                "pip install gpiod  # or  pip install RPi.GPIO"
            )

    def _power_on(self) -> None:
        """Power on device via GPIO."""
        if self._gpio_controller is None or self.config.power_gpio is None:
            return

        # Assuming active-high power control
        if hasattr(self._gpio_controller, "output"):
            # RPi.GPIO
            self._gpio_controller.output(self.config.power_gpio, True)
        # Add gpiod support if needed

    def _power_off(self) -> None:
        """Power off device via GPIO."""
        if self._gpio_controller is None or self.config.power_gpio is None:
            return

        if hasattr(self._gpio_controller, "output"):
            # RPi.GPIO
            self._gpio_controller.output(self.config.power_gpio, False)

    def _reset_device(self) -> None:
        """Reset device via GPIO pulse."""
        if self._gpio_controller is None or self.config.reset_gpio is None:
            return

        # Assuming active-low reset (pulse low to reset)
        if hasattr(self._gpio_controller, "output"):
            # RPi.GPIO
            self._gpio_controller.output(self.config.reset_gpio, False)
            time.sleep(self.config.reset_duration)
            self._gpio_controller.output(self.config.reset_gpio, True)

    def _connect(self) -> None:
        """Connect to hardware interface.

        Raises:
            ImportError: If required library is not installed.
            OSError: If connection fails.
        """
        if self.config.interface == InterfaceType.SERIAL:
            self._connect_serial()
        elif self.config.interface == InterfaceType.SOCKETCAN:
            self._connect_socketcan()
        elif self.config.interface == InterfaceType.USB:
            self._connect_usb()
        elif self.config.interface == InterfaceType.SPI:
            self._connect_spi()
        elif self.config.interface == InterfaceType.I2C:
            self._connect_i2c()

    def _connect_serial(self) -> None:
        """Connect to serial port.

        Raises:
            ImportError: If pyserial is not installed.
            OSError: If serial port cannot be opened.
        """
        self._connection = connect_serial_port(
            port=str(self.config.port),
            baud_rate=self.config.baud_rate,
            timeout=self.config.timeout,
        )

    def _connect_socketcan(self) -> None:
        """Connect to SocketCAN interface.

        Raises:
            ImportError: If python-can is not installed.
            OSError: If CAN interface cannot be opened.
        """
        if can is None:
            raise ImportError(
                "python-can is required for SocketCAN. Install with: pip install python-can"
            )

        if not isinstance(self.config.port, str):
            raise ValueError(f"CAN interface must be string, got {type(self.config.port)}")

        self._connection = can.interface.Bus(
            channel=self.config.port, interface="socketcan", receive_own_messages=False
        )

    def _connect_usb(self) -> None:
        """Connect to USB device.

        Raises:
            ImportError: If pyusb is not installed.
            OSError: If USB device not found or cannot be opened.
        """
        if getattr(usb, "core", None) is None:
            raise ImportError(
                "pyusb is required for USB interface. Install with: pip install pyusb"
            )

        if self.config.usb_vendor_id is None or self.config.usb_product_id is None:
            raise ValueError("usb_vendor_id and usb_product_id must be set for USB interface")

        dev = usb.core.find(
            idVendor=self.config.usb_vendor_id, idProduct=self.config.usb_product_id
        )
        if dev is None:
            raise OSError(
                f"USB device not found: {self.config.usb_vendor_id:04x}:"
                f"{self.config.usb_product_id:04x}"
            )

        self._connection = dev

    def _connect_spi(self) -> None:
        """Connect to SPI device.

        Raises:
            ImportError: If spidev is not installed.
            OSError: If SPI device cannot be opened.
        """
        if spidev is None:
            raise ImportError(
                "spidev is required for SPI interface. Install with: pip install spidev"
            )

        spi = spidev.SpiDev()
        spi.open(self.config.spi_bus, self.config.spi_device)
        spi.max_speed_hz = self.config.spi_speed_hz
        self._connection = spi

    def _connect_i2c(self) -> None:
        """Connect to I2C device.

        Raises:
            ImportError: If smbus2 is not installed.
            OSError: If I2C device cannot be opened.
        """
        if SMBus is None:
            raise ImportError(
                "smbus2 is required for I2C interface. Install with: pip install smbus2"
            )

        self._connection = SMBus(self.config.i2c_bus)

    def _send_receive(self, data: bytes, timeout: float) -> bytes | None:
        """Send data and receive response.

        Args:
            data: Data to send.
            timeout: Timeout in seconds.

        Returns:
            Response data, or None if timeout.

        Raises:
            OSError: If send/receive fails.
        """
        if self.config.dry_run:
            # In dry-run mode, echo the data back
            time.sleep(0.001)  # Simulate minimal latency
            return data

        if self.config.interface == InterfaceType.SERIAL:
            return self._send_receive_serial(data, timeout)
        elif self.config.interface == InterfaceType.SOCKETCAN:
            return self._send_receive_socketcan(data, timeout)
        elif self.config.interface == InterfaceType.USB:
            return self._send_receive_usb(data, timeout)
        elif self.config.interface == InterfaceType.SPI:
            return self._send_receive_spi(data)
        elif self.config.interface == InterfaceType.I2C:
            return self._send_receive_i2c(data)

        return None

    def _send_receive_serial(self, data: bytes, timeout: float) -> bytes | None:
        """Send/receive via serial port."""
        ser: serial.Serial = self._connection
        original_timeout = ser.timeout
        ser.timeout = timeout
        ser.reset_input_buffer()
        ser.write(data)
        ser.flush()

        # Read response
        response = ser.read(1024)
        ser.timeout = original_timeout
        return response if response else None

    def _send_receive_socketcan(self, data: bytes, timeout: float) -> bytes | None:
        """Send/receive via SocketCAN."""
        bus: CANBusProtocol = self._connection
        msg = can.Message(arbitration_id=0x123, data=data, is_extended_id=False)
        bus.send(msg)

        response_msg = bus.recv(timeout=timeout)
        return bytes(response_msg.data) if response_msg else None

    def _send_receive_usb(self, data: bytes, timeout: float) -> bytes | None:
        """Send/receive via USB bulk transfer."""
        dev = self._connection
        endpoint_out = 0x01  # Typically endpoint 1 OUT
        endpoint_in = 0x81  # Typically endpoint 1 IN

        # Send data
        dev.write(endpoint_out, data, int(timeout * 1000))

        # Receive response
        try:
            response = dev.read(endpoint_in, 1024, int(timeout * 1000))
            return bytes(response) if response else None
        except Exception:
            return None

    def _send_receive_spi(self, data: bytes) -> bytes | None:
        """Send/receive via SPI (full-duplex)."""
        spi = self._connection
        response = spi.xfer2(list(data))
        return bytes(response)

    def _send_receive_i2c(self, data: bytes) -> bytes | None:
        """Send/receive via I2C."""
        bus = self._connection
        # Write data
        for byte in data:
            bus.write_byte(self.config.i2c_address, byte)

        # Read response (assume same length as sent)
        response = []
        for _ in range(len(data)):
            response.append(bus.read_byte(self.config.i2c_address))

        return bytes(response)

    def _count_bit_errors(self, received: bytes, expected: bytes) -> int:
        """Count number of bit errors between received and expected data.

        Args:
            received: Received data.
            expected: Expected data.

        Returns:
            Number of bit errors (Hamming distance).
        """
        # Pad shorter sequence
        max_len = max(len(received), len(expected))
        received_padded = received + b"\x00" * (max_len - len(received))
        expected_padded = expected + b"\x00" * (max_len - len(expected))

        bit_errors = 0
        for r, e in zip(received_padded, expected_padded, strict=True):
            # Count differing bits
            xor = r ^ e
            while xor:
                bit_errors += xor & 1
                xor >>= 1

        return bit_errors

    def _export_pcap(self) -> None:
        """Export captured traffic to PCAP file.

        Requires scapy to be installed.
        """
        if wrpcap is None or IP is None or UDP is None or Packet is None:
            # Silently skip if scapy not available
            return

        packets: list[Any] = []
        for timestamp, sent_data, recv_data in self._pcap_packets:
            # Create UDP packet for sent data
            pkt = IP(dst="192.168.1.1") / UDP(dport=12345) / bytes(sent_data)
            pkt.time = timestamp
            packets.append(pkt)

            # Create UDP packet for received data if present
            if recv_data:
                pkt = IP(src="192.168.1.1") / UDP(sport=12345) / bytes(recv_data)
                pkt.time = timestamp + 0.001  # Slight offset
                packets.append(pkt)

        if packets:
            wrpcap(self.config.pcap_file, packets)


__all__ = [
    "HILConfig",
    "HILTestReport",
    "HILTestResult",
    "HILTester",
    "InterfaceType",
    "TestStatus",
]
