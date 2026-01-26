"""Unit tests for Hardware-in-Loop (HIL) testing framework.

This module tests the HIL testing framework including:
- Configuration validation
- Dry-run mode operation
- Test execution and result reporting
- Timing validation
- Error handling
- Multiple interface types (mocked)
- GPIO control (mocked)
- PCAP export (mocked)
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.validation.hil_testing import (
    HILConfig,
    HILTester,
    HILTestReport,
    HILTestResult,
    InterfaceType,
)
from oscura.validation.hil_testing import TestStatus as HILTestStatus


class TestHILConfig:
    """Test HILConfig dataclass."""

    def test_config_minimal(self) -> None:
        """Test HILConfig with minimal required parameters."""
        config = HILConfig(interface="serial", port="/dev/ttyUSB0")

        assert config.interface == InterfaceType.SERIAL
        assert config.port == "/dev/ttyUSB0"
        assert config.baud_rate == 115200
        assert config.timeout == 1.0
        assert config.dry_run is False

    def test_config_all_parameters(self) -> None:
        """Test HILConfig with all parameters specified."""
        config = HILConfig(
            interface="socketcan",
            port="can0",
            baud_rate=500000,
            timeout=2.0,
            reset_gpio=17,
            power_gpio=18,
            reset_duration=0.2,
            setup_delay=1.0,
            teardown_delay=0.5,
            dry_run=True,
            validate_timing=False,
            capture_pcap=True,
            pcap_file="test.pcap",
        )

        assert config.interface == InterfaceType.SOCKETCAN
        assert config.port == "can0"
        assert config.baud_rate == 500000
        assert config.timeout == 2.0
        assert config.reset_gpio == 17
        assert config.power_gpio == 18
        assert config.reset_duration == 0.2
        assert config.setup_delay == 1.0
        assert config.teardown_delay == 0.5
        assert config.dry_run is True
        assert config.validate_timing is False
        assert config.capture_pcap is True
        assert config.pcap_file == "test.pcap"

    def test_config_enum_conversion(self) -> None:
        """Test that string interface is converted to enum."""
        config = HILConfig(interface="spi", port=0)
        assert config.interface == InterfaceType.SPI
        assert isinstance(config.interface, InterfaceType)

    def test_config_invalid_interface(self) -> None:
        """Test that invalid interface raises ValueError."""
        with pytest.raises(ValueError, match="Invalid interface"):
            HILConfig(interface="invalid", port="/dev/null")  # type: ignore[arg-type]

    def test_config_invalid_timeout(self) -> None:
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            HILConfig(interface="serial", port="/dev/null", timeout=0)

    def test_config_invalid_baud_rate(self) -> None:
        """Test that invalid baud rate raises ValueError."""
        with pytest.raises(ValueError, match="baud_rate must be positive"):
            HILConfig(interface="serial", port="/dev/null", baud_rate=-1)

    def test_config_invalid_reset_duration(self) -> None:
        """Test that invalid reset duration raises ValueError."""
        with pytest.raises(ValueError, match="reset_duration must be non-negative"):
            HILConfig(interface="serial", port="/dev/null", reset_duration=-0.1)


class TestHILTestResult:
    """Test HILTestResult dataclass."""

    def test_result_passed(self) -> None:
        """Test HILTestResult for passed test."""
        result = HILTestResult(
            test_name="test1",
            status=HILTestStatus.PASSED,
            sent_data="0102",
            received_data="0304",
            expected_data="0304",
            latency=0.123,
        )

        assert result.test_name == "test1"
        assert result.status == HILTestStatus.PASSED
        assert result.passed is True
        assert result.sent_data == "0102"
        assert result.received_data == "0304"
        assert result.expected_data == "0304"
        assert result.latency == 0.123
        assert result.error is None
        assert result.bit_errors == 0

    def test_result_failed(self) -> None:
        """Test HILTestResult for failed test."""
        result = HILTestResult(
            test_name="test2",
            status=HILTestStatus.FAILED,
            sent_data="01",
            received_data="02",
            expected_data="03",
            latency=0.5,
            bit_errors=2,
        )

        assert result.status == HILTestStatus.FAILED
        assert result.passed is False
        assert result.bit_errors == 2

    def test_result_timeout(self) -> None:
        """Test HILTestResult for timeout."""
        result = HILTestResult(
            test_name="test3",
            status=HILTestStatus.TIMEOUT,
            sent_data="01",
            received_data=None,
            expected_data="02",
            latency=None,
        )

        assert result.status == HILTestStatus.TIMEOUT
        assert result.passed is False
        assert result.received_data is None
        assert result.latency is None

    def test_result_error(self) -> None:
        """Test HILTestResult for error."""
        result = HILTestResult(
            test_name="test4",
            status=HILTestStatus.ERROR,
            sent_data="01",
            received_data=None,
            expected_data=None,
            latency=None,
            error="OSError: Connection failed",
        )

        assert result.status == HILTestStatus.ERROR
        assert result.passed is False
        assert result.error == "OSError: Connection failed"


class TestHILTestReport:
    """Test HILTestReport dataclass."""

    def test_report_empty(self) -> None:
        """Test HILTestReport with no tests."""
        report = HILTestReport(
            test_results=[],
            total=0,
            passed=0,
            failed=0,
            errors=0,
            timeouts=0,
            skipped=0,
            hardware_info={},
            timing_statistics={},
            start_time=1000.0,
            end_time=1001.0,
            duration=1.0,
        )

        assert report.total == 0
        assert report.success_rate == 0.0

    def test_report_statistics(self) -> None:
        """Test HILTestReport statistics calculation."""
        results = [
            HILTestResult("t1", HILTestStatus.PASSED, "01", "02", "02", 0.1),
            HILTestResult("t2", HILTestStatus.PASSED, "03", "04", "04", 0.2),
            HILTestResult("t3", HILTestStatus.FAILED, "05", "06", "07", 0.3),
            HILTestResult("t4", HILTestStatus.TIMEOUT, "08", None, "09", None),
        ]

        report = HILTestReport(
            test_results=results,
            total=4,
            passed=2,
            failed=1,
            errors=0,
            timeouts=1,
            skipped=0,
            hardware_info={"interface": "serial"},
            timing_statistics={"avg_latency": 0.2},
            start_time=1000.0,
            end_time=1002.0,
            duration=2.0,
        )

        assert report.total == 4
        assert report.passed == 2
        assert report.failed == 1
        assert report.timeouts == 1
        assert report.success_rate == 0.5

    def test_report_to_dict(self) -> None:
        """Test HILTestReport to_dict export."""
        result = HILTestResult("test", HILTestStatus.PASSED, "01", "02", "02", 0.1)
        report = HILTestReport(
            test_results=[result],
            total=1,
            passed=1,
            failed=0,
            errors=0,
            timeouts=0,
            skipped=0,
            hardware_info={"interface": "serial"},
            timing_statistics={"avg_latency": 0.1},
            start_time=1000.0,
            end_time=1000.5,
            duration=0.5,
        )

        data = report.to_dict()

        assert "test_results" in data
        assert "summary" in data
        assert data["summary"]["total"] == 1
        assert data["summary"]["passed"] == 1
        assert data["summary"]["success_rate"] == 1.0
        assert data["hardware_info"]["interface"] == "serial"
        assert data["duration"] == 0.5


class TestHILTesterDryRun:
    """Test HILTester in dry-run mode (no hardware required)."""

    def test_dry_run_setup_teardown(self) -> None:
        """Test setup and teardown in dry-run mode."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)

        tester.setup()
        assert tester._is_setup is True

        tester.teardown()
        assert tester._is_setup is False

    def test_dry_run_context_manager(self) -> None:
        """Test HILTester as context manager in dry-run mode."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)

        with HILTester(config) as tester:
            assert tester._is_setup is True

        assert tester._is_setup is False

    def test_dry_run_single_test_pass(self) -> None:
        """Test single test execution in dry-run mode (echoes data)."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {
            "name": "echo_test",
            "send": b"\x01\x02\x03",
            "expect": b"\x01\x02\x03",  # Dry-run echoes
        }

        result = tester.run_test(test_case)

        assert result.test_name == "echo_test"
        assert result.status == HILTestStatus.PASSED
        assert result.sent_data == "010203"
        assert result.received_data == "010203"
        assert result.latency is not None
        assert result.latency < 0.1  # Should be very fast

        tester.teardown()

    def test_dry_run_single_test_fail(self) -> None:
        """Test single test failure in dry-run mode."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {
            "name": "fail_test",
            "send": b"\x01",
            "expect": b"\xff",  # Won't match echo
        }

        result = tester.run_test(test_case)

        assert result.status == HILTestStatus.FAILED
        assert result.bit_errors > 0

        tester.teardown()

    def test_dry_run_skip_test(self) -> None:
        """Test skipping a test."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {
            "name": "skipped_test",
            "send": b"\x01",
            "expect": b"\x02",
            "skip": True,
        }

        result = tester.run_test(test_case)

        assert result.status == HILTestStatus.SKIPPED
        assert result.latency is None

        tester.teardown()

    def test_dry_run_test_suite(self) -> None:
        """Test running a suite of tests in dry-run mode."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)

        test_cases = [
            {"name": "test1", "send": b"\x01", "expect": b"\x01"},
            {"name": "test2", "send": b"\x02", "expect": b"\x02"},
            {"name": "test3", "send": b"\x03", "expect": b"\xff"},  # Will fail
            {"name": "test4", "send": b"\x04", "skip": True},
        ]

        with HILTester(config) as tester:
            report = tester.run_tests(test_cases)

        assert report.total == 4
        assert report.passed == 2
        assert report.failed == 1
        assert report.skipped == 1
        assert report.success_rate == 0.5
        assert "avg_latency" in report.timing_statistics

    def test_dry_run_timing_validation(self) -> None:
        """Test timing validation in dry-run mode."""
        config = HILConfig(
            interface="serial",
            port="/dev/null",
            dry_run=True,
            validate_timing=True,
            setup_delay=0.0,
        )
        tester = HILTester(config)
        tester.setup()

        # Test with unrealistic max_latency (should fail timing)
        test_case = {
            "name": "timing_test",
            "send": b"\x01",
            "expect": b"\x01",
            "max_latency": 0.000001,  # 1 microsecond - unrealistic
        }

        result = tester.run_test(test_case)

        # Might pass or fail depending on system speed, but should have timing info
        assert result.timing_valid is not None

        tester.teardown()

    def test_not_setup_raises(self) -> None:
        """Test that running test without setup raises error."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True)
        tester = HILTester(config)

        with pytest.raises(RuntimeError, match="Not setup"):
            tester.run_test({"name": "test", "send": b"\x01"})

    def test_double_setup_raises(self) -> None:
        """Test that calling setup twice raises error."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)

        tester.setup()

        with pytest.raises(RuntimeError, match="Already setup"):
            tester.setup()

        tester.teardown()


class TestHILTesterMockedSerial:
    """Test HILTester with mocked serial interface."""

    @patch("oscura.validation.hil_testing.serial")
    def test_serial_connection(self, mock_serial: Mock) -> None:
        """Test serial connection establishment."""
        mock_serial_instance = MagicMock()
        mock_serial.Serial.return_value = mock_serial_instance

        config = HILConfig(interface="serial", port="/dev/ttyUSB0", baud_rate=9600, setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        mock_serial.Serial.assert_called_once_with(port="/dev/ttyUSB0", baudrate=9600, timeout=1.0)

        tester.teardown()
        mock_serial_instance.close.assert_called_once()

    @patch("oscura.validation.hil_testing.serial")
    def test_serial_send_receive(self, mock_serial: Mock) -> None:
        """Test serial send/receive operation."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.read.return_value = b"\x02\x03"
        mock_serial.Serial.return_value = mock_serial_instance

        config = HILConfig(interface="serial", port="/dev/ttyUSB0", setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {"name": "serial_test", "send": b"\x01\x02", "expect": b"\x02\x03"}
        result = tester.run_test(test_case)

        assert result.status == HILTestStatus.PASSED
        mock_serial_instance.write.assert_called_once_with(b"\x01\x02")
        mock_serial_instance.read.assert_called_once()

        tester.teardown()

    @patch("oscura.validation.hil_testing.serial")
    def test_serial_timeout(self, mock_serial: Mock) -> None:
        """Test serial timeout handling."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.read.return_value = b""  # Empty response = timeout
        mock_serial.Serial.return_value = mock_serial_instance

        config = HILConfig(interface="serial", port="/dev/ttyUSB0", setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {"name": "timeout_test", "send": b"\x01", "expect": b"\x02"}
        result = tester.run_test(test_case)

        assert result.status == HILTestStatus.TIMEOUT
        assert result.received_data is None

        tester.teardown()


class TestHILTesterMockedSocketCAN:
    """Test HILTester with mocked SocketCAN interface."""

    @patch("oscura.validation.hil_testing.can")
    def test_socketcan_connection(self, mock_can: Mock) -> None:
        """Test SocketCAN connection establishment."""
        mock_bus = MagicMock()
        mock_can.interface.Bus.return_value = mock_bus

        config = HILConfig(interface="socketcan", port="can0", setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        mock_can.interface.Bus.assert_called_once_with(
            channel="can0", interface="socketcan", receive_own_messages=False
        )

        tester.teardown()

    @patch("oscura.validation.hil_testing.can")
    def test_socketcan_send_receive(self, mock_can: Mock) -> None:
        """Test SocketCAN send/receive operation."""
        mock_bus = MagicMock()
        mock_msg = MagicMock()
        mock_msg.data = [0x02, 0x03, 0x04]
        mock_bus.recv.return_value = mock_msg
        mock_can.interface.Bus.return_value = mock_bus
        mock_can.Message = MagicMock()

        config = HILConfig(interface="socketcan", port="can0", setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {"name": "can_test", "send": b"\x01\x02", "expect": b"\x02\x03\x04"}
        result = tester.run_test(test_case)

        assert result.status == HILTestStatus.PASSED
        mock_bus.send.assert_called_once()
        mock_bus.recv.assert_called_once()

        tester.teardown()


class TestHILTesterMockedUSB:
    """Test HILTester with mocked USB interface."""

    @patch("oscura.validation.hil_testing.usb.core")
    def test_usb_connection(self, mock_usb_core: Mock) -> None:
        """Test USB connection establishment."""
        mock_device = MagicMock()
        mock_usb_core.find.return_value = mock_device

        config = HILConfig(
            interface="usb",
            port=0,
            usb_vendor_id=0x1234,
            usb_product_id=0x5678,
            setup_delay=0.0,
        )
        tester = HILTester(config)
        tester.setup()

        mock_usb_core.find.assert_called_once_with(idVendor=0x1234, idProduct=0x5678)

        tester.teardown()

    @patch("oscura.validation.hil_testing.usb.core")
    def test_usb_device_not_found(self, mock_usb_core: Mock) -> None:
        """Test USB device not found error."""
        mock_usb_core.find.return_value = None

        config = HILConfig(
            interface="usb",
            port=0,
            usb_vendor_id=0x1234,
            usb_product_id=0x5678,
            setup_delay=0.0,
        )
        tester = HILTester(config)

        with pytest.raises(OSError, match="USB device not found"):
            tester.setup()


class TestHILTesterMockedSPI:
    """Test HILTester with mocked SPI interface."""

    @patch("oscura.validation.hil_testing.spidev")
    def test_spi_connection(self, mock_spidev: Mock) -> None:
        """Test SPI connection establishment."""
        mock_spi = MagicMock()
        mock_spidev.SpiDev.return_value = mock_spi

        config = HILConfig(
            interface="spi", port=0, spi_bus=0, spi_device=0, spi_speed_hz=1000000, setup_delay=0.0
        )
        tester = HILTester(config)
        tester.setup()

        mock_spi.open.assert_called_once_with(0, 0)
        assert mock_spi.max_speed_hz == 1000000

        tester.teardown()

    @patch("oscura.validation.hil_testing.spidev")
    def test_spi_send_receive(self, mock_spidev: Mock) -> None:
        """Test SPI send/receive (full-duplex)."""
        mock_spi = MagicMock()
        mock_spi.xfer2.return_value = [0x02, 0x03]
        mock_spidev.SpiDev.return_value = mock_spi

        config = HILConfig(interface="spi", port=0, setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {"name": "spi_test", "send": b"\x01\x02", "expect": b"\x02\x03"}
        result = tester.run_test(test_case)

        assert result.status == HILTestStatus.PASSED
        mock_spi.xfer2.assert_called_once_with([0x01, 0x02])

        tester.teardown()


class TestHILTesterMockedI2C:
    """Test HILTester with mocked I2C interface."""

    @patch("oscura.validation.hil_testing.SMBus")
    def test_i2c_connection(self, mock_smbus: Mock) -> None:
        """Test I2C connection establishment."""
        mock_bus = MagicMock()
        mock_smbus.return_value = mock_bus

        config = HILConfig(interface="i2c", port=1, i2c_bus=1, i2c_address=0x50, setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        mock_smbus.assert_called_once_with(1)

        tester.teardown()

    @patch("oscura.validation.hil_testing.SMBus")
    def test_i2c_send_receive(self, mock_smbus: Mock) -> None:
        """Test I2C send/receive operation."""
        mock_bus = MagicMock()
        mock_bus.read_byte.side_effect = [0x02, 0x03]
        mock_smbus.return_value = mock_bus

        config = HILConfig(interface="i2c", port=1, i2c_address=0x50, setup_delay=0.0)
        tester = HILTester(config)
        tester.setup()

        test_case = {"name": "i2c_test", "send": b"\x01\x02", "expect": b"\x02\x03"}
        result = tester.run_test(test_case)

        assert result.status == HILTestStatus.PASSED
        assert mock_bus.write_byte.call_count == 2
        assert mock_bus.read_byte.call_count == 2

        tester.teardown()


class TestHILTesterGPIO:
    """Test HILTester GPIO control (mocked)."""

    @patch("oscura.validation.hil_testing.GPIO")
    def test_gpio_reset(self, mock_gpio: Mock) -> None:
        """Test GPIO reset pulse."""
        config = HILConfig(
            interface="serial",
            port="/dev/null",
            dry_run=True,
            reset_gpio=17,
            reset_duration=0.01,
            setup_delay=0.0,
        )

        with patch("time.sleep"):  # Speed up test
            tester = HILTester(config)
            tester.setup()

            # Check GPIO was configured and pulsed
            mock_gpio.setup.assert_called()
            assert mock_gpio.output.call_count >= 2  # Low then high

            tester.teardown()
            mock_gpio.cleanup.assert_called_once()

    @patch("oscura.validation.hil_testing.GPIO")
    def test_gpio_power_control(self, mock_gpio: Mock) -> None:
        """Test GPIO power control."""
        config = HILConfig(
            interface="serial",
            port="/dev/null",
            dry_run=True,
            power_gpio=18,
            setup_delay=0.0,
            teardown_delay=0.0,
        )

        tester = HILTester(config)
        tester.setup()

        # Power should be on
        calls = [c for c in mock_gpio.output.call_args_list if c[0][0] == 18]
        assert any(c[0][1] is True for c in calls)

        tester.teardown()

        # Power should be off
        calls = [c for c in mock_gpio.output.call_args_list if c[0][0] == 18]
        assert any(c[0][1] is False for c in calls)


class TestHILTesterPCAP:
    """Test HILTester PCAP capture (mocked)."""

    @patch("oscura.validation.hil_testing.wrpcap")
    @patch("oscura.validation.hil_testing.IP")
    @patch("oscura.validation.hil_testing.UDP")
    def test_pcap_export(self, mock_udp: Mock, mock_ip: Mock, mock_wrpcap: Mock) -> None:
        """Test PCAP export of captured traffic."""
        config = HILConfig(
            interface="serial",
            port="/dev/null",
            dry_run=True,
            capture_pcap=True,
            pcap_file="test.pcap",
            setup_delay=0.0,
        )

        with HILTester(config) as tester:
            test_cases = [
                {"name": "test1", "send": b"\x01\x02", "expect": b"\x01\x02"},
                {"name": "test2", "send": b"\x03\x04", "expect": b"\x03\x04"},
            ]
            tester.run_tests(test_cases)

        # Should have written PCAP file
        mock_wrpcap.assert_called_once()
        args = mock_wrpcap.call_args
        assert args[0][0] == "test.pcap"  # Filename
        assert len(args[0][1]) == 4  # 2 tests * 2 packets (send + receive)


class TestBitErrorCounting:
    """Test bit error counting functionality."""

    def test_count_bit_errors_perfect_match(self) -> None:
        """Test bit error counting with perfect match."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)

        errors = tester._count_bit_errors(b"\x01\x02\x03", b"\x01\x02\x03")
        assert errors == 0

    def test_count_bit_errors_single_bit(self) -> None:
        """Test bit error counting with single bit error."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)

        # 0x01 vs 0x03 differs by 1 bit (bit 1)
        errors = tester._count_bit_errors(b"\x01", b"\x03")
        assert errors == 1

    def test_count_bit_errors_multiple_bits(self) -> None:
        """Test bit error counting with multiple bit errors."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)

        # 0xFF vs 0x00 differs by 8 bits
        errors = tester._count_bit_errors(b"\xff", b"\x00")
        assert errors == 8

    def test_count_bit_errors_different_lengths(self) -> None:
        """Test bit error counting with different length data."""
        config = HILConfig(interface="serial", port="/dev/null", dry_run=True, setup_delay=0.0)
        tester = HILTester(config)

        # Shorter data is padded with zeros
        errors = tester._count_bit_errors(b"\x01", b"\x01\x02")
        assert errors > 0  # Padding mismatch


class TestImportErrors:
    """Test graceful handling when libraries are unavailable."""

    def test_serial_import_error(self) -> None:
        """Test ImportError when pyserial is not available."""
        config = HILConfig(interface="serial", port="/dev/ttyUSB0", setup_delay=0.0)
        tester = HILTester(config)

        with patch("oscura.validation.hil_testing.serial", None):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="pyserial is required"):
                    tester.setup()

    def test_socketcan_import_error(self) -> None:
        """Test ImportError when python-can is not available."""
        config = HILConfig(interface="socketcan", port="can0", setup_delay=0.0)
        tester = HILTester(config)

        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError, match="python-can is required"):
                tester.setup()

    def test_usb_import_error(self) -> None:
        """Test ImportError when pyusb is not available."""
        config = HILConfig(
            interface="usb", port=0, usb_vendor_id=0x1234, usb_product_id=0x5678, setup_delay=0.0
        )
        tester = HILTester(config)

        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError, match="pyusb is required"):
                tester.setup()

    def test_spi_import_error(self) -> None:
        """Test ImportError when spidev is not available."""
        config = HILConfig(interface="spi", port=0, setup_delay=0.0)
        tester = HILTester(config)

        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError, match="spidev is required"):
                tester.setup()

    def test_i2c_import_error(self) -> None:
        """Test ImportError when smbus2 is not available."""
        config = HILConfig(interface="i2c", port=1, setup_delay=0.0)
        tester = HILTester(config)

        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError, match="smbus2 is required"):
                tester.setup()

    def test_gpio_import_error(self) -> None:
        """Test ImportError when no GPIO library is available."""
        config = HILConfig(
            interface="serial", port="/dev/null", dry_run=True, reset_gpio=17, setup_delay=0.0
        )
        tester = HILTester(config)

        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError, match="No GPIO library available"):
                tester.setup()
