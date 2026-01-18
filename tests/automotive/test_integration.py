"""Integration tests for automotive CAN reverse engineering workflows.

This module tests end-to-end workflows for CAN bus analysis including:
- Discovery workflow (analyze -> test hypotheses -> document)
- DBC round-trip (document -> generate DBC -> load DBC -> decode)
- Multiple format loading (CSV -> analyze -> export)
- Signal correlation analysis
- Performance with large datasets

All tests use synthetic data only, following Oscura's testing standards.
Large dataset tests are marked with @pytest.mark.slow and require --runslow.

Test coverage:
- 9 integration tests
- 5 workflow scenarios
- 11,000+ synthetic CAN messages in performance tests
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import struct
import time

import pytest

from oscura.automotive.can.correlation import CorrelationAnalyzer
from oscura.automotive.can.discovery import (
    DiscoveryDocument,
    Hypothesis,
    MessageDiscovery,
    SignalDiscovery,
)
from oscura.automotive.can.models import CANMessage, CANMessageList, SignalDefinition
from oscura.automotive.can.session import CANSession
from oscura.automotive.dbc.generator import DBCGenerator

# Conditional imports for optional dependencies
try:
    from oscura.automotive.dbc.parser import load_dbc

    DBC_AVAILABLE = True
except ImportError:
    DBC_AVAILABLE = False

try:
    from oscura.automotive.loaders.csv_can import load_csv_can

    CSV_AVAILABLE = True
except ImportError:
    CSV_AVAILABLE = False


@pytest.mark.integration
class TestDiscoveryWorkflow:
    """Test complete discovery workflow from raw messages to documentation."""

    def test_discovery_workflow(self, tmp_path):
        """Test complete discovery workflow.

        Workflow:
        1. Generate synthetic CAN messages
        2. Create CANSession
        3. Analyze messages
        4. Test hypotheses
        5. Document signals
        6. Save to .tkcan
        7. Reload and verify
        """
        # Step 1: Generate synthetic CAN messages with known signals
        messages = CANMessageList()

        # Message 0x280: Engine RPM (bytes 2-3, big-endian, scale 0.25)
        for i in range(100):
            timestamp = i * 0.01
            rpm = 1000 + (3000 * i / 99)  # 1000 to 4000 RPM
            raw_rpm = int(rpm / 0.25)

            data = bytearray(8)
            data[0] = 0xAA  # Constant
            data[1] = 0xBB  # Constant
            data[2:4] = struct.pack(">H", raw_rpm)  # RPM
            data[4] = i % 256  # Counter
            data[5:8] = b"\x00\x00\x00"

            messages.append(
                CANMessage(
                    arbitration_id=0x280,
                    timestamp=timestamp,
                    data=bytes(data),
                    is_extended=False,
                )
            )

        # Message 0x300: Vehicle speed (bytes 0-1, big-endian, scale 0.01 km/h)
        for i in range(100):
            timestamp = i * 0.01
            speed_kmh = 50 + (100 * i / 99)  # 50 to 150 km/h
            raw_speed = int(speed_kmh / 0.01)

            data = bytearray(8)
            data[0:2] = struct.pack(">H", raw_speed)
            data[2:8] = b"\xff\xff\xff\xff\xff\xff"

            messages.append(
                CANMessage(
                    arbitration_id=0x300,
                    timestamp=timestamp,
                    data=bytes(data),
                    is_extended=False,
                )
            )

        # Step 2: Create CANSession
        session = CANSession(messages=messages)

        assert len(session) == 200
        assert len(session.unique_ids()) == 2

        # Step 3: Analyze messages
        inventory = session.inventory()
        assert len(inventory) == 2

        # Analyze specific message
        rpm_analysis = session.analyze_message(0x280)
        assert rpm_analysis.arbitration_id == 0x280
        assert rpm_analysis.message_count == 100
        assert len(rpm_analysis.byte_analyses) == 8

        # Verify counter detection in byte 4
        byte4_analysis = rpm_analysis.byte_analyses[4]
        # Counter byte should have high change rate or high entropy
        assert byte4_analysis.change_rate > 0.8 or byte4_analysis.entropy > 5.0

        # Step 4: Test hypotheses
        rpm_msg = session.message(0x280)

        # Test RPM hypothesis
        rpm_hypothesis = rpm_msg.test_hypothesis(
            signal_name="engine_rpm",
            start_byte=2,
            bit_length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            expected_min=800,
            expected_max=8000,
        )

        assert len(rpm_hypothesis.values) == 100
        assert 950 < rpm_hypothesis.min_value < 1050  # Should be ~1000
        assert 3950 < rpm_hypothesis.max_value < 4050  # Should be ~4000
        assert rpm_hypothesis.confidence > 0.9

        # Step 5: Document signals
        rpm_msg.document_signal(
            name="engine_rpm",
            start_bit=16,  # Byte 2
            length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            comment="Engine RPM from ECU",
        )

        speed_msg = session.message(0x300)
        speed_msg.document_signal(
            name="vehicle_speed",
            start_bit=0,  # Byte 0
            length=16,
            byte_order="big_endian",
            scale=0.01,
            unit="km/h",
            comment="Vehicle speed from ABS module",
        )

        # Verify signals documented
        rpm_signals = rpm_msg.get_documented_signals()
        assert "engine_rpm" in rpm_signals
        assert rpm_signals["engine_rpm"].scale == 0.25

        # Step 6: Save to .tkcan
        doc = DiscoveryDocument()
        doc.vehicle.make = "TestVehicle"
        doc.vehicle.model = "TestModel"
        doc.vehicle.year = "2024"

        # Add messages with signals
        rpm_sig = SignalDiscovery(
            name="engine_rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            min_value=rpm_hypothesis.min_value,
            max_value=rpm_hypothesis.max_value,
            confidence=rpm_hypothesis.confidence,
            evidence=["Statistical analysis", "Hypothesis validation"],
            comment="Engine RPM from ECU",
        )

        rpm_msg_discovery = MessageDiscovery(
            id=0x280,
            name="Engine_Status",
            length=8,
            cycle_time_ms=10.0,
            confidence=0.95,
            signals=[rpm_sig],
        )

        doc.add_message(rpm_msg_discovery)

        # Add hypothesis for future testing
        doc.add_hypothesis(
            Hypothesis(
                message_id=0x300,
                signal="vehicle_speed",
                hypothesis="Vehicle speed from ABS",
                status="confirmed",
                test_plan="Validated against known values",
            )
        )

        tkcan_path = tmp_path / "discoveries.tkcan"
        doc.save(tkcan_path)

        assert tkcan_path.exists()

        # Step 7: Reload and verify
        loaded_doc = DiscoveryDocument.load(tkcan_path)

        assert loaded_doc.vehicle.make == "TestVehicle"
        assert loaded_doc.vehicle.model == "TestModel"
        assert 0x280 in loaded_doc.messages
        assert loaded_doc.messages[0x280].name == "Engine_Status"
        assert len(loaded_doc.messages[0x280].signals) == 1
        assert loaded_doc.messages[0x280].signals[0].name == "engine_rpm"
        assert loaded_doc.messages[0x280].signals[0].scale == 0.25
        assert len(loaded_doc.hypotheses) == 1
        assert loaded_doc.hypotheses[0].message_id == 0x300
        assert loaded_doc.hypotheses[0].status == "confirmed"


@pytest.mark.integration
class TestDBCRoundtrip:
    """Test DBC file generation and round-trip."""

    def test_dbc_roundtrip(self, tmp_path):
        """Test DBC round-trip workflow.

        Workflow:
        1. Create session with documented signals
        2. Generate DBC file
        3. Load DBC file
        4. Decode messages using DBC
        5. Verify decoded values match expectations
        """
        # Step 1: Create session with documented signals
        messages = CANMessageList()

        # Generate messages with known signal values
        test_values = []
        for i in range(50):
            timestamp = i * 0.01
            rpm = 2000.0  # Constant RPM for easy verification
            speed = 100.0  # Constant speed

            # RPM in bytes 2-3, big-endian, scale 0.25
            raw_rpm = int(rpm / 0.25)
            # Speed in bytes 0-1, big-endian, scale 0.01
            raw_speed = int(speed / 0.01)

            # Message 0x280: Engine data
            rpm_data = bytearray(8)
            rpm_data[2:4] = struct.pack(">H", raw_rpm)
            messages.append(
                CANMessage(
                    arbitration_id=0x280,
                    timestamp=timestamp,
                    data=bytes(rpm_data),
                )
            )

            # Message 0x300: Speed data
            speed_data = bytearray(8)
            speed_data[0:2] = struct.pack(">H", raw_speed)
            messages.append(
                CANMessage(
                    arbitration_id=0x300,
                    timestamp=timestamp,
                    data=bytes(speed_data),
                )
            )

            test_values.append({"rpm": rpm, "speed": speed})

        session = CANSession(messages=messages)

        # Document signals
        # Note: We need to use a discovery document approach since
        # message wrappers don't persist documented signals across sessions
        doc = DiscoveryDocument()

        # Add RPM message with signal
        # Note: For DBC big-endian, start_bit is the MSB position
        # Bytes 2-3 = bits 16-31, so MSB is at bit 23
        rpm_sig = SignalDiscovery(
            name="engine_rpm",
            start_bit=23,  # MSB of 16-bit big-endian value at bytes 2-3
            length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            min_value=1900.0,
            max_value=2100.0,
            confidence=1.0,
        )

        rpm_msg_discovery = MessageDiscovery(
            id=0x280,
            name="Message_280",
            length=8,
            cycle_time_ms=10.0,
            signals=[rpm_sig],
            confidence=1.0,
        )

        doc.add_message(rpm_msg_discovery)

        # Add speed message with signal
        # Bytes 0-1 = bits 0-15, so MSB is at bit 7
        speed_sig = SignalDiscovery(
            name="vehicle_speed",
            start_bit=7,  # MSB of 16-bit big-endian value at bytes 0-1
            length=16,
            byte_order="big_endian",
            scale=0.01,
            unit="km/h",
            min_value=90.0,
            max_value=110.0,
            confidence=1.0,
        )

        speed_msg_discovery = MessageDiscovery(
            id=0x300,
            name="Message_300",
            length=8,
            cycle_time_ms=20.0,
            signals=[speed_sig],
            confidence=1.0,
        )

        doc.add_message(speed_msg_discovery)

        # Step 2: Generate DBC file from discovery document
        dbc_path = tmp_path / "test_vehicle.dbc"
        DBCGenerator.generate(doc, dbc_path)

        assert dbc_path.exists()
        assert dbc_path.stat().st_size > 0

        # Verify DBC content
        dbc_content = dbc_path.read_text()

        assert "BO_ 640 Message_280" in dbc_content  # 0x280 = 640
        assert "engine_rpm" in dbc_content
        assert "vehicle_speed" in dbc_content

        # Step 3: Load DBC file
        dbc = load_dbc(dbc_path)

        assert 0x280 in dbc.get_message_ids()
        assert 0x300 in dbc.get_message_ids()

        # Step 4 & 5: Decode messages and verify values
        rpm_messages = messages.filter_by_id(0x280)
        speed_messages = messages.filter_by_id(0x300)

        for msg in rpm_messages.messages[:10]:
            decoded = dbc.decode_message(msg)
            assert "engine_rpm" in decoded
            assert decoded["engine_rpm"].value == pytest.approx(2000.0, abs=1.0)
            assert decoded["engine_rpm"].unit == "rpm"

        for msg in speed_messages.messages[:10]:
            decoded = dbc.decode_message(msg)
            assert "vehicle_speed" in decoded
            assert decoded["vehicle_speed"].value == pytest.approx(100.0, abs=0.1)
            assert decoded["vehicle_speed"].unit == "km/h"


@pytest.mark.integration
class TestMultipleFormatLoading:
    """Test loading from different file formats."""

    def test_multiple_format_loading(self, tmp_path):
        """Test loading CAN data from CSV format.

        Workflow:
        1. Create CSV with CAN data
        2. Load via CSV loader
        3. Analyze
        4. Export to DBC
        5. Verify results
        """
        # Step 1: Create CSV with CAN data
        csv_path = tmp_path / "can_data.csv"
        csv_content = "timestamp,id,data\n"

        for i in range(100):
            timestamp = i * 0.01
            rpm = 1500 + (i * 10)
            raw_rpm = int(rpm / 0.25)

            data_bytes = bytearray(8)
            data_bytes[2:4] = struct.pack(">H", raw_rpm)
            data_hex = data_bytes.hex()

            csv_content += f"{timestamp:.6f},0x280,{data_hex}\n"

        csv_path.write_text(csv_content)

        # Step 2: Load via CSV loader
        messages = load_csv_can(csv_path)

        assert len(messages) == 100
        assert 0x280 in messages.unique_ids()

        # Step 3: Analyze
        session = CANSession(messages=messages)
        analysis = session.analyze_message(0x280)

        assert analysis.message_count == 100
        assert len(analysis.byte_analyses) == 8

        # Step 4: Export to DBC (if available)
        if DBC_AVAILABLE:
            # Create discovery document
            doc = DiscoveryDocument()

            rpm_sig = SignalDiscovery(
                name="engine_rpm",
                start_bit=23,  # MSB for big-endian at bytes 2-3
                length=16,
                byte_order="big_endian",
                scale=0.25,
                unit="rpm",
                confidence=1.0,
            )

            msg_discovery = MessageDiscovery(
                id=0x280,
                name="Message_280",
                length=8,
                signals=[rpm_sig],
                confidence=1.0,
            )

            doc.add_message(msg_discovery)

            dbc_path = tmp_path / "from_csv.dbc"
            DBCGenerator.generate(doc, dbc_path)

            assert dbc_path.exists()

            # Step 5: Verify results
            dbc = load_dbc(dbc_path)
            decoded = dbc.decode_message(messages.messages[0])

            assert "engine_rpm" in decoded
            # First message: rpm = 1500
            assert decoded["engine_rpm"].value == pytest.approx(1500.0, abs=1.0)


@pytest.mark.integration
class TestCorrelationWorkflow:
    """Test signal correlation analysis workflow."""

    def test_correlation_workflow(self):
        """Test signal correlation detection.

        Workflow:
        1. Create messages with correlated signals
        2. Detect correlation
        3. Validate correlation coefficient
        4. Test with OBD-II correlation use case
        """
        # Step 1: Create messages with correlated signals
        messages = CANMessageList()

        # Two messages with correlated data:
        # 0x280: Engine RPM
        # 0x300: Throttle position (correlated with RPM)
        for i in range(100):
            timestamp = i * 0.01
            rpm = 1000 + (i * 30)  # Increasing RPM

            # Throttle roughly correlates with RPM
            throttle_pct = 20 + (i * 0.6)  # 20% to 80%

            # Message 0x280: RPM in bytes 2-3
            raw_rpm = int(rpm / 0.25)
            rpm_data = bytearray(8)
            rpm_data[2:4] = struct.pack(">H", raw_rpm)
            messages.append(
                CANMessage(
                    arbitration_id=0x280,
                    timestamp=timestamp,
                    data=bytes(rpm_data),
                )
            )

            # Message 0x300: Throttle in byte 0
            raw_throttle = int(throttle_pct / 0.4)
            throttle_data = bytearray(8)
            throttle_data[0] = raw_throttle & 0xFF
            messages.append(
                CANMessage(
                    arbitration_id=0x300,
                    timestamp=timestamp,
                    data=bytes(throttle_data),
                )
            )

        session = CANSession(messages=messages)

        # Step 2: Detect correlation between RPM and throttle
        rpm_messages = session._messages.filter_by_id(0x280)
        throttle_messages = session._messages.filter_by_id(0x300)

        rpm_signal = SignalDefinition(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            scale=0.25,
        )

        throttle_signal = SignalDefinition(
            name="throttle",
            start_bit=0,
            length=8,
            byte_order="big_endian",
            scale=0.4,
        )

        # Step 3: Validate correlation coefficient
        correlation_result = CorrelationAnalyzer.correlate_signals(
            rpm_messages, rpm_signal, throttle_messages, throttle_signal
        )

        assert correlation_result["correlation"] > 0.95  # Strong positive correlation
        assert correlation_result["p_value"] < 0.01  # Statistically significant
        assert correlation_result["sample_count"] == 100

        # Step 4: Test message-level correlation finder
        correlations = CorrelationAnalyzer.find_correlated_messages(
            session, arbitration_id=0x280, threshold=0.7
        )

        # Should find 0x300 as correlated
        assert len(correlations) > 0

    def test_obd_correlation_use_case(self):
        """Test OBD-II correlation use case.

        Scenario: Vehicle has OBD-II responses (0x7E8) that should correlate
        with proprietary messages (0x280) for the same parameter.
        """
        messages = CANMessageList()

        # Simulate OBD-II PID 0x0C (Engine RPM) response
        # OBD-II format: [0x04, 0x41, 0x0C, RPM_HIGH, RPM_LOW, ...]
        # RPM = ((A*256)+B)/4

        # Proprietary message with same RPM
        for i in range(50):
            timestamp = i * 0.02
            rpm = 1500 + (i * 40)  # 1500 to 3460 RPM

            # OBD-II response (0x7E8)
            obd_rpm_raw = int(rpm * 4)
            rpm_high = (obd_rpm_raw >> 8) & 0xFF
            rpm_low = obd_rpm_raw & 0xFF

            obd_data = bytes([0x04, 0x41, 0x0C, rpm_high, rpm_low, 0x00, 0x00, 0x00])
            messages.append(
                CANMessage(
                    arbitration_id=0x7E8,
                    timestamp=timestamp,
                    data=obd_data,
                )
            )

            # Proprietary message (0x280)
            prop_rpm_raw = int(rpm / 0.25)
            prop_data = bytearray(8)
            prop_data[2:4] = struct.pack(">H", prop_rpm_raw)
            messages.append(
                CANMessage(
                    arbitration_id=0x280,
                    timestamp=timestamp,
                    data=bytes(prop_data),
                )
            )

        session = CANSession(messages=messages)

        # Define signals
        obd_signal = SignalDefinition(
            name="obd_rpm",
            start_bit=24,  # Byte 3
            length=16,
            byte_order="big_endian",
            scale=0.25,  # (value)/4 = value * 0.25
        )

        prop_signal = SignalDefinition(
            name="prop_rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            scale=0.25,
        )

        # Check correlation
        obd_messages = session._messages.filter_by_id(0x7E8)
        prop_messages = session._messages.filter_by_id(0x280)

        correlation_result = CorrelationAnalyzer.correlate_signals(
            obd_messages, obd_signal, prop_messages, prop_signal
        )

        # Should be perfectly correlated
        assert correlation_result["correlation"] > 0.99
        assert correlation_result["p_value"] < 0.001


@pytest.mark.integration
@pytest.mark.slow
class TestLargeDataset:
    """Test performance with large datasets."""

    def test_large_dataset(self):
        """Test analysis performance with large dataset.

        Tests:
        1. Generate 10,000+ messages
        2. Measure analysis time
        3. Verify memory usage reasonable
        4. Test analysis caching
        """
        # Step 1: Generate 10,000+ messages (multiple IDs)
        messages = CANMessageList()
        message_counts = {0x100: 3000, 0x200: 3000, 0x280: 2500, 0x300: 2500}

        for arb_id, count in message_counts.items():
            for i in range(count):
                timestamp = i * 0.001  # 1 kHz
                data = bytearray(8)

                # Add some pattern
                data[0] = i % 256  # Counter
                data[1] = (i // 256) % 256
                data[2:4] = struct.pack(">H", (i * 7) % 65536)
                data[4:8] = struct.pack(">I", i)

                messages.append(
                    CANMessage(
                        arbitration_id=arb_id,
                        timestamp=timestamp,
                        data=bytes(data),
                    )
                )

        assert len(messages) == 11000

        # Step 2: Measure analysis time
        session = CANSession(messages=messages)

        start_time = time.time()
        inventory = session.inventory()
        inventory_time = time.time() - start_time

        # Inventory should be fast (< 1 second for 11k messages)
        assert inventory_time < 1.0
        assert len(inventory) == 4

        # Analyze each message
        start_time = time.time()
        analyses = {}
        for arb_id in [0x100, 0x200, 0x280, 0x300]:
            analyses[arb_id] = session.analyze_message(arb_id)
        analysis_time = time.time() - start_time

        # Analysis should be reasonable (< 5 seconds for 4 messages)
        assert analysis_time < 5.0

        # Verify analysis results
        for arb_id, analysis in analyses.items():
            assert analysis.message_count == message_counts[arb_id]
            assert len(analysis.byte_analyses) == 8

        # Step 3: Test analysis caching
        start_time = time.time()
        cached_analysis = session.analyze_message(0x280)
        cache_time = time.time() - start_time

        # Cached access should be very fast (< 1ms)
        assert cache_time < 0.001
        assert cached_analysis is analyses[0x280]  # Same object

        # Force refresh should re-analyze
        start_time = time.time()
        refreshed = session.analyze_message(0x280, force_refresh=True)
        refresh_time = time.time() - start_time

        assert refresh_time > cache_time  # Slower than cached
        assert refreshed is not cached_analysis  # New object

    def test_large_dataset_filtering(self):
        """Test filtering performance on large dataset."""
        # Generate diverse message set
        messages = CANMessageList()

        for arb_id in range(0x100, 0x200):  # 256 different IDs
            for i in range(20):  # 20 messages each
                timestamp = (arb_id - 0x100) * 0.1 + i * 0.005
                data = bytes([arb_id & 0xFF, i] + [0] * 6)
                messages.append(CANMessage(arbitration_id=arb_id, timestamp=timestamp, data=data))

        session = CANSession(messages=messages)
        assert len(session) == 256 * 20  # 5,120 messages

        # Test filtering by IDs
        start_time = time.time()
        filtered = session.filter(arbitration_ids=[0x120, 0x130, 0x140])
        filter_time = time.time() - start_time

        assert filter_time < 0.5
        assert len(filtered.unique_ids()) == 3
        assert len(filtered) == 60  # 3 IDs * 20 messages

        # Test filtering by time range
        start_time = time.time()
        time_filtered = session.filter(time_range=(5.0, 10.0))
        time_filter_time = time.time() - start_time

        assert time_filter_time < 0.5
        time_start, time_end = time_filtered.time_range()
        assert time_start >= 5.0
        assert time_end <= 10.0

    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with large dataset."""
        # Generate minimal messages to test memory layout
        messages = CANMessageList()

        # Create 5,000 messages
        for i in range(5000):
            messages.append(
                CANMessage(
                    arbitration_id=0x123,
                    timestamp=i * 0.001,
                    data=bytes([i % 256] * 8),
                )
            )

        session = CANSession(messages=messages)

        # Should be able to access messages efficiently
        assert len(session) == 5000

        # Filter should not duplicate all data
        filtered = session.filter(time_range=(1.0, 2.0))
        assert len(filtered) > 0
        assert len(filtered) < 5000


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    def test_unknown_vehicle_reverse_engineering(self, tmp_path):
        """Test complete unknown vehicle reverse engineering scenario.

        Scenario: Reverse engineer an unknown vehicle's CAN bus.
        """
        # Capture synthetic data from "unknown" vehicle
        messages = CANMessageList()

        # Simulate realistic automotive CAN traffic
        for i in range(200):
            timestamp = i * 0.01

            # High-frequency message (100 Hz): Engine data
            rpm = 800 + (3200 * (i / 200))  # Idle to 4000 RPM
            throttle = 0 + (80 * (i / 200))  # 0% to 80%

            engine_data = bytearray(8)
            engine_data[0:2] = struct.pack(">H", int(rpm / 0.25))
            engine_data[2] = int(throttle / 0.4) & 0xFF
            engine_data[3] = i % 256  # Counter
            messages.append(
                CANMessage(arbitration_id=0x280, timestamp=timestamp, data=bytes(engine_data))
            )

            # Medium-frequency message (50 Hz): Vehicle speed, every other iteration
            if i % 2 == 0:
                speed = 0 + (120 * (i / 200))  # 0 to 120 km/h
                speed_data = bytearray(8)
                speed_data[0:2] = struct.pack(">H", int(speed / 0.01))
                messages.append(
                    CANMessage(arbitration_id=0x300, timestamp=timestamp, data=bytes(speed_data))
                )

        # Discovery workflow
        session = CANSession(messages=messages)

        # Step 1: Inventory
        inventory = session.inventory()
        assert len(inventory) == 2

        # Step 2: Analyze highest-frequency message first
        analysis_280 = session.analyze_message(0x280)
        assert analysis_280.frequency_hz == pytest.approx(100.0, abs=5.0)

        # Step 3: Test hypotheses for common automotive signals
        msg_280 = session.message(0x280)

        # Try RPM in different positions
        rpm_result = msg_280.test_hypothesis(
            signal_name="rpm",
            start_byte=0,
            bit_length=16,
            byte_order="big_endian",
            scale=0.25,
            expected_min=500,
            expected_max=8000,
        )

        # Should find valid RPM values
        assert rpm_result.min_value > 500
        assert rpm_result.max_value < 8000
        assert rpm_result.confidence > 0.8

        # Step 4: Document discoveries
        msg_280.document_signal(
            name="engine_rpm",
            start_bit=0,
            length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            comment="Discovered via statistical analysis",
        )

        # Step 5: Save findings
        doc = DiscoveryDocument()
        doc.vehicle.make = "Unknown"
        doc.vehicle.notes = "Reverse engineered from CAN capture"

        rpm_sig = SignalDiscovery.from_definition(
            msg_280.get_documented_signals()["engine_rpm"],
            confidence=rpm_result.confidence,
            evidence=["Statistical pattern analysis", "Value range validation"],
        )

        msg_discovery = MessageDiscovery(
            id=0x280,
            name="Engine_ECU",
            length=8,
            cycle_time_ms=10.0,
            signals=[rpm_sig],
        )

        doc.add_message(msg_discovery)

        # Save and verify
        save_path = tmp_path / "unknown_vehicle.tkcan"
        doc.save(save_path)

        # Reload
        loaded = DiscoveryDocument.load(save_path)
        assert len(loaded.messages) == 1
        assert loaded.messages[0x280].signals[0].name == "engine_rpm"
