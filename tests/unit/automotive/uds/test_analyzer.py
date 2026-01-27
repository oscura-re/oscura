"""Unit tests for UDS analyzer."""

from pathlib import Path

import pytest

from oscura.automotive.uds.analyzer import UDSECU, UDSAnalyzer, UDSMessage


class TestUDSMessage:
    """Test UDSMessage dataclass."""

    def test_request_message(self) -> None:
        """Test request message representation."""
        msg = UDSMessage(
            timestamp=1.0,
            service_id=0x10,
            service_name="DiagnosticSessionControl",
            is_response=False,
            sub_function=0x03,
        )
        assert msg.timestamp == 1.0
        assert msg.service_id == 0x10
        assert not msg.is_response
        assert msg.sub_function == 0x03

    def test_response_message(self) -> None:
        """Test response message representation."""
        msg = UDSMessage(
            timestamp=1.5,
            service_id=0x10,
            service_name="DiagnosticSessionControl",
            is_response=True,
            sub_function=0x03,
        )
        assert msg.is_response
        assert "Response" in repr(msg)

    def test_negative_response(self) -> None:
        """Test negative response representation."""
        msg = UDSMessage(
            timestamp=2.0,
            service_id=0x10,
            service_name="DiagnosticSessionControl",
            is_response=True,
            negative_response_code=0x33,
        )
        assert msg.negative_response_code == 0x33
        assert "NRC" in repr(msg)


class TestUDSECU:
    """Test UDSECU dataclass."""

    def test_default_state(self) -> None:
        """Test default ECU state."""
        ecu = UDSECU(ecu_id="ECU1")
        assert ecu.ecu_id == "ECU1"
        assert ecu.current_session == 0x01  # Default session
        assert ecu.security_level == 0  # Locked
        assert len(ecu.supported_services) == 0
        assert len(ecu.dtcs) == 0

    def test_state_updates(self) -> None:
        """Test ECU state updates."""
        ecu = UDSECU(ecu_id="ECU1")
        ecu.supported_services.add(0x10)
        ecu.supported_services.add(0x27)
        ecu.current_session = 0x03
        ecu.security_level = 1

        assert 0x10 in ecu.supported_services
        assert 0x27 in ecu.supported_services
        assert ecu.current_session == 0x03
        assert ecu.security_level == 1


class TestUDSAnalyzer:
    """Test UDSAnalyzer."""

    def test_init(self) -> None:
        """Test analyzer initialization."""
        analyzer = UDSAnalyzer()
        assert len(analyzer.messages) == 0
        assert len(analyzer.ecus) == 0

    def test_parse_empty_message(self) -> None:
        """Test parsing empty message raises error."""
        analyzer = UDSAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.parse_message(b"")

    def test_parse_diagnostic_session_control_request(self) -> None:
        """Test parsing diagnostic session control request."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x10, 0x03]), timestamp=1.0)

        assert msg.service_id == 0x10
        assert msg.service_name == "DiagnosticSessionControl"
        assert not msg.is_response
        assert msg.sub_function == 0x03
        assert msg.decoded["session_type"] == "ExtendedDiagnosticSession"

    def test_parse_diagnostic_session_control_response(self) -> None:
        """Test parsing diagnostic session control response."""
        analyzer = UDSAnalyzer()
        # Response: SID+0x40, sub-function, P2_server_max (2 bytes), P2*_server_max (2 bytes)
        data = bytes([0x50, 0x03, 0x00, 0x32, 0x01, 0xF4])
        msg = analyzer.parse_message(data, timestamp=1.1)

        assert msg.service_id == 0x10
        assert msg.is_response
        assert msg.sub_function == 0x03
        assert msg.decoded["p2_server_max_ms"] == 0x0032  # 50ms
        assert msg.decoded["p2_star_server_max_ms"] == 0x01F4  # 500ms

    def test_parse_ecu_reset_request(self) -> None:
        """Test parsing ECU reset request."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x11, 0x01]), timestamp=2.0)

        assert msg.service_id == 0x11
        assert msg.service_name == "ECUReset"
        assert msg.sub_function == 0x01
        assert msg.decoded["reset_type"] == "hardReset"

    def test_parse_negative_response(self) -> None:
        """Test parsing negative response."""
        analyzer = UDSAnalyzer()
        # Negative response: 0x7F, requested_SID, NRC
        msg = analyzer.parse_message(bytes([0x7F, 0x10, 0x33]), timestamp=1.5)

        assert msg.service_id == 0x10
        assert msg.is_response
        assert msg.negative_response_code == 0x33
        assert msg.decoded["nrc_name"] == "SecurityAccessDenied"

    def test_parse_negative_response_too_short(self) -> None:
        """Test parsing negative response that's too short."""
        analyzer = UDSAnalyzer()
        with pytest.raises(ValueError, match="too short"):
            analyzer.parse_message(bytes([0x7F, 0x10]))

    def test_parse_security_access_request_seed(self) -> None:
        """Test parsing security access request seed."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x27, 0x01]), timestamp=3.0)

        assert msg.service_id == 0x27
        assert msg.service_name == "SecurityAccess"
        assert msg.sub_function == 0x01
        assert msg.decoded["access_type"] == "requestSeed"
        assert msg.decoded["security_level"] == 1

    def test_parse_security_access_response_seed(self) -> None:
        """Test parsing security access response with seed."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x67, 0x01, 0x12, 0x34, 0x56, 0x78]), timestamp=3.1)

        assert msg.service_id == 0x27
        assert msg.is_response
        assert msg.decoded["access_type"] == "requestSeed"
        assert msg.decoded["seed"] == "12345678"

    def test_parse_security_access_send_key(self) -> None:
        """Test parsing security access send key."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x27, 0x02, 0xAB, 0xCD, 0xEF, 0x01]), timestamp=3.2)

        assert msg.service_id == 0x27
        assert msg.sub_function == 0x02
        assert msg.decoded["access_type"] == "sendKey"
        assert msg.decoded["security_level"] == 1
        assert msg.decoded["key"] == "abcdef01"

    def test_parse_security_access_suppress_response(self) -> None:
        """Test parsing security access with suppress response bit."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x27, 0x81]), timestamp=3.3)

        assert msg.sub_function == 0x01
        assert msg.decoded["suppress_positive_response"]

    def test_parse_read_dtc_request(self) -> None:
        """Test parsing read DTC information request."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x19, 0x02, 0xFF]), timestamp=4.0)

        assert msg.service_id == 0x19
        assert msg.service_name == "ReadDTCInformation"
        assert msg.sub_function == 0x02

    def test_parse_read_dtc_response(self) -> None:
        """Test parsing read DTC information response."""
        analyzer = UDSAnalyzer()
        # Response: SID+0x40, sub-function, availability_mask, DTCs...
        # DTC format: 3 bytes DTC code + 1 byte status
        data = bytes(
            [
                0x59,
                0x02,
                0xFF,  # Availability mask
                0xC1,
                0x23,
                0x45,
                0x0F,  # DTC C12345, status 0x0F
                0xD4,
                0x56,
                0x78,
                0x28,  # DTC D45678, status 0x28
            ]
        )
        msg = analyzer.parse_message(data, timestamp=4.1)

        assert msg.service_id == 0x19
        assert msg.is_response
        assert msg.decoded["dtc_count"] == 2

        dtcs = msg.decoded["dtcs"]
        assert len(dtcs) == 2
        assert dtcs[0]["dtc"] == "C12345"
        assert dtcs[0]["status"] == 0x0F
        assert dtcs[0]["test_failed"]
        assert dtcs[0]["confirmed"]
        assert dtcs[1]["dtc"] == "D45678"
        assert dtcs[1]["status"] == 0x28

    def test_parse_read_data_by_id_request(self) -> None:
        """Test parsing read data by identifier request."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x22, 0xF1, 0x90, 0xF1, 0x91]), timestamp=5.0)

        assert msg.service_id == 0x22
        assert msg.service_name == "ReadDataByIdentifier"
        assert msg.decoded["requested_dids"] == [0xF190, 0xF191]

    def test_parse_read_data_by_id_response(self) -> None:
        """Test parsing read data by identifier response."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(
            bytes([0x62, 0xF1, 0x90, 0x01, 0x02, 0x03, 0x04]), timestamp=5.1
        )

        assert msg.service_id == 0x22
        assert msg.is_response
        assert msg.decoded["did"] == 0xF190
        assert msg.decoded["did_data"] == "01020304"

    def test_parse_write_data_by_id_request(self) -> None:
        """Test parsing write data by identifier request."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x2E, 0xF1, 0x90, 0xAA, 0xBB, 0xCC]), timestamp=6.0)

        assert msg.service_id == 0x2E
        assert msg.service_name == "WriteDataByIdentifier"
        assert msg.decoded["did"] == 0xF190
        assert msg.decoded["did_data"] == "aabbcc"

    def test_parse_write_data_by_id_response(self) -> None:
        """Test parsing write data by identifier response."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x6E, 0xF1, 0x90]), timestamp=6.1)

        assert msg.service_id == 0x2E
        assert msg.is_response
        assert msg.decoded["did"] == 0xF190

    def test_parse_routine_control_start(self) -> None:
        """Test parsing routine control start routine."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x31, 0x01, 0x12, 0x34, 0xAA, 0xBB]), timestamp=7.0)

        assert msg.service_id == 0x31
        assert msg.service_name == "RoutineControl"
        assert msg.sub_function == 0x01
        assert msg.decoded["routine_type"] == "startRoutine"
        assert msg.decoded["routine_id"] == 0x1234
        assert msg.decoded["routine_option"] == "aabb"

    def test_parse_routine_control_response(self) -> None:
        """Test parsing routine control response."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x71, 0x01, 0x12, 0x34, 0x00]), timestamp=7.1)

        assert msg.service_id == 0x31
        assert msg.is_response
        assert msg.decoded["routine_id"] == 0x1234
        assert msg.decoded["status_record"] == "00"

    def test_parse_tester_present_request(self) -> None:
        """Test parsing tester present request."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x3E, 0x00]), timestamp=8.0)

        assert msg.service_id == 0x3E
        assert msg.service_name == "TesterPresent"
        assert msg.sub_function == 0x00

    def test_parse_tester_present_suppress_response(self) -> None:
        """Test parsing tester present with suppress response."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x3E, 0x80]), timestamp=8.1)

        assert msg.decoded["suppress_positive_response"]

    def test_parse_unknown_service(self) -> None:
        """Test parsing unknown service."""
        analyzer = UDSAnalyzer()
        msg = analyzer.parse_message(bytes([0x99, 0x01, 0x02]), timestamp=9.0)

        assert msg.service_id == 0x99
        assert "Unknown" in msg.service_name

    def test_ecu_state_tracking(self) -> None:
        """Test ECU state tracking."""
        analyzer = UDSAnalyzer()

        # Parse diagnostic session control request
        analyzer.parse_message(bytes([0x10, 0x03]), timestamp=1.0, ecu_id="ECU1")

        # Parse response
        analyzer.parse_message(
            bytes([0x50, 0x03, 0x00, 0x32, 0x01, 0xF4]), timestamp=1.1, ecu_id="ECU1"
        )

        ecu = analyzer.ecus["ECU1"]
        assert 0x10 in ecu.supported_services
        assert ecu.current_session == 0x03

    def test_ecu_security_level_tracking(self) -> None:
        """Test ECU security level tracking."""
        analyzer = UDSAnalyzer()

        # Security access request seed
        analyzer.parse_message(bytes([0x27, 0x01]), timestamp=2.0, ecu_id="ECU1")

        # Security access response with seed
        analyzer.parse_message(
            bytes([0x67, 0x01, 0x12, 0x34, 0x56, 0x78]), timestamp=2.1, ecu_id="ECU1"
        )

        # Security access send key
        analyzer.parse_message(
            bytes([0x27, 0x02, 0xAB, 0xCD, 0xEF, 0x01]), timestamp=2.2, ecu_id="ECU1"
        )

        # Positive response to send key
        analyzer.parse_message(bytes([0x67, 0x02]), timestamp=2.3, ecu_id="ECU1")

        ecu = analyzer.ecus["ECU1"]
        assert ecu.security_level == 1

    def test_ecu_dtc_storage(self) -> None:
        """Test ECU DTC storage."""
        analyzer = UDSAnalyzer()

        # Read DTCs response
        data = bytes(
            [
                0x59,
                0x02,
                0xFF,
                0xC1,
                0x23,
                0x45,
                0x0F,
            ]
        )
        analyzer.parse_message(data, timestamp=3.0, ecu_id="ECU1")

        ecu = analyzer.ecus["ECU1"]
        assert len(ecu.dtcs) == 1
        assert ecu.dtcs[0]["dtc"] == "C12345"

    def test_ecu_data_identifier_storage(self) -> None:
        """Test ECU data identifier storage."""
        analyzer = UDSAnalyzer()

        # Read data by identifier response
        analyzer.parse_message(
            bytes([0x62, 0xF1, 0x90, 0x01, 0x02, 0x03]), timestamp=4.0, ecu_id="ECU1"
        )

        ecu = analyzer.ecus["ECU1"]
        assert 0xF190 in ecu.data_identifiers
        assert ecu.data_identifiers[0xF190] == bytes([0x01, 0x02, 0x03])

    def test_multiple_ecus(self) -> None:
        """Test tracking multiple ECUs."""
        analyzer = UDSAnalyzer()

        analyzer.parse_message(bytes([0x10, 0x01]), timestamp=1.0, ecu_id="ECU1")
        analyzer.parse_message(bytes([0x10, 0x03]), timestamp=1.1, ecu_id="ECU2")

        assert "ECU1" in analyzer.ecus
        assert "ECU2" in analyzer.ecus
        assert len(analyzer.ecus) == 2

    def test_export_session_flows(self, tmp_path: Path) -> None:
        """Test exporting session flows to JSON."""
        analyzer = UDSAnalyzer()

        # Add some messages
        analyzer.parse_message(bytes([0x10, 0x03]), timestamp=1.0, ecu_id="ECU1")
        analyzer.parse_message(
            bytes([0x50, 0x03, 0x00, 0x32, 0x01, 0xF4]), timestamp=1.1, ecu_id="ECU1"
        )
        analyzer.parse_message(bytes([0x27, 0x01]), timestamp=2.0, ecu_id="ECU1")

        output_file = tmp_path / "flows.json"
        analyzer.export_session_flows(output_file)

        assert output_file.exists()

        # Verify JSON structure
        import json

        with output_file.open() as f:
            data = json.load(f)

        assert "messages" in data
        assert "ecus" in data
        assert len(data["messages"]) == 3
        assert "ECU1" in data["ecus"]
        assert data["ecus"]["ECU1"]["current_session"] == 0x03

    def test_service_constants(self) -> None:
        """Test service ID constants."""
        assert UDSAnalyzer.SERVICES[0x10] == "DiagnosticSessionControl"
        assert UDSAnalyzer.SERVICES[0x27] == "SecurityAccess"
        assert UDSAnalyzer.SERVICES[0x3E] == "TesterPresent"

    def test_diagnostic_session_constants(self) -> None:
        """Test diagnostic session constants."""
        assert UDSAnalyzer.DIAGNOSTIC_SESSIONS[0x01] == "DefaultSession"
        assert UDSAnalyzer.DIAGNOSTIC_SESSIONS[0x02] == "ProgrammingSession"
        assert UDSAnalyzer.DIAGNOSTIC_SESSIONS[0x03] == "ExtendedDiagnosticSession"

    def test_negative_response_code_constants(self) -> None:
        """Test negative response code constants."""
        assert UDSAnalyzer.NEGATIVE_RESPONSE_CODES[0x33] == "SecurityAccessDenied"
        assert UDSAnalyzer.NEGATIVE_RESPONSE_CODES[0x35] == "InvalidKey"
        assert (
            UDSAnalyzer.NEGATIVE_RESPONSE_CODES[0x78] == "RequestCorrectlyReceived-ResponsePending"
        )

    def test_all_services_covered(self) -> None:
        """Test that all documented services are in SERVICES dict."""
        expected_services = {
            0x10,
            0x11,
            0x14,
            0x19,
            0x22,
            0x23,
            0x24,
            0x27,
            0x28,
            0x2A,
            0x2C,
            0x2E,
            0x2F,
            0x31,
            0x34,
            0x35,
            0x36,
            0x37,
            0x38,
            0x3D,
            0x3E,
            0x83,
            0x84,
            0x85,
            0x86,
            0x87,
        }
        assert set(UDSAnalyzer.SERVICES.keys()) == expected_services

    def test_dtc_status_flags(self) -> None:
        """Test DTC status flag parsing."""
        analyzer = UDSAnalyzer()

        # Create DTC with all status bits set
        data = bytes([0x59, 0x02, 0xFF, 0x12, 0x34, 0x56, 0xFF])
        msg = analyzer.parse_message(data, timestamp=1.0)

        dtc = msg.decoded["dtcs"][0]
        assert dtc["test_failed"]
        assert dtc["test_failed_this_operation_cycle"]
        assert dtc["pending"]
        assert dtc["confirmed"]
        assert dtc["test_not_completed_since_last_clear"]
        assert dtc["test_failed_since_last_clear"]
        assert dtc["test_not_completed_this_operation_cycle"]
        assert dtc["warning_indicator_requested"]

    def test_message_list_tracking(self) -> None:
        """Test that all messages are tracked in messages list."""
        analyzer = UDSAnalyzer()

        analyzer.parse_message(bytes([0x10, 0x01]), timestamp=1.0)
        analyzer.parse_message(bytes([0x50, 0x01, 0x00, 0x32, 0x01, 0xF4]), timestamp=1.1)
        analyzer.parse_message(bytes([0x3E, 0x00]), timestamp=2.0)

        assert len(analyzer.messages) == 3
        assert analyzer.messages[0].service_id == 0x10
        assert analyzer.messages[1].service_id == 0x10
        assert analyzer.messages[1].is_response
        assert analyzer.messages[2].service_id == 0x3E

    def test_routine_control_all_types(self) -> None:
        """Test all routine control types."""
        analyzer = UDSAnalyzer()

        # Start routine
        msg1 = analyzer.parse_message(bytes([0x31, 0x01, 0x00, 0x01]), timestamp=1.0)
        assert msg1.decoded["routine_type"] == "startRoutine"

        # Stop routine
        msg2 = analyzer.parse_message(bytes([0x31, 0x02, 0x00, 0x01]), timestamp=2.0)
        assert msg2.decoded["routine_type"] == "stopRoutine"

        # Request routine results
        msg3 = analyzer.parse_message(bytes([0x31, 0x03, 0x00, 0x01]), timestamp=3.0)
        assert msg3.decoded["routine_type"] == "requestRoutineResults"

    def test_ecu_reset_all_types(self) -> None:
        """Test all ECU reset types."""
        analyzer = UDSAnalyzer()

        reset_types = [
            (0x01, "hardReset"),
            (0x02, "keyOffOnReset"),
            (0x03, "softReset"),
            (0x04, "enableRapidPowerShutDown"),
            (0x05, "disableRapidPowerShutDown"),
        ]

        for sub_func, expected_type in reset_types:
            msg = analyzer.parse_message(bytes([0x11, sub_func]), timestamp=1.0)
            assert msg.decoded["reset_type"] == expected_type

    def test_read_data_empty_response(self) -> None:
        """Test read data by identifier with minimal data."""
        analyzer = UDSAnalyzer()

        # Response with just DID, no data
        msg = analyzer.parse_message(bytes([0x62, 0xF1, 0x90]), timestamp=1.0)

        assert msg.decoded["did"] == 0xF190
        assert msg.decoded["did_data"] == ""

    def test_security_access_multiple_levels(self) -> None:
        """Test security access with multiple levels."""
        analyzer = UDSAnalyzer()

        # Level 1 request seed
        msg1 = analyzer.parse_message(bytes([0x27, 0x01]), timestamp=1.0)
        assert msg1.decoded["security_level"] == 1

        # Level 2 request seed
        msg2 = analyzer.parse_message(bytes([0x27, 0x03]), timestamp=2.0)
        assert msg2.decoded["security_level"] == 2

        # Level 5 request seed
        msg3 = analyzer.parse_message(bytes([0x27, 0x09]), timestamp=3.0)
        assert msg3.decoded["security_level"] == 5

    def test_comprehensive_session_flow(self) -> None:
        """Test comprehensive diagnostic session flow."""
        analyzer = UDSAnalyzer()

        # 1. Diagnostic session control
        analyzer.parse_message(bytes([0x10, 0x03]), timestamp=1.0, ecu_id="ECU1")
        analyzer.parse_message(
            bytes([0x50, 0x03, 0x00, 0x32, 0x01, 0xF4]), timestamp=1.1, ecu_id="ECU1"
        )

        # 2. Security access
        analyzer.parse_message(bytes([0x27, 0x01]), timestamp=2.0, ecu_id="ECU1")
        analyzer.parse_message(
            bytes([0x67, 0x01, 0x12, 0x34, 0x56, 0x78]), timestamp=2.1, ecu_id="ECU1"
        )
        analyzer.parse_message(bytes([0x27, 0x02, 0xAB, 0xCD]), timestamp=2.2, ecu_id="ECU1")
        analyzer.parse_message(bytes([0x67, 0x02]), timestamp=2.3, ecu_id="ECU1")

        # 3. Read DTCs
        analyzer.parse_message(bytes([0x19, 0x02, 0xFF]), timestamp=3.0, ecu_id="ECU1")
        analyzer.parse_message(
            bytes([0x59, 0x02, 0xFF, 0xC1, 0x23, 0x45, 0x08]), timestamp=3.1, ecu_id="ECU1"
        )

        # 4. Tester present
        analyzer.parse_message(bytes([0x3E, 0x80]), timestamp=4.0, ecu_id="ECU1")

        ecu = analyzer.ecus["ECU1"]
        assert ecu.current_session == 0x03
        assert ecu.security_level == 1
        assert len(ecu.dtcs) == 1
        assert 0x10 in ecu.supported_services
        assert 0x27 in ecu.supported_services
        assert 0x19 in ecu.supported_services
        assert 0x3E in ecu.supported_services
