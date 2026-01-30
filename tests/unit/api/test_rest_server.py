"""Tests for REST API server.

Comprehensive test suite for the FastAPI-based REST API server covering
session management, file upload, protocol analysis, and export endpoints.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oscura.api.rest_server import (
    AnalysisRequest,
    AnalysisResponse,
    ErrorResponse,
    ProtocolResponse,
    RESTAPIServer,
    SessionManager,
    SessionResponse,
)

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Skip all tests if FastAPI not available
pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def session_manager() -> SessionManager:
    """Create a session manager for testing."""
    return SessionManager(max_sessions=10, session_timeout=60.0)


@pytest.fixture
def mock_complete_re_result() -> Mock:
    """Create a mock CompleteREResult."""
    result = Mock()
    result.protocol_spec = Mock()
    result.protocol_spec.protocol_name = "UART"
    result.protocol_spec.messages = [b"\x01\x02\x03", b"\x04\x05\x06"]
    result.protocol_spec.fields = [
        Mock(name="header", offset=0, length=1, field_type="uint8", confidence=0.95),
        Mock(name="data", offset=1, length=2, field_type="bytes", confidence=0.90),
    ]
    result.confidence_score = 0.92
    result.dissector_path = Path("/tmp/dissector.lua")
    result.scapy_layer_path = Path("/tmp/layer.py")
    result.kaitai_path = Path("/tmp/spec.ksy")
    result.test_vectors_path = Path("/tmp/tests.json")
    result.report_path = Path("/tmp/report.html")
    result.validation_result = None
    result.warnings = []
    result.execution_time = 5.2
    result.partial_results = {}
    return result


@pytest.fixture
def api_server() -> RESTAPIServer:
    """Create a REST API server for testing."""
    return RESTAPIServer(
        host="127.0.0.1",
        port=8000,
        max_sessions=10,
        enable_cors=True,
    )


@pytest.fixture
def test_client(api_server: RESTAPIServer) -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(api_server.app)


@pytest.fixture
def sample_file_data() -> bytes:
    """Create sample file data for upload."""
    # Simple binary data simulating a capture
    return b"\x55\x00\x01\x02\x03\x04\x05\x06" * 10


# ============================================================================
# Request/Response Model Tests
# ============================================================================


def test_analysis_request_creation() -> None:
    """Test AnalysisRequest dataclass creation."""
    request = AnalysisRequest(
        file_data=b"\x01\x02\x03",
        filename="test.bin",
        protocol_hint="uart",
        auto_crc=True,
        detect_crypto=False,
        generate_tests=True,
        export_formats=["wireshark", "scapy"],
    )

    assert request.file_data == b"\x01\x02\x03"
    assert request.filename == "test.bin"
    assert request.protocol_hint == "uart"
    assert request.auto_crc is True
    assert request.detect_crypto is False
    assert request.generate_tests is True
    assert request.export_formats == ["wireshark", "scapy"]


def test_analysis_request_defaults() -> None:
    """Test AnalysisRequest default values."""
    request = AnalysisRequest(
        file_data=b"\x01\x02\x03",
        filename="test.bin",
    )

    assert request.protocol_hint is None
    assert request.auto_crc is True
    assert request.detect_crypto is True
    assert request.generate_tests is True
    assert request.export_formats == ["wireshark"]


def test_analysis_response_creation() -> None:
    """Test AnalysisResponse dataclass creation."""
    response = AnalysisResponse(
        session_id="abc-123",
        status="complete",
        protocols_found=["UART", "SPI"],
        confidence_scores={"UART": 0.95, "SPI": 0.80},
        message="Analysis complete",
        created_at="2026-01-25T00:00:00",
        estimated_duration=10.5,
    )

    assert response.session_id == "abc-123"
    assert response.status == "complete"
    assert response.protocols_found == ["UART", "SPI"]
    assert response.confidence_scores == {"UART": 0.95, "SPI": 0.80}
    assert response.message == "Analysis complete"
    assert response.estimated_duration == 10.5


def test_session_response_creation() -> None:
    """Test SessionResponse dataclass creation."""
    response = SessionResponse(
        session_id="xyz-456",
        status="processing",
        messages_decoded=42,
        fields_discovered=5,
        artifacts={"dissector": "/path/to/dissector.lua"},
        statistics={"execution_time": 5.2},
    )

    assert response.session_id == "xyz-456"
    assert response.status == "processing"
    assert response.messages_decoded == 42
    assert response.fields_discovered == 5
    assert response.artifacts == {"dissector": "/path/to/dissector.lua"}


def test_protocol_response_creation() -> None:
    """Test ProtocolResponse dataclass creation."""
    response = ProtocolResponse(
        protocol_name="UART",
        confidence=0.95,
        message_count=100,
        field_count=3,
        fields=[{"name": "header", "type": "uint8"}],
        crc_info={"polynomial": "0x1021"},
    )

    assert response.protocol_name == "UART"
    assert response.confidence == 0.95
    assert response.message_count == 100
    assert response.field_count == 3
    assert len(response.fields) == 1


def test_error_response_creation() -> None:
    """Test ErrorResponse dataclass creation."""
    response = ErrorResponse(
        error_code="INVALID_FILE",
        message="File format not supported",
        details={"format": "unknown"},
    )

    assert response.error_code == "INVALID_FILE"
    assert response.message == "File format not supported"
    assert response.details == {"format": "unknown"}
    assert "T" in response.timestamp  # ISO timestamp format


# ============================================================================
# SessionManager Tests
# ============================================================================


def test_session_manager_creation() -> None:
    """Test SessionManager initialization."""
    manager = SessionManager(max_sessions=50, session_timeout=3600.0)

    assert manager.max_sessions == 50
    assert manager.session_timeout == 3600.0
    assert len(manager.sessions) == 0


def test_create_session(session_manager: SessionManager) -> None:
    """Test session creation."""
    session_id = session_manager.create_session(
        filename="test.bin",
        file_data=b"\x01\x02\x03",
        options={"protocol_hint": "uart"},
    )

    assert isinstance(session_id, str)
    assert len(session_id) == 36  # UUID format
    assert session_id in session_manager.sessions

    session = session_manager.sessions[session_id]
    assert session["filename"] == "test.bin"
    assert session["file_data"] == b"\x01\x02\x03"
    assert session["options"] == {"protocol_hint": "uart"}
    assert session["status"] == "created"
    assert session["result"] is None
    assert session["error"] is None


def test_create_session_file_hash(session_manager: SessionManager) -> None:
    """Test session creation includes file hash."""
    session_id = session_manager.create_session(
        filename="test.bin",
        file_data=b"\x01\x02\x03",
        options={},
    )

    session = session_manager.sessions[session_id]
    assert "file_hash" in session
    assert len(session["file_hash"]) == 64  # SHA256 hex digest


def test_get_session(session_manager: SessionManager) -> None:
    """Test retrieving session by ID."""
    session_id = session_manager.create_session(
        filename="test.bin",
        file_data=b"\x01\x02\x03",
        options={},
    )

    session = session_manager.get_session(session_id)
    assert session is not None
    assert session["id"] == session_id


def test_get_nonexistent_session(session_manager: SessionManager) -> None:
    """Test retrieving non-existent session."""
    session = session_manager.get_session("nonexistent-id")
    assert session is None


def test_get_session_updates_accessed_time(session_manager: SessionManager) -> None:
    """Test that getting a session updates accessed_at timestamp."""
    session_id = session_manager.create_session(
        filename="test.bin",
        file_data=b"\x01\x02\x03",
        options={},
    )

    original_time = session_manager.sessions[session_id]["accessed_at"]
    time.sleep(0.01)
    session_manager.get_session(session_id)
    new_time = session_manager.sessions[session_id]["accessed_at"]

    assert new_time > original_time


def test_update_session(session_manager: SessionManager, mock_complete_re_result: Mock) -> None:
    """Test updating session status."""
    session_id = session_manager.create_session(
        filename="test.bin",
        file_data=b"\x01\x02\x03",
        options={},
    )

    session_manager.update_session(session_id, "complete", result=mock_complete_re_result)

    session = session_manager.sessions[session_id]
    assert session["status"] == "complete"
    assert session["result"] == mock_complete_re_result
    assert session["error"] is None


def test_update_session_with_error(session_manager: SessionManager) -> None:
    """Test updating session with error."""
    session_id = session_manager.create_session(
        filename="test.bin",
        file_data=b"\x01\x02\x03",
        options={},
    )

    session_manager.update_session(session_id, "error", error="Analysis failed")

    session = session_manager.sessions[session_id]
    assert session["status"] == "error"
    assert session["error"] == "Analysis failed"


def test_delete_session(session_manager: SessionManager) -> None:
    """Test deleting a session."""
    session_id = session_manager.create_session(
        filename="test.bin",
        file_data=b"\x01\x02\x03",
        options={},
    )

    deleted = session_manager.delete_session(session_id)
    assert deleted is True
    assert session_id not in session_manager.sessions


def test_delete_nonexistent_session(session_manager: SessionManager) -> None:
    """Test deleting non-existent session."""
    deleted = session_manager.delete_session("nonexistent-id")
    assert deleted is False


def test_list_sessions(session_manager: SessionManager) -> None:
    """Test listing all sessions."""
    # Create multiple sessions
    sid1 = session_manager.create_session("file1.bin", b"\x01", {})
    sid2 = session_manager.create_session("file2.bin", b"\x02", {})

    sessions = session_manager.list_sessions()

    assert len(sessions) == 2
    session_ids = [s["session_id"] for s in sessions]
    assert sid1 in session_ids
    assert sid2 in session_ids


def test_list_sessions_empty(session_manager: SessionManager) -> None:
    """Test listing sessions when none exist."""
    sessions = session_manager.list_sessions()
    assert len(sessions) == 0


def test_max_sessions_exceeded(session_manager: SessionManager) -> None:
    """Test that max sessions limit is enforced."""
    # Fill to capacity
    for i in range(session_manager.max_sessions):
        session_manager.create_session(f"file{i}.bin", b"\x01", {})

    # Next should raise error
    with pytest.raises(RuntimeError, match="Maximum sessions"):
        session_manager.create_session("overflow.bin", b"\x01", {})


def test_cleanup_old_sessions() -> None:
    """Test automatic cleanup of timed-out sessions."""
    manager = SessionManager(max_sessions=5, session_timeout=0.05)

    # Create sessions
    for i in range(5):
        manager.create_session(f"file{i}.bin", b"\x01", {})

    # Wait for timeout
    time.sleep(0.1)

    # Try to create new session (should trigger cleanup)
    new_id = manager.create_session("new.bin", b"\x02", {})

    # Old sessions should be removed
    assert len(manager.sessions) == 1
    assert new_id in manager.sessions


# ============================================================================
# REST API Endpoint Tests
# ============================================================================


def test_health_check(test_client: TestClient) -> None:
    """Test health check endpoint."""
    response = test_client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.7.0"
    assert "sessions_active" in data
    assert "timestamp" in data


def test_analyze_endpoint(test_client: TestClient, sample_file_data: bytes) -> None:
    """Test file upload and analysis endpoint."""
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}

    response = test_client.post("/api/v1/analyze", files=files)

    assert response.status_code == 202  # Accepted
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "processing"
    assert "created_at" in data


def test_analyze_with_protocol_hint(test_client: TestClient, sample_file_data: bytes) -> None:
    """Test analysis with protocol hint."""
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}
    params = {"protocol_hint": "uart", "auto_crc": False}

    response = test_client.post("/api/v1/analyze", files=files, params=params)

    assert response.status_code == 202
    data = response.json()
    assert "session_id" in data


def test_analyze_no_filename(test_client: TestClient) -> None:
    """Test analysis without filename raises error."""
    # Create file without filename (FastAPI rejects this with 422)
    files = {"file": ("", io.BytesIO(b"\x01\x02"), "application/octet-stream")}

    response = test_client.post("/api/v1/analyze", files=files)

    # FastAPI validation rejects before reaching endpoint
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_list_sessions_endpoint(test_client: TestClient) -> None:
    """Test listing sessions endpoint."""
    response = test_client.get("/api/v1/sessions")

    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert "count" in data
    assert isinstance(data["sessions"], list)


def test_get_session_endpoint(
    test_client: TestClient,
    api_server: RESTAPIServer,
    sample_file_data: bytes,
) -> None:
    """Test getting session details endpoint."""
    # Create a session first
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}
    create_response = test_client.post("/api/v1/analyze", files=files)
    session_id = create_response.json()["session_id"]

    # Update session status (without storing Mock result which can't be serialized)
    test_client.app.extra["session_manager"] = api_server.session_manager
    api_server.session_manager.update_session(session_id, "complete")

    # Get session
    response = test_client.get(f"/api/v1/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["status"] == "complete"


def test_get_session_not_found(test_client: TestClient) -> None:
    """Test getting non-existent session."""
    response = test_client.get("/api/v1/sessions/nonexistent-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_session_endpoint(test_client: TestClient, sample_file_data: bytes) -> None:
    """Test deleting session endpoint."""
    # Create a session first
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}
    create_response = test_client.post("/api/v1/analyze", files=files)
    session_id = create_response.json()["session_id"]

    # Delete session
    response = test_client.delete(f"/api/v1/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "deleted" in data["message"].lower()


def test_delete_session_not_found(test_client: TestClient) -> None:
    """Test deleting non-existent session."""
    response = test_client.delete("/api/v1/sessions/nonexistent-id")

    assert response.status_code == 404


def test_list_protocols_endpoint(test_client: TestClient) -> None:
    """Test listing discovered protocols endpoint."""
    response = test_client.get("/api/v1/protocols")

    assert response.status_code == 200
    data = response.json()
    assert "protocols" in data
    assert "count" in data
    assert isinstance(data["protocols"], list)


def test_export_endpoint(
    test_client: TestClient,
    api_server: RESTAPIServer,
    sample_file_data: bytes,
    mock_complete_re_result: Mock,
) -> None:
    """Test export endpoint."""
    # Create and complete a session
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}
    create_response = test_client.post("/api/v1/analyze", files=files)
    session_id = create_response.json()["session_id"]

    # Update session to complete with mock result
    api_server.session_manager.update_session(
        session_id, "complete", result=mock_complete_re_result
    )

    # Export as Wireshark dissector
    response = test_client.post(
        "/api/v1/export/wireshark",
        params={"session_id": session_id},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "wireshark"
    assert "file_path" in data


def test_export_invalid_format(
    test_client: TestClient,
    api_server: RESTAPIServer,
    sample_file_data: bytes,
    mock_complete_re_result: Mock,
) -> None:
    """Test export with invalid format."""
    # Create and complete a session
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}
    create_response = test_client.post("/api/v1/analyze", files=files)
    session_id = create_response.json()["session_id"]

    api_server.session_manager.update_session(
        session_id, "complete", result=mock_complete_re_result
    )

    # Try invalid format
    response = test_client.post(
        "/api/v1/export/invalid_format",
        params={"session_id": session_id},
    )

    assert response.status_code == 400
    assert "Invalid format" in response.json()["detail"]


@patch("oscura.api.rest_server.BackgroundTasks.add_task")
def test_export_incomplete_session(
    mock_add_task: Mock, test_client: TestClient, sample_file_data: bytes
) -> None:
    """Test export on incomplete session."""
    # Mock prevents background task from completing the session
    # Create session but don't complete it
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}
    create_response = test_client.post("/api/v1/analyze", files=files)
    session_id = create_response.json()["session_id"]

    # Try to export
    response = test_client.post(
        "/api/v1/export/wireshark",
        params={"session_id": session_id},
    )

    assert response.status_code == 400
    assert "not complete" in response.json()["detail"]


# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_workflow(
    test_client: TestClient,
    api_server: RESTAPIServer,
    sample_file_data: bytes,
    mock_complete_re_result: Mock,
) -> None:
    """Test complete workflow from upload to export."""
    # Mock the analysis function - patch where it's imported (inside _run_analysis)
    with patch("oscura.workflows.complete_re.full_protocol_re") as mock_full_re:
        mock_full_re.return_value = mock_complete_re_result

    # 1. Upload file
    files = {"file": ("test.bin", io.BytesIO(sample_file_data), "application/octet-stream")}
    create_response = test_client.post("/api/v1/analyze", files=files)
    assert create_response.status_code == 202
    session_id = create_response.json()["session_id"]

    # 2. Manually run analysis (background task not auto-executed in test)
    api_server._run_analysis(session_id)

    # 3. Check session status
    session_response = test_client.get(f"/api/v1/sessions/{session_id}")
    assert session_response.status_code == 200
    assert session_response.json()["status"] == "complete"

    # 4. Export results
    export_response = test_client.post(
        "/api/v1/export/wireshark",
        params={"session_id": session_id},
    )
    assert export_response.status_code == 200


def test_serialization_methods(api_server: RESTAPIServer, mock_complete_re_result: Mock) -> None:
    """Test protocol spec and artifact serialization."""
    # Test protocol spec serialization
    spec_dict = api_server._serialize_protocol_spec(mock_complete_re_result)
    assert spec_dict["protocol_name"] == "UART"
    assert spec_dict["message_count"] == 2
    assert spec_dict["field_count"] == 2
    assert len(spec_dict["fields"]) == 2

    # Test artifact serialization
    artifacts = api_server._serialize_artifacts(mock_complete_re_result)
    assert "dissector_path" in artifacts
    assert "scapy_layer_path" in artifacts
    assert "kaitai_path" in artifacts


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_api_server_creation_without_fastapi() -> None:
    """Test that server creation fails gracefully without FastAPI."""
    with patch("oscura.api.rest_server.HAS_FASTAPI", False):
        with pytest.raises(ImportError, match="FastAPI required"):
            RESTAPIServer()


def test_run_without_uvicorn() -> None:
    """Test that run() fails gracefully without uvicorn."""
    server = RESTAPIServer()

    # Patch the import inside the run() method to simulate missing uvicorn
    with patch.dict("sys.modules", {"uvicorn": None}):
        with pytest.raises(ImportError, match="uvicorn required"):
            server.run()


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_file_upload(test_client: TestClient) -> None:
    """Test uploading empty file."""
    files = {"file": ("empty.bin", io.BytesIO(b""), "application/octet-stream")}

    response = test_client.post("/api/v1/analyze", files=files)

    # Should accept but may fail during analysis
    assert response.status_code == 202


def test_large_file_upload(test_client: TestClient) -> None:
    """Test uploading large file."""
    # Create 1MB file
    large_data = b"\x00" * (1024 * 1024)
    files = {"file": ("large.bin", io.BytesIO(large_data), "application/octet-stream")}

    response = test_client.post("/api/v1/analyze", files=files)

    assert response.status_code == 202


def test_concurrent_sessions(session_manager: SessionManager) -> None:
    """Test handling multiple concurrent sessions."""
    session_ids = []

    for i in range(5):
        sid = session_manager.create_session(f"file{i}.bin", b"\x01" * i, {})
        session_ids.append(sid)

    assert len(session_manager.sessions) == 5
    assert all(sid in session_manager.sessions for sid in session_ids)


def test_session_timeout_precision(session_manager: SessionManager) -> None:
    """Test session timeout with high precision."""
    manager = SessionManager(max_sessions=5, session_timeout=0.02)

    sid = manager.create_session("test.bin", b"\x01", {})
    time.sleep(0.03)

    # Trigger cleanup
    manager._cleanup_old_sessions()

    assert sid not in manager.sessions
