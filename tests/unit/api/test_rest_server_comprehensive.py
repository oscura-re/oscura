"""Comprehensive tests for REST API server (enhanced coverage).

This module provides additional test coverage for the REST API server beyond
the basic tests in test_rest_server.py, targeting critical code paths including:
- Error handling and edge cases
- Session timeout and cleanup
- Export validation and artifact handling
- Background task execution
- Protocol serialization
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Skip all tests if FastAPI not available
pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")

if HAS_FASTAPI:
    from oscura.api.rest_server import (
        RESTAPIServer,
        SessionManager,
    )


# ============================================================================
# Session Manager Comprehensive Tests
# ============================================================================


@pytest.mark.unit
class TestSessionManagerComprehensive:
    """Comprehensive tests for SessionManager."""

    def test_session_cleanup_on_max_sessions(self) -> None:
        """Test that old sessions are cleaned up when max is reached."""
        manager = SessionManager(max_sessions=3, session_timeout=1.0)

        # Create 3 sessions
        sid1 = manager.create_session("file1.bin", b"data1", {})
        sid2 = manager.create_session("file2.bin", b"data2", {})
        sid3 = manager.create_session("file3.bin", b"data3", {})

        assert len(manager.sessions) == 3

        # Wait for timeout
        time.sleep(1.1)

        # Creating a 4th session should trigger cleanup
        sid4 = manager.create_session("file4.bin", b"data4", {})

        # Old sessions should be cleaned up
        assert len(manager.sessions) <= 3
        assert sid4 in manager.sessions

    def test_session_cleanup_removes_only_expired(self) -> None:
        """Test cleanup only removes timed-out sessions."""
        manager = SessionManager(max_sessions=10, session_timeout=0.5)

        # Create first session
        sid1 = manager.create_session("file1.bin", b"data1", {})

        # Wait a bit
        time.sleep(0.3)

        # Create second session (fresh)
        sid2 = manager.create_session("file2.bin", b"data2", {})

        # Wait for first session to timeout
        time.sleep(0.3)

        # Trigger cleanup by creating new session
        sid3 = manager.create_session("file3.bin", b"data3", {})

        # sid1 should be cleaned up, but sid2 and sid3 remain
        assert sid1 not in manager.sessions
        assert sid2 in manager.sessions
        assert sid3 in manager.sessions

    def test_get_session_updates_access_time(self) -> None:
        """Test that getting a session updates its access time."""
        manager = SessionManager(max_sessions=10, session_timeout=1.0)
        sid = manager.create_session("test.bin", b"data", {})

        # Get initial access time
        session = manager.get_session(sid)
        assert session is not None
        initial_access = session["accessed_at"]

        # Wait a bit
        time.sleep(0.1)

        # Access again
        session = manager.get_session(sid)
        assert session is not None
        new_access = session["accessed_at"]

        # Access time should be updated
        assert new_access > initial_access

    def test_update_session_updates_timestamps(self) -> None:
        """Test that updating a session updates both updated_at and accessed_at."""
        manager = SessionManager(max_sessions=10, session_timeout=10.0)
        sid = manager.create_session("test.bin", b"data", {})

        # Get initial timestamps
        session = manager.get_session(sid)
        assert session is not None
        initial_updated = session["updated_at"]
        initial_accessed = session["accessed_at"]

        # Wait a bit
        time.sleep(0.1)

        # Update session
        manager.update_session(sid, "processing", result="some_result")

        # Get updated session
        session = manager.get_session(sid)
        assert session is not None

        # Both timestamps should be updated
        assert session["updated_at"] > initial_updated
        assert session["accessed_at"] > initial_accessed
        assert session["status"] == "processing"
        assert session["result"] == "some_result"

    def test_update_session_with_error(self) -> None:
        """Test updating a session with an error."""
        manager = SessionManager(max_sessions=10, session_timeout=10.0)
        sid = manager.create_session("test.bin", b"data", {})

        manager.update_session(sid, "error", error="Something went wrong")

        session = manager.get_session(sid)
        assert session is not None
        assert session["status"] == "error"
        assert session["error"] == "Something went wrong"

    def test_delete_nonexistent_session_returns_false(self) -> None:
        """Test deleting a session that doesn't exist returns False."""
        manager = SessionManager(max_sessions=10, session_timeout=10.0)
        result = manager.delete_session("nonexistent_id")
        assert result is False

    def test_list_sessions_returns_summaries(self) -> None:
        """Test list_sessions returns correct session summaries."""
        manager = SessionManager(max_sessions=10, session_timeout=10.0)

        sid1 = manager.create_session("file1.bin", b"data1", {})
        sid2 = manager.create_session("file2.bin", b"data2", {})

        sessions = manager.list_sessions()

        assert len(sessions) == 2
        assert all("session_id" in s for s in sessions)
        assert all("status" in s for s in sessions)
        assert all("filename" in s for s in sessions)
        assert all("created_at" in s for s in sessions)
        assert all("updated_at" in s for s in sessions)

        filenames = [s["filename"] for s in sessions]
        assert "file1.bin" in filenames
        assert "file2.bin" in filenames

    def test_max_sessions_exceeded_raises_error(self) -> None:
        """Test that exceeding max sessions raises RuntimeError."""
        manager = SessionManager(max_sessions=2, session_timeout=100.0)

        # Create max sessions
        manager.create_session("file1.bin", b"data1", {})
        manager.create_session("file2.bin", b"data2", {})

        # Trying to create one more should raise (after cleanup fails)
        with pytest.raises(RuntimeError, match="Maximum sessions"):
            manager.create_session("file3.bin", b"data3", {})

    def test_session_file_hash_computed(self) -> None:
        """Test that file hash is computed correctly."""
        import hashlib

        manager = SessionManager(max_sessions=10, session_timeout=10.0)
        file_data = b"test_data_123"
        expected_hash = hashlib.sha256(file_data).hexdigest()

        sid = manager.create_session("test.bin", file_data, {})
        session = manager.get_session(sid)

        assert session is not None
        assert session["file_hash"] == expected_hash

    def test_update_nonexistent_session_does_nothing(self) -> None:
        """Test updating a session that doesn't exist does nothing."""
        manager = SessionManager(max_sessions=10, session_timeout=10.0)

        # This should not raise an error
        manager.update_session("nonexistent_id", "complete", result="data")

        # Session should still not exist
        assert manager.get_session("nonexistent_id") is None


# ============================================================================
# REST API Server Comprehensive Tests
# ============================================================================


@pytest.mark.unit
class TestRESTAPIServerComprehensive:
    """Comprehensive tests for RESTAPIServer endpoints."""

    @pytest.fixture
    def mock_full_protocol_re(self) -> Mock:
        """Create a mock for full_protocol_re."""
        result = Mock()
        result.protocol_spec = Mock()
        result.protocol_spec.protocol_name = "UART"
        result.protocol_spec.messages = [b"\x01\x02", b"\x03\x04"]
        result.protocol_spec.fields = [
            Mock(name="header", offset=0, length=1, field_type="uint8", confidence=0.9),
        ]
        result.confidence_score = 0.85
        result.dissector_path = Path("/tmp/dissector.lua")
        result.scapy_layer_path = Path("/tmp/layer.py")
        result.kaitai_path = Path("/tmp/spec.ksy")
        result.test_vectors_path = Path("/tmp/tests.json")
        result.report_path = Path("/tmp/report.html")
        return result

    def test_health_endpoint_returns_version(self) -> None:
        """Test health endpoint returns version and status."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "sessions_active" in data
        assert "timestamp" in data

    def test_analyze_endpoint_missing_filename_fails(self) -> None:
        """Test analyze endpoint with missing filename fails."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create file with no filename (FastAPI rejects with 422)
        files = {"file": ("", b"test_data", "application/octet-stream")}

        response = client.post("/api/v1/analyze", files=files)

        # FastAPI validation rejects before reaching endpoint
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_analyze_endpoint_max_sessions_exceeded(self) -> None:
        """Test analyze endpoint when max sessions exceeded."""
        server = RESTAPIServer(host="127.0.0.1", port=8000, max_sessions=1)
        client = TestClient(server.app)

        # Create first session
        files1 = {"file": ("test1.bin", b"data1", "application/octet-stream")}
        response1 = client.post("/api/v1/analyze", files=files1)
        assert response1.status_code == 202

        # Try to create second session (should fail)
        files2 = {"file": ("test2.bin", b"data2", "application/octet-stream")}
        response2 = client.post("/api/v1/analyze", files=files2)

        assert response2.status_code == 503
        assert "Maximum sessions" in response2.json()["detail"]

    def test_get_session_nonexistent_returns_404(self) -> None:
        """Test getting a nonexistent session returns 404."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        response = client.get("/api/v1/sessions/nonexistent_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_session_nonexistent_returns_404(self) -> None:
        """Test deleting a nonexistent session returns 404."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        response = client.delete("/api/v1/sessions/nonexistent_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_session_success(self) -> None:
        """Test successfully deleting a session."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create a session
        files = {"file": ("test.bin", b"test_data", "application/octet-stream")}
        response = client.post("/api/v1/analyze", files=files)
        session_id = response.json()["session_id"]

        # Delete it
        response = client.delete(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        assert response.json()["message"] == "Session deleted"
        assert response.json()["session_id"] == session_id

        # Verify it's gone
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 404

    def test_export_invalid_format_returns_400(self) -> None:
        """Test export with invalid format returns 400."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create a completed session
        sid = server.session_manager.create_session("test.bin", b"data", {})
        server.session_manager.update_session(sid, "complete", result=Mock())

        # Try to export with invalid format
        response = client.post("/api/v1/export/invalid_format", params={"session_id": sid})

        assert response.status_code == 400
        assert "Invalid format" in response.json()["detail"]

    def test_export_incomplete_session_returns_400(self) -> None:
        """Test export of incomplete session returns 400."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create a processing session
        sid = server.session_manager.create_session("test.bin", b"data", {})
        server.session_manager.update_session(sid, "processing")

        # Try to export
        response = client.post("/api/v1/export/wireshark", params={"session_id": sid})

        assert response.status_code == 400
        assert "not complete" in response.json()["detail"]

    def test_export_missing_artifact_returns_404(self, mock_full_protocol_re: Mock) -> None:
        """Test export when artifact is missing returns 404."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create a completed session with no dissector
        result = mock_full_protocol_re
        result.dissector_path = None  # No Wireshark dissector

        sid = server.session_manager.create_session("test.bin", b"data", {})
        server.session_manager.update_session(sid, "complete", result=result)

        # Try to export Wireshark dissector
        response = client.post("/api/v1/export/wireshark", params={"session_id": sid})

        assert response.status_code == 404
        assert "No wireshark artifact available" in response.json()["detail"]

    def test_list_sessions_returns_all_sessions(self) -> None:
        """Test list sessions endpoint returns all active sessions."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create multiple sessions
        files1 = {"file": ("test1.bin", b"data1", "application/octet-stream")}
        files2 = {"file": ("test2.bin", b"data2", "application/octet-stream")}

        client.post("/api/v1/analyze", files=files1)
        client.post("/api/v1/analyze", files=files2)

        # List sessions
        response = client.get("/api/v1/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["sessions"]) == 2
        assert "timestamp" in data

    def test_protocols_endpoint_returns_empty_initially(self) -> None:
        """Test protocols endpoint returns empty list initially."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        response = client.get("/api/v1/protocols")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["protocols"] == []

    def test_protocols_endpoint_returns_discovered_protocols(
        self, mock_full_protocol_re: Mock
    ) -> None:
        """Test protocols endpoint returns discovered protocols."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create a completed session with protocol
        sid = server.session_manager.create_session("test.bin", b"data", {})
        server.session_manager.update_session(sid, "complete", result=mock_full_protocol_re)

        # Get protocols
        response = client.get("/api/v1/protocols")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["protocols"]) == 1
        assert data["protocols"][0]["protocol_name"] == "UART"
        assert data["protocols"][0]["session_id"] == sid

    def test_session_response_includes_error(self) -> None:
        """Test session response includes error when analysis fails."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)
        client = TestClient(server.app)

        # Create a failed session
        sid = server.session_manager.create_session("test.bin", b"data", {})
        server.session_manager.update_session(sid, "error", error="Analysis failed: bad data")

        # Get session
        response = client.get(f"/api/v1/sessions/{sid}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["error"] == "Analysis failed: bad data"

    def test_run_analysis_handles_missing_session(self) -> None:
        """Test _run_analysis gracefully handles missing session."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)

        # Call with nonexistent session ID
        server._run_analysis("nonexistent_id")

        # Should not raise an error (logs error instead)
        # No assertion needed, just verifying it doesn't crash

    def test_run_analysis_handles_exception(self, tmp_path: Path) -> None:
        """Test _run_analysis handles exceptions during analysis."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)

        # Create a session with invalid data
        sid = server.session_manager.create_session("test.bin", b"invalid", {})

        # Mock full_protocol_re to raise an exception
        with patch("oscura.api.rest_server.full_protocol_re") as mock_func:
            mock_func.side_effect = ValueError("Invalid data format")

            # Run analysis
            server._run_analysis(sid)

            # Session should be in error state
            session = server.session_manager.get_session(sid)
            assert session is not None
            assert session["status"] == "error"
            assert "Invalid data format" in session["error"]

    @patch("oscura.api.rest_server.full_protocol_re")
    def test_run_analysis_success(
        self, mock_func: Mock, mock_full_protocol_re: Mock, tmp_path: Path
    ) -> None:
        """Test _run_analysis successfully completes analysis."""
        mock_func.return_value = mock_full_protocol_re

        server = RESTAPIServer(host="127.0.0.1", port=8000)

        # Create a session
        sid = server.session_manager.create_session(
            "test.bin", b"test_data", {"protocol_hint": "uart", "auto_crc": True}
        )

        # Run analysis
        server._run_analysis(sid)

        # Verify session is complete
        session = server.session_manager.get_session(sid)
        assert session is not None
        assert session["status"] == "complete"
        assert session["result"] == mock_full_protocol_re
        assert session["error"] is None

    def test_serialize_protocol_spec_handles_missing_attributes(self) -> None:
        """Test _serialize_protocol_spec handles missing attributes gracefully."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)

        # Create a result with minimal attributes
        result = Mock()
        result.protocol_spec = Mock()
        # Don't set any attributes

        # Should not raise an error
        spec = server._serialize_protocol_spec(result)

        assert spec["protocol_name"] == "unknown"
        assert spec["message_count"] == 0
        assert spec["field_count"] == 0
        assert spec["fields"] == []

    def test_serialize_artifacts_handles_none_paths(self) -> None:
        """Test _serialize_artifacts handles None paths gracefully."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)

        # Create a result with all None paths
        result = Mock()
        result.dissector_path = None
        result.scapy_layer_path = None
        result.kaitai_path = None
        result.test_vectors_path = None
        result.report_path = None

        # Should return empty dict
        artifacts = server._serialize_artifacts(result)

        assert artifacts == {}

    def test_build_session_response_with_result(self, mock_full_protocol_re: Mock) -> None:
        """Test _build_session_response includes result details."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)

        # Create a session with result
        session = {
            "id": "test_id",
            "status": "complete",
            "filename": "test.bin",
            "file_hash": "abc123",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:05:00",
            "result": mock_full_protocol_re,
            "error": None,
        }

        response = server._build_session_response(session)

        assert response["session_id"] == "test_id"
        assert response["status"] == "complete"
        assert response["filename"] == "test.bin"
        assert "protocol_spec" in response
        assert "artifacts" in response
        assert response["confidence_score"] == 0.85

    def test_extract_protocols_from_sessions_empty(self) -> None:
        """Test _extract_protocols_from_sessions with no sessions."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)

        protocols = server._extract_protocols_from_sessions()

        assert protocols == []

    def test_import_error_for_fastapi(self) -> None:
        """Test that ImportError is raised when FastAPI is not available."""
        with patch("oscura.api.rest_server.HAS_FASTAPI", False):
            with pytest.raises(ImportError, match="FastAPI required"):
                RESTAPIServer(host="127.0.0.1", port=8000)

    def test_run_requires_uvicorn(self) -> None:
        """Test that run() raises ImportError when uvicorn is not available."""
        server = RESTAPIServer(host="127.0.0.1", port=8000)

        with patch("oscura.api.rest_server.HAS_FASTAPI", True):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'uvicorn'")):
                with pytest.raises(ImportError, match="uvicorn required"):
                    server.run()

    def test_cors_middleware_configured(self) -> None:
        """Test that CORS middleware is properly configured."""
        server = RESTAPIServer(host="127.0.0.1", port=8000, enable_cors=True)

        # Check that middleware is added
        assert len(server.app.user_middleware) > 0

    def test_cors_disabled(self) -> None:
        """Test creating server without CORS."""
        server = RESTAPIServer(host="127.0.0.1", port=8000, enable_cors=False)

        # Server should still work
        assert server.app is not None

    def test_custom_cors_origins(self) -> None:
        """Test creating server with custom CORS origins."""
        origins = ["https://example.com", "https://api.example.com"]
        server = RESTAPIServer(host="127.0.0.1", port=8000, enable_cors=True, cors_origins=origins)

        assert server.app is not None


# ============================================================================
# Data Model Tests
# ============================================================================


@pytest.mark.unit
class TestDataModels:
    """Test request/response data models."""

    def test_analysis_request_creation(self) -> None:
        """Test creating AnalysisRequest."""
        from oscura.api.rest_server import AnalysisRequest

        request = AnalysisRequest(
            file_data=b"test",
            filename="test.bin",
            protocol_hint="uart",
            auto_crc=True,
            detect_crypto=False,
            generate_tests=True,
            export_formats=["wireshark", "scapy"],
        )

        assert request.file_data == b"test"
        assert request.filename == "test.bin"
        assert request.protocol_hint == "uart"
        assert request.auto_crc is True
        assert request.detect_crypto is False
        assert len(request.export_formats) == 2

    def test_analysis_response_defaults(self) -> None:
        """Test AnalysisResponse default values."""
        from oscura.api.rest_server import AnalysisResponse

        response = AnalysisResponse(session_id="test_id", status="processing")

        assert response.session_id == "test_id"
        assert response.status == "processing"
        assert response.protocols_found == []
        assert response.confidence_scores == {}
        assert response.message == ""
        assert response.estimated_duration == 0.0

    def test_session_response_creation(self) -> None:
        """Test creating SessionResponse."""
        from oscura.api.rest_server import SessionResponse

        response = SessionResponse(
            session_id="test_id", status="complete", messages_decoded=10, fields_discovered=5
        )

        assert response.session_id == "test_id"
        assert response.status == "complete"
        assert response.messages_decoded == 10
        assert response.fields_discovered == 5
        assert response.artifacts == {}

    def test_protocol_response_creation(self) -> None:
        """Test creating ProtocolResponse."""
        from oscura.api.rest_server import ProtocolResponse

        response = ProtocolResponse(
            protocol_name="UART",
            confidence=0.95,
            message_count=100,
            field_count=5,
            fields=[{"name": "header", "type": "uint8"}],
        )

        assert response.protocol_name == "UART"
        assert response.confidence == 0.95
        assert response.message_count == 100
        assert len(response.fields) == 1

    def test_error_response_with_timestamp(self) -> None:
        """Test ErrorResponse includes timestamp."""
        from oscura.api.rest_server import ErrorResponse

        error = ErrorResponse(
            error_code="TEST_ERROR", message="Test error message", details={"key": "value"}
        )

        assert error.error_code == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.details["key"] == "value"
        assert error.timestamp != ""  # Should have a timestamp
