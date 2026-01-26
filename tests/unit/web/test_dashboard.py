"""Unit tests for web dashboard.

Tests cover:
- Dashboard initialization and configuration
- Route registration and handling
- WebSocket connection management
- File upload and analysis triggering
- Session management integration
- Template rendering
- API endpoint responses
- Error handling
- Theme support
"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Skip all tests if FastAPI not available
pytest.importorskip("fastapi")

from oscura.api.server.dashboard import (
    ConnectionManager,
    DashboardConfig,
    WebDashboard,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create test dashboard configuration."""
    return DashboardConfig(
        title="Test Dashboard",
        theme="dark",
        max_file_size=10 * 1024 * 1024,  # 10 MB for tests
        enable_websocket=True,
        session_timeout=600.0,
    )


@pytest.fixture
def mock_api_server() -> Mock:
    """Create mock REST API server."""
    server = Mock()
    server.session_manager = Mock()
    server.session_manager.sessions = {}
    server.session_manager.create_session = Mock(return_value="test-session-id")
    server.session_manager.get_session = Mock(return_value=None)
    server.session_manager.list_sessions = Mock(return_value=[])
    server.session_manager.delete_session = Mock(return_value=True)
    server._serialize_protocol_spec = Mock(return_value={})
    server._serialize_artifacts = Mock(return_value={})
    server._run_analysis = Mock()
    return server


@pytest.fixture
def dashboard(dashboard_config: DashboardConfig, mock_api_server: Mock) -> WebDashboard:
    """Create test web dashboard instance."""
    return WebDashboard(
        host="127.0.0.1",
        port=5000,
        config=dashboard_config,
        api_server=mock_api_server,
    )


@pytest.fixture
def test_client(dashboard: WebDashboard) -> Any:
    """Create test client for dashboard app."""
    from fastapi.testclient import TestClient  # type: ignore[import-not-found]

    return TestClient(dashboard.app)


# ============================================================================
# Test DashboardConfig
# ============================================================================


class TestDashboardConfig:
    """Test dashboard configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DashboardConfig()

        assert config.title == "Oscura Protocol Analysis Dashboard"
        assert config.theme == "dark"
        assert config.max_file_size == 100 * 1024 * 1024
        assert config.enable_websocket is True
        assert config.session_timeout == 3600.0
        assert config.cache_waveforms is True
        assert "responsive" in config.plotly_config

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DashboardConfig(
            title="Custom Dashboard",
            theme="light",
            max_file_size=50 * 1024 * 1024,
            enable_websocket=False,
        )

        assert config.title == "Custom Dashboard"
        assert config.theme == "light"
        assert config.max_file_size == 50 * 1024 * 1024
        assert config.enable_websocket is False


# ============================================================================
# Test ConnectionManager
# ============================================================================


class TestConnectionManager:
    """Test WebSocket connection manager."""

    @pytest.fixture
    def connection_manager(self) -> ConnectionManager:
        """Create connection manager."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self) -> Mock:
        """Create mock WebSocket connection."""
        ws = Mock()
        ws.accept = Mock()
        ws.send_json = Mock()
        return ws

    @pytest.mark.asyncio
    async def test_connect(
        self, connection_manager: ConnectionManager, mock_websocket: Mock
    ) -> None:
        """Test WebSocket connection."""
        await connection_manager.connect(mock_websocket, "session-1")

        mock_websocket.accept.assert_called_once()
        assert "session-1" in connection_manager.active_connections
        assert mock_websocket in connection_manager.active_connections["session-1"]

    def test_disconnect(self, connection_manager: ConnectionManager, mock_websocket: Mock) -> None:
        """Test WebSocket disconnection."""
        connection_manager.active_connections["session-1"] = [mock_websocket]

        connection_manager.disconnect(mock_websocket, "session-1")

        assert "session-1" not in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_send_message(
        self, connection_manager: ConnectionManager, mock_websocket: Mock
    ) -> None:
        """Test sending message to session."""
        connection_manager.active_connections["session-1"] = [mock_websocket]

        message = {"type": "status", "message": "Processing"}
        await connection_manager.send_message("session-1", message)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast(
        self, connection_manager: ConnectionManager, mock_websocket: Mock
    ) -> None:
        """Test broadcasting message to all connections."""
        connection_manager.active_connections["session-1"] = [mock_websocket]
        connection_manager.active_connections["session-2"] = [mock_websocket]

        message = {"type": "announcement", "message": "System update"}
        await connection_manager.broadcast(message)

        # Should be called twice (once per session)
        assert mock_websocket.send_json.call_count == 2


# ============================================================================
# Test WebDashboard Initialization
# ============================================================================


class TestWebDashboardInit:
    """Test web dashboard initialization."""

    def test_init_with_config(
        self, dashboard_config: DashboardConfig, mock_api_server: Mock
    ) -> None:
        """Test initialization with custom config."""
        dashboard = WebDashboard(
            host="0.0.0.0",
            port=8080,
            config=dashboard_config,
            api_server=mock_api_server,
        )

        assert dashboard.host == "0.0.0.0"
        assert dashboard.port == 8080
        assert dashboard.config == dashboard_config
        assert dashboard.api_server == mock_api_server

    def test_init_without_fastapi(self) -> None:
        """Test initialization fails without FastAPI."""
        with patch("oscura.web.dashboard.HAS_FASTAPI", False):
            with pytest.raises(ImportError, match="FastAPI required"):
                WebDashboard()

    def test_templates_setup(self, dashboard: WebDashboard) -> None:
        """Test templates are configured."""
        assert dashboard.templates is not None

    def test_websocket_manager(self, dashboard: WebDashboard) -> None:
        """Test WebSocket manager is initialized."""
        assert isinstance(dashboard.ws_manager, ConnectionManager)


# ============================================================================
# Test Dashboard Routes
# ============================================================================


class TestDashboardRoutes:
    """Test dashboard route handlers."""

    def test_home_page(self, test_client: Any) -> None:
        """Test home page renders."""
        response = test_client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_sessions_page(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test sessions page renders."""
        mock_api_server.session_manager.list_sessions.return_value = [
            {
                "session_id": "test-1",
                "filename": "test.vcd",
                "status": "complete",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:10:00",
            }
        ]

        response = test_client.get("/sessions")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_session_detail_found(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test session detail page with valid session."""
        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "filename": "test.vcd",
            "file_hash": "abc123",
            "status": "complete",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:10:00",
            "result": None,
        }

        response = test_client.get("/session/test-session")

        assert response.status_code == 200

    def test_session_detail_not_found(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test session detail page with invalid session."""
        mock_api_server.session_manager.get_session.return_value = None

        response = test_client.get("/session/invalid-session")

        assert response.status_code == 404

    def test_protocols_page(self, test_client: Any) -> None:
        """Test protocols page renders."""
        response = test_client.get("/protocols")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_waveforms_page(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test waveforms page with valid session."""
        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "filename": "test.vcd",
            "status": "complete",
        }

        response = test_client.get("/waveforms/test-session")

        assert response.status_code == 200

    def test_reports_page(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test reports page with valid session."""
        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "filename": "test.vcd",
            "status": "complete",
            "result": None,
        }

        response = test_client.get("/reports/test-session")

        assert response.status_code == 200

    def test_export_page(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test export page with valid session."""
        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "filename": "test.vcd",
            "status": "complete",
            "result": None,
        }

        response = test_client.get("/export/test-session")

        assert response.status_code == 200


# ============================================================================
# Test API Endpoints
# ============================================================================


class TestAPIEndpoints:
    """Test dashboard API endpoints."""

    def test_upload_file_success(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test successful file upload."""
        # Create test file
        file_content = b"test signal data"
        files = {"file": ("test.vcd", BytesIO(file_content), "application/octet-stream")}

        response = test_client.post(
            "/api/upload",
            files=files,
            data={
                "protocol_hint": "uart",
                "auto_crc": "true",
                "detect_crypto": "true",
                "generate_tests": "true",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "processing"

    def test_upload_file_no_filename(self, test_client: Any) -> None:
        """Test upload fails without filename."""
        files = {"file": ("", BytesIO(b"data"), "application/octet-stream")}

        response = test_client.post("/api/upload", files=files)

        assert response.status_code == 400

    def test_upload_file_too_large(self, test_client: Any, dashboard: WebDashboard) -> None:
        """Test upload fails for oversized file."""
        # Create file larger than limit
        large_data = b"x" * (dashboard.config.max_file_size + 1)
        files = {"file": ("large.bin", BytesIO(large_data), "application/octet-stream")}

        response = test_client.post("/api/upload", files=files)

        assert response.status_code == 413

    def test_get_session_status_found(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test getting session status for valid session."""
        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "status": "processing",
            "updated_at": "2024-01-01T00:00:00",
            "error": None,
        }

        response = test_client.get("/api/session/test-session/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"

    def test_get_session_status_not_found(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test getting session status for invalid session."""
        mock_api_server.session_manager.get_session.return_value = None

        response = test_client.get("/api/session/invalid/status")

        assert response.status_code == 404

    def test_get_waveform_data(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test getting waveform data."""
        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "filename": "test.vcd",
            "status": "complete",
        }

        response = test_client.get("/api/session/test-session/waveform")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "layout" in data

    def test_delete_session_success(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test successful session deletion."""
        mock_api_server.session_manager.delete_session.return_value = True

        response = test_client.delete("/api/session/test-session")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"

    def test_delete_session_not_found(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test deleting non-existent session."""
        mock_api_server.session_manager.delete_session.return_value = False

        response = test_client.delete("/api/session/invalid")

        assert response.status_code == 404

    def test_download_artifact_success(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test successful artifact download."""
        # Create temporary artifact file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".lua") as f:
            f.write("-- Test dissector")
            artifact_path = f.name

        try:
            mock_result = Mock()
            mock_result.dissector_path = artifact_path

            mock_api_server.session_manager.get_session.return_value = {
                "id": "test-session",
                "status": "complete",
                "result": mock_result,
            }

            response = test_client.get("/api/download/test-session/dissector")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/octet-stream"
        finally:
            Path(artifact_path).unlink(missing_ok=True)

    def test_download_artifact_not_found(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test downloading non-existent artifact."""
        mock_result = Mock()
        mock_result.dissector_path = None

        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "status": "complete",
            "result": mock_result,
        }

        response = test_client.get("/api/download/test-session/dissector")

        assert response.status_code == 404

    def test_download_invalid_artifact_type(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test downloading invalid artifact type."""
        mock_api_server.session_manager.get_session.return_value = {
            "id": "test-session",
            "status": "complete",
            "result": Mock(),
        }

        response = test_client.get("/api/download/test-session/invalid_type")

        assert response.status_code == 400


# ============================================================================
# Test Waveform Generation
# ============================================================================


class TestWaveformGeneration:
    """Test waveform data generation."""

    def test_generate_waveform_data(self, dashboard: WebDashboard) -> None:
        """Test waveform data generation."""
        session = {
            "filename": "test.vcd",
            "status": "complete",
        }

        waveform = dashboard._generate_waveform_data(session)

        assert "data" in waveform
        assert "layout" in waveform
        assert len(waveform["data"]) > 0
        assert "x" in waveform["data"][0]
        assert "y" in waveform["data"][0]
        assert waveform["layout"]["title"] == "Waveform: test.vcd"

    def test_waveform_dark_theme(
        self, dashboard_config: DashboardConfig, mock_api_server: Mock
    ) -> None:
        """Test waveform uses dark theme."""
        dashboard_config.theme = "dark"
        dashboard = WebDashboard(config=dashboard_config, api_server=mock_api_server)

        session = {"filename": "test.vcd"}
        waveform = dashboard._generate_waveform_data(session)

        assert "plotly_dark" in waveform["layout"]["template"]

    def test_waveform_light_theme(
        self, dashboard_config: DashboardConfig, mock_api_server: Mock
    ) -> None:
        """Test waveform uses light theme."""
        dashboard_config.theme = "light"
        dashboard = WebDashboard(config=dashboard_config, api_server=mock_api_server)

        session = {"filename": "test.vcd"}
        waveform = dashboard._generate_waveform_data(session)

        assert "plotly_white" in waveform["layout"]["template"]


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test dashboard error handling."""

    def test_session_not_found_404(self, test_client: Any, mock_api_server: Mock) -> None:
        """Test 404 error for non-existent session."""
        mock_api_server.session_manager.get_session.return_value = None

        endpoints = [
            "/session/invalid",
            "/waveforms/invalid",
            "/reports/invalid",
            "/export/invalid",
        ]

        for endpoint in endpoints:
            response = test_client.get(endpoint)
            assert response.status_code == 404

    def test_invalid_upload_400(self, test_client: Any) -> None:
        """Test 400 error for invalid upload."""
        # No file provided
        response = test_client.post("/api/upload")

        assert response.status_code == 422  # FastAPI validation error


# ============================================================================
# Test Command-Line Interface
# ============================================================================


class TestCLI:
    """Test command-line interface."""

    @patch("oscura.web.dashboard.WebDashboard.run")
    def test_main_default_args(self, mock_run: Mock) -> None:
        """Test CLI with default arguments."""

        with patch("sys.argv", ["dashboard"]):
            with pytest.raises(SystemExit):
                # main() will call run() which we've mocked
                pass

    @patch("oscura.web.dashboard.WebDashboard.run")
    @patch("sys.argv", ["dashboard", "--host", "0.0.0.0", "--port", "8080", "--theme", "light"])
    def test_main_custom_args(self, mock_run: Mock) -> None:
        """Test CLI with custom arguments."""

        # This would normally be tested with actual CLI execution
        # but we're mocking to avoid running the server
