"""Interactive web dashboard for protocol analysis.

This module provides a comprehensive web-based UI for Oscura, enabling:
- File upload and analysis management
- Real-time progress tracking
- Interactive waveform visualization
- Protocol exploration and browsing
- Export to multiple formats (Wireshark, Scapy, Kaitai)
- Session management interface

The dashboard uses FastAPI for the backend with Jinja2 templates for
HTML rendering. Real-time updates are provided via WebSocket connections.

Example:
    >>> from oscura.api.server import WebDashboard
    >>> dashboard = WebDashboard(host="0.0.0.0", port=5000)
    >>> dashboard.run()  # Starts server on http://0.0.0.0:5000

Architecture:
    - FastAPI backend (REST API + WebSocket)
    - Jinja2 templates for HTML pages
    - Bootstrap CSS framework for responsive UI
    - Plotly.js for interactive waveform visualization
    - Vanilla JavaScript for client-side interactivity
    - Optional dark/light theme support
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import FastAPI and dependencies
try:
    from fastapi import (
        BackgroundTasks,
        FastAPI,
        HTTPException,
        Request,
        UploadFile,
        WebSocket,
        WebSocketDisconnect,
        status,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
    )
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not available. Install with: pip install 'fastapi[all]' uvicorn")


# ============================================================================
# Dashboard Configuration
# ============================================================================


@dataclass
class DashboardConfig:
    """Configuration for web dashboard.

    Attributes:
        title: Dashboard title displayed in UI.
        theme: Default theme (light or dark).
        max_file_size: Maximum upload file size in bytes.
        enable_websocket: Enable WebSocket for real-time updates.
        session_timeout: Session timeout in seconds.
        cache_waveforms: Cache waveform data for faster display.
        plotly_config: Configuration for Plotly.js charts.
    """

    title: str = "Oscura Protocol Analysis Dashboard"
    theme: str = "dark"
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    enable_websocket: bool = True
    session_timeout: float = 3600.0  # 1 hour
    cache_waveforms: bool = True
    plotly_config: dict[str, Any] = field(
        default_factory=lambda: {
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "oscura_waveform",
                "height": 800,
                "width": 1200,
            },
        }
    )


# ============================================================================
# WebSocket Connection Manager
# ============================================================================


class ConnectionManager:
    """Manages WebSocket connections for real-time updates.

    Attributes:
        active_connections: Dict mapping session_id to WebSocket connections.
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept WebSocket connection.

        Args:
            websocket: WebSocket connection.
            session_id: Session identifier.
        """
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Remove WebSocket connection.

        Args:
            websocket: WebSocket connection.
            session_id: Session identifier.
        """
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Send message to all connections for a session.

        Args:
            session_id: Session identifier.
            message: Message to send (will be JSON-encoded).
        """
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send WebSocket message: {e}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all active connections.

        Args:
            message: Message to send (will be JSON-encoded).
        """
        for connections in self.active_connections.values():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast WebSocket message: {e}")


# ============================================================================
# Web Dashboard
# ============================================================================


class WebDashboard:
    """Interactive web dashboard for Oscura protocol analysis.

    Provides comprehensive web-based UI for:
    - File upload and analysis management
    - Real-time progress tracking via WebSocket
    - Interactive waveform visualization with Plotly.js
    - Protocol exploration and browsing
    - Export to multiple formats
    - Session management interface
    - Dark/light theme toggle
    - Responsive mobile-friendly design

    Example:
        >>> dashboard = WebDashboard(host="0.0.0.0", port=5000)
        >>> dashboard.run()
        >>> # Visit http://0.0.0.0:5000 for web UI

    Architecture:
        - FastAPI backend for REST API endpoints
        - Jinja2 templates for server-side rendering
        - WebSocket for real-time progress updates
        - Bootstrap for responsive CSS framework
        - Plotly.js for interactive charts
        - Vanilla JavaScript for client-side logic
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        config: DashboardConfig | None = None,
        api_server: Any | None = None,
    ):
        """Initialize web dashboard.

        Args:
            host: Server host address.
            port: Server port number.
            config: Dashboard configuration.
            api_server: Optional RESTAPIServer instance (for session management).

        Raises:
            ImportError: If FastAPI is not available.
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI required. Install with: pip install 'fastapi[all]' uvicorn")

        self.host = host
        self.port = port
        self.config = config or DashboardConfig()

        # Import REST API server for session management
        if api_server is None:
            from oscura.api.rest_server import RESTAPIServer

            self.api_server = RESTAPIServer(host=host, port=8000)
        else:
            self.api_server = api_server

        # WebSocket connection manager
        self.ws_manager = ConnectionManager()

        # Create FastAPI app with dynamic version from package metadata (SSOT: pyproject.toml)
        try:
            from importlib.metadata import version

            app_version = version("oscura")
        except Exception:
            app_version = "0.0.0+dev"

        self.app = FastAPI(
            title=self.config.title,
            description="Interactive web dashboard for hardware reverse engineering",
            version=app_version,
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup templates and static files
        self._setup_templates()

        # Register routes
        self._register_routes()

    def _setup_templates(self) -> None:
        """Setup Jinja2 templates and static files."""
        # Get path to web module
        web_dir = Path(__file__).parent

        # Templates directory
        templates_dir = web_dir / "templates"
        if not templates_dir.exists():
            templates_dir.mkdir(parents=True)
            logger.warning(f"Templates directory created: {templates_dir}")

        self.templates = Jinja2Templates(directory=str(templates_dir))

        # Static files directory
        static_dir = web_dir / "static"
        if not static_dir.exists():
            static_dir.mkdir(parents=True)
            logger.warning(f"Static directory created: {static_dir}")

        # Mount static files
        try:
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")

    def _register_routes(self) -> None:
        """Register all dashboard routes and endpoints."""
        self._register_page_routes()
        self._register_api_routes()
        self._register_websocket_routes()

    def _register_page_routes(self) -> None:
        """Register HTML page routes."""
        self.app.get("/", response_class=HTMLResponse, tags=["Dashboard"])(self._route_home)
        self.app.get("/sessions", response_class=HTMLResponse, tags=["Dashboard"])(
            self._route_sessions
        )
        self.app.get("/session/{session_id}", response_class=HTMLResponse, tags=["Dashboard"])(
            self._route_session_detail
        )
        self.app.get("/protocols", response_class=HTMLResponse, tags=["Dashboard"])(
            self._route_protocols
        )
        self.app.get("/waveforms/{session_id}", response_class=HTMLResponse, tags=["Dashboard"])(
            self._route_waveforms
        )
        self.app.get("/reports/{session_id}", response_class=HTMLResponse, tags=["Dashboard"])(
            self._route_reports
        )
        self.app.get("/export/{session_id}", response_class=HTMLResponse, tags=["Dashboard"])(
            self._route_export
        )

    def _base_context(self, request: Request, **kwargs: Any) -> dict[str, Any]:
        """Build base template context."""
        return {
            "request": request,
            "title": self.config.title,
            "theme": self.config.theme,
            **kwargs,
        }

    async def _route_home(self, request: Request) -> HTMLResponse:
        """Dashboard home page with file upload."""
        return self.templates.TemplateResponse(
            "home.html",
            self._base_context(
                request, max_file_size_mb=self.config.max_file_size // (1024 * 1024)
            ),
        )

    async def _route_sessions(self, request: Request) -> HTMLResponse:
        """Sessions management page."""
        return self.templates.TemplateResponse(
            "sessions.html",
            self._base_context(request, sessions=self.api_server.session_manager.list_sessions()),
        )

    async def _route_session_detail(self, request: Request, session_id: str) -> HTMLResponse:
        """Session detail page with analysis results."""
        session = self._get_session_or_404(session_id)
        return self.templates.TemplateResponse(
            "session_detail.html",
            self._base_context(
                request, session=session, protocol_spec=self._extract_protocol_spec(session)
            ),
        )

    async def _route_protocols(self, request: Request) -> HTMLResponse:
        """Protocols browser page."""
        return self.templates.TemplateResponse(
            "protocols.html", self._base_context(request, protocols=self._gather_protocol_list())
        )

    async def _route_waveforms(self, request: Request, session_id: str) -> HTMLResponse:
        """Interactive waveform viewer page."""
        session = self._get_session_or_404(session_id)
        return self.templates.TemplateResponse(
            "waveforms.html",
            self._base_context(
                request,
                session_id=session_id,
                filename=session["filename"],
                plotly_config=json.dumps(self.config.plotly_config),
            ),
        )

    async def _route_reports(self, request: Request, session_id: str) -> HTMLResponse:
        """Analysis reports page."""
        session = self._get_session_or_404(session_id)
        report_path = self._extract_report_path(session)
        return self.templates.TemplateResponse(
            "reports.html",
            self._base_context(
                request,
                session_id=session_id,
                report_path=str(report_path) if report_path else None,
            ),
        )

    async def _route_export(self, request: Request, session_id: str) -> HTMLResponse:
        """Export/download page for generated artifacts."""
        session = self._get_session_or_404(session_id)
        return self.templates.TemplateResponse(
            "export.html",
            self._base_context(
                request, session_id=session_id, artifacts=self._extract_artifacts(session)
            ),
        )

    def _register_api_routes(self) -> None:
        """Register API endpoints for AJAX requests."""

        @self.app.post("/api/upload", tags=["API"])
        async def upload_file(
            file: UploadFile,
            background_tasks: BackgroundTasks,
            protocol_hint: str | None = None,
            auto_crc: bool = True,
            detect_crypto: bool = True,
            generate_tests: bool = True,
        ) -> JSONResponse:
            """Upload file and start analysis."""
            self._validate_upload_file(file)
            file_data = await self._read_and_validate_file_size(file)

            options = {
                "protocol_hint": protocol_hint,
                "auto_crc": auto_crc,
                "detect_crypto": detect_crypto,
                "generate_tests": generate_tests,
            }

            session_id = self._create_analysis_session(file.filename, file_data, options)

            # Add background task for analysis (FastAPI auto-injects BackgroundTasks)
            background_tasks.add_task(self._run_analysis_with_updates, session_id)

            return JSONResponse(
                {
                    "session_id": session_id,
                    "status": "processing",
                    "message": "Analysis started",
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

        @self.app.get("/api/session/{session_id}/status", tags=["API"])
        async def get_session_status(session_id: str) -> JSONResponse:
            """Get session status for AJAX polling."""
            session = self._get_session_or_404(session_id)
            return JSONResponse(
                {
                    "session_id": session_id,
                    "status": session["status"],
                    "updated_at": session["updated_at"],
                    "error": session.get("error"),
                }
            )

        @self.app.get("/api/session/{session_id}/waveform", tags=["API"])
        async def get_waveform_data(session_id: str) -> JSONResponse:
            """Get waveform data for Plotly.js visualization."""
            session = self._get_session_or_404(session_id)
            waveform_data = self._generate_waveform_data(session)
            return JSONResponse(waveform_data)

        @self.app.delete("/api/session/{session_id}", tags=["API"])
        async def delete_session_api(session_id: str) -> JSONResponse:
            """Delete session via AJAX."""
            deleted = self.api_server.session_manager.delete_session(session_id)
            if not deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {session_id} not found",
                )

            return JSONResponse(
                {
                    "message": "Session deleted",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        @self.app.get("/api/download/{session_id}/{artifact_type}", tags=["API"])
        async def download_artifact(session_id: str, artifact_type: str) -> FileResponse:
            """Download generated artifact."""
            session = self._get_session_or_404(session_id)
            artifact_path = self._get_artifact_path(session, artifact_type)
            return FileResponse(
                path=str(artifact_path),
                filename=Path(artifact_path).name,
                media_type="application/octet-stream",
            )

    def _register_websocket_routes(self) -> None:
        """Register WebSocket endpoints for real-time updates."""

        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
            """WebSocket connection for real-time analysis updates."""
            await self.ws_manager.connect(websocket, session_id)
            try:
                while True:
                    data = await websocket.receive_text()
                    logger.debug(f"WebSocket received: {data}")
            except WebSocketDisconnect:
                self.ws_manager.disconnect(websocket, session_id)

    def _get_session_or_404(self, session_id: str) -> dict[str, Any]:
        """Get session or raise 404 error.

        Args:
            session_id: Session identifier.

        Returns:
            Session data dictionary.

        Raises:
            HTTPException: If session not found.
        """
        session = self.api_server.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        return session

    def _extract_protocol_spec(self, session: dict[str, Any]) -> dict[str, Any] | None:
        """Extract protocol specification from session result.

        Args:
            session: Session data dictionary.

        Returns:
            Serialized protocol spec or None.
        """
        if session["result"]:
            return self.api_server._serialize_protocol_spec(session["result"])
        return None

    def _extract_report_path(self, session: dict[str, Any]) -> Path | None:
        """Extract report path from session result.

        Args:
            session: Session data dictionary.

        Returns:
            Report path or None.
        """
        if session["result"]:
            result = session["result"]
            return getattr(result, "report_path", None)
        return None

    def _extract_artifacts(self, session: dict[str, Any]) -> dict[str, Any]:
        """Extract artifacts from session result.

        Args:
            session: Session data dictionary.

        Returns:
            Dictionary of artifacts.
        """
        if session["result"]:
            return self.api_server._serialize_artifacts(session["result"])
        return {}

    def _gather_protocol_list(self) -> list[dict[str, Any]]:
        """Gather list of protocols from all sessions.

        Returns:
            List of protocol information dictionaries.
        """
        protocols = []
        for session in self.api_server.session_manager.sessions.values():
            if session["result"]:
                result = session["result"]
                spec = getattr(result, "protocol_spec", None)
                if spec:
                    protocols.append(
                        {
                            "session_id": session["id"],
                            "protocol_name": getattr(spec, "protocol_name", "unknown"),
                            "confidence": getattr(result, "confidence_score", 0.0),
                            "message_count": len(getattr(spec, "messages", [])),
                            "field_count": len(getattr(spec, "fields", [])),
                            "filename": session["filename"],
                            "created_at": session["created_at"],
                        }
                    )
        return protocols

    def _validate_upload_file(self, file: UploadFile) -> None:
        """Validate uploaded file has a filename.

        Args:
            file: Uploaded file object.

        Raises:
            HTTPException: If filename is missing.
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename required",
            )

    async def _read_and_validate_file_size(self, file: UploadFile) -> bytes:
        """Read and validate file size.

        Args:
            file: Uploaded file object.

        Returns:
            File data as bytes.

        Raises:
            HTTPException: If file exceeds maximum size.
        """
        file_data_raw = await file.read()
        # Ensure we have bytes (UploadFile.read() returns bytes | str based on mode)
        file_data = file_data_raw if isinstance(file_data_raw, bytes) else file_data_raw.encode()
        if len(file_data) > self.config.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {self.config.max_file_size} bytes)",
            )
        return file_data

    def _create_analysis_session(
        self, filename: str | None, file_data: bytes, options: dict[str, Any]
    ) -> str:
        """Create analysis session via API server.

        Args:
            filename: Name of uploaded file.
            file_data: File content as bytes.
            options: Analysis options dictionary.

        Returns:
            Session ID string.

        Raises:
            HTTPException: If session creation fails or filename is None.
        """
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename required",
            )
        try:
            return self.api_server.session_manager.create_session(filename, file_data, options)
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e

    def _get_artifact_path(self, session: dict[str, Any], artifact_type: str) -> Path:
        """Get path to specific artifact type.

        Args:
            session: Session data dictionary.
            artifact_type: Type of artifact (dissector, scapy, kaitai, report, tests).

        Returns:
            Path to artifact file.

        Raises:
            HTTPException: If artifact type invalid, result missing, or file not found.
        """
        if not session["result"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No results available",
            )

        artifact_map = {
            "dissector": "dissector_path",
            "scapy": "scapy_layer_path",
            "kaitai": "kaitai_path",
            "report": "report_path",
            "tests": "test_vectors_path",
        }

        if artifact_type not in artifact_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid artifact type: {artifact_type}",
            )

        result = session["result"]
        artifact_path = getattr(result, artifact_map[artifact_type], None)

        if not artifact_path or not Path(artifact_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact not found: {artifact_type}",
            )

        return Path(artifact_path)

    async def _run_analysis_with_updates(self, session_id: str) -> None:
        """Run analysis with WebSocket progress updates.

        Args:
            session_id: Session identifier.
        """
        # Send initial status
        await self.ws_manager.send_message(
            session_id,
            {
                "type": "status",
                "status": "processing",
                "message": "Starting analysis...",
                "progress": 0,
            },
        )

        # Run actual analysis via API server
        try:
            self.api_server._run_analysis(session_id)

            # Send completion message
            await self.ws_manager.send_message(
                session_id,
                {
                    "type": "status",
                    "status": "complete",
                    "message": "Analysis complete",
                    "progress": 100,
                },
            )
        except Exception as e:
            # Send error message
            await self.ws_manager.send_message(
                session_id,
                {
                    "type": "error",
                    "status": "error",
                    "message": str(e),
                    "progress": 0,
                },
            )

    def _generate_waveform_data(self, session: dict[str, Any]) -> dict[str, Any]:
        """Generate waveform data in Plotly.js format.

        Args:
            session: Session data.

        Returns:
            Dict with Plotly.js traces.
        """
        # Simplified waveform generation
        # In production, this would extract actual signal data from the capture
        import numpy as np

        # Generate sample waveform
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)

        return {
            "data": [
                {
                    "x": t.tolist(),
                    "y": signal.tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Signal",
                    "line": {"color": "#00d9ff", "width": 1},
                }
            ],
            "layout": {
                "title": f"Waveform: {session['filename']}",
                "xaxis": {"title": "Time (s)"},
                "yaxis": {"title": "Amplitude"},
                "template": "plotly_dark" if self.config.theme == "dark" else "plotly_white",
                "hovermode": "x unified",
            },
        }

    def run(self, reload: bool = False) -> None:
        """Start the web dashboard server.

        Args:
            reload: Enable auto-reload for development.
        """
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError("uvicorn required. Install with: pip install uvicorn") from e

        logger.info(f"Starting Oscura Web Dashboard on {self.host}:{self.port}")
        logger.info(f"Dashboard URL: http://{self.host}:{self.port}")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            log_level="info",
        )


# ============================================================================
# Command-Line Interface
# ============================================================================


def main() -> None:
    """Command-line interface for web dashboard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Oscura Web Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port number (default: 5000)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default="dark",
        help="UI theme (default: dark)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Create configuration
    config = DashboardConfig(theme=args.theme)

    # Create and run dashboard
    dashboard = WebDashboard(host=args.host, port=args.port, config=config)
    dashboard.run(reload=args.reload)


if __name__ == "__main__":
    main()
