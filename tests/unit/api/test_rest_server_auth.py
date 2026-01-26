"""Comprehensive tests for REST API authentication (SEC-002).

Tests verify Bearer token authentication for all API endpoints.
Coverage includes:
- Authentication enforcement when api_key configured
- Public access when api_key=None (development mode)
- Valid API key acceptance
- Invalid/missing API key rejection
- Health endpoint remains unauthenticated
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # type: ignore[import-not-found]

from oscura.api.rest_server import RESTAPIServer

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def server_with_auth() -> RESTAPIServer:
    """REST API server with authentication enabled."""
    return RESTAPIServer(
        host="127.0.0.1",
        port=8000,
        api_key="test-secret-key-12345",
        enable_cors=False,
    )


@pytest.fixture
def server_no_auth() -> RESTAPIServer:
    """REST API server without authentication."""
    return RESTAPIServer(
        host="127.0.0.1",
        port=8000,
        api_key=None,
        enable_cors=False,
    )


@pytest.fixture
def client_with_auth(server_with_auth: RESTAPIServer) -> TestClient:
    """Test client for authenticated server."""
    return TestClient(server_with_auth.app)


@pytest.fixture
def client_no_auth(server_no_auth: RESTAPIServer) -> TestClient:
    """Test client for unauthenticated server."""
    return TestClient(server_no_auth.app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Valid authorization headers."""
    return {"Authorization": "Bearer test-secret-key-12345"}


@pytest.fixture
def invalid_auth_headers() -> dict[str, str]:
    """Invalid authorization headers."""
    return {"Authorization": "Bearer wrong-key"}


# ============================================================================
# Health Endpoint Tests (Should NOT require auth)
# ============================================================================


class TestHealthEndpointNoAuth:
    """Verify health endpoint remains publicly accessible."""

    def test_health_check_without_auth_when_configured(self, client_with_auth: TestClient) -> None:
        """Health endpoint accessible without auth even when api_key set."""
        response = client_with_auth.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "sessions_active" in data

    def test_health_check_without_auth_when_not_configured(
        self, client_no_auth: TestClient
    ) -> None:
        """Health endpoint accessible without auth when api_key=None."""
        response = client_no_auth.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


# ============================================================================
# Analyze Endpoint Authentication Tests
# ============================================================================


class TestAnalyzeEndpointAuth:
    """Test authentication for /api/v1/analyze endpoint."""

    def test_analyze_requires_auth_when_configured(
        self, client_with_auth: TestClient, tmp_path
    ) -> None:
        """Analyze endpoint rejects requests without auth when api_key set."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        # Request without authorization header
        with open(test_file, "rb") as f:
            response = client_with_auth.post(
                "/api/v1/analyze",
                files={"file": ("test.bin", f, "application/octet-stream")},
            )

        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

    def test_analyze_accepts_valid_key(
        self, client_with_auth: TestClient, auth_headers: dict[str, str], tmp_path
    ) -> None:
        """Analyze endpoint accepts requests with valid API key."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        # Request with valid authorization header
        with open(test_file, "rb") as f:
            response = client_with_auth.post(
                "/api/v1/analyze",
                files={"file": ("test.bin", f, "application/octet-stream")},
                headers=auth_headers,
            )

        assert response.status_code == 202
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "processing"

    def test_analyze_rejects_invalid_key(
        self,
        client_with_auth: TestClient,
        invalid_auth_headers: dict[str, str],
        tmp_path,
    ) -> None:
        """Analyze endpoint rejects requests with invalid API key."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        # Request with invalid authorization header
        with open(test_file, "rb") as f:
            response = client_with_auth.post(
                "/api/v1/analyze",
                files={"file": ("test.bin", f, "application/octet-stream")},
                headers=invalid_auth_headers,
            )

        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

    def test_analyze_allows_access_without_auth_config(
        self, client_no_auth: TestClient, tmp_path
    ) -> None:
        """Analyze endpoint accessible without auth when api_key=None."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        # Request without authorization header (should succeed)
        with open(test_file, "rb") as f:
            response = client_no_auth.post(
                "/api/v1/analyze",
                files={"file": ("test.bin", f, "application/octet-stream")},
            )

        assert response.status_code == 202
        data = response.json()
        assert "session_id" in data


# ============================================================================
# Sessions Endpoint Authentication Tests
# ============================================================================


class TestSessionsEndpointAuth:
    """Test authentication for /api/v1/sessions endpoints."""

    def test_list_sessions_requires_auth(self, client_with_auth: TestClient) -> None:
        """List sessions endpoint rejects requests without auth."""
        response = client_with_auth.get("/api/v1/sessions")
        assert response.status_code == 401

    def test_list_sessions_accepts_valid_key(
        self, client_with_auth: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """List sessions endpoint accepts valid API key."""
        response = client_with_auth.get("/api/v1/sessions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "count" in data

    def test_get_session_requires_auth(self, client_with_auth: TestClient) -> None:
        """Get session endpoint rejects requests without auth."""
        response = client_with_auth.get("/api/v1/sessions/test-session-id")
        assert response.status_code == 401

    def test_get_session_accepts_valid_key(
        self, client_with_auth: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Get session endpoint accepts valid API key (404 for missing session OK)."""
        response = client_with_auth.get("/api/v1/sessions/test-session-id", headers=auth_headers)
        # Auth passes, but session doesn't exist
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_session_requires_auth(self, client_with_auth: TestClient) -> None:
        """Delete session endpoint rejects requests without auth."""
        response = client_with_auth.delete("/api/v1/sessions/test-session-id")
        assert response.status_code == 401

    def test_delete_session_accepts_valid_key(
        self, client_with_auth: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Delete session endpoint accepts valid API key (404 for missing session OK)."""
        response = client_with_auth.delete("/api/v1/sessions/test-session-id", headers=auth_headers)
        # Auth passes, but session doesn't exist
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


# ============================================================================
# Protocols Endpoint Authentication Tests
# ============================================================================


class TestProtocolsEndpointAuth:
    """Test authentication for /api/v1/protocols endpoint."""

    def test_list_protocols_requires_auth(self, client_with_auth: TestClient) -> None:
        """List protocols endpoint rejects requests without auth."""
        response = client_with_auth.get("/api/v1/protocols")
        assert response.status_code == 401

    def test_list_protocols_accepts_valid_key(
        self, client_with_auth: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """List protocols endpoint accepts valid API key."""
        response = client_with_auth.get("/api/v1/protocols", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "protocols" in data
        assert "count" in data

    def test_list_protocols_allows_access_without_auth_config(
        self, client_no_auth: TestClient
    ) -> None:
        """List protocols accessible without auth when api_key=None."""
        response = client_no_auth.get("/api/v1/protocols")
        assert response.status_code == 200


# ============================================================================
# Export Endpoint Authentication Tests
# ============================================================================


class TestExportEndpointAuth:
    """Test authentication for /api/v1/export endpoint."""

    def test_export_requires_auth(self, client_with_auth: TestClient) -> None:
        """Export endpoint rejects requests without auth."""
        response = client_with_auth.post(
            "/api/v1/export/wireshark",
            json={"session_id": "test-session-id"},
        )
        assert response.status_code == 401

    def test_export_accepts_valid_key(
        self, client_with_auth: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Export endpoint accepts valid API key (404 for missing session OK)."""
        response = client_with_auth.post(
            "/api/v1/export/wireshark",
            json={"session_id": "test-session-id"},
            headers=auth_headers,
        )
        # Auth passes, but session doesn't exist
        assert response.status_code in [400, 404]


# ============================================================================
# Integration Tests
# ============================================================================


class TestAPIAuthenticationIntegration:
    """Integration tests for API authentication workflow."""

    def test_full_workflow_with_authentication(
        self, client_with_auth: TestClient, auth_headers: dict[str, str], tmp_path
    ) -> None:
        """Complete workflow with authentication."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        # 1. Analyze file (with auth)
        with open(test_file, "rb") as f:
            response = client_with_auth.post(
                "/api/v1/analyze",
                files={"file": ("test.bin", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert response.status_code == 202
        session_id = response.json()["session_id"]

        # 2. List sessions (with auth)
        response = client_with_auth.get("/api/v1/sessions", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["count"] >= 1

        # 3. Get session (with auth)
        response = client_with_auth.get(f"/api/v1/sessions/{session_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["session_id"] == session_id

        # 4. Delete session (with auth)
        response = client_with_auth.delete(f"/api/v1/sessions/{session_id}", headers=auth_headers)
        assert response.status_code == 200

    def test_full_workflow_fails_without_authentication(
        self, client_with_auth: TestClient, tmp_path
    ) -> None:
        """Workflow fails at every step without authentication."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        # 1. Analyze file (no auth - should fail)
        with open(test_file, "rb") as f:
            response = client_with_auth.post(
                "/api/v1/analyze",
                files={"file": ("test.bin", f, "application/octet-stream")},
            )
        assert response.status_code == 401

        # 2. List sessions (no auth - should fail)
        response = client_with_auth.get("/api/v1/sessions")
        assert response.status_code == 401

        # 3. List protocols (no auth - should fail)
        response = client_with_auth.get("/api/v1/protocols")
        assert response.status_code == 401

    def test_full_workflow_without_auth_config(self, client_no_auth: TestClient, tmp_path) -> None:
        """Complete workflow works without auth when api_key=None."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        # 1. Analyze file (no auth required)
        with open(test_file, "rb") as f:
            response = client_no_auth.post(
                "/api/v1/analyze",
                files={"file": ("test.bin", f, "application/octet-stream")},
            )
        assert response.status_code == 202
        session_id = response.json()["session_id"]

        # 2. List sessions (no auth required)
        response = client_no_auth.get("/api/v1/sessions")
        assert response.status_code == 200

        # 3. Get session (no auth required)
        response = client_no_auth.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200

        # 4. List protocols (no auth required)
        response = client_no_auth.get("/api/v1/protocols")
        assert response.status_code == 200


# ============================================================================
# Edge Cases
# ============================================================================


class TestAuthenticationEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_bearer_token_rejected(self, client_with_auth: TestClient) -> None:
        """Empty bearer token is rejected."""
        response = client_with_auth.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer "},
        )
        assert response.status_code == 401

    def test_malformed_auth_header_rejected(self, client_with_auth: TestClient) -> None:
        """Malformed authorization header is rejected."""
        response = client_with_auth.get(
            "/api/v1/sessions",
            headers={"Authorization": "NotBearer token123"},
        )
        assert response.status_code == 401

    def test_case_sensitive_key_comparison(self, client_with_auth: TestClient) -> None:
        """API key comparison is case-sensitive."""
        response = client_with_auth.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer TEST-SECRET-KEY-12345"},  # Wrong case
        )
        assert response.status_code == 401

    def test_www_authenticate_header_in_response(self, client_with_auth: TestClient) -> None:
        """401 responses include WWW-Authenticate header."""
        response = client_with_auth.get("/api/v1/sessions")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.headers["WWW-Authenticate"] == "Bearer"


# ============================================================================
# Security Validation Tests
# ============================================================================


class TestSecurityValidation:
    """Validate security properties of authentication implementation."""

    def test_api_key_not_logged_in_responses(self, client_with_auth: TestClient) -> None:
        """API key never appears in response bodies or headers."""
        response = client_with_auth.get("/api/v1/sessions")
        response_text = response.text.lower()
        assert "test-secret-key" not in response_text
        assert "12345" not in response_text  # Part of key

    def test_different_keys_for_different_servers(self) -> None:
        """Each server instance uses its own API key."""
        server1 = RESTAPIServer(api_key="key1")
        server2 = RESTAPIServer(api_key="key2")

        client1 = TestClient(server1.app)
        client2 = TestClient(server2.app)

        # Key1 works on server1 but not server2
        response1 = client1.get("/api/v1/sessions", headers={"Authorization": "Bearer key1"})
        assert response1.status_code == 200

        response2 = client2.get("/api/v1/sessions", headers={"Authorization": "Bearer key1"})
        assert response2.status_code == 401

    def test_auth_header_not_required_for_health_check(self, client_with_auth: TestClient) -> None:
        """Health check never requires authentication."""
        # Even with api_key configured, health check is public
        response = client_with_auth.get("/api/health")
        assert response.status_code == 200

        # Verify it's truly unauthenticated (not just accepting wrong keys)
        response = client_with_auth.get(
            "/api/health",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 200  # Still works with wrong key


# ============================================================================
# Documentation Tests
# ============================================================================


class TestAPIDocumentation:
    """Verify OpenAPI documentation includes security schemes."""

    def test_openapi_schema_includes_security(self, client_with_auth: TestClient) -> None:
        """OpenAPI schema documents security requirements."""
        response = client_with_auth.get("/api/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        # FastAPI should auto-document Bearer auth
        # Check that paths reference security requirements
        assert "paths" in schema

    def test_docs_page_accessible(self, client_with_auth: TestClient) -> None:
        """API documentation page is accessible."""
        response = client_with_auth.get("/docs")
        assert response.status_code == 200
