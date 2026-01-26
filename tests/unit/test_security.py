"""Security tests for Oscura.

Tests for cryptographic features, tamper detection, and security boundaries.
"""

from __future__ import annotations

import gzip
import hashlib
import tempfile
from pathlib import Path

import pytest

from oscura.core.audit import AuditTrail
from oscura.core.exceptions import SecurityError
from oscura.sessions.legacy import Session, load_session

pytestmark = pytest.mark.unit


class TestSessionSecurity:
    """Test Session save/load security features."""

    def test_session_save_uncompressed_includes_signature(self) -> None:
        """Test that uncompressed sessions include HMAC signature."""
        session = Session(name="test")

        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False) as f:
            session_path = Path(f.name)

        try:
            # Save without compression
            session.save(str(session_path), compress=False)

            # Read raw file and verify signature is present
            with open(session_path, "rb") as f:
                content = f.read()

            # Should start with magic bytes
            assert content[:4] == b"OSC1"

            # Should have signature after magic bytes
            # Format: magic (4) + signature (32) + pickled data
            assert len(content) > 36  # At least magic + signature
        finally:
            session_path.unlink()

    def test_session_load_verifies_signature(self) -> None:
        """Test that loading sessions verifies HMAC signature."""
        session = Session(name="test")

        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False) as f:
            session_path = Path(f.name)

        try:
            # Save session (compressed by default)
            session.save(str(session_path))

            # Load should succeed with valid signature
            loaded = load_session(str(session_path))
            assert loaded.name == "test"
        finally:
            session_path.unlink()

    def test_tampered_uncompressed_session_rejected(self) -> None:
        """Test that tampered uncompressed sessions are rejected."""
        session = Session(name="test")

        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False) as f:
            session_path = Path(f.name)

        try:
            # Save without compression for easier tampering
            session.save(str(session_path), compress=False)

            # Tamper with the file
            with open(session_path, "rb") as f:
                content = bytearray(f.read())

            # Modify pickled data (after magic + signature)
            if len(content) > 40:
                content[-10] ^= 0xFF  # Flip some bits

            with open(session_path, "wb") as f:
                f.write(content)

            # Load should fail with SecurityError
            with pytest.raises(SecurityError, match="signature verification failed"):
                load_session(str(session_path))
        finally:
            session_path.unlink()


class TestAuditTrailSecurity:
    """Test AuditTrail HMAC chain security."""

    def test_audit_trail_hmac_chain(self) -> None:
        """Test that audit trail maintains HMAC chain."""
        trail = AuditTrail(secret_key=b"test-key")

        # Add first entry
        entry1 = trail.record_action("action1", {"param": "value1"})

        # Verify first entry has HMAC
        assert entry1.hmac != ""
        assert entry1.previous_hash == "GENESIS"  # First entry has GENESIS

        # Add second entry
        entry2 = trail.record_action("action2", {"param": "value2"})

        # Verify chain
        assert entry2.hmac != ""
        assert entry2.previous_hash == entry1.hmac  # Chained to previous

    def test_audit_trail_tamper_detection(self) -> None:
        """Test that audit trail detects tampering."""
        trail = AuditTrail(secret_key=b"test-key")

        trail.record_action("action1", {"param": "value1"})
        trail.record_action("action2", {"param": "value2"})
        trail.record_action("action3", {"param": "value3"})

        # Verify integrity before tampering
        assert trail.verify_integrity()

        # Tamper with middle entry
        entries = trail.get_entries()
        entries[1].details["param"] = "TAMPERED"

        # Verification should fail
        assert not trail.verify_integrity()

    def test_audit_trail_missing_entry_detection(self) -> None:
        """Test that audit trail detects missing entries."""
        trail = AuditTrail(secret_key=b"test-key")

        trail.record_action("action1", {"param": "value1"})
        trail.record_action("action2", {"param": "value2"})
        trail.record_action("action3", {"param": "value3"})

        # Verify before modification
        assert trail.verify_integrity()

        # Remove middle entry (access private attribute for testing)
        del trail._entries[1]

        # Verification should fail (broken chain)
        assert not trail.verify_integrity()

    def test_audit_trail_reordering_detection(self) -> None:
        """Test that audit trail detects reordered entries."""
        trail = AuditTrail(secret_key=b"test-key")

        trail.record_action("action1", {"param": "value1"})
        trail.record_action("action2", {"param": "value2"})
        trail.record_action("action3", {"param": "value3"})

        # Verify before modification
        assert trail.verify_integrity()

        # Swap entries (access private attribute for testing)
        trail._entries[0], trail._entries[1] = trail._entries[1], trail._entries[0]

        # Verification should fail
        assert not trail.verify_integrity()


class TestSecurityError:
    """Test SecurityError exception."""

    def test_security_error_raised(self) -> None:
        """Test that SecurityError can be raised."""
        with pytest.raises(SecurityError):
            raise SecurityError("Test security violation")

    def test_security_error_message(self) -> None:
        """Test SecurityError message handling."""
        try:
            raise SecurityError("Authentication failed")
        except SecurityError as e:
            assert "Authentication failed" in str(e)

    def test_security_error_inheritance(self) -> None:
        """Test that SecurityError inherits from correct base."""
        from oscura.core.exceptions import OscuraError

        err = SecurityError("test")
        assert isinstance(err, OscuraError)


class TestPickleSecurity:
    """Test pickle serialization security."""

    def test_hmac_key_is_constant(self) -> None:
        """Test that HMAC key is deterministic."""
        from oscura.sessions.legacy.session import _SECURITY_KEY

        # Key should be derived from constant string
        expected = hashlib.sha256(b"oscura-session-v1").digest()
        assert expected == _SECURITY_KEY

    def test_session_magic_bytes(self) -> None:
        """Test that session files have correct magic bytes."""
        from oscura.sessions.legacy.session import _SESSION_MAGIC

        assert _SESSION_MAGIC == b"OSC1"
        assert len(_SESSION_MAGIC) == 4

    def test_compressed_session_security(self) -> None:
        """Test that compressed sessions maintain security."""
        session = Session(name="test")

        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False) as f:
            session_path = Path(f.name)

        try:
            # Save with compression (default)
            session.save(str(session_path), compress=True)

            # File should be gzip compressed
            with gzip.open(session_path, "rb") as f:
                content = f.read()

            # Inside compressed file should have magic bytes
            assert content[:4] == b"OSC1"

            # Load should verify signature
            loaded = load_session(str(session_path))
            assert loaded.name == "test"
        finally:
            session_path.unlink()


class TestInputValidation:
    """Test security boundary validation."""

    def test_session_name_sanitization(self) -> None:
        """Test that session names are validated."""
        # Should accept normal names
        session = Session(name="valid_name_123")
        assert session.name == "valid_name_123"

        # Should accept names with spaces
        session = Session(name="Valid Session Name")
        assert session.name == "Valid Session Name"

    def test_file_path_validation(self) -> None:
        """Test that file paths are validated during load."""
        # Non-existent file should raise appropriate error
        with pytest.raises(FileNotFoundError):
            load_session("/nonexistent/path/file.tks")

    def test_invalid_session_format_rejected(self) -> None:
        """Test that invalid session formats are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False, mode="wb") as f:
            session_path = Path(f.name)
            # Write invalid content
            f.write(b"INVALID_DATA_NOT_A_SESSION_FILE")

        try:
            # Should raise SecurityError for files without HMAC signature
            with pytest.raises(SecurityError, match="HMAC signature"):
                load_session(str(session_path))
        finally:
            session_path.unlink()
