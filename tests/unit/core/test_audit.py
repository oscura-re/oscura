"""Tests for audit trail with HMAC chain verification.

Requirements tested:
"""

import json
from datetime import UTC, datetime

import pytest

from oscura.core.audit import (
    AuditEntry,
    AuditTrail,
    get_global_audit_trail,
    record_audit,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestAuditEntry:
    """Test AuditEntry dataclass."""

    def test_create_entry(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            timestamp="2025-12-21T10:00:00.000000Z",
            action="test_action",
            details={"param": "value"},
            user="testuser",
            host="testhost",
            previous_hash="GENESIS",
            hmac="abc123",
        )

        assert entry.timestamp == "2025-12-21T10:00:00.000000Z"
        assert entry.action == "test_action"
        assert entry.details == {"param": "value"}
        assert entry.user == "testuser"
        assert entry.host == "testhost"
        assert entry.previous_hash == "GENESIS"
        assert entry.hmac == "abc123"

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = AuditEntry(
            timestamp="2025-12-21T10:00:00.000000Z",
            action="test_action",
            details={"param": "value"},
            user="testuser",
            host="testhost",
            previous_hash="GENESIS",
            hmac="abc123",
        )

        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["timestamp"] == "2025-12-21T10:00:00.000000Z"
        assert d["action"] == "test_action"
        assert d["details"] == {"param": "value"}
        assert d["user"] == "testuser"

    def test_from_dict(self):
        """Test creating entry from dictionary."""
        d = {
            "timestamp": "2025-12-21T10:00:00.000000Z",
            "action": "test_action",
            "details": {"param": "value"},
            "user": "testuser",
            "host": "testhost",
            "previous_hash": "GENESIS",
            "hmac": "abc123",
        }

        entry = AuditEntry.from_dict(d)
        assert entry.timestamp == d["timestamp"]
        assert entry.action == d["action"]
        assert entry.details == d["details"]


class TestAuditTrail:
    """Test AuditTrail class."""

    def test_create_trail(self):
        """Test creating an audit trail."""
        audit = AuditTrail(secret_key=b"test-secret")
        assert isinstance(audit, AuditTrail)
        assert len(audit._entries) == 0

    def test_record_action_basic(self):
        """Test recording a basic action."""
        audit = AuditTrail(secret_key=b"test-secret")

        entry = audit.record_action(
            "test_action",
            {"param": "value"},
            user="alice",
        )

        assert isinstance(entry, AuditEntry)
        assert entry.action == "test_action"
        assert entry.details == {"param": "value"}
        assert entry.user == "alice"
        assert entry.previous_hash == "GENESIS"
        assert len(entry.hmac) > 0

    def test_record_multiple_actions(self):
        """Test recording multiple actions."""
        audit = AuditTrail(secret_key=b"test-secret")

        entry1 = audit.record_action("action1", {"a": 1})
        entry2 = audit.record_action("action2", {"b": 2})
        entry3 = audit.record_action("action3", {"c": 3})

        assert len(audit._entries) == 3
        assert entry1.previous_hash == "GENESIS"
        assert entry2.previous_hash == entry1.hmac
        assert entry3.previous_hash == entry2.hmac

    def test_verify_integrity_empty(self):
        """Test verifying empty trail."""
        audit = AuditTrail(secret_key=b"test-secret")
        assert audit.verify_integrity() is True

    def test_verify_integrity_valid(self):
        """Test verifying valid trail."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"a": 1})
        audit.record_action("action2", {"b": 2})
        audit.record_action("action3", {"c": 3})

        assert audit.verify_integrity() is True

    def test_verify_integrity_tampered_action(self):
        """Test verifying trail with tampered action."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"a": 1})
        audit.record_action("action2", {"b": 2})
        audit.record_action("action3", {"c": 3})

        # Tamper with middle entry
        audit._entries[1].action = "modified_action"

        assert audit.verify_integrity() is False

    def test_verify_integrity_tampered_details(self):
        """Test verifying trail with tampered details."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"a": 1})
        audit.record_action("action2", {"b": 2})

        # Tamper with details
        audit._entries[1].details["b"] = 999

        assert audit.verify_integrity() is False

    def test_verify_integrity_tampered_timestamp(self):
        """Test verifying trail with tampered timestamp."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"a": 1})
        audit.record_action("action2", {"b": 2})

        # Tamper with timestamp
        audit._entries[1].timestamp = "2000-01-01T00:00:00.000000Z"

        assert audit.verify_integrity() is False

    def test_verify_integrity_broken_chain(self):
        """Test verifying trail with broken chain."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"a": 1})
        audit.record_action("action2", {"b": 2})
        audit.record_action("action3", {"c": 3})

        # Break the chain by modifying previous_hash
        audit._entries[2].previous_hash = "invalid_hash"

        assert audit.verify_integrity() is False

    def test_get_entries_all(self):
        """Test getting all entries."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"a": 1})
        audit.record_action("action2", {"b": 2})
        audit.record_action("action3", {"c": 3})

        entries = audit.get_entries()
        assert len(entries) == 3
        assert entries[0].action == "action1"
        assert entries[1].action == "action2"
        assert entries[2].action == "action3"

    def test_get_entries_by_action_type(self):
        """Test filtering entries by action type."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("load", {"file": "a.bin"})
        audit.record_action("analyze", {"type": "fft"})
        audit.record_action("load", {"file": "b.bin"})
        audit.record_action("analyze", {"type": "thd"})

        load_entries = audit.get_entries(action_type="load")
        assert len(load_entries) == 2
        assert all(e.action == "load" for e in load_entries)

        analyze_entries = audit.get_entries(action_type="analyze")
        assert len(analyze_entries) == 2
        assert all(e.action == "analyze" for e in analyze_entries)

    def test_get_entries_since_timestamp(self):
        """Test filtering entries by timestamp."""
        audit = AuditTrail(secret_key=b"test-secret")

        # Record entries
        audit.record_action("action1", {"a": 1})

        # Get current time
        cutoff = datetime.now(UTC)

        # Record more entries after cutoff
        audit.record_action("action2", {"b": 2})
        audit.record_action("action3", {"c": 3})

        # Get entries since cutoff (should get last 2)
        recent = audit.get_entries(since=cutoff)
        # This might get 2 or 3 depending on timing, so just check >= 2
        assert len(recent) >= 2

    def test_export_json(self, tmp_path):
        """Test exporting audit trail as JSON."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"a": 1})
        audit.record_action("action2", {"b": 2})

        output_path = tmp_path / "audit.json"
        audit.export_audit_log(str(output_path), format="json")

        assert output_path.exists()

        # Verify JSON structure
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "version" in data
        assert "hash_algorithm" in data
        assert "entries" in data
        assert len(data["entries"]) == 2
        assert data["entries"][0]["action"] == "action1"
        assert data["entries"][1]["action"] == "action2"

    def test_export_csv(self, tmp_path):
        """Test exporting audit trail as CSV."""
        audit = AuditTrail(secret_key=b"test-secret")

        audit.record_action("action1", {"param1": "value1"})
        audit.record_action("action2", {"param2": "value2"})

        output_path = tmp_path / "audit.csv"
        audit.export_audit_log(str(output_path), format="csv")

        assert output_path.exists()

        # Verify CSV content
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Check header
        assert "timestamp" in lines[0]
        assert "action" in lines[0]
        assert "hmac" in lines[0]

        # Check we have data rows
        assert len(lines) >= 3  # Header + 2 entries

    def test_export_unsupported_format(self, tmp_path):
        """Test exporting with unsupported format."""
        audit = AuditTrail(secret_key=b"test-secret")
        audit.record_action("action1", {"a": 1})

        output_path = tmp_path / "audit.xml"

        with pytest.raises(ValueError, match="Unsupported format"):
            audit.export_audit_log(str(output_path), format="xml")  # type: ignore

    def test_hmac_sha256(self):
        """Test HMAC with SHA-256 algorithm."""
        audit = AuditTrail(secret_key=b"test-secret", hash_algorithm="sha256")
        entry = audit.record_action("test", {})

        # SHA-256 produces 64 hex characters
        assert len(entry.hmac) == 64

    def test_hmac_sha512(self):
        """Test HMAC with SHA-512 algorithm."""
        audit = AuditTrail(secret_key=b"test-secret", hash_algorithm="sha512")
        entry = audit.record_action("test", {})

        # SHA-512 produces 128 hex characters
        assert len(entry.hmac) == 128

    def test_different_secrets_different_hmacs(self):
        """Test that different secrets produce different HMACs."""
        audit1 = AuditTrail(secret_key=b"secret1")
        audit2 = AuditTrail(secret_key=b"secret2")

        entry1 = audit1.record_action("test", {"a": 1}, user="alice")
        entry2 = audit2.record_action("test", {"a": 1}, user="alice")

        # Same action, different secrets -> different HMACs
        assert entry1.hmac != entry2.hmac

    def test_default_secret_key(self):
        """Test that default secret key is generated."""
        audit1 = AuditTrail()
        audit2 = AuditTrail()

        # Each instance gets its own random key
        entry1 = audit1.record_action("test", {})
        entry2 = audit2.record_action("test", {})

        # Different random keys -> different HMACs
        assert entry1.hmac != entry2.hmac


class TestGlobalAuditTrail:
    """Test global audit trail functions."""

    def test_get_global_audit_trail(self):
        """Test getting global audit trail."""
        # Note: This uses a global singleton, so we need to be careful
        # about state between tests
        audit = get_global_audit_trail(secret_key=b"test-key")
        assert isinstance(audit, AuditTrail)

    def test_record_audit(self):
        """Test recording to global audit trail."""
        # Clear global state by creating a new instance
        import oscura.core.audit as audit_module

        audit_module._global_audit_trail = None

        entry = record_audit("test_action", {"test": "data"})
        assert isinstance(entry, AuditEntry)
        assert entry.action == "test_action"
        assert entry.details == {"test": "data"}

    def test_global_trail_persistence(self):
        """Test that global trail persists across calls."""
        # Clear global state
        import oscura.core.audit as audit_module

        audit_module._global_audit_trail = None

        # Record entries
        record_audit("action1", {})
        record_audit("action2", {})

        # Get trail and verify both entries exist
        audit = get_global_audit_trail()
        entries = audit.get_entries()

        assert len(entries) >= 2
        # Find our entries (there might be others from other tests)
        actions = [e.action for e in entries]
        assert "action1" in actions
        assert "action2" in actions


class TestAuditTrailIntegration:
    """Integration tests for audit trail."""

    def test_complete_workflow(self, tmp_path):
        """Test complete audit trail workflow."""
        # Create audit trail
        audit = AuditTrail(secret_key=b"my-secret-key")

        # Simulate analysis workflow
        audit.record_action(
            "load_trace",
            {"file": "data.bin", "size_bytes": 1024000},
            user="scientist",
        )

        audit.record_action(
            "compute_fft",
            {"samples": 1000000, "window": "hann"},
            user="scientist",
        )

        audit.record_action(
            "measure_thd",
            {"fundamental_freq": 1000.0, "thd_db": -60.5},
            user="scientist",
        )

        audit.record_action(
            "export_results",
            {"format": "csv", "file": "results.csv"},
            user="scientist",
        )

        # Verify integrity
        assert audit.verify_integrity() is True

        # Export
        json_path = tmp_path / "audit.json"
        audit.export_audit_log(str(json_path), format="json")

        csv_path = tmp_path / "audit.csv"
        audit.export_audit_log(str(csv_path), format="csv")

        # Verify exports exist
        assert json_path.exists()
        assert csv_path.exists()

        # Query specific actions
        fft_entries = audit.get_entries(action_type="compute_fft")
        assert len(fft_entries) == 1
        assert fft_entries[0].details["window"] == "hann"

    def test_tamper_detection(self):
        """Test that tampering is reliably detected."""
        audit = AuditTrail(secret_key=b"secure-key")

        # Create audit trail
        for i in range(10):
            audit.record_action(f"action_{i}", {"index": i})

        # Verify it's valid
        assert audit.verify_integrity() is True

        # Try different types of tampering
        original_action = audit._entries[5].action
        audit._entries[5].action = "tampered"
        assert audit.verify_integrity() is False
        audit._entries[5].action = original_action

        original_detail = audit._entries[7].details["index"]
        audit._entries[7].details["index"] = 999
        assert audit.verify_integrity() is False
        audit._entries[7].details["index"] = original_detail

        original_hash = audit._entries[3].hmac
        audit._entries[3].hmac = "fake_hash"
        assert audit.verify_integrity() is False
        audit._entries[3].hmac = original_hash

        # Should be valid again after restoring
        assert audit.verify_integrity() is True
