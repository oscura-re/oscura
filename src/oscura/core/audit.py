"""Audit trail with HMAC chain verification for compliance and tamper detection.

This module provides tamper-evident audit logging using HMAC signatures
to create a verifiable chain of audit entries.


Example:
    >>> from oscura.core.audit import AuditTrail
    >>> audit = AuditTrail(secret_key=b"my-secret-key")
    >>> audit.record_action("load_trace", {"file": "data.bin"}, user="alice")
    >>> audit.record_action("compute_fft", {"samples": 1000000}, user="alice")
    >>> # Verify integrity
    >>> is_valid = audit.verify_integrity()
    >>> # Export audit log
    >>> audit.export_audit_log("audit.json", format="json")

References:
    LOG-009: Comprehensive Audit Trail for Compliance
    HMAC-SHA256 for tamper detection
"""

from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from oscura.core.logging import format_timestamp


@dataclass
class AuditEntry:
    """Single audit trail entry with HMAC signature.

    Each entry records an auditable action and is linked to the previous
    entry via HMAC chaining for tamper detection.

    Attributes:
        timestamp: ISO 8601 timestamp (UTC) of the action.
        action: Action identifier (e.g., "load_trace", "compute_fft").
        details: Additional details about the action (parameters, results).
        user: Username who performed the action (defaults to current user).
        host: Hostname where the action was performed.
        previous_hash: HMAC of the previous entry (for chain verification).
        hmac: HMAC signature of this entry.

    References:
        LOG-009: Comprehensive Audit Trail for Compliance
    """

    timestamp: str
    action: str
    details: dict[str, Any]
    user: str
    host: str
    previous_hash: str
    hmac: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        """Convert audit entry to dictionary.

        Returns:
            Dictionary representation of the audit entry.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Create audit entry from dictionary.

        Args:
            data: Dictionary containing audit entry data.

        Returns:
            AuditEntry instance.
        """
        return cls(**data)


class AuditTrail:
    """Tamper-evident audit trail with HMAC chain verification.

    Maintains a chain of audit entries where each entry is cryptographically
    linked to the previous entry using HMAC signatures. This allows detection
    of any tampering or modification of the audit log.

    The HMAC chain works as follows:
    1. Each entry contains the HMAC of the previous entry
    2. Each entry's HMAC is computed over: timestamp + action + details + user + previous_hash
    3. Any modification to any entry breaks the chain and fails verification

    Args:
        secret_key: Secret key for HMAC computation (required for tamper detection).
        hash_algorithm: Hash algorithm to use (default: 'sha256').

    Example:
        >>> audit = AuditTrail(secret_key=b"my-secret")
        >>> audit.record_action("operation", {"param": "value"})
        >>> assert audit.verify_integrity()

    References:
        LOG-009: Comprehensive Audit Trail for Compliance
    """

    def __init__(
        self,
        secret_key: bytes | None = None,
        hash_algorithm: Literal["sha256", "sha512"] = "sha256",
    ):
        """Initialize audit trail.

        Args:
            secret_key: Secret key for HMAC computation. If None, a random key is generated.
            hash_algorithm: Hash algorithm to use (sha256 or sha512).
        """
        self._entries: list[AuditEntry] = []
        self._secret_key = secret_key or os.urandom(32)
        self._hash_algorithm = hash_algorithm

    def record_action(
        self,
        action: str,
        details: dict[str, Any],
        user: str | None = None,
    ) -> AuditEntry:
        """Record an auditable action.

        Creates a new audit entry with HMAC signature and adds it to the chain.

        Args:
            action: Action identifier (e.g., "load_trace", "compute_measurement").
            details: Dictionary of action details (parameters, results, etc.).
            user: Username who performed the action (defaults to current user).

        Returns:
            The created AuditEntry.

        Example:
            >>> audit = AuditTrail(secret_key=b"key")
            >>> entry = audit.record_action(
            ...     "load_trace",
            ...     {"file": "data.bin", "size_mb": 100},
            ...     user="alice"
            ... )

        References:
            LOG-009: Comprehensive Audit Trail for Compliance
        """
        # Get current user and host
        if user is None:
            try:
                user = getpass.getuser()
            except Exception:
                user = "unknown"

        try:
            host = socket.gethostname()
        except Exception:
            host = "unknown"

        # Get timestamp
        timestamp = format_timestamp(datetime.now(UTC), format="iso8601")

        # Get previous hash
        previous_hash = self._entries[-1].hmac if self._entries else "GENESIS"

        # Create entry (without HMAC initially)
        entry = AuditEntry(
            timestamp=timestamp,
            action=action,
            details=details,
            user=user,
            host=host,
            previous_hash=previous_hash,
            hmac="",
        )

        # Compute HMAC
        entry.hmac = self._compute_hmac(entry)

        # Add to chain
        self._entries.append(entry)

        return entry

    def verify_integrity(self) -> bool:
        """Verify HMAC chain integrity.

        Verifies that:
        1. Each entry's HMAC is valid
        2. Each entry's previous_hash matches the previous entry's HMAC
        3. No entries have been tampered with or removed

        Returns:
            True if the audit trail is intact and untampered, False otherwise.

        Example:
            >>> audit = AuditTrail(secret_key=b"key")
            >>> audit.record_action("action1", {})
            >>> audit.record_action("action2", {})
            >>> assert audit.verify_integrity()  # Should be True
            >>> # Tampering with an entry would break the chain
            >>> audit._entries[0].action = "modified"
            >>> assert not audit.verify_integrity()  # Should be False

        References:
            LOG-009: Comprehensive Audit Trail for Compliance
        """
        if not self._entries:
            return True  # Empty trail is valid

        for i, entry in enumerate(self._entries):
            # Verify HMAC
            expected_hmac = self._compute_hmac(entry)
            if entry.hmac != expected_hmac:
                return False

            # Verify previous hash linkage
            if i == 0:
                if entry.previous_hash != "GENESIS":
                    return False
            elif entry.previous_hash != self._entries[i - 1].hmac:
                return False

        return True

    def export_audit_log(
        self,
        path: str,
        format: Literal["json", "csv"] = "json",
    ) -> None:
        """Export audit trail to file.

        Args:
            path: Path to export file.
            format: Export format (json or csv).

        Raises:
            ValueError: If format is not supported.

        Example:
            >>> audit = AuditTrail(secret_key=b"key")
            >>> audit.record_action("test", {})
            >>> audit.export_audit_log("audit.json", format="json")

        References:
            LOG-009: Comprehensive Audit Trail for Compliance
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._export_json(path_obj)
        elif format == "csv":
            self._export_csv(path_obj)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_entries(
        self,
        since: datetime | None = None,
        action_type: str | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filtering.

        Args:
            since: Return only entries after this datetime (UTC).
            action_type: Return only entries with this action type.

        Returns:
            List of matching AuditEntry objects.

        Example:
            >>> from datetime import datetime, UTC, timedelta
            >>> audit = AuditTrail(secret_key=b"key")
            >>> audit.record_action("load", {})
            >>> audit.record_action("analyze", {})
            >>> # Get all load actions
            >>> loads = audit.get_entries(action_type="load")
            >>> # Get entries from last hour
            >>> recent = audit.get_entries(since=datetime.now(UTC) - timedelta(hours=1))

        References:
            LOG-009: Comprehensive Audit Trail for Compliance
        """
        results = self._entries.copy()

        # Filter by timestamp
        if since is not None:
            since_str = format_timestamp(since, format="iso8601")
            results = [e for e in results if e.timestamp >= since_str]

        # Filter by action type
        if action_type is not None:
            results = [e for e in results if e.action == action_type]

        return results

    def _compute_hmac(self, entry: AuditEntry) -> str:
        """Compute HMAC signature for an audit entry.

        Args:
            entry: Audit entry to sign.

        Returns:
            Hexadecimal HMAC signature.

        Raises:
            ValueError: If hash algorithm is unsupported.

        References:
            LOG-009: HMAC-based tamper detection
        """
        # Create canonical representation
        canonical = (
            f"{entry.timestamp}|{entry.action}|{json.dumps(entry.details, sort_keys=True)}"
            f"|{entry.user}|{entry.host}|{entry.previous_hash}"
        )

        # Compute HMAC
        if self._hash_algorithm == "sha256":
            h = hmac.new(self._secret_key, canonical.encode("utf-8"), hashlib.sha256)
        elif self._hash_algorithm == "sha512":
            h = hmac.new(self._secret_key, canonical.encode("utf-8"), hashlib.sha512)
        else:
            raise ValueError(f"Unsupported hash algorithm: {self._hash_algorithm}")

        return h.hexdigest()

    def _export_json(self, path: Path) -> None:
        """Export audit trail as JSON.

        Args:
            path: Path to JSON file.
        """
        data = {
            "version": "1.0",
            "hash_algorithm": self._hash_algorithm,
            "entries": [entry.to_dict() for entry in self._entries],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, path: Path) -> None:
        """Export audit trail as CSV.

        Args:
            path: Path to CSV file.
        """
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            if not self._entries:
                return

            # Get all possible detail keys
            detail_keys = set()  # type: ignore[var-annotated]
            for entry in self._entries:
                detail_keys.update(entry.details.keys())
            detail_keys = sorted(detail_keys)  # type: ignore[assignment]

            # Create CSV writer
            fieldnames = [
                "timestamp",
                "action",
                "user",
                "host",
                "previous_hash",
                "hmac",
            ] + [f"detail_{k}" for k in detail_keys]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write entries
            for entry in self._entries:
                row = {
                    "timestamp": entry.timestamp,
                    "action": entry.action,
                    "user": entry.user,
                    "host": entry.host,
                    "previous_hash": entry.previous_hash,
                    "hmac": entry.hmac,
                }
                # Add details
                for key in detail_keys:
                    value = entry.details.get(key)
                    row[f"detail_{key}"] = json.dumps(value) if value is not None else ""

                writer.writerow(row)


# Convenience function for global audit trail
_global_audit_trail: AuditTrail | None = None


def get_global_audit_trail(secret_key: bytes | None = None) -> AuditTrail:
    """Get or create the global audit trail.

    Args:
        secret_key: Secret key for HMAC computation (only used on first call).

    Returns:
        Global AuditTrail instance.

    Example:
        >>> from oscura.core.audit import get_global_audit_trail
        >>> audit = get_global_audit_trail(secret_key=b"my-key")
        >>> audit.record_action("test", {})

    References:
        LOG-009: Comprehensive Audit Trail for Compliance
    """
    global _global_audit_trail
    if _global_audit_trail is None:
        _global_audit_trail = AuditTrail(secret_key=secret_key)
    return _global_audit_trail


def record_audit(action: str, details: dict[str, Any], user: str | None = None) -> AuditEntry:
    """Record an action to the global audit trail.

    Convenience function for recording to the global audit trail.

    Args:
        action: Action identifier.
        details: Action details.
        user: Username (defaults to current user).

    Returns:
        Created AuditEntry.

    Example:
        >>> from oscura.core.audit import record_audit
        >>> record_audit("compute_fft", {"samples": 1000000})

    References:
        LOG-009: Comprehensive Audit Trail for Compliance
    """
    audit = get_global_audit_trail()
    return audit.record_action(action, details, user)


__all__ = [
    "AuditEntry",
    "AuditTrail",
    "get_global_audit_trail",
    "record_audit",
]
