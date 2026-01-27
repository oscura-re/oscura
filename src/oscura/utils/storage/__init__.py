"""Storage backends for persisting analysis results.

This module provides database backends for storing and querying
reverse engineering session data, protocol analysis results,
and decoded messages.

Example:
    >>> from oscura.utils.storage import DatabaseBackend, DatabaseConfig
    >>>
    >>> # Create database backend
    >>> config = DatabaseConfig(url="sqlite:///analysis.db")
    >>> db = DatabaseBackend(config)
    >>>
    >>> # Create project and session
    >>> project_id = db.create_project("IoT Device RE", "Unknown protocol analysis")
    >>> session_id = db.create_session(project_id, "blackbox", {"capture": "device.bin"})
    >>>
    >>> # Store protocol analysis
    >>> protocol_id = db.store_protocol(
    ...     session_id,
    ...     name="IoT Protocol",
    ...     spec_json={"fields": [...]},
    ...     confidence=0.85
    ... )
    >>>
    >>> # Store decoded messages
    >>> db.store_message(protocol_id, timestamp=0.0, data=b"\\xaa\\x55", decoded={"id": 1})
    >>>
    >>> # Query results
    >>> protocols = db.find_protocols(min_confidence=0.8)
    >>> sessions = db.get_sessions(project_id)
    >>> messages = db.query_messages(protocol_id, time_range=(0.0, 1.0))

Available classes:
    - DatabaseConfig: Configuration dataclass
    - DatabaseBackend: Main database interface
    - Project/Session/Protocol/Message: Result dataclasses
    - QueryResult: Paginated query results
"""

from oscura.utils.storage.database import (
    AnalysisResult,
    DatabaseBackend,
    DatabaseConfig,
    Message,
    Project,
    Protocol,
    QueryResult,
    Session,
)

__all__ = [
    "AnalysisResult",
    "DatabaseBackend",
    "DatabaseConfig",
    "Message",
    "Project",
    "Protocol",
    "QueryResult",
    "Session",
]
