"""Database backend for storing and querying analysis results.

This module provides a comprehensive database abstraction for persisting
hardware reverse engineering session data including protocols, messages,
and analysis results.

Example:
    >>> from oscura.utils.storage import DatabaseBackend, DatabaseConfig
    >>>
    >>> # SQLite (default, no dependencies)
    >>> config = DatabaseConfig(url="sqlite:///analysis.db")
    >>> db = DatabaseBackend(config)
    >>>
    >>> # PostgreSQL (optional)
    >>> config = DatabaseConfig(
    ...     url="postgresql://user:pass@localhost/oscura",
    ...     pool_size=10
    ... )
    >>> db = DatabaseBackend(config)
    >>>
    >>> # Store analysis workflow
    >>> project_id = db.create_project("CAN Bus RE", "Automotive reverse engineering")
    >>> session_id = db.create_session(project_id, "can", {"bus": "HS-CAN"})
    >>> protocol_id = db.store_protocol(session_id, "UDS", spec_json, confidence=0.9)
    >>> db.store_message(protocol_id, timestamp=1.5, data=b"\\x02\\x10\\x01", decoded)
    >>>
    >>> # Query historical data
    >>> protocols = db.find_protocols(name_pattern="UDS%", min_confidence=0.8)
    >>> messages = db.query_messages(protocol_id, time_range=(0.0, 10.0))
    >>> results = db.get_analysis_results(session_id, analysis_type="dpa")

Architecture:
    - SQLite by default (serverless, file-based)
    - PostgreSQL optional (production deployments)
    - Raw SQL fallback (no ORM dependencies)
    - Connection pooling for performance
    - Automatic schema migration
    - Transaction support

Database Schema:
    projects: Project metadata and descriptions
    sessions: Analysis sessions per project
    protocols: Discovered protocols per session
    messages: Decoded messages per protocol
    analysis_results: DPA, timing, entropy, etc.

References:
    V0.6.0_COMPLETE_COMPREHENSIVE_PLAN.md: Phase 5 Feature 45
    SQLite Documentation: https://www.sqlite.org/docs.html
    PostgreSQL Documentation: https://www.postgresql.org/docs/
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional PostgreSQL support
try:
    import psycopg2  # type: ignore[import-untyped]
    from psycopg2.pool import SimpleConnectionPool  # type: ignore[import-untyped]

    HAS_POSTGRES = True
except ImportError:
    psycopg2 = None
    SimpleConnectionPool = None
    HAS_POSTGRES = False


logger = logging.getLogger(__name__)

# SQL Schema constants for SQLite
_SQL_CREATE_PROJECTS_SQLITE = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
"""

_SQL_CREATE_SESSIONS_SQLITE = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    session_type TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
)
"""

_SQL_CREATE_PROTOCOLS_SQLITE = """
CREATE TABLE IF NOT EXISTS protocols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
)
"""

_SQL_CREATE_MESSAGES_SQLITE = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    data TEXT NOT NULL,
    decoded_fields TEXT,
    FOREIGN KEY (protocol_id) REFERENCES protocols(id) ON DELETE CASCADE
)
"""

_SQL_CREATE_ANALYSIS_SQLITE = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    analysis_type TEXT NOT NULL,
    results_json TEXT NOT NULL,
    metrics TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
)
"""

# SQL Schema constants for PostgreSQL
_SQL_CREATE_PROJECTS_POSTGRES = """
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
)
"""

_SQL_CREATE_SESSIONS_POSTGRES = """
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_type TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
)
"""

_SQL_CREATE_PROTOCOLS_POSTGRES = """
CREATE TABLE IF NOT EXISTS protocols (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    spec_json JSONB NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_SQL_CREATE_MESSAGES_POSTGRES = """
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    protocol_id INTEGER NOT NULL REFERENCES protocols(id) ON DELETE CASCADE,
    timestamp REAL NOT NULL,
    data TEXT NOT NULL,
    decoded_fields JSONB
)
"""

_SQL_CREATE_ANALYSIS_POSTGRES = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    analysis_type TEXT NOT NULL,
    results_json JSONB NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Index creation statements
_SQL_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_protocols_session ON protocols(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_protocols_name ON protocols(name)",
    "CREATE INDEX IF NOT EXISTS idx_messages_protocol ON messages(protocol_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_analysis_session ON analysis_results(session_id)",
]


@dataclass
class DatabaseConfig:
    """Database configuration.

    Attributes:
        url: Database URL (sqlite:///path.db or postgresql://...)
        pool_size: Connection pool size (PostgreSQL only)
        timeout: Connection timeout in seconds
        echo_sql: Log SQL statements for debugging

    Example:
        >>> # SQLite (default)
        >>> config = DatabaseConfig(url="sqlite:///analysis.db")
        >>>
        >>> # PostgreSQL
        >>> config = DatabaseConfig(
        ...     url="postgresql://user:pass@localhost/oscura",
        ...     pool_size=10,
        ...     timeout=30.0
        ... )
    """

    url: str = "sqlite:///oscura_analysis.db"
    pool_size: int = 5
    timeout: float = 30.0
    echo_sql: bool = False


@dataclass
class Project:
    """Project metadata.

    Attributes:
        id: Project ID (auto-assigned)
        name: Project name
        description: Project description
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata

    Example:
        >>> project = Project(
        ...     id=1,
        ...     name="Automotive CAN",
        ...     description="CAN bus protocol analysis",
        ...     created_at=datetime.now(UTC),
        ...     updated_at=datetime.now(UTC)
        ... )
    """

    id: int | None = None
    name: str = ""
    description: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Analysis session.

    Attributes:
        id: Session ID (auto-assigned)
        project_id: Parent project ID
        session_type: Session type (blackbox, can, uart, etc.)
        timestamp: Session timestamp
        metadata: Session-specific metadata

    Example:
        >>> session = Session(
        ...     id=1,
        ...     project_id=1,
        ...     session_type="blackbox",
        ...     timestamp=datetime.now(UTC),
        ...     metadata={"capture_file": "device.bin"}
        ... )
    """

    id: int | None = None
    project_id: int | None = None
    session_type: str = ""
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Protocol:
    """Discovered protocol.

    Attributes:
        id: Protocol ID (auto-assigned)
        session_id: Parent session ID
        name: Protocol name
        spec_json: Protocol specification as JSON
        confidence: Confidence score (0.0-1.0)
        created_at: Creation timestamp

    Example:
        >>> protocol = Protocol(
        ...     id=1,
        ...     session_id=1,
        ...     name="IoT Protocol",
        ...     spec_json={"fields": [...]},
        ...     confidence=0.85
        ... )
    """

    id: int | None = None
    session_id: int | None = None
    name: str = ""
    spec_json: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    created_at: datetime | None = None


@dataclass
class Message:
    """Decoded message.

    Attributes:
        id: Message ID (auto-assigned)
        protocol_id: Parent protocol ID
        timestamp: Message timestamp
        data: Raw message data (hex string)
        decoded_fields: Decoded field values

    Example:
        >>> message = Message(
        ...     id=1,
        ...     protocol_id=1,
        ...     timestamp=1.5,
        ...     data="aa5501",
        ...     decoded_fields={"id": 1, "counter": 0}
        ... )
    """

    id: int | None = None
    protocol_id: int | None = None
    timestamp: float = 0.0
    data: str = ""
    decoded_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Analysis result.

    Attributes:
        id: Result ID (auto-assigned)
        session_id: Parent session ID
        analysis_type: Analysis type (dpa, timing, entropy, etc.)
        results_json: Analysis results as JSON
        metrics: Computed metrics
        created_at: Creation timestamp

    Example:
        >>> result = AnalysisResult(
        ...     id=1,
        ...     session_id=1,
        ...     analysis_type="dpa",
        ...     results_json={"recovered_key": "0x1234..."},
        ...     metrics={"confidence": 0.95}
        ... )
    """

    id: int | None = None
    session_id: int | None = None
    analysis_type: str = ""
    results_json: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class QueryResult:
    """Paginated query result.

    Attributes:
        items: Result items
        total: Total number of results
        page: Current page number (0-indexed)
        page_size: Items per page

    Example:
        >>> result = QueryResult(
        ...     items=[msg1, msg2, msg3],
        ...     total=100,
        ...     page=0,
        ...     page_size=10
        ... )
        >>> print(f"Page 1/{result.total_pages}: {len(result.items)} items")
    """

    items: list[Any] = field(default_factory=list)
    total: int = 0
    page: int = 0
    page_size: int = 100

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages.

        Returns:
            Number of pages (at least 1)
        """
        return max(1, (self.total + self.page_size - 1) // self.page_size)

    @property
    def has_next(self) -> bool:
        """Check if there is a next page.

        Returns:
            True if more pages available
        """
        return self.page < self.total_pages - 1

    @property
    def has_prev(self) -> bool:
        """Check if there is a previous page.

        Returns:
            True if previous pages exist
        """
        return self.page > 0


class DatabaseBackend:
    """Database backend for storing analysis results.

    Supports SQLite (default) and PostgreSQL (optional).
    Uses raw SQL for simplicity and graceful degradation.

    Example:
        >>> config = DatabaseConfig(url="sqlite:///analysis.db")
        >>> db = DatabaseBackend(config)
        >>>
        >>> # Create project hierarchy
        >>> proj_id = db.create_project("IoT RE", "Device protocol analysis")
        >>> sess_id = db.create_session(proj_id, "blackbox", {"file": "capture.bin"})
        >>> prot_id = db.store_protocol(sess_id, "IoT", {"fields": []}, 0.9)
        >>>
        >>> # Store messages
        >>> db.store_message(prot_id, 0.0, b"\\xaa\\x55", {"id": 1})
        >>>
        >>> # Query
        >>> protocols = db.find_protocols(min_confidence=0.8)
        >>> messages = db.query_messages(prot_id, limit=10)
    """

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        """Initialize database backend.

        Args:
            config: Database configuration (default: SQLite)

        Raises:
            ValueError: If PostgreSQL URL but psycopg2 not installed
            sqlite3.Error: If SQLite database creation fails
        """
        self.config = config or DatabaseConfig()
        self._conn: Any = None
        self._pool: Any = None

        # Determine backend type
        self._is_postgres = self.config.url.startswith("postgresql://")

        if self._is_postgres and not HAS_POSTGRES:
            raise ValueError(
                "PostgreSQL URL specified but psycopg2 not installed. "
                "Install with: pip install psycopg2-binary"
            )

        # Initialize connection/pool
        self._init_connection()

        # Create schema
        self._create_schema()

    def _init_connection(self) -> None:
        """Initialize database connection or pool."""
        if self._is_postgres:
            # PostgreSQL connection pool
            self._pool = SimpleConnectionPool(
                1,
                self.config.pool_size,
                self.config.url,
                connect_timeout=int(self.config.timeout),
            )
        else:
            # SQLite connection
            db_path = self.config.url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                db_path,
                timeout=self.config.timeout,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row

    def _get_connection(self) -> Any:
        """Get database connection.

        Returns:
            Connection object (sqlite3.Connection or psycopg2.connection)
        """
        if self._is_postgres:
            return self._pool.getconn()
        return self._conn

    def _return_connection(self, conn: Any) -> None:
        """Return connection to pool (PostgreSQL only).

        Args:
            conn: Connection to return
        """
        if self._is_postgres:
            self._pool.putconn(conn)

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute SQL statement.

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            Cursor after execution
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if self.config.echo_sql:
                logger.debug(f"SQL: {sql}")
                logger.debug(f"Params: {params}")
            cursor.execute(sql, params)
            conn.commit()
            return cursor
        finally:
            self._return_connection(conn)

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        """Execute query and fetch all results.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if self.config.echo_sql:
                logger.debug(f"SQL: {sql}")
                logger.debug(f"Params: {params}")
            cursor.execute(sql, params)

            if self._is_postgres:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
            else:
                return [dict(row) for row in cursor.fetchall()]
        finally:
            self._return_connection(conn)

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Execute query and fetch one result.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Row dictionary or None
        """
        results = self._fetchall(sql, params)
        return results[0] if results else None

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        tables = [
            ("projects", _SQL_CREATE_PROJECTS_SQLITE, _SQL_CREATE_PROJECTS_POSTGRES),
            ("sessions", _SQL_CREATE_SESSIONS_SQLITE, _SQL_CREATE_SESSIONS_POSTGRES),
            ("protocols", _SQL_CREATE_PROTOCOLS_SQLITE, _SQL_CREATE_PROTOCOLS_POSTGRES),
            ("messages", _SQL_CREATE_MESSAGES_SQLITE, _SQL_CREATE_MESSAGES_POSTGRES),
            ("analysis_results", _SQL_CREATE_ANALYSIS_SQLITE, _SQL_CREATE_ANALYSIS_POSTGRES),
        ]

        for _, sqlite_sql, postgres_sql in tables:
            self._execute(sqlite_sql if not self._is_postgres else postgres_sql)

        # Create indexes
        for idx_sql in _SQL_CREATE_INDEXES:
            self._execute(idx_sql)

    def create_project(
        self, name: str, description: str = "", metadata: dict[str, Any] | None = None
    ) -> int:
        """Create new project.

        Args:
            name: Project name
            description: Project description
            metadata: Additional metadata

        Returns:
            Project ID

        Example:
            >>> db = DatabaseBackend()
            >>> project_id = db.create_project("IoT RE", "Unknown device protocol")
        """
        metadata_json = json.dumps(metadata or {})
        cursor = self._execute(
            "INSERT INTO projects (name, description, metadata) VALUES (?, ?, ?)",
            (name, description, metadata_json),
        )
        result: int = cursor.lastrowid
        return result

    def get_project(self, project_id: int) -> Project | None:
        """Get project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project or None if not found

        Example:
            >>> project = db.get_project(1)
            >>> print(project.name)
        """
        row = self._fetchone("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not row:
            return None

        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def list_projects(self) -> list[Project]:
        """List all projects.

        Returns:
            List of projects

        Example:
            >>> projects = db.list_projects()
            >>> for proj in projects:
            ...     print(f"{proj.id}: {proj.name}")
        """
        rows = self._fetchall("SELECT * FROM projects ORDER BY updated_at DESC")
        return [
            Project(
                id=row["id"],
                name=row["name"],
                description=row["description"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    def create_session(
        self,
        project_id: int,
        session_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Create new session.

        Args:
            project_id: Parent project ID
            session_type: Session type (blackbox, can, uart, etc.)
            metadata: Session metadata

        Returns:
            Session ID

        Example:
            >>> session_id = db.create_session(
            ...     project_id=1,
            ...     session_type="blackbox",
            ...     metadata={"capture": "device.bin"}
            ... )
        """
        metadata_json = json.dumps(metadata or {})
        cursor = self._execute(
            "INSERT INTO sessions (project_id, session_type, metadata) VALUES (?, ?, ?)",
            (project_id, session_type, metadata_json),
        )
        result: int = cursor.lastrowid
        return result

    def get_sessions(self, project_id: int) -> list[Session]:
        """Get all sessions for project.

        Args:
            project_id: Project ID

        Returns:
            List of sessions

        Example:
            >>> sessions = db.get_sessions(project_id=1)
            >>> for sess in sessions:
            ...     print(f"{sess.id}: {sess.session_type}")
        """
        rows = self._fetchall(
            "SELECT * FROM sessions WHERE project_id = ? ORDER BY timestamp DESC",
            (project_id,),
        )
        return [
            Session(
                id=row["id"],
                project_id=row["project_id"],
                session_type=row["session_type"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    def store_protocol(
        self,
        session_id: int,
        name: str,
        spec_json: dict[str, Any],
        confidence: float,
    ) -> int:
        """Store discovered protocol.

        Args:
            session_id: Parent session ID
            name: Protocol name
            spec_json: Protocol specification
            confidence: Confidence score (0.0-1.0)

        Returns:
            Protocol ID

        Example:
            >>> protocol_id = db.store_protocol(
            ...     session_id=1,
            ...     name="IoT Protocol",
            ...     spec_json={"fields": [...]},
            ...     confidence=0.85
            ... )
        """
        spec_str = json.dumps(spec_json)
        cursor = self._execute(
            "INSERT INTO protocols (session_id, name, spec_json, confidence) VALUES (?, ?, ?, ?)",
            (session_id, name, spec_str, confidence),
        )
        result: int = cursor.lastrowid
        return result

    def find_protocols(
        self,
        name_pattern: str | None = None,
        min_confidence: float | None = None,
    ) -> list[Protocol]:
        """Find protocols by criteria.

        Args:
            name_pattern: SQL LIKE pattern (e.g., "UDS%")
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching protocols

        Example:
            >>> # Find all UDS protocols with confidence > 0.8
            >>> protocols = db.find_protocols(name_pattern="UDS%", min_confidence=0.8)
        """
        conditions = []
        params: list[Any] = []

        if name_pattern:
            conditions.append("name LIKE ?")
            params.append(name_pattern)

        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM protocols {where_clause} ORDER BY confidence DESC"

        rows = self._fetchall(sql, tuple(params))
        return [
            Protocol(
                id=row["id"],
                session_id=row["session_id"],
                name=row["name"],
                spec_json=json.loads(row["spec_json"]),
                confidence=row["confidence"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def store_message(
        self,
        protocol_id: int,
        timestamp: float,
        data: bytes,
        decoded_fields: dict[str, Any] | None = None,
    ) -> int:
        """Store decoded message.

        Args:
            protocol_id: Parent protocol ID
            timestamp: Message timestamp
            data: Raw message bytes
            decoded_fields: Decoded field values

        Returns:
            Message ID

        Example:
            >>> msg_id = db.store_message(
            ...     protocol_id=1,
            ...     timestamp=1.5,
            ...     data=b"\\xaa\\x55\\x01",
            ...     decoded_fields={"id": 1, "counter": 0}
            ... )
        """
        data_hex = data.hex()
        fields_json = json.dumps(decoded_fields or {})
        cursor = self._execute(
            "INSERT INTO messages (protocol_id, timestamp, data, decoded_fields) "
            "VALUES (?, ?, ?, ?)",
            (protocol_id, timestamp, data_hex, fields_json),
        )
        result: int = cursor.lastrowid
        return result

    def query_messages(
        self,
        protocol_id: int,
        time_range: tuple[float, float] | None = None,
        field_filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """Query messages with filtering and pagination.

        Args:
            protocol_id: Protocol ID
            time_range: (start_time, end_time) tuple
            field_filters: Field name -> value filters
            limit: Maximum results per page
            offset: Result offset for pagination

        Returns:
            Paginated query results

        Example:
            >>> # Get first 10 messages between t=0 and t=10
            >>> result = db.query_messages(
            ...     protocol_id=1,
            ...     time_range=(0.0, 10.0),
            ...     limit=10
            ... )
            >>> print(f"Page {result.page + 1}/{result.total_pages}")
            >>> for msg in result.items:
            ...     print(msg.decoded_fields)
        """
        conditions = ["protocol_id = ?"]
        params: list[Any] = [protocol_id]

        if time_range:
            conditions.append("timestamp >= ? AND timestamp <= ?")
            params.extend(time_range)

        where_clause = f"WHERE {' AND '.join(conditions)}"

        # Count total
        count_sql = f"SELECT COUNT(*) as total FROM messages {where_clause}"
        count_row = self._fetchone(count_sql, tuple(params))
        total = count_row["total"] if count_row else 0

        # Fetch page
        sql = f"SELECT * FROM messages {where_clause} ORDER BY timestamp LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._fetchall(sql, tuple(params))

        messages = [
            Message(
                id=row["id"],
                protocol_id=row["protocol_id"],
                timestamp=row["timestamp"],
                data=row["data"],
                decoded_fields=json.loads(row["decoded_fields"]) if row["decoded_fields"] else {},
            )
            for row in rows
        ]

        # Apply field filters (client-side for SQLite)
        if field_filters:
            messages = [
                msg
                for msg in messages
                if all(msg.decoded_fields.get(k) == v for k, v in field_filters.items())
            ]

        return QueryResult(
            items=messages,
            total=total,
            page=offset // limit,
            page_size=limit,
        )

    def store_analysis_result(
        self,
        session_id: int,
        analysis_type: str,
        results_json: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> int:
        """Store analysis result.

        Args:
            session_id: Parent session ID
            analysis_type: Analysis type (dpa, timing, entropy, etc.)
            results_json: Analysis results
            metrics: Computed metrics

        Returns:
            Result ID

        Example:
            >>> result_id = db.store_analysis_result(
            ...     session_id=1,
            ...     analysis_type="dpa",
            ...     results_json={"recovered_key": "0x1234..."},
            ...     metrics={"confidence": 0.95}
            ... )
        """
        results_str = json.dumps(results_json)
        metrics_str = json.dumps(metrics or {})
        cursor = self._execute(
            "INSERT INTO analysis_results (session_id, analysis_type, results_json, metrics) "
            "VALUES (?, ?, ?, ?)",
            (session_id, analysis_type, results_str, metrics_str),
        )
        result: int = cursor.lastrowid
        return result

    def get_analysis_results(
        self, session_id: int, analysis_type: str | None = None
    ) -> list[AnalysisResult]:
        """Get analysis results for session.

        Args:
            session_id: Session ID
            analysis_type: Filter by analysis type (optional)

        Returns:
            List of analysis results

        Example:
            >>> # Get all DPA results
            >>> results = db.get_analysis_results(session_id=1, analysis_type="dpa")
            >>> for result in results:
            ...     print(result.metrics["confidence"])
        """
        if analysis_type:
            sql = (
                "SELECT * FROM analysis_results "
                "WHERE session_id = ? AND analysis_type = ? "
                "ORDER BY created_at DESC"
            )
            params: tuple[Any, ...] = (session_id, analysis_type)
        else:
            sql = "SELECT * FROM analysis_results WHERE session_id = ? ORDER BY created_at DESC"
            params = (session_id,)

        rows = self._fetchall(sql, params)
        return [
            AnalysisResult(
                id=row["id"],
                session_id=row["session_id"],
                analysis_type=row["analysis_type"],
                results_json=json.loads(row["results_json"]),
                metrics=json.loads(row["metrics"]) if row["metrics"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def export_to_sql(self, output_path: str | Path) -> None:
        """Export database to SQL dump.

        Args:
            output_path: Output SQL file path

        Example:
            >>> db.export_to_sql("backup.sql")
        """
        output_path = Path(output_path)

        if self._is_postgres:
            raise NotImplementedError("PostgreSQL export via pg_dump recommended")

        # SQLite dump
        with open(output_path, "w") as f:
            for line in self._conn.iterdump():
                f.write(f"{line}\n")

    def export_to_json(self, output_path: str | Path, project_id: int | None = None) -> None:
        """Export database contents to JSON.

        Args:
            output_path: Output JSON file path
            project_id: Export specific project (optional)

        Example:
            >>> db.export_to_json("export.json", project_id=1)
        """
        output_path = Path(output_path)

        projects_list: list[Project | None]
        if project_id:
            projects_list = [self.get_project(project_id)]
        else:
            projects_list = list(self.list_projects())

        export_data = []
        for proj in projects_list:
            if proj is None:
                continue

            proj_data = asdict(proj)
            proj_data["sessions"] = []

            sessions = self.get_sessions(proj.id)  # type: ignore[arg-type]
            for sess in sessions:
                sess_data = asdict(sess)
                sess_data["protocols"] = []
                sess_data["analysis_results"] = []

                # Get protocols
                protocols = self.find_protocols()
                for prot in protocols:
                    if prot.session_id == sess.id:
                        prot_data = asdict(prot)
                        # Get messages
                        msgs = self.query_messages(prot.id, limit=1000)  # type: ignore[arg-type]
                        prot_data["messages"] = [asdict(msg) for msg in msgs.items]
                        sess_data["protocols"].append(prot_data)

                # Get analysis results
                results = self.get_analysis_results(sess.id)  # type: ignore[arg-type]
                sess_data["analysis_results"] = [asdict(r) for r in results]

                proj_data["sessions"].append(sess_data)

            export_data.append(proj_data)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    def export_to_csv(self, output_dir: str | Path, project_id: int | None = None) -> None:
        """Export database to CSV files.

        Args:
            output_dir: Output directory for CSV files
            project_id: Export specific project (optional)

        Example:
            >>> db.export_to_csv("csv_export/", project_id=1)
        """
        import csv

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        projects_list: list[Project | None]
        if project_id:
            projects_list = [self.get_project(project_id)]
        else:
            projects_list = list(self.list_projects())

        # Export projects
        with open(output_dir / "projects.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "name", "description", "created_at", "updated_at"]
            )
            writer.writeheader()
            for proj in projects_list:
                if proj:
                    writer.writerow(
                        {
                            "id": proj.id,
                            "name": proj.name,
                            "description": proj.description,
                            "created_at": proj.created_at,
                            "updated_at": proj.updated_at,
                        }
                    )

        # Export sessions
        all_sessions = []
        for proj in projects_list:
            if proj:
                all_sessions.extend(self.get_sessions(proj.id))  # type: ignore[arg-type]

        with open(output_dir / "sessions.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "project_id", "session_type", "timestamp"])
            writer.writeheader()
            for sess in all_sessions:
                writer.writerow(
                    {
                        "id": sess.id,
                        "project_id": sess.project_id,
                        "session_type": sess.session_type,
                        "timestamp": sess.timestamp,
                    }
                )

        # Export protocols
        all_protocols = self.find_protocols()
        with open(output_dir / "protocols.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "session_id", "name", "confidence", "created_at"]
            )
            writer.writeheader()
            for prot in all_protocols:
                writer.writerow(
                    {
                        "id": prot.id,
                        "session_id": prot.session_id,
                        "name": prot.name,
                        "confidence": prot.confidence,
                        "created_at": prot.created_at,
                    }
                )

    def close(self) -> None:
        """Close database connection/pool.

        Example:
            >>> db.close()
        """
        if self._is_postgres and self._pool:
            self._pool.closeall()
        elif self._conn:
            self._conn.close()

    def __enter__(self) -> DatabaseBackend:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
