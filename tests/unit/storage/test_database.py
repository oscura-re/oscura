"""Tests for database backend.

This module tests the DatabaseBackend class for storing and querying
analysis results with comprehensive coverage of all features.

Test Coverage:
    - DatabaseConfig dataclass validation
    - Project CRUD operations
    - Session CRUD operations
    - Protocol storage and querying
    - Message storage and querying
    - Analysis result storage
    - QueryResult pagination
    - Export functionality (SQL, JSON, CSV)
    - Context manager support
    - Edge cases and error handling
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from oscura.utils.storage import (
    AnalysisResult,
    DatabaseBackend,
    DatabaseConfig,
    Message,
    Project,
    Protocol,
    QueryResult,
    Session,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def db_config(temp_db_path: Path) -> DatabaseConfig:
    """Database configuration for testing."""
    return DatabaseConfig(url=f"sqlite:///{temp_db_path}")


@pytest.fixture
def db(db_config: DatabaseConfig) -> DatabaseBackend:
    """Database backend instance."""
    backend = DatabaseBackend(db_config)
    yield backend
    backend.close()


@pytest.fixture
def populated_db(db: DatabaseBackend) -> DatabaseBackend:
    """Database with test data."""
    # Create project
    proj_id = db.create_project("Test Project", "Test description")

    # Create sessions
    sess1_id = db.create_session(proj_id, "blackbox", {"capture": "device1.bin"})
    sess2_id = db.create_session(proj_id, "can", {"bus": "HS-CAN"})

    # Create protocols
    prot1_id = db.store_protocol(
        sess1_id,
        "IoT Protocol",
        {"fields": [{"name": "id", "offset": 0, "length": 1}]},
        0.85,
    )
    prot2_id = db.store_protocol(
        sess2_id,
        "UDS",
        {"fields": [{"name": "service", "offset": 0, "length": 1}]},
        0.95,
    )

    # Store messages
    for i in range(20):
        db.store_message(
            prot1_id,
            timestamp=float(i),
            data=bytes([0xAA, 0x55, i]),
            decoded_fields={"id": i, "counter": i % 10},
        )

    # Store analysis result
    db.store_analysis_result(
        sess1_id,
        "dpa",
        {"recovered_key": "0x1234"},
        {"confidence": 0.95},
    )

    return db


# =============================================================================
# DatabaseConfig Tests
# =============================================================================


def test_database_config_defaults() -> None:
    """Test DatabaseConfig default values."""
    config = DatabaseConfig()
    assert config.url == "sqlite:///oscura_analysis.db"
    assert config.pool_size == 5
    assert config.timeout == 30.0
    assert config.echo_sql is False


def test_database_config_custom() -> None:
    """Test DatabaseConfig with custom values."""
    config = DatabaseConfig(
        url="postgresql://localhost/test",
        pool_size=10,
        timeout=60.0,
        echo_sql=True,
    )
    assert config.url == "postgresql://localhost/test"
    assert config.pool_size == 10
    assert config.timeout == 60.0
    assert config.echo_sql is True


# =============================================================================
# Project Tests
# =============================================================================


def test_project_dataclass() -> None:
    """Test Project dataclass."""
    now = datetime.now(UTC)
    project = Project(
        id=1,
        name="Test",
        description="Desc",
        created_at=now,
        updated_at=now,
        metadata={"key": "value"},
    )
    assert project.id == 1
    assert project.name == "Test"
    assert project.description == "Desc"
    assert project.metadata == {"key": "value"}


def test_create_project(db: DatabaseBackend) -> None:
    """Test creating a project."""
    proj_id = db.create_project("IoT RE", "Device protocol analysis")
    assert proj_id > 0


def test_create_project_with_metadata(db: DatabaseBackend) -> None:
    """Test creating project with metadata."""
    metadata = {"device": "ESP32", "version": "1.0"}
    proj_id = db.create_project("ESP32 RE", metadata=metadata)

    project = db.get_project(proj_id)
    assert project is not None
    assert project.name == "ESP32 RE"
    assert project.metadata == metadata


def test_get_project(db: DatabaseBackend) -> None:
    """Test retrieving a project."""
    proj_id = db.create_project("Test Project", "Description")
    project = db.get_project(proj_id)

    assert project is not None
    assert project.id == proj_id
    assert project.name == "Test Project"
    assert project.description == "Description"
    assert project.created_at is not None
    assert project.updated_at is not None


def test_get_nonexistent_project(db: DatabaseBackend) -> None:
    """Test getting project that doesn't exist."""
    project = db.get_project(9999)
    assert project is None


def test_list_projects(db: DatabaseBackend) -> None:
    """Test listing all projects."""
    db.create_project("Project 1", "Desc 1")
    db.create_project("Project 2", "Desc 2")
    db.create_project("Project 3", "Desc 3")

    projects = db.list_projects()
    assert len(projects) == 3
    assert all(isinstance(p, Project) for p in projects)
    # Verify all project names are present
    names = {p.name for p in projects}
    assert names == {"Project 1", "Project 2", "Project 3"}


# =============================================================================
# Session Tests
# =============================================================================


def test_session_dataclass() -> None:
    """Test Session dataclass."""
    now = datetime.now(UTC)
    session = Session(
        id=1,
        project_id=1,
        session_type="blackbox",
        timestamp=now,
        metadata={"file": "test.bin"},
    )
    assert session.id == 1
    assert session.project_id == 1
    assert session.session_type == "blackbox"
    assert session.metadata == {"file": "test.bin"}


def test_create_session(db: DatabaseBackend) -> None:
    """Test creating a session."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox", {"capture": "device.bin"})
    assert sess_id > 0


def test_get_sessions(db: DatabaseBackend) -> None:
    """Test retrieving sessions for a project."""
    proj_id = db.create_project("Project")
    sess1 = db.create_session(proj_id, "blackbox", {"file": "test1.bin"})
    sess2 = db.create_session(proj_id, "can", {"bus": "HS-CAN"})

    sessions = db.get_sessions(proj_id)
    assert len(sessions) == 2
    assert all(isinstance(s, Session) for s in sessions)
    assert sessions[0].session_type in ["blackbox", "can"]


def test_get_sessions_empty(db: DatabaseBackend) -> None:
    """Test getting sessions for project with no sessions."""
    proj_id = db.create_project("Empty Project")
    sessions = db.get_sessions(proj_id)
    assert sessions == []


# =============================================================================
# Protocol Tests
# =============================================================================


def test_protocol_dataclass() -> None:
    """Test Protocol dataclass."""
    now = datetime.now(UTC)
    protocol = Protocol(
        id=1,
        session_id=1,
        name="UDS",
        spec_json={"fields": []},
        confidence=0.9,
        created_at=now,
    )
    assert protocol.id == 1
    assert protocol.name == "UDS"
    assert protocol.confidence == 0.9


def test_store_protocol(db: DatabaseBackend) -> None:
    """Test storing a protocol."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")

    spec = {
        "fields": [
            {"name": "id", "offset": 0, "length": 1},
            {"name": "data", "offset": 1, "length": 4},
        ]
    }
    prot_id = db.store_protocol(sess_id, "IoT Protocol", spec, 0.85)
    assert prot_id > 0


def test_find_protocols_all(populated_db: DatabaseBackend) -> None:
    """Test finding all protocols."""
    protocols = populated_db.find_protocols()
    assert len(protocols) == 2
    assert all(isinstance(p, Protocol) for p in protocols)
    # Ordered by confidence DESC
    assert protocols[0].confidence >= protocols[1].confidence


def test_find_protocols_by_name(populated_db: DatabaseBackend) -> None:
    """Test finding protocols by name pattern."""
    protocols = populated_db.find_protocols(name_pattern="UDS%")
    assert len(protocols) == 1
    assert protocols[0].name == "UDS"


def test_find_protocols_by_confidence(populated_db: DatabaseBackend) -> None:
    """Test finding protocols by minimum confidence."""
    protocols = populated_db.find_protocols(min_confidence=0.9)
    assert len(protocols) == 1
    assert protocols[0].confidence >= 0.9


def test_find_protocols_combined_filters(populated_db: DatabaseBackend) -> None:
    """Test finding protocols with multiple filters."""
    protocols = populated_db.find_protocols(
        name_pattern="IoT%",
        min_confidence=0.8,
    )
    assert len(protocols) == 1
    assert protocols[0].name == "IoT Protocol"
    assert protocols[0].confidence >= 0.8


# =============================================================================
# Message Tests
# =============================================================================


def test_message_dataclass() -> None:
    """Test Message dataclass."""
    message = Message(
        id=1,
        protocol_id=1,
        timestamp=1.5,
        data="aa5501",
        decoded_fields={"id": 1, "counter": 0},
    )
    assert message.id == 1
    assert message.timestamp == 1.5
    assert message.data == "aa5501"
    assert message.decoded_fields == {"id": 1, "counter": 0}


def test_store_message(db: DatabaseBackend) -> None:
    """Test storing a message."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")
    prot_id = db.store_protocol(sess_id, "Protocol", {}, 0.8)

    msg_id = db.store_message(
        prot_id,
        timestamp=1.5,
        data=b"\xaa\x55\x01",
        decoded_fields={"id": 1},
    )
    assert msg_id > 0


def test_query_messages_basic(populated_db: DatabaseBackend) -> None:
    """Test basic message query."""
    protocols = populated_db.find_protocols(name_pattern="IoT%")
    prot_id = protocols[0].id

    result = populated_db.query_messages(prot_id)
    assert isinstance(result, QueryResult)
    assert len(result.items) > 0
    assert all(isinstance(m, Message) for m in result.items)


def test_query_messages_pagination(populated_db: DatabaseBackend) -> None:
    """Test message query with pagination."""
    protocols = populated_db.find_protocols(name_pattern="IoT%")
    prot_id = protocols[0].id

    # First page
    page1 = populated_db.query_messages(prot_id, limit=5, offset=0)
    assert len(page1.items) == 5
    assert page1.page == 0
    assert page1.total == 20
    assert page1.has_next
    assert not page1.has_prev

    # Second page
    page2 = populated_db.query_messages(prot_id, limit=5, offset=5)
    assert len(page2.items) == 5
    assert page2.page == 1
    assert page2.has_next
    assert page2.has_prev


def test_query_messages_time_range(populated_db: DatabaseBackend) -> None:
    """Test querying messages by time range."""
    protocols = populated_db.find_protocols(name_pattern="IoT%")
    prot_id = protocols[0].id

    result = populated_db.query_messages(prot_id, time_range=(5.0, 10.0))
    assert all(5.0 <= m.timestamp <= 10.0 for m in result.items)


def test_query_messages_field_filters(populated_db: DatabaseBackend) -> None:
    """Test querying messages with field filters."""
    protocols = populated_db.find_protocols(name_pattern="IoT%")
    prot_id = protocols[0].id

    result = populated_db.query_messages(
        prot_id,
        field_filters={"counter": 5},
    )
    # Messages with counter=5: id=5 and id=15
    assert all(m.decoded_fields["counter"] == 5 for m in result.items)


# =============================================================================
# QueryResult Tests
# =============================================================================


def test_query_result_total_pages() -> None:
    """Test QueryResult total_pages calculation."""
    result = QueryResult(items=[], total=100, page=0, page_size=10)
    assert result.total_pages == 10

    result = QueryResult(items=[], total=95, page=0, page_size=10)
    assert result.total_pages == 10

    result = QueryResult(items=[], total=0, page=0, page_size=10)
    assert result.total_pages == 1  # At least 1


def test_query_result_has_next() -> None:
    """Test QueryResult has_next property."""
    result = QueryResult(items=[], total=30, page=0, page_size=10)
    assert result.has_next

    result = QueryResult(items=[], total=30, page=2, page_size=10)
    assert not result.has_next


def test_query_result_has_prev() -> None:
    """Test QueryResult has_prev property."""
    result = QueryResult(items=[], total=30, page=0, page_size=10)
    assert not result.has_prev

    result = QueryResult(items=[], total=30, page=1, page_size=10)
    assert result.has_prev


# =============================================================================
# Analysis Result Tests
# =============================================================================


def test_analysis_result_dataclass() -> None:
    """Test AnalysisResult dataclass."""
    now = datetime.now(UTC)
    result = AnalysisResult(
        id=1,
        session_id=1,
        analysis_type="dpa",
        results_json={"key": "0x1234"},
        metrics={"confidence": 0.95},
        created_at=now,
    )
    assert result.id == 1
    assert result.analysis_type == "dpa"
    assert result.metrics["confidence"] == 0.95


def test_store_analysis_result(db: DatabaseBackend) -> None:
    """Test storing analysis result."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")

    result_id = db.store_analysis_result(
        sess_id,
        "dpa",
        {"recovered_key": "0x1234"},
        {"confidence": 0.95},
    )
    assert result_id > 0


def test_get_analysis_results_all(populated_db: DatabaseBackend) -> None:
    """Test getting all analysis results."""
    sessions = populated_db.get_sessions(1)
    sess_id = sessions[0].id

    results = populated_db.get_analysis_results(sess_id)
    assert len(results) == 1
    assert isinstance(results[0], AnalysisResult)


def test_get_analysis_results_by_type(populated_db: DatabaseBackend) -> None:
    """Test getting analysis results by type."""
    sessions = populated_db.get_sessions(1)
    sess_id = sessions[0].id

    # Store multiple types
    populated_db.store_analysis_result(sess_id, "timing", {}, {})
    populated_db.store_analysis_result(sess_id, "entropy", {}, {})

    dpa_results = populated_db.get_analysis_results(sess_id, "dpa")
    assert len(dpa_results) == 1
    assert dpa_results[0].analysis_type == "dpa"

    timing_results = populated_db.get_analysis_results(sess_id, "timing")
    assert len(timing_results) == 1


# =============================================================================
# Export Tests
# =============================================================================


def test_export_to_sql(populated_db: DatabaseBackend, tmp_path: Path) -> None:
    """Test SQL export."""
    output = tmp_path / "export.sql"
    populated_db.export_to_sql(output)

    assert output.exists()
    assert output.stat().st_size > 0

    # Verify SQL is valid by executing it
    conn = sqlite3.connect(":memory:")
    with open(output) as f:
        conn.executescript(f.read())
    conn.close()


def test_export_to_json(populated_db: DatabaseBackend, tmp_path: Path) -> None:
    """Test JSON export."""
    output = tmp_path / "export.json"
    populated_db.export_to_json(output)

    assert output.exists()

    with open(output) as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 1  # 1 project
    assert "sessions" in data[0]
    assert len(data[0]["sessions"]) == 2  # 2 sessions


def test_export_to_json_single_project(populated_db: DatabaseBackend, tmp_path: Path) -> None:
    """Test JSON export for single project."""
    output = tmp_path / "export.json"
    populated_db.export_to_json(output, project_id=1)

    with open(output) as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["id"] == 1


def test_export_to_csv(populated_db: DatabaseBackend, tmp_path: Path) -> None:
    """Test CSV export."""
    output_dir = tmp_path / "csv_export"
    populated_db.export_to_csv(output_dir)

    assert (output_dir / "projects.csv").exists()
    assert (output_dir / "sessions.csv").exists()
    assert (output_dir / "protocols.csv").exists()


def test_export_to_csv_single_project(populated_db: DatabaseBackend, tmp_path: Path) -> None:
    """Test CSV export for single project."""
    output_dir = tmp_path / "csv_export"
    populated_db.export_to_csv(output_dir, project_id=1)

    # Verify projects.csv has 1 project
    with open(output_dir / "projects.csv") as f:
        lines = f.readlines()
        assert len(lines) == 2  # Header + 1 row


# =============================================================================
# Context Manager Tests
# =============================================================================


def test_context_manager(temp_db_path: Path) -> None:
    """Test database context manager."""
    config = DatabaseConfig(url=f"sqlite:///{temp_db_path}")

    with DatabaseBackend(config) as db:
        proj_id = db.create_project("Test")
        assert proj_id > 0

    # Connection should be closed
    # Verify by opening new connection
    with DatabaseBackend(config) as db2:
        projects = db2.list_projects()
        assert len(projects) == 1


# =============================================================================
# Edge Cases
# =============================================================================


def test_empty_database(db: DatabaseBackend) -> None:
    """Test querying empty database."""
    projects = db.list_projects()
    assert projects == []

    protocols = db.find_protocols()
    assert protocols == []


def test_store_message_with_empty_fields(db: DatabaseBackend) -> None:
    """Test storing message with no decoded fields."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")
    prot_id = db.store_protocol(sess_id, "Protocol", {}, 0.8)

    msg_id = db.store_message(prot_id, 0.0, b"\xaa\x55")
    assert msg_id > 0

    result = db.query_messages(prot_id)
    assert result.items[0].decoded_fields == {}


def test_store_protocol_with_complex_spec(db: DatabaseBackend) -> None:
    """Test storing protocol with complex specification."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")

    complex_spec = {
        "fields": [
            {"name": "header", "offset": 0, "length": 2, "type": "constant"},
            {"name": "id", "offset": 2, "length": 2, "type": "uint16"},
            {"name": "data", "offset": 4, "length": 10, "type": "bytes"},
            {"name": "checksum", "offset": 14, "length": 2, "type": "crc16"},
        ],
        "state_machine": {
            "states": ["IDLE", "ACTIVE", "ERROR"],
            "transitions": [
                {"from": "IDLE", "to": "ACTIVE", "event": "start"},
            ],
        },
        "crc_info": {
            "polynomial": 0x1021,
            "init_value": 0xFFFF,
        },
    }

    prot_id = db.store_protocol(sess_id, "Complex Protocol", complex_spec, 0.9)
    protocols = db.find_protocols(name_pattern="Complex%")

    assert len(protocols) == 1
    assert protocols[0].spec_json == complex_spec


def test_query_messages_no_results(db: DatabaseBackend) -> None:
    """Test querying messages when none exist."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")
    prot_id = db.store_protocol(sess_id, "Protocol", {}, 0.8)

    result = db.query_messages(prot_id)
    assert result.items == []
    assert result.total == 0


def test_large_dataset_pagination(db: DatabaseBackend) -> None:
    """Test pagination with large dataset."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")
    prot_id = db.store_protocol(sess_id, "Protocol", {}, 0.8)

    # Store 100 messages
    for i in range(100):
        db.store_message(prot_id, float(i), bytes([i]), {"idx": i})

    # Query with small page size
    page = db.query_messages(prot_id, limit=10, offset=0)
    assert len(page.items) == 10
    assert page.total == 100
    assert page.total_pages == 10


def test_invalid_project_id_in_session(db: DatabaseBackend) -> None:
    """Test creating session with invalid project ID."""
    # SQLite will allow this (foreign key constraints off by default)
    # but it's still good to test the behavior
    try:
        sess_id = db.create_session(9999, "blackbox")
        assert sess_id > 0  # Will succeed in SQLite without FK enforcement
    except sqlite3.IntegrityError:
        pass  # Expected in PostgreSQL or SQLite with FK enabled


def test_find_protocols_empty_filters(db: DatabaseBackend) -> None:
    """Test finding protocols with no filters."""
    proj_id = db.create_project("Project")
    sess_id = db.create_session(proj_id, "blackbox")
    db.store_protocol(sess_id, "Protocol 1", {}, 0.8)
    db.store_protocol(sess_id, "Protocol 2", {}, 0.9)

    protocols = db.find_protocols()
    assert len(protocols) == 2
