"""Tests for advanced reporting features.

Tests advanced reporting capabilities including templates, interactive elements,
annotations, scheduling, distribution, archiving, search, versioning, approval,
compliance, localization, accessibility, encryption, watermarking, and audit trail.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oscura.reporting.advanced import (
    AccessibilityOptions,
    Annotation,
    AnnotationManager,
    ApprovalStatus,
    ApprovalWorkflow,
    AuditTrail,
    ComplianceChecker,
    CustomTemplate,
    DistributionChannel,
    DistributionConfig,
    InteractiveElement,
    InteractiveElementType,
    ReportArchive,
    ReportDistributor,
    ReportEncryption,
    ReportLocalizer,
    ReportSchedule,
    ReportScheduler,
    ReportSearchIndex,
    ReportVersionControl,
    ScheduleFrequency,
    TemplateField,
    Watermark,
    add_accessibility_features,
    add_watermark,
)

pytestmark = pytest.mark.unit


# =============================================================================
# =============================================================================


class TestTemplateField:
    """Test TemplateField dataclass."""

    def test_create_simple_field(self) -> None:
        """Test creating a simple template field."""
        field = TemplateField(name="company_name")
        assert field.name == "company_name"
        assert field.type == "text"
        assert field.default is None
        assert field.required is False

    def test_create_required_field(self) -> None:
        """Test creating a required field."""
        field = TemplateField(name="title", type="text", required=True, description="Report title")
        assert field.required is True
        assert field.description == "Report title"

    def test_create_field_with_validation(self) -> None:
        """Test creating field with validation."""
        field = TemplateField(name="email", validation=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        assert field.validation is not None


class TestCustomTemplate:
    """Test CustomTemplate class."""

    def test_create_simple_template(self) -> None:
        """Test creating a simple template."""
        template = CustomTemplate(name="basic_report")
        assert template.name == "basic_report"
        assert template.version == "1.0.0"
        assert len(template.fields) == 0

    def test_create_template_with_fields(self) -> None:
        """Test creating template with fields."""
        fields = [
            TemplateField("title", required=True),
            TemplateField("author"),
            TemplateField("logo", type="image"),
        ]
        template = CustomTemplate(name="detailed_report", fields=fields)
        assert len(template.fields) == 3

    def test_validate_data_all_required_present(self) -> None:
        """Test validation when all required fields are present."""
        template = CustomTemplate(
            name="test",
            fields=[
                TemplateField("title", required=True),
                TemplateField("author", required=True),
            ],
        )
        data = {"title": "Test Report", "author": "John Doe"}
        valid, errors = template.validate_data(data)
        assert valid is True
        assert len(errors) == 0

    def test_validate_data_missing_required(self) -> None:
        """Test validation when required field is missing."""
        template = CustomTemplate(name="test", fields=[TemplateField("title", required=True)])
        data = {"author": "John Doe"}
        valid, errors = template.validate_data(data)
        assert valid is False
        assert len(errors) == 1
        assert "title" in errors[0]

    def test_validate_data_validation_regex(self) -> None:
        """Test validation with regex pattern."""
        template = CustomTemplate(
            name="test",
            fields=[TemplateField("email", validation=r"^[\w\.-]+@[\w\.-]+\.\w+$")],
        )

        # Valid email
        valid, errors = template.validate_data({"email": "test@example.com"})
        assert valid is True

        # Invalid email
        valid, errors = template.validate_data({"email": "invalid-email"})
        assert valid is False
        assert len(errors) == 1

    def test_render_simple(self) -> None:
        """Test rendering template with simple placeholders."""
        template = CustomTemplate(
            name="test",
            fields=[TemplateField("name"), TemplateField("company")],
            layout={"template": "Hello {{name}} from {{company}}"},
        )
        data = {"name": "John", "company": "Acme"}
        rendered = template.render(data)
        assert rendered == "Hello John from Acme"

    def test_render_with_defaults(self) -> None:
        """Test rendering with default values."""
        template = CustomTemplate(
            name="test",
            fields=[TemplateField("name", default="Unknown")],
            layout={"template": "Hello {{name}}"},
        )
        rendered = template.render({})
        assert rendered == "Hello Unknown"


# =============================================================================
# =============================================================================


class TestInteractiveElement:
    """Test InteractiveElement class."""

    def test_create_simple_element(self) -> None:
        """Test creating a simple interactive element."""
        element = InteractiveElement(id="chart1", type=InteractiveElementType.ZOOMABLE_CHART)
        assert element.id == "chart1"
        assert element.type == InteractiveElementType.ZOOMABLE_CHART

    def test_create_collapsible_section(self) -> None:
        """Test creating collapsible section."""
        element = InteractiveElement(
            id="section1",
            type=InteractiveElementType.COLLAPSIBLE_SECTION,
            data="Section content",
            options={"title": "Click to expand"},
        )
        html = element.to_html()
        assert "collapsible" in html
        assert "Click to expand" in html
        assert "Section content" in html

    def test_create_sortable_table(self) -> None:
        """Test creating sortable table."""
        element = InteractiveElement(
            id="table1",
            type=InteractiveElementType.SORTABLE_TABLE,
            data="<tr><td>Data</td></tr>",
        )
        html = element.to_html()
        assert "sortable" in html
        assert "data-sort-enabled" in html

    def test_create_tooltip(self) -> None:
        """Test creating tooltip element."""
        element = InteractiveElement(
            id="tip1",
            type=InteractiveElementType.TOOLTIP,
            data="Hover me",
            options={"text": "Tooltip text"},
        )
        html = element.to_html()
        assert "tooltip" in html
        assert "Tooltip text" in html

    def test_to_html_contains_id(self) -> None:
        """Test that HTML output contains element ID."""
        element = InteractiveElement(id="test123", type=InteractiveElementType.TOGGLE)
        html = element.to_html()
        assert "test123" in html


# =============================================================================
# =============================================================================


class TestAnnotation:
    """Test Annotation dataclass."""

    def test_create_annotation(self) -> None:
        """Test creating an annotation."""
        annotation = Annotation(id="ann1", target="section1", text="Important note", author="John")
        assert annotation.id == "ann1"
        assert annotation.target == "section1"
        assert annotation.text == "Important note"

    def test_annotation_defaults(self) -> None:
        """Test annotation default values."""
        annotation = Annotation(id="ann1", target="sec1", text="Note")
        assert annotation.type == "note"
        assert annotation.author == ""
        assert isinstance(annotation.created, datetime)

    def test_to_dict(self) -> None:
        """Test converting annotation to dictionary."""
        annotation = Annotation(
            id="ann1", target="sec1", text="Note", author="John", type="warning"
        )
        data = annotation.to_dict()
        assert data["id"] == "ann1"
        assert data["target"] == "sec1"
        assert data["text"] == "Note"
        assert data["author"] == "John"
        assert data["type"] == "warning"
        assert "created" in data


class TestAnnotationManager:
    """Test AnnotationManager class."""

    def test_create_manager(self) -> None:
        """Test creating annotation manager."""
        manager = AnnotationManager("report123")
        assert manager.report_id == "report123"
        assert len(manager._annotations) == 0

    def test_add_annotation(self) -> None:
        """Test adding annotation."""
        manager = AnnotationManager("report1")
        ann = manager.add("section1", "Important note", "John", "warning")
        assert ann.target == "section1"
        assert ann.text == "Important note"
        assert ann.author == "John"
        assert ann.type == "warning"
        assert len(manager._annotations) == 1

    def test_remove_annotation(self) -> None:
        """Test removing annotation."""
        manager = AnnotationManager("report1")
        ann = manager.add("section1", "Note")
        assert len(manager._annotations) == 1

        removed = manager.remove(ann.id)
        assert removed is True
        assert len(manager._annotations) == 0

    def test_remove_nonexistent(self) -> None:
        """Test removing nonexistent annotation."""
        manager = AnnotationManager("report1")
        removed = manager.remove("nonexistent")
        assert removed is False

    def test_get_for_target(self) -> None:
        """Test getting annotations for specific target."""
        manager = AnnotationManager("report1")
        manager.add("section1", "Note 1")
        manager.add("section2", "Note 2")
        manager.add("section1", "Note 3")

        annotations = manager.get_for_target("section1")
        assert len(annotations) == 2
        assert all(a.target == "section1" for a in annotations)

    def test_export(self) -> None:
        """Test exporting annotations."""
        manager = AnnotationManager("report1")
        manager.add("section1", "Note 1", "Alice")
        manager.add("section2", "Note 2", "Bob")

        exported = manager.export()
        assert len(exported) == 2
        assert all(isinstance(item, dict) for item in exported)


# =============================================================================
# =============================================================================


class TestReportSchedule:
    """Test ReportSchedule dataclass."""

    def test_create_schedule(self) -> None:
        """Test creating a schedule."""
        schedule = ReportSchedule(
            id="sched1",
            report_config={"title": "Daily Report"},
            frequency=ScheduleFrequency.DAILY,
        )
        assert schedule.id == "sched1"
        assert schedule.frequency == ScheduleFrequency.DAILY
        assert schedule.enabled is True

    def test_calculate_next_run_hourly(self) -> None:
        """Test calculating next run for hourly schedule."""
        schedule = ReportSchedule(id="s1", report_config={}, frequency=ScheduleFrequency.HOURLY)
        now = datetime.now()
        next_run = schedule.calculate_next_run()
        assert next_run > now
        assert (next_run - now).total_seconds() >= 3600

    def test_calculate_next_run_daily(self) -> None:
        """Test calculating next run for daily schedule."""
        schedule = ReportSchedule(id="s1", report_config={}, frequency=ScheduleFrequency.DAILY)
        now = datetime.now()
        next_run = schedule.calculate_next_run()
        assert (next_run - now).days >= 1

    def test_calculate_next_run_weekly(self) -> None:
        """Test calculating next run for weekly schedule."""
        schedule = ReportSchedule(id="s1", report_config={}, frequency=ScheduleFrequency.WEEKLY)
        now = datetime.now()
        next_run = schedule.calculate_next_run()
        assert (next_run - now).days >= 7

    def test_calculate_next_run_monthly(self) -> None:
        """Test calculating next run for monthly schedule."""
        schedule = ReportSchedule(id="s1", report_config={}, frequency=ScheduleFrequency.MONTHLY)
        now = datetime.now()
        next_run = schedule.calculate_next_run()
        assert (next_run - now).days >= 30


class TestReportScheduler:
    """Test ReportScheduler class."""

    def test_create_scheduler(self) -> None:
        """Test creating scheduler."""
        scheduler = ReportScheduler()
        assert len(scheduler._schedules) == 0
        assert scheduler._running is False

    def test_add_schedule(self) -> None:
        """Test adding schedule."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule(
            report_config={"title": "Test"},
            frequency=ScheduleFrequency.DAILY,
            recipients=["test@example.com"],
        )
        assert schedule_id in scheduler._schedules
        assert scheduler._schedules[schedule_id].frequency == ScheduleFrequency.DAILY

    def test_remove_schedule(self) -> None:
        """Test removing schedule."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule({}, ScheduleFrequency.DAILY)

        removed = scheduler.remove_schedule(schedule_id)
        assert removed is True
        assert schedule_id not in scheduler._schedules

    def test_remove_nonexistent_schedule(self) -> None:
        """Test removing nonexistent schedule."""
        scheduler = ReportScheduler()
        removed = scheduler.remove_schedule("nonexistent")
        assert removed is False

    def test_get_pending_empty(self) -> None:
        """Test getting pending schedules when none are due."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule({}, ScheduleFrequency.DAILY)

        # Set next_run to future
        scheduler._schedules[schedule_id].next_run = datetime.now() + timedelta(hours=1)

        pending = scheduler.get_pending()
        assert len(pending) == 0

    def test_get_pending_with_due_schedule(self) -> None:
        """Test getting pending schedules when one is due."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule({}, ScheduleFrequency.DAILY)

        # Set next_run to past
        scheduler._schedules[schedule_id].next_run = datetime.now() - timedelta(hours=1)

        pending = scheduler.get_pending()
        assert len(pending) == 1

    def test_get_pending_disabled_schedule(self) -> None:
        """Test that disabled schedules are not returned."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule({}, ScheduleFrequency.DAILY)

        # Disable and set to past
        scheduler._schedules[schedule_id].enabled = False
        scheduler._schedules[schedule_id].next_run = datetime.now() - timedelta(hours=1)

        pending = scheduler.get_pending()
        assert len(pending) == 0

    def test_execute_pending(self) -> None:
        """Test executing pending schedules."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule({}, ScheduleFrequency.HOURLY)
        scheduler._schedules[schedule_id].next_run = datetime.now() - timedelta(hours=1)

        generator = MagicMock()
        executed = scheduler.execute_pending(generator)

        assert len(executed) == 1
        assert schedule_id in executed
        generator.assert_called_once()

    def test_execute_pending_updates_next_run(self) -> None:
        """Test that execution updates next_run time."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule({}, ScheduleFrequency.HOURLY)
        old_next_run = datetime.now() - timedelta(hours=1)
        scheduler._schedules[schedule_id].next_run = old_next_run

        generator = MagicMock()
        scheduler.execute_pending(generator)

        new_next_run = scheduler._schedules[schedule_id].next_run
        assert new_next_run > old_next_run

    def test_execute_pending_handles_errors(self) -> None:
        """Test that execution handles generator errors."""
        scheduler = ReportScheduler()
        schedule_id = scheduler.add_schedule({}, ScheduleFrequency.DAILY)
        scheduler._schedules[schedule_id].next_run = datetime.now() - timedelta(hours=1)

        generator = MagicMock(side_effect=Exception("Generation failed"))
        executed = scheduler.execute_pending(generator)

        assert len(executed) == 0


# =============================================================================
# =============================================================================


class TestDistributionConfig:
    """Test DistributionConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating distribution config."""
        config = DistributionConfig(
            channel=DistributionChannel.EMAIL,
            recipients=["user@example.com"],
            settings={"smtp_server": "mail.example.com"},
        )
        assert config.channel == DistributionChannel.EMAIL
        assert len(config.recipients) == 1


class TestReportDistributor:
    """Test ReportDistributor class."""

    def test_create_distributor(self) -> None:
        """Test creating distributor."""
        distributor = ReportDistributor()
        assert len(distributor._handlers) == 0

    def test_register_handler(self) -> None:
        """Test registering distribution handler."""
        distributor = ReportDistributor()
        handler = MagicMock()

        distributor.register_handler(DistributionChannel.EMAIL, handler)
        assert DistributionChannel.EMAIL in distributor._handlers

    def test_distribute_single_channel(self, tmp_path: Path) -> None:
        """Test distributing to single channel."""
        distributor = ReportDistributor()
        handler = MagicMock(return_value=True)
        distributor.register_handler(DistributionChannel.EMAIL, handler)

        report_path = tmp_path / "report.pdf"
        report_path.write_text("test report")

        config = DistributionConfig(
            channel=DistributionChannel.EMAIL, recipients=["test@example.com"]
        )
        results = distributor.distribute(report_path, [config])

        assert results["EMAIL"] is True
        handler.assert_called_once()

    def test_distribute_multiple_channels(self, tmp_path: Path) -> None:
        """Test distributing to multiple channels."""
        distributor = ReportDistributor()

        email_handler = MagicMock(return_value=True)
        s3_handler = MagicMock(return_value=True)

        distributor.register_handler(DistributionChannel.EMAIL, email_handler)
        distributor.register_handler(DistributionChannel.S3, s3_handler)

        report_path = tmp_path / "report.pdf"
        report_path.write_text("test")

        configs = [
            DistributionConfig(channel=DistributionChannel.EMAIL),
            DistributionConfig(channel=DistributionChannel.S3),
        ]

        results = distributor.distribute(report_path, configs)

        assert results["EMAIL"] is True
        assert results["S3"] is True
        assert email_handler.called
        assert s3_handler.called

    def test_distribute_no_handler(self, tmp_path: Path) -> None:
        """Test distributing when no handler registered."""
        distributor = ReportDistributor()
        report_path = tmp_path / "report.pdf"
        report_path.write_text("test")

        config = DistributionConfig(channel=DistributionChannel.EMAIL)
        results = distributor.distribute(report_path, [config])

        assert results["EMAIL"] is False

    def test_distribute_handler_error(self, tmp_path: Path) -> None:
        """Test distribution when handler raises error."""
        distributor = ReportDistributor()
        handler = MagicMock(side_effect=Exception("Connection failed"))
        distributor.register_handler(DistributionChannel.EMAIL, handler)

        report_path = tmp_path / "report.pdf"
        report_path.write_text("test")

        config = DistributionConfig(channel=DistributionChannel.EMAIL)
        results = distributor.distribute(report_path, [config])

        assert results["EMAIL"] is False


# =============================================================================
# =============================================================================


class TestReportArchive:
    """Test ReportArchive class."""

    def test_create_archive(self, tmp_path: Path) -> None:
        """Test creating archive."""
        archive_dir = tmp_path / "archive"
        archive = ReportArchive(archive_dir)
        assert archive_dir.exists()
        assert len(archive._index) == 0

    def test_archive_report(self, tmp_path: Path) -> None:
        """Test archiving a report."""
        archive = ReportArchive(tmp_path / "archive")

        # Create test report
        report_path = tmp_path / "report.pdf"
        report_path.write_bytes(b"test report content")

        report_id = archive.archive(report_path, {"author": "John"})

        assert report_id in archive._index
        archived = archive._index[report_id]
        assert archived.name == "report.pdf"
        assert archived.path.exists()
        assert archived.metadata["author"] == "John"

    def test_archive_calculates_checksum(self, tmp_path: Path) -> None:
        """Test that archiving calculates checksum."""
        archive = ReportArchive(tmp_path / "archive")
        report_path = tmp_path / "report.pdf"
        report_path.write_bytes(b"test content")

        report_id = archive.archive(report_path)
        archived = archive._index[report_id]

        assert len(archived.checksum) == 64  # SHA256 hex

    def test_retrieve_archived_report(self, tmp_path: Path) -> None:
        """Test retrieving archived report."""
        archive = ReportArchive(tmp_path / "archive")
        report_path = tmp_path / "report.pdf"
        report_path.write_bytes(b"content")

        report_id = archive.archive(report_path)
        retrieved_path = archive.retrieve(report_id)

        assert retrieved_path is not None
        assert retrieved_path.exists()
        assert retrieved_path.read_bytes() == b"content"

    def test_retrieve_nonexistent(self, tmp_path: Path) -> None:
        """Test retrieving nonexistent report."""
        archive = ReportArchive(tmp_path / "archive")
        retrieved = archive.retrieve("nonexistent")
        assert retrieved is None

    def test_cleanup_expired(self, tmp_path: Path) -> None:
        """Test cleaning up expired archives."""
        archive = ReportArchive(tmp_path / "archive")
        report_path = tmp_path / "report.pdf"
        report_path.write_bytes(b"content")

        report_id = archive.archive(report_path)

        # Manually set to expired
        archive._index[report_id].created = datetime.now() - timedelta(days=400)
        archive._index[report_id].retention_days = 365

        removed = archive.cleanup_expired()

        assert removed == 1
        assert report_id not in archive._index

    def test_cleanup_not_expired(self, tmp_path: Path) -> None:
        """Test that non-expired archives are not removed."""
        archive = ReportArchive(tmp_path / "archive")
        report_path = tmp_path / "report.pdf"
        report_path.write_bytes(b"content")

        report_id = archive.archive(report_path)
        removed = archive.cleanup_expired()

        assert removed == 0
        assert report_id in archive._index


# =============================================================================
# =============================================================================


class TestReportSearchIndex:
    """Test ReportSearchIndex class."""

    def test_create_index(self) -> None:
        """Test creating search index."""
        index = ReportSearchIndex()
        assert len(index._index) == 0

    def test_index_report(self) -> None:
        """Test indexing a report."""
        index = ReportSearchIndex()
        content = "This is a test report about signal analysis"
        metadata = {"title": "Test Report", "author": "John"}

        index.index_report("report1", content, metadata)

        assert "report1" in index._index
        assert "signal" in index._index["report1"]["words"]

    def test_search_simple(self) -> None:
        """Test simple search."""
        index = ReportSearchIndex()
        index.index_report("r1", "signal analysis report", {"name": "Report 1"})
        index.index_report("r2", "power measurement test", {"name": "Report 2"})

        results = index.search("signal")

        assert len(results) == 1
        assert results[0].report_id == "r1"

    def test_search_multiple_words(self) -> None:
        """Test search with multiple words."""
        index = ReportSearchIndex()
        index.index_report("r1", "signal analysis report", {"name": "Report 1"})
        index.index_report("r2", "signal power measurement", {"name": "Report 2"})

        results = index.search("signal power")

        assert len(results) == 2
        # r2 should score higher (matches both words)
        assert results[0].report_id == "r2"

    def test_search_case_insensitive(self) -> None:
        """Test case-insensitive search."""
        index = ReportSearchIndex()
        index.index_report("r1", "SIGNAL Analysis", {"name": "Report 1"})

        results = index.search("signal")
        assert len(results) == 1

    def test_search_with_limit(self) -> None:
        """Test search result limit."""
        index = ReportSearchIndex()
        for i in range(20):
            index.index_report(f"r{i}", "test report", {"name": f"Report {i}"})

        results = index.search("test", limit=5)
        assert len(results) == 5

    def test_search_no_matches(self) -> None:
        """Test search with no matches."""
        index = ReportSearchIndex()
        index.index_report("r1", "signal analysis", {})

        results = index.search("nonexistent")
        assert len(results) == 0

    def test_search_result_scoring(self) -> None:
        """Test that search results are scored correctly."""
        index = ReportSearchIndex()
        index.index_report("r1", "test", {"name": "R1"})
        index.index_report("r2", "test report analysis", {"name": "R2"})

        # Search for 2 words
        results = index.search("test report")

        # r2 should score higher (100% match vs 50%)
        assert results[0].report_id == "r2"
        assert results[0].score > results[1].score


# =============================================================================
# =============================================================================


class TestReportVersionControl:
    """Test ReportVersionControl class."""

    def test_create_version_control(self, tmp_path: Path) -> None:
        """Test creating version control."""
        storage_dir = tmp_path / "versions"
        vc = ReportVersionControl(storage_dir)
        assert storage_dir.exists()
        assert len(vc._versions) == 0

    def test_commit_first_version(self, tmp_path: Path) -> None:
        """Test committing first version."""
        vc = ReportVersionControl(tmp_path / "versions")
        report_path = tmp_path / "report.pdf"
        report_path.write_text("Version 1 content")

        version = vc.commit("report1", report_path, "Alice", "Initial version")

        assert version == 1
        assert "report1" in vc._versions
        assert len(vc._versions["report1"]) == 1

    def test_commit_multiple_versions(self, tmp_path: Path) -> None:
        """Test committing multiple versions."""
        vc = ReportVersionControl(tmp_path / "versions")
        report_path = tmp_path / "report.pdf"

        report_path.write_text("Version 1")
        v1 = vc.commit("report1", report_path, "Alice", "V1")

        report_path.write_text("Version 2")
        v2 = vc.commit("report1", report_path, "Bob", "V2")

        assert v1 == 1
        assert v2 == 2
        assert len(vc._versions["report1"]) == 2

    def test_get_version(self, tmp_path: Path) -> None:
        """Test getting specific version."""
        vc = ReportVersionControl(tmp_path / "versions")
        report_path = tmp_path / "report.pdf"

        report_path.write_text("Version 1")
        vc.commit("report1", report_path, "Alice", "V1")

        report_path.write_text("Version 2")
        vc.commit("report1", report_path, "Alice", "V2")

        v1_path = vc.get_version("report1", 1)
        assert v1_path is not None
        assert v1_path.read_text() == "Version 1"

    def test_get_version_nonexistent(self, tmp_path: Path) -> None:
        """Test getting nonexistent version."""
        vc = ReportVersionControl(tmp_path / "versions")
        path = vc.get_version("nonexistent", 1)
        assert path is None

    def test_get_history(self, tmp_path: Path) -> None:
        """Test getting version history."""
        vc = ReportVersionControl(tmp_path / "versions")
        report_path = tmp_path / "report.pdf"

        report_path.write_text("V1")
        vc.commit("report1", report_path, "Alice", "First")

        report_path.write_text("V2")
        vc.commit("report1", report_path, "Bob", "Second")

        history = vc.get_history("report1")
        assert len(history) == 2
        assert history[0].author == "Alice"
        assert history[1].author == "Bob"

    def test_diff_versions(self, tmp_path: Path) -> None:
        """Test getting diff between versions."""
        vc = ReportVersionControl(tmp_path / "versions")
        report_path = tmp_path / "report.txt"

        report_path.write_text("Line 1\nLine 2\n")
        vc.commit("report1", report_path, "Alice", "V1")

        report_path.write_text("Line 1\nLine 2 modified\nLine 3\n")
        vc.commit("report1", report_path, "Alice", "V2")

        diff = vc.diff("report1", 1, 2)
        assert "Line 2" in diff

    def test_diff_nonexistent_version(self, tmp_path: Path) -> None:
        """Test diff with nonexistent version."""
        vc = ReportVersionControl(tmp_path / "versions")
        diff = vc.diff("report1", 1, 2)
        assert "not found" in diff


# =============================================================================
# =============================================================================


class TestApprovalWorkflow:
    """Test ApprovalWorkflow class."""

    def test_create_workflow(self) -> None:
        """Test creating approval workflow."""
        workflow = ApprovalWorkflow()
        assert len(workflow._records) == 0

    def test_submit_for_review(self) -> None:
        """Test submitting report for review."""
        workflow = ApprovalWorkflow()
        record = workflow.submit_for_review("report1", "Alice")

        assert record.report_id == "report1"
        assert record.status == ApprovalStatus.PENDING_REVIEW
        assert record.submitter == "Alice"
        assert record.submitted_at is not None

    def test_approve_report(self) -> None:
        """Test approving report."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("report1", "Alice")

        record = workflow.approve("report1", "Bob", "Looks good")

        assert record.status == ApprovalStatus.APPROVED
        assert record.reviewer == "Bob"
        assert record.comments == "Looks good"
        assert record.reviewed_at is not None

    def test_reject_report(self) -> None:
        """Test rejecting report."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("report1", "Alice")

        record = workflow.reject("report1", "Bob", "Needs revision")

        assert record.status == ApprovalStatus.REJECTED
        assert record.reviewer == "Bob"
        assert record.comments == "Needs revision"

    def test_approve_nonexistent(self) -> None:
        """Test approving nonexistent report."""
        workflow = ApprovalWorkflow()

        with pytest.raises(ValueError, match="not in workflow"):
            workflow.approve("nonexistent", "Bob")

    def test_reject_nonexistent(self) -> None:
        """Test rejecting nonexistent report."""
        workflow = ApprovalWorkflow()

        with pytest.raises(ValueError, match="not in workflow"):
            workflow.reject("nonexistent", "Bob", "Bad")

    def test_on_status_change_callback(self) -> None:
        """Test status change callbacks."""
        workflow = ApprovalWorkflow()
        callback = MagicMock()

        workflow.on_status_change(ApprovalStatus.APPROVED, callback)
        workflow.submit_for_review("report1", "Alice")
        workflow.approve("report1", "Bob")

        callback.assert_called_once()

    def test_callback_error_handling(self) -> None:
        """Test that callback errors don't break workflow."""
        workflow = ApprovalWorkflow()
        bad_callback = MagicMock(side_effect=Exception("Callback failed"))

        workflow.on_status_change(ApprovalStatus.APPROVED, bad_callback)
        workflow.submit_for_review("report1", "Alice")

        # Should not raise exception
        workflow.approve("report1", "Bob")


# =============================================================================
# =============================================================================


class TestComplianceChecker:
    """Test ComplianceChecker class."""

    def test_create_checker(self) -> None:
        """Test creating compliance checker."""
        checker = ComplianceChecker()
        assert len(checker._rules) == 0

    def test_add_rule(self) -> None:
        """Test adding compliance rule."""
        checker = ComplianceChecker()

        def check_has_title(data: dict) -> bool:  # type: ignore[type-arg]
            return "title" in data

        checker.add_rule("has_title", check_has_title, "Report must have title")
        assert len(checker._rules) == 1

    def test_check_passing(self) -> None:
        """Test compliance check that passes."""
        checker = ComplianceChecker()
        checker.add_rule("has_title", lambda d: "title" in d)

        result = checker.check({"title": "Test Report"})

        assert result.passed is True
        assert len(result.violations) == 0

    def test_check_failing(self) -> None:
        """Test compliance check that fails."""
        checker = ComplianceChecker()
        checker.add_rule("has_title", lambda d: "title" in d, "Missing title")

        result = checker.check({"author": "John"})

        assert result.passed is False
        assert len(result.violations) == 1
        assert "has_title" in result.violations[0][0]

    def test_check_warnings(self) -> None:
        """Test compliance warnings."""
        checker = ComplianceChecker()
        checker.add_rule("has_logo", lambda d: "logo" in d, "Logo recommended", severity="warning")

        result = checker.check({"title": "Test"})

        assert result.passed is True  # Warnings don't fail
        assert len(result.warnings) == 1

    def test_check_multiple_rules(self) -> None:
        """Test multiple compliance rules."""
        checker = ComplianceChecker()
        checker.add_rule("has_title", lambda d: "title" in d)
        checker.add_rule("has_author", lambda d: "author" in d)
        checker.add_rule("has_date", lambda d: "date" in d)

        result = checker.check({"title": "Test"})

        assert result.passed is False
        assert len(result.violations) == 2  # Missing author and date

    def test_rule_exception_handling(self) -> None:
        """Test that rule exceptions are handled."""
        checker = ComplianceChecker()

        def bad_check(data: dict) -> bool:  # type: ignore[type-arg]
            raise Exception("Check failed")

        checker.add_rule("bad_rule", bad_check)

        # Should not raise exception
        result = checker.check({"title": "Test"})
        assert result.passed is True  # Failed checks are logged, not counted


# =============================================================================
# =============================================================================


class TestReportLocalizer:
    """Test ReportLocalizer class."""

    def test_create_localizer(self) -> None:
        """Test creating localizer."""
        localizer = ReportLocalizer()
        assert localizer.default_locale == "en_US"
        assert "en_US" in localizer._locales

    def test_get_string_default_locale(self) -> None:
        """Test getting string in default locale."""
        localizer = ReportLocalizer()
        text = localizer.get_string("title")
        assert text == "Report"

    def test_get_string_specific_locale(self) -> None:
        """Test getting string in specific locale."""
        localizer = ReportLocalizer()
        text = localizer.get_string("title", locale="de_DE")
        assert text == "Bericht"

    def test_get_string_missing_key(self) -> None:
        """Test getting missing string key."""
        localizer = ReportLocalizer()
        text = localizer.get_string("nonexistent")
        assert text == "nonexistent"  # Returns key if not found

    def test_format_number_en_us(self) -> None:
        """Test formatting number for en_US."""
        localizer = ReportLocalizer()
        formatted = localizer.format_number(1234.56, locale="en_US")
        assert "1,234.56" in formatted

    def test_format_number_de_de(self) -> None:
        """Test formatting number for de_DE."""
        localizer = ReportLocalizer()
        formatted = localizer.format_number(1234.56, locale="de_DE")
        # German uses comma for decimal, period for thousands
        assert "," in formatted


# =============================================================================
# =============================================================================


class TestAccessibilityFeatures:
    """Test accessibility features."""

    def test_add_aria_landmarks(self) -> None:
        """Test adding ARIA landmarks."""
        html = '<div class="report">Content</div>'
        options = AccessibilityOptions()

        enhanced = add_accessibility_features(html, options)

        assert 'role="main"' in enhanced
        assert 'aria-label="Report Content"' in enhanced

    def test_add_skip_navigation(self) -> None:
        """Test adding skip navigation link."""
        html = "<body><div>Content</div></body>"
        options = AccessibilityOptions()

        enhanced = add_accessibility_features(html, options)

        assert "Skip to main content" in enhanced
        assert "skip-link" in enhanced

    def test_high_contrast_mode(self) -> None:
        """Test high contrast mode."""
        html = "<html><head></head><body>Content</body></html>"
        options = AccessibilityOptions(high_contrast=True)

        enhanced = add_accessibility_features(html, options)

        assert "background: white" in enhanced
        assert "color: black" in enhanced

    def test_no_high_contrast(self) -> None:
        """Test without high contrast mode."""
        html = "<html><head></head><body>Content</body></html>"
        options = AccessibilityOptions(high_contrast=False)

        enhanced = add_accessibility_features(html, options)

        assert "background: white" not in enhanced


# =============================================================================
# =============================================================================


class TestReportEncryption:
    """Test ReportEncryption class."""

    def test_encrypt_decrypt_content(self) -> None:
        """Test encrypting and decrypting content."""
        content = b"Secret report content"
        password = "strongpassword"

        encrypted = ReportEncryption.encrypt_content(content, password)
        assert encrypted != content

        decrypted = ReportEncryption.decrypt_content(encrypted, password)
        assert decrypted == content

    def test_wrong_password(self) -> None:
        """Test decryption with wrong password."""
        content = b"Secret content"
        encrypted = ReportEncryption.encrypt_content(content, "password1")
        decrypted = ReportEncryption.decrypt_content(encrypted, "password2")

        assert decrypted != content

    def test_encrypt_decrypt_file(self, tmp_path: Path) -> None:
        """Test encrypting and decrypting files."""
        input_path = tmp_path / "report.txt"
        encrypted_path = tmp_path / "report.enc"
        decrypted_path = tmp_path / "report_dec.txt"

        input_path.write_bytes(b"Confidential report")

        ReportEncryption.encrypt_file(input_path, encrypted_path, "password")
        assert encrypted_path.exists()
        assert encrypted_path.read_bytes() != b"Confidential report"

        ReportEncryption.decrypt_file(encrypted_path, decrypted_path, "password")
        assert decrypted_path.read_bytes() == b"Confidential report"


# =============================================================================
# =============================================================================


class TestWatermarking:
    """Test watermarking functionality."""

    def test_add_watermark_default(self) -> None:
        """Test adding watermark with defaults."""
        html = "<html><head></head><body>Content</body></html>"
        watermark = Watermark()

        watermarked = add_watermark(html, watermark)

        assert "CONFIDENTIAL" in watermarked
        assert "watermark" in watermarked

    def test_add_watermark_custom_text(self) -> None:
        """Test adding watermark with custom text."""
        html = "<html><head></head><body>Content</body></html>"
        watermark = Watermark(text="DRAFT")

        watermarked = add_watermark(html, watermark)

        assert "DRAFT" in watermarked

    def test_add_watermark_custom_rotation(self) -> None:
        """Test watermark with custom rotation."""
        html = "<html><head></head><body>Content</body></html>"
        watermark = Watermark(rotation=-30)

        watermarked = add_watermark(html, watermark)

        assert "rotate(-30deg)" in watermarked

    def test_add_watermark_custom_opacity(self) -> None:
        """Test watermark with custom opacity."""
        html = "<html><head></head><body>Content</body></html>"
        watermark = Watermark(opacity=0.3)

        watermarked = add_watermark(html, watermark)

        assert "0.3" in watermarked


# =============================================================================
# =============================================================================


class TestAuditTrail:
    """Test AuditTrail class."""

    def test_create_audit_trail(self) -> None:
        """Test creating audit trail."""
        trail = AuditTrail()
        assert len(trail._entries) == 0

    def test_log_entry(self) -> None:
        """Test logging audit entry."""
        trail = AuditTrail()
        entry = trail.log("report1", "created", "Alice", {"version": 1})

        assert entry.report_id == "report1"
        assert entry.action == "created"
        assert entry.user == "Alice"
        assert entry.details["version"] == 1

    def test_get_for_report(self) -> None:
        """Test getting entries for specific report."""
        trail = AuditTrail()
        trail.log("report1", "created", "Alice")
        trail.log("report2", "created", "Bob")
        trail.log("report1", "modified", "Alice")

        entries = trail.get_for_report("report1")

        assert len(entries) == 2
        assert all(e.report_id == "report1" for e in entries)

    def test_get_by_user(self) -> None:
        """Test getting entries by user."""
        trail = AuditTrail()
        trail.log("report1", "created", "Alice")
        trail.log("report2", "created", "Bob")
        trail.log("report3", "created", "Alice")

        entries = trail.get_by_user("Alice")

        assert len(entries) == 2
        assert all(e.user == "Alice" for e in entries)

    def test_export_json(self) -> None:
        """Test exporting audit trail as JSON."""
        trail = AuditTrail()
        trail.log("report1", "created", "Alice")
        trail.log("report1", "approved", "Bob")

        json_str = trail.export("json")
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["action"] == "created"
        assert data[1]["action"] == "approved"

    def test_persist_to_file(self, tmp_path: Path) -> None:
        """Test persisting audit trail to file."""
        storage_path = tmp_path / "audit.json"
        trail = AuditTrail(storage_path=storage_path)

        trail.log("report1", "created", "Alice")

        assert storage_path.exists()
        data = json.loads(storage_path.read_text())
        assert len(data) == 1

    def test_no_persist_without_path(self) -> None:
        """Test that entries are not persisted without storage path."""
        trail = AuditTrail()
        trail.log("report1", "created", "Alice")
        # Should not raise error
