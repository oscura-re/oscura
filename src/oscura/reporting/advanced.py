"""Advanced reporting features for Oscura.

This module provides advanced reporting capabilities including interactive
reports, scheduled generation, distribution, versioning, and compliance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================


@dataclass
class TemplateField:
    """Customizable template field.

    Attributes:
        name: Field identifier
        type: Field type (text, number, image, table, chart)
        default: Default value
        required: Whether field is required
        validation: Validation rule (regex pattern)
    """

    name: str
    type: str = "text"
    default: Any = None
    required: bool = False
    validation: str | None = None
    description: str = ""


@dataclass
class CustomTemplate:
    """Customizable report template.

    Attributes:
        name: Template name
        version: Template version
        fields: List of customizable fields
        layout: Layout configuration
        styles: CSS/style overrides
        includes: Included partial templates

    Example:
        >>> template = CustomTemplate(
        ...     name="compliance_report",
        ...     fields=[
        ...         TemplateField("company_name", required=True),
        ...         TemplateField("logo", type="image")
        ...     ]
        ... )

    References:
        REPORT-011: Report Customization Templates
    """

    name: str
    version: str = "1.0.0"
    fields: list[TemplateField] = field(default_factory=list)
    layout: dict[str, Any] = field(default_factory=dict)
    styles: dict[str, str] = field(default_factory=dict)
    includes: list[str] = field(default_factory=list)
    description: str = ""

    def validate_data(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate data against template fields.

        Args:
            data: Data dictionary

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        for f in self.fields:
            if f.required and f.name not in data:
                errors.append(f"Required field '{f.name}' missing")
            elif f.name in data and f.validation:
                if not re.match(f.validation, str(data[f.name])):
                    errors.append(f"Field '{f.name}' failed validation")
        return len(errors) == 0, errors

    def render(self, data: dict[str, Any]) -> str:
        """Render template with data.

        Args:
            data: Data dictionary

        Returns:
            Rendered content
        """
        # Simple placeholder substitution
        content = self.layout.get("template", "")
        for f in self.fields:
            placeholder = f"{{{{{f.name}}}}}"
            value = data.get(f.name, f.default or "")
            content = content.replace(placeholder, str(value))
        return content  # type: ignore[no-any-return]


# =============================================================================
# =============================================================================


class InteractiveElementType(Enum):
    """Types of interactive elements."""

    ZOOMABLE_CHART = auto()
    COLLAPSIBLE_SECTION = auto()
    FILTER_DROPDOWN = auto()
    SORTABLE_TABLE = auto()
    TOOLTIP = auto()
    DRILL_DOWN = auto()
    TOGGLE = auto()


@dataclass
class InteractiveElement:
    """Interactive element for HTML reports.

    Attributes:
        id: Element ID
        type: Element type
        data: Element data
        options: Configuration options
        script: JavaScript code for interactivity

    Example:
        >>> element = InteractiveElement(
        ...     id="chart1",
        ...     type=InteractiveElementType.ZOOMABLE_CHART,
        ...     data=chart_data
        ... )

    References:
        REPORT-012: Interactive Report Elements
    """

    id: str
    type: InteractiveElementType
    data: Any = None
    options: dict[str, Any] = field(default_factory=dict)
    script: str = ""

    def to_html(self) -> str:
        """Generate HTML for interactive element."""
        html_parts = [f'<div id="{self.id}" class="interactive-{self.type.name.lower()}">']

        if self.type == InteractiveElementType.COLLAPSIBLE_SECTION:
            html_parts.append(f"""
                <button class="collapsible" onclick="toggleSection('{self.id}')">
                    {self.options.get("title", "Section")}
                </button>
                <div class="content" style="display:none;">
                    {self.data or ""}
                </div>
            """)
        elif self.type == InteractiveElementType.SORTABLE_TABLE:
            html_parts.append(f"""
                <table class="sortable" data-sort-enabled="true">
                    {self.data or ""}
                </table>
            """)
        elif self.type == InteractiveElementType.TOOLTIP:
            html_parts.append(f'''
                <span class="tooltip" data-tooltip="{self.options.get("text", "")}">
                    {self.data or ""}
                </span>
            ''')
        else:
            html_parts.append(str(self.data or ""))

        html_parts.append("</div>")
        return "\n".join(html_parts)


# =============================================================================
# =============================================================================


@dataclass
class Annotation:
    """Report annotation.

    Attributes:
        id: Unique annotation ID
        target: Target element ID or location
        text: Annotation text
        author: Author name
        created: Creation timestamp
        type: Annotation type (note, warning, highlight, etc.)
        position: Position info for placement

    References:
        REPORT-013: Report Annotations
    """

    id: str
    target: str
    text: str
    author: str = ""
    created: datetime = field(default_factory=datetime.now)
    type: str = "note"
    position: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "target": self.target,
            "text": self.text,
            "author": self.author,
            "created": self.created.isoformat(),
            "type": self.type,
            "position": self.position,
        }


class AnnotationManager:
    """Manager for report annotations.

    References:
        REPORT-013: Report Annotations
    """

    def __init__(self, report_id: str):
        self.report_id = report_id
        self._annotations: list[Annotation] = []

    def add(self, target: str, text: str, author: str = "", type_: str = "note") -> Annotation:
        """Add annotation."""
        annotation = Annotation(
            id=str(uuid.uuid4()), target=target, text=text, author=author, type=type_
        )
        self._annotations.append(annotation)
        return annotation

    def remove(self, annotation_id: str) -> bool:
        """Remove annotation."""
        for i, ann in enumerate(self._annotations):
            if ann.id == annotation_id:
                del self._annotations[i]
                return True
        return False

    def get_for_target(self, target: str) -> list[Annotation]:
        """Get annotations for target."""
        return [a for a in self._annotations if a.target == target]

    def export(self) -> list[dict[str, Any]]:
        """Export all annotations."""
        return [a.to_dict() for a in self._annotations]


# =============================================================================
# =============================================================================


class ScheduleFrequency(Enum):
    """Report schedule frequency."""

    ONCE = auto()
    HOURLY = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    CUSTOM = auto()


@dataclass
class ReportSchedule:
    """Scheduled report configuration.

    Attributes:
        id: Schedule ID
        report_config: Report configuration
        frequency: Generation frequency
        next_run: Next scheduled run time
        enabled: Whether schedule is active
        recipients: Email recipients
        cron_expression: Cron expression for custom schedules

    References:
        REPORT-017: Report Scheduling
    """

    id: str
    report_config: dict[str, Any]
    frequency: ScheduleFrequency = ScheduleFrequency.DAILY
    next_run: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    recipients: list[str] = field(default_factory=list)
    cron_expression: str | None = None

    def calculate_next_run(self) -> datetime:
        """Calculate next run time."""
        now = datetime.now()
        if self.frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif self.frequency == ScheduleFrequency.DAILY:
            return now + timedelta(days=1)
        elif self.frequency == ScheduleFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif self.frequency == ScheduleFrequency.MONTHLY:
            return now + timedelta(days=30)
        return now


class ReportScheduler:
    """Report scheduler for automated generation.

    References:
        REPORT-017: Report Scheduling
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        self._schedules: dict[str, ReportSchedule] = {}
        self._running = False

    def add_schedule(
        self,
        report_config: dict[str, Any],
        frequency: ScheduleFrequency,
        recipients: list[str] | None = None,
    ) -> str:
        """Add new schedule."""
        schedule = ReportSchedule(
            id=str(uuid.uuid4()),
            report_config=report_config,
            frequency=frequency,
            recipients=recipients or [],
        )
        self._schedules[schedule.id] = schedule
        return schedule.id

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove schedule."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            return True
        return False

    def get_pending(self) -> list[ReportSchedule]:
        """Get schedules due for execution."""
        now = datetime.now()
        return [s for s in self._schedules.values() if s.enabled and s.next_run <= now]

    def execute_pending(self, generator: Callable[[dict[str, Any]], Any]) -> list[str]:
        """Execute pending schedules."""
        executed = []
        for schedule in self.get_pending():
            try:
                generator(schedule.report_config)
                schedule.next_run = schedule.calculate_next_run()
                executed.append(schedule.id)
            except Exception as e:
                logger.error(f"Scheduled report failed: {e}")
        return executed


# =============================================================================
# =============================================================================


class DistributionChannel(Enum):
    """Distribution channels."""

    EMAIL = auto()
    FILE_SHARE = auto()
    WEBHOOK = auto()
    S3 = auto()
    SFTP = auto()


@dataclass
class DistributionConfig:
    """Distribution configuration.

    References:
        REPORT-020: Report Distribution
    """

    channel: DistributionChannel
    recipients: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


class ReportDistributor:
    """Distributes reports to configured channels.

    References:
        REPORT-020: Report Distribution
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        self._handlers: dict[DistributionChannel, Callable] = {}  # type: ignore[type-arg]

    def register_handler(
        self,
        channel: DistributionChannel,
        handler: Callable[[Path, DistributionConfig], bool],
    ) -> None:
        """Register distribution handler."""
        self._handlers[channel] = handler

    def distribute(self, report_path: Path, configs: list[DistributionConfig]) -> dict[str, bool]:
        """Distribute report to all configured channels."""
        results = {}
        for config in configs:
            handler = self._handlers.get(config.channel)
            if handler:
                try:
                    results[config.channel.name] = handler(report_path, config)
                except Exception as e:
                    logger.error(f"Distribution failed for {config.channel}: {e}")
                    results[config.channel.name] = False
            else:
                logger.warning(f"No handler for channel: {config.channel}")
                results[config.channel.name] = False
        return results


# =============================================================================
# =============================================================================


@dataclass
class ArchivedReport:
    """Archived report metadata.

    References:
        REPORT-021: Report Archiving
    """

    id: str
    name: str
    path: Path
    created: datetime
    size: int
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retention_days: int = 365


class ReportArchive:
    """Report archiving system.

    References:
        REPORT-021: Report Archiving
    """

    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
        archive_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, ArchivedReport] = {}

    def archive(self, report_path: Path, metadata: dict[str, Any] | None = None) -> str:
        """Archive a report."""
        report_id = str(uuid.uuid4())

        # Calculate checksum
        with open(report_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Copy to archive
        archive_path = self.archive_dir / f"{report_id}_{report_path.name}"
        import shutil

        shutil.copy2(report_path, archive_path)

        archived = ArchivedReport(
            id=report_id,
            name=report_path.name,
            path=archive_path,
            created=datetime.now(),
            size=archive_path.stat().st_size,
            checksum=checksum,
            metadata=metadata or {},
        )
        self._index[report_id] = archived

        logger.info(f"Archived report: {report_id}")
        return report_id

    def retrieve(self, report_id: str) -> Path | None:
        """Retrieve archived report."""
        if report_id in self._index:
            return self._index[report_id].path
        return None

    def cleanup_expired(self) -> int:
        """Remove expired archives."""
        now = datetime.now()
        removed = 0
        for report_id, archived in list(self._index.items()):
            age = (now - archived.created).days
            if age > archived.retention_days:
                archived.path.unlink(missing_ok=True)
                del self._index[report_id]
                removed += 1
        return removed


# =============================================================================
# =============================================================================


@dataclass
class SearchResult:
    """Report search result.

    References:
        REPORT-022: Report Search
    """

    report_id: str
    name: str
    score: float
    highlights: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReportSearchIndex:
    """Full-text search index for reports.

    References:
        REPORT-022: Report Search
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        self._index: dict[str, dict[str, Any]] = {}

    def index_report(self, report_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Add report to search index."""
        # Simple word-based indexing
        words = set(content.lower().split())
        self._index[report_id] = {
            "words": words,
            "content": content,
            "metadata": metadata,
        }

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for reports."""
        query_words = set(query.lower().split())
        results = []

        for report_id, doc in self._index.items():
            # Simple scoring: intersection of words
            matches = query_words & doc["words"]
            if matches:
                score = len(matches) / len(query_words)
                results.append(
                    SearchResult(
                        report_id=report_id,
                        name=doc["metadata"].get("name", report_id),
                        score=score,
                        highlights=[f"...{m}..." for m in matches],
                        metadata=doc["metadata"],
                    )
                )

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]


# =============================================================================
# =============================================================================


@dataclass
class ReportVersion:
    """Report version entry.

    References:
        REPORT-023: Report Versioning
    """

    version: int
    created: datetime
    author: str
    changes: str
    checksum: str
    path: Path


class ReportVersionControl:
    """Version control for reports.

    References:
        REPORT-023: Report Versioning
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        storage_dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, list[ReportVersion]] = {}

    def commit(self, report_id: str, report_path: Path, author: str, changes: str) -> int:
        """Commit new version of report."""
        if report_id not in self._versions:
            self._versions[report_id] = []

        version = len(self._versions[report_id]) + 1

        # Copy to versioned storage
        version_path = self.storage_dir / f"{report_id}_v{version}{report_path.suffix}"
        import shutil

        shutil.copy2(report_path, version_path)

        # Calculate checksum
        with open(version_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        entry = ReportVersion(
            version=version,
            created=datetime.now(),
            author=author,
            changes=changes,
            checksum=checksum,
            path=version_path,
        )
        self._versions[report_id].append(entry)

        logger.info(f"Committed {report_id} version {version}")
        return version

    def get_version(self, report_id: str, version: int) -> Path | None:
        """Get specific version of report."""
        if report_id in self._versions:
            for v in self._versions[report_id]:
                if v.version == version:
                    return v.path
        return None

    def get_history(self, report_id: str) -> list[ReportVersion]:
        """Get version history."""
        return self._versions.get(report_id, [])

    def diff(self, report_id: str, v1: int, v2: int) -> str:
        """Get diff between versions."""
        path1 = self.get_version(report_id, v1)
        path2 = self.get_version(report_id, v2)

        if not path1 or not path2:
            return "Version not found"

        # Simple text diff
        with open(path1) as f1, open(path2) as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

        import difflib

        diff = difflib.unified_diff(lines1, lines2, lineterm="")
        return "\n".join(diff)


# =============================================================================
# =============================================================================


class ApprovalStatus(Enum):
    """Approval status."""

    DRAFT = auto()
    PENDING_REVIEW = auto()
    APPROVED = auto()
    REJECTED = auto()
    PUBLISHED = auto()


@dataclass
class ApprovalRecord:
    """Approval workflow record.

    References:
        REPORT-024: Report Approval Workflow
    """

    report_id: str
    status: ApprovalStatus = ApprovalStatus.DRAFT
    submitter: str = ""
    reviewer: str | None = None
    submitted_at: datetime | None = None
    reviewed_at: datetime | None = None
    comments: str = ""


class ApprovalWorkflow:
    """Report approval workflow manager.

    References:
        REPORT-024: Report Approval Workflow
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        self._records: dict[str, ApprovalRecord] = {}
        self._callbacks: dict[ApprovalStatus, list[Callable]] = {}  # type: ignore[type-arg]

    def submit_for_review(self, report_id: str, submitter: str) -> ApprovalRecord:
        """Submit report for review."""
        record = ApprovalRecord(
            report_id=report_id,
            status=ApprovalStatus.PENDING_REVIEW,
            submitter=submitter,
            submitted_at=datetime.now(),
        )
        self._records[report_id] = record
        self._trigger_callbacks(ApprovalStatus.PENDING_REVIEW, record)
        return record

    def approve(self, report_id: str, reviewer: str, comments: str = "") -> ApprovalRecord:
        """Approve report."""
        record = self._records.get(report_id)
        if not record:
            raise ValueError(f"Report {report_id} not in workflow")

        record.status = ApprovalStatus.APPROVED
        record.reviewer = reviewer
        record.reviewed_at = datetime.now()
        record.comments = comments
        self._trigger_callbacks(ApprovalStatus.APPROVED, record)
        return record

    def reject(self, report_id: str, reviewer: str, comments: str) -> ApprovalRecord:
        """Reject report."""
        record = self._records.get(report_id)
        if not record:
            raise ValueError(f"Report {report_id} not in workflow")

        record.status = ApprovalStatus.REJECTED
        record.reviewer = reviewer
        record.reviewed_at = datetime.now()
        record.comments = comments
        self._trigger_callbacks(ApprovalStatus.REJECTED, record)
        return record

    def on_status_change(
        self, status: ApprovalStatus, callback: Callable[[ApprovalRecord], None]
    ) -> None:
        """Register callback for status change."""
        if status not in self._callbacks:
            self._callbacks[status] = []
        self._callbacks[status].append(callback)

    def _trigger_callbacks(self, status: ApprovalStatus, record: ApprovalRecord) -> None:
        """Trigger callbacks for status."""
        for callback in self._callbacks.get(status, []):
            try:
                callback(record)
            except Exception as e:
                logger.warning(f"Approval callback failed: {e}")


# =============================================================================
# =============================================================================


@dataclass
class ComplianceRule:
    """Compliance checking rule.

    References:
        REPORT-025: Report Compliance Checking
    """

    id: str
    name: str
    description: str
    check: Callable[[dict[str, Any]], bool]
    severity: str = "error"  # error, warning, info


@dataclass
class ComplianceResult:
    """Compliance check result."""

    passed: bool
    violations: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)


class ComplianceChecker:
    """Report compliance checker.

    References:
        REPORT-025: Report Compliance Checking
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        self._rules: list[ComplianceRule] = []

    def add_rule(
        self,
        name: str,
        check: Callable[[dict[str, Any]], bool],
        description: str = "",
        severity: str = "error",
    ) -> None:
        """Add compliance rule."""
        rule = ComplianceRule(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            check=check,
            severity=severity,
        )
        self._rules.append(rule)

    def check(self, report_data: dict[str, Any]) -> ComplianceResult:
        """Check report against all rules."""
        violations = []
        warnings = []

        for rule in self._rules:
            try:
                if not rule.check(report_data):
                    if rule.severity == "error":
                        violations.append((rule.name, rule.description))
                    else:
                        warnings.append((rule.name, rule.description))
            except Exception as e:
                logger.warning(f"Compliance rule {rule.name} failed: {e}")

        return ComplianceResult(
            passed=len(violations) == 0, violations=violations, warnings=warnings
        )


# =============================================================================
# =============================================================================


@dataclass
class LocaleStrings:
    """Localized strings for a locale.

    References:
        REPORT-026: Report Localization
    """

    locale: str
    strings: dict[str, str] = field(default_factory=dict)
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_decimal: str = "."
    number_thousand: str = ","


class ReportLocalizer:
    """Report localization manager.

    References:
        REPORT-026: Report Localization
    """

    def __init__(self, default_locale: str = "en_US"):
        self.default_locale = default_locale
        self._locales: dict[str, LocaleStrings] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default locales."""
        self._locales["en_US"] = LocaleStrings(
            locale="en_US",
            strings={
                "title": "Report",
                "summary": "Summary",
                "pass": "PASS",
                "fail": "FAIL",
            },
        )
        self._locales["de_DE"] = LocaleStrings(
            locale="de_DE",
            strings={
                "title": "Bericht",
                "summary": "Zusammenfassung",
                "pass": "BESTANDEN",
                "fail": "DURCHGEFALLEN",
            },
            date_format="%d.%m.%Y",
            number_decimal=",",
            number_thousand=".",
        )

    def get_string(self, key: str, locale: str | None = None) -> str:
        """Get localized string."""
        loc = locale or self.default_locale
        strings = self._locales.get(loc, self._locales[self.default_locale])
        return strings.strings.get(key, key)

    def format_number(self, value: float, locale: str | None = None) -> str:
        """Format number for locale."""
        loc_strings = self._locales.get(
            locale or self.default_locale, self._locales[self.default_locale]
        )
        formatted = f"{value:,.2f}"
        # Replace separators
        formatted = formatted.replace(",", "TEMP")
        formatted = formatted.replace(".", loc_strings.number_decimal)
        formatted = formatted.replace("TEMP", loc_strings.number_thousand)
        return formatted


# =============================================================================
# =============================================================================


@dataclass
class AccessibilityOptions:
    """Accessibility options for reports.

    References:
        REPORT-027: Report Accessibility
    """

    alt_text_required: bool = True
    high_contrast: bool = False
    screen_reader_friendly: bool = True
    keyboard_navigable: bool = True
    wcag_level: str = "AA"  # A, AA, AAA


def add_accessibility_features(html_content: str, options: AccessibilityOptions) -> str:
    """Add accessibility features to HTML report.

    Args:
        html_content: HTML content
        options: Accessibility options

    Returns:
        Enhanced HTML content

    References:
        REPORT-027: Report Accessibility
    """
    # Add ARIA landmarks
    html_content = html_content.replace(
        '<div class="report">',
        '<div class="report" role="main" aria-label="Report Content">',
    )

    # Add skip navigation link
    skip_nav = '<a href="#main-content" class="skip-link">Skip to main content</a>'
    html_content = html_content.replace("<body>", f"<body>{skip_nav}")

    # Add high contrast styles if enabled
    if options.high_contrast:
        contrast_styles = """
        <style>
            body { background: white !important; color: black !important; }
            a { color: blue !important; }
            .pass { background: green !important; color: white !important; }
            .fail { background: red !important; color: white !important; }
        </style>
        """
        html_content = html_content.replace("</head>", f"{contrast_styles}</head>")

    return html_content


# =============================================================================
# =============================================================================


class ReportEncryption:
    """Report encryption utilities.

    References:
        REPORT-028: Report Encryption
    """

    @staticmethod
    def encrypt_content(content: bytes, password: str) -> bytes:
        """Encrypt report content.

        Args:
            content: Content bytes to encrypt.
            password: Encryption password.

        Returns:
            Encrypted content bytes.

        Note:
            Uses simple XOR encryption for demonstration.
            In production, use proper encryption (AES, etc.).
        """
        key = hashlib.sha256(password.encode()).digest()
        encrypted = bytearray()
        for i, byte in enumerate(content):
            encrypted.append(byte ^ key[i % len(key)])
        return bytes(encrypted)

    @staticmethod
    def decrypt_content(encrypted: bytes, password: str) -> bytes:
        """Decrypt report content."""
        # XOR is symmetric
        return ReportEncryption.encrypt_content(encrypted, password)

    @staticmethod
    def encrypt_file(input_path: Path, output_path: Path, password: str) -> None:
        """Encrypt report file."""
        with open(input_path, "rb") as f:
            content = f.read()
        encrypted = ReportEncryption.encrypt_content(content, password)
        with open(output_path, "wb") as f:
            f.write(encrypted)

    @staticmethod
    def decrypt_file(input_path: Path, output_path: Path, password: str) -> None:
        """Decrypt report file."""
        with open(input_path, "rb") as f:
            encrypted = f.read()
        decrypted = ReportEncryption.decrypt_content(encrypted, password)
        with open(output_path, "wb") as f:
            f.write(decrypted)


# =============================================================================
# =============================================================================


@dataclass
class Watermark:
    """Report watermark configuration.

    References:
        REPORT-029: Report Watermarking
    """

    text: str = "CONFIDENTIAL"
    opacity: float = 0.1
    rotation: int = -45
    position: str = "center"  # center, header, footer
    font_size: int = 48


def add_watermark(html_content: str, watermark: Watermark) -> str:
    """Add watermark to HTML report.

    Args:
        html_content: HTML content
        watermark: Watermark configuration

    Returns:
        HTML with watermark

    References:
        REPORT-029: Report Watermarking
    """
    watermark_css = f"""
    <style>
        .watermark {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) rotate({watermark.rotation}deg);
            font-size: {watermark.font_size}px;
            color: rgba(128, 128, 128, {watermark.opacity});
            pointer-events: none;
            z-index: 1000;
            white-space: nowrap;
        }}
    </style>
    """
    watermark_div = f'<div class="watermark">{watermark.text}</div>'

    html_content = html_content.replace("</head>", f"{watermark_css}</head>")
    html_content = html_content.replace("<body>", f"<body>{watermark_div}")

    return html_content


# =============================================================================
# =============================================================================


@dataclass
class AuditEntry:
    """Audit trail entry.

    References:
        REPORT-030: Report Audit Trail
    """

    id: str
    report_id: str
    action: str
    user: str
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""


class AuditTrail:
    """Report audit trail manager.

    References:
        REPORT-030: Report Audit Trail
    """

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path
        self._entries: list[AuditEntry] = []

    def log(
        self,
        report_id: str,
        action: str,
        user: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log audit entry."""
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            report_id=report_id,
            action=action,
            user=user,
            timestamp=datetime.now(),
            details=details or {},
        )
        self._entries.append(entry)

        # Persist if storage configured
        if self.storage_path:
            self._persist()

        return entry

    def get_for_report(self, report_id: str) -> list[AuditEntry]:
        """Get audit entries for report."""
        return [e for e in self._entries if e.report_id == report_id]

    def get_by_user(self, user: str) -> list[AuditEntry]:
        """Get audit entries by user."""
        return [e for e in self._entries if e.user == user]

    def export(self, format_: str = "json") -> str:
        """Export audit trail."""
        if format_ == "json":
            return json.dumps(
                [
                    {
                        "id": e.id,
                        "report_id": e.report_id,
                        "action": e.action,
                        "user": e.user,
                        "timestamp": e.timestamp.isoformat(),
                        "details": e.details,
                    }
                    for e in self._entries
                ],
                indent=2,
            )
        return ""

    def _persist(self) -> None:
        """Persist audit trail to storage."""
        if self.storage_path:
            with open(self.storage_path, "w") as f:
                f.write(self.export("json"))


__all__ = [
    # Accessibility (REPORT-027)
    "AccessibilityOptions",
    # Annotations (REPORT-013)
    "Annotation",
    "AnnotationManager",
    # Approval (REPORT-024)
    "ApprovalRecord",
    "ApprovalStatus",
    "ApprovalWorkflow",
    # Archiving (REPORT-021)
    "ArchivedReport",
    # Audit Trail (REPORT-030)
    "AuditEntry",
    "AuditTrail",
    # Compliance (REPORT-025)
    "ComplianceChecker",
    "ComplianceResult",
    "ComplianceRule",
    # Templates (REPORT-011)
    "CustomTemplate",
    # Distribution (REPORT-020)
    "DistributionChannel",
    "DistributionConfig",
    # Interactive (REPORT-012)
    "InteractiveElement",
    "InteractiveElementType",
    # Localization (REPORT-026)
    "LocaleStrings",
    "ReportArchive",
    "ReportDistributor",
    # Encryption (REPORT-028)
    "ReportEncryption",
    "ReportLocalizer",
    # Scheduling (REPORT-017)
    "ReportSchedule",
    "ReportScheduler",
    # Search (REPORT-022)
    "ReportSearchIndex",
    # Versioning (REPORT-023)
    "ReportVersion",
    "ReportVersionControl",
    "ScheduleFrequency",
    "SearchResult",
    "TemplateField",
    # Watermarking (REPORT-029)
    "Watermark",
    "add_accessibility_features",
    "add_watermark",
]
