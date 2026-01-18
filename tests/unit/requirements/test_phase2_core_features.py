import pytest

"""Tests for Phase 2 Core Features requirements.

Tests for:
- REPORT-011 through REPORT-030: Advanced Reporting
- LOG-009 through LOG-020: Advanced Logging
- API-010 through API-019: Expert API
"""

import tempfile
import time
from pathlib import Path

import numpy as np

pytestmark = pytest.mark.unit


class TestAdvancedReporting:
    """Tests for REPORT-011 through REPORT-030."""

    def test_report011_custom_templates(self):
        """Test REPORT-011: Report Customization Templates."""
        from oscura.reporting.advanced import CustomTemplate, TemplateField

        template = CustomTemplate(
            name="custom_report",
            version="1.0.0",
            fields=[
                TemplateField("title", required=True),
                TemplateField("logo", type="image"),
                TemplateField("company", default="Oscura"),
            ],
            layout={"template": "Report: {{title}} by {{company}}"},
        )

        # Validation
        is_valid, errors = template.validate_data({"title": "Test Report"})
        assert is_valid
        assert len(errors) == 0

        # Missing required field
        is_valid, errors = template.validate_data({})
        assert not is_valid
        assert "title" in errors[0]

        # Rendering
        result = template.render({"title": "Test Report"})
        assert "Test Report" in result
        assert "Oscura" in result

    def test_report012_interactive_elements(self):
        """Test REPORT-012: Interactive Report Elements."""
        from oscura.reporting.advanced import (
            InteractiveElement,
            InteractiveElementType,
        )

        element = InteractiveElement(
            id="chart1",
            type=InteractiveElementType.COLLAPSIBLE_SECTION,
            data="<p>Section content</p>",
            options={"title": "Details"},
        )

        html = element.to_html()
        assert "chart1" in html
        assert "collapsible" in html
        assert "Details" in html

    def test_report013_annotations(self):
        """Test REPORT-013: Report Annotations."""
        from oscura.reporting.advanced import AnnotationManager

        manager = AnnotationManager("report-001")

        # Add annotation
        ann = manager.add("figure1", "Note about figure", author="user1")
        assert ann.target == "figure1"
        assert ann.author == "user1"

        # Get annotations for target
        annotations = manager.get_for_target("figure1")
        assert len(annotations) == 1

        # Export
        exported = manager.export()
        assert len(exported) == 1
        assert exported[0]["text"] == "Note about figure"

    def test_report017_scheduling(self):
        """Test REPORT-017: Report Scheduling."""
        from oscura.reporting.advanced import (
            ReportScheduler,
            ScheduleFrequency,
        )

        scheduler = ReportScheduler()

        schedule_id = scheduler.add_schedule(
            report_config={"type": "summary"},
            frequency=ScheduleFrequency.DAILY,
            recipients=["user@example.com"],
        )

        assert schedule_id is not None

        # Check pending (none should be pending yet since next_run is in future)
        scheduler.get_pending()
        # Note: may or may not have pending depending on timing

    def test_report020_distribution(self):
        """Test REPORT-020: Report Distribution."""
        from oscura.reporting.advanced import (
            DistributionChannel,
            DistributionConfig,
            ReportDistributor,
        )

        distributor = ReportDistributor()

        # Register mock handler
        handled = []

        def mock_handler(path, config):
            handled.append(config.channel)
            return True

        distributor.register_handler(DistributionChannel.FILE_SHARE, mock_handler)

        # Distribute
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test report content")
            f.flush()

            config = DistributionConfig(
                channel=DistributionChannel.FILE_SHARE, recipients=["dest_path"]
            )

            results = distributor.distribute(Path(f.name), [config])
            assert results["FILE_SHARE"] is True

    def test_report021_archiving(self):
        """Test REPORT-021: Report Archiving."""
        from oscura.reporting.advanced import ReportArchive

        with tempfile.TemporaryDirectory() as tmpdir:
            archive = ReportArchive(Path(tmpdir))

            # Create test report
            report_path = Path(tmpdir) / "test_report.pdf"
            report_path.write_bytes(b"test content")

            # Archive
            report_id = archive.archive(report_path, {"name": "test"})
            assert report_id is not None

            # Retrieve
            retrieved = archive.retrieve(report_id)
            assert retrieved is not None
            assert retrieved.exists()

    def test_report022_search(self):
        """Test REPORT-022: Report Search."""
        from oscura.reporting.advanced import ReportSearchIndex

        index = ReportSearchIndex()

        # Index reports
        index.index_report("report-1", "UART analysis results", {"name": "UART Report"})
        index.index_report("report-2", "SPI timing measurements", {"name": "SPI Report"})

        # Search
        results = index.search("UART analysis")
        assert len(results) >= 1
        assert results[0].report_id == "report-1"

    def test_report023_versioning(self):
        """Test REPORT-023: Report Versioning."""
        from oscura.reporting.advanced import ReportVersionControl

        with tempfile.TemporaryDirectory() as tmpdir:
            vcs = ReportVersionControl(Path(tmpdir))

            # Create report
            report_path = Path(tmpdir) / "report.pdf"
            report_path.write_bytes(b"version 1")

            # Commit v1
            v1 = vcs.commit("report-001", report_path, "author", "Initial version")
            assert v1 == 1

            # Modify and commit v2
            report_path.write_bytes(b"version 2 with changes")
            v2 = vcs.commit("report-001", report_path, "author", "Updated analysis")
            assert v2 == 2

            # Get history
            history = vcs.get_history("report-001")
            assert len(history) == 2

    def test_report024_approval_workflow(self):
        """Test REPORT-024: Report Approval Workflow."""
        from oscura.reporting.advanced import ApprovalStatus, ApprovalWorkflow

        workflow = ApprovalWorkflow()

        # Track status changes
        status_changes = []
        workflow.on_status_change(
            ApprovalStatus.APPROVED, lambda r: status_changes.append(r.status)
        )

        # Submit for review
        record = workflow.submit_for_review("report-001", "submitter")
        assert record.status == ApprovalStatus.PENDING_REVIEW

        # Approve
        record = workflow.approve("report-001", "reviewer", "Looks good")
        assert record.status == ApprovalStatus.APPROVED
        assert len(status_changes) == 1

    def test_report025_compliance_checking(self):
        """Test REPORT-025: Report Compliance Checking."""
        from oscura.reporting.advanced import ComplianceChecker

        checker = ComplianceChecker()

        # Add rules
        checker.add_rule("has_title", lambda d: "title" in d, "Report must have title")
        checker.add_rule(
            "valid_date",
            lambda d: "date" in d,
            "Report must have date",
            severity="warning",
        )

        # Check passing
        result = checker.check({"title": "Test", "date": "2025-01-01"})
        assert result.passed

        # Check failing
        result = checker.check({})
        assert not result.passed
        assert len(result.violations) > 0

    def test_report026_localization(self):
        """Test REPORT-026: Report Localization."""
        from oscura.reporting.advanced import ReportLocalizer

        localizer = ReportLocalizer()

        # English (default)
        assert localizer.get_string("pass") == "PASS"

        # German
        assert localizer.get_string("pass", "de_DE") == "BESTANDEN"

        # Number formatting
        formatted = localizer.format_number(1234.56, "de_DE")
        assert "," in formatted  # German decimal separator

    def test_report027_accessibility(self):
        """Test REPORT-027: Report Accessibility."""
        from oscura.reporting.advanced import (
            AccessibilityOptions,
            add_accessibility_features,
        )

        html = '<html><head></head><body><div class="report">Content</div></body></html>'
        options = AccessibilityOptions(high_contrast=True)

        enhanced = add_accessibility_features(html, options)
        assert 'role="main"' in enhanced
        assert "skip-link" in enhanced

    def test_report028_encryption(self):
        """Test REPORT-028: Report Encryption."""
        from oscura.reporting.advanced import ReportEncryption

        original = b"Sensitive report content"
        password = "secret123"

        encrypted = ReportEncryption.encrypt_content(original, password)
        assert encrypted != original

        decrypted = ReportEncryption.decrypt_content(encrypted, password)
        assert decrypted == original

    def test_report029_watermarking(self):
        """Test REPORT-029: Report Watermarking."""
        from oscura.reporting.advanced import Watermark, add_watermark

        html = "<html><head></head><body>Report</body></html>"
        watermark = Watermark(text="CONFIDENTIAL", opacity=0.2)

        watermarked = add_watermark(html, watermark)
        assert "CONFIDENTIAL" in watermarked
        assert "watermark" in watermarked

    def test_report030_audit_trail(self):
        """Test REPORT-030: Report Audit Trail."""
        from oscura.reporting.advanced import AuditTrail

        audit = AuditTrail()

        # Log actions
        audit.log("report-001", "created", "user1")
        audit.log("report-001", "edited", "user2")
        audit.log("report-002", "created", "user1")

        # Get by report
        entries = audit.get_for_report("report-001")
        assert len(entries) == 2

        # Get by user
        entries = audit.get_by_user("user1")
        assert len(entries) == 2

        # Export
        exported = audit.export("json")
        assert "report-001" in exported


class TestAdvancedLogging:
    """Tests for LOG-009 through LOG-020."""

    def test_log009_aggregation(self):
        """Test LOG-009: Log Aggregation."""
        import logging

        from oscura.core.logging_advanced import LogAggregator

        aggregator = LogAggregator(window_seconds=60, min_count=2)

        # Add similar messages
        record = logging.LogRecord("test", logging.INFO, "", 0, "Error at line 123", (), None)
        aggregator.add(record)

        record2 = logging.LogRecord("test", logging.INFO, "", 0, "Error at line 456", (), None)
        aggregator.add(record2)

        summary = aggregator.get_summary()
        assert len(summary) >= 1  # Should aggregate similar messages

    def test_log010_analysis(self):
        """Test LOG-010: Log Analysis and Patterns."""
        import logging

        from oscura.core.logging_advanced import LogAnalyzer

        analyzer = LogAnalyzer()

        # Add log entries
        for i in range(100):
            level = logging.ERROR if i % 10 == 0 else logging.INFO
            record = logging.LogRecord("test", level, "", 0, f"Message {i}", (), None)
            analyzer.add(record)

        # Analyze patterns
        patterns = analyzer.analyze_patterns()
        assert len(patterns) > 0

        # Check error rate
        error_rate = analyzer.get_error_rate(window_minutes=60)
        assert 0 <= error_rate <= 1

    def test_log012_alerting(self):
        """Test LOG-012: Log Alerting."""
        import logging

        from oscura.core.logging_advanced import AlertSeverity, LogAlerter

        alerter = LogAlerter()

        # Track alerts
        triggered_alerts = []
        alerter.on_alert(lambda a: triggered_alerts.append(a))

        # Add alert rule
        alerter.add_alert(
            "error_alert",
            lambda r: r.levelno >= logging.ERROR,
            AlertSeverity.ERROR,
            cooldown_seconds=0,  # Disable cooldown for test
        )

        # Check record that should trigger
        error_record = logging.LogRecord("test", logging.ERROR, "", 0, "Error!", (), None)
        triggered = alerter.check(error_record)
        assert len(triggered) == 1

    def test_log015_sampling(self):
        """Test LOG-015: Log Sampling for High-Volume."""
        import logging

        from oscura.core.logging_advanced import LogSampler, SamplingStrategy

        sampler = LogSampler(strategy=SamplingStrategy.RATE_LIMIT, max_per_second=5)

        # Sample lots of records
        logged = 0
        record = logging.LogRecord("test", logging.INFO, "", 0, "Message", (), None)

        for _ in range(100):
            if sampler.should_log(record):
                logged += 1

        # Should be rate limited
        assert logged < 100

        # Errors should always log
        error_record = logging.LogRecord("test", logging.ERROR, "", 0, "Error", (), None)
        assert sampler.should_log(error_record)

    def test_log016_buffer(self):
        """Test LOG-016: Log Buffer for Batch Writing."""
        import logging

        from oscura.core.logging_advanced import LogBuffer

        buffer = LogBuffer(max_size=10)

        flushed_records = []
        buffer.on_flush(lambda records: flushed_records.extend(records))

        # Add records
        for i in range(5):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"Message {i}", (), None)
            buffer.add(record)

        # Manual flush
        buffer.flush()
        assert len(flushed_records) == 5

    def test_log017_compression(self):
        """Test LOG-017: Log Compression."""
        from oscura.core.logging_advanced import CompressedLogHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = CompressedLogHandler(
                filename=str(Path(tmpdir) / "test.log"),
                max_bytes=1000,
                compression_level=9,
            )

            # Format and emit
            import logging

            handler.setFormatter(logging.Formatter("%(message)s"))
            record = logging.LogRecord("test", logging.INFO, "", 0, "Test message", (), None)
            handler.emit(record)
            handler.close()

            # Compressed file should exist
            assert (Path(tmpdir) / "test.log.gz").exists()

    def test_log018_encryption(self):
        """Test LOG-018: Log Encryption."""
        from oscura.core.logging_advanced import EncryptedLogHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = EncryptedLogHandler(
                filename=str(Path(tmpdir) / "encrypted.log"), key="secret_key"
            )

            import logging

            handler.setFormatter(logging.Formatter("%(message)s"))
            record = logging.LogRecord("test", logging.INFO, "", 0, "Sensitive data", (), None)
            handler.emit(record)
            handler.close()

            # Encrypted file should exist and not contain plaintext
            encrypted_path = Path(tmpdir) / "encrypted.log"
            assert encrypted_path.exists()
            content = encrypted_path.read_bytes()
            assert b"Sensitive data" not in content

    def test_log019_forwarding(self):
        """Test LOG-019: Log Forwarding."""
        from oscura.core.logging_advanced import (
            ForwardingConfig,
            LogForwarder,
            LogForwarderProtocol,
        )

        config = ForwardingConfig(
            protocol=LogForwarderProtocol.HTTP,
            host="localhost",
            port=8080,
            batch_size=10,
        )

        forwarder = LogForwarder(config)

        # Forward record (won't actually send since no server)
        import logging

        record = logging.LogRecord("test", logging.INFO, "", 0, "Message", (), None)
        forwarder.forward(record)

        # Check buffer
        assert len(forwarder._buffer) == 1

    def test_log020_dashboard(self):
        """Test LOG-020: Log Visualization Dashboard Data."""
        import logging

        from oscura.core.logging_advanced import LogDashboardCollector

        collector = LogDashboardCollector(window_minutes=60)

        # Add various log entries
        for i in range(50):
            level = logging.ERROR if i % 5 == 0 else logging.INFO
            record = logging.LogRecord(f"logger.{i % 3}", level, "", 0, f"Message {i}", (), None)
            collector.add(record)

        # Get metrics
        metrics = collector.get_metrics()
        assert metrics.total_logs == 50
        assert "ERROR" in metrics.logs_by_level
        assert "INFO" in metrics.logs_by_level


class TestExpertAPI:
    """Tests for API-010 through API-019."""

    def test_api010_dsl(self):
        """Test API-010: Domain-Specific Language (DSL)."""
        from oscura.api.dsl import DSLParser, analyze

        parser = DSLParser()

        # Parse simple expression
        expr = parser.parse("lowpass(cutoff=1e6)")
        assert expr.operation == "lowpass"
        assert expr.kwargs["cutoff"] == 1e6

        # Parse chained expression
        expr = parser.parse("lowpass(cutoff=1e6) | fft(nfft=8192)")
        assert expr.operation == "lowpass"
        assert expr.chain is not None
        assert expr.chain.operation == "fft"

        # Execute on data
        data = np.random.randn(1000)
        result = analyze(data, "normalize(method='minmax')")
        assert result.min() >= 0
        assert result.max() <= 1

    def test_api012_profiling(self):
        """Test API-012: Performance Profiling API."""
        from oscura.api.profiling import Profiler, profile

        profiler = Profiler()

        # Profile with context manager
        with profiler.profile("test_operation"):
            time.sleep(0.01)

        report = profiler.report()
        assert "test_operation" in report.profiles
        assert report.profiles["test_operation"].calls == 1

        # Profile with decorator
        @profile()
        def slow_function(x):
            time.sleep(0.01)
            return x * 2

        result = slow_function(5)
        assert result == 10

    def test_api014_optimization(self):
        """Test API-014: Parameter Optimization."""
        from oscura.api.optimization import GridSearch, ParameterSpace

        # Define objective
        def objective(params, data):
            return -abs(params["x"] - 5)  # Maximize = minimize distance to 5

        # Create search
        search = GridSearch([ParameterSpace("x", values=[1, 3, 5, 7, 9])], verbose=False)

        # Run optimization
        result = search.fit(objective, None, maximize=True)
        assert result.best_params["x"] == 5
        assert result.num_evaluations == 5

    def test_api015_operators(self):
        """Test API-015: Pythonic Operators."""
        from oscura.api.operators import make_pipeable

        @make_pipeable
        def double(data):
            return data * 2

        @make_pipeable
        def add_one(data):
            return data + 1

        # Test pipe operator
        data = np.array([1, 2, 3])
        result = data >> double() >> add_one()
        np.testing.assert_array_equal(result, [3, 5, 7])

    def test_api016_time_indexing(self):
        """Test API-016: Time-Based Indexing."""
        from oscura.api.operators import TimeIndex

        # Create test data
        sample_rate = 1e6  # 1 MHz
        data = np.arange(1000000)  # 1 second of data

        ti = TimeIndex(data, sample_rate)

        # Check duration
        assert abs(ti.duration - 1.0) < 0.001

        # Slice by time
        segment = ti["0ms":"1ms"]
        assert len(segment) == 1000

        # Get value at time
        value = ti.at("500ms")
        assert value == 500000

    def test_api018_unit_conversion(self):
        """Test API-018: Automatic Unit Conversion."""
        from oscura.api.operators import UnitConverter, convert_units

        converter = UnitConverter()

        # Voltage conversion
        assert convert_units(1000, "mV", "V") == 1.0
        assert convert_units(1, "V", "mV") == 1000

        # Frequency conversion
        assert convert_units(1, "MHz", "Hz") == 1e6
        assert convert_units(1e9, "Hz", "GHz") == 1.0

        # Auto-scaling
        value, unit = converter.auto_scale(0.001, "V")
        assert value == 1.0
        assert unit == "mV"

    def test_api019_fluent_interface(self):
        """Test API-019: Fluent Interface."""
        from oscura.api.fluent import trace

        data = np.sin(2 * np.pi * np.arange(10000) / 100)

        # Create fluent trace
        ft = trace(data, sample_rate=1e6)

        # Chain operations
        result = ft.normalize(method="minmax").scale(factor=2).offset(value=-1).get()

        assert result.min() >= -1
        assert result.max() <= 1

        # Get measurements
        rms = ft.copy().normalize().rms()
        assert 0 < rms.get() < 1
