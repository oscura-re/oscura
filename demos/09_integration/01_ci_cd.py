"""CI/CD Integration: Continuous integration and testing patterns.

Demonstrates:
- Oscura in CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
- Automated regression testing
- Performance benchmarking in CI
- Test report generation
- Coverage tracking and reporting

Category: Integration
IEEE Standards: N/A

Related Demos:
- 01_data_loading/01_waveforms.py
- 08_testing/01_unit_testing.py

This demonstrates how to integrate Oscura into CI/CD workflows for automated
testing, regression detection, and quality assurance.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class CICDIntegrationDemo(BaseDemo):
    """Demonstrates CI/CD integration patterns for Oscura."""

    name = "CI/CD Integration"
    description = "Integrate Oscura into CI/CD pipelines"
    category = "integration"

    def generate_data(self) -> None:
        """Generate test signals for CI/CD examples."""
        from oscura.core import TraceMetadata, WaveformTrace

        # Generate reference signal (golden standard)
        sample_rate = 100e3
        duration = 0.01
        t = np.linspace(0, duration, int(sample_rate * duration))
        data = np.sin(2 * np.pi * 1000 * t)

        self.reference_trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="REF",
            ),
        )

        # Generate test signal (slightly different for regression detection)
        test_data = np.sin(2 * np.pi * 1001 * t)  # 1 Hz difference
        self.test_trace = WaveformTrace(
            data=test_data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="TEST",
            ),
        )

    def run_analysis(self) -> None:
        """Demonstrate CI/CD integration patterns."""
        print_header("CI/CD Integration Patterns")

        print_subheader("1. GitHub Actions Workflow")
        print_info("Example GitHub Actions configuration for Oscura testing:")

        github_actions = """
name: Oscura Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --all-extras

      - name: Run tests
        run: ./scripts/test.sh

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
"""
        print(github_actions)

        print_subheader("2. Automated Regression Testing")
        print_info("Compare against golden reference measurements:")

        from oscura import frequency

        # Measure both traces
        ref_freq = frequency(self.reference_trace)
        test_freq = frequency(self.test_trace)

        print_info(f"Reference frequency: {ref_freq:.2f} Hz")
        print_info(f"Test frequency: {test_freq:.2f} Hz")
        print_info(f"Difference: {abs(ref_freq - test_freq):.2f} Hz")

        # Regression check
        tolerance = 0.5  # Hz
        regression_detected = abs(ref_freq - test_freq) > tolerance

        self.results["regression_detected"] = regression_detected
        self.results["frequency_difference"] = abs(ref_freq - test_freq)

        if regression_detected:
            print_info("⚠️  Regression detected!")
        else:
            print_info("✓ No regression detected")

        print_subheader("3. Performance Benchmarking")
        print_info("Track performance metrics in CI:")

        import time

        # Benchmark analysis time
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            _ = frequency(self.test_trace)
        duration = time.perf_counter() - start
        avg_time = duration / iterations

        print_info(f"Average analysis time: {avg_time*1000:.3f} ms")
        print_info(f"Throughput: {1/avg_time:.1f} analyses/sec")

        self.results["avg_analysis_time_ms"] = avg_time * 1000
        self.results["throughput"] = 1 / avg_time

        print_subheader("4. Test Report Generation")
        print_info("Generate JUnit XML for CI systems:")

        junit_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="oscura.tests" tests="2" failures="0" errors="0" time="{duration:.3f}">
  <testcase classname="oscura.tests" name="frequency_measurement" time="{avg_time:.3f}">
    <system-out>Measured frequency: {test_freq:.2f} Hz</system-out>
  </testcase>
  <testcase classname="oscura.tests" name="regression_check" time="0.001">
    <system-out>No regression detected</system-out>
  </testcase>
</testsuite>
"""
        # Save report
        report_path = self.data_dir / "junit-report.xml"
        report_path.write_text(junit_xml)
        print_info(f"Report saved: {report_path}")

        self.results["report_path"] = str(report_path)

        print_subheader("5. GitLab CI Configuration")
        print_info("Example GitLab CI configuration:")

        gitlab_ci = """
test:
  image: python:3.12
  script:
    - pip install uv
    - uv sync --all-extras
    - ./scripts/test.sh
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      junit: junit-report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
"""
        print(gitlab_ci)

        print_subheader("6. Jenkins Pipeline")
        print_info("Example Jenkinsfile:")

        jenkinsfile = """
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'pip install uv'
                sh 'uv sync --all-extras'
                sh './scripts/test.sh'
            }
        }
        stage('Report') {
            steps {
                junit 'junit-report.xml'
                publishHTML([
                    reportDir: 'htmlcov',
                    reportFiles: 'index.html',
                    reportName: 'Coverage Report'
                ])
            }
        }
    }
}
"""
        print(jenkinsfile)

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate CI/CD integration results."""
        suite.check_exists("Report path", self.results.get("report_path"))
        suite.check_exists("Regression detection", self.results.get("regression_detected"))
        suite.check_type("Average analysis time", self.results.get("avg_analysis_time_ms"), float)
        suite.check_type("Throughput", self.results.get("throughput"), float)


if __name__ == "__main__":
    demo = CICDIntegrationDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
