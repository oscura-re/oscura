"""Validation and test generation for protocol specifications.

This module provides tools for generating test vectors, validating protocol
implementations, fuzzing protocol parsers, and replaying messages to verify
reverse-engineered protocols.

Example:
    >>> from oscura.validation import GrammarTestGenerator, TestGenerationConfig
    >>> from oscura.sessions import ProtocolSpec
    >>>
    >>> config = TestGenerationConfig(strategy="coverage", num_tests=100)
    >>> generator = GrammarTestGenerator(config)
    >>> tests = generator.generate_tests(protocol_spec)
    >>> print(f"Generated {len(tests.valid_messages)} valid messages")
    >>>
    >>> # Replay validation
    >>> from oscura.validation import ReplayConfig, ReplayValidator
    >>> replay_config = ReplayConfig(interface="serial", port="/dev/ttyUSB0")
    >>> validator = ReplayValidator(replay_config)
    >>> result = validator.validate_protocol(protocol_spec, test_messages)
    >>>
    >>> # Fuzzing
    >>> from oscura.validation import ProtocolFuzzer, FuzzingConfig
    >>> fuzzer_config = FuzzingConfig(strategy="coverage_guided", max_iterations=1000)
    >>> fuzzer = ProtocolFuzzer(fuzzer_config)
    >>> report = fuzzer.fuzz_protocol(protocol_spec, seed_corpus)
    >>> print(f"Found {report.total_crashes} crashes")

Modules:
    grammar_tests: Grammar-based test vector generation and fuzzing
    replay: Protocol replay validation framework
    fuzzer: Advanced protocol fuzzing with coverage-guided mutation
"""

from oscura.validation.compliance_tests import (
    ComplianceConfig,
    ComplianceTestGenerator,
    ComplianceTestSuite,
    StandardType,
    TestCase,
    TestType,
)
from oscura.validation.fuzzer import (
    FuzzingConfig,
    FuzzingReport,
    FuzzingResult,
    FuzzingStrategy,
    MutationOperator,
    ProtocolFuzzer,
    TestResult,
)
from oscura.validation.grammar_tests import (
    GeneratedTests,
    GrammarTestGenerator,
    TestGenerationConfig,
)
from oscura.validation.grammar_validator import (
    ErrorSeverity,
    ErrorType,
    ProtocolGrammarValidator,
    ValidationError,
    ValidationReport,
)
from oscura.validation.hil_testing import (
    HILConfig,
    HILTester,
    HILTestReport,
    HILTestResult,
    InterfaceType,
    TestStatus,
)
from oscura.validation.regression_suite import (
    ComparisonMode,
    RegressionReport,
    RegressionTestResult,
    RegressionTestSuite,
)
from oscura.validation.replay import (
    ProtocolSpec,
    ReplayConfig,
    ReplayValidator,
    ValidationResult,
)

__all__ = [
    "ComparisonMode",
    "ComplianceConfig",
    "ComplianceTestGenerator",
    "ComplianceTestSuite",
    "ErrorSeverity",
    "ErrorType",
    "FuzzingConfig",
    "FuzzingReport",
    "FuzzingResult",
    "FuzzingStrategy",
    "GeneratedTests",
    "GrammarTestGenerator",
    "HILConfig",
    "HILTestReport",
    "HILTestResult",
    "HILTester",
    "InterfaceType",
    "MutationOperator",
    "ProtocolFuzzer",
    "ProtocolGrammarValidator",
    "ProtocolSpec",
    "RegressionReport",
    "RegressionTestResult",
    "RegressionTestSuite",
    "ReplayConfig",
    "ReplayValidator",
    "StandardType",
    "TestCase",
    "TestGenerationConfig",
    "TestResult",
    "TestStatus",
    "TestType",
    "ValidationError",
    "ValidationReport",
    "ValidationResult",
]
