"""Side-Channel Attack Detection and Vulnerability Assessment.

This module implements comprehensive side-channel vulnerability detection for
cryptographic implementations including timing attacks, power analysis, EM
emissions, and cache timing vulnerabilities.

Key capabilities:
- Timing-based leakage detection (variable-time operations)
- Power analysis vulnerability detection (data-dependent consumption)
- EM emission analysis for information leakage
- Cache timing attack detection
- Constant-time operation validation
- T-test for leakage detection (Welch's t-test)
- Mutual information calculation
- Statistical correlation analysis

Typical use cases:
- Evaluate cryptographic implementation security
- Detect non-constant-time operations
- Identify data-dependent branching
- Assess power consumption leakage
- Generate vulnerability reports for security audits

Example:
    >>> from oscura.hardware.security.side_channel_detector import SideChannelDetector
    >>> from oscura.side_channel.dpa import PowerTrace
    >>> import numpy as np
    >>> # Create detector
    >>> detector = SideChannelDetector(
    ...     timing_threshold=0.01,
    ...     power_threshold=0.7,
    ...     ttest_threshold=4.5
    ... )
    >>> # Analyze power traces
    >>> traces = [
    ...     PowerTrace(
    ...         timestamp=np.arange(1000),
    ...         power=np.random.randn(1000),
    ...         plaintext=bytes([i % 256 for i in range(16)])
    ...     )
    ...     for _ in range(100)
    ... ]
    >>> report = detector.analyze_power_traces(traces, fixed_key=bytes(16))
    >>> print(f"Found {len(report.vulnerabilities)} vulnerabilities")
    >>> # Check for timing vulnerabilities
    >>> timing_data = [(bytes([i]), 0.001 + i*1e-6) for i in range(256)]
    >>> result = detector.detect_timing_leakage(timing_data)
    >>> if result.severity != "low":
    ...     print(f"Timing vulnerability: {result.evidence}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from oscura.side_channel.dpa import PowerTrace

logger = logging.getLogger(__name__)


class VulnerabilityType(str, Enum):
    """Type of side-channel vulnerability detected."""

    TIMING = "timing"  # Timing-based leakage
    POWER = "power"  # Power consumption leakage
    EM = "electromagnetic"  # EM emission leakage
    CACHE = "cache"  # Cache timing leakage
    CONSTANT_TIME = "constant_time"  # Non-constant-time operations


class Severity(str, Enum):
    """Vulnerability severity level."""

    LOW = "low"  # Minor leakage, difficult to exploit
    MEDIUM = "medium"  # Moderate leakage, exploitable with effort
    HIGH = "high"  # Significant leakage, easily exploitable
    CRITICAL = "critical"  # Severe leakage, trivial to exploit


@dataclass
class SideChannelVulnerability:
    """Side-channel vulnerability finding.

    Attributes:
        vulnerability_type: Type of vulnerability detected.
        severity: Severity level (low/medium/high/critical).
        confidence: Confidence score (0.0-1.0) in detection.
        evidence: Evidence supporting the vulnerability (e.g., correlation value).
        description: Human-readable description of the vulnerability.
        mitigation_suggestions: List of mitigation recommendations.
        affected_operation: Operation or code location affected (optional).
        metadata: Additional context and metrics.

    Example:
        >>> vuln = SideChannelVulnerability(
        ...     vulnerability_type=VulnerabilityType.TIMING,
        ...     severity=Severity.HIGH,
        ...     confidence=0.95,
        ...     evidence="Timing variance: 125.3 ns",
        ...     description="Input-dependent execution time detected",
        ...     mitigation_suggestions=["Use constant-time comparison"]
        ... )
    """

    vulnerability_type: VulnerabilityType
    severity: Severity
    confidence: float
    evidence: str
    description: str
    mitigation_suggestions: list[str] = field(default_factory=list)
    affected_operation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VulnerabilityReport:
    """Comprehensive side-channel vulnerability assessment report.

    Attributes:
        vulnerabilities: List of detected vulnerabilities.
        summary_statistics: Summary metrics across all detections.
        analysis_config: Configuration used for analysis.
        recommendations: Overall security recommendations.
        timestamp: When the analysis was performed.

    Example:
        >>> report = detector.analyze_power_traces(traces)
        >>> print(f"Critical: {report.summary_statistics['critical_count']}")
        >>> for vuln in report.vulnerabilities:
        ...     if vuln.severity == Severity.CRITICAL:
        ...         print(f"  {vuln.description}")
    """

    vulnerabilities: list[SideChannelVulnerability]
    summary_statistics: dict[str, Any] = field(default_factory=dict)
    analysis_config: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = ""


class SideChannelDetector:
    """Side-channel vulnerability detection and assessment framework.

    This class implements multiple statistical tests and analysis methods for
    detecting side-channel vulnerabilities in cryptographic implementations.

    Detection Methods:
        - Welch's t-test for leakage detection (TVLA methodology)
        - Pearson correlation analysis for data dependencies
        - Mutual information calculation
        - Statistical timing analysis (variance, distribution)
        - Frequency-domain analysis for EM leakage

    Analysis Types:
        - Timing analysis: Variable-time operations, input correlations
        - Power analysis: Data-dependent consumption, DPA susceptibility
        - EM analysis: Emission patterns, frequency leakage
        - Cache timing: Data-dependent memory access patterns
        - Constant-time: Operation time validation

    Example:
        >>> # Basic timing analysis
        >>> detector = SideChannelDetector(timing_threshold=0.01)
        >>> timing_data = [(input_bytes, execution_time), ...]
        >>> vuln = detector.detect_timing_leakage(timing_data)
        >>> # Power trace analysis with t-test
        >>> report = detector.analyze_power_traces(
        ...     traces,
        ...     fixed_key=key,
        ...     use_ttest=True
        ... )
    """

    def __init__(
        self,
        timing_threshold: float = 0.01,
        power_threshold: float = 0.7,
        em_threshold: float = 0.6,
        cache_threshold: float = 0.05,
        ttest_threshold: float = 4.5,
        mutual_info_threshold: float = 0.1,
    ) -> None:
        """Initialize side-channel detector.

        Args:
            timing_threshold: Timing correlation threshold for vulnerability (0.0-1.0).
            power_threshold: Power correlation threshold for vulnerability (0.0-1.0).
            em_threshold: EM emission correlation threshold (0.0-1.0).
            cache_threshold: Cache timing threshold for vulnerability (0.0-1.0).
            ttest_threshold: T-test statistic threshold (typically 4.5 for p<0.00001).
            mutual_info_threshold: Mutual information threshold in bits (0.0-8.0).

        Example:
            >>> # Strict detection thresholds
            >>> detector = SideChannelDetector(
            ...     timing_threshold=0.005,
            ...     ttest_threshold=3.0
            ... )
        """
        self.timing_threshold = timing_threshold
        self.power_threshold = power_threshold
        self.em_threshold = em_threshold
        self.cache_threshold = cache_threshold
        self.ttest_threshold = ttest_threshold
        self.mutual_info_threshold = mutual_info_threshold

    def detect_timing_leakage(
        self,
        timing_data: Sequence[tuple[bytes, float]],
        operation_name: str = "operation",
    ) -> SideChannelVulnerability:
        """Detect timing-based side-channel leakage.

        Analyzes timing measurements for correlation with input data to detect
        non-constant-time operations.

        Algorithm:
            1. Calculate timing statistics (mean, variance, range)
            2. Compute correlation between input values and timing
            3. Perform Welch's t-test between input groups
            4. Assess severity based on correlation and variance

        Args:
            timing_data: List of (input_bytes, execution_time) tuples.
            operation_name: Name of operation being analyzed.

        Returns:
            SideChannelVulnerability with timing analysis results.

        Example:
            >>> # Measure encryption timing for different plaintexts
            >>> timing_data = [
            ...     (bytes([i]), measure_encryption_time(bytes([i])))
            ...     for i in range(256)
            ... ]
            >>> vuln = detector.detect_timing_leakage(timing_data, "AES_encrypt")
            >>> if vuln.severity in [Severity.HIGH, Severity.CRITICAL]:
            ...     print(f"Timing vulnerability: {vuln.evidence}")
        """
        if not timing_data:
            return self._create_empty_timing_result()

        timings, first_bytes = self._extract_timing_data(timing_data)
        stats_data = self._calculate_timing_statistics(timings)
        correlation = self._calculate_timing_correlation(first_bytes, timings)
        t_stat, p_value = self._perform_timing_ttest(timings, first_bytes)
        severity = self._assess_timing_severity(correlation, t_stat)
        confidence = self._calculate_timing_confidence(timings, stats_data)
        mitigations = self._generate_timing_mitigations(correlation, t_stat)

        return SideChannelVulnerability(
            vulnerability_type=VulnerabilityType.TIMING,
            severity=severity,
            confidence=float(confidence),
            evidence=self._format_timing_evidence(correlation, t_stat, stats_data),
            description=self._format_timing_description(operation_name, severity),
            mitigation_suggestions=mitigations,
            affected_operation=operation_name,
            metadata={
                **stats_data,
                "correlation": float(correlation),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "sample_count": len(timings),
            },
        )

    def _create_empty_timing_result(self) -> SideChannelVulnerability:
        """Create vulnerability result for empty timing data."""
        return SideChannelVulnerability(
            vulnerability_type=VulnerabilityType.TIMING,
            severity=Severity.LOW,
            confidence=0.0,
            evidence="No timing data provided",
            description="Insufficient data for timing analysis",
        )

    def _extract_timing_data(
        self, timing_data: Sequence[tuple[bytes, float]]
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Extract timing arrays from timing data."""
        timings = np.array([t for _, t in timing_data])
        first_bytes = np.array([data[0] if data else 0 for data, _ in timing_data])
        return timings, first_bytes

    def _calculate_timing_statistics(self, timings: NDArray[np.float64]) -> dict[str, float]:
        """Calculate timing statistics (mean, std, range)."""
        mean_time = float(np.mean(timings))
        std_time = float(np.std(timings))
        min_time = float(np.min(timings))
        max_time = float(np.max(timings))
        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "time_range": max_time - min_time,
        }

    def _calculate_timing_correlation(
        self, first_bytes: NDArray[np.int_], timings: NDArray[np.float64]
    ) -> float:
        """Calculate correlation between input bytes and timing."""
        if len(set(first_bytes)) <= 1:
            return 0.0
        correlation = abs(np.corrcoef(first_bytes, timings)[0, 1])
        return 0.0 if np.isnan(correlation) else correlation

    def _perform_timing_ttest(
        self, timings: NDArray[np.float64], first_bytes: NDArray[np.int_]
    ) -> tuple[float, float]:
        """Perform t-test between low and high input groups."""
        if len(timings) < 10:
            return 0.0, 1.0

        median_byte = np.median(first_bytes)
        low_group = timings[first_bytes <= median_byte]
        high_group = timings[first_bytes > median_byte]

        if len(low_group) == 0 or len(high_group) == 0:
            return 0.0, 1.0

        t_stat, p_value = stats.ttest_ind(low_group, high_group, equal_var=False)
        return abs(float(t_stat)), float(p_value)

    def _assess_timing_severity(self, correlation: float, t_stat: float) -> Severity:
        """Assess severity based on correlation and t-statistic."""
        if correlation <= self.timing_threshold and t_stat <= self.ttest_threshold:
            return Severity.LOW
        if correlation > 0.5 or t_stat > 10.0:
            return Severity.CRITICAL
        if correlation > 0.2 or t_stat > 7.0:
            return Severity.HIGH
        if correlation > 0.1 or t_stat > self.ttest_threshold:
            return Severity.MEDIUM
        return Severity.LOW

    def _calculate_timing_confidence(
        self, timings: NDArray[np.float64], stats_data: dict[str, float]
    ) -> float:
        """Calculate confidence based on sample size and variance."""
        confidence = min(1.0, len(timings) / 100.0)
        if stats_data["std_time"] / (stats_data["mean_time"] + 1e-10) < 0.01:
            confidence *= 0.5
        return confidence

    def _generate_timing_mitigations(self, correlation: float, t_stat: float) -> list[str]:
        """Generate mitigation suggestions based on findings."""
        mitigations = []
        if correlation > self.timing_threshold:
            mitigations.extend(
                [
                    "Implement constant-time operations",
                    "Use constant-time comparison functions",
                    "Avoid data-dependent branching",
                ]
            )
        if t_stat > self.ttest_threshold:
            mitigations.extend(
                [
                    "Add random delays to mask timing variations",
                    "Use blinding or masking techniques",
                ]
            )
        return mitigations

    def _format_timing_evidence(
        self, correlation: float, t_stat: float, stats_data: dict[str, float]
    ) -> str:
        """Format evidence string for timing vulnerability."""
        return (
            f"Correlation: {correlation:.4f}, T-statistic: {t_stat:.2f}, "
            f"Range: {stats_data['time_range'] * 1e9:.1f} ns, "
            f"Std: {stats_data['std_time'] * 1e9:.1f} ns"
        )

    def _format_timing_description(self, operation_name: str, severity: Severity) -> str:
        """Format description string for timing vulnerability."""
        significance = (
            "significant" if severity in [Severity.HIGH, Severity.CRITICAL] else "potential"
        )
        return f"Timing analysis of '{operation_name}' shows {significance} input-dependent execution time"

    def analyze_power_traces(
        self,
        traces: Sequence[PowerTrace],
        fixed_key: bytes | None = None,
        use_ttest: bool = True,
    ) -> VulnerabilityReport:
        """Analyze power traces for DPA/CPA vulnerabilities.

        Performs comprehensive power analysis to detect data-dependent power
        consumption patterns that could enable DPA or CPA attacks.

        Analysis Methods:
            1. Welch's t-test (TVLA) for first-order leakage
            2. Correlation analysis between power and hypothetical values
            3. Variance analysis across different inputs
            4. Frequency-domain analysis for EM leakage

        Args:
            traces: List of power consumption traces with plaintexts.
            fixed_key: Known key for hypothesis testing (optional).
            use_ttest: Whether to perform t-test analysis.

        Returns:
            VulnerabilityReport with all detected power-related vulnerabilities.

        Example:
            >>> # Analyze AES implementation for DPA vulnerabilities
            >>> traces = collect_power_traces(plaintexts, key)
            >>> report = detector.analyze_power_traces(traces, fixed_key=key)
            >>> critical = [v for v in report.vulnerabilities
            ...             if v.severity == Severity.CRITICAL]
            >>> print(f"Critical vulnerabilities: {len(critical)}")
        """
        if not traces:
            return VulnerabilityReport(
                vulnerabilities=[],
                summary_statistics={"error": "No traces provided"},
            )

        power_matrix = np.array([t.power for t in traces])
        num_traces, num_samples = power_matrix.shape

        vulnerabilities = self._collect_power_vulnerabilities(
            traces, power_matrix, use_ttest, fixed_key
        )
        summary_stats = self._build_summary_statistics(vulnerabilities, num_traces, num_samples)
        recommendations = self._generate_recommendations(vulnerabilities)

        return VulnerabilityReport(
            vulnerabilities=vulnerabilities,
            summary_statistics=summary_stats,
            analysis_config={
                "power_threshold": self.power_threshold,
                "ttest_threshold": self.ttest_threshold,
                "use_ttest": use_ttest,
            },
            recommendations=recommendations,
        )

    def _collect_power_vulnerabilities(
        self,
        traces: Sequence[PowerTrace],
        power_matrix: np.ndarray[Any, Any],
        use_ttest: bool,
        fixed_key: bytes | None,
    ) -> list[SideChannelVulnerability]:
        """Collect all power-related vulnerabilities from traces.

        Args:
            traces: Power consumption traces.
            power_matrix: 2D array of power measurements.
            use_ttest: Whether to perform t-test analysis.
            fixed_key: Known key for CPA analysis.

        Returns:
            List of detected vulnerabilities.
        """
        vulnerabilities: list[SideChannelVulnerability] = []

        # T-test analysis
        if use_ttest and len(traces) >= 10:
            ttest_vuln = self._analyze_ttest_leakage(traces, power_matrix)
            if ttest_vuln is not None:
                vulnerabilities.append(ttest_vuln)

        # Correlation analysis
        if fixed_key is not None and len(traces) >= 10:
            cpa_vuln = self._analyze_cpa_vulnerability(traces, fixed_key)
            if cpa_vuln is not None:
                vulnerabilities.append(cpa_vuln)

        # EM leakage analysis
        em_vuln = self._analyze_em_leakage(power_matrix)
        if em_vuln.severity != Severity.LOW:
            vulnerabilities.append(em_vuln)

        # Variance analysis
        variance_vuln = self._analyze_power_variance(traces)
        if variance_vuln.severity != Severity.LOW:
            vulnerabilities.append(variance_vuln)

        return vulnerabilities

    def _analyze_ttest_leakage(
        self, traces: Sequence[PowerTrace], power_matrix: np.ndarray[Any, Any]
    ) -> SideChannelVulnerability | None:
        """Analyze power traces using Welch's t-test for leakage.

        Args:
            traces: Power consumption traces.
            power_matrix: 2D array of power measurements.

        Returns:
            Vulnerability if leakage detected, None otherwise.
        """
        t_stats = self._perform_ttest_leakage(traces)
        if t_stats is None:
            return None

        num_samples = power_matrix.shape[1]
        max_t_stat = float(np.max(np.abs(t_stats)))
        leakage_points = int(np.sum(np.abs(t_stats) > self.ttest_threshold))

        if max_t_stat <= self.ttest_threshold:
            return None

        severity = self._assess_ttest_severity(max_t_stat)
        confidence = min(1.0, len(traces) / 100.0)

        return SideChannelVulnerability(
            vulnerability_type=VulnerabilityType.POWER,
            severity=severity,
            confidence=confidence,
            evidence=(
                f"Max T-statistic: {max_t_stat:.2f}, Leakage points: {leakage_points}/{num_samples}"
            ),
            description="Welch's t-test reveals significant first-order power leakage",
            mitigation_suggestions=[
                "Implement power-balanced logic gates",
                "Add random noise to power consumption",
                "Use masking or hiding countermeasures",
                "Employ dual-rail precharge logic (DPL)",
            ],
            metadata={
                "max_t_statistic": max_t_stat,
                "leakage_points": leakage_points,
                "threshold": self.ttest_threshold,
            },
        )

    def _assess_ttest_severity(self, max_t_stat: float) -> Severity:
        """Assess severity based on t-test statistic magnitude.

        Args:
            max_t_stat: Maximum t-statistic value.

        Returns:
            Severity level.
        """
        if max_t_stat > 20.0:
            return Severity.CRITICAL
        if max_t_stat > 10.0:
            return Severity.HIGH
        if max_t_stat > self.ttest_threshold:
            return Severity.MEDIUM
        return Severity.LOW

    def _analyze_cpa_vulnerability(
        self, traces: Sequence[PowerTrace], fixed_key: bytes
    ) -> SideChannelVulnerability | None:
        """Analyze CPA vulnerability using correlation analysis.

        Args:
            traces: Power consumption traces.
            fixed_key: Known encryption key.

        Returns:
            Vulnerability if CPA attack successful, None otherwise.
        """
        from oscura.side_channel.dpa import DPAAnalyzer

        analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")

        try:
            result = analyzer.cpa_attack(list(traces), target_byte=0)
            if result.correlation_traces is None:
                return None

            max_correlation = float(np.max(result.correlation_traces))
            if max_correlation <= self.power_threshold:
                return None

            severity = self._assess_correlation_severity(max_correlation)

            return SideChannelVulnerability(
                vulnerability_type=VulnerabilityType.POWER,
                severity=severity,
                confidence=result.confidence,
                evidence=(
                    f"Max correlation: {max_correlation:.4f}, "
                    f"Attack confidence: {result.confidence:.2%}"
                ),
                description=(
                    "CPA attack successful - power consumption correlates with Hamming weight"
                ),
                mitigation_suggestions=[
                    "Implement algorithmic masking (boolean/arithmetic)",
                    "Use shuffling to randomize operation order",
                    "Add random delays between operations",
                    "Employ constant-power hardware primitives",
                ],
                metadata={
                    "max_correlation": max_correlation,
                    "recovered_key_byte": int(result.recovered_key[0]),
                    "attack_successful": result.successful,
                },
            )
        except Exception as e:
            logger.warning(f"CPA analysis failed: {e}")
            return None

    def _assess_correlation_severity(self, max_correlation: float) -> Severity:
        """Assess severity based on correlation magnitude.

        Args:
            max_correlation: Maximum correlation value.

        Returns:
            Severity level.
        """
        if max_correlation > 0.95:
            return Severity.CRITICAL
        if max_correlation > 0.85:
            return Severity.HIGH
        if max_correlation > self.power_threshold:
            return Severity.MEDIUM
        return Severity.LOW

    def _build_summary_statistics(
        self,
        vulnerabilities: list[SideChannelVulnerability],
        num_traces: int,
        num_samples: int,
    ) -> dict[str, Any]:
        """Build summary statistics from detected vulnerabilities.

        Args:
            vulnerabilities: List of detected vulnerabilities.
            num_traces: Number of power traces analyzed.
            num_samples: Number of samples per trace.

        Returns:
            Dictionary of summary statistics.
        """
        return {
            "total_vulnerabilities": len(vulnerabilities),
            "critical_count": sum(1 for v in vulnerabilities if v.severity == Severity.CRITICAL),
            "high_count": sum(1 for v in vulnerabilities if v.severity == Severity.HIGH),
            "medium_count": sum(1 for v in vulnerabilities if v.severity == Severity.MEDIUM),
            "low_count": sum(1 for v in vulnerabilities if v.severity == Severity.LOW),
            "num_traces": num_traces,
            "num_samples": num_samples,
        }

    def _generate_recommendations(
        self, vulnerabilities: list[SideChannelVulnerability]
    ) -> list[str]:
        """Generate security recommendations based on vulnerabilities.

        Args:
            vulnerabilities: List of detected vulnerabilities.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if any(v.severity == Severity.CRITICAL for v in vulnerabilities):
            recommendations.append(
                "CRITICAL: Immediate countermeasures required - implementation is highly "
                "vulnerable to power analysis attacks"
            )
        if any(v.vulnerability_type == VulnerabilityType.POWER for v in vulnerabilities):
            recommendations.append(
                "Consider hardware countermeasures (noise generation, power filtering)"
            )
        if any(v.vulnerability_type == VulnerabilityType.EM for v in vulnerabilities):
            recommendations.append("Add EM shielding and filtering to reduce emissions")

        if not vulnerabilities:
            recommendations.append(
                "No significant vulnerabilities detected with current thresholds"
            )

        return recommendations

    def detect_constant_time_violation(
        self,
        timing_measurements: Sequence[tuple[Any, float]],
        input_extractor: Any = None,
    ) -> SideChannelVulnerability:
        """Detect non-constant-time operations.

        Identifies operations with execution time dependent on secret data,
        which can enable timing side-channel attacks.

        Args:
            timing_measurements: List of (input_data, execution_time) tuples.
            input_extractor: Function to extract relevant bits from input_data.

        Returns:
            SideChannelVulnerability for constant-time analysis.

        Example:
            >>> # Check if AES S-box lookup is constant-time
            >>> measurements = [
            ...     (input_byte, time_sbox_lookup(input_byte))
            ...     for input_byte in range(256)
            ... ]
            >>> vuln = detector.detect_constant_time_violation(measurements)
        """
        if not timing_measurements:
            return SideChannelVulnerability(
                vulnerability_type=VulnerabilityType.CONSTANT_TIME,
                severity=Severity.LOW,
                confidence=0.0,
                evidence="No timing measurements provided",
                description="Insufficient data for constant-time analysis",
            )

        timings = np.array([t for _, t in timing_measurements])

        # Calculate timing statistics
        mean_time = float(np.mean(timings))
        std_time = float(np.std(timings))
        cv = std_time / (mean_time + 1e-10)  # Coefficient of variation

        # Check for constant time (low variance)
        if cv < 0.001:  # Very low variation (< 0.1%)
            severity = Severity.LOW
            description = "Operation appears to be constant-time"
            mitigations: list[str] = []
        elif cv < 0.01:  # Low variation (< 1%)
            severity = Severity.LOW
            description = "Operation shows minimal timing variation"
            mitigations = ["Verify constant-time properties with formal methods"]
        elif cv < 0.05:  # Moderate variation (< 5%)
            severity = Severity.MEDIUM
            description = "Operation shows moderate timing variation"
            mitigations = [
                "Review for data-dependent branches",
                "Check for table lookups without cache protection",
            ]
        else:  # High variation (>= 5%)
            severity = Severity.HIGH
            description = "Operation shows significant timing variation (non-constant-time)"
            mitigations = [
                "Reimplement using constant-time algorithms",
                "Eliminate data-dependent branches",
                "Use constant-time table lookups",
            ]

        confidence = min(1.0, len(timings) / 50.0)

        return SideChannelVulnerability(
            vulnerability_type=VulnerabilityType.CONSTANT_TIME,
            severity=severity,
            confidence=confidence,
            evidence=f"Coefficient of variation: {cv:.6f}, Std: {std_time * 1e9:.2f} ns",
            description=description,
            mitigation_suggestions=mitigations,
            metadata={
                "mean_time": mean_time,
                "std_time": std_time,
                "coefficient_of_variation": float(cv),
                "sample_count": len(timings),
            },
        )

    def calculate_mutual_information(
        self,
        secret_data: NDArray[np.int_],
        observable_data: NDArray[np.float64],
        bins: int = 50,
    ) -> float:
        """Calculate mutual information between secret and observable data.

        Mutual information quantifies how much knowing the observable reduces
        uncertainty about the secret. Higher values indicate more leakage.

        Formula:
            I(Secret; Observable) = H(Secret) + H(Observable) - H(Secret, Observable)

        Args:
            secret_data: Secret data values (e.g., key bytes).
            observable_data: Observable measurements (e.g., timing, power).
            bins: Number of bins for histogram estimation.

        Returns:
            Mutual information in bits (0.0 to log2(len(secret_data))).

        Example:
            >>> # Calculate MI between key byte and execution time
            >>> key_bytes = np.array([key[0] for _ in range(1000)])
            >>> timings = np.array([measure_time(key[0]) for _ in range(1000)])
            >>> mi = detector.calculate_mutual_information(key_bytes, timings)
            >>> print(f"Mutual information: {mi:.4f} bits")
        """
        if len(secret_data) != len(observable_data):
            raise ValueError("Secret and observable data must have same length")

        # Discretize observable data into bins
        obs_binned = np.digitize(
            observable_data, np.linspace(observable_data.min(), observable_data.max(), bins)
        )

        # Calculate joint histogram
        hist_joint, _, _ = np.histogram2d(
            secret_data, obs_binned, bins=[len(np.unique(secret_data)), bins]
        )

        # Normalize to probabilities
        p_joint = hist_joint / hist_joint.sum()
        p_secret = p_joint.sum(axis=1)
        p_obs = p_joint.sum(axis=0)

        # Calculate entropies
        def entropy(p: NDArray[np.float64]) -> float:
            """Calculate Shannon entropy."""
            p_nonzero = p[p > 0]
            return float(-np.sum(p_nonzero * np.log2(p_nonzero)))

        h_secret = entropy(p_secret)
        h_obs = entropy(p_obs)
        h_joint = entropy(p_joint.flatten())

        # Mutual information
        mi = h_secret + h_obs - h_joint

        return float(max(0.0, mi))  # Ensure non-negative

    def export_report(
        self,
        report: VulnerabilityReport,
        output_path: Path,
        format: Literal["json", "html"] = "json",
    ) -> None:
        """Export vulnerability report to file.

        Args:
            report: VulnerabilityReport to export.
            output_path: Path to save report file.
            format: Export format ("json" or "html").

        Example:
            >>> report = detector.analyze_power_traces(traces)
            >>> detector.export_report(
            ...     report,
            ...     Path("security_audit.json"),
            ...     format="json"
            ... )
        """
        if format == "json":
            self._export_json(report, output_path)
        elif format == "html":
            self._export_html(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, report: VulnerabilityReport, output_path: Path) -> None:
        """Export report as JSON."""
        data = {
            "summary": report.summary_statistics,
            "vulnerabilities": [
                {
                    "type": v.vulnerability_type.value,
                    "severity": v.severity.value,
                    "confidence": v.confidence,
                    "evidence": v.evidence,
                    "description": v.description,
                    "mitigations": v.mitigation_suggestions,
                    "affected_operation": v.affected_operation,
                    "metadata": v.metadata,
                }
                for v in report.vulnerabilities
            ],
            "recommendations": report.recommendations,
            "config": report.analysis_config,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Vulnerability report exported to {output_path}")

    def _export_html(self, report: VulnerabilityReport, output_path: Path) -> None:
        """Export report as HTML."""
        html_content = [
            "<!DOCTYPE html>",
            "<html><head><title>Side-Channel Vulnerability Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".vulnerability { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }",
            ".critical { border-left: 5px solid #d9534f; }",
            ".high { border-left: 5px solid #f0ad4e; }",
            ".medium { border-left: 5px solid #5bc0de; }",
            ".low { border-left: 5px solid #5cb85c; }",
            "</style></head><body>",
            "<h1>Side-Channel Vulnerability Report</h1>",
            f"<h2>Summary: {report.summary_statistics.get('total_vulnerabilities', 0)} "
            "vulnerabilities found</h2>",
        ]

        # Summary statistics
        html_content.append("<h3>Summary</h3><ul>")
        for key, value in report.summary_statistics.items():
            html_content.append(f"<li><strong>{key}:</strong> {value}</li>")
        html_content.append("</ul>")

        # Vulnerabilities
        html_content.append("<h3>Vulnerabilities</h3>")
        for vuln in report.vulnerabilities:
            html_content.append(
                f'<div class="vulnerability {vuln.severity.value}">'
                f"<h4>{vuln.vulnerability_type.value.upper()} - {vuln.severity.value.upper()}</h4>"
                f"<p><strong>Confidence:</strong> {vuln.confidence:.2%}</p>"
                f"<p><strong>Description:</strong> {vuln.description}</p>"
                f"<p><strong>Evidence:</strong> {vuln.evidence}</p>"
            )
            if vuln.mitigation_suggestions:
                html_content.append("<p><strong>Mitigations:</strong></p><ul>")
                for mitigation in vuln.mitigation_suggestions:
                    html_content.append(f"<li>{mitigation}</li>")
                html_content.append("</ul>")
            html_content.append("</div>")

        # Recommendations
        if report.recommendations:
            html_content.append("<h3>Recommendations</h3><ul>")
            for rec in report.recommendations:
                html_content.append(f"<li>{rec}</li>")
            html_content.append("</ul>")

        html_content.append("</body></html>")

        with open(output_path, "w") as f:
            f.write("\n".join(html_content))

        logger.info(f"HTML report exported to {output_path}")

    def _perform_ttest_leakage(
        self,
        traces: Sequence[PowerTrace],
    ) -> NDArray[np.float64] | None:
        """Perform Welch's t-test for leakage detection (TVLA).

        Args:
            traces: Power traces with plaintexts.

        Returns:
            T-statistics for each sample point, or None if insufficient data.
        """
        if len(traces) < 10:
            return None

        # Partition traces by first plaintext byte (odd vs even)
        group0_traces = []
        group1_traces = []

        for trace in traces:
            if trace.plaintext is None or len(trace.plaintext) == 0:
                continue

            if trace.plaintext[0] % 2 == 0:
                group0_traces.append(trace.power)
            else:
                group1_traces.append(trace.power)

        if len(group0_traces) < 5 or len(group1_traces) < 5:
            return None

        power0 = np.array(group0_traces)
        power1 = np.array(group1_traces)

        # Perform Welch's t-test at each time point
        t_stats = np.zeros(power0.shape[1])

        for i in range(power0.shape[1]):
            t_stat, _ = stats.ttest_ind(power0[:, i], power1[:, i], equal_var=False)
            t_stats[i] = abs(t_stat) if not np.isnan(t_stat) else 0.0

        return t_stats

    def _analyze_em_leakage(
        self,
        power_matrix: NDArray[np.float64],
    ) -> SideChannelVulnerability:
        """Analyze electromagnetic emission leakage via frequency domain.

        Args:
            power_matrix: Power traces matrix (num_traces x num_samples).

        Returns:
            SideChannelVulnerability for EM analysis.
        """
        # Compute FFT for each trace
        fft_traces = np.fft.rfft(power_matrix, axis=1)
        magnitude_spectrum = np.abs(fft_traces)

        # Calculate variance across traces at each frequency
        freq_variance = np.var(magnitude_spectrum, axis=0)
        max_variance = float(np.max(freq_variance))
        mean_variance = float(np.mean(freq_variance))

        # Normalize by mean to get relative peaks
        if mean_variance > 0:
            peak_ratio = max_variance / mean_variance
        else:
            peak_ratio = 0.0

        # Assess EM leakage
        if peak_ratio > 10.0:
            severity = Severity.HIGH
        elif peak_ratio > 5.0:
            severity = Severity.MEDIUM
        elif peak_ratio > 3.0:
            severity = Severity.LOW
        else:
            severity = Severity.LOW

        confidence = min(1.0, power_matrix.shape[0] / 100.0)

        return SideChannelVulnerability(
            vulnerability_type=VulnerabilityType.EM,
            severity=severity,
            confidence=confidence,
            evidence=f"Peak frequency variance ratio: {peak_ratio:.2f}",
            description=(
                "Frequency-domain analysis shows potential EM emission leakage"
                if severity != Severity.LOW
                else "No significant EM leakage detected"
            ),
            mitigation_suggestions=[
                "Add EM shielding to device enclosure",
                "Use frequency-domain filtering",
                "Implement spread-spectrum techniques",
            ]
            if severity != Severity.LOW
            else [],
            metadata={
                "peak_variance_ratio": float(peak_ratio),
                "max_variance": max_variance,
                "mean_variance": mean_variance,
            },
        )

    def _analyze_power_variance(
        self,
        traces: Sequence[PowerTrace],
    ) -> SideChannelVulnerability:
        """Analyze power consumption variance across different inputs.

        Args:
            traces: Power traces with plaintexts.

        Returns:
            SideChannelVulnerability for variance analysis.
        """
        # Group traces by first plaintext byte
        power_by_input: dict[int, list[NDArray[np.float64]]] = {}

        for trace in traces:
            if trace.plaintext is None or len(trace.plaintext) == 0:
                continue

            first_byte = trace.plaintext[0]
            if first_byte not in power_by_input:
                power_by_input[first_byte] = []
            power_by_input[first_byte].append(trace.power)

        if len(power_by_input) < 2:
            return SideChannelVulnerability(
                vulnerability_type=VulnerabilityType.POWER,
                severity=Severity.LOW,
                confidence=0.0,
                evidence="Insufficient input variation",
                description="Cannot assess variance - need multiple input values",
            )

        # Calculate mean power for each input
        mean_powers = {}
        for byte_val, powers in power_by_input.items():
            power_array = np.array(powers)
            mean_powers[byte_val] = np.mean(power_array, axis=0)

        # Calculate variance across different inputs
        mean_power_matrix = np.array(list(mean_powers.values()))
        inter_input_variance = np.var(mean_power_matrix, axis=0)
        max_variance = float(np.max(inter_input_variance))

        # Assess severity
        if max_variance > 0.1:
            severity = Severity.MEDIUM
        elif max_variance > 0.01:
            severity = Severity.LOW
        else:
            severity = Severity.LOW

        confidence = min(1.0, len(traces) / 100.0)

        return SideChannelVulnerability(
            vulnerability_type=VulnerabilityType.POWER,
            severity=severity,
            confidence=confidence,
            evidence=f"Max power variance across inputs: {max_variance:.6f}",
            description=(
                "Power variance analysis shows data-dependent consumption"
                if severity != Severity.LOW
                else "Power variance across inputs is low"
            ),
            mitigation_suggestions=[
                "Implement power balancing techniques",
                "Use dual-rail encoding",
                "Add noise injection",
            ]
            if severity != Severity.LOW
            else [],
            metadata={
                "max_variance": max_variance,
                "num_unique_inputs": len(power_by_input),
            },
        )
