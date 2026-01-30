"""Tests for measurement interpretation."""

import pytest

from oscura.reporting.interpretation import (
    ComplianceStatus,
    MeasurementInterpretation,
    QualityLevel,
    compliance_check,
    generate_finding,
    interpret_measurement,
    interpret_results_batch,
    quality_score,
)


def test_interpret_rise_time():
    """Test rise time interpretation."""
    # Fast rise time
    interp = interpret_measurement("rise_time", 2e-9, "s")
    assert interp.quality == QualityLevel.GOOD
    assert "fast" in interp.interpretation.lower()

    # Slow rise time
    interp = interpret_measurement("rise_time", 200e-9, "s")
    assert interp.quality == QualityLevel.MARGINAL
    assert "slow" in interp.interpretation.lower()


def test_interpret_snr():
    """Test SNR interpretation."""
    # Excellent SNR
    interp = interpret_measurement("snr", 65, "dB")
    assert interp.quality == QualityLevel.EXCELLENT
    assert "excellent" in interp.interpretation.lower()

    # Poor SNR
    interp = interpret_measurement("snr", 15, "dB")
    assert interp.quality == QualityLevel.POOR
    assert len(interp.recommendations) > 0


def test_interpret_jitter():
    """Test jitter interpretation."""
    # Low jitter (5 ps)
    interp = interpret_measurement("jitter", 5e-12, "s")
    assert interp.quality == QualityLevel.EXCELLENT

    # High jitter (500 ps)
    interp = interpret_measurement("jitter", 500e-12, "s")
    assert interp.quality == QualityLevel.MARGINAL
    assert len(interp.recommendations) > 0


def test_interpret_bandwidth():
    """Test bandwidth interpretation."""
    # Wide bandwidth
    interp = interpret_measurement("bandwidth", 2e9, "Hz")
    assert interp.quality == QualityLevel.EXCELLENT
    assert "GHz" in interp.interpretation

    # Limited bandwidth
    interp = interpret_measurement("bandwidth", 5e6, "Hz")
    assert interp.quality == QualityLevel.MARGINAL


def test_interpret_with_specs():
    """Test interpretation with specifications."""
    # Within spec with good margin
    interp = interpret_measurement("voltage", 3.3, "V", spec_min=2.0, spec_max=5.0)
    assert interp.quality in (QualityLevel.GOOD, QualityLevel.ACCEPTABLE)

    # Below spec
    interp = interpret_measurement("voltage", 1.5, "V", spec_min=2.0, spec_max=5.0)
    assert interp.quality == QualityLevel.FAILED
    assert len(interp.recommendations) > 0

    # Above spec
    interp = interpret_measurement("voltage", 5.5, "V", spec_min=2.0, spec_max=5.0)
    assert interp.quality == QualityLevel.FAILED


def test_interpret_string_value():
    """Test interpretation of non-numeric value."""
    interp = interpret_measurement("status", "PASS", "")
    assert interp.value == "PASS"
    assert "non-numeric" in interp.interpretation.lower()


def test_generate_finding():
    """Test finding generation."""
    measurements = {"snr": 25.5, "thd": -60.2}

    finding = generate_finding("Signal Quality", measurements, severity="info")

    assert finding.title == "Signal Quality"
    assert "snr" in finding.description.lower()
    assert len(finding.measurements) == 2


def test_generate_finding_with_severity():
    """Test finding with different severity levels."""
    measurements = {"voltage": 1.0}

    # Warning
    finding = generate_finding("Low Voltage", measurements, severity="warning")
    assert finding.severity == "warning"
    assert len(finding.recommendation) > 0

    # Critical
    finding = generate_finding("Critical Failure", measurements, severity="critical")
    assert finding.severity == "critical"
    assert "immediate" in finding.recommendation.lower()


def test_quality_score_calculation():
    """Test quality score calculation."""
    measurements = {"snr": 90, "bandwidth": 80, "jitter": 70}

    score, level = quality_score(measurements)

    assert 0 <= score <= 100
    assert level in QualityLevel
    assert score == pytest.approx(80.0, abs=1.0)


def test_quality_score_with_weights():
    """Test quality score with weights."""
    measurements = {"snr": 90, "bandwidth": 50}
    weights = {"snr": 2.0, "bandwidth": 1.0}

    score, level = quality_score(measurements, weights)

    # Weighted average: (90*2 + 50*1) / 3 = 76.67
    assert score == pytest.approx(76.67, abs=1.0)


def test_quality_score_empty():
    """Test quality score with empty measurements."""
    score, level = quality_score({})
    assert score == 0.0
    assert level == QualityLevel.POOR


def test_quality_score_levels():
    """Test quality score level thresholds."""
    # Excellent
    score, level = quality_score({"m1": 95})
    assert level == QualityLevel.EXCELLENT

    # Good
    score, level = quality_score({"m1": 80})
    assert level == QualityLevel.GOOD

    # Acceptable
    score, level = quality_score({"m1": 65})
    assert level == QualityLevel.ACCEPTABLE

    # Marginal
    score, level = quality_score({"m1": 45})
    assert level == QualityLevel.MARGINAL

    # Poor
    score, level = quality_score({"m1": 30})
    assert level == QualityLevel.POOR


def test_compliance_check_pass():
    """Test compliance check passing."""
    status, msg = compliance_check("rise_time", 2.5e-9, "181", {"max": 5e-9})

    assert status == ComplianceStatus.COMPLIANT
    assert "compliant" in msg.lower()
    assert "181" in msg


def test_compliance_check_fail_min():
    """Test compliance check failing minimum."""
    status, msg = compliance_check("voltage", 1.5, "1057", {"min": 2.0, "max": 5.0})

    assert status == ComplianceStatus.NON_COMPLIANT
    assert "below" in msg.lower()


def test_compliance_check_fail_max():
    """Test compliance check failing maximum."""
    status, msg = compliance_check("voltage", 5.5, "1057", {"min": 2.0, "max": 5.0})

    assert status == ComplianceStatus.NON_COMPLIANT
    assert "exceeds" in msg.lower()


def test_compliance_check_marginal():
    """Test marginal compliance."""
    # Just within limits (5% margin)
    status, msg = compliance_check("voltage", 2.05, "1057", {"min": 2.0, "max": 3.0})

    assert status == ComplianceStatus.MARGINAL
    assert "marginal" in msg.lower()


def test_compliance_check_no_limits():
    """Test compliance check without limits."""
    status, msg = compliance_check("voltage", 3.3, "1057", None)

    assert status == ComplianceStatus.NOT_APPLICABLE
    assert "no limits" in msg.lower()


def test_interpret_results_batch():
    """Test batch interpretation."""
    results = {
        "rise_time": {"value": 2.5e-9, "units": "s", "spec_max": 5e-9},
        "snr": {"value": 45.2, "units": "dB"},
        "bandwidth": {"value": 1e9, "units": "Hz"},
    }

    interpretations = interpret_results_batch(results)

    assert len(interpretations) == 3
    assert "rise_time" in interpretations
    assert "snr" in interpretations
    assert "bandwidth" in interpretations

    # Check quality levels assigned
    assert all(isinstance(i.quality, QualityLevel) for i in interpretations.values())
