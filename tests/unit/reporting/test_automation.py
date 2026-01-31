"""Tests for automated analysis and reporting."""

from oscura.reporting.automation import (
    auto_interpret_results,
    flag_anomalies,
    generate_summary,
    identify_issues,
    suggest_follow_up_analyses,
)


def test_auto_interpret_results_flat():
    """Test auto interpretation of flat results."""
    results = {"snr": 45.2, "bandwidth": 1e9, "jitter": 50e-12}

    interpretations = auto_interpret_results(results)

    assert len(interpretations) == 3
    assert "snr" in interpretations
    assert "bandwidth" in interpretations
    assert "jitter" in interpretations


def test_auto_interpret_results_nested():
    """Test auto interpretation of nested results."""
    results = {
        "spectral": {"snr": {"value": 45.2, "units": "dB"}},
        "waveform": {"bandwidth": 1e9},
    }

    interpretations = auto_interpret_results(results)

    assert "snr" in interpretations
    assert "bandwidth" in interpretations


def test_generate_summary_basic():
    """Test summary generation."""
    results = {"snr": 45.2, "thd": -60.5, "bandwidth": 1e9}

    summary = generate_summary(results)

    assert "statistics" in summary
    assert "key_findings" in summary
    assert "recommendations" in summary
    assert "anomalies" in summary
    assert "interpretations" in summary


def test_generate_summary_statistics():
    """Test summary statistics."""
    results = {"m1": 10, "m2": 20, "m3": 30}

    summary = generate_summary(results)

    stats = summary["statistics"]
    assert stats["count"] == 3
    assert stats["mean"] == 20.0
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0


def test_flag_anomalies_outliers():
    """Test anomaly flagging for outliers."""
    measurements = {"m1": 10, "m2": 11, "m3": 12, "m4": 100}

    anomalies = flag_anomalies(measurements, threshold_std=1.5)

    assert len(anomalies) >= 1
    assert any(a["name"] == "m4" for a in anomalies)


def test_flag_anomalies_negative_snr():
    """Test anomaly flagging for invalid SNR."""
    measurements = {"snr": -5.0, "bandwidth": 1e9, "jitter": 10e-12}

    anomalies = flag_anomalies(measurements)

    assert len(anomalies) >= 1
    assert any("snr" in a["name"].lower() for a in anomalies)
    assert any(a.get("severity") == "critical" for a in anomalies)


def test_flag_anomalies_negative_bandwidth():
    """Test anomaly flagging for invalid bandwidth."""
    measurements = {"bandwidth": -100, "snr": 50, "jitter": 10e-12}

    anomalies = flag_anomalies(measurements)

    assert len(anomalies) >= 1
    assert any("bandwidth" in a["name"].lower() for a in anomalies)


def test_flag_anomalies_high_jitter():
    """Test anomaly flagging for unrealistic jitter."""
    measurements = {
        "jitter": 2.0,
        "snr": 50,
        "bandwidth": 1e9,
    }  # 2 seconds of jitter is unrealistic

    anomalies = flag_anomalies(measurements)

    # Should flag unrealistic jitter
    assert len(anomalies) >= 1
    assert any("jitter" in a["name"].lower() for a in anomalies)


def test_flag_anomalies_invalid_power_factor():
    """Test anomaly flagging for invalid power factor."""
    measurements = {
        "power_factor": 1.5,
        "voltage": 120,
        "current": 5,
    }  # Power factor must be <= 1

    anomalies = flag_anomalies(measurements)

    assert len(anomalies) >= 1
    assert any("power_factor" in a["name"].lower() for a in anomalies)


def test_flag_anomalies_insufficient_data():
    """Test anomaly flagging with insufficient data."""
    measurements = {"m1": 10}  # Only one measurement

    anomalies = flag_anomalies(measurements)

    # Should not crash, may return domain-specific anomalies only
    assert isinstance(anomalies, list)


def test_suggest_follow_up_analyses_low_snr():
    """Test follow-up suggestions for low SNR."""
    measurements = {"snr": 15.5}

    suggestions = suggest_follow_up_analyses(measurements)

    assert len(suggestions) > 0
    assert any("noise" in s.lower() for s in suggestions)


def test_suggest_follow_up_analyses_high_thd():
    """Test follow-up suggestions for high THD."""
    measurements = {"thd": -30}  # High distortion

    suggestions = suggest_follow_up_analyses(measurements)

    assert len(suggestions) > 0
    assert any("harmonic" in s.lower() for s in suggestions)


def test_suggest_follow_up_analyses_jitter():
    """Test follow-up suggestions for jitter."""
    measurements = {"rms_jitter": 100e-12}

    suggestions = suggest_follow_up_analyses(measurements)

    assert len(suggestions) > 0
    assert any("jitter" in s.lower() for s in suggestions)


def test_suggest_follow_up_analyses_power():
    """Test follow-up suggestions for power measurements."""
    measurements = {"active_power": 1000, "reactive_power": 200}

    suggestions = suggest_follow_up_analyses(measurements)

    assert len(suggestions) > 0
    assert any("power" in s.lower() for s in suggestions)


def test_identify_issues_from_anomalies():
    """Test issue identification from anomalies."""
    measurements = {"snr": -5}
    anomalies = [{"name": "snr", "value": -5, "severity": "critical", "reason": "Negative SNR"}]

    issues = identify_issues(measurements, anomalies)

    assert len(issues) >= 1
    assert any(i["severity"] == "critical" for i in issues)


def test_identify_issues_low_snr():
    """Test issue identification for low SNR."""
    measurements = {"snr": 15}
    anomalies = []

    issues = identify_issues(measurements, anomalies)

    assert len(issues) >= 1
    assert any(i["measurement"] == "snr" for i in issues)


def test_identify_issues_limited_bandwidth():
    """Test issue identification for limited bandwidth."""
    measurements = {"bandwidth": 5e6}  # 5 MHz
    anomalies = []

    issues = identify_issues(measurements, anomalies)

    assert len(issues) >= 1
    assert any(i["measurement"] == "bandwidth" for i in issues)


def test_identify_issues_empty():
    """Test issue identification with no issues."""
    measurements = {"snr": 60, "bandwidth": 1e9}
    anomalies = []

    issues = identify_issues(measurements, anomalies)

    # Should have no issues for good measurements
    assert isinstance(issues, list)


def test_auto_interpret_nested_complex():
    """Test complex nested structure interpretation."""
    results = {
        "digital": {
            "timing": {"rise_time": {"value": 2e-9, "units": "s"}},
            "quality": {"snr": 45.2},
        },
        "spectral": {"bandwidth": 1e9},
    }

    interpretations = auto_interpret_results(results)

    assert "rise_time" in interpretations
    assert "snr" in interpretations
    assert "bandwidth" in interpretations
