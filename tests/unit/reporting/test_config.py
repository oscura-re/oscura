import pytest

"""Tests for comprehensive analysis report configuration.

Tests verify the configuration dataclasses, enums, and analysis capabilities
registry work correctly.
"""

from oscura.reporting.config import (
    ANALYSIS_CAPABILITIES,
    AnalysisConfig,
    AnalysisDomain,
    AnalysisError,
    DataOutputConfig,
    DomainConfig,
    InputType,
    ProgressInfo,
    get_available_analyses,
)

pytestmark = pytest.mark.unit


class TestInputType:
    """Tests for InputType enum."""

    def test_all_input_types_defined(self):
        """Verify all expected input types are defined."""
        expected = {"waveform", "digital", "binary", "pcap", "iq", "packets", "sparams"}
        actual = {t.value for t in InputType}
        assert expected == actual

    def test_input_type_string_values(self):
        """Verify InputType values are lowercase strings."""
        for t in InputType:
            assert t.value == t.value.lower()


class TestAnalysisDomain:
    """Tests for AnalysisDomain enum."""

    def test_fourteen_domains_defined(self):
        """Verify exactly 14 analysis domains are defined."""
        assert len(AnalysisDomain) == 14

    def test_all_domains_defined(self):
        """Verify all expected domains are defined."""
        expected = {
            "waveform",
            "digital",
            "timing",
            "spectral",
            "statistics",
            "patterns",
            "jitter",
            "eye",
            "power",
            "protocols",
            "signal_integrity",
            "inference",
            "packet",
            "entropy",
        }
        actual = {d.value for d in AnalysisDomain}
        assert expected == actual


class TestDomainConfig:
    """Tests for DomainConfig dataclass."""

    def test_default_config(self):
        """Test default domain configuration."""
        config = DomainConfig()
        assert config.enabled is True
        assert config.parameters == {}
        assert config.timeout is None

    def test_custom_config(self):
        """Test custom domain configuration."""
        config = DomainConfig(
            enabled=False,
            parameters={"fft_size": 1024},
            timeout=60.0,
        )
        assert config.enabled is False
        assert config.parameters["fft_size"] == 1024
        assert config.timeout == 60.0


class TestDataOutputConfig:
    """Tests for DataOutputConfig dataclass."""

    def test_default_full_data_mode(self):
        """Test that default config enables full data mode."""
        config = DataOutputConfig()
        assert config.full_data_mode is True
        assert config.max_array_elements is None
        assert config.max_list_items is None
        assert config.max_bytes_sample is None
        assert config.smart_aggregation is True

    def test_custom_limits(self):
        """Test custom output limits configuration."""
        config = DataOutputConfig(
            full_data_mode=False,
            max_array_elements=1000,
            max_list_items=500,
            max_bytes_sample=100,
        )
        assert config.full_data_mode is False
        assert config.max_array_elements == 1000
        assert config.max_list_items == 500
        assert config.max_bytes_sample == 100


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_default_config(self):
        """Test default analysis configuration."""
        config = AnalysisConfig()
        assert config.domains is None  # All applicable
        assert config.exclude_domains == []
        assert config.generate_plots is True
        assert config.plot_format == "png"
        assert config.plot_dpi == 150
        assert config.continue_on_error is True

    def test_is_domain_enabled_default(self):
        """Test domain enabled check with default config."""
        config = AnalysisConfig()
        assert config.is_domain_enabled(AnalysisDomain.SPECTRAL) is True
        assert config.is_domain_enabled(AnalysisDomain.WAVEFORM) is True

    def test_is_domain_enabled_explicit_list(self):
        """Test domain enabled check with explicit domain list."""
        config = AnalysisConfig(domains=[AnalysisDomain.SPECTRAL, AnalysisDomain.WAVEFORM])
        assert config.is_domain_enabled(AnalysisDomain.SPECTRAL) is True
        assert config.is_domain_enabled(AnalysisDomain.JITTER) is False

    def test_is_domain_enabled_exclude(self):
        """Test domain exclusion."""
        config = AnalysisConfig(exclude_domains=[AnalysisDomain.JITTER, AnalysisDomain.EYE])
        assert config.is_domain_enabled(AnalysisDomain.SPECTRAL) is True
        assert config.is_domain_enabled(AnalysisDomain.JITTER) is False

    def test_is_domain_enabled_domain_config(self):
        """Test domain enabled via per-domain config."""
        config = AnalysisConfig(
            domain_config={
                AnalysisDomain.SPECTRAL: DomainConfig(enabled=False),
            }
        )
        assert config.is_domain_enabled(AnalysisDomain.SPECTRAL) is False
        assert config.is_domain_enabled(AnalysisDomain.WAVEFORM) is True

    def test_get_domain_config(self):
        """Test get domain config returns correct config or default."""
        custom = DomainConfig(timeout=60.0)
        config = AnalysisConfig(domain_config={AnalysisDomain.SPECTRAL: custom})

        spectral_config = config.get_domain_config(AnalysisDomain.SPECTRAL)
        assert spectral_config.timeout == 60.0

        waveform_config = config.get_domain_config(AnalysisDomain.WAVEFORM)
        assert waveform_config.timeout is None  # Default


class TestProgressInfo:
    """Tests for ProgressInfo dataclass."""

    def test_progress_info_creation(self):
        """Test creating progress info."""
        info = ProgressInfo(
            phase="analyzing",
            domain=AnalysisDomain.SPECTRAL,
            function="compute_fft",
            percent=50.0,
            message="Computing FFT",
            elapsed_seconds=5.0,
            estimated_remaining_seconds=5.0,
        )
        assert info.phase == "analyzing"
        assert info.domain == AnalysisDomain.SPECTRAL
        assert info.percent == 50.0


class TestAnalysisError:
    """Tests for AnalysisError dataclass."""

    def test_analysis_error_creation(self):
        """Test creating analysis error record."""
        error = AnalysisError(
            domain=AnalysisDomain.SPECTRAL,
            function="compute_fft",
            error_type="ValueError",
            error_message="Invalid data",
            traceback=None,
            duration_ms=100.0,
        )
        assert error.domain == AnalysisDomain.SPECTRAL
        assert error.function == "compute_fft"
        assert error.error_type == "ValueError"


class TestAnalysisCapabilities:
    """Tests for ANALYSIS_CAPABILITIES registry."""

    def test_all_domains_have_capabilities(self):
        """Verify all domains have capabilities defined."""
        for domain in AnalysisDomain:
            assert domain in ANALYSIS_CAPABILITIES, f"Missing capabilities for {domain}"

    def test_capabilities_structure(self):
        """Verify each capability has required fields."""
        for domain, cap in ANALYSIS_CAPABILITIES.items():
            assert "description" in cap, f"{domain} missing description"
            assert "modules" in cap, f"{domain} missing modules"
            assert "requires" in cap, f"{domain} missing requires"
            assert isinstance(cap["modules"], list), f"{domain} modules must be list"
            assert isinstance(cap["requires"], list), f"{domain} requires must be list"


class TestGetAvailableAnalyses:
    """Tests for get_available_analyses function."""

    def test_waveform_input_analyses(self):
        """Test analyses available for waveform input."""
        domains = get_available_analyses(InputType.WAVEFORM)
        assert AnalysisDomain.WAVEFORM in domains
        assert AnalysisDomain.SPECTRAL in domains
        assert AnalysisDomain.POWER in domains

    def test_binary_input_analyses(self):
        """Test analyses available for binary input."""
        domains = get_available_analyses(InputType.BINARY)
        assert AnalysisDomain.ENTROPY in domains
        assert AnalysisDomain.STATISTICS in domains

    def test_pcap_input_analyses(self):
        """Test analyses available for PCAP input."""
        domains = get_available_analyses(InputType.PCAP)
        assert AnalysisDomain.PACKET in domains

    def test_returns_list(self):
        """Verify function returns a list."""
        domains = get_available_analyses(InputType.WAVEFORM)
        assert isinstance(domains, list)
        assert all(isinstance(d, AnalysisDomain) for d in domains)
