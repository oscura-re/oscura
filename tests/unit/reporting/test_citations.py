"""Tests for IEEE citation system."""

from oscura.reporting.citations import (
    Citation,
    CitationManager,
    auto_cite_measurement,
    get_standard_info,
    list_available_standards,
)


def test_citation_format_inline():
    """Test inline citation formatting."""
    citation = Citation("181")
    assert citation.format_inline() == "[IEEE 181]"

    citation_with_section = Citation("181", section="3.1")
    assert citation_with_section.format_inline() == "[IEEE 181 §3.1]"


def test_citation_format_bibliography():
    """Test bibliography citation formatting."""
    citation = Citation("181")
    bib = citation.format_bibliography()

    assert "IEEE Std 181-2011" in bib
    assert "DOI" in bib
    assert "10.1109" in bib


def test_citation_get_url():
    """Test getting DOI URL."""
    citation = Citation("181")
    url = citation.get_url()

    assert url.startswith("https://doi.org/")
    assert "10.1109" in url


def test_citation_manager_add():
    """Test adding citations."""
    manager = CitationManager()

    cite1 = manager.add_citation("181", section="3.1", context="Rise time")
    assert cite1.standard_id == "181"
    assert cite1.section == "3.1"

    assert len(manager.citations) == 1


def test_citation_manager_unique():
    """Test unique citation filtering."""
    manager = CitationManager()

    manager.add_citation("181")
    manager.add_citation("181", section="3.1")
    manager.add_citation("1241")
    manager.add_citation("181", section="3.2")

    unique = manager.get_unique_citations()
    assert len(unique) == 2
    assert "181" in [c.standard_id for c in unique]
    assert "1241" in [c.standard_id for c in unique]


def test_citation_manager_bibliography_markdown():
    """Test Markdown bibliography generation."""
    manager = CitationManager()
    manager.add_citation("181")
    manager.add_citation("1241")

    md = manager.generate_bibliography_markdown()

    assert "## References" in md
    assert "IEEE Std 181" in md
    assert "IEEE Std 1241" in md


def test_citation_manager_bibliography_html():
    """Test HTML bibliography generation."""
    manager = CitationManager()
    manager.add_citation("181")

    html = manager.generate_bibliography_html()

    assert "<h2>References</h2>" in html
    assert "doi.org" in html
    assert "href=" in html


def test_citation_manager_context():
    """Test citation context tracking."""
    manager = CitationManager()
    manager.add_citation("181", context="Rise time measurement")
    manager.add_citation("181", context="Fall time measurement")
    manager.add_citation("1241", context="SNR calculation")

    contexts = manager.get_citation_context()

    assert len(contexts["181"]) == 2
    assert len(contexts["1241"]) == 1
    assert "Rise time" in contexts["181"][0]


def test_get_standard_info():
    """Test standard info retrieval."""
    info = get_standard_info("181")

    assert info["year"] == "2011"
    assert info["doi"] == "10.1109/IEEESTD.2011.6016359"
    assert "pulse" in info["scope"].lower()


def test_list_available_standards():
    """Test listing standards."""
    standards = list_available_standards()

    assert "181" in standards
    assert "1241" in standards
    assert "2414" in standards
    assert len(standards) >= 5


def test_auto_cite_measurement():
    """Test automatic citation selection."""
    # Pulse/waveform -> IEEE 181
    assert auto_cite_measurement("rise_time") == "181"
    assert auto_cite_measurement("fall_time") == "181"
    assert auto_cite_measurement("pulse_width") == "181"

    # ADC -> IEEE 1241
    assert auto_cite_measurement("snr") == "1241"
    assert auto_cite_measurement("sinad") == "1241"
    assert auto_cite_measurement("enob") == "1241"

    # Jitter -> IEEE 2414
    assert auto_cite_measurement("jitter") == "2414"
    assert auto_cite_measurement("phase_noise") == "2414"

    # Power -> IEEE 1459
    assert auto_cite_measurement("power") == "1459"
    assert auto_cite_measurement("rms") == "1459"

    # Unknown
    assert auto_cite_measurement("unknown_parameter") is None


def test_citation_empty_manager():
    """Test empty citation manager."""
    manager = CitationManager()

    assert manager.generate_bibliography_markdown() == ""
    assert manager.generate_bibliography_html() == ""
    assert manager.get_citation_context() == {}


def test_citation_unknown_standard():
    """Test citation with unknown standard."""
    citation = Citation("9999")

    bib = citation.format_bibliography()
    assert "not in database" in bib

    url = citation.get_url()
    assert url == ""
