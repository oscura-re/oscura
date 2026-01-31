"""IEEE standard citation system for professional reports.

This module provides automatic citation management for IEEE standards
referenced in analysis reports, with proper DOI links and bibliographic
formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# IEEE Standards Database with DOIs
IEEE_STANDARDS: dict[str, dict[str, str]] = {
    "181": {
        "title": "IEEE Standard for Transitions, Pulses, and Related Waveforms",
        "year": "2011",
        "doi": "10.1109/IEEESTD.2011.6016359",
        "url": "https://doi.org/10.1109/IEEESTD.2011.6016359",
        "full_name": "IEEE Std 181-2011",
        "scope": "Pulse measurement terminology, definitions, and algorithms",
    },
    "1241": {
        "title": "IEEE Standard for Terminology and Test Methods for Analog-to-Digital Converters",
        "year": "2010",
        "doi": "10.1109/IEEESTD.2011.5692956",
        "url": "https://doi.org/10.1109/IEEESTD.2011.5692956",
        "full_name": "IEEE Std 1241-2010",
        "scope": "ADC characterization including SNR, ENOB, and dynamic performance",
    },
    "1057": {
        "title": "IEEE Standard for Digitizing Waveform Recorders",
        "year": "2017",
        "doi": "10.1109/IEEESTD.2017.8291139",
        "url": "https://doi.org/10.1109/IEEESTD.2017.8291139",
        "full_name": "IEEE Std 1057-2017",
        "scope": "Oscilloscope and digitizer performance specifications",
    },
    "2414": {
        "title": "IEEE Standard for Jitter and Phase Noise",
        "year": "2020",
        "doi": "10.1109/IEEESTD.2020.9268529",
        "url": "https://doi.org/10.1109/IEEESTD.2020.9268529",
        "full_name": "IEEE Std 2414-2020",
        "scope": "Jitter measurement and analysis methodologies",
    },
    "1459": {
        "title": "IEEE Standard Definitions for the Measurement of Electric Power Quantities Under Sinusoidal, Nonsinusoidal, Balanced, or Unbalanced Conditions",
        "year": "2010",
        "doi": "10.1109/IEEESTD.2010.5439063",
        "url": "https://doi.org/10.1109/IEEESTD.2010.5439063",
        "full_name": "IEEE Std 1459-2010",
        "scope": "Power measurement definitions and algorithms",
    },
    "829": {
        "title": "IEEE Standard for Software and System Test Documentation",
        "year": "2008",
        "doi": "10.1109/IEEESTD.2008.4578383",
        "url": "https://doi.org/10.1109/IEEESTD.2008.4578383",
        "full_name": "IEEE Std 829-2008",
        "scope": "Test documentation and reporting standards",
    },
}


@dataclass
class Citation:
    """A citation to an IEEE standard.

    Attributes:
        standard_id: IEEE standard number (e.g., "181", "1241").
        section: Specific section if applicable.
        context: Context where citation is used.
        page: Page number if applicable.
    """

    standard_id: str
    section: str | None = None
    context: str = ""
    page: int | None = None

    def format_inline(self) -> str:
        """Format citation for inline use.

        Returns:
            Formatted citation string (e.g., "[IEEE 181]" or "[IEEE 181 §3.1]").

        Example:
            >>> citation = Citation("181", section="3.1")
            >>> citation.format_inline()
            '[IEEE 181 §3.1]'
        """
        if self.section:
            return f"[IEEE {self.standard_id} §{self.section}]"
        return f"[IEEE {self.standard_id}]"

    def format_bibliography(self) -> str:
        """Format citation for bibliography/references section.

        Returns:
            Formatted bibliography entry with DOI link.

        Example:
            >>> citation = Citation("181")
            >>> print(citation.format_bibliography())
            IEEE Std 181-2011, "IEEE Standard for Transitions, Pulses, and Related Waveforms," 2011. DOI: 10.1109/IEEESTD.2011.6016359
        """
        if self.standard_id not in IEEE_STANDARDS:
            return f"[IEEE {self.standard_id}] - Standard not in database"

        std = IEEE_STANDARDS[self.standard_id]
        return f'{std["full_name"]}, "{std["title"]}," {std["year"]}. DOI: {std["doi"]}'

    def get_url(self) -> str:
        """Get DOI URL for the standard.

        Returns:
            DOI URL string or empty string if not found.

        Example:
            >>> citation = Citation("181")
            >>> citation.get_url()
            'https://doi.org/10.1109/IEEESTD.2011.6016359'
        """
        if self.standard_id in IEEE_STANDARDS:
            return IEEE_STANDARDS[self.standard_id]["url"]
        return ""


@dataclass
class CitationManager:
    """Manages citations in a report.

    Tracks all citations used in a report and generates bibliographies.
    Automatically deduplicates and sorts citations.

    Attributes:
        citations: List of all citations in the report.

    Example:
        >>> manager = CitationManager()
        >>> manager.add_citation("181", section="3.1", context="Rise time measurement")
        >>> manager.add_citation("1241", context="SNR calculation")
        >>> html = manager.generate_bibliography_html()
    """

    citations: list[Citation] = field(default_factory=list)

    def add_citation(
        self,
        standard_id: str,
        section: str | None = None,
        context: str = "",
        page: int | None = None,
    ) -> Citation:
        """Add a citation to the report.

        Args:
            standard_id: IEEE standard number (e.g., "181").
            section: Specific section if applicable.
            context: Context where citation is used.
            page: Page number if applicable.

        Returns:
            The created Citation object.

        Example:
            >>> manager = CitationManager()
            >>> cite = manager.add_citation("181", section="3.1")
            >>> cite.format_inline()
            '[IEEE 181 §3.1]'
        """
        citation = Citation(
            standard_id=standard_id,
            section=section,
            context=context,
            page=page,
        )
        self.citations.append(citation)
        return citation

    def get_unique_citations(self) -> list[Citation]:
        """Get unique citations (deduplicated by standard_id).

        Returns:
            List of unique citations sorted by standard ID.

        Example:
            >>> manager = CitationManager()
            >>> manager.add_citation("181")
            >>> manager.add_citation("181", section="3.1")
            >>> manager.add_citation("1241")
            >>> len(manager.get_unique_citations())
            2
        """
        seen: set[str] = set()
        unique: list[Citation] = []

        for citation in sorted(self.citations, key=lambda c: c.standard_id):
            if citation.standard_id not in seen:
                seen.add(citation.standard_id)
                unique.append(citation)

        return unique

    def generate_bibliography_markdown(self) -> str:
        """Generate bibliography in Markdown format.

        Returns:
            Markdown-formatted bibliography section.

        Example:
            >>> manager = CitationManager()
            >>> manager.add_citation("181")
            >>> md = manager.generate_bibliography_markdown()
            >>> "IEEE 181" in md
            True
        """
        if not self.citations:
            return ""

        lines = ["## References", ""]

        for citation in self.get_unique_citations():
            lines.append(f"- {citation.format_bibliography()}")

        return "\n".join(lines)

    def generate_bibliography_html(self) -> str:
        """Generate bibliography in HTML format with DOI links.

        Returns:
            HTML-formatted bibliography section.

        Example:
            >>> manager = CitationManager()
            >>> manager.add_citation("181")
            >>> html = manager.generate_bibliography_html()
            >>> "doi.org" in html
            True
        """
        if not self.citations:
            return ""

        lines = ['<div class="references">', "<h2>References</h2>", "<ol>"]

        for citation in self.get_unique_citations():
            url = citation.get_url()
            bib = citation.format_bibliography()

            if url:
                lines.append(f'<li><a href="{url}" target="_blank">{bib}</a></li>')
            else:
                lines.append(f"<li>{bib}</li>")

        lines.extend(["</ol>", "</div>"])
        return "\n".join(lines)

    def get_citation_context(self) -> dict[str, list[str]]:
        """Get contexts where each standard was cited.

        Returns:
            Dictionary mapping standard IDs to list of contexts.

        Example:
            >>> manager = CitationManager()
            >>> manager.add_citation("181", context="Rise time")
            >>> manager.add_citation("181", context="Fall time")
            >>> contexts = manager.get_citation_context()
            >>> len(contexts["181"])
            2
        """
        context_map: dict[str, list[str]] = {}

        for citation in self.citations:
            if citation.context:
                if citation.standard_id not in context_map:
                    context_map[citation.standard_id] = []
                context_map[citation.standard_id].append(citation.context)

        return context_map


def get_standard_info(standard_id: str) -> dict[str, Any]:
    """Get information about an IEEE standard.

    Args:
        standard_id: IEEE standard number (e.g., "181").

    Returns:
        Dictionary with standard metadata or empty dict if not found.

    Example:
        >>> info = get_standard_info("181")
        >>> info["year"]
        '2011'
        >>> "pulse" in info["scope"].lower()
        True
    """
    return IEEE_STANDARDS.get(standard_id, {})


def list_available_standards() -> list[str]:
    """List all IEEE standards in the citation database.

    Returns:
        List of standard IDs.

    Example:
        >>> standards = list_available_standards()
        >>> "181" in standards
        True
        >>> len(standards) >= 5
        True
    """
    return sorted(IEEE_STANDARDS.keys())


def auto_cite_measurement(measurement_name: str) -> str | None:
    """Automatically determine which IEEE standard to cite for a measurement.

    Args:
        measurement_name: Name of measurement (e.g., "rise_time", "snr").

    Returns:
        Standard ID to cite, or None if no match.

    Example:
        >>> auto_cite_measurement("rise_time")
        '181'
        >>> auto_cite_measurement("snr")
        '1241'
        >>> auto_cite_measurement("jitter")
        '2414'
    """
    measurement_lower = measurement_name.lower()

    # Pulse/waveform measurements -> IEEE 181
    if any(
        term in measurement_lower
        for term in [
            "rise",
            "fall",
            "pulse",
            "transition",
            "overshoot",
            "settling",
            "slew",
        ]
    ):
        return "181"

    # ADC/quantization measurements -> IEEE 1241
    if any(
        term in measurement_lower
        for term in ["snr", "sinad", "enob", "thd", "sfdr", "quantization"]
    ):
        return "1241"

    # Jitter measurements -> IEEE 2414
    if any(term in measurement_lower for term in ["jitter", "phase_noise", "timing"]):
        return "2414"

    # Power measurements -> IEEE 1459
    if any(term in measurement_lower for term in ["power", "rms", "thd", "distortion", "harmonic"]):
        return "1459"

    # Oscilloscope/digitizer -> IEEE 1057
    if any(
        term in measurement_lower for term in ["bandwidth", "sample_rate", "resolution", "accuracy"]
    ):
        return "1057"

    return None
