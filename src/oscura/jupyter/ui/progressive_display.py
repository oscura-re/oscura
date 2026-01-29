"""Progressive disclosure UI pattern for information hierarchy.

This module implements progressive disclosure to display simple, high-level
results first with options to drill down into details on demand.


Example:
    >>> from oscura.jupyter.ui import ProgressiveDisplay
    >>> display = ProgressiveDisplay()
    >>> output = display.render(result)
    >>> print(output.summary())  # Level 1: Summary
    >>> print(output.details())   # Level 2: Intermediate
    >>> print(output.expert())    # Level 3: Expert

References:
    Oscura Auto-Discovery Specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    """Progressive disclosure section.

    Attributes:
        title: Section title.
        summary: Brief summary text.
        content: Full content (shown when expanded).
        visualization: Optional plot or chart.
        is_collapsed: Whether section is initially collapsed.
        detail_level: Minimum detail level to show (1-3).
    """

    title: str
    summary: str
    content: str = ""
    visualization: Any = None
    is_collapsed: bool = True
    detail_level: int = 2


@dataclass
class ProgressiveOutput:
    """Progressive disclosure output container.

    Attributes:
        level1_content: Summary level content (≤5 items, ≤100 words).
        level2_sections: Intermediate level sections.
        level3_data: Expert level data (raw values, debug info).
        current_level: Current display level (1-3).
    """

    level1_content: str
    level2_sections: list[Section] = field(default_factory=list)
    level3_data: dict[str, Any] = field(default_factory=dict)
    current_level: int = 1

    def summary(self) -> str:
        """Get Level 1 summary.

        Returns:
            Brief summary with 3-5 key items.
        """
        return self.level1_content

    def details(self, level: str = "intermediate") -> str:
        """Get detailed view.

        Args:
            level: Detail level ("intermediate" or "expert").

        Returns:
            Formatted details at requested level.
        """
        if level == "expert":
            return self.expert()

        # Level 2: Intermediate details
        output = self.level1_content + "\n\n"

        for section in self.level2_sections:
            output += f"\n{'=' * 60}\n"
            output += f"{section.title}\n"
            output += f"{'=' * 60}\n"

            if section.is_collapsed:
                output += f"{section.summary}\n"
                output += "[+] Expand for details\n"
            else:
                output += f"{section.content}\n"

        return output

    def expert(self) -> str:
        """Get Level 3 expert view.

        Returns:
            Complete data including raw values and debug info.
        """
        output = self.details(level="intermediate") + "\n\n"

        if self.level3_data:
            output += "\n" + "=" * 60 + "\n"
            output += "EXPERT DETAILS\n"
            output += "=" * 60 + "\n"

            for key, value in self.level3_data.items():
                output += f"\n{key}:\n"
                output += f"  {value}\n"

        return output

    def has_level3(self) -> bool:
        """Check if expert level data is available.

        Returns:
            True if level 3 data exists.
        """
        return bool(self.level3_data)

    def expand_section(self, title: str) -> None:
        """Expand a collapsed section.

        Args:
            title: Section title to expand.
        """
        for section in self.level2_sections:
            if section.title == title:
                section.is_collapsed = False
                break

    def collapse_section(self, title: str) -> None:
        """Collapse an expanded section.

        Args:
            title: Section title to collapse.
        """
        for section in self.level2_sections:
            if section.title == title:
                section.is_collapsed = True
                break

    def export(
        self,
        path: str,
        detail_level: str = "intermediate",
    ) -> None:
        """Export to file with specified detail level.

        Args:
            path: Output file path.
            detail_level: Detail level ("summary", "intermediate", "expert").
        """
        if detail_level == "summary":
            content = self.summary()
        elif detail_level == "expert":
            content = self.expert()
        else:
            content = self.details()

        with open(path, "w") as f:
            f.write(content)


class ProgressiveDisplay:
    """Progressive disclosure display manager.

    Manages hierarchical display of information with three detail levels:
    - L1: Summary (≤5 items, ≤100 words)
    - L2: Intermediate (≤20 items, includes charts)
    - L3: Expert (full data, raw values, debug info)

    Example:
        >>> display = ProgressiveDisplay(default_level="summary")
        >>> output = display.render(analysis_result)
        >>> print(output.summary())  # Simple overview
        >>> output.expand_section("Timing Analysis")
        >>> print(output.details())  # More detail

    References:
        DISC-011: Progressive Disclosure
    """

    def __init__(
        self,
        default_level: str = "summary",
        max_summary_items: int = 5,
        enable_collapsible_sections: bool = True,
    ):
        """Initialize progressive display manager.

        Args:
            default_level: Default detail level ("summary", "intermediate", "expert").
            max_summary_items: Maximum items in summary (3-10).
            enable_collapsible_sections: Enable section collapse/expand.
        """
        self.default_level = default_level
        self.max_summary_items = max(3, min(10, max_summary_items))
        self.enable_collapsible_sections = enable_collapsible_sections

    def render(self, result: Any) -> ProgressiveOutput:
        """Render result with progressive disclosure.

        Args:
            result: Analysis result to render.

        Returns:
            ProgressiveOutput with hierarchical content.

        Example:
            >>> output = display.render(characterization_result)
            >>> print(output.summary())
        """
        level1_content = self._build_level1_summary(result)
        level2_sections = self._build_level2_sections(result)
        level3_data = self._build_level3_expert_data(result)
        current_level = self._determine_current_level()

        return ProgressiveOutput(
            level1_content=level1_content,
            level2_sections=level2_sections,
            level3_data=level3_data,
            current_level=current_level,
        )

    def _build_level1_summary(self, result: Any) -> str:
        """Build Level 1 summary content.

        Args:
            result: Analysis result.

        Returns:
            Summary string with key items.
        """
        level1_items = []

        if hasattr(result, "signal_type"):
            level1_items.append(f"Signal Type: {result.signal_type}")

        if hasattr(result, "confidence"):
            confidence_pct = result.confidence * 100
            level1_items.append(f"Confidence: {confidence_pct:.0f}%")

        if hasattr(result, "quality"):
            level1_items.append(f"Quality: {result.quality}")

        if hasattr(result, "status"):
            level1_items.append(f"Status: {result.status}")

        level1_items = level1_items[: self.max_summary_items]

        if not level1_items:
            return "Analysis complete. Expand for details."

        return "\n".join(level1_items)

    def _build_level2_sections(self, result: Any) -> list[Section]:
        """Build Level 2 detailed sections.

        Args:
            result: Analysis result.

        Returns:
            List of Section objects.
        """
        sections = []

        params_section = self._build_parameters_section(result)
        if params_section:
            sections.append(params_section)

        quality_section = self._build_quality_metrics_section(result)
        if quality_section:
            sections.append(quality_section)

        findings_section = self._build_findings_section(result)
        if findings_section:
            sections.append(findings_section)

        return sections

    def _build_parameters_section(self, result: Any) -> Section | None:
        """Build parameters section.

        Args:
            result: Analysis result.

        Returns:
            Section or None if no parameters.
        """
        if not (hasattr(result, "parameters") and result.parameters):
            return None

        params = result.parameters
        summary = f"{len(params)} parameters detected"

        content = "Parameters:\n"
        for key, value in params.items():
            content += f"  {key}: {value}\n"

        return Section(
            title="Parameters",
            summary=summary,
            content=content,
            is_collapsed=self.enable_collapsible_sections,
            detail_level=2,
        )

    def _build_quality_metrics_section(self, result: Any) -> Section | None:
        """Build quality metrics section.

        Args:
            result: Analysis result.

        Returns:
            Section or None if no quality data.
        """
        if not (hasattr(result, "quality") or hasattr(result, "metrics")):
            return None

        metrics = getattr(result, "metrics", {})
        summary = "Quality assessment available"
        content = "Quality Metrics:\n"

        if hasattr(result, "quality"):
            content += f"  Overall: {result.quality}\n"

        if metrics:
            for key, value in metrics.items():
                content += f"  {key}: {value}\n"

        return Section(
            title="Quality Metrics",
            summary=summary,
            content=content,
            is_collapsed=self.enable_collapsible_sections,
            detail_level=2,
        )

    def _build_findings_section(self, result: Any) -> Section | None:
        """Build findings section.

        Args:
            result: Analysis result.

        Returns:
            Section or None if no findings.
        """
        if not (hasattr(result, "findings") and result.findings):
            return None

        findings = result.findings
        summary = f"{len(findings)} findings"
        content = "Findings:\n"

        for i, finding in enumerate(findings, 1):
            if hasattr(finding, "title") and hasattr(finding, "description"):
                content += f"\n{i}. {finding.title}\n"
                content += f"   {finding.description}\n"
            else:
                content += f"\n{i}. {finding}\n"

        return Section(
            title="Findings",
            summary=summary,
            content=content,
            is_collapsed=self.enable_collapsible_sections,
            detail_level=2,
        )

    def _build_level3_expert_data(self, result: Any) -> dict[str, Any]:
        """Build Level 3 expert data.

        Args:
            result: Analysis result.

        Returns:
            Dict of expert-level data.
        """
        level3_data = {}

        if hasattr(result, "raw_data"):
            level3_data["raw_data"] = result.raw_data

        if hasattr(result, "algorithm_config"):
            level3_data["algorithm_config"] = result.algorithm_config

        if hasattr(result, "debug_trace"):
            level3_data["debug_trace"] = result.debug_trace

        return level3_data

    def _determine_current_level(self) -> int:
        """Determine current detail level from default.

        Returns:
            Current level (1-3).
        """
        level_map = {"summary": 1, "intermediate": 2, "expert": 3}
        return level_map.get(self.default_level, 1)


__all__ = [
    "ProgressiveDisplay",
    "ProgressiveOutput",
    "Section",
]
