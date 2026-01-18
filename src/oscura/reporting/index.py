"""Index file generation for comprehensive analysis reports.

This module provides HTML and Markdown index generation from analysis results
using a simple template engine (no external dependencies like Jinja2).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.reporting.config import AnalysisResult
    from oscura.reporting.output import OutputManager


class TemplateEngine:
    """Simple template engine for variable substitution and control flow.

    Supports:
    - {{variable}} - Variable substitution
    - {{#if condition}}...{{/if}} - Conditional blocks
    - {{#each items}}...{{/each}} - Iteration blocks
    - {{this}} - Current item in iteration

    Requirements:
    """

    def __init__(self) -> None:
        """Initialize template engine."""
        self._var_pattern = re.compile(r"\{\{([^#/}][^}]*)\}\}")
        self._if_pattern = re.compile(r"\{\{#if\s+([^}]+)\}\}(.*?)\{\{/if\}\}", re.DOTALL)
        self._each_pattern = re.compile(r"\{\{#each\s+([^}]+)\}\}(.*?)\{\{/each\}\}", re.DOTALL)

    def render(self, template: str, context: dict[str, Any]) -> str:
        """Render template with context.

        Args:
            template: Template string with placeholders.
            context: Context dictionary for variable substitution.

        Returns:
            Rendered template string.

        Examples:
            >>> engine = TemplateEngine()
            >>> engine.render("Hello {{name}}", {"name": "World"})
            'Hello World'
            >>> engine.render("{{#if show}}visible{{/if}}", {"show": True})
            'visible'
            >>> engine.render("{{#each items}}{{this}} {{/each}}", {"items": [1, 2]})
            '1 2 '
        """
        # Process each blocks first (innermost to outermost)
        result = self._process_each_blocks(template, context)

        # Process if blocks
        result = self._process_if_blocks(result, context)

        # Process variables
        result = self._process_variables(result, context)

        return result

    def _process_variables(self, template: str, context: dict[str, Any]) -> str:
        """Replace {{variable}} with values from context.

        Args:
            template: Template string.
            context: Context dictionary.

        Returns:
            Template with variables replaced.
        """

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1).strip()

            # Handle "this" for current iteration item
            if var_name == "this":
                return str(context.get("_current_item", ""))

            # Handle nested access like "domain.value"
            value: Any = context
            for key in var_name.split("."):
                if isinstance(value, dict):
                    value = value.get(key, "")
                elif hasattr(value, key):
                    value = getattr(value, key)
                else:
                    value = ""
                    break

            # Handle enum values
            if hasattr(value, "value"):
                value = value.value

            return str(value) if value is not None else ""

        return self._var_pattern.sub(replace_var, template)

    def _process_if_blocks(self, template: str, context: dict[str, Any]) -> str:
        """Process {{#if condition}}...{{/if}} blocks.

        Args:
            template: Template string.
            context: Context dictionary.

        Returns:
            Template with conditionals processed.
        """

        def replace_if(match: re.Match[str]) -> str:
            condition = match.group(1).strip()
            content = match.group(2)

            # Evaluate condition
            value = context.get(condition, False)

            # Truthy check
            if value and value != 0 and value != "" and value != []:
                return content
            return ""

        return self._if_pattern.sub(replace_if, template)

    def _process_each_blocks(self, template: str, context: dict[str, Any]) -> str:
        """Process {{#each items}}...{{/each}} blocks.

        Args:
            template: Template string.
            context: Context dictionary.

        Returns:
            Template with iterations processed.
        """
        # Manually find and process each blocks to handle nesting
        result = []
        pos = 0

        while pos < len(template):
            # Look for next {{#each}}
            start_match = re.search(r"\{\{#each\s+([^}]+)\}\}", template[pos:])
            if not start_match:
                # No more each blocks
                result.append(template[pos:])
                break

            # Add everything before this block
            result.append(template[pos : pos + start_match.start()])

            # Find the matching {{/each}} accounting for nesting
            items_name = start_match.group(1).strip()
            block_start = pos + start_match.end()
            block_end = self._find_matching_end(template, block_start, "each")

            if block_end == -1:
                # No matching end tag, skip this
                result.append(start_match.group(0))
                pos = block_start
                continue

            # Extract the item template
            item_template = template[block_start:block_end]

            # Get the items
            items = context.get(items_name, [])
            if not items:
                # Empty result
                pass
            else:
                # Render each item
                for item in items:
                    # Create context for this iteration
                    if isinstance(item, dict):
                        item_context = {**context, **item, "_current_item": item}
                    else:
                        item_context = {**context, "this": item, "_current_item": item}

                    # Recursively process nested blocks
                    rendered = self._process_each_blocks(item_template, item_context)
                    rendered = self._process_if_blocks(rendered, item_context)
                    rendered = self._process_variables(rendered, item_context)
                    result.append(rendered)

            # Move past the {{/each}}
            pos = block_end + len("{{/each}}")

        return "".join(result)

    def _find_matching_end(self, template: str, start_pos: int, block_type: str) -> int:
        """Find matching end tag for a block, accounting for nesting.

        Args:
            template: Template string.
            start_pos: Position after the opening tag.
            block_type: Block type (e.g., "each", "if").

        Returns:
            Position of the start of the matching {{/block_type}} tag, or -1 if not found.
        """
        open_tag = f"{{{{#{block_type}"
        close_tag = f"{{{{/{block_type}}}}}"
        depth = 1
        pos = start_pos

        while pos < len(template) and depth > 0:
            # Look for next open or close tag
            next_open = template.find(open_tag, pos)
            next_close = template.find(close_tag, pos)

            if next_close == -1:
                # No closing tag found
                return -1

            if next_open != -1 and next_open < next_close:
                # Found nested open tag
                depth += 1
                pos = next_open + len(open_tag)
            else:
                # Found close tag
                depth -= 1
                if depth == 0:
                    return next_close
                pos = next_close + len(close_tag)

        return -1


class IndexGenerator:
    """Generate HTML and Markdown index files from analysis results.

    Creates navigable index pages that link to all analysis outputs including
    plots, data files, and domain-specific results.

    Attributes:
        output_manager: Output manager for file operations.

    Requirements:
    """

    def __init__(self, output_manager: OutputManager) -> None:
        """Initialize index generator.

        Args:
            output_manager: Output manager for file operations.

        Examples:
            >>> from pathlib import Path
            >>> om = OutputManager(Path("/tmp/output"), "test")
            >>> generator = IndexGenerator(om)
        """
        self._output_manager = output_manager
        self._engine = TemplateEngine()

        # Template directory
        self._template_dir = Path(__file__).parent / "templates"

    def generate(
        self,
        result: AnalysisResult,
        include_formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate index files in requested formats.

        Args:
            result: Analysis result containing all output metadata.
            include_formats: Formats to generate (e.g., ["html", "md"]).
                Defaults to ["html", "md"] if None.

        Returns:
            Dictionary mapping format name to generated file path.

        Requirements:

        Examples:
            >>> # result = AnalysisResult(...)
            >>> # generator = IndexGenerator(output_manager)
            >>> # paths = generator.generate(result, ["html", "md"])
            >>> # paths["html"]  # Path to index.html
        """
        if include_formats is None:
            include_formats = ["html", "md"]

        # Build context from result
        context = self._build_context(result)

        # Generate each format
        outputs: dict[str, Path] = {}

        if "html" in include_formats:
            html_content = self._render_html(context)
            html_path = self._output_manager.save_text("index.html", html_content)
            outputs["html"] = html_path

        if "md" in include_formats:
            md_content = self._render_markdown(context)
            md_path = self._output_manager.save_text("index.md", md_content)
            outputs["md"] = md_path

        return outputs

    def _build_context(self, result: AnalysisResult) -> dict[str, Any]:
        """Build template context from AnalysisResult.

        Args:
            result: Analysis result.

        Returns:
            Context dictionary for template rendering.

        Requirements:
        """
        # Extract timestamp properly from output_dir name
        # Format is: YYYYMMDD_HHMMSS_name_analysis
        dir_name = result.output_dir.name
        timestamp = "N/A"
        if "_" in dir_name:
            parts = dir_name.split("_")
            if len(parts) >= 2:
                date_part = parts[0]  # YYYYMMDD
                time_part = parts[1]  # HHMMSS
                if len(date_part) == 8 and len(time_part) == 6:
                    try:
                        timestamp = (
                            f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} "
                            f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                        )
                    except (IndexError, ValueError):
                        timestamp = f"{date_part}_{time_part}"

        # Basic metadata
        context: dict[str, Any] = {
            "title": "Analysis Report",
            "input_name": result.input_file or "In-Memory Data",
            "input_size": self._format_size(result.input_file),
            "input_type": result.input_type.value,
            "timestamp": timestamp,
            "duration": self._format_duration(result.duration_seconds),
            "total_analyses": result.total_analyses,
            "successful": result.successful_analyses,
            "failed": result.failed_analyses,
            "domains_count": len(result.domain_summaries),
            "has_errors": len(result.errors) > 0,
        }

        # Build domain information
        # domain_summaries contains {AnalysisDomain: {func_name: result, ...}}
        domains: list[dict[str, Any]] = []
        for domain, domain_results in result.domain_summaries.items():
            # Count successful analyses in this domain
            # domain_results is a dict of {function_name: result_value}
            analyses_count = len(domain_results) if isinstance(domain_results, dict) else 0

            # Find plots for this domain
            domain_plots = []
            if result.plot_paths:
                domain_id = domain.value
                for plot_path in result.plot_paths:
                    # Check if plot belongs to this domain
                    plot_str = str(plot_path)
                    if f"/{domain_id}/" in plot_str or plot_str.startswith(domain_id):
                        domain_plots.append(
                            {
                                "title": plot_path.stem.replace("_", " ").title(),
                                "path": str(plot_path.name)
                                if plot_path.parent == result.output_dir
                                else str(plot_path.relative_to(result.output_dir)),
                                "filename": plot_path.name,
                            }
                        )

            # Find data files for this domain
            domain_data_files = []
            domain_dir = result.domain_dirs.get(domain)
            if domain_dir and domain_dir.exists():
                for data_file in domain_dir.glob("*.json"):
                    domain_data_files.append(
                        {
                            "filename": data_file.name,
                            "path": str(data_file.relative_to(result.output_dir)),
                            "format": "JSON",
                        }
                    )

            # Build key findings from results
            key_findings = self._extract_key_findings(domain_results)

            domain_data: dict[str, Any] = {
                "domain_id": domain.value,
                "domain_name": domain.value.replace("_", " ").title(),
                "analyses_count": analyses_count,
                "plots_count": len(domain_plots),
                "data_files_count": len(domain_data_files),
                "key_findings": key_findings,
                "plots": domain_plots,
                "data_files": domain_data_files,
            }
            domains.append(domain_data)

        context["domains"] = domains

        # Build error information
        if result.errors:
            errors: list[dict[str, Any]] = []
            for error in result.errors:
                errors.append(
                    {
                        "domain": error.domain.value,
                        "analysis_name": error.function,
                        "error_message": error.error_message,
                    }
                )
            context["errors"] = errors

        return context

    def _extract_key_findings(self, domain_results: dict[str, Any]) -> list[str]:
        """Extract key findings from domain results for display.

        Args:
            domain_results: Dictionary of analysis function results.

        Returns:
            List of key finding strings.
        """
        findings = []
        for func_name, result in domain_results.items():
            # Extract function short name
            short_name = func_name.split(".")[-1].replace("_", " ").title()

            # Format result based on type
            if result is None:
                continue
            elif isinstance(result, int | float):
                if not (isinstance(result, float) and (result != result)):  # Check for NaN
                    findings.append(
                        f"{short_name}: {result:.4g}"
                        if isinstance(result, float)
                        else f"{short_name}: {result}"
                    )
            elif isinstance(result, dict) and len(result) <= 3:
                # Show small dicts inline
                items = [
                    f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in list(result.items())[:3]
                    if v is not None and not (isinstance(v, float) and v != v)
                ]
                if items:
                    findings.append(f"{short_name}: {', '.join(items)}")

        # Limit to most relevant findings
        return findings[:5]

    def _format_plots(self, plots: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Format plot information for templates.

        Args:
            plots: List of plot dictionaries.

        Returns:
            Formatted plot data.
        """
        formatted = []
        for plot in plots:
            formatted.append(
                {
                    "title": plot.get("title", "Untitled"),
                    "path": str(plot.get("path", "")),
                    "filename": Path(plot.get("path", "")).name,
                }
            )
        return formatted

    def _format_data_files(self, data_files: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Format data file information for templates.

        Args:
            data_files: List of data file dictionaries.

        Returns:
            Formatted data file data.
        """
        formatted = []
        for data_file in data_files:
            path = Path(data_file.get("path", ""))
            formatted.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "format": path.suffix.lstrip(".").upper() or "DATA",
                }
            )
        return formatted

    def _format_size(self, filepath: str | None) -> str:
        """Format file size in human-readable format.

        Args:
            filepath: Path to file.

        Returns:
            Formatted size string (e.g., "1.5 MB").
        """
        if not filepath:
            return "N/A"

        try:
            path = Path(filepath)
            if not path.exists():
                return "N/A"

            size_bytes = path.stat().st_size
            size_float = float(size_bytes)
            for unit in ["B", "KB", "MB", "GB"]:
                if size_float < 1024.0:
                    return f"{size_float:.1f} {unit}"
                size_float /= 1024.0
            return f"{size_float:.1f} TB"
        except Exception:
            return "N/A"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string (e.g., "1m 30s").
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _render_html(self, context: dict[str, Any]) -> str:
        """Render HTML index from template.

        Args:
            context: Template context.

        Returns:
            Rendered HTML string.

        Raises:
            FileNotFoundError: If HTML template file not found.

        Requirements:
        """
        template_path = self._template_dir / "index.html"
        if not template_path.exists():
            raise FileNotFoundError(f"HTML template not found: {template_path}")

        template = template_path.read_text()
        return self._engine.render(template, context)

    def _render_markdown(self, context: dict[str, Any]) -> str:
        """Render Markdown index from template.

        Args:
            context: Template context.

        Returns:
            Rendered Markdown string.

        Raises:
            FileNotFoundError: If Markdown template file not found.

        Requirements:
        """
        template_path = self._template_dir / "index.md"
        if not template_path.exists():
            raise FileNotFoundError(f"Markdown template not found: {template_path}")

        template = template_path.read_text()
        return self._engine.render(template, context)


__all__ = [
    "IndexGenerator",
    "TemplateEngine",
]
