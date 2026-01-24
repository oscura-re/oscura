#!/usr/bin/env python3
"""Validate documentation links and cross-references.

This script validates that all documentation files have valid internal links,
anchors, file references, and cross-references.

Validates:
- Internal links resolve to existing files
- Markdown anchors resolve to actual headings
- File references exist
- Config references match actual config.yaml structure
- Agent references match actual agent files
- Command references match actual command files
- Code blocks have language hints
- No references to deleted files
- Cross-reference symmetry (bidirectional links)

Exit codes:
    0: All documentation valid
    1: Broken links or references found
    2: Critical documentation errors

Version: 1.0.0
Created: 2026-01-22
"""

import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
CLAUDE_DIR = REPO_ROOT / ".claude"
AGENTS_DIR = CLAUDE_DIR / "agents"
COMMANDS_DIR = CLAUDE_DIR / "commands"
DOCS_DIR = CLAUDE_DIR / "docs"
CONFIG_FILE = CLAUDE_DIR / "config.yaml"

# Known deleted files that shouldn't be referenced
DELETED_FILES = [
    "research.md",
    "review.md",
    "orchestration-config.yaml",
    "agent-config.yaml",
]


def load_yaml_file(file_path: Path) -> dict[str, Any] | None:
    """Load YAML file if available.

    Args:
        file_path: Path to YAML file

    Returns:
        Dict if successful, None otherwise
    """
    if not HAS_YAML or not file_path.exists():
        return None

    try:
        with open(file_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def extract_headings(content: str) -> set[str]:
    """Extract all markdown headings from content.

    Args:
        content: Markdown file content

    Returns:
        Set of heading anchor IDs (lowercased, spaces->hyphens)
    """
    headings = set()

    # Find all markdown headings (## Heading Text)
    heading_pattern = r"^#{1,6}\s+(.+)$"
    matches = re.findall(heading_pattern, content, re.MULTILINE)

    for heading in matches:
        # Convert to anchor format (lowercase, spaces to hyphens, remove special chars)
        anchor = heading.lower()
        anchor = re.sub(r"[^\w\s-]", "", anchor)
        anchor = re.sub(r"[\s]+", "-", anchor)
        headings.add(anchor)

    return headings


def validate_internal_links(
    file_path: Path, content: str, all_markdown_files: set[Path]
) -> list[str]:
    """Validate internal markdown links.

    Args:
        file_path: Path to current file
        content: File content
        all_markdown_files: Set of all markdown file paths

    Returns:
        List of validation errors
    """
    errors = []

    # Find all markdown links [text](url), but not in code blocks
    # Skip links that are part of ` ` inline code or inside ``` ``` blocks
    link_pattern = r"(?<!`)\[([^\]]+)\]\(([^\)]+)\)(?!`)"
    matches = re.findall(link_pattern, content)

    for text, url in matches:
        # Skip external URLs
        if url.startswith(("http://", "https://", "mailto:", "#")):
            continue

        # Handle anchor-only links
        if url.startswith("#"):
            anchor = url[1:]
            headings = extract_headings(content)
            if anchor not in headings:
                errors.append(f"Broken anchor link: {url} (heading '#{anchor}' not found)")
            continue

        # Split URL into path and anchor
        if "#" in url:
            url_path, anchor = url.split("#", 1)
        else:
            url_path, anchor = url, None

        # Resolve relative path
        if url_path:
            target_path = (file_path.parent / url_path).resolve()

            # Check if file exists
            if not target_path.exists():
                errors.append(f"Broken link: [{text}]({url}) -> {target_path} (file not found)")
                continue

            # Check anchor if present
            if anchor:
                try:
                    target_content = target_path.read_text()
                    target_headings = extract_headings(target_content)
                    if anchor not in target_headings:
                        errors.append(
                            f"Broken anchor: [{text}]({url}) (heading '#{anchor}' not found in {target_path.name})"
                        )
                except Exception as e:
                    errors.append(f"Error reading target file {target_path}: {e}")

    return errors


def validate_file_references(file_path: Path, content: str) -> list[str]:
    """Validate file path references in documentation.

    Args:
        file_path: Path to current file
        content: File content

    Returns:
        List of validation errors
    """
    errors = []

    # Find file path references (e.g., `.claude/config.yaml`, `src/oscura/core.py`)
    file_ref_pattern = r"`([\.\/a-zA-Z0-9_\-]+\.[a-z]+)`"
    matches = re.findall(file_ref_pattern, content)

    for ref in matches:
        # Skip common code snippets that aren't file paths
        if ref in ["self.value", "some.module", "example.com", "test.py"]:
            continue

        # Skip example/placeholder paths
        if "path/to/" in ref or "example" in ref or "/foo/" in ref:
            continue

        # Only check .claude references (most important for infrastructure)
        if not ref.startswith(".claude/"):
            continue

        # Resolve path relative to repo root
        target_path = REPO_ROOT / ref

        # Check if file exists
        if not target_path.exists():
            # Check if it's a known deleted file
            if any(deleted in ref for deleted in DELETED_FILES):
                errors.append(f"Reference to deleted file: {ref} (should be removed or updated)")
            else:
                errors.append(f"Referenced file not found: {ref}")

    return errors


def validate_config_references(file_path: Path, content: str) -> list[str]:
    """Validate config.yaml path references.

    Args:
        file_path: Path to current file
        content: File content

    Returns:
        List of validation errors
    """
    errors = []

    # Load config if available
    config = load_yaml_file(CONFIG_FILE)
    if not config:
        return []  # Can't validate without config

    # Find config path references (e.g., config.yaml:orchestration.agents.max_concurrent)
    config_ref_pattern = r"config\.yaml:([a-z_\.]+)"
    matches = re.findall(config_ref_pattern, content)

    for ref in matches:
        # Parse path (e.g., "orchestration.agents.max_concurrent")
        path_parts = ref.split(".")

        # Navigate config structure
        current = config
        valid = True
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                valid = False
                break

        if not valid:
            errors.append(f"Invalid config reference: config.yaml:{ref} (path not found)")

    return errors


def validate_agent_references(file_path: Path, content: str) -> list[str]:
    """Validate agent name references.

    Args:
        file_path: Path to current file
        content: File content

    Returns:
        List of validation errors
    """
    errors = []

    # Get list of valid agents
    valid_agents = {f.stem for f in AGENTS_DIR.glob("*.md")}

    # Find agent references (e.g., `code_assistant`, `technical_writer`)
    # Match underscore_names in backticks or after keywords
    agent_ref_patterns = [
        r"`([a-z_]+)`",  # Backtick references
        r"agent:\s*([a-z_]+)",  # agent: field in JSON/YAML
        r"next_agent:\s*\"([a-z_]+)\"",  # next_agent field
    ]

    found_refs = set()
    for pattern in agent_ref_patterns:
        matches = re.findall(pattern, content)
        found_refs.update(matches)

    # Check each reference
    for ref in found_refs:
        # Skip common non-agent words and JSON fields
        common_words = {
            "string",
            "int",
            "bool",
            "dict",
            "list",
            "none",
            "true",
            "false",
            "config",
            "status",
            "error",
            "warning",
            "info",
            "debug",
            "next_agent",
            "task_id",
            "agent_id",
            "user_input",
            "file_path",
            "in_progress",
            "needs_review",
            "started_at",
            "completed_at",
            "handoff_context",
            "commits_created",
            "push_status",
            "validation_passed",
            "routing_keywords",
            "routing_decision",
            "config_version",
            "max_concurrent",
            "agent_outputs",
            "polling_interval_seconds",
            "max_batch_size",
            "checkpoint_threshold",
            "warning_threshold",
            "critical_threshold",
            "original_text",
            "modified_text",
            "file_contents",
            "potential_gaps",
            "files_with_matches",
            "partial_ratio",
            "snake_case",
            "locks_stale_minutes",
            "activity_check_hours",
            "node_modules",
            "check_subagent_stop",
            "schema_version",
            "max_age_days",
            "large_json_size_bytes",
            "check_stop",
            "data_file",
            "__pycache__",
            "cleanup_stale_agents",
            "recent_window_minutes",
            "min_compatible_version",
            "disk_space_warning_percent",
            "pre_compact_cleanup",
            "stale_threshold_hours",
            "max_stale_hours",
            "_meta",
            "use_json",
            "old_report_days",
            "health_check",
            "disk_space_critical_percent",
            "output_size_threshold_bytes",
            "last_updated",
            "test_auto_clock_recovery_fft",
            "max_samples",
            "max_file_size_mb",
        }
        if ref in common_words or len(ref) < 4:
            continue

        # Check if it's a valid agent
        if ref not in valid_agents and "_" in ref:
            # Likely meant to be an agent reference
            errors.append(f"Referenced agent '{ref}' not found in .claude/agents/")

    return errors


def validate_command_references(file_path: Path, content: str) -> list[str]:
    """Validate slash command references.

    Args:
        file_path: Path to current file
        content: File content

    Returns:
        List of validation errors
    """
    errors = []

    # Get list of valid commands
    valid_commands = {f.stem for f in COMMANDS_DIR.glob("*.md")}

    # Claude Code built-in commands (don't need .md files)
    builtin_commands = {
        "compact",
        "rewind",
        "feedback",
        "rename",
        "stats",
        "undo",
        "redo",
        "cancel",
        "interrupt",
    }

    # Find command references (e.g., `/ai`, `/review`, `/git`)
    command_pattern = r"`/([a-z]+)`"
    matches = re.findall(command_pattern, content)

    for cmd in matches:
        if cmd not in valid_commands and cmd not in builtin_commands:
            errors.append(f"Referenced command '/{cmd}' not found in .claude/commands/")

    return errors


def validate_code_blocks(file_path: Path, content: str) -> list[str]:
    """Validate code blocks have language hints.

    Args:
        file_path: Path to current file
        content: File content

    Returns:
        List of validation errors
    """
    errors = []

    # Track code blocks with fence depth (``` vs ```` for nested blocks)
    lines = content.split("\n")
    fence_stack: list[tuple[int, bool]] = []  # Stack of (fence_length, has_language)
    bad_blocks = 0

    for line in lines:
        stripped = line.strip()

        # Check for code fence
        fence_match = re.match(r"^(`{3,})\s*(\w*).*$", stripped)
        if fence_match:
            fence_chars = fence_match.group(1)
            fence_length = len(fence_chars)
            language = fence_match.group(2)

            if not fence_stack:
                # Opening a new top-level block
                if not language:
                    bad_blocks += 1
                fence_stack.append((fence_length, bool(language)))
            elif fence_length == fence_stack[-1][0]:
                # Closing the current block
                fence_stack.pop()
            elif fence_length != fence_stack[-1][0]:
                # Nested block or different fence length - push to stack
                if not language:
                    # Only count bad blocks at top level
                    if len(fence_stack) == 1:
                        # This is a nested block in a markdown/etc example - that's OK
                        pass
                fence_stack.append((fence_length, bool(language)))

    if bad_blocks > 0:
        errors.append(
            f"Found {bad_blocks} code block(s) without language hint. Use ```python, ```bash, etc."
        )

    return errors


def validate_cross_reference_symmetry(all_files: dict[Path, str]) -> list[str]:
    """Validate bidirectional cross-references.

    If doc A links to doc B with text "see X", check if doc B links back to A.

    Args:
        all_files: Dict mapping file_path -> content

    Returns:
        List of validation warnings (not errors)
    """
    warnings = []

    # Build link graph: file -> set of linked files
    link_graph: dict[Path, set[Path]] = {}

    for file_path, content in all_files.items():
        links = set()

        # Find all markdown links
        link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        matches = re.findall(link_pattern, content)

        for _text, url in matches:
            # Skip external URLs and anchors
            if url.startswith(("http://", "https://", "mailto:", "#")):
                continue

            # Resolve relative path
            if "#" in url:
                url = url.split("#")[0]

            if url:
                target_path = (file_path.parent / url).resolve()
                if target_path.exists():
                    links.add(target_path)

        link_graph[file_path] = links

    # Check for asymmetric links (A->B but not B->A)
    for file_a, links_from_a in link_graph.items():
        for file_b in links_from_a:
            links_from_b = link_graph.get(file_b, set())
            if file_a not in links_from_b:
                warnings.append(
                    f"Asymmetric link: {file_a.name} -> {file_b.name} (consider adding backlink)"
                )

    return warnings


def validate_markdown_file(file_path: Path, all_markdown_files: set[Path]) -> list[str]:
    """Validate single markdown file.

    Args:
        file_path: Path to markdown file
        all_markdown_files: Set of all markdown file paths

    Returns:
        List of validation errors
    """
    errors = []

    try:
        content = file_path.read_text()
    except Exception as e:
        return [f"Failed to read file: {e}"]

    # Run all validations
    errors.extend(validate_internal_links(file_path, content, all_markdown_files))
    errors.extend(validate_file_references(file_path, content))
    errors.extend(validate_config_references(file_path, content))
    errors.extend(validate_agent_references(file_path, content))
    errors.extend(validate_command_references(file_path, content))
    errors.extend(validate_code_blocks(file_path, content))

    # Prefix all errors with filename
    return [f"{file_path.name}: {error}" for error in errors]


def main() -> int:
    """Main validation function.

    Returns:
        Exit code (0 for success, 1 for errors, 2 for critical)
    """
    print("\n" + "=" * 70)
    print("  DOCUMENTATION VALIDATION")
    print("=" * 70 + "\n")

    # Check critical directories exist
    if not CLAUDE_DIR.exists():
        print(f"❌ CRITICAL: .claude directory not found: {CLAUDE_DIR}", file=sys.stderr)
        return 2

    # Collect all markdown files
    markdown_files: set[Path] = set()
    for pattern in ["agents/*.md", "commands/*.md", "docs/*.md"]:
        markdown_files.update(CLAUDE_DIR.glob(pattern))

    # Also include root markdown files
    markdown_files.update(REPO_ROOT.glob("*.md"))

    if not markdown_files:
        print("⚠ WARNING: No markdown files found", file=sys.stderr)
        return 0

    print(f"Validating {len(markdown_files)} markdown file(s)...\n")

    # Validate each file
    all_errors = []
    all_content = {}

    for md_file in sorted(markdown_files):
        try:
            content = md_file.read_text()
            all_content[md_file] = content

            errors = validate_markdown_file(md_file, markdown_files)
            all_errors.extend(errors)
        except Exception as e:
            all_errors.append(f"{md_file.name}: Failed to validate: {e}")

    # Check cross-reference symmetry (warnings only)
    warnings = validate_cross_reference_symmetry(all_content)

    # Report results
    print("=" * 70)

    if all_errors:
        print(f"\n❌ VALIDATION FAILED with {len(all_errors)} error(s):\n")
        for error in all_errors:
            print(f"  - {error}")
        print("\n" + "=" * 70)
        return 1

    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):\n")
        for warning in warnings:
            print(f"  - {warning}")

    print("\n✅ ALL CHECKS PASSED")
    print(f"  - {len(markdown_files)} markdown file(s) validated")
    print("  - All internal links valid")
    print("  - All file references exist")
    print("  - No deleted file references")
    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
