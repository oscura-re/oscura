#!/usr/bin/env python3
"""Validate all agent markdown files conform to standard template.

This script validates that all agent definition files in .claude/agents/
follow the standardized structure defined in .claude/templates/agent-definition.md.

Validates:
- Frontmatter presence and completeness (all required fields)
- Frontmatter types (routing_keywords is list, model is valid)
- Section presence (all required sections exist)
- Section completeness (no empty sections, no TODO placeholders)
- Status values (only standard values in completion reports)
- Keyword uniqueness (no duplicate keywords across agents)
- Cross-references (all referenced agents/commands/docs exist)
- Examples (at least 2 concrete examples present)
- Line count (warns if agent exceeds reasonable length)

Exit codes:
    0: All agents valid
    1: Validation errors found
    2: Critical structural errors

Version: 2.0.0
Last Updated: 2026-01-22
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
COMMANDS_DIR = REPO_ROOT / ".claude" / "commands"
DOCS_DIR = REPO_ROOT / ".claude" / "docs"

# Required frontmatter fields
REQUIRED_FRONTMATTER = ["name", "description", "tools", "model", "routing_keywords"]

# Required sections in agent markdown
REQUIRED_SECTIONS = [
    "Core Capabilities",
    "Routing Keywords",
    "Triggers",
    "Workflow",
    "Definition of Done",
    "Anti-Patterns",
    "Completion Report Format",
]

# Valid status values (standardized across all agents)
VALID_STATUS_VALUES = ["complete", "in_progress", "blocked", "needs_review", "failed"]

# Valid model values
VALID_MODELS = ["sonnet", "opus"]

# Maximum reasonable agent length (lines)
MAX_AGENT_LINES = 500

# Minimum examples required
MIN_EXAMPLES = 2

# Known intentional keyword overlaps (document here to avoid false positives)
ALLOWED_KEYWORD_OVERLAPS = {
    # Format: frozenset({agent1, agent2}): [keyword1, keyword2, ...]
    frozenset({"code_assistant", "technical_writer"}): ["write"],  # Both create content
    frozenset({"code_reviewer", "knowledge_researcher"}): ["quality"],  # Both assess quality
}


def extract_frontmatter(content: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract and parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Tuple of (frontmatter_dict, errors_list)
    """
    errors = []

    if not content.startswith("---"):
        errors.append("Missing YAML frontmatter (must start with '---')")
        return None, errors

    try:
        # Find end of frontmatter
        end_idx = content.index("---", 3)
        frontmatter_text = content[3:end_idx].strip()

        if not HAS_YAML:
            errors.append("PyYAML not available, cannot parse frontmatter")
            return None, errors

        # Parse YAML
        frontmatter = yaml.safe_load(frontmatter_text)

        if not isinstance(frontmatter, dict):
            errors.append("Frontmatter is not a valid YAML dictionary")
            return None, errors

        return frontmatter, errors

    except ValueError:
        errors.append("Missing closing '---' for frontmatter")
        return None, errors
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in frontmatter: {e}")
        return None, errors


def validate_frontmatter(frontmatter: dict[str, Any] | None, file_name: str) -> list[str]:
    """Validate frontmatter contains all required fields with correct types.

    Args:
        frontmatter: Parsed frontmatter dictionary
        file_name: Name of the file being validated

    Returns:
        List of validation errors
    """
    errors: list[str] = []

    if frontmatter is None:
        return errors  # Already caught by extract_frontmatter

    # Check required fields
    for field in REQUIRED_FRONTMATTER:
        if field not in frontmatter:
            errors.append(f"Missing frontmatter field '{field}'")

    # Validate specific field types
    if "routing_keywords" in frontmatter:
        if not isinstance(frontmatter["routing_keywords"], list):
            errors.append("'routing_keywords' must be a list")
        elif len(frontmatter["routing_keywords"]) == 0:
            errors.append("'routing_keywords' list is empty")
        else:
            # Check each keyword is a string
            for i, keyword in enumerate(frontmatter["routing_keywords"]):
                if not isinstance(keyword, str):
                    errors.append(f"'routing_keywords[{i}]' must be a string, got {type(keyword)}")

    if "model" in frontmatter:
        if frontmatter["model"] not in VALID_MODELS:
            errors.append(f"'model' must be one of {VALID_MODELS}, got '{frontmatter['model']}'")

    # Validate name matches filename
    if "name" in frontmatter:
        expected_name = file_name.replace(".md", "")
        if frontmatter["name"] != expected_name:
            errors.append(
                f"'name' field ('{frontmatter['name']}') must match filename ('{expected_name}')"
            )

    return errors


def validate_sections(content: str, file_name: str) -> list[str]:
    """Validate that all required sections are present and not empty.

    Args:
        content: Full markdown file content
        file_name: Name of the file being validated

    Returns:
        List of validation errors
    """
    errors = []

    # Split content into sections
    lines = content.split("\n")

    for section in REQUIRED_SECTIONS:
        # Check for section header (## Section Name)
        if f"## {section}" not in content:
            errors.append(f"Missing section '## {section}'")
            continue

        # Check section is not empty (has content after header)
        section_idx = None
        for i, line in enumerate(lines):
            if line.strip() == f"## {section}":
                section_idx = i
                break

        if section_idx is not None:
            # Find next section or end of file
            next_section_idx = len(lines)
            for i in range(section_idx + 1, len(lines)):
                if lines[i].startswith("## "):
                    next_section_idx = i
                    break

            # Check if there's meaningful content between section headers
            section_content = "\n".join(lines[section_idx + 1 : next_section_idx]).strip()

            if not section_content:
                errors.append(f"Section '## {section}' is empty")
            elif "TODO" in section_content or "TBD" in section_content:
                errors.append(f"Section '## {section}' contains TODO/TBD placeholders")

    return errors


def validate_status_values(content: str, file_name: str) -> list[str]:
    """Validate that completion report uses standardized status values.

    Args:
        content: Full markdown file content
        file_name: Name of the file being validated

    Returns:
        List of validation errors
    """
    errors = []

    # Check for status field in completion report
    if '"status":' in content:
        # Look for the status line pattern
        status_pattern = r'"status":\s*"([^"]+)"'
        matches = re.findall(status_pattern, content)

        for match in matches:
            # Split by | to handle the format "complete|in_progress|blocked|..."
            statuses = [s.strip() for s in match.split("|")]

            # Check if all values are valid
            invalid_statuses = [s for s in statuses if s not in VALID_STATUS_VALUES]

            if invalid_statuses:
                errors.append(
                    f"Invalid status value(s) in completion report: {invalid_statuses}. "
                    f"Must be one of: {VALID_STATUS_VALUES}"
                )

    return errors


def validate_cross_references(content: str, file_name: str) -> list[str]:
    """Validate that all cross-references to agents/commands/docs exist.

    Args:
        content: Full markdown file content
        file_name: Name of the file being validated

    Returns:
        List of validation errors
    """
    errors = []

    # Find all agent references in backticks (e.g., `code_assistant`, `technical_writer`)
    # Only check underscore_names in backticks to avoid false positives
    agent_pattern = r"`([a-z_]+_[a-z_]+)`"
    agent_refs = re.findall(agent_pattern, content)

    # Get list of valid agent names
    valid_agents = {f.stem for f in AGENTS_DIR.glob("*.md") if f.name != file_name}  # Exclude self

    for ref in agent_refs:
        if ref not in valid_agents:
            # Check if it's likely an agent name (has underscore, not a common variable/field)
            common_vars = {
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
                "max_concurrent",
                "agent_outputs",
            }
            if ref not in common_vars:
                errors.append(f"Referenced agent '{ref}' does not exist")

    # Find all command references (e.g., `/ai`, `/review`, `/git`)
    # Only check commands that are documented to exist
    command_pattern = r"`/([a-z]+)`"
    command_refs = re.findall(command_pattern, content)

    # Get list of valid commands (if directory exists)
    if COMMANDS_DIR.exists():
        valid_commands = {f.stem for f in COMMANDS_DIR.glob("*.md")}

        for ref in command_refs:
            if ref not in valid_commands:
                errors.append(f"Referenced command '/{ref}' does not exist")

    # Find all doc references (e.g., .claude/docs/routing-concepts.md)
    doc_pattern = r"\.claude/docs/([a-z-]+\.md)"
    doc_refs = re.findall(doc_pattern, content)

    for ref in doc_refs:
        doc_path = DOCS_DIR / ref
        if not doc_path.exists():
            errors.append(f"Referenced document '.claude/docs/{ref}' does not exist")

    return errors


def validate_examples(content: str, file_name: str) -> list[str]:
    """Validate that agent has sufficient concrete examples.

    Args:
        content: Full markdown file content
        file_name: Name of the file being validated

    Returns:
        List of validation errors
    """
    errors = []

    # Count example sections (### Example N:, **Example N:**, etc.)
    example_patterns = [
        r"###\s+Example\s+\d+",  # ### Example 1
        r"\*\*Example\s+\d+\*\*",  # **Example 1**
        r"##\s+Example\s+\d+",  # ## Example 1
    ]

    example_count = 0
    for pattern in example_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        example_count += len(matches)

    if example_count < MIN_EXAMPLES:
        errors.append(
            f"Insufficient examples ({example_count}/{MIN_EXAMPLES}). "
            "Add at least 2 concrete examples."
        )

    return errors


def validate_line_count(content: str, file_name: str) -> list[str]:
    """Validate that agent does not exceed reasonable length.

    Args:
        content: Full markdown file content
        file_name: Name of the file being validated

    Returns:
        List of validation warnings
    """
    warnings = []

    line_count = len(content.split("\n"))

    if line_count > MAX_AGENT_LINES:
        warnings.append(
            f"Agent is very long ({line_count} lines, max recommended: {MAX_AGENT_LINES}). "
            "Consider splitting or refactoring."
        )

    return warnings


def check_keyword_uniqueness(all_agents_data: dict[str, dict[str, Any]]) -> list[str]:
    """Check for duplicate routing keywords across agents.

    Args:
        all_agents_data: Dict mapping agent_name -> frontmatter

    Returns:
        List of validation errors
    """
    errors = []

    # Build keyword -> agents mapping
    keyword_to_agents: dict[str, list[str]] = defaultdict(list)

    for agent_name, frontmatter in all_agents_data.items():
        if frontmatter and "routing_keywords" in frontmatter:
            for keyword in frontmatter["routing_keywords"]:
                keyword_to_agents[keyword].append(agent_name)

    # Check for duplicates
    for keyword, agents in keyword_to_agents.items():
        if len(agents) > 1:
            # Check if this overlap is intentionally allowed
            agents_set = frozenset(agents)
            is_allowed = False

            for allowed_set, allowed_keywords in ALLOWED_KEYWORD_OVERLAPS.items():
                if agents_set == allowed_set and keyword in allowed_keywords:
                    is_allowed = True
                    break

            if not is_allowed:
                errors.append(
                    f"Duplicate routing keyword '{keyword}' found in agents: {', '.join(sorted(agents))}. "
                    "Either remove duplicates or document intentional overlap in ALLOWED_KEYWORD_OVERLAPS."
                )

    return errors


def validate_agent_file(agent_path: Path) -> tuple[list[str], dict[str, Any] | None]:
    """Validate single agent file against template.

    Args:
        agent_path: Path to agent markdown file

    Returns:
        Tuple of (errors_list, frontmatter_dict)
    """
    errors = []
    warnings = []
    file_name = agent_path.name

    try:
        content = agent_path.read_text()
    except Exception as e:
        return [f"Failed to read file: {e}"], None

    # Extract and validate frontmatter
    frontmatter, fm_errors = extract_frontmatter(content)
    errors.extend(fm_errors)

    if frontmatter is not None:
        errors.extend(validate_frontmatter(frontmatter, file_name))

    # Validate sections
    errors.extend(validate_sections(content, file_name))

    # Validate status values
    errors.extend(validate_status_values(content, file_name))

    # Validate cross-references
    errors.extend(validate_cross_references(content, file_name))

    # Validate examples
    errors.extend(validate_examples(content, file_name))

    # Validate line count (warnings only)
    warnings.extend(validate_line_count(content, file_name))

    # Prefix all errors with file name
    all_errors = [f"{file_name}: {error}" for error in errors]
    all_warnings = [f"{file_name}: {warning}" for warning in warnings]

    return all_errors + all_warnings, frontmatter


def main() -> int:
    """Main validation function.

    Returns:
        Exit code (0 for success, 1 for errors, 2 for critical errors)
    """
    print("\n" + "=" * 70)
    print("  AGENT VALIDATION")
    print("=" * 70 + "\n")

    # Get agents directory
    if not AGENTS_DIR.exists():
        print(f"❌ CRITICAL: Agents directory not found: {AGENTS_DIR}", file=sys.stderr)
        return 2

    # Collect all errors
    all_errors = []
    all_frontmatter = {}

    # Validate each agent file
    agent_files = sorted(AGENTS_DIR.glob("*.md"))

    if not agent_files:
        print(f"⚠ WARNING: No agent files found in {AGENTS_DIR}", file=sys.stderr)
        return 0

    print(f"Validating {len(agent_files)} agent file(s)...\n")

    for agent_file in agent_files:
        errors, frontmatter = validate_agent_file(agent_file)
        all_errors.extend(errors)

        if frontmatter:
            all_frontmatter[agent_file.stem] = frontmatter

    # Check keyword uniqueness across all agents
    keyword_errors = check_keyword_uniqueness(all_frontmatter)
    all_errors.extend(keyword_errors)

    # Report results
    print("=" * 70)
    if all_errors:
        # Separate errors from warnings
        errors = [e for e in all_errors if "very long" not in e]
        warnings = [e for e in all_errors if "very long" in e]

        if errors:
            print(f"\n❌ VALIDATION FAILED with {len(errors)} error(s):\n")
            for error in errors:
                print(f"  - {error}")

        if warnings:
            print(f"\n⚠ {len(warnings)} warning(s):\n")
            for warning in warnings:
                print(f"  - {warning}")

        print("\n" + "=" * 70)
        return 1 if errors else 0
    else:
        print("\n✅ ALL CHECKS PASSED")
        print(f"  - {len(agent_files)} agent file(s) validated")
        print("  - All required sections present")
        print("  - All cross-references valid")
        print("  - No duplicate keywords")
        print("\n" + "=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
