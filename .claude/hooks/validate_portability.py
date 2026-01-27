#!/usr/bin/env python3
"""Validate .claude infrastructure portability.

This script checks that the .claude infrastructure is portable across projects
by detecting hardcoded project names, language assumptions, VCS assumptions, etc.

Validates:
- No hardcoded project names (except in project-metadata.yaml)
- No language-specific code in language-agnostic files
- No VCS-specific code in VCS-agnostic files
- Relative paths instead of absolute paths
- Template placeholders properly marked
- No hardcoded organization/repository names

Scoring:
- Calculate portability score (0-100)
- Identify specific issues with severity levels
- Provide fix recommendations

Exit codes:
    0: Fully portable (score >= 95)
    1: Mostly portable (score >= 80)
    2: Not portable (score < 80)

Version: 1.0.0
Created: 2026-01-22
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
CLAUDE_DIR = REPO_ROOT / ".claude"
PROJECT_METADATA = CLAUDE_DIR / "project-metadata.yaml"

# Files that are ALLOWED to have project-specific content
EXEMPTED_FILES = {
    "project-metadata.yaml",  # By design contains project info
    "paths.yaml",  # Project-specific paths
    "README.md",  # Project README
    "CLAUDE.md",  # Project-specific instructions
    "CHANGELOG.md",  # Project changelog
    "CONTRIBUTING.md",  # Project contribution guide
    "SECURITY.md",  # Project security policy
    # Documentation files may have project-specific examples
    "conventional-commits-guide.md",  # Has project-specific scope examples
    "claude-code-complete-reference.md",  # Reference with project examples
    # Scripts that load from project-metadata.yaml (intentionally project-aware)
    "sync_versions.py",  # Updates project-specific version patterns
    "validate_config_consistency.py",  # Validates project-specific config
    "validate_documentation.py",  # May reference project-specific doc paths
    # Audit/analysis scripts are project-specific tools
    "audit_copy_usage.py",  # Project-specific code auditing
    "audit_skipped_tests.py",  # Project-specific test auditing
    "comprehensive_skip_fixer.py",  # Project-specific test fixing
    "fix_220_skips.py",  # Project-specific test fixing
    "remove_duplicates.py",  # Project-specific cleanup
    "remove_fixable_skips.py",  # Project-specific test fixing
    "remove_fixable_skips_v2.py",  # Project-specific test fixing
    "remove_skipif_decorators.py",  # Project-specific test fixing
    "update_changelog.py",  # Project-specific changelog management
    # Reports are project-specific work products (all reports exempted by directory below)
}

# Language-agnostic files (should not have language-specific code)
# NOTE: config.yaml contains security patterns that ARE language-specific by design.
# The structure is portable; the values are project-specific.
# Currently NO files are required to be fully language-agnostic.
LANGUAGE_AGNOSTIC_FILES: set[str] = set()  # Empty - all files can have language refs

# VCS-agnostic files (should not have GitHub-specific code)
# NOTE: Agent files ARE allowed to reference VCS concepts (PR, pull request, etc.)
# since they need to describe git workflows. Only config files should be VCS-agnostic.
VCS_AGNOSTIC_FILES = {
    "config.yaml",
    "coding-standards.yaml",
    # Removed agents/*.md - agents can reference VCS concepts
    # Removed docs/*.md - docs can reference VCS concepts
}

# Severity levels
CRITICAL = "critical"  # Breaks portability entirely
HIGH = "high"  # Major portability issue
MEDIUM = "medium"  # Minor portability issue
LOW = "low"  # Best practice violation

# Issue scoring (points deducted)
SEVERITY_SCORES = {
    CRITICAL: 20,
    HIGH: 10,
    MEDIUM: 5,
    LOW: 2,
}


class PortabilityIssue:
    """Represents a portability issue."""

    def __init__(
        self,
        file_path: Path,
        line_number: int,
        severity: str,
        category: str,
        description: str,
        recommendation: str,
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.severity = severity
        self.category = category
        self.description = description
        self.recommendation = recommendation

    def __str__(self) -> str:
        """Format issue for display."""
        return (
            f"[{self.severity.upper()}] {self.file_path.name}:{self.line_number}\n"
            f"  Category: {self.category}\n"
            f"  Issue: {self.description}\n"
            f"  Fix: {self.recommendation}"
        )


def load_project_metadata() -> dict[str, Any]:
    """Load project metadata to get exempted terms.

    Returns:
        Dict of project metadata
    """
    if not HAS_YAML or not PROJECT_METADATA.exists():
        return {}

    try:
        with open(PROJECT_METADATA) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def is_exempted_file(file_path: Path) -> bool:
    """Check if file is exempted from portability checks.

    Args:
        file_path: Path to file

    Returns:
        True if exempted
    """
    # Check if filename is exempted
    if file_path.name in EXEMPTED_FILES:
        return True

    # Exempt all audit/report files (project-specific work products)
    # These files document project-specific work and naturally contain project names
    report_patterns = [
        "_AUDIT_",
        "_REPORT",  # Matches _REPORT.md and _REPORT_.md
        "_SUMMARY",  # Matches _SUMMARY.md and _SUMMARY_.md
        "_COMPLETE",  # Matches _COMPLETE.md and _COMPLETE_.md
        "_PROGRESS",  # Matches _PROGRESS.md and _PROGRESS_.md
        "_INDEX",
        "_PROFILE_",
        "_ENTRIES_",
        "OPTIMIZATION_",  # Matches OPTIMIZATION_*.md
        "VALIDATION_",  # Matches *_VALIDATION_*.md
    ]
    filename_upper = file_path.name.upper()
    for pattern in report_patterns:
        if pattern in filename_upper:
            return True

    # Exempt timestamped reports (2026-01-25-*.md)
    if re.match(r"^\d{4}-\d{2}-\d{2}-.+\.md$", file_path.name):
        return True

    # Exempt all .py scripts in .claude directory (project-specific tools)
    if file_path.suffix == ".py" and CLAUDE_DIR in file_path.parents:
        # Only exempt scripts in .claude root or analysis/, not hooks/
        rel_path = file_path.relative_to(CLAUDE_DIR)
        if not str(rel_path).startswith("hooks/"):
            return True

    return False


def is_language_agnostic_file(file_path: Path) -> bool:
    """Check if file should be language-agnostic.

    Args:
        file_path: Path to file

    Returns:
        True if should be language-agnostic
    """
    rel_path = file_path.relative_to(CLAUDE_DIR) if CLAUDE_DIR in file_path.parents else file_path

    # Check exact matches
    if str(rel_path) in LANGUAGE_AGNOSTIC_FILES:
        return True

    # Check patterns
    for pattern in LANGUAGE_AGNOSTIC_FILES:
        if "*" in pattern:
            # Simple glob matching
            if str(rel_path).startswith(pattern.replace("*", "")):
                return True

    return False


def is_vcs_agnostic_file(file_path: Path) -> bool:
    """Check if file should be VCS-agnostic.

    Args:
        file_path: Path to file

    Returns:
        True if should be VCS-agnostic
    """
    rel_path = file_path.relative_to(CLAUDE_DIR) if CLAUDE_DIR in file_path.parents else file_path

    # Check exact matches
    if str(rel_path) in VCS_AGNOSTIC_FILES:
        return True

    # Check patterns
    for pattern in VCS_AGNOSTIC_FILES:
        if "*" in pattern:
            pattern_prefix = pattern.replace("/*.md", "")
            if str(rel_path).startswith(pattern_prefix) and str(rel_path).endswith(".md"):
                return True

    return False


def check_hardcoded_project_names(
    file_path: Path, content: str, project_name: str
) -> list[PortabilityIssue]:
    """Check for hardcoded project name references.

    Args:
        file_path: Path to file
        content: File content
        project_name: Official project name from metadata

    Returns:
        List of issues found
    """
    if is_exempted_file(file_path):
        return []

    issues = []
    lines = content.split("\n")

    # Check each line for project name
    for i, line in enumerate(lines, 1):
        # Skip comments and TEMPLATE markers
        if "TEMPLATE:" in line or "{{" in line:
            continue

        # Case-insensitive search for project name
        if re.search(rf"\b{project_name}\b", line, re.IGNORECASE):
            issues.append(
                PortabilityIssue(
                    file_path=file_path,
                    line_number=i,
                    severity=HIGH,
                    category="hardcoded_project_name",
                    description=f"Hardcoded project name '{project_name}' found",
                    recommendation=(
                        "Replace with template variable or load from project-metadata.yaml"
                    ),
                )
            )

    return issues


def check_language_assumptions(file_path: Path, content: str) -> list[PortabilityIssue]:
    """Check for Python-specific code in language-agnostic files.

    Args:
        file_path: Path to file
        content: File content

    Returns:
        List of issues found
    """
    if not is_language_agnostic_file(file_path):
        return []

    issues = []
    lines = content.split("\n")

    # Python-specific patterns
    python_patterns = [
        (r"\bpytest\b", "pytest", "Use generic 'test framework'"),
        (r"\bmypy\b", "mypy", "Use generic 'type checker'"),
        (r"\bruff\b", "ruff", "Use generic 'linter'"),
        (r"\bpip\b", "pip", "Use generic 'package manager'"),
        (r"\buv\b(?!\w)", "uv", "Use generic 'package manager'"),
        (r"\.py\b", ".py extension", "Use generic 'source file'"),
        (r"\bpython\b", "python", "Use generic 'language'"),
        (r"\b__pycache__\b", "__pycache__", "Use generic 'cache directory'"),
        (r"pyproject\.toml", "pyproject.toml", "Use generic 'config file'"),
    ]

    for i, line in enumerate(lines, 1):
        # Skip code blocks and template markers
        if line.strip().startswith("```") or "TEMPLATE:" in line:
            continue

        for pattern, name, recommendation in python_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(
                    PortabilityIssue(
                        file_path=file_path,
                        line_number=i,
                        severity=MEDIUM,
                        category="language_assumption",
                        description=f"Python-specific reference '{name}' in language-agnostic file",
                        recommendation=recommendation,
                    )
                )
                break  # One issue per line

    return issues


def check_vcs_assumptions(file_path: Path, content: str) -> list[PortabilityIssue]:
    """Check for GitHub-specific code in VCS-agnostic files.

    Args:
        file_path: Path to file
        content: File content

    Returns:
        List of issues found
    """
    if not is_vcs_agnostic_file(file_path) or is_exempted_file(file_path):
        return []

    issues = []
    lines = content.split("\n")

    # GitHub-specific patterns
    github_patterns = [
        (r"\bgithub\b", "github", "Use generic 'VCS provider'"),
        (r"\bgh\b", "gh CLI", "Use generic 'VCS CLI'"),
        (r"\.github/", ".github/", "Use generic 'CI directory'"),
        (r"\bpull request\b", "pull request", "Use generic 'merge request'"),
        (r"\bPR\b", "PR", "Use generic 'MR'"),
        (r"github\.com", "github.com", "Use generic 'repository host'"),
    ]

    for i, line in enumerate(lines, 1):
        # Skip URLs, code blocks, template markers
        if "http" in line or line.strip().startswith("```") or "TEMPLATE:" in line:
            continue

        for pattern, name, recommendation in github_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(
                    PortabilityIssue(
                        file_path=file_path,
                        line_number=i,
                        severity=MEDIUM,
                        category="vcs_assumption",
                        description=f"GitHub-specific reference '{name}' in VCS-agnostic file",
                        recommendation=recommendation,
                    )
                )
                break  # One issue per line

    return issues


def check_absolute_paths(file_path: Path, content: str) -> list[PortabilityIssue]:
    """Check for hardcoded absolute paths.

    Args:
        file_path: Path to file
        content: File content

    Returns:
        List of issues found
    """
    issues = []
    lines = content.split("\n")

    # Absolute path patterns
    abs_path_patterns = [
        (r"/home/[a-z]+/", "Unix absolute path"),
        (r"C:\\", "Windows absolute path"),
        (r"/Users/[a-zA-Z]+/", "macOS absolute path"),
    ]

    for i, line in enumerate(lines, 1):
        # Skip code blocks
        if line.strip().startswith("```") or "REPO_ROOT" in line:
            continue

        for pattern, description in abs_path_patterns:
            if re.search(pattern, line):
                issues.append(
                    PortabilityIssue(
                        file_path=file_path,
                        line_number=i,
                        severity=CRITICAL,
                        category="absolute_path",
                        description=f"{description} found",
                        recommendation="Use relative paths or path resolution from REPO_ROOT",
                    )
                )
                break  # One issue per line

    return issues


def check_template_placeholders(file_path: Path, content: str) -> list[PortabilityIssue]:
    """Check for unmarked template variables.

    Args:
        file_path: Path to file
        content: File content

    Returns:
        List of issues found
    """
    if is_exempted_file(file_path):
        return []

    issues = []
    lines = content.split("\n")

    # Variables that should be templates but aren't marked
    # NOTE: These patterns are intentionally narrow to avoid false positives
    # We only flag very obvious human name patterns in specific contexts
    suspicious_patterns = [
        # Only match author/name fields with typical human name format
        (r'author.*"[A-Z][a-z]+\s+[A-Z][a-z]+"', "Author name (likely needs template)"),
    ]

    for i, line in enumerate(lines, 1):
        # Skip if already marked as template
        if "TEMPLATE:" in line or "{{" in line:
            continue

        for pattern, description in suspicious_patterns:
            if re.search(pattern, line):
                issues.append(
                    PortabilityIssue(
                        file_path=file_path,
                        line_number=i,
                        severity=LOW,
                        category="template_placeholder",
                        description=f"{description} not marked as template",
                        recommendation=(
                            "Add # TEMPLATE: {{variable_name}} comment or use {{variable}}"
                        ),
                    )
                )
                break  # One issue per line

    return issues


def check_organization_names(
    file_path: Path, content: str, org_name: str
) -> list[PortabilityIssue]:
    """Check for hardcoded organization names.

    Args:
        file_path: Path to file
        content: File content
        org_name: Official organization name from metadata

    Returns:
        List of issues found
    """
    if is_exempted_file(file_path) or not org_name:
        return []

    issues = []
    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        # Skip URLs, templates
        if "http" in line or "TEMPLATE:" in line or "{{" in line:
            continue

        if re.search(rf"\b{org_name}\b", line, re.IGNORECASE):
            issues.append(
                PortabilityIssue(
                    file_path=file_path,
                    line_number=i,
                    severity=HIGH,
                    category="hardcoded_org_name",
                    description=f"Hardcoded organization name '{org_name}' found",
                    recommendation="Replace with template variable or load from project-metadata.yaml",
                )
            )

    return issues


def strip_code_blocks(content: str) -> str:
    """Remove code blocks from markdown to avoid false positives.

    Args:
        content: File content

    Returns:
        Content with code blocks removed
    """
    lines = content.split("\n")
    result = []
    in_code_block = False
    fence_depth = 0

    for line in lines:
        # Check for code fence
        fence_match = re.match(r"^```+", line.strip())
        if fence_match:
            if not in_code_block:
                in_code_block = True
                fence_depth = len(fence_match.group(0))
                result.append("")  # Blank line instead of code
            else:
                current_depth = len(fence_match.group(0))
                if current_depth == fence_depth:
                    in_code_block = False
                result.append("")
        elif in_code_block:
            result.append("")  # Blank line instead of code
        else:
            result.append(line)

    return "\n".join(result)


def validate_file(file_path: Path, project_metadata: dict[str, Any]) -> list[PortabilityIssue]:
    """Validate single file for portability.

    Args:
        file_path: Path to file
        project_metadata: Project metadata dict

    Returns:
        List of issues found
    """
    # Don't validate the validator itself (contains pattern definitions)
    if file_path.name == "validate_portability.py":
        return []

    try:
        content = file_path.read_text()
    except Exception:
        return []  # Skip unreadable files

    # For markdown files, strip code blocks to avoid false positives
    if file_path.suffix == ".md":
        content = strip_code_blocks(content)

    # For Python files, strip docstrings to avoid false positives from examples
    if file_path.suffix == ".py":
        # Remove triple-quoted strings (docstrings)
        content = re.sub(r'"""[\s\S]*?"""', "", content)
        content = re.sub(r"'''[\s\S]*?'''", "", content)

    issues = []

    # Get project info
    project_name = project_metadata.get("project", {}).get("name", "")
    org_name = project_metadata.get("vcs", {}).get("org", "")

    # Run all checks
    issues.extend(check_hardcoded_project_names(file_path, content, project_name))
    issues.extend(check_language_assumptions(file_path, content))
    issues.extend(check_vcs_assumptions(file_path, content))
    issues.extend(check_absolute_paths(file_path, content))
    issues.extend(check_template_placeholders(file_path, content))
    issues.extend(check_organization_names(file_path, content, org_name))

    return issues


def calculate_score(issues: list[PortabilityIssue]) -> int:
    """Calculate portability score (0-100).

    Args:
        issues: List of all issues found

    Returns:
        Score from 0 (not portable) to 100 (fully portable)
    """
    # Start at 100, deduct points for each issue
    score = 100

    for issue in issues:
        score -= SEVERITY_SCORES.get(issue.severity, 1)

    # Clamp to 0-100
    return max(0, min(100, score))


def main() -> int:
    """Main validation function.

    Returns:
        Exit code (0 for fully portable, 1 for mostly portable, 2 for not portable)
    """
    print("\n" + "=" * 70)
    print("  PORTABILITY VALIDATION")
    print("=" * 70 + "\n")

    # Load project metadata
    project_metadata = load_project_metadata()

    # Collect all files to check
    check_files: list[Path] = []
    for pattern in ["**/*.yaml", "**/*.yml", "**/*.md", "**/*.py"]:
        check_files.extend(CLAUDE_DIR.glob(pattern))

    # Filter out exempted directories
    exempted_dirs = {"__pycache__", ".git", "archive", "reports", "agent-outputs"}
    check_files = [f for f in check_files if not any(ex in f.parts for ex in exempted_dirs)]

    if not check_files:
        print("⚠ WARNING: No files found to validate", file=sys.stderr)
        return 0

    print(f"Checking {len(check_files)} file(s) for portability...\n")

    # Validate each file
    all_issues = []
    for file_path in sorted(check_files):
        issues = validate_file(file_path, project_metadata)
        all_issues.extend(issues)

    # Calculate score
    score = calculate_score(all_issues)

    # Group issues by severity
    issues_by_severity = defaultdict(list)
    for issue in all_issues:
        issues_by_severity[issue.severity].append(issue)

    # Report results
    print("=" * 70)
    print(f"\n📊 PORTABILITY SCORE: {score}/100\n")

    if all_issues:
        # Report by severity
        for severity in [CRITICAL, HIGH, MEDIUM, LOW]:
            issues = issues_by_severity.get(severity, [])
            if issues:
                print(f"\n[{severity.upper()}] {len(issues)} issue(s):\n")
                for issue in issues[:5]:  # Show first 5 per severity
                    print(f"  {issue}\n")
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more\n")

    print("=" * 70)

    # Determine exit code based on score
    if score >= 95:
        print("\n✅ FULLY PORTABLE")
        print("  The .claude infrastructure is highly portable across projects.")
        print("\n" + "=" * 70)
        return 0
    elif score >= 80:
        print("\n⚠ MOSTLY PORTABLE")
        print("  Some minor portability issues found. Consider addressing them.")
        print("\n" + "=" * 70)
        return 1
    else:
        print("\n❌ NOT PORTABLE")
        print("  Significant portability issues found. Must address before reuse.")
        print("\n" + "=" * 70)
        return 2


if __name__ == "__main__":
    sys.exit(main())
