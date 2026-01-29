#!/usr/bin/env python3
"""
Reformat markdown tables to compact format (unaligned pipes).

This script reformats markdown tables to use compact format where pipes
are not aligned, which is the Oscura standard for handling emoji characters
that have variable display widths.

Usage:
    python scripts/quality/reformat_tables.py <file.md>
    python scripts/quality/reformat_tables.py --all  # Reformat all .md files
"""

import re
import sys
from pathlib import Path


def reformat_table(table_text: str) -> str:
    """
    Reformat a markdown table to compact format (no pipe alignment).

    Args:
        table_text: The table text to reformat

    Returns:
        Reformatted table text
    """
    lines = table_text.strip().split("\n")
    if len(lines) < 2:
        return table_text

    reformatted = []
    for i, line in enumerate(lines):
        # Split by pipe, strip whitespace from each cell
        cells = [cell.strip() for cell in line.split("|")]

        # Remove empty first/last cells (from leading/trailing pipes)
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if i == 1:
            # Header separator line - use minimal dashes
            reformatted.append("|" + "|".join(["-" * 3 for _ in cells]) + "|")
        else:
            # Regular line - compact format
            reformatted.append("|" + "|".join(cells) + "|")

    return "\n".join(reformatted)


def reformat_file(file_path: Path, dry_run: bool = False) -> int:
    """
    Reformat all tables in a markdown file.

    Args:
        file_path: Path to the markdown file
        dry_run: If True, only report changes without modifying file

    Returns:
        Number of tables reformatted
    """
    content = file_path.read_text()

    # Match markdown tables (simplified pattern)
    # Matches: line starting with |, followed by separator line, followed by data lines
    table_pattern = re.compile(
        r"((?:^\|[^\n]+\|\n)+^\|[-:\s|]+\|\n(?:^\|[^\n]+\|\n)*)", re.MULTILINE
    )

    tables_found = 0
    new_content = content

    for match in table_pattern.finditer(content):
        table_text = match.group(1)
        reformatted = reformat_table(table_text)

        if reformatted != table_text:
            tables_found += 1
            if not dry_run:
                new_content = new_content.replace(table_text, reformatted)

    if tables_found > 0 and not dry_run:
        file_path.write_text(new_content)
        print(f"✅ {file_path}: Reformatted {tables_found} table(s)")
    elif tables_found > 0 and dry_run:
        print(f"📋 {file_path}: Would reformat {tables_found} table(s)")
    else:
        print(f"⏭️  {file_path}: No tables need reformatting")

    return tables_found


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python reformat_tables.py <file.md>")
        print("       python reformat_tables.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        # Reformat all markdown files in project
        project_root = Path(__file__).parent.parent.parent
        md_files = list(project_root.glob("*.md")) + list(project_root.glob("**/*.md"))

        # Filter out excluded directories
        excluded = {".venv", "node_modules", "build", "dist", ".git"}
        md_files = [f for f in md_files if not any(part in excluded for part in f.parts)]

        print(f"Found {len(md_files)} markdown files")
        total_tables = 0

        for md_file in sorted(md_files):
            total_tables += reformat_file(md_file, dry_run=False)

        print(f"\n✅ Total: Reformatted {total_tables} tables in {len(md_files)} files")
    else:
        # Reformat single file
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        reformat_file(file_path)


if __name__ == "__main__":
    main()
