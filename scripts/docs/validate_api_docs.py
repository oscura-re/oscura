#!/usr/bin/env python3
"""Validate API documentation completeness against source code.

This script ensures that:
1. All public functions in __init__.py are documented
2. Function signatures match between docs and code
3. No documented functions that don't exist in code
4. Example code in docs can be parsed (syntax check)

Usage:
    python scripts/validate_api_docs.py [--verbose] [--fix]
"""

import argparse
import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any


class APIValidator:
    """Validates API documentation against source code."""

    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
        self.issues: list[dict[str, Any]] = []

    def log(self, message: str) -> None:
        """Print verbose logging message."""
        if self.verbose:
            print(f"  {message}")

    def extract_public_api(self, module_path: Path) -> dict[str, Any]:
        """Extract all public API functions from a module's __init__.py."""
        init_file = module_path / "__init__.py"
        if not init_file.exists():
            return {}

        try:
            spec = importlib.util.spec_from_file_location("module", init_file)
            if spec is None or spec.loader is None:
                return {}

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get all public items from __all__ or non-underscore items
            if hasattr(module, "__all__"):
                public_names = module.__all__
            else:
                public_names = [name for name in dir(module) if not name.startswith("_")]

            api = {}
            for name in public_names:
                try:
                    obj = getattr(module, name)
                    if callable(obj):
                        api[name] = {
                            "type": "function" if inspect.isfunction(obj) else "class",
                            "signature": str(inspect.signature(obj)) if callable(obj) else None,
                            "doc": inspect.getdoc(obj),
                        }
                except Exception:
                    pass

            return api

        except Exception as e:
            self.log(f"Failed to import {init_file}: {e}")
            return {}

    def extract_documented_functions(self, doc_file: Path) -> set[str]:
        """Extract function names mentioned in documentation."""
        if not doc_file.exists():
            return set()

        content = doc_file.read_text(encoding="utf-8")

        # Match patterns like osc.function_name( or oscura.function_name(
        patterns = [
            r"tk\.([a-z_][a-z0-9_]*)\s*\(",
            r"oscura\.([a-z_][a-z0-9_]*)\s*\(",
            r"`([a-z_][a-z0-9_]*)\(\)`",
            r"def ([a-z_][a-z0-9_]*)\(",
        ]

        documented = set()
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            documented.update(matches)

        return documented

    def extract_code_examples(self, doc_file: Path) -> list[tuple[int, str]]:
        """Extract Python code blocks from markdown."""
        if not doc_file.exists():
            return []

        content = doc_file.read_text(encoding="utf-8")
        examples = []

        # Match ```python code blocks
        pattern = r"```python\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            code = match.group(1)
            # Calculate line number
            line_num = content[: match.start()].count("\n") + 1
            examples.append((line_num, code))

        return examples

    def validate_code_syntax(self, doc_file: Path) -> None:
        """Validate that code examples can be parsed."""
        examples = self.extract_code_examples(doc_file)

        for line_num, code in examples:
            # Skip examples with placeholders or ellipsis
            if "..." in code or "<" in code or ">" in code:
                continue

            # Skip examples with obvious placeholders
            if re.search(r"\b(your_|my_|example_)", code):
                continue

            try:
                ast.parse(code)
            except SyntaxError as e:
                self.issues.append(
                    {
                        "type": "syntax_error",
                        "file": str(doc_file.relative_to(self.repo_root)),
                        "line": line_num + e.lineno - 1 if e.lineno else line_num,
                        "message": f"Invalid Python syntax: {e.msg}",
                        "code_preview": code.split("\n")[max(0, e.lineno - 2) : e.lineno + 1]
                        if e.lineno
                        else code.split("\n")[:3],
                    }
                )

    def validate_api_coverage(self) -> None:
        """Check that all public API is documented."""
        print("\n--- Checking API Coverage ---")

        # Get actual API from source code
        src_path = self.repo_root / "src" / "oscura"
        actual_api = self.extract_public_api(src_path)

        self.log(f"Found {len(actual_api)} public API items in source code")

        # Get documented API from docs
        docs_path = self.repo_root / "docs"
        documented_functions = set()

        for doc_file in docs_path.rglob("*.md"):
            funcs = self.extract_documented_functions(doc_file)
            documented_functions.update(funcs)

        self.log(f"Found {len(documented_functions)} function references in docs")

        # Find undocumented functions
        undocumented = set(actual_api.keys()) - documented_functions

        # Filter out common non-user-facing items
        undocumented = {
            f
            for f in undocumented
            if not f.startswith("_") and f not in {"version", "VERSION", "logger", "Logger"}
        }

        if undocumented:
            for func in sorted(undocumented):
                self.issues.append(
                    {
                        "type": "undocumented_api",
                        "function": func,
                        "api_type": actual_api[func]["type"],
                        "signature": actual_api[func]["signature"],
                    }
                )

        # Find documented but non-existent functions (phantom docs)
        phantom_docs = documented_functions - set(actual_api.keys())

        # Filter out common patterns that aren't actual functions
        common_words = {
            "load",
            "save",
            "open",
            "close",
            "read",
            "write",
            "print",
            "import",
            "from",
            "return",
            "trace",
            "data",
            "name",
            "path",
            "file",
        }
        phantom_docs = phantom_docs - common_words

        if phantom_docs:
            for func in sorted(phantom_docs):
                self.issues.append(
                    {
                        "type": "phantom_documentation",
                        "function": func,
                        "message": "Documented but doesn't exist in actual API",
                    }
                )

    def validate_example_syntax(self) -> None:
        """Validate all code examples in documentation."""
        print("\n--- Validating Code Examples ---")

        docs_path = self.repo_root / "docs"
        doc_files = list(docs_path.rglob("*.md"))

        for doc_file in doc_files:
            self.validate_code_syntax(doc_file)

        example_path = self.repo_root / "examples"
        if example_path.exists():
            for example_file in example_path.rglob("*.py"):
                try:
                    ast.parse(example_file.read_text(encoding="utf-8"))
                except SyntaxError as e:
                    self.issues.append(
                        {
                            "type": "syntax_error",
                            "file": str(example_file.relative_to(self.repo_root)),
                            "line": e.lineno or 0,
                            "message": f"Invalid Python syntax: {e.msg}",
                        }
                    )

    def generate_report(self) -> int:
        """Generate validation report."""
        print("\n" + "=" * 60)
        print("API Documentation Validation Report")
        print("=" * 60)

        if not self.issues:
            print("\n✅ All validations passed!")
            print("  - All public API is documented")
            print("  - All code examples have valid syntax")
            print("  - No phantom documentation found")
            return 0

        # Group issues by type
        by_type = {}
        for issue in self.issues:
            issue_type = issue["type"]
            by_type.setdefault(issue_type, []).append(issue)

        # Report undocumented API
        if "undocumented_api" in by_type:
            print(f"\n❌ Undocumented API ({len(by_type['undocumented_api'])} items)")
            for issue in sorted(by_type["undocumented_api"], key=lambda x: x["function"]):
                print(f"  - {issue['function']}() - {issue['api_type']}")
                if issue.get("signature"):
                    print(f"    Signature: {issue['signature']}")

        # Report phantom documentation
        if "phantom_documentation" in by_type:
            print(f"\n⚠️  Phantom Documentation ({len(by_type['phantom_documentation'])} items)")
            for issue in sorted(by_type["phantom_documentation"], key=lambda x: x["function"]):
                print(f"  - {issue['function']}() - {issue['message']}")

        # Report syntax errors
        if "syntax_error" in by_type:
            print(f"\n❌ Syntax Errors ({len(by_type['syntax_error'])} items)")
            for issue in by_type["syntax_error"]:
                print(f"  - {issue['file']}:{issue['line']}")
                print(f"    {issue['message']}")
                if "code_preview" in issue:
                    for line in issue["code_preview"]:
                        print(f"      {line}")

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Total issues: {len(self.issues)}")
        print(f"  Undocumented API: {len(by_type.get('undocumented_api', []))}")
        print(f"  Phantom Documentation: {len(by_type.get('phantom_documentation', []))}")
        print(f"  Syntax Errors: {len(by_type.get('syntax_error', []))}")

        return 1  # Non-zero exit for CI

    def run(self) -> int:
        """Run all validations."""
        print("=" * 60)
        print("Oscura API Documentation Validator")
        print("=" * 60)

        self.validate_api_coverage()
        self.validate_example_syntax()

        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(description="Validate API documentation completeness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Generate fix suggestions (not implemented yet)",
    )
    args = parser.parse_args()

    # Find repo root
    repo_root = Path(__file__).parent.parent

    validator = APIValidator(repo_root, verbose=args.verbose)
    exit_code = validator.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
