#!/usr/bin/env python3
"""Batch refactoring tool for complex functions.

This script provides refactoring suggestions and can apply them automatically
for functions exceeding complexity/length thresholds.
"""

import argparse
import json
from pathlib import Path


def load_complexity_report() -> dict:
    """Load the complexity analysis report."""
    report_path = Path(
        "/home/lair-click-bats/development/oscura/.claude/agent-outputs/complexity_analysis_complete.json"
    )
    with open(report_path) as f:
        return json.load(f)


def generate_refactoring_plan(report: dict, batch_size: int = 10) -> list[dict]:
    """Generate refactoring plan for top N functions.

    Args:
        report: Complexity analysis report.
        batch_size: Number of functions to include in plan.

    Returns:
        List of function metadata dictionaries.
    """
    functions = report["functions"][:batch_size]
    return [
        {
            "name": f["name"],
            "file": f["file"],
            "lineno": f["lineno"],
            "lines": f["lines"],
            "complexity": f["complexity"],
            "priority": idx + 1,
        }
        for idx, f in enumerate(functions)
    ]


def print_refactoring_plan(plan: list[dict]) -> None:
    """Print refactoring plan to console."""
    print("\n" + "=" * 80)
    print("REFACTORING PLAN")
    print("=" * 80)
    for func in plan:
        relative_path = func["file"].replace("/home/lair-click-bats/development/oscura/", "")
        print(f"\n{func['priority']}. {func['name']}")
        print(f"   Location: {relative_path}:{func['lineno']}")
        print(f"   Metrics: {func['lines']} lines, complexity {func['complexity']}")
        print("   Target: <100 lines, <15 complexity")
        print(
            f"   Reduction needed: {func['lines'] - 100} lines, {func['complexity'] - 15} complexity"
        )


def save_refactoring_plan(plan: list[dict], output_path: Path) -> None:
    """Save refactoring plan to JSON file.

    Args:
        plan: Refactoring plan.
        output_path: Output file path.
    """
    with open(output_path, "w") as f:
        json.dump({"plan": plan, "total_functions": len(plan)}, f, indent=2)
    print(f"\nPlan saved to: {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate refactoring plan for complex functions")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of functions to plan")
    parser.add_argument("--output", type=Path, help="Output file for plan (JSON)")
    args = parser.parse_args()

    # Load report
    report = load_complexity_report()

    # Generate plan
    plan = generate_refactoring_plan(report, batch_size=args.batch_size)

    # Print plan
    print_refactoring_plan(plan)

    # Save plan if requested
    if args.output:
        save_refactoring_plan(plan, args.output)
    else:
        default_output = Path(
            "/home/lair-click-bats/development/oscura/.claude/agent-outputs/refactoring_plan.json"
        )
        save_refactoring_plan(plan, default_output)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Total functions to refactor: {report['problematic_functions']}")
    print(f"Batch size: {len(plan)}")
    print(f"Remaining after batch: {report['problematic_functions'] - len(plan)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
