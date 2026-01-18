"""
Stress Test Framework for Claude Code Orchestration

Comprehensive testing of:
- Configuration validation
- Hook execution
- Agent orchestration
- Context management
- Edge cases
- Performance benchmarks
"""

from pathlib import Path

# Test framework version
__version__ = "1.0.0"

# Base paths
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent.parent
HOOKS_DIR = PROJECT_ROOT / ".claude" / "hooks"
FIXTURES_DIR = TESTS_DIR / "fixtures"
SCENARIOS_DIR = TESTS_DIR / "scenarios"

# Ensure fixture directories exist
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
