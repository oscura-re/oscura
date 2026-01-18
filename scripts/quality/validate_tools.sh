#!/usr/bin/env bash
# =============================================================================
# validate_tools.sh - Validation Dashboard for Oscura Configurations
# =============================================================================
# Usage: ./scripts/validate_tools.sh [-h|--help] [-v|--verbose]
# =============================================================================
# Runs all linters/formatters on test files and reports results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common library
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Options
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -v | --verbose)
      VERBOSE=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [-h|--help] [-v|--verbose]"
      echo ""
      echo "Validation dashboard for Oscura configurations"
      echo ""
      echo "Runs all configured linters and formatters on test files to"
      echo "validate that tool configurations are working correctly."
      echo ""
      echo "Options:"
      echo "  -v, --verbose  Show detailed output"
      echo "  -h, --help     Show this help message"
      echo ""
      echo "Test files location: tests/unit/"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try '$0 --help' for usage information" >&2
      exit 2
      ;;
  esac
done

TEST_DIR="${REPO_ROOT}/tests/unit"

print_header "OSCURA - TOOL VALIDATION DASHBOARD"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Test Files:${NC} ${TEST_DIR}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

reset_counters

run_check() {
  local name="$1"
  local config="$2"
  local cmd="$3"

  print_section "${name}"
  echo -e "    ${DIM}Config:${NC} ${config}"

  if eval "${cmd}" &> /dev/null; then
    print_pass "Validation passed"
    increment_passed
    return 0
  else
    print_fail "Validation failed"
    increment_failed
    # Show actual output in verbose mode
    if ${VERBOSE}; then
      echo ""
      eval "${cmd}" 2>&1 | head -5 | sed 's/^/      /'
    fi
    return 1
  fi
}

skip_check() {
  local name="$1"
  local reason="$2"

  print_section "${name}"
  print_skip "${reason}"
  increment_skipped
}

# ==============================================================================
# Python
# ==============================================================================

print_header "Python (ruff + mypy)"

if has_tool "uv"; then
  if [[ -f "${TEST_DIR}/test_python.py" ]]; then
    run_check "Ruff Linting" "pyproject.toml [tool.ruff.lint]" \
      "cd '${REPO_ROOT}' && uv run ruff check '${TEST_DIR}/test_python.py'"

    run_check "Ruff Formatting" "pyproject.toml [tool.ruff.format]" \
      "cd '${REPO_ROOT}' && uv run ruff format --check '${TEST_DIR}/test_python.py'"

    run_check "Mypy Type Checking" "pyproject.toml [tool.mypy]" \
      "cd '${REPO_ROOT}' && uv run mypy '${TEST_DIR}/test_python.py'"
  else
    skip_check "Python Validation" "Test file not found: ${TEST_DIR}/test_python.py"
  fi
else
  skip_check "Python Tools" "uv not installed"
fi

# ==============================================================================
# Shell
# ==============================================================================

print_header "Shell (shellcheck + shfmt)"

if has_tool "shellcheck"; then
  if [[ -f "${TEST_DIR}/test_script.sh" ]]; then
    run_check "ShellCheck Linting" ".shellcheckrc" \
      "cd '${REPO_ROOT}' && shellcheck '${TEST_DIR}/test_script.sh'"
  else
    skip_check "ShellCheck" "Test file not found: ${TEST_DIR}/test_script.sh"
  fi
else
  skip_check "ShellCheck" "shellcheck not installed"
fi

if has_tool "shfmt"; then
  if [[ -f "${TEST_DIR}/test_script.sh" ]]; then
    run_check "shfmt Formatting" ".editorconfig" \
      "cd '${REPO_ROOT}' && shfmt -d '${TEST_DIR}/test_script.sh'"
  else
    skip_check "shfmt" "Test file not found: ${TEST_DIR}/test_script.sh"
  fi
else
  skip_check "shfmt" "shfmt not installed (go install mvdan.cc/sh/v3/cmd/shfmt@latest)"
fi

# ==============================================================================
# YAML
# ==============================================================================

print_header "YAML (yamllint)"

if has_tool "yamllint"; then
  if [[ -f "${TEST_DIR}/test_config.yaml" ]]; then
    run_check "YAML Linting" ".yamllint.yaml" \
      "cd '${REPO_ROOT}' && yamllint '${TEST_DIR}/test_config.yaml'"
  else
    skip_check "YAML Linting" "Test file not found: ${TEST_DIR}/test_config.yaml"
  fi
else
  skip_check "YAML Linting" "yamllint not installed"
fi

# ==============================================================================
# JSON/JSONC
# ==============================================================================

print_header "JSON/JSONC (jsonc.sh)"

if has_tool "jq" || has_tool "python3"; then
  if [[ -f "${TEST_DIR}/test_config.json" ]]; then
    run_check "JSON Validation" "scripts/tools/jsonc.sh" \
      "cd '${REPO_ROOT}' && '${SCRIPT_DIR}/tools/jsonc.sh' --check '${TEST_DIR}/test_config.json'"
  else
    skip_check "JSON Validation" "Test file not found: ${TEST_DIR}/test_config.json"
  fi

  # Test JSONC support with VS Code settings if they exist
  if [[ -f "${REPO_ROOT}/.vscode/settings.json" ]]; then
    run_check "JSONC Validation (VS Code)" "scripts/tools/jsonc.sh" \
      "cd '${REPO_ROOT}' && '${SCRIPT_DIR}/tools/jsonc.sh' --check '${REPO_ROOT}/.vscode/settings.json'"
  fi
else
  skip_check "JSON Validation" "jq or python3 not installed"
fi

# ==============================================================================
# Markdown
# ==============================================================================

print_header "Markdown (markdownlint)"

if has_tool "markdownlint"; then
  if [[ -f "${TEST_DIR}/test_markdown.md" ]]; then
    run_check "Markdown Linting" ".markdownlint.yaml" \
      "cd '${REPO_ROOT}' && markdownlint '${TEST_DIR}/test_markdown.md'"
  else
    skip_check "Markdown Linting" "Test file not found: ${TEST_DIR}/test_markdown.md"
  fi
else
  skip_check "Markdown Linting" "markdownlint not installed"
fi

# ==============================================================================
# Prettier (JSON, YAML, Markdown)
# ==============================================================================

print_header "Prettier (json, yaml)"

if has_tool "npx"; then
  if [[ -f "${TEST_DIR}/test_config.json" ]]; then
    run_check "JSON Formatting" ".prettierrc.yaml" \
      "cd '${REPO_ROOT}' && npx prettier --check '${TEST_DIR}/test_config.json'"
  else
    skip_check "JSON Formatting" "Test file not found: ${TEST_DIR}/test_config.json"
  fi

  if [[ -f "${TEST_DIR}/test_config.yaml" ]]; then
    run_check "YAML Formatting" ".prettierrc.yaml" \
      "cd '${REPO_ROOT}' && npx prettier --check '${TEST_DIR}/test_config.yaml'"
  else
    skip_check "YAML Formatting" "Test file not found: ${TEST_DIR}/test_config.yaml"
  fi
else
  skip_check "Prettier" "npx not installed"
fi

# ==============================================================================
# VS Code
# ==============================================================================

print_header "VS Code (vscode_extensions.sh)"

if has_tool "code"; then
  if [[ -f "${SCRIPT_DIR}/tools/vscode_extensions.sh" ]]; then
    run_check "VS Code Extensions" "scripts/tools/vscode_extensions.sh" \
      "cd '${REPO_ROOT}' && '${SCRIPT_DIR}/tools/vscode_extensions.sh' --check"
  else
    skip_check "VS Code Extensions" "Script not found"
  fi
else
  skip_check "VS Code Extensions" "VS Code CLI not installed"
fi

# ==============================================================================
# Summary
# ==============================================================================

print_header "VALIDATION SUMMARY"
echo ""
echo -e "  ${GREEN}Passed:${NC}  ${PASSED}"
echo -e "  ${RED}Failed:${NC}  ${FAILED}"
echo -e "  ${YELLOW}Skipped:${NC} ${SKIPPED}"
echo ""

total=$((PASSED + FAILED))
if [[ ${total} -gt 0 ]]; then
  pass_rate=$((PASSED * 100 / total))
  echo -e "  ${DIM}Pass rate:${NC} ${pass_rate}%"
  echo ""
fi

if [[ ${FAILED} -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}All validation checks passed!${NC}"
  exit 0
else
  echo -e "  ${RED}${BOLD}Some validation checks failed.${NC}"
  echo -e "  ${DIM}Run with -v for details${NC}"
  exit 1
fi
