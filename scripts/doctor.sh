#!/usr/bin/env bash
# =============================================================================
# doctor.sh - Development Environment Health Check
# =============================================================================
# Usage: ./scripts/doctor.sh [-v|--verbose] [--json] [-h|--help]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Defaults
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -v | --verbose)
      VERBOSE=true
      shift
      ;;
    --json)
      enable_json
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Development environment health check"
      echo ""
      echo "Options:"
      echo "  -v, --verbose  Show detailed version info"
      echo "  --json         Output machine-readable JSON"
      echo "  -h, --help     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

print_header "ENVIRONMENT HEALTH CHECK"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

reset_counters

# Tool categories
# shellcheck disable=SC2034
declare -A CORE_TOOLS=(
  ["git"]="Version control"
  ["bash"]="Shell interpreter"
)

# shellcheck disable=SC2034
declare -A PYTHON_TOOLS=(
  ["python3"]="Python interpreter"
  ["uv"]="Python package manager"
  ["ruff"]="Python linter/formatter"
  ["mypy"]="Python type checker"
)

# shellcheck disable=SC2034
declare -A JS_TOOLS=(
  ["node"]="Node.js runtime"
  ["npm"]="Node package manager"
  ["npx"]="Node package runner"
  ["prettier"]="Code formatter"
  ["markdownlint"]="Markdown linter"
)

# shellcheck disable=SC2034
declare -A SHELL_TOOLS=(
  ["shellcheck"]="Shell linter"
  ["shfmt"]="Shell formatter"
)

# shellcheck disable=SC2034
declare -A DATA_TOOLS=(
  ["yamllint"]="YAML linter"
  ["xmllint"]="XML validator"
  ["jq"]="JSON processor"
)

check_tools() {
  local category="$1"
  shift
  local -n tools=$1

  print_header "${category}"
  for tool in "${!tools[@]}"; do
    local desc="${tools[${tool}]}"
    local tool_found=false
    local version=""

    # For Python tools (mypy, yamllint), check via uv run if available
    if [[ "${tool}" =~ ^(mypy|yamllint)$ ]] && has_uv; then
      if uv run "${tool}" --version &> /dev/null; then
        tool_found=true
        if ${VERBOSE}; then
          version=$(uv run "${tool}" --version 2>&1 | head -1)
        fi
      fi
    fi

    # Otherwise check if tool is in PATH
    if ! ${tool_found} && has_tool "${tool}"; then
      tool_found=true
      if ${VERBOSE}; then
        case "${tool}" in
          python3) version=$(${tool} --version 2>&1 | head -1) ;;
          node) version=$(${tool} --version 2>&1) ;;
          git) version=$(${tool} --version 2>&1 | head -1) ;;
          uv) version=$(${tool} --version 2>&1 | head -1) ;;
          ruff) version=$(${tool} --version 2>&1 | head -1) ;;
          shfmt) version=$(${tool} --version 2>&1 | head -1) ;;
          jq) version=$(${tool} --version 2>&1) ;;
          *) version="installed" ;;
        esac
      fi
    fi

    if ${tool_found}; then
      if ${VERBOSE} && [[ -n "${version}" ]]; then
        print_pass "${tool}: ${desc} ${DIM}(${version})${NC}"
      else
        print_pass "${tool}: ${desc}"
      fi
      increment_passed
      json_result "${tool}" "pass" "${version}"
    else
      print_skip "${tool}: ${desc} (not installed)"
      increment_skipped
      json_result "${tool}" "skip" "Not installed"
    fi
  done
}

# Check each category
check_tools "Core Tools" CORE_TOOLS
check_tools "Python Tools" PYTHON_TOOLS
check_tools "JavaScript Tools" JS_TOOLS
check_tools "Shell Tools" SHELL_TOOLS
check_tools "Data Tools" DATA_TOOLS

# Check config files
print_header "Configuration Files"

configs=(
  "pyproject.toml:Python project config"
  ".editorconfig:Editor settings"
  ".gitattributes:Git attributes"
  ".gitignore:Git ignore patterns"
  ".shellcheckrc:ShellCheck config"
  ".yamllint.yaml:YAML lint config"
  ".markdownlint.yaml:Markdown lint config"
  ".prettierrc.yaml:Prettier config"
  ".prettierignore:Prettier ignore"
  ".pre-commit-config.yaml:Pre-commit hooks"
)

for config in "${configs[@]}"; do
  name="${config%%:*}"
  desc="${config#*:}"
  if [[ -f "${REPO_ROOT}/${name}" ]]; then
    print_pass "${name}: ${desc}"
    increment_passed
    json_result "${name}" "pass" ""
  else
    print_skip "${name}: ${desc} (not found)"
    increment_skipped
    json_result "${name}" "skip" "Not found"
  fi
done

# Summary (ignore return value - we use custom exit logic below)
print_summary "check" || true

# Recommendations for critical missing tools
missing_critical=false

if ! has_tool "uv"; then
  echo -e "  ${RED}[CRITICAL]${NC} Install uv: ${DIM}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
  missing_critical=true
fi
if ! has_tool "shellcheck"; then
  echo -e "  ${RED}[CRITICAL]${NC} Install shellcheck: ${DIM}apt install shellcheck${NC}"
  missing_critical=true
fi

# Recommendations for optional tools
if ! has_tool "shfmt"; then
  echo -e "  ${YELLOW}[OPTIONAL]${NC} Install shfmt: ${DIM}go install mvdan.cc/sh/v3/cmd/shfmt@latest${NC}"
fi
if ! has_tool "markdownlint"; then
  echo -e "  ${YELLOW}[OPTIONAL]${NC} Install markdownlint: ${DIM}npm install -g markdownlint-cli${NC}"
fi
if ! has_tool "prettier"; then
  echo -e "  ${YELLOW}[OPTIONAL]${NC} Install prettier: ${DIM}npm install -g prettier${NC}"
fi
if ! has_tool "jq"; then
  echo -e "  ${YELLOW}[OPTIONAL]${NC} Install jq: ${DIM}apt install jq${NC}"
fi

echo ""
if ${missing_critical}; then
  echo -e "  ${RED}${BOLD}Critical tools missing!${NC}"
  echo -e "  ${DIM}Run ./scripts/setup.sh for automated setup${NC}"
  exit 1
else
  echo -e "  ${GREEN}${BOLD}Environment is healthy!${NC}"
  exit 0
fi
