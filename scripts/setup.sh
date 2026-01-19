#!/usr/bin/env bash
# =============================================================================
# setup.sh - Development Environment Setup
# =============================================================================
# Usage: ./scripts/setup.sh [--dev] [--check-only] [-v|--verbose] [-h|--help]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Flags
DEV_MODE=false
CHECK_ONLY=false
VERBOSE=false

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev)
      DEV_MODE=true
      shift
      ;;
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    -v | --verbose)
      VERBOSE=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Set up development environment for Oscura"
      echo ""
      echo "Options:"
      echo "  --dev        Install development dependencies"
      echo "  --check-only Only check dependencies, don't install"
      echo "  -v, --verbose Show detailed output"
      echo "  -h, --help   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

print_header "OSCURA DEVELOPMENT SETUP"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

cd "${REPO_ROOT}"

# =============================================================================
# Dependency Checks
# =============================================================================

print_header "Checking Dependencies"

MISSING_CRITICAL=()

# Python 3.11+
print_section "Core Tools"
if has_tool python3; then
  PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
  PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)
  if [[ "${PYTHON_MAJOR}" -ge 3 && "${PYTHON_MINOR}" -ge 11 ]]; then
    print_pass "Python ${PYTHON_VERSION}"
  else
    print_info "Python ${PYTHON_VERSION} (3.11+ recommended)"
  fi
else
  print_fail "Python 3 not found"
  MISSING_CRITICAL+=("python3")
fi

# uv package manager
if has_tool uv; then
  if ${VERBOSE}; then
    UV_VERSION=$(uv --version 2> /dev/null | head -1)
    print_pass "uv: ${UV_VERSION}"
  else
    print_pass "uv"
  fi
else
  print_fail "uv not found"
  MISSING_CRITICAL+=("uv")
fi

# Git
if has_tool git; then
  if ${VERBOSE}; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    print_pass "Git ${GIT_VERSION}"
  else
    print_pass "git"
  fi
else
  print_fail "git not found"
  MISSING_CRITICAL+=("git")
fi

# Optional tools
print_section "Optional Tools"
if has_tool shellcheck; then
  print_pass "shellcheck"
else
  print_skip "shellcheck (apt install shellcheck)"
fi

if has_tool shfmt; then
  print_pass "shfmt"
else
  print_skip "shfmt (go install mvdan.cc/sh/v3/cmd/shfmt@latest)"
fi

if has_tool prettier; then
  print_pass "prettier"
else
  print_skip "prettier (npm install -g prettier)"
fi

if has_tool markdownlint; then
  print_pass "markdownlint"
else
  print_skip "markdownlint (npm install -g markdownlint-cli)"
fi

# =============================================================================
# Check for Missing Critical Dependencies
# =============================================================================

if [[ ${#MISSING_CRITICAL[@]} -gt 0 ]]; then
  print_header "MISSING CRITICAL DEPENDENCIES"
  echo ""
  echo -e "  ${RED}${BOLD}Cannot continue - missing: ${MISSING_CRITICAL[*]}${NC}"
  echo ""
  echo "  Installation instructions:"
  for dep in "${MISSING_CRITICAL[@]}"; do
    case ${dep} in
      python3)
        echo -e "    Python 3.11+: ${DIM}https://www.python.org/downloads/${NC}"
        ;;
      uv)
        echo -e "    uv: ${DIM}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        ;;
      git)
        echo -e "    Git: ${DIM}apt install git${NC}"
        ;;
    esac
  done
  exit 1
fi

if ${CHECK_ONLY}; then
  print_pass "All critical dependencies present"
  exit 0
fi

# =============================================================================
# Environment Setup
# =============================================================================

print_header "Python Environment"

print_section "Installing dependencies"

if ${DEV_MODE}; then
  print_info "Installing with development dependencies..."
  if ${VERBOSE}; then
    uv sync --dev
  else
    uv sync --dev &> /dev/null
  fi
else
  print_info "Installing production dependencies..."
  if ${VERBOSE}; then
    uv sync
  else
    uv sync &> /dev/null
  fi
fi

print_pass "Dependencies installed"

# =============================================================================
# Git Hooks Setup (REQUIRED)
# =============================================================================

print_header "Git Hooks"

print_section "Installing pre-commit and pre-push hooks"
if [[ -x "${SCRIPT_DIR}/setup/install-hooks.sh" ]]; then
  if ${VERBOSE}; then
    if "${SCRIPT_DIR}/setup/install-hooks.sh"; then
      print_pass "Git hooks installed (pre-commit + pre-push)"
    else
      print_fail "Failed to install git hooks"
    fi
  else
    if "${SCRIPT_DIR}/setup/install-hooks.sh" &> /dev/null; then
      print_pass "Git hooks installed (pre-commit + pre-push)"
    else
      print_fail "Failed to install git hooks"
    fi
  fi
else
  print_fail "setup/install-hooks.sh not found or not executable"
  print_info "Falling back to pre-commit only..."
  if [[ -f "${REPO_ROOT}/.pre-commit-config.yaml" ]]; then
    if uv run pre-commit install &> /dev/null; then
      print_pass "Pre-commit hooks installed (pre-push hook missing)"
    else
      print_fail "Failed to install pre-commit hooks"
    fi
  fi
fi

# =============================================================================
# Script Executability
# =============================================================================

print_header "Script Permissions"

scripts_fixed=0

fix_permissions() {
  local dir="$1"
  local pattern="${2:-*.sh}"

  if [[ -d "${dir}" ]]; then
    while IFS= read -r -d '' script; do
      if [[ ! -x "${script}" ]]; then
        chmod +x "${script}"
        ((scripts_fixed++)) || true
        ${VERBOSE} && print_info "Made executable: ${script}"
      fi
    done < <(find "${dir}" -maxdepth 1 -type f -name "${pattern}" -print0 2> /dev/null)
  fi
}

fix_permissions "${REPO_ROOT}/scripts"
fix_permissions "${REPO_ROOT}/scripts/tools"
fix_permissions "${REPO_ROOT}/scripts/git"
fix_permissions "${REPO_ROOT}/scripts/maintenance"
fix_permissions "${REPO_ROOT}/.claude/hooks" "*.sh"
fix_permissions "${REPO_ROOT}/.claude/hooks" "*.py"

if [[ ${scripts_fixed} -gt 0 ]]; then
  print_pass "Fixed ${scripts_fixed} script(s) permissions"
else
  print_pass "All scripts executable"
fi

# =============================================================================
# Summary
# =============================================================================

print_header "SETUP COMPLETE"
echo ""
echo -e "  ${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo -e "  ${DIM}Next steps:${NC}"
echo -e "    - Run ${CYAN}./scripts/check.sh${NC} to verify code quality"
echo -e "    - Run ${CYAN}./scripts/doctor.sh${NC} to check environment health"
echo -e "    - Run ${CYAN}uv run python -m pytest tests/unit -x --maxfail=5${NC} to run tests"
