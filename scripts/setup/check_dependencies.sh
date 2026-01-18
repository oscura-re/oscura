#!/usr/bin/env bash
# =============================================================================
# check_dependencies.sh - Dependency Checker for Oscura
# =============================================================================
# Verifies all required and optional tools are available
# Usage: ./scripts/check_dependencies.sh [-v]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Counters
REQUIRED_MISSING=0
VERBOSE=false

# =============================================================================
# Helper Functions
# =============================================================================

log_ok() {
  local tool=$1
  local version=${2:-""}
  if [[ -n "${version}" ]]; then
    echo -e "  ${GREEN}+${NC} ${tool} ${CYAN}(${version})${NC}"
  else
    echo -e "  ${GREEN}+${NC} ${tool}"
  fi
}

log_missing() {
  local tool=$1
  echo -e "  ${RED}x${NC} ${tool} ${RED}(not found)${NC}"
}

log_optional() {
  local tool=$1
  echo -e "  ${YELLOW}o${NC} ${tool} ${YELLOW}(optional, not found)${NC}"
}

check_command() {
  command -v "$1" &> /dev/null
}

get_version() {
  local cmd=$1
  local version_flag=${2:---version}
  "${cmd}" "${version_flag}" 2> /dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || echo "unknown"
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
  case $1 in
    -v | --verbose)
      VERBOSE=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [-v]"
      echo ""
      echo "Checks for required and optional development tools."
      echo ""
      echo "Options:"
      echo "  -v, --verbose  Show installation instructions"
      echo "  -h, --help     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# =============================================================================
# Required Dependencies
# =============================================================================

echo ""
echo -e "${BLUE}Required Dependencies${NC}"
echo "---------------------------------------------"

# Python
if check_command python3; then
  VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
  log_ok "Python 3" "${VERSION}"
else
  log_missing "Python 3"
  ((REQUIRED_MISSING++))
fi

# uv
if check_command uv; then
  VERSION=$(get_version uv)
  log_ok "uv" "${VERSION}"
else
  log_missing "uv"
  ((REQUIRED_MISSING++))
fi

# Git
if check_command git; then
  VERSION=$(get_version git)
  log_ok "Git" "${VERSION}"
else
  log_missing "Git"
  ((REQUIRED_MISSING++))
fi

# =============================================================================
# Python Packages (via uv)
# =============================================================================

echo ""
echo -e "${BLUE}Python Packages${NC}"
echo "---------------------------------------------"

if check_command uv; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

  if [[ -d "${REPO_ROOT}/.venv" ]]; then
    # Core packages
    if uv run python -c "import numpy" 2> /dev/null; then
      VERSION=$(uv run python -c "import numpy; print(numpy.__version__)" 2> /dev/null || echo "unknown")
      log_ok "numpy" "${VERSION}"
    else
      log_missing "numpy"
    fi

    if uv run python -c "import scipy" 2> /dev/null; then
      VERSION=$(uv run python -c "import scipy; print(scipy.__version__)" 2> /dev/null || echo "unknown")
      log_ok "scipy" "${VERSION}"
    else
      log_missing "scipy"
    fi

    # pytest (dev)
    if uv run python -c "import pytest" 2> /dev/null; then
      VERSION=$(uv run python -c "import pytest; print(pytest.__version__)" 2> /dev/null || echo "unknown")
      log_ok "pytest" "${VERSION}"
    else
      log_optional "pytest (dev dependency)"
    fi

    # ruff
    if uv run python -c "import ruff" 2> /dev/null || check_command ruff; then
      VERSION=$(ruff --version 2> /dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
      log_ok "ruff" "${VERSION}"
    else
      log_optional "ruff"
    fi
  else
    echo -e "  ${YELLOW}o${NC} Virtual environment not found. Run: uv sync"
  fi
else
  echo -e "  ${RED}x${NC} Cannot check packages without uv"
fi

# =============================================================================
# Shell Linting Tools
# =============================================================================

echo ""
echo -e "${BLUE}Shell Linting Tools${NC}"
echo "---------------------------------------------"

# ShellCheck
if check_command shellcheck; then
  VERSION=$(shellcheck --version 2> /dev/null | grep version: | awk '{print $2}' || echo "unknown")
  log_ok "shellcheck" "${VERSION}"
else
  log_optional "shellcheck"
fi

# shfmt
if check_command shfmt; then
  VERSION=$(shfmt --version 2> /dev/null || echo "unknown")
  log_ok "shfmt" "${VERSION}"
else
  log_optional "shfmt"
fi

# =============================================================================
# Development Tools (optional)
# =============================================================================

echo ""
echo -e "${BLUE}Optional Development Tools${NC}"
echo "---------------------------------------------"

# pre-commit
if check_command pre-commit; then
  VERSION=$(get_version pre-commit)
  log_ok "pre-commit" "${VERSION}"
else
  log_optional "pre-commit"
fi

# prettier
if check_command prettier; then
  VERSION=$(get_version prettier)
  log_ok "prettier" "${VERSION}"
else
  log_optional "prettier"
fi

# markdownlint
if check_command markdownlint; then
  log_ok "markdownlint"
else
  log_optional "markdownlint"
fi

# yamllint
if check_command yamllint; then
  VERSION=$(yamllint --version 2> /dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
  log_ok "yamllint" "${VERSION}"
else
  log_optional "yamllint"
fi

# jq
if check_command jq; then
  VERSION=$(jq --version 2> /dev/null | grep -oE '[0-9]+\.[0-9]+' || echo "unknown")
  log_ok "jq" "${VERSION}"
else
  log_optional "jq"
fi

# xmllint
if check_command xmllint; then
  VERSION=$(xmllint --version 2>&1 | grep -oE '[0-9]+' | head -1 || echo "unknown")
  log_ok "xmllint" "${VERSION}"
else
  log_optional "xmllint"
fi

# lychee (link checker)
if check_command lychee; then
  VERSION=$(lychee --version 2> /dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
  log_ok "lychee" "${VERSION}"
else
  log_optional "lychee"
fi

# =============================================================================
# Diagram Tools (optional)
# =============================================================================

echo ""
echo -e "${BLUE}Diagram Tools${NC}"
echo "---------------------------------------------"

# PlantUML
if check_command plantuml; then
  VERSION=$(plantuml -version 2> /dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "unknown")
  log_ok "plantuml" "${VERSION}"
else
  log_optional "plantuml"
fi

# Mermaid CLI
if check_command mmdc; then
  VERSION=$(mmdc --version 2> /dev/null || echo "unknown")
  log_ok "mermaid-cli (mmdc)" "${VERSION}"
else
  log_optional "mermaid-cli"
fi

# Graphviz
if check_command dot; then
  VERSION=$(dot -V 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
  log_ok "graphviz (dot)" "${VERSION}"
else
  log_optional "graphviz"
fi

# =============================================================================
# VS Code (for extension management)
# =============================================================================

echo ""
echo -e "${BLUE}VS Code / Editor${NC}"
echo "---------------------------------------------"

if check_command code; then
  VERSION=$(code --version 2> /dev/null | head -1 || echo "unknown")
  log_ok "VS Code" "${VERSION}"
elif check_command code-insiders; then
  VERSION=$(code-insiders --version 2> /dev/null | head -1 || echo "unknown")
  log_ok "VS Code Insiders" "${VERSION}"
elif check_command codium; then
  VERSION=$(codium --version 2> /dev/null | head -1 || echo "unknown")
  log_ok "VSCodium" "${VERSION}"
else
  log_optional "VS Code / VSCodium"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "---------------------------------------------"

if [[ ${REQUIRED_MISSING} -eq 0 ]]; then
  echo -e "${GREEN}All required dependencies are installed!${NC}"

  if ${VERBOSE}; then
    echo ""
    echo "To install optional tools:"
    echo "  pre-commit:       uv tool install pre-commit"
    echo "  prettier:         npm install -g prettier"
    echo "  markdownlint:     npm install -g markdownlint-cli"
    echo "  yamllint:         uv tool install yamllint"
    echo "  shellcheck:       apt install shellcheck"
    echo "  shfmt:            go install mvdan.cc/sh/v3/cmd/shfmt@latest"
    echo "  jq:               apt install jq"
    echo "  xmllint:          apt install libxml2-utils"
    echo "  lychee:           cargo install lychee"
    echo ""
    echo "Diagram tools:"
    echo "  plantuml:         apt install plantuml"
    echo "  mermaid-cli:      npm install -g @mermaid-js/mermaid-cli"
    echo "  graphviz:         apt install graphviz"
  fi
  exit 0
else
  echo -e "${RED}Missing ${REQUIRED_MISSING} required dependencies!${NC}"
  echo ""
  echo "Installation instructions:"
  echo "  Python 3.11+:  https://www.python.org/downloads/"
  echo "  uv:            curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  Git:           apt install git (or equivalent)"
  exit 1
fi
