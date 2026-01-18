#!/usr/bin/env bash
# =============================================================================
# VS Code Extension Installer for Workspace Template
# =============================================================================
# Installs all recommended VS Code extensions from .vscode/extensions.json
# Usage: ./scripts/install_extensions.sh [--check] [--list] [--json]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output (detect terminal capability)
if [[ -t 1 ]] && [[ -n "${TERM:-}" ]] && [[ "${TERM}" != "dumb" ]]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  DIM='\033[2m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BLUE='' DIM='' NC=''
fi

# =============================================================================
# Configuration
# =============================================================================

EXTENSIONS_FILE="${REPO_ROOT}/.vscode/extensions.json"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Parse extensions from extensions.json (handles JSONC - JSON with comments)
# Uses jq if available, otherwise falls back to grep/sed
parse_extensions_json() {
  local file="$1"

  if [[ ! -f "${file}" ]]; then
    return 1
  fi

  # Try jq first (most reliable)
  if command -v jq &> /dev/null; then
    # Strip comments and parse with jq
    sed 's|//.*||g' "${file}" | jq -r '.recommendations[]' 2> /dev/null
    return $?
  fi

  # Fallback: grep-based parsing
  # 1. Remove single-line comments (// ...)
  # 2. Extract strings that look like extension IDs (publisher.extension-name)
  # 3. Filter to only lines within "recommendations" context
  grep -E '"[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"' "${file}" \
    | sed 's|//.*||g' \
    | grep -oE '"[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"' \
    | tr -d '"' \
    | sort -u
}

# Load extensions from extensions.json into EXTENSIONS array
load_extensions() {
  EXTENSIONS=()

  if [[ ! -f "${EXTENSIONS_FILE}" ]]; then
    log_warning "No .vscode/extensions.json found"
    return 1
  fi

  local ext
  while IFS= read -r ext; do
    [[ -n "${ext}" ]] && EXTENSIONS+=("${ext}")
  done < <(parse_extensions_json "${EXTENSIONS_FILE}")

  if [[ ${#EXTENSIONS[@]} -eq 0 ]]; then
    log_warning "No extensions found in extensions.json"
    return 1
  fi

  return 0
}

# Detect VS Code command
detect_vscode() {
  if command -v code &> /dev/null; then
    echo "code"
  elif command -v code-insiders &> /dev/null; then
    echo "code-insiders"
  elif command -v codium &> /dev/null; then
    echo "codium"
  else
    echo ""
  fi
}

# Check if extension is installed (case-insensitive)
is_installed() {
  local vscode_cmd=$1
  local extension=$2
  # Normalize both to lowercase for comparison
  local extension_lower
  extension_lower=$(echo "${extension}" | tr '[:upper:]' '[:lower:]')
  "${vscode_cmd}" --list-extensions 2> /dev/null | tr '[:upper:]' '[:lower:]' | grep -q "^${extension_lower}$"
}

# =============================================================================
# Main Logic
# =============================================================================

show_help() {
  cat << 'EOF'
Usage: install_extensions.sh [OPTIONS]

Install recommended VS Code extensions from .vscode/extensions.json

OPTIONS:
    --check     Check which extensions are installed/missing
    --list      List all recommended extensions
    --json      Output machine-readable JSON
    -h, --help  Show this help message

Without options: Install all missing extensions

EXAMPLES:
    ./scripts/install_extensions.sh              # Install missing extensions
    ./scripts/install_extensions.sh --check      # Check status only
    ./scripts/install_extensions.sh --list       # List all recommendations

EXIT CODES:
    0 - Success (all installed or all checked)
    1 - Missing extensions (--check mode) or installation failures
    2 - Configuration error (no extensions.json, no VS Code)
EOF
}

list_extensions() {
  if ! load_extensions; then
    exit 2
  fi

  local count=${#EXTENSIONS[@]}
  echo "Recommended VS Code Extensions (${count} total):"
  echo ""
  echo "Source: .vscode/extensions.json"
  echo ""

  for ext in "${EXTENSIONS[@]}"; do
    echo "  - ${ext}"
  done
}

check_extensions() {
  local vscode_cmd=$1
  local installed=0
  local missing=0

  echo "Extension Status:"
  echo ""

  for ext in "${EXTENSIONS[@]}"; do
    if is_installed "${vscode_cmd}" "${ext}"; then
      log_success "${ext}"
      ((installed++)) || true
    else
      log_warning "${ext} (not installed)"
      ((missing++)) || true
    fi
  done

  echo ""
  echo "Summary: ${installed} installed, ${missing} missing"

  if ${JSON_MODE}; then
    printf '{"installed":%d,"missing":%d,"total":%d}\n' \
      "${installed}" "${missing}" "${#EXTENSIONS[@]}"
  fi

  return "${missing}"
}

install_extensions() {
  local vscode_cmd=$1
  local installed=0
  local skipped=0
  local failed=0

  log_info "Installing VS Code extensions..."
  echo ""

  for ext in "${EXTENSIONS[@]}"; do
    if is_installed "${vscode_cmd}" "${ext}"; then
      log_success "${ext} (already installed)"
      ((skipped++)) || true
    else
      log_info "Installing ${ext}..."
      if "${vscode_cmd}" --install-extension "${ext}" --force &> /dev/null; then
        log_success "${ext} installed"
        ((installed++)) || true
      else
        log_error "Failed to install ${ext}"
        ((failed++)) || true
      fi
    fi
  done

  echo ""
  echo "Summary: ${installed} installed, ${skipped} already present, ${failed} failed"

  if ${JSON_MODE}; then
    printf '{"installed":%d,"skipped":%d,"failed":%d}\n' \
      "${installed}" "${skipped}" "${failed}"
  fi

  if [[ ${failed} -gt 0 ]]; then
    return 1
  fi
  return 0
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
CHECK_ONLY=false
LIST_ONLY=false
JSON_MODE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      CHECK_ONLY=true
      shift
      ;;
    --list)
      LIST_ONLY=true
      shift
      ;;
    --json)
      JSON_MODE=true
      shift
      ;;
    -h | --help)
      show_help
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      show_help
      exit 2
      ;;
  esac
done

# List only mode (doesn't need VS Code)
if [[ "${LIST_ONLY}" == "true" ]]; then
  list_extensions
  exit 0
fi

# Load extensions from extensions.json
if ! load_extensions; then
  log_error "Cannot proceed without extensions.json"
  echo ""
  echo "Create .vscode/extensions.json with a 'recommendations' array."
  exit 2
fi

# Detect VS Code
VSCODE_CMD=$(detect_vscode)

if [[ -z "${VSCODE_CMD}" ]]; then
  log_error "VS Code command not found"
  echo ""
  echo "Make sure one of these is in your PATH:"
  echo "  - code (VS Code)"
  echo "  - code-insiders (VS Code Insiders)"
  echo "  - codium (VSCodium)"
  echo ""
  echo "To install extensions manually, run:"
  for ext in "${EXTENSIONS[@]}"; do
    echo "  code --install-extension ${ext}"
  done
  exit 2
fi

log_info "Using: ${VSCODE_CMD}"
log_info "Extensions source: ${EXTENSIONS_FILE} (${#EXTENSIONS[@]} extensions)"
echo ""

# Check or install
if [[ "${CHECK_ONLY}" == "true" ]]; then
  check_extensions "${VSCODE_CMD}"
  exit_code=$?
  [[ ${exit_code} -eq 0 ]] && exit 0 || exit 1
else
  install_extensions "${VSCODE_CMD}"
fi
