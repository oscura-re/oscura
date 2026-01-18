#!/usr/bin/env bash
# =============================================================================
# vscode_extensions.sh - VS Code Extension Detection and Validation
# =============================================================================
# Usage: ./scripts/tools/vscode_extensions.sh [--check|--list|--sync] [--json] [-v]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="VS Code Extensions"
TOOL_CMD="code"
INSTALL_HINT="Install VS Code and add 'code' to PATH"

# Defaults
MODE="check"
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      MODE="check"
      shift
      ;;
    --list)
      MODE="list"
      shift
      ;;
    --sync)
      MODE="sync"
      shift
      ;;
    --json)
      enable_json
      shift
      ;;
    -v)
      VERBOSE=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "VS Code extension detection and validation"
      echo ""
      echo "Options:"
      echo "  --check     Check installed vs recommended (default)"
      echo "  --list      List all installed extensions"
      echo "  --sync      Bidirectional sync check (missing AND extra)"
      echo "  --json      Output machine-readable JSON"
      echo "  -v          Verbose output"
      echo "  -h, --help  Show this help message"
      echo ""
      echo "Modes:"
      echo "  --check   Reports extensions in recommendations but not installed"
      echo "  --sync    Reports BOTH missing AND extra extensions (bidirectional)"
      echo "  --list    Simply lists all installed extensions"
      echo ""
      echo "Exit codes:"
      echo "  0 - All recommended extensions installed (or list mode)"
      echo "  1 - Missing recommended extensions (or sync issues)"
      echo "  2 - VS Code not installed or not in PATH"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      echo "Try '$0 --help' for usage information" >&2
      exit 2
      ;;
  esac
done

# Main
print_tool "${TOOL_NAME} (${MODE})"

# Check if VS Code CLI is available
if ! require_tool "${TOOL_CMD}" "${INSTALL_HINT}"; then
  json_result "vscode" "skip" "VS Code CLI not available"
  exit 0
fi

# Get installed extensions (normalized to lowercase for comparison)
installed_extensions_raw=$(code --list-extensions 2> /dev/null | sort)
installed_extensions=$(echo "${installed_extensions_raw}" | tr '[:upper:]' '[:lower:]')

if [[ -z "${installed_extensions}" ]]; then
  print_skip "No VS Code extensions installed"
  json_result "vscode" "skip" "No extensions installed"
  exit 0
fi

# Count installed extensions
extension_count=$(echo "${installed_extensions}" | wc -l | tr -d ' ')

# Mode: List
if [[ "${MODE}" == "list" ]]; then
  if ${VERBOSE}; then
    echo "    Installed extensions (${extension_count}):"
    # shellcheck disable=SC2001
    echo "${installed_extensions_raw}" | sed 's/^/      /'
  else
    print_pass "${extension_count} extensions installed"
  fi

  if is_json_mode; then
    # Build JSON array
    json_array="["
    first=true
    while IFS= read -r ext; do
      if ${first}; then
        json_array+="\"${ext}\""
        first=false
      else
        json_array+=",\"${ext}\""
      fi
    done <<< "${installed_extensions_raw}"
    json_array+="]"

    printf '{"tool":"vscode","status":"pass","count":%d,"extensions":%s}\n' \
      "${extension_count}" "${json_array}"
  else
    json_result "vscode" "pass" "${extension_count} extensions installed"
  fi

  exit 0
fi

# For check and sync modes, we need extensions.json
vscode_dir="${REPO_ROOT}/.vscode"
extensions_file="${vscode_dir}/extensions.json"

if [[ ! -f "${extensions_file}" ]]; then
  print_skip "No .vscode/extensions.json found (nothing to check)"
  json_result "vscode" "skip" "No extensions.json"
  exit 0
fi

# Parse recommended extensions from extensions.json (JSONC-aware)
# Normalize to lowercase for case-insensitive comparison
parse_extensions_json() {
  local file="$1"

  # Try jq first (most reliable)
  if command -v jq &> /dev/null; then
    # Strip comments and parse with jq
    sed 's|//.*||g' "${file}" | jq -r '.recommendations[]' 2> /dev/null | tr '[:upper:]' '[:lower:]'
    return $?
  fi

  # Fallback: grep-based parsing
  grep -E '"[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"' "${file}" \
    | sed 's|//.*||g' \
    | grep -oE '"[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"' \
    | tr -d '"' \
    | tr '[:upper:]' '[:lower:]' \
    | sort -u
}

recommended_extensions=$(parse_extensions_json "${extensions_file}" | sort)

if [[ -z "${recommended_extensions}" ]]; then
  print_skip "No recommended extensions in extensions.json"
  json_result "vscode" "skip" "No recommendations"
  exit 0
fi

recommended_count=$(echo "${recommended_extensions}" | wc -l | tr -d ' ')

# Check which recommended extensions are missing
missing_extensions=()
while IFS= read -r ext; do
  if ! echo "${installed_extensions}" | grep -q "^${ext}$"; then
    missing_extensions+=("${ext}")
  fi
done <<< "${recommended_extensions}"

# Mode: Check (only missing)
if [[ "${MODE}" == "check" ]]; then
  if [[ ${#missing_extensions[@]} -eq 0 ]]; then
    print_pass "All ${recommended_count} recommended extensions installed"

    if ${VERBOSE}; then
      echo "    Installed (${extension_count} total):"
      # shellcheck disable=SC2001
      echo "${installed_extensions_raw}" | sed 's/^/      /'
    fi

    json_result "vscode" "pass" "All recommended extensions installed"
    exit 0
  else
    missing_count=${#missing_extensions[@]}
    print_fail "Missing ${missing_count} of ${recommended_count} recommended extensions"

    if ${VERBOSE}; then
      echo "    Missing extensions:"
      for ext in "${missing_extensions[@]}"; do
        echo "      - ${ext}"
      done
      echo ""
      echo "    Install with:"
      for ext in "${missing_extensions[@]}"; do
        echo "      code --install-extension ${ext}"
      done
    else
      echo "    Run with -v to see missing extensions"
    fi

    if is_json_mode; then
      # Build JSON array of missing extensions
      json_missing="["
      first=true
      for ext in "${missing_extensions[@]}"; do
        if ${first}; then
          json_missing+="\"${ext}\""
          first=false
        else
          json_missing+=",\"${ext}\""
        fi
      done
      json_missing+="]"

      printf '{"tool":"vscode","status":"fail","missing_count":%d,"missing":%s}\n' \
        "${missing_count}" "${json_missing}"
    else
      json_result "vscode" "fail" "Missing ${missing_count} extensions"
    fi

    exit 1
  fi
fi

# Mode: Sync (bidirectional - missing AND extra)
if [[ "${MODE}" == "sync" ]]; then
  # Find extra extensions (installed but not recommended)
  extra_extensions=()
  while IFS= read -r ext; do
    if ! echo "${recommended_extensions}" | grep -q "^${ext}$"; then
      extra_extensions+=("${ext}")
    fi
  done <<< "${installed_extensions}"

  missing_count=${#missing_extensions[@]}
  extra_count=${#extra_extensions[@]}

  has_issues=false

  # Report missing
  if [[ ${missing_count} -gt 0 ]]; then
    has_issues=true
    print_fail "Missing ${missing_count} recommended extensions"
    if ${VERBOSE}; then
      for ext in "${missing_extensions[@]}"; do
        echo "      - ${ext}"
      done
    fi
  fi

  # Report extra
  if [[ ${extra_count} -gt 0 ]]; then
    has_issues=true
    echo ""
    print_info "Found ${extra_count} extensions not in recommendations:"
    if ${VERBOSE}; then
      for ext in "${extra_extensions[@]}"; do
        echo "      + ${ext}"
      done
      echo ""
      echo "    To add to recommendations, update .vscode/extensions.json"
      echo "    To uninstall, run:"
      for ext in "${extra_extensions[@]}"; do
        echo "      code --uninstall-extension ${ext}"
      done
    else
      echo "    Run with -v to see details"
    fi
  fi

  # Summary
  echo ""
  if ! ${has_issues}; then
    print_pass "Extensions fully synchronized (${recommended_count} recommended, ${extension_count} installed)"
    json_result "vscode" "pass" "Synchronized"
    exit 0
  else
    echo "    Summary: ${missing_count} missing, ${extra_count} extra"

    if is_json_mode; then
      # Build JSON arrays
      json_missing="["
      first=true
      for ext in "${missing_extensions[@]}"; do
        if ${first}; then
          json_missing+="\"${ext}\""
          first=false
        else
          json_missing+=",\"${ext}\""
        fi
      done
      json_missing+="]"

      json_extra="["
      first=true
      for ext in "${extra_extensions[@]}"; do
        if ${first}; then
          json_extra+="\"${ext}\""
          first=false
        else
          json_extra+=",\"${ext}\""
        fi
      done
      json_extra+="]"

      printf '{"tool":"vscode","status":"fail","missing_count":%d,"missing":%s,"extra_count":%d,"extra":%s}\n' \
        "${missing_count}" "${json_missing}" "${extra_count}" "${json_extra}"
    else
      json_result "vscode" "fail" "${missing_count} missing, ${extra_count} extra"
    fi

    exit 1
  fi
fi
