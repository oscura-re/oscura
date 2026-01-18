#!/usr/bin/env bash
# =============================================================================
# mermaid.sh - Mermaid Diagram Validation and Export
# =============================================================================
# Usage: ./scripts/tools/mermaid.sh [--check|--export] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="Mermaid"
TOOL_CMD="mmdc"
INSTALL_HINT="npm install -g @mermaid-js/mermaid-cli"

# Defaults
MODE="check"
VERBOSE=false
PATHS=()
OUTPUT_FORMAT="png"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      MODE="check"
      shift
      ;;
    --export)
      MODE="export"
      shift
      ;;
    --png)
      OUTPUT_FORMAT="png"
      shift
      ;;
    --svg)
      OUTPUT_FORMAT="svg"
      shift
      ;;
    --pdf)
      OUTPUT_FORMAT="pdf"
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
      echo "Usage: $0 [OPTIONS] [paths...]"
      echo ""
      echo "Mermaid diagram validation and export"
      echo ""
      echo "Options:"
      echo "  --check       Validate Mermaid syntax (default)"
      echo "  --export      Export diagrams to images"
      echo "  --png         Export as PNG (default for --export)"
      echo "  --svg         Export as SVG"
      echo "  --pdf         Export as PDF"
      echo "  --json        Output machine-readable JSON"
      echo "  -v            Verbose output"
      echo "  -h, --help    Show this help message"
      echo ""
      echo "File patterns: *.mmd, *.mermaid"
      echo ""
      echo "Exit codes:"
      echo "  0 - All diagrams valid / exported"
      echo "  1 - Syntax errors found"
      echo "  2 - Tool not installed"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      PATHS+=("$1")
      shift
      ;;
  esac
done

# Default to current directory
[[ ${#PATHS[@]} -eq 0 ]] && PATHS=(".")

# Main
print_tool "${TOOL_NAME} (${MODE})"

# Check tool installed
if ! require_tool "${TOOL_CMD}" "${INSTALL_HINT}"; then
  json_result "mermaid" "skip" "Tool not installed"
  exit 0
fi

# Find Mermaid files
mmd_files=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" ]]; then
    mmd_files+=("${path}")
  elif [[ -d "${path}" ]]; then
    while IFS= read -r -d '' file; do
      mmd_files+=("${file}")
    done < <(find "${path}" -type f \( -name "*.mmd" -o -name "*.mermaid" \) \
      -not -path "*/.git/*" -not -path "*/.venv/*" -not -path "*/node_modules/*" -print0 2> /dev/null)
  fi
done

if [[ ${#mmd_files[@]} -eq 0 ]]; then
  print_skip "No Mermaid files found"
  json_result "mermaid" "skip" "No files found"
  exit 0
fi

file_count=${#mmd_files[@]}
${VERBOSE} && print_info "Found ${file_count} Mermaid file(s)"

# Create temp directory for validation outputs
TEMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# Mode: Check (syntax validation)
if [[ "${MODE}" == "check" ]]; then
  has_errors=false
  error_count=0

  for file in "${mmd_files[@]}"; do
    # mmdc validates by attempting to render to /dev/null or temp file
    temp_output="${TEMP_DIR}/$(basename "${file}").png"
    if ${VERBOSE}; then
      if ! mmdc -i "${file}" -o "${temp_output}" 2>&1; then
        has_errors=true
        ((error_count++)) || true
      fi
    else
      if ! mmdc -i "${file}" -o "${temp_output}" &> /dev/null; then
        has_errors=true
        ((error_count++)) || true
        print_fail "Syntax error: $(basename "${file}")"
      fi
    fi
  done

  if ${has_errors}; then
    print_fail "${error_count} file(s) with syntax errors"
    json_result "mermaid" "fail" "${error_count} syntax errors"
    exit 1
  else
    print_pass "All ${file_count} Mermaid files valid"
    json_result "mermaid" "pass" ""
    exit 0
  fi
fi

# Mode: Export
if [[ "${MODE}" == "export" ]]; then
  has_errors=false
  exported_count=0

  for file in "${mmd_files[@]}"; do
    # Generate output filename next to input
    output_file="${file%.*}.${OUTPUT_FORMAT}"

    if ${VERBOSE}; then
      echo "    Exporting: $(basename "${file}") -> $(basename "${output_file}")"
      if mmdc -i "${file}" -o "${output_file}" -e "${OUTPUT_FORMAT}"; then
        ((exported_count++)) || true
      else
        has_errors=true
      fi
    else
      if mmdc -i "${file}" -o "${output_file}" -e "${OUTPUT_FORMAT}" &> /dev/null; then
        ((exported_count++)) || true
      else
        has_errors=true
        print_fail "Export failed: $(basename "${file}")"
      fi
    fi
  done

  if ${has_errors}; then
    print_fail "Some exports failed"
    json_result "mermaid" "fail" "Export errors"
    exit 1
  else
    print_pass "Exported ${exported_count} file(s) to ${OUTPUT_FORMAT}"
    json_result "mermaid" "pass" "Exported ${exported_count} files"
    exit 0
  fi
fi
