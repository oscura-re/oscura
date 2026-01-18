#!/usr/bin/env bash
# =============================================================================
# plantuml.sh - PlantUML Diagram Validation and Export
# =============================================================================
# Usage: ./scripts/tools/plantuml.sh [--check|--export] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="PlantUML"
TOOL_CMD="plantuml"
INSTALL_HINT="apt install plantuml / brew install plantuml"

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
      echo "PlantUML diagram validation and export"
      echo ""
      echo "Options:"
      echo "  --check       Validate PlantUML syntax (default)"
      echo "  --export      Export diagrams to images"
      echo "  --png         Export as PNG (default for --export)"
      echo "  --svg         Export as SVG"
      echo "  --json        Output machine-readable JSON"
      echo "  -v            Verbose output"
      echo "  -h, --help    Show this help message"
      echo ""
      echo "File patterns: *.puml, *.plantuml, *.pu"
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
  json_result "${TOOL_CMD}" "skip" "Tool not installed"
  exit 0
fi

# Find PlantUML files
puml_files=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" ]]; then
    puml_files+=("${path}")
  elif [[ -d "${path}" ]]; then
    while IFS= read -r -d '' file; do
      puml_files+=("${file}")
    done < <(find "${path}" -type f \( -name "*.puml" -o -name "*.plantuml" -o -name "*.pu" \) \
      -not -path "*/.git/*" -not -path "*/.venv/*" -not -path "*/node_modules/*" -print0 2> /dev/null)
  fi
done

if [[ ${#puml_files[@]} -eq 0 ]]; then
  print_skip "No PlantUML files found"
  json_result "${TOOL_CMD}" "skip" "No files found"
  exit 0
fi

file_count=${#puml_files[@]}
${VERBOSE} && print_info "Found ${file_count} PlantUML file(s)"

# Mode: Check (syntax validation)
if [[ "${MODE}" == "check" ]]; then
  has_errors=false
  error_count=0

  for file in "${puml_files[@]}"; do
    # PlantUML -syntax flag performs syntax check without generating output
    if ${VERBOSE}; then
      if ! plantuml -syntax "${file}" 2>&1; then
        has_errors=true
        ((error_count++)) || true
      fi
    else
      if ! plantuml -syntax "${file}" &> /dev/null; then
        has_errors=true
        ((error_count++)) || true
        print_fail "Syntax error: $(basename "${file}")"
      fi
    fi
  done

  if ${has_errors}; then
    print_fail "${error_count} file(s) with syntax errors"
    json_result "${TOOL_CMD}" "fail" "${error_count} syntax errors"
    exit 1
  else
    print_pass "All ${file_count} PlantUML files valid"
    json_result "${TOOL_CMD}" "pass" ""
    exit 0
  fi
fi

# Mode: Export
if [[ "${MODE}" == "export" ]]; then
  has_errors=false
  exported_count=0

  for file in "${puml_files[@]}"; do
    if ${VERBOSE}; then
      echo "    Exporting: $(basename "${file}")"
      if plantuml -t"${OUTPUT_FORMAT}" "${file}"; then
        ((exported_count++)) || true
      else
        has_errors=true
      fi
    else
      if plantuml -t"${OUTPUT_FORMAT}" "${file}" &> /dev/null; then
        ((exported_count++)) || true
      else
        has_errors=true
        print_fail "Export failed: $(basename "${file}")"
      fi
    fi
  done

  if ${has_errors}; then
    print_fail "Some exports failed"
    json_result "${TOOL_CMD}" "fail" "Export errors"
    exit 1
  else
    print_pass "Exported ${exported_count} file(s) to ${OUTPUT_FORMAT}"
    json_result "${TOOL_CMD}" "pass" "Exported ${exported_count} files"
    exit 0
  fi
fi
