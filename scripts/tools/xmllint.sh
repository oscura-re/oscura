#!/usr/bin/env bash
# =============================================================================
# xmllint.sh - XML Validation with xmllint
# =============================================================================
# Usage: ./scripts/tools/xmllint.sh [--check|--format] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="xmllint"
TOOL_CMD="xmllint"
INSTALL_HINT="apt install libxml2-utils"

# Defaults
MODE="check"
VERBOSE=false
PATHS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      MODE="check"
      shift
      ;;
    --format)
      MODE="format"
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
      echo "XML validation and formatting with xmllint"
      echo ""
      echo "Options:"
      echo "  --check       Validate XML syntax (default)"
      echo "  --format      Format XML files in place"
      echo "  --json        Output machine-readable JSON"
      echo "  -v            Verbose output"
      echo "  -h, --help    Show this help message"
      echo ""
      echo "File patterns: *.xml, *.xsd, *.xsl, *.xslt, *.svg"
      echo ""
      echo "Exit codes:"
      echo "  0 - All files valid / formatted"
      echo "  1 - Validation errors found"
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

# Find XML files
xml_files=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" ]]; then
    xml_files+=("${path}")
  elif [[ -d "${path}" ]]; then
    while IFS= read -r -d '' file; do
      xml_files+=("${file}")
    done < <(find "${path}" -type f \( -name "*.xml" -o -name "*.xsd" -o -name "*.xsl" -o -name "*.xslt" \) \
      -not -path "*/.git/*" -not -path "*/.venv/*" -not -path "*/node_modules/*" -print0 2> /dev/null)
  fi
done

if [[ ${#xml_files[@]} -eq 0 ]]; then
  print_skip "No XML files found"
  json_result "${TOOL_CMD}" "skip" "No files found"
  exit 0
fi

file_count=${#xml_files[@]}
${VERBOSE} && print_info "Found ${file_count} XML file(s)"

# Mode: Check (validation)
if [[ "${MODE}" == "check" ]]; then
  has_errors=false
  error_count=0

  for file in "${xml_files[@]}"; do
    if ${VERBOSE}; then
      if ! xmllint --noout "${file}" 2>&1; then
        has_errors=true
        ((error_count++)) || true
      fi
    else
      if ! xmllint --noout "${file}" 2> /dev/null; then
        has_errors=true
        ((error_count++)) || true
        print_fail "Invalid: $(basename "${file}")"
      fi
    fi
  done

  if ${has_errors}; then
    print_fail "${error_count} file(s) with errors"
    json_result "${TOOL_CMD}" "fail" "${error_count} errors"
    exit 1
  else
    print_pass "All ${file_count} XML files valid"
    json_result "${TOOL_CMD}" "pass" ""
    exit 0
  fi
fi

# Mode: Format
if [[ "${MODE}" == "format" ]]; then
  formatted_count=0

  for file in "${xml_files[@]}"; do
    # Format in place
    temp_file=$(mktemp)
    if xmllint --format "${file}" > "${temp_file}" 2> /dev/null; then
      if ! diff -q "${file}" "${temp_file}" &> /dev/null; then
        mv "${temp_file}" "${file}"
        ((formatted_count++)) || true
        ${VERBOSE} && echo "    Formatted: $(basename "${file}")"
      else
        rm "${temp_file}"
        ${VERBOSE} && echo "    Unchanged: $(basename "${file}")"
      fi
    else
      rm -f "${temp_file}"
      print_fail "Format failed: $(basename "${file}")"
    fi
  done

  print_pass "Formatted ${formatted_count} of ${file_count} files"
  json_result "${TOOL_CMD}" "pass" "Formatted ${formatted_count} files"
  exit 0
fi
