#!/usr/bin/env bash
# =============================================================================
# jsonc.sh - JSON/JSONC Validation and Formatting Tool
# =============================================================================
# Validates JSON files and JSONC (JSON with Comments) files used in VS Code
# configurations. Supports schema validation when check-jsonschema is available.
#
# Usage: ./scripts/tools/jsonc.sh [OPTIONS] [paths...]
#
# Examples:
#   ./scripts/tools/jsonc.sh --check                    # Validate all JSON files
#   ./scripts/tools/jsonc.sh --check .vscode/           # Validate VS Code configs
#   ./scripts/tools/jsonc.sh --schema schema.json file  # Validate with schema
#   ./scripts/tools/jsonc.sh --check --json             # JSON output for CI
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="JSON/JSONC Validator"
INSTALL_HINT="jq (apt install jq), python3, or check-jsonschema (uv pip install check-jsonschema)"

# Defaults
MODE="check"
VERBOSE=false
SCHEMA_FILE=""
PATHS=()

# Results tracking
VALIDATED=0
ISSUES=0
SKIPPED_FILES=0

# =============================================================================
# Help
# =============================================================================

show_script_help() {
  cat << 'EOF'
jsonc.sh - JSON/JSONC validation and formatting tool

USAGE:
    jsonc.sh [OPTIONS] [paths...]

DESCRIPTION:
    Validates JSON files for syntax errors and optionally against JSON schemas.
    Supports JSONC (JSON with Comments) files commonly used in VS Code
    configurations (.vscode/*.json).

    Detection hierarchy:
    1. jq (fast, reliable JSON validation)
    2. python3 json module (fallback, always available)
    3. check-jsonschema (required for schema validation)

OPTIONS:
    --check         Validate JSON files (default)
    --schema FILE   Validate against JSON schema (requires check-jsonschema)
    --json          Output machine-readable JSON
    -v, --verbose   Show detailed validation output
    -h, --help      Show this help message

    If no paths given, searches current directory for JSON files.

FILE TYPES:
    .json           Standard JSON files (validated as-is)
    .jsonc          JSON with Comments (comments stripped before validation)
    .vscode/*.json  VS Code settings (treated as JSONC)

REQUIREMENTS:
    Required (one of):
        jq          - Fast JSON processor (recommended)
        python3     - Python with json module (fallback)

    Optional:
        check-jsonschema - For schema validation
                          Install: uv pip install check-jsonschema

EXAMPLES:
    # Validate all JSON files in repository
    jsonc.sh --check

    # Validate specific file
    jsonc.sh --check package.json

    # Validate VS Code settings (JSONC)
    jsonc.sh --check .vscode/settings.json

    # Validate with schema
    jsonc.sh --check --schema tsconfig.schema.json tsconfig.json

    # Verbose output showing each file
    jsonc.sh --check -v

    # JSON output for CI pipelines
    jsonc.sh --check --json

EXIT CODES:
    0  All files valid
    1  Validation errors found
    2  Configuration error (missing tools, invalid options)

SEE ALSO:
    scripts/tools/yamllint.sh   - YAML validation
    scripts/tools/prettier.sh   - JSON/YAML formatting
    scripts/check.sh            - Run all quality checks
EOF
}

# =============================================================================
# JSONC Processing
# =============================================================================

# Strip comments from JSONC content for validation
# Handles // line comments and /* block comments */
strip_jsonc_comments() {
  local file="$1"

  if has_tool "python3"; then
    python3 -c "
import re
import sys

content = open(sys.argv[1], 'r').read()

# Remove single-line comments (// ...)
# Be careful not to remove // inside strings
result = []
in_string = False
i = 0
while i < len(content):
    char = content[i]

    if char == '\"' and (i == 0 or content[i-1] != '\\\\'):
        in_string = not in_string
        result.append(char)
    elif not in_string and content[i:i+2] == '//':
        # Skip to end of line
        while i < len(content) and content[i] != '\n':
            i += 1
        continue
    elif not in_string and content[i:i+2] == '/*':
        # Skip to end of block comment
        i += 2
        while i < len(content) - 1 and content[i:i+2] != '*/':
            i += 1
        i += 2
        continue
    else:
        result.append(char)
    i += 1

print(''.join(result))
" "${file}"
  else
    # Fallback: simple sed-based comment removal (less accurate)
    sed 's|//.*$||g; s|/\*.*\*/||g' "${file}"
  fi
}

# Determine if file should be treated as JSONC
is_jsonc_file() {
  local file="$1"

  # Explicit .jsonc extension
  [[ "${file}" == *.jsonc ]] && return 0

  # VS Code configuration files are JSONC
  [[ "${file}" == */.vscode/*.json ]] && return 0
  [[ "${file}" == .vscode/*.json ]] && return 0

  # tsconfig and jsconfig often have comments
  [[ "$(basename "${file}")" == "tsconfig.json" ]] && return 0
  [[ "$(basename "${file}")" == "jsconfig.json" ]] && return 0

  return 1
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_json_with_jq() {
  local file="$1"
  local content="$2"

  if [[ -n "${content}" ]]; then
    echo "${content}" | jq empty 2>&1
  else
    jq empty "${file}" 2>&1
  fi
}

validate_json_with_python() {
  local file="$1"
  local content="$2"

  if [[ -n "${content}" ]]; then
    echo "${content}" | python3 -c "import json, sys; json.load(sys.stdin)" 2>&1
  else
    python3 -c "import json; json.load(open('${file}'))" 2>&1
  fi
}

validate_with_schema() {
  local file="$1"
  local schema="$2"

  if ! has_tool "check-jsonschema"; then
    print_skip "Schema validation requires check-jsonschema"
    return 1
  fi

  check-jsonschema --schemafile "${schema}" "${file}" 2>&1
}

validate_file() {
  local file="$1"
  local result=""
  local content=""

  # Handle JSONC files
  if is_jsonc_file "${file}"; then
    content=$(strip_jsonc_comments "${file}")
    if ${VERBOSE}; then
      print_info "Treating as JSONC: ${file}"
    fi
  fi

  # Validate with best available tool
  if has_tool "jq"; then
    result=$(validate_json_with_jq "${file}" "${content}" 2>&1) || true
  elif has_tool "python3"; then
    result=$(validate_json_with_python "${file}" "${content}" 2>&1) || true
  else
    print_fail "No JSON validator available"
    return 2
  fi

  # Check result
  if [[ -z "${result}" ]]; then
    ((VALIDATED++)) || true
    if ${VERBOSE}; then
      print_pass "${file}"
    fi
    return 0
  else
    ((ISSUES++)) || true
    if ${VERBOSE}; then
      print_fail "${file}"
      print_info "${result}"
    fi
    return 1
  fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      MODE="check"
      shift
      ;;
    --schema)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --schema requires a file argument" >&2
        exit 2
      fi
      SCHEMA_FILE="$2"
      shift 2
      ;;
    --json)
      enable_json
      shift
      ;;
    -v | --verbose)
      VERBOSE=true
      shift
      ;;
    -h | --help)
      show_script_help
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Use -h for help" >&2
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

# =============================================================================
# Main
# =============================================================================

print_tool "${TOOL_NAME} (${MODE})"

# Check for available validator
if ! has_tool "jq" && ! has_tool "python3"; then
  print_skip "No JSON validator available (${INSTALL_HINT})"
  json_result "jsonc" "skip" "No validator available"
  exit 0
fi

# Report which validator is being used
if ${VERBOSE}; then
  if has_tool "jq"; then
    print_info "Using jq for validation"
  else
    print_info "Using python3 for validation"
  fi
fi

# Validate schema file if provided
if [[ -n "${SCHEMA_FILE}" ]]; then
  if [[ ! -f "${SCHEMA_FILE}" ]]; then
    print_fail "Schema file not found: ${SCHEMA_FILE}"
    json_result "jsonc" "fail" "Schema not found"
    exit 2
  fi
  if ! has_tool "check-jsonschema"; then
    print_fail "Schema validation requires check-jsonschema"
    print_info "Install: uv pip install check-jsonschema"
    json_result "jsonc" "fail" "check-jsonschema required"
    exit 2
  fi
fi

# Find JSON files
json_files=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" ]]; then
    json_files+=("${path}")
  elif [[ -d "${path}" ]]; then
    while IFS= read -r -d '' file; do
      json_files+=("${file}")
    done < <(find "${path}" -type f \( -name "*.json" -o -name "*.jsonc" \) \
      -not -path "*/.git/*" \
      -not -path "*/.venv/*" \
      -not -path "*/node_modules/*" \
      -not -path "*/__pycache__/*" \
      -not -name "package-lock.json" \
      -not -name "uv.lock" \
      -not -name "corrupt_registry.json" \
      -print0 2> /dev/null)
  fi
done

if [[ ${#json_files[@]} -eq 0 ]]; then
  print_skip "No JSON files found"
  json_result "jsonc" "skip" "No files found"
  exit 0
fi

if ${VERBOSE}; then
  print_info "Found ${#json_files[@]} JSON file(s)"
fi

# Validate each file
validation_failed=false
for file in "${json_files[@]}"; do
  if [[ -n "${SCHEMA_FILE}" ]]; then
    # Schema validation
    if ! validate_with_schema "${file}" "${SCHEMA_FILE}"; then
      validation_failed=true
    fi
  else
    # Syntax validation
    if ! validate_file "${file}"; then
      validation_failed=true
    fi
  fi
done

# Summary
if ${VERBOSE}; then
  echo ""
  print_info "Validated: ${VALIDATED}, Issues: ${ISSUES}"
fi

# Final result
if ${validation_failed} || [[ ${ISSUES} -gt 0 ]]; then
  if ! ${VERBOSE}; then
    print_fail "JSON validation issues found"
    print_info "Run with -v for details"
  else
    print_fail "Validation failed"
  fi
  json_result "jsonc" "fail" "${ISSUES} file(s) with issues"
  exit 1
else
  print_pass "All JSON files valid (${VALIDATED} files)"
  json_result "jsonc" "pass" ""
  exit 0
fi
