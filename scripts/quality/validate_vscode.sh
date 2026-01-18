#!/usr/bin/env bash
# =============================================================================
# validate_vscode.sh - VS Code Configuration Validation
# =============================================================================
# Usage: ./scripts/validate_vscode.sh [-h|--help] [-v|--verbose] [--skip-schema]
# =============================================================================
# Validates .vscode/ configuration files for correctness
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
VERBOSE=false
SKIP_SCHEMA=false

# VS Code settings schema URL
VSCODE_SETTINGS_SCHEMA="https://raw.githubusercontent.com/wraith13/vscode-schemas/master/en/latest/schemas/settings/machine.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h | --help)
      echo "Usage: $0 [-h|--help] [-v|--verbose] [--skip-schema]"
      echo ""
      echo "VS Code configuration validation"
      echo ""
      echo "Validates .vscode/ configuration files for correctness:"
      echo "  1. JSONC syntax (validates JSON with comments)"
      echo "  2. JSON Schema (validates against VS Code settings schema)"
      echo "  3. Extensions (cross-references installed vs recommended)"
      echo ""
      echo "Options:"
      echo "  -h, --help     Show this help message"
      echo "  -v, --verbose  Show detailed output"
      echo "  --skip-schema  Skip JSON schema validation"
      echo ""
      echo "Exit Codes:"
      echo "  0  All validations passed"
      echo "  1  Validation errors found"
      echo "  2  Script/argument error"
      exit 0
      ;;
    -v | --verbose)
      VERBOSE=true
      shift
      ;;
    --skip-schema)
      SKIP_SCHEMA=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try '$0 --help' for usage information" >&2
      exit 2
      ;;
  esac
done

VSCODE_DIR="${REPO_ROOT}/.vscode"

print_header "VS Code Configuration Validation"
echo "   Workspace: ${REPO_ROOT}"
echo ""

ERRORS=0
WARNINGS=0

# ============================================================================
# 1. Check .vscode directory exists
# ============================================================================
if [[ ! -d "${VSCODE_DIR}" ]]; then
  print_fail ".vscode/ directory not found"
  exit 1
fi

# ============================================================================
# 2. Validate JSONC files
# ============================================================================
print_section "Validating JSONC syntax"

for file in settings.json extensions.json tasks.json launch.json; do
  file_path="${VSCODE_DIR}/${file}"
  if [[ ! -f "${file_path}" ]]; then
    print_skip "${file} (not present)"
    continue
  fi

  # Use the jsonc tool script
  if "${SCRIPT_DIR}/tools/jsonc.sh" --check "${file_path}" &> /dev/null; then
    print_pass "${file}"
  else
    print_fail "${file} (invalid JSONC)"
    ((ERRORS++))
  fi
done
echo ""

# ============================================================================
# 3. JSON Schema Validation
# ============================================================================
print_section "Validating settings.json against schema"

if [[ "${SKIP_SCHEMA}" == "true" ]]; then
  print_skip "Schema validation skipped (--skip-schema)"
elif [[ ! -f "${VSCODE_DIR}/settings.json" ]]; then
  print_skip "settings.json not present"
elif ! has_tool "check-jsonschema"; then
  print_skip "check-jsonschema not installed"
  print_info "Install: uv pip install check-jsonschema"
  ((WARNINGS++))
else
  # Create temp file with comments stripped
  temp_file=$(mktemp)
  trap 'rm -f "${temp_file}"' EXIT

  # Strip comments and validate
  if python3 -c "
import json, re, sys

def strip_comments(content):
    lines = []
    for line in content.split('\n'):
        if '//' in line:
            idx = line.find('//')
            while idx != -1:
                if idx > 0 and line[idx-1] == ':':
                    idx = line.find('//', idx + 2)
                else:
                    line = line[:idx]
                    break
        lines.append(line)
    content = '\n'.join(lines)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return content

with open('${VSCODE_DIR}/settings.json', 'r') as f:
    clean = strip_comments(f.read())
    data = json.loads(clean)
    with open('${temp_file}', 'w') as out:
        json.dump(data, out)
" 2> /dev/null; then
    if ${VERBOSE}; then
      if check-jsonschema --schemafile "${VSCODE_SETTINGS_SCHEMA}" "${temp_file}" 2>&1; then
        print_pass "settings.json (schema valid)"
      else
        print_fail "settings.json (schema validation failed)"
        ((ERRORS++))
      fi
    else
      if check-jsonschema --schemafile "${VSCODE_SETTINGS_SCHEMA}" "${temp_file}" &> /dev/null; then
        print_pass "settings.json (schema valid)"
      else
        print_fail "settings.json (schema validation failed)"
        print_info "Run with -v for details"
        ((ERRORS++))
      fi
    fi
  else
    print_fail "settings.json (failed to parse)"
    ((ERRORS++))
  fi
fi
echo ""

# ============================================================================
# 4. Check extensions
# ============================================================================
print_section "Checking extensions"

if ${VERBOSE}; then
  "${SCRIPT_DIR}/tools/vscode_extensions.sh" --check -v || ((ERRORS++))
else
  "${SCRIPT_DIR}/tools/vscode_extensions.sh" --check || ((ERRORS++))
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "------------------------------------------------------------------------"
if [[ ${ERRORS} -eq 0 ]] && [[ ${WARNINGS} -eq 0 ]]; then
  print_pass "All VS Code validations passed"
  exit 0
elif [[ ${ERRORS} -eq 0 ]]; then
  print_info "Passed with ${WARNINGS} warning(s)"
  exit 0
else
  print_fail "Found ${ERRORS} validation error(s)"
  exit 1
fi
