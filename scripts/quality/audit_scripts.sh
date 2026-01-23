#!/usr/bin/env bash
# =============================================================================
# audit_scripts.sh - Script Interface Audit
# =============================================================================
# Usage: ./scripts/audit_scripts.sh [-h|--help]
# =============================================================================
# Analyzes all scripts for consistency and best practices

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h | --help)
      echo "Usage: $0 [-h|--help]"
      echo ""
      echo "Audit scripts for consistency and best practices"
      echo ""
      echo "Checks:"
      echo "  - Usage line present"
      echo "  - --help flag supported"
      echo "  - --json flag for tool scripts"
      echo "  - -v/--verbose flag"
      echo "  - Sources common.sh"
      echo "  - set -euo pipefail"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

print_header "SCRIPT INTERFACE AUDIT"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

# Find all shell scripts
mapfile -t scripts < <(find "${SCRIPT_DIR}" -type f -name "*.sh" ! -name "audit_scripts.sh" | sort)

echo ""
echo -e "  ${DIM}Found ${#scripts[@]} shell scripts${NC}"

# Analysis categories
declare -A has_usage
declare -A has_help
declare -A has_json
declare -A has_check_fix
declare -A has_verbose
declare -A sources_common
declare -A has_set_euo

# Analyze each script
for script in "${scripts[@]}"; do
  name="${script#"${SCRIPT_DIR}"/}"

  # Check for usage line
  if grep -q "^# Usage:" "${script}"; then
    has_usage["${name}"]=1
  fi

  # Check for --help
  if grep -q -- "--help" "${script}"; then
    has_help["${name}"]=1
  fi

  # Check for --json
  if grep -q -- "--json" "${script}"; then
    has_json["${name}"]=1
  fi

  # Check for --check/--fix
  if grep -qE -- "--(check|fix|format)" "${script}"; then
    has_check_fix["${name}"]=1
  fi

  # Check for -v/--verbose
  if grep -qE -- "(-v|--verbose)" "${script}"; then
    has_verbose["${name}"]=1
  fi

  # Check for sourcing common.sh
  if grep -q "source.*common.sh" "${script}"; then
    sources_common["${name}"]=1
  fi

  # Check for set -euo pipefail
  if grep -q "set -euo pipefail" "${script}"; then
    has_set_euo["${name}"]=1
  fi
done

# Helper function to count category
count_category() {
  local -n assoc=$1
  local count=0
  for key in "${!assoc[@]}"; do
    ((count++)) || true
  done
  echo "${count}"
}

# Print statistics
print_header "STATISTICS"
echo ""
printf "  %-20s %s/%d\n" "Usage line:" "$(count_category has_usage)" "${#scripts[@]}"
printf "  %-20s %s/%d\n" "--help flag:" "$(count_category has_help)" "${#scripts[@]}"
printf "  %-20s %s/%d\n" "--json flag:" "$(count_category has_json)" "${#scripts[@]}"
printf "  %-20s %s/%d\n" "--check/--fix:" "$(count_category has_check_fix)" "${#scripts[@]}"
printf "  %-20s %s/%d\n" "-v/--verbose:" "$(count_category has_verbose)" "${#scripts[@]}"
printf "  %-20s %s/%d\n" "Sources common.sh:" "$(count_category sources_common)" "${#scripts[@]}"
printf "  %-20s %s/%d\n" "set -euo pipefail:" "$(count_category has_set_euo)" "${#scripts[@]}"

# Report missing features
print_header "MISSING FEATURES"

print_section "Missing Usage Line"
for script in "${scripts[@]}"; do
  name="${script#"${SCRIPT_DIR}"/}"
  if [[ -z "${has_usage[${name}]:-}" ]]; then
    print_info "${name}"
  fi
done

print_section "Missing --help"
for script in "${scripts[@]}"; do
  name="${script#"${SCRIPT_DIR}"/}"
  if [[ -z "${has_help[${name}]:-}" ]]; then
    print_info "${name}"
  fi
done

print_section "Missing set -euo pipefail"
for script in "${scripts[@]}"; do
  name="${script#"${SCRIPT_DIR}"/}"
  if [[ -z "${has_set_euo[${name}]:-}" ]]; then
    print_fail "${name}"
  fi
done

# Tool script table
print_header "TOOL SCRIPT INTERFACES"
echo ""
echo -e "  ${DIM}Script                     Usage  Help  JSON  Mode  -v  Common${NC}"
echo -e "  ${DIM}------------------------   -----  ----  ----  ----  --  ------${NC}"

for script in "${scripts[@]}"; do
  name="${script#"${SCRIPT_DIR}"/}"
  # Only show tools/*
  if [[ "${name}" != tools/* ]]; then
    continue
  fi

  short_name="${name#tools/}"
  printf "  %-24s   " "${short_name}"
  printf "%s     " "${has_usage[${name}]:+Y}"
  printf "%s     " "${has_help[${name}]:+Y}"
  printf "%s     " "${has_json[${name}]:+Y}"
  printf "%s     " "${has_check_fix[${name}]:+Y}"
  printf "%s   " "${has_verbose[${name}]:+Y}"
  printf "%s" "${sources_common[${name}]:+Y}"
  echo ""
done

# Summary
print_header "SUMMARY"
echo ""

total_euo=$(count_category has_set_euo)
if [[ ${total_euo} -eq ${#scripts[@]} ]]; then
  echo -e "  ${GREEN}${BOLD}All scripts use strict mode!${NC}"
else
  missing=$((${#scripts[@]} - total_euo))
  echo -e "  ${YELLOW}${BOLD}${missing} script(s) missing strict mode${NC}"
fi
