#!/usr/bin/env bash
# =============================================================================
# Directory Tree Generator for Documentation
# =============================================================================
# Generates directory trees with folders first, then files, sorted alphabetically
# Safe: Only reads directories, never modifies files
# Usage: ./scripts/generate_tree.sh [path] [--depth N] [--exclude pattern]
# =============================================================================

set -euo pipefail

# Defaults
TARGET_PATH="."
MAX_DEPTH=3
EXCLUDE_PATTERNS=(".git" ".venv" "__pycache__" "node_modules" ".mypy_cache" ".ruff_cache" ".pytest_cache")

# Tree drawing characters
TEE="├──"
ELBOW="└──"
SPACE="    "
PIPE_SPACE="│   "

# =============================================================================
# Helper Functions
# =============================================================================

show_help() {
  echo "Usage: $0 [path] [options]"
  echo ""
  echo "Options:"
  echo "  --depth N       Maximum depth to traverse (default: 3)"
  echo "  --exclude PAT   Add pattern to exclude list"
  echo "  --no-default    Don't exclude default patterns (.git, .venv, etc.)"
  echo "  -h, --help      Show this help message"
  echo ""
  echo "Output:"
  echo "  Generates a directory tree with:"
  echo "  - Directories listed first, then files"
  echo "  - All entries sorted alphabetically"
  echo ""
  echo "Examples:"
  echo "  $0                          # Tree of current directory"
  echo "  $0 /path/to/dir             # Tree of specific directory"
  echo "  $0 --depth 5                # Deeper traversal"
  echo "  $0 --exclude '*.log'        # Add exclusion"
}

is_excluded() {
  local name=$1
  for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ "${name}" == "${pattern}" ]] || [[ "${name}" == *"${pattern}"* ]]; then
      return 0
    fi
  done
  return 1
}

# =============================================================================
# Tree Generation (Folders First, Alphabetically Sorted)
# =============================================================================

generate_tree() {
  local dir=$1
  local prefix=$2
  local depth=$3
  local max_depth=$4

  # Check depth limit
  if [[ ${depth} -gt ${max_depth} ]]; then
    return
  fi

  # Get directories and files separately, sorted alphabetically
  local dirs=()
  local files=()

  while IFS= read -r -d '' entry; do
    local name
    name=$(basename "${entry}")

    # Skip excluded patterns
    if is_excluded "${name}"; then
      continue
    fi

    if [[ -d "${entry}" ]]; then
      dirs+=("${entry}")
    else
      files+=("${entry}")
    fi
  done < <(find "${dir}" -maxdepth 1 -mindepth 1 -print0 2> /dev/null | sort -z)

  # Combine: directories first, then files
  local all_entries=("${dirs[@]}" "${files[@]}")
  local total=${#all_entries[@]}
  local count=0

  for entry in "${all_entries[@]}"; do
    ((++count)) # Pre-increment to avoid exit code 1 when count starts at 0
    local name is_last connector new_prefix
    name=$(basename "${entry}")
    is_last=$([[ ${count} -eq ${total} ]] && echo "true" || echo "false")
    connector=$([[ "${is_last}" == "true" ]] && echo "${ELBOW}" || echo "${TEE}")
    new_prefix=$([[ "${is_last}" == "true" ]] && echo "${prefix}${SPACE}" || echo "${prefix}${PIPE_SPACE}")

    if [[ -d "${entry}" ]]; then
      echo "${prefix}${connector} ${name}/"
      generate_tree "${entry}" "${new_prefix}" $((depth + 1)) "${max_depth}"
    else
      echo "${prefix}${connector} ${name}"
    fi
  done
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
  case $1 in
    --depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --exclude)
      EXCLUDE_PATTERNS+=("$2")
      shift 2
      ;;
    --no-default)
      EXCLUDE_PATTERNS=()
      shift
      ;;
    -h | --help)
      show_help
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
    *)
      TARGET_PATH="$1"
      shift
      ;;
  esac
done

# Resolve path
if [[ ! -d "${TARGET_PATH}" ]]; then
  echo "Error: '${TARGET_PATH}' is not a directory"
  exit 1
fi

TARGET_PATH=$(cd "${TARGET_PATH}" && pwd)

# =============================================================================
# Generate Output
# =============================================================================

echo "$(basename "${TARGET_PATH}")/"
generate_tree "${TARGET_PATH}" "" 1 "${MAX_DEPTH}"
