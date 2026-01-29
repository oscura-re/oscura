#!/usr/bin/env bash
# shellcheck disable=SC2250,SC2292
# Post-compact recovery hook - validates critical files exist after compaction
# Enhanced version with optional checks, better recovery suggestions, and hook validation

set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
HOOKS_DIR="$PROJECT_DIR/.claude/hooks"
LOG_FILE="$HOOKS_DIR/compaction.log"

# Ensure log directory exists
mkdir -p "$HOOKS_DIR"

log() {
  echo "[$(date -Iseconds)] POST_COMPACT: $*" >> "$LOG_FILE"
}

warn() {
  echo "[$(date -Iseconds)] POST_COMPACT_WARN: $*" >> "$LOG_FILE"
  echo "WARNING: $*" >&2
}

# ============================================================
# Configuration
# ============================================================

# Critical files that SHOULD exist (warn if missing, don't fail)
CRITICAL_FILES_WARN=(
  "$PROJECT_DIR/CLAUDE.md"
)

# Critical files that MUST exist (fail if missing)
CRITICAL_FILES_REQUIRED=()

# Critical directories (warn if missing or empty, don't fail)
CRITICAL_DIRS_WARN=(
  "$PROJECT_DIR/.claude/agents"
  "$PROJECT_DIR/.claude/commands"
)

# Critical directories that MUST exist
CRITICAL_DIRS_REQUIRED=(
  "$PROJECT_DIR/.claude"
  "$PROJECT_DIR/.claude/hooks"
)

# Spec directory for coordination
SPEC_DIR="$PROJECT_DIR/.coordination/spec"

# ============================================================
# Validation Functions
# ============================================================

WARNINGS=()
ERRORS=()

check_file() {
  local file="$1"
  local required="${2:-false}"

  if [ ! -f "$file" ]; then
    if [ "$required" = "true" ]; then
      ERRORS+=("Missing required file: $file")
      log "ERROR: Missing required file: $file"
    else
      WARNINGS+=("Missing file: $file")
      warn "Missing file: $file"
    fi
    return 1
  fi
  return 0
}

check_dir() {
  local dir="$1"
  local required="${2:-false}"
  local check_empty="${3:-true}"

  if [ ! -d "$dir" ]; then
    if [ "$required" = "true" ]; then
      ERRORS+=("Missing required directory: $dir")
      log "ERROR: Missing required directory: $dir"
    else
      WARNINGS+=("Missing directory: $dir")
      warn "Missing directory: $dir"
    fi
    return 1
  elif [ "$check_empty" = "true" ] && [ -z "$(ls -A "$dir" 2> /dev/null)" ]; then
    if [ "$required" = "true" ]; then
      ERRORS+=("Empty required directory: $dir")
      log "ERROR: Empty required directory: $dir"
    else
      WARNINGS+=("Empty directory: $dir")
      warn "Empty directory: $dir"
    fi
    return 1
  fi
  return 0
}

check_hook_executable() {
  local hook="$1"
  local hook_path="$HOOKS_DIR/$hook"

  if [ -f "$hook_path" ]; then
    if [ ! -x "$hook_path" ]; then
      WARNINGS+=("Hook not executable: $hook")
      warn "Hook not executable: $hook - fixing..."
      chmod +x "$hook_path" 2> /dev/null || true
    fi
  fi
}

# ============================================================
# Run Checks
# ============================================================

# Check required files
for file in "${CRITICAL_FILES_REQUIRED[@]}"; do
  check_file "$file" "true"
done

# Check recommended files (warn only)
for file in "${CRITICAL_FILES_WARN[@]}"; do
  check_file "$file" "false" || true
done

# Check required directories
for dir in "${CRITICAL_DIRS_REQUIRED[@]}"; do
  check_dir "$dir" "true" "false" || [ $? -eq 1 ] || exit $?
done

# Check recommended directories (warn only)
for dir in "${CRITICAL_DIRS_WARN[@]}"; do
  check_dir "$dir" "false" "true" || true
done

# ============================================================
# NEW: Verify .coordination/spec/ directory
# ============================================================
if [ -d "$SPEC_DIR" ]; then
  SPEC_COUNT=$(find "$SPEC_DIR" -name "*.yaml" -type f 2> /dev/null | wc -l)
  log "Spec directory found with $SPEC_COUNT YAML files"
else
  WARNINGS+=("Spec directory missing: $SPEC_DIR")
  warn "Spec directory missing - consider restoring from git or backup"
fi

# ============================================================
# NEW: Verify hook executability
# ============================================================
HOOKS_TO_CHECK=(
  "pre_compact_cleanup.sh"
  "post_compact_recovery.sh"
  "session_cleanup.sh"
  "check_context_usage.sh"
)

for hook in "${HOOKS_TO_CHECK[@]}"; do
  check_hook_executable "$hook"
done

# ============================================================
# Count agents and commands
# ============================================================
AGENT_COUNT=0
CMD_COUNT=0

if [ -d "$PROJECT_DIR/.claude/agents" ]; then
  AGENT_COUNT=$(find "$PROJECT_DIR/.claude/agents" -name "*.md" -not -path "*/archive/*" 2> /dev/null | wc -l)
fi

if [ -d "$PROJECT_DIR/.claude/commands" ]; then
  CMD_COUNT=$(find "$PROJECT_DIR/.claude/commands" -name "*.md" -not -path "*/archive/*" 2> /dev/null | wc -l)
fi

log "Recovery check: $AGENT_COUNT agents, $CMD_COUNT commands, ${#WARNINGS[@]} warnings, ${#ERRORS[@]} errors"

# ============================================================
# Generate Output
# ============================================================

# Build recovery suggestions
RECOVERY_SUGGESTIONS=()

if [ ${#WARNINGS[@]} -gt 0 ] || [ ${#ERRORS[@]} -gt 0 ]; then
  # Check if CLAUDE.md is missing
  if [[ " ${WARNINGS[*]} " =~ CLAUDE\.md ]] || [[ " ${ERRORS[*]} " =~ CLAUDE\.md ]]; then
    RECOVERY_SUGGESTIONS+=("CLAUDE.md missing: Check git history with 'git checkout HEAD -- CLAUDE.md'")
  fi

  # Check if agents directory issue
  if [[ " ${WARNINGS[*]} " =~ agents ]] || [[ " ${ERRORS[*]} " =~ agents ]]; then
    RECOVERY_SUGGESTIONS+=("Agents missing: Restore from .claude/agents/ backup or git")
  fi

  # Check if spec directory issue
  if [[ " ${WARNINGS[*]} " =~ spec ]]; then
    RECOVERY_SUGGESTIONS+=("Spec files missing: Restore from .coordination/spec/ backup")
  fi
fi

# Only fail if there are actual errors (required items missing)
if [ ${#ERRORS[@]} -gt 0 ]; then
  # Output error JSON for Claude to see
  cat << EOF
{
    "ok": false,
    "error": "critical_files_missing",
    "errors": $(printf '%s\n' "${ERRORS[@]}" | jq -R . | jq -s .),
    "warnings": $(printf '%s\n' "${WARNINGS[@]}" | jq -R . 2> /dev/null | jq -s . 2> /dev/null || echo '[]'),
    "recovery_suggestions": $(printf '%s\n' "${RECOVERY_SUGGESTIONS[@]}" | jq -R . 2> /dev/null | jq -s . 2> /dev/null || echo '[]'),
    "agents": $AGENT_COUNT,
    "commands": $CMD_COUNT
}
EOF
  exit 1
fi

# ============================================================
# Record compaction metrics and generate summaries (v2.0 enhancements)
# ============================================================

# Record compaction event in metrics
if [ -f "$HOOKS_DIR/manage_agent_registry.py" ]; then
  python3 "$HOOKS_DIR/manage_agent_registry.py" record-compaction 2> /dev/null || true

  # Generate any missing summaries after recovery (helps with context restoration)
  SUMMARY_RESULT=$(python3 "$HOOKS_DIR/manage_agent_registry.py" generate-all-summaries 2> /dev/null || echo '{}')
  SUMMARIES_GENERATED=$(echo "$SUMMARY_RESULT" | jq -r '.generated // 0' 2> /dev/null || echo 0)
  if [ "$SUMMARIES_GENERATED" -gt 0 ]; then
    log "Generated $SUMMARIES_GENERATED missing summaries"
  fi
fi

# Success (possibly with warnings)
log "Post-compact recovery complete"

if [ ${#WARNINGS[@]} -gt 0 ]; then
  cat << EOF
{
    "ok": true,
    "warnings": $(printf '%s\n' "${WARNINGS[@]}" | jq -R . 2> /dev/null | jq -s . 2> /dev/null || echo '[]'),
    "recovery_suggestions": $(printf '%s\n' "${RECOVERY_SUGGESTIONS[@]}" | jq -R . 2> /dev/null | jq -s . 2> /dev/null || echo '[]'),
    "agents": $AGENT_COUNT,
    "commands": $CMD_COUNT,
    "message": "Post-compact recovery complete with ${#WARNINGS[@]} warnings"
}
EOF
else
  echo '{"ok": true, "agents": '"$AGENT_COUNT"', "commands": '"$CMD_COUNT"', "message": "All critical files present"}'
fi

exit 0
