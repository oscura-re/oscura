#!/usr/bin/env bash
# shellcheck disable=SC2250,SC2292
set -e

# SessionEnd Cleanup Hook
# Performs cleanup tasks when a session ends
# Removes temporary files, stale locks, and orphaned chunks

CLAUDE_PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
LOG_FILE="$CLAUDE_PROJECT_DIR/.claude/hooks/sessions.log"
COORD_DIR="$CLAUDE_PROJECT_DIR/.coordination"

# Ensure log directory exists
mkdir -p "$CLAUDE_PROJECT_DIR/.claude/hooks"

# Remove temporary files (comprehensive patterns)
TEMP_COUNT=0
if [ -d "$COORD_DIR" ]; then
  TEMP_COUNT=$(find "$COORD_DIR" \( \
    -name "*.tmp" -o \
    -name "*.temp" -o \
    -name "*.bak" -o \
    -name "*.backup" -o \
    -name "*.partial" -o \
    -name "*.swp" -o \
    -name "*~" \
    \) 2> /dev/null | wc -l)
  find "$COORD_DIR" \( \
    -name "*.tmp" -o \
    -name "*.temp" -o \
    -name "*.bak" -o \
    -name "*.backup" -o \
    -name "*.partial" -o \
    -name "*.swp" -o \
    -name "*~" \
    \) -delete 2> /dev/null || true
fi

# Remove expired lock files (check expires_at field in JSON)
LOCK_COUNT=0
if [ -d "$COORD_DIR/locks" ]; then
  shopt -s nullglob
  for lock_file in "$COORD_DIR/locks"/*.json; do
    [ -f "$lock_file" ] || continue

    # Try to read expires_at from JSON (handle corrupted JSON gracefully)
    expires_at=""
    if command -v jq &> /dev/null; then
      expires_at=$(jq -r '.expires_at // empty' "$lock_file" 2> /dev/null || true)
    fi

    if [ -n "$expires_at" ]; then
      # Convert ISO-8601 to epoch and compare
      expires_epoch=$(date -d "$expires_at" +%s 2> /dev/null || echo 0)
      now_epoch=$(date +%s)

      if [ "$expires_epoch" -gt 0 ] && [ "$now_epoch" -gt "$expires_epoch" ]; then
        # Lock has expired based on expires_at field
        rm -f "$lock_file"
        LOCK_COUNT=$((LOCK_COUNT + 1))
      fi
    else
      # No expires_at field or invalid JSON - fallback to mtime (>60 minutes)
      if [ -n "$(find "$lock_file" -mmin +60 2> /dev/null)" ]; then
        rm -f "$lock_file"
        LOCK_COUNT=$((LOCK_COUNT + 1))
      fi
    fi
  done
  shopt -u nullglob
fi

# Clean up orphaned translation chunks (chunk-* without corresponding -translated)
CHUNK_COUNT=0
if [ -d "$COORD_DIR/translation" ]; then
  shopt -s nullglob
  for chunk in "$COORD_DIR"/translation/*/chunk-*.md; do
    [ -f "$chunk" ] || continue
    translated="${chunk%.md}-translated.md"
    # If chunk is older than 24 hours and no translated version exists, remove it
    if [ ! -f "$translated" ] && [ -n "$(find "$chunk" -mmin +1440 2> /dev/null)" ]; then
      rm -f "$chunk" 2> /dev/null || true
      CHUNK_COUNT=$((CHUNK_COUNT + 1))
    fi
  done
  shopt -u nullglob
fi

# Log session end
echo "[$(date -Iseconds)] SessionEnd: Cleaned $TEMP_COUNT temp, $LOCK_COUNT locks, $CHUNK_COUNT orphan chunks" >> "$LOG_FILE"

exit 0
