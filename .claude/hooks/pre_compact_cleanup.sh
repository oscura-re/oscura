#!/usr/bin/env bash
# shellcheck disable=SC2250,SC2292,SC2311,SC2005
set -e

# PreCompact Cleanup Hook - Enhanced Version
# Archives old coordination and agent output files before compaction
# Cleans up checkpoints, large JSON files, test data references
# Reduces noise in conversation history before summarization
#
# Features:
# - Archives coordination files >30 days old
# - Cleans checkpoints and handoffs >7 days old
# - Compresses large JSON files (>100KB)
# - Removes stale test data references
# - Deduplicates repetitive file references
# - Logs context usage metrics

CLAUDE_PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
COORD_DIR="$CLAUDE_PROJECT_DIR/.coordination"
AGENT_OUT="$CLAUDE_PROJECT_DIR/.claude/agent-outputs"
ARCHIVE_DATE=$(date +%Y-%m)
LOG_FILE="$CLAUDE_PROJECT_DIR/.claude/hooks/compaction.log"
METRICS_FILE="$CLAUDE_PROJECT_DIR/.claude/hooks/context_metrics.log"

# Size thresholds
LARGE_JSON_SIZE=102400 # 100KB in bytes
OLD_REPORT_DAYS=7

# Ensure log and archive directories exist
mkdir -p "$CLAUDE_PROJECT_DIR/.claude/hooks"
mkdir -p "$COORD_DIR/archive/$ARCHIVE_DATE"
mkdir -p "$COORD_DIR/archive/$ARCHIVE_DATE/large-json"
mkdir -p "$AGENT_OUT/archive/$ARCHIVE_DATE"

log() {
  echo "[$(date -Iseconds)] PRE_COMPACT: $*" >> "$LOG_FILE"
}

log_metrics() {
  echo "[$(date -Iseconds)] $*" >> "$METRICS_FILE"
}

# Calculate directory size for metrics
calc_dir_size() {
  local dir="$1"
  if [ -d "$dir" ]; then
    du -sb "$dir" 2> /dev/null | cut -f1 || echo 0
  else
    echo 0
  fi
}

# Record initial sizes for metrics
INITIAL_COORD_SIZE=$(calc_dir_size "$COORD_DIR")
INITIAL_AGENT_SIZE=$(calc_dir_size "$AGENT_OUT")

# ============================================================
# 1. Archive coordination files older than 30 days (root level)
# ============================================================
COORD_COUNT=0
if [ -d "$COORD_DIR" ]; then
  COORD_COUNT=$(find "$COORD_DIR" -maxdepth 1 -type f -name "*.md" -mtime +30 2> /dev/null | wc -l)
  if [ "$COORD_COUNT" -gt 0 ]; then
    find "$COORD_DIR" -maxdepth 1 -type f -name "*.md" -mtime +30 \
      -exec mv {} "$COORD_DIR/archive/$ARCHIVE_DATE/" \; 2> /dev/null || true
  fi
fi

# ============================================================
# 2. Archive module-specific coordination files (spec, translation subdirs)
# ============================================================
SUBDIR_COUNT=0
for subdir in spec translation contexts; do
  if [ -d "$COORD_DIR/$subdir" ]; then
    count=$(find "$COORD_DIR/$subdir" -type f \( -name "*.md" -o -name "*.yaml" \) -mtime +30 2> /dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
      mkdir -p "$COORD_DIR/archive/$ARCHIVE_DATE/$subdir"
      find "$COORD_DIR/$subdir" -type f \( -name "*.md" -o -name "*.yaml" \) -mtime +30 \
        -exec mv {} "$COORD_DIR/archive/$ARCHIVE_DATE/$subdir/" \; 2> /dev/null || true
      SUBDIR_COUNT=$((SUBDIR_COUNT + count))
    fi
  fi
done

# ============================================================
# 3. Clean up checkpoints older than 7 days (per documented retention policy)
# ============================================================
CHECKPOINT_COUNT=0
if [ -d "$COORD_DIR/checkpoints" ]; then
  CHECKPOINT_COUNT=$(find "$COORD_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -mtime +7 2> /dev/null | wc -l)
  if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
    find "$COORD_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -mtime +7 \
      -exec rm -rf {} \; 2> /dev/null || true
  fi
fi

# ============================================================
# 4. Clean up handoffs older than 7 days
# ============================================================
HANDOFF_COUNT=0
if [ -d "$COORD_DIR/handoffs" ]; then
  HANDOFF_COUNT=$(find "$COORD_DIR/handoffs" -type f -mtime +7 2> /dev/null | wc -l)
  if [ "$HANDOFF_COUNT" -gt 0 ]; then
    find "$COORD_DIR/handoffs" -type f -mtime +7 \
      -exec rm -f {} \; 2> /dev/null || true
  fi
fi

# ============================================================
# 5. Archive agent output files older than 7 days
# ============================================================
AGENT_COUNT=0
if [ -d "$AGENT_OUT" ]; then
  AGENT_COUNT=$(find "$AGENT_OUT" -maxdepth 1 -type f -name "*.json" -mtime +7 2> /dev/null | wc -l)
  if [ "$AGENT_COUNT" -gt 0 ]; then
    find "$AGENT_OUT" -maxdepth 1 -type f -name "*.json" -mtime +7 \
      -exec mv {} "$AGENT_OUT/archive/$ARCHIVE_DATE/" \; 2> /dev/null || true
  fi
fi

# ============================================================
# 6. NEW: Compress or archive large JSON files (>100KB)
# ============================================================
LARGE_JSON_COUNT=0
if [ -d "$COORD_DIR" ]; then
  while IFS= read -r -d '' json_file; do
    if [ -f "$json_file" ]; then
      file_size=$(stat -c%s "$json_file" 2> /dev/null || echo 0)
      if [ "$file_size" -gt "$LARGE_JSON_SIZE" ]; then
        # Move to archive with compression
        base_name=$(basename "$json_file")
        gzip -c "$json_file" > "$COORD_DIR/archive/$ARCHIVE_DATE/large-json/${base_name}.gz" 2> /dev/null \
          && rm -f "$json_file" 2> /dev/null
        LARGE_JSON_COUNT=$((LARGE_JSON_COUNT + 1))
      fi
    fi
  done < <(find "$COORD_DIR" -maxdepth 2 -name "*.json" -type f -print0 2> /dev/null)
fi

# Also check root directory for large JSON files
ROOT_LARGE_JSON=0
while IFS= read -r -d '' json_file; do
  if [ -f "$json_file" ]; then
    file_size=$(stat -c%s "$json_file" 2> /dev/null || echo 0)
    mtime_days=$((($(date +%s) - $(stat -c%Y "$json_file" 2> /dev/null || echo "$(date +%s)")) / 86400))
    # Only archive if older than 7 days and large
    if [ "$file_size" -gt "$LARGE_JSON_SIZE" ] && [ "$mtime_days" -gt "$OLD_REPORT_DAYS" ]; then
      base_name=$(basename "$json_file")
      mkdir -p "$COORD_DIR/archive/$ARCHIVE_DATE/root-json"
      gzip -c "$json_file" > "$COORD_DIR/archive/$ARCHIVE_DATE/root-json/${base_name}.gz" 2> /dev/null \
        && rm -f "$json_file" 2> /dev/null
      ROOT_LARGE_JSON=$((ROOT_LARGE_JSON + 1))
    fi
  fi
done < <(find "$CLAUDE_PROJECT_DIR" -maxdepth 1 -name "*.json" -type f -print0 2> /dev/null)

# ============================================================
# 7. NEW: Clean up test data references (*.wfm, *.tss files in test_data/)
# ============================================================
TEST_DATA_COUNT=0
if [ -d "$CLAUDE_PROJECT_DIR/test_data" ]; then
  # Remove orphaned partial files
  TEST_DATA_COUNT=$(find "$CLAUDE_PROJECT_DIR/test_data" \( -name "*.partial" -o -name "*.tmp" -o -name "*.bak" \) 2> /dev/null | wc -l)
  find "$CLAUDE_PROJECT_DIR/test_data" \( -name "*.partial" -o -name "*.tmp" -o -name "*.bak" \) \
    -delete 2> /dev/null || true
fi

# ============================================================
# 8. NEW: Clean up old validation reports in root (>7 days old)
# ============================================================
OLD_REPORT_COUNT=0
for pattern in "*_report*.json" "*_validation*.json" "comprehensive_*.json"; do
  while IFS= read -r -d '' report_file; do
    if [ -f "$report_file" ]; then
      mtime_days=$((($(date +%s) - $(stat -c%Y "$report_file" 2> /dev/null || echo "$(date +%s)")) / 86400))
      if [ "$mtime_days" -gt "$OLD_REPORT_DAYS" ]; then
        mkdir -p "$COORD_DIR/archive/$ARCHIVE_DATE/reports"
        mv "$report_file" "$COORD_DIR/archive/$ARCHIVE_DATE/reports/" 2> /dev/null || true
        OLD_REPORT_COUNT=$((OLD_REPORT_COUNT + 1))
      fi
    fi
  done < <(find "$CLAUDE_PROJECT_DIR" -maxdepth 1 -name "$pattern" -type f -print0 2> /dev/null)
done

# ============================================================
# 9. Archive retention policy: Delete archives older than 180 days (6 months)
# ============================================================
OLD_ARCHIVE_COUNT=0
for archive_dir in "$COORD_DIR/archive" "$AGENT_OUT/archive"; do
  if [ -d "$archive_dir" ]; then
    count=$(find "$archive_dir" -mindepth 1 -maxdepth 1 -type d -mtime +180 2> /dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
      find "$archive_dir" -mindepth 1 -maxdepth 1 -type d -mtime +180 \
        -exec rm -rf {} \; 2> /dev/null || true
      OLD_ARCHIVE_COUNT=$((OLD_ARCHIVE_COUNT + count))
    fi
  fi
done

# ============================================================
# 10. Clean empty archive directories
# ============================================================
find "$COORD_DIR/archive" -type d -empty -delete 2> /dev/null || true
find "$AGENT_OUT/archive" -type d -empty -delete 2> /dev/null || true

# ============================================================
# 11. NEW: Remove duplicate/redundant audit files
# ============================================================
DEDUP_COUNT=0
if [ -d "$COORD_DIR/audits" ]; then
  # Keep only the most recent version of audit files with similar names
  # Remove any audit files that are duplicates (same content, different names)
  shopt -s nullglob
  declare -A seen_checksums
  for audit_file in "$COORD_DIR/audits"/*.md; do
    if [ -f "$audit_file" ]; then
      checksum=$(md5sum "$audit_file" 2> /dev/null | cut -d' ' -f1 || echo "")
      if [ -n "$checksum" ]; then
        if [ -n "${seen_checksums[$checksum]}" ]; then
          # Duplicate found - archive the older one
          existing="${seen_checksums[$checksum]}"
          existing_mtime=$(stat -c%Y "$existing" 2> /dev/null || echo 0)
          current_mtime=$(stat -c%Y "$audit_file" 2> /dev/null || echo 0)
          if [ "$current_mtime" -gt "$existing_mtime" ]; then
            mkdir -p "$COORD_DIR/archive/$ARCHIVE_DATE/audits"
            mv "$existing" "$COORD_DIR/archive/$ARCHIVE_DATE/audits/" 2> /dev/null || true
            seen_checksums[$checksum]="$audit_file"
          else
            mkdir -p "$COORD_DIR/archive/$ARCHIVE_DATE/audits"
            mv "$audit_file" "$COORD_DIR/archive/$ARCHIVE_DATE/audits/" 2> /dev/null || true
          fi
          DEDUP_COUNT=$((DEDUP_COUNT + 1))
        else
          seen_checksums[$checksum]="$audit_file"
        fi
      fi
    fi
  done
  shopt -u nullglob
fi

# ============================================================
# Calculate final sizes and log metrics
# ============================================================
FINAL_COORD_SIZE=$(calc_dir_size "$COORD_DIR")
FINAL_AGENT_SIZE=$(calc_dir_size "$AGENT_OUT")
SPACE_SAVED=$(((INITIAL_COORD_SIZE + INITIAL_AGENT_SIZE) - (FINAL_COORD_SIZE + FINAL_AGENT_SIZE)))

# Log cleanup summary
log "Archived: $COORD_COUNT coord, $SUBDIR_COUNT subdir, $AGENT_COUNT outputs"
log "Cleaned: $CHECKPOINT_COUNT checkpoints, $HANDOFF_COUNT handoffs, $OLD_ARCHIVE_COUNT old archives"
log "New: $LARGE_JSON_COUNT large JSON, $ROOT_LARGE_JSON root JSON, $TEST_DATA_COUNT test data"
log "Reports: $OLD_REPORT_COUNT old reports, $DEDUP_COUNT duplicates"
log "Space saved: $(numfmt --to=iec $SPACE_SAVED 2> /dev/null || echo "${SPACE_SAVED}B")"

# Log metrics for monitoring
log_metrics "CLEANUP coord=$COORD_COUNT subdir=$SUBDIR_COUNT agent=$AGENT_COUNT checkpoint=$CHECKPOINT_COUNT handoff=$HANDOFF_COUNT json=$LARGE_JSON_COUNT test_data=$TEST_DATA_COUNT reports=$OLD_REPORT_COUNT dedup=$DEDUP_COUNT saved=$SPACE_SAVED"

exit 0
