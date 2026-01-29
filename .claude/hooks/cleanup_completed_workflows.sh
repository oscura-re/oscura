#!/usr/bin/env bash
# shellcheck disable=SC2250,SC2292
# =============================================================================
# Cleanup Completed Workflows Hook
# Archives workflow-progress.json files when workflows complete
# Retention: 0 days (archive immediately on completion)
#
# Version: 1.0.0
# Created: 2025-12-25
# =============================================================================

set -euo pipefail

# Resolve to absolute path from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$REPO_ROOT}"
LOG_FILE="$PROJECT_DIR/.claude/hooks/hook.log"

log() {
  echo "[$(date -Iseconds)] CLEANUP_WORKFLOWS: $*" >> "$LOG_FILE"
}

log "Starting workflow cleanup"

ARCHIVED=0

# Archive workflow-progress.json if it exists
WORKFLOW_FILE="$PROJECT_DIR/.claude/workflow-progress.json"
if [ -f "$WORKFLOW_FILE" ]; then
  TIMESTAMP=$(date +%Y%m%d%H%M%S)
  ARCHIVE_NAME="$PROJECT_DIR/.claude/workflow-progress-archived-$TIMESTAMP.json"

  mv "$WORKFLOW_FILE" "$ARCHIVE_NAME"
  log "Archived $WORKFLOW_FILE to $ARCHIVE_NAME"
  ARCHIVED=$((ARCHIVED + 1))
fi

echo "{\"ok\": true, \"archived\": $ARCHIVED}"
log "Cleanup complete: $ARCHIVED files archived"
exit 0
