#!/usr/bin/env bash
# =============================================================================
# Git Reset (Preserve Ignored Files)
# =============================================================================
# Usage: ./scripts/git_reset_preserve_ignored.sh [branch] [-h|--help]
# =============================================================================
# Reset local repository to match remote while preserving .gitignored files.
# Useful for syncing work between machines while keeping IDE settings, build
# caches, and other ignored files intact.
#
# WHAT IT DOES:
#   1. Downloads latest commits from remote (no merge yet)
#   2. Discards all local commits and changes, resets to remote state
#   3. Removes untracked files/directories (but keeps .gitignored files)
#
# WARNING: Destroys all local commits and uncommitted changes!
#          Untracked files (not in .gitignore) will be deleted!
#
# EXAMPLES:
#   ./scripts/git_reset_preserve_ignored.sh        # Reset to origin/main
#   ./scripts/git_reset_preserve_ignored.sh dev    # Reset to origin/dev
#
# RESULT:
#   - All tracked files match remote exactly
#   - Untracked files/directories deleted
#   - .gitignored files preserved (IDE settings, build artifacts, etc.)
# =============================================================================

set -euo pipefail

# Parse arguments
BRANCH="main"
while [[ $# -gt 0 ]]; do
  case $1 in
    -h | --help)
      echo "Usage: $0 [branch] [-h|--help]"
      echo ""
      echo "Git Reset (Preserve Ignored Files)"
      echo ""
      echo "Reset local repository to match remote while preserving .gitignored files."
      echo ""
      echo "WARNING: This is DESTRUCTIVE! It will:"
      echo "  - Discard all local commits and changes"
      echo "  - Remove untracked files"
      echo "  BUT keeps .gitignored files (.env, IDE settings, caches, etc.)"
      echo ""
      echo "Arguments:"
      echo "  branch      Target branch (default: main)"
      echo ""
      echo "Options:"
      echo "  -h, --help  Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0              # Reset to origin/main"
      echo "  $0 dev          # Reset to origin/dev"
      exit 0
      ;;
    *)
      BRANCH="$1"
      shift
      ;;
  esac
done

echo "=== Git Reset (Preserving Ignored Files) ==="
echo ""
echo "Target: origin/${BRANCH}"
echo ""

# Confirm before proceeding
read -r -p "This will discard ALL local commits and changes. Continue? [y/N] " response
if [[ ! "${response}" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
echo "Fetching from remote..."
git fetch origin

echo "Resetting to origin/${BRANCH}..."
git reset --hard "origin/${BRANCH}"

echo "Cleaning untracked files and directories (preserving .gitignored files)..."
git clean -fd

echo ""
echo "Done! Local repo now matches origin/${BRANCH} (.gitignored files preserved)."
