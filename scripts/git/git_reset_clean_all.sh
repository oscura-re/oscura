#!/usr/bin/env bash
# =============================================================================
# Git Reset (Clean All - Nuclear Option)
# =============================================================================
# Usage: ./scripts/git_reset_clean_all.sh [branch] [-h|--help]
# =============================================================================
# Reset local repository to match remote EXACTLY with complete clean slate.
# Use this for a pristine checkout state, removing EVERYTHING not in remote.
#
# WHAT IT DOES:
#   1. Downloads latest commits from remote (no merge yet)
#   2. Discards all local commits and changes, resets to remote state
#   3. Removes ALL untracked files/directories INCLUDING .gitignored files
#
# WARNING: Destroys all local commits, uncommitted changes, AND ignored files!
#          This includes IDE settings, build caches, .env files, etc.
#          Essentially equivalent to deleting the repo and cloning fresh.
#
# EXAMPLES:
#   ./scripts/git_reset_clean_all.sh        # Reset to origin/main
#   ./scripts/git_reset_clean_all.sh dev    # Reset to origin/dev
#
# RESULT:
#   - All tracked files match remote exactly
#   - ALL untracked files/directories deleted
#   - ALL .gitignored files deleted
#   - Essentially a fresh clone state
# =============================================================================

set -euo pipefail

# Parse arguments
BRANCH="main"
while [[ $# -gt 0 ]]; do
  case $1 in
    -h | --help)
      echo "Usage: $0 [branch] [-h|--help]"
      echo ""
      echo "Git Reset (Clean All - Nuclear Option)"
      echo ""
      echo "Reset local repository to match remote EXACTLY."
      echo ""
      echo "WARNING: This is DESTRUCTIVE! It will:"
      echo "  - Discard all local commits and changes"
      echo "  - Remove ALL .gitignored files (.env, caches, etc.)"
      echo "  - Essentially delete the repo and clone fresh"
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

echo "=== Git Reset (NUCLEAR - Clean All) ==="
echo ""
echo "Target: origin/${BRANCH}"
echo ""
echo "WARNING: This will remove:"
echo "  - All uncommitted changes"
echo "  - All local commits not pushed"
echo "  - ALL .gitignored files (.env, caches, IDE settings, etc.)"
echo ""

# Confirm before proceeding
read -r -p "This is DESTRUCTIVE. Type 'yes' to confirm: " response
if [[ "${response}" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
echo "Fetching from remote..."
git fetch origin

echo "Resetting to origin/${BRANCH}..."
git reset --hard "origin/${BRANCH}"

echo "Cleaning ALL untracked files, directories, and .gitignored files..."
git clean -fdx

echo ""
echo "Done! Local repo now matches origin/${BRANCH} exactly (fresh clone state)."
