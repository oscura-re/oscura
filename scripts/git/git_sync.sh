#!/usr/bin/env bash
# =============================================================================
# Git Sync - Smart Repository Synchronization
# =============================================================================
# Synchronize local repository with remote, with options for preserving
# or cleaning gitignored files, stashing changes, and branch selection.
#
# Usage: ./scripts/git/git_sync.sh [OPTIONS]
#
# Examples:
#   ./scripts/git/git_sync.sh                      # Sync to origin/main, preserve ignored
#   ./scripts/git/git_sync.sh --clean              # Full clean including ignored files
#   ./scripts/git/git_sync.sh --branch dev         # Sync to origin/dev
#   ./scripts/git/git_sync.sh --stash              # Stash changes before sync
# =============================================================================
#
# NOTE: VS Code shellcheck extension may show SC2154 false positives for
# variables sourced from lib/common.sh. These are extension bugs, not code
# issues. CLI shellcheck validates correctly:
#   cd /path/to/workspace_template && shellcheck scripts/git/git_sync.sh
# =============================================================================

# shellcheck source=../lib/common.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# Options
PRESERVE_IGNORED=true
DO_CLEAN=false
BRANCH=""
STASH_CHANGES=false
FORCE=false
DRY_RUN=false
VERBOSE=false

# =============================================================================
# Help
# =============================================================================

show_script_help() {
  cat << 'EOF'
git_sync.sh - Smart repository synchronization with remote

USAGE:
    git_sync.sh [OPTIONS]

DESCRIPTION:
    Synchronizes local repository to match remote exactly. By default,
    preserves .gitignored files (IDE settings, build caches, venvs).
    Useful for syncing work between multiple machines.

OPTIONS:
    --preserve      Preserve .gitignored files (default)
    --clean         Remove ALL untracked files including .gitignored
    --branch NAME   Sync to specific branch (default: current or main)
    --stash         Stash local changes before sync (can recover with git stash pop)
    --force         Skip confirmation prompts
    -n, --dry-run   Show what would happen without making changes
    -v, --verbose   Show detailed output
    --json          Output machine-readable JSON
    -h, --help      Show this help message

MODES:

    --preserve (default):
        - Discards all local commits and uncommitted changes
        - Removes untracked files NOT in .gitignore
        - Keeps .gitignored files (.venv, .idea, __pycache__, etc.)
        - Use when syncing between machines

    --clean:
        - Nuclear option - resets everything
        - Removes ALL untracked files including .gitignored
        - Use for fresh clone-like state

    --stash:
        - Saves current changes to stash before syncing
        - Can recover with: git stash pop
        - Useful when you want to preserve work-in-progress

WARNING:
    This script DESTROYS local commits and uncommitted changes!
    Use --stash to preserve work-in-progress.
    Use --dry-run to preview changes.

EXAMPLES:
    # Standard sync - preserve IDE settings and venvs
    git_sync.sh

    # Preview what would happen
    git_sync.sh --dry-run

    # Sync to development branch
    git_sync.sh --branch develop

    # Save WIP before sync
    git_sync.sh --stash

    # Full clean (like fresh clone)
    git_sync.sh --clean

    # Non-interactive (for scripts)
    git_sync.sh --force

GIT COMMANDS USED:

    --preserve mode:
        git fetch origin
        git reset --hard origin/<branch>
        git clean -fd              # Remove untracked, keep ignored

    --clean mode:
        git fetch origin
        git reset --hard origin/<branch>
        git clean -fdx             # Remove everything untracked

EXIT CODES:
    0  Success
    1  Sync failed
    2  Configuration error

SEE ALSO:
    git stash - Save changes for later
    git status - Check repository state
    rsync - File synchronization
EOF
}

# =============================================================================
# Git Operations
# =============================================================================

check_git_repo() {
  if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    print_fail "Not inside a Git repository"
    return 1
  fi
  return 0
}

get_current_branch() {
  git branch --show-current 2> /dev/null || echo "main"
}

get_default_branch() {
  # Try to get default branch from remote
  local default
  default=$(git symbolic-ref refs/remotes/origin/HEAD 2> /dev/null | sed 's@^refs/remotes/origin/@@')

  if [[ -z "${default}" ]]; then
    # Fallback to common defaults
    if git show-ref --verify --quiet refs/remotes/origin/main; then
      default="main"
    elif git show-ref --verify --quiet refs/remotes/origin/master; then
      default="master"
    else
      default="main"
    fi
  fi

  echo "${default}"
}

check_uncommitted_changes() {
  if [[ -n "$(git status --porcelain 2> /dev/null)" ]]; then
    return 0 # Has changes
  fi
  return 1 # Clean
}

check_unpushed_commits() {
  local branch="$1"
  local count
  count=$(git rev-list --count "origin/${branch}..HEAD" 2> /dev/null || echo "0")
  [[ "${count}" -gt 0 ]]
}

stash_changes() {
  print_info "Stashing local changes..."

  local stash_message="git_sync auto-stash $(date '+%Y-%m-%d %H:%M:%S')"

  if git stash push -m "${stash_message}" --include-untracked; then
    print_pass "Changes stashed: ${stash_message}"
    print_info "Recover with: git stash pop"
    return 0
  else
    print_fail "Failed to stash changes"
    return 1
  fi
}

do_sync() {
  local target_branch="$1"
  local preserve="$2"
  local dry_run="$3"

  # Fetch
  print_section "Fetching from origin..."
  if ${dry_run}; then
    print_info "[DRY RUN] Would run: git fetch origin"
  else
    if ! git fetch origin; then
      print_fail "Failed to fetch from origin"
      return 1
    fi
    print_pass "Fetch complete"
  fi

  # Check if branch exists on remote
  if ! git show-ref --verify --quiet "refs/remotes/origin/${target_branch}"; then
    print_fail "Branch not found on remote: origin/${target_branch}"
    print_info "Available branches:"
    git branch -r | head -10
    return 1
  fi

  # Reset to remote
  print_section "Resetting to origin/${target_branch}..."
  if ${dry_run}; then
    print_info "[DRY RUN] Would run: git reset --hard origin/${target_branch}"
  else
    if ! git reset --hard "origin/${target_branch}"; then
      print_fail "Failed to reset to origin/${target_branch}"
      return 1
    fi
    print_pass "Reset complete"
  fi

  # Clean untracked files
  print_section "Cleaning untracked files..."
  local clean_opts=(-fd)
  if ! ${preserve}; then
    clean_opts+=(-x) # Also remove ignored files
  fi

  if ${dry_run}; then
    print_info "[DRY RUN] Would run: git clean ${clean_opts[*]}"
    echo ""
    print_info "Files that would be removed:"
    git clean -n "${clean_opts[@]}" | head -20
  else
    if ! git clean "${clean_opts[@]}"; then
      print_fail "Failed to clean untracked files"
      return 1
    fi
    print_pass "Clean complete"
  fi

  return 0
}

# =============================================================================
# Main
# =============================================================================

main() {
  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help)
        show_script_help
        exit 0
        ;;
      --preserve)
        PRESERVE_IGNORED=true
        DO_CLEAN=false
        shift
        ;;
      --clean)
        DO_CLEAN=true
        PRESERVE_IGNORED=false
        shift
        ;;
      --branch)
        BRANCH="$2"
        shift 2
        ;;
      --stash)
        STASH_CHANGES=true
        shift
        ;;
      --force)
        FORCE=true
        shift
        ;;
      -n | --dry-run)
        DRY_RUN=true
        shift
        ;;
      -v | --verbose)
        VERBOSE=true
        shift
        ;;
      --json)
        enable_json
        shift
        ;;
      -*)
        echo "Unknown option: $1" >&2
        echo "Use -h for help" >&2
        exit 2
        ;;
      *)
        echo "Unexpected argument: $1" >&2
        exit 2
        ;;
    esac
  done

  # Verify we're in a git repo
  if ! check_git_repo; then
    exit 2
  fi

  # Determine target branch
  if [[ -z "${BRANCH}" ]]; then
    BRANCH=$(get_current_branch)
    # If detached HEAD, use default branch
    if [[ -z "${BRANCH}" ]]; then
      BRANCH=$(get_default_branch)
    fi
  fi

  print_header "GIT SYNC"
  echo ""
  print_info "Repository: ${REPO_ROOT}"
  print_info "Target:     origin/${BRANCH}"
  if ${PRESERVE_IGNORED}; then
    print_info "Mode:       Preserve .gitignored files"
  else
    print_info "Mode:       Clean ALL (including .gitignored)"
  fi
  ${DRY_RUN} && print_info "Dry run:    Yes (no changes will be made)"
  echo ""

  # Check for uncommitted changes
  local has_changes=false
  if check_uncommitted_changes; then
    has_changes=true
    print_section "Uncommitted changes detected"
    echo ""
    git status --short | head -10
    echo ""
  fi

  # Check for unpushed commits
  local has_unpushed=false
  if check_unpushed_commits "${BRANCH}"; then
    has_unpushed=true
    local count
    count=$(git rev-list --count "origin/${BRANCH}..HEAD" 2> /dev/null)
    print_section "Warning: ${count} unpushed commit(s) will be lost!"
    echo ""
    git log --oneline "origin/${BRANCH}..HEAD" | head -5
    echo ""
  fi

  # Stash if requested
  if ${STASH_CHANGES} && ${has_changes}; then
    if ! stash_changes; then
      exit 1
    fi
    has_changes=false
  fi

  # Confirmation
  if ! ${FORCE} && ! ${DRY_RUN}; then
    if ${has_changes} || ${has_unpushed}; then
      echo ""
      echo -e "${YELLOW}WARNING: This will DESTROY local commits and uncommitted changes!${NC}"
      if ! ${PRESERVE_IGNORED}; then
        echo -e "${YELLOW}WARNING: This will also DELETE all .gitignored files!${NC}"
      fi
      echo ""
      read -r -p "Continue? [y/N] " response
      if [[ ! "${response}" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Aborted"
        exit 0
      fi
    fi
  fi

  # Perform sync
  if ! do_sync "${BRANCH}" "${PRESERVE_IGNORED}" "${DRY_RUN}"; then
    print_fail "Sync failed"
    json_result "sync" "fail" "Sync failed"
    exit 1
  fi

  # Summary
  echo ""
  if ${DRY_RUN}; then
    print_pass "Dry run complete - no changes made"
    json_result "sync" "dry-run" "No changes"
  else
    print_pass "Sync complete!"
    print_info "Local repository now matches origin/${BRANCH}"
    if ${PRESERVE_IGNORED}; then
      print_info ".gitignored files were preserved"
    fi
    if ${STASH_CHANGES}; then
      print_info "Previous changes saved in stash"
    fi
    json_result "sync" "pass" "origin/${BRANCH}"
  fi
}

main "$@"
