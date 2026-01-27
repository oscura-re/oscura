#!/bin/bash
# Cleanup script: Remove obsolete demos/ directory after successful migration
#
# This script should only be run after:
# 1. Migration is complete (migrate_examples.py executed successfully)
# 2. Validation passes (python demonstrations/validate_all.py)
# 3. User confirms all examples work

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DEMOS_DIR="$ROOT_DIR/demos"

echo "================================================================================"
echo "CLEANUP: Remove obsolete demos/ directory"
echo "================================================================================"
echo ""

# Safety check
if [ ! -d "$DEMOS_DIR" ]; then
  echo "✓ demos/ directory already removed"
  exit 0
fi

# Count files before
DEMO_FILES=$(find "$DEMOS_DIR" -type f -name "*.py" | wc -l)
echo "Found $DEMO_FILES Python files in demos/"
echo ""

# Confirm with user
echo "This will DELETE the entire demos/ directory."
echo "All content has been migrated to demonstrations/"
echo ""
read -p "Proceed with deletion? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^yes$ ]]; then
  echo "Cleanup cancelled"
  exit 1
fi

# Create archive before deletion
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_FILE="$ROOT_DIR/.claude/demos_archived_$TIMESTAMP.tar.gz"

echo "Creating archive: $ARCHIVE_FILE"
tar -czf "$ARCHIVE_FILE" -C "$ROOT_DIR" demos/
echo "✓ Archive created"
echo ""

# Delete demos directory
echo "Deleting demos/ directory..."
rm -rf "$DEMOS_DIR"
echo "✓ demos/ directory removed"
echo ""

# Update .gitignore to remove demos/ references if they exist
GITIGNORE="$ROOT_DIR/.gitignore"
if [ -f "$GITIGNORE" ]; then
  # Remove lines referencing demos/
  sed -i.bak '/^demos\//d' "$GITIGNORE"
  rm -f "$GITIGNORE.bak"
  echo "✓ Updated .gitignore"
fi

echo ""
echo "================================================================================"
echo "CLEANUP COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  - Deleted: demos/ directory ($DEMO_FILES Python files)"
echo "  - Archived to: $ARCHIVE_FILE"
echo "  - Updated: .gitignore"
echo ""
echo "Next steps:"
echo "  1. Update CONTRIBUTING.md to reference only demonstrations/"
echo "  2. Update CHANGELOG.md"
echo "  3. Commit changes: git add -A && git commit -m 'feat: consolidate examples into single demonstrations/ directory'"
echo ""
