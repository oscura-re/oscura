#!/usr/bin/env bash
#
# Fix hypothesis test skips by replacing pytest.skip() with assume()
#
# These skips are in hypothesis property tests that skip when generated data
# doesn't meet preconditions. The proper way is to use hypothesis.assume()
# instead of pytest.skip().
#

set -e

# Files to fix (from analysis)
FILES=(
  "tests/unit/loaders/test_preprocessing_hypothesis.py"
  "tests/unit/analyzers/packet/test_checksum_hypothesis.py"
  "tests/unit/analyzers/power/test_measurement_hypothesis.py"
  "tests/unit/inference/test_sequences_hypothesis.py"
  "tests/unit/analyzers/digital/test_clock_hypothesis.py"
  "tests/unit/analyzers/jitter/test_measurement_hypothesis.py"
  "tests/unit/analyzers/patterns/test_search_hypothesis.py"
  "tests/unit/analyzers/patterns/test_matching_hypothesis.py"
  "tests/unit/analyzers/patterns/test_repetition_hypothesis.py"
)

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "⚠ Skipping $file - not found"
    continue
  fi

  echo "Processing $file..."

  # Add assume import if not present
  if ! grep -q "from hypothesis import assume" "$file"; then
    # Add assume to existing hypothesis import
    if grep -q "from hypothesis import" "$file"; then
      sed -i 's/from hypothesis import \(.*\)/from hypothesis import assume, \1/' "$file"
    else
      # Add new import after other imports
      sed -i '/import pytest/a from hypothesis import assume' "$file"
    fi
  fi

  # Replace pytest.skip patterns with assume
  # Pattern 1: if len(X) == 0: pytest.skip("Empty ...")
  sed -i 's/if len(\([^)]*\)) == 0:\s*$/assume(len(\1) > 0)/' "$file"
  sed -i '/assume(len/,+1 {/pytest.skip/d}' "$file"

  # Pattern 2: if condition: pytest.skip()
  sed -i 's/if \(.*\) < 2:\s*$/assume(\1 >= 2)/' "$file"
  sed -i '/assume.*>=/,+1 {/pytest.skip/d}' "$file"

  # Pattern 3: Direct replacements for common patterns
  sed -i 's/if len(\([^)]*\)) == 0:.*pytest.skip/assume(len(\1) > 0)/' "$file"
  sed -i 's/if len(\([^)]*\)) < \([0-9]\+\):.*pytest.skip/assume(len(\1) >= \2)/' "$file"

  echo "  ✓ Fixed $file"
done

echo ""
echo "====================================================================="
echo "Hypothesis test skips fixed!"
echo "====================================================================="
echo ""
echo "Next: Run tests to verify"
echo "  ./scripts/test.sh tests/unit/analyzers/statistical/test_distribution_hypothesis.py"
