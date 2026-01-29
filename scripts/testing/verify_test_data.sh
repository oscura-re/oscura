#!/bin/bash
# Verify test data exists and is valid
# This script is called from GitHub Actions workflows to validate test data integrity

set -e

echo "🔍 Verifying test data..."

# Check if test_data directory exists
if [[ ! -d "test_data" ]]; then
  echo "❌ Error: test_data directory not found"
  exit 1
fi

# Check if manifest.json exists
if [[ ! -f "test_data/manifest.json" ]]; then
  echo "❌ Error: test_data/manifest.json not found"
  exit 1
fi

# Validate manifest.json is valid JSON
if ! python3 -c "import json; json.load(open('test_data/manifest.json'))" 2> /dev/null; then
  echo "❌ Error: test_data/manifest.json is not valid JSON"
  exit 1
fi

# Check for expected subdirectories
expected_dirs=(
  "binary"
  "waveforms"
  "pcap"
  "real_captures"
  "power_analysis"
  "signal_integrity"
  "statistical"
)

missing_dirs=()
for dir in "${expected_dirs[@]}"; do
  if [[ ! -d "test_data/${dir}" ]]; then
    missing_dirs+=("${dir}")
  fi
done

if [[ ${#missing_dirs[@]} -ne 0 ]]; then
  echo "⚠️  Warning: Some expected directories are missing: ${missing_dirs[*]}"
else
  echo "✅ All expected test data directories present"
fi

# Get test data size
test_data_size=$(du -sh test_data 2> /dev/null | cut -f1)
echo "📊 Test data size: ${test_data_size}"

# Count total files
file_count=$(find test_data -type f | wc -l)
echo "📁 Total files in test_data: ${file_count}"

echo "✅ Test data verification complete"
exit 0
