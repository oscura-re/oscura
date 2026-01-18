#!/bin/bash
# Check name availability across PyPI, GitHub, and domains
# Usage: ./check-name-availability.sh <name>

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <name> [name2] [name3] ..."
  echo "Example: $0 signalforge scopex sigrev"
  exit 1
fi

echo "=== Name Availability Checker ==="
echo "Date: $(date -Iseconds)"
echo ""

# Function to check a single name
check_name() {
  local name="$1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Checking: $name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  # 1. PyPI availability
  echo "[1/4] PyPI Package Availability"
  if curl -s "https://pypi.org/pypi/${name}/json" | grep -q '"name"'; then
    echo "  ❌ TAKEN - Package exists on PyPI"
    PYPI_URL="https://pypi.org/project/${name}/"
    echo "  URL: $PYPI_URL"
  else
    echo "  ✅ AVAILABLE - Not found on PyPI"
  fi
  echo ""

  # 2. GitHub repository
  echo "[2/4] GitHub Repository"
  GITHUB_COUNT=$(gh search repos "${name}" --limit 5 --json nameWithOwner --jq 'length' 2>/dev/null || echo "0")
  if [ "$GITHUB_COUNT" -eq 0 ]; then
    echo "  ✅ AVAILABLE - No exact matches found"
  elif [ "$GITHUB_COUNT" -lt 3 ]; then
    echo "  ⚠️  LOW COMPETITION - $GITHUB_COUNT similar repos found"
  else
    echo "  ❌ HIGH COMPETITION - $GITHUB_COUNT+ similar repos found"
  fi
  echo ""

  # 3. Domain availability (using dig for basic check)
  echo "[3/4] Domain Availability (Basic DNS Check)"
  for ext in com io dev tools; do
    domain="${name}.${ext}"
    if dig +short "$domain" | grep -q .; then
      echo "  ❌ ${domain} - TAKEN (has DNS records)"
    else
      echo "  ✅ ${domain} - LIKELY AVAILABLE (no DNS records)"
    fi
  done
  echo ""
  echo "  Note: Use https://instantdomainsearch.com/ for definitive check"
  echo ""

  # 4. PyPI name variants
  echo "[4/4] PyPI Name Variants"
  for variant in "${name}-py" "py${name}" "${name}kit" "${name}-signal"; do
    if curl -s "https://pypi.org/pypi/${variant}/json" | grep -q '"name"'; then
      echo "  ❌ ${variant} - Taken"
    else
      echo "  ✅ ${variant} - Available"
    fi
  done
  echo ""
}

# Check each name provided
for name in "$@"; do
  # Convert to lowercase
  name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  check_name "$name_lower"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Check complete for $# name(s)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
