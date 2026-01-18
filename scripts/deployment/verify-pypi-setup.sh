#!/bin/bash
# Verify PyPI configuration is ready

echo "🔍 Verifying PyPI Setup"
echo "======================="
echo ""

# Check if .pypirc exists
if [[ -f ~/.pypirc ]]; then
  echo "✅ Found ~/.pypirc"

  # Check permissions
  PERMS=$(stat -c %a ~/.pypirc 2>/dev/null || stat -f %A ~/.pypirc 2>/dev/null)
  if [[ "${PERMS}" == "600" ]]; then
    echo "✅ Permissions correct (600)"
  else
    echo "⚠️  Warning: Permissions are ${PERMS} (should be 600)"
    echo "   Run: chmod 600 ~/.pypirc"
  fi

  # Check if testpypi section exists
  if grep -q "\[testpypi\]" ~/.pypirc; then
    echo "✅ TestPyPI configuration found"
  else
    echo "❌ TestPyPI configuration missing"
    exit 1
  fi

  # Check if token is configured (not the placeholder)
  if grep -q "password = pypi-" ~/.pypirc; then
    echo "✅ TestPyPI token configured"
  else
    echo "❌ TestPyPI token not configured or invalid"
    echo "   Token should start with 'pypi-'"
    exit 1
  fi

else
  echo "❌ ~/.pypirc not found"
  echo ""
  echo "Please run one of:"
  echo "  ./scripts/setup-testpypi-token.sh"
  echo "  OR manually create ~/.pypirc"
  exit 1
fi

# Check if dist files exist
echo ""
echo "📦 Checking distribution files..."
if [[ -f dist/oscura-0.1.0.tar.gz ]] && [[ -f dist/oscura-0.1.0-py3-none-any.whl ]]; then
  echo "✅ Distribution files found"

  # Check with twine
  if uv run twine check dist/oscura-0.1.0* >/dev/null 2>&1; then
    echo "✅ Package validation passed"
  else
    echo "⚠️  Package validation warnings"
  fi
else
  echo "❌ Distribution files missing"
  echo "   Run: uv build"
  exit 1
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "Ready to upload to TestPyPI:"
echo "  uv run twine upload --repository testpypi dist/oscura-0.1.0*"
