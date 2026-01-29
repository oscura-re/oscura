#!/bin/bash
# Oscura v0.1.0 PyPI Publishing Script
# Run this after setting up credentials in ~/.pypirc

set -e

echo "🚀 Oscura v0.1.0 PyPI Publishing"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .pypirc exists
if [[ ! -f ~/.pypirc ]]; then
  echo -e "${RED}❌ Error: ~/.pypirc not found${NC}"
  echo ""
  echo "Please create ~/.pypirc with your API tokens."
  echo "See: /tmp/pypi-upload-instructions.md"
  exit 1
fi

# Check dist files exist
if [[ ! -f dist/oscura-0.1.0.tar.gz ]] || [[ ! -f dist/oscura-0.1.0-py3-none-any.whl ]]; then
  echo -e "${RED}❌ Error: Distribution files not found${NC}"
  echo "Run: uv build"
  exit 1
fi

echo "✅ Found distribution files"
echo ""

# Step 1: Upload to TestPyPI
echo -e "${YELLOW}Step 1: Uploading to TestPyPI...${NC}"
if uv run twine upload --repository testpypi dist/oscura-0.1.0*; then
  echo -e "${GREEN}✅ Successfully uploaded to TestPyPI${NC}"
  echo ""
  echo "View at: https://test.pypi.org/project/oscura/0.1.0/"
  echo ""
else
  echo -e "${RED}❌ TestPyPI upload failed${NC}"
  exit 1
fi

# Step 2: Test installation from TestPyPI
echo -e "${YELLOW}Step 2: Testing installation from TestPyPI...${NC}"
echo "This will test install in a temporary environment"
read -r -p "Press Enter to continue or Ctrl+C to abort..."

# Create temp venv for testing
TEMP_VENV=$(mktemp -d)/test-install
python3 -m venv "${TEMP_VENV}"
source "${TEMP_VENV}/bin/activate"

if pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oscura==0.1.0; then
  echo -e "${GREEN}✅ Test installation successful${NC}"
  python -c "import oscura; print(f'Oscura version: {oscura.__version__}')"
  deactivate
  rm -rf "$(dirname "${TEMP_VENV}")"
else
  echo -e "${RED}❌ Test installation failed${NC}"
  deactivate
  rm -rf "$(dirname "${TEMP_VENV}")"
  exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: Ready to upload to production PyPI${NC}"
echo "⚠️  This will publish Oscura v0.1.0 to the public PyPI."
echo ""
read -r -p "Are you sure you want to proceed? (yes/no): " confirm

if [[ "${confirm}" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
echo -e "${YELLOW}Uploading to production PyPI...${NC}"
if uv run twine upload dist/oscura-0.1.0*; then
  echo ""
  echo -e "${GREEN}🎉 Successfully published Oscura v0.1.0 to PyPI!${NC}"
  echo ""
  echo "📦 Package: https://pypi.org/project/oscura/0.1.0/"
  echo "📚 GitHub: https://github.com/lair-click-bats/oscura/releases/tag/v0.1.0"
  echo ""
  echo "Install with: pip install oscura==0.1.0"
else
  echo -e "${RED}❌ Production PyPI upload failed${NC}"
  exit 1
fi
