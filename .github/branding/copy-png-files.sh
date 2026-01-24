#!/bin/bash
# Copy PNG branding files from knowledge-base to oscura repo
# Run this script from the oscura repository root

SOURCE_DIR="/home/lair-click-bats/development/knowledge-base/assets/brand/oscura-redesign/github-branding"
DEST_DIR="/home/lair-click-bats/development/oscura/.github/branding"

echo "Copying PNG branding files..."

cp "$SOURCE_DIR/oscura-org-avatar.png" "$DEST_DIR/"
cp "$SOURCE_DIR/oscura-org-avatar-1024.png" "$DEST_DIR/"
cp "$SOURCE_DIR/oscura-repo-social.png" "$DEST_DIR/"
cp "$SOURCE_DIR/oscura-readme-header.png" "$DEST_DIR/"

echo "Done. PNG files copied to $DEST_DIR"
echo ""
echo "Files copied:"
ls -la "$DEST_DIR"/*.png 2> /dev/null || echo "No PNG files found"
