# Repository Configuration Export

**Source Repository**: oscura-re/oscura
**Export Date**: 20260127_193000
**Script Version**: 1.0.0

## Exported Files

### Fully Recreatable via CLI/API

- `basic-settings.json` - Name, description, visibility, etc.
- `features.json` - Issues, Wiki, Projects, Discussions toggles
- `merge-settings.json` - Squash/merge/rebase settings
- `topics.json` - Repository topics/tags
- `branch-protection.json` - Branch protection rules
- `rulesets.json` - Repository rulesets (if any)
- `collaborators.json` - Team/user access permissions
- `webhooks.json` - Webhook configurations
- `variables.json` - Actions variables (names AND values)
- `pages-config.json` - GitHub Pages configuration

### Partially Exportable

- `secret-names.txt` - Secret names only (VALUES CANNOT BE EXPORTED)
- `deploy-keys.json` - Metadata only (PRIVATE KEYS CANNOT BE EXPORTED)

### NOT Exportable via API

- Social preview image (must re-upload via web UI)
- Some security scanning settings (web UI only)
- Actions workflow permissions (some require web UI)
- Stars/Watchers/Forks counts (cannot be transferred)

## Using This Export

See `import-repo-config.sh` to recreate these settings in a new repository.

## Warnings

1. **Secrets**: You must manually recreate secrets - values cannot be exported
2. **Deploy Keys**: You must have the original private keys to recreate
3. **Social Preview**: Must re-upload image via Settings > Social Preview
4. **History**: This export does NOT include commit history, issues, PRs, or discussions
