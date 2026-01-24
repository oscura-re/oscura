# Security Policy

## Supported Versions

|Version|Supported|
|---|---||0.1.x|Yes||< 0.1|No|---

## Reporting a Vulnerability

**We take security seriously.** If you discover a security vulnerability, please report it responsibly.

### How to Report

**Preferred**: [Private security advisory](https://github.com/oscura-re/oscura/security/advisories/new) on GitHub

**Alternative**: Email security@oscura-re.dev

**Do not** report vulnerabilities through public issues unless already disclosed.

### What to Include

- Type of vulnerability
- Affected source fil es and locations
- Step-by-step reproduction
- Impact assessment
- Proof-of-concept (if available)
- Suggested fi x (if you have one)

### Response Timeline

- **Initial response**: Within 48 hours
- **Severity assessment**: Within 7 days
- **Patch release**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 60 days
- **Public disclosure**: After patch release

---

## Security Measures

Oscura implements the following security practices:

### Code Security

- Static analysis with `bandit` for Python security issues
- Dependency vulnerability scanning with `safety`
- No execution of untrusted code from input files
- Input validation for all file loaders

### CI/CD Security

- Automated security scanning in CI pipeline
- Dependency updates monitored
- No secrets in repository

### Data Handling

- File loaders validate format before parsing
- Memory limits enforced for large file processing
- No network operations in core library

### Session File Security

**Session files (.tks) use HMAC-SHA256 signatures** for integrity verification:

- **New format (v0.2.0+)**: All session files include cryptographic signatures
- **Integrity check**: HMAC signature verified on load (detects tampering)
- **Backward compatible**: Legacy files load with warning
- **Security note**: Session files still use pickle serialization - only load from trusted sources

**Best practices**:

````python
from oscura.session import Session, load_session

# Save with signature (default)
session.save('analysis.tks')  # Includes HMAC signature

# Load with verification (default)
session = load_session('analysis.tks')  # Verifies signature

# Legacy files trigger warning
session = load_session('old_file.tks')  # UserWarning: no signature
```python

**Security warnings**:

- ⚠️ **Only load .tks files from trusted sources** (pickle deserialization)
- ✅ Signature verification prevents tampering but not malicious content
- 🔄 Re-save legacy files to add signatures: `session.save('updated.tks')`
- 🚫 Never load .tks files from untrusted or unknown sources

For **secure data exchange**, use alternative formats:

```python
# Instead of pickle-based .tks files
session.export_to_json('safe_export.json')  # Human-readable, safe
session.export_to_hdf5('safe_export.h5')    # Efficient, safe
```markdown

---

## Threat Model

### Untrusted Inputs

Oscura treats the following as untrusted:

- User-provided waveform files (may contain malformed data)
- User-provided file paths (path traversal risk)
- Protocol definitions (pattern matching only, no code execution)
- CSV/binary data imports

### Attack Vectors

1. **Malformed files**: Buffer overflows, memory exhaustion
2. **Path traversal**: Unsanitized file paths
3. **Session file deserialization**: Malicious .tks files (pickle-based)
   - **Mitigation**: HMAC signatures detect tampering
   - **Limitation**: Only load from trusted sources
4. **DoS**: Large files, complex patterns, memory exhaustion
5. **Dependency vulnerabilities**: Third-party package issues

---

## Security Best Practices

### For Users

**1. Validate inputs:**

```python
from pathlib import Path

# Good: Validate paths
if not Path(user_path).resolve().is_relative_to(safe_dir):
    raise ValueError("Path outside safe directory")

# Bad: Trust user input
loader.load(user_input_path)  # Don't do this without validation
```python

**2. Set memory limits for large files:**

```python
from oscura.loaders import load_with_limits

# Set maximum file size and sample count
trace = load_with_limits(
    filepath,
    max_file_size_mb=100,
    max_samples=10_000_000
)
```bash

**3. Run vulnerability scans:**

```bash
uv pip install safety pip-audit
safety check
pip-audit
```bash

### For Deployment

**Production checklist:**

- [ ] Validate all user-provided file paths
- [ ] Set file size limits for uploads
- [ ] Run vulnerability scans: `safety check`
- [ ] Keep dependencies updated: `uv pip install --upgrade oscura`
- [ ] Use principle of least privilege for file system access
- [ ] Monitor memory usage during batch processing

**Docker example:**

```dockerfile
FROM python:3.12-slim
RUN uv pip install oscura
USER nonroot  # Don't run as root
COPY --chown=nonroot:nonroot . /app
WORKDIR /app
```markdown

---

## Dependency Security

### Automated Scanning

Oscura uses:

- **Dependabot**: Automatic dependency updates (when enabled)
- **CI scanning**: Vulnerability monitoring in CI pipeline

### Manual Scanning

Check your installation:

```bash
uv pip install safety pip-audit
safety check
pip-audit
```markdown

---

## Known Limitations

### Binary File Parsing

- **Risk**: Malformed binary files may cause excessive memory usage
- **Mitigation**: Use with trusted data sources or implement memory limits
- **User action**: Set `max_samples` and `max_file_size_mb` parameters

### Protocol Decoders

- **Risk**: Complex patterns may cause CPU exhaustion
- **Mitigation**: Pattern matching only, not arbitrary code execution
- **User action**: Set timeout limits for decoding operations

### Large File Processing

- **Risk**: Processing very large files may exhaust memory
- **Mitigation**: Use chunked processing APIs
- **User action**: Use streaming/chunked APIs for files >100MB

---

## Known Dependency Vulnerabilities

### CVE-2025-53000: nbconvert Uncontrolled Search Path (Windows)

**Status**: No patch available (as of 2026-01-16)
**Severity**: High (CVSS 8.5)
**Affected**: nbconvert ≤ 7.16.6 (transitive dependency via `jupyter` extra)
**Platform**: Windows only
**Attack vector**: Local, requires user interaction

**Description**: On Windows, converting notebooks with SVG output to PDF via `jupyter nbconvert --to pdf` can execute arbitrary code if a malicious `inkscape.bat` file exists in the working directory.

**Impact**: Windows users of Oscura's optional `jupyter` extra who convert notebooks to PDF.

**Mitigation**:

- **Recommended**: Avoid using `jupyter nbconvert --to pdf` with SVG content on Windows until patched
- **Alternative**: Use Linux/macOS, or convert to HTML instead of PDF
- **Workaround**: Ensure working directory contains no untrusted `.bat` files before conversion

**Tracking**:

- GitHub Advisory: [GHSA-xm59-rqc7-hhvf](https://github.com/advisories/GHSA-xm59-rqc7-hhvf)
- Oscura Issue: https://github.com/oscura-re/oscura/security/dependabot/1

**Resolution**: Will update to patched version when available.

---

## Out of Scope

The following are **not** security vulnerabilities in Oscura:

### 1. Resource Usage with Large Files

Processing large waveform files (>1GB) is **resource-intensive by design**, not a vulnerability.

**User's responsibility**: Set reasonable limits in your application
**Oscura provides**: Chunked processing and streaming APIs

### 2. Invalid Measurement Results

Incorrect measurements from malformed input data is a **data quality issue**, not a security vulnerability.

**User's responsibility**: Validate input data quality
**Oscura provides**: Data validation utilities

---

## Disclosure Policy

- We follow **coordinated disclosure**
- Security fixes are released ASAP
- CVEs assigned when applicable
- Public disclosure after patch + reasonable upgrade time
- Credits given to researchers who report responsibly

---

## Security Contact

- **GitHub Security Advisories**: [Create advisory](https://github.com/oscura-re/oscura/security/advisories/new)
- **Email**: security@oscura-re.dev
- **Issues**: Use "security" label for non-sensitive issues

---

## Security Updates

Security updates are released as patch versions and announced in:

- GitHub Security Advisories
- CHANGELOG.md

Subscribe to repository notifications for security alerts.
````
