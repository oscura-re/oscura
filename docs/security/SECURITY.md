# Security Considerations

This document outlines security considerations, best practices, and vulnerability disclosure procedures for Oscura.

## Security Model

Oscura is a hardware reverse engineering framework designed for **security research** and **analysis of untrusted systems**. The security model assumes:

1. **Input files are untrusted**: Waveform captures, protocol specifications, and firmware binaries may come from untrusted sources
2. **Analysis environment is trusted**: The machine running Oscura should be under your control
3. **Output artifacts are for research**: Generated dissectors, protocol specs, and reports are for analysis purposes

## Cryptography

### LoRaWAN Cryptographic Operations

**Module**: `src/oscura/iot/lorawan/crypto.py`

The LoRaWAN cryptographic implementation uses the modern **cryptography** library (not deprecated PyCrypto):

- **AES-128 encryption**: CTR mode for payload encryption following LoRaWAN Specification 1.0.3
- **CMAC**: AES-CMAC for Message Integrity Code (MIC) computation
- **ECB mode usage**: Required by LoRaWAN spec for counter block encryption (safe because each block is unique)

**Security notes**:

- ECB mode is only used for encrypting unique counter blocks (Section 4.3.3 of LoRaWAN spec)
- All cryptographic operations follow IEEE and LoRaWAN Alliance specifications
- Keys must be 16 bytes (128 bits) as required by LoRaWAN

**Dependencies**:

```bash
pip install oscura[iot]  # Installs cryptography>=42.0.0
```

### Hash Functions

**MD5 usage**: MD5 is used in several modules for **non-cryptographic purposes only**:

- **Cache key generation** (`src/oscura/performance/caching.py`): Hashing numpy arrays and function arguments for cache keys
- **Data checksums** (`src/oscura/utils/memory_advanced.py`, `src/oscura/utils/memory_extensions.py`): Cache invalidation via data comparison

All MD5 calls include `usedforsecurity=False` to explicitly mark non-cryptographic usage:

```python
# Correct usage for cache keys (NOT for security)
cache_key = hashlib.md5(data, usedforsecurity=False).hexdigest()

# For security-critical operations, use SHA256 or better
secure_hash = hashlib.sha256(data).hexdigest()
```

**When to use each**:

- MD5 with `usedforsecurity=False`: Cache keys, checksums, non-security data comparison
- SHA256: Secure hashing, key derivation, integrity checks, security-critical operations
- BLAKE2: High-performance alternative to SHA256 for large data

## Deserialization

### Pickle Usage

**Security risk**: Python's `pickle` module can execute arbitrary code during deserialization.

**Oscura's approach**: Pickle is used only for **trusted local cache files**:

1. **Disk cache** (`src/oscura/performance/caching.py`): Stores expensive computation results in `~/.cache/oscura/`
2. **Memory cache** (`src/oscura/utils/memory_advanced.py`): Temporary cache in system temp directory
3. **Session files** (`src/oscura/session/session.py`): User-created analysis sessions

**Security notes**:

- ✅ **Safe**: Loading cache files from trusted local directories (your own cache)
- ❌ **Unsafe**: Loading pickle files from network, email attachments, or untrusted sources
- ✅ **Safe**: Session files you created yourself
- ❌ **Unsafe**: Session files from unknown sources

**Best practices**:

```python
# DO: Load your own cache/session files
session = BlackBoxSession.load("my_analysis.pkl")

# DON'T: Load session files from untrusted sources
# session = BlackBoxSession.load(downloaded_file)  # DANGEROUS!

# If you must load untrusted data, use JSON instead:
with open("untrusted_data.json") as f:
    data = json.load(f)  # Safe - no code execution
```

**Recommendations**:

1. Never load pickle files from untrusted sources
2. Use JSON export for sharing analysis results: `session.export_json("results.json")`
3. Clear cache if system is compromised: `rm -rf ~/.cache/oscura/`

## Input Validation

### File Format Parsers

Oscura parses multiple binary file formats (WFM, PCAP, BLF, HDF5, etc.) from potentially untrusted sources.

**Protections in place**:

- Size limits on uploaded files
- Format validation before parsing
- Exception handling for malformed files
- Memory limits to prevent OOM attacks

**Recommendations**:

1. Run Oscura in isolated environment (VM, container) when analyzing untrusted captures
2. Set memory limits: `ulimit -v $((8 * 1024 * 1024))  # 8GB limit`
3. Use sandboxing for firmware analysis: `firejail oscura analyze firmware.bin`

### XML Parsing

**Current status**: XML parsing (FlexRay FIBEX) uses standard `xml.etree.ElementTree` (flagged by bandit as MEDIUM).

**Future improvement**: Consider `defusedxml` for parsing untrusted XML files.

**Workaround**:

```python
# For now, only parse FIBEX files from trusted sources
# Future: Integrate defusedxml
```

## Network Operations

### REST API Server

**Module**: `src/oscura/api/rest_server.py`

**Security features**:

- Optional API key authentication
- Optional rate limiting
- CORS configuration for web dashboards
- File upload size limits
- Session timeout and cleanup

**Production deployment checklist**:

- [ ] Enable API key authentication: `--api-key your-secret-key`
- [ ] Enable rate limiting: `--rate-limit 100` (requests/minute)
- [ ] Use HTTPS reverse proxy (nginx, caddy)
- [ ] Configure CORS origins: `--cors-origins https://dashboard.example.com`
- [ ] Set file upload limits: `--max-upload-mb 100`
- [ ] Run as non-root user
- [ ] Enable request logging for audit trail

**Example production config**:

```bash
oscura serve \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$(openssl rand -hex 32)" \
  --rate-limit 100 \
  --max-sessions 50 \
  --cors-origins "https://dashboard.example.com" \
  --max-upload-mb 100
```

### Binding to All Interfaces

**Current status**: Some components bind to `0.0.0.0` for convenience (flagged as MEDIUM).

**Recommendation**: For production, bind to `127.0.0.1` (localhost) and use reverse proxy for external access.

## Dependency Security

### Security Scanning

Oscura uses multiple tools for security scanning:

- **bandit**: Python security scanner (checks for common security issues)
- **pip-audit**: Scans dependencies for known CVEs
- **GitHub Dependabot**: Automated dependency updates

**Current status** (as of 2026-01-25):

- ✅ **0 HIGH** severity issues (all resolved)
- ⚠️ **16 MEDIUM** severity issues (documented, acceptable risk)
  - 7× Pickle deserialization (only from trusted local cache)
  - 3× URL open (internal use only)
  - 3× SQL string construction (parameterized queries)
  - 1× ECB mode (required by LoRaWAN spec, safe usage)
  - 1× XML parsing (only trusted FIBEX files)
  - 1× Exec detection (Jupyter magic, controlled environment)

### Known CVEs in Dependencies

**nbconvert** (Jupyter export):

- **CVE-2025-53000**: Windows-only code execution via PDF+SVG conversion
- **Risk**: Limited to Windows, specific workflow (PDF with SVG)
- **Mitigation**: Constraint added `nbconvert>=7.0.0,<8.0.0` to avoid vulnerable versions
- **Status**: Tracked, fix pending in nbconvert 8.x release

### Dependency Update Policy

1. Security updates applied within **7 days** of disclosure
2. Automated Dependabot PRs reviewed weekly
3. Breaking changes tested with full test suite
4. Security-critical dependencies pinned with version constraints

## Vulnerability Disclosure

### Reporting Security Issues

**DO NOT** file public GitHub issues for security vulnerabilities.

**Contact**: security@oscura.dev (if available) or create a private security advisory on GitHub.

**What to include**:

1. Description of the vulnerability
2. Steps to reproduce
3. Impact assessment
4. Suggested fix (if available)
5. Your name/handle for credit (optional)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Fix release**: Within 30 days for HIGH, 90 days for MEDIUM
- **Public disclosure**: After fix is released and users have time to update

### Security Advisories

Published security advisories: https://github.com/yourusername/oscura/security/advisories

### Hall of Fame

We maintain a security researchers hall of fame for responsible disclosures.

## Security Best Practices

### For Users

1. **Keep Oscura updated**: `pip install --upgrade oscura`
2. **Use virtual environments**: Isolate dependencies
3. **Validate input files**: Check file hashes, sources
4. **Use sandboxing**: Run in VM/container for untrusted data
5. **Review generated code**: Inspect Wireshark dissectors before use
6. **Secure API deployments**: Enable authentication, HTTPS, rate limiting

### For Developers

1. **Input validation**: Validate all user inputs, file formats
2. **Avoid pickle for untrusted data**: Use JSON, protobuf, or msgpack
3. **Use parameterized queries**: Prevent SQL injection
4. **Sanitize outputs**: Escape HTML, shell commands
5. **Security scanning**: Run `bandit` and `pip-audit` before commits
6. **Dependency updates**: Review security advisories weekly
7. **Code review**: All PRs reviewed for security issues

### For Researchers

1. **Responsible disclosure**: Report vulnerabilities privately first
2. **Isolated environment**: Use VMs for analyzing malicious hardware
3. **Data privacy**: Don't share captures with sensitive data
4. **Legal compliance**: Follow laws regarding RE in your jurisdiction

## References

- **LoRaWAN Security**: https://lora-alliance.org/resource_hub/lorawan-specification-v1-0-3/
- **Python Security**: https://python.readthedocs.io/en/stable/library/security_warnings.html
- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **CWE/SANS Top 25**: https://cwe.mitre.org/top25/archive/2023/2023_top25_list.html

## Compliance

Oscura follows security best practices from:

- **NIST Cybersecurity Framework**: Identify, Protect, Detect, Respond, Recover
- **OWASP Secure Coding Practices**: Input validation, authentication, session management
- **IEEE 1685-2009**: Profiling methodology
- **ISO/IEC 27001**: Information security management

## License

Security documentation is part of Oscura and licensed under MIT License.

---

**Last Updated**: 2026-01-25
**Version**: 0.6.0
