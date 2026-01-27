# Security Audit Report - Oscura v0.6.0

**Date**: 2026-01-25  
**Auditor**: AI Security Analysis (Claude Code)  
**Scope**: Complete codebase security assessment for v0.6.0 release  
**Status**: COMPLETED - 7 findings (2 HIGH, 3 MEDIUM, 2 LOW)

---

## Executive Summary

This comprehensive security audit of the Oscura hardware reverse engineering framework identified **7 security issues** across input validation, dependency management, authentication, and deserialization security.

**Critical Findings**:

- **NONE** - No critical vulnerabilities requiring immediate remediation

**High Priority Findings**:

- 2 HIGH severity issues requiring fixes before v0.6.0 release
- 3 MEDIUM severity issues to address soon
- 2 LOW severity informational items

**Overall Security Posture**: **GOOD** - The codebase demonstrates strong security practices in most areas. Key strengths include:

- Secure use of MD5 (marked with `usedforsecurity=False`)
- Path resolution for preventing path traversal
- No hardcoded secrets found
- No SQL injection vectors (parameterized queries used)
- HTTPS/TLS enforcement (no `verify=False` found)

---

## Vulnerability Summary

| ID | Severity | Category | Component | Status |
|----|----------|----------|-----------|--------|
| SEC-001 | HIGH | Arbitrary Code Execution | jupyter/magic.py | ACCEPTED RISK |
| SEC-002 | HIGH | Authentication | api/rest_server.py | FIX REQUIRED |
| SEC-003 | MEDIUM | Deserialization | utils/performance/caching.py | FIX REQUIRED |
| SEC-004 | MEDIUM | Command Injection | cli/config_cmd.py | FIX REQUIRED |
| SEC-005 | MEDIUM | Dependency Vulnerabilities | cryptography, nbconvert, py | UPDATE REQUIRED |
| SEC-006 | LOW | Subprocess Security | export/* | MITIGATED |
| SEC-007 | LOW | CORS Configuration | api/rest_server.py | DOCUMENT RISK |

---

## Detailed Findings

### SEC-001: Arbitrary Code Execution in Jupyter Magic (HIGH - ACCEPTED RISK)

**File**: `src/oscura/jupyter/magic.py:310`

**Issue**:
The `%%oscura` Jupyter magic command uses `exec()` to execute user-provided cell content directly.

```python
# Execute cell
exec(cell, namespace)
```

**Risk**:

- **Impact**: HIGH - Arbitrary code execution
- **Likelihood**: LOW - Jupyter notebooks are inherently trusted environments
- **Exploitability**: User must already have access to Jupyter environment

**Justification for Acceptance**:
This is the **intended behavior** for Jupyter magic commands. All Jupyter cells execute arbitrary Python code by design. This is not a vulnerability in the traditional sense, as:

1. Users running Jupyter have already accepted code execution risks
2. The magic command provides no additional attack surface beyond normal Jupyter
3. Standard IPython magic commands use the same pattern

**Recommendation**:
**ACCEPT RISK** - Document this behavior in security documentation. No fix required.

**Status**: ✅ ACCEPTED - Working as designed

---

### SEC-002: Missing API Authentication (HIGH)

**File**: `src/oscura/api/rest_server.py`

**Issue**:
REST API server accepts an `api_key` parameter but **does not enforce it**. All routes are publicly accessible without authentication.

```python
def __init__(
    self,
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str | None = None,  # Stored but never checked
    ...
):
    self.api_key = api_key
    # Routes registered without authentication middleware
```

**Risk**:

- **Impact**: HIGH - Unauthorized access to analysis, file uploads, session management
- **Likelihood**: HIGH - Default configuration is insecure
- **Exploitability**: Trivial - No authentication required

**Attack Scenario**:

```bash
# Attacker uploads malicious file without authentication
curl -X POST http://victim:8000/api/v1/analyze \
  -F "file=@malicious.bin"

# No API key required - request succeeds
```

**Recommendation**:
**FIX REQUIRED** - Implement authentication middleware:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security)
) -> None:
    if not self.api_key:
        return  # No auth required if not configured

    if not credentials or credentials.credentials != self.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Apply to all routes
@self.app.post("/api/v1/analyze", dependencies=[Depends(verify_api_key)])
async def analyze(...):
    ...
```

**Status**: ⚠️ FIX REQUIRED

---

### SEC-003: Unsafe Pickle Deserialization (MEDIUM)

**Files**:

- `src/oscura/utils/performance/caching.py:484`
- `src/oscura/core/cache.py:470`
- `src/oscura/utils/memory_advanced.py:988`

**Issue**:
Cache systems use `pickle.load()` to deserialize cached data without validation. While caches are in trusted directories, pickle deserialization can execute arbitrary code if cache files are compromised.

```python
with open(cache_file, "rb") as f:
    # No validation before deserialization
    loaded_entry: CacheEntry = pickle.load(f)
```

**Risk**:

- **Impact**: HIGH - Arbitrary code execution if cache poisoned
- **Likelihood**: LOW - Attacker needs write access to cache directory
- **Exploitability**: Moderate - Requires local file write access

**Attack Scenario**:

1. Attacker gains write access to `~/.cache/oscura/` (e.g., via another vulnerability)
2. Attacker creates malicious pickle file that executes code on load
3. Next time Oscura runs, malicious code executes

**Recommendation**:
**FIX REQUIRED** - Add HMAC validation to cache entries:

```python
import hmac
import hashlib

# Generate cache signature
def _save_cache(self, key: str, value: Any) -> None:
    data = pickle.dumps(value)
    sig = hmac.new(self._cache_key, data, hashlib.sha256).digest()
    with open(cache_file, "wb") as f:
        f.write(sig)
        f.write(data)

# Validate before load
def _load_cache(self, key: str) -> Any:
    with open(cache_file, "rb") as f:
        sig = f.read(32)
        data = f.read()

    expected = hmac.new(self._cache_key, data, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise SecurityError("Cache integrity check failed")

    return pickle.loads(data)
```

**Note**: `sessions/legacy.py:750` already implements HMAC validation correctly.

**Status**: ⚠️ FIX REQUIRED

---

### SEC-004: Command Injection in Config Editor (MEDIUM)

**File**: `src/oscura/cli/config_cmd.py:269`

**Issue**:
Config edit command passes user-controlled `$EDITOR` environment variable directly to `subprocess.run()` without validation.

```python
editor = os.environ.get("EDITOR", "nano")
subprocess.run([editor, str(config_path)], check=True)
```

**Risk**:

- **Impact**: MEDIUM - Command execution with user privileges
- **Likelihood**: LOW - Requires attacker to control user's environment
- **Exploitability**: Low - Limited attack surface (CLI tool)

**Attack Scenario**:

```bash
# Attacker sets malicious EDITOR
export EDITOR="rm -rf / #"
oscura config edit
# Attempts to execute malicious command
```

**Recommendation**:
**FIX REQUIRED** - Validate editor against allowlist:

```python
ALLOWED_EDITORS = {
    "nano", "vim", "vi", "emacs", "code", "subl", "gedit", "kate"
}

def _get_safe_editor() -> str:
    editor = os.environ.get("EDITOR", "nano")
    # Extract base command (remove args)
    editor_cmd = Path(editor).name.split()[0]

    if editor_cmd not in ALLOWED_EDITORS:
        logger.warning(f"Untrusted editor '{editor}', falling back to nano")
        return "nano"

    return editor

# Use validated editor
editor = _get_safe_editor()
subprocess.run([editor, str(config_path)], check=True)
```

**Status**: ⚠️ FIX REQUIRED

---

### SEC-005: Dependency Vulnerabilities (MEDIUM)

**Issue**:
Safety scan identified 3 vulnerable dependencies:

1. **cryptography 43.0.3** (CVE-2024-12797)
   - Severity: MEDIUM
   - Impact: OpenSSL vulnerability in static binary
   - Fix: Update to >= 44.0.1 when available

2. **nbconvert 7.16.6** (CVE-2025-53000)
   - Severity: MEDIUM  
   - Impact: Uncontrolled search path (Windows only)
   - Fix: Update when patch available
   - Mitigation: Linux/macOS not affected

3. **py 1.11.0** (CVE-2022-42969 - DISPUTED)
   - Severity: LOW
   - Impact: ReDoS in SVN info parsing
   - Fix: Remove dependency (deprecated package)
   - Note: Oscura doesn't use SVN features

**Recommendation**:
**UPDATE REQUIRED** - Update dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    # Remove deprecated 'py' package (pytest no longer needs it)
    # "py>=1.11.0",  # REMOVE
]

jupyter = [
    "nbconvert>=7.17.0",  # UPDATE when available
]

# Monitor cryptography updates
dependencies = [
    "cryptography>=44.0.1",  # UPDATE when available
]
```

**Status**: ⚠️ UPDATE IN PROGRESS

---

### SEC-006: Subprocess Command Injection Risk (LOW - MITIGATED)

**Files**:

- `src/oscura/export/kaitai_struct.py:488`
- `src/oscura/export/wireshark_dissector.py:624`
- `src/oscura/export/wireshark/validator.py:33`

**Issue**:
External tools (`ksc`, `luac`) are executed via `subprocess.run()`. Code is **already secure** with proper use of list arguments and timeouts.

```python
# SECURE - Uses list arguments (no shell=True)
result = subprocess.run(
    ["luac", "-p", "-"],
    input=lua_code.encode("utf-8"),
    capture_output=True,
    timeout=5,  # Timeout protection
    check=False,
)
```

**Risk**:

- **Impact**: LOW - No injection possible (list args used correctly)
- **Likelihood**: NONE - Code is secure
- **Exploitability**: None

**Recommendation**:
**NO ACTION REQUIRED** - Code follows security best practices:

- ✅ List arguments prevent shell injection
- ✅ Timeout prevents DoS
- ✅ No user input in command path
- ✅ Graceful fallback if tool missing

**Status**: ✅ SECURE - No changes needed

---

### SEC-007: CORS Wildcard Configuration (LOW)

**File**: `src/oscura/api/rest_server.py:389`

**Issue**:
Default CORS configuration allows all origins (`["*"]`), which could enable CSRF attacks if deployed publicly.

```python
if enable_cors:
    origins = cors_origins or ["*"]  # Wildcard default
    self.app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

**Risk**:

- **Impact**: LOW - CSRF attacks possible
- **Likelihood**: LOW - Server defaults to localhost
- **Exploitability**: Low - Requires public deployment

**Recommendation**:
**DOCUMENT RISK** - Add security warning to documentation:

```python
# Add docstring warning
"""
Security Warning:
    Default CORS configuration allows all origins. For production deployments,
    explicitly configure allowed origins:

        server = RESTAPIServer(
            enable_cors=True,
            cors_origins=["https://trusted-domain.com"]
        )
"""
```

**Status**: ⚠️ DOCUMENT - Add to security guide

---

## Security Strengths

The codebase demonstrates **strong security practices** in several areas:

### ✅ Secure Cryptography Usage

- **MD5 usage is safe**: All MD5 uses explicitly marked `usedforsecurity=False` for cache keys
- **No weak crypto**: No use of broken algorithms for security purposes
- **HMAC validation**: Session files use HMAC-SHA256 for integrity (legacy.py:745)

### ✅ SQL Injection Prevention

- **Parameterized queries**: All database queries use parameter binding
- **No string interpolation**: Zero instances of f-strings in SQL

```python
# Correct parameterized query pattern
cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
```

### ✅ Path Traversal Prevention

- **Path resolution**: Paths resolved with `.resolve()` to prevent `../` attacks
- **Validation**: File paths validated before operations

### ✅ No Hardcoded Secrets

- **Environment variables**: API keys read from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- **No credentials**: Zero hardcoded passwords, tokens, or API keys found

### ✅ HTTPS Enforcement

- **No SSL bypass**: Zero instances of `verify=False` in requests
- **Secure defaults**: No insecure HTTP for external connections

---

## Recommendations Summary

### Immediate Actions (Before v0.6.0 Release)

1. **SEC-002**: Implement REST API authentication middleware (2-4 hours)
2. **SEC-003**: Add HMAC validation to cache deserialization (4-6 hours)
3. **SEC-004**: Add editor validation in config command (1 hour)
4. **SEC-005**: Update vulnerable dependencies when patches available (ongoing)

### Documentation Updates

1. **SEC-001**: Document Jupyter magic security considerations
2. **SEC-007**: Add CORS security guidance to deployment docs

### Long-term Improvements

1. Consider replacing pickle with safer serialization (JSON/msgpack) for caches
2. Add security testing to CI pipeline (bandit, safety scans)
3. Implement rate limiting for REST API endpoints
4. Add security.md file with responsible disclosure policy

---

## Testing Recommendations

### Security Test Cases to Add

```python
# Test API authentication
def test_api_requires_authentication():
    """Verify API key is enforced when configured."""
    server = RESTAPIServer(api_key="secret")
    response = client.post("/api/v1/analyze", headers={})
    assert response.status_code == 401

# Test cache integrity
def test_cache_rejects_tampered_data():
    """Verify HMAC prevents cache poisoning."""
    cache.set("key", {"data": "trusted"})
    # Tamper with cache file
    with open(cache_file, "wb") as f:
        f.write(b"malicious_pickle_data")

    with pytest.raises(SecurityError):
        cache.get("key")

# Test editor validation
def test_config_edit_rejects_malicious_editor():
    """Verify untrusted editors are rejected."""
    os.environ["EDITOR"] = "rm -rf /"
    with pytest.raises(SecurityError):
        config_edit(config_path)
```

---

## Compliance

### OWASP Top 10 (2021) Coverage

| Category | Status | Notes |
|----------|--------|-------|
| A01: Broken Access Control | ⚠️ PARTIAL | SEC-002: API auth missing |
| A02: Cryptographic Failures | ✅ GOOD | Secure crypto usage |
| A03: Injection | ✅ GOOD | No SQL injection, parameterized queries |
| A04: Insecure Design | ✅ GOOD | Strong architecture |
| A05: Security Misconfiguration | ⚠️ PARTIAL | SEC-007: CORS config |
| A06: Vulnerable Components | ⚠️ PARTIAL | SEC-005: Dependency updates needed |
| A07: Auth/Identity Failures | ⚠️ PARTIAL | SEC-002: Auth not enforced |
| A08: Data Integrity Failures | ⚠️ PARTIAL | SEC-003: Pickle validation |
| A09: Logging/Monitoring | ✅ GOOD | Comprehensive logging |
| A10: Server-Side Request Forgery | ✅ N/A | No SSRF vectors |

---

## Conclusion

**Overall Assessment**: The Oscura codebase demonstrates **strong security fundamentals** with no critical vulnerabilities. The identified issues are manageable and can be addressed before the v0.6.0 release.

**Security Score**: 7.5/10

- Strong: Crypto, SQL injection prevention, secrets management
- Needs improvement: API authentication, deserialization security

**Recommendation**: **APPROVE v0.6.0 release** after addressing HIGH and MEDIUM severity findings (estimated 8-12 hours of work).

---

## Appendix A: Audit Methodology

### Tools Used

- `safety check` - Dependency vulnerability scanning
- `grep/ripgrep` - Pattern-based vulnerability detection
- Manual code review - Security-critical functions

### Coverage

- 100% of Python source files in `src/oscura/`
- Focused review of:
  - File I/O operations (path traversal)
  - subprocess calls (command injection)
  - Deserialization (pickle security)
  - Database queries (SQL injection)
  - API endpoints (authentication)
  - Cryptography usage

### Excluded from Scope

- Third-party dependencies (internal code only)
- Test code security (tests/* directory)
- Performance/availability issues
- Code quality (separate code review)

---

## Appendix B: References

- [OWASP Top 10 (2021)](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [CWE-502: Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)
- [CWE-78: OS Command Injection](https://cwe.mitre.org/data/definitions/78.html)
- [Pickle Security Documentation](https://docs.python.org/3/library/pickle.html#module-pickle)

---

**Report Generated**: 2026-01-25  
**Version**: 1.0  
**Signed**: AI Security Analysis (Claude Code)
