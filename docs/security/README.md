# Security Documentation

This directory contains security-related documentation for the Oscura project.

## Documents

### Security Audits

- **[Security Audit 2026-01-25](security-audit-2026-01-25.md)** - Comprehensive security assessment for v0.6.0 release
  - 7 findings identified (2 HIGH, 3 MEDIUM, 2 LOW)
  - Zero critical vulnerabilities
  - Overall security score: 7.5/10
  - Status: Approved for v0.6.0 after addressing HIGH/MEDIUM findings

## Security Contacts

For security issues, please see our responsible disclosure policy in SECURITY.md (root directory).

## Security Best Practices

### For Contributors

When contributing code to Oscura, please follow these security guidelines:

1. **Input Validation**
   - Always validate and sanitize user inputs
   - Use parameterized queries for database operations
   - Validate file paths to prevent directory traversal

2. **Secrets Management**
   - Never hardcode API keys, passwords, or tokens
   - Use environment variables for sensitive configuration
   - Mark cryptographic operations appropriately (`usedforsecurity` flag)

3. **Deserialization**
   - Avoid pickle for untrusted data
   - Use HMAC validation for cache integrity
   - Prefer JSON/msgpack for data exchange

4. **Authentication**
   - Implement proper authentication for API endpoints
   - Use industry-standard auth patterns (Bearer tokens, API keys)
   - Never disable SSL/TLS verification

5. **Dependencies**
   - Keep dependencies updated
   - Run `safety check` regularly
   - Monitor security advisories

### For Users

When deploying Oscura in production:

1. **API Security**
   - Always configure API keys for REST API server
   - Use HTTPS for all external connections
   - Configure CORS with explicit allowed origins (not wildcard)

2. **File Permissions**
   - Restrict access to cache directories
   - Use appropriate file permissions (0600 for sensitive data)
   - Validate uploaded files before processing

3. **Network Security**
   - Deploy behind firewall/reverse proxy
   - Implement rate limiting
   - Monitor for unusual activity

## Reporting Vulnerabilities

See `SECURITY.md` in the repository root for our responsible disclosure policy.

## Security Changelog

- **2026-01-25**: Initial comprehensive security audit completed
  - 7 findings identified
  - 0 critical vulnerabilities
  - HIGH priority fixes in progress
