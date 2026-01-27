"""Web dashboard for Oscura protocol analysis.

This module provides interactive web-based UI for protocol analysis,
real-time visualization, and session management.

Example:
    >>> from oscura.api.server import WebDashboard
    >>> dashboard = WebDashboard(host="0.0.0.0", port=5000)
    >>> dashboard.run()
    >>> # Visit http://0.0.0.0:5000 for web UI
"""

from __future__ import annotations

from oscura.api.server.dashboard import WebDashboard

__all__ = ["WebDashboard"]
