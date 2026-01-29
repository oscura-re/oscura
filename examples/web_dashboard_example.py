"""Example: Using the Oscura Web Dashboard.

This example demonstrates how to start the web dashboard server
and access the interactive UI for protocol analysis.

# SKIP_VALIDATION: oscura.web module not yet implemented

Requirements:
    pip install 'fastapi[all]' uvicorn

Usage:
    python examples/web_dashboard_example.py

Then visit http://127.0.0.1:5000 in your browser.
"""

from oscura.web import WebDashboard
from oscura.web.dashboard import DashboardConfig


def main() -> None:
    """Run web dashboard server."""
    # Configure dashboard
    config = DashboardConfig(
        title="Oscura Protocol Analysis",
        theme="dark",  # or "light"
        max_file_size=100 * 1024 * 1024,  # 100 MB
        enable_websocket=True,
        session_timeout=3600.0,  # 1 hour
    )

    # Create dashboard instance
    dashboard = WebDashboard(
        host="127.0.0.1",
        port=5000,
        config=config,
    )

    # Start server
    print("Starting Oscura Web Dashboard...")
    print("Visit http://127.0.0.1:5000 in your browser")
    print("Press Ctrl+C to stop")

    dashboard.run(reload=True)  # reload=True for development


if __name__ == "__main__":
    main()
