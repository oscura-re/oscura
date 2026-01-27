"""Oscura Export Command - Export Analysis Results.

Provides CLI for exporting analysis sessions and results to various formats.


Example:
    $ oscura export json session.tks --output results.json
    $ oscura export html session.tks --output report.html
    $ oscura export wireshark session.tks --output dissector.lua
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("oscura.cli.export")


@click.command()
@click.argument(
    "format",
    type=click.Choice(
        ["json", "html", "csv", "matlab", "wireshark", "scapy", "kaitai"], case_sensitive=False
    ),
)
@click.argument("session", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output file path.",
)
@click.option(
    "--include-traces",
    is_flag=True,
    default=True,
    help="Include trace data in export (default: True).",
)
@click.pass_context
def export(
    ctx: click.Context,
    format: str,
    session: str,
    output: str,
    include_traces: bool,
) -> None:
    """Export analysis session to various formats.

    Supports exporting to JSON, HTML, CSV, MATLAB, Wireshark dissector,
    Scapy layer, and Kaitai struct formats.

    Args:
        ctx: Click context object.
        format: Export format.
        session: Path to session file (.tks).
        output: Output file path.
        include_traces: Include trace data in export.

    Examples:

        \b
        # Export to JSON
        $ oscura export json session.tks --output results.json

        \b
        # Generate HTML report
        $ oscura export html session.tks --output report.html

        \b
        # Generate Wireshark dissector
        $ oscura export wireshark session.tks --output protocol.lua
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Exporting session {session} to {format}")

    try:
        # Note: Session loading functionality has been redesigned
        # For now, this is a placeholder - session export needs to be reimplemented
        # using the new AnalysisSession architecture
        raise NotImplementedError(
            "Session export has been redesigned. Use the new AnalysisSession API instead."
        )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _export_json(session: Any, output_path: Path, include_traces: bool) -> None:
    """Export session to JSON.

    Args:
        session: Session object.
        output_path: Output file path.
        include_traces: Include trace data.
    """
    import json

    data = session._to_dict(include_traces=include_traces)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _export_html(session: Any, output_path: Path) -> None:
    """Export session to HTML report.

    Args:
        session: Session object.
        output_path: Output file path.
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Oscura Analysis Report - {session.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #4CAF50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metadata {{ background: #f9f9f9; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Oscura Analysis Report</h1>
    <div class="metadata">
        <h2>Session: {session.name}</h2>
        <p>Created: {session.created_at}</p>
        <p>Modified: {session.modified_at}</p>
    </div>

    <h2>Traces ({len(session.traces)})</h2>
    <ul>
"""

    for trace_name in session.list_traces():
        html_content += f"        <li>{trace_name}</li>\n"

    html_content += """    </ul>

    <h2>Measurements</h2>
    <table>
        <tr><th>Measurement</th><th>Value</th><th>Unit</th></tr>
"""

    for name, meas in session.get_measurements().items():
        value = meas.get("value", "N/A")
        unit = meas.get("unit", "")
        html_content += f"        <tr><td>{name}</td><td>{value}</td><td>{unit}</td></tr>\n"

    html_content += """    </table>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html_content)


def _export_csv(session: Any, output_path: Path) -> None:
    """Export measurements to CSV.

    Args:
        session: Session object.
        output_path: Output file path.
    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Measurement", "Value", "Unit", "Trace"])

        for name, meas in session.get_measurements().items():
            writer.writerow(
                [
                    name,
                    meas.get("value", ""),
                    meas.get("unit", ""),
                    meas.get("trace", ""),
                ]
            )


def _export_matlab(session: Any, output_path: Path) -> None:
    """Export to MATLAB format.

    Args:
        session: Session object.
        output_path: Output file path.
    """
    # MATLAB export has been redesigned - use new AnalysisSession API
    raise NotImplementedError("MATLAB export needs reimplementation with new API")


def _export_wireshark(session: Any, output_path: Path) -> None:
    """Export to Wireshark dissector.

    Args:
        session: Session object.
        output_path: Output file path.
    """
    # Placeholder - requires protocol specification
    lua_template = """-- Wireshark Dissector for Unknown Protocol
-- Generated by Oscura

local proto = Proto("unknown", "Unknown Protocol")

function proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = "UNKNOWN"
    local subtree = tree:add(proto, buffer(), "Unknown Protocol Data")
    subtree:add(buffer(0), "Raw data: " .. buffer():bytes():tohex())
end

-- Register dissector
local udp_table = DissectorTable.get("udp.port")
-- udp_table:add(YOUR_PORT, proto)
"""

    with open(output_path, "w") as f:
        f.write(lua_template)


def _export_scapy(session: Any, output_path: Path) -> None:
    """Export to Scapy layer.

    Args:
        session: Session object.
        output_path: Output file path.
    """
    # Placeholder - requires protocol specification
    scapy_template = """# Scapy Layer for Unknown Protocol
# Generated by Oscura

from scapy.all import *

class UnknownProtocol(Packet):
    name = "UnknownProtocol"
    fields_desc = [
        # Define fields here
        XByteField("data", 0x00),
    ]

# Bind to lower layer if needed
# bind_layers(UDP, UnknownProtocol, dport=YOUR_PORT)
"""

    with open(output_path, "w") as f:
        f.write(scapy_template)


def _export_kaitai(session: Any, output_path: Path) -> None:
    """Export to Kaitai struct.

    Args:
        session: Session object.
        output_path: Output file path.
    """
    # Placeholder - requires protocol specification
    kaitai_template = """meta:
  id: unknown_protocol
  title: Unknown Protocol
  file-extension: dat
  endian: le

seq:
  - id: header
    type: u1
  - id: data
    size-eos: true
"""

    with open(output_path, "w") as f:
        f.write(kaitai_template)
