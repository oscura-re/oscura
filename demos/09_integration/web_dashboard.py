"""Web Dashboard: Web-based interactive analysis interface.

Demonstrates:
- Web dashboard architecture patterns
- REST API for analysis operations
- WebSocket for real-time updates
- Interactive visualization in browser
- Session management and security

Category: Integration
IEEE Standards: N/A

Related Demos:
- 02_basic_analysis/01_measurements.py
- 10_export_visualization/03_plotting.py

This demonstrates patterns for building web-based interfaces for Oscura,
including REST APIs, real-time updates, and interactive visualizations.

Note: This is a conceptual demo showing patterns. Full web implementation
requires FastAPI/Flask and frontend framework (React, Vue, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class WebDashboardDemo(BaseDemo):
    """Demonstrates web dashboard integration patterns."""

    name = "Web Dashboard"
    description = "Web-based interactive analysis interface"
    category = "integration"

    def generate_data(self) -> None:
        """Generate sample data for dashboard demo."""
        from oscura.core import TraceMetadata, WaveformTrace

        t = np.linspace(0, 0.01, 1000)
        data = np.sin(2 * np.pi * 1000 * t)

        self.trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=100e3,
                channel_name="CH1",
            ),
        )

    def run_analysis(self) -> None:
        """Demonstrate web dashboard patterns."""
        print_header("Web Dashboard Patterns")

        print_subheader("1. REST API Design")
        print_info("Example FastAPI application for Oscura:")

        fastapi_example = '''
from fastapi import FastAPI, UploadFile, File
from oscura import frequency, amplitude
import numpy as np

app = FastAPI(title="Oscura API")

@app.post("/api/analyze")
async def analyze_trace(file: UploadFile = File(...)):
    """Analyze uploaded trace file."""
    # Load trace from file
    data = await file.read()
    # ... parse and create WaveformTrace ...

    # Perform analysis
    freq = frequency(trace)
    amp = amplitude(trace)

    return {
        "frequency": freq,
        "amplitude": amp,
        "status": "success"
    }

@app.get("/api/measurements/{trace_id}")
async def get_measurements(trace_id: str):
    """Get measurements for trace."""
    # ... load trace by ID ...
    return {"measurements": {...}}
'''

        print(fastapi_example)

        print_subheader("2. WebSocket for Real-Time Updates")
        print_info("WebSocket endpoint for streaming analysis:")

        websocket_example = '''
from fastapi import WebSocket
import asyncio

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Stream real-time analysis updates."""
    await websocket.accept()

    try:
        while True:
            # Acquire new data
            trace = await acquire_from_hardware()

            # Analyze
            measurements = {
                "timestamp": time.time(),
                "frequency": frequency(trace),
                "amplitude": amplitude(trace)
            }

            # Send to client
            await websocket.send_json(measurements)
            await asyncio.sleep(0.1)  # 10 Hz update rate

    except Exception as e:
        await websocket.close()
'''

        print(websocket_example)

        print_subheader("3. Frontend Integration")
        print_info("React component example for dashboard:")

        react_example = '''
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis } from 'recharts';

function OscuraDashboard() {
  const [measurements, setMeasurements] = useState({});
  const [waveformData, setWaveformData] = useState([]);

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket('ws://localhost:8000/ws/stream');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMeasurements(data);
    };

    return () => ws.close();
  }, []);

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();
    setMeasurements(result);
  };

  return (
    <div>
      <h1>Oscura Dashboard</h1>
      <div>Frequency: {measurements.frequency} Hz</div>
      <LineChart data={waveformData}>
        <Line type="monotone" dataKey="value" stroke="#8884d8" />
        <XAxis dataKey="time" />
        <YAxis />
      </LineChart>
    </div>
  );
}
'''

        print(react_example)

        print_subheader("4. API Response Format")
        print_info("Standardized JSON response format:")

        from oscura import amplitude, frequency

        freq = frequency(self.trace)
        amp = amplitude(self.trace)

        api_response = {
            "status": "success",
            "data": {
                "trace_id": "trace_001",
                "measurements": {"frequency": freq, "amplitude": amp, "rms": amp / np.sqrt(2)},
                "metadata": {
                    "sample_rate": self.trace.metadata.sample_rate,
                    "channel": self.trace.metadata.channel_name,
                    "duration": len(self.trace.data) / self.trace.metadata.sample_rate,
                },
            },
            "timestamp": "2024-01-29T12:00:00Z",
        }

        print(json.dumps(api_response, indent=2))

        # Save example response
        response_path = self.data_dir / "api_response.json"
        response_path.write_text(json.dumps(api_response, indent=2))
        print_info(f"✓ Example response saved: {response_path}")

        self.results["response_path"] = str(response_path)

        print_subheader("5. Session Management")
        print_info("User session and authentication pattern:")

        session_example = '''
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/secure/data")
async def secure_endpoint(user = Depends(verify_token)):
    """Protected endpoint requiring authentication."""
    return {"data": "sensitive information", "user": user["username"]}
'''

        print(session_example)

        print_subheader("6. Configuration Example")
        print_info("Dashboard configuration file:")

        config = {
            "server": {
                "host": "127.0.0.1",
                "port": 8000,
                "workers": 4,
                "reload": False,
            },
            "dashboard": {
                "title": "Oscura Analysis Dashboard",
                "theme": "dark",
                "max_file_size_mb": 100,
                "session_timeout_seconds": 3600,
            },
            "websocket": {"enabled": True, "update_rate_hz": 10, "max_connections": 100},
        }

        config_path = self.data_dir / "dashboard_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        print(json.dumps(config, indent=2))
        print_info(f"✓ Config saved: {config_path}")

        self.results["config_path"] = str(config_path)

        print_subheader("7. Running the Dashboard")
        print_info("To run a full dashboard (requires FastAPI):")
        print_info("  1. Install: pip install fastapi uvicorn")
        print_info("  2. Create: dashboard.py with FastAPI app")
        print_info("  3. Run: uvicorn dashboard:app --reload")
        print_info("  4. Visit: http://127.0.0.1:8000")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate web dashboard results."""
        suite.check_exists("API response path", self.results.get("response_path"))
        suite.check_exists("Config path", self.results.get("config_path"))


if __name__ == "__main__":
    demo = WebDashboardDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
