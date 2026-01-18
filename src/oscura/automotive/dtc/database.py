"""Diagnostic Trouble Code (DTC) database for automotive diagnostics.

This module provides a comprehensive database of standardized DTCs following
SAE J2012 and ISO 14229 standards. DTCs are used to identify vehicle system
faults and malfunctions.

DTC Format:
    - First character: Category (P/C/B/U)
        - P: Powertrain (engine, transmission, emissions)
        - C: Chassis (brakes, steering, suspension)
        - B: Body (lighting, HVAC, security)
        - U: Network/Communication (CAN, LIN, FlexRay)
    - Second character: Code type
        - 0: Generic (SAE defined)
        - 1-3: Manufacturer specific
    - Remaining 3 digits: Specific fault code

Example:
    >>> from oscura.automotive.dtc import DTCDatabase
    >>> # Look up a specific code
    >>> info = DTCDatabase.lookup("P0420")
    >>> print(f"{info.code}: {info.description}")
    P0420: Catalyst System Efficiency Below Threshold (Bank 1)
    >>> # Search for related codes
    >>> results = DTCDatabase.search("oxygen sensor")
    >>> print(f"Found {len(results)} oxygen sensor codes")
    >>> # Get all powertrain codes
    >>> powertrain = DTCDatabase.get_by_category("Powertrain")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DTCInfo:
    """Information about a Diagnostic Trouble Code.

    Attributes:
        code: DTC code (e.g., "P0420", "C0035", "B1234", "U0100")
        description: Human-readable description of the fault
        category: DTC category ("Powertrain", "Chassis", "Body", "Network")
        severity: Fault severity ("Critical", "High", "Medium", "Low")
        system: Specific system affected (e.g., "Emissions Control", "ABS")
        possible_causes: List of common causes for this DTC
    """

    code: str
    description: str
    category: str
    severity: str
    system: str
    possible_causes: list[str]


# Comprehensive DTC Database (200+ codes)
DTCS: dict[str, DTCInfo] = {
    # ============================================================================
    # POWERTRAIN (P) CODES - Generic SAE (P0xxx)
    # ============================================================================
    # P00xx - Fuel and Air Metering
    "P0001": DTCInfo(
        code="P0001",
        description="Fuel Volume Regulator Control Circuit/Open",
        category="Powertrain",
        severity="Medium",
        system="Fuel System",
        possible_causes=[
            "Faulty fuel volume regulator",
            "Wiring harness open or shorted",
            "Poor electrical connection",
            "Faulty fuel pump control module",
        ],
    ),
    "P0002": DTCInfo(
        code="P0002",
        description="Fuel Volume Regulator Control Circuit Range/Performance",
        category="Powertrain",
        severity="Medium",
        system="Fuel System",
        possible_causes=[
            "Fuel volume regulator stuck",
            "Fuel pressure sensor malfunction",
            "Fuel pump performance issue",
            "Electrical circuit resistance",
        ],
    ),
    "P0010": DTCInfo(
        code="P0010",
        description="Intake Camshaft Position Actuator Circuit/Open (Bank 1)",
        category="Powertrain",
        severity="Medium",
        system="Variable Valve Timing",
        possible_causes=[
            "Faulty VVT actuator solenoid",
            "Engine oil level low or dirty",
            "Wiring harness problem",
            "ECM fault",
        ],
    ),
    "P0011": DTCInfo(
        code="P0011",
        description="Intake Camshaft Position Timing Over-Advanced (Bank 1)",
        category="Powertrain",
        severity="Medium",
        system="Variable Valve Timing",
        possible_causes=[
            "Camshaft position sensor malfunction",
            "Engine oil viscosity too low",
            "VVT actuator stuck in advanced position",
            "Timing chain stretched",
        ],
    ),
    "P0016": DTCInfo(
        code="P0016",
        description="Crankshaft/Camshaft Position Correlation (Bank 1 Sensor A)",
        category="Powertrain",
        severity="High",
        system="Engine Timing",
        possible_causes=[
            "Timing chain/belt jumped or broken",
            "Crankshaft position sensor fault",
            "Camshaft position sensor fault",
            "Engine mechanical problem",
        ],
    ),
    # P01xx - Fuel and Air Metering
    "P0101": DTCInfo(
        code="P0101",
        description="Mass Air Flow (MAF) Circuit Range/Performance",
        category="Powertrain",
        severity="Medium",
        system="Air Metering",
        possible_causes=[
            "Dirty or contaminated MAF sensor",
            "Air leak after MAF sensor",
            "Faulty MAF sensor",
            "Exhaust leak",
        ],
    ),
    "P0102": DTCInfo(
        code="P0102",
        description="Mass Air Flow (MAF) Circuit Low Input",
        category="Powertrain",
        severity="Medium",
        system="Air Metering",
        possible_causes=[
            "Faulty MAF sensor",
            "MAF sensor circuit shorted to ground",
            "Intake air leak",
            "Poor electrical connection",
        ],
    ),
    "P0103": DTCInfo(
        code="P0103",
        description="Mass Air Flow (MAF) Circuit High Input",
        category="Powertrain",
        severity="Medium",
        system="Air Metering",
        possible_causes=[
            "Faulty MAF sensor",
            "MAF sensor circuit shorted to voltage",
            "Wiring harness fault",
            "ECM problem",
        ],
    ),
    "P0106": DTCInfo(
        code="P0106",
        description="Manifold Absolute Pressure (MAP) Circuit Range/Performance",
        category="Powertrain",
        severity="Medium",
        system="Air Metering",
        possible_causes=[
            "Faulty MAP sensor",
            "Vacuum leak",
            "MAP sensor hose disconnected",
            "Intake manifold gasket leak",
        ],
    ),
    "P0107": DTCInfo(
        code="P0107",
        description="Manifold Absolute Pressure (MAP) Circuit Low Input",
        category="Powertrain",
        severity="Medium",
        system="Air Metering",
        possible_causes=[
            "MAP sensor circuit shorted to ground",
            "Faulty MAP sensor",
            "Wiring harness fault",
            "Poor electrical connection",
        ],
    ),
    "P0108": DTCInfo(
        code="P0108",
        description="Manifold Absolute Pressure (MAP) Circuit High Input",
        category="Powertrain",
        severity="Medium",
        system="Air Metering",
        possible_causes=[
            "MAP sensor circuit shorted to voltage",
            "Faulty MAP sensor",
            "Blocked MAP sensor hose",
            "Wiring harness fault",
        ],
    ),
    "P0110": DTCInfo(
        code="P0110",
        description="Intake Air Temperature (IAT) Circuit Malfunction",
        category="Powertrain",
        severity="Low",
        system="Air Metering",
        possible_causes=[
            "Faulty IAT sensor",
            "IAT sensor circuit open or short",
            "Poor electrical connection",
            "ECM fault",
        ],
    ),
    "P0115": DTCInfo(
        code="P0115",
        description="Engine Coolant Temperature (ECT) Circuit Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Engine Temperature",
        possible_causes=[
            "Faulty coolant temperature sensor",
            "Wiring harness open or shorted",
            "Coolant level low",
            "Thermostat stuck open",
        ],
    ),
    "P0116": DTCInfo(
        code="P0116",
        description="Engine Coolant Temperature Circuit Range/Performance",
        category="Powertrain",
        severity="Medium",
        system="Engine Temperature",
        possible_causes=[
            "Coolant temperature sensor drift",
            "Thermostat malfunction",
            "Coolant flow restriction",
            "ECM problem",
        ],
    ),
    "P0117": DTCInfo(
        code="P0117",
        description="Engine Coolant Temperature Circuit Low Input",
        category="Powertrain",
        severity="Medium",
        system="Engine Temperature",
        possible_causes=[
            "Coolant sensor circuit shorted to ground",
            "Faulty coolant temperature sensor",
            "Wiring harness fault",
            "Poor electrical connection",
        ],
    ),
    "P0118": DTCInfo(
        code="P0118",
        description="Engine Coolant Temperature Circuit High Input",
        category="Powertrain",
        severity="Medium",
        system="Engine Temperature",
        possible_causes=[
            "Coolant sensor circuit open or shorted to voltage",
            "Faulty coolant temperature sensor",
            "Wiring harness fault",
            "Engine overheating",
        ],
    ),
    "P0120": DTCInfo(
        code="P0120",
        description="Throttle Position Sensor/Switch Circuit Malfunction",
        category="Powertrain",
        severity="High",
        system="Throttle Control",
        possible_causes=[
            "Faulty throttle position sensor",
            "Wiring harness problem",
            "Throttle body malfunction",
            "ECM fault",
        ],
    ),
    "P0121": DTCInfo(
        code="P0121",
        description="Throttle Position Sensor Circuit Range/Performance",
        category="Powertrain",
        severity="High",
        system="Throttle Control",
        possible_causes=[
            "TPS sensor out of range",
            "Throttle body dirty or stuck",
            "Accelerator pedal position sensor fault",
            "Wiring problem",
        ],
    ),
    "P0122": DTCInfo(
        code="P0122",
        description="Throttle Position Sensor Circuit Low Input",
        category="Powertrain",
        severity="High",
        system="Throttle Control",
        possible_causes=[
            "TPS circuit shorted to ground",
            "Faulty TPS sensor",
            "Wiring harness fault",
            "Poor electrical connection",
        ],
    ),
    "P0123": DTCInfo(
        code="P0123",
        description="Throttle Position Sensor Circuit High Input",
        category="Powertrain",
        severity="High",
        system="Throttle Control",
        possible_causes=[
            "TPS circuit shorted to voltage",
            "Faulty TPS sensor",
            "Wiring harness open",
            "ECM problem",
        ],
    ),
    "P0125": DTCInfo(
        code="P0125",
        description="Insufficient Coolant Temperature for Closed Loop Fuel Control",
        category="Powertrain",
        severity="Low",
        system="Engine Temperature",
        possible_causes=[
            "Thermostat stuck open",
            "Coolant temperature sensor fault",
            "Low coolant level",
            "Engine not reaching operating temperature",
        ],
    ),
    "P0128": DTCInfo(
        code="P0128",
        description="Coolant Thermostat (Coolant Temperature Below Thermostat Regulating Temperature)",
        category="Powertrain",
        severity="Low",
        system="Engine Temperature",
        possible_causes=[
            "Thermostat stuck open",
            "Coolant temperature sensor fault",
            "Low ambient temperature operation",
            "Cooling fan stuck on",
        ],
    ),
    "P0130": DTCInfo(
        code="P0130",
        description="O2 Sensor Circuit Malfunction (Bank 1 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Faulty oxygen sensor",
            "Oxygen sensor heater circuit fault",
            "Wiring harness problem",
            "Exhaust leak near sensor",
        ],
    ),
    "P0131": DTCInfo(
        code="P0131",
        description="O2 Sensor Circuit Low Voltage (Bank 1 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Oxygen sensor contaminated",
            "Exhaust leak near sensor",
            "Oxygen sensor circuit shorted to ground",
            "Faulty oxygen sensor",
        ],
    ),
    "P0132": DTCInfo(
        code="P0132",
        description="O2 Sensor Circuit High Voltage (Bank 1 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Oxygen sensor circuit shorted to voltage",
            "Fuel pressure too high",
            "Faulty oxygen sensor",
            "Wiring harness fault",
        ],
    ),
    "P0133": DTCInfo(
        code="P0133",
        description="O2 Sensor Circuit Slow Response (Bank 1 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Oxygen sensor aged or contaminated",
            "Exhaust leak",
            "Engine vacuum leak",
            "Faulty MAF sensor",
        ],
    ),
    "P0134": DTCInfo(
        code="P0134",
        description="O2 Sensor Circuit No Activity Detected (Bank 1 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Faulty oxygen sensor",
            "Oxygen sensor heater circuit fault",
            "Wiring harness open circuit",
            "Poor electrical connection",
        ],
    ),
    "P0135": DTCInfo(
        code="P0135",
        description="O2 Sensor Heater Circuit Malfunction (Bank 1 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Faulty oxygen sensor heater",
            "Heater circuit fuse blown",
            "Wiring harness open or shorted",
            "ECM fault",
        ],
    ),
    "P0136": DTCInfo(
        code="P0136",
        description="O2 Sensor Circuit Malfunction (Bank 1 Sensor 2)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Faulty oxygen sensor",
            "Oxygen sensor heater circuit fault",
            "Wiring harness problem",
            "Exhaust leak near sensor",
        ],
    ),
    "P0137": DTCInfo(
        code="P0137",
        description="O2 Sensor Circuit Low Voltage (Bank 1 Sensor 2)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Oxygen sensor circuit shorted to ground",
            "Faulty catalytic converter",
            "Exhaust leak",
            "Faulty oxygen sensor",
        ],
    ),
    "P0138": DTCInfo(
        code="P0138",
        description="O2 Sensor Circuit High Voltage (Bank 1 Sensor 2)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Oxygen sensor circuit shorted to voltage",
            "Faulty catalytic converter",
            "Fuel pressure too high",
            "Faulty oxygen sensor",
        ],
    ),
    "P0140": DTCInfo(
        code="P0140",
        description="O2 Sensor Circuit No Activity Detected (Bank 1 Sensor 2)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Faulty oxygen sensor",
            "Oxygen sensor heater circuit fault",
            "Wiring harness open circuit",
            "Poor electrical connection",
        ],
    ),
    "P0141": DTCInfo(
        code="P0141",
        description="O2 Sensor Heater Circuit Malfunction (Bank 1 Sensor 2)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Faulty oxygen sensor heater",
            "Heater circuit fuse blown",
            "Wiring harness open or shorted",
            "ECM fault",
        ],
    ),
    "P0150": DTCInfo(
        code="P0150",
        description="O2 Sensor Circuit Malfunction (Bank 2 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Faulty oxygen sensor",
            "Oxygen sensor heater circuit fault",
            "Wiring harness problem",
            "Exhaust leak near sensor",
        ],
    ),
    "P0151": DTCInfo(
        code="P0151",
        description="O2 Sensor Circuit Low Voltage (Bank 2 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Oxygen sensor contaminated",
            "Exhaust leak near sensor",
            "Oxygen sensor circuit shorted to ground",
            "Faulty oxygen sensor",
        ],
    ),
    "P0152": DTCInfo(
        code="P0152",
        description="O2 Sensor Circuit High Voltage (Bank 2 Sensor 1)",
        category="Powertrain",
        severity="Medium",
        system="Oxygen Sensors",
        possible_causes=[
            "Oxygen sensor circuit shorted to voltage",
            "Fuel pressure too high",
            "Faulty oxygen sensor",
            "Wiring harness fault",
        ],
    ),
    "P0171": DTCInfo(
        code="P0171",
        description="System Too Lean (Bank 1)",
        category="Powertrain",
        severity="Medium",
        system="Fuel Trim",
        possible_causes=[
            "Vacuum leak",
            "MAF sensor dirty or faulty",
            "Fuel pressure low",
            "Injector clogged",
            "Oxygen sensor fault",
        ],
    ),
    "P0172": DTCInfo(
        code="P0172",
        description="System Too Rich (Bank 1)",
        category="Powertrain",
        severity="Medium",
        system="Fuel Trim",
        possible_causes=[
            "Fuel pressure too high",
            "Leaking fuel injector",
            "Dirty air filter",
            "MAF sensor fault",
            "Oxygen sensor fault",
        ],
    ),
    "P0174": DTCInfo(
        code="P0174",
        description="System Too Lean (Bank 2)",
        category="Powertrain",
        severity="Medium",
        system="Fuel Trim",
        possible_causes=[
            "Vacuum leak",
            "MAF sensor dirty or faulty",
            "Fuel pressure low",
            "Injector clogged",
            "Oxygen sensor fault",
        ],
    ),
    "P0175": DTCInfo(
        code="P0175",
        description="System Too Rich (Bank 2)",
        category="Powertrain",
        severity="Medium",
        system="Fuel Trim",
        possible_causes=[
            "Fuel pressure too high",
            "Leaking fuel injector",
            "Dirty air filter",
            "MAF sensor fault",
            "Oxygen sensor fault",
        ],
    ),
    # P02xx - Fuel Injection
    "P0200": DTCInfo(
        code="P0200",
        description="Injector Circuit Malfunction",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty fuel injector",
            "Injector circuit open or shorted",
            "ECM fault",
            "Wiring harness problem",
        ],
    ),
    "P0201": DTCInfo(
        code="P0201",
        description="Injector Circuit Malfunction - Cylinder 1",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 1 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    "P0202": DTCInfo(
        code="P0202",
        description="Injector Circuit Malfunction - Cylinder 2",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 2 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    "P0203": DTCInfo(
        code="P0203",
        description="Injector Circuit Malfunction - Cylinder 3",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 3 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    "P0204": DTCInfo(
        code="P0204",
        description="Injector Circuit Malfunction - Cylinder 4",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 4 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    "P0205": DTCInfo(
        code="P0205",
        description="Injector Circuit Malfunction - Cylinder 5",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 5 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    "P0206": DTCInfo(
        code="P0206",
        description="Injector Circuit Malfunction - Cylinder 6",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 6 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    "P0207": DTCInfo(
        code="P0207",
        description="Injector Circuit Malfunction - Cylinder 7",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 7 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    "P0208": DTCInfo(
        code="P0208",
        description="Injector Circuit Malfunction - Cylinder 8",
        category="Powertrain",
        severity="High",
        system="Fuel Injection",
        possible_causes=[
            "Faulty cylinder 8 injector",
            "Injector circuit open or shorted",
            "Poor electrical connection",
            "ECM driver circuit fault",
        ],
    ),
    # P03xx - Ignition System / Misfire
    "P0300": DTCInfo(
        code="P0300",
        description="Random/Multiple Cylinder Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Worn spark plugs",
            "Faulty ignition coils",
            "Vacuum leak",
            "Low fuel pressure",
            "Dirty fuel injectors",
            "Engine mechanical problem",
        ],
    ),
    "P0301": DTCInfo(
        code="P0301",
        description="Cylinder 1 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 1)",
            "Faulty ignition coil (cylinder 1)",
            "Fuel injector problem (cylinder 1)",
            "Low compression (cylinder 1)",
            "Vacuum leak at cylinder 1",
        ],
    ),
    "P0302": DTCInfo(
        code="P0302",
        description="Cylinder 2 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 2)",
            "Faulty ignition coil (cylinder 2)",
            "Fuel injector problem (cylinder 2)",
            "Low compression (cylinder 2)",
            "Vacuum leak at cylinder 2",
        ],
    ),
    "P0303": DTCInfo(
        code="P0303",
        description="Cylinder 3 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 3)",
            "Faulty ignition coil (cylinder 3)",
            "Fuel injector problem (cylinder 3)",
            "Low compression (cylinder 3)",
            "Vacuum leak at cylinder 3",
        ],
    ),
    "P0304": DTCInfo(
        code="P0304",
        description="Cylinder 4 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 4)",
            "Faulty ignition coil (cylinder 4)",
            "Fuel injector problem (cylinder 4)",
            "Low compression (cylinder 4)",
            "Vacuum leak at cylinder 4",
        ],
    ),
    "P0305": DTCInfo(
        code="P0305",
        description="Cylinder 5 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 5)",
            "Faulty ignition coil (cylinder 5)",
            "Fuel injector problem (cylinder 5)",
            "Low compression (cylinder 5)",
            "Vacuum leak at cylinder 5",
        ],
    ),
    "P0306": DTCInfo(
        code="P0306",
        description="Cylinder 6 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 6)",
            "Faulty ignition coil (cylinder 6)",
            "Fuel injector problem (cylinder 6)",
            "Low compression (cylinder 6)",
            "Vacuum leak at cylinder 6",
        ],
    ),
    "P0307": DTCInfo(
        code="P0307",
        description="Cylinder 7 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 7)",
            "Faulty ignition coil (cylinder 7)",
            "Fuel injector problem (cylinder 7)",
            "Low compression (cylinder 7)",
            "Vacuum leak at cylinder 7",
        ],
    ),
    "P0308": DTCInfo(
        code="P0308",
        description="Cylinder 8 Misfire Detected",
        category="Powertrain",
        severity="High",
        system="Ignition System",
        possible_causes=[
            "Faulty spark plug (cylinder 8)",
            "Faulty ignition coil (cylinder 8)",
            "Fuel injector problem (cylinder 8)",
            "Low compression (cylinder 8)",
            "Vacuum leak at cylinder 8",
        ],
    ),
    "P0320": DTCInfo(
        code="P0320",
        description="Ignition/Distributor Engine Speed Input Circuit Malfunction",
        category="Powertrain",
        severity="Critical",
        system="Ignition System",
        possible_causes=[
            "Faulty crankshaft position sensor",
            "Wiring harness problem",
            "Poor electrical connection",
            "ECM fault",
        ],
    ),
    "P0325": DTCInfo(
        code="P0325",
        description="Knock Sensor 1 Circuit Malfunction (Bank 1)",
        category="Powertrain",
        severity="Medium",
        system="Ignition System",
        possible_causes=[
            "Faulty knock sensor",
            "Wiring harness open or shorted",
            "Poor electrical connection",
            "ECM fault",
        ],
    ),
    "P0335": DTCInfo(
        code="P0335",
        description="Crankshaft Position Sensor Circuit Malfunction",
        category="Powertrain",
        severity="Critical",
        system="Engine Position Sensors",
        possible_causes=[
            "Faulty crankshaft position sensor",
            "Sensor reluctor wheel damaged",
            "Wiring harness problem",
            "Poor electrical connection",
        ],
    ),
    "P0336": DTCInfo(
        code="P0336",
        description="Crankshaft Position Sensor Range/Performance",
        category="Powertrain",
        severity="Critical",
        system="Engine Position Sensors",
        possible_causes=[
            "Crankshaft position sensor misaligned",
            "Reluctor wheel damaged or dirty",
            "Sensor air gap too large",
            "Timing belt/chain problem",
        ],
    ),
    "P0340": DTCInfo(
        code="P0340",
        description="Camshaft Position Sensor Circuit Malfunction (Bank 1)",
        category="Powertrain",
        severity="High",
        system="Engine Position Sensors",
        possible_causes=[
            "Faulty camshaft position sensor",
            "Wiring harness problem",
            "Poor electrical connection",
            "Timing belt/chain problem",
        ],
    ),
    "P0341": DTCInfo(
        code="P0341",
        description="Camshaft Position Sensor Range/Performance (Bank 1)",
        category="Powertrain",
        severity="High",
        system="Engine Position Sensors",
        possible_causes=[
            "Camshaft position sensor misaligned",
            "Timing belt/chain jumped",
            "Faulty camshaft position sensor",
            "Reluctor wheel damaged",
        ],
    ),
    # P04xx - Emissions Control
    "P0400": DTCInfo(
        code="P0400",
        description="Exhaust Gas Recirculation (EGR) Flow Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Emissions Control",
        possible_causes=[
            "EGR valve stuck closed",
            "EGR passages clogged",
            "Faulty EGR valve",
            "Vacuum leak",
        ],
    ),
    "P0401": DTCInfo(
        code="P0401",
        description="Exhaust Gas Recirculation (EGR) Flow Insufficient",
        category="Powertrain",
        severity="Medium",
        system="Emissions Control",
        possible_causes=[
            "EGR valve stuck partially closed",
            "EGR passages restricted with carbon",
            "Faulty EGR valve position sensor",
            "Vacuum supply issue",
        ],
    ),
    "P0402": DTCInfo(
        code="P0402",
        description="Exhaust Gas Recirculation (EGR) Flow Excessive",
        category="Powertrain",
        severity="Medium",
        system="Emissions Control",
        possible_causes=[
            "EGR valve stuck open",
            "Faulty EGR valve",
            "EGR vacuum solenoid fault",
            "ECM problem",
        ],
    ),
    "P0403": DTCInfo(
        code="P0403",
        description="Exhaust Gas Recirculation (EGR) Control Circuit Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Emissions Control",
        possible_causes=[
            "Faulty EGR solenoid",
            "Wiring harness open or shorted",
            "Poor electrical connection",
            "ECM fault",
        ],
    ),
    "P0420": DTCInfo(
        code="P0420",
        description="Catalyst System Efficiency Below Threshold (Bank 1)",
        category="Powertrain",
        severity="Medium",
        system="Emissions Control",
        possible_causes=[
            "Faulty catalytic converter",
            "Oxygen sensor degraded",
            "Exhaust leak",
            "Engine misfire",
            "Fuel system problem",
        ],
    ),
    "P0421": DTCInfo(
        code="P0421",
        description="Warm Up Catalyst Efficiency Below Threshold (Bank 1)",
        category="Powertrain",
        severity="Medium",
        system="Emissions Control",
        possible_causes=[
            "Faulty warm-up catalytic converter",
            "Oxygen sensor fault",
            "Exhaust leak",
            "Engine running too rich or lean",
        ],
    ),
    "P0430": DTCInfo(
        code="P0430",
        description="Catalyst System Efficiency Below Threshold (Bank 2)",
        category="Powertrain",
        severity="Medium",
        system="Emissions Control",
        possible_causes=[
            "Faulty catalytic converter (bank 2)",
            "Oxygen sensor degraded",
            "Exhaust leak",
            "Engine misfire",
            "Fuel system problem",
        ],
    ),
    "P0440": DTCInfo(
        code="P0440",
        description="Evaporative Emission Control System Malfunction",
        category="Powertrain",
        severity="Low",
        system="Emissions Control",
        possible_causes=[
            "Loose or missing fuel cap",
            "EVAP system leak",
            "Faulty purge valve",
            "Faulty vent valve",
        ],
    ),
    "P0441": DTCInfo(
        code="P0441",
        description="Evaporative Emission Control System Incorrect Purge Flow",
        category="Powertrain",
        severity="Low",
        system="Emissions Control",
        possible_causes=[
            "Faulty purge valve",
            "Purge valve stuck open or closed",
            "EVAP system leak",
            "Vacuum line disconnected",
        ],
    ),
    "P0442": DTCInfo(
        code="P0442",
        description="Evaporative Emission Control System Leak Detected (Small Leak)",
        category="Powertrain",
        severity="Low",
        system="Emissions Control",
        possible_causes=[
            "Loose fuel cap",
            "EVAP hose cracked or disconnected",
            "Faulty fuel tank pressure sensor",
            "Small leak in EVAP system",
        ],
    ),
    "P0443": DTCInfo(
        code="P0443",
        description="Evaporative Emission Control System Purge Control Valve Circuit Malfunction",
        category="Powertrain",
        severity="Low",
        system="Emissions Control",
        possible_causes=[
            "Faulty purge solenoid valve",
            "Wiring harness open or shorted",
            "Poor electrical connection",
            "ECM fault",
        ],
    ),
    "P0446": DTCInfo(
        code="P0446",
        description="Evaporative Emission Control System Vent Control Circuit Malfunction",
        category="Powertrain",
        severity="Low",
        system="Emissions Control",
        possible_causes=[
            "Faulty vent control valve",
            "Vent valve stuck closed",
            "Wiring harness problem",
            "Charcoal canister clogged",
        ],
    ),
    "P0455": DTCInfo(
        code="P0455",
        description="Evaporative Emission Control System Leak Detected (Large Leak)",
        category="Powertrain",
        severity="Low",
        system="Emissions Control",
        possible_causes=[
            "Missing or loose fuel cap",
            "EVAP hose disconnected",
            "Fuel tank crack",
            "Large leak in EVAP system",
        ],
    ),
    "P0456": DTCInfo(
        code="P0456",
        description="Evaporative Emission Control System Leak Detected (Very Small Leak)",
        category="Powertrain",
        severity="Low",
        system="Emissions Control",
        possible_causes=[
            "Loose fuel cap",
            "Fuel cap seal damaged",
            "Very small EVAP system leak",
            "Faulty fuel tank pressure sensor",
        ],
    ),
    # P05xx - Speed and Idle Control
    "P0500": DTCInfo(
        code="P0500",
        description="Vehicle Speed Sensor Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Speed Sensors",
        possible_causes=[
            "Faulty vehicle speed sensor",
            "Wiring harness problem",
            "Poor electrical connection",
            "ECM fault",
        ],
    ),
    "P0505": DTCInfo(
        code="P0505",
        description="Idle Control System Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Idle Control",
        possible_causes=[
            "Faulty idle air control valve",
            "IAC valve stuck or carbon buildup",
            "Vacuum leak",
            "Throttle body dirty",
        ],
    ),
    "P0506": DTCInfo(
        code="P0506",
        description="Idle Control System RPM Lower Than Expected",
        category="Powertrain",
        severity="Low",
        system="Idle Control",
        possible_causes=[
            "Vacuum leak",
            "IAC valve fault",
            "Dirty throttle body",
            "PCV valve problem",
        ],
    ),
    "P0507": DTCInfo(
        code="P0507",
        description="Idle Control System RPM Higher Than Expected",
        category="Powertrain",
        severity="Low",
        system="Idle Control",
        possible_causes=[
            "Vacuum leak",
            "IAC valve stuck open",
            "PCV valve stuck open",
            "EVAP purge valve leaking",
        ],
    ),
    # P06xx - Computer/ECM
    "P0600": DTCInfo(
        code="P0600",
        description="Serial Communication Link Malfunction",
        category="Powertrain",
        severity="High",
        system="Engine Control Module",
        possible_causes=[
            "ECM internal fault",
            "CAN bus communication problem",
            "Wiring harness fault",
            "Ground connection problem",
        ],
    ),
    "P0601": DTCInfo(
        code="P0601",
        description="Internal Control Module Memory Check Sum Error",
        category="Powertrain",
        severity="Critical",
        system="Engine Control Module",
        possible_causes=[
            "ECM internal memory corruption",
            "ECM needs reprogramming",
            "ECM hardware failure",
            "Power supply problem",
        ],
    ),
    "P0602": DTCInfo(
        code="P0602",
        description="Control Module Programming Error",
        category="Powertrain",
        severity="Critical",
        system="Engine Control Module",
        possible_causes=[
            "ECM not programmed",
            "ECM programming incomplete",
            "Wrong software version",
            "ECM fault",
        ],
    ),
    "P0603": DTCInfo(
        code="P0603",
        description="Internal Control Module Keep Alive Memory (KAM) Error",
        category="Powertrain",
        severity="High",
        system="Engine Control Module",
        possible_causes=[
            "ECM battery voltage lost",
            "ECM internal fault",
            "Poor battery connection",
            "ECM needs replacement",
        ],
    ),
    "P0604": DTCInfo(
        code="P0604",
        description="Internal Control Module Random Access Memory (RAM) Error",
        category="Powertrain",
        severity="Critical",
        system="Engine Control Module",
        possible_causes=[
            "ECM internal RAM failure",
            "ECM needs replacement",
            "Power supply problem",
            "EMI interference",
        ],
    ),
    "P0605": DTCInfo(
        code="P0605",
        description="Internal Control Module Read Only Memory (ROM) Error",
        category="Powertrain",
        severity="Critical",
        system="Engine Control Module",
        possible_causes=[
            "ECM internal ROM failure",
            "ECM needs replacement",
            "ECM programming corruption",
            "Hardware failure",
        ],
    ),
    "P0606": DTCInfo(
        code="P0606",
        description="ECM/PCM Processor Fault",
        category="Powertrain",
        severity="Critical",
        system="Engine Control Module",
        possible_causes=[
            "ECM processor failure",
            "ECM internal fault",
            "Power supply problem",
            "ECM needs replacement",
        ],
    ),
    "P0607": DTCInfo(
        code="P0607",
        description="Control Module Performance",
        category="Powertrain",
        severity="High",
        system="Engine Control Module",
        possible_causes=[
            "ECM performance degraded",
            "ECM internal fault",
            "Software glitch",
            "Electrical interference",
        ],
    ),
    "P0620": DTCInfo(
        code="P0620",
        description="Generator Control Circuit Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Charging System",
        possible_causes=[
            "Faulty alternator",
            "Wiring harness problem",
            "Poor electrical connection",
            "ECM fault",
        ],
    ),
    "P0625": DTCInfo(
        code="P0625",
        description="Generator Field Terminal Circuit Low",
        category="Powertrain",
        severity="Medium",
        system="Charging System",
        possible_causes=[
            "Alternator field circuit shorted to ground",
            "Faulty alternator",
            "Wiring harness fault",
            "ECM driver circuit fault",
        ],
    ),
    "P0630": DTCInfo(
        code="P0630",
        description="VIN Not Programmed or Mismatch - ECM/PCM",
        category="Powertrain",
        severity="Medium",
        system="Engine Control Module",
        possible_causes=[
            "VIN not programmed in ECM",
            "VIN mismatch between modules",
            "ECM replaced without programming",
            "ECM fault",
        ],
    ),
    # P07xx - Transmission
    "P0700": DTCInfo(
        code="P0700",
        description="Transmission Control System Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Transmission fault detected",
            "TCM communication error",
            "Check TCM for additional codes",
            "Wiring harness problem",
        ],
    ),
    "P0705": DTCInfo(
        code="P0705",
        description="Transmission Range Sensor Circuit Malfunction (PRNDL Input)",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Faulty transmission range sensor",
            "Wiring harness problem",
            "Shift linkage misadjusted",
            "TCM fault",
        ],
    ),
    "P0710": DTCInfo(
        code="P0710",
        description="Transmission Fluid Temperature Sensor Circuit Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Transmission",
        possible_causes=[
            "Faulty transmission fluid temperature sensor",
            "Wiring harness open or shorted",
            "Poor electrical connection",
            "TCM fault",
        ],
    ),
    "P0715": DTCInfo(
        code="P0715",
        description="Input/Turbine Speed Sensor Circuit Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Faulty input speed sensor",
            "Wiring harness problem",
            "Sensor reluctor damaged",
            "TCM fault",
        ],
    ),
    "P0720": DTCInfo(
        code="P0720",
        description="Output Speed Sensor Circuit Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Faulty output speed sensor",
            "Wiring harness problem",
            "Sensor reluctor damaged",
            "TCM fault",
        ],
    ),
    "P0725": DTCInfo(
        code="P0725",
        description="Engine Speed Input Circuit Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "CAN communication problem",
            "Faulty crankshaft position sensor",
            "TCM fault",
            "Wiring harness problem",
        ],
    ),
    "P0730": DTCInfo(
        code="P0730",
        description="Incorrect Gear Ratio",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Low transmission fluid",
            "Transmission internal problem",
            "Faulty transmission solenoid",
            "TCM fault",
        ],
    ),
    "P0740": DTCInfo(
        code="P0740",
        description="Torque Converter Clutch Circuit Malfunction",
        category="Powertrain",
        severity="Medium",
        system="Transmission",
        possible_causes=[
            "Faulty torque converter clutch solenoid",
            "Transmission fluid dirty or low",
            "Wiring harness problem",
            "Internal transmission fault",
        ],
    ),
    "P0750": DTCInfo(
        code="P0750",
        description="Shift Solenoid A Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Faulty shift solenoid A",
            "Transmission fluid dirty or low",
            "Wiring harness problem",
            "TCM fault",
        ],
    ),
    "P0755": DTCInfo(
        code="P0755",
        description="Shift Solenoid B Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Faulty shift solenoid B",
            "Transmission fluid dirty or low",
            "Wiring harness problem",
            "TCM fault",
        ],
    ),
    "P0760": DTCInfo(
        code="P0760",
        description="Shift Solenoid C Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Faulty shift solenoid C",
            "Transmission fluid dirty or low",
            "Wiring harness problem",
            "TCM fault",
        ],
    ),
    "P0765": DTCInfo(
        code="P0765",
        description="Shift Solenoid D Malfunction",
        category="Powertrain",
        severity="High",
        system="Transmission",
        possible_causes=[
            "Faulty shift solenoid D",
            "Transmission fluid dirty or low",
            "Wiring harness problem",
            "TCM fault",
        ],
    ),
    # ============================================================================
    # CHASSIS (C) CODES
    # ============================================================================
    "C0035": DTCInfo(
        code="C0035",
        description="Left Front Wheel Speed Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Faulty wheel speed sensor",
            "Damaged sensor reluctor ring",
            "Wiring harness problem",
            "Poor electrical connection",
        ],
    ),
    "C0040": DTCInfo(
        code="C0040",
        description="Right Front Wheel Speed Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Faulty wheel speed sensor",
            "Damaged sensor reluctor ring",
            "Wiring harness problem",
            "Poor electrical connection",
        ],
    ),
    "C0045": DTCInfo(
        code="C0045",
        description="Left Rear Wheel Speed Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Faulty wheel speed sensor",
            "Damaged sensor reluctor ring",
            "Wiring harness problem",
            "Poor electrical connection",
        ],
    ),
    "C0050": DTCInfo(
        code="C0050",
        description="Right Rear Wheel Speed Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Faulty wheel speed sensor",
            "Damaged sensor reluctor ring",
            "Wiring harness problem",
            "Poor electrical connection",
        ],
    ),
    "C0060": DTCInfo(
        code="C0060",
        description="ABS Pump Motor Circuit Malfunction",
        category="Chassis",
        severity="Critical",
        system="ABS",
        possible_causes=[
            "Faulty ABS pump motor",
            "Wiring harness open or shorted",
            "ABS module fault",
            "Motor relay failure",
        ],
    ),
    "C0070": DTCInfo(
        code="C0070",
        description="ABS Control Module Malfunction",
        category="Chassis",
        severity="Critical",
        system="ABS",
        possible_causes=[
            "ABS module internal fault",
            "ABS module needs replacement",
            "Power supply problem",
            "Ground connection fault",
        ],
    ),
    "C0110": DTCInfo(
        code="C0110",
        description="Pump Motor Circuit Malfunction",
        category="Chassis",
        severity="Critical",
        system="ABS",
        possible_causes=[
            "Faulty pump motor",
            "Motor circuit open or shorted",
            "Poor electrical connection",
            "ABS module fault",
        ],
    ),
    "C0121": DTCInfo(
        code="C0121",
        description="Valve Relay Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Faulty valve relay",
            "Relay circuit problem",
            "ABS module fault",
            "Wiring harness issue",
        ],
    ),
    "C0161": DTCInfo(
        code="C0161",
        description="ABS/TCS Brake Switch Circuit Malfunction",
        category="Chassis",
        severity="Medium",
        system="ABS",
        possible_causes=[
            "Faulty brake light switch",
            "Wiring harness problem",
            "Brake switch misadjusted",
            "Poor electrical connection",
        ],
    ),
    "C0200": DTCInfo(
        code="C0200",
        description="Steering Angle Sensor Malfunction",
        category="Chassis",
        severity="High",
        system="Stability Control",
        possible_causes=[
            "Steering angle sensor not calibrated",
            "Faulty steering angle sensor",
            "Wiring harness problem",
            "ESC module fault",
        ],
    ),
    "C0201": DTCInfo(
        code="C0201",
        description="Yaw Rate Sensor Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="Stability Control",
        possible_causes=[
            "Faulty yaw rate sensor",
            "Sensor not calibrated",
            "Wiring harness problem",
            "ESC module fault",
        ],
    ),
    "C0202": DTCInfo(
        code="C0202",
        description="Lateral Acceleration Sensor Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="Stability Control",
        possible_causes=[
            "Faulty lateral acceleration sensor",
            "Sensor not calibrated",
            "Wiring harness problem",
            "ESC module fault",
        ],
    ),
    "C0210": DTCInfo(
        code="C0210",
        description="Park Brake Switch Circuit Malfunction",
        category="Chassis",
        severity="Low",
        system="Parking Brake",
        possible_causes=[
            "Faulty park brake switch",
            "Wiring harness problem",
            "Switch misadjusted",
            "Poor electrical connection",
        ],
    ),
    "C0221": DTCInfo(
        code="C0221",
        description="Right Front Wheel Speed Sensor Input Signal Missing",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel speed sensor not connected",
            "Faulty wheel speed sensor",
            "Wiring harness open circuit",
            "Reluctor ring damaged",
        ],
    ),
    "C0222": DTCInfo(
        code="C0222",
        description="Right Front Wheel Speed Signal Erratic",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel bearing worn",
            "Reluctor ring damaged",
            "Sensor air gap too large",
            "Faulty wheel speed sensor",
        ],
    ),
    "C0225": DTCInfo(
        code="C0225",
        description="Left Front Wheel Speed Sensor Input Signal Missing",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel speed sensor not connected",
            "Faulty wheel speed sensor",
            "Wiring harness open circuit",
            "Reluctor ring damaged",
        ],
    ),
    "C0226": DTCInfo(
        code="C0226",
        description="Left Front Wheel Speed Signal Erratic",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel bearing worn",
            "Reluctor ring damaged",
            "Sensor air gap too large",
            "Faulty wheel speed sensor",
        ],
    ),
    "C0229": DTCInfo(
        code="C0229",
        description="Right Rear Wheel Speed Sensor Input Signal Missing",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel speed sensor not connected",
            "Faulty wheel speed sensor",
            "Wiring harness open circuit",
            "Reluctor ring damaged",
        ],
    ),
    "C0230": DTCInfo(
        code="C0230",
        description="Right Rear Wheel Speed Signal Erratic",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel bearing worn",
            "Reluctor ring damaged",
            "Sensor air gap too large",
            "Faulty wheel speed sensor",
        ],
    ),
    "C0233": DTCInfo(
        code="C0233",
        description="Left Rear Wheel Speed Sensor Input Signal Missing",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel speed sensor not connected",
            "Faulty wheel speed sensor",
            "Wiring harness open circuit",
            "Reluctor ring damaged",
        ],
    ),
    "C0234": DTCInfo(
        code="C0234",
        description="Left Rear Wheel Speed Signal Erratic",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wheel bearing worn",
            "Reluctor ring damaged",
            "Sensor air gap too large",
            "Faulty wheel speed sensor",
        ],
    ),
    "C0245": DTCInfo(
        code="C0245",
        description="Wheel Speed Sensor Frequency Error",
        category="Chassis",
        severity="High",
        system="ABS",
        possible_causes=[
            "Wrong tire size installed",
            "Mismatched tire sizes",
            "Wheel speed sensor fault",
            "Reluctor ring incorrect",
        ],
    ),
    "C0265": DTCInfo(
        code="C0265",
        description="Electronic Brake Control Module Relay Circuit",
        category="Chassis",
        severity="Critical",
        system="ABS",
        possible_causes=[
            "Faulty EBCM relay",
            "Wiring harness problem",
            "ABS module fault",
            "Poor electrical connection",
        ],
    ),
    "C0267": DTCInfo(
        code="C0267",
        description="Pump Motor Circuit Open",
        category="Chassis",
        severity="Critical",
        system="ABS",
        possible_causes=[
            "ABS pump motor circuit open",
            "Faulty pump motor",
            "Wiring harness open circuit",
            "ABS module fault",
        ],
    ),
    "C0268": DTCInfo(
        code="C0268",
        description="Pump Motor Circuit Shorted to Ground",
        category="Chassis",
        severity="Critical",
        system="ABS",
        possible_causes=[
            "Pump motor circuit shorted to ground",
            "Faulty pump motor",
            "Wiring harness short",
            "ABS module fault",
        ],
    ),
    "C0269": DTCInfo(
        code="C0269",
        description="Pump Motor Circuit Shorted to Voltage",
        category="Chassis",
        severity="Critical",
        system="ABS",
        possible_causes=[
            "Pump motor circuit shorted to battery",
            "Faulty pump motor",
            "Wiring harness short",
            "ABS module fault",
        ],
    ),
    "C0278": DTCInfo(
        code="C0278",
        description="Stability Control System Temporarily Disabled",
        category="Chassis",
        severity="Medium",
        system="Stability Control",
        possible_causes=[
            "Sensor calibration needed",
            "System self-test failed",
            "Low battery voltage",
            "Temporary sensor fault",
        ],
    ),
    "C0279": DTCInfo(
        code="C0279",
        description="Stability Control System Permanently Disabled",
        category="Chassis",
        severity="High",
        system="Stability Control",
        possible_causes=[
            "ESC module fault",
            "Multiple sensor faults",
            "System configuration error",
            "Hardware failure",
        ],
    ),
    "C0281": DTCInfo(
        code="C0281",
        description="Brake Booster Pressure Sensor Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="Brake Booster",
        possible_causes=[
            "Faulty brake booster pressure sensor",
            "Wiring harness problem",
            "Vacuum leak",
            "Brake booster fault",
        ],
    ),
    "C0300": DTCInfo(
        code="C0300",
        description="Air Suspension System Malfunction",
        category="Chassis",
        severity="Medium",
        system="Suspension",
        possible_causes=[
            "Air suspension compressor fault",
            "Air leak in suspension",
            "Faulty height sensor",
            "Control module fault",
        ],
    ),
    "C0321": DTCInfo(
        code="C0321",
        description="Front Left Height Sensor Circuit Malfunction",
        category="Chassis",
        severity="Medium",
        system="Suspension",
        possible_causes=[
            "Faulty height sensor",
            "Wiring harness problem",
            "Sensor linkage disconnected",
            "Poor electrical connection",
        ],
    ),
    "C0325": DTCInfo(
        code="C0325",
        description="Front Right Height Sensor Circuit Malfunction",
        category="Chassis",
        severity="Medium",
        system="Suspension",
        possible_causes=[
            "Faulty height sensor",
            "Wiring harness problem",
            "Sensor linkage disconnected",
            "Poor electrical connection",
        ],
    ),
    "C0327": DTCInfo(
        code="C0327",
        description="Rear Left Height Sensor Circuit Malfunction",
        category="Chassis",
        severity="Medium",
        system="Suspension",
        possible_causes=[
            "Faulty height sensor",
            "Wiring harness problem",
            "Sensor linkage disconnected",
            "Poor electrical connection",
        ],
    ),
    "C0330": DTCInfo(
        code="C0330",
        description="Rear Right Height Sensor Circuit Malfunction",
        category="Chassis",
        severity="Medium",
        system="Suspension",
        possible_causes=[
            "Faulty height sensor",
            "Wiring harness problem",
            "Sensor linkage disconnected",
            "Poor electrical connection",
        ],
    ),
    "C0460": DTCInfo(
        code="C0460",
        description="Steering Angle Sensor 1 Circuit Malfunction",
        category="Chassis",
        severity="High",
        system="Power Steering",
        possible_causes=[
            "Faulty steering angle sensor",
            "Sensor not calibrated",
            "Wiring harness problem",
            "EPS control module fault",
        ],
    ),
    "C0550": DTCInfo(
        code="C0550",
        description="ECU Malfunction",
        category="Chassis",
        severity="Critical",
        system="Chassis Control",
        possible_causes=[
            "Control module internal fault",
            "Module needs replacement",
            "Software corruption",
            "Power supply problem",
        ],
    ),
    "C0561": DTCInfo(
        code="C0561",
        description="System Disabled Information Stored",
        category="Chassis",
        severity="Medium",
        system="Chassis Control",
        possible_causes=[
            "System disabled by user",
            "System disabled due to fault",
            "Configuration error",
            "Module needs programming",
        ],
    ),
    "C0710": DTCInfo(
        code="C0710",
        description="Steering Position Sensor Not Learned",
        category="Chassis",
        severity="Medium",
        system="Power Steering",
        possible_causes=[
            "Steering angle sensor calibration required",
            "Sensor learning procedure not performed",
            "Battery disconnected",
            "Sensor replaced without calibration",
        ],
    ),
    "C0750": DTCInfo(
        code="C0750",
        description="Electronic Stability Program (ESP) Control Module Performance",
        category="Chassis",
        severity="High",
        system="Stability Control",
        possible_causes=[
            "ESP module performance degraded",
            "Software issue",
            "Sensor inputs out of range",
            "Module needs replacement",
        ],
    ),
    "C0800": DTCInfo(
        code="C0800",
        description="Device Voltage Low",
        category="Chassis",
        severity="Medium",
        system="Power Supply",
        possible_causes=[
            "Low battery voltage",
            "Faulty alternator",
            "Poor battery connection",
            "Excessive electrical load",
        ],
    ),
    "C0801": DTCInfo(
        code="C0801",
        description="Device Voltage High",
        category="Chassis",
        severity="Medium",
        system="Power Supply",
        possible_causes=[
            "Alternator overcharging",
            "Faulty voltage regulator",
            "Poor ground connection",
            "Charging system problem",
        ],
    ),
    # ============================================================================
    # BODY (B) CODES
    # ============================================================================
    "B0001": DTCInfo(
        code="B0001",
        description="Driver Airbag Circuit Malfunction",
        category="Body",
        severity="Critical",
        system="Airbag System",
        possible_causes=[
            "Faulty driver airbag",
            "Airbag clockspring fault",
            "Wiring harness problem",
            "SRS control module fault",
        ],
    ),
    "B0002": DTCInfo(
        code="B0002",
        description="Passenger Airbag Circuit Malfunction",
        category="Body",
        severity="Critical",
        system="Airbag System",
        possible_causes=[
            "Faulty passenger airbag",
            "Wiring harness open or shorted",
            "Poor electrical connection",
            "SRS control module fault",
        ],
    ),
    "B0010": DTCInfo(
        code="B0010",
        description="Left Side Airbag Circuit Malfunction",
        category="Body",
        severity="Critical",
        system="Airbag System",
        possible_causes=[
            "Faulty side airbag",
            "Wiring harness problem",
            "Seat connector disconnected",
            "SRS control module fault",
        ],
    ),
    "B0011": DTCInfo(
        code="B0011",
        description="Right Side Airbag Circuit Malfunction",
        category="Body",
        severity="Critical",
        system="Airbag System",
        possible_causes=[
            "Faulty side airbag",
            "Wiring harness problem",
            "Seat connector disconnected",
            "SRS control module fault",
        ],
    ),
    "B0015": DTCInfo(
        code="B0015",
        description="Driver Seat Belt Pretensioner Circuit Malfunction",
        category="Body",
        severity="High",
        system="Airbag System",
        possible_causes=[
            "Faulty seat belt pretensioner",
            "Wiring harness problem",
            "Connector under seat disconnected",
            "SRS control module fault",
        ],
    ),
    "B0016": DTCInfo(
        code="B0016",
        description="Passenger Seat Belt Pretensioner Circuit Malfunction",
        category="Body",
        severity="High",
        system="Airbag System",
        possible_causes=[
            "Faulty seat belt pretensioner",
            "Wiring harness problem",
            "Connector under seat disconnected",
            "SRS control module fault",
        ],
    ),
    "B0020": DTCInfo(
        code="B0020",
        description="Front Impact Sensor Circuit Malfunction",
        category="Body",
        severity="High",
        system="Airbag System",
        possible_causes=[
            "Faulty impact sensor",
            "Wiring harness problem",
            "Sensor mounting damaged",
            "Poor electrical connection",
        ],
    ),
    "B0050": DTCInfo(
        code="B0050",
        description="Driver Seat Belt Buckle Switch Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Seat Belt System",
        possible_causes=[
            "Faulty seat belt buckle switch",
            "Wiring harness problem",
            "Switch stuck or damaged",
            "Poor electrical connection",
        ],
    ),
    "B0051": DTCInfo(
        code="B0051",
        description="Passenger Seat Belt Buckle Switch Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Seat Belt System",
        possible_causes=[
            "Faulty seat belt buckle switch",
            "Wiring harness problem",
            "Switch stuck or damaged",
            "Poor electrical connection",
        ],
    ),
    "B0100": DTCInfo(
        code="B0100",
        description="Electronic Control Unit Malfunction",
        category="Body",
        severity="Critical",
        system="Body Control Module",
        possible_causes=[
            "BCM internal fault",
            "Module needs replacement",
            "Software corruption",
            "Power supply problem",
        ],
    ),
    "B0101": DTCInfo(
        code="B0101",
        description="SRS Control Module Malfunction",
        category="Body",
        severity="Critical",
        system="Airbag System",
        possible_causes=[
            "SRS module internal fault",
            "Module needs replacement",
            "Crash data stored",
            "Module needs reset",
        ],
    ),
    "B0132": DTCInfo(
        code="B0132",
        description="Passenger Airbag Deactivation Indicator Circuit Malfunction",
        category="Body",
        severity="Medium",
        system="Airbag System",
        possible_causes=[
            "Faulty airbag indicator lamp",
            "Wiring harness problem",
            "Instrument cluster fault",
            "SRS module fault",
        ],
    ),
    "B0200": DTCInfo(
        code="B0200",
        description="Anti-Theft System Malfunction",
        category="Body",
        severity="Medium",
        system="Security System",
        possible_causes=[
            "Immobilizer key not recognized",
            "Faulty immobilizer antenna",
            "Security module fault",
            "Key needs programming",
        ],
    ),
    "B0201": DTCInfo(
        code="B0201",
        description="Immobilizer Malfunction",
        category="Body",
        severity="Critical",
        system="Security System",
        possible_causes=[
            "Immobilizer not programmed",
            "Key synchronization lost",
            "Immobilizer module fault",
            "Transponder key fault",
        ],
    ),
    "B0300": DTCInfo(
        code="B0300",
        description="Door Ajar Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Door Control",
        possible_causes=[
            "Faulty door ajar switch",
            "Door not closing properly",
            "Wiring harness problem",
            "Door latch worn",
        ],
    ),
    "B0301": DTCInfo(
        code="B0301",
        description="Driver Door Ajar Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Door Control",
        possible_causes=[
            "Faulty driver door ajar switch",
            "Door not closing properly",
            "Wiring harness problem",
            "Door latch worn",
        ],
    ),
    "B0302": DTCInfo(
        code="B0302",
        description="Passenger Door Ajar Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Door Control",
        possible_causes=[
            "Faulty passenger door ajar switch",
            "Door not closing properly",
            "Wiring harness problem",
            "Door latch worn",
        ],
    ),
    "B0400": DTCInfo(
        code="B0400",
        description="Power Window Motor Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Power Windows",
        possible_causes=[
            "Faulty window motor",
            "Window regulator jammed",
            "Wiring harness problem",
            "Window switch fault",
        ],
    ),
    "B0401": DTCInfo(
        code="B0401",
        description="Driver Power Window Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Power Windows",
        possible_causes=[
            "Faulty driver window motor",
            "Window regulator stuck",
            "Wiring harness problem",
            "Window switch fault",
        ],
    ),
    "B0500": DTCInfo(
        code="B0500",
        description="Central Locking System Malfunction",
        category="Body",
        severity="Low",
        system="Door Locks",
        possible_causes=[
            "Faulty door lock actuator",
            "Wiring harness problem",
            "Central locking module fault",
            "Door lock mechanism jammed",
        ],
    ),
    "B0501": DTCInfo(
        code="B0501",
        description="Driver Door Lock Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Door Locks",
        possible_causes=[
            "Faulty driver door lock actuator",
            "Wiring harness problem",
            "Lock mechanism stuck",
            "Poor electrical connection",
        ],
    ),
    "B0600": DTCInfo(
        code="B0600",
        description="Lamp Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Lighting System",
        possible_causes=[
            "Burned out bulb",
            "Wiring harness problem",
            "Lamp socket corrosion",
            "BCM fault",
        ],
    ),
    "B0601": DTCInfo(
        code="B0601",
        description="Headlamp Circuit Malfunction",
        category="Body",
        severity="Medium",
        system="Lighting System",
        possible_causes=[
            "Burned out headlamp bulb",
            "Faulty headlamp ballast (HID)",
            "Wiring harness problem",
            "Headlamp module fault",
        ],
    ),
    "B0602": DTCInfo(
        code="B0602",
        description="Left Turn Signal Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Lighting System",
        possible_causes=[
            "Burned out turn signal bulb",
            "Wiring harness problem",
            "Flasher relay fault",
            "BCM fault",
        ],
    ),
    "B0603": DTCInfo(
        code="B0603",
        description="Right Turn Signal Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Lighting System",
        possible_causes=[
            "Burned out turn signal bulb",
            "Wiring harness problem",
            "Flasher relay fault",
            "BCM fault",
        ],
    ),
    "B0604": DTCInfo(
        code="B0604",
        description="Brake Lamp Circuit Malfunction",
        category="Body",
        severity="High",
        system="Lighting System",
        possible_causes=[
            "Burned out brake light bulb",
            "Faulty brake light switch",
            "Wiring harness problem",
            "BCM fault",
        ],
    ),
    "B0700": DTCInfo(
        code="B0700",
        description="HVAC System Malfunction",
        category="Body",
        severity="Low",
        system="Climate Control",
        possible_causes=[
            "HVAC control module fault",
            "Sensor malfunction",
            "Actuator motor fault",
            "Wiring harness problem",
        ],
    ),
    "B0701": DTCInfo(
        code="B0701",
        description="Cabin Temperature Sensor Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Climate Control",
        possible_causes=[
            "Faulty cabin temperature sensor",
            "Wiring harness problem",
            "HVAC module fault",
            "Poor electrical connection",
        ],
    ),
    "B0702": DTCInfo(
        code="B0702",
        description="Ambient Temperature Sensor Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Climate Control",
        possible_causes=[
            "Faulty ambient temperature sensor",
            "Wiring harness problem",
            "Sensor location contaminated",
            "Poor electrical connection",
        ],
    ),
    "B0703": DTCInfo(
        code="B0703",
        description="Evaporator Temperature Sensor Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Climate Control",
        possible_causes=[
            "Faulty evaporator temperature sensor",
            "Wiring harness problem",
            "HVAC module fault",
            "A/C system issue",
        ],
    ),
    "B0710": DTCInfo(
        code="B0710",
        description="Blower Motor Circuit Malfunction",
        category="Body",
        severity="Low",
        system="Climate Control",
        possible_causes=[
            "Faulty blower motor",
            "Blower motor resistor fault",
            "Wiring harness problem",
            "HVAC control module fault",
        ],
    ),
    "B0800": DTCInfo(
        code="B0800",
        description="Instrument Cluster Malfunction",
        category="Body",
        severity="Medium",
        system="Instrument Cluster",
        possible_causes=[
            "Instrument cluster internal fault",
            "CAN communication problem",
            "Power supply issue",
            "Cluster needs replacement",
        ],
    ),
    "B1000": DTCInfo(
        code="B1000",
        description="Body Control Module Malfunction",
        category="Body",
        severity="High",
        system="Body Control Module",
        possible_causes=[
            "BCM internal fault",
            "Module needs replacement",
            "Software corruption",
            "Power supply problem",
        ],
    ),
    "B1200": DTCInfo(
        code="B1200",
        description="Keyless Entry System Malfunction",
        category="Body",
        severity="Low",
        system="Keyless Entry",
        possible_causes=[
            "Key fob battery weak",
            "Key fob not synchronized",
            "BCM fault",
            "Receiver antenna fault",
        ],
    ),
    "B1300": DTCInfo(
        code="B1300",
        description="Tire Pressure Monitor System Malfunction",
        category="Body",
        severity="Low",
        system="TPMS",
        possible_causes=[
            "TPMS sensor battery dead",
            "Faulty TPMS sensor",
            "TPMS receiver fault",
            "Sensor not programmed",
        ],
    ),
    "B1301": DTCInfo(
        code="B1301",
        description="Left Front TPMS Sensor Malfunction",
        category="Body",
        severity="Low",
        system="TPMS",
        possible_causes=[
            "Left front TPMS sensor battery dead",
            "Faulty TPMS sensor",
            "Signal not received",
            "Sensor damaged during tire service",
        ],
    ),
    "B1302": DTCInfo(
        code="B1302",
        description="Right Front TPMS Sensor Malfunction",
        category="Body",
        severity="Low",
        system="TPMS",
        possible_causes=[
            "Right front TPMS sensor battery dead",
            "Faulty TPMS sensor",
            "Signal not received",
            "Sensor damaged during tire service",
        ],
    ),
    "B1303": DTCInfo(
        code="B1303",
        description="Left Rear TPMS Sensor Malfunction",
        category="Body",
        severity="Low",
        system="TPMS",
        possible_causes=[
            "Left rear TPMS sensor battery dead",
            "Faulty TPMS sensor",
            "Signal not received",
            "Sensor damaged during tire service",
        ],
    ),
    "B1304": DTCInfo(
        code="B1304",
        description="Right Rear TPMS Sensor Malfunction",
        category="Body",
        severity="Low",
        system="TPMS",
        possible_causes=[
            "Right rear TPMS sensor battery dead",
            "Faulty TPMS sensor",
            "Signal not received",
            "Sensor damaged during tire service",
        ],
    ),
    "B1305": DTCInfo(
        code="B1305",
        description="Spare Tire TPMS Sensor Malfunction",
        category="Body",
        severity="Low",
        system="TPMS",
        possible_causes=[
            "Spare tire TPMS sensor battery dead",
            "Faulty TPMS sensor",
            "Signal not received",
            "Sensor not programmed",
        ],
    ),
    # ============================================================================
    # NETWORK/COMMUNICATION (U) CODES
    # ============================================================================
    "U0100": DTCInfo(
        code="U0100",
        description="Lost Communication With ECM/PCM",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "ECM not powered",
            "CAN bus wiring problem",
            "ECM internal fault",
            "CAN bus termination issue",
        ],
    ),
    "U0101": DTCInfo(
        code="U0101",
        description="Lost Communication With TCM",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "TCM not powered",
            "CAN bus wiring problem",
            "TCM internal fault",
            "CAN bus short circuit",
        ],
    ),
    "U0102": DTCInfo(
        code="U0102",
        description="Lost Communication With Transfer Case Control Module",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "Transfer case module not powered",
            "CAN bus wiring problem",
            "Module internal fault",
            "Poor ground connection",
        ],
    ),
    "U0103": DTCInfo(
        code="U0103",
        description="Lost Communication With Gear Shift Module",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "Gear shift module not powered",
            "CAN bus wiring problem",
            "Module internal fault",
            "Connector problem",
        ],
    ),
    "U0121": DTCInfo(
        code="U0121",
        description="Lost Communication With ABS Control Module",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "ABS module not powered",
            "CAN bus wiring problem",
            "ABS module internal fault",
            "Poor ground connection",
        ],
    ),
    "U0122": DTCInfo(
        code="U0122",
        description="Lost Communication With Vehicle Dynamics Control Module",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "VDC module not powered",
            "CAN bus wiring problem",
            "Module internal fault",
            "CAN bus termination problem",
        ],
    ),
    "U0123": DTCInfo(
        code="U0123",
        description="Lost Communication With Yaw Rate Sensor Module",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "Yaw rate sensor not powered",
            "CAN bus wiring problem",
            "Sensor module fault",
            "Poor electrical connection",
        ],
    ),
    "U0140": DTCInfo(
        code="U0140",
        description="Lost Communication With Body Control Module",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "BCM not powered",
            "CAN bus wiring problem",
            "BCM internal fault",
            "Ground connection issue",
        ],
    ),
    "U0141": DTCInfo(
        code="U0141",
        description="Lost Communication With Body Control Module 'B'",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "BCM not powered",
            "CAN bus wiring problem",
            "Module internal fault",
            "Connector problem",
        ],
    ),
    "U0151": DTCInfo(
        code="U0151",
        description="Lost Communication With SRS Control Module",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "SRS module not powered",
            "CAN bus wiring problem",
            "SRS module internal fault",
            "Ground connection problem",
        ],
    ),
    "U0155": DTCInfo(
        code="U0155",
        description="Lost Communication With Instrument Panel Cluster",
        category="Network",
        severity="Medium",
        system="CAN Bus",
        possible_causes=[
            "Instrument cluster not powered",
            "CAN bus wiring problem",
            "Cluster internal fault",
            "Poor electrical connection",
        ],
    ),
    "U0164": DTCInfo(
        code="U0164",
        description="Lost Communication With HVAC Control Module",
        category="Network",
        severity="Low",
        system="CAN Bus",
        possible_causes=[
            "HVAC module not powered",
            "CAN bus wiring problem",
            "Module internal fault",
            "Connector problem",
        ],
    ),
    "U0168": DTCInfo(
        code="U0168",
        description="Lost Communication With Vehicle Security Control Module",
        category="Network",
        severity="Medium",
        system="CAN Bus",
        possible_causes=[
            "Security module not powered",
            "CAN bus wiring problem",
            "Module internal fault",
            "Ground connection problem",
        ],
    ),
    "U0184": DTCInfo(
        code="U0184",
        description="Lost Communication With Radio",
        category="Network",
        severity="Low",
        system="CAN Bus",
        possible_causes=[
            "Radio not powered",
            "CAN bus wiring problem",
            "Radio internal fault",
            "Antenna amplifier problem",
        ],
    ),
    "U0200": DTCInfo(
        code="U0200",
        description="Lost Communication With Infotainment System",
        category="Network",
        severity="Low",
        system="CAN Bus",
        possible_causes=[
            "Infotainment system not powered",
            "CAN bus wiring problem",
            "System internal fault",
            "Software issue",
        ],
    ),
    "U0300": DTCInfo(
        code="U0300",
        description="Internal Control Module Software Incompatibility",
        category="Network",
        severity="High",
        system="Module Programming",
        possible_causes=[
            "Module software version mismatch",
            "Module needs programming",
            "Wrong software installed",
            "Module replacement needed",
        ],
    ),
    "U0401": DTCInfo(
        code="U0401",
        description="Invalid Data Received From ECM/PCM",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "ECM transmitting incorrect data",
            "CAN bus signal integrity problem",
            "EMI interference",
            "ECM internal fault",
        ],
    ),
    "U0402": DTCInfo(
        code="U0402",
        description="Invalid Data Received From TCM",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "TCM transmitting incorrect data",
            "CAN bus signal integrity problem",
            "EMI interference",
            "TCM internal fault",
        ],
    ),
    "U1000": DTCInfo(
        code="U1000",
        description="CAN Bus Open",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "CAN High or Low wire open circuit",
            "Module not connected to bus",
            "Wiring harness damage",
            "Connector problem",
        ],
    ),
    "U1001": DTCInfo(
        code="U1001",
        description="CAN Bus Short to Ground",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "CAN High or Low shorted to ground",
            "Wiring harness damage",
            "Module internal fault",
            "Connector corrosion",
        ],
    ),
    "U1002": DTCInfo(
        code="U1002",
        description="CAN Bus Short to Battery",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "CAN High or Low shorted to voltage",
            "Wiring harness damage",
            "Module internal fault",
            "Chafed wiring",
        ],
    ),
    "U1003": DTCInfo(
        code="U1003",
        description="CAN Bus Lines Shorted Together",
        category="Network",
        severity="Critical",
        system="CAN Bus",
        possible_causes=[
            "CAN High and Low shorted together",
            "Wiring harness damage",
            "Connector problem",
            "Module fault",
        ],
    ),
    "U1004": DTCInfo(
        code="U1004",
        description="CAN Bus Communication Error",
        category="Network",
        severity="High",
        system="CAN Bus",
        possible_causes=[
            "CAN bus termination problem",
            "Too many errors on bus",
            "EMI interference",
            "Faulty module transmitting errors",
        ],
    ),
    "U1005": DTCInfo(
        code="U1005",
        description="CAN Bus Message Counter Error",
        category="Network",
        severity="Medium",
        system="CAN Bus",
        possible_causes=[
            "Module transmitting incorrect counters",
            "Message timing problem",
            "Software issue",
            "Module needs programming",
        ],
    ),
}


class DTCDatabase:
    """Database for looking up Diagnostic Trouble Codes (DTCs).

    This class provides methods to search and retrieve DTC information from
    a comprehensive database of standardized automotive fault codes.
    """

    @staticmethod
    def lookup(code: str) -> DTCInfo | None:
        """Look up a DTC by its code.

        Args:
            code: DTC code to look up (e.g., "P0420", "p0420")
                  Case-insensitive

        Returns:
            DTCInfo object if found, None if code not in database

        Example:
            >>> info = DTCDatabase.lookup("P0420")
            >>> if info:
            ...     print(f"{info.code}: {info.description}")
            P0420: Catalyst System Efficiency Below Threshold (Bank 1)
        """
        return DTCS.get(code.upper())

    @staticmethod
    def search(keyword: str) -> list[DTCInfo]:
        """Search DTCs by keyword in description or possible causes.

        Searches are case-insensitive and match partial words.

        Args:
            keyword: Search term (e.g., "oxygen sensor", "misfire", "ABS")

        Returns:
            List of matching DTCInfo objects, sorted by code

        Example:
            >>> results = DTCDatabase.search("oxygen sensor")
            >>> for dtc in results[:3]:
            ...     print(f"{dtc.code}: {dtc.description}")
            P0130: O2 Sensor Circuit Malfunction (Bank 1 Sensor 1)
            P0131: O2 Sensor Circuit Low Voltage (Bank 1 Sensor 1)
            P0132: O2 Sensor Circuit High Voltage (Bank 1 Sensor 1)
        """
        keyword_lower = keyword.lower()
        results = []

        for dtc in DTCS.values():
            # Search in description
            if keyword_lower in dtc.description.lower():
                results.append(dtc)
                continue

            # Search in system
            if keyword_lower in dtc.system.lower():
                results.append(dtc)
                continue

            # Search in possible causes
            for cause in dtc.possible_causes:
                if keyword_lower in cause.lower():
                    results.append(dtc)
                    break

        # Sort by code
        results.sort(key=lambda x: x.code)
        return results

    @staticmethod
    def get_by_category(category: str) -> list[DTCInfo]:
        """Get all DTCs in a specific category.

        Args:
            category: Category name ("Powertrain", "Chassis", "Body", "Network")
                     Case-insensitive

        Returns:
            List of DTCInfo objects in the category, sorted by code

        Example:
            >>> chassis_codes = DTCDatabase.get_by_category("Chassis")
            >>> print(f"Found {len(chassis_codes)} chassis codes")
            Found 42 chassis codes
        """
        category_lower = category.lower()
        results = [dtc for dtc in DTCS.values() if dtc.category.lower() == category_lower]

        # Sort by code
        results.sort(key=lambda x: x.code)
        return results

    @staticmethod
    def get_by_system(system: str) -> list[DTCInfo]:
        """Get all DTCs for a specific system.

        Args:
            system: System name (e.g., "ABS", "Oxygen Sensors", "Fuel System")
                   Case-insensitive

        Returns:
            List of DTCInfo objects for the system, sorted by code

        Example:
            >>> abs_codes = DTCDatabase.get_by_system("ABS")
            >>> for dtc in abs_codes[:3]:
            ...     print(f"{dtc.code}: {dtc.description}")
        """
        system_lower = system.lower()
        results = [dtc for dtc in DTCS.values() if dtc.system.lower() == system_lower]

        # Sort by code
        results.sort(key=lambda x: x.code)
        return results

    @staticmethod
    def parse_dtc(code: str) -> tuple[str, str, str] | None:
        """Parse a DTC code into its components.

        Args:
            code: DTC code to parse (e.g., "P0420")

        Returns:
            Tuple of (category, code_type, fault_code) or None if invalid
            - category: "Powertrain", "Chassis", "Body", or "Network"
            - code_type: "Generic" (0) or "Manufacturer" (1-3)
            - fault_code: Remaining 3 digits

        Example:
            >>> result = DTCDatabase.parse_dtc("P0420")
            >>> if result:
            ...     category, code_type, fault_code = result
            ...     print(f"Category: {category}, Type: {code_type}, Code: {fault_code}")
            Category: Powertrain, Type: Generic, Code: 420
        """
        code = code.upper().strip()

        # Validate format
        if len(code) != 5:
            return None

        # Parse category
        category_map = {
            "P": "Powertrain",
            "C": "Chassis",
            "B": "Body",
            "U": "Network",
        }
        category = category_map.get(code[0])
        if not category:
            return None

        # Parse code type
        try:
            type_digit = int(code[1])
        except ValueError:
            return None

        code_type = "Generic" if type_digit == 0 else "Manufacturer"

        # Parse fault code
        try:
            fault_code = code[2:5]
            int(fault_code)  # Validate it's numeric
        except ValueError:
            return None

        return (category, code_type, fault_code)

    @staticmethod
    def get_all_codes() -> list[str]:
        """Get a list of all DTC codes in the database.

        Returns:
            Sorted list of all DTC codes

        Example:
            >>> all_codes = DTCDatabase.get_all_codes()
            >>> print(f"Database contains {len(all_codes)} codes")
            >>> print(f"First code: {all_codes[0]}")
            >>> print(f"Last code: {all_codes[-1]}")
        """
        return sorted(DTCS.keys())

    @staticmethod
    def get_stats() -> dict[str, int]:
        """Get statistics about the DTC database.

        Returns:
            Dictionary with counts by category and total

        Example:
            >>> stats = DTCDatabase.get_stats()
            >>> for category, count in stats.items():
            ...     print(f"{category}: {count}")
            Powertrain: 100
            Chassis: 42
            Body: 45
            Network: 23
            Total: 210
        """
        stats = {"Powertrain": 0, "Chassis": 0, "Body": 0, "Network": 0}

        for dtc in DTCS.values():
            stats[dtc.category] += 1

        stats["Total"] = len(DTCS)
        return stats
