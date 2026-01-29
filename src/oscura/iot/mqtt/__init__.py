"""MQTT protocol analysis for versions 3.1.1 and 5.0.

This module provides MQTT (Message Queuing Telemetry Transport) protocol
analysis including packet parsing, topic hierarchy discovery, and session tracking.

Example:
    >>> from oscura.iot.mqtt import MQTTAnalyzer, MQTTPacket
    >>> analyzer = MQTTAnalyzer()
    >>> packet = analyzer.parse_packet(mqtt_data, timestamp=0.0)
    >>> topology = analyzer.get_topic_hierarchy()

References:
    MQTT 3.1.1: http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/
    MQTT 5.0: https://docs.oasis-open.org/mqtt/mqtt/v5.0/
"""

from oscura.iot.mqtt.analyzer import (
    MQTTAnalyzer,
    MQTTPacket,
    MQTTSession,
)

__all__ = [
    "MQTTAnalyzer",
    "MQTTPacket",
    "MQTTSession",
]
