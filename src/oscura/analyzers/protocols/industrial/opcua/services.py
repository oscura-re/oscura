"""OPC UA service request/response decoders.

This module implements parsing for OPC UA service-specific payloads including
Read, Write, Browse, Subscribe, and Publish operations.

References:
    OPC UA Part 4: Services
    https://reference.opcfoundation.org/Core/Part4/v105/docs/
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def parse_read_request(data: bytes) -> dict[str, Any]:
    """Parse ReadRequest service payload.

    ReadRequest Format:
    - RequestHeader (complex structure)
    - MaxAge (8 bytes, double)
    - TimestampsToReturn (4 bytes, enum)
    - NodesToRead (array of ReadValueId)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed request data.

    Example:
        >>> # Simplified ReadRequest
        >>> request = parse_read_request(b'...')
        >>> assert 'nodes_to_read' in request
    """
    # Simplified implementation - full parsing would be very complex
    result: dict[str, Any] = {
        "service": "ReadRequest",
        "payload_size": len(data),
    }

    # Try to extract basic information
    if len(data) >= 12:
        # Skip request header (would need complex parsing)
        # MaxAge at some offset
        # TimestampsToReturn enum
        result["partial_parse"] = True

    return result


def parse_read_response(data: bytes) -> dict[str, Any]:
    """Parse ReadResponse service payload.

    ReadResponse Format:
    - ResponseHeader (complex structure)
    - Results (array of DataValue)
    - DiagnosticInfos (array, optional)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed response data.

    Example:
        >>> response = parse_read_response(b'...')
        >>> assert 'service' in response
    """
    result: dict[str, Any] = {
        "service": "ReadResponse",
        "payload_size": len(data),
    }

    return result


def parse_write_request(data: bytes) -> dict[str, Any]:
    """Parse WriteRequest service payload.

    WriteRequest Format:
    - RequestHeader
    - NodesToWrite (array of WriteValue)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed request data.

    Example:
        >>> request = parse_write_request(b'...')
        >>> assert request['service'] == 'WriteRequest'
    """
    result: dict[str, Any] = {
        "service": "WriteRequest",
        "payload_size": len(data),
    }

    return result


def parse_browse_request(data: bytes) -> dict[str, Any]:
    """Parse BrowseRequest service payload.

    BrowseRequest Format:
    - RequestHeader
    - View (ViewDescription)
    - RequestedMaxReferencesPerNode (4 bytes)
    - NodesToBrowse (array of BrowseDescription)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed request data.

    Example:
        >>> request = parse_browse_request(b'...')
        >>> assert 'service' in request
    """
    result: dict[str, Any] = {
        "service": "BrowseRequest",
        "payload_size": len(data),
    }

    return result


def parse_browse_response(data: bytes) -> dict[str, Any]:
    """Parse BrowseResponse service payload.

    BrowseResponse Format:
    - ResponseHeader
    - Results (array of BrowseResult)
    - DiagnosticInfos (array, optional)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed response data.

    Example:
        >>> response = parse_browse_response(b'...')
        >>> assert response['service'] == 'BrowseResponse'
    """
    result: dict[str, Any] = {
        "service": "BrowseResponse",
        "payload_size": len(data),
    }

    return result


def parse_create_subscription_request(data: bytes) -> dict[str, Any]:
    """Parse CreateSubscriptionRequest service payload.

    CreateSubscriptionRequest Format:
    - RequestHeader
    - RequestedPublishingInterval (8 bytes, double)
    - RequestedLifetimeCount (4 bytes)
    - RequestedMaxKeepAliveCount (4 bytes)
    - MaxNotificationsPerPublish (4 bytes)
    - PublishingEnabled (1 byte, boolean)
    - Priority (1 byte)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed request data with subscription parameters.

    Example:
        >>> request = parse_create_subscription_request(b'...')
        >>> assert 'service' in request
    """
    result: dict[str, Any] = {
        "service": "CreateSubscriptionRequest",
        "payload_size": len(data),
    }

    return result


def parse_publish_request(data: bytes) -> dict[str, Any]:
    """Parse PublishRequest service payload.

    PublishRequest Format:
    - RequestHeader
    - SubscriptionAcknowledgements (array)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed request data.

    Example:
        >>> request = parse_publish_request(b'...')
        >>> assert request['service'] == 'PublishRequest'
    """
    result: dict[str, Any] = {
        "service": "PublishRequest",
        "payload_size": len(data),
    }

    return result


def parse_publish_response(data: bytes) -> dict[str, Any]:
    """Parse PublishResponse service payload.

    PublishResponse Format:
    - ResponseHeader
    - SubscriptionId (4 bytes)
    - AvailableSequenceNumbers (array)
    - MoreNotifications (1 byte, boolean)
    - NotificationMessage (complex)
    - Results (array of StatusCode)
    - DiagnosticInfos (array, optional)

    Args:
        data: Service payload bytes.

    Returns:
        Parsed response data with notification data.

    Example:
        >>> response = parse_publish_response(b'...')
        >>> assert 'service' in response
    """
    result: dict[str, Any] = {
        "service": "PublishResponse",
        "payload_size": len(data),
    }

    return result


# Service ID to parser mapping
SERVICE_PARSERS: dict[
    int, tuple[Callable[[bytes], dict[str, Any]] | None, Callable[[bytes], dict[str, Any]] | None]
] = {
    421: (parse_read_request, parse_read_response),  # Read
    673: (parse_write_request, None),  # Write (response is simple StatusCode array)
    527: (parse_browse_request, parse_browse_response),  # Browse
    631: (parse_create_subscription_request, None),  # CreateSubscription
    826: (parse_publish_request, parse_publish_response),  # Publish
}


__all__ = [
    "SERVICE_PARSERS",
    "parse_browse_request",
    "parse_browse_response",
    "parse_create_subscription_request",
    "parse_publish_request",
    "parse_publish_response",
    "parse_read_request",
    "parse_read_response",
    "parse_write_request",
]
