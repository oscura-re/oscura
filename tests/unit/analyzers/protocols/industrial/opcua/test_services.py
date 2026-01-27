"""Tests for OPC UA service parsers."""

from __future__ import annotations

from oscura.analyzers.protocols.industrial.opcua.services import (
    SERVICE_PARSERS,
    parse_browse_request,
    parse_browse_response,
    parse_create_subscription_request,
    parse_publish_request,
    parse_publish_response,
    parse_read_request,
    parse_read_response,
    parse_write_request,
)


class TestServiceParsers:
    """Test suite for OPC UA service parsers."""

    def test_parse_read_request(self) -> None:
        """Test parsing ReadRequest service."""
        # Simplified payload
        data = b"\x00" * 20
        result = parse_read_request(data)

        assert result["service"] == "ReadRequest"
        assert result["payload_size"] == 20

    def test_parse_read_response(self) -> None:
        """Test parsing ReadResponse service."""
        data = b"\x00" * 16
        result = parse_read_response(data)

        assert result["service"] == "ReadResponse"
        assert result["payload_size"] == 16

    def test_parse_write_request(self) -> None:
        """Test parsing WriteRequest service."""
        data = b"\x00" * 24
        result = parse_write_request(data)

        assert result["service"] == "WriteRequest"
        assert result["payload_size"] == 24

    def test_parse_browse_request(self) -> None:
        """Test parsing BrowseRequest service."""
        data = b"\x00" * 32
        result = parse_browse_request(data)

        assert result["service"] == "BrowseRequest"
        assert result["payload_size"] == 32

    def test_parse_browse_response(self) -> None:
        """Test parsing BrowseResponse service."""
        data = b"\x00" * 28
        result = parse_browse_response(data)

        assert result["service"] == "BrowseResponse"
        assert result["payload_size"] == 28

    def test_parse_create_subscription_request(self) -> None:
        """Test parsing CreateSubscriptionRequest service."""
        data = b"\x00" * 40
        result = parse_create_subscription_request(data)

        assert result["service"] == "CreateSubscriptionRequest"
        assert result["payload_size"] == 40

    def test_parse_publish_request(self) -> None:
        """Test parsing PublishRequest service."""
        data = b"\x00" * 12
        result = parse_publish_request(data)

        assert result["service"] == "PublishRequest"
        assert result["payload_size"] == 12

    def test_parse_publish_response(self) -> None:
        """Test parsing PublishResponse service."""
        data = b"\x00" * 36
        result = parse_publish_response(data)

        assert result["service"] == "PublishResponse"
        assert result["payload_size"] == 36

    def test_service_parsers_mapping(self) -> None:
        """Test SERVICE_PARSERS mapping contains expected entries."""
        # ReadRequest/Response
        assert 421 in SERVICE_PARSERS
        assert SERVICE_PARSERS[421][0] is parse_read_request
        assert SERVICE_PARSERS[421][1] is parse_read_response

        # WriteRequest
        assert 673 in SERVICE_PARSERS
        assert SERVICE_PARSERS[673][0] is parse_write_request

        # BrowseRequest/Response
        assert 527 in SERVICE_PARSERS
        assert SERVICE_PARSERS[527][0] is parse_browse_request
        assert SERVICE_PARSERS[527][1] is parse_browse_response

        # CreateSubscriptionRequest
        assert 631 in SERVICE_PARSERS
        assert SERVICE_PARSERS[631][0] is parse_create_subscription_request

        # PublishRequest/Response
        assert 826 in SERVICE_PARSERS
        assert SERVICE_PARSERS[826][0] is parse_publish_request
        assert SERVICE_PARSERS[826][1] is parse_publish_response

    def test_parse_empty_payload(self) -> None:
        """Test parsing with empty payload."""
        result = parse_read_request(b"")
        assert result["service"] == "ReadRequest"
        assert result["payload_size"] == 0

    def test_parse_minimal_payloads(self) -> None:
        """Test parsing all services with minimal payloads."""
        minimal_data = b"\x00" * 8

        # Test each service parser
        assert parse_read_request(minimal_data)["service"] == "ReadRequest"
        assert parse_read_response(minimal_data)["service"] == "ReadResponse"
        assert parse_write_request(minimal_data)["service"] == "WriteRequest"
        assert parse_browse_request(minimal_data)["service"] == "BrowseRequest"
        assert parse_browse_response(minimal_data)["service"] == "BrowseResponse"
        assert (
            parse_create_subscription_request(minimal_data)["service"]
            == "CreateSubscriptionRequest"
        )
        assert parse_publish_request(minimal_data)["service"] == "PublishRequest"
        assert parse_publish_response(minimal_data)["service"] == "PublishResponse"
