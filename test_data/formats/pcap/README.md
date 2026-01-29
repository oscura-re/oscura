# PCAP Test Data

This directory contains network packet capture files for testing Oscura's PCAP loading and network analysis capabilities.

## Directory Structure

```
pcap/
├── tcp/           # TCP-based protocols
│   ├── http/      # HTTP traffic
│   ├── https/     # HTTPS/TLS traffic (ssl.pcap)
│   ├── ftp/       # FTP file transfers
│   ├── ssh/       # SSH sessions
│   ├── smtp/      # Email (SMTP) traffic
│   ├── pop3/      # Email (POP3) traffic
│   ├── imap/      # Email (IMAP) traffic
│   └── telnet/    # Telnet sessions
├── udp/           # UDP-based protocols
│   ├── dns/       # DNS queries/responses
│   ├── ntp/       # NTP time sync
│   ├── syslog/    # Syslog messages
│   └── tftp/      # TFTP file transfers
├── industrial/    # Industrial protocols
│   ├── modbus_tcp/# Modbus/TCP
│   ├── dnp3/      # DNP3 SCADA protocol
│   └── opcua/     # OPC-UA industrial automation
├── iot/           # IoT protocols
│   ├── mqtt/      # MQTT messaging
│   └── coap/      # CoAP requests
└── malformed/     # Invalid/corrupted PCAPs for error handling tests
```

## Known Issues

### Corrupted Files (Download Errors)

The following files were incorrectly downloaded and contain HTML error pages instead of
actual PCAP data. They need to be re-downloaded from Wireshark's sample captures:

- `tcp/http/http.pcap` - Contains GitLab HTML page instead of PCAP
- `tcp/ftp/ftp.pcap` - Contains GitLab HTML page instead of PCAP
- `tcp/ssh/ssh.pcap` - Contains GitLab HTML page instead of PCAP
- `industrial/modbus_tcp/modbus.pcap` - Contains GitLab HTML page instead of PCAP

**To fix**: Re-download from https://wiki.wireshark.org/SampleCaptures using the
direct download links, not the wiki page links.

### Synthetic Test Credentials

Some PCAP files contain simulated credentials in the captured traffic for testing
protocol decoding. These are **synthetic test values**, not real credentials:

- `tcp/telnet/telnet_session.pcap` - Contains simulated login: admin/secret123
- `tcp/pop3/pop3_session.pcap` - Contains simulated login: admin/secret123
- `tcp/imap/imap_session.pcap` - Contains simulated login: admin/secret123

These values are intentionally included for protocol decoder testing and are
clearly documented as synthetic.

## Valid PCAP Files

The following files are verified valid PCAP format:

### TCP Protocol Captures

- `tcp/https/ssl.pcap` - TLS/SSL handshake and encrypted traffic
- `tcp/smtp/smtp.pcap` - SMTP email session
- `tcp/pop3/pop3_session.pcap` - POP3 email retrieval (synthetic)
- `tcp/imap/imap_session.pcap` - IMAP email session (synthetic)
- `tcp/telnet/telnet_session.pcap` - Telnet interactive session (synthetic)

### UDP Protocol Captures

- `udp/dns/dns.pcap` - DNS query/response pairs
- `udp/ntp/ntp.pcap` - NTP time synchronization
- `udp/syslog/syslog.pcap` - Syslog message examples
- `udp/tftp/tftp_transfer.pcap` - TFTP file transfer

### Industrial Protocol Captures

- `industrial/dnp3/dnp3_poll.pcap` - DNP3 SCADA polling
- `industrial/opcua/opcua_connect.pcap` - OPC-UA connection setup

### IoT Protocol Captures

- `iot/mqtt/mqtt_packets_tcpdump.pcap` - MQTT publish/subscribe
- `iot/mqtt/mqtt_packets_RedHat61.pcap` - MQTT session from RedHat
- `iot/coap/coap_request.pcap` - CoAP request/response

### Test Files

- `malformed/truncated_packet.pcap` - Intentionally truncated for error handling

## Sources

- **Wireshark Sample Captures**: https://wiki.wireshark.org/SampleCaptures
- **MQTT Captures**: https://github.com/pradeesi/MQTT-Wireshark-Capture
- **Synthetic**: Generated internally for specific test scenarios

## Usage

```python
from oscura.loaders.pcap import load_pcap

# Load a valid PCAP file
packets = load_pcap("test_data/pcap/udp/dns/dns.pcap")

# Analyze packets
for packet in packets:
    print(f"Time: {packet.timestamp}, Length: {packet.length}")
```

## Maintenance

When adding new PCAP files:

1. Verify the file is valid PCAP format (starts with magic bytes `\xd4\xc3\xb2\xa1` or `\xa1\xb2\xc3\xd4`)
2. Add entry to this README documenting the source and contents
3. If the file contains any credential-like strings, document them as synthetic test data
4. Update the test_data/README.md if adding new protocol categories
