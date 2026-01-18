#!/bin/bash
# Download comprehensive test data for Oscura validation
# Generated: 2025-12-25

set -e

TEST_DATA_DIR="${1:-test_data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "🔍 Oscura Test Data Download"
echo "================================"
echo "Target directory: ${PROJECT_ROOT}/${TEST_DATA_DIR}"
echo

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p "${PROJECT_ROOT}/${TEST_DATA_DIR}"/{waveforms/{tektronix/{analog/{single_channel,multi_channel},digital/{logic_analyzer,mixed_signal},legacy/{wfm003,tds_series}},rigol,keysight,synthetic},logic_analyzer/{sigrok/{uart,spi,i2c,can,usb,jtag,parallel},vcd},pcap/{tcp/{http,https,ftp,ssh,smtp},udp/{dns,ntp,sip_voip,custom_binary},industrial/{modbus_tcp,dnp3,bacnet,opcua,ethernet_ip},iot/{mqtt,coap,zigbee_ip},mixed,malformed},binary/{synthetic,can,serial},statistical/{entropy,patterns},ground_truth/{annotations,decoded}}

cd "${PROJECT_ROOT}"

# Counter for downloaded files
DOWNLOAD_COUNT=0

# =============================================================================
# 1. Wireshark Sample Captures
# =============================================================================
echo "📦 Downloading Wireshark sample captures..."

declare -A WIRESHARK_SAMPLES=(
  ["${TEST_DATA_DIR}/pcap/tcp/http/http.pcap"]="https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/http.cap"
  ["${TEST_DATA_DIR}/pcap/udp/dns/dns.pcap"]="https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/dns.cap"
  ["${TEST_DATA_DIR}/pcap/tcp/ftp/ftp.pcap"]="https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/ftp.pcap"
  ["${TEST_DATA_DIR}/pcap/tcp/smtp/smtp.pcap"]="https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/smtp.pcap"
  ["${TEST_DATA_DIR}/pcap/tcp/ssh/ssh.pcap"]="https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/ssh.cap"
  ["${TEST_DATA_DIR}/pcap/udp/ntp/ntp.pcap"]="https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/NTP_sync.pcap"
  ["${TEST_DATA_DIR}/pcap/industrial/modbus_tcp/modbus.pcap"]="https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/modbus.pcap"
)

for dest in "${!WIRESHARK_SAMPLES[@]}"; do
  url="${WIRESHARK_SAMPLES[${dest}]}"
  if [[ ! -f "${dest}" ]]; then
    echo "  ↓ $(basename "${dest}")"
    curl -L -s -o "${dest}" "${url}"
    ((DOWNLOAD_COUNT++))
  else
    echo "  ✓ $(basename "${dest}") (already exists)"
  fi
done

# =============================================================================
# 2. MQTT Captures
# =============================================================================
echo "📦 Downloading MQTT captures..."

declare -A MQTT_SAMPLES=(
  ["${TEST_DATA_DIR}/pcap/iot/mqtt/mqtt_packets_tcpdump.pcap"]="https://github.com/pradeesi/MQTT-Wireshark-Capture/raw/master/mqtt_packets_tcpdump.pcap"
  ["${TEST_DATA_DIR}/pcap/iot/mqtt/mqtt_packets_RedHat61.pcap"]="https://github.com/pradeesi/MQTT-Wireshark-Capture/raw/master/mqtt_packets_RedHat61_tcpdump.pcap"
)

for dest in "${!MQTT_SAMPLES[@]}"; do
  url="${MQTT_SAMPLES[${dest}]}"
  if [[ ! -f "${dest}" ]]; then
    echo "  ↓ $(basename "${dest}")"
    curl -L -s -o "${dest}" "${url}"
    ((DOWNLOAD_COUNT++))
  else
    echo "  ✓ $(basename "${dest}") (already exists)"
  fi
done

# =============================================================================
# 3. Tektronix tm_data_types
# =============================================================================
echo "📦 Downloading Tektronix WFM files..."

if [[ ! -d "/tmp/tm_data_types" ]]; then
  echo "  ↓ Cloning tm_data_types repository..."
  git clone --depth 1 https://github.com/tektronix/tm_data_types.git /tmp/tm_data_types > /dev/null 2>&1
fi

if [[ -d "/tmp/tm_data_types/tests/waveforms" ]]; then
  echo "  ↓ Copying WFM test files..."
  cp /tmp/tm_data_types/tests/waveforms/*.wfm "${TEST_DATA_DIR}/waveforms/tektronix/analog/single_channel/" 2> /dev/null || true
  cp /tmp/tm_data_types/examples/samples/*.wfm "${TEST_DATA_DIR}/waveforms/tektronix/analog/single_channel/" 2> /dev/null || true
  cp -r /tmp/tm_data_types/tests/waveforms/invalid_waveforms "${TEST_DATA_DIR}/waveforms/tektronix/" 2> /dev/null || true
  WFM_COUNT=$(find "${TEST_DATA_DIR}/waveforms/tektronix" -name "*.wfm" | wc -l)
  echo "  ✓ Copied ${WFM_COUNT} WFM files"
  ((DOWNLOAD_COUNT += WFM_COUNT))
fi

# =============================================================================
# 4. sigrok-dumps
# =============================================================================
echo "📦 Downloading sigrok protocol captures..."

if [[ ! -d "/tmp/sigrok-dumps" ]]; then
  echo "  ↓ Cloning sigrok-dumps repository (this may take a moment)..."
  git clone --depth 1 https://github.com/sigrokproject/sigrok-dumps.git /tmp/sigrok-dumps > /dev/null 2>&1
fi

if [[ -d "/tmp/sigrok-dumps" ]]; then
  echo "  ↓ Copying protocol captures..."

  # Copy protocol-specific captures
  cp -r /tmp/sigrok-dumps/uart/* "${TEST_DATA_DIR}/logic_analyzer/sigrok/uart/" 2> /dev/null || true
  cp -r /tmp/sigrok-dumps/spi/* "${TEST_DATA_DIR}/logic_analyzer/sigrok/spi/" 2> /dev/null || true
  cp -r /tmp/sigrok-dumps/i2c/* "${TEST_DATA_DIR}/logic_analyzer/sigrok/i2c/" 2> /dev/null || true
  cp -r /tmp/sigrok-dumps/can/* "${TEST_DATA_DIR}/logic_analyzer/sigrok/can/" 2> /dev/null || true
  cp -r /tmp/sigrok-dumps/usb/* "${TEST_DATA_DIR}/logic_analyzer/sigrok/usb/" 2> /dev/null || true
  cp -r /tmp/sigrok-dumps/jtag/* "${TEST_DATA_DIR}/logic_analyzer/sigrok/jtag/" 2> /dev/null || true

  SIGROK_COUNT=$(find "${TEST_DATA_DIR}/logic_analyzer/sigrok" -name "*.sr" | wc -l)
  echo "  ✓ Copied ${SIGROK_COUNT} sigrok captures"
  ((DOWNLOAD_COUNT += SIGROK_COUNT))
fi

# =============================================================================
# 5. Generate Synthetic Data
# =============================================================================
echo "🔧 Generating synthetic test data..."

if [[ -f "${SCRIPT_DIR}/generate_comprehensive_test_data.py" ]]; then
  uv run python "${SCRIPT_DIR}/generate_comprehensive_test_data.py" "${TEST_DATA_DIR}" > /dev/null 2>&1
  SYNTHETIC_COUNT=$(find "${TEST_DATA_DIR}/binary/synthetic" -type f | wc -l)
  echo "  ✓ Generated ${SYNTHETIC_COUNT} synthetic files"
  ((DOWNLOAD_COUNT += SYNTHETIC_COUNT))
else
  echo "  ⚠ generate_comprehensive_test_data.py not found, skipping synthetic data generation"
fi

# =============================================================================
# Summary
# =============================================================================
echo
echo "✨ Download Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total files: ${DOWNLOAD_COUNT}"
echo

# File counts
echo "📊 Summary by category:"
printf "  • Sigrok captures: %d files\n" "$(find "${TEST_DATA_DIR}/logic_analyzer/sigrok" -name "*.sr" 2> /dev/null | wc -l)"
printf "  • PCAP files: %d files\n" "$(find "${TEST_DATA_DIR}/pcap" -name "*.pcap" 2> /dev/null | wc -l)"
printf "  • Tektronix WFM: %d files\n" "$(find "${TEST_DATA_DIR}/waveforms/tektronix" -name "*.wfm" 2> /dev/null | wc -l)"
printf "  • Synthetic data: %d files\n" "$(find "${TEST_DATA_DIR}/binary/synthetic" -type f 2> /dev/null | wc -l)"
printf "  • Ground truth: %d files\n" "$(find "${TEST_DATA_DIR}/ground_truth" -name "*.json" 2> /dev/null | wc -l)"
echo

# Storage usage
TOTAL_SIZE=$(du -sh "${TEST_DATA_DIR}" 2> /dev/null | cut -f1)
echo "💾 Total storage: ${TOTAL_SIZE}"
echo

echo "📖 Next steps:"
echo "  1. Review manifest: cat ${TEST_DATA_DIR}/manifest.yaml"
echo "  2. Check README: cat ${TEST_DATA_DIR}/README.md"
echo "  3. Run validation tests: uv run pytest tests/"
echo

# Cleanup temporary files
echo "🧹 Cleaning up temporary files..."
rm -rf /tmp/tm_data_types /tmp/sigrok-dumps
echo "  ✓ Cleanup complete"
echo

echo "✅ All done! Test data ready for validation."
