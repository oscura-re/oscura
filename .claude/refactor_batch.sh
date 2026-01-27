#!/bin/bash
# Automated refactoring for demonstration files

set -e

FILES=(
  "demonstrations/03_protocol_decoding/i2s.py"
  "demonstrations/03_protocol_decoding/jtag.py"
  "demonstrations/03_protocol_decoding/manchester.py"
  "demonstrations/03_protocol_decoding/onewire.py"
  "demonstrations/03_protocol_decoding/protocol_comprehensive.py"
  "demonstrations/03_protocol_decoding/swd.py"
  "demonstrations/03_protocol_decoding/udp_analysis.py"
  "demonstrations/03_protocol_decoding/usb.py"
  "demonstrations/04_advanced_analysis/jitter_bathtub.py"
  "demonstrations/04_advanced_analysis/jitter_ddj_dcd.py"
  "demonstrations/04_advanced_analysis/power_dcdc.py"
  "demonstrations/04_advanced_analysis/power_ripple.py"
  "demonstrations/04_advanced_analysis/signal_integrity_sparams.py"
  "demonstrations/04_advanced_analysis/signal_integrity_tdr.py"
  "demonstrations/04_advanced_analysis/signal_integrity_timing.py"
  "demonstrations/05_domain_specific/automotive_comprehensive.py"
  "demonstrations/05_domain_specific/automotive_flexray.py"
  "demonstrations/05_domain_specific/automotive_lin.py"
  "demonstrations/05_domain_specific/emc_comprehensive.py"
  "demonstrations/05_domain_specific/timing_ieee181.py"
  "demonstrations/06_reverse_engineering/crc_reverse.py"
  "demonstrations/06_reverse_engineering/inference_active_learning.py"
  "demonstrations/06_reverse_engineering/inference_bayesian.py"
  "demonstrations/06_reverse_engineering/inference_dsl.py"
  "demonstrations/06_reverse_engineering/re_comprehensive.py"
  "demonstrations/06_reverse_engineering/state_machine_learning.py"
  "demonstrations/06_reverse_engineering/wireshark_dissector.py"
  "demonstrations/11_integration/04_configuration_files.py"
  "demonstrations/16_complete_workflows/automotive_workflow.py"
  "demonstrations/16_complete_workflows/network_workflow.py"
  "demonstrations/16_complete_workflows/unknown_signal_workflow.py"
)

for file in "${FILES[@]}"; do
  echo "Processing $file..."

  if [[ ! -f "$file" ]]; then
    echo "  ERROR: File not found"
    continue
  fi

  # Step 1: Rename methods
  sed -i 's/def generate_data(self) -> None:/def generate_test_data(self) -> dict:/' "$file"
  sed -i 's/def run_analysis(self) -> None:/def run_demonstration(self, data: dict) -> dict:/' "$file"
  sed -i 's/def validate_results(self, suite: ValidationSuite) -> None:/def validate(self, results: dict) -> bool:/' "$file"

  echo "  ✓ Renamed methods"
done

echo "Done! Manual fixups still needed for each file."
