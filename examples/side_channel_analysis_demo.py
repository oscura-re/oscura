#!/usr/bin/env python3
"""Side-Channel Analysis Demo - Differential Power Analysis (DPA).

This demo shows how to use the DPA framework to recover cryptographic keys
from power consumption traces using CPA (Correlation Power Analysis).

# SKIP_VALIDATION - This demo takes >60 seconds due to cryptographic computations

Example:
    python examples/side_channel_analysis_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oscura.side_channel.dpa import DPAAnalyzer, PowerTrace


def generate_simulated_aes_traces(
    num_traces: int = 200,
    num_samples: int = 1000,
    true_key: bytes = b"\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c",
    noise_level: float = 0.5,
) -> list[PowerTrace]:
    """Generate simulated AES power traces with Hamming weight leakage.

    Simulates a real AES implementation that leaks power consumption
    proportional to the Hamming weight of the S-box output.

    Args:
        num_traces: Number of power traces to generate.
        num_samples: Number of time samples per trace.
        true_key: 16-byte AES-128 key.
        noise_level: Gaussian noise standard deviation.

    Returns:
        List of PowerTrace objects.
    """
    print(f"Generating {num_traces} simulated AES power traces...")
    print(f"True key: {true_key.hex()}")
    print(f"Noise level: {noise_level}")

    traces = []
    rng = np.random.RandomState(42)  # Deterministic for reproducibility

    # AES S-box (same as in dpa.py)
    from oscura.side_channel.dpa import AES_SBOX

    for i in range(num_traces):
        # Random plaintext (normally from test vectors)
        plaintext = bytes(rng.randint(0, 256, 16))

        # Generate baseline power trace (noise)
        power = rng.randn(num_samples) * noise_level

        # Add realistic leakage for each AES round
        # We'll add leakage at sample points 100-115 (one per byte)
        for byte_idx in range(16):
            # Calculate S-box output (first round of AES)
            intermediate = AES_SBOX[plaintext[byte_idx] ^ true_key[byte_idx]]

            # Hamming weight (number of 1 bits)
            hw = bin(intermediate).count("1")

            # Add leakage at specific time point
            leakage_point = 100 + byte_idx * 5
            if leakage_point < num_samples:
                # Strong signal proportional to Hamming weight
                power[leakage_point] += hw * 2.0

        traces.append(
            PowerTrace(
                timestamp=np.linspace(0, 1e-6, num_samples),  # 1 microsecond total
                power=power,
                plaintext=plaintext,
                metadata={"trace_id": i, "device": "AES_target", "temperature": 25.0},
            )
        )

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_traces} traces...")

    print("✓ Trace generation complete\n")
    return traces


def main() -> None:
    """Run side-channel analysis demo."""
    print("=" * 70)
    print("SIDE-CHANNEL ANALYSIS DEMO - Differential Power Analysis")
    print("=" * 70)
    print()

    # Step 1: Generate simulated traces
    true_key = b"\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c"
    traces = generate_simulated_aes_traces(
        num_traces=50,  # Reduced from 200 for faster execution in tests
        num_samples=500,  # Reduced from 1000 for faster execution in tests
        true_key=true_key,
        noise_level=0.3,  # Low noise for good attack success
    )

    # Step 2: Create CPA analyzer
    print("Initializing CPA analyzer...")
    analyzer = DPAAnalyzer(
        attack_type="cpa",
        leakage_model="hamming_weight",
    )
    print("✓ Analyzer ready\n")

    # Step 3: Perform attack on all 16 key bytes
    print("Performing CPA attack on all 16 AES key bytes...")
    print("-" * 70)

    recovered_key = bytearray(16)
    successful_bytes = 0

    for byte_idx in range(16):
        print(f"\nAttacking byte {byte_idx}...")

        result = analyzer.perform_attack(
            traces=traces,
            target_byte=byte_idx,
            algorithm="aes128",
        )

        recovered_key[byte_idx] = result.recovered_key[0]

        # Check if correct
        correct = result.recovered_key[0] == true_key[byte_idx]
        status = "✓ CORRECT" if correct else "✗ WRONG"

        print(f"  True key byte:      0x{true_key[byte_idx]:02X}")
        print(f"  Recovered key byte: 0x{result.recovered_key[0]:02X}")
        print(f"  Confidence:         {result.confidence:.2%}")
        print(f"  Status:             {status}")

        if correct:
            successful_bytes += 1

    print("\n" + "=" * 70)
    print("ATTACK RESULTS")
    print("=" * 70)
    print(f"True key:      {true_key.hex()}")
    print(f"Recovered key: {bytes(recovered_key).hex()}")
    print(f"Success rate:  {successful_bytes}/16 bytes ({successful_bytes / 16:.0%})")
    print()

    # Step 4: Visualize attack for first byte
    print("Generating visualization for byte 0...")
    result_byte0 = analyzer.perform_attack(traces, target_byte=0)

    output_dir = Path("examples/outputs")
    output_dir.mkdir(exist_ok=True)

    viz_path = output_dir / "dpa_attack_byte0.png"
    analyzer.visualize_attack(result_byte0, viz_path)
    print(f"✓ Visualization saved to {viz_path}")

    # Step 5: Export results
    json_path = output_dir / "dpa_results_byte0.json"
    analyzer.export_results(result_byte0, json_path)
    print(f"✓ Results exported to {json_path}")

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Try different noise levels (increase noise_level parameter)")
    print("  2. Reduce number of traces to see attack degradation")
    print("  3. Try DPA attack: DPAAnalyzer(attack_type='dpa')")
    print("  4. Implement your own leakage model")
    print()


if __name__ == "__main__":
    main()
