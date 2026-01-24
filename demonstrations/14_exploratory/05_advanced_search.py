"""Advanced Pattern Search: Binary regex, multi-pattern, and fuzzy matching

Demonstrates:
- oscura.analyzers.patterns.matching.binary_regex_search() - Regex-style patterns
- oscura.analyzers.patterns.matching.multi_pattern_search() - Multi-pattern (Aho-Corasick)
- oscura.analyzers.patterns.matching.fuzzy_search() - Approximate matching
- oscura.analyzers.patterns.matching.find_similar_sequences() - Similarity search
- Pattern discovery in unknown protocols
- Optimized search strategies

IEEE Standards: N/A
Related Demos:
- 14_exploratory/02_fuzzy_matching.py
- 06_reverse_engineering/01_unknown_protocol.py

This demonstration shows how to efficiently search binary data for patterns using
regex-like syntax, multi-pattern algorithms, and fuzzy matching for protocols
with variations or errors.

This is a P1 HIGH feature - demonstrates advanced search capability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.analyzers.patterns.matching import (
    BinaryRegex,
    find_similar_sequences,
    fuzzy_search,
    multi_pattern_search,
)


class AdvancedSearchDemo(BaseDemo):
    """Demonstrate advanced pattern search algorithms."""

    def __init__(self) -> None:
        """Initialize advanced search demonstration."""
        super().__init__(
            name="advanced_search",
            description="Advanced pattern search: regex, multi-pattern, fuzzy matching",
            capabilities=[
                "oscura.analyzers.patterns.matching.binary_regex_search",
                "oscura.analyzers.patterns.matching.multi_pattern_search",
                "oscura.analyzers.patterns.matching.fuzzy_search",
                "oscura.analyzers.patterns.matching.find_similar_sequences",
            ],
            related_demos=[
                "14_exploratory/02_fuzzy_matching.py",
                "06_reverse_engineering/01_unknown_protocol.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test data with known patterns.

        Creates binary data with:
        - Known patterns (headers, delimiters)
        - Variations of patterns (for fuzzy matching)
        - Multiple interleaved patterns
        - Similar sequences

        Returns:
            Dictionary with test data and ground truth
        """
        self.info("Generating binary test data with patterns...")

        # ===== Pattern 1: Protocol Headers =====
        HEADER_MAIN = b"\xaa\x55"
        HEADER_ALT = b"\xaa\x54"  # Similar (1 bit different)
        DELIMITER = b"\xff\x00"

        # ===== Pattern 2: Message Types =====
        MSG_REQUEST = b"\x01\x10"
        MSG_RESPONSE = b"\x01\x20"
        MSG_ERROR = b"\x01\xf0"

        # ===== Pattern 3: Data Patterns =====
        DATA_COUNTER = bytes(range(16))  # Incrementing sequence
        DATA_TOGGLE = b"\x55\xaa" * 4  # Alternating pattern

        # Build test data stream
        data = bytearray()

        # Add several packets with variations
        self.subsection("Test Data Structure")

        # Packet 1: Clean pattern
        self.info("Packet 1: Clean header + request")
        data.extend(HEADER_MAIN)
        data.extend(MSG_REQUEST)
        data.extend(DATA_COUNTER)
        data.extend(DELIMITER)

        # Packet 2: Alternative header (fuzzy match target)
        self.info("Packet 2: Alternative header + response")
        data.extend(HEADER_ALT)
        data.extend(MSG_RESPONSE)
        data.extend(DATA_TOGGLE)
        data.extend(DELIMITER)

        # Random noise
        data.extend(np.random.randint(0, 256, 10, dtype=np.uint8).tobytes())

        # Packet 3: Another clean packet
        self.info("Packet 3: Clean header + error")
        data.extend(HEADER_MAIN)
        data.extend(MSG_ERROR)
        data.extend(DATA_COUNTER)
        data.extend(DELIMITER)

        # Add similar sequences for similarity search
        data.extend(b"\xaa\x56\x01\x11")  # Similar to header+request

        # Random noise
        data.extend(np.random.randint(0, 256, 8, dtype=np.uint8).tobytes())

        # Packet 4: Clean pattern
        self.info("Packet 4: Clean header + request")
        data.extend(HEADER_MAIN)
        data.extend(MSG_REQUEST)
        data.extend(DATA_TOGGLE)
        data.extend(DELIMITER)

        # More similar sequences
        data.extend(b"\xaa\x55\x01\x1f")  # Similar to header+request

        self.result("Total data size", len(data), "bytes")
        self.result("Known header occurrences", 3)
        self.result("Known fuzzy matches", 2)

        return {
            "data": bytes(data),
            "patterns": {
                "HEADER_MAIN": HEADER_MAIN,
                "HEADER_ALT": HEADER_ALT,
                "DELIMITER": DELIMITER,
                "MSG_REQUEST": MSG_REQUEST,
                "MSG_RESPONSE": MSG_RESPONSE,
                "MSG_ERROR": MSG_ERROR,
                "DATA_COUNTER": DATA_COUNTER,
                "DATA_TOGGLE": DATA_TOGGLE,
            },
            "expected_exact": 3,  # HEADER_MAIN
            "expected_fuzzy": 2,  # HEADER_ALT + similar sequences
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute advanced search demonstration."""
        results: dict[str, Any] = {}

        test_data = data["data"]
        patterns = data["patterns"]

        # ===== Phase 1: Exact Pattern Search =====
        self.section("Part 1: Exact Pattern Search")

        self.subsection("Simple Binary Search")
        header_matches = test_data.count(patterns["HEADER_MAIN"])
        self.result("HEADER_MAIN occurrences", header_matches)

        results["exact_header_count"] = header_matches

        # ===== Phase 2: Binary Regex Search =====
        self.section("Part 2: Binary Regex Pattern Matching")

        self.subsection("Wildcard Pattern: Header + ?? (Any 2 Bytes)")
        # Pattern: \xAA\x55 followed by any 2 bytes using ?? wildcard
        wildcard_pattern = "\\xaa\\x55??"
        wildcard_regex = BinaryRegex(pattern=wildcard_pattern, name="wildcard")
        wildcard_matches = wildcard_regex.findall(test_data)

        self.result("Wildcard matches found", len(wildcard_matches))

        for i, match in enumerate(wildcard_matches[:5], 1):
            self.info(
                f"  Match {i}: offset={match.offset:3d}, data={match.matched_data.hex().upper()}"
            )

        results["regex_matches"] = len(wildcard_matches)

        # Multi-byte wildcard pattern
        self.subsection("Multi-Wildcard Pattern: Header + ???? + Delimiter")
        multi_wildcard_pattern = "\\xaa\\x55????\\xff\\x00"
        multi_regex = BinaryRegex(pattern=multi_wildcard_pattern, name="multi_wildcard")
        multi_matches = multi_regex.findall(test_data)

        self.result("Multi-wildcard matches", len(multi_matches))

        results["wildcard_matches"] = len(multi_matches)

        # ===== Phase 3: Multi-Pattern Search (Aho-Corasick) =====
        self.section("Part 3: Multi-Pattern Search (Aho-Corasick)")

        self.subsection("Searching for Multiple Patterns Simultaneously")

        # Define multiple patterns to search
        search_patterns = {
            "HEADER_MAIN": patterns["HEADER_MAIN"],
            "DELIMITER": patterns["DELIMITER"],
            "MSG_REQUEST": patterns["MSG_REQUEST"],
            "MSG_RESPONSE": patterns["MSG_RESPONSE"],
            "MSG_ERROR": patterns["MSG_ERROR"],
        }

        multi_results = multi_pattern_search(test_data, search_patterns)

        self.info("Multi-pattern search results:")
        total_multi_matches = 0
        for pattern_name, matches in multi_results.items():
            self.info(f"  {pattern_name:15s}: {len(matches)} occurrences")
            total_multi_matches += len(matches)

        self.result("Total patterns found", total_multi_matches)

        results["multi_pattern_total"] = total_multi_matches
        results["multi_pattern_details"] = {
            name: len(matches) for name, matches in multi_results.items()
        }

        # Show efficiency comparison
        self.subsection("Efficiency Analysis")
        self.info("Multi-pattern search (Aho-Corasick):")
        self.info(f"  - Searched for {len(search_patterns)} patterns")
        self.info(f"  - Single pass through {len(test_data)} bytes")
        self.info("  - Time complexity: O(n + m + z)")
        self.info(f"    n={len(test_data)} (text length)")
        self.info(f"    m={sum(len(p) for p in search_patterns.values())} (total pattern length)")
        self.info(f"    z={total_multi_matches} (matches found)")

        # ===== Phase 4: Fuzzy Pattern Matching =====
        self.section("Part 4: Fuzzy Pattern Matching (Edit Distance)")

        self.subsection("Exact vs Fuzzy Matching")

        # Search for exact header
        exact_pattern = patterns["HEADER_MAIN"]
        exact_count = test_data.count(exact_pattern)
        self.info(f"Exact matches for {exact_pattern.hex().upper()}: {exact_count}")

        # Fuzzy search (allow 1 byte difference)
        self.subsection("Fuzzy Search (max_distance=1)")
        fuzzy_matches = fuzzy_search(
            test_data,
            exact_pattern,
            max_distance=1,
            name="fuzzy_header",
        )

        self.result("Fuzzy matches found", len(fuzzy_matches))

        for i, match in enumerate(fuzzy_matches[:10], 1):
            self.info(
                f"  Match {i}: offset={match.offset:3d}, "
                f"data={match.matched_data.hex().upper()}, "
                f"similarity={match.similarity:.2f}, "
                f"edit_distance={match.edit_distance}"
            )

        results["fuzzy_matches"] = len(fuzzy_matches)
        results["fuzzy_unique"] = len(fuzzy_matches) - exact_count

        self.info(f"\nFuzzy matching found {results['fuzzy_unique']} additional near-matches!")

        # ===== Phase 5: Similarity-Based Search =====
        self.section("Part 5: Similarity-Based Sequence Discovery")

        self.subsection("Finding Similar Sequences (Automated)")

        # Find all similar sequences in data
        similar_seqs = find_similar_sequences(
            test_data,
            min_length=4,
            max_distance=2,
        )

        self.result("Similar sequence pairs", len(similar_seqs))

        if similar_seqs:
            self.info("\nTop similar sequences:")
            for i, (offset1, offset2, similarity) in enumerate(similar_seqs[:5], 1):
                seq1 = test_data[offset1 : offset1 + 4].hex().upper()
                seq2 = test_data[offset2 : offset2 + 4].hex().upper()
                self.info(
                    f"  {i}. Offset {offset1:3d} ({seq1}) <-> "
                    f"Offset {offset2:3d} ({seq2}), "
                    f"similarity={similarity:.2f}"
                )

        results["similar_sequences"] = len(similar_seqs)

        # ===== Phase 6: Practical Workflow - Unknown Protocol Analysis =====
        self.section("Part 6: Practical Workflow - Unknown Protocol Pattern Discovery")

        self.subsection("Step 1: Find Repeated Patterns")

        # Use Aho-Corasick to find all 2-byte sequences
        byte_pairs: dict[bytes, int] = {}
        for i in range(len(test_data) - 1):
            pair = test_data[i : i + 2]
            byte_pairs[pair] = byte_pairs.get(pair, 0) + 1

        # Find most frequent pairs
        frequent_pairs = sorted(byte_pairs.items(), key=lambda x: x[1], reverse=True)[:5]

        self.info("Most frequent 2-byte sequences:")
        for pair, count in frequent_pairs:
            self.info(f"  {pair.hex().upper()}: {count} occurrences")

        self.subsection("Step 2: Fuzzy Search for Variations")

        # Take most frequent pattern and find variations
        most_frequent_pattern = frequent_pairs[0][0]
        variations = fuzzy_search(test_data, most_frequent_pattern, max_distance=1)

        self.info(f"\nVariations of {most_frequent_pattern.hex().upper()}:")
        variation_set = set()
        for match in variations:
            variation_set.add(match.matched_data)

        for i, var in enumerate(sorted(variation_set), 1):
            count = sum(1 for m in variations if m.matched_data == var)
            self.info(f"  {i}. {var.hex().upper()}: {count} occurrences")

        self.subsection("Step 3: Identify Delimiters")

        # Look for patterns that appear at regular intervals
        delimiter_candidates = {k: v for k, v in byte_pairs.items() if v >= 3}
        self.info(f"\nDelimiter candidates (>=3 occurrences): {len(delimiter_candidates)}")

        results["workflow_completed"] = True
        results["frequent_patterns"] = len(frequent_pairs)
        results["pattern_variations"] = len(variation_set)

        # ===== Summary =====
        self.section("Advanced Search Summary")

        self.result("Exact matches", results.get("exact_header_count", 0))
        self.result("Wildcard matches", results.get("regex_matches", 0))
        self.result("Multi-pattern total", results.get("multi_pattern_total", 0))
        self.result("Fuzzy matches", results.get("fuzzy_matches", 0))
        self.result("Similar sequences", results.get("similar_sequences", 0))

        self.success("All advanced search techniques demonstrated!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate search results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        all_valid = True

        # Validate exact search
        self.subsection("Exact Search Validation")
        if results.get("exact_header_count", 0) >= 3:
            self.success(
                f"Found {results['exact_header_count']} exact header matches (expected >=3)"
            )
        else:
            self.error(f"Exact header count: {results.get('exact_header_count', 0)} (expected >=3)")
            all_valid = False

        # Validate wildcard/regex search
        self.subsection("Wildcard Search Validation")
        if results.get("regex_matches", 0) >= 2:
            self.success(f"Found {results['regex_matches']} wildcard pattern matches")
        else:
            self.error(f"Wildcard matches: {results.get('regex_matches', 0)} (expected >=2)")
            all_valid = False

        # Validate multi-pattern search
        self.subsection("Multi-Pattern Search Validation")
        if results.get("multi_pattern_total", 0) >= 10:
            self.success(f"Found {results['multi_pattern_total']} total multi-pattern matches")

            # Check individual patterns
            details = results.get("multi_pattern_details", {})
            if details.get("HEADER_MAIN", 0) >= 3:
                self.success(f"  HEADER_MAIN: {details['HEADER_MAIN']} matches")
            if details.get("DELIMITER", 0) >= 3:
                self.success(f"  DELIMITER: {details['DELIMITER']} matches")
        else:
            self.error(
                f"Multi-pattern total: {results.get('multi_pattern_total', 0)} (expected >=10)"
            )
            all_valid = False

        # Validate fuzzy matching
        self.subsection("Fuzzy Matching Validation")
        if results.get("fuzzy_matches", 0) > results.get("exact_header_count", 0):
            additional = results.get("fuzzy_unique", 0)
            self.success(
                f"Fuzzy matching found {additional} additional near-matches "
                f"(total: {results['fuzzy_matches']})"
            )
        else:
            self.warning("Fuzzy matching did not find additional near-matches")

        # Validate similarity search
        self.subsection("Similarity Search Validation")
        if results.get("similar_sequences", 0) > 0:
            self.success(f"Found {results['similar_sequences']} similar sequence pairs")
        else:
            self.warning("No similar sequences found")

        # Validate workflow
        self.subsection("Workflow Validation")
        if results.get("workflow_completed", False):
            self.success("Protocol discovery workflow completed")
            if results.get("frequent_patterns", 0) >= 3:
                self.success(f"  Found {results['frequent_patterns']} frequent patterns")
            if results.get("pattern_variations", 0) >= 2:
                self.success(f"  Found {results['pattern_variations']} pattern variations")
        else:
            self.error("Protocol discovery workflow incomplete")
            all_valid = False

        # Overall validation
        if all_valid:
            self.success("\nAll advanced search validations passed!")
            self.info("\nKey takeaways:")
            self.info("  - Binary regex enables flexible pattern matching")
            self.info("  - Multi-pattern search (Aho-Corasick) is efficient for multiple patterns")
            self.info("  - Fuzzy matching handles protocol variations and errors")
            self.info("  - Similarity search discovers related patterns automatically")
            self.info("\nNext steps:")
            self.info("  - Apply to real protocol captures")
            self.info("  - Combine with statistical analysis (entropy, n-grams)")
            self.info("  - Try 06_reverse_engineering demos for complete workflows")
        else:
            self.error("\nSome advanced search validations failed")

        return all_valid


if __name__ == "__main__":
    demo = AdvancedSearchDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
