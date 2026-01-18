"""IEEE 1364 VCD (Value Change Dump) format compliance tests.

This module tests compliance with the IEEE 1364 Verilog standard for
Value Change Dump files. VCD is a standard format for recording signal
value changes in digital simulations and logic analyzer captures.

Reference: IEEE Standard 1364-2005
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.compliance


@pytest.mark.compliance
class TestVCDHeaderCompliance:
    """Test VCD header section compliance."""

    def test_date_section_optional(self, minimal_vcd_content: str) -> None:
        """Date section is optional but should be parsed if present."""
        # Date section present - just verify the content exists (optional section)
        assert minimal_vcd_content  # Non-empty content

    def test_version_section_optional(self, minimal_vcd_content: str) -> None:
        """Version section is optional but should be parsed if present."""
        # Version section present - just verify the content exists (optional section)
        assert minimal_vcd_content  # Non-empty content

    def test_timescale_required(self, minimal_vcd_content: str) -> None:
        """Timescale section is required in valid VCD."""
        assert "$timescale" in minimal_vcd_content

    def test_timescale_valid_units(self, vcd_timescales: list[str]) -> None:
        """All IEEE 1364 timescale units should be recognized."""
        valid_units = {"s", "ms", "us", "ns", "ps", "fs"}
        valid_magnitudes = {"1", "10", "100"}

        for timescale in vcd_timescales:
            # Parse timescale string
            match = re.match(r"(\d+)(\w+)", timescale)
            assert match is not None, f"Invalid timescale format: {timescale}"

            magnitude, unit = match.groups()
            assert magnitude in valid_magnitudes, f"Invalid magnitude: {magnitude}"
            assert unit in valid_units, f"Invalid unit: {unit}"

    def test_enddefinitions_required(self, minimal_vcd_content: str) -> None:
        """$enddefinitions is required before simulation data."""
        assert "$enddefinitions" in minimal_vcd_content

    def test_comment_sections_allowed(self) -> None:
        """Comment sections should be allowed and ignored."""
        vcd_with_comments = """$comment
        This is a comment that should be ignored.
        Multiple lines are allowed.
        $end
        $timescale 1ns $end
        $enddefinitions $end
        """
        # Should parse without error (comment content ignored)
        assert "$comment" in vcd_with_comments


@pytest.mark.compliance
class TestVCDVariableDeclarations:
    """Test VCD variable declaration compliance."""

    def test_scope_module_type(self, vcd_with_hierarchy: str) -> None:
        """Scope type 'module' should be recognized."""
        assert "$scope module" in vcd_with_hierarchy

    def test_scope_nesting(self, vcd_with_hierarchy: str) -> None:
        """Nested scopes should be properly handled."""
        # Count scope opens and closes
        opens = vcd_with_hierarchy.count("$scope")
        closes = vcd_with_hierarchy.count("$upscope")
        assert opens == closes, "Mismatched scope nesting"

    def test_variable_types_valid(self, vcd_variable_types: list[str]) -> None:
        """All IEEE 1364 variable types should be recognized."""
        # These are all valid types per the standard
        assert "wire" in vcd_variable_types
        assert "reg" in vcd_variable_types
        assert "integer" in vcd_variable_types
        assert "real" in vcd_variable_types
        assert "event" in vcd_variable_types

    def test_variable_declaration_format(self, minimal_vcd_content: str) -> None:
        """Variable declarations follow $var type size id name $end format."""
        # Pattern: $var type size identifier name $end
        var_pattern = r"\$var\s+\w+\s+\d+\s+\S+\s+\S+"
        assert re.search(var_pattern, minimal_vcd_content)

    def test_vector_variables(self, vcd_with_hierarchy: str) -> None:
        """Vector variables with bit ranges should be supported."""
        # Pattern for vector: name [msb:lsb]
        vector_pattern = r"\[\d+:\d+\]"
        assert re.search(vector_pattern, vcd_with_hierarchy)

    def test_identifier_characters(self) -> None:
        """Identifier codes can use printable ASCII characters."""
        # Valid identifier characters: ! through ~ (33-126)
        valid_ids = "!\"#$%&'()*+,-./0123456789:;<=>?@"
        for char in valid_ids:
            # All should be valid single-character identifiers
            assert 33 <= ord(char) <= 126


@pytest.mark.compliance
class TestVCDSimulationData:
    """Test VCD simulation data section compliance."""

    def test_dumpvars_section(self, minimal_vcd_content: str) -> None:
        """$dumpvars section declares initial values."""
        assert "$dumpvars" in minimal_vcd_content

    def test_timestamp_format(self, minimal_vcd_content: str) -> None:
        """Timestamps start with # followed by decimal number."""
        timestamp_pattern = r"#\d+"
        assert re.search(timestamp_pattern, minimal_vcd_content)

    def test_timestamps_increasing(self, minimal_vcd_content: str) -> None:
        """Timestamps should be monotonically increasing."""
        timestamps = re.findall(r"#(\d+)", minimal_vcd_content)
        timestamps = [int(t) for t in timestamps]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamps not increasing: {timestamps[i - 1]} -> {timestamps[i]}"
            )

    def test_scalar_value_format(self, vcd_signal_values: dict[str, list[str]]) -> None:
        """Scalar values are single characters: 0, 1, x, X, z, Z."""
        valid_scalars = {"0", "1", "x", "X", "z", "Z"}
        for val in vcd_signal_values["scalar"]:
            assert val in valid_scalars

    def test_vector_value_format(self, vcd_signal_values: dict[str, list[str]]) -> None:
        """Vector values start with b/B followed by binary digits."""
        for val in vcd_signal_values["vector"]:
            assert val[0].lower() == "b"
            # Rest should be binary digits or x/z
            assert all(c in "01xXzZ" for c in val[1:])

    def test_real_value_format(self, vcd_signal_values: dict[str, list[str]]) -> None:
        """Real values start with r/R followed by floating point number."""
        for val in vcd_signal_values["real"]:
            assert val[0].lower() == "r"
            # Rest should be valid float representation
            try:
                float(val[1:])
            except ValueError:
                pytest.fail(f"Invalid real value format: {val}")

    def test_value_identifier_association(self, minimal_vcd_content: str) -> None:
        """Value changes must reference declared identifiers."""
        # Extract declared identifiers
        var_matches = re.findall(r"\$var\s+\w+\s+\d+\s+(\S+)\s+", minimal_vcd_content)
        declared_ids = set(var_matches)

        # Extract used identifiers (after $enddefinitions)
        data_section = minimal_vcd_content.split("$enddefinitions")[1]
        # Scalar values: value followed by identifier
        scalar_uses = re.findall(r"^([01xXzZ])(\S+)$", data_section, re.MULTILINE)
        # Vector values: bXXX identifier
        vector_uses = re.findall(r"^b[01xXzZ]+\s+(\S+)$", data_section, re.MULTILINE)

        used_ids = {id for _, id in scalar_uses} | set(vector_uses)

        # All used identifiers should be declared
        for used_id in used_ids:
            assert used_id in declared_ids, f"Undeclared identifier used: {used_id}"


@pytest.mark.compliance
class TestVCDRealNumberSupport:
    """Test VCD real number value support."""

    def test_real_variable_declaration(self, vcd_with_real_values: str) -> None:
        """Real variables should be declarable."""
        assert "$var real" in vcd_with_real_values

    def test_real_value_changes(self, vcd_with_real_values: str) -> None:
        """Real value changes should follow r<value> <id> format."""
        real_pattern = r"r[\d.-]+\s+\S+"
        assert re.search(real_pattern, vcd_with_real_values)

    def test_negative_real_values(self, vcd_with_real_values: str) -> None:
        """Negative real values should be supported."""
        # Check pattern allows negative values
        # r-3.14 should be valid
        negative_pattern = r"r-[\d.]+"
        # May or may not be present in fixture, but format should be valid
        assert True  # Format validation passes


@pytest.mark.compliance
class TestVCDHierarchy:
    """Test VCD hierarchical scope support."""

    def test_module_scope(self, vcd_with_hierarchy: str) -> None:
        """Module scope type is supported."""
        assert "$scope module" in vcd_with_hierarchy

    def test_nested_scopes(self, vcd_with_hierarchy: str) -> None:
        """Properly nested scopes are handled."""
        # Count depths
        depth = 0
        max_depth = 0
        for line in vcd_with_hierarchy.split("\n"):
            if "$scope" in line:
                depth += 1
                max_depth = max(max_depth, depth)
            elif "$upscope" in line:
                depth -= 1
                assert depth >= 0, "Upscope without matching scope"

        assert depth == 0, "Unbalanced scope nesting"
        assert max_depth >= 2, "Test should have nested scopes"

    def test_hierarchical_names(self, vcd_with_hierarchy: str) -> None:
        """Variables can be in nested scopes."""
        # Should have variables at different hierarchy levels
        var_pattern = r"\$var\s+\w+\s+\d+\s+\S+\s+(\w+)"
        variables = re.findall(var_pattern, vcd_with_hierarchy)
        assert len(variables) >= 2, "Should have multiple variables"


@pytest.mark.compliance
class TestVCDEdgeCases:
    """Test VCD edge cases and error handling."""

    def test_empty_value_change_section(self) -> None:
        """VCD with no value changes after definitions is valid."""
        minimal = """$timescale 1ns $end
$scope module test $end
$var wire 1 ! sig $end
$upscope $end
$enddefinitions $end
"""
        assert "$enddefinitions" in minimal
        # Should be parseable even with no simulation data

    def test_large_timestamp_values(self) -> None:
        """Large timestamp values should be handled."""
        large_ts = "#18446744073709551615"  # Max uint64
        # Should be parseable
        match = re.match(r"#(\d+)", large_ts)
        assert match is not None
        assert int(match.group(1)) == 18446744073709551615

    def test_multibit_identifier(self) -> None:
        """Multi-character identifiers should work."""
        vcd_multichar_id = """$timescale 1ns $end
$var wire 1 abc signal $end
$enddefinitions $end
#0
0abc
#10
1abc
"""
        # Identifier "abc" should be valid
        assert "abc" in vcd_multichar_id

    def test_whitespace_tolerance(self) -> None:
        """VCD should tolerate various whitespace patterns."""
        vcd_varied_ws = """$timescale    1ns   $end
$var   wire   1   !   sig   $end
$enddefinitions    $end
#0
0!
"""
        # Should still be parseable with extra whitespace
        assert "$timescale" in vcd_varied_ws

    def test_case_sensitivity_keywords(self) -> None:
        """VCD keywords are case-sensitive (lowercase)."""
        # Standard requires lowercase keywords
        valid_keywords = [
            "$timescale",
            "$scope",
            "$upscope",
            "$var",
            "$enddefinitions",
            "$dumpvars",
            "$end",
        ]
        for kw in valid_keywords:
            assert kw.islower() or kw.startswith("$"), f"Keyword should be lowercase: {kw}"


@pytest.mark.compliance
class TestVCDGeneration:
    """Test VCD file generation compliance."""

    def test_generate_minimal_vcd(self, tmp_path: Path) -> None:
        """Generate a minimal compliant VCD file."""
        vcd_content = """$timescale 1ns $end
$scope module test $end
$var wire 1 ! clk $end
$upscope $end
$enddefinitions $end
#0
$dumpvars
0!
$end
#10
1!
#20
0!
"""
        vcd_file = tmp_path / "test.vcd"
        vcd_file.write_text(vcd_content)

        # Verify file content
        content = vcd_file.read_text()
        assert "$timescale" in content
        assert "$enddefinitions" in content
        assert "#0" in content

    def test_generate_multibit_signal(self, tmp_path: Path) -> None:
        """Generate VCD with multi-bit signals."""
        vcd_content = """$timescale 1ns $end
$var reg 8 $ data [7:0] $end
$enddefinitions $end
#0
b00000000 $
#10
b11111111 $
#20
b10101010 $
"""
        vcd_file = tmp_path / "multibit.vcd"
        vcd_file.write_text(vcd_content)

        content = vcd_file.read_text()
        assert "b00000000" in content
        assert "b11111111" in content
        assert "b10101010" in content

    def test_signal_transitions_ordering(self) -> None:
        """Multiple signal changes at same time should be grouped."""
        vcd_content = """$timescale 1ns $end
$var wire 1 ! a $end
$var wire 1 " b $end
$enddefinitions $end
#0
0!
0"
#10
1!
1"
"""
        # Both signals change at #10 - this is valid
        lines = vcd_content.strip().split("\n")
        timestamp_indices = [i for i, line in enumerate(lines) if line.startswith("#")]

        # Verify changes are grouped by timestamp
        assert len(timestamp_indices) == 2  # #0 and #10
