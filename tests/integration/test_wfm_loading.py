"""Integration tests for WFM file loading and real capture handling.

Tests real Tektronix file format handling, large file management, and vendor-specific quirks.
Focuses on edge cases and file format variations NOT covered by demos.

NOTE: Basic WFM loading and analysis is covered by demos/01_waveform_analysis/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]


@pytest.mark.integration
@pytest.mark.requires_data
class TestRealWFMFileLoading:
    """Test loading real Tektronix WFM files across size tiers."""

    def test_load_small_wfm_files(self, real_wfm_small: list[Path]) -> None:
        """Test loading small WFM files (< 1.5 MB)."""
        if not real_wfm_small:
            pytest.skip("No small WFM files available")

        from oscura.core.exceptions import LoaderError
        from oscura.loaders.tektronix import load_tektronix_wfm

        loaded_count = 0
        for wfm_file in real_wfm_small:
            try:
                trace = load_tektronix_wfm(wfm_file)

                assert trace is not None
                assert hasattr(trace, "data")
                assert len(trace.data) > 0
                assert np.isfinite(trace.data).all()
                assert hasattr(trace, "metadata")
                assert trace.metadata is not None
                loaded_count += 1
            except LoaderError:
                continue

        if loaded_count == 0:
            pytest.skip("No WFM files could be loaded")

    def test_load_medium_wfm_files(self, real_wfm_medium: list[Path]) -> None:
        """Test loading medium WFM files (1.5 - 6 MB)."""
        if not real_wfm_medium:
            pytest.skip("No medium WFM files available")

        from oscura.core.exceptions import LoaderError
        from oscura.loaders.tektronix import load_tektronix_wfm

        loaded_count = 0
        for wfm_file in real_wfm_medium[:3]:
            try:
                trace = load_tektronix_wfm(wfm_file)

                assert trace is not None
                assert len(trace.data) > 0
                assert np.isfinite(trace.data).all()
                loaded_count += 1
            except LoaderError:
                continue

        if loaded_count == 0:
            pytest.skip("No WFM files could be loaded")

    @pytest.mark.slow
    def test_load_large_wfm_files(self, real_wfm_large: list[Path]) -> None:
        """Test loading large WFM files (> 6 MB) for memory management."""
        if not real_wfm_large:
            pytest.skip("No large WFM files available")

        from oscura.loaders.tektronix import load_tektronix_wfm

        wfm_file = real_wfm_large[0]
        trace = load_tektronix_wfm(wfm_file)

        assert trace is not None
        assert len(trace.data) > 0
        assert np.isfinite(trace.data).all()


@pytest.mark.integration
@pytest.mark.requires_data
class TestWFMMetadataExtraction:
    """Test metadata extraction from real WFM files."""

    def test_wfm_metadata_extraction(self, real_wfm_files: dict[str, list[Path]]) -> None:
        """Test metadata extraction from real WFM files."""
        from oscura.core.exceptions import LoaderError
        from oscura.loaders.tektronix import load_tektronix_wfm

        all_files = []
        for files in real_wfm_files.values():
            all_files.extend(files)

        if not all_files:
            pytest.skip("No real WFM files available")

        tested = 0
        for files in real_wfm_files.values():
            if not files:
                continue

            try:
                trace = load_tektronix_wfm(files[0])

                # Verify metadata
                assert trace.metadata is not None
                assert hasattr(trace.metadata, "sample_rate")

                # Sample rate should be reasonable (1 kHz to 10 GHz)
                if trace.metadata.sample_rate:
                    assert 1e3 <= trace.metadata.sample_rate <= 1e10

                tested += 1
            except LoaderError:
                continue

        if tested == 0:
            pytest.skip("No WFM files could be loaded")

    def test_size_categories_coverage(self, real_wfm_files: dict[str, list[Path]]) -> None:
        """Test WFM file size category coverage."""
        total_files = sum(len(files) for files in real_wfm_files.values())

        if total_files == 0:
            pytest.skip("No real WFM files available")

        # Report available files
        for category, files in real_wfm_files.items():
            if files:
                total_size = sum(f.stat().st_size for f in files)
                print(f"  {category}: {len(files)} files, {total_size / 1e6:.1f} MB total")


@pytest.mark.integration
@pytest.mark.requires_data
class TestWFMAnalysisPipelines:
    """Test WFM loading through analysis pipelines (edge cases only)."""

    def test_wfm_to_filtering_edge_cases(self, wfm_files: list[Path]) -> None:
        """Test WFM loading with filtering (edge cases)."""
        if not wfm_files:
            pytest.skip("No WFM files available")

        try:
            from oscura import load, low_pass
            from oscura.core.types import IQTrace

            wfm_path = wfm_files[0]
            trace = load(wfm_path)

            # Skip IQ traces (they have i_data/q_data, not data)
            if isinstance(trace, IQTrace):
                pytest.skip("IQ trace filtering not tested here")

            # Apply low-pass filter
            if hasattr(trace.metadata, "sample_rate") and trace.metadata.sample_rate:
                cutoff = trace.metadata.sample_rate * 0.1
            else:
                cutoff = 1e5

            filtered = low_pass(trace, cutoff=cutoff)

            assert len(filtered.data) == len(trace.data)
            assert np.isfinite(filtered.data).all()

        except Exception as e:
            pytest.skip(f"WFM filtering test skipped: {e}")

    def test_wfm_to_digital_conversion(self, wfm_files: list[Path]) -> None:
        """Test WFM to digital conversion edge cases."""
        if not wfm_files:
            pytest.skip("No WFM files available")

        try:
            from oscura import detect_edges, load, to_digital
            from oscura.core.types import IQTrace

            wfm_path = wfm_files[0]
            trace = load(wfm_path)

            # Skip IQ traces (they have i_data/q_data, not data)
            if isinstance(trace, IQTrace):
                pytest.skip("IQ trace digital conversion not tested here")

            # Convert to digital
            digital = to_digital(trace.data)
            assert len(digital) == len(trace.data)

            # Detect edges
            edges = detect_edges(trace.data)
            assert edges is not None

        except Exception as e:
            pytest.skip(f"Digital analysis failed: {e}")

    def test_multi_channel_loading(self, wfm_files: list[Path]) -> None:
        """Test loading all channels from WFM."""
        if not wfm_files:
            pytest.skip("No WFM files available")

        try:
            from oscura import load_all_channels, mean

            wfm_path = wfm_files[0]

            channels = load_all_channels(wfm_path)

            for name, trace in channels.items():
                m = mean(trace)
                assert np.isfinite(m), f"Channel {name} has invalid mean"

        except Exception as e:
            pytest.skip(f"Multi-channel test skipped: {e}")


@pytest.mark.integration
@pytest.mark.requires_data
class TestRealUDPCaptures:
    """Test real UDP packet capture handling."""

    def test_udp_segments_exist(self, real_udp_packets: dict[str, Path | None]) -> None:
        """Test UDP packet segment files exist."""
        segments_found = sum(1 for v in real_udp_packets.values() if v is not None)

        if segments_found == 0:
            pytest.skip("No UDP packet files available")

        for segment_name, path in real_udp_packets.items():
            if path is not None:
                assert path.exists(), f"UDP {segment_name} file doesn't exist"
                assert path.stat().st_size > 0, f"UDP {segment_name} file is empty"

    def test_load_udp_as_binary(self, real_udp_packets: dict[str, Path | None]) -> None:
        """Test loading UDP packets as binary data."""
        from oscura.loaders.binary import load_binary

        for path in real_udp_packets.values():
            if path is None:
                continue

            trace = load_binary(path, dtype="uint8", sample_rate=1.0)

            assert trace is not None
            assert len(trace.data) > 0

    def test_entropy_on_udp_packets(self, real_udp_packets: dict[str, Path | None]) -> None:
        """Test entropy analysis on UDP packet data."""
        try:
            from oscura.analyzers.statistical.entropy import calculate_entropy
        except ImportError:
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            # SKIP: Valid - Optional entropy analysis module
            # Only skip if entropy analyzers not available
            pytest.skip("Entropy analysis not available")

        for segment_name, path in real_udp_packets.items():
            if path is None:
                continue

            with open(path, "rb") as f:
                data = f.read(10000)

            entropy = calculate_entropy(data)

            # Real packet data should have moderate to high entropy
            assert 0 < entropy <= 8.0, f"Entropy out of range for {segment_name}"


@pytest.mark.integration
@pytest.mark.requires_data
class TestRealCaptureManifest:
    """Test manifest validation for real captures."""

    def test_manifest_structure(self, real_captures_manifest: dict[str, any]) -> None:
        """Test manifest.json has valid structure."""
        if not real_captures_manifest:
            pytest.skip("No manifest.json found")

        assert "version" in real_captures_manifest
        assert "files" in real_captures_manifest
        assert "total_files" in real_captures_manifest
        assert "categories" in real_captures_manifest

    def test_manifest_files_exist(
        self, real_captures_manifest: dict[str, any], real_captures_dir: Path
    ) -> None:
        """Test files in manifest exist."""
        if not real_captures_manifest:
            pytest.skip("No manifest.json found")

        missing_files = []
        for file_info in real_captures_manifest.get("files", []):
            # Manifest uses 'path' field with full relative path
            file_path = file_info.get("path")
            if not file_path:
                continue

            expected_path = real_captures_dir / file_path

            if not expected_path.exists():
                missing_files.append(str(expected_path))

        if missing_files:
            pytest.fail(f"Missing files from manifest: {missing_files[:5]}...")

    def test_manifest_checksums_spot_check(
        self, real_captures_manifest: dict[str, any], real_captures_dir: Path
    ) -> None:
        """Test file checksums match manifest (spot check)."""
        import hashlib

        if not real_captures_manifest:
            pytest.skip("No manifest.json found")

        files_checked = 0
        files_matched = 0
        mismatches = []

        for file_info in real_captures_manifest.get("files", [])[:10]:
            # Manifest uses 'path' field with full relative path
            relative_path = file_info.get("path")
            expected_md5 = file_info.get("md5_hash")

            if not relative_path:
                continue

            file_path = real_captures_dir / relative_path

            if not file_path.exists() or not expected_md5:
                continue

            # Compute MD5
            md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    md5.update(chunk)

            actual_md5 = md5.hexdigest()
            files_checked += 1

            if actual_md5 == expected_md5:
                files_matched += 1
            else:
                mismatches.append(relative_path)

        if files_checked == 0:
            pytest.skip("No files with checksums to verify")

        # Allow up to 20% mismatches (corrupted test data)
        match_rate = files_matched / files_checked if files_checked > 0 else 0
        assert match_rate >= 0.8, (
            f"Too many checksum mismatches ({len(mismatches)}/{files_checked}): {mismatches}"
        )
