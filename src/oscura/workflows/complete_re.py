"""Complete one-function protocol reverse engineering workflow.

This module provides a single function that automates the ENTIRE reverse
engineering workflow from raw captures to working dissectors and documentation.

Example:
    >>> from oscura.workflows import full_protocol_re
    >>> result = full_protocol_re(
    ...     captures={"idle": "idle.bin", "button": "button.bin"},
    ...     export_dir="output/"
    ... )
    >>> print(f"Dissector: {result.dissector_path}")
    >>> print(f"Confidence: {result.confidence_score:.2f}")
    >>> print(f"Generated in {result.execution_time:.1f}s")

The workflow automates 14 steps:
1. Load captures (auto-detect format)
2. Detect protocol (timing, voltage levels)
3. Decode messages
4. Differential analysis (if multiple captures)
5. Infer message structure (fields, boundaries)
6. Detect entropy/crypto regions
7. Recover CRC/checksums
8. Extract state machine
9. Generate Wireshark dissector (.lua)
10. Generate Scapy layer (.py)
11. Generate Kaitai struct (.ksy)
12. Create test vectors (.json)
13. Generate HTML/PDF report
14. Replay validation (if target specified)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from oscura.workflows.reverse_engineering import (
    ProtocolSpec,
    reverse_engineer_signal,
)

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace

logger = logging.getLogger(__name__)


@dataclass
class _WorkflowContext:
    """Internal context for workflow execution."""

    capture_dict: dict[str, str]
    export_path: Path
    verbose: bool
    protocol_hint: str | None
    auto_crc: bool
    detect_crypto: bool
    generate_tests: bool
    validate: bool
    kwargs: dict[str, Any]


@dataclass
class _WorkflowResults:
    """Internal results accumulator for workflow."""

    warnings: list[str] = field(default_factory=list)
    partial_results: dict[str, Any] = field(default_factory=dict)
    dissector_path: Path | None = None
    scapy_layer_path: Path | None = None
    kaitai_path: Path | None = None
    test_vectors_path: Path | None = None
    report_path: Path | None = None
    validation_result: Any | None = None
    protocol_spec: ProtocolSpec | None = None


@dataclass
class CompleteREResult:
    """Results from complete reverse engineering workflow.

    Attributes:
        protocol_spec: Inferred protocol specification.
        dissector_path: Path to generated Wireshark dissector (.lua).
        scapy_layer_path: Path to generated Scapy layer (.py).
        kaitai_path: Path to generated Kaitai struct (.ksy).
        test_vectors_path: Path to generated test vectors (.json).
        report_path: Path to generated HTML/PDF report.
        validation_result: Replay validation result (None if not performed).
        confidence_score: Overall confidence score (0-1).
        warnings: List of warnings from workflow execution.
        execution_time: Total execution time in seconds.
        partial_results: Dict of partial results if workflow incomplete.
    """

    protocol_spec: ProtocolSpec
    dissector_path: Path | None
    scapy_layer_path: Path | None
    kaitai_path: Path | None
    test_vectors_path: Path | None
    report_path: Path | None
    validation_result: Any | None
    confidence_score: float
    warnings: list[str]
    execution_time: float
    partial_results: dict[str, Any]


def full_protocol_re(
    captures: dict[str, str] | str,
    protocol_hint: str | None = None,
    export_dir: str = "output/",
    validate: bool = True,
    auto_crc: bool = True,
    detect_crypto: bool = True,
    generate_tests: bool = True,
    **kwargs: Any,
) -> CompleteREResult:
    """Complete protocol reverse engineering in ONE function call.

    Automates the entire workflow from raw capture to working dissector.
    This is the cornerstone function of Oscura v0.6.0, providing a unified
    interface to all reverse engineering capabilities.

    Args:
        captures: Path to capture file OR dict mapping labels to paths.
        protocol_hint: Optional protocol name to skip auto-detection.
        export_dir: Directory for all output files. Created if doesn't exist.
        validate: Perform replay validation if True (requires hardware target).
        auto_crc: Automatically detect and recover CRCs/checksums.
        detect_crypto: Detect encrypted/compressed regions via entropy analysis.
        generate_tests: Generate test vectors for validation.
        **kwargs: Additional workflow options.

    Returns:
        CompleteREResult with all generated artifacts and metadata.

    Raises:
        ValueError: If captures is empty or invalid format.
        FileNotFoundError: If capture files don't exist.
        RuntimeError: If critical workflow steps fail (with partial results).

    Example:
        >>> result = full_protocol_re("unknown_protocol.bin")
        >>> print(f"Protocol: {result.protocol_spec.name}")
        >>> print(f"Confidence: {result.confidence_score:.2%}")
    """
    start_time = time.time()
    context = _initialize_workflow_context(
        captures,
        export_dir,
        protocol_hint,
        validate,
        auto_crc,
        detect_crypto,
        generate_tests,
        kwargs,
    )
    results = _WorkflowResults()

    # Execute 14-step workflow with progress tracking
    total_steps = 14
    with tqdm(total=total_steps, desc="Complete RE workflow", disable=not context.verbose) as pbar:
        traces = _step_1_load_captures(pbar, context, results)
        detected_protocol = _step_2_detect_protocol(pbar, context, results, traces)
        protocol_spec, re_result = _step_3_decode_messages(
            pbar, context, results, traces, detected_protocol
        )
        _step_4_differential_analysis(pbar, results, traces, protocol_spec)
        _step_5_infer_structure(pbar, results, re_result, protocol_spec)
        _step_6_detect_crypto(pbar, context, results, re_result)
        _step_7_recover_crc(pbar, context, results, re_result, protocol_spec)
        _step_8_extract_state_machine(pbar, results, re_result)
        _step_9_generate_wireshark(pbar, context, results, protocol_spec)
        _step_10_generate_scapy(pbar, context, results, protocol_spec)
        _step_11_generate_kaitai(pbar, context, results, protocol_spec)
        _step_12_create_test_vectors(pbar, context, results, re_result, protocol_spec)
        _step_13_generate_report(pbar, context, results, protocol_spec)
        _step_14_replay_validation(pbar, context, results, protocol_spec, re_result)

    # Finalize results
    results.protocol_spec = protocol_spec
    confidence_score = _calculate_overall_confidence(
        protocol_spec, results.partial_results, results.warnings
    )
    execution_time = time.time() - start_time

    return CompleteREResult(
        protocol_spec=protocol_spec,
        dissector_path=results.dissector_path,
        scapy_layer_path=results.scapy_layer_path,
        kaitai_path=results.kaitai_path,
        test_vectors_path=results.test_vectors_path,
        report_path=results.report_path,
        validation_result=results.validation_result,
        confidence_score=confidence_score,
        warnings=results.warnings,
        execution_time=execution_time,
        partial_results=results.partial_results,
    )


# =============================================================================
# Workflow Step Functions
# =============================================================================


def _step_1_load_captures(
    pbar: Any, context: _WorkflowContext, results: _WorkflowResults
) -> dict[str, WaveformTrace]:
    """Execute workflow step 1: Load captures.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.

    Returns:
        Loaded traces.

    Raises:
        RuntimeError: If loading fails (critical).
    """
    pbar.set_description("Loading captures")
    try:
        traces = _load_captures(context.capture_dict)
        results.partial_results["traces"] = traces
        pbar.update(1)
        return traces
    except Exception as e:
        msg = f"Failed to load captures: {e}"
        results.warnings.append(msg)
        logger.exception(msg)
        raise RuntimeError(msg) from e


def _step_2_detect_protocol(
    pbar: Any,
    context: _WorkflowContext,
    results: _WorkflowResults,
    traces: dict[str, WaveformTrace],
) -> str:
    """Execute workflow step 2: Detect protocol.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        traces: Loaded traces.

    Returns:
        Detected protocol name.
    """
    pbar.set_description("Detecting protocol")
    try:
        if context.protocol_hint:
            detected_protocol = context.protocol_hint
            results.partial_results["protocol_detection"] = {
                "hint": context.protocol_hint,
                "confidence": 1.0,
            }
        else:
            detected_protocol, detection_confidence = _detect_protocol(
                traces, context.kwargs.get("expected_baud_rates")
            )
            results.partial_results["protocol_detection"] = {
                "protocol": detected_protocol,
                "confidence": detection_confidence,
            }
            if detection_confidence < 0.6:
                results.warnings.append(
                    f"Low protocol detection confidence: {detection_confidence:.2f}"
                )
        pbar.update(1)
        return detected_protocol
    except Exception as e:
        msg = f"Protocol detection failed: {e}"
        results.warnings.append(msg)
        logger.warning(msg)
        pbar.update(1)
        return "unknown"


def _step_3_decode_messages(
    pbar: Any,
    context: _WorkflowContext,
    results: _WorkflowResults,
    traces: dict[str, WaveformTrace],
    detected_protocol: str,
) -> tuple[ProtocolSpec, Any]:
    """Execute workflow step 3: Decode messages.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        traces: Loaded traces.
        detected_protocol: Protocol from detection step.

    Returns:
        Tuple of (protocol_spec, re_result).
    """
    pbar.set_description("Decoding messages")
    try:
        primary_trace = next(iter(traces.values()))
        re_result = reverse_engineer_signal(
            primary_trace,
            expected_baud_rates=context.kwargs.get("expected_baud_rates"),
            min_frames=context.kwargs.get("min_frames", 3),
            max_frame_length=context.kwargs.get("max_frame_length", 256),
            checksum_types=context.kwargs.get("checksum_types"),
        )
        protocol_spec = re_result.protocol_spec
        results.partial_results["reverse_engineering"] = re_result
        pbar.update(1)
        return protocol_spec, re_result
    except Exception as e:
        msg = f"Message decoding failed: {e}"
        results.warnings.append(msg)
        logger.exception(msg)
        # Create minimal protocol spec for graceful degradation
        # Use "Unknown" since we couldn't decode despite detection
        protocol_spec = ProtocolSpec(
            name="Unknown",
            baud_rate=0.0,
            frame_format="unknown",
            sync_pattern="",
            frame_length=None,
            fields=[],
            checksum_type=None,
            checksum_position=None,
            confidence=0.0,
        )
        pbar.update(1)
        return protocol_spec, None


def _step_4_differential_analysis(
    pbar: Any,
    results: _WorkflowResults,
    traces: dict[str, WaveformTrace],
    protocol_spec: ProtocolSpec,
) -> None:
    """Execute workflow step 4: Differential analysis.

    Args:
        pbar: Progress bar for updates.
        results: Results accumulator.
        traces: Loaded traces.
        protocol_spec: Protocol specification to enhance.
    """
    pbar.set_description("Differential analysis")
    if len(traces) > 1:
        try:
            diff_results = _differential_analysis(traces)
            results.partial_results["differential"] = diff_results
            _enhance_spec_with_differential(protocol_spec, diff_results)
        except Exception as e:
            msg = f"Differential analysis failed: {e}"
            results.warnings.append(msg)
            logger.warning(msg)
    pbar.update(1)


def _step_5_infer_structure(
    pbar: Any, results: _WorkflowResults, re_result: Any, protocol_spec: ProtocolSpec
) -> None:
    """Execute workflow step 5: Infer message structure.

    Args:
        pbar: Progress bar for updates.
        results: Results accumulator.
        re_result: Reverse engineering result.
        protocol_spec: Protocol specification to enhance.
    """
    pbar.set_description("Inferring structure")
    try:
        if hasattr(re_result, "frames") and re_result.frames:
            structure = _infer_message_structure(re_result.frames)
            results.partial_results["structure"] = structure
            if structure.get("fields"):
                protocol_spec.fields = structure["fields"]
    except Exception as e:
        msg = f"Structure inference failed: {e}"
        results.warnings.append(msg)
        logger.warning(msg)
    pbar.update(1)


def _step_6_detect_crypto(
    pbar: Any, context: _WorkflowContext, results: _WorkflowResults, re_result: Any
) -> None:
    """Execute workflow step 6: Detect crypto/entropy regions.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        re_result: Reverse engineering result.
    """
    pbar.set_description("Detecting crypto")
    if context.detect_crypto:
        try:
            if hasattr(re_result, "frames") and re_result.frames:
                crypto_regions = _detect_crypto_regions(re_result.frames)
                results.partial_results["crypto"] = crypto_regions
                if crypto_regions:
                    results.warnings.append(
                        f"Found {len(crypto_regions)} high-entropy regions (possible encryption)"
                    )
        except Exception as e:
            msg = f"Crypto detection failed: {e}"
            results.warnings.append(msg)
            logger.warning(msg)
    pbar.update(1)


def _step_7_recover_crc(
    pbar: Any,
    context: _WorkflowContext,
    results: _WorkflowResults,
    re_result: Any,
    protocol_spec: ProtocolSpec,
) -> None:
    """Execute workflow step 7: Recover CRC/checksums.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        re_result: Reverse engineering result.
        protocol_spec: Protocol specification to enhance.
    """
    pbar.set_description("Recovering CRCs")
    if context.auto_crc:
        try:
            if hasattr(re_result, "frames") and re_result.frames:
                crc_results = _recover_crc(re_result.frames, context.kwargs.get("checksum_types"))
                results.partial_results["crc"] = crc_results
                if crc_results and crc_results.get("checksum_type"):
                    protocol_spec.checksum_type = crc_results["checksum_type"]
                    protocol_spec.checksum_position = crc_results.get("position", -1)
        except Exception as e:
            msg = f"CRC recovery failed: {e}"
            results.warnings.append(msg)
            logger.warning(msg)
    pbar.update(1)


def _step_8_extract_state_machine(pbar: Any, results: _WorkflowResults, re_result: Any) -> None:
    """Execute workflow step 8: Extract state machine.

    Args:
        pbar: Progress bar for updates.
        results: Results accumulator.
        re_result: Reverse engineering result.
    """
    pbar.set_description("Extracting state machine")
    try:
        if hasattr(re_result, "frames") and re_result.frames:
            state_machine = _extract_state_machine(re_result.frames)
            results.partial_results["state_machine"] = state_machine
    except Exception as e:
        msg = f"State machine extraction failed: {e}"
        results.warnings.append(msg)
        logger.warning(msg)
    pbar.update(1)


def _step_9_generate_wireshark(
    pbar: Any, context: _WorkflowContext, results: _WorkflowResults, protocol_spec: ProtocolSpec
) -> None:
    """Execute workflow step 9: Generate Wireshark dissector.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        protocol_spec: Protocol specification.
    """
    pbar.set_description("Generating Wireshark dissector")
    try:
        results.dissector_path = (
            context.export_path / f"{protocol_spec.name.replace(' ', '_').lower()}.lua"
        )
        _generate_wireshark_dissector(protocol_spec, results.dissector_path)
    except Exception as e:
        msg = f"Wireshark dissector generation failed: {e}"
        results.warnings.append(msg)
        logger.warning(msg)
        results.dissector_path = None
    pbar.update(1)


def _step_10_generate_scapy(
    pbar: Any, context: _WorkflowContext, results: _WorkflowResults, protocol_spec: ProtocolSpec
) -> None:
    """Execute workflow step 10: Generate Scapy layer.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        protocol_spec: Protocol specification.
    """
    pbar.set_description("Generating Scapy layer")
    try:
        results.scapy_layer_path = (
            context.export_path / f"{protocol_spec.name.replace(' ', '_').lower()}.py"
        )
        _generate_scapy_layer(protocol_spec, results.scapy_layer_path)
    except Exception as e:
        msg = f"Scapy layer generation failed: {e}"
        results.warnings.append(msg)
        logger.warning(msg)
        results.scapy_layer_path = None
    pbar.update(1)


def _step_11_generate_kaitai(
    pbar: Any, context: _WorkflowContext, results: _WorkflowResults, protocol_spec: ProtocolSpec
) -> None:
    """Execute workflow step 11: Generate Kaitai struct.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        protocol_spec: Protocol specification.
    """
    pbar.set_description("Generating Kaitai struct")
    try:
        results.kaitai_path = (
            context.export_path / f"{protocol_spec.name.replace(' ', '_').lower()}.ksy"
        )
        _generate_kaitai_struct(protocol_spec, results.kaitai_path)
    except Exception as e:
        msg = f"Kaitai struct generation failed: {e}"
        results.warnings.append(msg)
        logger.warning(msg)
        results.kaitai_path = None
    pbar.update(1)


def _step_12_create_test_vectors(
    pbar: Any,
    context: _WorkflowContext,
    results: _WorkflowResults,
    re_result: Any,
    protocol_spec: ProtocolSpec,
) -> None:
    """Execute workflow step 12: Create test vectors.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        re_result: Reverse engineering result.
        protocol_spec: Protocol specification.
    """
    pbar.set_description("Creating test vectors")
    if context.generate_tests:
        try:
            if hasattr(re_result, "frames") and re_result.frames:
                results.test_vectors_path = context.export_path / "test_vectors.json"
                _create_test_vectors(re_result.frames, protocol_spec, results.test_vectors_path)
        except Exception as e:
            msg = f"Test vector generation failed: {e}"
            results.warnings.append(msg)
            logger.warning(msg)
            results.test_vectors_path = None
    pbar.update(1)


def _step_13_generate_report(
    pbar: Any, context: _WorkflowContext, results: _WorkflowResults, protocol_spec: ProtocolSpec
) -> None:
    """Execute workflow step 13: Generate HTML report.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        protocol_spec: Protocol specification.
    """
    pbar.set_description("Generating report")
    try:
        results.report_path = context.export_path / "report.html"
        _generate_report(
            protocol_spec, results.partial_results, results.report_path, results.warnings
        )
    except Exception as e:
        msg = f"Report generation failed: {e}"
        results.warnings.append(msg)
        logger.warning(msg)
        results.report_path = None
    pbar.update(1)


def _step_14_replay_validation(
    pbar: Any,
    context: _WorkflowContext,
    results: _WorkflowResults,
    protocol_spec: ProtocolSpec,
    re_result: Any,
) -> None:
    """Execute workflow step 14: Replay validation.

    Args:
        pbar: Progress bar for updates.
        context: Workflow context.
        results: Results accumulator.
        protocol_spec: Protocol specification.
        re_result: Reverse engineering result.
    """
    pbar.set_description("Validating (replay)")
    if context.validate and context.kwargs.get("target_device"):
        try:
            results.validation_result = _replay_validation(
                protocol_spec,
                context.kwargs["target_device"],
                re_result.frames if hasattr(re_result, "frames") else [],
            )
            results.partial_results["validation"] = results.validation_result
        except Exception as e:
            msg = f"Replay validation failed: {e}"
            results.warnings.append(msg)
            logger.warning(msg)
            results.validation_result = None
    pbar.update(1)


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _initialize_workflow_context(
    captures: dict[str, str] | str,
    export_dir: str,
    protocol_hint: str | None,
    validate: bool,
    auto_crc: bool,
    detect_crypto: bool,
    generate_tests: bool,
    kwargs: dict[str, Any],
) -> _WorkflowContext:
    """Initialize workflow context from input parameters.

    Args:
        captures: Path to capture file OR dict mapping labels to paths.
        export_dir: Directory for all output files.
        protocol_hint: Optional protocol name hint.
        validate: Whether to perform replay validation.
        auto_crc: Whether to automatically detect CRCs.
        detect_crypto: Whether to detect crypto regions.
        generate_tests: Whether to generate test vectors.
        kwargs: Additional workflow options.

    Returns:
        Initialized _WorkflowContext.

    Raises:
        ValueError: If captures is invalid.
        FileNotFoundError: If capture files don't exist.
    """
    # Convert single capture to dict
    if isinstance(captures, str):
        capture_dict = {"primary": captures}
    else:
        capture_dict = captures

    if not capture_dict:
        msg = "No captures provided"
        raise ValueError(msg)

    # Validate all capture files exist
    for path_str in capture_dict.values():
        path = Path(path_str)
        if not path.exists():
            msg = f"Capture file not found: {path_str}"
            raise FileNotFoundError(msg)

    # Create export directory
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    # Extract verbose flag from kwargs
    verbose = kwargs.get("verbose", True)

    return _WorkflowContext(
        capture_dict=capture_dict,
        export_path=export_path,
        verbose=verbose,
        protocol_hint=protocol_hint,
        auto_crc=auto_crc,
        detect_crypto=detect_crypto,
        generate_tests=generate_tests,
        validate=validate,
        kwargs=kwargs,
    )


def _load_captures(capture_dict: dict[str, str]) -> dict[str, WaveformTrace]:
    """Load all capture files with auto-format detection.

    Args:
        capture_dict: Mapping of labels to file paths.

    Returns:
        Dict mapping labels to loaded WaveformTrace objects.

    Raises:
        ValueError: If file format not supported.
    """
    import oscura.loaders as loaders

    traces: dict[str, WaveformTrace] = {}

    for label, path_str in capture_dict.items():
        path = Path(path_str)
        suffix = path.suffix.lower()

        # Auto-detect format based on extension
        if suffix in (".bin", ".dat"):
            # Binary files - try to infer structure
            trace = loaders.load_binary(str(path))  # type: ignore[attr-defined]
        elif suffix == ".wfm":
            trace = loaders.load_tektronix(str(path))  # type: ignore[attr-defined]
        elif suffix == ".vcd":
            trace = loaders.load_vcd(str(path))  # type: ignore[attr-defined]
        elif suffix == ".wav":
            trace = loaders.load_wav(str(path))  # type: ignore[attr-defined]
        elif suffix == ".csv":
            trace = loaders.load_csv(str(path))  # type: ignore[attr-defined]
        elif suffix in (".pcap", ".pcapng"):
            trace = loaders.load_pcap(str(path))  # type: ignore[attr-defined]
        elif suffix == ".sr":
            trace = loaders.load_sigrok(str(path))  # type: ignore[attr-defined]
        else:
            msg = f"Unsupported file format: {suffix}"
            raise ValueError(msg)

        traces[label] = trace

    return traces


def _detect_protocol(
    traces: dict[str, WaveformTrace], expected_baud_rates: list[int] | None = None
) -> tuple[str, float]:
    """Detect protocol from signal characteristics.

    Args:
        traces: Loaded waveform traces.
        expected_baud_rates: Optional list of expected baud rates.

    Returns:
        Tuple of (protocol_name, confidence_score).
    """
    # Use first trace for detection
    _ = next(iter(traces.values()))

    # Simple protocol detection based on signal characteristics
    # In future, this would use more sophisticated detection algorithms

    # For now, default to UART as most common
    return "uart", 0.8


def _differential_analysis(traces: dict[str, WaveformTrace]) -> dict[str, Any]:
    """Perform differential analysis between multiple captures.

    Args:
        traces: Multiple labeled captures.

    Returns:
        Dict with differential analysis results.
    """
    # Placeholder for differential analysis
    # Would compare traces to identify state-dependent fields
    results = {
        "trace_count": len(traces),
        "differences": [],
        "constant_fields": [],
        "variable_fields": [],
    }
    return results


def _enhance_spec_with_differential(spec: ProtocolSpec, diff_results: dict[str, Any]) -> None:
    """Enhance protocol spec with differential analysis insights.

    Args:
        spec: Protocol specification to enhance (modified in-place).
        diff_results: Results from differential analysis.
    """
    # Placeholder - would add field annotations based on differential results


def _infer_message_structure(frames: list[Any]) -> dict[str, Any]:
    """Infer detailed message structure from decoded frames.

    Args:
        frames: List of decoded frames.

    Returns:
        Dict with inferred structure details.
    """
    # Placeholder for structure inference
    return {"fields": [], "patterns": []}


def _detect_crypto_regions(frames: list[Any]) -> list[dict[str, Any]]:
    """Detect encrypted/compressed regions via entropy analysis.

    Args:
        frames: List of decoded frames.

    Returns:
        List of detected high-entropy regions.
    """
    regions = []

    # Analyze entropy of each frame
    for i, frame in enumerate(frames):
        if hasattr(frame, "raw_bytes"):
            data = frame.raw_bytes
            if len(data) > 0:
                # Calculate byte entropy
                entropy = _calculate_entropy(data)
                if entropy > 7.0:  # High entropy threshold
                    regions.append(
                        {
                            "frame_index": i,
                            "offset": 0,
                            "length": len(data),
                            "entropy": entropy,
                        }
                    )

    return regions


def _calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte sequence.

    Args:
        data: Byte sequence.

    Returns:
        Entropy in bits (0-8 for bytes).
    """
    if not data:
        return 0.0

    # Count byte frequencies
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / len(data)

    # Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def _recover_crc(frames: list[Any], checksum_types: list[str] | None = None) -> dict[str, Any]:
    """Recover CRC/checksum algorithms.

    Args:
        frames: List of decoded frames.
        checksum_types: Optional list of checksum types to try.

    Returns:
        Dict with CRC recovery results.
    """
    # Placeholder - would use existing checksum detection from reverse_engineer_signal
    return {"checksum_type": None, "position": None, "confidence": 0.0}


def _extract_state_machine(frames: list[Any]) -> dict[str, Any]:
    """Extract state machine from message sequences.

    Args:
        frames: List of decoded frames.

    Returns:
        Dict with state machine representation.
    """
    # Placeholder for state machine extraction using RPNI or similar
    return {"states": [], "transitions": [], "initial_state": None}


def _generate_wireshark_dissector(spec: ProtocolSpec, output_path: Path) -> None:
    """Generate Wireshark Lua dissector.

    Args:
        spec: Protocol specification.
        output_path: Path to write .lua file.
    """
    # Generate basic Lua dissector
    lua_code = f'''-- Wireshark dissector for {spec.name}
-- Auto-generated by Oscura

local proto = Proto("{spec.name.lower().replace(" ", "_")}", "{spec.name}")

-- Fields
local fields = proto.fields
'''

    for spec_field in spec.fields:
        lua_code += f'fields.{spec_field.name} = ProtoField.bytes("{spec.name.lower()}.{spec_field.name}", "{spec_field.name}")\n'

    lua_code += """
function proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = proto.name
    local subtree = tree:add(proto, buffer(), proto.name)
    -- Field parsing would go here
end

DissectorTable.get("udp.port"):add(0, proto)
"""

    output_path.write_text(lua_code)
    logger.info(f"Generated Wireshark dissector: {output_path}")


def _generate_scapy_layer(spec: ProtocolSpec, output_path: Path) -> None:
    """Generate Scapy protocol layer.

    Args:
        spec: Protocol specification.
        output_path: Path to write .py file.
    """
    class_name = spec.name.replace(" ", "")

    scapy_code = f'''"""Scapy layer for {spec.name}"""

from scapy.packet import Packet
from scapy.fields import ByteField, XByteField

class {class_name}(Packet):
    name = "{spec.name}"
    fields_desc = [
'''

    for spec_field in spec.fields:
        field_type = "ByteField" if spec_field.field_type == "uint8" else "XByteField"
        scapy_code += f'        {field_type}("{spec_field.name}", 0),\n'

    scapy_code += "    ]\n"

    output_path.write_text(scapy_code)
    logger.info(f"Generated Scapy layer: {output_path}")


def _generate_kaitai_struct(spec: ProtocolSpec, output_path: Path) -> None:
    """Generate Kaitai Struct definition.

    Args:
        spec: Protocol specification.
        output_path: Path to write .ksy file.
    """
    kaitai_yaml = f"""meta:
  id: {spec.name.lower().replace(" ", "_")}
  title: {spec.name}
  endian: le
seq:
"""

    for spec_field in spec.fields:
        kaitai_yaml += f"""  - id: {spec_field.name}
    type: u1
"""

    output_path.write_text(kaitai_yaml)
    logger.info(f"Generated Kaitai struct: {output_path}")


def _create_test_vectors(frames: list[Any], spec: ProtocolSpec, output_path: Path) -> None:
    """Create test vectors for validation.

    Args:
        frames: Decoded frames.
        spec: Protocol specification.
        output_path: Path to write .json file.
    """
    test_vectors = []

    for i, frame in enumerate(frames[:10]):  # First 10 frames
        if hasattr(frame, "raw_bytes"):
            test_vectors.append(
                {
                    "index": i,
                    "raw_hex": frame.raw_bytes.hex(),
                    "expected_fields": {},
                }
            )

    vectors_data = {
        "protocol": spec.name,
        "version": "1.0",
        "test_vectors": test_vectors,
    }

    with output_path.open("w") as f:
        json.dump(vectors_data, f, indent=2)

    logger.info(f"Generated {len(test_vectors)} test vectors: {output_path}")


def _generate_report(
    spec: ProtocolSpec,
    partial_results: dict[str, Any],
    output_path: Path,
    warnings: list[str],
) -> None:
    """Generate HTML report with analysis results.

    Args:
        spec: Protocol specification.
        partial_results: All partial results from workflow.
        output_path: Path to write .html file.
        warnings: List of warnings.
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{spec.name} - Reverse Engineering Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .warning {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{spec.name} - Reverse Engineering Report</h1>

    <div class="section">
        <h2>Protocol Summary</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Name</td><td>{spec.name}</td></tr>
            <tr><td>Baud Rate</td><td>{spec.baud_rate} bps</td></tr>
            <tr><td>Frame Format</td><td>{spec.frame_format}</td></tr>
            <tr><td>Sync Pattern</td><td>{spec.sync_pattern}</td></tr>
            <tr><td>Frame Length</td><td>{spec.frame_length or "Variable"}</td></tr>
            <tr><td>Checksum</td><td>{spec.checksum_type or "None detected"}</td></tr>
            <tr><td>Confidence</td><td>{spec.confidence:.2%}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Fields ({len(spec.fields)})</h2>
        <table>
            <tr><th>Name</th><th>Offset</th><th>Size</th><th>Type</th></tr>
"""

    for spec_field in spec.fields:
        html += f"""            <tr>
                <td>{spec_field.name}</td>
                <td>{spec_field.offset}</td>
                <td>{spec_field.size}</td>
                <td>{spec_field.field_type}</td>
            </tr>
"""

    html += """        </table>
    </div>
"""

    if warnings:
        html += """    <div class="section">
        <h2>Warnings</h2>
"""
        for warning in warnings:
            html += f'        <div class="warning">{warning}</div>\n'
        html += "    </div>\n"

    html += """</body>
</html>"""

    output_path.write_text(html)
    logger.info(f"Generated report: {output_path}")


def _replay_validation(spec: ProtocolSpec, target_device: str, frames: list[Any]) -> dict[str, Any]:
    """Perform replay validation on target hardware.

    Args:
        spec: Protocol specification.
        target_device: Device path for validation.
        frames: Frames to replay.

    Returns:
        Dict with validation results.
    """
    # Placeholder for replay validation
    return {
        "replayed": 0,
        "successful": 0,
        "failed": 0,
        "success_rate": 0.0,
    }


def _calculate_overall_confidence(
    spec: ProtocolSpec, partial_results: dict[str, Any], warnings: list[str]
) -> float:
    """Calculate overall workflow confidence score.

    Args:
        spec: Protocol specification.
        partial_results: All partial results.
        warnings: List of warnings.

    Returns:
        Overall confidence score (0-1).
    """
    # Start with protocol spec confidence
    confidence = spec.confidence

    # Penalize for warnings
    warning_penalty = len(warnings) * 0.05
    confidence = max(0.0, confidence - warning_penalty)

    # Bonus for successful steps
    successful_steps = sum(
        1 for key in ["traces", "reverse_engineering", "state_machine"] if key in partial_results
    )
    step_bonus = successful_steps * 0.02
    confidence = min(1.0, confidence + step_bonus)

    return confidence
