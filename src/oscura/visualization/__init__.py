"""Visualization module for Oscura.

Provides plotting functions for waveforms, spectra, and other signal data,
plus optimization utilities, style presets, and intelligent rendering.

**Requires matplotlib:**
This module requires matplotlib to be installed. Install with:
    pip install oscura[visualization]    # Just visualization
    pip install oscura[standard]         # Recommended
    pip install oscura[all]              # Everything

Example:
    >>> from oscura.visualization import plot_waveform, plot_spectrum
    >>> from oscura.visualization import plot_timing, plot_eye
    >>> from oscura.visualization import plot_bode, plot_histogram
    >>> from oscura.visualization import apply_style_preset
    >>> import matplotlib.pyplot as plt
    >>> with apply_style_preset("publication"):
    ...     plot_waveform(trace, time_unit="us")
    ...     plt.savefig("figure.pdf")
"""

# NOTE: Matplotlib is optional - individual functions will check and raise
# helpful errors if matplotlib is not installed when they're called.
# The module itself can be imported without matplotlib.

# Import plot module as namespace for DSL compatibility
from oscura.visualization import plot
from oscura.visualization.accessibility import (
    FAIL_SYMBOL,
    LINE_STYLES,
    PASS_SYMBOL,
    KeyboardHandler,
    add_plot_aria_attributes,
    format_pass_fail,
    generate_alt_text,
    get_colorblind_palette,
    get_multi_line_styles,
)

# Phase 30 enhancements
from oscura.visualization.annotations import (
    Annotation as EnhancedAnnotation,
)
from oscura.visualization.annotations import (
    PlacedAnnotation as EnhancedPlacedAnnotation,
)
from oscura.visualization.annotations import (
    create_priority_annotation,
    filter_by_zoom_level,
    place_annotations,
)
from oscura.visualization.axis_scaling import (
    calculate_axis_limits,
    calculate_multi_channel_limits,
    suggest_tick_spacing,
)
from oscura.visualization.colors import (
    COLORBLIND_SAFE_QUALITATIVE,
    DIVERGING_COOLWARM,
    SEQUENTIAL_VIRIDIS,
    select_optimal_palette,
)
from oscura.visualization.digital import (
    plot_logic_analyzer,
    plot_timing,
)
from oscura.visualization.eye import (
    plot_bathtub,
    plot_eye,
)
from oscura.visualization.histogram import (
    calculate_bin_edges,
    calculate_optimal_bins,
)
from oscura.visualization.interactive import (
    CursorMeasurement,
    ZoomState,
    add_measurement_cursors,
    enable_zoom_pan,
    plot_bode,
    plot_histogram,
    plot_phase,
    plot_waterfall,
    plot_with_cursors,
)
from oscura.visualization.layout import (
    Annotation,
    ChannelLayout,
    PlacedAnnotation,
    layout_stacked_channels,
    optimize_annotation_placement,
)
from oscura.visualization.optimization import (
    InterestingRegion,
    calculate_grid_spacing,
    calculate_optimal_x_window,
    calculate_optimal_y_range,
    decimate_for_display,
    detect_interesting_regions,
    optimize_db_range,
)
from oscura.visualization.presets import (
    DARK_THEME_PRESET,
    IEEE_DOUBLE_COLUMN_PRESET,
    IEEE_PUBLICATION_PRESET,
    VisualizationPreset,
    apply_preset,
    get_preset_colors,
)
from oscura.visualization.presets import (
    create_custom_preset as create_custom_visualization_preset,
)
from oscura.visualization.presets import (
    list_presets as list_visualization_presets,
)
from oscura.visualization.protocols import (
    plot_can_decode,
    plot_i2c_decode,
    plot_protocol_decode,
    plot_spi_decode,
    plot_uart_decode,
)
from oscura.visualization.render import (
    RenderPreset,
    apply_rendering_config,
    configure_dpi_rendering,
)
from oscura.visualization.rendering import (
    StreamingRenderer,
    downsample_for_memory,
    estimate_memory_usage,
    progressive_render,
    render_with_lod,
)

# Reverse Engineering Visualizations (HIGH-2)
from oscura.visualization.reverse_engineering import (
    plot_crc_parameters,
    plot_field_confidence_heatmap,
    plot_message_field_layout,
    plot_message_type_distribution,
    plot_pipeline_timing,
    plot_protocol_candidates,
    plot_re_summary,
)
from oscura.visualization.specialized import (
    ProtocolSignal,
    StateTransition,
    plot_protocol_timing,
    plot_state_machine,
)
from oscura.visualization.spectral import (
    plot_fft,
    plot_psd,
    plot_quality_summary,
    plot_spectrogram,
    plot_spectrum,
    plot_thd_bars,
)
from oscura.visualization.styles import (
    PRESENTATION_PRESET,
    PRINT_PRESET,
    PUBLICATION_PRESET,
    SCREEN_PRESET,
    StylePreset,
    apply_style_preset,
    create_custom_preset,
)
from oscura.visualization.thumbnails import (
    render_thumbnail,
    render_thumbnail_multichannel,
)
from oscura.visualization.time_axis import (
    TimeUnit,
    calculate_major_ticks,
    convert_time_values,
    create_relative_time,
    format_cursor_readout,
    format_time_labels,
    select_time_unit,
)
from oscura.visualization.waveform import (
    plot_multi_channel,
    plot_waveform,
    plot_xy,
)

__all__ = [
    "COLORBLIND_SAFE_QUALITATIVE",
    "DARK_THEME_PRESET",
    "DIVERGING_COOLWARM",
    "FAIL_SYMBOL",
    "IEEE_DOUBLE_COLUMN_PRESET",
    "IEEE_PUBLICATION_PRESET",
    "LINE_STYLES",
    "PASS_SYMBOL",
    "PRESENTATION_PRESET",
    "PRINT_PRESET",
    "PUBLICATION_PRESET",
    "SCREEN_PRESET",
    "SEQUENTIAL_VIRIDIS",
    "Annotation",
    # Layout functions (VIS-015, VIS-016)
    "ChannelLayout",
    # Interactive (VIS-008)
    "CursorMeasurement",
    # Phase 30: Enhanced modules
    "EnhancedAnnotation",
    "EnhancedPlacedAnnotation",
    "InterestingRegion",
    "KeyboardHandler",
    "PlacedAnnotation",
    "ProtocolSignal",
    "RenderPreset",
    "StateTransition",
    "StreamingRenderer",
    "StylePreset",
    "TimeUnit",
    "VisualizationPreset",
    # Interactive (VIS-007)
    "ZoomState",
    # Interactive (VIS-008)
    "add_measurement_cursors",
    "add_plot_aria_attributes",
    "apply_preset",
    "apply_rendering_config",
    # Styles
    "apply_style_preset",
    "calculate_axis_limits",
    "calculate_bin_edges",
    "calculate_grid_spacing",
    "calculate_major_ticks",
    "calculate_multi_channel_limits",
    # Histogram
    "calculate_optimal_bins",
    "calculate_optimal_x_window",
    # Optimization functions (VIS-013, VIS-014, VIS-019, VIS-020, VIS-022)
    "calculate_optimal_y_range",
    # Rendering functions (VIS-017)
    "configure_dpi_rendering",
    "convert_time_values",
    "create_custom_preset",
    "create_custom_visualization_preset",
    "create_priority_annotation",
    "create_relative_time",
    "decimate_for_display",
    "detect_interesting_regions",
    "downsample_for_memory",
    # Interactive (VIS-007)
    "enable_zoom_pan",
    "estimate_memory_usage",
    "filter_by_zoom_level",
    "format_cursor_readout",
    "format_pass_fail",
    "format_time_labels",
    "generate_alt_text",
    # Accessibility (ACC-001, ACC-002, ACC-003)
    "get_colorblind_palette",
    "get_multi_line_styles",
    "get_preset_colors",
    "layout_stacked_channels",
    "list_visualization_presets",
    "optimize_annotation_placement",
    "optimize_db_range",
    "place_annotations",
    "plot",
    # Eye diagram (VIS-006)
    "plot_bathtub",
    # Interactive (VIS-010)
    "plot_bode",
    # Protocol decode visualization (VIS-030)
    "plot_can_decode",
    # Reverse Engineering Visualization (HIGH-2)
    "plot_crc_parameters",
    "plot_eye",
    # Spectral plotting
    "plot_fft",
    # Reverse Engineering Visualization (HIGH-2)
    "plot_field_confidence_heatmap",
    # Interactive (VIS-012)
    "plot_histogram",
    # Protocol decode visualization (VIS-030)
    "plot_i2c_decode",
    # Digital visualization (VIS-005)
    "plot_logic_analyzer",
    # Reverse Engineering Visualization (HIGH-2)
    "plot_message_field_layout",
    "plot_message_type_distribution",
    "plot_multi_channel",
    # Interactive (VIS-009)
    "plot_phase",
    # Reverse Engineering Visualization (HIGH-2)
    "plot_pipeline_timing",
    # Protocol decode visualization (VIS-030)
    "plot_protocol_candidates",
    "plot_protocol_decode",
    "plot_protocol_timing",
    "plot_psd",
    # Signal quality summary (IEEE 1241-2010)
    "plot_quality_summary",
    # Reverse Engineering Visualization (HIGH-2)
    "plot_re_summary",
    "plot_spectrogram",
    "plot_spectrum",
    # Protocol decode visualization (VIS-030)
    "plot_spi_decode",
    "plot_state_machine",
    # THD harmonic bar chart
    "plot_thd_bars",
    "plot_timing",
    # Protocol decode visualization (VIS-030)
    "plot_uart_decode",
    # Interactive (VIS-011)
    "plot_waterfall",
    # Waveform plotting
    "plot_waveform",
    # Interactive (VIS-008)
    "plot_with_cursors",
    "plot_xy",
    "progressive_render",
    # Thumbnails
    "render_thumbnail",
    "render_thumbnail_multichannel",
    "render_with_lod",
    # Colors
    "select_optimal_palette",
    "select_time_unit",
    "suggest_tick_spacing",
]
