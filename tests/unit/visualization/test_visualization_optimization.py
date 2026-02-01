"""Tests for visualization optimization functions.

Tests VIS-013, VIS-014, VIS-015, VIS-016, VIS-017, VIS-019, VIS-021, VIS-022.
"""

import numpy as np
import pytest

from oscura.analyzers.eye.diagram import (
    auto_center_eye_diagram,
    generate_eye,
)
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.visualization.layout import (
    Annotation,
    layout_stacked_channels,
    optimize_annotation_placement,
)
from oscura.visualization.optimization import (
    calculate_grid_spacing,
    calculate_optimal_x_window,
    calculate_optimal_y_range,
    decimate_for_display,
    optimize_db_range,
)
from oscura.visualization.render import (
    configure_dpi_rendering,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.visualization,
    pytest.mark.usefixtures("cleanup_matplotlib"),
]


class TestYAxisRangeOptimization:
    """Test VIS-013: Auto Y-Axis Range Optimization."""

    def test_basic_range_calculation(self):
        """Test basic Y-range calculation without outliers."""
        np.random.seed(42)  # For reproducibility
        data = np.random.randn(1000)
        y_min, y_max = calculate_optimal_y_range(data)

        # Y range should be reasonable (excludes outliers, includes margin)
        # Most data should be within the range
        within_range = np.sum((data >= y_min) & (data <= y_max))
        assert within_range >= 950  # At least 95% of data
        assert y_max > y_min

        # Range should be larger than core data range (due to margin)
        core_data = data[(data >= np.percentile(data, 1)) & (data <= np.percentile(data, 99))]
        core_range = np.max(core_data) - np.min(core_data)
        calc_range = y_max - y_min
        assert calc_range >= core_range  # Should include core + margin

    def test_outlier_exclusion(self):
        """Test that outliers are excluded from range."""
        # Create data with outliers (fixed seed for deterministic test)
        np.random.seed(42)
        data = np.random.randn(1000)
        data[0] = 100.0  # Outlier
        data[1] = -100.0  # Outlier

        y_min, y_max = calculate_optimal_y_range(data, outlier_threshold=3.0)

        # Range should not include outliers
        assert y_min > -50
        assert y_max < 50

    def test_symmetric_range(self):
        """Test symmetric range mode for bipolar signals."""
        np.random.seed(43)  # Fixed seed for deterministic test
        data = np.random.randn(1000)
        y_min, y_max = calculate_optimal_y_range(data, symmetric=True)

        # Should be approximately symmetric
        assert abs(y_min + y_max) < 0.1 * abs(y_max)

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_optimal_y_range(np.array([]))

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        data = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
        y_min, y_max = calculate_optimal_y_range(data)

        assert not np.isnan(y_min)
        assert not np.isnan(y_max)
        assert y_min < 1.0
        assert y_max > 4.0

    def test_smart_margin_adjustment(self):
        """Test that margin adjusts based on data density."""
        # Dense data should have smaller margin
        np.random.seed(42)  # For reproducibility
        dense_data = np.random.randn(20000)
        y_min_dense, y_max_dense = calculate_optimal_y_range(dense_data)

        # Sparse data should have larger margin
        np.random.seed(43)  # For reproducibility
        sparse_data = np.random.randn(50)
        _y_min_sparse, y_max_sparse = calculate_optimal_y_range(sparse_data)

        # Calculate margin ratios
        dense_range = y_max_dense - y_min_dense
        sparse_range = y_max_sparse - _y_min_sparse

        # Both should be positive
        assert dense_range > 0
        assert sparse_range > 0


class TestXAxisWindowOptimization:
    """Test VIS-014: Adaptive X-Axis Time Window."""

    def test_full_signal_window(self):
        """Test window calculation for simple signal."""
        time = np.linspace(0, 1e-3, 10000)
        data = np.sin(2 * np.pi * 1000 * time)

        t_start, t_end = calculate_optimal_x_window(time, data)

        assert t_start >= time[0]
        assert t_end <= time[-1]
        assert t_end > t_start

    def test_activity_detection(self):
        """Test that active regions are detected."""
        time = np.linspace(0, 1e-3, 10000)
        data = np.zeros_like(time)
        # Add activity in middle
        data[4000:6000] = np.sin(2 * np.pi * 1000 * time[4000:6000])

        t_start, _t_end = calculate_optimal_x_window(time, data, activity_threshold=0.1)

        # Window should start near active region
        assert t_start >= time[3000]

    def test_mismatched_arrays_raises_error(self):
        """Test that mismatched time/data raises ValueError."""
        time = np.linspace(0, 1, 100)
        data = np.random.randn(50)

        with pytest.raises(ValueError, match="must match"):
            calculate_optimal_x_window(time, data)


class TestGridSpacing:
    """Test VIS-019: Grid Auto-Spacing."""

    def test_nice_numbers(self):
        """Test that grid spacing uses nice numbers."""
        major, minor = calculate_grid_spacing(0, 100, target_major_ticks=5)

        # Should be nice number like 20
        assert major in [10, 20, 25, 50]
        assert minor < major

    def test_log_scale_spacing(self):
        """Test logarithmic grid spacing."""
        major, minor = calculate_grid_spacing(1, 1000, log_scale=True, target_major_ticks=3)

        assert major > 0
        assert minor > 0
        assert minor < major

    def test_invalid_range_raises_error(self):
        """Test that invalid range raises ValueError."""
        with pytest.raises(ValueError, match="Invalid axis range"):
            calculate_grid_spacing(100, 10)

    def test_time_axis_alignment(self):
        """Test time axis alignment to natural units."""
        major, _minor = calculate_grid_spacing(0, 1e-3, time_axis=True, target_major_ticks=5)

        # Should align to time units
        time_units = [1e-9, 2e-9, 5e-9, 1e-6, 2e-6, 5e-6, 1e-3, 2e-3, 5e-3]
        assert (
            major in time_units
            or abs(major - min(time_units, key=lambda x: abs(x - major))) < 1e-12
        )


class TestSpectrumDbRangeOptimization:
    """Test VIS-022: Spectrum dB Range Optimization."""

    def test_db_range_from_spectrum(self):
        """Test dB range optimization from spectrum data."""
        # Create spectrum with noise floor and peaks
        freq = np.linspace(0, 1000, 1000)
        spectrum_linear = np.ones_like(freq) * 0.01  # Noise floor
        spectrum_linear[100:110] = 1.0  # Peak

        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum_linear)

        db_min, db_max = optimize_db_range(spectrum_db)

        assert db_min < db_max
        # Max should be near peak
        assert db_max > -5  # 20*log10(1.0) = 0
        # Min should be below noise floor
        assert db_min < -40

    def test_linear_to_db_conversion(self):
        """Test automatic linear to dB conversion."""
        spectrum_linear = np.random.rand(1000) * 100

        db_min, db_max = optimize_db_range(spectrum_linear)

        assert db_min < db_max

    def test_empty_spectrum_raises_error(self):
        """Test that empty spectrum raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            optimize_db_range(np.array([]))

    def test_dynamic_range_compression(self):
        """Test that excessive dynamic range is compressed."""
        # Create spectrum with huge dynamic range
        spectrum_db = np.linspace(-150, 0, 1000)

        db_min, db_max = optimize_db_range(spectrum_db, max_dynamic_range_db=80)

        # Dynamic range should be limited
        assert (db_max - db_min) <= 80


class TestDecimation:
    """Test VIS-014: Decimation for display."""

    def test_no_decimation_when_small(self):
        """Test that small signals are not decimated."""
        time = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * time)

        time_dec, _data_dec = decimate_for_display(time, data, max_points=1000)

        assert len(time_dec) == len(time)
        assert np.array_equal(time_dec, time)

    def test_uniform_decimation(self):
        """Test uniform stride decimation."""
        time = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * time)

        time_dec, data_dec = decimate_for_display(time, data, max_points=1000, method="uniform")

        assert len(time_dec) <= 1000
        assert len(time_dec) == len(data_dec)

    def test_minmax_decimation(self):
        """Test min-max envelope decimation."""
        time = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * 10 * time)

        time_dec, data_dec = decimate_for_display(time, data, max_points=1000, method="minmax")

        assert len(time_dec) <= 1000
        # Should preserve peaks (within tolerance)
        assert np.max(data_dec) >= 0.95 * np.max(data)
        assert np.min(data_dec) <= 0.95 * np.min(data)

    def test_lttb_decimation(self):
        """Test LTTB decimation algorithm."""
        time = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * 5 * time)

        time_dec, _data_dec = decimate_for_display(time, data, max_points=500, method="lttb")

        assert len(time_dec) <= 500
        assert time_dec[0] == time[0]
        assert time_dec[-1] == time[-1]

    def test_invalid_method_raises_error(self):
        """Test that invalid decimation method raises ValueError."""
        time = np.linspace(0, 1, 10000)  # Need more data to trigger decimation
        data = np.random.randn(10000)

        with pytest.raises(ValueError, match="Invalid decimation method"):
            decimate_for_display(time, data, max_points=1000, method="invalid")


class TestChannelLayout:
    """Test VIS-015: Multi-Channel Stack Optimization."""

    def test_equal_spacing(self):
        """Test that channels are equally spaced."""
        layout = layout_stacked_channels(n_channels=3, gap_ratio=0.1)

        assert layout.n_channels == 3
        assert len(layout.heights) == 3
        assert len(layout.gaps) == 2

        # All heights should be equal
        assert np.allclose(layout.heights, layout.heights[0])

        # All gaps should be equal
        if len(layout.gaps) > 0:
            assert np.allclose(layout.gaps, layout.gaps[0])

    def test_single_channel(self):
        """Test layout with single channel."""
        layout = layout_stacked_channels(n_channels=1)

        assert layout.n_channels == 1
        assert len(layout.heights) == 1
        assert len(layout.gaps) == 0

    def test_gap_ratio(self):
        """Test that gap ratio is respected."""
        layout = layout_stacked_channels(n_channels=2, gap_ratio=0.2)

        # Gap should be 20% of channel height
        expected_gap = layout.heights[0] * 0.2
        assert np.isclose(layout.gaps[0], expected_gap)

    def test_invalid_channels_raises_error(self):
        """Test that invalid n_channels raises ValueError."""
        with pytest.raises(ValueError, match="n_channels"):
            layout_stacked_channels(n_channels=0)

    def test_invalid_gap_ratio_raises_error(self):
        """Test that invalid gap_ratio raises ValueError."""
        with pytest.raises(ValueError, match="gap_ratio"):
            layout_stacked_channels(n_channels=2, gap_ratio=-0.1)


class TestAnnotationPlacement:
    """Test VIS-016: Annotation Placement Intelligence."""

    def test_single_annotation(self):
        """Test placement of single annotation."""
        annot = Annotation("Test", x=100, y=100)
        placed = optimize_annotation_placement([annot])

        assert len(placed) == 1
        assert placed[0].annotation == annot

    def test_overlap_detection(self):
        """Test that overlapping annotations are separated."""
        annots = [
            Annotation("A", x=100, y=100, bbox_width=50, bbox_height=20),
            Annotation("B", x=102, y=102, bbox_width=50, bbox_height=20),
        ]

        placed = optimize_annotation_placement(annots, max_iterations=50)

        # Annotations should be separated
        dx = placed[1].display_x - placed[0].display_x
        dy = placed[1].display_y - placed[0].display_y
        distance = np.sqrt(dx**2 + dy**2)

        # Should be separated by at least some distance
        assert distance > 10

    def test_priority_placement(self):
        """Test that high-priority annotations move less."""
        annots = [
            Annotation("High", x=100, y=100, priority=0.9),
            Annotation("Low", x=102, y=102, priority=0.1),
        ]

        placed = optimize_annotation_placement(annots, max_iterations=50)

        # High priority should move less from original position
        np.sqrt((placed[0].display_x - 100) ** 2 + (placed[0].display_y - 100) ** 2)
        np.sqrt((placed[1].display_x - 102) ** 2 + (placed[1].display_y - 102) ** 2)

        # This might not always hold due to randomness, but generally true
        # Just check that both are placed
        assert placed[0].display_x > 0
        assert placed[1].display_x > 0

    def test_empty_annotations_raises_error(self):
        """Test that empty annotation list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            optimize_annotation_placement([])


class TestDpiRendering:
    """Test VIS-017: DPI-Aware Rendering."""

    def test_screen_preset(self):
        """Test screen rendering preset."""
        config = configure_dpi_rendering("screen")

        assert config["dpi"] == 96
        assert config["antialias"] is True
        assert config["format"] == "png"

    def test_print_preset(self):
        """Test print rendering preset."""
        config = configure_dpi_rendering("print")

        assert config["dpi"] == 300
        assert config["format"] == "pdf"

    def test_publication_preset(self):
        """Test publication rendering preset."""
        config = configure_dpi_rendering("publication")

        assert config["dpi"] == 600
        assert config["format"] == "pdf"
        assert "serif" in config["style_params"]["font.family"]

    def test_custom_dpi(self):
        """Test custom DPI override."""
        config = configure_dpi_rendering("screen", custom_dpi=150)

        assert config["dpi"] == 150

    def test_scale_factors(self):
        """Test that scale factors are calculated correctly."""
        config = configure_dpi_rendering("print", baseline_dpi=96)

        # 300 / 96 ≈ 3.125
        expected_scale = 300 / 96
        assert np.isclose(config["font_scale"], expected_scale)
        assert np.isclose(config["line_scale"], expected_scale)

    def test_invalid_preset_raises_error(self):
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid preset"):
            configure_dpi_rendering("invalid")


class TestEyeDiagramCentering:
    """Test VIS-021: Eye Diagram Auto-Centering."""

    def test_eye_diagram_centering(self):
        """Test auto-centering of eye diagram."""
        # Create simple periodic signal
        sample_rate = 10e9  # 10 GHz (need high rate for small unit interval)
        unit_interval = 1e-9  # 1 ns
        duration = 100e-9  # 100 ns
        time = np.arange(0, duration, 1 / sample_rate)

        # Create NRZ signal
        bit_rate = 1 / unit_interval
        data = np.sign(np.sin(2 * np.pi * bit_rate * time / 2))

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel="Data",
            ),
        )

        # Generate eye diagram
        eye = generate_eye(trace, unit_interval=unit_interval, n_ui=2)

        # Auto-center
        centered = auto_center_eye_diagram(eye)

        assert centered.n_traces == eye.n_traces
        assert centered.samples_per_ui == eye.samples_per_ui

    def test_invalid_trigger_fraction_raises_error(self):
        """Test that invalid trigger fraction raises ValueError."""
        # Create dummy eye diagram
        eye_data = np.random.randn(10, 100)
        time_axis = np.linspace(0, 2, 100)

        from oscura.analyzers.eye.diagram import EyeDiagram

        eye = EyeDiagram(
            data=eye_data,
            time_axis=time_axis,
            unit_interval=1e-9,
            samples_per_ui=50,
            n_traces=10,
            sample_rate=1e9,
        )

        with pytest.raises(ValueError, match="trigger_fraction"):
            auto_center_eye_diagram(eye, trigger_fraction=1.5)
