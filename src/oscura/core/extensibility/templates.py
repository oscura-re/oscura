"""Plugin template generation for creating new Oscura plugins.

This module provides tools for generating plugin skeletons with all necessary
boilerplate code, tests, and documentation.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

# Plugin type definitions
PluginType = Literal["analyzer", "loader", "exporter", "decoder"]


@dataclass
class PluginTemplate:
    """Configuration for plugin template generation.

    Attributes:
        name: Plugin name (e.g., 'my_custom_decoder').
        plugin_type: Type of plugin ('analyzer', 'loader', 'exporter', 'decoder').
        output_dir: Directory where plugin will be generated.
        author: Plugin author name.
        description: Brief description of plugin functionality.
        version: Initial plugin version (default: '0.1.0').

    Example:
        >>> template = PluginTemplate(
        ...     name='flexray_decoder',
        ...     plugin_type='decoder',
        ...     output_dir=Path('plugins/flexray'),
        ...     author='John Doe',
        ...     description='FlexRay protocol decoder'
        ... )

    References:
        PLUG-008: Plugin Template Generator
    """

    name: str
    plugin_type: PluginType
    output_dir: Path
    author: str = "Plugin Author"
    description: str = "Custom Oscura plugin"
    version: str = "0.1.0"


def generate_plugin_template(
    name: str,
    plugin_type: PluginType,
    output_dir: Path,
    *,
    author: str = "Plugin Author",
    description: str | None = None,
    version: str = "0.1.0",
) -> Path:
    """Generate a plugin skeleton with all necessary boilerplate.

    Creates a complete plugin package structure including:
    - __init__.py with plugin metadata
    - Main module with stub implementation
    - tests/ directory with test stubs
    - README.md with usage instructions
    - pyproject.toml for packaging

    Args:
        name: Plugin name (will be converted to snake_case).
        plugin_type: Type of plugin to generate.
        output_dir: Directory where plugin will be created.
        author: Plugin author name.
        description: Plugin description (auto-generated if None).
        version: Initial plugin version.

    Returns:
        Path to the generated plugin directory.

    Raises:
        ValueError: If plugin_type is invalid.

    Example:
        >>> from pathlib import Path
        >>> plugin_dir = generate_plugin_template(
        ...     name='flexray_decoder',
        ...     plugin_type='decoder',
        ...     output_dir=Path('plugins/flexray'),
        ...     author='John Doe',
        ...     description='FlexRay protocol decoder'
        ... )
        >>> print(f"Plugin generated at {plugin_dir}")

    Plugin Structure:
        ```
        plugins/flexray/
        ├── __init__.py           # Plugin metadata and entry point
        ├── flexray_decoder.py    # Main implementation
        ├── tests/
        │   ├── __init__.py
        │   └── test_flexray_decoder.py
        ├── README.md             # Usage documentation
        └── pyproject.toml        # Packaging configuration
        ```

    References:
        PLUG-008: Plugin Template Generator
    """
    # Validate plugin type
    valid_types: set[PluginType] = {"analyzer", "loader", "exporter", "decoder"}
    if plugin_type not in valid_types:
        raise ValueError(
            f"Invalid plugin_type '{plugin_type}'. Must be one of: {', '.join(valid_types)}"
        )

    # Generate default description if not provided
    if description is None:
        description = f"Custom {plugin_type} plugin for Oscura"

    # Create template configuration
    template = PluginTemplate(
        name=name,
        plugin_type=plugin_type,
        output_dir=output_dir,
        author=author,
        description=description,
        version=version,
    )

    # Generate plugin directory structure
    _generate_plugin_structure(template)

    return output_dir


def _generate_plugin_structure(template: PluginTemplate) -> None:
    """Generate complete plugin directory structure.

    Args:
        template: Plugin template configuration.

    Raises:
        FileExistsError: If plugin directory already exists.
    """
    output_dir = template.output_dir

    # Check if directory exists
    if output_dir.exists():
        raise FileExistsError(
            f"Plugin directory already exists: {output_dir}\n"
            f"Remove it or choose a different output_dir."
        )

    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=False)
    tests_dir = output_dir / "tests"
    tests_dir.mkdir(exist_ok=False)

    # Generate files
    _write_init_py(template)
    _write_main_module(template)
    _write_test_init(template)
    _write_test_module(template)
    _write_readme(template)
    _write_pyproject_toml(template)


def _write_init_py(template: PluginTemplate) -> None:
    """Write plugin __init__.py with metadata.

    Args:
        template: Plugin template configuration.
    """
    content = textwrap.dedent(f'''\
        """{template.description}

        This plugin integrates with Oscura via entry points.

        Plugin Metadata:
            Name: {template.name}
            Type: {template.plugin_type}
            Version: {template.version}
            Author: {template.author}

        Installation:
            pip install -e .

        Usage:
            import oscura as osc
            # Plugin auto-discovered via entry points
            # See README.md for usage examples

        References:
            PLUG-008: Plugin Template Generator
        """
        from .{template.name} import {_get_class_name(template)}

        __version__ = "{template.version}"

        __all__ = [
            "{_get_class_name(template)}",
        ]
    ''')

    (template.output_dir / "__init__.py").write_text(content)


def _write_main_module(template: PluginTemplate) -> None:
    """Write main plugin module with stub implementation.

    Args:
        template: Plugin template configuration.
    """
    class_name = _get_class_name(template)

    if template.plugin_type == "decoder":
        content = _generate_decoder_stub(template, class_name)
    elif template.plugin_type == "analyzer":
        content = _generate_analyzer_stub(template, class_name)
    elif template.plugin_type == "loader":
        content = _generate_loader_stub(template, class_name)
    elif template.plugin_type == "exporter":
        content = _generate_exporter_stub(template, class_name)
    else:
        # Fallback generic stub
        content = _generate_generic_stub(template, class_name)  # type: ignore[unreachable]

    (template.output_dir / f"{template.name}.py").write_text(content)


def _write_test_init(template: PluginTemplate) -> None:
    """Write tests/__init__.py.

    Args:
        template: Plugin template configuration.
    """
    content = '"""Test suite for plugin."""\n'
    (template.output_dir / "tests" / "__init__.py").write_text(content)


def _write_test_module(template: PluginTemplate) -> None:
    """Write test module with example tests.

    Args:
        template: Plugin template configuration.
    """
    class_name = _get_class_name(template)

    content = textwrap.dedent(f'''\
        """Tests for {template.name} plugin.

        This module contains unit tests for the plugin implementation.
        """
        import numpy as np
        import pytest

        from {template.name} import {class_name}


        def test_{template.name}_initialization():
            """Test plugin can be instantiated."""
            plugin = {class_name}()
            assert plugin is not None


        def test_{template.name}_basic_functionality():
            """Test basic plugin functionality."""
            plugin = {class_name}()

            # USER: Implement test for your plugin's main functionality
            # Example:
            # result = plugin.process(test_data)
            # assert result is not None

            # Placeholder assertion - replace with actual tests
            assert True


        def test_{template.name}_error_handling():
            """Test plugin handles errors gracefully."""
            plugin = {class_name}()

            # USER: Implement error condition tests
            # Example:
            # with pytest.raises(ValueError):
            #     plugin.process(invalid_data)

            # Placeholder assertion - replace with actual tests
            assert True


        @pytest.mark.parametrize("param", [1, 2, 3])
        def test_{template.name}_parametrized(param):
            """Example parametrized test."""
            plugin = {class_name}()

            # USER: Implement parametrized test logic
            assert param > 0
    ''')

    (template.output_dir / "tests" / f"test_{template.name}.py").write_text(content)


def _write_readme(template: PluginTemplate) -> None:
    """Write README.md with usage instructions.

    Args:
        template: Plugin template configuration.
    """
    class_name = _get_class_name(template)
    entry_point_group = _get_entry_point_group(template.plugin_type)

    content = _generate_readme_content(template, class_name, entry_point_group)
    (template.output_dir / "README.md").write_text(content)


def _generate_readme_content(
    template: PluginTemplate, class_name: str, entry_point_group: str
) -> str:
    """Generate README.md content.

    Args:
        template: Plugin template configuration.
        class_name: PascalCase class name.
        entry_point_group: Entry point group name.

    Returns:
        README content as string.
    """
    header = _generate_readme_header(template, class_name)
    usage = _generate_readme_usage(template, class_name)
    dev = _generate_readme_development(template)
    metadata = _generate_readme_metadata(template, entry_point_group)
    return f"{header}\n{usage}\n{dev}\n{metadata}"


def _generate_readme_header(template: PluginTemplate, class_name: str) -> str:
    """Generate README header and installation section.

    Args:
        template: Plugin template configuration.
        class_name: Class name.

    Returns:
        Header section content.
    """
    return textwrap.dedent(f"""\
        # {class_name}

        {template.description}

        ## Installation

        Install in development mode:

        ```bash
        cd {template.output_dir.name}
        pip install -e .
        ```
    """)


def _generate_readme_usage(template: PluginTemplate, class_name: str) -> str:
    """Generate usage section.

    Args:
        template: Plugin template configuration.
        class_name: Class name.

    Returns:
        Usage section content.
    """
    return textwrap.dedent(f"""\
        ## Usage

        The plugin integrates automatically with Oscura via entry points:

        ```python
        import oscura as osc

        # Plugin is automatically discovered
        # USER: Add usage examples specific to your plugin
        ```

        ### Direct Usage

        ```python
        from {template.name} import {class_name}

        # Create instance
        plugin = {class_name}()

        # USER: Add direct usage examples
        ```

        ## CLI Integration

        After installation, the plugin is available in Oscura CLI:

        ```bash
        # List installed plugins
        oscura plugin list

        # Show plugin info
        oscura plugin info {template.name}
        ```
    """)


def _generate_readme_development(template: PluginTemplate) -> str:
    """Generate development section.

    Args:
        template: Plugin template configuration.

    Returns:
        Development section content.
    """
    return textwrap.dedent(f"""\
        ## Development

        ### Running Tests

        ```bash
        pytest tests/
        ```

        ### Code Quality

        ```bash
        # Linting
        ruff check {template.name}.py

        # Formatting
        ruff format {template.name}.py

        # Type checking
        mypy {template.name}.py
        ```
    """)


def _generate_readme_metadata(template: PluginTemplate, entry_point_group: str) -> str:
    """Generate metadata section.

    Args:
        template: Plugin template configuration.
        entry_point_group: Entry point group name.

    Returns:
        Metadata section content.
    """
    return textwrap.dedent(f"""\
        ## Plugin Type: {template.plugin_type}

        This is a **{template.plugin_type}** plugin for Oscura.

        ### Entry Point

        Registered in `{entry_point_group}` entry point group.

        ## Requirements

        - Oscura >= 0.1.0
        - Python >= 3.12

        ## License

        MIT

        ## Author

        {template.author}

        ## Version

        {template.version}
    """)


def _write_pyproject_toml(template: PluginTemplate) -> None:
    """Write pyproject.toml for plugin packaging.

    Args:
        template: Plugin template configuration.
    """
    entry_point_group = _get_entry_point_group(template.plugin_type)
    class_name = _get_class_name(template)

    content = textwrap.dedent(f'''\
        [project]
        name = "{template.name}"
        version = "{template.version}"
        description = "{template.description}"
        readme = "README.md"
        license = {{ text = "MIT" }}
        requires-python = ">=3.12"
        authors = [
            {{ name = "{template.author}" }}
        ]
        keywords = ["oscura", "plugin", "{template.plugin_type}"]
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
        ]

        dependencies = [
            "oscura>=0.1.0",
            "numpy>=1.26.0",
        ]

        [project.optional-dependencies]
        dev = [
            "pytest>=8.3.0",
            "pytest-cov>=6.0.0",
            "ruff>=0.8.0",
            "mypy>=1.13.0",
        ]

        # Oscura plugin entry point
        [project.entry-points."{entry_point_group}"]
        {template.name} = "{template.name}:{class_name}"

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        python_files = ["test_*.py"]
        python_classes = ["Test*"]
        python_functions = ["test_*"]

        [tool.ruff]
        line-length = 88
        target-version = "py312"

        [tool.ruff.lint]
        select = ["E", "F", "W", "I", "N", "UP", "YTT", "B", "A", "C4", "T10", "RUF"]

        [tool.mypy]
        python_version = "3.12"
        warn_return_any = true
        warn_unused_configs = true
        disallow_untyped_defs = true
    ''')

    (template.output_dir / "pyproject.toml").write_text(content)


def _get_class_name(template: PluginTemplate) -> str:
    """Generate class name from plugin name.

    Args:
        template: Plugin template configuration.

    Returns:
        PascalCase class name.

    Example:
        >>> template = PluginTemplate('my_decoder', 'decoder', Path('.'))
        >>> _get_class_name(template)
        'MyDecoder'
    """
    # Convert snake_case to PascalCase
    parts = template.name.split("_")
    return "".join(word.capitalize() for word in parts)


def _get_entry_point_group(plugin_type: PluginType) -> str:
    """Get entry point group for plugin type.

    Args:
        plugin_type: Type of plugin.

    Returns:
        Entry point group name.
    """
    return f"oscura.{plugin_type}s"


def _generate_decoder_stub(template: PluginTemplate, class_name: str) -> str:
    """Generate decoder plugin stub.

    Args:
        template: Plugin template configuration.
        class_name: Class name for the decoder.

    Returns:
        Python source code for decoder stub.
    """
    return textwrap.dedent(f'''\
        """{template.description}

        This decoder implements protocol decoding for Oscura.

        References:
            PLUG-008: Plugin Template Generator
        """
        from __future__ import annotations

        import numpy as np
        from numpy.typing import NDArray


        class {class_name}:
            """Protocol decoder implementation.

            Attributes:
                sample_rate: Sample rate of input signal in Hz.

            Example:
                >>> decoder = {class_name}(sample_rate=1_000_000)
                >>> frames = decoder.decode(digital_signal)

            References:
                PLUG-008: Plugin Template Generator
            """
            def __init__(self, *, sample_rate: float = 1_000_000.0) -> None:
                """Initialize decoder.

                Args:
                    sample_rate: Sample rate in Hz.
                """
                self.sample_rate = sample_rate

            def decode(
                self,
                signal: NDArray[np.uint8],
            ) -> list[dict[str, object]]:
                """Decode protocol frames from digital signal.

                Args:
                    signal: Digital signal (0/1 values).

                Returns:
                    List of decoded frames, each a dictionary with frame data.

                Raises:
                    ValueError: If signal is empty or invalid.

                Example:
                    >>> signal = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
                    >>> frames = decoder.decode(signal)
                """
                if len(signal) == 0:
                    raise ValueError("Signal cannot be empty")

                # USER: Implement protocol decoding logic here
                # This stub returns an empty list - replace with actual decoding
                frames: list[dict[str, object]] = []

                return frames

            def configure(self, **params: object) -> None:
                """Configure decoder parameters.

                Args:
                    **params: Decoder-specific parameters.

                Example:
                    >>> decoder.configure(baudrate=115200, parity='none')
                """
                # USER: Implement configuration logic here
                # Store parameters as instance attributes
                for key, value in params.items():
                    setattr(self, key, value)
    ''')


def _generate_analyzer_stub(template: PluginTemplate, class_name: str) -> str:
    """Generate analyzer plugin stub.

    Args:
        template: Plugin template configuration.
        class_name: Class name for the analyzer.

    Returns:
        Python source code for analyzer stub.
    """
    return textwrap.dedent(f'''\
        """{template.description}

        This analyzer implements custom signal analysis for Oscura.

        References:
            PLUG-008: Plugin Template Generator
        """
        import numpy as np
        from numpy.typing import NDArray


        class {class_name}:
            """Signal analyzer implementation.

            Example:
                >>> analyzer = {class_name}()
                >>> result = analyzer.analyze(signal)

            References:
                PLUG-008: Plugin Template Generator
            """
            def __init__(self) -> None:
                """Initialize analyzer."""
                pass

            def analyze(
                self,
                signal: NDArray[np.float64],
                *,
                sample_rate: float = 1_000_000.0,
            ) -> dict[str, object]:
                """Analyze signal and extract features.

                Args:
                    signal: Input signal array.
                    sample_rate: Sample rate in Hz.

                Returns:
                    Dictionary containing analysis results.

                Raises:
                    ValueError: If signal is empty or invalid.

                Example:
                    >>> signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 1000))
                    >>> result = analyzer.analyze(signal, sample_rate=1000)
                """
                if len(signal) == 0:
                    raise ValueError("Signal cannot be empty")

                # USER: Implement analysis logic here
                # This stub returns placeholder results - replace with actual analysis
                result: dict[str, object] = {{
                    "status": "not_implemented",
                    "sample_count": len(signal),
                    "sample_rate": sample_rate,
                }}

                return result
    ''')


def _generate_loader_stub(template: PluginTemplate, class_name: str) -> str:
    """Generate loader plugin stub.

    Args:
        template: Plugin template configuration.
        class_name: Class name for the loader.

    Returns:
        Python source code for loader stub.
    """
    return textwrap.dedent(f'''\
        """{template.description}

        This loader implements file format loading for Oscura.

        References:
            PLUG-008: Plugin Template Generator
        """
        import numpy as np
        from numpy.typing import NDArray
        from pathlib import Path


        class {class_name}:
            """File format loader implementation.

            Example:
                >>> loader = {class_name}()
                >>> data = loader.load(Path("capture.dat"))

            References:
                PLUG-008: Plugin Template Generator
            """
            def __init__(self) -> None:
                """Initialize loader."""
                pass

            def load(self, file_path: Path) -> dict[str, NDArray[np.float64]]:
                """Load data from file.

                Args:
                    file_path: Path to file to load.

                Returns:
                    Dictionary mapping channel names to signal arrays.

                Raises:
                    FileNotFoundError: If file does not exist.
                    ValueError: If file format is invalid.

                Example:
                    >>> data = loader.load(Path("capture.dat"))
                    >>> print(f"Loaded {{len(data)}} channels")
                """
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {{file_path}}")

                # USER: Implement file loading logic here
                # This stub returns empty data - replace with actual loading
                data: dict[str, NDArray[np.float64]] = {{}}

                return data

            @staticmethod
            def can_load(file_path: Path) -> bool:
                """Check if this loader can handle the file.

                Args:
                    file_path: Path to file.

                Returns:
                    True if loader can handle this file format.

                Example:
                    >>> if loader.can_load(Path("capture.dat")):
                    ...     data = loader.load(Path("capture.dat"))
                """
                # USER: Implement format detection here
                # Check file extension, magic bytes, etc.
                return file_path.suffix == ".dat"
    ''')


def _generate_exporter_stub(template: PluginTemplate, class_name: str) -> str:
    """Generate exporter plugin stub.

    Args:
        template: Plugin template configuration.
        class_name: Class name for the exporter.

    Returns:
        Python source code for exporter stub.
    """
    return textwrap.dedent(f'''\
        """{template.description}

        This exporter implements custom export format for Oscura.

        References:
            PLUG-008: Plugin Template Generator
        """
        import numpy as np
        from numpy.typing import NDArray
        from pathlib import Path


        class {class_name}:
            """Export format implementation.

            Example:
                >>> exporter = {class_name}()
                >>> exporter.export(data, Path("output.dat"))

            References:
                PLUG-008: Plugin Template Generator
            """
            def __init__(self) -> None:
                """Initialize exporter."""
                pass

            def export(
                self,
                data: dict[str, NDArray[np.float64]],
                output_path: Path,
            ) -> None:
                """Export data to file.

                Args:
                    data: Dictionary mapping channel names to signal arrays.
                    output_path: Path where file will be written.

                Raises:
                    ValueError: If data is invalid.
                    OSError: If file cannot be written.

                Example:
                    >>> data = {{"ch1": np.sin(np.linspace(0, 10, 100))}}
                    >>> exporter.export(data, Path("output.dat"))
                """
                if not data:
                    raise ValueError("Data dictionary cannot be empty")

                # USER: Implement export logic here
                # Write data to output_path in your custom format

                # Placeholder implementation - replace with actual export
                with output_path.open("w") as f:
                    f.write("# USER: Implement export format\\n")
                    for name, values in data.items():
                        f.write(f"# Channel: {{name}}, samples: {{len(values)}}\\n")

            @staticmethod
            def supports_format(format_name: str) -> bool:
                """Check if this exporter supports the format.

                Args:
                    format_name: Name of export format.

                Returns:
                    True if format is supported.

                Example:
                    >>> if exporter.supports_format("custom"):
                    ...     exporter.export(data, path)
                """
                # USER: Implement format support detection here
                return format_name == "custom"
    ''')


def _generate_generic_stub(template: PluginTemplate, class_name: str) -> str:
    """Generate generic plugin stub.

    Args:
        template: Plugin template configuration.
        class_name: Class name for the plugin.

    Returns:
        Python source code for generic stub.
    """
    return textwrap.dedent(f'''\
        """{template.description}

        This is a generic plugin implementation for Oscura.

        References:
            PLUG-008: Plugin Template Generator
        """

        class {class_name}:
            """Generic plugin implementation.

            Example:
                >>> plugin = {class_name}()
                >>> result = plugin.process()

            References:
                PLUG-008: Plugin Template Generator
            """
            def __init__(self) -> None:
                """Initialize plugin."""
                pass

            def process(self) -> dict[str, object]:
                """Process data or perform plugin function.

                Returns:
                    Dictionary containing results.

                Example:
                    >>> result = plugin.process()
                """
                # USER: Implement plugin logic here
                result: dict[str, object] = {{
                    "status": "not_implemented",
                }}

                return result
    ''')


__all__ = [
    "PluginTemplate",
    "PluginType",
    "generate_plugin_template",
]
