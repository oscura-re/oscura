"""Extension documentation auto-generation system.

This module provides automatic documentation generation for Oscura extensions
including API reference, usage examples, and metadata extraction from docstrings.


Example:
    >>> from oscura.core.extensibility.docs import generate_extension_docs
    >>> from pathlib import Path
    >>>
    >>> # Generate documentation for an extension
    >>> docs = generate_extension_docs(Path("my_plugin/"))
    >>> print(docs.markdown)
"""

from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FunctionDoc:
    """Documentation for a function or method.

    Attributes:
        name: Function name
        signature: Full function signature
        docstring: Function docstring
        parameters: List of parameter descriptions
        returns: Return value description
        examples: Code examples from docstring
    """

    name: str
    signature: str = ""
    docstring: str = ""
    parameters: list[tuple[str, str]] = field(default_factory=list)
    returns: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class ClassDoc:
    """Documentation for a class.

    Attributes:
        name: Class name
        docstring: Class docstring
        methods: List of public method documentation
        attributes: List of class/instance attributes
        bases: List of base class names
    """

    name: str
    docstring: str = ""
    methods: list[FunctionDoc] = field(default_factory=list)
    attributes: list[tuple[str, str]] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)


@dataclass
class ModuleDoc:
    """Documentation for a Python module.

    Attributes:
        name: Module name
        docstring: Module docstring
        classes: List of class documentation
        functions: List of function documentation
        path: Source file path
    """

    name: str
    docstring: str = ""
    classes: list[ClassDoc] = field(default_factory=list)
    functions: list[FunctionDoc] = field(default_factory=list)
    path: str = ""


@dataclass
class ExtensionDocs:
    """Complete documentation for an extension.

    Attributes:
        name: Extension name
        version: Extension version
        description: Extension description
        author: Extension author
        modules: List of module documentation
        metadata: Extension metadata
        markdown: Generated markdown documentation
        html: Generated HTML documentation
    """

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    modules: list[ModuleDoc] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    markdown: str = ""
    html: str = ""


def generate_extension_docs(
    extension_path: Path,
    *,
    include_private: bool = False,
    include_examples: bool = True,
    output_format: str = "markdown",
) -> ExtensionDocs:
    """Generate documentation for an extension.

    Extracts documentation from Python modules, docstrings, and metadata files
    to create comprehensive API documentation.

    Args:
        extension_path: Path to extension directory
        include_private: Include private members (starting with _)
        include_examples: Extract examples from docstrings
        output_format: Output format ("markdown" or "html")

    Returns:
        ExtensionDocs object with generated documentation

    Example:
        >>> from pathlib import Path
        >>> docs = generate_extension_docs(Path("plugins/my_decoder/"))
        >>> print(docs.markdown)
        >>> with open("docs/my_decoder.md", "w") as f:
        ...     f.write(docs.markdown)

    References:
        EXT-006: Extension Documentation
    """
    docs = ExtensionDocs(name=extension_path.name)

    # Extract metadata
    _extract_metadata(extension_path, docs)

    # Document Python modules
    _document_modules(extension_path, docs, include_private, include_examples)

    # Generate output
    if output_format == "markdown":
        docs.markdown = _generate_markdown(docs)
    elif output_format == "html":
        docs.html = _generate_html(docs)

    return docs


def generate_decoder_docs(
    decoder_class: type,
    *,
    include_examples: bool = True,
) -> str:
    """Generate documentation for a decoder class.

    Args:
        decoder_class: Decoder class to document
        include_examples: Include usage examples

    Returns:
        Markdown documentation string

    Example:
        >>> class MyDecoder:
        ...     '''Custom UART decoder.
        ...
        ...     Example:
        ...         >>> decoder = MyDecoder()
        ...         >>> frames = decoder.decode(signal)
        ...     '''
        ...     def decode(self, signal):
        ...         '''Decode signal.'''
        ...         return []
        >>> docs = generate_decoder_docs(MyDecoder)
        >>> print(docs)

    References:
        EXT-006: Extension Documentation
    """
    class_doc = _document_class(
        decoder_class, include_private=False, include_examples=include_examples
    )

    # Generate markdown
    lines = []
    lines.append(f"# {class_doc.name}")
    lines.append("")

    if class_doc.docstring:
        lines.append(class_doc.docstring)
        lines.append("")

    if class_doc.bases:
        lines.append(f"**Inherits from:** {', '.join(class_doc.bases)}")
        lines.append("")

    # Attributes
    if class_doc.attributes:
        lines.append("## Attributes")
        lines.append("")
        for name, desc in class_doc.attributes:
            lines.append(f"- **{name}**: {desc}")
        lines.append("")

    # Methods
    if class_doc.methods:
        lines.append("## Methods")
        lines.append("")
        for method in class_doc.methods:
            lines.append(f"### {method.name}")
            lines.append("")
            if method.signature:
                lines.append("```python")
                lines.append(f"{method.signature}")
                lines.append("```")
                lines.append("")
            if method.docstring:
                lines.append(method.docstring)
                lines.append("")
            if method.parameters:
                lines.append("**Parameters:**")
                lines.append("")
                for param_name, param_desc in method.parameters:
                    lines.append(f"- **{param_name}**: {param_desc}")
                lines.append("")
            if method.returns:
                lines.append(f"**Returns:** {method.returns}")
                lines.append("")
            if method.examples:
                lines.append("**Example:**")
                lines.append("")
                for example in method.examples:
                    lines.append("```python")
                    lines.append(example)
                    lines.append("```")
                    lines.append("")

    return "\n".join(lines)


def extract_plugin_metadata(
    extension_path: Path,
) -> dict[str, Any]:
    """Extract metadata from extension directory.

    Args:
        extension_path: Path to extension directory

    Returns:
        Dictionary with metadata fields

    Example:
        >>> metadata = extract_plugin_metadata(Path("plugins/my_plugin/"))
        >>> print(metadata["name"])
        >>> print(metadata["version"])

    References:
        EXT-006: Extension Documentation
    """
    metadata: dict[str, Any] = {}

    # Try pyproject.toml
    pyproject = extension_path / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib

            with open(pyproject, "rb") as f:
                data = tomllib.load(f)

            if "project" in data:
                project = data["project"]
                metadata.update(
                    {
                        "name": project.get("name", ""),
                        "version": project.get("version", ""),
                        "description": project.get("description", ""),
                        "authors": project.get("authors", []),
                        "dependencies": project.get("dependencies", []),
                        "entry_points": project.get("entry-points", {}),
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml: {e}")

    # Try plugin.yaml
    plugin_yaml = extension_path / "plugin.yaml"
    if plugin_yaml.exists():
        try:
            import yaml

            with open(plugin_yaml, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data:
                metadata.update(data)

        except Exception as e:
            logger.warning(f"Failed to parse plugin.yaml: {e}")

    return metadata


def _extract_metadata(extension_path: Path, docs: ExtensionDocs) -> None:
    """Extract metadata from extension files.

    Args:
        extension_path: Path to extension directory
        docs: ExtensionDocs to populate
    """
    metadata = extract_plugin_metadata(extension_path)

    docs.name = metadata.get("name", extension_path.name)
    docs.version = metadata.get("version", "0.1.0")
    docs.description = metadata.get("description", "")
    docs.metadata = metadata

    # Extract author
    authors = metadata.get("authors", [])
    if authors and isinstance(authors, list) and len(authors) > 0:
        if isinstance(authors[0], dict):
            docs.author = authors[0].get("name", "")
        else:
            docs.author = str(authors[0])


def _document_modules(
    extension_path: Path,
    docs: ExtensionDocs,
    include_private: bool,
    include_examples: bool,
) -> None:
    """Document all Python modules in extension.

    Args:
        extension_path: Path to extension directory
        docs: ExtensionDocs to populate
        include_private: Include private members
        include_examples: Extract examples from docstrings
    """
    # Find Python files
    py_files = list(extension_path.glob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]

    for py_file in py_files:
        try:
            module_doc = _document_module(py_file, include_private, include_examples)
            docs.modules.append(module_doc)
        except Exception as e:
            logger.warning(f"Failed to document {py_file}: {e}")


def _document_module(
    module_path: Path,
    include_private: bool,
    include_examples: bool,
) -> ModuleDoc:
    """Document a Python module.

    Args:
        module_path: Path to Python file
        include_private: Include private members
        include_examples: Extract examples

    Returns:
        ModuleDoc with extracted documentation
    """
    with open(module_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    module_doc = ModuleDoc(
        name=module_path.stem,
        path=str(module_path),
        docstring=ast.get_docstring(tree) or "",
    )

    # Extract classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if include_private or not node.name.startswith("_"):
                class_doc = _document_class_ast(node, include_private, include_examples)
                module_doc.classes.append(class_doc)

        elif isinstance(node, ast.FunctionDef):
            if include_private or not node.name.startswith("_"):
                func_doc = _document_function_ast(node, include_examples)
                module_doc.functions.append(func_doc)

    return module_doc


def _document_class(
    cls: type,
    include_private: bool,
    include_examples: bool,
) -> ClassDoc:
    """Document a class from runtime object.

    Args:
        cls: Class to document
        include_private: Include private members
        include_examples: Extract examples

    Returns:
        ClassDoc with extracted documentation
    """
    class_doc = ClassDoc(
        name=cls.__name__,
        docstring=inspect.getdoc(cls) or "",
        bases=[base.__name__ for base in cls.__bases__ if base is not object],
    )

    # Document methods
    for name, obj in inspect.getmembers(cls):
        if include_private or not name.startswith("_"):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                try:
                    sig = str(inspect.signature(obj))
                    func_doc = FunctionDoc(
                        name=name,
                        signature=f"def {name}{sig}",
                        docstring=inspect.getdoc(obj) or "",
                    )
                    class_doc.methods.append(func_doc)
                except Exception:
                    pass

    return class_doc


def _document_class_ast(
    node: ast.ClassDef,
    include_private: bool,
    include_examples: bool,
) -> ClassDoc:
    """Document a class from AST node.

    Args:
        node: AST ClassDef node
        include_private: Include private members
        include_examples: Extract examples

    Returns:
        ClassDoc with extracted documentation
    """
    class_doc = ClassDoc(
        name=node.name,
        docstring=ast.get_docstring(node) or "",
        bases=[_get_name_from_ast(base) for base in node.bases],
    )

    # Document methods
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            if include_private or not item.name.startswith("_"):
                func_doc = _document_function_ast(item, include_examples)
                class_doc.methods.append(func_doc)

    return class_doc


def _document_function_ast(
    node: ast.FunctionDef,
    include_examples: bool,
) -> FunctionDoc:
    """Document a function from AST node.

    Args:
        node: AST FunctionDef node
        include_examples: Extract examples

    Returns:
        FunctionDoc with extracted documentation
    """
    # Build signature
    args = []
    for arg in node.args.args:
        args.append(arg.arg)

    signature = f"def {node.name}({', '.join(args)})"

    func_doc = FunctionDoc(
        name=node.name,
        signature=signature,
        docstring=ast.get_docstring(node) or "",
    )

    # Parse docstring for parameters, returns, examples
    if func_doc.docstring:
        _parse_docstring(func_doc, include_examples)

    return func_doc


def _parse_docstring(func_doc: FunctionDoc, include_examples: bool) -> None:
    """Parse Google-style docstring for structured information.

    Args:
        func_doc: FunctionDoc to populate
        include_examples: Extract examples
    """
    lines = func_doc.docstring.split("\n")
    current_section = None
    section_content: list[str] = []

    for line in lines:
        line_stripped = line.strip()

        # Detect sections
        if line_stripped.endswith(":") and line_stripped[:-1] in [
            "Args",
            "Arguments",
            "Parameters",
            "Returns",
            "Return",
            "Example",
            "Examples",
        ]:
            # Process previous section
            if current_section:
                _process_section(func_doc, current_section, section_content, include_examples)

            current_section = line_stripped[:-1].lower()
            section_content = []
        else:
            section_content.append(line)

    # Process final section
    if current_section:
        _process_section(func_doc, current_section, section_content, include_examples)


def _process_section(
    func_doc: FunctionDoc,
    section: str,
    content: list[str],
    include_examples: bool,
) -> None:
    """Process a docstring section.

    Args:
        func_doc: FunctionDoc to populate
        section: Section name
        content: Section content lines
        include_examples: Extract examples
    """
    if section in ["args", "arguments", "parameters"]:
        _process_parameters_section(func_doc, content)
    elif section in ["returns", "return"]:
        _process_returns_section(func_doc, content)
    elif section in ["example", "examples"] and include_examples:
        _process_examples_section(func_doc, content)


def _process_parameters_section(func_doc: FunctionDoc, content: list[str]) -> None:
    """Process parameters section.

    Args:
        func_doc: FunctionDoc to populate.
        content: Section content lines.
    """
    for line in content:
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            param_name = parts[0].strip()
            param_desc = parts[1].strip()
            func_doc.parameters.append((param_name, param_desc))


def _process_returns_section(func_doc: FunctionDoc, content: list[str]) -> None:
    """Process returns section.

    Args:
        func_doc: FunctionDoc to populate.
        content: Section content lines.
    """
    func_doc.returns = "\n".join(content).strip()


def _process_examples_section(func_doc: FunctionDoc, content: list[str]) -> None:
    """Process examples section.

    Args:
        func_doc: FunctionDoc to populate.
        content: Section content lines.
    """
    in_code = False
    code_lines = []

    for line in content:
        if ">>>" in line or "..." in line:
            in_code = True
            code_lines.append(line.strip())
        elif in_code:
            if line.strip() and not line.strip().startswith("#"):
                if not (">>>" in line or "..." in line):
                    in_code = False
                    if code_lines:
                        func_doc.examples.append("\n".join(code_lines))
                        code_lines = []
                else:
                    code_lines.append(line.strip())

    if code_lines:
        func_doc.examples.append("\n".join(code_lines))


def _get_name_from_ast(node: ast.expr) -> str:
    """Extract name from AST expression.

    Args:
        node: AST expression node

    Returns:
        Name string
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    else:
        return str(node)


def _generate_markdown(docs: ExtensionDocs) -> str:
    """Generate markdown documentation.

    Args:
        docs: ExtensionDocs to render

    Returns:
        Markdown string
    """
    lines: list[str] = []

    # Title and metadata
    _add_markdown_header(lines, docs)

    # Dependencies
    _add_markdown_dependencies(lines, docs)

    # Modules
    _add_markdown_modules(lines, docs)

    return "\n".join(lines)


def _add_markdown_header(lines: list[str], docs: ExtensionDocs) -> None:
    """Add header section to markdown.

    Args:
        lines: List to append lines to.
        docs: ExtensionDocs to render.
    """
    lines.append(f"# {docs.name}")
    lines.append("")

    if docs.version:
        lines.append(f"**Version:** {docs.version}")
        lines.append("")
    if docs.author:
        lines.append(f"**Author:** {docs.author}")
        lines.append("")
    if docs.description:
        lines.append(docs.description)
        lines.append("")


def _add_markdown_dependencies(lines: list[str], docs: ExtensionDocs) -> None:
    """Add dependencies section to markdown.

    Args:
        lines: List to append lines to.
        docs: ExtensionDocs to render.
    """
    if "dependencies" in docs.metadata:
        lines.append("## Dependencies")
        lines.append("")
        for dep in docs.metadata["dependencies"]:
            lines.append(f"- {dep}")
        lines.append("")


def _add_markdown_modules(lines: list[str], docs: ExtensionDocs) -> None:
    """Add modules section to markdown.

    Args:
        lines: List to append lines to.
        docs: ExtensionDocs to render.
    """
    for module in docs.modules:
        lines.append(f"## Module: {module.name}")
        lines.append("")
        if module.docstring:
            lines.append(module.docstring)
            lines.append("")

        # Classes
        _add_markdown_classes(lines, module)

        # Functions
        _add_markdown_functions(lines, module)


def _add_markdown_classes(lines: list[str], module: ModuleDoc) -> None:
    """Add classes section to markdown.

    Args:
        lines: List to append lines to.
        module: ModuleDoc to render.
    """
    for cls in module.classes:
        lines.append(f"### Class: {cls.name}")
        lines.append("")
        if cls.docstring:
            lines.append(cls.docstring)
            lines.append("")

        # Methods
        if cls.methods:
            lines.append("#### Methods")
            lines.append("")
            _add_markdown_methods(lines, cls.methods)


def _add_markdown_methods(lines: list[str], methods: list[FunctionDoc]) -> None:
    """Add methods section to markdown.

    Args:
        lines: List to append lines to.
        methods: List of method documentation.
    """
    for method in methods:
        lines.append(f"##### {method.name}")
        lines.append("")
        _add_markdown_function_details(lines, method)


def _add_markdown_functions(lines: list[str], module: ModuleDoc) -> None:
    """Add functions section to markdown.

    Args:
        lines: List to append lines to.
        module: ModuleDoc to render.
    """
    for func in module.functions:
        lines.append(f"### Function: {func.name}")
        lines.append("")
        _add_markdown_function_details(lines, func)


def _add_markdown_function_details(lines: list[str], func: FunctionDoc) -> None:
    """Add function/method details to markdown.

    Args:
        lines: List to append lines to.
        func: FunctionDoc to render.
    """
    if func.signature:
        lines.append("```python")
        lines.append(func.signature)
        lines.append("```")
        lines.append("")
    if func.docstring:
        lines.append(func.docstring)
        lines.append("")


def _generate_html(docs: ExtensionDocs) -> str:
    """Generate HTML documentation.

    Args:
        docs: ExtensionDocs to render

    Returns:
        HTML string
    """
    # Convert markdown to HTML (simple conversion)
    markdown = _generate_markdown(docs)

    # Simple markdown-to-HTML conversion
    html_lines = ["<!DOCTYPE html>", "<html>", "<head>"]
    html_lines.append(f"<title>{docs.name} Documentation</title>")
    html_lines.append("<style>")
    html_lines.append("body { font-family: Arial, sans-serif; margin: 40px; }")
    html_lines.append("code { background: #f4f4f4; padding: 2px 4px; }")
    html_lines.append("pre { background: #f4f4f4; padding: 10px; }")
    html_lines.append("</style>")
    html_lines.append("</head>")
    html_lines.append("<body>")

    # Very simple markdown-to-HTML
    for line in markdown.split("\n"):
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("```"):
            # Toggle code block
            if "<pre><code>" not in html_lines[-1] if html_lines else "":
                html_lines.append("<pre><code>")
            else:
                html_lines.append("</code></pre>")
        elif line.strip():
            html_lines.append(f"<p>{line}</p>")

    html_lines.append("</body>")
    html_lines.append("</html>")

    return "\n".join(html_lines)


__all__ = [
    "ClassDoc",
    "ExtensionDocs",
    "FunctionDoc",
    "ModuleDoc",
    "extract_plugin_metadata",
    "generate_decoder_docs",
    "generate_extension_docs",
]
