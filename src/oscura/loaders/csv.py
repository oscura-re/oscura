"""Alias module for CSV loader - provides expected import path.

Why two CSV loader files exist:
    - `oscura/loaders/csv_loader.py`: Canonical implementation with full CSV
      parsing logic, format detection, and validation
    - `oscura/loaders/csv.py` (this file): Convenience re-export to support
      natural import syntax like `from oscura.loaders.csv import load_csv`

The name `csv_loader.py` avoids shadowing Python's built-in `csv` module
within the implementation file. This alias module provides a cleaner import
path for external users.

Usage patterns:
    # Recommended - explicit module name
    from oscura.loaders.csv_loader import load_csv

    # Also supported - natural import path via this alias
    from oscura.loaders.csv import load_csv

    # Via loaders package __init__
    from oscura.loaders import load_csv
"""

from oscura.loaders.csv_loader import load_csv

__all__ = ["load_csv"]
