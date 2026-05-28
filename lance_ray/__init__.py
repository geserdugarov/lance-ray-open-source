"""
Lance-Ray: Ray integration for Lance columnar format.

This package provides integration between Ray and Lance for distributed
columnar data processing.
"""

__version__ = "0.4.2"
__author__ = "LanceDB Devs"
__email__ = "dev@lancedb.com"
from .compaction import compact_database, compact_files

# Main imports
from .datasink import LanceFragmentCommitter

# Fragment API imports
from .fragment import LanceFragmentWriter
from .index import create_index, create_scalar_index, optimize_indices
from .io import add_columns, read_lance, write_lance
from .pdf import convert_pdf_to_markdown

__all__ = [
    "read_lance",
    "write_lance",
    "add_columns",
    "create_scalar_index",
    "create_index",
    "optimize_indices",
    "compact_files",
    "compact_database",
    "convert_pdf_to_markdown",
    "LanceFragmentWriter",
    "LanceFragmentCommitter",
]
