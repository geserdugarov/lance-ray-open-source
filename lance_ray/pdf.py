"""PDF-to-Markdown conversion utilities.

Thin wrapper around the `opendataloader-pdf <https://github.com/opendataloader-project/opendataloader-pdf>`_
Python package, exposing a focused API for converting PDF files to Markdown.

The ``opendataloader-pdf`` package is an optional dependency. Install it with::

    pip install opendataloader-pdf

or as an extra of ``lance-ray``::

    pip install "lance-ray[pdf]"
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Literal, Optional


def _normalize_input_paths(
    input_path: str | os.PathLike | Iterable[str | os.PathLike],
) -> list[str]:
    """Coerce the user-supplied input into a list of string paths."""
    if isinstance(input_path, (str, os.PathLike)):
        return [os.fspath(input_path)]
    paths = [os.fspath(p) for p in input_path]
    if not paths:
        raise ValueError(
            "'input_path' must contain at least one PDF file or directory."
        )
    return paths


def _import_opendataloader_pdf():
    """Import the optional ``opendataloader_pdf`` dependency with a clear error."""
    try:
        import opendataloader_pdf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via tests with stub
        raise ImportError(
            "PDF-to-Markdown conversion requires the 'opendataloader-pdf' package. "
            "Install it with `pip install opendataloader-pdf` or "
            '`pip install "lance-ray[pdf]"`.'
        ) from exc
    return opendataloader_pdf


def convert_pdf_to_markdown(
    input_path: str | os.PathLike | Iterable[str | os.PathLike],
    output_dir: str | os.PathLike,
    *,
    image_output: Literal["off", "embedded", "external"] = "external",
    image_format: Literal["png", "jpeg"] = "png",
    use_struct_tree: bool = False,
    hybrid: Optional[str] = None,
    sanitize: bool = False,
) -> None:
    """Convert one or more PDF files to Markdown using ``opendataloader-pdf``.

    The underlying library spawns a JVM process per call, so prefer batching
    every PDF into a single invocation rather than calling this function in a
    loop.

    Examples:
        Convert a single PDF::

            import lance_ray as lr

            lr.convert_pdf_to_markdown(
                "./books/sample.pdf",
                output_dir="./out",
            )

        Convert every PDF under a directory with embedded images::

            lr.convert_pdf_to_markdown(
                ["./books"],
                output_dir="./out",
                image_output="embedded",
                image_format="jpeg",
            )

    Args:
        input_path: A single PDF file, a directory containing PDFs, or an
            iterable of such paths. Directories are processed recursively by
            ``opendataloader-pdf``.
        output_dir: Directory where the generated Markdown files (and any
            extracted images) will be written.
        image_output: How extracted images are stored. ``"external"`` writes
            image files next to the Markdown output, ``"embedded"`` inlines
            them as base64 within the Markdown, and ``"off"`` skips images.
        image_format: Image encoding when ``image_output`` is not ``"off"``.
        use_struct_tree: Use the PDF's native structure tags when available,
            which improves heading and table extraction on well-tagged PDFs.
        hybrid: Optional hybrid backend name (e.g. ``"docling-fast"``) that
            enables OCR / AI-assisted extraction. Requires the hybrid extra
            of ``opendataloader-pdf``.
        sanitize: Strip hidden text and other prompt-injection vectors from
            the output.

    Raises:
        ImportError: If the optional ``opendataloader-pdf`` package is not
            installed.
        ValueError: If ``input_path`` is an empty iterable.
    """
    paths = _normalize_input_paths(input_path)
    output_dir = os.fspath(output_dir)

    opendataloader_pdf = _import_opendataloader_pdf()
    opendataloader_pdf.convert(
        input_path=paths,
        output_dir=output_dir,
        format="markdown",
        image_output=image_output,
        image_format=image_format,
        use_struct_tree=use_struct_tree,
        hybrid=hybrid,
        sanitize=sanitize,
    )
