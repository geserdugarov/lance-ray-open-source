"""Tests for the PDF-to-Markdown conversion wrapper."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lance_ray as lr
import pytest
from lance_ray import pdf as lr_pdf


@pytest.fixture
def fake_opendataloader_pdf():
    """Stand in for the real opendataloader_pdf module and capture convert() calls."""
    calls = []

    def convert(**kwargs):
        calls.append(kwargs)

    fake = SimpleNamespace(convert=convert)
    with patch.object(lr_pdf, "_import_opendataloader_pdf", return_value=fake):
        yield calls


def test_public_export():
    assert lr.convert_pdf_to_markdown is lr_pdf.convert_pdf_to_markdown


def test_convert_single_string_path(fake_opendataloader_pdf, tmp_path):
    lr.convert_pdf_to_markdown("books/sample.pdf", tmp_path)

    assert len(fake_opendataloader_pdf) == 1
    call = fake_opendataloader_pdf[0]
    assert call["input_path"] == ["books/sample.pdf"]
    assert call["output_dir"] == str(tmp_path)
    assert call["format"] == "markdown"
    # Defaults forwarded verbatim
    assert call["image_output"] == "external"
    assert call["image_format"] == "png"
    assert call["use_struct_tree"] is False
    assert call["hybrid"] is None
    assert call["sanitize"] is False


def test_convert_accepts_pathlike(fake_opendataloader_pdf, tmp_path):
    lr.convert_pdf_to_markdown(Path("books/sample.pdf"), Path(tmp_path))

    call = fake_opendataloader_pdf[0]
    assert call["input_path"] == ["books/sample.pdf"]
    assert call["output_dir"] == str(tmp_path)


def test_convert_iterable_of_paths(fake_opendataloader_pdf, tmp_path):
    lr.convert_pdf_to_markdown(
        ["books/a.pdf", Path("books/b.pdf"), "books/"],
        tmp_path,
    )

    call = fake_opendataloader_pdf[0]
    assert call["input_path"] == ["books/a.pdf", "books/b.pdf", "books/"]


def test_convert_forwards_options(fake_opendataloader_pdf, tmp_path):
    lr.convert_pdf_to_markdown(
        "books/sample.pdf",
        tmp_path,
        image_output="embedded",
        image_format="jpeg",
        use_struct_tree=True,
        hybrid="docling-fast",
        sanitize=True,
    )

    call = fake_opendataloader_pdf[0]
    assert call["image_output"] == "embedded"
    assert call["image_format"] == "jpeg"
    assert call["use_struct_tree"] is True
    assert call["hybrid"] == "docling-fast"
    assert call["sanitize"] is True
    # Format is always pinned to markdown regardless of other options.
    assert call["format"] == "markdown"


def test_convert_rejects_empty_iterable(fake_opendataloader_pdf, tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        lr.convert_pdf_to_markdown([], tmp_path)
    assert fake_opendataloader_pdf == []


def test_missing_dependency_raises_clear_error(tmp_path):
    """If opendataloader_pdf is not installed, surface a helpful ImportError."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "opendataloader_pdf":
            raise ImportError("No module named 'opendataloader_pdf'")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        pytest.raises(ImportError, match="opendataloader-pdf"),
    ):
        lr.convert_pdf_to_markdown("books/sample.pdf", tmp_path)
