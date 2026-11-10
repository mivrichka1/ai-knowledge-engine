"""Bounded document extraction. PDF page numbers are one-based."""

from dataclasses import dataclass
from io import BytesIO
from zipfile import BadZipFile, ZipFile

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

PDF_PARSER_VERSION = "pdf-columns-v1"


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    page: int | None
    text: str


def _pdf_page_text(page) -> str:
    """Use column order only for a clear two-column layout; otherwise use PyMuPDF's order."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append((line["bbox"], text))
    midpoint = page.rect.width / 2
    left = [line for line in lines if line[0][2] < midpoint]
    right = [line for line in lines if line[0][0] > midpoint]
    crossing = [line for line in lines if line[0][0] <= midpoint <= line[0][2]]
    if len(left) < 2 or len(right) < 2:
        return page.get_text(sort=True)
    overlap_top = max(min(x[0][1] for x in left), min(x[0][1] for x in right))
    overlap_bottom = min(max(x[0][3] for x in left), max(x[0][3] for x in right))
    gap = min(x[0][0] for x in right) - max(x[0][2] for x in left)
    if overlap_bottom <= overlap_top or gap < page.rect.width * 0.02:
        return page.get_text(sort=True)
    if any(x[0][3] > overlap_top and x[0][1] < overlap_bottom for x in crossing):
        return page.get_text(sort=True)
    before = [x for x in crossing if x[0][3] <= overlap_top]
    after = [x for x in crossing if x[0][1] >= overlap_bottom]
    ordered = []
    for group in (before, left, right, after):
        ordered.extend(sorted(group, key=lambda x: (x[0][1], x[0][0])))
    return "\n".join(x[1] for x in ordered)


def parse_document(
    data: bytes,
    format: str,
    max_bytes: int = 10_000_000,
    max_pages: int = 500,
    max_text_chars: int = 2_000_000,
) -> list[Page]:
    if max_bytes <= 0 or max_text_chars <= 0 or not data or len(data) > max_bytes:
        raise ParseError("Empty document or file size limit exceeded")
    if format in {"txt", "md"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ParseError("Text must be UTF-8") from exc
        if "\x00" in text:
            raise ParseError("Invalid text content")
        if len(text) > max_text_chars:
            raise ParseError("Extracted text limit exceeded")
        return [Page(None, text)]
    if format == "pdf":
        if not data.startswith(b"%PDF-"):
            raise ParseError("Invalid PDF signature")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise ParseError("Encrypted PDF is unsupported")
                if not 0 < len(doc) <= max_pages:
                    raise ParseError("PDF page limit exceeded or no pages")
                pages = []
                text_chars = 0
                for i, page in enumerate(doc):
                    page_text = _pdf_page_text(page)
                    text_chars += len(page_text)
                    if text_chars > max_text_chars:
                        raise ParseError("Extracted text limit exceeded")
                    pages.append(Page(i + 1, page_text))
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError("Invalid or unreadable PDF") from exc
        if not any(page.text.strip() for page in pages):
            raise ParseError("No extractable text layer; OCR is required")
        return pages
    if format == "docx":
        try:
            with ZipFile(BytesIO(data)) as archive:
                members = archive.infolist()
                if len(members) > 10000 or sum(m.file_size for m in members) > max_bytes * 10:
                    raise ParseError("DOCX expanded size limit exceeded")
                names = {m.filename for m in members}
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise ParseError("Invalid DOCX structure")
                if any(name.lower().endswith("vbaproject.bin") for name in names):
                    raise ParseError("Macro-enabled documents are unsupported")
            document = Document(BytesIO(data))
            lines = []
            for block in document.iter_inner_content():
                if isinstance(block, Paragraph):
                    lines.append(block.text)
                elif isinstance(block, Table):
                    lines.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
            text = "\n".join(lines)
            if len(text) > max_text_chars:
                raise ParseError("Extracted text limit exceeded")
            return [Page(None, text)]
        except ParseError:
            raise
        except (BadZipFile, Exception) as exc:
            raise ParseError("Invalid or unreadable DOCX") from exc
    raise ParseError("Unsupported document format")
