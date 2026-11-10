from io import BytesIO

import pymupdf
import pytest
from docx import Document

from ake_backend.parsers import ParseError, parse_document


def test_pdf_pages_and_scanned_detection():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "First page")
    doc.new_page().insert_text((72, 72), "Second page")
    data = doc.tobytes()
    doc.close()
    pages = parse_document(data, "pdf")
    assert [(p.page, p.text.strip()) for p in pages] == [(1, "First page"), (2, "Second page")]
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    with pytest.raises(ParseError, match="OCR"):
        parse_document(data, "pdf")


def test_docx_paragraphs_and_table_preserve_order():
    doc = Document()
    doc.add_paragraph("Before")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    doc.add_paragraph("After")
    buffer = BytesIO()
    doc.save(buffer)
    pages = parse_document(buffer.getvalue(), "docx")
    assert pages[0].text == "Before\nA\tB\nAfter"


def test_reject_invalid_and_limit():
    with pytest.raises(ParseError):
        parse_document(b"not a pdf", "pdf")
    with pytest.raises(ParseError):
        parse_document(b"x" * 21, "txt", max_bytes=20)
    with pytest.raises(ParseError):
        parse_document(b"\x00", "txt")


def test_pdf_page_limit_and_encryption():
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page().insert_text((72, 72), "Searchable content")
    plain = doc.tobytes()
    encrypted = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner-secret", user_pw="user-secret"
    )
    doc.close()
    with pytest.raises(ParseError, match="page limit"):
        parse_document(plain, "pdf", max_pages=1)
    with pytest.raises(ParseError, match="Encrypted"):
        parse_document(encrypted, "pdf")


def test_docx_macro_and_decompression_limits():
    from zipfile import ZIP_DEFLATED, ZipFile

    data = BytesIO()
    with ZipFile(data, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "x")
        archive.writestr("word/document.xml", "x")
        archive.writestr("word/vbaProject.bin", "macro")
    with pytest.raises(ParseError, match="Macro"):
        parse_document(data.getvalue(), "docx")
    compressed = BytesIO()
    with ZipFile(compressed, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "x")
        archive.writestr("word/document.xml", "a" * 50000)
    with pytest.raises(ParseError, match="expanded size"):
        parse_document(compressed.getvalue(), "docx", max_bytes=2000)


def test_plain_text_malformed_and_bom():
    assert parse_document(b"\xef\xbb\xbfHello", "md")[0].text == "Hello"
    with pytest.raises(ParseError, match="UTF-8"):
        parse_document(b"\xff", "txt")
    with pytest.raises(ParseError, match="Unsupported"):
        parse_document(b"text", "other")


def test_extracted_text_limit_applies_to_text_and_pdf():
    with pytest.raises(ParseError, match="text limit"):
        parse_document(b"abcdefgh", "txt", max_text_chars=4)
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "This document is longer than four characters.")
    data = pdf.tobytes()
    pdf.close()
    with pytest.raises(ParseError, match="text limit"):
        parse_document(data, "pdf", max_text_chars=4)


def test_two_column_pdf_is_read_column_by_column():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 50), "Document title")
    page.insert_text((40, 120), "LEFT FIRST")
    page.insert_text((340, 120), "RIGHT FIRST")
    page.insert_text((40, 180), "LEFT SECOND")
    page.insert_text((340, 180), "RIGHT SECOND")
    data = doc.tobytes()
    doc.close()
    text = parse_document(data, "pdf")[0].text
    assert (
        text.index("LEFT FIRST")
        < text.index("LEFT SECOND")
        < text.index("RIGHT FIRST")
        < text.index("RIGHT SECOND")
    )


def test_single_column_pdf_keeps_reading_order():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 100), "First paragraph")
    page.insert_text((40, 150), "Second paragraph")
    page.insert_text((40, 200), "Third paragraph")
    data = doc.tobytes()
    doc.close()
    text = parse_document(data, "pdf")[0].text
    assert (
        text.index("First paragraph")
        < text.index("Second paragraph")
        < text.index("Third paragraph")
    )


def test_pdf_with_full_width_middle_content_keeps_default_layout():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 100), "LEFT FIRST")
    page.insert_text((340, 100), "RIGHT FIRST")
    page.insert_text((40, 140), "LEFT SECOND")
    page.insert_text((340, 140), "RIGHT SECOND")
    page.insert_text((180, 120), "FULL WIDTH CONTENT THAT CROSSES THE MIDDLE OF THE PAGE")
    expected = page.get_text(sort=True)
    data = doc.tobytes()
    doc.close()
    assert parse_document(data, "pdf")[0].text == expected
