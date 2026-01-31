"""Tests for epub chapter extraction."""

from pathlib import Path

from vocab.epub import extract_chapters
from vocab.models import Chapter


class TestExtractChapters:
    """Tests for extract_chapters function."""

    def test_extracts_all_chapters(self, sample_epub: Path) -> None:
        """Should extract all non-empty chapters from epub."""
        chapters = list(extract_chapters(sample_epub))

        assert len(chapters) == 3

    def test_chapters_are_in_reading_order(self, sample_epub: Path) -> None:
        """Should yield chapters in spine order with sequential indices."""
        chapters = list(extract_chapters(sample_epub))

        assert chapters[0].index == 0
        assert chapters[1].index == 1
        assert chapters[2].index == 2

    def test_extracts_chapter_titles_from_toc(self, sample_epub: Path) -> None:
        """Should extract chapter titles from table of contents."""
        chapters = list(extract_chapters(sample_epub))

        assert chapters[0].title == "Premier Chapitre"
        assert chapters[1].title == "Deuxième Chapitre"
        assert chapters[2].title == "Troisième Chapitre"

    def test_extracts_text_content(self, sample_epub: Path) -> None:
        """Should extract plain text from HTML content."""
        chapters = list(extract_chapters(sample_epub))

        # First chapter should have paragraph text
        assert "Ceci est le premier paragraphe" in chapters[0].text
        assert "Voici le deuxième paragraphe" in chapters[0].text

    def test_strips_html_tags(self, sample_epub: Path) -> None:
        """Should remove HTML tags from content."""
        chapters = list(extract_chapters(sample_epub))

        # Should not contain any HTML tags
        assert "<p>" not in chapters[0].text
        assert "<h1>" not in chapters[0].text
        assert "</body>" not in chapters[0].text

    def test_removes_script_and_style(self, sample_epub: Path) -> None:
        """Should remove script and style content."""
        chapters = list(extract_chapters(sample_epub))

        # Chapter 2 has script and style elements
        assert "var x = 1" not in chapters[1].text
        assert ".hidden" not in chapters[1].text
        assert "display: none" not in chapters[1].text

    def test_normalizes_whitespace(self, sample_epub: Path) -> None:
        """Should collapse multiple whitespace, preserving single newlines between paragraphs."""
        chapters = list(extract_chapters(sample_epub))

        # Should not have multiple consecutive spaces
        assert "  " not in chapters[0].text
        # Should not have multiple consecutive newlines
        assert "\n\n" not in chapters[0].text
        # But should preserve single newlines between paragraphs
        assert "\n" in chapters[0].text

    def test_returns_chapter_dataclass(self, sample_epub: Path) -> None:
        """Should return Chapter dataclass instances."""
        chapters = list(extract_chapters(sample_epub))

        assert all(isinstance(ch, Chapter) for ch in chapters)

    def test_skips_empty_chapters(self, empty_chapter_epub: Path) -> None:
        """Should skip chapters that contain only whitespace."""
        chapters = list(extract_chapters(empty_chapter_epub))

        # Should only have 2 chapters (empty one skipped)
        assert len(chapters) == 2
        assert chapters[0].title == "Chapter One"
        assert chapters[1].title == "Chapter Three"

    def test_empty_chapter_indices_are_sequential(self, empty_chapter_epub: Path) -> None:
        """Should maintain sequential indices even when skipping empty chapters."""
        chapters = list(extract_chapters(empty_chapter_epub))

        # Indices should be 0, 1 (not 0, 2)
        assert chapters[0].index == 0
        assert chapters[1].index == 1


class TestNestedToc:
    """Tests for nested table of contents handling."""

    def test_extracts_titles_from_nested_toc(self, tmp_path: Path) -> None:
        """Should extract titles from nested TOC sections."""
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-nested")
        book.set_title("Test")
        book.set_language("en")

        ch1 = epub.EpubHtml(title="Part 1", file_name="part1.xhtml", lang="en")
        ch1.content = "<html><body><p>Part 1 intro.</p></body></html>"
        book.add_item(ch1)

        ch2 = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
        ch2.content = "<html><body><p>Chapter 1 content.</p></body></html>"
        book.add_item(ch2)

        ch3 = epub.EpubHtml(title="Chapter 2", file_name="ch2.xhtml", lang="en")
        ch3.content = "<html><body><p>Chapter 2 content.</p></body></html>"
        book.add_item(ch3)

        # Nested TOC structure: Part 1 -> [Chapter 1, Chapter 2]
        book.toc = [
            (
                epub.Section("Part One", "part1.xhtml"),
                [
                    epub.Link("ch1.xhtml", "Chapter One", "ch1"),
                    epub.Link("ch2.xhtml", "Chapter Two", "ch2"),
                ],
            )
        ]

        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1, ch2, ch3]

        epub_path = tmp_path / "nested_toc.epub"
        epub.write_epub(str(epub_path), book)

        chapters = list(extract_chapters(epub_path))

        assert len(chapters) == 3
        assert chapters[0].title == "Part One"
        assert chapters[1].title == "Chapter One"
        assert chapters[2].title == "Chapter Two"


class TestTextExtraction:
    """Tests for text extraction preserving paragraph boundaries."""

    def test_preserves_paragraph_boundaries_as_newlines(self, tmp_path: Path) -> None:
        """Should insert newlines between paragraphs for sentence detection."""
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-paragraphs")
        book.set_title("Test")
        book.set_language("fr")

        ch1 = epub.EpubHtml(title="Chapter", file_name="ch1.xhtml", lang="fr")
        ch1.content = """
        <html>
        <body>
            <p>Premier paragraphe.</p>
            <p>Deuxième paragraphe.</p>
            <p>Troisième paragraphe.</p>
        </body>
        </html>
        """
        book.add_item(ch1)

        book.toc = [epub.Link("ch1.xhtml", "Chapter", "ch1")]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1]

        epub_path = tmp_path / "paragraphs.epub"
        epub.write_epub(str(epub_path), book)

        chapters = list(extract_chapters(epub_path))

        # Paragraphs should be separated by newlines, not just spaces
        assert "Premier paragraphe.\n" in chapters[0].text
        assert "Deuxième paragraphe.\n" in chapters[0].text


class TestMalformedEpub:
    """Tests for handling malformed epub files."""

    def test_skips_spine_items_with_missing_ids(self, tmp_path: Path) -> None:
        """Should skip spine entries that reference non-existent items."""
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-malformed")
        book.set_title("Test")
        book.set_language("en")

        ch1 = epub.EpubHtml(title="Chapter", file_name="ch1.xhtml", lang="en")
        ch1.content = "<html><body><p>Valid content.</p></body></html>"
        book.add_item(ch1)

        book.toc = [epub.Link("ch1.xhtml", "Chapter", "ch1")]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Add a fake item_id that doesn't exist
        book.spine = ["nav", ch1]
        book.spine.append(("nonexistent_item_id", "yes"))

        epub_path = tmp_path / "malformed.epub"
        epub.write_epub(str(epub_path), book)

        chapters = list(extract_chapters(epub_path))

        # Should only get the valid chapter, skipping the non-existent one
        assert len(chapters) == 1
        assert "Valid content" in chapters[0].text

    def test_skips_non_document_items_in_spine(self, tmp_path: Path) -> None:
        """Should skip spine entries that are not document items (e.g., CSS)."""
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-non-document")
        book.set_title("Test")
        book.set_language("en")

        ch1 = epub.EpubHtml(title="Chapter", file_name="ch1.xhtml", lang="en")
        ch1.content = "<html><body><p>Valid content.</p></body></html>"
        book.add_item(ch1)

        # Add a CSS item (non-document type)
        css = epub.EpubItem(
            uid="style",
            file_name="style.css",
            media_type="text/css",
            content=b"body { color: red; }",
        )
        book.add_item(css)

        book.toc = [epub.Link("ch1.xhtml", "Chapter", "ch1")]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Include the CSS item in the spine (malformed but possible)
        book.spine = ["nav", ch1]
        book.spine.append(("style", "yes"))

        epub_path = tmp_path / "non_document_spine.epub"
        epub.write_epub(str(epub_path), book)

        chapters = list(extract_chapters(epub_path))

        # Should only get the valid chapter, skipping the CSS item
        assert len(chapters) == 1
        assert "Valid content" in chapters[0].text


class TestChapterTitleExtraction:
    def test_title_from_h1_when_no_toc(self, tmp_path: Path) -> None:
        """Should fall back to h1 tag when not in TOC."""
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-003")
        book.set_title("Test")
        book.set_language("en")

        ch1 = epub.EpubHtml(title="Chapter", file_name="ch1.xhtml", lang="en")
        ch1.content = """
        <html>
        <body>
            <h1>Title from H1</h1>
            <p>Content.</p>
        </body>
        </html>
        """
        book.add_item(ch1)

        # Empty TOC - no title mapping
        book.toc = []
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1]

        epub_path = tmp_path / "no_toc.epub"
        epub.write_epub(str(epub_path), book)

        chapters = list(extract_chapters(epub_path))

        assert chapters[0].title == "Title from H1"

    def test_title_from_title_tag_fallback(self) -> None:
        """Should fall back to title tag when no h1.

        Note: This tests _extract_title_from_html directly because ebooklib
        strips <title> tags from head when writing epubs. Real-world epubs
        from other tools may preserve title tags.
        """
        from vocab.epub import _extract_title_from_html

        html = b"""
        <html>
        <head><title>Title from Title Tag</title></head>
        <body>
            <p>Content without h1.</p>
        </body>
        </html>
        """

        title = _extract_title_from_html(html)

        assert title == "Title from Title Tag"

    def test_none_title_when_no_title_source(self, tmp_path: Path) -> None:
        """Should return None when no title source available."""
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-005")
        book.set_title("Test")
        book.set_language("en")

        ch1 = epub.EpubHtml(title="Chapter", file_name="ch1.xhtml", lang="en")
        ch1.content = """
        <html>
        <body>
            <p>Content with no title anywhere.</p>
        </body>
        </html>
        """
        book.add_item(ch1)

        book.toc = []
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1]

        epub_path = tmp_path / "no_title.epub"
        epub.write_epub(str(epub_path), book)

        chapters = list(extract_chapters(epub_path))

        assert chapters[0].title is None
