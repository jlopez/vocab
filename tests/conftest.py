"""Pytest fixtures for vocab tests."""

from pathlib import Path

import pytest
from ebooklib import epub


@pytest.fixture
def sample_epub(tmp_path: Path) -> Path:
    """Create a sample epub file for testing.

    Creates an epub with:
    - 3 chapters with titles and content
    - A table of contents
    - Various HTML elements to test text extraction

    Returns:
        Path to the created epub file.
    """
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier("test-book-001")
    book.set_title("Test Book")
    book.set_language("fr")
    book.add_author("Test Author")

    # Chapter 1
    ch1 = epub.EpubHtml(title="Premier Chapitre", file_name="ch1.xhtml", lang="fr")
    ch1.content = """
    <html>
    <head><title>Premier Chapitre</title></head>
    <body>
        <h1>Premier Chapitre</h1>
        <p>Ceci est le premier paragraphe.</p>
        <p>Voici le deuxième paragraphe avec plus de texte.</p>
    </body>
    </html>
    """
    book.add_item(ch1)

    # Chapter 2
    ch2 = epub.EpubHtml(title="Deuxième Chapitre", file_name="ch2.xhtml", lang="fr")
    ch2.content = """
    <html>
    <head><title>Deuxième Chapitre</title></head>
    <body>
        <h1>Deuxième Chapitre</h1>
        <p>Le texte du deuxième chapitre.</p>
        <script>var x = 1;</script>
        <style>.hidden { display: none; }</style>
        <p>Encore du texte après le script.</p>
    </body>
    </html>
    """
    book.add_item(ch2)

    # Chapter 3 - no h1, title from TOC only
    ch3 = epub.EpubHtml(title="Troisième Chapitre", file_name="ch3.xhtml", lang="fr")
    ch3.content = """
    <html>
    <head><title>Troisième Chapitre</title></head>
    <body>
        <p>Ce chapitre n'a pas de titre h1.</p>
        <p>Mais il a du contenu intéressant.</p>
    </body>
    </html>
    """
    book.add_item(ch3)

    # Table of contents
    book.toc = [
        epub.Link("ch1.xhtml", "Premier Chapitre", "ch1"),
        epub.Link("ch2.xhtml", "Deuxième Chapitre", "ch2"),
        epub.Link("ch3.xhtml", "Troisième Chapitre", "ch3"),
    ]

    # Add navigation files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Spine (reading order)
    book.spine = ["nav", ch1, ch2, ch3]

    # Write epub
    epub_path = tmp_path / "test_book.epub"
    epub.write_epub(str(epub_path), book)

    return epub_path


@pytest.fixture
def empty_chapter_epub(tmp_path: Path) -> Path:
    """Create an epub with an empty chapter that should be skipped.

    Returns:
        Path to the created epub file.
    """
    book = epub.EpubBook()
    book.set_identifier("test-book-002")
    book.set_title("Test Book with Empty Chapter")
    book.set_language("fr")

    # Non-empty chapter
    ch1 = epub.EpubHtml(title="Chapter One", file_name="ch1.xhtml", lang="fr")
    ch1.content = """
    <html>
    <body>
        <h1>Chapter One</h1>
        <p>Some content here.</p>
    </body>
    </html>
    """
    book.add_item(ch1)

    # Empty chapter (only whitespace)
    ch2 = epub.EpubHtml(title="Empty Chapter", file_name="ch2.xhtml", lang="fr")
    ch2.content = """
    <html>
    <body>
        <p>   </p>
    </body>
    </html>
    """
    book.add_item(ch2)

    # Another non-empty chapter
    ch3 = epub.EpubHtml(title="Chapter Three", file_name="ch3.xhtml", lang="fr")
    ch3.content = """
    <html>
    <body>
        <h1>Chapter Three</h1>
        <p>More content.</p>
    </body>
    </html>
    """
    book.add_item(ch3)

    book.toc = [
        epub.Link("ch1.xhtml", "Chapter One", "ch1"),
        epub.Link("ch2.xhtml", "Empty Chapter", "ch2"),
        epub.Link("ch3.xhtml", "Chapter Three", "ch3"),
    ]

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", ch1, ch2, ch3]

    epub_path = tmp_path / "empty_chapter.epub"
    epub.write_epub(str(epub_path), book)

    return epub_path
