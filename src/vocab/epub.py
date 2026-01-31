"""Epub chapter extraction (Layer 0)."""

import re
import warnings
from collections.abc import Generator
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import ITEM_DOCUMENT, epub

from vocab.models import Chapter

# Suppress warning about parsing XHTML as HTML - lxml handles this fine
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def extract_chapters(epub_path: Path) -> Generator[Chapter, None, None]:
    """Extract chapters from an epub file.

    Yields Chapter objects containing the text content, index, and title
    (if available) for each document in the epub's reading order.

    Args:
        epub_path: Path to the epub file.

    Yields:
        Chapter objects in reading order.
    """
    book = epub.read_epub(str(epub_path))

    # Build a mapping from item file name to title from TOC
    toc_titles = _extract_toc_titles(book)

    # Iterate through spine (reading order)
    index = 0
    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None:
            continue

        # Skip non-document items
        if item.get_type() != ITEM_DOCUMENT:
            continue

        # Also skip items that are EpubNav or EpubNcx (navigation)
        if isinstance(item, (epub.EpubNav, epub.EpubNcx)):
            continue

        content = item.get_content()
        text = _extract_text(content)

        # Skip empty chapters
        if not text.strip():
            continue

        # Try to get title from TOC, then from HTML
        file_name = item.get_name()
        title = toc_titles.get(file_name) or _extract_title_from_html(content)

        yield Chapter(text=text, index=index, title=title)
        index += 1


def _extract_toc_titles(book: epub.EpubBook) -> dict[str, str]:
    """Extract chapter titles from the table of contents.

    Args:
        book: The epub book object.

    Returns:
        Mapping from file name to title.
    """
    titles: dict[str, str] = {}

    def process_toc_item(item: epub.Link | tuple[epub.Section, list]) -> None:  # type: ignore[type-arg]
        if isinstance(item, epub.Link):
            # Remove fragment identifier if present (e.g., "chapter1.xhtml#section1")
            href = item.href.split("#")[0]
            titles[href] = item.title
        elif isinstance(item, tuple):
            # Section with nested items
            section, children = item
            if isinstance(section, epub.Section):
                href = section.href.split("#")[0] if section.href else None
                if href:
                    titles[href] = section.title
            for child in children:
                process_toc_item(child)

    for item in book.toc:
        process_toc_item(item)

    return titles


def _extract_text(html_content: bytes) -> str:
    """Extract plain text from HTML content.

    Preserves paragraph boundaries as newlines for proper sentence detection.

    Args:
        html_content: Raw HTML bytes.

    Returns:
        Plain text with paragraph boundaries preserved as newlines.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()

    # Insert newlines before block elements to preserve paragraph boundaries
    block_elements = ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]
    for tag in soup.find_all(block_elements):
        tag.insert_before("\n")

    # Get text with space separator to preserve word boundaries within elements
    text = soup.get_text(separator=" ")

    # Normalize: collapse multiple spaces (but not newlines) into single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Clean up spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Collapse multiple newlines into single newline (must be after space cleanup)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def _extract_title_from_html(html_content: bytes) -> str | None:
    """Extract title from HTML heading or title tag.

    Args:
        html_content: Raw HTML bytes.

    Returns:
        Title string or None if not found.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Try h1 first
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
        if title:
            return title

    # Fall back to title tag
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        if title:
            return title

    return None
