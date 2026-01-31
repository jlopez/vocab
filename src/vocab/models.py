"""Data models for vocabulary extraction."""

from dataclasses import dataclass


@dataclass
class Chapter:
    """A chapter extracted from an epub file.

    Attributes:
        text: The plain text content of the chapter with HTML stripped.
        index: Zero-based index of the chapter in reading order.
        title: Chapter title if available, otherwise None.
    """

    text: str
    index: int
    title: str | None


@dataclass
class Sentence:
    """A sentence extracted from text.

    Attributes:
        text: The sentence text.
        index: Zero-based index of the sentence within the source text.
    """

    text: str
    index: int


@dataclass
class SentenceLocation:
    """Location of a sentence within an epub.

    Attributes:
        chapter_index: Zero-based index of the chapter.
        chapter_title: Title of the chapter, if available.
        sentence_index: Zero-based index of the sentence within the chapter.
    """

    chapter_index: int
    chapter_title: str | None
    sentence_index: int


@dataclass
class Token:
    """A token extracted from text with lemma and context.

    Attributes:
        lemma: The lemmatized form of the token.
        original: The original text of the token as it appears.
        sentence: The full sentence containing the token.
        location: Location of the sentence within the epub.
    """

    lemma: str
    original: str
    sentence: str
    location: SentenceLocation
