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
