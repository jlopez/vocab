"""Data models for vocabulary extraction."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any


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
        pos: Universal part-of-speech tag (e.g., "NOUN", "VERB").
        morph: Morphological features as a dictionary (e.g., {"Gender": "Masc"}).
        original: The original text of the token as it appears.
        sentence: The full sentence containing the token.
        location: Location of the sentence within the epub.
    """

    lemma: str
    pos: str
    morph: dict[str, str]
    original: str
    sentence: str
    location: SentenceLocation


@dataclass
class Example:
    """An example sentence for a lemma.

    Attributes:
        sentence: The example sentence text.
        location: Location of the sentence within the epub.
    """

    sentence: str
    location: SentenceLocation


@dataclass
class LemmaEntry:
    """Aggregated data for a single lemma with a specific part-of-speech.

    Attributes:
        lemma: The lemmatized form.
        pos: Universal part-of-speech tag (e.g., "NOUN", "VERB").
        frequency: Total count of occurrences.
        forms: Mapping of original forms to their counts.
        examples: List of example sentences with their locations.
    """

    lemma: str
    pos: str
    frequency: int
    forms: dict[str, int]
    examples: list[Example]


@dataclass
class Vocabulary:
    """Vocabulary extracted from a document.

    Attributes:
        entries: Nested mapping of lemma -> pos -> LemmaEntry.
        language: Language code of the vocabulary.
    """

    entries: dict[str, dict[str, LemmaEntry]]
    language: str

    def __iter__(self) -> Iterator[LemmaEntry]:
        """Iterate over all LemmaEntry objects in the vocabulary.

        Yields:
            Each LemmaEntry in the vocabulary.
        """
        for pos_dict in self.entries.values():
            yield from pos_dict.values()

    def to_dict(self) -> dict[str, Any]:
        """Export vocabulary as a JSON-serializable dictionary.

        Returns:
            Dictionary with language and entries, where each entry
            contains lemma, pos, frequency, forms, and examples.
        """
        return asdict(self)
