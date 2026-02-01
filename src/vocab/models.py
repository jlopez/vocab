"""Data models for vocabulary extraction."""

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
    """Aggregated data for a single lemma.

    Attributes:
        lemma: The lemmatized form.
        frequency: Total count of occurrences.
        forms: Mapping of original forms to their counts.
        examples: List of example sentences with their locations.
    """

    lemma: str
    frequency: int
    forms: dict[str, int]
    examples: list[Example]


@dataclass
class Vocabulary:
    """Vocabulary extracted from a document.

    Attributes:
        entries: Mapping of lemma to LemmaEntry.
        language: Language code of the vocabulary.
    """

    entries: dict[str, LemmaEntry]
    language: str

    def top(self, n: int) -> list[LemmaEntry]:
        """Return top n lemmas by frequency.

        Args:
            n: Number of top entries to return (must be >= 1).

        Returns:
            List of LemmaEntry objects sorted by frequency (descending).

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.frequency,
            reverse=True,
        )
        return sorted_entries[:n]

    def to_dict(self) -> dict[str, Any]:
        """Export vocabulary as a JSON-serializable dictionary.

        Returns:
            Dictionary with language and entries, where each entry
            contains lemma, frequency, forms, and examples.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vocabulary":
        """Load vocabulary from a JSON-serializable dictionary.

        Args:
            data: Dictionary with 'language' and 'entries' keys,
                  as produced by to_dict().

        Returns:
            Vocabulary instance.
        """
        entries: dict[str, LemmaEntry] = {}
        for lemma, entry_data in data["entries"].items():
            examples = [
                Example(
                    sentence=ex["sentence"],
                    location=SentenceLocation(
                        chapter_index=ex["location"]["chapter_index"],
                        chapter_title=ex["location"]["chapter_title"],
                        sentence_index=ex["location"]["sentence_index"],
                    ),
                )
                for ex in entry_data["examples"]
            ]
            entries[lemma] = LemmaEntry(
                lemma=entry_data["lemma"],
                frequency=entry_data["frequency"],
                forms=entry_data["forms"],
                examples=examples,
            )
        return cls(entries=entries, language=data["language"])
