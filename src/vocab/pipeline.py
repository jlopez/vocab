"""Pipeline for generating Anki flashcards from vocabulary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from vocab.dictionary import SPACY_TO_KAIKKI, Dictionary, DictionaryEntry
from vocab.models import LemmaEntry, Vocabulary


@dataclass
class EnrichedLemma:
    """A lemma enriched with dictionary data.

    Attributes:
        lemma: The LemmaEntry from the vocabulary.
        words: List of matching dictionary entries (invariant: len >= 1).
    """

    lemma: LemmaEntry
    words: list[DictionaryEntry]

    def __post_init__(self) -> None:
        """Validate invariants."""
        if not self.words:
            raise ValueError("words must have at least one entry")


def generate_enriched_lemmas(
    vocabulary: Vocabulary,
    dictionary: Dictionary,
) -> Iterator[EnrichedLemma]:
    """Generate enriched lemmas from vocabulary.

    For each LemmaEntry in the vocabulary, looks up matching dictionary
    entries by (word, POS). Only yields entries with at least one match.

    Args:
        vocabulary: Vocabulary to process.
        dictionary: Dictionary for lookups.

    Yields:
        EnrichedLemma for each lemma with dictionary matches.
    """
    for lemma_by_pos in vocabulary.entries.values():
        for lemma_entry in lemma_by_pos.values():
            kaikki_pos = SPACY_TO_KAIKKI.get(lemma_entry.pos, [])
            words = dictionary.lookup(lemma_entry.lemma, pos=kaikki_pos or None)
            if words:
                yield EnrichedLemma(lemma=lemma_entry, words=words)
