"""Vocabulary filtering utilities."""

from vocab.models import LemmaEntry, Vocabulary


def filter_vocabulary(
    vocab: Vocabulary,
    *,
    min_freq: int = 1,
    max_freq: int | None = None,
    exclude_proper_nouns: bool = True,
) -> list[LemmaEntry]:
    """Filter vocabulary entries based on frequency and type.

    Args:
        vocab: Vocabulary to filter.
        min_freq: Minimum frequency (inclusive). Default 1.
        max_freq: Maximum frequency (inclusive). None for no limit.
        exclude_proper_nouns: If True, exclude lemmas that appear to be
            proper nouns (start with uppercase letter).

    Returns:
        List of LemmaEntry objects matching the criteria,
        sorted by frequency descending.
    """
    entries = list(vocab.entries.values())

    # Filter by minimum frequency
    entries = [e for e in entries if e.frequency >= min_freq]

    # Filter by maximum frequency if specified
    if max_freq is not None:
        entries = [e for e in entries if e.frequency <= max_freq]

    # Exclude proper nouns if requested
    if exclude_proper_nouns:
        entries = [e for e in entries if not e.lemma[0].isupper()]

    # Sort by frequency descending
    entries.sort(key=lambda e: e.frequency, reverse=True)

    return entries
