"""Vocabulary aggregation from epub files (Layer 3)."""

from pathlib import Path

from vocab.epub import extract_chapters
from vocab.models import Example, LemmaEntry, SentenceLocation, Vocabulary
from vocab.sentences import extract_sentences
from vocab.tokens import extract_tokens

# POS tags to skip during vocabulary building
SKIP_POS: set[str] = {""}


def build_vocabulary(
    epub_path: Path,
    language: str,
    max_examples: int = 3,
) -> Vocabulary:
    """Build vocabulary frequency data from an epub.

    Processes the entire epub file, extracting chapters, sentences, and tokens,
    then aggregates them into a vocabulary with frequency counts and examples.

    Args:
        epub_path: Path to the epub file.
        language: Language code (e.g., "fr" for French).
        max_examples: Maximum example sentences to keep per lemma (0 for none).

    Returns:
        Vocabulary object with aggregated frequency data.

    Raises:
        SpacyModelNotFoundError: If the spaCy model is not installed.
        ValueError: If the language code is not supported or max_examples < 0.
    """
    if max_examples < 0:
        raise ValueError("max_examples must be >= 0")

    # Track data during aggregation, keyed by (lemma, pos) tuple
    frequency: dict[tuple[str, str], int] = {}
    forms: dict[tuple[str, str], dict[str, int]] = {}
    examples: dict[tuple[str, str], list[Example]] = {}

    for chapter in extract_chapters(epub_path):
        location_base = SentenceLocation(
            chapter_index=chapter.index,
            chapter_title=chapter.title,
            sentence_index=0,  # Will be updated per sentence
        )

        for sentence in extract_sentences(chapter.text, language):
            location = SentenceLocation(
                chapter_index=location_base.chapter_index,
                chapter_title=location_base.chapter_title,
                sentence_index=sentence.index,
            )

            for token in extract_tokens(sentence.text, location, language):
                # Skip tokens with blocked POS
                if token.pos in SKIP_POS:
                    continue

                key = (token.lemma, token.pos)

                # Update frequency
                frequency[key] = frequency.get(key, 0) + 1

                # Update forms
                if key not in forms:
                    forms[key] = {}
                key_forms = forms[key]
                key_forms[token.original] = key_forms.get(token.original, 0) + 1

                # Add example if under limit and not a duplicate sentence
                if key not in examples:
                    examples[key] = []
                key_examples = examples[key]
                if len(key_examples) < max_examples and not any(
                    ex.sentence == token.sentence for ex in key_examples
                ):
                    key_examples.append(Example(sentence=token.sentence, location=token.location))

    # Build final vocabulary with nested structure: entries[lemma][pos] = LemmaEntry
    entries: dict[str, dict[str, LemmaEntry]] = {}
    for (lemma, pos), freq in frequency.items():
        if lemma not in entries:
            entries[lemma] = {}
        entries[lemma][pos] = LemmaEntry(
            lemma=lemma,
            pos=pos,
            frequency=freq,
            forms=forms[(lemma, pos)],
            examples=examples[(lemma, pos)],
        )

    return Vocabulary(entries=entries, language=language)
