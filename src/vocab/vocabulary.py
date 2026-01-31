"""Vocabulary aggregation from epub files (Layer 3)."""

from pathlib import Path

from vocab.epub import extract_chapters
from vocab.models import Example, LemmaEntry, SentenceLocation, Vocabulary
from vocab.sentences import extract_sentences
from vocab.tokens import extract_tokens


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

    # Track lemma data during aggregation
    lemma_frequency: dict[str, int] = {}
    lemma_forms: dict[str, dict[str, int]] = {}
    lemma_examples: dict[str, list[Example]] = {}

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
                lemma = token.lemma

                # Update frequency
                lemma_frequency[lemma] = lemma_frequency.get(lemma, 0) + 1

                # Update forms
                if lemma not in lemma_forms:
                    lemma_forms[lemma] = {}
                forms = lemma_forms[lemma]
                forms[token.original] = forms.get(token.original, 0) + 1

                # Add example if under limit and not a duplicate sentence
                if lemma not in lemma_examples:
                    lemma_examples[lemma] = []
                examples = lemma_examples[lemma]
                if len(examples) < max_examples and not any(
                    ex.sentence == token.sentence for ex in examples
                ):
                    examples.append(Example(sentence=token.sentence, location=token.location))

    # Build final vocabulary
    entries: dict[str, LemmaEntry] = {}
    for lemma, frequency in lemma_frequency.items():
        entries[lemma] = LemmaEntry(
            lemma=lemma,
            frequency=frequency,
            forms=lemma_forms[lemma],
            examples=lemma_examples[lemma],
        )

    return Vocabulary(entries=entries, language=language)
