"""Vocabulary aggregation from epub files (Layer 3)."""

from pathlib import Path

from vocab.epub import extract_chapters
from vocab.models import Example, LemmaEntry, SentenceLocation, Token, Vocabulary
from vocab.sentences import extract_sentences
from vocab.tokens import extract_tokens

# POS tags to skip during vocabulary building
SKIP_POS: set[str] = {""}


class VocabularyBuilder:
    """Incrementally builds vocabulary from tokens.

    Accumulates tokens via add(), then call build() to produce
    a Vocabulary object with frequency counts and examples.
    """

    def __init__(self, language: str, max_examples: int = 3) -> None:
        """Initialize the vocabulary builder.

        Args:
            language: Language code (e.g., "fr" for French).
            max_examples: Maximum example sentences to keep per lemma (0 for none).

        Raises:
            ValueError: If max_examples < 0.
        """
        if max_examples < 0:
            raise ValueError("max_examples must be >= 0")

        self._language = language
        self._max_examples = max_examples
        self._frequency: dict[tuple[str, str], int] = {}
        self._forms: dict[tuple[str, str], dict[str, int]] = {}
        self._examples: dict[tuple[str, str], list[Example]] = {}
        self._built = False

    def add(self, token: Token) -> None:
        """Add a token to the vocabulary.

        Tokens with POS in SKIP_POS are silently ignored.

        Args:
            token: The token to add.

        Raises:
            RuntimeError: If build() has already been called.
        """
        if self._built:
            raise RuntimeError("Cannot add tokens after build() has been called")
        if token.pos in SKIP_POS:
            return

        key = (token.lemma, token.pos)

        # Update frequency
        self._frequency[key] = self._frequency.get(key, 0) + 1

        # Update forms
        if key not in self._forms:
            self._forms[key] = {}
        key_forms = self._forms[key]
        key_forms[token.original] = key_forms.get(token.original, 0) + 1

        # Add example if under limit and not a duplicate sentence
        if key not in self._examples:
            self._examples[key] = []
        key_examples = self._examples[key]
        if len(key_examples) < self._max_examples and not any(
            ex.sentence == token.sentence for ex in key_examples
        ):
            key_examples.append(Example(sentence=token.sentence, location=token.location))

    def build(self) -> Vocabulary:
        """Build the final Vocabulary object from accumulated tokens.

        This is a terminal operation — after calling build(), further
        add() or build() calls will raise RuntimeError.

        Returns:
            Vocabulary object with aggregated frequency data.

        Raises:
            RuntimeError: If build() has already been called.
        """
        if self._built:
            raise RuntimeError("build() has already been called")
        self._built = True
        entries: dict[str, dict[str, LemmaEntry]] = {}
        for (lemma, pos), freq in self._frequency.items():
            if lemma not in entries:
                entries[lemma] = {}
            entries[lemma][pos] = LemmaEntry(
                lemma=lemma,
                pos=pos,
                frequency=freq,
                forms=self._forms[(lemma, pos)],
                examples=self._examples[(lemma, pos)],
            )

        return Vocabulary(entries=entries, language=self._language)


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
    builder = VocabularyBuilder(language=language, max_examples=max_examples)

    for chapter in extract_chapters(epub_path):
        for sentence in extract_sentences(chapter.text, language):
            location = SentenceLocation(
                chapter_index=chapter.index,
                chapter_title=chapter.title,
                sentence_index=sentence.index,
            )
            for token in extract_tokens(sentence.text, location, language):
                builder.add(token)

    return builder.build()
