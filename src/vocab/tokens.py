"""Token extraction from sentences."""

from collections.abc import Generator

from vocab.models import SentenceLocation, Token
from vocab.sentences import get_model


def extract_tokens(
    sentence_text: str,
    location: SentenceLocation,
    language: str,
    *,
    filter_non_alpha: bool = True,
) -> Generator[Token, None, None]:
    """Extract lemmatized tokens from a sentence.

    Args:
        sentence_text: The sentence text to tokenize.
        location: Location of the sentence within the source document.
        language: Language code (e.g., "fr" for French).
        filter_non_alpha: If True (default), only yield tokens whose lemma
            contains only alphabetic characters. This filters out numbers,
            symbols, and other non-vocabulary tokens.

    Yields:
        Token objects with lemma, original form, sentence text, and location.

    Raises:
        SpacyModelNotFoundError: If the spaCy model is not installed.
        ValueError: If the language code is not supported.
    """
    if not sentence_text.strip():
        return

    nlp = get_model(language)
    doc = nlp(sentence_text)

    for token in doc:
        if token.is_punct or token.is_space:
            continue
        # Defensive assertion: spaCy's is_space should catch whitespace tokens
        assert token.text.strip(), f"Unexpected empty token after is_space check: {token!r}"

        # Use original text as fallback if lemma is empty
        lemma = token.lemma_ if token.lemma_ else token.text

        # Filter non-alphabetic lemmas (numbers, symbols, etc.)
        if filter_non_alpha and not lemma.isalpha():
            continue

        yield Token(
            lemma=lemma,
            original=token.text,
            sentence=sentence_text,
            location=location,
        )
