"""Sentence extraction using spaCy."""

from collections.abc import Generator

import spacy
from spacy.language import Language

from vocab.models import Sentence

# Mapping of language codes to spaCy model names
_LANGUAGE_MODELS: dict[str, str] = {
    "fr": "fr_core_news_sm",
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
}

# Cache for loaded spaCy models
_model_cache: dict[str, Language] = {}


class SpacyModelNotFoundError(Exception):
    """Raised when a spaCy model is not installed."""

    def __init__(self, language: str, model_name: str) -> None:
        self.language = language
        self.model_name = model_name
        super().__init__(
            f"spaCy model '{model_name}' for language '{language}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        )


def get_model(language: str) -> Language:
    """Get or load a spaCy model for the given language.

    Args:
        language: Language code (e.g., "fr" for French).

    Returns:
        Loaded spaCy Language model.

    Raises:
        SpacyModelNotFoundError: If the model is not installed.
        ValueError: If the language code is not supported.
    """
    if language in _model_cache:
        return _model_cache[language]

    if language not in _LANGUAGE_MODELS:
        supported = ", ".join(sorted(_LANGUAGE_MODELS.keys()))
        raise ValueError(
            f"Unsupported language code '{language}'. Supported languages: {supported}"
        )

    model_name = _LANGUAGE_MODELS[language]

    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise SpacyModelNotFoundError(language, model_name) from e

    _model_cache[language] = nlp
    return nlp


def _is_punctuation_only(text: str) -> bool:
    """Check if text contains only punctuation and whitespace."""
    return all(not c.isalnum() for c in text)


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: replace newlines with spaces, collapse multiple spaces."""
    return " ".join(text.split())


def extract_sentences(
    text: str,
    language: str,
    *,
    filter_punctuation: bool = True,
) -> Generator[Sentence, None, None]:
    """Extract sentences from text using spaCy's sentence boundary detection.

    Args:
        text: Input text to split into sentences.
        language: Language code (e.g., "fr" for French).
        filter_punctuation: If True (default), skip sentences containing only
            punctuation and whitespace.

    Yields:
        Sentence objects with text and index within input.

    Raises:
        SpacyModelNotFoundError: If the spaCy model is not installed.
        ValueError: If the language code is not supported.
    """
    if not text.strip():
        return

    nlp = get_model(language)
    doc = nlp(text)
    index = 0

    for sent in doc.sents:
        sentence_text = _normalize_whitespace(sent.text)
        if not sentence_text:
            continue
        if filter_punctuation and _is_punctuation_only(sentence_text):
            continue
        yield Sentence(text=sentence_text, index=index)
        index += 1
