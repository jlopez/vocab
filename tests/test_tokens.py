"""Tests for token extraction."""

from unittest.mock import patch

import pytest
import spacy
from spacy.language import Language

from vocab.models import SentenceLocation, Token
from vocab.tokens import extract_tokens


@pytest.fixture
def mock_spacy_model_with_lemmatizer() -> Language:
    """Create a mock spaCy model for testing.

    This uses a blank French model. Note that a blank model without trained
    components returns token.text as the lemma. This is intentional: our tests
    verify that extract_tokens correctly packages spaCy output into Token
    objects, not that spaCy performs correct lemmatization. Testing spaCy's
    lemmatization quality is outside the scope of this module's unit tests.

    Returns:
        The blank spaCy model.
    """
    nlp = spacy.blank("fr")
    return nlp


@pytest.fixture
def sample_location() -> SentenceLocation:
    """Create a sample location for testing."""
    return SentenceLocation(
        chapter_index=0,
        chapter_title="Test Chapter",
        sentence_index=0,
    )


class TestExtractTokens:
    """Tests for extract_tokens function."""

    def test_yields_token_objects(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should yield Token dataclass instances."""
        sentence = "Ceci est une phrase simple."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    def test_token_has_correct_attributes(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should yield tokens with all required attributes."""
        sentence = "Ceci est une phrase simple."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        token = tokens[0]
        assert isinstance(token.lemma, str)
        assert isinstance(token.original, str)
        assert isinstance(token.sentence, str)
        assert isinstance(token.location, SentenceLocation)

    def test_location_is_passed_through(self, mock_spacy_model_with_lemmatizer: Language) -> None:
        """Should use the provided location in all output tokens."""
        sentence = "Une phrase simple."
        location = SentenceLocation(
            chapter_index=5,
            chapter_title="Chapitre Cinq",
            sentence_index=10,
        )
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, location, "fr"))

        for token in tokens:
            assert token.location.chapter_index == 5
            assert token.location.chapter_title == "Chapitre Cinq"
            assert token.location.sentence_index == 10

    def test_sentence_text_is_preserved(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should preserve the original sentence text in all tokens."""
        sentence = "Ceci est une phrase simple."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        for token in tokens:
            assert token.sentence == sentence

    def test_filters_punctuation_tokens(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should not yield punctuation-only tokens."""
        sentence = "Bonjour! Comment allez-vous?"
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        # No token should be only punctuation
        for token in tokens:
            assert any(c.isalnum() for c in token.original), (
                f"Found punctuation-only token: {token.original!r}"
            )

    def test_filters_whitespace_tokens(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should not yield whitespace-only tokens."""
        sentence = "Une   phrase   avec   espaces."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        for token in tokens:
            assert token.original.strip(), f"Found whitespace-only token: {token.original!r}"

    def test_empty_sentence_yields_nothing(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should yield nothing for empty sentence."""
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens("", sample_location, "fr"))

        assert len(tokens) == 0

    def test_whitespace_only_yields_nothing(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should yield nothing for whitespace-only sentence."""
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens("   \n\n   ", sample_location, "fr"))

        assert len(tokens) == 0

    def test_extracts_multiple_tokens(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should extract all word tokens from a sentence."""
        sentence = "Le chat noir dort."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        # Should have tokens for: Le, chat, noir, dort (4 words, no punctuation)
        assert len(tokens) == 4
        originals = [t.original for t in tokens]
        assert "Le" in originals
        assert "chat" in originals
        assert "noir" in originals
        assert "dort" in originals


class TestExtractTokensLemmatization:
    """Tests for lemmatization behavior."""

    def test_lemma_is_string(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Lemma should always be a string."""
        sentence = "Ceci est une phrase simple."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        for token in tokens:
            assert isinstance(token.lemma, str)
            assert len(token.lemma) > 0

    def test_original_form_preserved(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Original token text should be preserved."""
        sentence = "Ceci est une phrase."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        originals = {t.original for t in tokens}
        assert "Ceci" in originals


class TestExtractTokensErrors:
    """Tests for error handling in extract_tokens."""

    def test_raises_for_unsupported_language(self, sample_location: SentenceLocation) -> None:
        """Should raise ValueError for unsupported language codes."""
        with pytest.raises(ValueError, match="Unsupported language code"):
            list(extract_tokens("Une phrase.", sample_location, "xyz"))


class TestExtractTokensWithNoneTitle:
    """Tests for handling None chapter titles."""

    def test_handles_none_chapter_title(self, mock_spacy_model_with_lemmatizer: Language) -> None:
        """Should handle location with None chapter title."""
        sentence = "Une phrase simple."
        location = SentenceLocation(
            chapter_index=0,
            chapter_title=None,
            sentence_index=0,
        )
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, location, "fr"))

        assert len(tokens) > 0
        for token in tokens:
            assert token.location.chapter_title is None


class TestFilterNonAlpha:
    """Tests for filter_non_alpha parameter."""

    def test_filters_numeric_tokens_by_default(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should filter out numeric tokens when filter_non_alpha is True (default)."""
        sentence = "Il a 42 ans en 2024."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        lemmas = [t.lemma for t in tokens]
        assert "42" not in lemmas
        assert "2024" not in lemmas
        # Alphabetic tokens should still be present
        assert "Il" in lemmas
        assert "a" in lemmas
        assert "ans" in lemmas

    def test_can_include_non_alpha_tokens(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should include numeric tokens when filter_non_alpha is False."""
        sentence = "Il a 42 ans."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr", filter_non_alpha=False))

        lemmas = [t.lemma for t in tokens]
        assert "42" in lemmas
        assert "Il" in lemmas

    def test_filters_mixed_alphanumeric_tokens(
        self, mock_spacy_model_with_lemmatizer: Language, sample_location: SentenceLocation
    ) -> None:
        """Should filter tokens with mixed alphanumeric characters."""
        sentence = "Version v2 du produit ABC123."
        with patch("vocab.tokens.get_model", return_value=mock_spacy_model_with_lemmatizer):
            tokens = list(extract_tokens(sentence, sample_location, "fr"))

        lemmas = [t.lemma for t in tokens]
        # Pure alphabetic tokens should be present
        assert "Version" in lemmas
        assert "du" in lemmas
        assert "produit" in lemmas
        # Mixed alphanumeric should be filtered
        assert "v2" not in lemmas
        assert "ABC123" not in lemmas
