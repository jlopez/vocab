"""Tests for sentence extraction."""

from unittest.mock import MagicMock, patch

import pytest
from spacy.language import Language

from vocab.models import Sentence
from vocab.sentences import (
    SpacyModelNotFoundError,
    _model_cache,
    extract_sentences,
    get_model,
)


class TestExtractSentences:
    """Tests for extract_sentences function."""

    def test_extracts_simple_sentences(self, mock_spacy_model: Language) -> None:
        """Should split text into sentences with correct indices."""
        text = "Ceci est une phrase. Voici une autre phrase. Et une troisième."
        sentences = list(extract_sentences(text, "fr"))

        assert len(sentences) == 3
        assert sentences[0].text == "Ceci est une phrase."
        assert sentences[0].index == 0
        assert sentences[1].text == "Voici une autre phrase."
        assert sentences[1].index == 1
        assert sentences[2].text == "Et une troisième."
        assert sentences[2].index == 2

    def test_returns_sentence_dataclass(self, mock_spacy_model: Language) -> None:
        """Should return Sentence dataclass instances."""
        text = "Une phrase simple."
        sentences = list(extract_sentences(text, "fr"))

        assert all(isinstance(s, Sentence) for s in sentences)

    def test_handles_multiple_punctuation(self, mock_spacy_model: Language) -> None:
        """Should handle sentences with multiple punctuation marks."""
        text = "Vraiment?! Oui, vraiment. C'est incroyable..."
        sentences = list(extract_sentences(text, "fr"))

        # Sentencizer splits on punctuation - verify content is preserved
        assert len(sentences) >= 1
        full_text = " ".join(s.text for s in sentences)
        assert "Vraiment" in full_text
        assert "vraiment" in full_text

    def test_skips_empty_sentences(self, mock_spacy_model: Language) -> None:
        """Should skip sentences that are only whitespace."""
        text = "Une phrase.    \n\n    Autre phrase."
        sentences = list(extract_sentences(text, "fr"))

        assert len(sentences) == 2
        assert sentences[0].text == "Une phrase."
        assert sentences[1].text == "Autre phrase."

    def test_strips_whitespace_from_sentences(self, mock_spacy_model: Language) -> None:
        """Should strip leading/trailing whitespace from sentences."""
        text = "   Une phrase avec espaces.   "
        sentences = list(extract_sentences(text, "fr"))

        assert sentences[0].text == "Une phrase avec espaces."

    def test_empty_text_yields_nothing(self, mock_spacy_model: Language) -> None:
        """Should yield nothing for empty text."""
        sentences = list(extract_sentences("", "fr"))

        assert len(sentences) == 0

    def test_whitespace_only_yields_nothing(self, mock_spacy_model: Language) -> None:
        """Should yield nothing for whitespace-only text."""
        sentences = list(extract_sentences("   \n\n   ", "fr"))

        assert len(sentences) == 0

    def test_handles_newlines_between_sentences(self, mock_spacy_model: Language) -> None:
        """Should handle text with newlines between sentences."""
        text = "Première phrase.\nDeuxième phrase.\nTroisième phrase."
        sentences = list(extract_sentences(text, "fr"))

        assert len(sentences) >= 1
        full_text = " ".join(s.text for s in sentences)
        assert "Première" in full_text
        assert "Deuxième" in full_text
        assert "Troisième" in full_text

    def test_handles_dialogue_quotes(self, mock_spacy_model: Language) -> None:
        """Should handle French dialogue with guillemets."""
        text = "« Bonjour », dit-il. « Comment allez-vous ? »"
        sentences = list(extract_sentences(text, "fr"))

        # Verify content preserved
        assert len(sentences) >= 1
        full_text = " ".join(s.text for s in sentences)
        assert "Bonjour" in full_text
        assert "Comment allez-vous" in full_text

    def test_filters_punctuation_only_sentences_by_default(
        self, mock_spacy_model: Language
    ) -> None:
        """Should filter out sentences containing only punctuation."""
        # The sentencizer treats "!!!" as its own sentence before "Bonjour."
        text = "!!! Bonjour. Au revoir."
        sentences = list(extract_sentences(text, "fr"))

        # The "!!!" sentence should be filtered out
        assert len(sentences) == 2
        assert sentences[0].text == "Bonjour."
        assert sentences[1].text == "Au revoir."

    def test_punctuation_filtering_can_be_disabled(self, mock_spacy_model: Language) -> None:
        """Should keep punctuation-only sentences when filter_punctuation=False."""
        # The sentencizer treats "!!!" as its own sentence
        text = "!!! Bonjour."
        sentences_filtered = list(extract_sentences(text, "fr", filter_punctuation=True))
        sentences_unfiltered = list(extract_sentences(text, "fr", filter_punctuation=False))

        # With filtering enabled, "!!!" is removed
        assert len(sentences_filtered) == 1
        assert sentences_filtered[0].text == "Bonjour."

        # With filtering disabled, "!!!" is kept
        assert len(sentences_unfiltered) == 2
        assert sentences_unfiltered[0].text == "!!!"
        assert sentences_unfiltered[1].text == "Bonjour."

    def test_normalizes_internal_whitespace(self, mock_spacy_model: Language) -> None:
        """Should replace newlines with spaces and collapse multiple spaces."""
        text = "Une phrase\navec des\n\nretours à la ligne."
        sentences = list(extract_sentences(text, "fr"))

        # Newlines should be replaced with single spaces
        assert sentences[0].text == "Une phrase avec des retours à la ligne."

    def test_dialogue_with_em_dash(self, mock_spacy_model: Language) -> None:
        """Document behavior with French dialogue using em-dash.

        French novels use em-dash (—) for dialogue attribution.
        """
        text = "« Je pars demain. » — Vraiment ? — Oui."
        sentences = list(extract_sentences(text, "fr"))

        # Verify the actual dialogue content is preserved
        full_text = " ".join(s.text for s in sentences)
        assert "Je pars demain" in full_text
        assert "Vraiment" in full_text
        assert "Oui" in full_text


class TestGetModel:
    """Tests for get_model function."""

    def test_returns_spacy_model(self) -> None:
        """Should return a spaCy Language model."""
        mock_nlp = MagicMock()
        mock_nlp.pipe = MagicMock()
        mock_nlp.max_length = 1000000

        with patch("vocab.sentences.spacy.load", return_value=mock_nlp):
            _model_cache.clear()
            nlp = get_model("fr")

            assert hasattr(nlp, "pipe")
            assert hasattr(nlp, "max_length")

    def test_caches_loaded_models(self) -> None:
        """Should cache models and return same instance on subsequent calls."""
        mock_nlp = MagicMock()

        with patch("vocab.sentences.spacy.load", return_value=mock_nlp) as mock_load:
            _model_cache.clear()

            nlp1 = get_model("fr")
            nlp2 = get_model("fr")

            assert nlp1 is nlp2
            # spacy.load should only be called once due to caching
            mock_load.assert_called_once()

    def test_raises_for_unsupported_language(self) -> None:
        """Should raise ValueError for unsupported language codes."""
        with pytest.raises(ValueError, match="Unsupported language code 'xyz'"):
            get_model("xyz")

    def test_error_message_suggests_full_model_name(self) -> None:
        """Should suggest using full model name in error message."""
        with pytest.raises(ValueError, match="or pass a full spaCy model name"):
            get_model("xyz")

    def test_loads_full_model_name_directly(self) -> None:
        """Should load full model names without using the language mapping."""
        mock_nlp = MagicMock()

        with patch("vocab.sentences.spacy.load", return_value=mock_nlp) as mock_load:
            _model_cache.clear()
            nlp = get_model("fr_core_news_lg")

            # Should call spacy.load with the exact model name
            mock_load.assert_called_once_with("fr_core_news_lg")
            assert nlp is mock_nlp

    def test_caches_full_model_names(self) -> None:
        """Should cache models loaded by full name."""
        mock_nlp = MagicMock()

        with patch("vocab.sentences.spacy.load", return_value=mock_nlp) as mock_load:
            _model_cache.clear()

            nlp1 = get_model("fr_core_news_lg")
            nlp2 = get_model("fr_core_news_lg")

            assert nlp1 is nlp2
            mock_load.assert_called_once()

    def test_raises_for_missing_full_model_name(self) -> None:
        """Should raise SpacyModelNotFoundError when full model name not installed."""
        with patch("vocab.sentences.spacy.load", side_effect=OSError("Model not found")):
            _model_cache.clear()
            with pytest.raises(SpacyModelNotFoundError) as exc_info:
                get_model("fr_custom_model_lg")

            assert "fr_custom_model_lg" in str(exc_info.value)
            assert "python -m spacy download" in str(exc_info.value)

    def test_raises_for_missing_model(self) -> None:
        """Should raise SpacyModelNotFoundError when model not installed."""
        with patch.dict(
            "vocab.sentences._LANGUAGE_MODELS",
            {"fake": "fake_model_that_does_not_exist"},
        ):
            with pytest.raises(SpacyModelNotFoundError) as exc_info:
                get_model("fake")

            assert "fake_model_that_does_not_exist" in str(exc_info.value)
            assert "python -m spacy download" in str(exc_info.value)


class TestSpacyModelNotFoundError:
    """Tests for SpacyModelNotFoundError exception."""

    def test_error_message_contains_model_name(self) -> None:
        """Should include model name in error message."""
        error = SpacyModelNotFoundError("fr", "fr_core_news_lg")

        assert "fr_core_news_lg" in str(error)

    def test_error_message_contains_language(self) -> None:
        """Should include language code in error message."""
        error = SpacyModelNotFoundError("fr", "fr_core_news_lg")

        assert "'fr'" in str(error)

    def test_error_message_contains_install_instructions(self) -> None:
        """Should include installation instructions."""
        error = SpacyModelNotFoundError("fr", "fr_core_news_lg")

        assert "python -m spacy download fr_core_news_lg" in str(error)

    def test_stores_language_and_model(self) -> None:
        """Should store language and model_name as attributes."""
        error = SpacyModelNotFoundError("fr", "fr_core_news_lg")

        assert error.language == "fr"
        assert error.model_name == "fr_core_news_lg"


class TestLongText:
    """Tests for handling long text."""

    def test_handles_long_text(self, mock_spacy_model: Language) -> None:
        """Should handle long text with many sentences."""
        text = "Une phrase simple. " * 100

        sentences = list(extract_sentences(text, "fr"))

        # Should extract all sentences
        assert len(sentences) == 100

    def test_long_text_sentences_have_continuous_indices(self, mock_spacy_model: Language) -> None:
        """Long text should have continuous sentence indices."""
        text = "Première. Deuxième. Troisième. Quatrième. Cinquième."
        sentences = list(extract_sentences(text, "fr"))

        indices = [s.index for s in sentences]
        assert indices == list(range(len(sentences)))
