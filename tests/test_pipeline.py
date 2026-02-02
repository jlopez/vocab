"""Tests for the pipeline module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vocab.dictionary import DictionaryEntry, DictionarySense
from vocab.models import Example, LemmaEntry, SentenceLocation
from vocab.pipeline import (
    DisambiguationResponse,
    EnrichedLemma,
    SenseAssignment,
    SentenceAssignment,
    _build_prompt,
    _parse_llm_response,
    assign_single_sense,
    disambiguate_senses,
    enrich_lemma,
    needs_disambiguation,
)


def make_lemma_entry(
    lemma: str,
    pos: str,
    frequency: int = 1,
) -> LemmaEntry:
    """Create a LemmaEntry for testing."""
    return LemmaEntry(
        lemma=lemma,
        pos=pos,
        frequency=frequency,
        forms={lemma: frequency},
        examples=[
            Example(
                sentence=f"Example with {lemma}.",
                location=SentenceLocation(
                    chapter_index=0,
                    chapter_title="Chapter 1",
                    sentence_index=0,
                ),
            )
        ],
    )


def make_dictionary_entry(
    word: str,
    pos: str,
    translation: str = "translation",
) -> DictionaryEntry:
    """Create a DictionaryEntry for testing."""
    return DictionaryEntry(
        word=word,
        pos=pos,
        ipa="/test/",
        etymology=None,
        senses=[
            DictionarySense(
                id=f"{word}-{pos}-1",
                translation=translation,
                example=None,
            )
        ],
    )


class TestEnrichedLemma:
    """Tests for EnrichedLemma dataclass."""

    def test_has_lemma_and_words(self) -> None:
        """Test EnrichedLemma has required fields."""
        lemma = make_lemma_entry("chien", "NOUN")
        words = [make_dictionary_entry("chien", "noun", "dog")]

        enriched = EnrichedLemma(lemma=lemma, words=words)

        assert enriched.lemma == lemma
        assert enriched.words == words
        assert len(enriched.words) == 1

    def test_rejects_empty_words(self) -> None:
        """Test that EnrichedLemma rejects empty words list."""
        import pytest

        lemma = make_lemma_entry("chien", "NOUN")

        with pytest.raises(ValueError, match="words must have at least one entry"):
            EnrichedLemma(lemma=lemma, words=[])

    def test_rejects_words_with_no_senses(self) -> None:
        """Test that EnrichedLemma rejects words with zero senses."""
        lemma = make_lemma_entry("chien", "NOUN")
        words = [make_dictionary_entry_no_senses("chien", "noun")]

        with pytest.raises(ValueError, match="all words must have at least one sense"):
            EnrichedLemma(lemma=lemma, words=words)


class TestEnrichLemma:
    """Tests for enrich_lemma function."""

    def test_returns_enriched_lemma_for_match(self) -> None:
        """Test that matching lemma returns EnrichedLemma."""
        lemma = make_lemma_entry("chien", "NOUN")

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("chien", "noun", "dog")]

        result = enrich_lemma(lemma, mock_dict)

        assert result is not None
        assert result.lemma.lemma == "chien"
        assert result.words[0].senses[0].translation == "dog"

    def test_returns_none_without_match(self) -> None:
        """Test that lemma without dictionary match returns None."""
        lemma = make_lemma_entry("xyznonexistent", "NOUN")

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = []

        result = enrich_lemma(lemma, mock_dict)

        assert result is None

    def test_uses_spacy_to_kaikki_mapping(self) -> None:
        """Test that POS is mapped from spaCy to kaikki format."""
        lemma = make_lemma_entry("manger", "VERB")

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("manger", "verb", "to eat")]

        enrich_lemma(lemma, mock_dict)

        mock_dict.lookup.assert_called_once_with("manger", pos=["verb"])

    def test_handles_unmapped_pos(self) -> None:
        """Test that unmapped POS tags pass None to lookup."""
        lemma = make_lemma_entry("test", "UNKNOWN_POS")

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("test", "noun")]

        enrich_lemma(lemma, mock_dict)

        mock_dict.lookup.assert_called_once_with("test", pos=None)

    def test_returns_multiple_dictionary_entries(self) -> None:
        """Test that multiple dictionary entries are preserved."""
        lemma = make_lemma_entry("faux", "NOUN")

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [
            make_dictionary_entry("faux", "noun", "forgery"),
            make_dictionary_entry("faux", "noun", "scythe"),
        ]

        result = enrich_lemma(lemma, mock_dict)

        assert result is not None
        assert len(result.words) == 2
        translations = {w.senses[0].translation for w in result.words}
        assert translations == {"forgery", "scythe"}

    def test_adp_maps_to_multiple_pos(self) -> None:
        """Test that ADP maps to prep, prep_phrase, postp."""
        lemma = make_lemma_entry("dans", "ADP")

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("dans", "prep", "in")]

        enrich_lemma(lemma, mock_dict)

        call_args = mock_dict.lookup.call_args
        assert call_args[0][0] == "dans"
        assert set(call_args[1]["pos"]) == {"prep", "prep_phrase", "postp"}


def make_lemma_with_examples(
    lemma: str,
    pos: str,
    sentences: list[str],
) -> LemmaEntry:
    """Create a LemmaEntry with multiple examples."""
    return LemmaEntry(
        lemma=lemma,
        pos=pos,
        frequency=len(sentences),
        forms={lemma: len(sentences)},
        examples=[
            Example(
                sentence=s,
                location=SentenceLocation(
                    chapter_index=0,
                    chapter_title="Chapter 1",
                    sentence_index=i,
                ),
            )
            for i, s in enumerate(sentences)
        ],
    )


def make_dictionary_entry_with_senses(
    word: str,
    pos: str,
    translations: list[str],
    etymology: str | None = None,
) -> DictionaryEntry:
    """Create a DictionaryEntry with multiple senses."""
    return DictionaryEntry(
        word=word,
        pos=pos,
        ipa="/test/",
        etymology=etymology,
        senses=[
            DictionarySense(
                id=f"{word}-{pos}-{i}",
                translation=t,
                example=None,
            )
            for i, t in enumerate(translations)
        ],
    )


def make_dictionary_entry_no_senses(word: str, pos: str) -> DictionaryEntry:
    """Create a DictionaryEntry with zero senses (malformed data)."""
    return DictionaryEntry(
        word=word,
        pos=pos,
        ipa="/test/",
        etymology=None,
        senses=[],
    )


class TestSenseAssignment:
    """Tests for SenseAssignment dataclass."""

    def test_has_required_fields(self) -> None:
        """Test SenseAssignment has required fields."""
        lemma = make_lemma_entry("chien", "NOUN")
        word = make_dictionary_entry("chien", "noun", "dog")

        assignment = SenseAssignment(
            lemma=lemma,
            examples=[0],
            word=word,
            sense=0,
        )

        assert assignment.lemma == lemma
        assert assignment.examples == [0]
        assert assignment.word == word
        assert assignment.sense == 0

    def test_rejects_invalid_sense_index(self) -> None:
        """Test that SenseAssignment rejects sense index out of bounds."""
        lemma = make_lemma_entry("chien", "NOUN")
        word = make_dictionary_entry("chien", "noun", "dog")  # has 1 sense (index 0)

        with pytest.raises(ValueError, match="sense index 1 out of bounds"):
            SenseAssignment(lemma=lemma, examples=[0], word=word, sense=1)

    def test_rejects_negative_sense_index(self) -> None:
        """Test that SenseAssignment rejects negative sense index."""
        lemma = make_lemma_entry("chien", "NOUN")
        word = make_dictionary_entry("chien", "noun", "dog")

        with pytest.raises(ValueError, match="sense index -1 out of bounds"):
            SenseAssignment(lemma=lemma, examples=[0], word=word, sense=-1)

    def test_rejects_invalid_example_index(self) -> None:
        """Test that SenseAssignment rejects example index out of bounds."""
        lemma = make_lemma_entry("chien", "NOUN")  # has 1 example (index 0)
        word = make_dictionary_entry("chien", "noun", "dog")

        with pytest.raises(ValueError, match="example index 1 out of bounds"):
            SenseAssignment(lemma=lemma, examples=[0, 1], word=word, sense=0)

    def test_rejects_negative_example_index(self) -> None:
        """Test that SenseAssignment rejects negative example index."""
        lemma = make_lemma_entry("chien", "NOUN")
        word = make_dictionary_entry("chien", "noun", "dog")

        with pytest.raises(ValueError, match="example index -1 out of bounds"):
            SenseAssignment(lemma=lemma, examples=[-1], word=word, sense=0)


class TestNeedsDisambiguation:
    """Tests for needs_disambiguation function."""

    def test_single_word_single_sense_returns_false(self) -> None:
        """Test that single word with single sense doesn't need disambiguation."""
        lemma = make_lemma_entry("chien", "NOUN")
        words = [make_dictionary_entry("chien", "noun", "dog")]
        entry = EnrichedLemma(lemma=lemma, words=words)

        assert needs_disambiguation(entry) is False

    def test_multiple_words_returns_true(self) -> None:
        """Test that multiple words need disambiguation."""
        lemma = make_lemma_entry("faux", "NOUN")
        words = [
            make_dictionary_entry("faux", "noun", "forgery"),
            make_dictionary_entry("faux", "noun", "scythe"),
        ]
        entry = EnrichedLemma(lemma=lemma, words=words)

        assert needs_disambiguation(entry) is True

    def test_single_word_multiple_senses_returns_true(self) -> None:
        """Test that single word with multiple senses needs disambiguation."""
        lemma = make_lemma_entry("chien", "NOUN")
        words = [make_dictionary_entry_with_senses("chien", "noun", ["dog", "hammer"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        assert needs_disambiguation(entry) is True


class TestAssignSingleSense:
    """Tests for assign_single_sense function."""

    def test_assigns_all_examples_to_single_sense(self) -> None:
        """Test that all examples are assigned to the single sense."""
        lemma = make_lemma_with_examples("chien", "NOUN", ["Le chien aboie.", "Mon chien dort."])
        words = [make_dictionary_entry("chien", "noun", "dog")]
        entry = EnrichedLemma(lemma=lemma, words=words)

        assignment = assign_single_sense(entry)

        assert assignment.lemma == lemma
        assert assignment.examples == [0, 1]
        assert assignment.word == words[0]
        assert assignment.sense == 0

    def test_raises_for_multi_sense_entry(self) -> None:
        """Test that AssertionError is raised for multi-sense entries."""
        lemma = make_lemma_entry("chien", "NOUN")
        words = [make_dictionary_entry_with_senses("chien", "noun", ["dog", "hammer"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        with pytest.raises(AssertionError, match="Use disambiguate_senses"):
            assign_single_sense(entry)

    def test_raises_for_multi_word_entry(self) -> None:
        """Test that AssertionError is raised for multi-word entries."""
        lemma = make_lemma_entry("faux", "NOUN")
        words = [
            make_dictionary_entry("faux", "noun", "forgery"),
            make_dictionary_entry("faux", "noun", "scythe"),
        ]
        entry = EnrichedLemma(lemma=lemma, words=words)

        with pytest.raises(AssertionError, match="Use disambiguate_senses"):
            assign_single_sense(entry)


class TestBuildPrompt:
    """Tests for _build_prompt function."""

    def test_builds_prompt_with_single_word_multiple_senses(self) -> None:
        """Test prompt building with multiple senses."""
        lemma = make_lemma_with_examples("chien", "NOUN", ["Le chien aboie."])
        words = [
            make_dictionary_entry_with_senses(
                "chien", "noun", ["dog", "hammer"], etymology="From Latin canis"
            )
        ]
        entry = EnrichedLemma(lemma=lemma, words=words)

        prompt = _build_prompt(entry, "French")

        assert "Language: French" in prompt
        assert "Word: chien" in prompt
        assert "1. [word=chien, etymology=From Latin canis] dog" in prompt
        assert "2. [word=chien, etymology=From Latin canis] hammer" in prompt
        assert "1. Le chien aboie." in prompt

    def test_builds_prompt_with_multiple_words(self) -> None:
        """Test prompt building with multiple words."""
        lemma = make_lemma_with_examples("faux", "NOUN", ["Un faux document."])
        words = [
            make_dictionary_entry_with_senses(
                "faux", "noun", ["forgery"], etymology="From Latin falsus"
            ),
            make_dictionary_entry_with_senses(
                "faux", "noun", ["scythe"], etymology="From Latin falx"
            ),
        ]
        entry = EnrichedLemma(lemma=lemma, words=words)

        prompt = _build_prompt(entry, "French")

        assert "1. [word=faux, etymology=From Latin falsus] forgery" in prompt
        assert "2. [word=faux, etymology=From Latin falx] scythe" in prompt

    def test_uses_unknown_for_missing_etymology(self) -> None:
        """Test that missing etymology shows as 'unknown'."""
        lemma = make_lemma_with_examples("test", "NOUN", ["A test."])
        words = [make_dictionary_entry_with_senses("test", "noun", ["a test"], etymology=None)]
        entry = EnrichedLemma(lemma=lemma, words=words)

        prompt = _build_prompt(entry, "French")

        assert "etymology=unknown" in prompt


class TestParseLlmResponse:
    """Tests for _parse_llm_response function."""

    def test_parses_valid_response(self) -> None:
        """Test parsing a valid LLM response."""
        lemma = make_lemma_with_examples("faux", "NOUN", ["Doc faux.", "La faux."])
        words = [
            make_dictionary_entry_with_senses("faux", "noun", ["forgery"]),
            make_dictionary_entry_with_senses("faux", "noun", ["scythe"]),
        ]
        entry = EnrichedLemma(lemma=lemma, words=words)

        response = DisambiguationResponse(
            assignments=[
                SentenceAssignment(sentence=1, sense=1),  # forgery
                SentenceAssignment(sentence=2, sense=2),  # scythe
            ]
        )

        results = _parse_llm_response(response, entry)

        assert len(results) == 2
        # Check first assignment (forgery)
        forgery_assignment = next(r for r in results if r.word.senses[0].translation == "forgery")
        assert forgery_assignment.examples == [0]
        assert forgery_assignment.sense == 0

        # Check second assignment (scythe)
        scythe_assignment = next(r for r in results if r.word.senses[0].translation == "scythe")
        assert scythe_assignment.examples == [1]
        assert scythe_assignment.sense == 0

    def test_groups_examples_by_sense(self) -> None:
        """Test that multiple examples for same sense are grouped."""
        lemma = make_lemma_with_examples("chien", "NOUN", ["Le chien 1.", "Le chien 2."])
        words = [make_dictionary_entry_with_senses("chien", "noun", ["dog", "hammer"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        response = DisambiguationResponse(
            assignments=[
                SentenceAssignment(sentence=1, sense=1),  # both are dogs
                SentenceAssignment(sentence=2, sense=1),
            ]
        )

        results = _parse_llm_response(response, entry)

        assert len(results) == 1
        assert results[0].examples == [0, 1]
        assert results[0].sense == 0  # dog sense

    def test_skips_null_sense(self) -> None:
        """Test that null senses are skipped with warning."""
        lemma = make_lemma_with_examples("test", "NOUN", ["Test sentence."])
        words = [make_dictionary_entry_with_senses("test", "noun", ["a test"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        response = DisambiguationResponse(assignments=[SentenceAssignment(sentence=1, sense=None)])

        results = _parse_llm_response(response, entry)

        assert len(results) == 0

    def test_skips_invalid_sense_number(self) -> None:
        """Test that invalid sense numbers are skipped."""
        lemma = make_lemma_with_examples("test", "NOUN", ["Test sentence."])
        words = [make_dictionary_entry_with_senses("test", "noun", ["a test"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        response = DisambiguationResponse(
            assignments=[SentenceAssignment(sentence=1, sense=99)]  # Invalid
        )

        results = _parse_llm_response(response, entry)

        assert len(results) == 0

    def test_skips_invalid_sentence_number(self) -> None:
        """Test that invalid sentence numbers are skipped."""
        lemma = make_lemma_with_examples("test", "NOUN", ["Test sentence."])
        words = [make_dictionary_entry_with_senses("test", "noun", ["a test"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        response = DisambiguationResponse(
            assignments=[SentenceAssignment(sentence=99, sense=1)]  # Invalid
        )

        results = _parse_llm_response(response, entry)

        assert len(results) == 0


class TestDisambiguateSenses:
    """Tests for disambiguate_senses async function."""

    @pytest.mark.asyncio
    async def test_raises_for_trivial_entry(self) -> None:
        """Test that AssertionError is raised for trivial entries."""
        lemma = make_lemma_entry("chien", "NOUN")
        words = [make_dictionary_entry("chien", "noun", "dog")]
        entry = EnrichedLemma(lemma=lemma, words=words)

        with pytest.raises(AssertionError, match="Use assign_single_sense"):
            await disambiguate_senses(entry, language="fr")

    @pytest.mark.asyncio
    async def test_raises_for_unknown_language_code(self) -> None:
        """Test that ValueError is raised for unknown language codes."""
        lemma = make_lemma_entry("chien", "NOUN")
        words = [make_dictionary_entry_with_senses("chien", "noun", ["dog", "hammer"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        with pytest.raises(ValueError, match="Unknown language code: xyz"):
            await disambiguate_senses(entry, language="xyz")

    @pytest.mark.asyncio
    async def test_calls_llm_and_parses_response(self) -> None:
        """Test that LLM is called and response is parsed."""
        lemma = make_lemma_with_examples("faux", "NOUN", ["Doc faux."])
        words = [
            make_dictionary_entry_with_senses("faux", "noun", ["forgery"]),
            make_dictionary_entry_with_senses("faux", "noun", ["scythe"]),
        ]
        entry = EnrichedLemma(lemma=lemma, words=words)

        # Mock the Anthropic client
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "submit_assignments"
        mock_tool_use.input = {"assignments": [{"sentence": 1, "sense": 1}]}

        mock_message = MagicMock()
        mock_message.content = [mock_tool_use]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with patch("vocab.pipeline.anthropic.AsyncAnthropic", return_value=mock_client):
            results = await disambiguate_senses(entry, language="fr")

        assert len(results) == 1
        assert results[0].examples == [0]
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_correct_model_mapping(self) -> None:
        """Test that model names are mapped correctly."""
        lemma = make_lemma_with_examples("test", "NOUN", ["Test."])
        words = [make_dictionary_entry_with_senses("test", "noun", ["a", "b"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "submit_assignments"
        mock_tool_use.input = {"assignments": [{"sentence": 1, "sense": 1}]}

        mock_message = MagicMock()
        mock_message.content = [mock_tool_use]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with patch("vocab.pipeline.anthropic.AsyncAnthropic", return_value=mock_client):
            await disambiguate_senses(entry, language="fr", model="claude-haiku")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-haiku-latest"

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_tool_call(self) -> None:
        """Test that empty list is returned when LLM doesn't use tool."""
        lemma = make_lemma_with_examples("test", "NOUN", ["Test."])
        words = [make_dictionary_entry_with_senses("test", "noun", ["a", "b"])]
        entry = EnrichedLemma(lemma=lemma, words=words)

        mock_text = MagicMock()
        mock_text.type = "text"

        mock_message = MagicMock()
        mock_message.content = [mock_text]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with patch("vocab.pipeline.anthropic.AsyncAnthropic", return_value=mock_client):
            results = await disambiguate_senses(entry, language="fr")

        assert results == []
