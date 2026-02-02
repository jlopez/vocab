"""Tests for the pipeline module."""

from unittest.mock import MagicMock

from vocab.dictionary import DictionaryEntry, DictionarySense
from vocab.models import Example, LemmaEntry, SentenceLocation, Vocabulary
from vocab.pipeline import EnrichedLemma, generate_enriched_lemmas


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


class TestGenerateEnrichedLemmas:
    """Tests for generate_enriched_lemmas function."""

    def test_yields_enriched_lemmas_for_matches(self) -> None:
        """Test that matching lemmas are yielded."""
        vocab = Vocabulary(
            entries={
                "chien": {"NOUN": make_lemma_entry("chien", "NOUN")},
            },
            language="fr",
        )

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("chien", "noun", "dog")]

        results = list(generate_enriched_lemmas(vocab, mock_dict))

        assert len(results) == 1
        assert results[0].lemma.lemma == "chien"
        assert results[0].words[0].senses[0].translation == "dog"

    def test_skips_lemmas_without_matches(self) -> None:
        """Test that lemmas without dictionary matches are skipped."""
        vocab = Vocabulary(
            entries={
                "xyznonexistent": {"NOUN": make_lemma_entry("xyznonexistent", "NOUN")},
            },
            language="fr",
        )

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = []  # No matches

        results = list(generate_enriched_lemmas(vocab, mock_dict))

        assert len(results) == 0

    def test_uses_spacy_to_kaikki_mapping(self) -> None:
        """Test that POS is mapped from spaCy to kaikki format."""
        vocab = Vocabulary(
            entries={
                "manger": {"VERB": make_lemma_entry("manger", "VERB")},
            },
            language="fr",
        )

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("manger", "verb", "to eat")]

        list(generate_enriched_lemmas(vocab, mock_dict))

        # Should have called lookup with kaikki POS ["verb"]
        mock_dict.lookup.assert_called_once_with("manger", pos=["verb"])

    def test_handles_unmapped_pos(self) -> None:
        """Test that unmapped POS tags pass None to lookup."""
        vocab = Vocabulary(
            entries={
                "test": {"UNKNOWN_POS": make_lemma_entry("test", "UNKNOWN_POS")},
            },
            language="fr",
        )

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("test", "noun")]

        list(generate_enriched_lemmas(vocab, mock_dict))

        # Should have called lookup with pos=None for unmapped POS
        mock_dict.lookup.assert_called_once_with("test", pos=None)

    def test_processes_multiple_lemmas(self) -> None:
        """Test processing vocabulary with multiple lemmas."""
        vocab = Vocabulary(
            entries={
                "chien": {"NOUN": make_lemma_entry("chien", "NOUN")},
                "manger": {"VERB": make_lemma_entry("manger", "VERB")},
                "beau": {"ADJ": make_lemma_entry("beau", "ADJ")},
            },
            language="fr",
        )

        mock_dict = MagicMock()
        mock_dict.lookup.side_effect = [
            [make_dictionary_entry("chien", "noun", "dog")],
            [make_dictionary_entry("manger", "verb", "to eat")],
            [],  # beau has no match
        ]

        results = list(generate_enriched_lemmas(vocab, mock_dict))

        assert len(results) == 2
        lemmas = {r.lemma.lemma for r in results}
        assert lemmas == {"chien", "manger"}

    def test_processes_same_lemma_different_pos(self) -> None:
        """Test processing same lemma with different POS tags."""
        vocab = Vocabulary(
            entries={
                "faux": {
                    "NOUN": make_lemma_entry("faux", "NOUN"),
                    "ADJ": make_lemma_entry("faux", "ADJ"),
                },
            },
            language="fr",
        )

        mock_dict = MagicMock()
        mock_dict.lookup.side_effect = [
            [make_dictionary_entry("faux", "noun", "forgery")],
            [make_dictionary_entry("faux", "adj", "false")],
        ]

        results = list(generate_enriched_lemmas(vocab, mock_dict))

        assert len(results) == 2
        pos_set = {r.lemma.pos for r in results}
        assert pos_set == {"NOUN", "ADJ"}

    def test_returns_multiple_dictionary_entries(self) -> None:
        """Test that multiple dictionary entries are preserved."""
        vocab = Vocabulary(
            entries={
                "faux": {"NOUN": make_lemma_entry("faux", "NOUN")},
            },
            language="fr",
        )

        mock_dict = MagicMock()
        # Two different etymologies for "faux" as noun
        mock_dict.lookup.return_value = [
            make_dictionary_entry("faux", "noun", "forgery"),
            make_dictionary_entry("faux", "noun", "scythe"),
        ]

        results = list(generate_enriched_lemmas(vocab, mock_dict))

        assert len(results) == 1
        assert len(results[0].words) == 2
        translations = {w.senses[0].translation for w in results[0].words}
        assert translations == {"forgery", "scythe"}

    def test_empty_vocabulary_yields_nothing(self) -> None:
        """Test that empty vocabulary yields no results."""
        vocab = Vocabulary(entries={}, language="fr")
        mock_dict = MagicMock()

        results = list(generate_enriched_lemmas(vocab, mock_dict))

        assert len(results) == 0
        mock_dict.lookup.assert_not_called()

    def test_adp_maps_to_multiple_pos(self) -> None:
        """Test that ADP maps to prep, prep_phrase, postp."""
        vocab = Vocabulary(
            entries={
                "dans": {"ADP": make_lemma_entry("dans", "ADP")},
            },
            language="fr",
        )

        mock_dict = MagicMock()
        mock_dict.lookup.return_value = [make_dictionary_entry("dans", "prep", "in")]

        list(generate_enriched_lemmas(vocab, mock_dict))

        # Should call with all mapped POS tags
        call_args = mock_dict.lookup.call_args
        assert call_args[0][0] == "dans"
        assert set(call_args[1]["pos"]) == {"prep", "prep_phrase", "postp"}
