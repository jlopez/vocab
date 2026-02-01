"""Tests for vocabulary filtering."""

import pytest

from vocab import Vocabulary, filter_vocabulary
from vocab.models import Example, LemmaEntry, SentenceLocation


@pytest.fixture
def sample_location() -> SentenceLocation:
    """Create a sample sentence location for testing."""
    return SentenceLocation(chapter_index=0, chapter_title="Chapter 1", sentence_index=0)


@pytest.fixture
def sample_vocabulary(sample_location: SentenceLocation) -> Vocabulary:
    """Create a sample vocabulary for testing filtering."""
    return Vocabulary(
        language="fr",
        entries={
            "chat": LemmaEntry(
                lemma="chat",
                frequency=10,
                forms={"chat": 8, "chats": 2},
                examples=[
                    Example(sentence="Le chat dort.", location=sample_location),
                ],
            ),
            "chien": LemmaEntry(
                lemma="chien",
                frequency=5,
                forms={"chien": 5},
                examples=[
                    Example(sentence="Le chien court.", location=sample_location),
                ],
            ),
            "manger": LemmaEntry(
                lemma="manger",
                frequency=3,
                forms={"manger": 2, "mange": 1},
                examples=[
                    Example(sentence="Il mange.", location=sample_location),
                ],
            ),
            "petit": LemmaEntry(
                lemma="petit",
                frequency=1,
                forms={"petit": 1},
                examples=[
                    Example(sentence="Le petit chat.", location=sample_location),
                ],
            ),
            "Paris": LemmaEntry(
                lemma="Paris",
                frequency=8,
                forms={"Paris": 8},
                examples=[
                    Example(sentence="Il vit a Paris.", location=sample_location),
                ],
            ),
            "Marie": LemmaEntry(
                lemma="Marie",
                frequency=4,
                forms={"Marie": 4},
                examples=[
                    Example(sentence="Marie chante.", location=sample_location),
                ],
            ),
        },
    )


class TestVocabularyFromDict:
    """Tests for Vocabulary.from_dict() method."""

    def test_roundtrip_preserves_language(self, sample_vocabulary: Vocabulary) -> None:
        """Should preserve language through to_dict/from_dict roundtrip."""
        vocab_dict = sample_vocabulary.to_dict()
        restored = Vocabulary.from_dict(vocab_dict)
        assert restored.language == sample_vocabulary.language

    def test_roundtrip_preserves_entries(self, sample_vocabulary: Vocabulary) -> None:
        """Should preserve all entries through roundtrip."""
        vocab_dict = sample_vocabulary.to_dict()
        restored = Vocabulary.from_dict(vocab_dict)
        assert set(restored.entries.keys()) == set(sample_vocabulary.entries.keys())

    def test_roundtrip_preserves_lemma_entry_fields(self, sample_vocabulary: Vocabulary) -> None:
        """Should preserve LemmaEntry fields through roundtrip."""
        vocab_dict = sample_vocabulary.to_dict()
        restored = Vocabulary.from_dict(vocab_dict)

        for lemma, original in sample_vocabulary.entries.items():
            restored_entry = restored.entries[lemma]
            assert restored_entry.lemma == original.lemma
            assert restored_entry.frequency == original.frequency
            assert restored_entry.forms == original.forms
            assert len(restored_entry.examples) == len(original.examples)

    def test_roundtrip_preserves_example_structure(self, sample_vocabulary: Vocabulary) -> None:
        """Should preserve Example and SentenceLocation through roundtrip."""
        vocab_dict = sample_vocabulary.to_dict()
        restored = Vocabulary.from_dict(vocab_dict)

        for lemma in sample_vocabulary.entries:
            original_examples = sample_vocabulary.entries[lemma].examples
            restored_examples = restored.entries[lemma].examples

            for orig, rest in zip(original_examples, restored_examples, strict=True):
                assert rest.sentence == orig.sentence
                assert rest.location.chapter_index == orig.location.chapter_index
                assert rest.location.chapter_title == orig.location.chapter_title
                assert rest.location.sentence_index == orig.location.sentence_index

    def test_from_dict_returns_vocabulary_instance(self, sample_vocabulary: Vocabulary) -> None:
        """Should return Vocabulary instance."""
        vocab_dict = sample_vocabulary.to_dict()
        restored = Vocabulary.from_dict(vocab_dict)
        assert isinstance(restored, Vocabulary)

    def test_from_dict_entries_are_lemma_entry_instances(
        self, sample_vocabulary: Vocabulary
    ) -> None:
        """Should create LemmaEntry instances for entries."""
        vocab_dict = sample_vocabulary.to_dict()
        restored = Vocabulary.from_dict(vocab_dict)

        for entry in restored.entries.values():
            assert isinstance(entry, LemmaEntry)

    def test_from_dict_examples_are_example_instances(self, sample_vocabulary: Vocabulary) -> None:
        """Should create Example instances for examples."""
        vocab_dict = sample_vocabulary.to_dict()
        restored = Vocabulary.from_dict(vocab_dict)

        for entry in restored.entries.values():
            for ex in entry.examples:
                assert isinstance(ex, Example)
                assert isinstance(ex.location, SentenceLocation)


class TestFilterVocabulary:
    """Tests for filter_vocabulary function."""

    def test_returns_list_of_lemma_entries(self, sample_vocabulary: Vocabulary) -> None:
        """Should return list of LemmaEntry objects."""
        result = filter_vocabulary(sample_vocabulary)
        assert isinstance(result, list)
        assert all(isinstance(e, LemmaEntry) for e in result)

    def test_default_excludes_proper_nouns(self, sample_vocabulary: Vocabulary) -> None:
        """Should exclude proper nouns by default."""
        result = filter_vocabulary(sample_vocabulary)
        lemmas = [e.lemma for e in result]
        assert "Paris" not in lemmas
        assert "Marie" not in lemmas
        assert "chat" in lemmas

    def test_include_proper_nouns(self, sample_vocabulary: Vocabulary) -> None:
        """Should include proper nouns when exclude_proper_nouns=False."""
        result = filter_vocabulary(sample_vocabulary, exclude_proper_nouns=False)
        lemmas = [e.lemma for e in result]
        assert "Paris" in lemmas
        assert "Marie" in lemmas

    def test_min_freq_filters_low_frequency(self, sample_vocabulary: Vocabulary) -> None:
        """Should exclude entries below min_freq."""
        result = filter_vocabulary(sample_vocabulary, min_freq=3)
        lemmas = [e.lemma for e in result]
        assert "petit" not in lemmas  # frequency 1
        assert "chat" in lemmas  # frequency 10

    def test_max_freq_filters_high_frequency(self, sample_vocabulary: Vocabulary) -> None:
        """Should exclude entries above max_freq."""
        result = filter_vocabulary(sample_vocabulary, max_freq=5, exclude_proper_nouns=False)
        lemmas = [e.lemma for e in result]
        assert "chat" not in lemmas  # frequency 10
        assert "chien" in lemmas  # frequency 5
        assert "petit" in lemmas  # frequency 1

    def test_min_and_max_freq_combined(self, sample_vocabulary: Vocabulary) -> None:
        """Should filter by both min and max frequency."""
        result = filter_vocabulary(
            sample_vocabulary, min_freq=3, max_freq=8, exclude_proper_nouns=False
        )
        frequencies = [e.frequency for e in result]
        assert all(3 <= f <= 8 for f in frequencies)

    def test_results_sorted_by_frequency_descending(self, sample_vocabulary: Vocabulary) -> None:
        """Should return results sorted by frequency descending."""
        result = filter_vocabulary(sample_vocabulary, exclude_proper_nouns=False)
        frequencies = [e.frequency for e in result]
        assert frequencies == sorted(frequencies, reverse=True)

    def test_min_freq_inclusive(self, sample_vocabulary: Vocabulary) -> None:
        """Should include entries with exactly min_freq."""
        result = filter_vocabulary(sample_vocabulary, min_freq=5)
        lemmas = [e.lemma for e in result]
        assert "chien" in lemmas  # frequency exactly 5

    def test_max_freq_inclusive(self, sample_vocabulary: Vocabulary) -> None:
        """Should include entries with exactly max_freq."""
        result = filter_vocabulary(sample_vocabulary, max_freq=10, exclude_proper_nouns=False)
        lemmas = [e.lemma for e in result]
        assert "chat" in lemmas  # frequency exactly 10

    def test_empty_vocabulary(self) -> None:
        """Should handle empty vocabulary."""
        vocab = Vocabulary(language="fr", entries={})
        result = filter_vocabulary(vocab)
        assert result == []

    def test_all_filtered_out(self, sample_vocabulary: Vocabulary) -> None:
        """Should return empty list when all entries filtered."""
        result = filter_vocabulary(sample_vocabulary, min_freq=100)
        assert result == []

    def test_default_min_freq_is_1(self, sample_vocabulary: Vocabulary) -> None:
        """Should use min_freq=1 by default (include all frequencies)."""
        result = filter_vocabulary(sample_vocabulary, exclude_proper_nouns=False)
        lemmas = [e.lemma for e in result]
        assert "petit" in lemmas  # frequency 1

    def test_preserves_entry_data(self, sample_vocabulary: Vocabulary) -> None:
        """Should preserve all data in filtered entries."""
        result = filter_vocabulary(sample_vocabulary)
        chat_entry = next(e for e in result if e.lemma == "chat")

        original = sample_vocabulary.entries["chat"]
        assert chat_entry.frequency == original.frequency
        assert chat_entry.forms == original.forms
        assert len(chat_entry.examples) == len(original.examples)
