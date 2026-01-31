"""Tests for vocabulary aggregation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import spacy
from ebooklib import epub
from spacy.language import Language

from vocab.models import Example, LemmaEntry, SentenceLocation, Vocabulary
from vocab.vocabulary import build_vocabulary


@pytest.fixture
def mock_spacy_model_for_vocab() -> Language:
    """Create a mock spaCy model for testing vocabulary building.

    This uses a blank French model. A blank model returns token.text as the lemma,
    which is sufficient for testing the vocabulary aggregation logic.

    Returns:
        The blank spaCy model.
    """
    nlp = spacy.blank("fr")
    nlp.add_pipe("sentencizer")
    return nlp


@pytest.fixture
def simple_vocab_epub(tmp_path: Path) -> Path:
    """Create a simple epub for testing vocabulary building.

    Contains repeated words to test frequency counting.

    Returns:
        Path to the created epub file.
    """
    book = epub.EpubBook()
    book.set_identifier("vocab-test-001")
    book.set_title("Vocabulary Test Book")
    book.set_language("fr")

    # Chapter with repeated words for frequency testing
    ch1 = epub.EpubHtml(title="Chapitre Un", file_name="ch1.xhtml", lang="fr")
    ch1.content = """
    <html>
    <body>
        <h1>Chapitre Un</h1>
        <p>Le chat dort. Le chat mange. Le chien court.</p>
    </body>
    </html>
    """
    book.add_item(ch1)

    # Chapter 2 with more occurrences
    ch2 = epub.EpubHtml(title="Chapitre Deux", file_name="ch2.xhtml", lang="fr")
    ch2.content = """
    <html>
    <body>
        <h1>Chapitre Deux</h1>
        <p>Le chat dort encore. Le petit chat joue.</p>
    </body>
    </html>
    """
    book.add_item(ch2)

    book.toc = [
        epub.Link("ch1.xhtml", "Chapitre Un", "ch1"),
        epub.Link("ch2.xhtml", "Chapitre Deux", "ch2"),
    ]

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", ch1, ch2]

    epub_path = tmp_path / "vocab_test.epub"
    epub.write_epub(str(epub_path), book)

    return epub_path


@pytest.fixture
def epub_for_examples(tmp_path: Path) -> Path:
    """Create an epub with many sentences for testing example limits.

    Returns:
        Path to the created epub file.
    """
    book = epub.EpubBook()
    book.set_identifier("examples-test-001")
    book.set_title("Examples Test Book")
    book.set_language("fr")

    # Multiple sentences with the same word
    ch1 = epub.EpubHtml(title="Chapter One", file_name="ch1.xhtml", lang="fr")
    ch1.content = """
    <html>
    <body>
        <p>Le chat dort.</p>
        <p>Le chat mange.</p>
        <p>Le chat joue.</p>
        <p>Le chat court.</p>
        <p>Le chat saute.</p>
    </body>
    </html>
    """
    book.add_item(ch1)

    book.toc = [epub.Link("ch1.xhtml", "Chapter One", "ch1")]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", ch1]

    epub_path = tmp_path / "examples_test.epub"
    epub.write_epub(str(epub_path), book)

    return epub_path


class TestBuildVocabulary:
    """Tests for build_vocabulary function."""

    def test_returns_vocabulary_object(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should return a Vocabulary instance."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        assert isinstance(vocab, Vocabulary)
        assert vocab.language == "fr"

    def test_tracks_frequencies(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should correctly count token frequencies."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        # "Le" appears multiple times across both chapters
        assert "Le" in vocab.entries
        assert vocab.entries["Le"].frequency > 1

        # "chat" appears multiple times
        assert "chat" in vocab.entries
        assert vocab.entries["chat"].frequency > 1

    def test_tracks_forms(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should track all original forms for each lemma."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        # Check forms dict exists and has correct structure
        entry = vocab.entries["chat"]
        assert isinstance(entry.forms, dict)
        assert "chat" in entry.forms
        assert entry.forms["chat"] == entry.frequency  # In our mock, lemma == original

    def test_caps_examples_at_max(
        self, mock_spacy_model_for_vocab: Language, epub_for_examples: Path
    ) -> None:
        """Should limit examples to max_examples per lemma."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(epub_for_examples, "fr", max_examples=2)

        # "chat" appears 5 times but should only have 2 examples
        entry = vocab.entries["chat"]
        assert len(entry.examples) <= 2

    def test_examples_are_unique_sentences(
        self, mock_spacy_model_for_vocab: Language, epub_for_examples: Path
    ) -> None:
        """Should not include duplicate sentences in examples."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(epub_for_examples, "fr", max_examples=10)

        for entry in vocab.entries.values():
            sentences = [ex.sentence for ex in entry.examples]
            assert len(sentences) == len(set(sentences)), f"Duplicate sentences for {entry.lemma}"

    def test_entries_contain_lemma_entry_objects(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should have LemmaEntry objects in entries dict."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        for lemma, entry in vocab.entries.items():
            assert isinstance(entry, LemmaEntry)
            assert entry.lemma == lemma

    def test_examples_have_correct_location(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should include proper SentenceLocation in examples."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        for entry in vocab.entries.values():
            for ex in entry.examples:
                assert isinstance(ex, Example)
                assert isinstance(ex.sentence, str)
                assert isinstance(ex.location, SentenceLocation)
                assert isinstance(ex.location.chapter_index, int)
                assert isinstance(ex.location.sentence_index, int)


class TestVocabularyTop:
    """Tests for Vocabulary.top() method."""

    def test_top_returns_sorted_by_frequency(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should return entries sorted by frequency descending."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        top_entries = vocab.top(10)

        # Verify sorted by frequency descending
        for i in range(len(top_entries) - 1):
            assert top_entries[i].frequency >= top_entries[i + 1].frequency

    def test_top_limits_results(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should return at most n entries."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        top_3 = vocab.top(3)
        assert len(top_3) <= 3

    def test_top_with_n_larger_than_entries(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should return all entries when n > total entries."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        top_1000 = vocab.top(1000)
        assert len(top_1000) == len(vocab.entries)

    def test_top_returns_lemma_entry_objects(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should return list of LemmaEntry objects."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        top_entries = vocab.top(5)
        assert all(isinstance(e, LemmaEntry) for e in top_entries)


class TestVocabularyToDict:
    """Tests for Vocabulary.to_dict() method."""

    def test_to_dict_is_json_serializable(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should produce JSON-serializable output."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()

        # Should not raise
        json_str = json.dumps(vocab_dict)
        assert json_str

    def test_to_dict_contains_language(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should include language in output."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()
        assert vocab_dict["language"] == "fr"

    def test_to_dict_contains_entries(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should include all entries."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()
        assert "entries" in vocab_dict
        assert len(vocab_dict["entries"]) == len(vocab.entries)

    def test_to_dict_entry_structure(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should have correct structure for each entry."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()

        for lemma, entry_dict in vocab_dict["entries"].items():
            assert entry_dict["lemma"] == lemma
            assert "frequency" in entry_dict
            assert "forms" in entry_dict
            assert "examples" in entry_dict

    def test_to_dict_example_structure(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should have correct structure for examples."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()

        for entry_dict in vocab_dict["entries"].values():
            for example in entry_dict["examples"]:
                assert "sentence" in example
                assert "location" in example
                assert "chapter_index" in example["location"]
                assert "chapter_title" in example["location"]
                assert "sentence_index" in example["location"]


class TestEmptyVocabulary:
    """Tests for edge cases with empty or minimal input."""

    def test_empty_epub_produces_empty_vocabulary(
        self, mock_spacy_model_for_vocab: Language, tmp_path: Path
    ) -> None:
        """Should handle epub with no extractable text."""
        book = epub.EpubBook()
        book.set_identifier("empty-001")
        book.set_title("Empty Book")
        book.set_language("fr")

        # Chapter with only whitespace
        ch1 = epub.EpubHtml(title="Empty", file_name="ch1.xhtml", lang="fr")
        ch1.content = "<html><body><p>   </p></body></html>"
        book.add_item(ch1)

        book.toc = []
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1]

        epub_path = tmp_path / "empty.epub"
        epub.write_epub(str(epub_path), book)

        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(epub_path, "fr")

        assert isinstance(vocab, Vocabulary)
        assert len(vocab.entries) == 0

    def test_top_on_empty_vocabulary(
        self, mock_spacy_model_for_vocab: Language, tmp_path: Path
    ) -> None:
        """Should return empty list for top() on empty vocabulary."""
        book = epub.EpubBook()
        book.set_identifier("empty-002")
        book.set_title("Empty Book")
        book.set_language("fr")

        ch1 = epub.EpubHtml(title="Empty", file_name="ch1.xhtml", lang="fr")
        ch1.content = "<html><body><p>   </p></body></html>"
        book.add_item(ch1)

        book.toc = []
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1]

        epub_path = tmp_path / "empty2.epub"
        epub.write_epub(str(epub_path), book)

        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(epub_path, "fr")

        assert vocab.top(10) == []

    def test_to_dict_on_empty_vocabulary(
        self, mock_spacy_model_for_vocab: Language, tmp_path: Path
    ) -> None:
        """Should produce valid dict for empty vocabulary."""
        book = epub.EpubBook()
        book.set_identifier("empty-003")
        book.set_title("Empty Book")
        book.set_language("fr")

        ch1 = epub.EpubHtml(title="Empty", file_name="ch1.xhtml", lang="fr")
        ch1.content = "<html><body><p>   </p></body></html>"
        book.add_item(ch1)

        book.toc = []
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1]

        epub_path = tmp_path / "empty3.epub"
        epub.write_epub(str(epub_path), book)

        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(epub_path, "fr")

        vocab_dict = vocab.to_dict()
        assert vocab_dict["language"] == "fr"
        assert vocab_dict["entries"] == {}
        assert json.dumps(vocab_dict)  # Should be serializable


class TestMaxExamplesValidation:
    """Tests for max_examples parameter validation."""

    def test_max_examples_zero_is_valid(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should accept max_examples=0 (no examples collected)."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr", max_examples=0)

        # Should have no examples for any entry
        for entry in vocab.entries.values():
            assert len(entry.examples) == 0

    def test_max_examples_negative_raises_value_error(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should raise ValueError when max_examples is negative."""
        with pytest.raises(ValueError, match="max_examples must be >= 0"):
            build_vocabulary(simple_vocab_epub, "fr", max_examples=-1)

    def test_max_examples_one_is_valid(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should accept max_examples=1 as valid."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr", max_examples=1)

        # Should have at most 1 example per entry
        for entry in vocab.entries.values():
            assert len(entry.examples) <= 1


class TestTopNValidation:
    """Tests for Vocabulary.top(n) parameter validation."""

    def test_top_zero_raises_value_error(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should raise ValueError when n is 0."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        with pytest.raises(ValueError, match="n must be >= 1"):
            vocab.top(0)

    def test_top_negative_raises_value_error(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should raise ValueError when n is negative."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        with pytest.raises(ValueError, match="n must be >= 1"):
            vocab.top(-1)

    def test_top_one_is_valid(
        self, mock_spacy_model_for_vocab: Language, simple_vocab_epub: Path
    ) -> None:
        """Should accept n=1 as valid."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        top_1 = vocab.top(1)
        assert len(top_1) == 1


class TestDuplicateTokenInSentence:
    """Tests for handling duplicate tokens within the same sentence."""

    @pytest.fixture
    def epub_with_duplicate_word(self, tmp_path: Path) -> Path:
        """Create an epub with a sentence containing the same word twice.

        Returns:
            Path to the created epub file.
        """
        book = epub.EpubBook()
        book.set_identifier("duplicate-test-001")
        book.set_title("Duplicate Word Test")
        book.set_language("fr")

        ch1 = epub.EpubHtml(title="Chapter One", file_name="ch1.xhtml", lang="fr")
        # "chat" appears twice in the same sentence
        ch1.content = """
        <html>
        <body>
            <p>Le chat voit un autre chat.</p>
        </body>
        </html>
        """
        book.add_item(ch1)

        book.toc = [epub.Link("ch1.xhtml", "Chapter One", "ch1")]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", ch1]

        epub_path = tmp_path / "duplicate_word.epub"
        epub.write_epub(str(epub_path), book)

        return epub_path

    def test_duplicate_word_counts_frequency_correctly(
        self, mock_spacy_model_for_vocab: Language, epub_with_duplicate_word: Path
    ) -> None:
        """Should count each occurrence for frequency."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(epub_with_duplicate_word, "fr")

        # "chat" appears twice, frequency should be 2
        assert vocab.entries["chat"].frequency == 2

    def test_duplicate_word_has_single_example(
        self, mock_spacy_model_for_vocab: Language, epub_with_duplicate_word: Path
    ) -> None:
        """Should only include one example for duplicate word in same sentence."""
        with (
            patch("vocab.sentences.get_model", return_value=mock_spacy_model_for_vocab),
            patch("vocab.tokens.get_model", return_value=mock_spacy_model_for_vocab),
        ):
            vocab = build_vocabulary(epub_with_duplicate_word, "fr", max_examples=10)

        # Despite "chat" appearing twice, should only have 1 example (same sentence)
        assert len(vocab.entries["chat"].examples) == 1
