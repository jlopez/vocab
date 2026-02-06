"""Tests for vocabulary aggregation."""

import json
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ebooklib import epub

from vocab.models import Example, LemmaEntry, SentenceLocation, Token, Vocabulary
from vocab.vocabulary import SKIP_POS, VocabularyBuilder, build_vocabulary


@contextmanager
def mock_vocab_pipeline(
    tokens: Sequence[Token] | Callable[..., list[Token]],
    sentences: list[str] | None = None,
    chapter_title: str = "Test",
    chapter_index: int = 0,
) -> Generator[None, None, None]:
    """Context manager to mock the vocabulary building pipeline.

    Args:
        tokens: Either a list of tokens to return, or a callable (side_effect)
            that takes (text, location, language) and returns tokens.
        sentences: List of sentence texts. Defaults to ["sentence"].
        chapter_title: Title for the mock chapter.
        chapter_index: Index for the mock chapter.

    Yields:
        None - use build_vocabulary inside the context.
    """
    if sentences is None:
        sentences = ["sentence"]

    mock_sentences_list = [MagicMock(text=s, index=i) for i, s in enumerate(sentences)]

    token_config: dict[str, Any] = {}
    if callable(tokens) and not isinstance(tokens, (list, tuple)):
        token_config["side_effect"] = tokens
    else:
        token_config["return_value"] = tokens

    with (
        patch("vocab.vocabulary.extract_chapters") as mock_chapters,
        patch("vocab.vocabulary.extract_sentences") as mock_sentences,
        patch("vocab.vocabulary.extract_tokens", **token_config),
    ):
        mock_chapters.return_value = [
            MagicMock(index=chapter_index, title=chapter_title, text="text")
        ]
        mock_sentences.return_value = mock_sentences_list
        yield


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


def create_mock_token(
    lemma: str, pos: str, original: str, sentence: str, location: SentenceLocation
) -> Token:
    """Create a Token with specified values."""
    return Token(
        lemma=lemma,
        pos=pos,
        morph={},
        original=original,
        sentence=sentence,
        location=location,
    )


class TestBuildVocabulary:
    """Tests for build_vocabulary function."""

    def test_returns_vocabulary_object(self, simple_vocab_epub: Path) -> None:
        """Should return a Vocabulary instance."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        assert isinstance(vocab, Vocabulary)
        assert vocab.language == "fr"

    def test_tracks_frequencies(self, simple_vocab_epub: Path) -> None:
        """Should correctly count token frequencies."""

        def mock_extract_tokens(text: str, loc: SentenceLocation, lang: str) -> list[Token]:
            if "chat mange" in text:
                return [
                    create_mock_token("le", "DET", "Le", text, loc),
                    create_mock_token("chat", "NOUN", "chat", text, loc),
                    create_mock_token("manger", "VERB", "mange", text, loc),
                ]
            elif "chat dort" in text:
                return [
                    create_mock_token("le", "DET", "Le", text, loc),
                    create_mock_token("chat", "NOUN", "chat", text, loc),
                    create_mock_token("dormir", "VERB", "dort", text, loc),
                ]
            return []

        with mock_vocab_pipeline(
            mock_extract_tokens, sentences=["Le chat dort.", "Le chat mange."]
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        # "chat" appears twice as NOUN
        assert "chat" in vocab.entries
        assert "NOUN" in vocab.entries["chat"]
        assert vocab.entries["chat"]["NOUN"].frequency == 2

        # "le" appears twice as DET
        assert vocab.entries["le"]["DET"].frequency == 2

    def test_tracks_forms(self, simple_vocab_epub: Path) -> None:
        """Should track all original forms for each lemma."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [
            create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location),
            create_mock_token("chat", "NOUN", "chats", "Les chats dorment.", location),
        ]

        with mock_vocab_pipeline(tokens):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        entry = vocab.entries["chat"]["NOUN"]
        assert isinstance(entry.forms, dict)
        assert "chat" in entry.forms
        assert "chats" in entry.forms
        assert entry.forms["chat"] == 1
        assert entry.forms["chats"] == 1

    def test_caps_examples_at_max(self, epub_for_examples: Path) -> None:
        """Should limit examples to max_examples per lemma."""
        sentences = [
            "Le chat dort.",
            "Le chat mange.",
            "Le chat joue.",
            "Le chat court.",
            "Le chat saute.",
        ]

        call_count = [0]

        def mock_extract_tokens(text: str, loc: SentenceLocation, lang: str) -> list[Token]:
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(sentences):
                return [create_mock_token("chat", "NOUN", "chat", sentences[idx], loc)]
            return []

        with mock_vocab_pipeline(mock_extract_tokens, sentences=sentences):
            vocab = build_vocabulary(epub_for_examples, "fr", max_examples=2)

        entry = vocab.entries["chat"]["NOUN"]
        assert len(entry.examples) <= 2

    def test_examples_are_unique_sentences(self, epub_for_examples: Path) -> None:
        """Should not include duplicate sentences in examples."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [
            create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location),
            create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location),
        ]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(epub_for_examples, "fr", max_examples=10)

        entry = vocab.entries["chat"]["NOUN"]
        sentences = [ex.sentence for ex in entry.examples]
        assert len(sentences) == len(set(sentences))

    def test_entries_contain_lemma_entry_objects(self, simple_vocab_epub: Path) -> None:
        """Should have LemmaEntry objects in entries dict."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [
            create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location),
            create_mock_token("dormir", "VERB", "dort", "Le chat dort.", location),
        ]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        for lemma, pos_dict in vocab.entries.items():
            for pos, entry in pos_dict.items():
                assert isinstance(entry, LemmaEntry)
                assert entry.lemma == lemma
                assert entry.pos == pos

    def test_examples_have_correct_location(self, simple_vocab_epub: Path) -> None:
        """Should include proper SentenceLocation in examples."""
        location = SentenceLocation(chapter_index=1, chapter_title="Chapter 2", sentence_index=3)
        tokens = [create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)]

        with mock_vocab_pipeline(
            tokens, sentences=["Le chat dort."], chapter_title="Chapter 2", chapter_index=1
        ):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        entry = vocab.entries["chat"]["NOUN"]
        assert len(entry.examples) == 1
        ex = entry.examples[0]
        assert isinstance(ex, Example)
        assert isinstance(ex.sentence, str)
        assert isinstance(ex.location, SentenceLocation)
        assert ex.location.chapter_index == 1
        assert ex.location.chapter_title == "Chapter 2"
        assert ex.location.sentence_index == 3

    def test_same_lemma_different_pos_creates_separate_entries(
        self, simple_vocab_epub: Path
    ) -> None:
        """Same lemma with different POS should create separate entries."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [
            create_mock_token("run", "NOUN", "run", "That was a good run.", location),
            create_mock_token("run", "VERB", "run", "I like to run.", location),
            create_mock_token("run", "VERB", "runs", "She runs fast.", location),
        ]

        with mock_vocab_pipeline(tokens):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        # Should have "run" with two different POS
        assert "run" in vocab.entries
        assert "NOUN" in vocab.entries["run"]
        assert "VERB" in vocab.entries["run"]

        # Separate frequency counts
        assert vocab.entries["run"]["NOUN"].frequency == 1
        assert vocab.entries["run"]["VERB"].frequency == 2


class TestVocabularyToDict:
    """Tests for Vocabulary.to_dict() method."""

    def test_to_dict_is_json_serializable(self, simple_vocab_epub: Path) -> None:
        """Should produce JSON-serializable output."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()
        json_str = json.dumps(vocab_dict)
        assert json_str

    def test_to_dict_contains_language(self, simple_vocab_epub: Path) -> None:
        """Should include language in output."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()
        assert vocab_dict["language"] == "fr"

    def test_to_dict_contains_nested_entries(self, simple_vocab_epub: Path) -> None:
        """Should include entries with nested lemma -> pos -> entry structure."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [
            create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location),
            create_mock_token("dormir", "VERB", "dort", "Le chat dort.", location),
        ]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()
        assert "entries" in vocab_dict
        assert "chat" in vocab_dict["entries"]
        assert "NOUN" in vocab_dict["entries"]["chat"]

    def test_to_dict_entry_structure(self, simple_vocab_epub: Path) -> None:
        """Should have correct structure for each entry."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()

        for lemma, pos_dict in vocab_dict["entries"].items():
            for pos, entry_dict in pos_dict.items():
                assert entry_dict["lemma"] == lemma
                assert entry_dict["pos"] == pos
                assert "frequency" in entry_dict
                assert "forms" in entry_dict
                assert "examples" in entry_dict

    def test_to_dict_example_structure(self, simple_vocab_epub: Path) -> None:
        """Should have correct structure for examples."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()

        for pos_dict in vocab_dict["entries"].values():
            for entry_dict in pos_dict.values():
                for example in entry_dict["examples"]:
                    assert "sentence" in example
                    assert "location" in example
                    assert "chapter_index" in example["location"]
                    assert "chapter_title" in example["location"]
                    assert "sentence_index" in example["location"]


class TestEmptyVocabulary:
    """Tests for edge cases with empty or minimal input."""

    def test_empty_epub_produces_empty_vocabulary(self, simple_vocab_epub: Path) -> None:
        """Should handle epub with no extractable text."""
        with mock_vocab_pipeline([], sentences=[], chapter_title="Empty"):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        assert isinstance(vocab, Vocabulary)
        assert len(vocab.entries) == 0

    def test_to_dict_on_empty_vocabulary(self, simple_vocab_epub: Path) -> None:
        """Should produce valid dict for empty vocabulary."""
        with mock_vocab_pipeline([], sentences=[], chapter_title="Empty"):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        vocab_dict = vocab.to_dict()
        assert vocab_dict["language"] == "fr"
        assert vocab_dict["entries"] == {}
        assert json.dumps(vocab_dict)  # Should be serializable


class TestMaxExamplesValidation:
    """Tests for max_examples parameter validation."""

    def test_max_examples_zero_is_valid(self, simple_vocab_epub: Path) -> None:
        """Should accept max_examples=0 (no examples collected)."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr", max_examples=0)

        # Should have no examples for any entry
        for pos_dict in vocab.entries.values():
            for entry in pos_dict.values():
                assert len(entry.examples) == 0

    def test_max_examples_negative_raises_value_error(self, simple_vocab_epub: Path) -> None:
        """Should raise ValueError when max_examples is negative."""
        with pytest.raises(ValueError, match="max_examples must be >= 0"):
            build_vocabulary(simple_vocab_epub, "fr", max_examples=-1)

    def test_max_examples_one_is_valid(self, simple_vocab_epub: Path) -> None:
        """Should accept max_examples=1 as valid."""
        sentences = ["Le chat dort.", "Le chat mange."]
        call_count = [0]

        def mock_extract_tokens(text: str, loc: SentenceLocation, lang: str) -> list[Token]:
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(sentences):
                return [create_mock_token("chat", "NOUN", "chat", sentences[idx], loc)]
            return []

        with mock_vocab_pipeline(mock_extract_tokens, sentences=sentences):
            vocab = build_vocabulary(simple_vocab_epub, "fr", max_examples=1)

        # Should have at most 1 example per entry
        for pos_dict in vocab.entries.values():
            for entry in pos_dict.values():
                assert len(entry.examples) <= 1


class TestDuplicateTokenInSentence:
    """Tests for handling duplicate tokens within the same sentence."""

    def test_duplicate_word_counts_frequency_correctly(self, simple_vocab_epub: Path) -> None:
        """Should count each occurrence for frequency."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        sentence = "Le chat voit un autre chat."
        tokens = [
            create_mock_token("chat", "NOUN", "chat", sentence, location),
            create_mock_token("chat", "NOUN", "chat", sentence, location),
        ]

        with mock_vocab_pipeline(tokens, sentences=[sentence]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        # "chat" appears twice, frequency should be 2
        assert vocab.entries["chat"]["NOUN"].frequency == 2

    def test_duplicate_word_has_single_example(self, simple_vocab_epub: Path) -> None:
        """Should only include one example for duplicate word in same sentence."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        sentence = "Le chat voit un autre chat."
        tokens = [
            create_mock_token("chat", "NOUN", "chat", sentence, location),
            create_mock_token("chat", "NOUN", "chat", sentence, location),
        ]

        with mock_vocab_pipeline(tokens, sentences=[sentence]):
            vocab = build_vocabulary(simple_vocab_epub, "fr", max_examples=10)

        # Despite "chat" appearing twice, should only have 1 example (same sentence)
        assert len(vocab.entries["chat"]["NOUN"].examples) == 1


class TestVocabularyIteration:
    """Tests for Vocabulary iteration support."""

    def test_iterating_yields_all_lemma_entries(self) -> None:
        """Iterating over Vocabulary should yield all LemmaEntry objects."""
        entries = {
            "chat": {
                "NOUN": LemmaEntry(
                    lemma="chat", pos="NOUN", frequency=5, forms={"chat": 5}, examples=[]
                ),
            },
            "run": {
                "VERB": LemmaEntry(
                    lemma="run", pos="VERB", frequency=3, forms={"run": 2, "runs": 1}, examples=[]
                ),
                "NOUN": LemmaEntry(
                    lemma="run", pos="NOUN", frequency=1, forms={"run": 1}, examples=[]
                ),
            },
        }
        vocab = Vocabulary(entries=entries, language="en")

        result = list(vocab)

        assert len(result) == 3
        assert all(isinstance(entry, LemmaEntry) for entry in result)
        # Check all entries are present (order not guaranteed)
        lemma_pos_pairs = {(e.lemma, e.pos) for e in result}
        assert lemma_pos_pairs == {("chat", "NOUN"), ("run", "VERB"), ("run", "NOUN")}

    def test_iterating_empty_vocabulary(self) -> None:
        """Iterating over empty Vocabulary should yield nothing."""
        vocab = Vocabulary(entries={}, language="en")

        result = list(vocab)

        assert result == []


class TestSkipEmptyPos:
    """Tests for skipping tokens with empty POS."""

    def test_skip_pos_constant_contains_empty_string(self) -> None:
        """SKIP_POS should contain empty string."""
        assert "" in SKIP_POS

    def test_tokens_with_empty_pos_are_skipped(self, simple_vocab_epub: Path) -> None:
        """Tokens with empty POS should not be included in vocabulary."""
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        tokens = [
            create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location),
            # Token with empty POS - should be skipped
            create_mock_token("unknown", "", "unknown", "Le chat dort.", location),
        ]

        with mock_vocab_pipeline(tokens, sentences=["Le chat dort."]):
            vocab = build_vocabulary(simple_vocab_epub, "fr")

        # "chat" should be in vocabulary
        assert "chat" in vocab.entries
        # "unknown" should NOT be in vocabulary (empty POS)
        assert "unknown" not in vocab.entries


class TestVocabularyBuilder:
    """Tests for VocabularyBuilder class."""

    def test_empty_builder_produces_empty_vocabulary(self) -> None:
        """Empty builder should produce a Vocabulary with no entries."""
        builder = VocabularyBuilder(language="fr")
        vocab = builder.build()

        assert isinstance(vocab, Vocabulary)
        assert vocab.language == "fr"
        assert len(vocab.entries) == 0

    def test_add_single_token(self) -> None:
        """Adding a single token should produce correct vocabulary entry."""
        builder = VocabularyBuilder(language="fr")
        location = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        token = create_mock_token("chat", "NOUN", "chat", "Le chat dort.", location)

        builder.add(token)
        vocab = builder.build()

        assert "chat" in vocab.entries
        assert "NOUN" in vocab.entries["chat"]
        entry = vocab.entries["chat"]["NOUN"]
        assert entry.frequency == 1
        assert entry.forms == {"chat": 1}
        assert len(entry.examples) == 1
        assert entry.examples[0].sentence == "Le chat dort."

    def test_add_tracks_frequency(self) -> None:
        """Multiple tokens with same lemma/POS should increment frequency."""
        builder = VocabularyBuilder(language="fr")
        loc0 = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        loc1 = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=1)

        builder.add(create_mock_token("chat", "NOUN", "chat", "Le chat dort.", loc0))
        builder.add(create_mock_token("chat", "NOUN", "chats", "Les chats mangent.", loc1))
        vocab = builder.build()

        assert vocab.entries["chat"]["NOUN"].frequency == 2

    def test_add_tracks_forms(self) -> None:
        """Different original forms of same lemma should be tracked."""
        builder = VocabularyBuilder(language="fr")
        loc0 = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        loc1 = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=1)

        builder.add(create_mock_token("chat", "NOUN", "chat", "Le chat dort.", loc0))
        builder.add(create_mock_token("chat", "NOUN", "chats", "Les chats mangent.", loc1))
        vocab = builder.build()

        forms = vocab.entries["chat"]["NOUN"].forms
        assert forms == {"chat": 1, "chats": 1}

    def test_max_examples_respected(self) -> None:
        """Should not collect more than max_examples examples."""
        builder = VocabularyBuilder(language="fr", max_examples=2)

        for i in range(5):
            loc = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=i)
            builder.add(create_mock_token("chat", "NOUN", "chat", f"Sentence {i}.", loc))
        vocab = builder.build()

        assert len(vocab.entries["chat"]["NOUN"].examples) == 2

    def test_max_examples_zero(self) -> None:
        """max_examples=0 should collect no examples."""
        builder = VocabularyBuilder(language="fr", max_examples=0)
        loc = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        builder.add(create_mock_token("chat", "NOUN", "chat", "Le chat dort.", loc))
        vocab = builder.build()

        assert len(vocab.entries["chat"]["NOUN"].examples) == 0

    def test_max_examples_negative_raises(self) -> None:
        """Negative max_examples should raise ValueError."""
        with pytest.raises(ValueError, match="max_examples must be >= 0"):
            VocabularyBuilder(language="fr", max_examples=-1)

    def test_duplicate_sentence_not_added(self) -> None:
        """Same sentence appearing twice should only be added once as example."""
        builder = VocabularyBuilder(language="fr", max_examples=10)
        loc = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)

        builder.add(create_mock_token("chat", "NOUN", "chat", "Le chat dort.", loc))
        builder.add(create_mock_token("chat", "NOUN", "chat", "Le chat dort.", loc))
        vocab = builder.build()

        assert len(vocab.entries["chat"]["NOUN"].examples) == 1

    def test_skip_empty_pos(self) -> None:
        """Tokens with empty POS should be skipped."""
        builder = VocabularyBuilder(language="fr")
        loc = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        builder.add(create_mock_token("unknown", "", "unknown", "Text.", loc))
        vocab = builder.build()

        assert len(vocab.entries) == 0

    def test_same_lemma_different_pos(self) -> None:
        """Same lemma with different POS should create separate entries."""
        builder = VocabularyBuilder(language="fr")
        loc0 = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        loc1 = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=1)

        builder.add(create_mock_token("run", "NOUN", "run", "A good run.", loc0))
        builder.add(create_mock_token("run", "VERB", "run", "I run fast.", loc1))
        vocab = builder.build()

        assert "NOUN" in vocab.entries["run"]
        assert "VERB" in vocab.entries["run"]
        assert vocab.entries["run"]["NOUN"].frequency == 1
        assert vocab.entries["run"]["VERB"].frequency == 1

    def test_example_location_preserved(self) -> None:
        """Example locations should match the token locations."""
        builder = VocabularyBuilder(language="fr")
        loc = SentenceLocation(chapter_index=2, chapter_title="Ch 3", sentence_index=5)
        builder.add(create_mock_token("chat", "NOUN", "chat", "Le chat dort.", loc))
        vocab = builder.build()

        example = vocab.entries["chat"]["NOUN"].examples[0]
        assert example.location.chapter_index == 2
        assert example.location.chapter_title == "Ch 3"
        assert example.location.sentence_index == 5

    def test_add_after_build_raises(self) -> None:
        """Calling add() after build() should raise RuntimeError."""
        builder = VocabularyBuilder(language="fr")
        loc = SentenceLocation(chapter_index=0, chapter_title="Test", sentence_index=0)
        builder.add(create_mock_token("chat", "NOUN", "chat", "Le chat dort.", loc))
        builder.build()

        with pytest.raises(RuntimeError, match="Cannot add tokens after build"):
            builder.add(create_mock_token("chien", "NOUN", "chien", "Le chien dort.", loc))

    def test_build_twice_raises(self) -> None:
        """Calling build() twice should raise RuntimeError."""
        builder = VocabularyBuilder(language="fr")
        builder.build()

        with pytest.raises(RuntimeError, match="build\\(\\) has already been called"):
            builder.build()
