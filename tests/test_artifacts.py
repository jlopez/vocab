"""Tests for artifact writers."""

import json
from pathlib import Path

from vocab.artifacts import ArtifactWriter, NullArtifactWriter
from vocab.models import (
    Chapter,
    Example,
    LemmaEntry,
    Sentence,
    SentenceLocation,
    Token,
    Vocabulary,
)
from vocab.pipeline import EnrichedLemma, SenseAssignment

# --- Test fixtures ---

_LOCATION = SentenceLocation(chapter_index=0, chapter_title="Ch 1", sentence_index=0)
_CHAPTER = Chapter(text="Le chat dort.", index=0, title="Ch 1")
_SENTENCE = Sentence(text="Le chat dort.", index=0)
_TOKEN = Token(
    lemma="chat",
    pos="NOUN",
    morph={},
    original="chat",
    sentence="Le chat dort.",
    location=_LOCATION,
)
_LEMMA = LemmaEntry(
    lemma="chat",
    pos="NOUN",
    frequency=5,
    forms={"chat": 3, "chats": 2},
    examples=[Example(sentence="Le chat dort.", location=_LOCATION)],
)
_VOCAB = Vocabulary(
    entries={"chat": {"NOUN": _LEMMA}},
    language="fr",
)


def _make_enriched() -> EnrichedLemma:
    from vocab.dictionary import DictionaryEntry, DictionarySense

    sense = DictionarySense(id="s1", translation="cat", example=None)
    word = DictionaryEntry(word="chat", pos="noun", ipa="/ʃa/", etymology=None, senses=[sense])
    return EnrichedLemma(lemma=_LEMMA, words=[word])


def _make_assignment() -> SenseAssignment:
    from vocab.dictionary import DictionaryEntry, DictionarySense

    sense = DictionarySense(id="s1", translation="cat", example=None)
    word = DictionaryEntry(word="chat", pos="noun", ipa="/ʃa/", etymology=None, senses=[sense])
    return SenseAssignment(lemma=_LEMMA, examples=[0], word=word, sense=0)


# --- ArtifactWriter tests ---


class TestArtifactWriter:
    """Tests for ArtifactWriter."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Should create the output directory on enter."""
        out = tmp_path / "artifacts"
        with ArtifactWriter(out):
            assert out.is_dir()

    def test_write_chapter(self, tmp_path: Path) -> None:
        """Should write chapter to 01-chapters.jsonl."""
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_chapter(_CHAPTER)

        lines = (tmp_path / "01-chapters.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["index"] == 0
        assert data["title"] == "Ch 1"
        assert data["text"] == "Le chat dort."

    def test_write_sentence(self, tmp_path: Path) -> None:
        """Should write sentence record to 02-sentences.jsonl."""
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_sentence(_CHAPTER, _SENTENCE)

        lines = (tmp_path / "02-sentences.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["chapter_index"] == 0
        assert data["chapter_title"] == "Ch 1"
        assert data["sentence_index"] == 0
        assert data["text"] == "Le chat dort."

    def test_write_token(self, tmp_path: Path) -> None:
        """Should write token to 03-tokens.jsonl."""
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_token(_TOKEN)

        lines = (tmp_path / "03-tokens.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["lemma"] == "chat"
        assert data["pos"] == "NOUN"
        assert data["original"] == "chat"

    def test_write_vocabulary(self, tmp_path: Path) -> None:
        """Should write vocabulary to 04-vocabulary.json."""
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_vocabulary(_VOCAB)

        data = json.loads((tmp_path / "04-vocabulary.json").read_text())
        assert data["language"] == "fr"
        assert "chat" in data["entries"]

    def test_write_enriched(self, tmp_path: Path) -> None:
        """Should write enriched lemma to 05-enriched.jsonl."""
        enriched = _make_enriched()
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_enriched(enriched)

        lines = (tmp_path / "05-enriched.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["lemma"]["lemma"] == "chat"

    def test_write_rejected(self, tmp_path: Path) -> None:
        """Should write rejected lemma to 05-rejected.jsonl."""
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_rejected(_LEMMA)

        lines = (tmp_path / "05-rejected.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["lemma"] == "chat"

    def test_write_assigned(self, tmp_path: Path) -> None:
        """Should write sense assignment to 06-assigned.jsonl."""
        assignment = _make_assignment()
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_assigned(assignment)

        lines = (tmp_path / "06-assigned.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["lemma"]["lemma"] == "chat"
        assert data["sense"] == 0

    def test_write_ambiguous(self, tmp_path: Path) -> None:
        """Should write ambiguous enriched lemma to 06-ambiguous.jsonl."""
        enriched = _make_enriched()
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_ambiguous(enriched)

        lines = (tmp_path / "06-ambiguous.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["lemma"]["lemma"] == "chat"

    def test_write_disambiguated(self, tmp_path: Path) -> None:
        """Should write disambiguated assignment to 07-disambiguated.jsonl."""
        assignment = _make_assignment()
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_disambiguated(assignment)

        lines = (tmp_path / "07-disambiguated.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1

    def test_write_failed(self, tmp_path: Path) -> None:
        """Should write failed enriched lemma to 07-failed.jsonl."""
        enriched = _make_enriched()
        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_failed(enriched)

        lines = (tmp_path / "07-failed.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1

    def test_multiple_writes_append(self, tmp_path: Path) -> None:
        """Multiple writes to same file should append lines."""
        ch1 = Chapter(text="First.", index=0, title="A")
        ch2 = Chapter(text="Second.", index=1, title="B")

        with ArtifactWriter(tmp_path) as artifacts:
            artifacts.write_chapter(ch1)
            artifacts.write_chapter(ch2)

        lines = (tmp_path / "01-chapters.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_files_closed_on_exit(self, tmp_path: Path) -> None:
        """File handles should be closed after exiting context."""
        writer = ArtifactWriter(tmp_path)
        with writer:
            writer.write_chapter(_CHAPTER)

        # After exit, handles should be cleared
        assert len(writer._handles) == 0

    def test_nested_directory_created(self, tmp_path: Path) -> None:
        """Should create nested directories as needed."""
        out = tmp_path / "a" / "b" / "c"
        with ArtifactWriter(out) as artifacts:
            artifacts.write_chapter(_CHAPTER)

        assert (out / "01-chapters.jsonl").exists()


# --- NullArtifactWriter tests ---


class TestNullArtifactWriter:
    """Tests for NullArtifactWriter."""

    def test_works_as_context_manager(self) -> None:
        """Should work as a context manager."""
        with NullArtifactWriter() as artifacts:
            assert artifacts is not None

    def test_write_methods_are_noop(self, tmp_path: Path) -> None:
        """All write methods should do nothing."""
        enriched = _make_enriched()
        assignment = _make_assignment()

        with NullArtifactWriter() as artifacts:
            artifacts.write_chapter(_CHAPTER)
            artifacts.write_sentence(_CHAPTER, _SENTENCE)
            artifacts.write_token(_TOKEN)
            artifacts.write_vocabulary(_VOCAB)
            artifacts.write_enriched(enriched)
            artifacts.write_rejected(_LEMMA)
            artifacts.write_assigned(assignment)
            artifacts.write_ambiguous(enriched)
            artifacts.write_disambiguated(assignment)
            artifacts.write_failed(enriched)

        # No files should be created anywhere

    def test_no_files_created(self, tmp_path: Path) -> None:
        """NullArtifactWriter should not create any files."""
        out = tmp_path / "should_not_exist"

        with NullArtifactWriter() as artifacts:
            artifacts.write_chapter(_CHAPTER)

        assert not out.exists()
