"""Tests for the CLI."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from vocab.cli import _resolve_deck_name, _validate_language, app
from vocab.dictionary import DictionaryEntry, DictionarySense
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

runner = CliRunner()

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
    sense = DictionarySense(id="s1", translation="cat", example=None)
    word = DictionaryEntry(word="chat", pos="noun", ipa="/ʃa/", etymology=None, senses=[sense])
    return EnrichedLemma(lemma=_LEMMA, words=[word])


def _make_assignment() -> SenseAssignment:
    sense = DictionarySense(id="s1", translation="cat", example=None)
    word = DictionaryEntry(word="chat", pos="noun", ipa="/ʃa/", etymology=None, senses=[sense])
    return SenseAssignment(lemma=_LEMMA, examples=[0], word=word, sense=0)


# --- Validation tests ---


class TestValidateLanguage:
    """Tests for _validate_language."""

    def test_valid_language(self) -> None:
        assert _validate_language("fr") == "fr"

    def test_invalid_language(self) -> None:
        import pytest
        import typer

        with pytest.raises(typer.BadParameter, match="Unsupported language"):
            _validate_language("xx")


class TestResolveDeckName:
    """Tests for _resolve_deck_name."""

    def test_explicit_deck_name(self) -> None:
        assert _resolve_deck_name("My Deck", "fr") == "My Deck"

    def test_default_french(self) -> None:
        assert _resolve_deck_name(None, "fr") == "French Vocabulary"

    def test_default_german(self) -> None:
        assert _resolve_deck_name(None, "de") == "German Vocabulary"

    def test_unknown_language_uses_code(self) -> None:
        assert _resolve_deck_name(None, "xx") == "XX Vocabulary"


# --- CLI invocation tests ---


def _enter_pipeline_mocks(
    stack: ExitStack,
    *,
    enriched_return: EnrichedLemma | None = None,
    assignment_return: SenseAssignment | None = None,
    needs_disambig: bool = False,
    tokens: list[Token] | None = None,
    no_env: bool = True,
) -> None:
    """Enter all pipeline mocks into an ExitStack."""
    if enriched_return is None:
        enriched_return = _make_enriched()
    if assignment_return is None:
        assignment_return = _make_assignment()

    stack.enter_context(patch("vocab.cli.load_dotenv"))
    stack.enter_context(patch("vocab.cli.extract_chapters", return_value=[_CHAPTER]))
    stack.enter_context(patch("vocab.cli.extract_sentences", return_value=[_SENTENCE]))
    stack.enter_context(
        patch("vocab.cli.extract_tokens", return_value=tokens if tokens else [_TOKEN])
    )
    stack.enter_context(patch("vocab.cli.Dictionary"))
    stack.enter_context(patch("vocab.cli.enrich_lemma", return_value=enriched_return))
    stack.enter_context(patch("vocab.cli.assign_single_sense", return_value=assignment_return))
    stack.enter_context(patch("vocab.cli.needs_disambiguation", return_value=needs_disambig))
    if no_env:
        stack.enter_context(patch.dict("os.environ", {}, clear=True))


def _mock_anki_builder() -> AsyncMock:
    """Create a mock AnkiDeckBuilder."""
    mock_builder = AsyncMock()
    mock_builder.cards_added = 1
    mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
    mock_builder.__aexit__ = AsyncMock(return_value=None)
    return mock_builder


class TestBuildCommand:
    """Tests for the build command."""

    def test_missing_epub_fails(self) -> None:
        result = runner.invoke(app, ["build", "nonexistent.epub", "-l", "fr"])
        assert result.exit_code != 0

    def test_help(self) -> None:
        result = runner.invoke(app, ["build", "--help"])
        assert result.exit_code == 0
        assert "--language" in result.output
        assert "--output" in result.output
        assert "--no-disambiguation" in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "Usage" in result.output

    def test_missing_api_key_with_disambiguation(self, tmp_path: Path) -> None:
        """Should fail early when API key is missing and disambiguation is on."""
        epub = tmp_path / "test.epub"
        epub.touch()

        with patch("vocab.cli.load_dotenv"), patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["build", str(epub), "-l", "fr"])

        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output

    def test_no_disambiguation_skips_api_check(self, tmp_path: Path) -> None:
        """--no-disambiguation should not require API key."""
        epub = tmp_path / "test.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack)
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-disambiguation", "--no-audio"],
            )

        assert result.exit_code == 0, result.output

    def test_default_output_path(self, tmp_path: Path) -> None:
        """Output should default to epub basename with .apkg extension."""
        epub = tmp_path / "my-book.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack)
            mock_cls = stack.enter_context(
                patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder)
            )
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-disambiguation", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["path"] == epub.with_suffix(".apkg")

    def test_custom_output_path(self, tmp_path: Path) -> None:
        """--output should override the default path."""
        epub = tmp_path / "book.epub"
        epub.touch()
        custom_output = tmp_path / "custom.apkg"

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack)
            mock_cls = stack.enter_context(
                patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder)
            )
            result = runner.invoke(
                app,
                [
                    "build",
                    str(epub),
                    "-l",
                    "fr",
                    "-o",
                    str(custom_output),
                    "--no-disambiguation",
                    "--no-audio",
                ],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["path"] == custom_output

    def test_artifacts_written(self, tmp_path: Path) -> None:
        """--artifacts should write intermediate files."""
        epub = tmp_path / "book.epub"
        epub.touch()
        artifacts_dir = tmp_path / "artifacts"

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack)
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                [
                    "build",
                    str(epub),
                    "-l",
                    "fr",
                    "--artifacts",
                    str(artifacts_dir),
                    "--no-disambiguation",
                    "--no-audio",
                ],
            )

        assert result.exit_code == 0, result.output
        assert artifacts_dir.is_dir()
        assert (artifacts_dir / "01-chapters.jsonl").exists()
        assert (artifacts_dir / "02-sentences.jsonl").exists()
        assert (artifacts_dir / "03-tokens.jsonl").exists()
        assert (artifacts_dir / "04-vocabulary.json").exists()
        assert (artifacts_dir / "05-enriched.jsonl").exists()
        assert (artifacts_dir / "06-assigned.jsonl").exists()

    def test_deck_name_default(self, tmp_path: Path) -> None:
        """Default deck name should be '{Language} Vocabulary'."""
        epub = tmp_path / "book.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack)
            mock_cls = stack.enter_context(
                patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder)
            )
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-disambiguation", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["deck_name"] == "French Vocabulary"

    def test_no_audio_passes_skip_audio_flag(self, tmp_path: Path) -> None:
        """--no-audio should pass skip_audio=True to AnkiDeckBuilder."""
        epub = tmp_path / "book.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack)
            mock_cls = stack.enter_context(
                patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder)
            )
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-disambiguation", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["skip_audio"] is True

    def test_pos_filtering(self, tmp_path: Path) -> None:
        """--pos should filter lemmas by POS tag."""
        epub = tmp_path / "book.epub"
        epub.touch()

        verb_token = Token(
            lemma="dormir",
            pos="VERB",
            morph={},
            original="dort",
            sentence="Le chat dort.",
            location=_LOCATION,
        )

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack, tokens=[_TOKEN, verb_token])
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                [
                    "build",
                    str(epub),
                    "-l",
                    "fr",
                    "--pos",
                    "NOUN",
                    "--no-disambiguation",
                    "--no-audio",
                ],
            )

        assert result.exit_code == 0, result.output

    def test_top_filtering(self, tmp_path: Path) -> None:
        """--top should keep only the N most frequent lemmas after POS filtering."""
        epub = tmp_path / "book.epub"
        epub.touch()

        loc = _LOCATION
        # "chat" appears 3 times, "chien" appears 1 time
        chat = Token(
            lemma="chat",
            pos="NOUN",
            morph={},
            original="chat",
            sentence="S1.",
            location=loc,
        )
        chien = Token(
            lemma="chien",
            pos="NOUN",
            morph={},
            original="chien",
            sentence="S4.",
            location=loc,
        )
        tokens = [chat, chat, chat, chien]

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack, tokens=tokens)
            mock_enrich = stack.enter_context(
                patch("vocab.cli.enrich_lemma", return_value=_make_enriched())
            )
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                [
                    "build",
                    str(epub),
                    "-l",
                    "fr",
                    "--top",
                    "1",
                    "--no-disambiguation",
                    "--no-audio",
                ],
            )

        assert result.exit_code == 0, result.output
        # enrich_lemma should only be called once (for "chat", the most frequent)
        assert mock_enrich.call_count == 1

    def test_enrich_returns_none_rejects_lemma(self, tmp_path: Path) -> None:
        """When enrich_lemma returns None, the lemma should be skipped."""
        epub = tmp_path / "book.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack)
            # Override: enrich_lemma returns None
            stack.enter_context(patch("vocab.cli.enrich_lemma", return_value=None))
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-disambiguation", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        assert "Enriched 0 lemmas" in result.output

    def test_disambiguation_success(self, tmp_path: Path) -> None:
        """Disambiguation should be invoked for ambiguous lemmas."""
        epub = tmp_path / "book.epub"
        epub.touch()

        assignment = _make_assignment()
        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack, needs_disambig=True, no_env=False)
            stack.enter_context(patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}))
            mock_disambig = stack.enter_context(
                patch(
                    "vocab.cli.disambiguate_senses",
                    new_callable=AsyncMock,
                    return_value=[assignment],
                )
            )
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        mock_disambig.assert_called_once()
        assert "1 succeeded" in result.output

    def test_disambiguation_empty_result(self, tmp_path: Path) -> None:
        """Empty disambiguation result should count as failed."""
        epub = tmp_path / "book.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack, needs_disambig=True, no_env=False)
            stack.enter_context(patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}))
            stack.enter_context(
                patch("vocab.cli.disambiguate_senses", new_callable=AsyncMock, return_value=[])
            )
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        assert "1 failed" in result.output

    def test_disambiguation_exception(self, tmp_path: Path) -> None:
        """Disambiguation exception should be caught and counted as failed."""
        epub = tmp_path / "book.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack, needs_disambig=True, no_env=False)
            stack.enter_context(patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}))
            stack.enter_context(
                patch(
                    "vocab.cli.disambiguate_senses",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("LLM error"),
                )
            )
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        assert "1 failed" in result.output

    def test_ambiguous_skipped_with_no_disambiguation(self, tmp_path: Path) -> None:
        """--no-disambiguation should skip ambiguous lemmas with a message."""
        epub = tmp_path / "book.epub"
        epub.touch()

        mock_builder = _mock_anki_builder()
        with ExitStack() as stack:
            _enter_pipeline_mocks(stack, needs_disambig=True)
            stack.enter_context(patch("vocab.cli.AnkiDeckBuilder", return_value=mock_builder))
            result = runner.invoke(
                app,
                ["build", str(epub), "-l", "fr", "--no-disambiguation", "--no-audio"],
            )

        assert result.exit_code == 0, result.output
        assert "Skipping disambiguation" in result.output
