"""Intermediate artifact writers for pipeline debugging."""

from __future__ import annotations

import json
from dataclasses import asdict
from io import TextIOWrapper
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from vocab.models import Chapter, LemmaEntry, Sentence, Token, Vocabulary
from vocab.pipeline import EnrichedLemma, SenseAssignment


class ArtifactWriter:
    """Context manager that writes numbered JSONL/JSON pipeline artifacts.

    Writes intermediate pipeline outputs to a directory for debugging
    and reproducibility. Files are named with numeric prefixes to reflect
    the pipeline stage order.

    Usage:
        with ArtifactWriter(Path("artifacts/")) as artifacts:
            artifacts.write_chapter(chapter)
            ...
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the artifact writer.

        Args:
            directory: Directory to write artifacts to. Created if it doesn't exist.
        """
        self._directory = directory
        self._handles: dict[str, TextIOWrapper] = {}

    def __enter__(self) -> Self:
        """Enter context: create directory."""
        self._directory.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context: close all open file handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _get_handle(self, filename: str) -> TextIOWrapper:
        """Get or open a file handle for the given filename."""
        if filename not in self._handles:
            path = self._directory / filename
            self._handles[filename] = open(path, "w", encoding="utf-8")  # noqa: SIM115
        return self._handles[filename]

    def _write_jsonl(self, filename: str, data: Any) -> None:
        """Write a single JSON line to the named file."""
        handle = self._get_handle(filename)
        handle.write(json.dumps(asdict(data), ensure_ascii=False) + "\n")

    def write_chapter(self, chapter: Chapter) -> None:
        """Write a chapter to 01-chapters.jsonl."""
        self._write_jsonl("01-chapters.jsonl", chapter)

    def write_sentence(self, chapter: Chapter, sentence: Sentence) -> None:
        """Write a sentence record to 02-sentences.jsonl."""
        record = {
            "chapter_index": chapter.index,
            "chapter_title": chapter.title,
            "sentence_index": sentence.index,
            "text": sentence.text,
        }
        handle = self._get_handle("02-sentences.jsonl")
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_token(self, token: Token) -> None:
        """Write a token to 03-tokens.jsonl."""
        self._write_jsonl("03-tokens.jsonl", token)

    def write_vocabulary(self, vocab: Vocabulary) -> None:
        """Write vocabulary to 04-vocabulary.json."""
        handle = self._get_handle("04-vocabulary.json")
        json.dump(vocab.to_dict(), handle, ensure_ascii=False, indent=2)

    def write_enriched(self, enriched: EnrichedLemma) -> None:
        """Write an enriched lemma to 05-enriched.jsonl."""
        self._write_jsonl("05-enriched.jsonl", enriched)

    def write_rejected(self, entry: LemmaEntry) -> None:
        """Write a rejected lemma to 05-rejected.jsonl."""
        self._write_jsonl("05-rejected.jsonl", entry)

    def write_assigned(self, assignment: SenseAssignment) -> None:
        """Write an unambiguous sense assignment to 06-assigned.jsonl."""
        self._write_jsonl("06-assigned.jsonl", assignment)

    def write_ambiguous(self, enriched: EnrichedLemma) -> None:
        """Write an ambiguous enriched lemma to 06-ambiguous.jsonl."""
        self._write_jsonl("06-ambiguous.jsonl", enriched)

    def write_disambiguated(self, assignment: SenseAssignment) -> None:
        """Write a disambiguated sense assignment to 07-disambiguated.jsonl."""
        self._write_jsonl("07-disambiguated.jsonl", assignment)

    def write_failed(self, enriched: EnrichedLemma) -> None:
        """Write a failed disambiguation to 07-failed.jsonl."""
        self._write_jsonl("07-failed.jsonl", enriched)


class NullArtifactWriter:
    """No-op artifact writer (null object pattern).

    Drop-in replacement for ArtifactWriter that writes nothing.
    Used when --artifacts is not provided.
    """

    def __enter__(self) -> Self:
        """Enter context (no-op)."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context (no-op)."""

    def write_chapter(self, chapter: Chapter) -> None:
        """No-op."""

    def write_sentence(self, chapter: Chapter, sentence: Sentence) -> None:
        """No-op."""

    def write_token(self, token: Token) -> None:
        """No-op."""

    def write_vocabulary(self, vocab: Vocabulary) -> None:
        """No-op."""

    def write_enriched(self, enriched: EnrichedLemma) -> None:
        """No-op."""

    def write_rejected(self, entry: LemmaEntry) -> None:
        """No-op."""

    def write_assigned(self, assignment: SenseAssignment) -> None:
        """No-op."""

    def write_ambiguous(self, enriched: EnrichedLemma) -> None:
        """No-op."""

    def write_disambiguated(self, assignment: SenseAssignment) -> None:
        """No-op."""

    def write_failed(self, enriched: EnrichedLemma) -> None:
        """No-op."""
