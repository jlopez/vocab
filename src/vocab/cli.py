"""CLI for vocab — build Anki decks from ePub files."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from vocab.anki import AnkiDeckBuilder
from vocab.artifacts import ArtifactWriter, NullArtifactWriter
from vocab.dictionary import KAIKKI_URLS, LANGUAGE_NAMES, Dictionary
from vocab.epub import extract_chapters
from vocab.models import LemmaEntry, SentenceLocation
from vocab.pipeline import (
    assign_single_sense,
    disambiguate_senses,
    enrich_lemma,
    needs_disambiguation,
)
from vocab.sentences import extract_sentences
from vocab.tokens import extract_tokens
from vocab.vocabulary import VocabularyBuilder

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="vocab",
    no_args_is_help=True,
)


@app.callback()
def _callback() -> None:
    """Build Anki decks from ePub files."""


# Default POS tags to include in the deck
DEFAULT_POS = "NOUN,VERB,ADJ,ADV"


def _validate_language(language: str) -> str:
    """Validate language code is supported by both dictionary and LLM pipeline."""
    if language not in KAIKKI_URLS:
        supported = ", ".join(sorted(KAIKKI_URLS.keys()))
        raise typer.BadParameter(f"Unsupported language: {language}. Supported: {supported}")
    return language


def _resolve_deck_name(deck_name: str | None, language: str) -> str:
    """Resolve deck name, defaulting to '{Language} Vocabulary'."""
    if deck_name:
        return deck_name
    language_name = LANGUAGE_NAMES.get(language, language.upper())
    return f"{language_name} Vocabulary"


@app.command()
def build(
    epub: Annotated[
        Path,
        typer.Argument(help="Path to the input ePub file.", exists=True, readable=True),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="2-letter language code (e.g., fr, de, es)."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output .apkg file path."),
    ] = None,
    deck_name: Annotated[
        str | None,
        typer.Option("--deck-name", "-d", help="Anki deck name."),
    ] = None,
    artifacts: Annotated[
        Path | None,
        typer.Option("--artifacts", help="Directory for intermediate pipeline artifacts."),
    ] = None,
    top: Annotated[
        int | None,
        typer.Option("--top", "-t", help="Limit to N most frequent lemmas.", min=1),
    ] = None,
    max_examples: Annotated[
        int,
        typer.Option("--max-examples", "-n", help="Max example sentences per lemma.", min=0),
    ] = 3,
    pos: Annotated[
        str,
        typer.Option("--pos", help="Comma-separated POS tags to include."),
    ] = DEFAULT_POS,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model for disambiguation."),
    ] = "claude-haiku",
    no_audio: Annotated[
        bool,
        typer.Option("--no-audio", help="Skip audio downloads."),
    ] = False,
    no_disambiguation: Annotated[
        bool,
        typer.Option("--no-disambiguation", help="Skip LLM disambiguation step."),
    ] = False,
) -> None:
    """Build an Anki deck from an ePub file.

    Runs the full pipeline: extract text, build vocabulary, enrich with
    dictionary definitions, disambiguate senses, and generate flashcards.
    """
    # Load .env for ANTHROPIC_API_KEY
    load_dotenv()

    # Validate inputs
    language = _validate_language(language)
    output_path = output or epub.with_suffix(".apkg")
    resolved_deck_name = _resolve_deck_name(deck_name, language)
    pos_tags = {p.strip().upper() for p in pos.split(",")}

    # Check API key if disambiguation is enabled
    if not no_disambiguation:
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            typer.echo(
                "Error: ANTHROPIC_API_KEY not set. "
                "Set it in .env or environment, or use --no-disambiguation.",
                err=True,
            )
            raise typer.Exit(1)

    asyncio.run(
        _build_pipeline(
            epub_path=epub,
            language=language,
            output_path=output_path,
            deck_name=resolved_deck_name,
            artifacts_dir=artifacts,
            top_n=top,
            max_examples=max_examples,
            pos_tags=pos_tags,
            model=model,
            no_audio=no_audio,
            no_disambiguation=no_disambiguation,
        )
    )


async def _build_pipeline(
    *,
    epub_path: Path,
    language: str,
    output_path: Path,
    deck_name: str,
    artifacts_dir: Path | None,
    top_n: int | None,
    max_examples: int,
    pos_tags: set[str],
    model: str,
    no_audio: bool,
    no_disambiguation: bool,
) -> None:
    """Run the full build pipeline."""
    from vocab.pipeline import EnrichedLemma, SenseAssignment

    # Setup artifact writer
    artifact_writer = ArtifactWriter(artifacts_dir) if artifacts_dir else NullArtifactWriter()

    with artifact_writer:
        # --- Phase 1-4: Build vocabulary ---
        typer.echo(f"Building vocabulary from {epub_path.name}...")
        builder = VocabularyBuilder(language=language, max_examples=max_examples)

        for chapter in extract_chapters(epub_path):
            artifact_writer.write_chapter(chapter)
            for sentence in extract_sentences(chapter.text, language):
                artifact_writer.write_sentence(chapter, sentence)
                location = SentenceLocation(
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    sentence_index=sentence.index,
                )
                for token in extract_tokens(sentence.text, location, language):
                    artifact_writer.write_token(token)
                    builder.add(token)

        vocab = builder.build()
        artifact_writer.write_vocabulary(vocab)
        typer.echo(f"Found {len(vocab.entries)} unique lemmas.")

        # --- Phase 5: Enrich with dictionary ---
        typer.echo("Enriching with dictionary definitions...")
        dictionary = Dictionary(language)

        enriched_lemmas: list[EnrichedLemma] = []

        # Identify which entries to keep (top-N by frequency if requested)
        if top_n is not None:
            pos_filtered = [e for e in vocab if e.pos in pos_tags]
            pos_filtered.sort(key=lambda e: e.frequency, reverse=True)
            keep = {(e.lemma, e.pos) for e in pos_filtered[:top_n]}
        else:
            keep = None

        # Iterate in insertion order (follows book narrative)
        candidates: list[LemmaEntry] = []
        for lemma_entry in vocab:
            if lemma_entry.pos not in pos_tags:
                artifact_writer.write_rejected(lemma_entry)
                continue
            if keep is not None and (lemma_entry.lemma, lemma_entry.pos) not in keep:
                continue
            candidates.append(lemma_entry)

        for lemma_entry in candidates:
            enriched = enrich_lemma(lemma_entry, dictionary)
            if enriched:
                enriched_lemmas.append(enriched)
                artifact_writer.write_enriched(enriched)
            else:
                artifact_writer.write_rejected(lemma_entry)

        typer.echo(f"Enriched {len(enriched_lemmas)} lemmas.")

        # --- Phase 6: Triage ---
        assignments: list[SenseAssignment] = []
        ambiguous: list[EnrichedLemma] = []

        for enriched in enriched_lemmas:
            if not needs_disambiguation(enriched):
                assignment = assign_single_sense(enriched)
                assignments.append(assignment)
                artifact_writer.write_assigned(assignment)
            else:
                ambiguous.append(enriched)
                artifact_writer.write_ambiguous(enriched)

        typer.echo(f"Triage: {len(assignments)} unambiguous, {len(ambiguous)} ambiguous.")

        # --- Phase 7: Disambiguate ---
        if not no_disambiguation and ambiguous:
            typer.echo(f"Disambiguating {len(ambiguous)} lemmas with {model}...")
            disambiguated_count = 0
            failed_count = 0

            for enriched in ambiguous:
                try:
                    results = await disambiguate_senses(enriched, language=language, model=model)
                    if results:
                        for assignment in results:
                            assignments.append(assignment)
                            artifact_writer.write_disambiguated(assignment)
                        disambiguated_count += 1
                    else:
                        artifact_writer.write_failed(enriched)
                        failed_count += 1
                except Exception:
                    logger.warning("Failed to disambiguate %s", enriched.lemma.lemma, exc_info=True)
                    artifact_writer.write_failed(enriched)
                    failed_count += 1

            typer.echo(f"Disambiguation: {disambiguated_count} succeeded, {failed_count} failed.")
        elif ambiguous:
            typer.echo(f"Skipping disambiguation for {len(ambiguous)} ambiguous lemmas.")

        # --- Phase 8: Build Anki deck ---
        typer.echo(f"Building Anki deck ({len(assignments)} cards)...")

        async with AnkiDeckBuilder(
            path=output_path,
            deck_name=deck_name,
            source_language=language,
            skip_audio=no_audio,
        ) as deck:
            for assignment in assignments:
                await deck.add(assignment)

        typer.echo(f"Generated: {output_path} ({deck.cards_added} cards)")


def main() -> None:  # pragma: no cover
    """Entry point for the CLI."""
    # Configure logging — only show warnings by default
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    app()
