"""Anki deck generation from sense assignments."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import uuid
from pathlib import Path
from types import TracebackType

import genanki

from vocab.media_cache import fetch_media
from vocab.pipeline import SenseAssignment

logger = logging.getLogger(__name__)

# CSS styling for cards
CARD_CSS = """\
.card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  line-height: 1.5;
}
.front {
  font-size: 24px;
  text-align: center;
}
.front .example-translation {
  font-size: 16px;
  font-style: italic;
  color: #666;
  margin-top: 15px;
}
.back {
  text-align: center;
}
.word {
  font-size: 28px;
  font-weight: bold;
  text-align: center;
}
.ipa {
  font-family: "Doulos SIL", "Noto Sans", serif;
  color: #555;
  text-align: center;
  margin: 10px 0;
}
.word-display {
  font-size: 14px;
  color: #666;
  text-align: center;
  margin: 5px 0;
}
.etymology {
  font-size: 14px;
  color: #777;
  text-align: center;
  margin: 10px 0;
}
.examples {
  font-style: italic;
  margin: 15px 0;
  text-align: left;
}
.examples .example {
  margin: 8px 0;
}
.forms {
  font-size: 14px;
  color: #888;
  text-align: center;
}
"""

# Card template
FRONT_TEMPLATE = """\
<div class="front">
  <div>{{Translation}}</div>
  {{#ExampleTranslation}}
  <div class="example-translation">{{ExampleTranslation}}</div>
  {{/ExampleTranslation}}
</div>
"""

BACK_TEMPLATE = """\
<div class="back">
  <div class="word">{{Word}}</div>
  {{#IPA}}<div class="ipa">{{IPA}}</div>{{/IPA}}
  {{Audio}}
  {{#WordDisplay}}<div class="word-display">{{WordDisplay}}</div>{{/WordDisplay}}
  {{#Etymology}}<div class="etymology">{{Etymology}}</div>{{/Etymology}}
  {{#Examples}}<div class="examples">{{Examples}}</div>{{/Examples}}
  {{#Forms}}<div class="forms">Forms: {{Forms}}</div>{{/Forms}}
</div>
"""


# Namespace UUID for generating deterministic IDs (derived from project URL)
_VOCAB_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/jlopez/vocab")


def _generate_model_id(deck_name: str, language: str) -> int:
    """Generate a deterministic model ID based on deck properties.

    Uses UUIDv5 to ensure the same deck name and language always
    produce the same model ID, preventing duplicate models in Anki.
    """
    name = f"{deck_name}:{language}:model"
    uid = uuid.uuid5(_VOCAB_NAMESPACE, name)
    return int(uid) >> 96  # Use top 32 bits


def _generate_deck_id(deck_name: str, language: str) -> int:
    """Generate a deterministic deck ID based on deck properties.

    Uses UUIDv5 to ensure the same deck name and language always
    produce the same deck ID, preventing duplicate decks in Anki.
    """
    name = f"{deck_name}:{language}:deck"
    uid = uuid.uuid5(_VOCAB_NAMESPACE, name)
    return int(uid) >> 96  # Use top 32 bits


def _generate_note_guid(deck_name: str, language: str, sense_id: str) -> str:
    """Generate a stable GUID for a note based on deck context and sense ID.

    Uses UUIDv5 to ensure the same sense in the same deck always produces
    the same GUID. This allows Anki to recognize updated cards (e.g.,
    improved examples) as the same note, preserving review history.

    Args:
        deck_name: Name of the Anki deck.
        language: Source language code (e.g., "fr").
        sense_id: The Kaikki sense ID (e.g., "en-table-fr-noun-7IF1wlIK").

    Returns:
        A stable GUID string for use with genanki.
    """
    name = f"{deck_name}:{language}:sense:{sense_id}"
    uid = uuid.uuid5(_VOCAB_NAMESPACE, name)
    return str(uid)


def _format_examples(assignment: SenseAssignment) -> str:
    """Format example sentences with highlighted word.

    Includes the dictionary example (if available) followed by book examples.

    Args:
        assignment: SenseAssignment containing examples.

    Returns:
        HTML-formatted examples string.
    """
    sentences: list[str] = []
    word = assignment.lemma.lemma

    # Dictionary example first (if available)
    sense = assignment.word.senses[assignment.sense]
    if sense.example and sense.example.text:
        sentences.append(sense.example.text)

    # Book examples
    for idx in assignment.examples:
        sentences.append(assignment.lemma.examples[idx].sentence)

    # Format all uniformly
    examples_html: list[str] = []
    for sentence in sentences:
        highlighted = _highlight_word(sentence, word)
        examples_html.append(f'<div class="example">{highlighted}</div>')

    return "\n".join(examples_html)


def _format_example_translation(assignment: SenseAssignment) -> str:
    """Format the dictionary example translation for the front of the card.

    Args:
        assignment: SenseAssignment containing the sense.

    Returns:
        HTML-formatted example translation, or empty string if not available.
    """
    sense = assignment.word.senses[assignment.sense]
    if not sense.example or not sense.example.translation:
        return ""

    word = assignment.lemma.lemma
    highlighted = _highlight_word(sense.example.translation, word)
    return highlighted


def _highlight_word(sentence: str, word: str) -> str:
    """Highlight occurrences of a word in a sentence.

    HTML-escapes the sentence first, then applies highlighting.

    Args:
        sentence: The sentence text.
        word: The word to highlight.

    Returns:
        HTML-escaped sentence with word wrapped in <b> tags.
    """
    escaped = html.escape(sentence)
    escaped_word = html.escape(word)
    return re.sub(
        re.escape(escaped_word),
        lambda m: f"<b>{m.group()}</b>",
        escaped,
        flags=re.IGNORECASE,
    )


def _format_forms(assignment: SenseAssignment) -> str:
    """Format the forms used in examples.

    Args:
        assignment: SenseAssignment containing lemma with forms.

    Returns:
        Comma-separated list of forms.
    """
    return ", ".join(sorted(assignment.lemma.forms.keys()))


class AnkiDeckBuilder:
    """Async context manager for building an Anki deck with audio support.

    Usage:
        async with AnkiDeckBuilder(path, deck_name, language) as deck:
            await deck.add(sense_assignment)
        # Deck is written on context exit
    """

    def __init__(
        self,
        path: Path,
        deck_name: str,
        source_language: str,
        max_concurrent_downloads: int = 16,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the deck builder.

        Args:
            path: Output path for the .apkg file.
            deck_name: Name for the Anki deck.
            source_language: Source language code (e.g., "fr").
            max_concurrent_downloads: Max concurrent audio downloads (default 16).
            cache_dir: Cache directory for audio files. Defaults to ~/.cache/vocab/media/
        """
        self._path = path
        self._deck_name = deck_name
        self._source_language = source_language
        self._cache_dir = cache_dir
        self._download_semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self._media_files: list[str] = []

        # Create model and deck
        self._model = genanki.Model(
            _generate_model_id(deck_name, source_language),
            f"Vocab {source_language}",
            fields=[
                {"name": "Translation"},
                {"name": "ExampleTranslation"},
                {"name": "Word"},
                {"name": "IPA"},
                {"name": "WordDisplay"},
                {"name": "Etymology"},
                {"name": "Examples"},
                {"name": "Forms"},
                {"name": "Audio"},
            ],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": FRONT_TEMPLATE,
                    "afmt": BACK_TEMPLATE,
                },
            ],
            css=CARD_CSS,
        )

        self._deck = genanki.Deck(_generate_deck_id(deck_name, source_language), deck_name)
        self._cards_added = 0

    async def add(self, entry: SenseAssignment) -> None:
        """Add a card for this sense assignment.

        Downloads audio if available (with concurrency limiting).

        Creates a card with:
        - Front: English translation (from sense)
        - Back: Word, IPA, audio, examples, forms

        Args:
            entry: SenseAssignment to create a card from.
        """
        sense = entry.word.senses[entry.sense]

        # Build field values (escape plain text fields for HTML safety)
        translation = html.escape(sense.translation)
        example_translation = _format_example_translation(entry)
        word = html.escape(entry.word.word)
        ipa = html.escape(entry.word.ipa or "")
        word_display = html.escape(entry.word.word_display or "")
        etymology = html.escape(entry.word.etymology or "")
        examples = _format_examples(entry)
        forms = html.escape(_format_forms(entry))

        # Use sense ID for stable GUID so updates preserve review history
        guid = _generate_note_guid(self._deck_name, self._source_language, sense.id)

        # Download audio if available
        audio = ""
        if entry.word.audio_url:
            audio = await self._download_audio(entry.word.audio_url, guid)

        note = genanki.Note(
            model=self._model,
            fields=[
                translation,
                example_translation,
                word,
                ipa,
                word_display,
                etymology,
                examples,
                forms,
                audio,
            ],
            guid=guid,
        )
        self._deck.add_note(note)
        self._cards_added += 1

    async def _download_audio(self, url: str, guid: str) -> str:
        """Download audio file and return Anki sound reference.

        Args:
            url: URL to download audio from.
            guid: Card GUID to use as filename base.

        Returns:
            Anki sound reference "[sound:{filename}]" or empty string on failure.
        """
        filename = f"{guid}.mp3"
        try:
            async with self._download_semaphore:
                path = await fetch_media(url, filename, cache_dir=self._cache_dir)
            if path is None:
                logger.debug("Audio not found: %s", url)
                return ""
            self._media_files.append(str(path))
            return f"[sound:{filename}]"
        except Exception:
            logger.warning("Failed to download audio from %s", url, exc_info=True)
            return ""

    @property
    def cards_added(self) -> int:
        """Return the number of cards added to the deck."""
        return self._cards_added

    async def __aenter__(self) -> AnkiDeckBuilder:
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Write the .apkg file and clean up."""
        if exc_type is None:
            # Only write if no exception occurred
            package = genanki.Package(self._deck)
            package.media_files = self._media_files
            # Note: write_to_file is synchronous and may block the event loop
            # for large decks. This is a genanki limitation.
            package.write_to_file(str(self._path))
