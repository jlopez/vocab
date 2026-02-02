"""Anki deck generation from sense assignments."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from types import TracebackType

import genanki

from vocab.pipeline import SenseAssignment

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
<div class="front">{{Translation}}</div>
"""

BACK_TEMPLATE = """\
<div class="back">
  <div class="word">{{Word}}</div>
  {{#IPA}}<div class="ipa">{{IPA}}</div>{{/IPA}}
  {{#Examples}}<div class="examples">{{Examples}}</div>{{/Examples}}
  {{#Forms}}<div class="forms">Forms: {{Forms}}</div>{{/Forms}}
</div>
"""


# Namespace UUID for generating deterministic IDs
_VOCAB_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


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


def _format_examples(assignment: SenseAssignment) -> str:
    """Format example sentences with guillemets and highlighted word.

    Args:
        assignment: SenseAssignment containing examples.

    Returns:
        HTML-formatted examples string.
    """
    examples_html: list[str] = []
    word = assignment.lemma.lemma

    for idx in assignment.examples:
        sentence = assignment.lemma.examples[idx].sentence
        # Highlight the word in the sentence (case-insensitive)
        highlighted = _highlight_word(sentence, word)
        examples_html.append(f'<div class="example">« {highlighted} »</div>')

    return "\n".join(examples_html)


def _highlight_word(sentence: str, word: str) -> str:
    """Highlight occurrences of a word in a sentence.

    Args:
        sentence: The sentence text.
        word: The word to highlight.

    Returns:
        Sentence with word wrapped in <b> tags.
    """
    return re.sub(
        re.escape(word),
        lambda m: f"<b>{m.group()}</b>",
        sentence,
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
    """Context manager for building an Anki deck.

    Usage:
        with AnkiDeckBuilder(path, deck_name, language) as deck:
            deck.add(sense_assignment)
        # Deck is written on context exit
    """

    def __init__(
        self,
        path: Path,
        deck_name: str,
        source_language: str,
    ) -> None:
        """Initialize the deck builder.

        Args:
            path: Output path for the .apkg file.
            deck_name: Name for the Anki deck.
            source_language: Source language code (e.g., "fr").
        """
        self._path = path
        self._deck_name = deck_name
        self._source_language = source_language

        # Create model and deck
        self._model = genanki.Model(
            _generate_model_id(deck_name, source_language),
            f"Vocab {source_language}",
            fields=[
                {"name": "Translation"},
                {"name": "Word"},
                {"name": "IPA"},
                {"name": "Examples"},
                {"name": "Forms"},
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

    def add(self, entry: SenseAssignment) -> None:
        """Add a card for this sense assignment.

        Creates a card with:
        - Front: English translation (from sense)
        - Back: Word, IPA, examples, forms

        Args:
            entry: SenseAssignment to create a card from.
        """
        sense = entry.word.senses[entry.sense]

        # Build field values
        translation = sense.translation
        word = entry.word.word
        ipa = entry.word.ipa or ""
        examples = _format_examples(entry)
        forms = _format_forms(entry)

        note = genanki.Note(
            model=self._model,
            fields=[translation, word, ipa, examples, forms],
        )
        self._deck.add_note(note)
        self._cards_added += 1

    @property
    def cards_added(self) -> int:
        """Return the number of cards added to the deck."""
        return self._cards_added

    def __enter__(self) -> AnkiDeckBuilder:
        """Enter context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Write the .apkg file and clean up."""
        if exc_type is None:
            # Only write if no exception occurred
            package = genanki.Package(self._deck)
            package.write_to_file(str(self._path))
