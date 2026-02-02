# Anki Flashcard Generation Pipeline

## Overview

Build a pipeline to generate Anki flashcards from extracted vocabulary. The pipeline enriches vocabulary entries with Wiktionary data, disambiguates word senses using an LLM, and exports to Anki's `.apkg` format.

The pipeline processes vocabulary in three stages:
1. **Enrichment**: Match lemmas to dictionary entries by (word, POS)
2. **Disambiguation**: Assign each example sentence to a specific word sense
3. **Export**: Generate Anki cards from sense assignments

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  generate_enriched  │     │  disambiguate       │     │  AnkiDeckBuilder    │
│  _lemmas()          │────▶│  _senses()          │────▶│  .add()             │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
   Vocabulary + Dict           LLM (when needed)           genanki
   → EnrichedLemma             → SenseAssignment           → .apkg
```

### Stage 1: Enrichment

- **Input**: `Vocabulary`, `Dictionary`
- **Output**: `Iterator[EnrichedLemma]`
- **Responsibility**: Look up each lemma in the dictionary by (word, POS), yield only those with matches

### Stage 2: Disambiguation

- **Input**: `EnrichedLemma`
- **Output**: `list[SenseAssignment]`
- **Responsibility**: Assign each example to a specific (word, sense) pair

Two paths:
- **Trivial**: Single word with single sense → direct assignment, no LLM
- **LLM**: Multiple words or senses → use LLM to disambiguate

### Stage 3: Export

- **Input**: `SenseAssignment` (streamed via `.add()`)
- **Output**: `.apkg` file
- **Responsibility**: Build styled Anki cards, write deck on context exit

## Data Models

```python
# === Dictionary Models (dictionary.py) ===

@dataclass
class DictionaryExample:
    """Example sentence from Wiktionary."""
    text: str  # Example in source language (e.g., French)
    translation: str  # English translation

@dataclass
class DictionarySense:
    """A single sense of a dictionary entry."""
    id: str  # Wiktionary sense ID
    translation: str  # English gloss (from .glosses[0])
    example: DictionaryExample | None  # From .examples[0] if available

@dataclass
class DictionaryEntry:
    """A dictionary entry from kaikki (one per etymology)."""
    word: str  # The headword (from .word)
    pos: str  # Part of speech (kaikki format: "noun", "verb", etc.)
    ipa: str | None  # From first .sounds[].ipa
    etymology: str | None  # From .etymology_text
    senses: list[DictionarySense]  # From .senses[]


# === Pipeline Models (pipeline.py) ===

@dataclass
class EnrichedLemma:
    """A lemma enriched with dictionary data."""
    lemma: LemmaEntry  # From Vocabulary
    words: list[DictionaryEntry]  # Invariant: len >= 1

@dataclass
class SenseAssignment:
    """A sense assignment mapping examples to a specific dictionary sense."""
    lemma: LemmaEntry  # Reference to EnrichedLemma.lemma
    examples: list[int]  # Indices into lemma.examples[]
    word: DictionaryEntry  # Reference to one of EnrichedLemma.words
    sense: int  # Index into word.senses[]
```

## POS Mapping

spaCy uses Universal POS tags; kaikki uses lowercase names. Mapping (1-to-many):

```python
SPACY_TO_KAIKKI: dict[str, list[str]] = {
    "NOUN": ["noun"],
    "VERB": ["verb"],
    "ADJ": ["adj"],
    "ADV": ["adv"],
    "PROPN": ["name"],
    "INTJ": ["intj"],
    "ADP": ["prep", "prep_phrase", "postp"],
    "PRON": ["pron"],
    "DET": ["det", "article"],
    "CONJ": ["conj"],
    "CCONJ": ["conj"],
    "SCONJ": ["conj"],
    "NUM": ["num"],
    "PART": ["particle"],
    "PUNCT": ["punct"],
    "SYM": ["symbol"],
    "X": ["phrase", "proverb", "contraction", "character"],
}
```

## Card Format

**Front (question):**
```
forgery, fabrication
```

**Back (answer):**
```
le faux
/fo/

« Ce tableau est un *faux* vendu comme une œuvre originale. »

Forms: faux
```

## Dependencies

**New packages:**
- `genanki` - Anki deck generation
- `anthropic` - LLM API calls

**Existing packages (already added):**
- `httpx` - Dictionary download

---

## Phase 1: Dictionary Restructuring

### Goal

Restructure `Dictionary` to support POS-filtered lookups and return the new `DictionaryEntry` model with senses.

### Deliverables

- Restructured `DictionaryEntry`, new `DictionarySense`, `DictionaryExample` dataclasses
- `Dictionary.lookup()` returns `list[DictionaryEntry]` filtered by POS
- POS mapping constant `SPACY_TO_KAIKKI`
- Updated tests

### Implementation Details

```python
# src/vocab/dictionary.py

SPACY_TO_KAIKKI: dict[str, list[str]] = {
    "NOUN": ["noun"],
    "VERB": ["verb"],
    "ADJ": ["adj"],
    "ADV": ["adv"],
    "PROPN": ["name"],
    "INTJ": ["intj"],
    "ADP": ["prep", "prep_phrase", "postp"],
    "PRON": ["pron"],
    "DET": ["det", "article"],
    "CONJ": ["conj"],
    "CCONJ": ["conj"],
    "SCONJ": ["conj"],
    "NUM": ["num"],
    "PART": ["particle"],
    "PUNCT": ["punct"],
    "SYM": ["symbol"],
    "X": ["phrase", "proverb", "contraction", "character"],
}


@dataclass
class DictionaryExample:
    """Example sentence from Wiktionary."""
    text: str
    translation: str

    @classmethod
    def from_kaikki(cls, raw: dict[str, Any]) -> "DictionaryExample":
        """Parse a kaikki example object."""
        return cls(
            text=raw.get("text", ""),
            translation=raw.get("translation") or raw.get("english", ""),
        )


@dataclass
class DictionarySense:
    """A single sense of a dictionary entry."""
    id: str
    translation: str
    example: DictionaryExample | None

    @classmethod
    def from_kaikki(cls, raw: dict[str, Any]) -> "DictionarySense":
        """Parse a kaikki sense object."""
        examples = raw.get("examples", [])
        example = DictionaryExample.from_kaikki(examples[0]) if examples else None

        glosses = raw.get("glosses", [])
        translation = glosses[0] if glosses else ""

        return cls(
            id=raw.get("id", ""),
            translation=translation,
            example=example,
        )


@dataclass
class DictionaryEntry:
    """A dictionary entry from kaikki (one per etymology)."""
    word: str
    pos: str
    ipa: str | None
    etymology: str | None
    senses: list[DictionarySense]

    @classmethod
    def from_kaikki(cls, raw: dict[str, Any]) -> "DictionaryEntry":
        """Parse a kaikki dictionary entry."""
        return cls(
            word=raw.get("word", ""),
            pos=raw.get("pos", ""),
            ipa=cls._extract_ipa(raw),
            etymology=raw.get("etymology_text"),
            senses=[DictionarySense.from_kaikki(s) for s in raw.get("senses", [])],
        )

    @staticmethod
    def _extract_ipa(raw: dict[str, Any]) -> str | None:
        """Extract first available IPA pronunciation."""
        for sound in raw.get("sounds", []):
            if ipa := sound.get("ipa"):
                return ipa
        return None


class Dictionary:
    def lookup(self, word: str, pos: list[str] | None = None) -> list[DictionaryEntry]:
        """Look up a word, optionally filtering by POS.

        Args:
            word: Word to look up.
            pos: List of kaikki POS tags to filter by (e.g., ["noun", "name"]).
                 If None, returns all entries for the word.

        Returns:
            List of DictionaryEntry objects, one per kaikki entry matching
            the word and POS filter. Empty list if no matches.
        """
        self._ensure_loaded()
        raw_entries = self._data.get(word, [])

        entries = [DictionaryEntry.from_kaikki(raw) for raw in raw_entries]

        if pos:
            entries = [e for e in entries if e.pos in pos]

        return entries
```

**Conversion pattern:**

Each dataclass has a `from_kaikki()` class method that encapsulates parsing logic:
- `DictionaryExample.from_kaikki()` - parses example, prefers `.translation` over `.english`
- `DictionarySense.from_kaikki()` - parses sense with id, first gloss, optional example
- `DictionaryEntry.from_kaikki()` - parses full entry, delegates to nested `from_kaikki()` calls

This pattern makes it easy to add fields later (e.g., gender): add field to dataclass, update `from_kaikki()`, add `_extract_*()` helper if complex.

**Testing the conversion:**

```python
def test_dictionary_entry_from_kaikki():
    raw = {
        "word": "chien",
        "pos": "noun",
        "sounds": [{"ipa": "/ʃjɛ̃/"}],
        "etymology_text": "From Latin canis.",
        "senses": [{"id": "fr-noun-1", "glosses": ["dog"]}],
    }
    entry = DictionaryEntry.from_kaikki(raw)
    assert entry.word == "chien"
    assert entry.ipa == "/ʃjɛ̃/"
    assert entry.etymology == "From Latin canis."
    assert len(entry.senses) == 1
    assert entry.senses[0].translation == "dog"
```

### Files to Modify

- `src/vocab/dictionary.py` - restructure models and lookup
- `tests/test_dictionary.py` - update for new API

### Acceptance Criteria

- [x] `DictionaryEntry` has `word`, `pos`, `ipa`, `etymology`, `senses` fields
- [x] `DictionarySense` has `id`, `translation`, `example` fields
- [x] `DictionaryExample` has `text`, `translation` fields
- [x] Each dataclass has `from_kaikki()` class method for parsing
- [x] `DictionaryEntry._extract_ipa()` extracts first available IPA
- [x] `lookup()` returns `list[DictionaryEntry]`
- [x] `lookup(word, pos=["noun"])` filters by POS
- [x] `lookup(word)` (no POS) returns all entries for word
- [x] `SPACY_TO_KAIKKI` mapping is exported
- [x] Unit tests cover `from_kaikki()` parsing with realistic kaikki structures
- [x] `uv run ruff check .` passes
- [x] `uv run mypy .` passes
- [x] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Phase 2: Enrichment Stage

### Goal

Implement `generate_enriched_lemmas()` to produce `EnrichedLemma` objects from a `Vocabulary`.

### Deliverables

- `src/vocab/pipeline.py` with `EnrichedLemma` dataclass and `generate_enriched_lemmas()`
- Unit tests
- Harness script `data/phase-2.py`

### Implementation Details

```python
# src/vocab/pipeline.py

@dataclass
class EnrichedLemma:
    """A lemma enriched with dictionary data."""
    lemma: LemmaEntry
    words: list[DictionaryEntry]  # Invariant: len >= 1


def generate_enriched_lemmas(
    vocabulary: Vocabulary,
    dictionary: Dictionary,
) -> Iterator[EnrichedLemma]:
    """Generate enriched lemmas from vocabulary.

    For each LemmaEntry in the vocabulary, looks up matching dictionary
    entries by (word, POS). Only yields entries with at least one match.

    Args:
        vocabulary: Vocabulary to process.
        dictionary: Dictionary for lookups.

    Yields:
        EnrichedLemma for each lemma with dictionary matches.
    """
    for lemma_by_pos in vocabulary.entries.values():
        for lemma_entry in lemma_by_pos.values():
            kaikki_pos = SPACY_TO_KAIKKI.get(lemma_entry.pos, [])
            words = dictionary.lookup(lemma_entry.lemma, pos=kaikki_pos or None)
            if words:
                yield EnrichedLemma(lemma=lemma_entry, words=words)
```

### Files to Create

- `src/vocab/pipeline.py`
- `tests/test_pipeline.py`

### Files to Modify

- `src/vocab/__init__.py` (exports)

### Harness Script (data/phase-2.py)

```python
#!/usr/bin/env python3
"""Test enrichment stage."""
import json
from vocab import Vocabulary
from vocab.dictionary import Dictionary
from vocab.pipeline import generate_enriched_lemmas

with open("phase4-lg.json") as f:
    vocab = Vocabulary.from_dict(json.load(f))

dictionary = Dictionary("fr")

# Count statistics
total = 0
matched = 0
multi_word = 0
multi_sense = 0

for enriched in generate_enriched_lemmas(vocab, dictionary):
    total += 1
    matched += 1
    if len(enriched.words) > 1:
        multi_word += 1
    if any(len(w.senses) > 1 for w in enriched.words):
        multi_sense += 1

print(f"Vocabulary entries: {sum(len(v) for v in vocab.entries.values())}")
print(f"Enriched lemmas: {matched}")
print(f"Multiple dictionary entries: {multi_word}")
print(f"Multiple senses: {multi_sense}")

# Show sample
for i, enriched in enumerate(generate_enriched_lemmas(vocab, dictionary)):
    if i >= 5:
        break
    print(f"\n{enriched.lemma.lemma} ({enriched.lemma.pos}):")
    for w in enriched.words:
        print(f"  {w.word} [{w.pos}] - {len(w.senses)} senses")
        for s in w.senses[:2]:
            print(f"    - {s.translation[:50]}")
```

### Acceptance Criteria

- [x] `EnrichedLemma` dataclass with `lemma` and `words` fields
- [x] `generate_enriched_lemmas()` yields only lemmas with dictionary matches
- [x] POS mapping correctly converts spaCy → kaikki
- [x] Lemmas with no dictionary match are skipped (not yielded)
- [x] `uv run ruff check .` passes
- [x] `uv run mypy .` passes
- [x] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Phase 3: Disambiguation Stage

### Goal

Implement sense disambiguation with trivial path and LLM path.

### Deliverables

- `SenseAssignment` dataclass
- `needs_disambiguation()`, `assign_single_sense()`, `disambiguate_senses()`
- Unit tests (mocked LLM)
- Harness script `data/phase-3.py`

### Implementation Details

```python
# src/vocab/pipeline.py (additions)

@dataclass
class SenseAssignment:
    """A sense assignment mapping examples to a specific dictionary sense."""
    lemma: LemmaEntry
    examples: list[int]  # Indices into lemma.examples[]
    word: DictionaryEntry
    sense: int  # Index into word.senses[]


def needs_disambiguation(entry: EnrichedLemma) -> bool:
    """Return True if LLM disambiguation is needed.

    Disambiguation is needed when there are multiple words or
    any word has multiple senses.
    """
    if len(entry.words) > 1:
        return True
    return len(entry.words[0].senses) > 1


def assign_single_sense(entry: EnrichedLemma) -> SenseAssignment:
    """Assign all examples to the single available sense.

    Args:
        entry: EnrichedLemma with exactly one word and one sense.

    Returns:
        SenseAssignment with all example indices.

    Raises:
        AssertionError: If entry has multiple words or senses.
    """
    assert not needs_disambiguation(entry), "Use disambiguate_senses() for this entry"
    return SenseAssignment(
        lemma=entry.lemma,
        examples=list(range(len(entry.lemma.examples))),
        word=entry.words[0],
        sense=0,
    )


async def disambiguate_senses(
    entry: EnrichedLemma,
    *,
    model: str = "claude-haiku",
) -> list[SenseAssignment]:
    """Use LLM to assign examples to senses.

    This function's signature is provider-agnostic: domain types in, domain
    types out. The LLM provider (currently Anthropic) is an implementation
    detail; switching providers requires only changing this function's internals.

    Args:
        entry: EnrichedLemma with multiple words or senses.
        model: Model identifier ("claude-haiku" or "claude-sonnet").

    Returns:
        List of SenseAssignments, one per unique (word, sense) used.
        Examples that couldn't be assigned are logged and omitted.

    Raises:
        AssertionError: If entry has only one sense.
    """
    assert needs_disambiguation(entry), "Use assign_single_sense() for this entry"
    ...
```

**Response Schema (Pydantic):**

```python
class SentenceAssignment(BaseModel):
    """Assignment of a sentence to a sense."""
    sentence: int  # 1-indexed sentence number
    sense: int | None  # 1-indexed sense number, None if unknown

class DisambiguationResponse(BaseModel):
    """LLM response for sense disambiguation."""
    assignments: list[SentenceAssignment]
```

**LLM Prompt Template:**

```
You are helping associate sentences with word meanings.

Language: {language}
Word: {word}

Available senses:
{senses}

Sentences from the source text:
{sentences}

For each sentence, indicate which sense is being used.
If you cannot confidently determine the sense, use null for the sense.
```

Template variables:
- `{language}` - Full language name (e.g., "French")
- `{word}` - The lemma being disambiguated
- `{senses}` - Numbered list, one per line: `{i}. [word={word}, etymology={etymology}] {translation}`
- `{sentences}` - Numbered list, one per line: `{i}. {sentence_text}`

**Rendered Example:**

```
You are helping associate sentences with word meanings.

Language: French
Word: faux

Available senses:
1. [word=faux, etymology=From Latin falsus] forgery, fabrication
2. [word=faux, etymology=From Latin falx] scythe

Sentences from the source text:
1. Le paysan affûte sa *faux* avant de couper l'herbe au lever du soleil.
2. Ce tableau est un *faux* vendu comme une œuvre originale.

For each sentence, indicate which sense is being used.
If you cannot confidently determine the sense, use null for the sense.
```

**Model mapping:**
- `claude-haiku` → `claude-3-5-haiku-latest`
- `claude-sonnet` → `claude-sonnet-4-20250514`

### Files to Modify

- `src/vocab/pipeline.py` (add disambiguation functions)
- `tests/test_pipeline.py` (add disambiguation tests)
- `pyproject.toml` (add anthropic dependency)

### Harness Script (data/phase-3.py)

```python
#!/usr/bin/env python3
"""Test disambiguation stage."""
import asyncio
import json
from vocab import Vocabulary
from vocab.dictionary import Dictionary
from vocab.pipeline import (
    generate_enriched_lemmas,
    needs_disambiguation,
    assign_single_sense,
    disambiguate_senses,
)

PARALLELISM = 16


async def main():
    with open("phase4-lg.json") as f:
        vocab = Vocabulary.from_dict(json.load(f))

    dictionary = Dictionary("fr")

    trivial_count = 0
    llm_count = 0
    assignments: list[SenseAssignment] = []

    llm_batch: list[EnrichedLemma] = []

    for enriched in generate_enriched_lemmas(vocab, dictionary):
        if not needs_disambiguation(enriched):
            trivial_count += 1
            assignments.append(assign_single_sense(enriched))
        else:
            llm_count += 1
            llm_batch.append(enriched)

            if len(llm_batch) >= PARALLELISM:
                results = await asyncio.gather(
                    *[disambiguate_senses(e) for e in llm_batch]
                )
                for result in results:
                    assignments.extend(result)
                llm_batch = []

        # Limit for testing
        if trivial_count + llm_count >= 100:
            break

    # Flush remaining
    if llm_batch:
        results = await asyncio.gather(*[disambiguate_senses(e) for e in llm_batch])
        for result in results:
            assignments.extend(result)

    print(f"Trivial: {trivial_count}")
    print(f"LLM: {llm_count}")
    print(f"Total assignments: {len(assignments)}")

    # Show sample
    for a in assignments[:5]:
        sense = a.word.senses[a.sense]
        print(f"\n{a.lemma.lemma}: {sense.translation[:40]}")
        print(f"  Examples: {a.examples}")


asyncio.run(main())
```

### Acceptance Criteria

- [ ] `SenseAssignment` dataclass with `lemma`, `examples`, `word`, `sense` fields
- [ ] `needs_disambiguation()` returns True for multi-word or multi-sense entries
- [ ] `assign_single_sense()` works for trivial cases
- [ ] `assign_single_sense()` raises AssertionError for non-trivial cases
- [ ] `disambiguate_senses()` calls LLM and parses response
- [ ] `disambiguate_senses()` raises AssertionError for trivial cases
- [ ] Unassignable examples are logged and omitted
- [ ] Unit tests mock LLM responses
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy .` passes
- [ ] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Phase 4: Anki Export

### Goal

Implement `AnkiDeckBuilder` context manager to generate `.apkg` files.

### Deliverables

- `src/vocab/anki.py` with `AnkiDeckBuilder` class
- Card template with CSS styling
- Unit tests
- Harness script `data/phase-4.py`

### Implementation Details

```python
# src/vocab/anki.py

class AnkiDeckBuilder:
    """Context manager for building an Anki deck."""

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
        ...

    def add(self, entry: SenseAssignment) -> None:
        """Add a card for this sense assignment.

        Creates a card with:
        - Front: English translation (from sense)
        - Back: Word with article (if noun), IPA, examples, forms

        Args:
            entry: SenseAssignment to create a card from.
        """
        ...

    def __enter__(self) -> "AnkiDeckBuilder":
        """Enter context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Write the .apkg file and clean up."""
        ...
```

**Card template:**

Front:
```html
<div class="front">{{Translation}}</div>
```

Back:
```html
<div class="back">
  <div class="word">{{Word}}</div>
  <div class="ipa">{{IPA}}</div>
  <div class="examples">{{Examples}}</div>
  <div class="forms">{{Forms}}</div>
</div>
```

CSS:
```css
.front {
  font-size: 24px;
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
}
.examples .example {
  margin: 8px 0;
}
.forms {
  font-size: 14px;
  color: #888;
  text-align: center;
}
```

**Note on gender:** Punted for now. Can be added later by extracting gender from kaikki data and prepending article to nouns.

### Files to Create

- `src/vocab/anki.py`
- `tests/test_anki.py`

### Files to Modify

- `src/vocab/__init__.py` (exports)
- `pyproject.toml` (add genanki dependency)

### Harness Script (data/phase-4.py)

```python
#!/usr/bin/env python3
"""Test full pipeline with Anki export."""
import asyncio
import json
from pathlib import Path
from vocab import Vocabulary
from vocab.dictionary import Dictionary
from vocab.pipeline import (
    generate_enriched_lemmas,
    needs_disambiguation,
    assign_single_sense,
    disambiguate_senses,
)
from vocab.anki import AnkiDeckBuilder

PARALLELISM = 16


async def main():
    with open("phase4-lg.json") as f:
        vocab = Vocabulary.from_dict(json.load(f))

    dictionary = Dictionary("fr")

    with AnkiDeckBuilder(
        path=Path("test-deck.apkg"),
        deck_name="French Vocabulary",
        source_language="fr",
    ) as deck:
        llm_batch: list = []

        for enriched in generate_enriched_lemmas(vocab, dictionary):
            if not needs_disambiguation(enriched):
                deck.add(assign_single_sense(enriched))
            else:
                llm_batch.append(enriched)

                if len(llm_batch) >= PARALLELISM:
                    results = await asyncio.gather(
                        *[disambiguate_senses(e) for e in llm_batch]
                    )
                    for assignments in results:
                        for a in assignments:
                            deck.add(a)
                    llm_batch = []

        # Flush remaining
        if llm_batch:
            results = await asyncio.gather(
                *[disambiguate_senses(e) for e in llm_batch]
            )
            for assignments in results:
                for a in assignments:
                    deck.add(a)

    print(f"Generated: test-deck.apkg")


asyncio.run(main())
```

### Acceptance Criteria

- [ ] `AnkiDeckBuilder` is a context manager
- [ ] `.add()` accepts `SenseAssignment` and queues a card
- [ ] On context exit, writes valid `.apkg` file
- [ ] Card front shows English translation
- [ ] Card back shows word, IPA, examples, forms
- [ ] Examples are formatted with guillemets and word highlighted
- [ ] CSS styling applied correctly
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy .` passes
- [ ] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Future Considerations (Out of Scope)

- **Gender/articles**: Extract gender from kaikki, prepend "le/la" to nouns
- **CLI interface**: Command-line tool for running the full pipeline
- **Target language translations**: If kaikki adds them, or via separate LLM call
- **Audio pronunciation**: Embed TTS audio in cards
- **Progress reporting**: Callbacks or progress bars during processing
- **Caching**: Cache LLM disambiguation results for reruns
- **Card deduplication**: Handle same sense appearing from different lemmas
