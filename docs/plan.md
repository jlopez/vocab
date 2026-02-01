# Anki Flashcard Generation Pipeline

## Overview

Build a pipeline to generate Anki flashcards from extracted vocabulary for language learning. The cards are designed for active recall training: the front shows a meaning in the target language (e.g., Spanish), and the back shows the source language word (e.g., French) with IPA pronunciation, gender (for nouns), example sentences, and variant forms.

The pipeline is language-agnostic, supporting any source/target language pair available in Wiktionary.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  L0: Vocabulary │     │  L1: Dictionary │     │  L2: Translation│     │   L3: Anki      │
│    Filtering    │────▶│     Lookup      │────▶│   Refinement    │────▶│    Export       │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
   frequency cutoffs      Wiktionary data          LLM (configurable)     genanki
   proper noun removal    IPA, gender, POS         meaning grouping        .apkg output
                          translations              concise translations
```

### Layer 0: Vocabulary Filtering
- **Input**: Vocabulary object (from JSON or in-memory)
- **Output**: Filtered list of LemmaEntry
- **Responsibility**: Apply frequency thresholds, remove proper nouns

### Layer 1: Dictionary Lookup
- **Input**: Lemma, source language, target language
- **Output**: DictionaryEntry (IPA, gender, POS, translations)
- **Dependencies**: kaikki.org Wiktionary dump (auto-downloaded)
- **Responsibility**: Provide pronunciation and raw translation data

### Layer 2: Translation Refinement
- **Input**: Lemma, DictionaryEntry, example sentences, source/target languages
- **Output**: RefinedTranslation with meanings grouped by semantic usage
- **Dependencies**: Anthropic API (configurable model)
- **Responsibility**: Group examples by meaning, provide concise translations

### Layer 3: Anki Export
- **Input**: List of CardData
- **Output**: .apkg file
- **Dependencies**: genanki
- **Responsibility**: Generate Anki deck with styled cards

## Data Models

```python
# Existing (models.py) - add from_dict()
@dataclass
class Vocabulary:
    entries: dict[str, LemmaEntry]
    language: str

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary": ...

# New (dictionary.py)
@dataclass
class DictionaryEntry:
    """Dictionary entry with pronunciation and translations."""
    lemma: str
    language: str  # Source language code, e.g., "fr"
    ipa: str | None
    gender: str | None  # "m", "f", or None
    pos: str | None  # "noun", "verb", "adj", etc.
    translations_en: list[str]  # Always present (from Wiktionary glosses)
    target_language: str | None  # e.g., "es", None if not requested
    target_translations: list[str]  # Translations in target language

# New (translation.py)
@dataclass
class TranslationMeaning:
    """A single meaning of a word with its translation and examples."""
    translation: str  # Single word/short phrase in target language
    examples: list[str]  # Example sentences using this meaning

@dataclass
class RefinedTranslation:
    """Result of LLM translation refinement."""
    lemma: str
    source_language: str
    target_language: str
    meanings: list[TranslationMeaning]

# New (anki.py)
@dataclass
class CardData:
    """Data for a single Anki flashcard."""
    front: str  # Target language meaning (e.g., Spanish)
    back_word: str  # Source word (with article if noun)
    back_ipa: str | None  # IPA pronunciation
    back_examples: list[str]  # Example sentences for this meaning
    back_forms: list[str]  # Variant forms seen
    tags: list[str]  # e.g., ["noun", "freq:50-100"]
```

## Card Format

**Front (question):**
```
perro
```

**Back (answer):**
```
le chien
/ʃjɛ̃/

« Il a adopté un chien errant. »
« Le chien aboyait sans cesse. »

Forms: chien, chiens
```

Styling:
- Article in bold for nouns
- IPA in a distinct font/color
- Example sentences in italics with guillemets
- Forms in smaller text

## Dependencies

**New packages to add:**
- `genanki` - Anki deck generation
- `anthropic` - LLM API calls
- `httpx` - Dictionary download (async-capable)
- `pycountry` - Language code to name mapping

**External data (auto-downloaded):**
- kaikki.org Wiktionary dumps (~200-300MB per language, cached at `~/.cache/vocab/`)

---

## Phase 1: Vocabulary Filtering (L0)

### Deliverables
- `Vocabulary.from_dict()` class method in `models.py`
- `src/vocab/filtering.py` with `filter_vocabulary()` function
- Unit tests
- Harness script `data/phaseB-1.py`

### Implementation Details

```python
# src/vocab/models.py - add to Vocabulary class
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Vocabulary":
    """Load vocabulary from a JSON-serializable dictionary.

    Args:
        data: Dictionary with 'language' and 'entries' keys,
              as produced by to_dict().

    Returns:
        Vocabulary instance.
    """

# src/vocab/filtering.py
def filter_vocabulary(
    vocab: Vocabulary,
    *,
    min_freq: int = 1,
    max_freq: int | None = None,
    exclude_proper_nouns: bool = True,
) -> list[LemmaEntry]:
    """Filter vocabulary entries based on frequency and type.

    Args:
        vocab: Vocabulary to filter.
        min_freq: Minimum frequency (inclusive). Default 1.
        max_freq: Maximum frequency (inclusive). None for no limit.
        exclude_proper_nouns: If True, exclude lemmas that appear to be
            proper nouns (start with uppercase letter).

    Returns:
        List of LemmaEntry objects matching the criteria,
        sorted by frequency descending.
    """
```

**Proper noun detection:**
- Lemma starts with uppercase letter
- Simple heuristic; may have false positives (sentence-initial words that got lemmatized with capital)
- Good enough for filtering; user can review

### Files to Create
- `src/vocab/filtering.py`
- `tests/test_filtering.py`
- `data/phaseB-1.py` (not checked in)

### Files to Modify
- `src/vocab/models.py` (add `from_dict`)
- `src/vocab/__init__.py` (exports)

### Harness Script (data/phaseB-1.py)
```python
# Load vocabulary from JSON, apply filters, print statistics
import json
from vocab import Vocabulary, filter_vocabulary

with open("phase4-lg.json") as f:
    vocab = Vocabulary.from_dict(json.load(f))

print(f"Total lemmas: {len(vocab.entries)}")

filtered = filter_vocabulary(vocab, min_freq=2, max_freq=500, exclude_proper_nouns=True)
print(f"After filtering: {len(filtered)}")

# Show sample
for entry in filtered[:20]:
    print(f"  {entry.lemma}: {entry.frequency}")
```

### Acceptance Criteria
- [x] `Vocabulary.from_dict()` correctly reconstructs Vocabulary from `to_dict()` output
- [x] `filter_vocabulary()` filters by min/max frequency
- [x] `filter_vocabulary()` excludes proper nouns when enabled
- [x] Results sorted by frequency descending
- [x] `uv run ruff check .` passes
- [x] `uv run mypy .` passes
- [x] `uv run pytest --cov=vocab --cov-fail-under=90` passes

**Phase 1 completed.**

---

## Phase 2: Dictionary Layer (L1)

### Deliverables
- `src/vocab/dictionary.py` with `Dictionary` class
- Auto-download and caching of Wiktionary data
- Unit tests
- Harness script `data/phaseB-2.py`

### Implementation Details

```python
# src/vocab/dictionary.py
@dataclass
class DictionaryEntry:
    """Dictionary entry with pronunciation and translations.

    Attributes:
        lemma: The word being looked up.
        language: Source language code (e.g., "fr").
        ipa: IPA pronunciation, if available.
        gender: Grammatical gender ("m" or "f") for nouns, None otherwise.
        pos: Part of speech (noun, verb, adj, adv, etc.).
        translations_en: English translations (from Wiktionary glosses).
        target_language: Target language code if translations requested, else None.
        target_translations: Translations in target language.
    """
    lemma: str
    language: str
    ipa: str | None
    gender: str | None
    pos: str | None
    translations_en: list[str]
    target_language: str | None
    target_translations: list[str]


# Mapping of language codes to kaikki.org URLs
KAIKKI_URLS = {
    "fr": "https://kaikki.org/dictionary/French/kaikki.org-dictionary-French.jsonl",
    "de": "https://kaikki.org/dictionary/German/kaikki.org-dictionary-German.jsonl",
    "es": "https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.jsonl",
    "it": "https://kaikki.org/dictionary/Italian/kaikki.org-dictionary-Italian.jsonl",
    "pt": "https://kaikki.org/dictionary/Portuguese/kaikki.org-dictionary-Portuguese.jsonl",
}


class Dictionary:
    """Dictionary backed by Wiktionary data.

    Data is automatically downloaded from kaikki.org on first use
    and cached locally.
    """

    def __init__(self, language: str, cache_dir: Path | None = None):
        """Initialize the dictionary for a language.

        Args:
            language: Language code (e.g., "fr", "de", "es").
            cache_dir: Directory for cached data. Defaults to ~/.cache/vocab/

        Raises:
            ValueError: If language is not supported.
        """

    @classmethod
    def supported_languages(cls) -> list[str]:
        """Return list of supported language codes."""
        return list(KAIKKI_URLS.keys())

    def lookup(
        self,
        lemma: str,
        target_language: str | None = None,
    ) -> DictionaryEntry | None:
        """Look up a word.

        Args:
            lemma: Word to look up.
            target_language: Optional target language for translations (e.g., "es").
                If provided, target_translations will be populated.

        Returns:
            DictionaryEntry if found, None otherwise.
        """
```

**Data source:**
- kaikki.org provides Wiktionary extracts as JSONL files
- Download once per language, parse into in-memory dict keyed by lemma
- Cache the parsed data for faster subsequent loads

**Parsing strategy:**
- Each line is a JSON object with word info
- Extract: `word`, `sounds` (for IPA), `senses` (for translations), `pos`
- Gender from `tags` or `head_templates` for nouns
- Target language translations from `senses[].translations` where `lang` matches
- English translations from `senses[].glosses`

### Files to Create
- `src/vocab/dictionary.py`
- `tests/test_dictionary.py`

### Files to Modify
- `src/vocab/__init__.py` (exports)
- `pyproject.toml` (add httpx dependency)

### Harness Script (data/phaseB-2.py)
```python
# Test dictionary lookups and coverage
import json
from vocab import Vocabulary, filter_vocabulary
from vocab.dictionary import Dictionary

# Load filtered vocabulary
with open("phase4-lg.json") as f:
    vocab = Vocabulary.from_dict(json.load(f))
filtered = filter_vocabulary(vocab, min_freq=2, exclude_proper_nouns=True)

# Initialize dictionary (will download on first run)
dictionary = Dictionary("fr")

# Check coverage
found = 0
missing = []
for entry in filtered[:500]:
    result = dictionary.lookup(entry.lemma, target_language="es")
    if result:
        found += 1
    else:
        missing.append(entry.lemma)

print(f"Coverage: {found}/500 ({found/5:.1f}%)")
print(f"Missing: {missing[:20]}")

# Show sample entries
for lemma in ["chien", "manger", "beau", "hormis"]:
    entry = dictionary.lookup(lemma, target_language="es")
    if entry:
        print(f"\n{lemma}:")
        print(f"  IPA: {entry.ipa}")
        print(f"  Gender: {entry.gender}")
        print(f"  EN: {entry.translations_en[:3]}")
        print(f"  ES: {entry.target_translations[:3]}")
```

### Acceptance Criteria
- [x] Dictionary auto-downloads on first use
- [x] Downloaded data cached at `~/.cache/vocab/`
- [x] Supports multiple source languages (fr, de, es, it, pt)
- [x] `lookup()` returns `DictionaryEntry` for known words
- [x] `lookup()` returns `None` for unknown words
- [x] `lookup()` populates target_translations when target_language provided
- [x] IPA correctly extracted from Wiktionary data
- [x] Gender correctly extracted for nouns
- [x] English translations always extracted
- [x] `uv run ruff check .` passes
- [x] `uv run mypy .` passes
- [x] `uv run pytest --cov=vocab --cov-fail-under=90` passes

**Phase 2 completed.**

---

## Phase 3: Translation Refinement (L2)

### Deliverables
- `src/vocab/translation.py` with `refine_translation()` async function
- `TranslationMeaning` and `RefinedTranslation` dataclasses
- Unit tests (with mocked LLM responses)
- Harness script `data/phaseB-3.py`

### Implementation Details

```python
# src/vocab/translation.py
import pycountry

@dataclass
class TranslationMeaning:
    """A single meaning of a word with its translation and examples."""
    translation: str  # Single word/short phrase in target language
    examples: list[str]  # Example sentences using this meaning


@dataclass
class RefinedTranslation:
    """Result of LLM translation refinement."""
    lemma: str
    source_language: str
    target_language: str
    meanings: list[TranslationMeaning]


def get_language_name(code: str) -> str:
    """Convert language code to full name.

    Args:
        code: ISO 639-1 language code (e.g., "fr").

    Returns:
        Full language name (e.g., "French").

    Raises:
        ValueError: If language code is not recognized.
    """
    lang = pycountry.languages.get(alpha_2=code)
    if lang is None:
        raise ValueError(f"Unknown language code: {code}")
    return lang.name


async def refine_translation(
    lemma: str,
    dict_entry: DictionaryEntry | None,
    examples: list[str],
    *,
    source_language: str,
    target_language: str,
    model: str = "claude-haiku",
) -> RefinedTranslation:
    """Refine translations using an LLM, grouping examples by meaning.

    Given a lemma with dictionary data and example sentences, groups the
    examples by semantic meaning and provides a concise translation for each.

    Args:
        lemma: The word to translate.
        dict_entry: Dictionary entry with raw translations, or None if not found.
        examples: List of example sentences from the source text.
        source_language: Source language code (e.g., "fr").
        target_language: Target language code (e.g., "es").
        model: Model identifier (e.g., "claude-haiku", "claude-sonnet").

    Returns:
        RefinedTranslation with meanings grouped by semantic usage.

    Raises:
        ValueError: If dict_entry is provided but doesn't match lemma/language.
    """
    # Validate dict_entry if provided
    if dict_entry is not None:
        if dict_entry.lemma != lemma:
            raise ValueError(
                f"dict_entry.lemma ({dict_entry.lemma}) != lemma ({lemma})"
            )
        if dict_entry.language != source_language:
            raise ValueError(
                f"dict_entry.language ({dict_entry.language}) != "
                f"source_language ({source_language})"
            )
```

**Prompt strategy:**
```
You are helping create {target_language_name}-to-{source_language_name} flashcards
for a {target_language_name} speaker learning {source_language_name}.

{source_language_name} word: {lemma}
Dictionary translations ({target_language_name}): {target_translations}
Dictionary translations (English): {translations_en}

Example sentences from the source text:
1. {example_1}
2. {example_2}
...

Group these examples by meaning. For each distinct meaning used in the examples:
1. Provide the single best {target_language_name} translation (1 word if possible, 2-3 max)
2. List which example numbers use that meaning

If all examples use the same meaning, return a single group.

Output JSON:
{
  "meanings": [
    {"translation": "...", "example_indices": [1, 2]},
    {"translation": "...", "example_indices": [3]}
  ]
}
```

**Model mapping:**
- `claude-haiku` → `claude-3-5-haiku-latest`
- `claude-sonnet` → `claude-sonnet-4-20250514`
- Allow full model IDs as well

### Files to Create
- `src/vocab/translation.py`
- `tests/test_translation.py`

### Files to Modify
- `src/vocab/__init__.py` (exports)
- `pyproject.toml` (add anthropic, pycountry dependencies)

### Harness Script (data/phaseB-3.py)
```python
# Test translation refinement on sample words with async parallelization
import asyncio
import json
from vocab import Vocabulary, filter_vocabulary
from vocab.dictionary import Dictionary
from vocab.translation import refine_translation

CONCURRENCY = 16

async def main():
    with open("phase4-lg.json") as f:
        vocab = Vocabulary.from_dict(json.load(f))
    filtered = filter_vocabulary(vocab, min_freq=5, exclude_proper_nouns=True)

    dictionary = Dictionary("fr")
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def process_one(entry):
        async with semaphore:
            dict_entry = dictionary.lookup(entry.lemma, target_language="es")
            examples = [ex.sentence for ex in entry.examples]

            result = await refine_translation(
                entry.lemma,
                dict_entry,
                examples,
                source_language="fr",
                target_language="es",
                model="claude-haiku",
            )
            return result

    # Process sample
    tasks = [process_one(entry) for entry in filtered[100:120]]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"\n{result.lemma}:")
        for meaning in result.meanings:
            print(f"  {meaning.translation}")
            for ex in meaning.examples[:2]:
                print(f"    - {ex[:60]}...")

asyncio.run(main())
```

### Acceptance Criteria
- [ ] `refine_translation()` is async
- [ ] Returns `RefinedTranslation` with grouped meanings
- [ ] Groups examples by semantic meaning
- [ ] Each meaning has concise translation (1-3 words)
- [ ] Validates dict_entry matches lemma and source_language
- [ ] Works with dict_entry=None (uses LLM knowledge only)
- [ ] Model parameter correctly maps to Anthropic model IDs
- [ ] Handles API errors gracefully
- [ ] Unit tests use mocked responses (no real API calls in CI)
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy .` passes
- [ ] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Phase 4: Anki Export + CLI (L3)

### Deliverables
- `src/vocab/anki.py` with `generate_deck()` function
- `src/vocab/cli.py` with command-line interface
- Card template with CSS styling
- Unit tests
- Harness script `data/phaseB-4.py`

### Implementation Details

```python
# src/vocab/anki.py
@dataclass
class CardData:
    """Data for a single Anki flashcard.

    Attributes:
        front: The question side (target language meaning).
        back_word: Source word, with article if noun (e.g., "le chien").
        back_ipa: IPA pronunciation, if available.
        back_examples: Example sentences for this specific meaning.
        back_forms: List of variant forms seen in source.
        tags: Tags for the card (e.g., ["noun", "freq:10-50"]).
    """
    front: str
    back_word: str
    back_ipa: str | None
    back_examples: list[str]
    back_forms: list[str]
    tags: list[str]


def generate_deck(
    cards: list[CardData],
    deck_name: str,
    output_path: Path,
) -> Path:
    """Generate an Anki deck from card data.

    Args:
        cards: List of CardData objects.
        deck_name: Name for the Anki deck.
        output_path: Path for the output .apkg file.

    Returns:
        Path to the generated .apkg file.
    """


# src/vocab/cli.py
def main() -> None:
    """CLI entry point for vocab commands."""
```

**CLI interface:**
```bash
uv run vocab anki input.json -o deck.apkg \
    --source-language fr \
    --target-language es \
    --deck-name "French Vocabulary" \
    --min-freq 2 \
    --max-freq 500 \
    --model claude-haiku
```

**Arguments:**
- `input.json`: Vocabulary JSON file (from phase 4 of previous plan)
- `-o, --output`: Output .apkg path (default: `output.apkg`)
- `--source-language`: Source language code (default: inferred from vocabulary)
- `--target-language`: Target language code (required)
- `--deck-name`: Name for the deck (default: "{Source} Vocabulary")
- `--min-freq`: Minimum frequency filter (default: 1)
- `--max-freq`: Maximum frequency filter (default: None)
- `--model`: LLM model for translation (default: `claude-haiku`)
- `-j, --parallelization`: Number of concurrent LLM calls (default: 16)
- `--exclude-proper-nouns / --include-proper-nouns`: Proper noun handling (default: exclude)

**Card template (HTML/CSS):**
```html
<!-- Front -->
<div class="front">{{Front}}</div>

<!-- Back -->
<div class="back">
  <div class="word">{{Word}}</div>
  <div class="ipa">{{IPA}}</div>
  <div class="examples">{{Examples}}</div>
  <div class="forms">{{Forms}}</div>
</div>
```

```css
.front { font-size: 24px; text-align: center; }
.word { font-size: 28px; font-weight: bold; }
.word .article { color: #666; }
.ipa { font-family: "Doulos SIL", serif; color: #555; margin: 10px 0; }
.examples { font-style: italic; margin: 15px 0; }
.examples .example { margin: 5px 0; }
.forms { font-size: 14px; color: #888; }
```

**Note on multiple meanings:**
When a word has multiple meanings (from `RefinedTranslation.meanings`), each meaning
becomes a separate card. This ensures each card tests a single meaning with its
relevant examples.

### Files to Create
- `src/vocab/anki.py`
- `src/vocab/cli.py`
- `tests/test_anki.py`
- `tests/test_cli.py`

### Files to Modify
- `src/vocab/__init__.py` (exports)
- `pyproject.toml` (add genanki, add CLI entry point)
- `README.md` (document CLI usage)

### Harness Script (data/phaseB-4.py)
```python
# Full pipeline test: JSON → filtered → translated → .apkg
import asyncio
import json
from pathlib import Path
from vocab import Vocabulary, filter_vocabulary
from vocab.dictionary import Dictionary
from vocab.translation import refine_translation
from vocab.anki import CardData, generate_deck

CONCURRENCY = 16

async def main():
    # Load and filter
    with open("phase4-lg.json") as f:
        vocab = Vocabulary.from_dict(json.load(f))
    filtered = filter_vocabulary(vocab, min_freq=3, max_freq=200, exclude_proper_nouns=True)
    print(f"Filtered to {len(filtered)} lemmas")

    # Build cards with async translation
    dictionary = Dictionary("fr")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    cards = []

    async def process_one(entry):
        async with semaphore:
            dict_entry = dictionary.lookup(entry.lemma, target_language="es")
            examples = [ex.sentence for ex in entry.examples]

            refined = await refine_translation(
                entry.lemma,
                dict_entry,
                examples,
                source_language="fr",
                target_language="es",
                model="claude-haiku",
            )

            # Build back_word with article for nouns
            if dict_entry and dict_entry.gender:
                article = "le" if dict_entry.gender == "m" else "la"
                back_word = f"{article} {entry.lemma}"
            else:
                back_word = entry.lemma

            # Create one card per meaning
            result_cards = []
            for meaning in refined.meanings:
                result_cards.append(CardData(
                    front=meaning.translation,
                    back_word=back_word,
                    back_ipa=dict_entry.ipa if dict_entry else None,
                    back_examples=meaning.examples,
                    back_forms=list(entry.forms.keys()),
                    tags=[],
                ))
            return result_cards

    # Process sample (limit for testing)
    tasks = [process_one(entry) for entry in filtered[:50]]
    results = await asyncio.gather(*tasks)

    for card_list in results:
        cards.extend(card_list)

    print(f"Generated {len(cards)} cards from {len(filtered[:50])} lemmas")

    # Generate deck
    output = generate_deck(cards, "Test French Deck", Path("test-deck.apkg"))
    print(f"Generated: {output}")

asyncio.run(main())
```

### Acceptance Criteria
- [ ] `CardData` correctly holds all card fields
- [ ] `generate_deck()` produces valid .apkg file
- [ ] One card per meaning (multi-meaning words produce multiple cards)
- [ ] Card template renders correctly in Anki
- [ ] CLI parses all arguments correctly
- [ ] CLI supports --source-language and --target-language
- [ ] CLI supports -j/--parallelization for concurrent LLM calls
- [ ] CLI runs full pipeline: load → filter → translate → export
- [ ] Progress output during processing
- [ ] README.md updated with CLI documentation
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy .` passes
- [ ] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Future Considerations (Out of Scope)

- **FR → ES cards**: Reverse direction for passive recall; can add by creating second card type
- **Audio pronunciation**: Embed TTS audio files in .apkg
- **Batch LLM calls**: Optimize API usage by batching multiple words per call
- **Interactive review**: TSV export for manual review/editing before final .apkg generation
- **Multiple source files**: Combine vocabulary from multiple books
- **Custom card templates**: User-configurable HTML/CSS
- **Spaced repetition tuning**: Custom intervals, ease factors
- **GUI/web interface**: Visual tool for card review and editing
