# vocab

![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

Extract vocabulary with frequency data from ePub files for language learning.

## Overview

vocab is a Python library that extracts words from ePub files, performs lemmatization using spaCy, and builds frequency lists. The primary use case is generating vocabulary lists for creating Anki flashcards, prioritizing the most frequently used words.

## Features

- Extract text from ePub files (epub2 and epub3)
- Preserve chapter context (title, index)
- Language-aware sentence splitting
- Context-aware lemmatization (e.g., French "suis" → "être" or "suivre" based on context)
- Track word frequencies with original forms and example sentences
- Dictionary lookups via Wiktionary (kaikki.org) with POS filtering
- LLM-powered sense disambiguation for polysemous words
- Export to Anki `.apkg` format with styled flashcards

## Installation

```bash
# Using uv
uv add vocab

# Using pip
pip install vocab
```

### spaCy Models

You'll need to download spaCy language models for the languages you want to process:

```bash
# French
python -m spacy download fr_core_news_lg

# Other languages
python -m spacy download en_core_web_lg  # English
python -m spacy download de_core_news_lg  # German
python -m spacy download es_core_news_lg  # Spanish
```

We use the large (`_lg`) models rather than small (`_sm`) models because they produce significantly better lemmatization. In testing with French text, the small model generated truncated lemmas (e.g., "couvertur" instead of "couverture"), spurious verb forms (e.g., "villa" → "viller"), and inconsistent case normalization. The large models produce ~12% fewer unique lemmas through better consolidation, meaning cleaner vocabulary lists with fewer errors to manually correct. The trade-off is larger download size (~500MB vs ~15MB per model).

## Usage

### Extract Chapters from ePub

```python
from pathlib import Path
from vocab import extract_chapters

for chapter in extract_chapters(Path("book.epub")):
    print(f"Chapter {chapter.index}: {chapter.title}")
    print(f"Text preview: {chapter.text[:100]}...")
```

### Extract Sentences from Text

```python
from vocab import extract_sentences

text = "Bonjour! Comment allez-vous? Je vais très bien."
for sentence in extract_sentences(text, "fr"):
    print(f"Sentence {sentence.index}: {sentence.text}")
```

### Extract Tokens with Lemmatization

```python
from vocab import extract_chapters, extract_sentences, extract_tokens, SentenceLocation

for chapter in extract_chapters(Path("book.epub")):
    for sentence in extract_sentences(chapter.text, "fr"):
        location = SentenceLocation(
            chapter_index=chapter.index,
            chapter_title=chapter.title,
            sentence_index=sentence.index,
        )
        for token in extract_tokens(sentence.text, location, "fr"):
            print(f"{token.original} → {token.lemma}")
```

### Build Vocabulary with Frequency Analysis

```python
from pathlib import Path
from vocab import build_vocabulary

# Build vocabulary from an ePub (max_examples must be >= 0)
vocab = build_vocabulary(Path("book.epub"), "fr", max_examples=3)

# Get the top 10 most frequent lemmas (n must be >= 1)
for entry in vocab.top(10):
    print(f"{entry.lemma}: {entry.frequency} occurrences")
    print(f"  Forms: {entry.forms}")
    if entry.examples:
        print(f"  Example: {entry.examples[0].sentence}")

# Export as JSON-serializable dict
import json
with open("vocabulary.json", "w") as f:
    json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)
```

**Note:** Processing very large ePubs builds the entire vocabulary in memory. For typical books this is not an issue, but extremely large documents may require significant memory.

### Dictionary Lookups

```python
from vocab import Dictionary, SPACY_TO_KAIKKI

# Initialize dictionary (downloads and caches from kaikki.org)
dictionary = Dictionary("fr")

# Look up a word with POS filtering
kaikki_pos = SPACY_TO_KAIKKI.get("NOUN", [])
entries = dictionary.lookup("faux", pos=kaikki_pos)

for entry in entries:
    print(f"{entry.word} [{entry.pos}] - IPA: {entry.ipa}")
    for sense in entry.senses:
        print(f"  - {sense.translation}")
```

### Enrichment and Disambiguation Pipeline

```python
import asyncio
from vocab import (
    build_vocabulary,
    Dictionary,
    enrich_lemma,
    needs_disambiguation,
    assign_single_sense,
    disambiguate_senses,
)
from pathlib import Path

async def process_vocabulary():
    vocab = build_vocabulary(Path("book.epub"), "fr", max_examples=3)
    dictionary = Dictionary("fr")

    for lemma_entry in vocab:
        enriched = enrich_lemma(lemma_entry, dictionary)
        if not enriched:
            continue  # No dictionary match

        if not needs_disambiguation(enriched):
            # Single sense - trivial assignment
            assignment = assign_single_sense(enriched)
        else:
            # Multiple senses - use LLM
            assignments = await disambiguate_senses(enriched)
            # Process assignments...

asyncio.run(process_vocabulary())
```

### Export to Anki

```python
import asyncio
from pathlib import Path
from vocab import (
    build_vocabulary,
    Dictionary,
    AnkiDeckBuilder,
    enrich_lemma,
    needs_disambiguation,
    assign_single_sense,
    disambiguate_senses,
)

async def create_anki_deck():
    vocab = build_vocabulary(Path("book.epub"), "fr", max_examples=3)
    dictionary = Dictionary("fr")

    with AnkiDeckBuilder(
        path=Path("vocabulary.apkg"),
        deck_name="French Vocabulary",
        source_language="fr",
    ) as deck:
        for lemma_entry in vocab.top(100):
            enriched = enrich_lemma(lemma_entry, dictionary)
            if not enriched:
                continue

            if not needs_disambiguation(enriched):
                deck.add(assign_single_sense(enriched))
            else:
                for assignment in await disambiguate_senses(enriched):
                    deck.add(assignment)

    print("Generated: vocabulary.apkg")

asyncio.run(create_anki_deck())
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/jesusla/vocab.git
cd vocab

# Install dependencies (including dev)
uv sync
```

### Running Tests

```bash
uv run pytest
```

### Linting and Type Checking

```bash
uv run ruff check src tests
uv run mypy src
```

## Architecture

The library is built as a layered pipeline:

| Layer | Module | Function | Description |
|-------|--------|----------|-------------|
| L0 | `epub.py` | `extract_chapters()` | ePub → chapters with text |
| L1 | `sentences.py` | `extract_sentences()` | Text → sentences (spaCy) |
| L2 | `tokens.py` | `extract_tokens()` | Sentences → lemmatized tokens |
| L3 | `vocabulary.py` | `build_vocabulary()` | Tokens → frequency data |
| L4 | `dictionary.py` | `Dictionary.lookup()` | Wiktionary lookups (kaikki.org) |
| L5 | `pipeline.py` | `enrich_lemma()`, `disambiguate_senses()` | Enrichment + LLM disambiguation |
| L6 | `anki.py` | `AnkiDeckBuilder` | Export to Anki `.apkg` format |

## License

MIT
