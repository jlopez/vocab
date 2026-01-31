# vocab

![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

Extract vocabulary with frequency data from ePub files for language learning.

## Overview

vocab is a Python library that extracts words from ePub files, performs lemmatization using spaCy, and builds frequency lists. The primary use case is generating vocabulary lists for creating Anki flashcards, prioritizing the most frequently used words.

## Features

- Extract text from ePub files (epub2 and epub3)
- Preserve chapter context (title, index)
- Language-aware sentence splitting
- Context-aware lemmatization (e.g., French "suis" → "être" or "suivre" based on context)
- Track word frequencies with original forms and example sentences

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
python -m spacy download fr_core_news_sm

# Other languages
python -m spacy download en_core_web_sm  # English
python -m spacy download de_core_news_sm  # German
python -m spacy download es_core_news_sm  # Spanish
```

## Usage

### Extract Chapters from ePub

```python
from pathlib import Path
from vocab import extract_chapters

for chapter in extract_chapters(Path("book.epub")):
    print(f"Chapter {chapter.index}: {chapter.title}")
    print(f"Text preview: {chapter.text[:100]}...")
```

### Coming Soon

- Sentence extraction with spaCy
- Token extraction with lemmatization
- Vocabulary building with frequency analysis
- Export to Anki-compatible formats

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

## License

MIT
