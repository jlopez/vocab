# Vocabulary Extraction Pipeline

## Overview

Build a pipeline to extract vocabulary with frequency data from ePub files for language learning (Anki flashcard generation). The pipeline extracts sentences with chapter context, performs lemmatization using spaCy, and tracks lemma frequencies with original forms and example sentences.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   L0: Epub      │     │   L1: Text      │     │   L2: Token     │     │  L3: Vocabulary │
│   Chapters      │────▶│   Sentences     │────▶│   Extraction    │────▶│   Aggregation   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
     ebooklib            spaCy sentencizer         spaCy nlp              dict grouping
     BeautifulSoup
```

### Layer 0: Chapter Extraction
- **Input**: Path to epub file
- **Output**: Generator of `Chapter` (text, index, title)
- **Dependencies**: ebooklib, beautifulsoup4, lxml
- **Responsibility**: Parse epub, extract chapter text, strip HTML

### Layer 1: Sentence Extraction
- **Input**: Text string, language code
- **Output**: Generator of `Sentence` (text, index)
- **Dependencies**: spaCy
- **Responsibility**: Split text into sentences using language-aware rules

### Layer 2: Token Extraction
- **Input**: Epub path, language code
- **Output**: Generator of `Token` (lemma, original, sentence, location)
- **Dependencies**: Layers 0 + 1
- **Responsibility**: Compose L0 and L1, yield tokens with full context

### Layer 3: Vocabulary Building
- **Input**: Epub path, language code
- **Output**: `Vocabulary` (lemma → frequency, original forms, example sentences)
- **Dependencies**: Layer 2
- **Responsibility**: Aggregate tokens into frequency data

## Data Models

```python
@dataclass
class Chapter:
    text: str
    index: int
    title: str | None

@dataclass
class Sentence:
    text: str
    index: int  # within chapter

@dataclass
class SentenceLocation:
    chapter_index: int
    chapter_title: str | None
    sentence_index: int

@dataclass
class Token:
    lemma: str
    original: str
    sentence: str
    location: SentenceLocation

@dataclass
class LemmaEntry:
    lemma: str
    frequency: int
    forms: dict[str, int]  # original form → count
    examples: list[tuple[str, SentenceLocation]]  # (sentence, location)

@dataclass
class Vocabulary:
    entries: dict[str, LemmaEntry]  # lemma → entry
    language: str
```

## Dependencies

**Existing:**
- ebooklib
- beautifulsoup4
- lxml

**To add:**
- spacy

**spaCy models (not pip packages, downloaded separately):**
- fr_core_news_sm (French, for initial implementation)
- Additional models as needed for other languages

---

## Phase 1: Chapter Extraction (L0)

### Deliverables
- `src/vocab/epub.py` with `extract_chapters()` generator
- `src/vocab/models.py` with `Chapter` dataclass
- Unit tests with a test epub fixture

### Implementation Details

```python
# src/vocab/models.py
@dataclass
class Chapter:
    text: str
    index: int
    title: str | None

# src/vocab/epub.py
def extract_chapters(epub_path: Path) -> Generator[Chapter, None, None]:
    """Extract chapters from an epub file.

    Yields Chapter objects containing the text content, index, and title
    (if available) for each document in the epub's reading order.
    """
```

**Chapter title extraction strategy:**
1. Check epub's Table of Contents (TOC) for item mapping
2. Fall back to first `<h1>` or `<title>` tag in chapter HTML
3. Use `None` if no title found

**Text extraction:**
- Use BeautifulSoup to parse chapter HTML
- Extract text with `get_text(separator=' ')` to preserve word boundaries
- Strip excessive whitespace

### Files to Create
- `src/vocab/models.py`
- `src/vocab/epub.py`
- `tests/conftest.py` (pytest fixtures)
- `tests/test_epub.py`
- `tests/fixtures/` (test epub file)

### Files to Modify
- `src/vocab/__init__.py` (exports)

### Acceptance Criteria
- [x] `extract_chapters()` yields `Chapter` objects in reading order
- [x] Chapter text has HTML stripped, whitespace normalized
- [x] Chapter titles extracted from TOC or heading tags when available
- [x] Works with epub2 and epub3 formats
- [x] `uv run ruff check src tests` passes
- [x] `uv run mypy src` passes
- [x] `uv run pytest --cov=vocab --cov-fail-under=90` passes (97% coverage)

---

## Phase 2: Sentence Extraction (L1)

### Deliverables
- `src/vocab/sentences.py` with `extract_sentences()` generator
- `Sentence` dataclass in models.py
- Unit tests

### Implementation Details

```python
# src/vocab/models.py
@dataclass
class Sentence:
    text: str
    index: int

# src/vocab/sentences.py
def extract_sentences(text: str, language: str) -> Generator[Sentence, None, None]:
    """Extract sentences from text using spaCy's sentence boundary detection.

    Args:
        text: Input text to split into sentences
        language: Language code (e.g., "fr" for French)

    Yields:
        Sentence objects with text and index within input
    """
```

**spaCy model loading:**
- Map language codes to model names (e.g., "fr" → "fr_core_news_sm")
- Cache loaded models to avoid reloading
- Raise clear error if model not installed

**Sentence filtering:**
- Skip empty sentences
- Strip whitespace from sentence text

### Files to Create
- `src/vocab/sentences.py`
- `tests/test_sentences.py`

### Files to Modify
- `src/vocab/models.py` (add Sentence)
- `src/vocab/__init__.py` (exports)
- `pyproject.toml` (add spacy dependency)

### Acceptance Criteria
- [ ] `extract_sentences()` yields `Sentence` objects with correct indices
- [ ] Handles French sentence boundaries correctly (M., Mme., etc.)
- [ ] Model caching works (same model not loaded twice)
- [ ] Clear error message when spaCy model not installed
- [ ] `uv run ruff check src tests` passes
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Phase 3: Token Extraction (L2)

### Deliverables
- `src/vocab/tokens.py` with `extract_tokens()` generator
- `SentenceLocation` and `Token` dataclasses
- Unit tests

### Implementation Details

```python
# src/vocab/models.py
@dataclass
class SentenceLocation:
    chapter_index: int
    chapter_title: str | None
    sentence_index: int

@dataclass
class Token:
    lemma: str
    original: str
    sentence: str
    location: SentenceLocation

# src/vocab/tokens.py
def extract_tokens(epub_path: Path, language: str) -> Generator[Token, None, None]:
    """Extract lemmatized tokens from an epub with full context.

    Composes chapter extraction (L0) and sentence extraction (L1),
    then runs spaCy lemmatization on each sentence.

    Args:
        epub_path: Path to the epub file
        language: Language code (e.g., "fr" for French)

    Yields:
        Token objects with lemma, original form, sentence text, and location
    """
```

**Token filtering:**
- Skip punctuation tokens
- Skip whitespace-only tokens
- Consider: skip stopwords? (make configurable?)

### Files to Create
- `src/vocab/tokens.py`
- `tests/test_tokens.py`

### Files to Modify
- `src/vocab/models.py` (add SentenceLocation, Token)
- `src/vocab/__init__.py` (exports)

### Acceptance Criteria
- [ ] `extract_tokens()` yields `Token` objects with correct lemmas
- [ ] Location correctly tracks chapter and sentence indices
- [ ] Punctuation and whitespace tokens filtered out
- [ ] Context-aware lemmatization works (e.g., "suis" → "être" vs "suivre")
- [ ] `uv run ruff check src tests` passes
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Phase 4: Vocabulary Aggregation (L3)

### Deliverables
- `src/vocab/vocabulary.py` with `build_vocabulary()` function
- `LemmaEntry` and `Vocabulary` dataclasses
- Unit tests

### Implementation Details

```python
# src/vocab/models.py
@dataclass
class LemmaEntry:
    lemma: str
    frequency: int
    forms: dict[str, int]  # original form → count
    examples: list[tuple[str, SentenceLocation]]  # (sentence, location)

@dataclass
class Vocabulary:
    entries: dict[str, LemmaEntry]
    language: str

    def top(self, n: int) -> list[LemmaEntry]:
        """Return top n lemmas by frequency."""

    def to_dict(self) -> dict:
        """Export as serializable dict."""

# src/vocab/vocabulary.py
def build_vocabulary(
    epub_path: Path,
    language: str,
    max_examples: int = 3
) -> Vocabulary:
    """Build vocabulary frequency data from an epub.

    Args:
        epub_path: Path to the epub file
        language: Language code (e.g., "fr" for French)
        max_examples: Maximum example sentences to keep per lemma

    Returns:
        Vocabulary object with aggregated frequency data
    """
```

**Example selection strategy:**
- Keep first N examples encountered (simplest)
- Could later add: prefer shorter sentences, diverse chapters

### Files to Create
- `src/vocab/vocabulary.py`
- `tests/test_vocabulary.py`

### Files to Modify
- `src/vocab/models.py` (add LemmaEntry, Vocabulary)
- `src/vocab/__init__.py` (exports)

### Acceptance Criteria
- [ ] `build_vocabulary()` returns `Vocabulary` with correct frequencies
- [ ] Forms dict tracks all original forms seen
- [ ] Examples capped at `max_examples` per lemma
- [ ] `top()` method returns lemmas sorted by frequency
- [ ] `to_dict()` produces JSON-serializable output
- [ ] `uv run ruff check src tests` passes
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest --cov=vocab --cov-fail-under=90` passes

---

## Future Considerations (Not in Scope)

- CLI interface for running from command line
- Anki export format (.apkg or CSV)
- Multiple epub processing
- Stopword filtering (configurable)
- Part-of-speech filtering (only nouns/verbs/adjectives)
- Progress reporting for large files
- PDF support via different L0 implementation
