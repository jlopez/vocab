# CLI Implementation

## Overview

Add a CLI (`vocab build`) using typer that runs the full epub-to-Anki pipeline with
configurable parameters and optional intermediate artifact output.

### Key Design Decisions

| Decision | Resolution |
|----------|------------|
| CLI framework | typer (with rich for progress output) |
| Entry point | `vocab` → `src/vocab/cli.py:app` |
| Subcommand | `build` — runs full epub→Anki pipeline |
| Vocabulary building | New `VocabularyBuilder` class (builder pattern, no context manager) |
| Artifact output | Null object pattern: `ArtifactWriter` / `NullArtifactWriter` |
| Artifact naming | Numbered prefixes (`01-chapters.jsonl`, `02-sentences.jsonl`, ...) |
| POS filtering | Applied before `--top` so top-N reflects useful cards only |
| .env loading | `python-dotenv` loads `ANTHROPIC_API_KEY` at CLI startup |

### CLI Interface

```
vocab build EPUB -l LANG [OPTIONS]
```

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `EPUB` | | (required) | Path to input epub file |
| `--language` | `-l` | (required) | 2-letter language code |
| `--output` | `-o` | `{epub_stem}.apkg` | Output .apkg path |
| `--deck-name` | `-d` | `"{Language} Vocabulary"` | Anki deck name |
| `--artifacts` | | (disabled) | Directory for intermediate artifacts |
| `--top` | `-t` | (all) | Limit to N most frequent lemmas |
| `--max-examples` | `-n` | 3 | Max example sentences per lemma |
| `--pos` | | `NOUN,VERB,ADJ,ADV` | Comma-separated POS tags to include |
| `--model` | `-m` | `claude-haiku` | LLM model for disambiguation |
| `--no-audio` | | false | Skip audio downloads |
| `--no-disambiguation` | | false | Skip LLM disambiguation step |

### Artifact Files

When `--artifacts DIR` is provided:

```
DIR/
  01-chapters.jsonl
  02-sentences.jsonl
  03-tokens.jsonl
  04-vocabulary.json
  05-enriched.jsonl
  05-rejected.jsonl
  06-assigned.jsonl
  06-ambiguous.jsonl
  07-disambiguated.jsonl
  07-failed.jsonl
```

### Pipeline Flow

```
epub → extract_chapters → extract_sentences → extract_tokens → VocabularyBuilder
  → enrich_lemma (POS filter + top-N) → triage (needs_disambiguation?)
  → assign_single_sense / disambiguate_senses → AnkiDeckBuilder → .apkg
```

---

## Phase 1: VocabularyBuilder Refactor

Extract the token accumulation logic from `build_vocabulary()` into a standalone
`VocabularyBuilder` class with `add()` and `build()` methods.

### Changes

**src/vocab/vocabulary.py**
- Add `VocabularyBuilder` class:
  ```python
  class VocabularyBuilder:
      def __init__(self, language: str, max_examples: int = 3): ...
      def add(self, token: Token) -> None: ...
      def build(self) -> Vocabulary: ...
  ```
  - `add()` encapsulates: frequency counting, form tracking, example collection, POS skip
  - `build()` converts accumulated state into a `Vocabulary` object
- Refactor `build_vocabulary()` to use `VocabularyBuilder` internally

**src/vocab/__init__.py**
- Export `VocabularyBuilder`

**tests/test_vocabulary.py**
- Add tests for `VocabularyBuilder`:
  - `add()` accumulates tokens correctly
  - `build()` produces identical output to current `build_vocabulary()`
  - `max_examples` limit is respected
  - Empty builder produces empty vocabulary

### Acceptance Criteria

- [x] `VocabularyBuilder` class with `add()` and `build()` methods
- [x] `build_vocabulary()` reimplemented as thin wrapper around `VocabularyBuilder`
- [x] Existing tests continue to pass unchanged
- [x] New unit tests for `VocabularyBuilder`
- [x] ruff, mypy, pytest pass

---

## Phase 2: Artifacts System

Create the artifact writer classes using the null object pattern.

### Changes

**src/vocab/artifacts.py** (new)
- `ArtifactWriter` — context manager that writes numbered JSONL/JSON files:
  - `__init__(self, directory: Path)` — creates directory, opens file handles
  - `__enter__` / `__exit__` — manages file handle lifecycle
  - `write_chapter(chapter)` → `01-chapters.jsonl`
  - `write_sentence(chapter, sentence)` → `02-sentences.jsonl`
  - `write_token(token)` → `03-tokens.jsonl`
  - `write_vocabulary(vocab)` → `04-vocabulary.json`
  - `write_enriched(enriched)` → `05-enriched.jsonl`
  - `write_rejected(entry)` → `05-rejected.jsonl`
  - `write_assigned(assignment)` → `06-assigned.jsonl`
  - `write_ambiguous(enriched)` → `06-ambiguous.jsonl`
  - `write_disambiguated(assignment)` → `07-disambiguated.jsonl`
  - `write_failed(enriched)` → `07-failed.jsonl`
- `NullArtifactWriter` — same interface, all methods are no-ops
  - Also a context manager (enter/exit do nothing)

**src/vocab/__init__.py**
- Export `ArtifactWriter`, `NullArtifactWriter`

**tests/test_artifacts.py** (new)
- `ArtifactWriter` creates directory and writes correct files
- Each `write_*` method produces valid JSONL/JSON
- `NullArtifactWriter` writes nothing, works as context manager
- Files are properly closed on `__exit__`

### Acceptance Criteria

- [x] `ArtifactWriter` writes all 10 artifact files with correct numbering
- [x] `NullArtifactWriter` is a drop-in replacement that writes nothing
- [x] Both work as context managers
- [x] ruff, mypy, pytest pass

---

## Phase 3: CLI Implementation

Create the typer CLI with the `build` subcommand wiring the full pipeline together.

### Changes

**src/vocab/cli.py** (new)
- `app = typer.Typer()` with `@app.command("build")`
- Loads `.env` via `python-dotenv`
- Implements the full pipeline:
  1. Initialize `VocabularyBuilder` + artifact writer
  2. Triple loop: chapters → sentences → tokens (with artifact output)
  3. Build vocabulary, apply POS filter, apply top-N
  4. Enrich with dictionary
  5. Triage ambiguous/unambiguous
  6. Disambiguate (unless `--no-disambiguation`)
  7. Build Anki deck (with or without audio)
- Progress output via rich/typer (card count, stage indicators)
- Early validation: check epub exists, language supported, API key present if needed

**pyproject.toml**
- Add dependencies: `typer>=0.15`
- Move `python-dotenv` from dev to regular dependencies
- Add entry point: `[project.scripts] vocab = "vocab.cli:app"`

**tests/test_cli.py** (new)
- Test CLI argument parsing and validation
- Test full pipeline with mocked dependencies (no real epub/API calls)
- Test `--no-disambiguation` skips LLM step
- Test `--artifacts` triggers artifact writing
- Test default values for output path and deck name

### Acceptance Criteria

- [x] `vocab build book.epub -l fr` runs the full pipeline
- [x] All CLI options work as specified
- [x] `--artifacts dir/` writes all numbered artifact files
- [x] `--no-disambiguation` works without API key
- [x] `--no-audio` skips audio downloads
- [x] Early validation with clear error messages
- [x] Running CLI with `--artifacts` against the same epub/language as `data/phaseN.py` produces artifacts identical to those in `data/` (phases 1–6; phase 7 is non-deterministic due to LLM)
- [x] ruff, mypy, pytest pass with coverage ≥ 90%

---

## Phase 4: README Update

Update documentation to reflect the new CLI and refactored API.

### Changes

**README.md**
- Add CLI section (installation, usage examples):
  - Minimal usage
  - Full control with all options
  - Quick mode (no LLM, no audio)
  - Artifacts mode for debugging
- Update Architecture table to include CLI layer
- Add `VocabularyBuilder` to usage examples or at least mention it
- Update coverage badge if needed
- Verify all existing examples still accurate

### Acceptance Criteria

- [x] CLI documented with practical usage examples
- [x] Architecture table updated
- [x] Coverage badge accurate
- [x] All examples in README are correct and runnable
