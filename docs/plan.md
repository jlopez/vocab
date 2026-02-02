# Plan: Index Vocabulary by (Lemma, POS)

## Overview

Restructure `Vocabulary` to index entries by both lemma and part-of-speech (POS), treating the same lemma with different POS as distinct entries (e.g., "run" as noun vs verb).

### Architecture

- **Indexing**: `dict[str, dict[str, LemmaEntry]]` (lemma -> pos -> entry)
- **LemmaEntry**: Add `pos: str` field
- **Token filtering**: Skip tokens with empty POS via blocklist
- **Remove**: `Vocabulary.top()` method

### Files to Modify

| File | Changes |
|------|---------|
| `src/vocab/models.py` | Add `pos` to `LemmaEntry`, change `Vocabulary.entries` type, remove `top()` |
| `src/vocab/vocabulary.py` | Add `SKIP_POS` blocklist, key aggregation by `(lemma, pos)` |
| `tests/test_vocabulary.py` | Update all tests for new structure, delete `TestVocabularyTop` class |

---

## Phase 1: Update Data Models ✓

### Deliverables

1. Add `pos: str` field to `LemmaEntry` dataclass (after `lemma`)
2. Change `Vocabulary.entries` type annotation to `dict[str, dict[str, LemmaEntry]]`
3. Update `Vocabulary` docstring to reflect nested structure
4. Remove `Vocabulary.top()` method entirely
5. `to_dict()` remains unchanged (relies on `asdict()` which handles nesting)

### Acceptance Criteria

- [x] `LemmaEntry` has `pos: str` field
- [x] `Vocabulary.entries` is typed as `dict[str, dict[str, LemmaEntry]]`
- [x] `top()` method is removed
- [x] Code passes `mypy`

---

## Phase 2: Update Vocabulary Builder ✓

### Deliverables

1. Add `SKIP_POS: set[str] = {""}` constant at module level
2. Update `build_vocabulary()` to:
   - Skip tokens where `token.pos` is in `SKIP_POS`
   - Key aggregation dicts by `(lemma, pos)` tuple internally
   - Build nested `entries` dict: `entries[lemma][pos] = LemmaEntry(...)`
3. Include `pos` when constructing `LemmaEntry`

### Acceptance Criteria

- [x] Tokens with empty POS are skipped
- [x] Vocabulary entries are keyed by lemma then POS
- [x] Each `LemmaEntry` has correct `pos` value
- [x] Code passes `mypy`

---

## Phase 3: Update Tests ✓

### Deliverables

1. Delete `TestVocabularyTop` class entirely
2. Delete `TestTopNValidation` class entirely
3. Update mock spaCy fixture to return controlled POS values (mock `pos_` attribute on tokens)
4. Update all tests that access `vocab.entries[lemma]` to use `vocab.entries[lemma][pos]`
5. Update assertions that check `entry.lemma == lemma` to also verify `entry.pos`
6. Update `to_dict` tests for nested structure
7. Add test: tokens with empty POS are not included in vocabulary

### Acceptance Criteria

- [x] All tests pass
- [x] No tests reference `top()` method
- [x] Tests verify POS is correctly captured
- [x] Test coverage remains >= 90%

---

## Phase 4: Final Validation ✓

### Deliverables

1. Run full test suite with coverage
2. Run `mypy` and `ruff`
3. Update coverage badge in README.md if needed

### Acceptance Criteria

- [x] `uv run pytest --cov` passes with >= 90% coverage (100%)
- [x] `uv run mypy src/ tests/` passes with no errors
- [x] `uv run ruff check .` passes
- [x] Coverage badge is accurate (100% brightgreen)
