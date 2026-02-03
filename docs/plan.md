# Audio Support for Anki Cards

## Overview

Add audio playback to generated Anki cards by extracting audio URLs from kaikki dictionary data and embedding MP3 files in the `.apkg` package.

## Architecture

### Data Flow

```
kaikki sounds[] → DictionaryEntry.audio_url → AnkiDeckBuilder.add() → download MP3 → embed in .apkg
```

### Key Decisions

| Decision | Resolution |
|----------|------------|
| Audio format | MP3 (universal Anki support) |
| Which audio | First `mp3_url` in kaikki `sounds[]` |
| Filename | `{card_guid}.mp3` |
| URL streaming | Not viable - must embed files |
| Download strategy | Async with semaphore-limited concurrency |
| Concurrency limit | 16 concurrent downloads (configurable) |
| Caching | `~/.cache/vocab/audio/` to avoid re-downloads |
| Error handling | Log warning, card works without audio |
| Replay | Built-in Anki behavior (R or F5 hotkey) |

---

## Phase 1: Data Model

Add audio URL extraction to the dictionary layer.

### Changes

**src/vocab/dictionary.py**
- Add `audio_url: str | None` field to `DictionaryEntry`
- Add `_extract_audio_url()` static method that returns first `mp3_url` from `sounds[]`
- Update `from_kaikki()` to call the new extractor

**tests/test_dictionary.py**
- Add test for `_extract_audio_url()` with various sound configurations:
  - Entry with multiple audio files (returns first mp3_url)
  - Entry with only IPA (no audio) → returns None
  - Entry with ogg_url but no mp3_url → returns None
  - Empty sounds array → returns None

### Acceptance Criteria

- [x] `DictionaryEntry` has `audio_url: str | None` field
- [x] First available `mp3_url` is extracted from kaikki data
- [x] Entries without audio have `audio_url=None`
- [x] All tests pass, ruff/mypy clean

---

## Phase 2: Media Cache Module

Create a reusable module for downloading and caching media files.

### Changes

**src/vocab/media_cache.py** (new file)
- `DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vocab" / "media"`
- `async def fetch_media(url: str, filename: str, cache_dir: Path | None = None) -> Path`
  - Computes tiered path: `{cache_dir}/{filename[:2]}/{filename}`
  - Returns cached path immediately if file exists
  - Downloads via `httpx.AsyncClient` if not cached
  - Creates parent directories as needed
  - Returns the local file path
- Uses tiered directory structure to avoid filesystem slowdown with many files

**tests/test_media_cache.py** (new file)
- Test successful download and caching
- Test cache hit (no network request on second call)
- Test tiered directory structure (`ab/abcdef.mp3`)
- Test download failure raises appropriate exception
- Test custom cache_dir parameter

### Acceptance Criteria

- [x] `fetch_media()` downloads and caches files
- [x] Tiered directory structure: `{cache_dir}/{filename[:2]}/{filename}`
- [x] Cache hits skip network requests
- [x] All tests pass, ruff/mypy clean

---

## Phase 3: Async Anki Builder with Audio

Make `AnkiDeckBuilder` async and integrate audio download/embedding.

### Changes

**src/vocab/anki.py**
- Convert `AnkiDeckBuilder` to async context manager (`__aenter__`/`__aexit__`)
- Add `max_concurrent_downloads: int = 16` constructor parameter
- Add `_download_semaphore: asyncio.Semaphore` instance variable
- Add `_media_files: list[str]` to track downloaded audio paths
- Convert `add()` to `async def add()`:
  - If `entry.word.audio_url` exists:
    - Compute filename as `{guid}.mp3`
    - Acquire semaphore, call `fetch_media()` from `media_cache` module
    - Add path to `_media_files`
    - Set `Audio` field to `[sound:{filename}]`
  - Handle download failures gracefully (log warning, continue without audio)
  - Add note to deck (unchanged)
- Update `__aexit__` to pass `media_files` to `genanki.Package`
- Add `Audio` field to model fields list
- Update `BACK_TEMPLATE` to include `{{Audio}}` near the word/IPA

**tests/test_anki.py**
- Update existing tests to use `async with` and `await deck.add()`
- Add test for audio download and embedding:
  - Mock `fetch_media` to return a path
  - Verify `[sound:...]` appears in card field
  - Verify media file is included in package
- Add test for missing audio URL (card still works)
- Add test for download failure (logs warning, card still works)

**README.md**
- Update "Export to Anki" example to use `async with` and `await`
- Mention audio support in Features section

### Acceptance Criteria

- [x] `AnkiDeckBuilder` is an async context manager
- [x] `add()` is async and downloads audio with concurrency limit (default 16)
- [x] Audio downloaded via `media_cache.fetch_media()`
- [x] Cards include `[sound:{guid}.mp3]` when audio available
- [x] Cards work normally when audio unavailable or download fails
- [x] All tests pass, ruff/mypy clean, coverage maintained
- [x] README updated with async usage

---

## Future Considerations (Out of Scope)

- Multiple audio files per card (random selection would require JS)
- Audio quality preferences (some kaikki entries have multiple recordings)
- Offline mode / pre-download all audio for a language
