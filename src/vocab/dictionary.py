"""Dictionary lookup using Wiktionary data from kaikki.org."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Mapping from spaCy Universal POS tags to kaikki.org POS tags
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

# Mapping of language codes to kaikki.org URLs
KAIKKI_URLS: dict[str, str] = {
    "fr": "https://kaikki.org/dictionary/French/kaikki.org-dictionary-French.jsonl",
    "de": "https://kaikki.org/dictionary/German/kaikki.org-dictionary-German.jsonl",
    "es": "https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.jsonl",
    "it": "https://kaikki.org/dictionary/Italian/kaikki.org-dictionary-Italian.jsonl",
    "pt": "https://kaikki.org/dictionary/Portuguese/kaikki.org-dictionary-Portuguese.jsonl",
}

# Map language codes to their full names (used in Wiktionary translation tags)
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vocab"


@dataclass
class DictionaryExample:
    """Example sentence from Wiktionary."""

    text: str
    translation: str

    @classmethod
    def from_kaikki(cls, raw: dict[str, Any]) -> DictionaryExample:
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
    def from_kaikki(cls, raw: dict[str, Any]) -> DictionarySense:
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
    """A dictionary entry from kaikki (one per etymology).

    Attributes:
        word: The headword.
        pos: Part of speech (kaikki format: "noun", "verb", etc.).
        ipa: IPA pronunciation, if available.
        etymology: Etymology text, if available.
        senses: List of word senses with translations and examples.
    """

    word: str
    pos: str
    ipa: str | None
    etymology: str | None
    senses: list[DictionarySense]

    @classmethod
    def from_kaikki(cls, raw: dict[str, Any]) -> DictionaryEntry:
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
            ipa = sound.get("ipa")
            # Check for string type (kaikki sometimes has lists) and non-empty
            if isinstance(ipa, str) and ipa:
                return ipa
        return None


class Dictionary:
    """Dictionary backed by Wiktionary data from kaikki.org.

    Data is automatically downloaded on first use and cached locally.
    The dictionary is indexed by lemma for fast lookups.
    """

    def __init__(self, language: str, cache_dir: Path | None = None) -> None:
        """Initialize the dictionary for a language.

        Args:
            language: Language code (e.g., "fr", "de", "es").
            cache_dir: Directory for cached data. Defaults to ~/.cache/vocab/

        Raises:
            ValueError: If language is not supported.
        """
        if language not in KAIKKI_URLS:
            supported = ", ".join(sorted(KAIKKI_URLS.keys()))
            raise ValueError(f"Unsupported language: {language}. Supported languages: {supported}")

        self._language = language
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded data
        self._data: dict[str, list[dict[str, Any]]] | None = None

    @property
    def language(self) -> str:
        """Return the language code for this dictionary."""
        return self._language

    @classmethod
    def supported_languages(cls) -> list[str]:
        """Return list of supported language codes."""
        return list(KAIKKI_URLS.keys())

    def _get_cache_path(self) -> Path:
        """Return the path to the cached JSONL file."""
        return self._cache_dir / f"kaikki-{self._language}.jsonl"

    def _get_index_path(self) -> Path:
        """Return the path to the cached index file."""
        return self._cache_dir / f"kaikki-{self._language}-index.json"

    def _download_data(self) -> None:
        """Download the Wiktionary data from kaikki.org."""
        url = KAIKKI_URLS[self._language]
        cache_path = self._get_cache_path()

        logger.info("Downloading Wiktionary data for %s from %s", self._language, url)
        print(f"Downloading Wiktionary data for {self._language}...")

        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with cache_path.open("wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\rDownloading: {pct}%", end="", flush=True)

        print("\nDownload complete.")
        logger.info("Downloaded %d bytes to %s", downloaded, cache_path)

    def _build_index(self) -> dict[str, list[dict[str, Any]]]:
        """Build an index from lemma to list of entries.

        Returns:
            Dictionary mapping lemmas to their entry data.
        """
        cache_path = self._get_cache_path()
        index: dict[str, list[dict[str, Any]]] = {}

        logger.info("Building index for %s", self._language)
        print(f"Building index for {self._language}...")

        with cache_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    word = entry.get("word", "")
                    if word:
                        if word not in index:
                            index[word] = []
                        index[word].append(entry)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse line %d", line_num)

        print(f"Indexed {len(index)} unique lemmas.")
        logger.info("Built index with %d entries", len(index))
        return index

    def _save_index(self, index: dict[str, list[dict[str, Any]]]) -> None:
        """Save the index to disk for faster future loads."""
        index_path = self._get_index_path()
        logger.info("Saving index to %s", index_path)
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index, f)

    def _load_index(self) -> dict[str, list[dict[str, Any]]] | None:
        """Load the index from disk if available and valid.

        Returns:
            The loaded index, or None if not available or outdated.
        """
        index_path = self._get_index_path()
        cache_path = self._get_cache_path()

        if not index_path.exists() or not cache_path.exists():
            return None

        # Check if index is newer than cache
        if index_path.stat().st_mtime < cache_path.stat().st_mtime:
            return None

        try:
            logger.info("Loading index from %s", index_path)
            with index_path.open(encoding="utf-8") as f:
                data: dict[str, list[dict[str, Any]]] = json.load(f)
                return data
        except (json.JSONDecodeError, OSError):
            return None

    def _ensure_loaded(self) -> None:
        """Ensure the dictionary data is loaded, downloading if necessary."""
        if self._data is not None:
            return

        cache_path = self._get_cache_path()

        # Try to load existing index
        index = self._load_index()
        if index is not None:
            self._data = index
            return

        # Download if cache doesn't exist
        if not cache_path.exists():
            self._download_data()

        # Build and save index
        index = self._build_index()
        self._save_index(index)
        self._data = index

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
        assert self._data is not None

        raw_entries = self._data.get(word, [])
        entries = [DictionaryEntry.from_kaikki(raw) for raw in raw_entries]

        if pos:
            entries = [e for e in entries if e.pos in pos]

        assert all(entry.senses for entry in entries), "Entry with no senses found"
        return entries
