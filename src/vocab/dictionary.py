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

    def lookup(
        self,
        lemma: str,
        target_language: str | None = None,
    ) -> DictionaryEntry | None:
        """Look up a word in the dictionary.

        Args:
            lemma: Word to look up.
            target_language: Optional target language for translations (e.g., "es").
                If provided, target_translations will be populated.

        Returns:
            DictionaryEntry if found, None otherwise.
        """
        self._ensure_loaded()
        assert self._data is not None

        entries = self._data.get(lemma)
        if not entries:
            return None

        # Aggregate data from all entries (same word may have multiple POS)
        ipa = self._extract_ipa(entries)
        gender = self._extract_gender(entries)
        pos = self._extract_pos(entries)
        translations_en = self._extract_english_translations(entries)
        target_translations = (
            self._extract_target_translations(entries, target_language) if target_language else []
        )

        return DictionaryEntry(
            lemma=lemma,
            language=self._language,
            ipa=ipa,
            gender=gender,
            pos=pos,
            translations_en=translations_en,
            target_language=target_language,
            target_translations=target_translations,
        )

    def _extract_ipa(self, entries: list[dict[str, Any]]) -> str | None:
        """Extract IPA pronunciation from entries."""
        for entry in entries:
            sounds = entry.get("sounds", [])
            for sound in sounds:
                ipa = sound.get("ipa")
                if isinstance(ipa, str) and ipa:
                    return ipa
        return None

    def _extract_gender(self, entries: list[dict[str, Any]]) -> str | None:
        """Extract grammatical gender from entries (for nouns)."""
        for entry in entries:
            # Check tags
            tags = entry.get("tags", [])
            if "masculine" in tags:
                return "m"
            if "feminine" in tags:
                return "f"

            # Check head_templates
            for template in entry.get("head_templates", []):
                args = template.get("args", {})
                # Common pattern: g or g1 contains gender
                for key in ["g", "g1", "1"]:
                    gender = args.get(key, "")
                    if gender in ("m", "m-p", "mf", "m-f"):
                        return "m"
                    if gender in ("f", "f-p"):
                        return "f"

            # Check forms for gender markers
            forms = entry.get("forms", [])
            for form in forms:
                form_tags = form.get("tags", [])
                if "masculine" in form_tags:
                    return "m"
                if "feminine" in form_tags:
                    return "f"

        return None

    def _extract_pos(self, entries: list[dict[str, Any]]) -> str | None:
        """Extract primary part of speech from entries."""
        # Prefer certain POS over others
        pos_priority = ["noun", "verb", "adj", "adv", "prep", "conj", "pron", "det"]

        found_pos: set[str] = set()
        for entry in entries:
            pos = entry.get("pos")
            if pos:
                found_pos.add(pos)

        for preferred in pos_priority:
            if preferred in found_pos:
                return preferred

        # Return first found if none in priority list
        return next(iter(found_pos), None)

    def _extract_english_translations(self, entries: list[dict[str, Any]]) -> list[str]:
        """Extract English translations from glosses."""
        translations: list[str] = []
        seen: set[str] = set()

        for entry in entries:
            for sense in entry.get("senses", []):
                for gloss in sense.get("glosses", []):
                    # Clean up the gloss
                    gloss = gloss.strip()
                    if gloss and gloss not in seen:
                        translations.append(gloss)
                        seen.add(gloss)

        return translations

    def _extract_target_translations(
        self, entries: list[dict[str, Any]], target_language: str
    ) -> list[str]:
        """Extract translations in the target language."""
        target_name = LANGUAGE_NAMES.get(target_language, target_language)
        translations: list[str] = []
        seen: set[str] = set()

        for entry in entries:
            for sense in entry.get("senses", []):
                for trans in sense.get("translations", []):
                    # Check if this translation is for the target language
                    lang = trans.get("lang", "")
                    if lang.lower() == target_name.lower() or trans.get("code") == target_language:
                        word = trans.get("word", "").strip()
                        if word and word not in seen:
                            translations.append(word)
                            seen.add(word)

        return translations
