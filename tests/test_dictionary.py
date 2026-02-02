"""Tests for the dictionary module."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vocab.dictionary import (
    DEFAULT_CACHE_DIR,
    KAIKKI_URLS,
    SPACY_TO_KAIKKI,
    Dictionary,
    DictionaryEntry,
    DictionaryExample,
    DictionarySense,
)

# Sample Wiktionary entries for testing
SAMPLE_ENTRIES: list[dict[str, Any]] = [
    {
        "word": "chien",
        "pos": "noun",
        "sounds": [{"ipa": "/ʃjɛ̃/"}],
        "etymology_text": "From Latin canis.",
        "senses": [
            {
                "id": "fr-noun-chien-1",
                "glosses": ["dog", "hound"],
                "examples": [
                    {"text": "Le chien aboie.", "translation": "The dog barks."},
                ],
            },
            {
                "id": "fr-noun-chien-2",
                "glosses": ["hammer (of a firearm)"],
            },
        ],
    },
    {
        "word": "manger",
        "pos": "verb",
        "sounds": [{"ipa": "/mɑ̃.ʒe/"}],
        "senses": [
            {
                "id": "fr-verb-manger-1",
                "glosses": ["to eat"],
                "examples": [
                    {"text": "Je mange une pomme.", "english": "I eat an apple."},
                ],
            },
        ],
    },
    {
        "word": "beau",
        "pos": "adj",
        "sounds": [{"ipa": "/bo/"}],
        "senses": [
            {
                "id": "fr-adj-beau-1",
                "glosses": ["beautiful", "handsome", "fine"],
            },
        ],
    },
    # Entry without IPA
    {
        "word": "hormis",
        "pos": "prep",
        "senses": [
            {
                "glosses": ["except", "save"],
            },
        ],
    },
    # Multiple entries for same word (different etymologies)
    {
        "word": "faux",
        "pos": "noun",
        "sounds": [{"ipa": "/fo/"}],
        "etymology_text": "From Latin falsus.",
        "senses": [
            {
                "id": "fr-noun-faux-1",
                "glosses": ["forgery", "fabrication"],
            },
        ],
    },
    {
        "word": "faux",
        "pos": "noun",
        "sounds": [{"ipa": "/fo/"}],
        "etymology_text": "From Latin falx.",
        "senses": [
            {
                "id": "fr-noun-faux-2",
                "glosses": ["scythe"],
            },
        ],
    },
    {
        "word": "faux",
        "pos": "adj",
        "sounds": [{"ipa": "/fo/"}],
        "senses": [
            {
                "id": "fr-adj-faux-1",
                "glosses": ["false", "fake"],
            },
        ],
    },
]


def create_sample_jsonl(entries: list[dict[str, Any]]) -> str:
    """Create a JSONL string from entries."""
    return "\n".join(json.dumps(entry) for entry in entries)


class TestDictionaryExample:
    """Tests for DictionaryExample dataclass."""

    def test_from_kaikki_with_translation(self) -> None:
        """Test parsing example with translation field."""
        raw = {"text": "Le chien aboie.", "translation": "The dog barks."}
        example = DictionaryExample.from_kaikki(raw)
        assert example.text == "Le chien aboie."
        assert example.translation == "The dog barks."

    def test_from_kaikki_with_english(self) -> None:
        """Test parsing example with english field (fallback)."""
        raw = {"text": "Je mange.", "english": "I eat."}
        example = DictionaryExample.from_kaikki(raw)
        assert example.text == "Je mange."
        assert example.translation == "I eat."

    def test_from_kaikki_prefers_translation_over_english(self) -> None:
        """Test that translation field is preferred over english."""
        raw = {"text": "Test", "translation": "preferred", "english": "fallback"}
        example = DictionaryExample.from_kaikki(raw)
        assert example.translation == "preferred"

    def test_from_kaikki_missing_fields(self) -> None:
        """Test parsing example with missing fields."""
        raw: dict[str, Any] = {}
        example = DictionaryExample.from_kaikki(raw)
        assert example.text == ""
        assert example.translation == ""


class TestDictionarySense:
    """Tests for DictionarySense dataclass."""

    def test_from_kaikki_full(self) -> None:
        """Test parsing sense with all fields."""
        raw = {
            "id": "fr-noun-1",
            "glosses": ["dog", "hound"],
            "examples": [{"text": "Le chien.", "translation": "The dog."}],
        }
        sense = DictionarySense.from_kaikki(raw)
        assert sense.id == "fr-noun-1"
        assert sense.translation == "dog"  # First gloss
        assert sense.example is not None
        assert sense.example.text == "Le chien."

    def test_from_kaikki_no_examples(self) -> None:
        """Test parsing sense without examples."""
        raw = {"id": "fr-noun-2", "glosses": ["hammer"]}
        sense = DictionarySense.from_kaikki(raw)
        assert sense.id == "fr-noun-2"
        assert sense.translation == "hammer"
        assert sense.example is None

    def test_from_kaikki_empty_glosses(self) -> None:
        """Test parsing sense with empty glosses."""
        raw: dict[str, Any] = {"id": "test", "glosses": []}
        sense = DictionarySense.from_kaikki(raw)
        assert sense.translation == ""

    def test_from_kaikki_missing_fields(self) -> None:
        """Test parsing sense with missing fields."""
        raw: dict[str, Any] = {}
        sense = DictionarySense.from_kaikki(raw)
        assert sense.id == ""
        assert sense.translation == ""
        assert sense.example is None


class TestDictionaryEntry:
    """Tests for DictionaryEntry dataclass."""

    def test_from_kaikki_full(self) -> None:
        """Test parsing entry with all fields."""
        raw = {
            "word": "chien",
            "pos": "noun",
            "sounds": [{"ipa": "/ʃjɛ̃/"}],
            "etymology_text": "From Latin canis.",
            "senses": [{"id": "fr-noun-1", "glosses": ["dog"]}],
        }
        entry = DictionaryEntry.from_kaikki(raw)
        assert entry.word == "chien"
        assert entry.pos == "noun"
        assert entry.ipa == "/ʃjɛ̃/"
        assert entry.etymology == "From Latin canis."
        assert len(entry.senses) == 1
        assert entry.senses[0].translation == "dog"

    def test_from_kaikki_no_ipa(self) -> None:
        """Test parsing entry without IPA."""
        raw = {"word": "test", "pos": "noun", "senses": []}
        entry = DictionaryEntry.from_kaikki(raw)
        assert entry.ipa is None

    def test_from_kaikki_ipa_from_multiple_sounds(self) -> None:
        """Test extracting IPA from first sound with ipa field."""
        raw = {
            "word": "test",
            "pos": "noun",
            "sounds": [
                {"audio": "file.mp3"},  # No IPA
                {"ipa": "/tɛst/"},
                {"ipa": "/tɛːst/"},  # Should use first IPA
            ],
            "senses": [],
        }
        entry = DictionaryEntry.from_kaikki(raw)
        assert entry.ipa == "/tɛst/"

    def test_from_kaikki_no_etymology(self) -> None:
        """Test parsing entry without etymology."""
        raw = {"word": "test", "pos": "noun", "senses": []}
        entry = DictionaryEntry.from_kaikki(raw)
        assert entry.etymology is None

    def test_from_kaikki_missing_fields(self) -> None:
        """Test parsing entry with missing fields."""
        raw: dict[str, Any] = {}
        entry = DictionaryEntry.from_kaikki(raw)
        assert entry.word == ""
        assert entry.pos == ""
        assert entry.ipa is None
        assert entry.etymology is None
        assert entry.senses == []


class TestSpacyToKaikkiMapping:
    """Tests for SPACY_TO_KAIKKI mapping."""

    def test_common_pos_mapped(self) -> None:
        """Test that common POS tags are mapped."""
        assert SPACY_TO_KAIKKI["NOUN"] == ["noun"]
        assert SPACY_TO_KAIKKI["VERB"] == ["verb"]
        assert SPACY_TO_KAIKKI["ADJ"] == ["adj"]
        assert SPACY_TO_KAIKKI["ADV"] == ["adv"]

    def test_adp_maps_to_multiple(self) -> None:
        """Test that ADP maps to multiple kaikki tags."""
        assert "prep" in SPACY_TO_KAIKKI["ADP"]
        assert "postp" in SPACY_TO_KAIKKI["ADP"]

    def test_conj_variants_map_same(self) -> None:
        """Test that CONJ, CCONJ, SCONJ all map to conj."""
        assert SPACY_TO_KAIKKI["CONJ"] == ["conj"]
        assert SPACY_TO_KAIKKI["CCONJ"] == ["conj"]
        assert SPACY_TO_KAIKKI["SCONJ"] == ["conj"]


class TestDictionaryInit:
    """Tests for Dictionary initialization."""

    def test_supported_languages(self) -> None:
        """Test that supported_languages returns all available languages."""
        supported = Dictionary.supported_languages()
        assert "fr" in supported
        assert "de" in supported
        assert "es" in supported
        assert "it" in supported
        assert "pt" in supported

    def test_unsupported_language_raises(self, tmp_path: Path) -> None:
        """Test that unsupported language raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported language: xx"):
            Dictionary("xx", cache_dir=tmp_path)

    def test_default_cache_dir(self) -> None:
        """Test that default cache dir is used when not specified."""
        assert Path.home() / ".cache" / "vocab" == DEFAULT_CACHE_DIR

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test that custom cache dir is used and created."""
        cache_dir = tmp_path / "custom_cache"
        assert not cache_dir.exists()

        dictionary = Dictionary("fr", cache_dir=cache_dir)

        assert cache_dir.exists()
        assert dictionary._cache_dir == cache_dir


class TestDictionaryDownload:
    """Tests for Dictionary download functionality."""

    def test_download_on_first_access(self, tmp_path: Path) -> None:
        """Test that data is downloaded on first access."""
        jsonl_content = create_sample_jsonl(SAMPLE_ENTRIES)

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            dictionary = Dictionary("fr", cache_dir=tmp_path)
            result = dictionary.lookup("chien")

            assert len(result) == 1
            assert result[0].word == "chien"
            mock_stream.assert_called_once()

    def test_uses_cache_on_second_access(self, tmp_path: Path) -> None:
        """Test that cached data is used on subsequent accesses."""
        jsonl_content = create_sample_jsonl(SAMPLE_ENTRIES)

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            # First access - downloads
            dictionary1 = Dictionary("fr", cache_dir=tmp_path)
            dictionary1.lookup("chien")

            # Second access - uses cache
            dictionary2 = Dictionary("fr", cache_dir=tmp_path)
            result = dictionary2.lookup("chien")

            assert len(result) == 1
            # Should only have downloaded once
            assert mock_stream.call_count == 1


class TestDictionaryLookup:
    """Tests for Dictionary lookup functionality."""

    @pytest.fixture
    def dictionary(self, tmp_path: Path) -> Dictionary:
        """Create a dictionary with sample data."""
        jsonl_content = create_sample_jsonl(SAMPLE_ENTRIES)

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response
            dictionary = Dictionary("fr", cache_dir=tmp_path)
            # Force load
            dictionary._ensure_loaded()
            return dictionary

    def test_lookup_returns_list(self, dictionary: Dictionary) -> None:
        """Test lookup returns a list of entries."""
        result = dictionary.lookup("chien")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_lookup_known_word(self, dictionary: Dictionary) -> None:
        """Test lookup of a known word."""
        result = dictionary.lookup("chien")

        assert len(result) == 1
        entry = result[0]
        assert entry.word == "chien"
        assert entry.pos == "noun"
        assert entry.ipa == "/ʃjɛ̃/"
        assert entry.etymology == "From Latin canis."
        assert len(entry.senses) == 2
        assert entry.senses[0].translation == "dog"
        assert entry.senses[0].example is not None
        assert entry.senses[0].example.text == "Le chien aboie."

    def test_lookup_unknown_word(self, dictionary: Dictionary) -> None:
        """Test lookup of an unknown word returns empty list."""
        result = dictionary.lookup("xyznonexistent")
        assert result == []

    def test_lookup_word_with_multiple_entries(self, dictionary: Dictionary) -> None:
        """Test lookup of word with multiple kaikki entries."""
        result = dictionary.lookup("faux")

        # Should have 3 entries (2 nouns, 1 adj)
        assert len(result) == 3
        pos_counts = {"noun": 0, "adj": 0}
        for entry in result:
            pos_counts[entry.pos] = pos_counts.get(entry.pos, 0) + 1
        assert pos_counts["noun"] == 2
        assert pos_counts["adj"] == 1

    def test_lookup_with_pos_filter(self, dictionary: Dictionary) -> None:
        """Test lookup with POS filter."""
        result = dictionary.lookup("faux", pos=["noun"])

        assert len(result) == 2
        for entry in result:
            assert entry.pos == "noun"

    def test_lookup_with_pos_filter_multiple_pos(self, dictionary: Dictionary) -> None:
        """Test lookup with multiple POS in filter."""
        result = dictionary.lookup("faux", pos=["noun", "adj"])

        assert len(result) == 3

    def test_lookup_with_pos_filter_no_match(self, dictionary: Dictionary) -> None:
        """Test lookup with POS filter that matches nothing."""
        result = dictionary.lookup("faux", pos=["verb"])

        assert result == []

    def test_lookup_verb(self, dictionary: Dictionary) -> None:
        """Test lookup of a verb."""
        result = dictionary.lookup("manger")

        assert len(result) == 1
        entry = result[0]
        assert entry.pos == "verb"
        assert entry.ipa == "/mɑ̃.ʒe/"
        assert entry.senses[0].translation == "to eat"
        # Example uses 'english' field fallback
        assert entry.senses[0].example is not None
        assert entry.senses[0].example.translation == "I eat an apple."

    def test_lookup_word_without_ipa(self, dictionary: Dictionary) -> None:
        """Test lookup of a word without IPA."""
        result = dictionary.lookup("hormis")

        assert len(result) == 1
        assert result[0].ipa is None
        assert result[0].pos == "prep"


class TestDictionaryEdgeCases:
    """Tests for edge cases in dictionary functionality."""

    def test_empty_jsonl_line(self, tmp_path: Path) -> None:
        """Test that empty lines in JSONL are handled."""
        jsonl_content = "\n".join(
            [
                json.dumps(SAMPLE_ENTRIES[0]),
                "",  # Empty line
                json.dumps(SAMPLE_ENTRIES[1]),
            ]
        )

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            dictionary = Dictionary("fr", cache_dir=tmp_path)
            result = dictionary.lookup("chien")

            assert len(result) == 1

    def test_malformed_json_line_skipped(self, tmp_path: Path) -> None:
        """Test that malformed JSON lines are skipped."""
        jsonl_content = "\n".join(
            [
                json.dumps(SAMPLE_ENTRIES[0]),
                "not valid json {{{",
                json.dumps(SAMPLE_ENTRIES[1]),
            ]
        )

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            dictionary = Dictionary("fr", cache_dir=tmp_path)
            # Should still work, skipping the bad line
            result = dictionary.lookup("chien")

            assert len(result) == 1

    def test_language_property(self, tmp_path: Path) -> None:
        """Test the language property."""
        dictionary = Dictionary("fr", cache_dir=tmp_path)
        assert dictionary.language == "fr"


class TestKaikkiUrls:
    """Tests for KAIKKI_URLS configuration."""

    def test_all_urls_are_valid(self) -> None:
        """Test that all URLs in KAIKKI_URLS are properly formatted."""
        for lang, url in KAIKKI_URLS.items():
            assert url.startswith("https://kaikki.org/dictionary/")
            assert url.endswith(".jsonl")
            assert lang in Dictionary.supported_languages()


class TestDictionaryIndexInvalidation:
    """Tests for index cache invalidation."""

    def test_rebuilds_index_when_older_than_cache(self, tmp_path: Path) -> None:
        """Test that stale index is rebuilt when cache is newer."""
        jsonl_content = create_sample_jsonl(SAMPLE_ENTRIES)

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            # First access - creates cache and index
            dict1 = Dictionary("fr", cache_dir=tmp_path)
            dict1._ensure_loaded()

            # Make cache newer than index
            cache_path = tmp_path / "kaikki-fr.jsonl"
            cache_path.touch()

            # Second instance should rebuild index
            dict2 = Dictionary("fr", cache_dir=tmp_path)
            dict2._ensure_loaded()

            assert dict2.lookup("chien")

    def test_rebuilds_index_when_corrupted(self, tmp_path: Path) -> None:
        """Test that corrupted index triggers rebuild."""
        jsonl_content = create_sample_jsonl(SAMPLE_ENTRIES)

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            # First access - creates cache and index
            dict1 = Dictionary("fr", cache_dir=tmp_path)
            dict1._ensure_loaded()

            # Corrupt the index
            index_path = tmp_path / "kaikki-fr-index.json"
            index_path.write_text("not valid json {{{")

            # Second instance should rebuild
            dict2 = Dictionary("fr", cache_dir=tmp_path)
            dict2._ensure_loaded()

            assert dict2.lookup("chien")
