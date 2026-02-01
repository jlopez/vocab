"""Tests for the dictionary module."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vocab.dictionary import (
    DEFAULT_CACHE_DIR,
    KAIKKI_URLS,
    Dictionary,
    DictionaryEntry,
)

# Sample Wiktionary entries for testing
SAMPLE_ENTRIES: list[dict[str, Any]] = [
    {
        "word": "chien",
        "pos": "noun",
        "sounds": [{"ipa": "/ʃjɛ̃/"}],
        "tags": ["masculine"],
        "senses": [
            {
                "glosses": ["dog", "hound"],
                "translations": [
                    {"lang": "Spanish", "code": "es", "word": "perro"},
                    {"lang": "Spanish", "code": "es", "word": "can"},
                    {"lang": "Italian", "code": "it", "word": "cane"},
                ],
            },
            {
                "glosses": ["hammer (of a firearm)"],
                "translations": [
                    {"lang": "Spanish", "code": "es", "word": "martillo"},
                ],
            },
        ],
    },
    {
        "word": "manger",
        "pos": "verb",
        "sounds": [{"ipa": "/mɑ̃.ʒe/"}],
        "senses": [
            {
                "glosses": ["to eat"],
                "translations": [
                    {"lang": "Spanish", "code": "es", "word": "comer"},
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
                "glosses": ["beautiful", "handsome", "fine"],
                "translations": [
                    {"lang": "Spanish", "code": "es", "word": "hermoso"},
                    {"lang": "Spanish", "code": "es", "word": "bello"},
                ],
            },
        ],
    },
    # Entry with gender in head_templates
    {
        "word": "maison",
        "pos": "noun",
        "sounds": [{"ipa": "/mɛ.zɔ̃/"}],
        "head_templates": [{"args": {"g": "f"}}],
        "senses": [
            {
                "glosses": ["house", "home"],
                "translations": [
                    {"lang": "Spanish", "code": "es", "word": "casa"},
                ],
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
]


def create_sample_jsonl(entries: list[dict[str, Any]]) -> str:
    """Create a JSONL string from entries."""
    return "\n".join(json.dumps(entry) for entry in entries)


class TestDictionaryEntry:
    """Tests for DictionaryEntry dataclass."""

    def test_creation(self) -> None:
        """Test basic creation of DictionaryEntry."""
        entry = DictionaryEntry(
            lemma="test",
            language="fr",
            ipa="/tɛst/",
            gender="m",
            pos="noun",
            translations_en=["test", "trial"],
            target_language="es",
            target_translations=["prueba", "ensayo"],
        )
        assert entry.lemma == "test"
        assert entry.language == "fr"
        assert entry.ipa == "/tɛst/"
        assert entry.gender == "m"
        assert entry.pos == "noun"
        assert entry.translations_en == ["test", "trial"]
        assert entry.target_language == "es"
        assert entry.target_translations == ["prueba", "ensayo"]

    def test_creation_without_optional_fields(self) -> None:
        """Test creation with None values for optional fields."""
        entry = DictionaryEntry(
            lemma="test",
            language="fr",
            ipa=None,
            gender=None,
            pos=None,
            translations_en=[],
            target_language=None,
            target_translations=[],
        )
        assert entry.ipa is None
        assert entry.gender is None
        assert entry.pos is None


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

        # Just init, don't load data
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

            assert result is not None
            assert result.lemma == "chien"
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

            assert result is not None
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

    def test_lookup_known_word(self, dictionary: Dictionary) -> None:
        """Test lookup of a known word."""
        result = dictionary.lookup("chien")

        assert result is not None
        assert result.lemma == "chien"
        assert result.language == "fr"
        assert result.ipa == "/ʃjɛ̃/"
        assert result.gender == "m"
        assert result.pos == "noun"
        assert "dog" in result.translations_en
        assert "hound" in result.translations_en

    def test_lookup_unknown_word(self, dictionary: Dictionary) -> None:
        """Test lookup of an unknown word returns None."""
        result = dictionary.lookup("xyznonexistent")
        assert result is None

    def test_lookup_with_target_language(self, dictionary: Dictionary) -> None:
        """Test lookup with target language translations."""
        result = dictionary.lookup("chien", target_language="es")

        assert result is not None
        assert result.target_language == "es"
        assert "perro" in result.target_translations
        assert "can" in result.target_translations

    def test_lookup_without_target_language(self, dictionary: Dictionary) -> None:
        """Test lookup without target language has empty target_translations."""
        result = dictionary.lookup("chien")

        assert result is not None
        assert result.target_language is None
        assert result.target_translations == []

    def test_lookup_verb(self, dictionary: Dictionary) -> None:
        """Test lookup of a verb."""
        result = dictionary.lookup("manger")

        assert result is not None
        assert result.pos == "verb"
        assert result.gender is None
        assert result.ipa == "/mɑ̃.ʒe/"
        assert "to eat" in result.translations_en

    def test_lookup_adjective(self, dictionary: Dictionary) -> None:
        """Test lookup of an adjective."""
        result = dictionary.lookup("beau")

        assert result is not None
        assert result.pos == "adj"
        assert result.gender is None
        assert "beautiful" in result.translations_en

    def test_lookup_feminine_noun(self, dictionary: Dictionary) -> None:
        """Test lookup of a feminine noun."""
        result = dictionary.lookup("maison")

        assert result is not None
        assert result.gender == "f"
        assert result.pos == "noun"

    def test_lookup_word_without_ipa(self, dictionary: Dictionary) -> None:
        """Test lookup of a word without IPA."""
        result = dictionary.lookup("hormis")

        assert result is not None
        assert result.ipa is None
        assert result.pos == "prep"


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

            assert result is not None

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

            assert result is not None

    def test_multiple_pos_entries(self, tmp_path: Path) -> None:
        """Test handling of words with multiple parts of speech."""
        entries = [
            {
                "word": "test",
                "pos": "adv",
                "senses": [{"glosses": ["as adverb"]}],
            },
            {
                "word": "test",
                "pos": "noun",
                "senses": [{"glosses": ["as noun"]}],
            },
        ]
        jsonl_content = create_sample_jsonl(entries)

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(jsonl_content))}
        mock_response.iter_bytes = lambda chunk_size: [jsonl_content.encode()]

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            dictionary = Dictionary("fr", cache_dir=tmp_path)
            result = dictionary.lookup("test")

            assert result is not None
            # Should prefer noun over adv based on priority
            assert result.pos == "noun"
            # Should have both glosses
            assert "as noun" in result.translations_en
            assert "as adverb" in result.translations_en

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
