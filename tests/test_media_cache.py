"""Tests for the media_cache module."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vocab.media_cache import (
    CHUNK_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_TIMEOUT,
    USER_AGENT,
    fetch_media,
)


def _make_mock_client(content: bytes) -> MagicMock:
    """Create a mock httpx client with streaming response.

    Returns a MagicMock that simulates httpx.AsyncClient with a
    stream() method returning an async context manager.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    # Create async iterator for aiter_bytes
    async def aiter_bytes(chunk_size: int = CHUNK_SIZE) -> AsyncIterator[bytes]:
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    mock_response.aiter_bytes = aiter_bytes

    mock_client = MagicMock()

    @asynccontextmanager
    async def mock_stream(method: str, url: str, **kwargs: object) -> AsyncIterator[MagicMock]:
        yield mock_response

    mock_client.stream = mock_stream
    mock_client._response = mock_response  # For test assertions
    return mock_client


class TestFetchMedia:
    """Tests for fetch_media function."""

    async def test_downloads_and_caches_file(self, tmp_path: Path) -> None:
        """Test successful download and caching."""
        mock_client = _make_mock_client(b"fake audio content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await fetch_media(
                "https://example.com/audio.mp3",
                "abc123.mp3",
                cache_dir=tmp_path,
            )

            assert result == tmp_path / "ab" / "abc123.mp3"
            assert result.exists()
            assert result.read_bytes() == b"fake audio content"

    async def test_cache_hit_skips_download(self, tmp_path: Path) -> None:
        """Test that cached files are returned without network request."""
        # Pre-create cached file
        cached_path = tmp_path / "ab" / "abc123.mp3"
        cached_path.parent.mkdir(parents=True)
        cached_path.write_bytes(b"cached content")

        with patch("httpx.AsyncClient") as mock_client_class:
            result = await fetch_media(
                "https://example.com/audio.mp3",
                "abc123.mp3",
                cache_dir=tmp_path,
            )

            assert result == cached_path
            assert result.read_bytes() == b"cached content"
            # Should not have made any network request
            mock_client_class.assert_not_called()

    async def test_tiered_directory_structure(self, tmp_path: Path) -> None:
        """Test that files are stored in tiered directories."""
        mock_client = _make_mock_client(b"content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await fetch_media(
                "https://example.com/audio.mp3",
                "xyz789.mp3",
                cache_dir=tmp_path,
            )

            # Should be in xy/ subdirectory
            assert result == tmp_path / "xy" / "xyz789.mp3"
            assert result.parent.name == "xy"

    async def test_short_filename_tiering(self, tmp_path: Path) -> None:
        """Test tiering with single-character filename prefix."""
        mock_client = _make_mock_client(b"content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await fetch_media(
                "https://example.com/x",
                "x",
                cache_dir=tmp_path,
            )

            # Single char filename uses itself as tier
            assert result == tmp_path / "x" / "x"

    async def test_404_returns_none(self, tmp_path: Path) -> None:
        """Test that 404 returns None instead of raising."""
        mock_client = MagicMock()
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404

        @asynccontextmanager
        async def mock_stream_error(
            method: str, url: str, **kwargs: object
        ) -> AsyncIterator[MagicMock]:
            raise httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", url),
                response=mock_response_404,
            )
            yield

        mock_client.stream = mock_stream_error

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await fetch_media(
                "https://example.com/missing.mp3",
                "missing.mp3",
                cache_dir=tmp_path,
            )

            assert result is None

    async def test_server_error_raises(self, tmp_path: Path) -> None:
        """Test that server errors (5xx) still raise."""
        mock_client = MagicMock()
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500

        @asynccontextmanager
        async def mock_stream_error(
            method: str, url: str, **kwargs: object
        ) -> AsyncIterator[MagicMock]:
            raise httpx.HTTPStatusError(
                "Internal Server Error",
                request=httpx.Request("GET", url),
                response=mock_response_500,
            )
            yield

        mock_client.stream = mock_stream_error

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await fetch_media(
                    "https://example.com/error.mp3",
                    "error.mp3",
                    cache_dir=tmp_path,
                )

    async def test_network_error_raises(self, tmp_path: Path) -> None:
        """Test that network errors are propagated."""
        mock_client = MagicMock()

        @asynccontextmanager
        async def mock_stream_network_error(
            method: str, url: str, **kwargs: object
        ) -> AsyncIterator[MagicMock]:
            raise httpx.ConnectError("Connection refused")
            yield  # pragma: no cover

        mock_client.stream = mock_stream_network_error

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await fetch_media(
                    "https://example.com/audio.mp3",
                    "audio.mp3",
                    cache_dir=tmp_path,
                )

    async def test_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test using custom cache directory."""
        custom_cache = tmp_path / "custom" / "cache"
        mock_client = _make_mock_client(b"content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await fetch_media(
                "https://example.com/audio.mp3",
                "test.mp3",
                cache_dir=custom_cache,
            )

            assert result is not None
            assert result.is_relative_to(custom_cache)
            assert result.exists()

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created as needed."""
        deep_cache = tmp_path / "deep" / "nested" / "cache"
        assert not deep_cache.exists()

        mock_client = _make_mock_client(b"content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await fetch_media(
                "https://example.com/audio.mp3",
                "file.mp3",
                cache_dir=deep_cache,
            )

            assert result is not None
            assert result.exists()
            assert deep_cache.exists()

    async def test_empty_filename_raises(self, tmp_path: Path) -> None:
        """Test that empty filename raises ValueError."""
        with pytest.raises(ValueError, match="filename must not be empty"):
            await fetch_media(
                "https://example.com/audio.mp3",
                "",
                cache_dir=tmp_path,
            )

    async def test_custom_timeout(self, tmp_path: Path) -> None:
        """Test that custom timeout is passed to httpx client."""
        mock_client = _make_mock_client(b"content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await fetch_media(
                "https://example.com/audio.mp3",
                "test.mp3",
                cache_dir=tmp_path,
                timeout=60.0,
            )

            mock_client_class.assert_called_once_with(
                timeout=60.0, headers={"User-Agent": USER_AGENT}
            )

    async def test_default_timeout(self, tmp_path: Path) -> None:
        """Test that default timeout is used when not specified."""
        mock_client = _make_mock_client(b"content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await fetch_media(
                "https://example.com/audio.mp3",
                "test.mp3",
                cache_dir=tmp_path,
            )

            mock_client_class.assert_called_once_with(
                timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}
            )


class TestDefaultCacheDir:
    """Tests for DEFAULT_CACHE_DIR constant."""

    def test_default_cache_dir_location(self) -> None:
        """Test that default cache dir is in expected location."""
        assert Path.home() / ".cache" / "vocab" / "media" == DEFAULT_CACHE_DIR


class TestRetryBehavior:
    """Tests for 429 retry behavior."""

    async def test_retries_on_429_then_succeeds(self, tmp_path: Path) -> None:
        """Test that 429 errors trigger retry and eventual success."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {}

        call_count = 0

        async def aiter_bytes(chunk_size: int = CHUNK_SIZE) -> AsyncIterator[bytes]:
            yield b"audio content"

        @asynccontextmanager
        async def mock_stream(method: str, url: str, **kwargs: object) -> AsyncIterator[MagicMock]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response_429)
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_bytes = aiter_bytes
            yield mock_response

        mock_client = MagicMock()
        mock_client.stream = mock_stream

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("vocab.media_cache.asyncio.sleep") as mock_sleep,
        ):
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await fetch_media(
                "https://example.com/audio.mp3",
                "test.mp3",
                cache_dir=tmp_path,
            )

            assert result is not None
            assert result.exists()
            assert call_count == 2
            mock_sleep.assert_called_once()

    async def test_respects_retry_after_header(self, tmp_path: Path) -> None:
        """Test that Retry-After header value is used for delay."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "5"}

        call_count = 0

        async def aiter_bytes(chunk_size: int = CHUNK_SIZE) -> AsyncIterator[bytes]:
            yield b"content"

        @asynccontextmanager
        async def mock_stream(method: str, url: str, **kwargs: object) -> AsyncIterator[MagicMock]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response_429)
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_bytes = aiter_bytes
            yield mock_response

        mock_client = MagicMock()
        mock_client.stream = mock_stream

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("vocab.media_cache.asyncio.sleep") as mock_sleep,
        ):
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await fetch_media(
                "https://example.com/audio.mp3",
                "test.mp3",
                cache_dir=tmp_path,
            )

            # Should use Retry-After value of 5 seconds
            mock_sleep.assert_called_once_with(5.0)

    async def test_gives_up_after_max_retries(self, tmp_path: Path) -> None:
        """Test that error is raised after max retries exhausted."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {}

        @asynccontextmanager
        async def mock_stream(method: str, url: str, **kwargs: object) -> AsyncIterator[MagicMock]:
            raise httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response_429)
            yield

        mock_client = MagicMock()
        mock_client.stream = mock_stream

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("vocab.media_cache.asyncio.sleep"),
        ):
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await fetch_media(
                    "https://example.com/audio.mp3",
                    "test.mp3",
                    cache_dir=tmp_path,
                )
