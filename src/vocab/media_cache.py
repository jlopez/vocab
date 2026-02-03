"""Media file downloading and caching."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vocab" / "media"
DEFAULT_TIMEOUT = 30.0
CHUNK_SIZE = 8192
USER_AGENT = "vocab/1.0 (https://github.com/jlopez/vocab)"
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds


async def fetch_media(
    url: str,
    filename: str,
    cache_dir: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path | None:
    """Download and cache a media file.

    Uses a tiered directory structure ({cache_dir}/{filename[:2]}/{filename})
    to avoid filesystem slowdown with many files.

    Args:
        url: URL to download from.
        filename: Filename to save as (e.g., "abc123.mp3"). Must not be empty.
        cache_dir: Cache directory. Defaults to ~/.cache/vocab/media/
        timeout: Request timeout in seconds. Defaults to 30.

    Returns:
        Path to the cached file, or None if the file was not found (404).

    Raises:
        httpx.HTTPStatusError: If download fails with non-2xx status (except 404).
        httpx.RequestError: If network request fails.
        ValueError: If filename is empty.
    """
    if not filename:
        raise ValueError("filename must not be empty")

    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Tiered path: first 2 chars of filename as subdirectory
    tier = filename[:2] if len(filename) >= 2 else filename
    file_path = cache_dir / tier / filename

    # Return immediately if cached
    if file_path.exists():
        return file_path

    # Download and cache atomically
    file_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        found = await _download_with_retry(client, url, file_path)

    return file_path if found else None


async def _download_with_retry(client: httpx.AsyncClient, url: str, file_path: Path) -> bool:
    """Download a file with retry logic for rate limiting.

    Retries with exponential backoff on 429 errors, respecting Retry-After header.

    Args:
        client: httpx async client to use.
        url: URL to download from.
        file_path: Destination path for the file.

    Returns:
        True if download succeeded, False if file not found (404).

    Raises:
        httpx.HTTPStatusError: If download fails with non-2xx status (except 404).
    """
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()

                # Write to temp file, then atomically rename
                with tempfile.NamedTemporaryFile(dir=file_path.parent, delete=False) as tmp:
                    async for chunk in response.aiter_bytes(CHUNK_SIZE):
                        tmp.write(chunk)
                    tmp_path = Path(tmp.name)

                tmp_path.rename(file_path)
                return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            if e.response.status_code != 429 or attempt == MAX_RETRIES:
                raise

            # Get retry delay from header or use exponential backoff
            retry_after = e.response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = backoff
                backoff *= 2  # Exponential backoff

            logger.debug("Rate limited, retrying in %.1fs (attempt %d)", delay, attempt + 1)
            await asyncio.sleep(delay)

    return False  # Should not reach here, but satisfies type checker
