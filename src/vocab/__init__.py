"""Vocabulary extraction from ePub files."""

from vocab.epub import extract_chapters
from vocab.models import Chapter

__all__ = ["Chapter", "extract_chapters"]
