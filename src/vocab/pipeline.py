"""Pipeline for generating Anki flashcards from vocabulary."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from vocab.dictionary import LANGUAGE_NAMES, SPACY_TO_KAIKKI, Dictionary, DictionaryEntry
from vocab.models import LemmaEntry

logger = logging.getLogger(__name__)

# Model name mapping
MODEL_MAPPING: dict[str, str] = {
    "claude-haiku": "claude-3-5-haiku-latest",
    "claude-sonnet": "claude-sonnet-4-20250514",
}


@dataclass
class EnrichedLemma:
    """A lemma enriched with dictionary data.

    Attributes:
        lemma: The LemmaEntry from the vocabulary.
        words: List of matching dictionary entries (invariant: len >= 1).
    """

    lemma: LemmaEntry
    words: list[DictionaryEntry]

    def __post_init__(self) -> None:
        """Validate invariants."""
        if not self.words:
            raise ValueError("words must have at least one entry")
        if any(not word.senses for word in self.words):
            raise ValueError("all words must have at least one sense")


@dataclass
class SenseAssignment:
    """A sense assignment mapping examples to a specific dictionary sense.

    Attributes:
        lemma: The LemmaEntry from the vocabulary.
        examples: Indices into lemma.examples[].
        word: The DictionaryEntry containing the matched sense.
        sense: Index into word.senses[].
    """

    lemma: LemmaEntry
    examples: list[int]
    word: DictionaryEntry
    sense: int

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.sense < 0 or self.sense >= len(self.word.senses):
            raise ValueError(f"sense index {self.sense} out of bounds")
        for idx in self.examples:
            if idx < 0 or idx >= len(self.lemma.examples):
                raise ValueError(f"example index {idx} out of bounds")


class SentenceAssignment(BaseModel):
    """Assignment of a sentence to a sense."""

    sentence: int  # 1-indexed sentence number
    sense: int | None  # 1-indexed sense number, None if unknown


class DisambiguationResponse(BaseModel):
    """LLM response for sense disambiguation."""

    assignments: list[SentenceAssignment]


def enrich_lemma(
    lemma_entry: LemmaEntry,
    dictionary: Dictionary,
) -> EnrichedLemma | None:
    """Enrich a single lemma with dictionary data.

    Looks up matching dictionary entries by (word, POS).

    Args:
        lemma_entry: The lemma entry to enrich.
        dictionary: Dictionary for lookups.

    Returns:
        EnrichedLemma if dictionary matches exist, None otherwise.
    """
    kaikki_pos = SPACY_TO_KAIKKI.get(lemma_entry.pos, [])
    words = dictionary.lookup(lemma_entry.lemma, pos=kaikki_pos or None)
    if words:
        return EnrichedLemma(lemma=lemma_entry, words=words)
    return None


def needs_disambiguation(entry: EnrichedLemma) -> bool:
    """Return True if LLM disambiguation is needed.

    Disambiguation is needed when there are multiple words or
    any word has multiple senses.

    Args:
        entry: EnrichedLemma to check.

    Returns:
        True if disambiguation is needed, False otherwise.
    """
    if len(entry.words) > 1:
        return True
    return len(entry.words[0].senses) > 1


def assign_single_sense(entry: EnrichedLemma) -> SenseAssignment:
    """Assign all examples to the single available sense.

    Args:
        entry: EnrichedLemma with exactly one word and one sense.

    Returns:
        SenseAssignment with all example indices.

    Raises:
        AssertionError: If entry has multiple words or senses.
    """
    assert not needs_disambiguation(entry), "Use disambiguate_senses() for this entry"
    return SenseAssignment(
        lemma=entry.lemma,
        examples=list(range(len(entry.lemma.examples))),
        word=entry.words[0],
        sense=0,
    )


def _build_prompt(entry: EnrichedLemma, language: str) -> str:
    """Build the disambiguation prompt for the LLM.

    Args:
        entry: EnrichedLemma with multiple words or senses.
        language: Full language name (e.g., "French").

    Returns:
        Formatted prompt string.
    """
    # Build senses list
    senses_lines: list[str] = []
    sense_idx = 1
    for word in entry.words:
        for sense in word.senses:
            etymology = word.etymology or "unknown"
            senses_lines.append(
                f"{sense_idx}. [word={word.word}, etymology={etymology}] {sense.translation}"
            )
            sense_idx += 1

    # Build sentences list
    sentences_lines: list[str] = []
    for i, example in enumerate(entry.lemma.examples, 1):
        sentences_lines.append(f"{i}. {example.sentence}")

    return f"""You are helping associate sentences with word meanings.

Language: {language}
Word: {entry.lemma.lemma}

Available senses:
{chr(10).join(senses_lines)}

Sentences from the source text:
{chr(10).join(sentences_lines)}

For each sentence, indicate which sense is being used.
If you cannot confidently determine the sense, use null for the sense."""


def _parse_llm_response(
    response: DisambiguationResponse,
    entry: EnrichedLemma,
) -> list[SenseAssignment]:
    """Parse LLM response into SenseAssignments.

    Args:
        response: Parsed LLM response.
        entry: EnrichedLemma being disambiguated.

    Returns:
        List of SenseAssignments, one per unique (word, sense) used.
    """
    # Build flat list of (word_idx, word, sense_idx) for 1-indexed sense numbers
    sense_map: list[tuple[int, DictionaryEntry, int]] = []
    for word_idx, word in enumerate(entry.words):
        for sense_idx in range(len(word.senses)):
            sense_map.append((word_idx, word, sense_idx))

    # Group examples by (word, sense)
    assignments_by_sense: dict[tuple[int, int], list[int]] = {}

    for assignment in response.assignments:
        if assignment.sense is None:
            logger.warning(
                "Could not disambiguate sentence %d for %s",
                assignment.sentence,
                entry.lemma.lemma,
            )
            continue

        # Convert 1-indexed to 0-indexed
        sentence_idx = assignment.sentence - 1
        sense_idx = assignment.sense - 1

        if sense_idx < 0 or sense_idx >= len(sense_map):
            logger.warning(
                "Invalid sense %d for %s (max %d)",
                assignment.sense,
                entry.lemma.lemma,
                len(sense_map),
            )
            continue

        if sentence_idx < 0 or sentence_idx >= len(entry.lemma.examples):
            logger.warning(
                "Invalid sentence %d for %s (max %d)",
                assignment.sentence,
                entry.lemma.lemma,
                len(entry.lemma.examples),
            )
            continue

        # Find which word this sense belongs to
        word_idx, _, local_sense_idx = sense_map[sense_idx]

        key = (word_idx, local_sense_idx)
        if key not in assignments_by_sense:
            assignments_by_sense[key] = []
        assignments_by_sense[key].append(sentence_idx)

    # Convert to SenseAssignments
    results: list[SenseAssignment] = []
    for (word_idx, sense_idx), examples in assignments_by_sense.items():
        results.append(
            SenseAssignment(
                lemma=entry.lemma,
                examples=examples,
                word=entry.words[word_idx],
                sense=sense_idx,
            )
        )

    return results


async def disambiguate_senses(
    entry: EnrichedLemma,
    *,
    language: str,
    model: str = "claude-haiku",
) -> list[SenseAssignment]:
    """Use LLM to assign examples to senses.

    This function's signature is provider-agnostic: domain types in, domain
    types out. The LLM provider (currently Anthropic) is an implementation
    detail; switching providers requires only changing this function's internals.

    Args:
        entry: EnrichedLemma with multiple words or senses.
        language: Two-letter language code (e.g., "fr").
        model: Model identifier ("claude-haiku" or "claude-sonnet").

    Returns:
        List of SenseAssignments, one per unique (word, sense) used.
        Examples that couldn't be assigned are logged and omitted.

    Raises:
        AssertionError: If entry has only one sense.
        ValueError: If language code is not recognized.
    """
    assert needs_disambiguation(entry), "Use assign_single_sense() for this entry"

    # Resolve language name from code
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unknown language code: {language}")
    language_name = LANGUAGE_NAMES[language]

    # Build prompt
    prompt = _build_prompt(entry, language_name)

    # Call LLM
    model_id = MODEL_MAPPING.get(model, model)
    client = anthropic.AsyncAnthropic()

    message = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "name": "submit_assignments",
                "description": "Submit the sense assignments for each sentence",
                "input_schema": DisambiguationResponse.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "submit_assignments"},
    )

    # Extract tool call result
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_assignments":
            response = DisambiguationResponse.model_validate(block.input)
            return _parse_llm_response(response, entry)

    logger.warning("No tool call in LLM response for %s", entry.lemma.lemma)
    return []
