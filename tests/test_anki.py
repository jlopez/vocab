"""Tests for the anki module."""

import zipfile
from pathlib import Path

from vocab.anki import (
    AnkiDeckBuilder,
    _format_example_translation,
    _format_examples,
    _format_forms,
    _generate_deck_id,
    _generate_model_id,
    _highlight_word,
)
from vocab.dictionary import DictionaryEntry, DictionaryExample, DictionarySense
from vocab.models import Example, LemmaEntry, SentenceLocation
from vocab.pipeline import SenseAssignment


def make_sense_assignment(
    word: str = "chien",
    pos: str = "noun",
    translation: str = "dog",
    ipa: str = "/ʃjɛ̃/",
    etymology: str | None = None,
    examples: list[str] | None = None,
    example_indices: list[int] | None = None,
    dict_example_text: str | None = None,
    dict_example_translation: str | None = None,
) -> SenseAssignment:
    """Create a SenseAssignment for testing."""
    if examples is None:
        examples = ["Le chien aboie."]
    if example_indices is None:
        example_indices = list(range(len(examples)))

    lemma = LemmaEntry(
        lemma=word,
        pos=pos.upper(),
        frequency=len(examples),
        forms={word: len(examples)},
        examples=[
            Example(
                sentence=s,
                location=SentenceLocation(
                    chapter_index=0,
                    chapter_title="Chapter 1",
                    sentence_index=i,
                ),
            )
            for i, s in enumerate(examples)
        ],
    )

    # Build dictionary example if provided
    dict_example: DictionaryExample | None = None
    if dict_example_text is not None:
        dict_example = DictionaryExample(
            text=dict_example_text,
            translation=dict_example_translation or "",
        )

    dict_entry = DictionaryEntry(
        word=word,
        pos=pos,
        ipa=ipa,
        etymology=etymology,
        senses=[
            DictionarySense(
                id=f"{word}-{pos}-1",
                translation=translation,
                example=dict_example,
            )
        ],
    )

    return SenseAssignment(
        lemma=lemma,
        examples=example_indices,
        word=dict_entry,
        sense=0,
    )


class TestDeterministicIds:
    """Tests for deterministic ID generation."""

    def test_model_id_is_deterministic(self) -> None:
        """Test that model ID is the same for same inputs."""
        id1 = _generate_model_id("Test Deck", "fr")
        id2 = _generate_model_id("Test Deck", "fr")
        assert id1 == id2

    def test_deck_id_is_deterministic(self) -> None:
        """Test that deck ID is the same for same inputs."""
        id1 = _generate_deck_id("Test Deck", "fr")
        id2 = _generate_deck_id("Test Deck", "fr")
        assert id1 == id2

    def test_different_deck_names_produce_different_ids(self) -> None:
        """Test that different deck names produce different IDs."""
        id1 = _generate_model_id("Deck A", "fr")
        id2 = _generate_model_id("Deck B", "fr")
        assert id1 != id2

    def test_different_languages_produce_different_ids(self) -> None:
        """Test that different languages produce different IDs."""
        id1 = _generate_model_id("Test Deck", "fr")
        id2 = _generate_model_id("Test Deck", "es")
        assert id1 != id2

    def test_model_and_deck_ids_are_different(self) -> None:
        """Test that model and deck IDs differ for same inputs."""
        model_id = _generate_model_id("Test Deck", "fr")
        deck_id = _generate_deck_id("Test Deck", "fr")
        assert model_id != deck_id


class TestHighlightWord:
    """Tests for _highlight_word function."""

    def test_highlights_exact_match(self) -> None:
        """Test highlighting exact word match."""
        result = _highlight_word("Le chien aboie.", "chien")
        assert result == "Le <b>chien</b> aboie."

    def test_case_insensitive(self) -> None:
        """Test case-insensitive highlighting."""
        result = _highlight_word("Le Chien aboie.", "chien")
        assert result == "Le <b>Chien</b> aboie."

    def test_multiple_occurrences(self) -> None:
        """Test highlighting multiple occurrences."""
        result = _highlight_word("Un chien voit un autre chien.", "chien")
        assert result == "Un <b>chien</b> voit un autre <b>chien</b>."

    def test_no_match(self) -> None:
        """Test when word is not in sentence."""
        result = _highlight_word("Le chat miaule.", "chien")
        assert result == "Le chat miaule."


class TestHtmlEscaping:
    """Tests for HTML escaping in card fields."""

    def test_format_examples_escapes_html_in_sentence(self) -> None:
        """Test that angle brackets in sentences are escaped."""
        assignment = make_sense_assignment(
            word="test",
            examples=["This has <angle> brackets."],
        )
        result = _format_examples(assignment)
        assert "&lt;angle&gt;" in result
        assert "<angle>" not in result

    def test_format_examples_escapes_html_in_dict_example(self) -> None:
        """Test that angle brackets in dictionary examples are escaped."""
        assignment = make_sense_assignment(
            word="test",
            examples=["Normal sentence."],
            dict_example_text="From <Latin> root.",
        )
        result = _format_examples(assignment)
        assert "&lt;Latin&gt;" in result
        assert "<Latin>" not in result

    def test_format_example_translation_escapes_html(self) -> None:
        """Test that angle brackets in translation are escaped."""
        assignment = make_sense_assignment(
            word="test",
            dict_example_text="Example.",
            dict_example_translation="This means <something>.",
        )
        result = _format_example_translation(assignment)
        assert "&lt;something&gt;" in result
        assert "<something>" not in result

    def test_highlight_preserved_after_escaping(self) -> None:
        """Test that highlighting still works after escaping."""
        assignment = make_sense_assignment(
            word="test",
            examples=["A <weird> test case."],
        )
        result = _format_examples(assignment)
        # Should have both escaping and highlighting
        assert "&lt;weird&gt;" in result
        assert "<b>test</b>" in result


class TestFormatExamples:
    """Tests for _format_examples function."""

    def test_formats_single_example(self) -> None:
        """Test formatting a single example."""
        assignment = make_sense_assignment(examples=["Le chien aboie."])
        result = _format_examples(assignment)
        assert "Le <b>chien</b> aboie." in result
        assert 'class="example"' in result

    def test_formats_multiple_examples(self) -> None:
        """Test formatting multiple examples."""
        assignment = make_sense_assignment(
            examples=["Le chien aboie.", "Mon chien dort."],
            example_indices=[0, 1],
        )
        result = _format_examples(assignment)
        assert "Le <b>chien</b> aboie." in result
        assert "Mon <b>chien</b> dort." in result

    def test_uses_only_specified_indices(self) -> None:
        """Test that only specified example indices are used."""
        assignment = make_sense_assignment(
            examples=["First.", "Second.", "Third."],
            example_indices=[0, 2],  # Skip index 1
            word="test",
        )
        result = _format_examples(assignment)
        assert "First" in result
        assert "Second" not in result
        assert "Third" in result

    def test_empty_examples_returns_empty_string(self) -> None:
        """Test that empty examples list returns empty string."""
        assignment = make_sense_assignment(
            examples=["Unused sentence."],
            example_indices=[],  # No examples selected
        )
        result = _format_examples(assignment)
        assert result == ""

    def test_includes_dictionary_example_first(self) -> None:
        """Test that dictionary example appears before book examples."""
        assignment = make_sense_assignment(
            examples=["Le chien aboie."],
            dict_example_text="Mon chien est gentil.",
        )
        result = _format_examples(assignment)
        # Dictionary example should come first
        dict_pos = result.find("Mon <b>chien</b> est gentil.")
        book_pos = result.find("Le <b>chien</b> aboie.")
        assert dict_pos < book_pos

    def test_dictionary_example_only_when_no_book_examples(self) -> None:
        """Test formatting with only dictionary example."""
        assignment = make_sense_assignment(
            examples=["Unused."],
            example_indices=[],  # No book examples
            word="chien",
            dict_example_text="Le chien dort.",
        )
        result = _format_examples(assignment)
        assert "Le <b>chien</b> dort." in result


class TestFormatExampleTranslation:
    """Tests for _format_example_translation function."""

    def test_returns_empty_when_no_example(self) -> None:
        """Test returns empty string when sense has no example."""
        assignment = make_sense_assignment()
        result = _format_example_translation(assignment)
        assert result == ""

    def test_returns_empty_when_no_translation(self) -> None:
        """Test returns empty when example has no translation."""
        assignment = make_sense_assignment(
            dict_example_text="Le chien dort.",
            dict_example_translation="",
        )
        result = _format_example_translation(assignment)
        assert result == ""

    def test_formats_translation(self) -> None:
        """Test formatting example translation."""
        assignment = make_sense_assignment(
            word="chien",
            dict_example_text="Le chien dort.",
            dict_example_translation="The dog sleeps.",
        )
        result = _format_example_translation(assignment)
        assert "The dog sleeps." in result

    def test_highlights_word_in_translation(self) -> None:
        """Test that source word is highlighted if present in translation."""
        assignment = make_sense_assignment(
            word="table",
            translation="table",
            dict_example_text="La table est grande.",
            dict_example_translation="The table is big.",
        )
        result = _format_example_translation(assignment)
        assert "<b>table</b>" in result


class TestFormatForms:
    """Tests for _format_forms function."""

    def test_formats_single_form(self) -> None:
        """Test formatting a single form."""
        assignment = make_sense_assignment()
        result = _format_forms(assignment)
        assert result == "chien"

    def test_formats_multiple_forms_sorted(self) -> None:
        """Test formatting multiple forms in sorted order."""
        lemma = LemmaEntry(
            lemma="manger",
            pos="VERB",
            frequency=3,
            forms={"mange": 1, "mangeons": 1, "manger": 1},
            examples=[
                Example(
                    sentence="Test",
                    location=SentenceLocation(
                        chapter_index=0, chapter_title=None, sentence_index=0
                    ),
                )
            ],
        )
        dict_entry = DictionaryEntry(
            word="manger",
            pos="verb",
            ipa="/mɑ̃.ʒe/",
            etymology=None,
            senses=[DictionarySense(id="1", translation="to eat", example=None)],
        )
        assignment = SenseAssignment(lemma=lemma, examples=[0], word=dict_entry, sense=0)

        result = _format_forms(assignment)
        assert result == "mange, mangeons, manger"


class TestAnkiDeckBuilder:
    """Tests for AnkiDeckBuilder class."""

    def test_is_context_manager(self, tmp_path: Path) -> None:
        """Test that AnkiDeckBuilder is a context manager."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            assert deck is not None

    def test_writes_apkg_on_exit(self, tmp_path: Path) -> None:
        """Test that .apkg file is written on context exit."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment())

        assert output_path.exists()

    def test_apkg_is_valid_zip(self, tmp_path: Path) -> None:
        """Test that .apkg file is a valid zip archive."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment())

        assert zipfile.is_zipfile(output_path)

    def test_apkg_contains_expected_files(self, tmp_path: Path) -> None:
        """Test that .apkg contains expected Anki files."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment())

        with zipfile.ZipFile(output_path) as zf:
            names = zf.namelist()
            # Anki packages should contain these files
            assert any("collection.anki2" in n for n in names)

    def test_add_increments_cards_count(self, tmp_path: Path) -> None:
        """Test that adding cards increments the count."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            assert deck.cards_added == 0
            deck.add(make_sense_assignment())
            assert deck.cards_added == 1
            deck.add(make_sense_assignment(word="chat", translation="cat"))
            assert deck.cards_added == 2

    def test_card_has_translation_on_front(self, tmp_path: Path) -> None:
        """Test that card front contains the translation."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment(translation="dog"))
            # Check that the model has the expected template
            assert "{{Translation}}" in deck._model.templates[0]["qfmt"]

    def test_card_has_word_on_back(self, tmp_path: Path) -> None:
        """Test that card back contains the word."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment(word="chien"))
            # Check that the model has the expected template
            assert "{{Word}}" in deck._model.templates[0]["afmt"]

    def test_card_has_ipa_on_back(self, tmp_path: Path) -> None:
        """Test that card back contains IPA."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment(ipa="/ʃjɛ̃/"))
            assert "{{IPA}}" in deck._model.templates[0]["afmt"]

    def test_handles_missing_ipa(self, tmp_path: Path) -> None:
        """Test that missing IPA is handled gracefully."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            assignment = make_sense_assignment(ipa=None)  # type: ignore[arg-type]
            assignment.word = DictionaryEntry(
                word="test",
                pos="noun",
                ipa=None,
                etymology=None,
                senses=[DictionarySense(id="1", translation="test", example=None)],
            )
            deck.add(assignment)
        assert output_path.exists()

    def test_card_has_etymology_on_back(self, tmp_path: Path) -> None:
        """Test that card back contains etymology field."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment(etymology="From Latin canis."))
            assert "{{Etymology}}" in deck._model.templates[0]["afmt"]

    def test_handles_missing_etymology(self, tmp_path: Path) -> None:
        """Test that missing etymology is handled gracefully."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            assignment = make_sense_assignment(etymology=None)
            deck.add(assignment)
        assert output_path.exists()

    def test_does_not_write_on_exception(self, tmp_path: Path) -> None:
        """Test that file is not written if an exception occurs."""
        output_path = tmp_path / "test.apkg"
        try:
            with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
                deck.add(make_sense_assignment())
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert not output_path.exists()

    def test_handles_empty_examples(self, tmp_path: Path) -> None:
        """Test that cards with no examples are created correctly."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            assignment = make_sense_assignment(
                examples=["Unused."],
                example_indices=[],  # No examples
            )
            deck.add(assignment)

        assert output_path.exists()
        assert deck.cards_added == 1

    def test_css_styling_applied(self, tmp_path: Path) -> None:
        """Test that CSS styling is applied to the model."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            assert ".front" in deck._model.css
            assert ".word" in deck._model.css
            assert ".ipa" in deck._model.css
            assert ".examples" in deck._model.css
            assert ".forms" in deck._model.css


class TestAnkiDeckBuilderMultipleCards:
    """Tests for adding multiple cards to a deck."""

    def test_adds_multiple_cards(self, tmp_path: Path) -> None:
        """Test adding multiple cards to a deck."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment(word="chien", translation="dog"))
            deck.add(make_sense_assignment(word="chat", translation="cat"))
            deck.add(make_sense_assignment(word="oiseau", translation="bird"))

        assert deck.cards_added == 3

    def test_different_senses_same_word(self, tmp_path: Path) -> None:
        """Test adding cards for different senses of the same word."""
        output_path = tmp_path / "test.apkg"
        with AnkiDeckBuilder(output_path, "Test Deck", "fr") as deck:
            deck.add(make_sense_assignment(word="faux", translation="forgery"))
            deck.add(make_sense_assignment(word="faux", translation="scythe"))

        assert deck.cards_added == 2
