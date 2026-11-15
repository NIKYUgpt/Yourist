import pytest

from app.evaluation.restoration_eval import outside_unchanged
from app.modules.deid.restoration import (
    Replacement,
    apply_replacements,
    inflect_person,
    restore_grammar,
)


def test_two_people_and_multiple_cases_keep_their_identity():
    mapping = {"[ФИО_1]": "Иванов Иван Иванович", "[ФИО_2]": "Петров Сергей Петрович"}
    text = "От [ФИО_1] к [ФИО_2]. Документы для [ФИО_1]: 3 рабочих дня, ч. 1 ст. 62."
    result = restore_grammar(text, mapping)
    assert result.text == (
        "От Иванова Ивана Ивановича к Петрову Сергею Петровичу. "
        "Документы для Иванова Ивана Ивановича: 3 рабочих дня, ч. 1 ст. 62."
    )
    assert result.changed == 3 and outside_unchanged(text, result.text, mapping)
    assert mapping["[ФИО_1]"] == "Иванов Иван Иванович"


def test_ambiguous_and_unrecognized_forms_keep_original_values():
    assert inflect_person("Петрова", "gent") is None
    assert inflect_person("Иванов И. И.", "gent") is None
    result = restore_grammar("Для [ФИО_1].", {"[ФИО_1]": "Петрова"})
    assert result.text == "Для Петрова." and result.unknown == 1


def test_oblique_first_surface_and_female_forms():
    assert inflect_person("Иванова Ивана Ивановича", "datv") == "Иванову Ивану Ивановичу"
    assert inflect_person("Иванова Мария Ивановна", "gent") == "Ивановой Марии Ивановны"


def test_unknown_labels_and_non_person_entities_are_literal():
    result = restore_grammar(
        "[ФИО_99], [ТЕЛЕФОН_1], [EMAIL_1]", {"[ТЕЛЕФОН_1]": "+7 900 000-00-00"}
    )
    assert result.text == "[ФИО_99], +7 900 000-00-00, [EMAIL_1]"
    assert result.changed == 0


@pytest.mark.parametrize(
    "edit",
    [
        Replacement(0, 7, "[ФИО_1]", "Петрову Сергею Петровичу"),
        Replacement(0, 8, "[ФИО_1]", "Иванов Иван Иванович"),
        Replacement(0, 7, "[ФИО_2]", "Иванов Иван Иванович"),
        Replacement(0, 7, "[ФИО_1]", "Иванов Иван Иванович не обязан платить"),
    ],
)
def test_correction_cannot_rewrite_law_or_swap_entities(edit):
    with pytest.raises(ValueError):
        apply_replacements(
            "[ФИО_1] обязан выплатить 1000 рублей.", {"[ФИО_1]": "Иванов Иван Иванович"}, [edit]
        )


def test_invalid_proposal_falls_back_and_does_not_log_values(monkeypatch):
    def invalid(*args):
        raise ValueError("bad span")

    monkeypatch.setattr("app.modules.deid.restoration.apply_replacements", invalid)
    result = restore_grammar("Для [ФИО_1].", {"[ФИО_1]": "Иванов Иван Иванович"})
    assert result.fallback and result.text == "Для Иванов Иван Иванович."
    assert "Иванов" not in str(result.notes)


def test_preservation_check_rejects_changed_numbers_negation_and_punctuation():
    mapping = {"[ФИО_1]": "Иванов Иван Иванович"}
    text = "Для [ФИО_1]: не позднее 3 рабочих дней."
    corrected = restore_grammar(text, mapping).text
    assert outside_unchanged(text, corrected, mapping)
    for changed in (corrected.replace("3", "30"), corrected.replace("не ", ""), corrected[:-1]):
        assert not outside_unchanged(text, changed, mapping)
