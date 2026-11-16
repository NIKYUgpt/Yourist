import hashlib
import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.modules.ai.profiles import ModelRef
from app.modules.deid.identity import (
    IdentityBindingError,
    IdentityRequired,
    PersonMapping,
    PersonResolution,
)
from app.modules.deid.service import deidentify, reidentify
from app.modules.deid.sessions import SessionLimit, SessionStore
from tests.deid.identity_helpers import resolution
from tests.fakes import FakeProvider

NAME = "Иванов Иван Иванович"
TEXT = f"Два разных работника: {NAME} из отдела А и {NAME} из отдела Б."


async def test_initial_namesakes_stop_without_mutating_state_or_exposing_names():
    state = PersonMapping()
    with pytest.raises(IdentityRequired) as required:
        await deidentify(TEXT, mapping=state)
    detail = required.value.detail
    assert detail["code"] == "person_identity_required"
    assert detail["masked_text"] == (
        "Два разных работника: [ФИО_1] из отдела А и [ФИО_2] из отдела Б."
    )
    mentions = detail["groups"][0]["mentions"]
    assert len(mentions) == 2
    assert all(TEXT[m["start"] : m["end"]] == NAME for m in mentions)
    assert "Иванов" not in json.dumps(detail, ensure_ascii=False)
    assert "Иванов" not in str(required.value)
    assert not state and not state.actors and not state.aliases


@pytest.mark.parametrize("participants,labels", [(("a", "b"), (1, 2)), (("a", "a"), (1, 1))])
async def test_explicit_same_or_different_people_preserve_roles(participants, labels):
    if participants[0] == participants[1]:
        text = f"Работник {NAME} пришёл. Затем {NAME} получил копию."
    else:
        text = TEXT
    state = PersonMapping()
    result = await deidentify(
        text,
        mapping=state,
        person_resolution=resolution(text, (NAME, participants[0]), (NAME, participants[1])),
    )
    first, suffix = text.split(NAME, 1)
    middle, last = suffix.split(NAME, 1)
    assert result.text == f"{first}[ФИО_{labels[0]}]{middle}[ФИО_{labels[1]}]{last}"
    assert reidentify(result.text, result.mapping) == text
    assert state.actors[participants[0]] == f"[ФИО_{labels[0]}]"
    assert state.actors[participants[1]] == f"[ФИО_{labels[1]}]"


async def test_capitalized_role_does_not_split_full_name_and_bypass_clarification():
    text = f"Директор {NAME} уволил работника {NAME}."
    with pytest.raises(IdentityRequired) as required:
        await deidentify(text)
    mentions = required.value.detail["groups"][0]["mentions"]
    assert len(mentions) == 2
    assert all(text[m["start"] : m["end"]] == NAME for m in mentions)


async def test_confirmed_actors_survive_turns_but_unbound_names_still_need_confirmation():
    store = SessionStore()
    info = store.create(uuid4())
    async with store.use(info.id, info.user_id) as state:
        await deidentify(
            TEXT, mapping=state, person_resolution=resolution(TEXT, (NAME, "a"), (NAME, "b"))
        )
    next_text = "Копию передали Иванову Ивану Ивановичу."
    before = store.entries[info.id].mapping.copy()
    with pytest.raises(IdentityRequired) as required:
        async with store.use(info.id, info.user_id) as state:
            await deidentify(next_text, mapping=state)
    assert required.value.detail["groups"][0]["candidates"] == ["[ФИО_1]", "[ФИО_2]"]
    assert store.entries[info.id].mapping == before
    assert store.entries[info.id].mapping.actors == before.actors
    async with store.use(info.id, info.user_id) as state:
        result = await deidentify(
            next_text,
            mapping=state,
            person_resolution=resolution(next_text, ("Иванову Ивану Ивановичу", "b")),
        )
    assert result.text == "Копию передали [ФИО_2]."
    assert len(result.mapping) == 2
    kept_state = store.entries[info.id].mapping
    store.remove(info.id, info.user_id)
    assert not kept_state and not kept_state.actors and not kept_state.aliases


async def test_alias_confirmed_by_user_remains_a_candidate_after_next_turn():
    state = PersonMapping()
    text = f"{NAME} пришёл. Затем Иван явился."
    await deidentify(
        text, mapping=state, person_resolution=resolution(text, (NAME, "a"), ("Иван", "a"))
    )
    with pytest.raises(IdentityRequired):
        await deidentify("Иван просит копию.", mapping=state)


@pytest.mark.parametrize(
    "case", ["stale", "outside", "partial", "unknown_label", "provisional_label"]
)
async def test_invalid_bindings_do_not_change_mapping(case):
    state = PersonMapping()
    binding = resolution(TEXT, (NAME, "a"), (NAME, "b")).model_dump()
    if case == "stale":
        binding["text_sha256"] = "0" * 64
    elif case == "outside":
        binding["bindings"][1]["end"] = len(TEXT) + 1
    elif case == "partial":
        binding["bindings"][0]["end"] -= 1
    else:
        binding["bindings"][1]["participant"] = "[ФИО_99]" if case == "unknown_label" else "[ФИО_1]"
    with pytest.raises(IdentityBindingError):
        await deidentify(TEXT, mapping=state, person_resolution=PersonResolution(**binding))
    assert not state and not state.actors and not state.aliases


def test_overlapping_bindings_are_rejected_before_masking():
    raw = resolution(TEXT, (NAME, "a"), (NAME, "b")).model_dump()
    raw["bindings"][1]["start"] = raw["bindings"][0]["start"] + 1
    with pytest.raises(ValidationError):
        PersonResolution(**raw)


async def test_llm_mentions_use_original_offsets_after_dictionary_replacements():
    text = "Петров Сергей Петрович позвал шурик и шурик."
    provider = FakeProvider("ollama")
    provider.reply = lambda messages: '{"entities": [{"text": "шурик", "type": "ФИО"}]}'
    with pytest.raises(IdentityRequired) as required:
        await deidentify(text, (provider, ModelRef(provider="ollama", model="fixture")))
    detail = required.value.detail
    assert len(detail["groups"]) == 1
    mentions = detail["groups"][0]["mentions"]
    assert len(mentions) == 2 and all(text[m["start"] : m["end"]] == "шурик" for m in mentions)
    assert detail["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert "Петров" not in detail["masked_text"] and "шурик" not in detail["masked_text"]
    resolved = await deidentify(
        text, person_resolution=resolution(text, ("шурик", "a"), ("шурик", "b"))
    )
    assert resolved.text == "[ФИО_1] позвал [ФИО_2] и [ФИО_3]."


async def test_identity_metadata_obeys_session_limits_and_rolls_back():
    store = SessionStore(max_entities=1)
    info = store.create(uuid4())
    with pytest.raises(SessionLimit):
        async with store.use(info.id, info.user_id) as state:
            await deidentify(
                TEXT, mapping=state, person_resolution=resolution(TEXT, (NAME, "a"), (NAME, "b"))
            )
    kept = store.entries[info.id].mapping
    assert not kept and not kept.actors and not kept.aliases


async def test_evaluation_does_not_count_withheld_identity_pairs_as_success():
    from app.evaluation.restoration_eval import evaluate

    report = await evaluate(
        {
            "status": "synthetic",
            "restoration": [],
            "masking": [
                {
                    "id": "namesakes",
                    "text": TEXT,
                    "entities": [{"text": NAME, "person": "a"}, {"text": NAME, "person": "b"}],
                }
            ],
        }
    )
    row = report["masking"][0]
    assert row["clarification_required"]
    assert row["identity_merges"] is None and row["identity_splits"] is None
    assert row["identity_pairs_evaluated"] == 0 and row["identity_pairs_unresolved"] == 1
    assert report["summary"]["identity_clarifications"] == 1
