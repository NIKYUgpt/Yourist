import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from httpx import ASGITransport

from app.deid_main import create_app
from app.modules.deid.api import DeidClient, DeidSettings
from app.modules.deid.identity import IdentityRequired
from app.modules.deid.service import deidentify
from app.modules.deid.sessions import SessionBusy, SessionExpired, SessionLimit, SessionStore
from tests.deid.identity_helpers import resolution
from tests.fakes import fake_registry


async def test_stable_labels_and_isolation():
    store = SessionStore()
    owner = uuid4()
    first = store.create(owner)
    async with store.use(first.id, owner) as mapping:
        one = await deidentify("Телефон 89161234567", mapping=mapping)
    async with store.use(first.id, owner) as mapping:
        two = await deidentify("Другой телефон 89167654321, старый 89161234567", mapping=mapping)
    assert one.text == "Телефон [ТЕЛЕФОН_1]"
    assert two.text == "Другой телефон [ТЕЛЕФОН_2], старый [ТЕЛЕФОН_1]"
    assert two.mapping == {"[ТЕЛЕФОН_1]": "89161234567", "[ТЕЛЕФОН_2]": "89167654321"}
    second = store.create(owner)
    async with store.use(second.id, owner) as mapping:
        result = await deidentify("Телефон 89167654321", mapping=mapping)
    assert result.text == "Телефон [ТЕЛЕФОН_1]"
    with pytest.raises(SessionExpired):
        async with store.use(first.id, uuid4()):
            pytest.fail("чужая таблица меток доступна")


async def test_expiration_and_restart_never_recreate_mapping():
    now = datetime(2026, 11, 5, tzinfo=UTC)
    store = SessionStore(ttl_seconds=60, clock=lambda: now)
    info = store.create(uuid4())
    async with store.use(info.id, info.user_id) as mapping:
        await deidentify("Телефон 89161234567", mapping=mapping)
    now += timedelta(seconds=61)
    with pytest.raises(SessionExpired):
        async with store.use(info.id, info.user_id):
            pass
    assert store.entries == {}
    with pytest.raises(SessionExpired):
        async with SessionStore().use(info.id, info.user_id):
            pass


async def test_concurrent_use_delete_and_failed_operation():
    store = SessionStore()
    info = store.create(uuid4())
    with pytest.raises(RuntimeError):
        async with store.use(info.id, info.user_id) as mapping:
            mapping["[ТЕЛЕФОН_1]"] = "89161234567"
            raise RuntimeError("failed before commit")
    assert store.entries[info.id].mapping == {}
    with pytest.raises(SessionExpired):
        async with store.use(info.id, info.user_id) as mapping:
            with pytest.raises(SessionBusy):
                async with store.use(info.id, info.user_id):
                    pass
            mapping["[ТЕЛЕФОН_1]"] = "89161234567"
            store.remove(info.id, info.user_id)
    assert not store.entries
    assert mapping == {}


async def test_limits_do_not_evict_or_partially_update():
    store = SessionStore(max_sessions=1, max_entities=1, max_chars=12)
    info = store.create(uuid4())
    with pytest.raises(SessionLimit):
        store.create(uuid4())
    with pytest.raises(SessionLimit):
        async with store.use(info.id, info.user_id) as mapping:
            await deidentify("89161234567 и 89167654321", mapping=mapping)
    assert store.entries[info.id].mapping == {}
    with pytest.raises(SessionLimit):
        async with store.use(info.id, info.user_id) as mapping:
            mapping["[ФИО_1]"] = "Очень длинное исходное значение"
    assert store.entries[info.id].mapping == {}
    store.start()
    await asyncio.sleep(0)
    await store.aclose()
    assert store.entries == {}


async def test_literal_placeholder_cannot_capture_a_new_person():
    result = await deidentify("[ТЕЛЕФОН_1] и номер 89161234567")
    assert result.mapping == {"[ТЕЛЕФОН_2]": "89161234567"}


async def test_known_namesakes_remain_ambiguous_between_turns():
    store = SessionStore()
    info = store.create(uuid4())
    async with store.use(info.id, info.user_id) as mapping:
        mapping.update({"[ФИО_1]": "Иванов Иван Иванович", "[ФИО_2]": "Иванов Иван Иванович"})
    for number in (3, 4):
        text = "Копию передали Иванову Ивану Ивановичу."
        with pytest.raises(IdentityRequired):
            async with store.use(info.id, info.user_id) as mapping:
                await deidentify(text, mapping=mapping)
        async with store.use(info.id, info.user_id) as mapping:
            result = await deidentify(
                text,
                mapping=mapping,
                person_resolution=resolution(text, ("Иванову Ивану Ивановичу", f"worker-{number}")),
            )
        assert result.text == f"Копию передали [ФИО_{number}]."
    assert store.entries[info.id].mapping["[ФИО_1]"] == "Иванов Иван Иванович"
    assert store.entries[info.id].mapping["[ФИО_2]"] == "Иванов Иван Иванович"
    await store.aclose()
    assert not store.entries


@pytest.mark.parametrize("remote", [False, True])
async def test_namesake_ambiguity_survives_service_boundary(remote):
    registry = fake_registry()
    app = create_app()
    app.state.ai_registry = registry
    client = DeidClient(
        DeidSettings(deid_url="http://deid" if remote else ""),
        registry,
        transport=ASGITransport(app=app),
    )
    try:
        owner = uuid4()
        info = await client.create_session(owner)
        store = app.state.deid_sessions if remote else client.sessions
        async with store.use(info.id, owner) as mapping:
            mapping.update({"[ФИО_1]": "Иванов Иван Иванович", "[ФИО_2]": "Иванов Иван Иванович"})
        text = "Копию передали Иванову Ивану Ивановичу."
        with pytest.raises(IdentityRequired):
            await client.deidentify(text, None, session_id=info.id, user_id=owner)
        result = await client.deidentify(
            text,
            None,
            session_id=info.id,
            user_id=owner,
            person_resolution=resolution(text, ("Иванову Ивану Ивановичу", "[ФИО_2]")),
        )
        assert result.text == "Копию передали [ФИО_2]."
        assert result.mapping["[ФИО_1]"] == result.mapping["[ФИО_2]"] == "Иванов Иван Иванович"
        await client.delete_session(info.id, owner)
        assert not store.entries
    finally:
        await client.aclose()


@pytest.mark.parametrize("remote", [False, True])
async def test_session_client_and_deletion(remote):
    registry = fake_registry()
    app = create_app()
    app.state.ai_registry = registry
    client = DeidClient(
        DeidSettings(deid_url="http://deid" if remote else ""),
        registry,
        transport=ASGITransport(app=app),
    )
    try:
        owner = uuid4()
        info = await client.create_session(owner)
        for value, label in [("89161234567", 1), ("89167654321", 2), ("89161234567", 1)]:
            result = await client.deidentify(
                "Телефон " + value, None, session_id=info.id, user_id=owner
            )
            assert result.text == f"Телефон [ТЕЛЕФОН_{label}]"
        with pytest.raises(SessionExpired):
            await client.deidentify(
                "Телефон 89161234567", None, session_id=info.id, user_id=uuid4()
            )
        await client.delete_session(info.id, owner)
        await client.delete_session(info.id, owner)
        with pytest.raises(SessionExpired):
            await client.deidentify("Телефон 89161234567", None, session_id=info.id, user_id=owner)
    finally:
        await client.aclose()
