"""Identity challenges traverse the deid transport without raw values or session confusion."""

import hashlib
import re
from uuid import uuid4

import pytest
from httpx import ASGITransport

from app.deid_main import create_app
from app.modules.ai.profiles import ModelRef
from app.modules.deid.api import DeidClient, DeidSettings
from app.modules.deid.identity import IdentityBindingError, IdentityRequired, PersonResolution
from app.modules.deid.media import with_document
from app.modules.deid.sessions import SessionExpired
from tests.fakes import fake_registry

PERSON = "Иванов Иван Иванович"
TEXT = f"директор {PERSON} уволил работника {PERSON}. Законно ли увольнение?"


def resolution(text, *, same=False):
    return PersonResolution.model_validate(
        {
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "bindings": [
                {
                    "start": match.start(),
                    "end": match.end(),
                    "participant": "participant-a" if same or index == 0 else "participant-b",
                }
                for index, match in enumerate(re.finditer(PERSON, text))
            ],
        }
    )


@pytest.fixture(params=[False, True], ids=["in-process", "http"])
async def identity_client(request):
    registry = fake_registry()
    server = create_app()
    server.state.ai_registry = registry
    client = DeidClient(
        DeidSettings(deid_url="http://deid" if request.param else ""),
        registry,
        transport=ASGITransport(app=server) if request.param else None,
    )
    yield client, registry
    await client.aclose()


@pytest.mark.parametrize("same", [False, True])
async def test_challenge_and_explicit_identity_retry_cross_transport(identity_client, same):
    client, registry = identity_client
    owner = uuid4()
    session = await client.create_session(owner)
    context = {"session_id": session.id, "user_id": owner}
    with pytest.raises(IdentityRequired) as raised:
        await client.deidentify(TEXT, None, **context)
    detail = raised.value.detail
    assert detail["code"] == "person_identity_required"
    assert detail["text_sha256"] == hashlib.sha256(TEXT.encode()).hexdigest()
    assert PERSON not in str(detail) and "Иванов" not in str(raised.value)
    assert not registry.get("ollama").chat_calls
    result = await client.deidentify(
        TEXT, None, person_resolution=resolution(TEXT, same=same), **context
    )
    labels = [item["placeholder"] for item in result.entities if item["type"] == "ФИО"]
    assert len(labels) == 2 and len(set(labels)) == (1 if same else 2)
    assert PERSON not in result.text
    await client.delete_session(session.id, owner)
    with pytest.raises(SessionExpired):
        await client.deidentify(TEXT, None, **context)


async def test_stale_resolution_is_safe_validation_error(identity_client):
    client, _ = identity_client
    stale = resolution(TEXT).model_copy(update={"text_sha256": "0" * 64})
    with pytest.raises(IdentityBindingError) as raised:
        await client.deidentify(TEXT, None, person_resolution=stale)
    assert PERSON not in str(raised.value)


@pytest.mark.parametrize("kind", ["voice", "photo"])
async def test_media_challenge_and_retry_resolve_recognized_text(identity_client, kind):
    client, registry = identity_client
    registry.get("whisper").transcript = TEXT
    registry.get("ollama").document_text = TEXT
    recognized = TEXT if kind == "voice" else with_document("Законно ли это?", TEXT)

    async def send(binding=None):
        options = {"person_resolution": binding} if binding is not None else {}
        if kind == "voice":
            return await client.from_audio(
                b"audio",
                "voice.ogg",
                ModelRef(provider="whisper", model="fixture"),
                None,
                **options,
            )
        return await client.from_image(
            b"image",
            "document.jpg",
            "image/jpeg",
            "Законно ли это?",
            ModelRef(provider="ollama", model="fixture"),
            None,
            **options,
        )

    with pytest.raises(IdentityRequired) as raised:
        await send()
    assert raised.value.detail["text_sha256"] == hashlib.sha256(recognized.encode()).hexdigest()
    assert PERSON not in str(raised.value.detail)
    result = await send(resolution(recognized))
    assert "[ФИО_1]" in result.deid.text and "[ФИО_2]" in result.deid.text
    assert not registry.get("ollama").chat_calls
