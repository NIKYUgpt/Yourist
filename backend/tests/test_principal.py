from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.principal import Principal, PrincipalDep, require_role
from app.core.roles import Role
from app.core.security import ACCESS, create_token
from tests.conftest import bearer, login, register_verified, set_role

probe = FastAPI()


@probe.get("/whoami")
async def whoami(principal: PrincipalDep):
    return {"user_id": str(principal.user_id), "role": principal.role}


@probe.get("/admin-only")
async def admin_only(principal: Principal = Depends(require_role(Role.ADMIN))):
    return {"ok": True}


@pytest.fixture
async def probe_client():
    async with AsyncClient(transport=ASGITransport(app=probe), base_url="http://probe") as c:
        yield c


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_role_claim_from_real_login(client, mailer, db, probe_client):
    user = await register_verified(client, mailer, "claims@example.com")
    await set_role(db, "claims@example.com", Role.SUPPORT)
    tokens = await login(client, "claims@example.com")

    resp = await probe_client.get("/whoami", headers=bearer(tokens))
    assert resp.json() == {"user_id": user["id"], "role": "support"}
    assert (await probe_client.get("/admin-only", headers=bearer(tokens))).status_code == 403


async def test_admin_claim_passes_role_check(probe_client):
    token = create_token(uuid4(), ACCESS, timedelta(minutes=5), role="admin")
    assert (await probe_client.get("/admin-only", headers=_auth(token))).status_code == 200


@pytest.mark.parametrize(
    "token",
    [
        create_token(uuid4(), ACCESS, timedelta(minutes=5)),  # нет роли
        create_token(uuid4(), ACCESS, timedelta(minutes=5), role="root"),  # неизвестная роль
        create_token(uuid4(), "verify_email", timedelta(minutes=5), role="admin"),  # не access
        create_token(uuid4(), ACCESS, timedelta(minutes=-1), role="admin"),  # истёк
        "garbage",
    ],
)
async def test_bad_tokens_rejected(probe_client, token):
    assert (await probe_client.get("/whoami", headers=_auth(token))).status_code == 401


async def test_missing_token_rejected(probe_client):
    assert (await probe_client.get("/whoami")).status_code == 401
