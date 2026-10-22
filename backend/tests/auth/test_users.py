from app.modules.auth.models import Role
from tests.conftest import bearer, login, register, register_verified, set_role


async def _admin(client, mailer, db, email="admin@example.com") -> dict:
    user = await register_verified(client, mailer, email)
    await set_role(db, email, Role.ADMIN)
    return {**user, "tokens": await login(client, email)}


async def test_regular_user_cannot_list_users(client, mailer):
    await register_verified(client, mailer, "user@example.com")
    tokens = await login(client, "user@example.com")
    resp = await client.get("/api/v1/users", headers=bearer(tokens))
    assert resp.status_code == 403


async def test_support_can_list_but_not_edit(client, mailer, db):
    target = await register(client, "client@example.com")
    await register_verified(client, mailer, "support@example.com")
    await set_role(db, "support@example.com", Role.SUPPORT)
    tokens = await login(client, "support@example.com")

    resp = await client.get("/api/v1/users", headers=bearer(tokens))
    assert resp.status_code == 200
    assert resp.json()["total"] == 2

    resp = await client.patch(
        f"/api/v1/users/{target['id']}", json={"role": "admin"}, headers=bearer(tokens)
    )
    assert resp.status_code == 403


async def test_list_users_filters_by_email(client, mailer, db):
    admin = await _admin(client, mailer, db)
    await register(client, "petrov@example.com")
    await register(client, "sidorov@example.com")

    resp = await client.get(
        "/api/v1/users", params={"email": "PETROV"}, headers=bearer(admin["tokens"])
    )
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["email"] == "petrov@example.com"


async def test_admin_changes_role(client, mailer, db):
    admin = await _admin(client, mailer, db)
    target = await register(client, "helper@example.com")

    resp = await client.patch(
        f"/api/v1/users/{target['id']}", json={"role": "support"}, headers=bearer(admin["tokens"])
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "support"


async def test_admin_cannot_demote_or_block_self(client, mailer, db):
    admin = await _admin(client, mailer, db)
    for payload in ({"role": "user"}, {"is_active": False}):
        resp = await client.patch(
            f"/api/v1/users/{admin['id']}", json=payload, headers=bearer(admin["tokens"])
        )
        assert resp.status_code == 400


async def test_blocked_user_loses_access(client, mailer, db):
    admin = await _admin(client, mailer, db)
    user = await register_verified(client, mailer, "bad@example.com")
    tokens = await login(client, "bad@example.com")

    resp = await client.patch(
        f"/api/v1/users/{user['id']}", json={"is_active": False}, headers=bearer(admin["tokens"])
    )
    assert resp.status_code == 200

    assert (await client.get("/api/v1/users/me", headers=bearer(tokens))).status_code == 401
    refresh = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert refresh.status_code == 401


async def test_patch_unknown_user_returns_404(client, mailer, db):
    admin = await _admin(client, mailer, db)
    resp = await client.patch(
        "/api/v1/users/00000000-0000-0000-0000-000000000000",
        json={"role": "user"},
        headers=bearer(admin["tokens"]),
    )
    assert resp.status_code == 404
