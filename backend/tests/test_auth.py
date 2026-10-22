from datetime import timedelta
from uuid import uuid4

from app.core.security import create_token
from app.services.auth import VERIFY_EMAIL
from tests.conftest import (
    PASSWORD,
    bearer,
    last_email_token,
    login,
    register,
    register_verified,
)


async def test_register_sends_verification_email(client, mailer):
    user = await register(client, "Ivan@Example.com")

    assert user["email"] == "ivan@example.com"
    assert user["role"] == "user"
    assert user["is_verified"] is False
    assert "password" not in str(user)
    assert mailer.outbox[-1]["To"] == "ivan@example.com"
    assert last_email_token(mailer)


async def test_register_requires_pd_consent(client):
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "a@example.com", "password": PASSWORD, "pd_consent": False},
    )
    assert resp.status_code == 422


async def test_register_rejects_short_password(client):
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "a@example.com", "password": "short", "pd_consent": True},
    )
    assert resp.status_code == 422


async def test_register_duplicate_email_case_insensitive(client):
    await register(client, "dup@example.com")
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "DUP@example.com", "password": PASSWORD, "pd_consent": True},
    )
    assert resp.status_code == 409


async def test_login_requires_verified_email(client):
    await register(client, "new@example.com")
    resp = await client.post(
        "/api/v1/auth/login", json={"email": "new@example.com", "password": PASSWORD}
    )
    assert resp.status_code == 403


async def test_verify_email_then_login(client, mailer):
    user = await register_verified(client, mailer, "ok@example.com")
    assert user["is_verified"] is True

    tokens = await login(client, "OK@example.com")
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] > 0

    me = await client.get("/api/v1/users/me", headers=bearer(tokens))
    assert me.status_code == 200
    assert me.json()["email"] == "ok@example.com"


async def test_verify_email_rejects_garbage_token(client):
    resp = await client.post("/api/v1/auth/verify-email", json={"token": "not-a-token"})
    assert resp.status_code == 400


async def test_verify_email_rejects_token_for_other_email(client, mailer):
    user = await register(client, "first@example.com")
    forged = create_token(
        user["id"], VERIFY_EMAIL, timedelta(hours=1), email="someone-else@example.com"
    )
    resp = await client.post("/api/v1/auth/verify-email", json={"token": forged})
    assert resp.status_code == 400


async def test_resend_verification_does_not_leak_accounts(client, mailer):
    await register(client, "wait@example.com")
    sent_before = len(mailer.outbox)

    known = await client.post(
        "/api/v1/auth/resend-verification", json={"email": "wait@example.com"}
    )
    unknown = await client.post(
        "/api/v1/auth/resend-verification", json={"email": "nobody@example.com"}
    )

    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()
    assert len(mailer.outbox) == sent_before + 1


async def test_login_wrong_password_and_unknown_email_look_the_same(client, mailer):
    await register_verified(client, mailer, "me@example.com")
    wrong = await client.post(
        "/api/v1/auth/login", json={"email": "me@example.com", "password": "wrong-password"}
    )
    unknown = await client.post(
        "/api/v1/auth/login", json={"email": "ghost@example.com", "password": "wrong-password"}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


async def test_refresh_rotates_tokens(client, mailer):
    await register_verified(client, mailer, "rot@example.com")
    first = await login(client, "rot@example.com")

    resp = await client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert resp.status_code == 200
    second = resp.json()
    assert second["refresh_token"] != first["refresh_token"]

    me = await client.get("/api/v1/users/me", headers=bearer(second))
    assert me.status_code == 200


async def test_reused_refresh_token_revokes_all_sessions(client, mailer):
    await register_verified(client, mailer, "theft@example.com")
    first = await login(client, "theft@example.com")
    second = (
        await client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    ).json()

    # кто-то повторно предъявил уже использованный токен
    reuse = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
    )
    assert reuse.status_code == 401

    # и легитимная новая сессия тоже закрыта
    after = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": second["refresh_token"]}
    )
    assert after.status_code == 401


async def test_logout_revokes_refresh_token(client, mailer):
    await register_verified(client, mailer, "bye@example.com")
    tokens = await login(client, "bye@example.com")

    resp = await client.post("/api/v1/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 204

    resp = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert resp.status_code == 401


async def test_me_requires_token(client):
    resp = await client.get("/api/v1/users/me")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


async def test_other_token_types_are_not_accepted_as_access(client, mailer):
    await register_verified(client, mailer, "typ@example.com")
    tokens = await login(client, "typ@example.com")

    # refresh-токен и токен из письма не должны работать как access
    for token in (tokens["refresh_token"], last_email_token(mailer)):
        resp = await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401


async def test_access_token_of_deleted_user_is_rejected(client):
    token = create_token(uuid4(), "access", timedelta(minutes=5))
    resp = await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
