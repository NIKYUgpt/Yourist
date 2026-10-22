from app import __version__


async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok", "version": __version__}
