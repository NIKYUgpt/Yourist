import json
from datetime import date

import httpx
import pytest

from app.modules.corpus.client import PravoClient, SourceError
from app.modules.corpus.config import CorpusSettings

SETTINGS = CorpusSettings(request_delay_sec=0, request_retries=3, default_acts=[])


def make_client(handler) -> PravoClient:
    return PravoClient(SETTINGS, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    # повторы ждут 2, 4 с — в тестах не ждём
    async def instant(_):
        return None

    monkeypatch.setattr("app.modules.corpus.client.asyncio.sleep", instant)


async def test_redactions_request_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "docid": 64284,
                "redactions": [
                    {
                        "redid": 492286,
                        "reddate": "20260901",
                        "reddatetimed": "2026-09-01T09:55:54",
                        "redcaption": " с 01.09.2026, актуальная",
                        "actual": True,
                        "redofficial": True,
                        "hascontent": True,
                    }
                ],
                "error": "",
            },
        )

    client = make_client(handler)
    [red] = await client.redactions("abc")
    await client.aclose()

    assert seen["path"].endswith("/redactions/")
    assert json.loads(seen["params"]["t"]) == {"hash": "abc", "ttl": 4}
    assert red.external_id == "492286"
    assert red.valid_from == date(2026, 9, 1)
    assert red.caption == "с 01.09.2026, актуальная"
    assert red.is_actual and red.is_official and red.has_content


async def test_card_parsing():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "docid": 64284,
                "dochash": "abc",
                "docname": "Трудовой кодекс Российской Федерации",
                "docpassing": "Кодекс Российской Федерации от\xa030.12.2001\xa0№\xa0197-ФЗ",
                "docstate": "Действует с изменениями",
                "adoptions": [{"onumber": "197-ФЗ", "odate": "30.12.2001", "type": "Кодекс"}],
                "error": "",
            },
        )

    client = make_client(handler)
    card = await client.card("abc")
    await client.aclose()
    assert card.passing == "Кодекс Российской Федерации от 30.12.2001 № 197-ФЗ"
    assert (card.kind, card.number, card.adopted_on) == ("Кодекс", "197-ФЗ", date(2001, 12, 30))


async def test_search_sends_words_with_plus():
    seen = {}

    def handler(request):
        seen["q"] = json.loads(request.url.params["q"])
        return httpx.Response(200, json={"docs": [], "docscount": 0, "error": ""})

    client = make_client(handler)
    assert await client.search("Трудовой  Кодекс") == []
    await client.aclose()
    assert seen["q"][0]["Words"] == ["+трудовой", "+кодекс"]


async def test_retries_on_5xx_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500, json={"Message": "An error has occurred."})
        return httpx.Response(200, json={"redtext": "<html><body><p>x</p></body></html>"})

    client = make_client(handler)
    assert "<body>" in await client.redaction_html("1")
    await client.aclose()
    assert len(calls) == 3


async def test_gives_up_after_retries():
    def handler(request):
        raise httpx.ConnectError("boom")

    client = make_client(handler)
    with pytest.raises(SourceError, match="недоступен"):
        await client.redaction_html("1")
    assert client.requests_made == 3
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["list"]),
        httpx.Response(200, json={"error": "Документ не найден"}),
        httpx.Response(200, json={"redtext": None}),
    ],
)
async def test_bad_responses_raise_source_error(response):
    client = make_client(lambda request: response)
    with pytest.raises(SourceError):
        await client.redaction_html("1")
    await client.aclose()


async def test_incomplete_toc_rejected():
    client = make_client(
        lambda r: httpx.Response(200, json={"data": [{"id": "x"}], "complete": False, "status": 0})
    )
    with pytest.raises(SourceError, match="неполное"):
        await client.redaction_toc("1")
    await client.aclose()
