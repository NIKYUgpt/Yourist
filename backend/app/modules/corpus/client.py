"""Клиент actual.pravo.gov.ru.

API недокументированный — форматы запросов взяты из кода самого сайта.
Поэтому ответы проверяются явно, а любая неожиданность превращается в SourceError.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from app.modules.corpus.config import CorpusSettings

logger = logging.getLogger(__name__)


class SourceError(Exception):
    """Источник ответил ошибкой или данными неожиданного вида."""


@dataclass(frozen=True)
class SearchHit:
    external_id: str
    external_hash: str
    title: str
    passing: str
    state: str


@dataclass(frozen=True)
class ActCard:
    external_id: str
    external_hash: str
    title: str
    passing: str
    state: str
    kind: str | None
    number: str | None
    adopted_on: date | None


@dataclass(frozen=True)
class SourceRedaction:
    external_id: str
    valid_from: date
    # время формирования редакции на стороне источника — меняется при её пересборке
    source_timestamp: str | None
    caption: str
    is_actual: bool
    is_official: bool
    has_content: bool


def _parse_compact_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _parse_ru_date(value: str | None) -> date | None:
    try:
        return datetime.strptime(value, "%d.%m.%Y").date() if value else None
    except ValueError:
        return None


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


class PravoClient:
    def __init__(self, settings: CorpusSettings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.source_url.rstrip("/") + "/",
            timeout=settings.request_timeout_sec,
            headers={
                "User-Agent": settings.user_agent,
                "Referer": "http://actual.pravo.gov.ru/",
            },
            transport=transport,
        )
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self.requests_made = 0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET с паузой между запросами и повторами при сетевых сбоях и 5xx."""
        attempts = max(1, self.settings.request_retries)
        for attempt in range(1, attempts + 1):
            async with self._lock:
                wait = self.settings.request_delay_sec - (time.monotonic() - self._last_request)
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    resp = await self._http.get(path, params=params)
                    failure = f"HTTP {resp.status_code}" if resp.status_code >= 500 else None
                except httpx.TransportError as exc:
                    resp, failure = None, f"{type(exc).__name__}: {exc}"
                finally:
                    self._last_request = time.monotonic()
                    self.requests_made += 1

            if resp is not None and failure is None:
                return self._decode(path, resp)
            if attempt < attempts:
                logger.warning("Источник %s: %s, попытка %d/%d", path, failure, attempt, attempts)
                await asyncio.sleep(2**attempt)
        raise SourceError(f"{path}: источник недоступен ({failure})")

    @staticmethod
    def _decode(path: str, resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code >= 400:
            raise SourceError(f"{path}: HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise SourceError(f"{path}: ответ не JSON") from exc
        if not isinstance(data, dict):
            raise SourceError(f"{path}: неожиданный формат ответа")
        if data.get("error"):
            raise SourceError(f"{path}: {data['error']}")
        return data

    async def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        words = [f"+{w}" for w in query.lower().split() if w]
        if not words:
            return []
        q = [
            {"AttrId": 7, "AttrMode": 9, "Words": words},
            # параметры выдачи в том виде, как их шлёт сайт: размер, сортировка, страница
            {"AttrId": 999, "AttrMode": 1, "Words": [limit, "type", "20220701", 0, 1]},
        ]
        data = await self._get(
            "attrsearch/", {"bpa": "ebpi", "q": json.dumps(q, ensure_ascii=False)}
        )
        try:
            return [
                SearchHit(
                    external_id=str(doc["docid"]),
                    external_hash=doc["dochash"],
                    title=_clean(doc.get("docnames")),
                    passing=_clean(doc.get("docpassing")),
                    state=_clean(doc.get("docstate")),
                )
                for doc in data.get("docs") or []
            ]
        except (KeyError, TypeError) as exc:
            raise SourceError(f"attrsearch: неожиданный формат ({exc})") from exc

    async def card(self, doc_hash: str) -> ActCard:
        data = await self._get("card/", {"bpa": "ebpi", "t": json.dumps({"hash": doc_hash})})
        try:
            adoption = (data.get("adoptions") or [{}])[0]
            return ActCard(
                external_id=str(data["docid"]),
                external_hash=data["dochash"],
                title=_clean(data.get("docname")),
                passing=_clean(data.get("docpassing")),
                state=_clean(data.get("docstate")),
                kind=_clean(adoption.get("type")) or None,
                number=_clean(adoption.get("onumber")) or None,
                adopted_on=_parse_ru_date(adoption.get("odate")),
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise SourceError(f"card: неожиданный формат ({exc})") from exc

    async def redactions(self, doc_hash: str) -> list[SourceRedaction]:
        # ttl=4 — режим «тексты с внесёнными изменениями», как на самом сайте
        t = json.dumps({"hash": doc_hash, "ttl": 4})
        data = await self._get("redactions/", {"bpa": "ebpi", "t": t})
        try:
            return [
                SourceRedaction(
                    external_id=str(r["redid"]),
                    valid_from=_parse_compact_date(r["reddate"]),
                    source_timestamp=r.get("reddatetimed"),
                    caption=_clean(r.get("redcaption")),
                    is_actual=bool(r.get("actual")),
                    is_official=bool(r.get("redofficial")),
                    has_content=bool(r.get("hascontent")),
                )
                for r in data.get("redactions") or []
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceError(f"redactions: неожиданный формат ({exc})") from exc

    async def redaction_html(self, redaction_id: str) -> str:
        data = await self._get("redtext", {"bpa": "ebpi", "t": redaction_id, "ttl": 0})
        text = data.get("redtext")
        if not isinstance(text, str) or "<body" not in text:
            raise SourceError("redtext: в ответе нет HTML документа")
        return text

    async def redaction_toc(self, redaction_id: str) -> list[dict[str, Any]]:
        data = await self._get("getcontent/", {"bpa": "ebpi", "rdk": redaction_id})
        toc = data.get("data")
        if data.get("status") or not isinstance(toc, list) or not toc:
            raise SourceError(f"getcontent: оглавление не получено (status={data.get('status')})")
        if not data.get("complete", True):
            raise SourceError("getcontent: оглавление неполное")
        return toc
