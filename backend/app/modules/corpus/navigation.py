"""Чтение замороженного корпуса с единым бюджетом и трассой. Без сети и записи БД."""

import json
import time
from dataclasses import dataclass

from app.evaluation.answer_dataset import fingerprint
from app.evaluation.frozen_corpus import Snapshot
from app.modules.search.retrieval import explicit_article_keys


class NavigationError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    calls: int = 4
    characters: int = 24000
    seconds: float = 60
    article_page: int = 6000

    def __post_init__(self):
        if not (
            1 <= self.calls <= 20
            and 100 <= self.characters <= 100000
            and 0 < self.seconds <= 300
            and 100 <= self.article_page <= 12000
        ):
            raise ValueError("invalid navigation limits")


class CorpusNavigator:
    def __init__(self, snapshot: Snapshot, limits: Limits | None = None):
        snapshot.verify()
        self.snapshot = snapshot
        self.limits = limits or Limits()
        self.started = time.monotonic()
        self.characters = 0
        self.trace: list[dict] = []
        self.articles = {(a.act, a.key): a for a in snapshot.articles}
        self.sections: dict[str, dict] = {}
        for article in snapshot.articles:
            section_id = fingerprint([article.act, article.path])[:16]
            entry = self.sections.setdefault(
                section_id,
                {
                    "id": section_id,
                    "act": article.act,
                    "path": article.path,
                    "articles": [],
                },
            )
            entry["articles"].append(article)

    def call(self, tool: str, arguments: dict | None = None) -> dict:
        args = arguments or {}
        started = time.monotonic()
        entry = {"tool": tool, "arguments": args, "status": "error"}
        previous = len(self.trace)
        self.trace.append(entry)  # Неудачные и неизвестные вызовы тоже расходуют бюджет.
        try:
            if previous >= self.limits.calls:
                raise NavigationError("call_budget_exceeded")
            if started - self.started >= self.limits.seconds:
                raise NavigationError("time_budget_exceeded")
            allowed = {
                "contents": set(),
                "section": {"section_id"},
                "article": {"act", "key", "offset"},
                "crossrefs": {"act", "key"},
                "redaction": {"act"},
            }
            if tool not in allowed or set(args) - allowed[tool]:
                raise NavigationError("unknown_tool_or_arguments")
            result = getattr(self, f"_{tool}")(**args)
            result = {
                "legal_as_of": self.snapshot.legal_as_of.isoformat(),
                "source_status": self.snapshot.source_status,
                **result,
            }
            size = len(json.dumps(result, ensure_ascii=False))
            if self.characters + size > self.limits.characters:
                raise NavigationError("output_budget_exceeded")
            if time.monotonic() - self.started >= self.limits.seconds:
                raise NavigationError("time_budget_exceeded")
            self.characters += size
            entry.update(status="ok", result=result, characters=size)
            return result
        except (TypeError, KeyError, ValueError) as exc:
            # Сообщение не содержит произвольных текстов вопроса или исходных ПД.
            error = str(exc) if isinstance(exc, NavigationError) else "invalid_arguments"
            entry["error"] = error
            raise NavigationError(error) from exc
        finally:
            entry["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)

    def _contents(self) -> dict:
        return {
            "sections": [
                {
                    "id": value["id"],
                    "act": value["act"],
                    "title": value["path"][-1] if value["path"] else "Без раздела",
                    "count": len(value["articles"]),
                }
                for value in self.sections.values()
            ]
        }

    def _section(self, section_id: str) -> dict:
        section = self.sections[section_id]
        return {
            "section_id": section_id,
            "path": section["path"],
            "articles": [
                {"act": a.act, "key": a.key, "heading": a.heading, "is_repealed": a.is_repealed}
                for a in section["articles"]
            ],
        }

    def _get(self, act: str, key: str):
        article = self.articles.get((act, key))
        if article is None or article.is_repealed:
            raise NavigationError("article_unavailable_on_snapshot_date")
        return article

    def _article(self, act: str, key: str, offset: int = 0) -> dict:
        article = self._get(act, key)
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset < len(article.text)
        ):
            raise NavigationError("invalid_offset")
        end = min(offset + self.limits.article_page, len(article.text))
        return {
            "article": {
                "act": act,
                "key": key,
                "heading": article.heading,
                "redaction": article.redaction,
                "valid_from": article.valid_from.isoformat(),
                "text": article.text[offset:end],
                "offset": offset,
                "next_offset": end if end < len(article.text) else None,
                "content_sha256": article.content_sha256,
                "source_url": article.source_url,
            }
        }

    def _crossrefs(self, act: str, key: str) -> dict:
        article = self._get(act, key)
        return {
            "status": "heuristic_same_act_candidates_not_exhaustive",
            "links": [
                {
                    "act": act,
                    "key": ref,
                    "exists_in_snapshot": (act, ref) in self.articles,
                    "applicability": "unresolved",
                }
                for ref in explicit_article_keys(article.text)
                if ref != key
            ],
        }

    def _redaction(self, act: str) -> dict:
        articles = [a for a in self.snapshot.articles if a.act == act]
        if not articles:
            raise NavigationError("act_unavailable")
        return {
            "act": act,
            "redactions": sorted({(a.redaction, str(a.valid_from)) for a in articles}),
            "coverage": "only_the_frozen_date_not_complete_redaction_history",
        }
