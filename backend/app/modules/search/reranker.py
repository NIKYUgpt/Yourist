"""Необязательный CPU cross-encoder. Только локальные веса, без сети в запросе."""

import asyncio
import math
import threading
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class RerankSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )
    rerank_model_path: str = ""
    rerank_candidates: int = Field(24, ge=8, le=80)
    rerank_max_length: int = Field(512, ge=64, le=1024)
    rerank_batch_size: int = Field(4, ge=1, le=16)
    rerank_threads: int = Field(4, ge=1, le=16)
    rerank_timeout_seconds: float = Field(30, gt=0, le=300)


class RerankUnavailable(Exception):
    """Код причины, без исходного текста и внутренних путей."""


class LocalReranker:
    def __init__(self, settings: RerankSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        location = self.settings.rerank_model_path
        if not location or not Path(location).is_dir():
            raise RerankUnavailable("model_not_prepared")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RerankUnavailable("dependencies_missing") from exc
        torch.set_num_threads(self.settings.rerank_threads)
        tokenizer = AutoTokenizer.from_pretrained(
            location, local_files_only=True, trust_remote_code=False
        )
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                location,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=torch.float32,
            )
            .to("cpu")
            .eval()
        )
        self._tokenizer, self._model = tokenizer, model

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        self._load()
        import torch

        values = []
        size = self.settings.rerank_batch_size
        # Ограничиваем и токены, и исходный объём. Последняя реплика стоит первой
        # в history-query, поэтому не вытесняется длинной предысторией.
        with torch.inference_mode():
            for start in range(0, len(passages), size):
                batch = [[query[:6000], p[:12000]] for p in passages[start : start + size]]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.settings.rerank_max_length,
                    return_tensors="pt",
                )
                values.extend(self._model(**inputs).logits.view(-1).float().tolist())
        return values

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if len(passages) > 80:
            raise RerankUnavailable("candidate_limit")
        if not self._lock.acquire(blocking=False):
            raise RerankUnavailable("busy")

        def work():
            try:
                scores = self._predict(query, passages)
                if len(scores) != len(passages) or not all(math.isfinite(s) for s in scores):
                    raise RerankUnavailable("invalid_scores")
                return scores
            except RerankUnavailable:
                raise
            except Exception as exc:
                raise RerankUnavailable("inference_failed") from exc
            finally:
                self._lock.release()

        task = asyncio.create_task(asyncio.to_thread(work))
        # При timeout поток дорабатывает и удерживает lock: новые запросы не создают
        # очередь CPU-инференса. Исключение фоновой задачи всегда считывается.
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=self.settings.rerank_timeout_seconds
            )
        except TimeoutError as exc:
            raise RerankUnavailable("timeout") from exc


@lru_cache
def get_reranker() -> LocalReranker:
    return LocalReranker(RerankSettings())


def order(scores: list[float], explicit: list[bool]) -> list[int]:
    """Явные пользовательские ссылки остаются впереди; при равенстве — исходный ранг."""
    if len(scores) != len(explicit) or not all(math.isfinite(s) for s in scores):
        raise RerankUnavailable("invalid_scores")
    return sorted(range(len(scores)), key=lambda i: (not explicit[i], -scores[i], i))


async def rerank_hits(query, hits, k, *, per_article=2, explicit_limit=None):
    """Переставляет только уже отфильтрованные по дате фрагменты; не добавляет источники."""
    from dataclasses import replace

    ranker = get_reranker()
    scores = await ranker.score(query, [h.context + "\n" + h.text for h in hits])
    selected, counts = [], {}
    for i in order(scores, [h.explicit for h in hits]):
        h = hits[i]
        key = h.act_id, h.key
        limit = (explicit_limit or max(per_article, k // 2)) if h.explicit else per_article
        if counts.get(key, 0) >= limit:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(replace(h, rerank_score=scores[i]))
        if len(selected) == k:
            break
    return selected
