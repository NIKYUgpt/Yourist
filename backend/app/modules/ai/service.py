"""Единая точка вызова нейросетей для остальных модулей.

Модули просят «ответь на вопрос» или «посчитай векторы», не зная, какая модель
стоит за задачей: это решает текущая конфигурация из админки.
"""

from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Request
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import SessionDep
from app.modules.ai.models import AISettings
from app.modules.ai.profiles import AIConfig, Mode, ModelRef, Task, embedding_key, with_defaults
from app.modules.ai.providers.base import ChatResult, EmbedKind, Message, Provider
from app.modules.ai.registry import Registry


async def load_config(session: AsyncSession) -> AIConfig:
    row = await session.get(AISettings, 1)
    return with_defaults(row.config if row else None)


async def save_config(session: AsyncSession, config: AIConfig, user_id: UUID | None) -> None:
    data = config.model_dump(mode="json")
    stmt = insert(AISettings).values(id=1, config=data, updated_by=user_id)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[AISettings.id], set_={"config": data, "updated_by": user_id}
        )
    )
    await session.commit()


class AIGateway:
    def __init__(self, registry: Registry, config: AIConfig):
        self.registry = registry
        self.config = config

    def resolve(self, task: Task) -> tuple[Mode, ModelRef, Provider]:
        mode, ref = self.config.resolve(task)
        return mode, ref, self.registry.get(ref.provider)

    async def chat(
        self,
        task: Task,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult:
        _, ref, provider = self.resolve(task)
        return await provider.chat(
            ref.model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            options=ref.options,
        )

    async def embed(
        self, texts: list[str], kind: EmbedKind = EmbedKind.DOCUMENT
    ) -> list[list[float]]:
        _, ref, provider = self.resolve(Task.EMBED)
        return await provider.embed(ref.model, texts, kind=kind, options=ref.options)

    @property
    def embedding_key(self) -> str:
        return embedding_key(self.config.resolve(Task.EMBED)[1])

    def describe(self, task: Task) -> dict[str, Any]:
        mode, ref = self.config.resolve(task)
        return {"mode": mode.value, "provider": ref.provider, "model": ref.model}


def get_registry(request: Request) -> Registry:
    return request.app.state.ai_registry


RegistryDep = Annotated[Registry, Depends(get_registry)]


async def get_ai(session: SessionDep, registry: RegistryDep) -> AIGateway:
    return AIGateway(registry, await load_config(session))


AIDep = Annotated[AIGateway, Depends(get_ai)]
