"""GigaChat (Сбер).

Авторизация в два шага: по «ключу авторизации» получаем токен доступа
(POST /api/v2/oauth, живёт 30 минут), с ним вызываем OpenAI-совместимый API
https://api.giga.chat/v1 (developers.sber.ru/docs/ru/gigachat/api/authorization).
"""

import asyncio
import ssl
import time
import uuid
from typing import Any

import certifi

from app.modules.ai.providers.base import Kind, ProviderError, ProviderInfo
from app.modules.ai.providers.openai_compat import CHAT_AND_EMBED, OpenAICompatProvider

# обновляем токен заранее, чтобы он не истёк посреди запроса
_TOKEN_MARGIN_SEC = 60


def gigachat_verify(ca_bundle: str, verify: bool) -> ssl.SSLContext | bool:
    """Стандартные корневые сертификаты плюс сертификат НУЦ Минцифры, если он задан."""
    if not verify:
        return False
    if not ca_bundle:
        return True
    context = ssl.create_default_context(cafile=certifi.where())
    context.load_verify_locations(cafile=ca_bundle)
    return context


class GigaChatProvider(OpenAICompatProvider):
    info = ProviderInfo(
        name="gigachat",
        title="GigaChat (Сбер)",
        kind=Kind.API,
        capabilities=CHAT_AND_EMBED,
        location="ru",
    )
    embed_batch = 16
    # response_format в документации GigaChat не описан — JSON просим в промпте
    supports_json_mode = False

    def __init__(self, base_url: str, credentials: str, scope: str, auth_url: str, **kwargs: Any):
        super().__init__(base_url, api_key="", **kwargs)
        self.credentials = credentials
        self.scope = scope
        self.auth_url = auth_url
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    def configured(self) -> bool:
        return bool(self.credentials)

    async def _token_value(self) -> str:
        async with self._lock:
            if self._token is None or time.time() > self._expires_at - _TOKEN_MARGIN_SEC:
                await self._refresh()
            return self._token

    async def _refresh(self) -> None:
        try:
            resp = await self.http.post(
                self.auth_url,
                data={"scope": self.scope},
                headers={
                    "Authorization": f"Basic {self.credentials}",
                    "RqUID": str(uuid.uuid4()),
                    "Accept": "application/json",
                },
            )
        except Exception as exc:
            raise ProviderError(self.info.name, f"токен не получен: {exc}") from exc
        if resp.status_code != 200:
            raise ProviderError(
                self.info.name,
                f"токен не получен: HTTP {resp.status_code}: {resp.text[:300]}",
                status=resp.status_code,
            )
        data = resp.json()
        expires_at = float(data.get("expires_at") or 0)
        # в примерах документации встречаются и секунды, и миллисекунды
        if expires_at > 1e11:
            expires_at /= 1000
        self._token = data["access_token"]
        self._expires_at = expires_at or time.time() + 30 * 60

    async def _headers(self) -> dict[str, str]:
        if not self.configured():
            raise ProviderError(self.info.name, "не задан ключ авторизации")
        return {"Authorization": f"Bearer {await self._token_value()}"}

    async def _on_unauthorized(self) -> bool:
        self._token = None  # токен отозван раньше срока — получим новый
        return True
