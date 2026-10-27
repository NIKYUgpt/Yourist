"""Провайдеры: какие запросы уходят и как разбираются ответы.

Живых ключей в тестах нет: ответы сервисов подменены httpx.MockTransport,
формат взят из документации и официальных SDK провайдеров.
"""

import json
import time

import httpx
import pytest

from app.modules.ai.config import ProviderSettings
from app.modules.ai.providers.base import EmbedKind, NotSupported, ProviderError, merge_options
from app.modules.ai.registry import build_registry

MESSAGES = [
    {"role": "system", "content": "Ты юрист."},
    {"role": "user", "content": "Сколько длится отпуск?"},
]


class Recorder:
    """Подменный сервер: запоминает запросы, отвечает по сценарию."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def registry(handler, **settings):
    recorder = Recorder(handler)
    values = {
        "yandex_api_key": "yc-key",
        "yandex_folder_id": "b1gfolder",
        "gigachat_credentials": "Z2lnYS1rZXk=",
        "openai_api_key": "sk-openai",
        "anthropic_api_key": "sk-ant",
        "gemini_api_key": "gm-key",
        "deepseek_api_key": "ds-key",
        "qwen_api_key": "qw-key",
        "ai_retries": 1,
        **settings,
    }
    reg = build_registry(
        ProviderSettings(_env_file=None, **values), transport=httpx.MockTransport(recorder)
    )
    return reg, recorder


def openai_chat_response(text: str = "28 календарных дней", model: str = "m") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        },
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("app.modules.ai.providers.base.asyncio.sleep", instant)


def test_merge_options_overrides_and_removes():
    body = {"temperature": 0.1, "max_tokens": 10}
    assert merge_options(body, {"temperature": None, "top_p": 0.5}) == {
        "max_tokens": 10,
        "top_p": 0.5,
    }


async def test_ollama_chat_and_embed():
    def handler(request):
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "model": "qwen3:8b",
                    "message": {"role": "assistant", "content": "Москва"},
                    "done_reason": "stop",
                    "prompt_eval_count": 20,
                    "eval_count": 2,
                },
            )
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    reg, rec = registry(handler)
    ollama = reg.get("ollama")
    result = await ollama.chat(
        "qwen3:8b", MESSAGES, json_mode=True, options={"think": False, "top_k": 20}
    )
    body = rec.body()
    assert str(rec.requests[-1].url) == "http://127.0.0.1:11434/api/chat"
    assert body["think"] is False and body["format"] == "json" and body["stream"] is False
    assert body["options"] == {
        "temperature": 0.1,
        "num_ctx": 8192,
        "num_predict": 2048,
        "top_k": 20,
    }
    assert (result.text, result.usage.prompt_tokens, result.usage.completion_tokens) == (
        "Москва",
        20,
        2,
    )

    vectors = await ollama.embed("bge-m3", ["а", "б"])
    assert rec.body() == {"model": "bge-m3", "input": ["а", "б"], "keep_alive": "30m"}
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_yandex_uses_model_uri_and_disables_logging():
    def handler(request):
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            data = [{"index": i, "embedding": [float(i)]} for i in range(len(body["input"]))]
            return httpx.Response(200, json={"data": data})
        return openai_chat_response()

    reg, rec = registry(handler)
    yandex = reg.get("yandex")
    await yandex.chat("yandexgpt", MESSAGES)
    req = rec.requests[-1]
    assert str(req.url) == "https://llm.api.cloud.yandex.net/v1/chat/completions"
    assert req.headers["Authorization"] == "Api-Key yc-key"
    assert req.headers["OpenAI-Project"] == "b1gfolder"
    assert req.headers["x-data-logging-enabled"] == "false"
    assert rec.body()["model"] == "gpt://b1gfolder/yandexgpt/latest"

    await yandex.chat("yandexgpt-lite/rc", MESSAGES)
    assert rec.body()["model"] == "gpt://b1gfolder/yandexgpt-lite/rc"

    await yandex.embed("text-search", ["документ"], kind=EmbedKind.DOCUMENT)
    assert rec.body()["model"] == "emb://b1gfolder/text-search-doc/latest"
    await yandex.embed("text-search", ["запрос"], kind=EmbedKind.QUERY)
    assert rec.body()["model"] == "emb://b1gfolder/text-search-query/latest"


async def test_gigachat_gets_token_once_and_refreshes_on_401():
    tokens = iter(["token-1", "token-2"])
    reject_next = {"flag": False}

    def handler(request):
        if request.url.host == "ngw.devices.sberbank.ru":
            # срок в миллисекундах — так в части примеров документации
            expires = int((time.time() + 1800) * 1000)
            return httpx.Response(200, json={"access_token": next(tokens), "expires_at": expires})
        if reject_next["flag"]:
            reject_next["flag"] = False
            return httpx.Response(401, json={"message": "Token has expired"})
        return openai_chat_response()

    reg, rec = registry(handler)
    giga = reg.get("gigachat")
    await giga.chat("GigaChat-2-Max", MESSAGES)
    await giga.chat("GigaChat-2-Max", MESSAGES)

    oauth = [r for r in rec.requests if r.url.host == "ngw.devices.sberbank.ru"]
    assert len(oauth) == 1, "токен должен переиспользоваться"
    assert str(oauth[0].url) == "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    assert oauth[0].headers["Authorization"] == "Basic Z2lnYS1rZXk="
    assert len(oauth[0].headers["RqUID"]) == 36
    assert oauth[0].content == b"scope=GIGACHAT_API_PERS"
    chat_req = rec.requests[-1]
    assert str(chat_req.url) == "https://api.giga.chat/v1/chat/completions"
    assert chat_req.headers["Authorization"] == "Bearer token-1"
    assert "response_format" not in rec.body()

    reject_next["flag"] = True
    await giga.chat("GigaChat-2-Max", MESSAGES, json_mode=True)
    assert rec.requests[-1].headers["Authorization"] == "Bearer token-2"
    assert len([r for r in rec.requests if r.url.host == "ngw.devices.sberbank.ru"]) == 2


async def test_gigachat_token_error_is_reported():
    reg, _ = registry(lambda request: httpx.Response(401, json={"message": "bad key"}))
    with pytest.raises(ProviderError, match="токен не получен: HTTP 401"):
        await reg.get("gigachat").chat("GigaChat", MESSAGES)


async def test_openai_uses_max_completion_tokens_and_json_mode():
    reg, rec = registry(lambda request: openai_chat_response())
    await reg.get("openai").chat(
        "gpt-5-mini", MESSAGES, max_tokens=300, json_mode=True, options={"temperature": None}
    )
    req, body = rec.requests[-1], rec.body()
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-openai"
    assert body["max_completion_tokens"] == 300 and "max_tokens" not in body
    assert "temperature" not in body
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("name", "url", "key"),
    [
        ("deepseek", "https://api.deepseek.com/chat/completions", "ds-key"),
        (
            "qwen",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            "qw-key",
        ),
        (
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "gm-key",
        ),
    ],
)
async def test_openai_compatible_endpoints(name, url, key):
    reg, rec = registry(lambda request: openai_chat_response("ок"))
    result = await reg.get(name).chat("some-model", MESSAGES, max_tokens=50)
    assert str(rec.requests[-1].url) == url
    assert rec.requests[-1].headers["Authorization"] == f"Bearer {key}"
    assert rec.body()["max_tokens"] == 50 and rec.body()["messages"] == MESSAGES
    assert result.text == "ок" and result.provider == name


async def test_qwen_embeddings_are_batched_by_ten():
    def handler(request):
        body = json.loads(request.content)
        # индексы приходят не по порядку — провайдер должен их отсортировать
        data = [{"index": i, "embedding": [len(t)]} for i, t in enumerate(body["input"])][::-1]
        return httpx.Response(200, json={"data": data})

    reg, rec = registry(handler)
    texts = ["x" * i for i in range(1, 24)]
    vectors = await reg.get("qwen").embed("text-embedding-v4", texts)
    assert [len(json.loads(r.content)["input"]) for r in rec.requests] == [10, 10, 3]
    assert vectors == [[len(t)] for t in texts]


async def test_anthropic_moves_system_prompt():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "28 дней"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 30, "output_tokens": 4},
            },
        )

    reg, rec = registry(handler)
    result = await reg.get("anthropic").chat("claude-sonnet-5-5", MESSAGES)
    req, body = rec.requests[-1], rec.body()
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "sk-ant"
    assert req.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "Ты юрист."
    assert body["messages"] == [MESSAGES[1]]
    assert (result.text, result.usage.prompt_tokens, result.finish_reason) == (
        "28 дней",
        30,
        "end_turn",
    )
    with pytest.raises(NotSupported):
        await reg.get("anthropic").embed("any", ["текст"])


async def test_retries_server_errors_then_fails():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="overloaded")

    reg, _ = registry(handler)
    with pytest.raises(ProviderError, match="HTTP 503"):
        await reg.get("deepseek").chat("deepseek-chat", MESSAGES)
    assert calls["n"] == 2  # первая попытка + один повтор (ai_retries=1)


async def test_unconfigured_provider_is_not_called():
    reg, rec = registry(lambda request: openai_chat_response(), openai_api_key="")
    assert not reg.get("openai").configured()
    with pytest.raises(ProviderError, match="не заданы ключи"):
        await reg.get("openai").chat("gpt", MESSAGES)
    assert rec.requests == []


async def test_proxy_only_for_listed_providers():
    reg, _ = registry(lambda r: openai_chat_response(), ai_proxy_url="http://proxy.local:3128")
    assert reg.get("openai").proxy == "http://proxy.local:3128"
    assert reg.get("anthropic").proxy == "http://proxy.local:3128"
    assert reg.get("yandex").proxy is None
    assert reg.get("ollama").proxy is None
