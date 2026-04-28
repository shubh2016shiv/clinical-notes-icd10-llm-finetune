from __future__ import annotations

from clinical_note_generation_v3.config.settings import V3PipelineSettings
from clinical_note_generation_v3.infrastructure.llm_provider import llm_client_factory
from clinical_note_generation_v3.infrastructure.llm_provider.openai_llm_client import (
    OpenAIJSONClient,
)


class FakeMessage:
    content = '{"ok": true}'


class FakeChoice:
    message = FakeMessage()


class FakeCompletionResponse:
    choices = [FakeChoice()]


class FakeChatCompletions:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeCompletionResponse()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeChatCompletions()


class FakeOpenAISdkClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_openai_json_client_uses_json_mode() -> None:
    client = OpenAIJSONClient.__new__(OpenAIJSONClient)
    client.model_name = "gpt-4o-mini"
    client._client = FakeOpenAISdkClient()

    result = client.generate_json("Return JSON.")

    assert result == {"ok": True}
    assert client._client.chat.completions.last_kwargs["model"] == "gpt-4o-mini"
    assert client._client.chat.completions.last_kwargs["response_format"] == {"type": "json_object"}


def test_default_llm_factory_uses_deepseek_primary_and_openai_fallback(monkeypatch) -> None:
    constructed: list[str] = []

    class FakeDeepSeek:
        provider_name = "deepseek"

        def __init__(self, api_key: str, model_name: str, base_url: str) -> None:
            constructed.append(f"deepseek:{model_name}:{base_url}:{api_key}")
            self.model_name = model_name

        def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
            raise RuntimeError("deepseek unavailable")

    class FakeOpenAI:
        provider_name = "openai"

        def __init__(self, api_key: str, model_name: str) -> None:
            constructed.append(f"openai:{model_name}:{api_key}")
            self.model_name = model_name

        def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
            return {"fallback": True}

    monkeypatch.setattr(llm_client_factory, "DeepSeekJSONClient", FakeDeepSeek)
    monkeypatch.setattr(llm_client_factory, "OpenAIJSONClient", FakeOpenAI)

    settings = V3PipelineSettings(
        deepseek_api_key="deepseek-key",
        openai_api_key="openai-key",
        deepseek_model="deepseek-v4-flash",
        openai_generation_model="gpt-4o-mini",
    )
    client = llm_client_factory.create_default_json_generation_client(settings)

    assert constructed == [
        "deepseek:deepseek-v4-flash:https://api.deepseek.com:deepseek-key",
        "openai:gpt-4o-mini:openai-key",
    ]
    assert client.generate_json("Return JSON.") == {"fallback": True}
    assert client.provider_name == "openai"


def test_default_llm_factory_uses_openai_when_deepseek_init_fails(monkeypatch) -> None:
    constructed: list[str] = []

    class BrokenDeepSeek:
        def __init__(self, api_key: str, model_name: str, base_url: str) -> None:
            constructed.append("deepseek")
            raise RuntimeError("missing deepseek key")

    class FakeOpenAI:
        provider_name = "openai"

        def __init__(self, api_key: str, model_name: str) -> None:
            constructed.append(f"openai:{model_name}:{api_key}")
            self.model_name = model_name

        def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
            return {"openai_only": True}

    monkeypatch.setattr(llm_client_factory, "DeepSeekJSONClient", BrokenDeepSeek)
    monkeypatch.setattr(llm_client_factory, "OpenAIJSONClient", FakeOpenAI)

    settings = V3PipelineSettings(
        deepseek_api_key=None,
        openai_api_key="openai-key",
        openai_generation_model="gpt-4o-mini",
    )
    client = llm_client_factory.create_default_json_generation_client(settings)

    assert constructed == ["deepseek", "openai:gpt-4o-mini:openai-key"]
    assert client.generate_json("Return JSON.") == {"openai_only": True}
    assert client.provider_name == "openai"
