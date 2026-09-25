# -*- coding: utf-8 -*-
"""Slice 11B.1b.3 -- Yandex Knowledge QA provider adapter (PURE, offline).

NO network, NO credentials: the SDK client is INJECTED as a fake. The real
client-construction path is exercised by monkeypatching ``openai.OpenAI``.
"""

from __future__ import annotations

import json

import pytest

from support import knowledge_qa_yandex as ry
from support.knowledge import KnowledgeItem
from support.knowledge_qa import KnowledgeQAStatus, answer_question


# ----------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------

class _FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class _FakeResponses:
    def __init__(self, output_text=None, raises=None):
        self.requests: list[dict] = []
        self._output_text = output_text
        self._raises = raises

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return _FakeResponse(self._output_text)


class _FakeClient:
    def __init__(self, output_text=None, raises=None):
        self.responses = _FakeResponses(output_text=output_text, raises=raises)


class _TransportBoom(Exception):
    """Stand-in for an SDK / auth / network failure."""


class _FakeBackend:
    def __init__(self, items=()):
        self._items = list(items)

    def search(self, query, *, max_num_results):
        return list(self._items)


def _provider(client) -> ry.YandexKnowledgeQAProvider:
    return ry.YandexKnowledgeQAProvider(client=client, folder_id="test-folder")


def _json_answer(text):
    return json.dumps({"answer": text})


# ================================================================
# 1. PINNED CONFIG + PUBLIC API
# ================================================================

def test_pinned_endpoint_model_and_envelope():
    assert ry.DEFAULT_BASE_URL == "https://ai.api.cloud.yandex.net/v1"
    assert ry.MODEL_ID_TEMPLATE == "gpt://{folder_id}/aliceai-llm-flash/latest"
    assert ry.build_qa_model_id("  my-folder  ") == (
        "gpt://my-folder/aliceai-llm-flash/latest"
    )
    assert ry.ANSWER_FIELD == "answer"
    assert ry.ANSWER_FORMAT_NAME == "angioplus_knowledge_qa_answer"


def test_public_api_surface():
    assert set(ry.__all__) == {
        "YandexKnowledgeQAProvider",
        "build_qa_model_id",
        "DEFAULT_BASE_URL",
        "MODEL_ID_TEMPLATE",
        "DEFAULT_TIMEOUT_SECONDS",
        "DEFAULT_MAX_OUTPUT_TOKENS",
        "ANSWER_FIELD",
        "ANSWER_FORMAT_NAME",
    }


# ================================================================
# 2. REQUEST SHAPE / RAW PASS-THROUGH
# ================================================================

def test_request_is_pinned_and_prompt_is_sent_verbatim():
    prompt = "ПОЛНЫЙ QA промпт\n<<EVIDENCE\nPACS DICOM\nEVIDENCE\n"
    client = _FakeClient(output_text=_json_answer("ответ"))

    assert _provider(client)(prompt) == _json_answer("ответ")

    assert len(client.responses.requests) == 1
    request = client.responses.requests[0]
    assert request == {
        "model": "gpt://test-folder/aliceai-llm-flash/latest",
        "temperature": 0,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": ry.DEFAULT_MAX_OUTPUT_TOKENS,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "angioplus_knowledge_qa_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            }
        },
    }


def test_raw_provider_text_is_returned_unchanged():
    raw = '  {"answer": "  not touched  "}\n'
    client = _FakeClient(output_text=raw)
    assert _provider(client)("prompt") is raw


def test_response_without_output_text_is_a_transport_failure():
    client = _FakeClient(output_text=None)
    with pytest.raises(RuntimeError):
        _provider(client)("prompt")


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None, 42])
def test_blank_prompt_is_refused_without_a_call(blank):
    client = _FakeClient(output_text=_json_answer("x"))
    with pytest.raises(ValueError):
        _provider(client)(blank)
    assert client.responses.requests == []


def test_transport_failure_propagates_unchanged():
    client = _FakeClient(raises=_TransportBoom("429"))
    with pytest.raises(_TransportBoom):
        _provider(client)("prompt")
    assert len(client.responses.requests) == 1


# ================================================================
# 3. LAZY CLIENT (no HTTP)
# ================================================================

def test_real_client_is_built_lazily_once_with_pinned_config(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")

    client = _FakeClient(output_text=_json_answer("ok"))
    built: list[dict] = []

    def _factory(**kwargs):
        built.append(kwargs)
        return client

    monkeypatch.setattr("openai.OpenAI", _factory)

    provider = ry.YandexKnowledgeQAProvider()
    assert built == []  # nothing is built at construction time

    provider("prompt")
    provider("prompt")

    assert built == [
        {
            "api_key": "test-key",
            "base_url": "https://ai.api.cloud.yandex.net/v1",
            "project": "env-folder",
            "timeout": ry.DEFAULT_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
    ]
    assert len(client.responses.requests) == 2


# ================================================================
# 4. END-TO-END THROUGH THE QA ENGINE
# ================================================================

def test_adapter_drives_the_qa_engine_end_to_end():
    client = _FakeClient(output_text=_json_answer("PACS — это архив изображений."))
    backend = _FakeBackend(items=[KnowledgeItem(text="PACS stores DICOM studies.")])

    result = answer_question(
        "Что такое PACS?", backend=backend, provider=_provider(client)
    )

    assert result.status is KnowledgeQAStatus.ANSWERED
    assert result.text == "PACS — это архив изображений."
    assert len(client.responses.requests) == 1
    sent = client.responses.requests[0]["input"][0]["content"]
    assert "Что такое PACS?" in sent
    assert "PACS stores DICOM studies." in sent
