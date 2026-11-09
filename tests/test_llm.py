import base64
import io

import pytest
from fakes import FakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image
from pydantic import BaseModel

from creative_agent.errors import LLMOutputError
from creative_agent.llm import ask_structured, ask_text, image_part, make_llm, message_text


class Answer(BaseModel):
    n: int


def _need_positive(a: Answer) -> list[str]:
    return [] if a.n > 0 else ["n must be positive"]


META = {"node": "test"}


async def test_returns_on_first_valid_answer():
    llm = FakeChatModel(structured=[{"n": 3}])
    result = await ask_structured(llm, Answer, [HumanMessage(content="hi")], meta=META, validate=_need_positive)
    assert result.n == 3
    assert len(llm.calls) == 1
    assert llm.calls[0].schema is Answer
    assert llm.calls[0].config["metadata"] == META


async def test_repair_message_appended_then_second_answer_returned():
    llm = FakeChatModel(structured=[{"n": 0}, {"n": 5}])
    result = await ask_structured(llm, Answer, [HumanMessage(content="hi")], meta=META, validate=_need_positive)
    assert result.n == 5
    assert len(llm.calls) == 2
    assert len(llm.calls[0].messages) == 1
    second = llm.calls[1].messages
    assert len(second) == 2
    assert "previous answer had these problems" in second[-1].content
    assert "n must be positive" in second[-1].content


async def test_raises_after_repairs_plus_one_failures():
    llm = FakeChatModel(structured=[{"n": 0}, {"n": -1}, {"n": 9}])
    with pytest.raises(LLMOutputError) as e:
        await ask_structured(llm, Answer, [HumanMessage(content="hi")], meta=META, validate=_need_positive)
    assert e.value.errors == ["n must be positive"]
    assert len(llm.calls) == 2          # repairs=1 → two attempts, the third answer is never requested


async def test_repairs_zero_fails_immediately():
    llm = FakeChatModel(structured=[{"n": 0}])
    with pytest.raises(LLMOutputError):
        await ask_structured(llm, Answer, [], meta=META, validate=_need_positive, repairs=0)
    assert len(llm.calls) == 1


async def test_pydantic_validation_error_counts_as_failure():
    llm = FakeChatModel(structured=[{"n": "not a number"}, {"n": 2}])
    result = await ask_structured(llm, Answer, [HumanMessage(content="hi")], meta=META)
    assert result.n == 2
    assert "previous answer had these problems" in llm.calls[1].messages[-1].content


async def test_ask_text_returns_text_and_merges_metadata():
    llm = FakeChatModel(texts=["hello"])
    out = await ask_text(llm, [HumanMessage(content="hi")], meta=META, config={"metadata": {"a": 1}})
    assert out == "hello"
    assert llm.calls[0].config["metadata"] == {"a": 1, "node": "test"}


def test_message_text_string_and_list_content():
    assert message_text(AIMessage(content="abc")) == "abc"
    parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]
    assert message_text(AIMessage(content=parts)) == "ab"
    assert message_text(None) == ""


def test_image_part_downscales_to_max_side(tmp_path):
    p = tmp_path / "big.png"
    Image.new("RGB", (1080, 1920), (64, 199, 6)).save(p)
    part = image_part(p, max_side=1024)
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(im.size) == 1024
    assert im.size == (576, 1024)


def test_image_part_without_max_side_keeps_original_bytes(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (10, 10)).save(p)
    url = image_part(p)["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == p.read_bytes()


def test_make_llm_receives_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
    llm = make_llm("anthropic:claude-3-5-sonnet-latest")
    assert llm.anthropic_api_key.get_secret_value() == "sk-ant-test-key-12345"
