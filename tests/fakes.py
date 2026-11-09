"""Test doubles for chat models. Duck-typed; no LangChain subclassing needed by project code."""
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel


@dataclass
class Call:
    kind: str                       # "text" or "structured"
    messages: list
    config: Any
    schema: type | None = None


@dataclass
class FakeChatModel:
    texts: list[str] = field(default_factory=list)
    structured: list[Any] = field(default_factory=list)   # dict | BaseModel | Exception
    calls: list[Call] = field(default_factory=list)

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls.append(Call("text", list(messages), config))
        return AIMessage(content=self.texts.pop(0))

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructured(self, schema)


class _FakeStructured:
    def __init__(self, parent: FakeChatModel, schema: type[BaseModel]):
        self.parent = parent
        self.schema = schema

    async def ainvoke(self, messages, config=None, **kwargs):
        self.parent.calls.append(Call("structured", list(messages), config, self.schema))
        item = self.parent.structured.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return self.schema.model_validate(item)
        return item
