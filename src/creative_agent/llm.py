"""LLM helpers: model construction, image message parts, and structured-output calls with a repair loop."""
import base64
import io
import mimetypes
from pathlib import Path
from typing import Callable, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image
from pydantic import BaseModel, ValidationError

from .errors import LLMOutputError

# Pydantic schema type returned by ask_structured.
T = TypeVar("T", bound=BaseModel)


def make_llm(model: str) -> BaseChatModel:
    """Create a chat model from a ``"provider:model"`` string (e.g. ``"anthropic:claude-sonnet-5-5"``)."""
    return init_chat_model(model, max_tokens=16384)   # the baseline returns five ads in one answer


def image_part(path: Path, max_side: int | None = None) -> dict:
    """An image message part. Downscale first when max_side is given, so coordinates refer to a size we know.

    Args:
        path: Image file on disk.
        max_side: If set, shrink the image (aspect ratio kept) so neither side exceeds this many pixels, and
            re-encode as PNG. If ``None``, the original bytes are sent untouched.

    Returns:
        A LangChain ``image_url`` content block holding a base64 data URI.
    """
    if max_side:
        with Image.open(path) as im:
            im = im.convert("RGB")           # drop alpha / palette modes so PNG encoding is predictable
            im.thumbnail((max_side, max_side))   # in-place, preserves aspect ratio, never upscales
            buf = io.BytesIO()
            im.save(buf, "PNG")
            data, mime = buf.getvalue(), "image/png"
    else:
        data, mime = Path(path).read_bytes(), mimetypes.guess_type(str(path))[0] or "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def message_text(msg: BaseMessage | None) -> str:
    """Extract plain text from a model reply.

    Message ``content`` is either a string or a list of content blocks (text, tool use, ...); in the latter
    case only the ``text`` of dict blocks is concatenated. ``None`` yields an empty string.
    """
    if msg is None:
        return ""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict))


def _config(config: RunnableConfig | None, meta: dict) -> RunnableConfig:
    """Copy ``config`` and merge ``meta`` into its metadata (used by tracing/cost callbacks to attribute calls)."""
    cfg: dict = dict(config or {})
    cfg["metadata"] = {**cfg.get("metadata", {}), **meta}
    return cfg  # type: ignore[return-value]


async def ask_text(llm: BaseChatModel, messages: list, *, meta: dict, config: RunnableConfig | None = None) -> str:
    """Invoke the model and return its reply as plain text.

    Args:
        llm: Chat model to call.
        messages: LangChain messages to send.
        meta: Metadata (e.g. node name, size) attached to the run for tracing and cost attribution.
        config: Optional parent runnable config (callbacks etc.) to propagate.
    """
    return message_text(await llm.ainvoke(messages, config=_config(config, meta)))


async def ask_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list,
    *,
    meta: dict,
    config: RunnableConfig | None = None,
    validate: Callable[[T], list[str]] | None = None,
    repairs: int = 1,
) -> T:
    """Structured output plus domain validation. On problems, retry once with the problems listed.

    Args:
        llm: Chat model to call.
        schema: Pydantic model describing the required answer shape.
        messages: LangChain messages to send.
        meta: Metadata attached to every call for tracing and cost attribution.
        config: Optional parent runnable config to propagate.
        validate: Optional domain check run on the parsed result; returns a list of problem descriptions
            (empty list = valid). Use it for rules the schema cannot express.
        repairs: How many extra attempts to allow after the first one fails.

    Returns:
        A valid instance of ``schema``.

    Raises:
        LLMOutputError: If the answer is still invalid after the first attempt plus ``repairs`` retries.
    """
    msgs = list(messages)
    errors: list[str] = []
    for _ in range(repairs + 1):
        try:
            result = await llm.with_structured_output(schema).ainvoke(msgs, config=_config(config, meta))
            errors = validate(result) if validate else []
            if not errors:
                return result
        except ValidationError as e:
            # The model's output didn't even parse into the schema; feed the parser error back to it.
            errors = [str(e)]
        # Append the problems so the next attempt can correct them (capped to keep the prompt small).
        msgs = msgs + [HumanMessage(content="Your previous answer had these problems:\n- "
                                    + "\n- ".join(errors[:20]) + "\nReturn a corrected answer.")]
    raise LLMOutputError(errors)
