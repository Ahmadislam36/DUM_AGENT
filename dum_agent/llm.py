import time
from functools import lru_cache
from typing import List

from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI

from . import config
from .logger import get_logger

log = get_logger("llm")


@lru_cache(maxsize=1)
def get_llm() -> AzureChatOpenAI:
    required = {
        "AZURE_OPENAI_API_KEY": config.AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": config.AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT": config.AZURE_OPENAI_DEPLOYMENT,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError("Missing in .env: " + ", ".join(missing))

    kwargs = {}
    # Optional: some Azure models (o-series / gpt-5) only accept the default temperature.
    if config.AZURE_OPENAI_TEMPERATURE not in (None, ""):
        kwargs["temperature"] = float(config.AZURE_OPENAI_TEMPERATURE)

    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        max_retries=3,
        **kwargs,
    )


class _LoggedStructured:
    """Wraps a structured-output runnable and logs each call + duration."""

    def __init__(self, runnable, label: str):
        self._runnable, self._label = runnable, label

    def invoke(self, messages):
        log.info("LLM call -> structured output (%s)", self._label)
        start = time.perf_counter()
        try:
            out = self._runnable.invoke(messages)
        except Exception as e:
            log.error("LLM call failed (%s): %s: %s", self._label, type(e).__name__, str(e)[:200])
            raise
        log.info("LLM done in %.1fs (%s)", time.perf_counter() - start, self._label)
        return out


def structured(schema):
    # function_calling works on every Azure OpenAI deployment / API version.
    runnable = get_llm().with_structured_output(schema, method="function_calling")
    return _LoggedStructured(runnable, schema.__name__)


def llm_text(messages: List[BaseMessage]) -> str:
    """Plain LLM call -> string."""
    log.info("LLM call -> plain text")
    start = time.perf_counter()
    try:
        content = get_llm().invoke(messages).content
    except Exception as e:
        log.error("LLM call failed (text): %s: %s", type(e).__name__, str(e)[:200])
        raise
    log.info("LLM done in %.1fs (text)", time.perf_counter() - start)
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ).strip()
    return str(content).strip()
