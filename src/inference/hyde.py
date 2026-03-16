import asyncio
import json
import aiohttp
import requests
from omegaconf import DictConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_HYDE_PROMPT = (
    "Write a detailed passage that directly answers the following question. "
    "Be factual and concise.\n\nQuestion: {query}\n\nPassage:"
)


def _vllm_payload(query: str, cfg: DictConfig) -> dict:
    return {
        "model": cfg.model.hyde_model,
        "prompt": _HYDE_PROMPT.format(query=query),
        "max_tokens": cfg.model.hyde_max_tokens,
        "temperature": 0.0,
    }


def _ollama_payload(query: str, cfg: DictConfig) -> dict:
    return {
        "model": cfg.model.hyde_model.split("/")[-1],  # e.g. "Meta-Llama-3-8B-Instruct"
        "prompt": _HYDE_PROMPT.format(query=query),
        "stream": False,
        "options": {"num_predict": cfg.model.hyde_max_tokens, "temperature": 0},
    }


def generate_hypothetical_doc(query: str, cfg: DictConfig) -> str:
    """
    Generate a hypothetical passage for the query using vLLM (remote) or Ollama (local).
    Falls back to the original query if the server is unavailable or times out.
    """
    is_remote = cfg.environment == "remote"
    url = cfg.model.hyde_server_remote if is_remote else cfg.model.hyde_server_local
    timeout = cfg.model.hyde_timeout

    try:
        if is_remote:
            payload = _vllm_payload(query, cfg)
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"].strip()
        else:
            payload = _ollama_payload(query, cfg)
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

    except Exception as e:
        logger.warning("HyDE failed for query %r: %s — using original query.", query, e)
        return query


async def _generate_one(
    session: aiohttp.ClientSession,
    query: str,
    cfg: DictConfig,
) -> str:
    url = cfg.model.hyde_server_remote
    payload = _vllm_payload(query, cfg)
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=cfg.model.hyde_timeout)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["text"].strip()
    except Exception as e:
        logger.warning("Async HyDE failed for %r: %s", query, e)
        return query


async def generate_hypothetical_docs_batch(
    queries: list[str],
    cfg: DictConfig,
    concurrency: int = 32,
) -> list[str]:
    """
    Async batch HyDE using vLLM's continuous batching.
    Used during evaluation to process 6,980 dev queries efficiently.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(q: str) -> str:
        async with semaphore:
            return await _generate_one(session, q, cfg)

    async with aiohttp.ClientSession() as session:
        tasks = [_bounded(q) for q in queries]
        results = await asyncio.gather(*tasks)

    return list(results)


def generate_hypothetical_docs_sync(
    queries: list[str],
    cfg: DictConfig,
    concurrency: int = 32,
) -> list[str]:
    """Synchronous wrapper around the async batch function."""
    return asyncio.run(generate_hypothetical_docs_batch(queries, cfg, concurrency))
