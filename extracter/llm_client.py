from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from threading import Lock
from time import monotonic, sleep
from urllib import error, request

from .configs import LLMConfig


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._last_request_time = 0.0
        self._throttle_lock = Lock()

    async def generate_json(self, request_payload: LLMRequest, *, max_qps: float) -> dict:
        return await asyncio.to_thread(self._generate_json_sync, request_payload, max_qps)

    def _generate_json_sync(self, request_payload: LLMRequest, max_qps: float) -> dict:
        if not self._config.base_url or not self._config.api_key or not self._config.model:
            raise LLMClientError("LLM configuration is incomplete.")
        self._throttle(max_qps)
        url = _resolve_chat_completions_url(self._config.base_url)
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": request_payload.system_prompt},
                {"role": "user", "content": request_payload.user_prompt},
            ],
            "temperature": 0.2,
        }
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._config.timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"LLM request failed: {exc.reason}") from exc

        try:
            payload_json = json.loads(body)
            content = payload_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMClientError("LLM response is not a valid chat completions payload.") from exc

        try:
            return json.loads(_extract_json_object(content))
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM content is not valid JSON.") from exc

    def _throttle(self, max_qps: float) -> None:
        if max_qps <= 0:
            return
        interval = 1.0 / max_qps
        with self._throttle_lock:
            now = monotonic()
            elapsed = now - self._last_request_time
            if elapsed < interval:
                sleep(interval - elapsed)
            self._last_request_time = monotonic()


def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found.", content, 0)
    return content[start : end + 1]
