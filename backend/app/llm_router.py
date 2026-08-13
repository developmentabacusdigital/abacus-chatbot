"""
Abacus Digital Chatbot - LLM Router
OpenRouter multi-model routing with cost tracking and stronger-model fallback (PRD 9.3).
"""

import asyncio
import httpx
import json
import logging
import re
from typing import Optional, Dict, Any, List

from .config import settings, MODEL_ROUTING

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[Any]:
    """
    Pull a JSON object out of a model response.

    Cheap models frequently wrap JSON in prose or code fences, so try, in order:
    the raw string, a fenced block, then the outermost {...} span.
    """
    if not text:
        return None

    candidates = [text.strip()]

    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class LLMRouter:
    """Routes LLM requests to OpenRouter with model selection per task type."""

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MAX_RETRIES = 2

    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self._client: Optional[httpx.AsyncClient] = None
        # Rolling spend counter, surfaced on /health for cost monitoring (PRD 8)
        self.total_cost: float = 0.0
        self.call_count: int = 0

    async def initialize(self):
        """Create the HTTP client."""
        if self._client:
            return
        self.api_key = settings.openrouter_api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://www.abacusdigital.net",
                "X-Title": "Abacus Digital Chatbot",
                "Content-Type": "application/json",
            },
        )

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def configured(self) -> bool:
        """True only for a key that could plausibly work — a placeholder is not one."""
        key = (self.api_key or "").strip()
        return bool(key) and not key.startswith("your_") and key.lower() != "changeme"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        task_type: str = "general",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a response using the appropriate model for the task.

        Returns: {"content", "model_used", "cost", "usage", "ok"}
        """
        routing = MODEL_ROUTING.get(task_type, MODEL_ROUTING["general"])

        if not self.configured:
            logger.error("OPENROUTER_API_KEY is not set; cannot call any model")
            return self._error_result()

        for model in (routing["primary"], routing["fallback"]):
            result = await self._call_model(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            if result and result.get("content", "").strip():
                return result
            logger.warning(f"Model {model} produced no usable output for task '{task_type}'")

        return self._error_result()

    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        task_type: str = "general",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate and parse a JSON object, escalating to the fallback model if the
        cheap model returns malformed output (PRD 9.3 fallback logic).

        Returns: {"data": dict|None, "model_used", "cost", "raw"}
        """
        routing = MODEL_ROUTING.get(task_type, MODEL_ROUTING["general"])
        total_cost = 0.0

        if not self.configured:
            return {"data": None, "model_used": "error", "cost": 0.0, "raw": ""}

        for model in (routing["primary"], routing["fallback"]):
            result = await self._call_model(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            if not result:
                continue
            total_cost += result.get("cost", 0.0)
            parsed = extract_json(result.get("content", ""))
            if isinstance(parsed, dict):
                return {
                    "data": parsed,
                    "model_used": model,
                    "cost": total_cost,
                    "raw": result.get("content", ""),
                }
            logger.warning(f"{model} returned non-JSON for task '{task_type}', escalating")

        return {"data": None, "model_used": "error", "cost": total_cost, "raw": ""}

    def _error_result(self) -> Dict[str, Any]:
        return {
            "content": (
                "I'm having trouble processing your request right now. Please try again in a "
                "moment, or reach out to our team directly at https://www.abacusdigital.net/contact."
            ),
            "model_used": "error",
            "cost": 0.0,
            "usage": {},
            "ok": False,
        }

    async def _call_model(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Make a single API call to OpenRouter, retrying transient failures."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Ask OpenRouter to return real spend so cost-per-conversation is accurate
            "usage": {"include": True},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self._client.post(self.OPENROUTER_API_URL, json=payload)

                # Rate limited or upstream hiccup: back off once, then give up on this model
                if response.status_code in (429, 502, 503, 504):
                    if attempt + 1 < self.MAX_RETRIES:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    logger.error(f"{model} unavailable: HTTP {response.status_code}")
                    return None

                response.raise_for_status()
                data = response.json()

                if data.get("error"):
                    logger.error(f"OpenRouter error for {model}: {data['error']}")
                    return None

                choices = data.get("choices") or []
                if not choices:
                    logger.error(f"No choices in response from {model}: {data}")
                    return None

                content = (choices[0].get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                cost = float(usage.get("cost", 0.0) or 0.0)

                self.total_cost += cost
                self.call_count += 1

                return {
                    "content": content,
                    "model_used": model,
                    "cost": cost,
                    "usage": usage,
                    "ok": True,
                }

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error from OpenRouter ({model}): "
                    f"{e.response.status_code} - {e.response.text[:300]}"
                )
                return None
            except httpx.TimeoutException:
                logger.error(f"Timeout calling OpenRouter ({model}) attempt {attempt + 1}")
                if attempt + 1 >= self.MAX_RETRIES:
                    return None
            except Exception as e:
                logger.error(f"Unexpected error calling OpenRouter ({model}): {e}")
                return None

        return None

    async def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        task_type: str = "general",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Stream a response from OpenRouter (for real-time chat).
        Yields content chunks as they arrive.
        """
        routing = MODEL_ROUTING.get(task_type, MODEL_ROUTING["general"])
        model = routing["primary"]

        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            async with self._client.stream(
                "POST", self.OPENROUTER_API_URL, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Streaming error from OpenRouter ({model}): {e}")
            yield "I'm having trouble right now. Please try again."


# Singleton
llm_router = LLMRouter()
