"""Classify a synthetic project request as a locally validated JSON intent."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# Version labels help identify which prompt and output contract produced a result.
PROMPT_VERSION = "project-intent-v1"
SCHEMA_VERSION = "1.0"

# Defaults for the Gemini request and the paid-tier text cost estimate.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 20
INPUT_USD_PER_MILLION = 0.25
OUTPUT_USD_PER_MILLION = 1.50
TOKENS_PER_MILLION = 1_000_000

# This made-up example is sent in live mode; it contains no personal project data.
SYNTHETIC_REQUEST = (
    "Please prepare a one-page project plan for a fictional community garden. "
    "Include a schedule, materials, and three success measures. Do not write code."
)

SYSTEM_PROMPT = f"""You classify synthetic project requests. Return JSON only, matching the schema.
Never return executable code, scripts, shell commands, or instructions to execute code.
The schema_version must be {SCHEMA_VERSION}. Keep the intent concise and descriptive."""

# This contract is checked locally, even when the provider returns structured JSON.
INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
        "intent_type": {
            "type": "string",
            "enum": ["project_plan", "research", "writing", "other"],
        },
        "summary": {"type": "string"},
        "deliverables": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "schema_version",
        "intent_type",
        "summary",
        "deliverables",
        "constraints",
    ],
    "additionalProperties": False,
}


class IntentError(Exception):
    """Base class for errors that can be reported safely to the caller."""


class ProviderError(IntentError):
    """The Gemini API rejected the request or refused to process it."""


class RequestTimeout(IntentError):
    """The Gemini API did not respond before the timeout."""


class InvalidOutput(IntentError):
    """The response could not be parsed or did not match the intent contract."""


@dataclass
class Result:
    """Validated intent and useful metadata for a successful run."""

    intent: dict[str, Any]
    model_id: str
    prompt_version: str
    duration_seconds: float
    usage: dict[str, int] | None
    estimated_cost_usd: float | None


def validate_intent(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate an intent, rejecting missing or unexpected fields."""
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidOutput("provider returned invalid JSON") from exc

    if not isinstance(value, dict):
        raise InvalidOutput("intent must be a JSON object")

    required_fields = set(INTENT_SCHEMA["required"])
    if set(value) != required_fields:
        raise InvalidOutput("intent has missing or unexpected fields")

    if value["schema_version"] != SCHEMA_VERSION:
        raise InvalidOutput("unsupported schema_version")

    allowed_types = INTENT_SCHEMA["properties"]["intent_type"]["enum"]
    if value["intent_type"] not in allowed_types:
        raise InvalidOutput("unsupported intent_type")

    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise InvalidOutput("summary must be a non-empty string")

    for field in ("deliverables", "constraints"):
        items = value[field]
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise InvalidOutput(f"{field} must be an array of non-empty strings")

    return value


def estimate_cost(usage: dict[str, int] | None) -> float | None:
    """Estimate paid-tier cost from token counts, or return None if unavailable."""
    if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
        return None

    input_cost = usage["input_tokens"] * INPUT_USD_PER_MILLION / TOKENS_PER_MILLION
    output_cost = usage["output_tokens"] * OUTPUT_USD_PER_MILLION / TOKENS_PER_MILLION
    return round(input_cost + output_cost, 10)


def _get_gemini_schema() -> dict[str, Any]:
    """Return the provider-compatible schema; local validation stays stricter."""
    schema = dict(INTENT_SCHEMA)
    # Gemini does not accept this keyword, so validate extra fields locally instead.
    schema.pop("additionalProperties", None)
    return schema


def _build_request(request_text: str, api_key: str, model_id: str) -> urllib.request.Request:
    """Create the HTTPS request Gemini's generateContent endpoint expects."""
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": request_text}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _get_gemini_schema(),
        },
    }

    return urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _provider_error_message(error: urllib.error.HTTPError) -> str:
    """Read a short API diagnostic without exposing request headers or the key."""
    try:
        body = json.loads(error.read().decode("utf-8"))
        detail = body.get("error", {}).get("message", "")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        detail = ""

    return " ".join(str(detail).split())[:500]


def _send_request(
    request: urllib.request.Request,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    """Send the API request and return its JSON body plus elapsed seconds."""
    started = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # HTTPError is also a URLError, so catch it first to retain the HTTP status.
        detail = _provider_error_message(exc)
        message = f"Gemini returned HTTP {exc.code}"
        if detail:
            message = f"{message}: {detail}"
        raise ProviderError(message) from exc
    except (TimeoutError, urllib.error.URLError) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
            raise RequestTimeout(f"provider timed out after {timeout:g}s") from exc

        safe_reason = " ".join(str(reason).split())
        raise ProviderError(f"could not reach the model provider: {safe_reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidOutput("provider returned a malformed response envelope") from exc

    duration = time.monotonic() - started
    return body, duration


def _extract_intent_text(response: dict[str, Any]) -> str:
    """Find the generated JSON text or explain why Gemini supplied no text."""
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        feedback = response.get("promptFeedback", {})
        if feedback.get("blockReason"):
            raise ProviderError("Gemini blocked the request") from exc
        raise InvalidOutput("Gemini returned no JSON text") from exc


def _extract_usage(response: dict[str, Any]) -> dict[str, int] | None:
    """Translate Gemini usage fields into the program's common token names."""
    usage = response.get("usageMetadata")
    if not isinstance(usage, dict):
        return None

    if "promptTokenCount" not in usage or "candidatesTokenCount" not in usage:
        return None

    token_counts = {
        "input_tokens": usage["promptTokenCount"],
        "output_tokens": usage["candidatesTokenCount"],
    }
    if "cachedContentTokenCount" in usage:
        token_counts["cached_input_tokens"] = usage["cachedContentTokenCount"]
    return token_counts


def classify_request(
    request: str = SYNTHETIC_REQUEST,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Result:
    """Call Gemini, validate its JSON response, and collect run metadata."""
    # Read the secret from the environment unless a caller supplies it directly.
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ProviderError(
            "GEMINI_API_KEY is not set; use --demo or configure the environment"
        )

    model_id = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    api_request = _build_request(request, key, model_id)
    response, duration = _send_request(api_request, timeout)

    # Validate the model's output before the rest of the program can use it.
    intent_text = _extract_intent_text(response)
    intent = validate_intent(intent_text)

    usage = _extract_usage(response)
    return Result(
        intent=intent,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        duration_seconds=round(duration, 3),
        usage=usage,
        estimated_cost_usd=estimate_cost(usage),
    )


def demo_result() -> Result:
    """Return a fabricated example without credentials or a network request."""
    sample_intent = {
        "schema_version": SCHEMA_VERSION,
        "intent_type": "project_plan",
        "summary": "Plan a fictional community garden project.",
        "deliverables": ["Schedule", "Materials list", "Three success measures"],
        "constraints": ["One page", "No code"],
    }

    return Result(
        intent=validate_intent(sample_intent),
        model_id="offline-demo",
        prompt_version=PROMPT_VERSION,
        duration_seconds=0.0,
        usage=None,
        estimated_cost_usd=None,
    )


def main(argv: list[str] | None = None) -> int:
    """Run demo or live mode, print a JSON report, and return an exit code."""
    args = sys.argv[1:] if argv is None else argv

    try:
        result = demo_result() if "--demo" in args else classify_request()
    except IntentError as exc:
        error = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        print(json.dumps(error))
        return 2

    report = {
        "ok": True,
        "intent": result.intent,
        "model_id": result.model_id,
        "prompt_version": result.prompt_version,
        "duration_seconds": result.duration_seconds,
        "usage": result.usage,
        "estimated_cost_usd": result.estimated_cost_usd,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
