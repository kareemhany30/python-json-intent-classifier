"""Ask a model to classify a synthetic project request into a versioned intent."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# These values identify the prompt/output formats and set the default API behavior.
# Cost rates are USD per million text tokens for the selected Gemini model.
PROMPT_VERSION = "project-intent-v1"
SCHEMA_VERSION = "1.0"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 20
INPUT_USD_PER_MILLION = 0.25
OUTPUT_USD_PER_MILLION = 1.50

# This made-up request is sent in live mode; it contains no personal or project data.
SYNTHETIC_REQUEST = (
    "Please prepare a one-page project plan for a fictional community garden. "
    "Include a schedule, materials, and three success measures. Do not write code."
)

SYSTEM_PROMPT = f"""You classify synthetic project requests. Return JSON only, matching the schema.
Never return executable code, scripts, shell commands, or instructions to execute code.
The schema_version must be {SCHEMA_VERSION}. Keep the intent concise and descriptive."""

# This is the contract for the JSON intent. Local validation uses the full schema;
# the API request below omits one keyword Gemini does not support.
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
    "required": ["schema_version", "intent_type", "summary", "deliverables", "constraints"],
    "additionalProperties": False,
}


# Named errors let main() report failures as readable JSON with a nonzero exit code.
class IntentError(Exception):
    """Base class for errors that can be reported safely to the caller."""


class ProviderError(IntentError):
    pass


class RequestTimeout(IntentError):
    pass


class InvalidOutput(IntentError):
    pass


@dataclass
class Result:
    """Data printed after a successful API or offline demo run."""
    intent: dict[str, Any]
    model_id: str
    prompt_version: str
    duration_seconds: float
    usage: dict[str, int] | None
    estimated_cost_usd: float | None


def validate_intent(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse and locally validate the versioned intent; reject all extra fields."""
    try:
        # API output arrives as text; tests and the demo can pass a dictionary.
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidOutput("provider returned invalid JSON") from exc

    # Check every field locally, even though the API was asked to follow a schema.
    if not isinstance(value, dict):
        raise InvalidOutput("intent must be a JSON object")
    expected = set(INTENT_SCHEMA["required"])
    if set(value) != expected:
        raise InvalidOutput("intent has missing or unexpected fields")
    if value["schema_version"] != SCHEMA_VERSION:
        raise InvalidOutput("unsupported schema_version")
    if value["intent_type"] not in INTENT_SCHEMA["properties"]["intent_type"]["enum"]:
        raise InvalidOutput("unsupported intent_type")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise InvalidOutput("summary must be a non-empty string")
    for field in ("deliverables", "constraints"):
        if not isinstance(value[field], list) or any(
            not isinstance(item, str) or not item.strip() for item in value[field]
        ):
            raise InvalidOutput(f"{field} must be an array of non-empty strings")
    return value
def estimate_cost(usage: dict[str, int] | None) -> float | None:
    """Estimate the paid-tier cost from provider token counts, if available."""
    if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    return round(
        usage["input_tokens"] * INPUT_USD_PER_MILLION / 1_000_000
        + usage["output_tokens"] * OUTPUT_USD_PER_MILLION / 1_000_000,
        10,
    )


def classify_request(
    request: str = SYNTHETIC_REQUEST,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Result:
    """Send the synthetic request to Gemini and return only locally checked data."""
    # Prefer an explicitly passed key (useful for callers/tests), otherwise read
    # it from the environment so no secret needs to be stored in this source file.
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set; use --demo or configure the environment")
    model_id = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    # Gemini's constrained schema subset does not accept additionalProperties;
    # local validation below still rejects unexpected fields in the response.
    gemini_schema = dict(INTENT_SCHEMA)
    gemini_schema.pop("additionalProperties", None)
    # Ask Gemini to produce JSON matching the response schema, without giving it tools.
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": request}]},
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": gemini_schema,
        },
    }
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    # Measure the network/API time and keep the key in an HTTPS header, not the URL.
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # HTTPError is also a URLError, so handle it first to preserve status codes.
        # Gemini's error envelope helps diagnose invalid requests; never print headers.
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            detail = error_body.get("error", {}).get("message", "")
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            detail = ""
        detail = " ".join(str(detail).split())[:500]
        suffix = f": {detail}" if detail else ""
        raise ProviderError(f"Gemini returned HTTP {exc.code}{suffix}") from exc
    except (TimeoutError, urllib.error.URLError) as exc:
        if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
            raise RequestTimeout(f"provider timed out after {timeout:g}s") from exc
        reason = getattr(exc, "reason", exc)
        # Include the transport cause (often DNS, proxy, or firewall) without secrets.
        safe_reason = str(reason).replace("\r", " ").replace("\n", " ")
        raise ProviderError(f"could not reach the model provider: {safe_reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidOutput("provider returned a malformed response envelope") from exc

    # Read the model's text, then validate it before the program accepts or prints it.
    duration = time.monotonic() - started
    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        prompt_feedback = body.get("promptFeedback", {})
        if prompt_feedback.get("blockReason"):
            raise ProviderError("Gemini blocked the request") from exc
        raise InvalidOutput("Gemini returned no JSON text") from exc
    intent = validate_intent(text)
    # Gemini may report token usage. If it does, save it for the cost estimate/output.
    raw_usage = body.get("usageMetadata")
    usage = None
    if isinstance(raw_usage, dict):
        if "promptTokenCount" in raw_usage and "candidatesTokenCount" in raw_usage:
            usage = {
                "input_tokens": raw_usage["promptTokenCount"],
                "output_tokens": raw_usage["candidatesTokenCount"],
            }
            if "cachedContentTokenCount" in raw_usage:
                usage["cached_input_tokens"] = raw_usage["cachedContentTokenCount"]
    return Result(intent, model_id, PROMPT_VERSION, round(duration, 3), usage, estimate_cost(usage))

def demo_result() -> Result:
    """Offline demonstration uses fabricated output and no credentials/network."""
    # Fixed sample data makes it possible to inspect the output without an API key.
    raw = {
        "schema_version": SCHEMA_VERSION,
        "intent_type": "project_plan",
        "summary": "Plan a fictional community garden project.",
        "deliverables": ["Schedule", "Materials list", "Three success measures"],
        "constraints": ["One page", "No code"],
    }
    return Result(validate_intent(raw), "offline-demo", PROMPT_VERSION, 0.0, None, None)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        # --demo stays offline; without it, the program makes a live Gemini request.
        result = demo_result() if "--demo" in args else classify_request()
    except IntentError as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}))
        return 2
    # Print one JSON report with the intent and metadata for this run.
    print(json.dumps({
        "ok": True,
        "intent": result.intent,
        "model_id": result.model_id,
        "prompt_version": result.prompt_version,
        "duration_seconds": result.duration_seconds,
        "usage": result.usage,
        "estimated_cost_usd": result.estimated_cost_usd,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
