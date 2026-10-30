# Synthetic project intent example

This small Python program sends one fabricated project request to OpenAI and expects a versioned JSON intent. The result is descriptive metadata; it is never executed as code. The local validator rejects malformed JSON, unsupported versions, missing or extra fields, and incorrect types.

## Run without a key

Python 3.10 or newer is sufficient; the program uses only the standard library.

```powershell
python intent_demo.py --demo
python -m unittest -v
```

Demo mode uses fabricated output, makes no network request, and prints `offline-demo` as the model ID. The tests run without an API key.

## Run with OpenAI

Create an API key in your OpenAI account, then set it in the shell environment. Do not paste it into this source tree.

```powershell
$env:OPENAI_API_KEY = 'your-key'
python intent_demo.py
```

The default provider is OpenAI's Responses API and the default model is `gpt-6-luna`. Override the model without changing this program using `$env:OPENAI_MODEL = 'model-id'`. Set `OPENAI_API_KEY` through your platform's secret manager or shell environment.

The program asks for strict structured JSON, then validates the response locally. It prints the model ID, prompt version (`project-intent-v1`), elapsed request time, input/output token usage if supplied, and an estimated USD cost. Cached token usage is recorded when supplied by the provider.

## Provider and cost assumptions

The selected model is `gpt-6-luna`, OpenAI's efficient model for focused, high-volume tasks. The local estimate uses the published standard text rates of **$0.05 per million input tokens** and **$0.25 per million output tokens**. Estimate formula: `(input_tokens × 0.05 + output_tokens × 0.25) / 1,000,000`. It is an estimate, not billing data; it excludes taxes, nonstandard processing, and any account-specific adjustments. When usage counts are missing, estimated cost is `null` rather than an invented figure. Check current [model details](https://developers.openai.com/api/docs/models/gpt-6-luna) and [pricing](https://developers.openai.com/api/docs/pricing) before relying on the estimate.

The API call uses `store: false`; it sends only the synthetic request embedded in the program. Timeout, provider, and malformed-output failures are surfaced as structured error JSON with a nonzero exit code.

