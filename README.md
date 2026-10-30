# Synthetic project intent example

This small Python program sends one fabricated project request to Google's Gemini API and expects a versioned JSON intent. The result is descriptive metadata; it is never executed as code. The local validator rejects malformed JSON, unsupported versions, missing or extra fields, and incorrect types.

## Run without a key

Python 3.10 or newer is sufficient; the program uses only the standard library.

```powershell
python intent_demo.py --demo
python -m unittest -v
```

Demo mode uses fabricated output, makes no network request, and prints `offline-demo` as the model ID. The tests run without an API key.

## Run with Gemini

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey), then save it as a Windows user environment variable so you do not need to enter it each run:

1. Search Windows for **Edit environment variables for your account** and open it.
2. Under **User variables**, choose **New**.
3. Set the variable name to `GEMINI_API_KEY` and its value to your key. Keep the value private; never add it to this project or commit it to Git.
4. Click **OK**, then close and reopen VS Code so it loads the new variable.

In the VS Code terminal, run `python intent_demo.py`. The editor's **Run Python File** action also runs the live API version by default. To run offline instead, use `python intent_demo.py --demo` in the terminal.

### Temporary terminal option

To use the key only in the current PowerShell terminal (without saving it in Windows settings), enter these commands. The secure prompt masks the key as you type. The variable is cleared when you close that terminal.

```powershell
$env:OPENAI_API_KEY = Read-Host "Enter your OpenAI API key"
python intent_demo.py
```

The provider is Gemini's `generateContent` API, called directly over HTTPS with Python's standard library. The default model is `gemini-3.1-flash-lite`. Override it without changing this program using `$env:GEMINI_MODEL = 'model-id'`. The API key is sent in an HTTPS header and is never printed.

The program asks for strict structured JSON, then validates the response locally. It prints the model ID, prompt version (`project-intent-v1`), elapsed request time, input/output token usage if supplied, and an estimated USD cost. Cached token usage is recorded when supplied by the provider.

## Provider and cost assumptions

The selected model is `gemini-3.1-flash-lite`. The local estimate uses Google's published standard paid-tier text rates of **$0.25 per million input tokens** and **$1.50 per million output tokens**. Estimate formula: `(input_tokens × 0.25 + output_tokens × 1.50) / 1,000,000`. Free-tier eligibility and pricing can depend on the account and may change; this estimate assumes paid standard-tier rates, excludes taxes and adjustments, and is not billing data. When usage counts are missing, estimated cost is `null` rather than an invented figure. Check current [model details](https://ai.google.dev/gemini-api/docs/models) and [pricing](https://ai.google.dev/gemini-api/docs/pricing) before relying on the estimate.

The program sends only the synthetic request embedded in the source. It requests a JSON response using Gemini's structured-output configuration, then validates the response locally. Timeout, provider, and malformed-output failures are surfaced as structured error JSON with a nonzero exit code.

