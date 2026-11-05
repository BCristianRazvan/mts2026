# 05 — Cloud LLM After Local STT (Group 2)

**Difficulty:** Easy–Medium
**Estimated effort:** 3–4 sessions of about 3 hours
**Team size:** 6 students
**Goal:** Keep the existing offline Sherpa-ONNX speech-to-text model. Send the transcripts produced on the phone to Claude (in Microsoft Foundry), let Claude correct transcription errors, and prove with measurements whether the corrected text is better than the phone text.

Read the [workshop README](README.md) first. It describes the test set, the tools, the result files and the rules.

## Objective

In this scenario you do not replace or retrain the STT model. The existing model continues to convert speech into text on the phone. A PC script then sends the recognized text to Claude, which is asked to correct transcription errors without changing the speaker's intended meaning.

You compare the phone transcript and the Claude transcript against the reference transcript in `transcripts.tsv`. A fluent rewrite is not necessarily a more accurate transcript, so the reference decides, not your opinion of which text sounds better.

```text
WAV file -> app on the phone (local STT) -> phone transcript ---------------------> WER against the reference
                                                   |
                                                   +-> PC script -> Claude -> Claude text -> WER against the reference
```

The Android app itself does not change. It keeps working on its own and offline: open it, tap **Start listening**, speak, and the text appears. Claude is called only from the PC script, so no API key is ever inside the app.

## Starting point

- The Android app and its local Sherpa-ONNX STT model.
- The test set in `docs/stt-model-replacement/testset/`: 10 hard synthetic WAV files and the reference text in `transcripts.tsv`.
- The tools: `device_benchmark.py` (benchmark on the phone) and `compare_runs.py` (comparison and verdict).
- A Microsoft Foundry resource with a Claude deployment, prepared by the organizer, and its endpoint and API key.
- Network access from the PC to the Foundry endpoint.

## Roles

Work in three pairs. Every pair takes part in every stage, but each pair owns one area.

| Pair | Owns | Main tasks |
|---|---|---|
| A — Benchmark | Measurements and results | Runs `device_benchmark.py` and `compare_runs.py`, keeps `results/group2/`, writes `REPORT.md` |
| B — Claude integration | The Claude script | Creates and runs `claude_correct.py`, handles errors, measures latency and cost |
| C — Review | Meaning review and failure tests | Reviews every changed transcript, runs the failure tests, checks the app on the phone |

## Work plan

### Stage 1 — Test the current application

Owner: pair A.

1. Create the group branch `group2/work` from `main` (README rule 5). Agree on the limits for the verdict (README, *Compare two runs*) and write them at the top of `REPORT.md`.
2. Build and install the unmodified debug app on the test phone (root README, sections 6–8).
3. Check normal use: speak two sentences into the app and check that the text appears.
4. Run the benchmark on the phone:
   ```powershell
   uv run --no-project --python 3.12 --with jiwer `
       python docs/stt-model-replacement/tools/device_benchmark.py `
       --serial <phone-serial> --run phone `
       --out-dir docs/stt-model-replacement/results/group2
   ```
5. Check that you get the setup-check result from the README (35.3% WER).

The command stores the phone transcripts in `results/group2/phone.tsv` (column `hypothesis`) and the phone measurements in `results/group2/phone.summary.tsv`. Do not edit these files: they are the input for every Claude run, so every run uses the same text.

The current model makes errors like these on the test set: `devil particulate filter` for `diesel particulate filter`, `crews control` for `cruise control`, `break hard` for `brake hard`, `kilobots` for `kilowatts`. Some of them can be corrected from the context, and some cannot (place names that the model did not recognize at all).

### Stage 2 — Add the code that connects to Claude

Owner: pair B, on the group branch `group2/work`. Pair B can do 2a and 2d already in session 1, while pair A runs Stage 1.

#### 2a. Claude access in Microsoft Foundry

The organizer creates a Microsoft Foundry resource, deploys a Claude model in it, and gives the group two values outside of git (for example in a private message):

| Setting | Example | Where to find it |
|---|---|---|
| Endpoint | `https://<resource-name>.services.ai.azure.com/anthropic/` | Foundry portal, the resource, **Keys and Endpoint** |
| API key | (secret) | Same page |
| Deployment name | `claude-opus-5-5` | The name of the Claude deployment. It is often the same as the model name. |

Set the values as environment variables in the PowerShell window where you run the script. Copy the endpoint exactly as the organizer sends it, including the final `/`:

```powershell
$env:ANTHROPIC_FOUNDRY_BASE_URL = "https://<resource-name>.services.ai.azure.com/anthropic/"
$env:ANTHROPIC_FOUNDRY_API_KEY = "<key from the organizer>"
$env:CLAUDE_MODEL = "claude-opus-5-5"   # the deployment name
```

The variables exist only in this window. In every new PowerShell window, set them again. Check them with `echo $env:CLAUDE_MODEL` and `echo $env:ANTHROPIC_FOUNDRY_BASE_URL`, but never `echo` the key, so that it does not appear on screen or in a screenshot. Never write the key into a file in the repository.

#### 2b. The prompt

Use this prompt, exactly as written, as the system prompt:

> Correct transcription errors only. Preserve the original meaning. Do not add information or infer words that are not supported by the input.

Do not change it during the measurements. If you want to try a variant, do it after the main measurement, save it under a new run name, and report it separately.

#### 2c. Suggested implementation: `tools/claude_correct.py`

This is one suggestion. You may write your own version, but it must read the same input, write the same output columns, and use the same prompt.

The script uses the official Anthropic Python SDK (`anthropic`) and its Microsoft Foundry client, `AnthropicFoundry`. The client reads the endpoint from `ANTHROPIC_FOUNDRY_BASE_URL` and the key from `ANTHROPIC_FOUNDRY_API_KEY`, and calls the Claude Messages API at `<endpoint>v1/messages`.

Create the file `docs/stt-model-replacement/tools/claude_correct.py` with the content below. Copy the whole block and save it as UTF-8 (VS Code does this by default). Adding this one new file to `tools/` is allowed; do not edit the four existing tools.

```python
"""Send each phone transcript to Claude on Microsoft Foundry and save the corrected text.

Reads a result TSV from the phone benchmark (columns `file` and `hypothesis`),
sends every transcript to Claude, and writes a new TSV with the columns
`file`, `stt`, `hypothesis` (Claude's text), `latency_ms`, `input_tokens`,
`output_tokens` and `status`. If a request fails, the phone text is kept and
`status` says why, so the comparison never loses a file.

Set these environment variables first (never commit them):

    ANTHROPIC_FOUNDRY_BASE_URL  https://<resource>.services.ai.azure.com/anthropic/
    ANTHROPIC_FOUNDRY_API_KEY   key of the Foundry resource (Keys and Endpoint)
    CLAUDE_MODEL                deployment name (default: claude-opus-5-5)

Example (from the repository root):

    uv run --no-project --python 3.12 --with anthropic \
        python docs/stt-model-replacement/tools/claude_correct.py \
        docs/stt-model-replacement/results/group2/phone.tsv \
        docs/stt-model-replacement/results/group2/claude.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import anthropic
from anthropic import AnthropicFoundry, BetaFallbackState, BetaRefusalFallbackMiddleware

# The prompt given by the workshop. Do not change it during the measurements.
SYSTEM_PROMPT = (
    "Correct transcription errors only. Preserve the original meaning. "
    "Do not add information or infer words that are not supported by the input."
)

# Structured output: Claude must answer with {"corrected_text": "..."}.
# This keeps explanations or quotes out of the text, which would count as WER errors.
OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"corrected_text": {"type": "string"}},
        "required": ["corrected_text"],
        "additionalProperties": False,
    },
}


def create_client() -> AnthropicFoundry:
    """Reads ANTHROPIC_FOUNDRY_BASE_URL and ANTHROPIC_FOUNDRY_API_KEY from the environment."""
    # If Claude declines a request (stop_reason "refusal"), the SDK retries it once on
    # the fallback model. The fallback model must also be deployed in the Foundry resource.
    fallback = os.environ.get("CLAUDE_FALLBACK_MODEL", "claude-opus-4-8")
    return AnthropicFoundry(
        timeout=30.0,
        max_retries=2,  # retries network errors, 429 and 5xx automatically
        middleware=[BetaRefusalFallbackMiddleware([{"model": fallback}])],
    )


def correct(client: AnthropicFoundry, model: str, text: str) -> dict:
    """Sends one transcript to Claude and returns the corrected text and usage."""
    start = time.perf_counter()
    with (
        BetaFallbackState()
    ):  # one state per transcript: each request is its own conversation
        response = client.beta.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            # Low effort: a short correction task does not need deep reasoning.
            output_config={"effort": "low", "format": OUTPUT_FORMAT},
        )
    latency_ms = (time.perf_counter() - start) * 1000
    if response.stop_reason == "refusal":
        raise ValueError("Claude declined the request")
    if response.stop_reason == "max_tokens":
        raise ValueError("answer was cut off (max_tokens)")
    answer = next(block.text for block in response.content if block.type == "text")
    corrected = json.loads(answer)["corrected_text"].strip()
    if not corrected:
        raise ValueError("empty answer")
    return {
        "text": corrected,
        "latency_ms": latency_ms,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "phone_tsv", type=Path, help="Result TSV from the phone benchmark."
    )
    parser.add_argument("out_tsv", type=Path)
    args = parser.parse_args()

    model = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
    try:
        client = create_client()
    except (anthropic.AnthropicError, ValueError) as error:
        raise SystemExit(f"Cannot create the Claude client: {error}") from None

    if args.out_tsv.exists():
        raise SystemExit(f"{args.out_tsv} already exists. Choose a new output name.")

    # utf-8-sig: also reads files that PowerShell saved with a BOM.
    with args.phone_tsv.open(encoding="utf-8-sig", newline="") as handle:
        phone_rows = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for row in phone_rows:
        original = row["hypothesis"]
        result = {
            "text": original,
            "latency_ms": 0,
            "input_tokens": "",
            "output_tokens": "",
        }
        status = "ok"
        if not original:
            status = "skipped: empty transcript"
        else:
            try:
                result = correct(client, model, original)
            except anthropic.AuthenticationError:
                status = "error 401: wrong API key"
            except anthropic.PermissionDeniedError:
                status = "error 403: key has no access"
            except anthropic.NotFoundError:
                status = f"error 404: deployment '{model}' not found"
            except anthropic.RateLimitError:
                status = "error 429: rate limit (retries used up)"
            except anthropic.APIStatusError as error:
                status = f"error {error.status_code}"
            except anthropic.APIConnectionError as error:  # includes timeouts
                status = f"network error: {error}"
            except (ValueError, KeyError, StopIteration) as error:
                status = f"bad answer: {error}"
        rows.append(
            {
                "file": row["file"],
                "stt": original,
                "hypothesis": result["text"],
                "latency_ms": f"{result['latency_ms']:.0f}",
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "status": status,
            }
        )
        print(f"{row['file']}  {status}  {result['latency_ms']:.0f} ms")
        print(f"   phone : {original}")
        print(f"   claude: {result['text']}")

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(1 for row in rows if row["status"] != "ok")
    print(f"\nSaved: {args.out_tsv}  ({len(rows)} files, {failed} failed)")


if __name__ == "__main__":
    main()
```

What each part does:

- **`SYSTEM_PROMPT`:** the workshop prompt, sent as the `system` parameter. The phone transcript is the only user message.
- **`OUTPUT_FORMAT` (structured output):** Claude must answer with JSON `{"corrected_text": "..."}`. Without it, Claude may add a sentence such as "Here is the corrected text:", and every added word counts as a WER error.
- **`effort: "low"`:** correcting a short sentence needs little reasoning. Low effort keeps latency and cost down. `claude-opus-5-5` does not accept `temperature`, so the script does not send it.
- **`BetaRefusalFallbackMiddleware`:** if Claude declines a request (`stop_reason` `refusal`), the SDK sends it once more to a fallback model (`CLAUDE_FALLBACK_MODEL`, default `claude-opus-4-8`). The fallback model must also be deployed in the Foundry resource. Otherwise a declined request ends with the status `error 404`, and the phone text is kept.
- **`max_retries=2`, `timeout=30.0`:** the SDK retries network errors, HTTP 429 and HTTP 5xx automatically, and stops waiting after 30 seconds.
- **Error handling:** every failure ends in the `status` column, and the phone text is kept in `hypothesis`. The script never stops in the middle, so the comparison always has all 10 files.
- **Usage:** `input_tokens` and `output_tokens` come from `response.usage`. `compare_runs.py` uses them to calculate the cost.

#### 2d. Test the connection with one file

Before you run all files, test the connection with one file. In PowerShell, from the repository root:

```powershell
Get-Content docs\stt-model-replacement\results\group2\phone.tsv -TotalCount 2 |
    Set-Content docs\stt-model-replacement\results\group2\one.tsv

uv run --no-project --python 3.12 --with anthropic `
    python docs/stt-model-replacement/tools/claude_correct.py `
    docs/stt-model-replacement/results/group2/one.tsv `
    docs/stt-model-replacement/results/group2/test-one.tsv
```

`one.tsv` contains the header and the first file. Check that the `status` printed for `test_01.wav` is `ok`. `phone.tsv` stays unchanged.

### Stage 3 — Send the phone transcripts to Claude

Owner: pair B.

```powershell
uv run --no-project --python 3.12 --with anthropic `
    python docs/stt-model-replacement/tools/claude_correct.py `
    docs/stt-model-replacement/results/group2/phone.tsv `
    docs/stt-model-replacement/results/group2/claude.tsv
```

The script prints the phone text and the Claude text of every file, and stores the results in `results/group2/claude.tsv` with the columns `file`, `stt` (phone text), `hypothesis` (Claude text), `latency_ms`, `input_tokens`, `output_tokens` and `status`.

If any `status` is not `ok`, find the cause and run the script again with a new output name (for example `claude-2.tsv`). The script refuses an output name that already exists.

Run the script a second time with the output `claude-repeat.tsv`, and compare the two Claude runs:

```powershell
uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/compare_runs.py `
    docs/stt-model-replacement/results/group2/claude.tsv `
    docs/stt-model-replacement/results/group2/claude-repeat.tsv `
    --label-a run1 --label-b run2
```

Here, ignore the verdict. Read only the line `Files improved / unchanged / worse` and the numbers of errors fixed and introduced. If all files are unchanged and both numbers are 0, Claude gave the same result twice. Report how many files changed between the two runs.

**Failure tests (pair C).** The script must keep the phone text and report the problem in every case. Test each case with `one.tsv` from 2d and a separate output file, `fail-<case>.tsv` (for example `fail-key.tsv`). After each test, set the key and `CLAUDE_MODEL` back to the correct values. Record the `status`:

| Case | How to create it | Expected `status` |
|---|---|---|
| Wrong key | Change one character of `ANTHROPIC_FOUNDRY_API_KEY` | `error 401: wrong API key` |
| Wrong deployment name | Set `CLAUDE_MODEL` to a name that does not exist | `error 404: deployment '...' not found` |
| No network | Disconnect the PC from the network | `network error: ...` (takes up to about 2 minutes, because the SDK retries twice with a 30 s timeout) |
| Empty transcript | In a copy of `one.tsv` (`empty.tsv`), delete the text in the `hypothesis` column | `skipped: empty transcript` |

### Stage 4 — Compare the phone transcript with the Claude transcript

Owner: whole group.

```powershell
uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/compare_runs.py `
    docs/stt-model-replacement/results/group2/phone.tsv `
    docs/stt-model-replacement/results/group2/claude.tsv `
    --label-a phone --label-b claude
```

The script prints the comparison in the terminal and writes `results/group2/compare_phone_vs_claude.html`. It shows:

- the WER of the phone text and of the Claude text, both against the reference;
- the number of files improved, unchanged and worse, the errors fixed and the errors introduced by Claude, and the files that were correct before and that Claude broke;
- the median and maximum latency, the failed requests, and the cost per request and per 1,000 requests (default prices for `claude-opus-5-5`: USD 4 per 1M input tokens and USD 20 per 1M output tokens; check the current Foundry prices and change them with `--price-in` and `--price-out`);
- the verdict: Claude is better only if its WER is at least 1 point lower, the median latency is below 3,000 ms, **and** no request failed. If a request failed, fix the cause and run again with a new output name.

**Meaning review (pair C).** WER does not show every problem. A changed number or command can have a small WER effect but a large effect for the driver. Two students review every file where Claude changed the text, independently, and then compare their answers:

| File | Phone WER | Claude WER | Words changed by Claude | Meaning preserved? | Information added? | Number or command changed? |
|---|---:|---:|---|---|---|---|
| `test_01.wav` | | | | | | |
| ... | | | | | | |

How to fill the review table:

- **Words changed by Claude:** the words that differ between the phone text and the Claude text. The *Transcripts* section of the HTML page shows both texts with the wrong words marked.
- **Meaning preserved?** yes or no. **Information added?** and **Number or command changed?** yes or no.
- A file has a **changed meaning** if either reviewer answered *no* to the first question or *yes* to the second or third question.

The verdict of `compare_runs.py` does not check the meaning. If any file has a changed meaning, your recommendation is "adopt with conditions" or "reject", never a plain "adopt".

In `REPORT.md`, fill this table:

| Metric | Phone only | Phone + Claude |
|---|---:|---:|
| WER (%) | | |
| Files improved / unchanged / worse | n/a | |
| Errors fixed / introduced | n/a | |
| Files with a changed meaning (review) | n/a | |
| Extra latency, median (ms) | 0 | |
| Failed requests | n/a | |
| Cost per request (USD) | 0 | |
| Works offline | Yes | No: needs the network |
| **Verdict of `compare_runs.py`** | | |

Then answer the question: **does Claude make the transcript more correct, without changing its meaning, at an acceptable latency and cost?** Give your recommendation: adopt, adopt with conditions, or reject.

## Security and privacy requirements

- The API key is a **test key** with a spending limit. The organizer rotates or deletes it after the workshop.
- Keep the key only in environment variables of your terminal session. Do not commit it to Git, and do not put it in a file in the repository, in a screenshot, or in the report.
- The key is never in the Android app. The app does not call Claude.
- The transcript text goes to an external cloud service. Send only the test transcripts. Do not send audio, only the text from the phone.

## Risks and things to watch

- Claude may rewrite or embellish the text instead of correcting it.
- Claude may change proper nouns, vehicle terms, numbers and short commands.
- Claude may answer a question in the transcript instead of correcting it. Check it in the meaning review.
- Claude may write numbers as digits (`130`) where the reference has words (`one hundred and thirty`). This counts as a WER error. Report how often it happens.
- Network latency and API availability affect the user experience. In a real car, the original transcript must stay usable when the network is down.
- API limits (HTTP 429), token limits and costs apply.
- Compare outputs against the reference transcripts, not by judging which text sounds more natural.

## Deliverables

1. `tools/claude_correct.py` (the suggested version or your own).
2. The result files of every test in `results/group2/`: `phone.tsv`, `phone.summary.tsv`, `claude.tsv`, `claude-repeat.tsv`.
3. The HTML comparison page and the terminal output of `compare_runs.py`.
4. The meaning review and the failure test results.
5. `REPORT.md` with the tables, examples of improvements and regressions, the conclusion, and the presentation.

## Acceptance criteria

- The app works on its own on the phone, without any script or network.
- The phone transcripts are recorded before Claude is used.
- Every phone transcript is sent to Claude with the workshop prompt, and the answer is stored in a file.
- API and network failures do not stop the script, and the phone text is kept for every failed file.
- `compare_runs.py` compares the phone and the Claude transcripts against the same reference, and its verdict is in the report.
- The report contains WER before and after, latency, cost and the meaning review.
- No API key is committed to Git.
