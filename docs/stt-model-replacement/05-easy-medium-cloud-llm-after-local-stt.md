# 05 — Cloud LLM After Local STT (Group 2)

**Difficulty:** Easy–Medium
**Estimated effort:** 3–4 sessions of about 3 hours
**Team size:** 6 students
**Goal:** Keep the existing offline Sherpa-ONNX speech-to-text model. Send the transcripts produced on the phone to an OpenAI LLM through the OpenAI API, let the LLM correct transcription errors, and prove with measurements whether the corrected text is better than the phone text.

Read the [workshop README](README.md) first. It describes the test set, the tools, the result files and the rules.

## Objective

In this scenario you do not replace or retrain the STT model. The existing model continues to convert speech into text on the phone. A PC script then sends the recognized text to an OpenAI LLM (default: `gpt-6-astra`), which is asked to correct transcription errors without changing the speaker's intended meaning. In this guideline, "the LLM" always means this OpenAI model.

You compare the phone transcript and the LLM transcript against the reference transcript in `transcripts.tsv`. A fluent rewrite is not necessarily a more accurate transcript, so the reference decides, not your opinion of which text sounds better.

```text
WAV file -> app on the phone (local STT) -> phone transcript ---------------------> WER against the reference
                                                   |
                                                   +-> PC script -> OpenAI API -> LLM text -> WER against the reference
```

The Android app itself does not change. It keeps working on its own and offline: open it, tap **Start listening**, speak, and the text appears. The LLM is called only from the PC script, so no API key is ever inside the app.

## Starting point

- The Android app and its local Sherpa-ONNX STT model.
- The test set in `docs/stt-model-replacement/testset/`: 10 hard synthetic WAV files and the reference text in `transcripts.tsv`.
- The tools: `device_benchmark.py` (benchmark on the phone) and `compare_runs.py` (comparison and verdict).
- An OpenAI API project with an API key, prepared by the organizer. API usage is billed to the API project on the OpenAI Platform, separately from a ChatGPT plan, so the project needs API credit.
- Network access from the PC to `https://api.openai.com`.

## Roles

Work in three pairs. Every pair takes part in every stage, but each pair owns one area.

| Pair | Owns | Main tasks |
|---|---|---|
| A — Benchmark | Measurements and results | Runs `device_benchmark.py` and `compare_runs.py`, keeps `results/group2/`, writes `REPORT.md` |
| B — LLM integration | The OpenAI script | Creates and runs `openai_correct.py`, handles errors, measures latency and cost |
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

The command stores the phone transcripts in `results/group2/phone.tsv` (column `hypothesis`) and the phone measurements in `results/group2/phone.summary.tsv`. Do not edit these files: they are the input for every LLM run, so every run uses the same text.

The current model makes errors like these on the test set: `devil particulate filter` for `diesel particulate filter`, `crews control` for `cruise control`, `break hard` for `brake hard`, `kilobots` for `kilowatts`. Some of them can be corrected from the context, and some cannot (place names that the model did not recognize at all).

### Stage 2 — Add the code that connects to the OpenAI API

Owner: pair B, on the group branch `group2/work`. Pair B can do 2a and 2d already in session 1, while pair A runs Stage 1.

#### 2a. OpenAI API access

The organizer creates a project on the OpenAI Platform, sets a budget for it, creates a project API key, and gives the group these values outside of git (for example in a private message):

| Setting | Example | Where to find it |
|---|---|---|
| API key | (secret, starts with `sk-`) | OpenAI Platform, the project, **API keys** |
| Model name | `gpt-6-astra` | OpenAI model list. The project must have access to this model. |

Set the values as environment variables in the PowerShell window where you run the script:

```powershell
$env:OPENAI_API_KEY = "<key from the organizer>"
$env:OPENAI_MODEL = "gpt-6-astra"
```

The OpenAI SDK reads the key from `OPENAI_API_KEY` automatically. The variables exist only in this window. In every new PowerShell window, set them again. Do not use `setx` for the key: `setx` stores it permanently in your Windows user profile. Check the model with `echo $env:OPENAI_MODEL`, but never `echo` the key, so that it does not appear on screen or in a screenshot. Never write the key into a file in the repository.

#### 2b. The prompt

Use this prompt, exactly as written, as the system instructions:

> Correct transcription errors only. Preserve the original meaning. Do not add information or infer words that are not supported by the input.

Do not change it during the measurements. If you want to try a variant, do it after the main measurement, save it under a new run name, and report it separately.

#### 2c. Suggested implementation: `tools/openai_correct.py`

This is one suggestion. You may write your own version, but it must read the same input, write the same output columns, and use the same prompt.

The script uses the official OpenAI Python SDK (`openai`) and the Responses API (`client.responses.create`, endpoint `v1/responses`). The client reads the key from `OPENAI_API_KEY`.

Create the file `docs/stt-model-replacement/tools/openai_correct.py` with the content below. Copy the whole block and save it as UTF-8 (VS Code does this by default). Adding this one new file to `tools/` is allowed; do not edit the four existing tools.

```python
"""Send each phone transcript to an OpenAI LLM and save the corrected text.

Reads a result TSV from the phone benchmark (columns `file` and `hypothesis`),
sends every transcript to the OpenAI Responses API, and writes a new TSV with
the columns `file`, `stt`, `hypothesis` (the LLM text), `latency_ms`,
`input_tokens`, `output_tokens` and `status`. If a request fails, the phone
text is kept and `status` says why, so the comparison never loses a file.

Set these environment variables first (never commit them):

    OPENAI_API_KEY  project API key from the OpenAI Platform
    OPENAI_MODEL    model name (default: gpt-6-astra)

Example (from the repository root):

    uv run --no-project --python 3.12 --with openai \
        python docs/stt-model-replacement/tools/openai_correct.py \
        docs/stt-model-replacement/results/group2/phone.tsv \
        docs/stt-model-replacement/results/group2/openai.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import openai
from openai import OpenAI

# The prompt given by the workshop. Do not change it during the measurements.
SYSTEM_PROMPT = (
    "Correct transcription errors only. Preserve the original meaning. "
    "Do not add information or infer words that are not supported by the input."
)

# Structured output: the LLM must answer with {"corrected_text": "..."}.
# This keeps explanations or quotes out of the text, which would count as WER errors.
OUTPUT_FORMAT = {
    "type": "json_schema",
    "name": "corrected_transcript",
    "schema": {
        "type": "object",
        "properties": {"corrected_text": {"type": "string"}},
        "required": ["corrected_text"],
        "additionalProperties": False,
    },
    "strict": True,
}


def create_client() -> OpenAI:
    """Reads OPENAI_API_KEY from the environment."""
    return OpenAI(
        timeout=30.0,
        max_retries=2,  # retries network errors, 429 and 5xx automatically
    )


def correct(client: OpenAI, model: str, text: str) -> dict:
    """Sends one transcript to the LLM and returns the corrected text and usage."""
    start = time.perf_counter()
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=text,
        # Low effort: a short correction task does not need deep reasoning.
        reasoning={"effort": "low"},
        text={"format": OUTPUT_FORMAT},
        # Reasoning tokens also count against this limit.
        max_output_tokens=2048,
        # Do not keep the transcript on the OpenAI side after the answer.
        store=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    if response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details else "unknown"
        raise ValueError(f"answer incomplete ({reason})")
    for item in response.output:
        if item.type != "message":
            continue
        for part in item.content:
            if part.type == "refusal":
                raise ValueError(f"the LLM declined the request: {part.refusal}")
    corrected = json.loads(response.output_text)["corrected_text"].strip()
    if not corrected:
        raise ValueError("empty answer")
    return {
        "text": corrected,
        "latency_ms": latency_ms,
        "input_tokens": response.usage.input_tokens,
        # Includes the reasoning tokens, which are billed as output tokens.
        "output_tokens": response.usage.output_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "phone_tsv", type=Path, help="Result TSV from the phone benchmark."
    )
    parser.add_argument("out_tsv", type=Path)
    args = parser.parse_args()

    model = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
    try:
        client = create_client()
    except openai.OpenAIError as error:
        raise SystemExit(f"Cannot create the OpenAI client: {error}") from None

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
            except openai.AuthenticationError:
                status = "error 401: wrong API key"
            except openai.PermissionDeniedError:
                status = "error 403: key has no access"
            except openai.NotFoundError:
                status = f"error: model '{model}' not found"
            except openai.RateLimitError as error:
                if error.code == "insufficient_quota":
                    status = "error 429: no API credit (insufficient_quota)"
                else:
                    status = "error 429: rate limit (retries used up)"
            except openai.APIStatusError as error:
                if error.code == "model_not_found":
                    status = f"error: model '{model}' not found"
                else:
                    status = f"error {error.status_code}"
            except openai.APIConnectionError as error:  # includes timeouts
                status = f"network error: {error}"
            except (ValueError, KeyError) as error:
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
        print(f"   phone: {original}")
        print(f"   llm  : {result['text']}")

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

- **`SYSTEM_PROMPT`:** the workshop prompt, sent as the `instructions` parameter. The phone transcript is the only `input`.
- **`OUTPUT_FORMAT` (structured output):** the LLM must answer with JSON `{"corrected_text": "..."}` (`text.format` with `type` `json_schema` and `strict` `True`). Without it, the LLM may add a sentence such as "Here is the corrected text:", and every added word counts as a WER error.
- **`reasoning={"effort": "low"}`:** `gpt-6-astra` is a reasoning model. Correcting a short sentence needs little reasoning, and low effort keeps latency and cost down. The script does not send `temperature`. The repeat run in Stage 3 measures how stable the answers are.
- **`max_output_tokens=2048`:** the reasoning tokens and the answer share this limit. If the limit is reached, the response has the status `incomplete` and the script records `bad answer: answer incomplete (max_output_tokens)`.
- **`store=False`:** OpenAI does not keep the response for later retrieval.
- **Refusal:** if the LLM declines a request, the answer contains a `refusal` part instead of the JSON. The script records `bad answer: the LLM declined the request: ...` and keeps the phone text.
- **`max_retries=2`, `timeout=30.0`:** the SDK retries network errors, HTTP 429 and HTTP 5xx automatically, and stops waiting after 30 seconds.
- **Error handling:** every failure ends in the `status` column, and the phone text is kept in `hypothesis`. The script never stops in the middle, so the comparison always has all 10 files.
- **Usage:** `input_tokens` and `output_tokens` come from `response.usage`. The output tokens include the reasoning tokens, because OpenAI bills them as output tokens. `compare_runs.py` uses both columns to calculate the cost.

#### 2d. Test the connection with one file

Before you run all files, test the connection with one file. In PowerShell, from the repository root:

```powershell
Get-Content docs\stt-model-replacement\results\group2\phone.tsv -TotalCount 2 |
    Set-Content docs\stt-model-replacement\results\group2\one.tsv

uv run --no-project --python 3.12 --with openai `
    python docs/stt-model-replacement/tools/openai_correct.py `
    docs/stt-model-replacement/results/group2/one.tsv `
    docs/stt-model-replacement/results/group2/test-one.tsv
```

`one.tsv` contains the header and the first file. Check that the `status` printed for `test_01.wav` is `ok`. `phone.tsv` stays unchanged. If the status is `error 429: no API credit (insufficient_quota)`, the API project has no credit: tell the organizer.

### Stage 3 — Send the phone transcripts to the LLM

Owner: pair B.

```powershell
uv run --no-project --python 3.12 --with openai `
    python docs/stt-model-replacement/tools/openai_correct.py `
    docs/stt-model-replacement/results/group2/phone.tsv `
    docs/stt-model-replacement/results/group2/openai.tsv
```

The script prints the phone text and the LLM text of every file, and stores the results in `results/group2/openai.tsv` with the columns `file`, `stt` (phone text), `hypothesis` (LLM text), `latency_ms`, `input_tokens`, `output_tokens` and `status`.

If any `status` is not `ok`, find the cause and run the script again with a new output name (for example `openai-2.tsv`). The script refuses an output name that already exists.

Run the script a second time with the output `openai-repeat.tsv`, and compare the two LLM runs:

```powershell
uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/compare_runs.py `
    docs/stt-model-replacement/results/group2/openai.tsv `
    docs/stt-model-replacement/results/group2/openai-repeat.tsv `
    --label-a run1 --label-b run2
```

Here, ignore the verdict. Read only the line `Files improved / unchanged / worse` and the numbers of errors fixed and introduced. If all files are unchanged and both numbers are 0, the LLM gave the same result twice. Report how many files changed between the two runs.

**Failure tests (pair C).** The script must keep the phone text and report the problem in every case. Test each case with `one.tsv` from 2d and a separate output file, `fail-<case>.tsv` (for example `fail-key.tsv`). After each test, set the key and `OPENAI_MODEL` back to the correct values. Record the `status`:

| Case | How to create it | Expected `status` |
|---|---|---|
| Wrong key | Change one character of `OPENAI_API_KEY` | `error 401: wrong API key` |
| Wrong model name | Set `OPENAI_MODEL` to a name that does not exist | `error: model '...' not found` |
| No network | Disconnect the PC from the network | `network error: ...` (takes up to about 2 minutes, because the SDK retries twice with a 30 s timeout) |
| Empty transcript | In a copy of `one.tsv` (`empty.tsv`), delete the text in the `hypothesis` column | `skipped: empty transcript` |

### Stage 4 — Compare the phone transcript with the LLM transcript

Owner: whole group.

`compare_runs.py` has default prices for a different model. Give the current OpenAI prices of your model with `--price-in` and `--price-out` (USD per 1M tokens). For `gpt-6-astra`, the standard prices are USD 10 per 1M input tokens and USD 50 per 1M output tokens. Check the current prices on the OpenAI pricing page before you run the command.

```powershell
uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/compare_runs.py `
    docs/stt-model-replacement/results/group2/phone.tsv `
    docs/stt-model-replacement/results/group2/openai.tsv `
    --label-a phone --label-b openai `
    --price-in 10 --price-out 50
```

The script prints the comparison in the terminal and writes `results/group2/compare_phone_vs_openai.html`. It shows:

- the WER of the phone text and of the LLM text, both against the reference;
- the number of files improved, unchanged and worse, the errors fixed and the errors introduced by the LLM, and the files that were correct before and that the LLM broke;
- the median and maximum latency, the failed requests, and the cost per request and per 1,000 requests;
- the verdict: the LLM is better only if its WER is at least 1 point lower, the median latency is below 3,000 ms, **and** no request failed. If a request failed, fix the cause and run again with a new output name.

**Meaning review (pair C).** WER does not show every problem. A changed number or command can have a small WER effect but a large effect for the driver. Two students review every file where the LLM changed the text, independently, and then compare their answers:

| File | Phone WER | LLM WER | Words changed by the LLM | Meaning preserved? | Information added? | Number or command changed? |
|---|---:|---:|---|---|---|---|
| `test_01.wav` | | | | | | |
| ... | | | | | | |

How to fill the review table:

- **Words changed by the LLM:** the words that differ between the phone text and the LLM text. The *Transcripts* section of the HTML page shows both texts with the wrong words marked.
- **Meaning preserved?** yes or no. **Information added?** and **Number or command changed?** yes or no.
- A file has a **changed meaning** if either reviewer answered *no* to the first question or *yes* to the second or third question.

The verdict of `compare_runs.py` does not check the meaning. If any file has a changed meaning, your recommendation is "adopt with conditions" or "reject", never a plain "adopt".

In `REPORT.md`, fill this table:

| Metric | Phone only | Phone + LLM |
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

Then answer the question: **does the LLM make the transcript more correct, without changing its meaning, at an acceptable latency and cost?** Give your recommendation: adopt, adopt with conditions, or reject.

## Security and privacy requirements

- The API key is a **test key** of a project with a budget. The organizer revokes it after the workshop.
- Keep the key only in environment variables of your terminal session. Do not commit it to Git, and do not put it in a file in the repository, in a screenshot, or in the report.
- The key is never in the Android app. The app does not call the OpenAI API.
- The transcript text goes to an external cloud service. Send only the test transcripts. Do not send audio, only the text from the phone.

## Risks and things to watch

- The LLM may rewrite or embellish the text instead of correcting it.
- The LLM may change proper nouns, vehicle terms, numbers and short commands.
- The LLM may answer a question in the transcript instead of correcting it. Check it in the meaning review.
- The LLM may write numbers as digits (`130`) where the reference has words (`one hundred and thirty`). This counts as a WER error. Report how often it happens.
- Network latency and API availability affect the user experience. In a real car, the original transcript must stay usable when the network is down.
- API limits (HTTP 429), API credit, token limits and costs apply. Reasoning tokens are billed as output tokens.
- Compare outputs against the reference transcripts, not by judging which text sounds more natural.

## Deliverables

1. `tools/openai_correct.py` (the suggested version or your own).
2. The result files of every test in `results/group2/`: `phone.tsv`, `phone.summary.tsv`, `openai.tsv`, `openai-repeat.tsv`.
3. The HTML comparison page and the terminal output of `compare_runs.py`.
4. The meaning review and the failure test results.
5. `REPORT.md` with the tables, examples of improvements and regressions, the conclusion, and the presentation.

## Acceptance criteria

- The app works on its own on the phone, without any script or network.
- The phone transcripts are recorded before the LLM is used.
- Every phone transcript is sent to the LLM with the workshop prompt, and the answer is stored in a file.
- API and network failures do not stop the script, and the phone text is kept for every failed file.
- `compare_runs.py` compares the phone and the LLM transcripts against the same reference, and its verdict is in the report.
- The report contains WER before and after, latency, cost and the meaning review.
- No API key is committed to Git.
