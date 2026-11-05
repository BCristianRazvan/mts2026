# Speech-to-text workshop

This folder contains the guidelines for the speech-to-text (STT) workshop. Read this page first. It explains the two groups, the shared test data, the benchmark tools and the rules that both groups follow.

## The application

The Android app (`app/`) records the microphone and turns speech into text **on the phone**, without a network connection.

- **Normal use:** open the app, tap **Start listening**, speak, and the text appears in the text box. No PC, script or benchmark is needed.
- **Benchmark mode:** a PC script starts the app with special settings over USB. The app then runs the test WAV files through the same recognizer, saves the results, and returns to normal use. Only debug builds accept this mode.

| Part | Where |
|---|---|
| Recognition logic | `app/src/main/java/com/example/sttchat/OfflineSpeechRecognizer.java` |
| Model configuration | `OfflineSpeechRecognizer.createRecognizer()`, used by normal use and by benchmark mode |
| Benchmark mode | `app/src/main/java/com/example/sttchat/SttBenchmark.java` |
| User interface | `app/src/main/java/com/example/sttchat/SttChatActivity.java` |
| STT library | `app/libs/sherpa-onnx-1.13.3.aar` (sherpa-onnx 1.13.3) |
| STT model | `app/src/main/assets/models/stt/`: streaming Zipformer transducer, English, int8, about 69 MB |
| Hotwords (car-domain phrases) | `app/src/main/assets/models/stt/hotwords.txt` |

Build and install steps are in the root [README](../../README.md).

## Groups

There are 12 students in two groups of 6. Each group has its own guideline.

| | Group 1 | Group 2 |
|---|---|---|
| Guideline | [04 — Upgrade to a more accurate streaming model](04-medium-upgrade-to-more-accurate-streaming-model.md) | [05 — Cloud LLM after local STT](05-easy-medium-cloud-llm-after-local-stt.md) |
| Question | Does a better offline model give more accurate text than the current model, and is it worth the cost in speed, memory and size? | Does Claude improve the text from the phone without changing its meaning? |
| Stage 1 | Test the current app on the phone | Test the current app on the phone |
| Stage 2 | Replace the offline model with a better one | Add the code that connects to Claude (API key and endpoint) |
| Stage 3 | Test the new model on the phone | Send the phone transcripts to Claude and save the improved text |
| Stage 4 | Compare the results and decide which model is better | Compare the phone transcripts with the Claude text |

Both groups produce **measurable results**: every test writes its results to a file, and the comparison script turns two result files into a table in the terminal, an HTML page, and a verdict.

## The test set

[`testset/`](testset/) contains 10 synthetic WAV files and their reference text in `transcripts.tsv` (`file<TAB>text`, one line per file). Everybody uses the same files, so the results are comparable.

The sentences are hard on purpose: car-domain commands with rare words (`defroster`, `diesel particulate filter`, `turbocharger`), Romanian place names (`strada mihai eminescu`, `pitesti`, `cluj napoca`), numbers, a license plate spelled letter by letter, and a homophone (`brake` and `break`). The speech is faster than normal, and car noise is mixed in.

| How the files are made | File |
|---|---|
| Text of each file | `transcripts.tsv` |
| Voice, speaking rate, noise level (SNR in dB, lower is more noise) and microphone band filter of each file | `conditions.tsv` |
| Step 1: Windows speech synthesizer, writes clean speech to `testset/clean/` | `generate_testset.ps1` |
| Step 2: adds car noise with a fixed random seed, writes `testset/test_XX.wav` | `add_noise.py` |

To create the files again (for example after a change in `transcripts.tsv` or `conditions.tsv`):

```powershell
powershell -ExecutionPolicy Bypass -File docs\stt-model-replacement\testset\generate_testset.ps1
uv run --no-project --python 3.12 --with numpy python docs\stt-model-replacement\testset\add_noise.py
```

All files are 16 kHz, mono, 16-bit PCM, the format the app records. Numbers are written as words in `transcripts.tsv`, because the models output spoken numbers as words.

**Limitation:** synthetic speech is not real speech. Use the test set to compare systems with each other, not to predict the accuracy for real drivers.

## Tools

All tools are in [`tools/`](tools/). You need Python 3.12 and [uv](https://docs.astral.sh/uv/). Run all commands from the repository root, in PowerShell.

| Tool | Who | What it does |
|---|---|---|
| [`device_benchmark.py`](tools/device_benchmark.py) | Both groups | Copies the WAV files to the phone, runs the benchmark on the phone, and copies the result files back to the PC. Prints the measurements and the WER. |
| [`compare_runs.py`](tools/compare_runs.py) | Both groups | Compares two result files. Prints a table in the terminal, writes an HTML page, and gives a verdict. |
| [`stt_benchmark.py`](tools/stt_benchmark.py) | Group 1, optional | Runs the WAV files through a model on the PC, without building the app. Use it to screen candidate models quickly. |
| [`wer_report.py`](tools/wer_report.py) | Both groups | WER per file for any number of result files. Also contains the shared text normalization. |

### Word Error Rate (WER)

All accuracy numbers are WER: the number of substituted, deleted and inserted words, divided by the number of words in the reference. Lower is better. Before WER is calculated, every text is normalized in the same way: lowercase, no punctuation, single spaces. Without this step, `"Flat tire."` and `FLAT TIRE` count as two errors.

### Benchmark on the phone

1. Build and install the **debug** app (root README, sections 6–8).
2. Connect the phone with USB and find its serial number with `adb devices`.
3. Run:
   ```powershell
   uv run --no-project --python 3.12 --with jiwer `
       python docs/stt-model-replacement/tools/device_benchmark.py `
       --serial <phone-serial> --run baseline `
       --out-dir docs/stt-model-replacement/results/group1
   ```

What happens:

| Step | Where |
|---|---|
| The WAV files are copied to `/sdcard/Android/data/com.example.sttchat/files/benchmark/input/testset/`. | PC to phone |
| The app starts in benchmark mode. It loads the model with `createRecognizer()`, the same method that normal use calls. | Phone |
| Each file is decoded 3 times (`--repeat`). The audio goes to the recognizer in chunks of the microphone buffer size, like in normal use. The first file is decoded once before timing starts (warm-up). | Phone |
| The app writes `<run>.tsv`, `<run>.summary.tsv` and `<run>.done`. | Phone |
| The files are copied to `--out-dir`, and the measurements and the WER are printed. | Phone to PC |

The phone screen shows the progress and stays on during the run. One run takes about 2 minutes. Then the result summary stays in the text box, the app loads the model again, and the button is enabled for normal use.

The result files:

| File | Content |
|---|---|
| `<run>.tsv` | One row per WAV file: `file`, `hypothesis` (the recognized text), `audio_s`, `decode_ms` (median of 3), `rtf`, `first_endpoint_s` |
| `<run>.summary.tsv` | Two columns, `key` and `value`, one measurement per row: `device`, `android`, `model_load_ms`, `total_audio_s`, `total_decode_ms`, `rtf`, `pss_before_load_mb`, `pss_after_load_mb`, `pss_peak_mb` and the run settings |
| `<run>-processed.tsv` | Only for an optional processing step in the app. Not used in this workshop; ignore it if it appears. |

What the numbers mean:

| Value | Meaning |
|---|---|
| `rtf` | Real-time factor: decode time divided by audio length. RTF 0.4 means that 1 second of audio needs 0.4 seconds of CPU. Near 1 or above, normal use cannot keep up with the microphone. |
| `model_load_ms` | Time to load the model. The user waits this long after the app starts. |
| `pss_peak_mb` | Highest RAM use of the app (`TOTAL PSS`). The test phone has 3 GB of RAM in total. |
| `first_endpoint_s` | Where normal use would detect the end of the sentence. Normal use stops there, the benchmark decodes the whole file. If the value is much smaller than `audio_s`, normal use would cut the text. |

Options: `--serial` can be left out when exactly one phone is connected. `--repeat 5` for more timing runs, `--apk <path>` to install the APK first, `--skip-push` if the WAV files are already on the phone (it does not copy model files), `--testset <folder>` for another test set.

In PowerShell, a command that continues on the next line ends with a backtick (`` ` ``) and no space after it. Run all commands from the repository root (the folder that contains `gradlew.bat`).

**Setup check:** the current app gives **35.3% WER (67 errors / 190 words)** on the phone. The WER must match exactly. RTF, load time and memory vary by about 10% between runs. If the WER is different, your app or test set is different: reinstall the APK, and check with `git status` that nothing in `testset/` changed.

#### When the benchmark stops with an error

| Message | What to do |
|---|---|
| `adb not found` | Add `%LOCALAPPDATA%\Android\Sdk\platform-tools` to `PATH`, or set `$env:ADB` to the full path of `adb.exe`. |
| `Phone state is 'offline'` or `'unauthorized'` | Unplug and replug the cable, unlock the phone, accept the USB debugging prompt, then check with `adb devices`. |
| `Could not copy <file> to the phone` | The USB connection dropped. Replug the cable (or use another cable or port) and run again. |
| `<run>.tsv already exists` | Results are never overwritten. Choose a new `--run` name. |
| `Timeout. Check the phone screen ...` | The app crashed or the screen was locked. Unlock the phone, check `adb logcat -s SttBenchmark OfflineSpeechRecognizer`, and run again with a new `--run` name. |
| `Benchmark failed on the phone: <reason>` | Read the reason (for example a missing model file). Check `adb logcat -s SttBenchmark OfflineSpeechRecognizer sherpa-onnx`. If you cannot solve it, send the line to the organizer. |

### Compare two runs

```powershell
uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/compare_runs.py `
    docs/stt-model-replacement/results/group1/baseline.tsv `
    docs/stt-model-replacement/results/group1/<new-run>.tsv
```

The script prints a table with all metrics, a table with the WER of every file, and a verdict. It also writes `compare_<A>_vs_<B>.html` next to run B. Open the HTML page in a browser: it shows the verdict, the key metrics, a WER chart per file, and the transcripts with the wrong words marked.

The verdict follows a fixed rule. **B is better** if its WER is at least 1 percentage point lower than A, and B passes every limit that was measured for it:

| Limit for B | Default | Option |
|---|---|---|
| Minimum WER gain | 1.0 points | `--min-gain` |
| RTF on the phone | 0.8 | `--max-rtf` |
| Peak memory | 1024 MB | `--max-pss-mb` |
| Model load time | 15,000 ms | `--max-load-ms` |
| Median LLM latency | 3,000 ms | `--max-latency-ms` |
| Failed LLM requests | 0 | `--max-failed` |

Agree on the limits as a group in session 1, **before** you see any results, and write them in the first lines of `REPORT.md`. Use the defaults if you have no reason to change them.

The verdict is a rule based on numbers only. It does not check the meaning of the text. Report it together with your own review: if a number or command changed its meaning, write "adopt with conditions" or "reject", never a plain "adopt".

## Rules for both groups

1. **Record the baseline first.** Test the unmodified app before you change anything. Without a baseline you cannot prove an improvement.
2. **Do not change the reference data.** Do not edit the files in `testset/`. Do not edit the four shared tools in `tools/`. Adding a new file is allowed (Group 2 adds `tools/claude_correct.py`). If you find a bug in a tool, tell the organizer, who fixes it for both groups.
3. **Use the same conditions.** Two runs are comparable only if all of these are true. Copy the list into `REPORT.md` and tick it for every run:
   - [ ] Same phone and same USB cable.
   - [ ] Same `--repeat` value (default 3).
   - [ ] Same test set (`testset/` not edited).
   - [ ] Debug build. The only difference between the two builds is the change you test (Group 1: the model files and the model lines in `createRecognizer()`).
   - [ ] Battery above 50% and the phone plugged in.
   - [ ] Screen unlocked, no other app open.
   - [ ] At least 3 minutes since the previous run, and the phone is not warm.

   If the RTF of the same app differs by more than 10% between two runs, the phone was hot or busy: wait and run again.
4. **Store every result, never overwrite.** Every run gets a new `--run` name, for example `baseline`, `baseline-2`, `nemo-480ms`, `nemo-480ms-2`. `device_benchmark.py` refuses a name that already exists in `--out-dir`. If you repeat a run, report the first valid run and mention the repeat.
5. **One branch per group.** Create `group1/work` or `group2/work` from `main`, and never commit to `main`. Pair A commits only under `results/<group>/`, pair C (Group 1) only under `app/`, pair B only its new files. Run `git pull --rebase` before every push. At the end, the group opens one pull request from its branch, with code, results and `REPORT.md`.
6. **Never commit secrets or large files:** API keys, `local.properties`, or files above 100 MB (GitHub rejects them).

## Phone notes (Huawei P Smart 2019, EMUI)

- `adb install` can wait for an **"Install via USB"** prompt on the phone. Accept it, or the command does not finish.
- `INSTALL_FAILED_UPDATE_INCOMPATIBLE`: an APK built on another PC is installed. Run `adb uninstall com.example.sttchat` once, then install again. **Uninstalling deletes `/sdcard/Android/data/com.example.sttchat/`**, including model files that you copied there with `adb push`. Copy them again before the next benchmark.
- `adb devices` shows `offline`: unplug and replug the cable, unlock the phone, and accept the USB debugging prompt. If the connection drops during large transfers, try another cable or USB port.

## Suggested schedule

A session is about 3 hours.

| Session | Group 1 | Group 2 |
|---|---|---|
| 1 | Setup, agree on the limits, Stage 1: baseline on the phone, start the candidate search | Setup, agree on the limits, Stage 1: baseline on the phone. Pair B in parallel: Stage 2a and 2d (Claude access test) |
| 2 | Stage 2: candidate screening, integrate the best candidate | Stage 2: Claude script, Stage 3: first Claude run |
| 3 | Stage 3: test the new model (at most 2 candidates), Stage 4: compare | Stage 3: repeat run and failure tests, Stage 4: compare and meaning review |
| 4 | Report and presentation | Report and presentation |

## Final presentation (both groups)

Each group gives a 10-minute presentation:

1. **Question:** what you tried to prove.
2. **Setup:** phone, models or Claude deployment, settings, limits.
3. **Baseline:** results of the current app.
4. **Change:** what you changed, and why.
5. **Results:** the comparison table and the HTML page from `compare_runs.py`.
6. **Examples:** two improvements, one case without change, one regression.
7. **Conclusion:** your recommendation (adopt, adopt with conditions, or reject), supported by the numbers.

## REPORT.md

Each group writes `docs/stt-model-replacement/results/<group>/REPORT.md` with these sections, in this order:

1. **Setup:** phone model and Android version (from `<run>.summary.tsv`), the git commit hash (`git rev-parse --short HEAD`), and the limits you agreed on.
2. **Conditions:** the ticked checklist from rule 3, for every run.
3. **Baseline:** the results of the current app.
4. **Change:** what you changed, and why.
5. **Results:** the results table from your guideline, filled from the `compare_runs.py` output, and the name of the HTML file.
6. **Examples:** four files by name: two improved, one unchanged, one worse. Quote the reference and both texts.
7. **Trade-offs:** speed, memory, size, latency, cost, offline use.
8. **Recommendation:** adopt, adopt with conditions, or reject, with the two or three numbers that support it.
9. **Problems:** what went wrong, and how you solved it.

The presentation follows the same order.

## For the organizer

The preparation checklist and the organizer's own test results are in [ORGANIZER.md](ORGANIZER.md). Students do not need it.
