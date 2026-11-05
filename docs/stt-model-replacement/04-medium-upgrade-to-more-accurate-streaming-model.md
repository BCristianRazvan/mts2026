# 04 — Upgrade to a More Accurate Streaming Model (Group 1)

**Difficulty:** Medium
**Estimated effort:** 3–4 sessions of about 3 hours
**Team size:** 6 students
**Goal:** Replace the current Sherpa-ONNX streaming speech-recognition model with a more capable compatible model, then prove with measurements on the phone whether it is better than the existing model.

Read the [workshop README](README.md) first. It describes the test set, the tools, the result files and the rules.

## Objective

The application already performs offline speech-to-text using Sherpa-ONNX. In this scenario, you first benchmark the existing model on the phone, then integrate a stronger candidate model and repeat the same benchmark.

The goal is to decide, with numbers, whether the new model improves transcription quality, and whether that improvement justifies any additional latency, memory use, CPU consumption or APK/model size.

## Starting point

- The Android app and its microphone and UI flow. The app must keep working on its own: open it, tap **Start listening**, speak, and the text appears.
- The current model: a Sherpa-ONNX streaming Zipformer transducer in `app/src/main/assets/models/stt/`.
- The test set in `docs/stt-model-replacement/testset/`: 10 hard synthetic WAV files and the reference text in `transcripts.tsv`.
- The tools: `device_benchmark.py` (benchmark on the phone), `compare_runs.py` (comparison and verdict), and `stt_benchmark.py` (optional screening on the PC).

Keep the existing implementation as the baseline. Do not change the test files or the reference transcripts between runs.

## Roles

Work in three pairs. Every pair takes part in every stage, but each pair owns one area.

| Pair | Owns | Main tasks |
|---|---|---|
| A — Benchmark | Measurements and results | Runs `device_benchmark.py` and `compare_runs.py`, keeps `results/group1/`, writes `REPORT.md` |
| B — Model research | Candidate selection | Reads model pages, downloads candidates, screens them on the PC, proposes the model to integrate |
| C — Android | App integration | Changes `createRecognizer()`, builds and installs, checks that normal use still works |

## What we have today

| File in `app/src/main/assets/models/stt/` | Size | Purpose |
|---|---|---|
| `encoder.int8.onnx` | 68 MB | Encoder |
| `decoder.int8.onnx` | 1.2 MB | Decoder |
| `joiner.int8.onnx` | 253 KB | Joiner |
| `tokens.txt` | 5.4 KB | 500 BPE tokens, uppercase |
| `bpe.vocab` | 12.8 KB | Needed for hotwords |
| `hotwords.txt` | 2.7 KB | Car-domain phrases, uppercase |

The file sizes and the token list suggest the model `sherpa-onnx-streaming-zipformer-en-2023-06-26`. No file in the repository confirms this.

The model is configured in `OfflineSpeechRecognizer.createRecognizer()`. Normal use and benchmark mode both call this method, so a model change applies to both:

```java
String modelDirectory = "models/stt/";
config.getModelConfig().getTransducer().setEncoder(modelDirectory + "encoder.int8.onnx");
config.getModelConfig().getTransducer().setDecoder(modelDirectory + "decoder.int8.onnx");
config.getModelConfig().getTransducer().setJoiner(modelDirectory + "joiner.int8.onnx");
config.getModelConfig().setTokens(modelDirectory + "tokens.txt");
config.getModelConfig().setModelingUnit("bpe");
config.getModelConfig().setBpeVocab(modelDirectory + "bpe.vocab");
config.setHotwordsFile(modelDirectory + "hotwords.txt");
config.setHotwordsScore(1.5f);
config.getModelConfig().setNumThreads(4);
config.setDecodingMethod("modified_beam_search");
config.setMaxActivePaths(4);

return new OnlineRecognizer(context.getAssets(), config);
```

The app uses a **streaming** recognizer (`OnlineRecognizer`). It shows partial text while the user speaks and detects the end of a sentence itself. A streaming candidate keeps this behavior; only the lines above change.

## Work plan

### Stage 1 — Test the current application

Owner: pairs A and C.

1. Create the group branch `group1/work` from `main` (README rule 5). Agree on the limits for the verdict (README, *Compare two runs*) and write them at the top of `REPORT.md`.
2. Build and install the unmodified debug app on the test phone (root README, sections 6–8).
3. Check normal use: speak two sentences into the app and check that the text appears.
4. Run the benchmark on the phone:
   ```powershell
   uv run --no-project --python 3.12 --with jiwer `
       python docs/stt-model-replacement/tools/device_benchmark.py `
       --serial <phone-serial> --run baseline `
       --out-dir docs/stt-model-replacement/results/group1
   ```
5. Check that you get the setup-check result from the README (35.3% WER). The command stores the results in `results/group1/baseline.tsv` and `results/group1/baseline.summary.tsv`.
6. Write the APK size (`app\build\outputs\apk\debug\app-debug.apk`) and the model folder size in `REPORT.md`.

This is the baseline. Keep the raw files so that the results can be reproduced.

### Stage 2 — Replace the offline model with a better one

#### 2a. Find candidates (pair B)

All models are on the sherpa-onnx release page: `https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models`. Each model has a docs page under `https://k2-fsa.github.io/sherpa/onnx/pretrained_models/` with its training data, files and example command. Read it before you download.

English streaming candidates on the release page:

| Model | Family (Java setter) | Hotwords | Notes |
|---|---|---|---|
| `sherpa-onnx-streaming-zipformer-en-2023-06-21` | Zipformer transducer (`getTransducer()`) | Yes, with a new `bpe.vocab` | Same family as today. Check its training data. Also a `-mobile` variant. |
| `sherpa-onnx-streaming-zipformer-en-2023-02-21` | Zipformer transducer | Yes, with a new `bpe.vocab` | Same family, older. |
| `sherpa-onnx-streaming-conformer-en-2023-05-09` | Conformer transducer (`getTransducer()`) | Yes, with a new `bpe.vocab` | Different encoder type. |
| `sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-{80,480,1040}ms-int8` | NeMo transducer (`getTransducer()`) | **No.** sherpa-onnx 1.13.3 accepts only `greedy_search` for NeMo transducers. | Encoder about 125 MB: **over 100 MB, use 2d**. Larger chunks need less CPU but give partial text later. |
| `sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-{80,480,1040}ms-int8` | NeMo CTC (`getNeMoCtc()`) | No | One `model.onnx` file. Uses `greedy_search`. Check the file size: if it is over 90 MB, use 2d. |
| `sherpa-onnx-nemotron-speech-streaming-en-0.6b-{80,160,560,1120}ms-int8-2026-04-25` | NeMo transducer (`getTransducer()`) | No (`greedy_search`) | 0.6B parameters. Encoder about 623 MB, about 785 MB RAM on a PC: **over 100 MB, use 2d**. |

**Runtime version:** the app contains sherpa-onnx **1.13.3** (released 2026-06-15). Models published after that date may need a newer runtime. If you upgrade the AAR, you change two things at once: then benchmark the **current** model again with the new runtime, so that you can separate the runtime effect from the model effect.

Do not assume that a model is more accurate because of its name or size. Confirm that it supports English and the vocabulary in the test data.

#### 2b. Screen candidates on the PC (pair B, optional but recommended)

An Android integration takes hours. Screening a model on the PC takes minutes. Keep downloaded models **outside** the repository, for example in `C:\models\`.

```powershell
# Zipformer or Conformer transducer: use the real file names from the archive.
uv run --no-project --python 3.12 --with sherpa-onnx==1.13.3 --with jiwer --with numpy --with psutil `
    python docs/stt-model-replacement/tools/stt_benchmark.py `
    --model-dir C:\models\<model-name> `
    --encoder <encoder-file>.int8.onnx --decoder <decoder-file>.onnx --joiner <joiner-file>.int8.onnx `
    --out docs/stt-model-replacement/results/group1/pc-<model-name>.tsv

# NeMo or Nemotron transducer: greedy_search only.
uv run --no-project --python 3.12 --with sherpa-onnx==1.13.3 --with jiwer --with numpy --with psutil `
    python docs/stt-model-replacement/tools/stt_benchmark.py `
    --model-dir C:\models\<model-name> --decoding-method greedy_search `
    --out docs/stt-model-replacement/results/group1/pc-<model-name>.tsv

# NeMo CTC: one model file.
uv run --no-project --python 3.12 --with sherpa-onnx==1.13.3 --with jiwer --with numpy --with psutil `
    python docs/stt-model-replacement/tools/stt_benchmark.py `
    --model-dir C:\models\<model-name> --family nemo_ctc --model model.int8.onnx `
    --out docs/stt-model-replacement/results/group1/pc-<model-name>.tsv
```

`<model-name>` is the folder name of the extracted archive, for example `sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8`.

Run the current model on the PC once, as the PC baseline, and compare each candidate with it:

```powershell
uv run --no-project --python 3.12 --with sherpa-onnx==1.13.3 --with jiwer --with numpy --with psutil `
    python docs/stt-model-replacement/tools/stt_benchmark.py `
    --model-dir app/src/main/assets/models/stt --hotwords `
    --out docs/stt-model-replacement/results/group1/pc-baseline.tsv

uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/compare_runs.py `
    docs/stt-model-replacement/results/group1/pc-baseline.tsv `
    docs/stt-model-replacement/results/group1/pc-<model-name>.tsv
```

On the PC, the verdict checks only the WER, because the PC does not measure the phone limits.

Choose the model to integrate by the WER first. Then check that it can run on the phone:

- **Speed:** the phone is about 3.5 times slower than a PC. A PC RTF above about 0.25 probably fails the phone RTF limit (0.8).
- **Load time:** it grows with the size of the encoder. The current 68 MB encoder already needs about 8–11 s on the phone, and the limit is 15 s. An encoder above about 100 MB will probably fail this limit.
- **Memory:** compare the PC memory of the candidate with the current model.

Pair B tells the group these risks **before** pair C starts the integration. A candidate that fails a limit is still a valid result, but plan to finish at least one candidate that can pass.

**Stop condition:** integrate and test at most **2 candidates** on the phone. Choose them in session 2, and do not add a third.

#### 2c. Integrate the model in the app (pair C)

1. Work on the group branch `group1/work`. Write in `REPORT.md` which candidate you integrate and why.
2. **Check the file sizes first** (`dir C:\models\<model-name>`). If any file is larger than 90 MB, do **not** put the model in `assets/`: use 2d instead, and keep the files out of git. Otherwise put the model files in a new folder, for example `app/src/main/assets/models/stt-new/`. Copy the encoder, decoder, joiner (or the CTC `model.onnx`) and `tokens.txt` from the **same archive**. When the new model works, remove the old `models/stt/` folder, otherwise both models are in the APK.
3. Change the model section of `createRecognizer()`. The code block in *What we have today* shows only the model lines. Before them, `createRecognizer()` also sets the features (`setSampleRate`, `setFeatureDim`) and the endpoint rules: do not delete those. Replace only the lines from `String modelDirectory` to `config.setMaxActivePaths(4);`.
   - **Zipformer or Conformer transducer:** change only the folder and file names.
   - **NeMo or Nemotron transducer:** change the file names, remove the BPE and hotword lines, and use greedy search:
     ```java
     String modelDirectory = "models/stt-new/";
     config.getModelConfig().getTransducer().setEncoder(modelDirectory + "encoder.int8.onnx");
     config.getModelConfig().getTransducer().setDecoder(modelDirectory + "decoder.int8.onnx");
     config.getModelConfig().getTransducer().setJoiner(modelDirectory + "joiner.int8.onnx");
     config.getModelConfig().setTokens(modelDirectory + "tokens.txt");
     config.getModelConfig().setNumThreads(4);
     config.setDecodingMethod("greedy_search");
     ```
   - **NeMo CTC:**
     ```java
     String modelDirectory = "models/stt-new/";
     config.getModelConfig().getNeMoCtc().setModel(modelDirectory + "model.int8.onnx");
     config.getModelConfig().setTokens(modelDirectory + "tokens.txt");
     config.getModelConfig().setNumThreads(4);
     config.setDecodingMethod("greedy_search");
     ```
   Use the model's docs page as the reference for every setting.
4. Check the features: `setSampleRate(16000)` and `setFeatureDim(80)` are correct for most English models. If the model page shows a different value, change it.
5. Hotwords (Zipformer or Conformer only): hotwords need a `bpe.vocab` that matches the **new** model's `bpe.model`. Download the script `https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/scripts/export_bpe_vocab.py` to a folder outside the repository, for example `C:\models\tools\`, then run:
   ```powershell
   uv run --no-project --python 3.12 --with sentencepiece `
       python C:\models\tools\export_bpe_vocab.py --bpe-model C:\models\<model-name>\bpe.model
   ```
   The script writes `bpe.vocab` next to `bpe.model`. Copy `bpe.vocab` and `hotwords.txt` to the new model folder, and check that `hotwords.txt` uses the same casing as the new `tokens.txt` (the current files are uppercase).
6. NeMo models output lowercase text; the current model outputs uppercase. Check that nothing in `SttChatActivity` depends on the case.
7. Create `SOURCE.txt` in the model folder with the model name, download URL and date.
8. Build, install and check the logs:
   ```powershell
   .\gradlew :app:assembleDebug
   adb install -r app\build\outputs\apk\debug\app-debug.apk
   adb logcat -s OfflineSpeechRecognizer SttBenchmark sherpa-onnx
   ```
   If the app shows "Offline speech model could not start", the log shows the real error. Common causes: wrong file name, `tokens.txt` from another model, wrong config object for the model family, or `modified_beam_search` with a NeMo model.
9. Check normal use: speak two sentences and check that the text appears.

#### 2d. Models above 100 MB

GitHub rejects files above 100 MB, and a large APK is slow to install. Keep large models on the phone storage:

```powershell
adb shell mkdir -p /sdcard/Android/data/com.example.sttchat/files/models/stt-new
adb push C:\models\<model-name>\. /sdcard/Android/data/com.example.sttchat/files/models/stt-new/
```

In `createRecognizer()`, build absolute paths and pass `null` instead of `context.getAssets()`. sherpa-onnx then reads the files from the file system:

```java
String modelDirectory = context.getExternalFilesDir(null) + "/models/stt-new/";
// ... the same setters as above, with modelDirectory ...
return new OnlineRecognizer(null, config);
```

`adb uninstall com.example.sttchat` deletes this folder. After an uninstall, run the `adb push` command again. `device_benchmark.py --skip-push` skips only the WAV files, never the model files.

Before you create the recognizer, check that every file exists. If a file is missing, throw an `IllegalStateException` with the expected path: the app then shows the error, and the benchmark writes it to `<run>.done`. Write the download and `adb push` steps in `REPORT.md`.

### Stage 3 — Test the new model

Owner: pairs A and C.

Run exactly the same benchmark as in Stage 1, with the same phone, the same `--repeat` value and a new run name:

```powershell
uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/device_benchmark.py `
    --serial <phone-serial> --run <model-name> `
    --out-dir docs/stt-model-replacement/results/group1
```

The results are stored in `results/group1/<model-name>.tsv` and `<model-name>.summary.tsv`. Do not overwrite the baseline. Also write the new APK size and model size in `REPORT.md`.

If time permits, integrate and test a second candidate the same way (at most 2 candidates). Each candidate gets its own run name.

To repeat a run (for example because the RTF changed by more than 10%), use the name with `-2`, for example `--run nemo-480ms-2`. Report the first valid run and mention the repeat.

### Stage 4 — Compare and decide

Owner: whole group.

```powershell
uv run --no-project --python 3.12 --with jiwer `
    python docs/stt-model-replacement/tools/compare_runs.py `
    docs/stt-model-replacement/results/group1/baseline.tsv `
    docs/stt-model-replacement/results/group1/<model-name>.tsv `
    --label-a current --label-b <model-name>
```

The script prints the comparison in the terminal and writes `results/group1/compare_baseline_vs_<model-name>.html`. The verdict uses the rule in the [README](README.md#compare-two-runs): the new model is better only if its WER is at least 1 point lower **and** it passes the limits for RTF, memory and load time on the phone.

In `REPORT.md`, fill this table from the terminal output and your size measurements:

| Metric | Current model | New model | Difference |
|---|---:|---:|---:|
| WER (%) | | | |
| Files without errors | | | |
| RTF on the phone | | | |
| Model load time (ms) | | | |
| Peak memory (MB) | | | |
| Model size (MB) | | | |
| APK size (MB) | | | |
| Live partial results | Yes | | |
| Hotwords | Yes | | |
| **Verdict of `compare_runs.py`** | | | |

How to fill the rows that `compare_runs.py` does not measure:

- **Model size:** the total size of the files in the model folder, in MB with one decimal (Explorer, folder **Properties**).
- **APK size:** in PowerShell, `(Get-Item app\build\outputs\apk\debug\app-debug.apk).Length / 1MB`.
- **Live partial results:** speak into the app. Yes, if text appears in the box before you finish the sentence.
- **Hotwords:** yes, if the model configuration contains `setHotwordsFile`.

Then explain:

1. Whether the new model improves WER, and in which files (use the per-file table and the marked transcripts in the HTML page).
2. Examples: two files that improved, one without change, one that became worse.
3. The trade-offs: speed, memory, load time and size.
4. Your recommendation: adopt, adopt with conditions, or reject. A candidate that loses is a valid result if it is measured correctly.

## Benchmark guidance

- Use the shared normalization (built into the tools) for both models and the references.
- Report the overall WER and the per-file WER.
- Use the same timing definition for both models: the benchmark measures the decode time per file (median of 3) and the RTF.
- Run the benchmark twice for each model if time permits (`--run <name>` and `--run <name>-2`). If the RTF changes by more than 10%, the phone was probably hot or busy: wait and run again.
- Keep all test conditions and model settings documented.
- Treat memory, CPU and size measurements as comparative: results vary by device.

## Risks and things to watch

- **Mixed files:** the encoder, decoder, joiner, `tokens.txt` and `bpe.vocab` must come from the same archive. Mixed files sometimes load without an error but produce nonsense.
- **Wrong config object:** a CTC model in `getTransducer()` does not load. Match the model family to the setter in the table.
- **Decoding method:** NeMo transducers and CTC models need `greedy_search`. `modified_beam_search` fails with `Unsupported decoding method`.
- **Hotwords:** remove `setHotwordsFile`, `setModelingUnit` and `setBpeVocab` for models without hotword support. Do not claim hotword parity unless you verified it.
- **Chunk size against CPU:** 80 ms chunks give the fastest partial text but use the most CPU. If the RTF on the phone goes above 1, the app falls behind the microphone.
- **`noCompress 'onnx'`** in `app/build.gradle` must stay. Without it, models are compressed in the APK and load more slowly.
- **A larger model** may use more memory, load more slowly, or run too slowly on the phone.
- **A runtime upgrade alone** does not guarantee a lower WER. Separate runtime changes from model changes.

## Deliverables

1. A working Android build with the chosen model (or a documented reason why no candidate was good enough). Normal use still works without any script.
2. The result files of every test in `results/group1/`: `baseline.tsv`, `baseline.summary.tsv`, and the same for each candidate.
3. The HTML comparison page and the terminal output of `compare_runs.py`.
4. A short summary of the model and configuration changes, including `SOURCE.txt`.
5. `REPORT.md` with the table, the examples, the trade-offs and the conclusion, and the presentation.

## Acceptance criteria

- The baseline is recorded on the phone before any model change.
- The candidate model runs in the Android app, in normal use and in benchmark mode.
- All 10 test files are run with both models under the same conditions.
- Every test result is stored in a file, and no result is overwritten.
- `compare_runs.py` compares the baseline and the new model, and its verdict is in the report.
- The conclusion is supported by the measured results.
