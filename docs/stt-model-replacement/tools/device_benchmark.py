"""Run the STT benchmark on a connected Android phone and copy the results to the PC.

Steps:
1. Push the WAV files of a test set to the phone.
2. Start the app in benchmark mode (SttBenchmark.java) with adb.
3. Wait until the phone writes `<run>.done`.
4. Pull `<run>.tsv`, `<run>.summary.tsv` (and `<run>-processed.tsv`, if any).
5. Print the phone measurements and the WER against transcripts.tsv.

The app must be a debug build and installed already (or pass --apk). The result
TSV has the same `file` and `hypothesis` columns as stt_benchmark.py, so
wer_report.py works on both.

Example (from the repository root):

    uv run --no-project --python 3.12 --with jiwer \
        python docs/stt-model-replacement/tools/device_benchmark.py \
        --serial 39V4C20423000474 --run baseline \
        --out-dir docs/stt-model-replacement/results/group1
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = "com.example.sttchat"
ACTIVITY = f"{PACKAGE}/.SttChatActivity"
DEVICE_BASE = f"/sdcard/Android/data/{PACKAGE}/files/benchmark"
# Same rule as SttBenchmark.isValidName() in the app.
SAFE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}")


def find_adb() -> str:
    """Return the adb executable: $ADB, adb on PATH, or the default Android SDK path."""
    candidates = [os.environ.get("ADB"), shutil.which("adb")]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            str(Path(local_app_data) / "Android/Sdk/platform-tools/adb.exe")
        )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("adb not found. Add platform-tools to PATH or set ADB.")


class Adb:
    def __init__(self, executable: str, serial: str | None) -> None:
        self.prefix = [executable] + (["-s", serial] if serial else [])

    def run(self, *args: str, check: bool = True, timeout: int = 600) -> str:
        result = subprocess.run(
            self.prefix + list(args),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise SystemExit(
                f"adb {' '.join(args)} failed:\n{result.stdout}{result.stderr}"
            )
        return result.stdout.strip()

    def shell(self, command: str, check: bool = True) -> str:
        return self.run("shell", command, check=check, timeout=60)


def push_with_retry(adb: Adb, local: Path, remote: str, attempts: int = 3) -> None:
    """Pushes one file and checks its size on the phone. USB connections can drop."""
    expected = local.stat().st_size
    for attempt in range(1, attempts + 1):
        adb.run("push", str(local), remote, check=False, timeout=300)
        size = adb.shell(f"stat -c %s '{remote}' 2>/dev/null", check=False)
        if size == str(expected):
            return
        print(
            f"  {local.name}: copy incomplete (attempt {attempt} of {attempts}), retrying"
        )
        time.sleep(2)
    raise SystemExit(
        f"Could not copy {local.name} to the phone. Reconnect the USB cable and run again."
    )


def parse_args() -> argparse.Namespace:
    default_testset = Path(__file__).resolve().parent.parent / "testset"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serial", help="Device serial from `adb devices`.")
    parser.add_argument("--testset", default=default_testset, type=Path)
    parser.add_argument(
        "--input-name",
        help="Folder name on the phone. Default: name of the --testset folder.",
    )
    parser.add_argument("--run", required=True, help="Run name, e.g. baseline.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--repeat", type=int, default=3, help="Decodes per file (1-10)."
    )
    parser.add_argument("--apk", type=Path, help="Install this APK first.")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds to wait.")
    parser.add_argument(
        "--skip-push", action="store_true", help="WAV files are already on the phone."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adb = Adb(find_adb(), args.serial)
    input_name = args.input_name or args.testset.name
    input_dir = f"{DEVICE_BASE}/input/{input_name}"
    output_dir = f"{DEVICE_BASE}/output"
    for name in (input_name, args.run):
        if not SAFE_NAME.fullmatch(name):
            raise SystemExit(
                f"Invalid name '{name}': use letters, digits, '.', '_', '-'."
            )

    existing = args.out_dir / f"{args.run}.tsv"
    if existing.exists():
        raise SystemExit(
            f"{existing} already exists. Results are never overwritten: "
            "choose a new --run name, for example baseline-2."
        )

    state = adb.run("get-state", check=False)
    if state != "device":
        raise SystemExit(
            f"Phone state is '{state or 'not found'}'. Reconnect USB, unlock the phone "
            "and accept the USB debugging prompt."
        )
    model = adb.shell("getprop ro.product.model")
    android = adb.shell("getprop ro.build.version.release")
    print(f"Phone: {model}, Android {android}")

    if args.apk:
        print(
            f"Installing {args.apk} (accept the prompt on the phone if one appears)..."
        )
        adb.run("install", "-r", str(args.apk), timeout=900)

    if not args.skip_push:
        wav_files = sorted(args.testset.glob("*.wav"))
        if not wav_files:
            raise SystemExit(f"No WAV files in {args.testset}")
        print(f"Pushing {len(wav_files)} WAV files to {input_dir} ...")
        adb.shell(f"rm -rf '{input_dir}' && mkdir -p '{input_dir}' '{output_dir}'")
        for wav in wav_files:
            push_with_retry(adb, wav, f"{input_dir}/{wav.name}")

    adb.shell(f"rm -f '{output_dir}/{args.run}'.*", check=False)
    adb.shell(f"am force-stop {PACKAGE}")
    adb.shell(
        f"am start -n {ACTIVITY} --es benchmark_input {input_name} "
        f"--es benchmark_run {args.run} --ei benchmark_repeat {args.repeat}"
    )
    print("Benchmark running on the phone. Keep it plugged in, unlocked and cool.")

    start = time.monotonic()
    done = ""
    while time.monotonic() - start < args.timeout:
        done = adb.shell(f"cat '{output_dir}/{args.run}.done' 2>/dev/null", check=False)
        if done:
            break
        elapsed = int(time.monotonic() - start)
        print(f"\r  waiting... {elapsed} s", end="", flush=True)
        time.sleep(3)
    print()
    if not done:
        raise SystemExit(
            "Timeout. Check the phone screen and `adb logcat -s SttBenchmark`."
        )
    if done != "ok":
        raise SystemExit(f"Benchmark failed on the phone: {done}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pulled = []
    for suffix in (".tsv", ".summary.tsv", "-processed.tsv"):
        remote = f"{output_dir}/{args.run}{suffix}"
        if adb.shell(f"ls '{remote}' 2>/dev/null", check=False):
            local = args.out_dir / f"{args.run}{suffix}"
            adb.run("pull", remote, str(local))
            pulled.append(local)

    summary_file = args.out_dir / f"{args.run}.summary.tsv"
    print()
    for line in summary_file.read_text(encoding="utf-8").splitlines()[1:]:
        key, value = line.split("\t", 1)
        print(f"  {key:20} {value}")
    if args.apk:
        print(f"  {'apk_mb':20} {args.apk.stat().st_size / (1024 * 1024):.1f}")

    reference = args.testset / "transcripts.tsv"
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from wer_report import load_tsv, word_errors
    except ImportError:
        print("\nInstall jiwer (uv run --with jiwer ...) to print WER.")
    else:
        references = load_tsv(reference)
        for path in pulled:
            if path.name.endswith(".summary.tsv"):
                continue
            hypotheses = load_tsv(path)
            errors = words = 0
            for name, text in references.items():
                if name in hypotheses:
                    e, w = word_errors(text, hypotheses[name])
                    errors += e
                    words += w
            if words:
                print(f"  WER {path.name:28} {errors / words:.2%} ({errors}/{words})")
    print("\nSaved:", ", ".join(str(path) for path in pulled))


if __name__ == "__main__":
    main()
