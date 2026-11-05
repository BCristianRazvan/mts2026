"""Run every WAV in the test set through a sherpa-onnx streaming model.

Writes one TSV with the recognized text and timing for each file, then prints
WER against transcripts.tsv. Both workshop groups use this script, so results
from different machines and models stay comparable.

Default settings match OfflineSpeechRecognizer.prepare() in the Android app
(transducer, modified_beam_search, 4 active paths, hotwords score 1.5).

Example (current app model, from the repository root):

    uv run --no-project --python 3.12 \
        --with sherpa-onnx==1.13.3 --with jiwer --with numpy --with psutil \
        python docs/stt-model-replacement/tools/stt_benchmark.py \
        --model-dir app/src/main/assets/models/stt \
        --hotwords \
        --out docs/stt-model-replacement/results/group1/pc-baseline.tsv
"""

from __future__ import annotations

import argparse
import csv
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx
from wer_report import load_tsv, normalize, word_errors

SAMPLE_RATE = 16000
TAIL_PADDING_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    default_testset = Path(__file__).resolve().parent.parent / "testset"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--testset", default=default_testset, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output TSV path.")
    parser.add_argument(
        "--family",
        choices=["transducer", "nemo_ctc", "zipformer2_ctc"],
        default="transducer",
    )
    parser.add_argument("--encoder", default="encoder.int8.onnx")
    parser.add_argument("--decoder", default="decoder.int8.onnx")
    parser.add_argument("--joiner", default="joiner.int8.onnx")
    parser.add_argument("--model", default="model.int8.onnx", help="CTC model file.")
    parser.add_argument("--tokens", default="tokens.txt")
    parser.add_argument(
        "--decoding-method",
        default=None,
        help="Default: modified_beam_search for transducer, greedy_search for CTC.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--hotwords",
        action="store_true",
        help="Use hotwords.txt and bpe.vocab from --model-dir, like the app.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Repeat count for timing.")
    return parser.parse_args()


def create_recognizer(args: argparse.Namespace) -> sherpa_onnx.OnlineRecognizer:
    model_dir = args.model_dir
    common = {
        "tokens": str(model_dir / args.tokens),
        "num_threads": args.threads,
        "sample_rate": SAMPLE_RATE,
        "feature_dim": 80,
    }
    if args.family == "transducer":
        extra = {}
        if args.hotwords:
            extra = {
                "hotwords_file": str(model_dir / "hotwords.txt"),
                "hotwords_score": 1.5,
                "modeling_unit": "bpe",
                "bpe_vocab": str(model_dir / "bpe.vocab"),
            }
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(model_dir / args.encoder),
            decoder=str(model_dir / args.decoder),
            joiner=str(model_dir / args.joiner),
            decoding_method=args.decoding_method or "modified_beam_search",
            max_active_paths=4,
            **common,
            **extra,
        )
    if args.hotwords:
        raise SystemExit("Hotwords work only with transducer models.")
    factory = (
        sherpa_onnx.OnlineRecognizer.from_nemo_ctc
        if args.family == "nemo_ctc"
        else sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc
    )
    return factory(
        model=str(model_dir / args.model),
        decoding_method=args.decoding_method or "greedy_search",
        **common,
    )


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            2,
        ):
            raise SystemExit(f"{path.name}: expected 16 kHz, mono, 16-bit PCM")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768


def recognize(recognizer: sherpa_onnx.OnlineRecognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    padding = np.zeros(int(TAIL_PADDING_SECONDS * SAMPLE_RATE), dtype=np.float32)
    stream.accept_waveform(SAMPLE_RATE, padding)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return recognizer.get_result(stream).strip()


def peak_memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss  # peak_wset: Windows only
    return peak / (1024 * 1024)


def main() -> None:
    args = parse_args()
    references = load_tsv(args.testset / "transcripts.tsv")

    load_start = time.perf_counter()
    recognizer = create_recognizer(args)
    load_ms = (time.perf_counter() - load_start) * 1000

    rows = []
    total_errors = total_words = 0
    total_audio = total_decode = 0.0
    for name, reference in references.items():
        samples = read_wav(args.testset / name)
        audio_seconds = len(samples) / SAMPLE_RATE
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            hypothesis = recognize(recognizer, samples)
            timings.append(time.perf_counter() - start)
        decode_seconds = float(np.median(timings))
        errors, words = word_errors(reference, hypothesis)
        total_errors += errors
        total_words += words
        total_audio += audio_seconds
        total_decode += decode_seconds
        rows.append(
            {
                "file": name,
                "hypothesis": hypothesis,
                "audio_s": f"{audio_seconds:.2f}",
                "decode_ms": f"{decode_seconds * 1000:.0f}",
                "rtf": f"{decode_seconds / audio_seconds:.3f}",
                "wer": f"{errors / words:.2%}",
            }
        )
        print(f"{name}  WER {errors / words:6.2%}  REF: {normalize(reference)}")
        print(f"{'':11}  {'':10}  HYP: {normalize(hypothesis)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    model_mb = sum(f.stat().st_size for f in args.model_dir.iterdir() if f.is_file())
    memory = peak_memory_mb()
    print()
    print(f"Model folder size : {model_mb / (1024 * 1024):.1f} MB")
    print(f"Model load time   : {load_ms:.0f} ms (PC)")
    print(
        f"Total decode time : {total_decode * 1000:.0f} ms for {total_audio:.1f} s audio"
    )
    print(f"RTF (PC)          : {total_decode / total_audio:.3f}")
    if memory is not None:
        print(f"Peak memory (PC)  : {memory:.0f} MB")
    print(
        f"WER               : {total_errors / total_words:.2%} "
        f"({total_errors} errors / {total_words} reference words)"
    )
    print(f"Saved             : {args.out}")


if __name__ == "__main__":
    main()
