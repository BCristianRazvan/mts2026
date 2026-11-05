"""Step 2 of 2: adds car noise to the clean test files and writes the final WAVs.

Reads testset/clean/<file> (from generate_testset.ps1) and the settings in
conditions.tsv, and writes testset/<file>:

- noise `car`: low-frequency road rumble plus a weak engine hum, mixed at
  `snr_db` (signal-to-noise ratio; lower means more noise).
- bandpass `yes`: keeps only 300-3400 Hz, like a cheap or distant microphone.

The noise uses a fixed random seed per file, so the output is the same on every
PC. Output: 16 kHz, mono, 16-bit PCM.

Usage (from the repository root):

    uv run --no-project --python 3.12 --with numpy \
        python docs/stt-model-replacement/testset/add_noise.py
"""

from __future__ import annotations

import csv
import wave
import zlib
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
LEAD_SILENCE_SECONDS = 0.3
TAIL_SILENCE_SECONDS = 0.5


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            2,
        ):
            raise SystemExit(f"{path.name}: expected 16 kHz, mono, 16-bit PCM")
        data = wav.readframes(wav.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def band_filter(samples: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    spectrum[(frequencies < low_hz) | (frequencies > high_hz)] = 0
    return np.fft.irfft(spectrum, n=len(samples))


def car_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Road rumble (brown noise, mostly below 500 Hz) plus a weak engine hum."""
    brown = np.cumsum(rng.standard_normal(length))
    brown -= np.convolve(brown, np.ones(801) / 801, mode="same")  # remove drift
    rumble = band_filter(brown, 20, 500)
    hiss = band_filter(rng.standard_normal(length), 500, 4000) * 0.15
    t = np.arange(length) / SAMPLE_RATE
    hum = 0.3 * np.sin(2 * np.pi * 95 * t) + 0.15 * np.sin(2 * np.pi * 190 * t)
    noise = rumble / np.std(rumble) + hiss / np.std(hiss) * 0.3 + hum
    return noise / np.std(noise)


def main() -> None:
    testset = Path(__file__).resolve().parent
    with (testset / "conditions.tsv").open(encoding="utf-8", newline="") as handle:
        conditions = list(csv.DictReader(handle, delimiter="\t"))

    for row in conditions:
        speech = read_wav(testset / "clean" / row["file"])
        speech = np.concatenate(
            [
                np.zeros(int(LEAD_SILENCE_SECONDS * SAMPLE_RATE)),
                speech,
                np.zeros(int(TAIL_SILENCE_SECONDS * SAMPLE_RATE)),
            ]
        )
        if row["bandpass"] == "yes":
            speech = band_filter(speech, 300, 3400)

        mixed = speech
        if row["noise"] == "car":
            rng = np.random.default_rng(zlib.crc32(row["file"].encode()))
            noise = car_noise(len(speech), rng)
            voiced = np.abs(speech) > 0.01
            speech_power = np.mean(speech[voiced] ** 2)
            noise_power = speech_power / 10 ** (float(row["snr_db"]) / 10)
            mixed = speech + noise * np.sqrt(noise_power)

        peak = np.max(np.abs(mixed))
        mixed = mixed * (0.7 / peak)  # same peak level for every file
        write_wav(testset / row["file"], mixed)
        print(
            f"{row['file']}  noise={row['noise']} snr={row['snr_db']} dB "
            f"bandpass={row['bandpass']}  {len(mixed) / SAMPLE_RATE:.1f} s"
        )


if __name__ == "__main__":
    main()
