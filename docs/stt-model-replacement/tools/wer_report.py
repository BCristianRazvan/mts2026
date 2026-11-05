"""Compare one or more result TSV files against transcripts.tsv.

Every result file must have a header row with the columns `file` and
`hypothesis` (stt_benchmark.py and llm_correct.py write this format).
The script prints a Markdown table with per-file WER and the overall WER,
so the output can be pasted into the group report.

Example:

    uv run --no-project --python 3.12 --with jiwer \
        python docs/stt-model-replacement/tools/wer_report.py \
        results/group1/baseline.tsv results/group1/candidate.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jiwer

_normalize = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def normalize(text: str) -> str:
    """Apply the shared normalization: lowercase, no punctuation, single spaces."""
    return _normalize(text)


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (substitutions + deletions + insertions, reference word count)."""
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)
    words = len(reference.split())
    if not hypothesis:
        return words, words
    output = jiwer.process_words(reference, hypothesis)
    return output.substitutions + output.deletions + output.insertions, words


def load_tsv(path: Path) -> dict[str, str]:
    """Load transcripts.tsv (no header) or a result TSV (header with `file`)."""
    # utf-8-sig: also reads files that PowerShell saved with a BOM.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        lines = [line.rstrip("\r\n") for line in handle if line.strip()]
    if lines and lines[0].split("\t")[0] == "file":
        reader = csv.DictReader(lines, delimiter="\t")
        return {row["file"]: row.get("hypothesis") or "" for row in reader}
    return dict(line.split("\t", 1) for line in lines)


def main() -> None:
    default_reference = (
        Path(__file__).resolve().parent.parent / "testset" / "transcripts.tsv"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--reference", default=default_reference, type=Path)
    args = parser.parse_args()

    references = load_tsv(args.reference)
    stems = [path.stem for path in args.results]
    results = {
        # Use folder/name when two files have the same name (e.g. PC and phone runs).
        (
            f"{path.parent.name}/{path.stem}"
            if stems.count(path.stem) > 1
            else path.stem
        ): (load_tsv(path))
        for path in args.results
    }
    totals = {name: [0, 0] for name in results}

    print("| File | " + " | ".join(f"{name} WER" for name in results) + " |")
    print("|---|" + "---:|" * len(results))
    for file_name, reference in references.items():
        cells = []
        for name, hypotheses in results.items():
            if file_name not in hypotheses:
                cells.append("missing")
                continue
            errors, words = word_errors(reference, hypotheses[file_name])
            totals[name][0] += errors
            totals[name][1] += words
            cells.append(f"{errors / words:.1%}")
        print(f"| `{file_name}` | " + " | ".join(cells) + " |")
    overall = [f"**{e / w:.1%}**" if w else "n/a" for e, w in totals.values()]
    print("| **Overall** | " + " | ".join(overall) + " |")


if __name__ == "__main__":
    main()
