#!/usr/bin/env python3
"""
merge_majority_vote.py
----------------------
Merge 2 to 5 Amazon Transcribe JSON outputs for the *same* audio into a single transcript
via a simple majority-vote alignment. This can reduce random mis-hearings. Designed
to work with JSONs produced by Amazon Transcribe (standard jobs).

Approach (simple, robust, not perfect):
- Extract word tokens with (start_time, content) from each JSON.
- Round start_time to a fixed resolution bucket (default 0.2s) to align words across runs.
- For each bucket, compute the most frequent normalized token (casefolded) with a tie-breaker by highest average confidence.
- Re-inject punctuation by the most common punctuation token that appeared *after* the last word in that bucket window.
- Output a plain-text transcript (or JSON with confidences if --json-out).

Ambiguities / caveats:
- If two services drift in timing significantly, bucket-based alignment may misalign. Tune --bucket-sec accordingly.
- This ignores speaker attribution. After merging, you can run diarization_to_markdown.py against a single best JSON to get speaker turns.
- If your three inputs were different *services* (e.g., Transcribe + Whisper), ensure they all have per-word timestamps; otherwise this script will skip ones without them.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

def extract_words(obj: Dict[str, Any]) -> List[Tuple[float, str, float]]:
    """Return list of (start_time, word, confidence)."""
    out = []
    for it in obj.get("results", {}).get("items", []):
        if it.get("type") == "pronunciation":
            try:
                start = float(it["start_time"])
                word = it["alternatives"][0]["content"]
                conf = float(it["alternatives"][0].get("confidence", "0.0"))
                out.append((start, word, conf))
            except Exception:
                continue
    return out

def extract_punct(obj: Dict[str, Any]) -> List[Tuple[float, str]]:
    """Heuristic: associate punctuation with the *end_time* of the previous word when available.
    Amazon Transcribe punctuation items do not have timestamps; we attach them to the previous word's time
    during pass-through collection.
    """
    puncts = []
    last_word_time = None
    for it in obj.get("results", {}).get("items", []):
        if it.get("type") == "pronunciation":
            try:
                last_word_time = float(it["start_time"])
            except Exception:
                pass
        else:
            if last_word_time is not None:
                puncts.append((last_word_time, it["alternatives"][0]["content"]))
    return puncts

def bucketize(t: float, bucket: float) -> float:
    return round(t / bucket) * bucket

def majority_vote_word(buckets: Dict[float, List[Tuple[str, float]]]) -> List[Tuple[float, str, float]]:
    merged = []
    for b in sorted(buckets.keys()):
        tokens = buckets[b]  # list of (word, conf)
        if not tokens:
            continue
        counts = Counter([w.casefold() for w, _ in tokens])
        best_norm, _ = counts.most_common(1)[0]
        # Among candidates that match best_norm, average confidence as tie-breaker
        confs = [c for (w, c) in tokens if w.casefold() == best_norm]
        avg_conf = sum(confs) / max(1, len(confs))
        # Use the original casing from the first occurrence with that norm
        original = next((w for (w, c) in tokens if w.casefold() == best_norm), best_norm)
        merged.append((b, original, avg_conf))
    return merged

def majority_vote_punct(punct_buckets: Dict[float, List[str]]) -> Dict[float, str]:
    out = {}
    for b in sorted(punct_buckets.keys()):
        if punct_buckets[b]:
            counts = Counter(punct_buckets[b])
            out[b] = counts.most_common(1)[0][0]
    return out

def compose_text(words: List[Tuple[float, str, float]], punct_at: Dict[float, str]) -> str:
    pieces = []
    last_added_bucket = None
    for (b, w, _c) in words:
        if pieces:
            pieces.append(" ")
        pieces.append(w)
        last_added_bucket = b
        if b in punct_at:
            pieces.append(punct_at[b])
    return "".join(pieces)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="2–5 Transcribe JSONs for the same audio")
    ap.add_argument("--out", default="merged_transcript.txt", help="Output text file")
    ap.add_argument("--bucket-sec", type=float, default=0.2, help="Time bucket size in seconds")
    ap.add_argument("--json-out", action="store_true", help="Write JSON with per-word confidence instead of text")
    args = ap.parse_args()

    if not (2 <= len(args.inputs) <= 5):
        raise SystemExit("Please provide 2–5 inputs.")

    word_buckets: Dict[float, List[Tuple[str, float]]] = defaultdict(list)
    punct_buckets: Dict[float, List[str]] = defaultdict(list)

    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        words = extract_words(obj)
        puncts = extract_punct(obj)
        for (t, w, c) in words:
            b = bucketize(t, args.bucket_sec)
            word_buckets[b].append((w, c))
        for (t, p) in puncts:
            b = bucketize(t, args.bucket_sec)
            punct_buckets[b].append(p)

    merged_words = majority_vote_word(word_buckets)
    merged_punct = majority_vote_punct(punct_buckets)

    if args.json_out:
        data = [{"time": t, "word": w, "avg_conf": c, "punct": merged_punct.get(t)} for (t, w, c) in merged_words]
        Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        text = compose_text(merged_words, merged_punct)
        Path(args.out).write_text(text, encoding="utf-8")

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
