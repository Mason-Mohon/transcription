#!/usr/bin/env python3
"""
diarization_to_markdown.py
--------------------------
Convert an Amazon Transcribe JSON into a speaker-named Markdown transcript.

New features:
- --keep-top-speakers N : Keep the N speakers with the most total speaking time; bucket the rest.
- --other-label TEXT     : Label for non-top speakers (default: "Other").
- --names "A,B,C"        : Assign these names to the top-N speakers in order of speaking time.
- --map mapping.json     : Explicit mapping { "spk_0": "Host", ... } (honored for top speakers).
- If neither --map nor --names is provided, top-N are named "Speaker 1..N" by duration order.

Notes:
- Amazon Transcribe doesn't identify real people; it labels speakers like spk_0..spk_k.
- If diarization is noisy (e.g., ads), keeping top-N collapses brief voices into "Other".
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def load_mapping(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if path.endswith((".yaml",".yml")):
        try:
            import yaml
        except ImportError:
            raise SystemExit("PyYAML not installed. Run: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def build_speaker_timeline(speaker_labels: Dict[str, Any]) -> List[Tuple[float, float, str]]:
    timeline = []
    for seg in speaker_labels.get("segments", []):
        spk = seg["speaker_label"]
        start = float(seg["start_time"])
        end = float(seg["end_time"])
        timeline.append((start, end, spk))
    return timeline

def find_speaker(timeline: List[Tuple[float,float,str]], t: float) -> Optional[str]:
    for start, end, spk in timeline:
        if start <= t <= end:
            return spk
    return None

def speaking_durations(timeline: List[Tuple[float,float,str]]) -> Dict[str, float]:
    dur: Dict[str, float] = {}
    for s, e, spk in timeline:
        dur[spk] = dur.get(spk, 0.0) + max(0.0, e - s)
    return dur

def assign_names_for_topN(sorted_speakers: List[str], names_csv: Optional[str], keep_top: int) -> Dict[str, str]:
    if not keep_top or keep_top <= 0:
        return {}
    top = sorted_speakers[:keep_top]
    mapping: Dict[str, str] = {}
    if names_csv:
        names = [n.strip() for n in names_csv.split(",") if n.strip()]
        for i, spk in enumerate(top):
            label = names[i] if i < len(names) else f"Speaker {i+1}"
            mapping[spk] = label
    else:
        for i, spk in enumerate(top):
            mapping[spk] = f"Speaker {i+1}"
    return mapping

def to_markdown(json_obj: Dict[str, Any],
                explicit_map: Dict[str, str],
                names_csv: Optional[str],
                keep_top: Optional[int],
                other_label: str) -> str:
    res = json_obj["results"]
    timeline = build_speaker_timeline(res.get("speaker_labels", {}))
    items = res.get("items", [])

    # Rank speakers by total duration
    durs = speaking_durations(timeline)
    ranked = sorted(durs.keys(), key=lambda k: durs[k], reverse=True)

    # Build name map
    name_map = dict(explicit_map) if explicit_map else {}
    if keep_top and keep_top > 0:
        auto = assign_names_for_topN(ranked, names_csv, keep_top)
        for spk, label in auto.items():
            name_map.setdefault(spk, label)
    for spk in durs.keys():
        if spk not in name_map:
            name_map[spk] = spk

    allowed: Optional[set] = None
    if keep_top and keep_top > 0:
        allowed = set(ranked[:keep_top])

    utterances: List[str] = []
    cur_spk: Optional[str] = None
    cur_words: List[str] = []

    def flush():
        nonlocal cur_spk, cur_words
        if cur_spk is not None and cur_words:
            label = name_map.get(cur_spk, cur_spk)
            if allowed is not None and cur_spk not in allowed:
                label = other_label
            text = ''.join(cur_words).strip()
            if text:
                if utterances and utterances[-1].startswith(f"**{label}:** "):
                    prev = utterances.pop()
                    utterances.append(prev + " " + text)
                else:
                    utterances.append(f"**{label}:** {text}")
        cur_spk, cur_words = None, []

    last_was_word = False
    for it in items:
        if it["type"] == "pronunciation":
            start_time = float(it["start_time"])
            spk = find_speaker(timeline, start_time) or cur_spk
            if spk != cur_spk:
                flush()
                cur_spk = spk
                last_was_word = False
            if last_was_word:
                cur_words.append(' ')
            cur_words.append(it["alternatives"][0]["content"])
            last_was_word = True
        else:
            cur_words.append(it["alternatives"][0]["content"])
            last_was_word = False

    flush()
    return "\n\n".join(utterances)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Transcribe JSON path")
    ap.add_argument("--out", default="transcript.md", help="Output Markdown path")
    ap.add_argument("--map", help="JSON or YAML mapping file of speaker_label -> real name")
    ap.add_argument("--names", help="Comma-separated names for top-N speakers (duration order)")
    ap.add_argument("--keep-top-speakers", type=int, default=None, help="Keep the N speakers with the most speaking time")
    ap.add_argument("--other-label", default="Other", help="Label for non-top speakers")
    args = ap.parse_args()

    obj = json.loads(Path(args.json).read_text(encoding="utf-8"))
    explicit_map = load_mapping(args.map) if args.map else {}

    md = to_markdown(
        obj,
        explicit_map=explicit_map,
        names_csv=args.names,
        keep_top=args.keep_top_speakers,
        other_label=args.other_label
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()


"""
/Users/mason/opt/anaconda3/envs/transcriptions/bin/python diarization_to_markdown.py \
 --json /Users/mason/Desktop/transcription/-EFLive-test-job-1.json \
 --out /Users/mason/Desktop/transcription/outputs/transcript.md \
 --map /Users/mason/Desktop/transcription/map.json \
 --keep-top-speakers 3 \
 --names "Bill Hayes,Phyllis Schlafly,Thomas Sowell"
"""