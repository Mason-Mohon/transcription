#!/usr/bin/env python3
"""
diarization_to_markdown.py
--------------------------
Convert an Amazon Transcribe JSON into a speaker-named transcript.

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

def to_transcript(json_obj: Dict[str, Any],
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
    segment_start_time: Optional[float] = None
    segment_end_time: Optional[float] = None

    def flush():
        nonlocal cur_spk, cur_words, segment_start_time, segment_end_time
        if cur_spk is not None and cur_words:
            label = name_map.get(cur_spk, cur_spk)
            if allowed is not None and cur_spk not in allowed:
                label = other_label
            text = ''.join(cur_words).strip()
            if text:
                # Format timestamps
                time_info = ""
                if segment_start_time is not None and segment_end_time is not None:
                    start_min = int(segment_start_time // 60)
                    start_sec = segment_start_time % 60
                    end_min = int(segment_end_time // 60)
                    end_sec = segment_end_time % 60
                    time_info = f" [{start_min:02d}:{start_sec:05.2f} - {end_min:02d}:{end_sec:05.2f}]"
                
                if utterances and utterances[-1].split(']')[-1].strip().startswith(f"**{label}:** "):
                    # Continuing same speaker - merge text but update end time
                    prev = utterances.pop()
                    # Extract the existing time range and update end time
                    if '[' in prev and ']' in prev:
                        start_part = prev.split('[')[1].split(' - ')[0]
                        end_min = int(segment_end_time // 60)
                        end_sec = segment_end_time % 60
                        new_time_info = f" [{start_part} - {end_min:02d}:{end_sec:05.2f}]"
                        label_and_text = prev.split(']', 1)[1]
                        utterances.append(f"**{label}:**{new_time_info}{label_and_text} {text}")
                    else:
                        utterances.append(prev + " " + text)
                else:
                    utterances.append(f"**{label}:**{time_info} {text}")
        cur_spk, cur_words = None, []
        segment_start_time, segment_end_time = None, None

    last_was_word = False
    for it in items:
        if it["type"] == "pronunciation":
            start_time = float(it["start_time"])
            end_time = float(it["end_time"])
            spk = find_speaker(timeline, start_time) or cur_spk
            if spk != cur_spk:
                flush()
                cur_spk = spk
                segment_start_time = start_time
                last_was_word = False
            # Update segment end time with the latest word's end time
            segment_end_time = end_time
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
    ap.add_argument("--out", default="transcript.txt", help="Output transcript path")
    ap.add_argument("--map", help="JSON or YAML mapping file of speaker_label -> real name")
    ap.add_argument("--names", help="Comma-separated names for top-N speakers (duration order)")
    ap.add_argument("--keep-top-speakers", type=int, default=None, help="Keep the N speakers with the most speaking time")
    ap.add_argument("--other-label", default="Other", help="Label for non-top speakers")
    args = ap.parse_args()

    obj = json.loads(Path(args.json).read_text(encoding="utf-8"))
    explicit_map = load_mapping(args.map) if args.map else {}

    transcript = to_transcript(
        obj,
        explicit_map=explicit_map,
        names_csv=args.names,
        keep_top=args.keep_top_speakers,
        other_label=args.other_label
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(transcript, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()


"""
/Users/mason/opt/anaconda3/envs/transcriptions/bin/python diarization_to_markdown.py \
 --json /Users/mason/Desktop/transcription/-EFLive-test-job-1.json \
 --out /Users/mason/Desktop/transcription/outputs/transcript.txt \
"""