#!/usr/bin/env python3
"""
diarization_to_markdown.py
--------------------------
Convert a single Amazon Transcribe JSON output into a speaker-named Markdown transcript.

Inputs:
- --json           : path to local Transcribe JSON file (downloaded from S3)
- --out            : output Markdown path (default: transcript.md)
- --map            : YAML or JSON file mapping speakers to real names, e.g. {"spk_0":"Host","spk_1":"Guest: Anne Cori"}
- --host-name      : fallback host name (used by "first speaker is host" heuristic if --map is missing)
- --guest-name     : fallback guest name (used if there are exactly two speakers and no --map provided)
- --channel-mode   : if set, prefers channel-based labeling when present (useful if you recorded multichannel)
- --names          : comma-separated real names mapped by first-seen order (e.g., "Host,Guest A,Guest B")

Ambiguities noted:
- If there are >2 speakers and no --map provided, we keep labels as spk_X unless --names is used.
- If timestamps overlap strangely, we choose the first segment that contains the token start_time.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def load_mapping(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml  # optional
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
    # Linear scan is simpler; replace with binary search if performance is an issue.
    for start, end, spk in timeline:
        if start <= t <= end:
            return spk
    return None

def build_map_from_names(json_obj: Dict[str, Any], names_csv: str) -> Dict[str, str]:
    """Map first-seen speakers to provided names in order."""
    names = [n.strip() for n in names_csv.split(",") if n.strip()]
    if not names:
        return {}
    res = json_obj.get("results", {})
    labels = res.get("speaker_labels", {}).get("segments", [])
    order = []
    seen = set()
    for seg in labels:
        spk = seg.get("speaker_label")
        if spk not in seen:
            seen.add(spk)
            order.append(spk)
    mapping = {}
    for i, spk in enumerate(order):
        if i < len(names):
            mapping[spk] = names[i]
        else:
            break
    return mapping

def to_markdown(json_obj: Dict[str, Any], speaker_name_map: Dict[str, str]) -> str:
    res = json_obj["results"]
    timeline = build_speaker_timeline(res.get("speaker_labels", {}))
    items = res.get("items", [])

    utterances: List[str] = []
    cur_spk: Optional[str] = None
    cur_words: List[str] = []

    def flush():
        nonlocal cur_spk, cur_words
        if cur_spk is not None and cur_words:
            name = speaker_name_map.get(cur_spk, cur_spk)
            text = ''.join(cur_words).strip()
            if text:
                utterances.append(f"**{name}:** {text}")
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

def guess_name_map(json_obj: Dict[str, Any], host_name: Optional[str], guest_name: Optional[str]) -> Dict[str, str]:
    """Heuristic: if exactly two speakers, map first-seen speaker to host_name (or 'Host'),
    the other to guest_name (or 'Guest'). Otherwise, return empty and rely on --map or --names.
    """
    res = json_obj.get("results", {})
    labels = res.get("speaker_labels", {}).get("segments", [])
    order = []
    seen = set()
    for seg in labels:
        spk = seg.get("speaker_label")
        if spk not in seen:
            seen.add(spk)
            order.append(spk)
    if len(order) == 2:
        h = host_name or "Host"
        g = guest_name or "Guest"
        return {order[0]: h, order[1]: g}
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Transcribe JSON path")
    ap.add_argument("--out", default="transcript.md", help="Output Markdown path")
    ap.add_argument("--map", help="JSON or YAML mapping file of speaker_label -> real name")
    ap.add_argument("--host-name", help="Fallback host name if using heuristic")
    ap.add_argument("--guest-name", help="Fallback guest name if using heuristic")
    ap.add_argument("--channel-mode", action="store_true", help="Prefer channel labeling if present (not implemented: uses diarization by default)")
    ap.add_argument("--names", help="Comma-separated real names mapped by first-seen speaker order (e.g., 'Host,Guest A,Guest B')")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping = {}
    if args.map:
        mapping = load_mapping(args.map)
    if args.names and not mapping:
        mapping = build_map_from_names(obj, args.names)
    if not mapping:
        mapping = guess_name_map(obj, args.host_name, args.guest_name)

    md = to_markdown(obj, mapping)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
