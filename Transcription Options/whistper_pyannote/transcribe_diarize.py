import os
import math
from pathlib import Path
from typing import List, Tuple

# Transcription
from faster_whisper import WhisperModel

# Diarization
from pyannote.audio import Pipeline

# Utilities
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word]


def sec_to_srt_time(t: float) -> str:
    t = max(0.0, t)
    hours = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def pick_speaker_for_span(diarization, t0: float, t1: float) -> str:
    """Choose the speaker label with the most overlap over [t0, t1]."""
    from pyannote.core import Segment as PSeg
    window = PSeg(t0, t1)
    overlaps = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        ov = turn & window
        if ov is not None:
            dur = ov.duration
            overlaps[speaker] = overlaps.get(speaker, 0.0) + dur
    if not overlaps:
        return "SPK?"
    label = max(overlaps, key=overlaps.get)
    return str(label)


def merge_contiguous_by_speaker(lines: List[Tuple[str, float, float, str]], max_gap: float = 0.6) -> List[Tuple[str, float, float, str]]:
    """
    Merge adjacent lines if same speaker and the gap is small.
    """
    if not lines:
        return []
    merged = [list(lines[0])]
    for spk, s, e, text in lines[1:]:
        prev = merged[-1]
        if spk == prev[0] and s - prev[2] <= max_gap:
            prev[2] = e
            prev[3] = (prev[3] + " " + text).strip()
        else:
            merged.append([spk, s, e, text])
    return [tuple(x) for x in merged]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe with Whisper (faster-whisper) and diarize with pyannote.")
    parser.add_argument("audio", help="Path to audio/video file (mp3, wav, mp4, etc.)")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (env HF_TOKEN if omitted)")
    parser.add_argument("--output", default=None, help="Output base path (without extension). Defaults to alongside input.")
    parser.add_argument("--language", default=None, help="Language hint, e.g., 'en'.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true", help="Enable VAD filtering in faster-whisper.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    out_base = Path(args.output) if args.output else audio_path.with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # 1) Transcribe with faster-whisper
    print("Loading Whisper model...")
    # Favor int8 on CPU for speed/memory; float16 on GPU
    compute_type = "int8" if args.device == "cpu" else "float16"
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    print("Transcribing...")
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=args.beam_size,
        language=args.language,
        vad_filter=args.vad_filter,
        word_timestamps=True,
    )

    w_segments: List[Segment] = []
    for seg in segments_iter:
        words: List[Word] = []
        if seg.words:
            for w in seg.words:
                if w.start is not None and w.end is not None:
                    words.append(Word(start=float(w.start), end=float(w.end), text=w.word))
        w_segments.append(Segment(start=float(seg.start), end=float(seg.end), text=seg.text.strip(), words=words))

    # 2) Diarize with pyannote
    print("Loading pyannote pipeline (speaker diarization)...")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("No Hugging Face token provided. Use --hf-token or set HF_TOKEN in your .env.")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    print("Running diarization...")
    diarization = pipeline(str(audio_path))

    # 3) Assign a speaker to each Whisper segment (use overlap majority)
    print("Assigning speakers to segments...")
    spk_lines: List[Tuple[str, float, float, str]] = []
    for seg in w_segments:
        if seg.words:
            from collections import Counter
            votes = Counter()
            for w in seg.words:
                mid = (w.start + w.end) / 2.0
                spk = pick_speaker_for_span(diarization, mid - 0.05, mid + 0.05)
                votes[spk] += 1
            speaker = votes.most_common(1)[0][0] if votes else pick_speaker_for_span(diarization, seg.start, seg.end)
        else:
            speaker = pick_speaker_for_span(diarization, seg.start, seg.end)
        spk_lines.append((speaker, seg.start, seg.end, seg.text))

    # Normalize speaker labels to SPK00, SPK01, ... in order of first appearance
    spk_map = {}
    next_id = 0
    normalized_lines = []
    for spk, s, e, text in spk_lines:
        if spk not in spk_map:
            spk_map[spk] = f"SPK{next_id:02d}"
            next_id += 1
        normalized_lines.append((spk_map[spk], s, e, text))

    # Merge nearby lines by same speaker
    merged = merge_contiguous_by_speaker(normalized_lines, max_gap=0.6)

    # 4) Write outputs
    txt_path = Path(f"{out_base}_diarized.txt")
    with txt_path.open("w", encoding="utf-8") as f:
        for spk, s, e, text in merged:
            f.write(f"[{spk}] {sec_to_srt_time(s)} - {sec_to_srt_time(e)}  {text}\n")

    srt_path = Path(f"{out_base}_diarized.srt")
    with srt_path.open("w", encoding="utf-8") as f:
        for i, (spk, s, e, text) in enumerate(merged, start=1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_srt_time(s)} --> {sec_to_srt_time(e)}\n")
            f.write(f"{spk}: {text.strip()}\n\n")

    csv_path = None
    try:
        import pandas as pd
        df_rows = [{"speaker": spk, "start": s, "end": e, "text": text} for spk, s, e, text in merged]
        if df_rows:
            df = pd.DataFrame(df_rows)
            csv_path = Path(f"{out_base}_diarized.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
    except Exception as e:
        print("CSV export skipped:", e)

    # 5) Friendly summary
    print("\n=== TRANSCRIPTION COMPLETE ===")
    print(f"Text file: {txt_path}")
    print(f"SRT file:  {srt_path}")
    if csv_path is not None:
        print(f"CSV file:  {csv_path}")
    print("================================\n")


if __name__ == "__main__":
    main()
