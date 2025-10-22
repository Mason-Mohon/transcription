import os, glob, sys
from openai import OpenAI

# If you keep your API key in .env, load it:
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

client = OpenAI()  # uses OPENAI_API_KEY env var

CHUNK_DIR = "/Users/mason/Desktop/Transcription Options/chunks"
OUT_DIR   = "/Users/mason/Desktop/Transcription Options/Other Transcripts"
MODE      = "srt"   # "text", "srt", "vtt", or "verbose_json"

paths = sorted(glob.glob(os.path.join(CHUNK_DIR, "2011_*.m4a")))
if not paths:
    print("No chunks found. Check CHUNK_DIR pattern.")
    sys.exit(1)

full_txt = []
srt_parts = []
counter = 0

def renumber_srt(srt_text, start_index):
    out = []
    idx = start_index
    for block in srt_text.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2: 
            continue
        # replace first line (index) with our running counter
        lines[0] = str(idx)
        out.append("\n".join(lines))
        idx += 1
    return "\n\n".join(out), idx

for p in paths:
    with open(p, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format=MODE
        )

    if MODE == "text":
        # Append to a big text file
        full_txt.append(resp)
    elif MODE == "srt":
        part = resp  # already a string
        part_renum, counter = renumber_srt(part, counter + 1)
        srt_parts.append(part_renum)
    elif MODE == "vtt":
        # Simple concat is usually fine for VTT
        srt_parts.append(resp)
    else:
        # verbose_json – write one JSON per chunk
        base = os.path.splitext(os.path.basename(p))[0]
        outp = os.path.join(OUT_DIR, f"{base}.json")
        with open(outp, "w", encoding="utf-8") as fo:
            fo.write(resp.model_dump_json(indent=2))

# Write combined outputs
if MODE == "text" and full_txt:
    with open(os.path.join(OUT_DIR, "2011_full.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n\n".join(full_txt))

if MODE in ("srt", "vtt") and srt_parts:
    ext = MODE
    with open(os.path.join(OUT_DIR, f"2011_full.{ext}"), "w", encoding="utf-8") as fo:
        # For SRT we renumbered cues; for VTT this is just concatenation
        fo.write("\n\n".join(srt_parts))

print("Done.")
