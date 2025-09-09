# Bulk Transcription Toolkit (Amazon Transcribe)

This toolkit gives you three executables to minimize human input on large batches of recordings:

1. **`transcribe_batch.py`** — Start Amazon Transcribe (or Call Analytics) jobs in bulk from a CSV index.
2. **`diarization_to_markdown.py`** — Convert a Transcribe JSON into a clean, speaker-named Markdown transcript.
3. **`merge_majority_vote.py`** — Merge 2–5 JSON transcripts (e.g., multiple Transcribe runs) by majority vote.

## 1) transcribe_batch.py

### CSV Columns
- `s3_uri` (required) — `s3://bucket/key` for the audio file.
- `guest_name` (required) — used for tagging and downstream mapping.
- `job_name` (optional) — unique job name. If omitted, one is generated from the s3_uri + guest.
- `language_code` (optional, default `en-US`).
- `vocab_name` (optional) — custom vocabulary to boost recognition.
- `categories` (optional, Call Analytics) — comma-separated category names.
- `channel_identification` (optional) — `true`/`false`. If `true`, uses channel identification (simpler attribution if you recorded multichannel).
- `max_speakers` (optional) — default `2` when diarization is used.
- `show_pii` (optional) — `true`/`false`. If true, enables simple content redaction (PII).
- `show_transcript` (optional) — `true`/`false` (only used in `--wait` mode).

### Examples
Standard transcription with diarization:
```bash
python transcribe_batch.py --csv index.csv --output-bucket my-bucket --output-prefix transcripts/ --wait --show-transcript
```

Call Analytics (requires an IAM role ARN with S3/Transcribe access):
```bash
python transcribe_batch.py --csv index.csv --output-bucket my-bucket --output-prefix analytics/ --call-analytics --role-arn arn:aws:iam::123456789012:role/MyTranscribeRole --wait
```

### Notes / Ambiguities
- EventBridge/Lambda orchestration is not included; this script does optional polling with `--wait`.
- If you consistently record stereo with known channels, prefer `channel_identification=true` per row or `--force-channel`.
- If you need specific redaction entities, adjust the `ContentRedaction` block in code.
- Region, quotas, and concurrency are your environment’s concern; tune `--region` and job batching as needed.

## 2) diarization_to_markdown.py

Turn one JSON into a Markdown transcript with real speaker names.
You can pass a mapping file (JSON or YAML):
```bash
python diarization_to_markdown.py --json output.json --map speaker_map.yaml --out transcript.md
```

Or rely on a heuristic when you have exactly two speakers:
```bash
python diarization_to_markdown.py --json output.json --host-name "Host" --guest-name "Anne Cori" --out transcript.md
```

## 3) merge_majority_vote.py

Combine multiple JSON transcripts of the *same* audio (e.g., several Transcribe runs with different settings) into a single text:
```bash
python merge_majority_vote.py --inputs run1.json run2.json run3.json --out merged.txt --bucket-sec 0.2
```

If you want a structured output:
```bash
python merge_majority_vote.py --inputs run1.json run2.json run3.json --json-out --out merged.json
```

### Caveats
- Uses time-bucket alignment (default 0.2s). If your runs drift, adjust `--bucket-sec`.
- Speaker attribution is not handled here; do diarization on a single JSON and/or apply name mapping later.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

Make sure your AWS credentials are configured (env vars, ~/.aws/credentials, or instance profile).

---

**Workflow Suggestion**
1. Run `transcribe_batch.py` from your CSV to start all jobs.
2. When jobs finish, download the JSONs (S3) you want to merge; optionally run `merge_majority_vote.py`.
3. For the final document, run `diarization_to_markdown.py` against a chosen JSON and supply a name mapping (from your CSV) so the transcript reads like `**Host:** ...  **Thomas Sowell:** ...`.

If you want these scripts to *also* download results automatically from S3 and write local files, we can add a downloader step next.
