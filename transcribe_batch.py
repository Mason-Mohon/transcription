#!/usr/bin/env python3
"""
transcribe_batch.py
-------------------
Bulk-start Amazon Transcribe jobs using a CSV "index" of your audio files.

Your CSV should (at minimum) include:
- s3_uri           : s3://bucket/key of the audio file
- guest_name       : e.g., "Thomas Sowell" (used for tagging and downstream speaker mapping)
Optional columns (if present, they will be used):
- job_name         : unique job name (if omitted, one will be generated)
- language_code    : e.g., "en-US" (default: en-US)
- vocab_name       : name of a custom vocabulary to boost recognition
- categories       : comma-separated Call Analytics category names (only used if --call-analytics is set)
- channel_identification : "true"/"false" (if "true", uses channel identification instead of diarization)
- max_speakers     : int, e.g., 2 (used when diarization is on)
- show_pii         : "true"/"false" (if true, enables content redaction with default settings)  # Ambiguity: adjust to your needs later
- show_transcript  : "true"/"false" (if true, prints job JSON on completion in --wait mode)

USAGE (standard transcription with diarization):
    python transcribe_batch.py --csv index.csv --output-bucket my-output-bucket --output-prefix transcripts/

USAGE (Call Analytics):
    python transcribe_batch.py --csv index.csv --output-bucket my-output-bucket --output-prefix analytics/ --call-analytics

Notes on ambiguities / to specify later:
- If you want to use a specific IAM role for Call Analytics (DataAccessRoleArn), set via --role-arn.
- If you prefer EventBridge-based completion instead of --wait polling, wire that externally (this script only does optional polling).
- If you record true multi-channel audio and want deterministic speaker=channel, pass channel_identification=true per-row OR --force-channel.
- Pricing/quotas/regions are environment-dependent; adjust --region and concurrency as needed.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from typing import Dict, Any, Optional, List

import boto3
from dotenv import load_dotenv

import botocore

def slugify_jobname(name: str) -> str:
    out = ''.join(ch if ch.isalnum() or ch in '-_' else '-' for ch in name)
    return out[:200]  # Transcribe has a max length; keep it safe

def default_job_name(s3_uri: str, guest_name: str) -> str:
    h = hashlib.sha1(s3_uri.encode('utf-8')).hexdigest()[:8]
    g = slugify_jobname(guest_name) if guest_name else "guest"
    base = f"job-{g}-{h}"
    return base

def boolish(val: Optional[str], default: bool=False) -> bool:
    if val is None: return default
    return str(val).strip().lower() in ("1","true","yes","y")

def start_standard_job(client, row: Dict[str, str], args) -> str:
    s3_uri = row["s3_uri"].strip()
    guest = row.get("guest_name","").strip()
    job_name = row.get("job_name") or default_job_name(s3_uri, guest)
    job_name = slugify_jobname(job_name)

    language_code = row.get("language_code") or args.language_code
    vocab_name = row.get("vocab_name") or None
    channel_ident = boolish(row.get("channel_identification")) or args.force_channel
    max_speakers = int(row.get("max_speakers") or (args.max_speakers if not channel_ident else 2))
    show_pii = boolish(row.get("show_pii")) or args.redact_pii

    settings = {
        "ShowSpeakerLabels": (not channel_ident),
        "MaxSpeakerLabels": max_speakers
    }
    if vocab_name:
        settings["VocabularyName"] = vocab_name
    if show_pii:
        # Ambiguity: tune content redaction / PII entity types later if needed.
        settings["ContentRedaction"] = {
            "RedactionType": "PII",
            "RedactionOutput": "redacted"
        }

    kwargs = {
        "TranscriptionJobName": job_name,
        "LanguageCode": language_code,
        "Media": {"MediaFileUri": s3_uri},
        "OutputBucketName": args.output_bucket,
        "Settings": settings,
        "Tags": [
            {"Key":"guest","Value": guest or "unknown"},
            {"Key":"source_csv","Value": os.path.basename(args.csv)},
        ]
    }
    if args.output_prefix:
        kwargs["OutputKey"] = args.output_prefix.rstrip("/") + f"/{job_name}.json"

    if channel_ident:
        kwargs["ChannelIdentification"] = True

    resp = client.start_transcription_job(**kwargs)
    return resp["TranscriptionJob"]["TranscriptionJobName"]

def start_analytics_job(client, row: Dict[str, str], args) -> str:
    s3_uri = row["s3_uri"].strip()
    guest = row.get("guest_name","").strip()
    job_name = row.get("job_name") or default_job_name(s3_uri, guest)
    job_name = slugify_jobname(job_name)

    # Ambiguity: If you need categories per-file, put them in "categories" column (comma-separated).
    categories = [c.strip() for c in (row.get("categories") or "").split(",") if c.strip()]
    vocab_name = row.get("vocab_name") or None
    language_code = row.get("language_code") or args.language_code

    kwargs = {
        "CallAnalyticsJobName": job_name,
        "Media": {"MediaFileUri": s3_uri},
        "OutputLocation": f"s3://{args.output_bucket}/{args.output_prefix or ''}".rstrip("/"),
        "ChannelDefinitions": [{"ChannelId": 0}, {"ChannelId": 1}],  # Ambiguity: adjust channels to your recordings
        "DataAccessRoleArn": args.role_arn,  # REQUIRED for Call Analytics
        "Settings": {"LanguageCode": language_code},
        "CallAnalyticsJobSettings": {}
    }
    if vocab_name:
        kwargs["CallAnalyticsJobSettings"]["VocabularyName"] = vocab_name
    if categories:
        kwargs["CallAnalyticsJobSettings"]["CallAnalyticsCategoryNames"] = categories

    resp = client.start_call_analytics_job(**kwargs)
    return resp["CallAnalyticsJob"]["CallAnalyticsJobName"]

def wait_for_jobs(client, job_names: List[str], analytics: bool=False, show_transcript: bool=False, poll_sec: int=15):
    get_fn = (client.get_call_analytics_job if analytics else client.get_transcription_job)
    status_key = ("CallAnalyticsJob" if analytics else "TranscriptionJob")
    while job_names:
        remaining = []
        for name in job_names:
            try:
                r = get_fn(**({"CallAnalyticsJobName": name} if analytics else {"TranscriptionJobName": name}))
                job = r[status_key]
                status = job["CallAnalyticsJobStatus" if analytics else "TranscriptionJobStatus"]
                print(f"[{name}] status: {status}")
                if status in ("COMPLETED","FAILED"):
                    if status == "COMPLETED" and show_transcript and not analytics:
                        # Print a small snippet of the transcript JSON URL for debugging.
                        print(f"  TranscriptFileUri: {job.get('Transcript',{}).get('TranscriptFileUri')}")
                else:
                    remaining.append(name)
            except Exception as e:
                print(f"[{name}] ERROR: {e}")
        job_names = remaining
        if job_names:
            time.sleep(poll_sec)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV index with at least s3_uri,guest_name columns")
    ap.add_argument("--output-bucket", required=True, help="S3 bucket where transcripts will be written")
    ap.add_argument("--output-prefix", default="", help="S3 prefix/folder for outputs")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION","us-east-2"))
    ap.add_argument("--profile", default="my-transcribe", help="AWS CLI profile name (overrides env vars if set)")
    ap.add_argument("--language-code", default="en-US")
    ap.add_argument("--force-channel", action="store_true", help="Force ChannelIdentification=True (overrides diarization)")
    ap.add_argument("--max-speakers", type=int, default=2, help="Max speaker labels when diarization is used")
    ap.add_argument("--redact-pii", action="store_true", help="Enable PII content redaction (basic settings)")
    ap.add_argument("--wait", action="store_true", help="Poll for completion (simple)")
    ap.add_argument("--poll-seconds", type=int, default=15)
    ap.add_argument("--show-transcript", action="store_true", help="In --wait mode, print TranscriptFileUri for completed jobs")
    ap.add_argument("--call-analytics", action="store_true", help="Use Call Analytics instead of standard Transcribe")
    ap.add_argument("--role-arn", default=None, help="IAM role ARN (REQUIRED for Call Analytics)")

    args = ap.parse_args()

    session = boto3.Session(region_name=args.region, profile_name=args.profile)
    client = session.client("transcribe")

    started = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"s3_uri","guest_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"ERROR: CSV is missing required columns: {missing}", file=sys.stderr)
            sys.exit(2)

        for row in reader:
            if args.call_analytics:
                if not args.role_arn:
                    print("ERROR: --role-arn is required for --call-analytics", file=sys.stderr)
                    sys.exit(2)
                name = start_analytics_job(client, row, args)
            else:
                name = start_standard_job(client, row, args)
            started.append(name)
            print(f"Started job: {name}")

    if args.wait:
        wait_for_jobs(client, started, analytics=args.call_analytics, show_transcript=args.show_transcript, poll_sec=args.poll_seconds)

if __name__ == "__main__":
    main()

    """
export AWS_PROFILE=my-transcribe
/Users/mason/opt/anaconda3/envs/transcriptions/bin/python transcribe_batch.py \
  --profile my-transcribe \
  --csv index.csv \
  --output-bucket pse-audio-files \
  --output-prefix outputs/ \
  --language-code en-US \
  --max-speakers 4 \
  --wait --show-transcript
    """