"""
Batch transcribe MP4/MKV files using faster-whisper.

Scans the given folder (and all subfolders) for video files and writes one
TXT file per video into a single output folder. Each transcript is named
after the original video (e.g. "my clip.mp4" -> "my clip.txt") and is written
for a downstream pipeline, containing:

  * a parseable YAML front-matter header (source_date, video_type, title,
    duration, instruments, content hash, ...), and
  * one line per segment, each prefixed with an [HH:MM:SS] start timestamp.

The chronological/type metadata the pipeline needs lives in the header, so
the filename can stay a simple, recognizable match to the source video.

A tracking file records which videos have already been transcribed, so you
can drop new videos into the folder over time and re-run to transcribe only
the new ones. Existing output files are also detected, so the batch is
resumable even if the tracking file is lost. Use --force to redo everything.

Usage:
    python transcribe.py /path/to/your/videos
    python transcribe.py /path/to/videos --video-type daily_plan
    python transcribe.py /path/to/videos --video-type live_stream
    python transcribe.py /path/to/videos --source-date 2024-03-15   # single file
    python transcribe.py /path/to/your/videos --output-dir /path/to/output
    python transcribe.py /path/to/your/videos --force
    python transcribe.py /path/to/your/videos --model small --language auto
"""

import re
import sys
import os
import json
import hashlib
import argparse
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, date as _date
from pathlib import Path
from faster_whisper import WhisperModel


def load_tracking(path):
    """Load the record of already-transcribed videos.

    Returns a dict keyed by each video's path relative to the input folder.
    Returns an empty dict if the file is missing or unreadable.
    """
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        processed = data.get("processed", {})
        if isinstance(processed, dict):
            return processed
    except (json.JSONDecodeError, OSError, AttributeError) as e:
        print(f"Warning: could not read tracking file '{path}' ({e}); starting fresh.")
    return {}


def save_tracking(path, processed):
    """Write the tracking record. Writes to a temp file first, then replaces,
    so an interrupted run can't corrupt the existing tracking file."""
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": processed}, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def file_signature(video):
    """Return (size, mtime) used to detect whether a file changed since last run."""
    st = video.stat()
    return st.st_size, st.st_mtime


def is_unchanged(record, size, mtime):
    """True if a tracked file looks identical to when it was last transcribed."""
    return record.get("size") == size and abs(record.get("mtime", 0) - mtime) < 1.0


def unique_output_name(base_name, output_dir, used_names):
    """Pick a '<base>.txt' name that doesn't collide with an already-used name
    or an existing file on disk, adding a numeric suffix if needed.

    Accepts either a bare stem or a full '<name>.txt' base name.
    """
    stem = base_name[:-4] if base_name.lower().endswith(".txt") else base_name
    candidate = stem + ".txt"
    n = 1
    while candidate.lower() in used_names or (output_dir / candidate).exists():
        candidate = f"{stem}_{n}.txt"
        n += 1
    return candidate


def seconds_to_hms(seconds):
    """Format a number of seconds as a zero-padded HH:MM:SS string."""
    total = int(seconds or 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _valid_ymd(y, m, d):
    try:
        _date(int(y), int(m), int(d))
        return True
    except ValueError:
        return False


def extract_date_from_name(name):
    """Best-effort extraction of an original recording date (YYYY-MM-DD) from a
    filename. Tries ISO-like, compact, then US ordering. Returns None if none."""
    # ISO-like: 2024-03-15, 2024_03_15, 2024.03.15
    m = re.search(r"(?<!\d)(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})(?!\d)", name)
    if m and _valid_ymd(*m.groups()):
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    # Compact: 20240315
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", name)
    if m and _valid_ymd(*m.groups()):
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    # US: 03-15-2024, 03_15_2024
    m = re.search(r"(?<!\d)(\d{1,2})[-_.](\d{1,2})[-_.](20\d{2})(?!\d)", name)
    if m:
        mo, d, y = m.groups()
        if _valid_ymd(y, mo, d):
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return None


def infer_video_type(name):
    """Infer 'daily_plan' or 'live_stream' from a filename, or None."""
    low = name.lower()
    if any(k in low for k in ("live", "stream")):
        return "live_stream"
    if any(k in low for k in ("daily", "plan", "premarket", "pre-market")):
        return "daily_plan"
    return None


def detect_instruments(filename, text):
    """Detect traded instruments (ES / NQ) from the filename or transcript.

    Conservative: only matches uppercase tickers as whole words, plus common
    synonyms. Returns 'ES', 'NQ', 'ES, NQ', or '' when undeterminable.
    """
    found = []
    if (re.search(r"\bES\b", filename) or re.search(r"\bES\b", text)
            or re.search(r"\bE-?mini\b", text, re.I)
            or re.search(r"S\s*&\s*P|S and P", text, re.I)):
        found.append("ES")
    if (re.search(r"\bNQ\b", filename) or re.search(r"\bNQ\b", text)
            or re.search(r"nasdaq", text, re.I)):
        found.append("NQ")
    return ", ".join(found)


def content_hash(text):
    """SHA-256 of the given text (prefixed 'sha256:'), for idempotent ingestion."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def yaml_quote(value):
    """Double-quote and escape a value so the header is unambiguous YAML; every
    field is emitted as a string for predictable parsing across the pipeline."""
    s = "" if value is None else str(value)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# Header field order is fixed so downstream parsing is stable.
HEADER_FIELDS = [
    "source_date", "video_type", "title", "original_filename", "duration",
    "instruments", "content_sha256", "author", "model", "language",
    "transcribed_at",
]


def build_header(meta):
    """Render the metadata dict as a YAML front-matter block."""
    lines = ["---"]
    for key in HEADER_FIELDS:
        lines.append(f"{key}: {yaml_quote(meta.get(key, ''))}")
    lines.append("---")
    return "\n".join(lines)


def format_segments(segments):
    """One line per segment, each prefixed with its [HH:MM:SS] start time.

    This is the single place segments become text, so the timestamps cannot be
    dropped by any later formatting pass.
    """
    return "\n".join(f"[{seconds_to_hms(start)}] {text}" for start, text in segments)


def strip_timestamps(body):
    """Return just the spoken words from a transcript body: drop the leading
    [HH:MM:SS] markers and collapse all whitespace to single spaces.

    content_sha256 is hashed over THIS, not the timestamped text, so a
    re-transcription with slightly shifted timestamps or different segment
    boundaries still yields the same hash (identical speech -> identical hash),
    and the downstream pipeline won't mistake it for a brand-new transcript.
    Downstream can reproduce the hash from a saved file by applying these same
    two steps to the body (the part after the YAML header).
    """
    no_marks = re.sub(r"(?m)^\[\d{2,}:\d{2}:\d{2}\]\s*", "", body)
    return re.sub(r"\s+", " ", no_marks).strip()


def check_ffmpeg():
    """Make sure ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except FileNotFoundError:
        print("Error: ffmpeg not found on PATH.")
        print("Download from https://www.gyan.dev/ffmpeg/builds/")
        print("and add the bin folder to your system PATH.")
        sys.exit(1)


def extract_audio(input_file):
    """Extract audio to a temporary 16kHz mono WAV (what Whisper expects)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    subprocess.run(
        [
            "ffmpeg", "-i", str(input_file),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            "-y", "-loglevel", "error", tmp_path,
        ],
        check=True,
    )
    return tmp_path


def transcribe_video(model, input_file, language):
    """Transcribe a single file.

    Returns (segments, info) where segments is a list of (start_seconds, text)
    tuples. Per-segment start times are kept so the writer can emit [HH:MM:SS]
    markers; the engine/quality settings below are unchanged.
    """
    print(f"\nTranscribing: {input_file}")

    tmp_path = extract_audio(input_file)

    try:
        transcribe_opts = {
            "beam_size": 5,
            "vad_filter": True,          # skip silence, speeds things up
            "vad_parameters": {
                "min_silence_duration_ms": 500,
            },
        }
        if language != "auto":
            transcribe_opts["language"] = language

        segments, info = model.transcribe(tmp_path, **transcribe_opts)
        print(
            f"  Language: {info.language} "
            f"(probability {info.language_probability:.2f})"
        )

        results = []
        for i, segment in enumerate(segments, 1):
            text = segment.text.strip()
            if text:
                results.append((segment.start, text))
            if i % 50 == 0:
                print(f"  ...{i} segments done")

        print(f"  Finished! {len(results)} segment(s).")
        return results, info
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe MP4/MKV files using faster-whisper"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing video files, scanned recursively "
             "(default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder to write the .txt transcripts into "
             "(default: a 'transcriptions' folder inside the input folder)",
    )
    parser.add_argument(
        "--tracking-file",
        default=None,
        help="JSON file recording which videos are already transcribed "
             "(default: '.transcribed.json' inside the output folder)",
    )
    parser.add_argument(
        "--video-type",
        choices=["daily_plan", "live_stream"],
        default=None,
        help="Video type for all files this run. If omitted, it is inferred "
             "from each filename (falling back to daily_plan).",
    )
    parser.add_argument(
        "--source-date",
        default=None,
        help="Original recording date YYYY-MM-DD applied to all files this "
             "run (intended for single-file runs). If omitted, the date is "
             "taken from each filename, falling back to the file's mod/time.",
    )
    parser.add_argument(
        "--author",
        default="severin",
        help="Author/source tag recorded in each file's header (default: severin)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe every video, even ones already in the tracking file",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3, highest accuracy)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code, or 'auto' for auto-detection (default: en)",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: folder '{folder}' not found")
        sys.exit(1)

    # All transcripts go into a single output folder.
    output_dir = Path(args.output_dir) if args.output_dir else folder / "transcriptions"
    output_dir.mkdir(parents=True, exist_ok=True)

    check_ffmpeg()

    # Find all MP4 and MKV files, including those in subfolders.
    video_files = sorted(folder.rglob("*.mp4")) + sorted(folder.rglob("*.mkv"))
    if not video_files:
        print(f"No .mp4 or .mkv files found in {folder}")
        sys.exit(1)

    print(f"Found {len(video_files)} video file(s)")

    # Validate an explicit --source-date up front (it applies to all files).
    if args.source_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.source_date):
        print(f"Error: --source-date must be YYYY-MM-DD, got '{args.source_date}'")
        sys.exit(1)

    # Load tracking so we transcribe only videos we haven't done before.
    tracking_file = (
        Path(args.tracking_file) if args.tracking_file
        else output_dir / ".transcribed.json"
    )
    processed = load_tracking(tracking_file)

    # Reserve output names already claimed by previously transcribed videos, so
    # a newly added video can never overwrite an existing transcript.
    used_names = {
        rec["output"].lower()
        for rec in processed.values()
        if isinstance(rec, dict) and rec.get("output")
    }

    # Count duplicate stems so we know when an existing "<stem>.txt" on disk
    # unambiguously belongs to this exact video (used for crash recovery below).
    stem_counts = Counter(v.stem.lower() for v in video_files)

    model = None  # loaded lazily, only if there is actually new work to do
    transcribed = 0
    skipped = 0
    failed = []

    for video in video_files:
        relative_path = video.relative_to(folder)
        key = relative_path.as_posix()
        size, mtime = file_signature(video)
        record = processed.get(key)

        # Resumable + idempotent. The primary signal is the tracking file, keyed
        # by source path, so it's never confused by two videos sharing a name.
        # As a fallback when the tracking file is lost, treat an existing
        # "<original name>.txt" as done -- but only when that stem is unique, so
        # we can't wrongly skip a different video that happens to share its name.
        if not args.force:
            if record and is_unchanged(record, size, mtime):
                skipped += 1
                continue
            if (stem_counts[video.stem.lower()] == 1
                    and (output_dir / (video.stem + ".txt")).exists()):
                print(f"  Skipping {relative_path}: '{video.stem}.txt' already exists.")
                skipped += 1
                continue

        if model is None:
            print(f"Loading model '{args.model}' "
                  "(this may take a minute the first time)...")
            model = WhisperModel(args.model, device="cpu", compute_type="int8")

        try:
            segments, info = transcribe_video(model, video, args.language)
        except Exception as e:
            print(f"  ERROR on {relative_path}: {e}")
            failed.append(relative_path)
            continue

        # Resolve the header metadata. source_date is the ORIGINAL recording
        # date -- from the filename or --source-date, falling back to the file's
        # modified time -- and is never the transcription date.
        source_date = args.source_date or extract_date_from_name(video.name)
        if not source_date:
            source_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            print(f"  WARNING: no date found in '{video.name}'; using file date "
                  f"{source_date}. Pass --source-date or put a date in the "
                  f"filename for correct chronological ordering downstream.")
        video_type = args.video_type or infer_video_type(video.name) or "daily_plan"

        # Build the timestamped body first, then wrap the metadata header around
        # it. Timestamps live in the body and are never stripped afterwards.
        body = format_segments(segments)
        duration = info.duration if getattr(info, "duration", None) else (
            segments[-1][0] if segments else 0
        )
        meta = {
            "source_date": source_date,
            "video_type": video_type,
            "title": video.stem,
            "original_filename": video.name,
            "duration": seconds_to_hms(duration),
            "instruments": detect_instruments(video.name, body),
            "content_sha256": content_hash(strip_timestamps(body)),
            "author": args.author,
            "model": args.model,
            "language": getattr(info, "language", "") or "",
            "transcribed_at": datetime.now().isoformat(timespec="seconds"),
        }
        content = build_header(meta) + "\n\n" + body + "\n"

        # Name the .txt after the original video. A known file (in the tracking
        # record) is overwritten in place. With no record but a uniquely-named
        # transcript already on disk, overwrite that too -- it's this video's
        # (this is the --force path; without --force it was skipped above). Only
        # a genuine same-name clash from another subfolder gets a numeric
        # suffix, so nothing is ever silently overwritten.
        if record and record.get("output"):
            out_name = record["output"]
        elif (stem_counts[video.stem.lower()] == 1
                and (output_dir / (video.stem + ".txt")).exists()):
            out_name = video.stem + ".txt"
            used_names.add(out_name.lower())
        else:
            out_name = unique_output_name(video.stem, output_dir, used_names)
            used_names.add(out_name.lower())

        out_path = output_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Saved: {out_path}")

        # Record progress immediately so an interrupted run resumes cleanly.
        processed[key] = {
            "output": out_name,
            "size": size,
            "mtime": mtime,
            "source_date": source_date,
            "video_type": video_type,
            "content_sha256": meta["content_sha256"],
            "transcribed_at": meta["transcribed_at"],
        }
        save_tracking(tracking_file, processed)
        transcribed += 1

    print(f"\n--- Done! {transcribed} transcribed, {skipped} skipped "
          f"(already done), {len(failed)} failed ---")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    print(f"Output folder: {output_dir}")
    print(f"Tracking file: {tracking_file}")


if __name__ == "__main__":
    main()
