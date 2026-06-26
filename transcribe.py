"""
Batch transcribe MP4/MKV files using faster-whisper.

Scans the given folder (and all subfolders) for video files and writes one
TXT file per video. Each TXT is named after the original video and all of
them are placed together in a single output folder.

A small tracking file records which videos have already been transcribed, so
you can drop new videos into the folder over time and re-run to transcribe
only the new ones. Use --force to re-transcribe everything.

Usage:
    python transcribe.py /path/to/your/videos
    python transcribe.py /path/to/your/videos --output-dir /path/to/output
    python transcribe.py /path/to/your/videos --force
    python transcribe.py /path/to/your/videos --model large-v3
    python transcribe.py /path/to/your/videos --model small --language auto
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
from datetime import datetime
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


def unique_output_name(stem, output_dir, used_names):
    """Pick a '<stem>.txt' name that doesn't collide with an already-used name
    or an existing file on disk, adding a numeric suffix if needed."""
    candidate = stem + ".txt"
    n = 1
    while candidate.lower() in used_names or (output_dir / candidate).exists():
        candidate = f"{stem}_{n}.txt"
        n += 1
    return candidate


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
    """Transcribe a single file and return the text."""
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

        i = 0
        text_parts = []
        for i, segment in enumerate(segments, 1):
            text_parts.append(segment.text.strip())
            if i % 50 == 0:
                print(f"  ...{i} segments done")

        print(f"  Finished! {i} segment(s) written.")
        return " ".join(text_parts)
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

    model = None  # loaded lazily, only if there is actually new work to do
    transcribed = 0
    skipped = 0
    failed = []

    for video in video_files:
        relative_path = video.relative_to(folder)
        key = relative_path.as_posix()
        size, mtime = file_signature(video)
        record = processed.get(key)

        # Skip videos already transcribed that haven't changed since.
        if record and not args.force and is_unchanged(record, size, mtime):
            skipped += 1
            continue

        if model is None:
            print(f"Loading model '{args.model}' "
                  "(this may take a minute the first time)...")
            model = WhisperModel(args.model, device="cpu", compute_type="int8")

        try:
            text = transcribe_video(model, video, args.language)
        except Exception as e:
            print(f"  ERROR on {relative_path}: {e}")
            failed.append(relative_path)
            continue

        # Re-transcribing a known file reuses its name (overwrites in place);
        # a new file gets a fresh, collision-free '<original name>.txt'.
        if record and record.get("output"):
            out_name = record["output"]
        else:
            out_name = unique_output_name(video.stem, output_dir, used_names)
            used_names.add(out_name.lower())

        out_path = output_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"  Saved: {out_path}")

        # Record progress immediately so an interrupted run resumes cleanly.
        processed[key] = {
            "output": out_name,
            "size": size,
            "mtime": mtime,
            "transcribed_at": datetime.now().isoformat(timespec="seconds"),
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
