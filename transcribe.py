"""
Batch transcribe MP4/MKV files using faster-whisper.

Scans the given folder (and all subfolders) for video files and writes one
TXT file per video. Each TXT is named after the original video and all of
them are placed together in a single output folder.

Usage:
    python transcribe.py /path/to/your/videos
    python transcribe.py /path/to/your/videos --output-dir /path/to/output
    python transcribe.py /path/to/your/videos --model large-v3
    python transcribe.py /path/to/your/videos --model small --language auto
"""

import sys
import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel


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
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)",
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
    print(f"Loading model '{args.model}' (this may take a minute the first time)...\n")

    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    succeeded = 0
    failed = []
    used_names = set()

    for video in video_files:
        relative_path = video.relative_to(folder)
        try:
            text = transcribe_video(model, video, args.language)
        except Exception as e:
            print(f"  ERROR on {relative_path}: {e}")
            failed.append(relative_path)
            continue

        # Name the .txt after the original video. If two videos in different
        # subfolders share a name, add a numeric suffix so nothing is overwritten.
        out_name = video.stem + ".txt"
        if out_name.lower() in used_names:
            n = 1
            while f"{video.stem}_{n}.txt".lower() in used_names:
                n += 1
            out_name = f"{video.stem}_{n}.txt"
        used_names.add(out_name.lower())

        out_path = output_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"  Saved: {out_path}")
        succeeded += 1

    print(f"\n--- Done! {succeeded}/{len(video_files)} files transcribed ---")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
