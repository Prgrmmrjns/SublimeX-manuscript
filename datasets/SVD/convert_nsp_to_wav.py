#!/usr/bin/env python3
"""
Recursively convert .nsp (IFF) audio files to .wav using ffmpeg,
processing multiple files in parallel and showing a progress bar.

Examples:
  python convert_nsp_to_wav.py .
  python convert_nsp_to_wav.py . --resample 44100 --workers 6
  python convert_nsp_to_wav.py . --dry-run
  python convert_nsp_to_wav.py . --overwrite
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


def iter_nsp_files(root: Path):
    for p in root.rglob("*.nsp"):
        if p.is_file():
            yield p


def convert_one(infile: str, resample: int | None, overwrite: bool, dry_run: bool) -> Tuple[str, str]:
    """
    Worker function. Returns (status, path).
    status in {"ok","skip_exists","dry_run","fail"}
    """
    p = Path(infile)
    out = p.with_suffix(".wav")

    if out.exists() and not overwrite:
        return ("skip_exists", infile)

    if dry_run:
        return ("dry_run", infile)

    # Build ffmpeg command: force WAV with PCM 16-bit; preserve SR unless --resample given
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(p),
        "-c:a",
        "pcm_s16le",
    ]
    if resample:
        cmd += ["-ar", str(resample)]
    cmd.append(str(out))

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode == 0:
            return ("ok", infile)
        tail = (completed.stderr or "").strip().splitlines()[-1:] or [""]
        return ("fail", f"{infile} :: {tail[0]}")
    except Exception as e:  # pragma: no cover
        return ("fail", f"{infile} :: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parallel .nsp ➜ .wav converter (with progress).")
    parser.add_argument("root", type=Path, help="Root directory to scan recursively")
    parser.add_argument("--resample", type=int, default=None, help="Target sample rate (e.g., 44100 or 48000)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV files")
    parser.add_argument("--dry-run", action="store_true", help="List planned actions without converting")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        sys.exit(f"Path not found: {root}")

    files = list(iter_nsp_files(root))
    total = len(files)
    if total == 0:
        print("No .nsp files found.")
        return

    try:
        from tqdm import tqdm
        use_bar = True
    except Exception:
        use_bar = False
        print(f"Converting {total} file(s)... (install tqdm for a progress bar)")

    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers and args.workers > 0 else cpu_count

    converted = 0
    skipped = 0
    failed = 0

    if use_bar:
        pbar = tqdm(total=total, unit="file", desc="Converting", ncols=80)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(convert_one, str(f), args.resample, args.overwrite, args.dry_run)
            for f in files
        ]

        for fut in as_completed(futures):
            status, info = fut.result()
            if status == "ok":
                converted += 1
            elif status in ("skip_exists", "dry_run"):
                skipped += 1
            else:
                failed += 1
                print(f"[fail] {info}", file=sys.stderr)

            if use_bar:
                pbar.update(1)

    if use_bar:
        pbar.close()

    print("\nSummary")
    print("-------")
    print(f"Total    : {total}")
    print(f"Converted: {converted}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {failed}")
    print(f"Workers  : {workers}")


if __name__ == "__main__":
    main()
