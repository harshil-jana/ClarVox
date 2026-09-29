"""
Resamples audio files to 16kHz mono WITHOUT needing ffmpeg -- uses only
soundfile + scipy, which are already in requirements.txt. Nothing extra to
install.

Usage:
    Fix everything in data/speech and data/noise that isn't 16kHz mono:
        python fix_sample_rates.py

    Or convert one specific file:
        python fix_sample_rates.py path\\to\\input.wav path\\to\\output.wav

Run with no arguments, it scans data/speech and data/noise and overwrites
any file that isn't already 16kHz mono with a resampled version, in place
(same filename). Anything it touches is backed up first into a sibling
`_originals` folder, so nothing is lost if something looks wrong.
"""
import glob
import math
import os
import shutil
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".flac")


def resample_to_target(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr == TARGET_SR:
        return audio.astype(np.float32)
    g = math.gcd(TARGET_SR, sr)
    up, down = TARGET_SR // g, sr // g
    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def convert_file(path, out_path=None, backup_dir=None):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 1 and sr == TARGET_SR:
        return False  # already fine, nothing to do
    fixed = resample_to_target(audio, sr)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    sf.write(out_path or path, fixed, TARGET_SR)
    return True


def scan_and_fix(folder):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    backup_dir = os.path.join(folder, "_originals")
    n_fixed = 0
    for f in sorted(files):
        try:
            changed = convert_file(f, backup_dir=backup_dir)
            if changed:
                print(f"  fixed: {os.path.basename(f)}")
                n_fixed += 1
        except Exception as e:
            print(f"  ERROR on {os.path.basename(f)}: {e}")
    print(f"{folder}: {n_fixed} file(s) resampled to {TARGET_SR}Hz mono "
          f"(originals backed up in _originals/)")


def main():
    args = sys.argv[1:]
    if len(args) == 2:
        convert_file(args[0], out_path=args[1])
        print(f"Wrote {args[1]} at {TARGET_SR}Hz mono")
        return
    if len(args) != 0:
        sys.exit("Usage: python fix_sample_rates.py   OR   "
                  "python fix_sample_rates.py in.wav out.wav")

    here = os.path.dirname(os.path.abspath(__file__))
    for sub in ("speech", "noise"):
        folder = os.path.join(here, "data", sub)
        if os.path.isdir(folder):
            print(f"Scanning {folder} ...")
            scan_and_fix(folder)
        else:
            print(f"(skipping {folder} -- doesn't exist)")
    print("\nDone. Run `python check_data.py` again to confirm everything's 16kHz now.")


if __name__ == "__main__":
    main()
