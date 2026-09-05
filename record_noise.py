"""
Record a short custom noise clip from your microphone and save it directly
into data/noise/, ready to be used the next time you run train.py.

Usage:
    python record_noise.py <name> [seconds]

Examples:
    python record_noise.py finger_snap 20
    python record_noise.py keyboard_typing 20
    python record_noise.py chair_creak 15

Tips:
    - Record 15-30 seconds, making the sound SEVERAL times with natural
      pauses in between (e.g. snap your fingers ~8-10 times spread across
      the recording), rather than one instant sound. This gives the
      training code real variety to sample from within one file.
    - Do a separate recording (separate <name>) for each distinct sound you
      want the model to learn -- finger snaps and keyboard typing are very
      different sounds and should NOT be in the same file.
    - Record in the same kind of environment/distance-from-mic you expect
      during the actual demo, for the closest match.
"""
import os
import sys
import time

import numpy as np
import soundfile as sf

from data import SR

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is not installed. Run: pip install sounddevice")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python record_noise.py <name> [seconds]\n"
                  "Example: python record_noise.py finger_snap 20")
    name = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "data", "noise")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.wav")

    if os.path.exists(out_path):
        print(f"Note: {out_path} already exists and will be overwritten.")

    print(f"About to record '{name}' for {seconds:.0f} seconds at {SR}Hz.")
    print("Make the sound SEVERAL times, spread out naturally, during the recording.")
    print()
    for i in (3, 2, 1):
        print(i)
        time.sleep(1)
    print("Recording now...")

    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        print("WARNING: recording is nearly silent -- check your microphone / input device "
              "before trusting this file.")

    sf.write(out_path, audio, SR)
    print(f"Saved {out_path}  (peak level: {peak:.3f})")
    print()
    print("Repeat this command with a different <name> for each sound you want to add.")
    print("When you're done recording, re-run train.py to retrain including these files.")


if __name__ == "__main__":
    main()
