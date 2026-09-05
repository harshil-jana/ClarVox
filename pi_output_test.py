"""
Minimal live-output test -- NOT the denoiser, just a raw tone played through
sounddevice to isolate whether sounddevice/PortAudio can drive the Pi's
3.5mm output at all during live playback, separate from the WAV-recording
path (which we already know works).

Run this on the Pi:
    source ~/anc-venv/bin/activate
    python pi_output_test.py

You should hear a clear 440Hz tone for 3 seconds through the AUX earphones
plugged into the 3.5mm jack. Report back whether you heard it.
"""
import sys
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is not installed. Run: pip install sounddevice")


def pick_output():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if "bcm2835" in d["name"].lower() and d["max_output_channels"] > 0:
            return i, devices
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            return i, devices
    return None, devices


def main():
    output_idx, devices = pick_output()
    if output_idx is None:
        sys.exit("No output device found.")

    sr = int(round(devices[output_idx]["default_samplerate"]))
    print(f"Output device: [{output_idx}] {devices[output_idx]['name']}")
    print(f"Sample rate: {sr}Hz")

    # --- Test 1: simple blocking playback (sd.play), no callback, no stream ---
    print("\n[Test 1] Playing 440Hz tone for 3s via sd.play (blocking)...")
    duration = 3.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sd.play(tone, samplerate=sr, device=output_idx)
    sd.wait()
    print("[Test 1] Done. Did you hear it? (this uses the same device index "
          "our real script picks)")

    # --- Test 2: same tone through a callback-based OutputStream, matching
    # how the real script drives audio (continuous callback, small blocks) ---
    print("\n[Test 2] Playing the same tone for 3s via an OutputStream "
          "callback (matches the real script's approach)...")
    blocksize = 768  # same HW_HOP the real script computes at 48kHz/16kHz
    phase = {"i": 0}

    def callback(outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        idx = phase["i"]
        block_t = (np.arange(frames) + idx) / sr
        block = (0.3 * np.sin(2 * np.pi * 440 * block_t)).astype(np.float32)
        outdata[:, 0] = block
        phase["i"] += frames

    with sd.OutputStream(samplerate=sr, blocksize=blocksize, channels=1,
                          dtype="float32", device=output_idx, callback=callback):
        sd.sleep(int(duration * 1000))
    print("[Test 2] Done. Did you hear it too?")


if __name__ == "__main__":
    main()
