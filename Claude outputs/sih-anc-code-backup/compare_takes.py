"""
Quick numeric comparison of an input/enhanced take pair, to tell for certain
whether the denoiser is actually changing anything (vs. by-ear judgment,
which can be misleading for short/quiet noise bursts).

Usage:
    python compare_takes.py pi_input_take3_...wav pi_enhanced_take3_...wav

Run this on the Pi (it only needs numpy, already installed there), or copy
both wav files to the laptop and run it there with plain Python + numpy.
"""
import sys
import wave
import numpy as np


def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python compare_takes.py <input.wav> <enhanced.wav>")

    in_path, out_path = sys.argv[1], sys.argv[2]
    in_audio, in_sr = load_wav(in_path)
    out_audio, out_sr = load_wav(out_path)

    print(f"Input:    {in_path}  ({len(in_audio)/in_sr:.2f}s @ {in_sr}Hz, {len(in_audio)} samples)")
    print(f"Enhanced: {out_path}  ({len(out_audio)/out_sr:.2f}s @ {out_sr}Hz, {len(out_audio)} samples)")
    print()

    n = min(len(in_audio), len(out_audio))
    a, b = in_audio[:n], out_audio[:n]

    identical = np.array_equal(a, b)
    max_abs_diff = float(np.max(np.abs(a - b)))
    rms_diff = float(np.sqrt(np.mean((a - b) ** 2)))
    rms_in = float(np.sqrt(np.mean(a ** 2)))
    rms_out = float(np.sqrt(np.mean(b ** 2)))
    peak_in = float(np.max(np.abs(a)))
    peak_out = float(np.max(np.abs(b)))

    print(f"Bit-for-bit identical: {identical}")
    print(f"Max sample difference: {max_abs_diff:.6f}")
    print(f"RMS of the difference: {rms_diff:.6f}")
    print()
    print(f"Input  RMS: {rms_in:.4f}   peak: {peak_in:.4f}")
    print(f"Output RMS: {rms_out:.4f}   peak: {peak_out:.4f}")
    if rms_in > 1e-9:
        print(f"Output/Input RMS ratio: {rms_out/rms_in:.3f} "
              f"(1.0 = same loudness, <1.0 = enhanced is quieter overall)")

    print()
    if identical:
        print(">>> These files are EXACTLY the same. That IS a real bug -- "
              "the denoiser isn't being applied to what's saved, or the wrong "
              "array is being recorded. Worth digging into the code, not the mic setup.")
    elif max_abs_diff < 0.01:
        print(">>> Files differ, but only very slightly (near-silent difference). "
              "The pipeline IS doing something, but the effect is small enough that "
              "it may not be audible for this particular noise/recording -- "
              "likely a tuning issue (MASK_GAMMA/MASK_FLOOR) or the noise wasn't "
              "loud/distinct enough in this take, not a bug in the save logic.")
    else:
        print(">>> Files clearly differ numerically. If they still sound the same "
              "to your ear, the difference may be too subtle to notice for this "
              "noise type -- worth trying a louder/more distinct noise source, or "
              "increasing MASK_GAMMA / lowering MASK_FLOOR won't help since floor "
              "already zeroes weak masks; try boosting suppression strength instead.")


if __name__ == "__main__":
    main()
