"""
Offline sanity check: takes one speech clip + one noise clip, mixes them,
runs the trained model over the whole mixture, reconstructs a cleaned-up
wav with overlap-add, and writes both files to disk so you can listen to
before/after.

Run this AFTER train.py (needs denoise_net_best.pt to exist).

Usage:
    python enhance_test.py path/to/speech.wav path/to/noise.wav [snr_db]

If you don't pass paths, it picks the first file it finds in data/speech
and data/noise, mixed at 5 dB SNR.
"""
import glob
import os
import sys

import numpy as np
import torch
import soundfile as sf

from data import SR, N_FFT, HOP, WIN, CONTEXT, load_mono, mix_at_snr
from model import DenoiseNet, N_FREQ

# ---------------------------------------------------------------------------
# Mask sharpening -- lets you hear the effect of more aggressive suppression
# WITHOUT retraining. 1.0 = the model's raw output. Try 1.5-2.5 if the
# enhanced audio still sounds too close to the noisy input.
# ---------------------------------------------------------------------------
MASK_GAMMA = 2.0
MASK_FLOOR = 0.08  # mask values below this are forced fully silent -- stops
                    # the AGC makeup-gain step below from re-amplifying
                    # leftover low-level noise back up to audible levels

# Makeup gain: suppression only ever removes energy, so the output ends up
# quieter than the input even when it's working correctly. This rescales
# the whole enhanced clip to roughly match the input mix's loudness.
#
# Set to 1.0 (no boost at all) so noisy-vs-enhanced keeps an honest, clearly
# audible loudness difference -- this is what you want for demos/jury
# comparisons, since it makes the suppression obvious. For a "field realistic"
# version (matching what live_test.py / pi_live_test.py actually do so a
# user's own voice doesn't sound artificially weak), raise this back up
# toward 3.0.
AGC_MAX_GAIN = 1.0


def sharpen_mask(mask, gamma=MASK_GAMMA, floor=MASK_FLOOR):
    m = np.clip(mask, 0.0, 1.0) ** gamma
    if floor > 0:
        m = np.where(m < floor, 0.0, m)
    return m


def stft_frames(audio):
    """Same framing as data.spectrogram, but keeps the complex STFT (not
    just magnitude) so we can reconstruct a waveform afterward."""
    pad = np.zeros(N_FFT, dtype=np.float32)
    padded = np.concatenate([pad, audio]).astype(np.float32)
    n = (len(padded) - N_FFT) // HOP
    spec = np.zeros((n, N_FFT // 2 + 1), dtype=np.complex64)
    for i in range(n):
        frame = padded[i * HOP:i * HOP + N_FFT] * WIN
        spec[i] = np.fft.rfft(frame)
    return spec, len(padded)


def istft_overlap_add(spec, padded_len):
    """Inverse of stft_frames via Hann-Hann overlap-add with COLA
    normalization (window-sum accumulation) -- this is the same math
    validated earlier against a pure passthrough (all-ones mask
    reconstructs the original signal to numerical precision)."""
    n = spec.shape[0]
    out = np.zeros(padded_len, dtype=np.float32)
    wsum = np.zeros(padded_len, dtype=np.float32)
    for i in range(n):
        frame = np.fft.irfft(spec[i], n=N_FFT).astype(np.float32) * WIN
        start = i * HOP
        out[start:start + N_FFT] += frame
        wsum[start:start + N_FFT] += WIN ** 2
    wsum[wsum < 1e-8] = 1e-8
    out = out / wsum
    return out[N_FFT:]  # undo the leading zero-pad stft_frames added


def enhance(model, mix, device):
    mix_spec, padded_len = stft_frames(mix)
    mix_mag = np.abs(mix_spec)
    n_frames, n_freq = mix_mag.shape

    # Pad the front with a repeated first frame so every real frame gets a
    # full CONTEXT-length window -- mirrors the padding used for very short
    # clips in data.make_training_example.
    pad_needed = CONTEXT - 1
    mix_mag_padded = np.concatenate([np.tile(mix_mag[:1], (pad_needed, 1)), mix_mag])

    masks = np.zeros((n_frames, n_freq), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for t in range(n_frames):
            context = mix_mag_padded[t: t + CONTEXT]  # (CONTEXT, n_freq)
            context_t = torch.from_numpy(context[None]).to(device)  # (1, CONTEXT, n_freq)
            mask = model(context_t).cpu().numpy()[0]
            masks[t] = mask

    masks = sharpen_mask(masks)

    # Apply the (real-valued) mask to the complex mixture spectrum -- this
    # scales magnitude and keeps the noisy signal's phase, same approach
    # RNNoise/DeepFilterNet-style mask models use.
    enhanced_spec = mix_spec * masks
    return istft_overlap_add(enhanced_spec, padded_len)


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        speech_path, noise_path = args[0], args[1]
        snr_db = float(args[2]) if len(args) > 2 else 5.0
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        speech_candidates = sorted(
            glob.glob(os.path.join(here, "data", "speech", "*.wav"))
            + glob.glob(os.path.join(here, "data", "speech", "*.flac"))
        )
        noise_candidates = sorted(
            glob.glob(os.path.join(here, "data", "noise", "*.wav"))
            + glob.glob(os.path.join(here, "data", "noise", "*.flac"))
        )
        if not speech_candidates or not noise_candidates:
            sys.exit("No test files given and none found in data/speech or data/noise.\n"
                      "Usage: python enhance_test.py speech.wav noise.wav [snr_db]")
        speech_path, noise_path = speech_candidates[0], noise_candidates[0]
        snr_db = 5.0
        print(f"No files given -- using {speech_path} + {noise_path} at {snr_db} dB SNR")

    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(here, "denoise_net_best.pt")
    if not os.path.exists(weights_path):
        sys.exit(f"{weights_path} not found -- run train.py first.")

    speech = load_mono(speech_path)
    noise = load_mono(noise_path)
    mix, _ = mix_at_snr(speech, noise, snr_db)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoiseNet(n_freq=N_FREQ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    enhanced = enhance(model, mix, device)

    # Trim/pad to matching length (OLA can be a few samples longer/shorter
    # than the input depending on framing).
    n = min(len(mix), len(enhanced))
    mix_trim = mix[:n]
    enhanced_trim = enhanced[:n]

    # Makeup gain -- boost the enhanced clip back toward the input's overall
    # loudness (suppression alone only ever makes it quieter).
    mix_rms = float(np.sqrt(np.mean(mix_trim.astype(np.float64) ** 2) + 1e-12))
    enh_rms = float(np.sqrt(np.mean(enhanced_trim.astype(np.float64) ** 2) + 1e-12))
    agc_gain = min(mix_rms / (enh_rms + 1e-8), AGC_MAX_GAIN)
    agc_gain = max(agc_gain, 1.0)
    enhanced_trim = np.clip(enhanced_trim * agc_gain, -1.0, 1.0).astype(np.float32)

    out_mix = os.path.join(here, "test_noisy.wav")
    out_enh = os.path.join(here, "test_enhanced.wav")
    sf.write(out_mix, mix_trim, SR)
    sf.write(out_enh, enhanced_trim, SR)

    def rms_db(x):
        return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)

    print(f"MASK_GAMMA={MASK_GAMMA}  MASK_FLOOR={MASK_FLOOR}  AGC_gain_applied={agc_gain:.2f}x")
    print(f"Wrote {out_mix}")
    print(f"Wrote {out_enh}")
    print(f"Noisy level:    {rms_db(mix_trim):.1f} dB (RMS)")
    print(f"Enhanced level: {rms_db(enhanced_trim):.1f} dB (RMS)")
    print("Listen to both files -- the enhanced one should sound noticeably cleaner.")
    print("If it sounds distorted, muffled, or silent, the model likely needs more")
    print("training epochs/data before it's ready -- that's useful signal, not failure.")


if __name__ == "__main__":
    main()
