"""
Live microphone test: speak into your laptop mic (with real background
noise -- a fan, music, typing, road noise, whatever's around you) and hear
the model's enhanced output through your headphones, in real time.

This runs the SAME streaming/causal processing pattern the Raspberry Pi
will eventually run -- so this is a genuine preview of real-time behaviour,
not just an offline file test.

IMPORTANT: wear headphones. If the enhanced output plays out of your
laptop speakers, the mic will pick it up again and you'll get feedback/
howling.

Setup (one-time):
    pip install sounddevice

Usage:
    python live_test.py

Press Ctrl+C to stop the whole session.

Recording (manual, take-based):
    Recording is OFF by default while the session runs -- audio flows
    through live (mic -> model -> headphones) but nothing is saved until
    you say so. While the script is running:
        Press ENTER to START recording a take.
        Press ENTER again to STOP that take and save it.
    You can do this as many times as you like in one session -- each
    start/stop pair is saved as its own pair of WAV files next to this
    script:
        live_input_take<N>_<timestamp>.wav     -- raw mic audio for take N
        live_enhanced_take<N>_<timestamp>.wav  -- enhanced output for take N
    This way you only capture the moments you actually care about (e.g.
    "play the gunshot sound now"), not dead air before/after.
    If you Ctrl+C while a take is still running, it gets saved automatically
    before exit. Set RECORD = False below to disable this entirely.

Tuning:
    MASK_GAMMA below controls how aggressively noise is suppressed WITHOUT
    retraining -- raising it pushes the mask's low (noisy) values further
    toward 0. Try 1.0 first (the model's raw output), then 1.5 or 2.0 if
    you want to hear a stronger effect while more epochs of training run
    in the background.
"""
import sys
import time
import threading
import wave
from datetime import datetime

import numpy as np
import torch

from data import SR, N_FFT, HOP, WIN, CONTEXT
from model import DenoiseNet, N_FREQ

RECORD = True  # set False to disable the manual recording feature entirely

# ---------------------------------------------------------------------------
MASK_GAMMA = 2.0   # chosen after listening to gamma 1.0 vs 2.0 side by side
MASK_FLOOR = 0.08  # mask values below this are forced fully silent -- stops
                    # the AGC makeup-gain step below from re-amplifying
                    # leftover low-level noise back up to audible levels

# Makeup gain: the model only ever REMOVES energy, it never adds any -- so
# suppression naturally makes the output quieter than the input, even when
# it's working correctly. This tracks a short-term running loudness of the
# input vs. the (pre-gain) output and boosts the output back up to roughly
# match, so speech doesn't end up sounding weaker than it started.
AGC_ENABLED = True
AGC_SMOOTHING = 0.98   # closer to 1.0 = slower/smoother gain changes (less pumping)

# Set to 1.0 (no boost at all) so your recorded input/enhanced takes keep an
# honest, clearly audible loudness difference -- this is what you want for
# demos/jury comparisons, since it makes the suppression obvious instead of
# masking it by re-boosting the enhanced clip back up to match input loudness.
# For a "field realistic" feel (own voice doesn't sound artificially weak
# during actual use), raise this back up toward 3.0.
AGC_MAX_GAIN = 1.0     # cap how much boost is allowed, so near-silence doesn't get amplified into hiss
# ---------------------------------------------------------------------------

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is not installed. Run: pip install sounddevice")


def save_wav(path, blocks, sr=SR):
    """blocks: list of float32 (-1..1) arrays -> writes a 16-bit PCM WAV file.
    Returns True if something was actually written."""
    if not blocks:
        return False
    audio = np.concatenate(blocks)
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    return True


def sharpen(mask, gamma=MASK_GAMMA, floor=MASK_FLOOR):
    m = np.clip(mask, 0.0, 1.0) ** gamma
    if floor > 0:
        m = np.where(m < floor, 0.0, m)
    return m


class StreamingDenoiser:
    """Causal, block-by-block denoiser. Feed it HOP samples at a time via
    process(), get HOP samples of enhanced audio back. Internally keeps a
    rolling raw-audio tail (to build full N_FFT analysis frames from
    successive HOP-sized blocks) and a rolling context of recent magnitude
    frames (what the model conditions on), plus an overlap-add accumulator
    for reconstructing the output waveform."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.prev_samples = np.zeros(N_FFT - HOP, dtype=np.float32)
        self.context_buf = np.zeros((CONTEXT, N_FREQ), dtype=np.float32)
        self.out_tail = np.zeros(N_FFT, dtype=np.float32)
        self.wsum_tail = np.zeros(N_FFT, dtype=np.float32)
        self.in_power_ema = 1e-6
        self.out_power_ema = 1e-6

    def process(self, block):
        # block: (HOP,) float32 raw audio
        frame = np.concatenate([self.prev_samples, block])  # (N_FFT,)
        self.prev_samples = frame[HOP:].copy()

        windowed = frame * WIN
        spec = np.fft.rfft(windowed)          # complex, (N_FREQ,)
        mag = np.abs(spec).astype(np.float32)

        self.context_buf = np.concatenate([self.context_buf[1:], mag[None, :]], axis=0)

        with torch.no_grad():
            ctx_t = torch.from_numpy(self.context_buf[None]).to(self.device)  # (1, CONTEXT, N_FREQ)
            mask = self.model(ctx_t).cpu().numpy()[0]

        mask = sharpen(mask)
        masked_spec = spec * mask
        frame_out = (np.fft.irfft(masked_spec, n=N_FFT).astype(np.float32)) * WIN

        self.out_tail[:N_FFT] += frame_out
        self.wsum_tail[:N_FFT] += (WIN ** 2)

        wsum_safe = np.maximum(self.wsum_tail[:HOP], 1e-8)
        out_block = self.out_tail[:HOP] / wsum_safe

        # slide the accumulator forward by HOP
        self.out_tail = np.concatenate([self.out_tail[HOP:], np.zeros(HOP, dtype=np.float32)])
        self.wsum_tail = np.concatenate([self.wsum_tail[HOP:], np.zeros(HOP, dtype=np.float32)])

        if AGC_ENABLED:
            in_power = float(np.mean(block ** 2))
            out_power = float(np.mean(out_block ** 2))
            a = AGC_SMOOTHING
            self.in_power_ema = a * self.in_power_ema + (1 - a) * in_power
            self.out_power_ema = a * self.out_power_ema + (1 - a) * out_power
            gain = np.sqrt(self.in_power_ema / (self.out_power_ema + 1e-8))
            gain = float(np.clip(gain, 1.0, AGC_MAX_GAIN))
            out_block = np.clip(out_block * gain, -1.0, 1.0)

        return out_block


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(here, "denoise_net_best.pt")
    if not os.path.exists(weights_path):
        sys.exit(f"{weights_path} not found -- run train.py first.")

    device = torch.device("cpu")  # keep it simple/predictable for real-time on a laptop CPU
    model = DenoiseNet(n_freq=N_FREQ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    denoiser = StreamingDenoiser(model, device)

    print(f"Loaded {weights_path}")
    print(f"Sample rate: {SR}Hz, block size: {HOP} samples ({HOP/SR*1000:.0f}ms per block)")
    print(f"MASK_GAMMA={MASK_GAMMA}  MASK_FLOOR={MASK_FLOOR}")
    print()
    print("WEAR HEADPHONES to avoid feedback.")
    print("Speak normally, with whatever background noise is around you.")
    print("Press Ctrl+C to stop.")
    if RECORD:
        print()
        print("Recording is manual: press ENTER to start a take, ENTER again to stop")
        print("and save it. Do this as many times as you like during the session.")
    print()

    # --- recording state (shared between the audio callback and the ENTER-key
    # listener thread below) -----------------------------------------------
    recording_lock = threading.Lock()
    state = {"is_recording": False, "take_num": 0, "input": [], "output": []}

    def save_take():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        in_path = os.path.join(here, f"live_input_take{state['take_num']}_{stamp}.wav")
        out_path = os.path.join(here, f"live_enhanced_take{state['take_num']}_{stamp}.wav")
        saved = save_wav(in_path, state["input"])
        save_wav(out_path, state["output"])
        if not saved:
            print(f">>> Take {state['take_num']} was empty -- nothing to save.")
            return
        dur = sum(len(b) for b in state["input"]) / SR
        print(f">>> Saved take {state['take_num']} ({dur:.1f}s):")
        print(f"      {in_path}")
        print(f"      {out_path}")

    def toggle_listener():
        while True:
            try:
                input()
            except EOFError:
                return
            with recording_lock:
                if not state["is_recording"]:
                    state["is_recording"] = True
                    state["input"] = []
                    state["output"] = []
                    print(">>> Recording... press ENTER again to stop and save.")
                    continue
                state["is_recording"] = False
                state["take_num"] += 1
            save_take()

    if RECORD:
        listener = threading.Thread(target=toggle_listener, daemon=True)
        listener.start()

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        block = indata[:, 0].astype(np.float32)
        try:
            out_block = denoiser.process(block)
        except Exception as e:
            print(f"Processing error: {e}", file=sys.stderr)
            out_block = np.zeros(HOP, dtype=np.float32)
        outdata[:, 0] = out_block
        if RECORD and state["is_recording"]:
            state["input"].append(block.copy())
            state["output"].append(out_block.copy())

    try:
        with sd.Stream(samplerate=SR, blocksize=HOP, channels=1, dtype="float32",
                        callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
        if RECORD and state["is_recording"]:
            state["take_num"] += 1
            print(">>> A take was still running -- saving it before exit...")
            save_take()
    except Exception as e:
        print(f"\nCould not open audio stream: {e}")
        print("Run `python -c \"import sounddevice as sd; print(sd.query_devices())\"` "
              "to see available input/output devices, and check your default mic/headphone "
              "devices in Windows sound settings.")


if __name__ == "__main__":
    main()
