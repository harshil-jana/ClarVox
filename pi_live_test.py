"""
Raspberry Pi real-time denoiser -- ONNX Runtime version of live_test.py.

Runs the exact same causal, block-by-block processing already validated on
the laptop (live_test.py), but using onnxruntime instead of PyTorch, so the
Pi doesn't need the full PyTorch install -- just onnxruntime + numpy +
sounddevice.

Setup (one-time, on the Pi):
    source ~/anc-venv/bin/activate
    pip install onnxruntime numpy sounddevice

Usage:
    python pi_live_test.py

Copy denoise_net.onnx into this same folder before running (produced on the
laptop by export_onnx.py).

Press Ctrl+C to stop.

Recording (manual, take-based -- same as live_test.py on the laptop):
    Recording is OFF by default while the stream runs. While the script is
    running:
        Press ENTER to START recording a take.
        Press ENTER again to STOP that take and save it.
    Each start/stop pair is saved next to this script as:
        pi_input_take<N>_<timestamp>.wav
        pi_enhanced_take<N>_<timestamp>.wav
    Set RECORD = False below to disable this entirely.
"""
import os
import sys
import time
import threading
import wave
from datetime import datetime

import numpy as np
import onnxruntime as ort

SR = 16000
N_FFT = 512
HOP = 256
CONTEXT = 8
N_FREQ = N_FFT // 2 + 1  # 257
WIN = np.hanning(N_FFT).astype(np.float32)

RECORD = True  # set False to disable the manual recording feature entirely

# ---------------------------------------------------------------------------
# Same tuning as live_test.py / enhance_test.py on the laptop -- keep these
# in sync if they change there.
MASK_GAMMA = 2.0
MASK_FLOOR = 0.08

AGC_ENABLED = True
AGC_SMOOTHING = 0.98

# Set to 1.0 (no boost at all) so input-vs-enhanced keeps an honest, clearly
# audible loudness difference for demos/jury comparisons -- matches the same
# fix applied to live_test.py on the laptop. For "field realistic" use (own
# voice doesn't sound artificially weak), raise this back up toward 3.0.
AGC_MAX_GAIN = 1.0
# ---------------------------------------------------------------------------

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is not installed. Run: pip install sounddevice")


def sharpen(mask, gamma=MASK_GAMMA, floor=MASK_FLOOR):
    m = np.clip(mask, 0.0, 1.0) ** gamma
    if floor > 0:
        m = np.where(m < floor, 0.0, m)
    return m


def resample_block(x, target_len):
    """Simple linear-interpolation resample of one block to a target length.
    Not as clean as a proper polyphase resampler, but dependency-free (no
    scipy needed on the Pi) and good enough for real-time speech + noise-
    suppression audio at these block sizes. Used to bridge between whatever
    sample rate the USB mic hardware actually supports (often 44100/48000,
    not our model's native 16000) and the model's fixed 16kHz processing."""
    if len(x) == target_len:
        return x.astype(np.float32)
    orig_t = np.linspace(0.0, 1.0, len(x), endpoint=False)
    target_t = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(target_t, orig_t, x).astype(np.float32)


def pick_devices():
    """Auto-picks input/output device indices instead of relying on the
    system's 'default' device, which can point at a device with 0 input
    channels (e.g. the Pi's own headphone jack) even when a real USB
    microphone is plugged in. Input: first device with any input channels
    (your USB earphones/mic). Output: prefers the Pi's own built-in jack
    (bcm2835) since that's where your separate 3.5mm output earphones are
    plugged in; falls back to any device with output channels otherwise."""
    devices = sd.query_devices()
    input_idx = None
    output_idx = None
    for i, d in enumerate(devices):
        if input_idx is None and d["max_input_channels"] > 0:
            input_idx = i
        if "bcm2835" in d["name"].lower() and d["max_output_channels"] > 0:
            output_idx = i
    if output_idx is None:
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                output_idx = i
                break
    return input_idx, output_idx, devices


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


class StreamingDenoiserONNX:
    """Same algorithm as live_test.py's StreamingDenoiser, with the PyTorch
    forward pass swapped for an onnxruntime session call."""

    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name    # "context"
        self.output_name = self.sess.get_outputs()[0].name  # "mask"

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

        ctx = self.context_buf[None].astype(np.float32)  # (1, CONTEXT, N_FREQ)
        mask = self.sess.run([self.output_name], {self.input_name: ctx})[0][0]

        mask = sharpen(mask)
        masked_spec = spec * mask
        frame_out = (np.fft.irfft(masked_spec, n=N_FFT).astype(np.float32)) * WIN

        self.out_tail[:N_FFT] += frame_out
        self.wsum_tail[:N_FFT] += (WIN ** 2)

        wsum_safe = np.maximum(self.wsum_tail[:HOP], 1e-8)
        out_block = self.out_tail[:HOP] / wsum_safe

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
    here = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(here, "denoise_net.onnx")
    if not os.path.exists(onnx_path):
        sys.exit(f"{onnx_path} not found -- copy it over from the laptop "
                  f"(produced there by export_onnx.py) before running this.")

    denoiser = StreamingDenoiserONNX(onnx_path)

    input_idx, output_idx, devices = pick_devices()
    if input_idx is None:
        sys.exit("No input (microphone) device found. Plug in your USB mic/earphones and "
                  "re-run `python -c \"import sounddevice as sd; print(sd.query_devices())\"` "
                  "to confirm it shows up with input channels > 0 before trying again.")
    # The USB mic hardware often can't run at the model's native 16000Hz --
    # cheap USB audio chips are frequently locked to 44100/48000Hz only. So
    # we open the stream at whatever rate the input device actually reports
    # as its default (virtually guaranteed to be supported), and resample
    # to/from 16000Hz per-block so the model itself is unaffected.
    HW_SR = int(round(devices[input_idx]["default_samplerate"]))
    HW_HOP = int(round(HOP * HW_SR / SR))

    print(f"Loaded {onnx_path}")
    print(f"Model sample rate: {SR}Hz, block size: {HOP} samples ({HOP/SR*1000:.0f}ms per block)")
    print(f"Hardware sample rate: {HW_SR}Hz, block size: {HW_HOP} samples "
          f"(resampled to/from {SR}Hz automatically)")
    print(f"MASK_GAMMA={MASK_GAMMA}  MASK_FLOOR={MASK_FLOOR}  AGC_MAX_GAIN={AGC_MAX_GAIN}")
    print(f"Input device:  [{input_idx}] {devices[input_idx]['name']}")
    print(f"Output device: [{output_idx}] {devices[output_idx]['name']}")
    print()
    print("WEAR HEADPHONES / USE A USB HEADSET to avoid feedback.")
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
        in_path = os.path.join(here, f"pi_input_take{state['take_num']}_{stamp}.wav")
        out_path = os.path.join(here, f"pi_enhanced_take{state['take_num']}_{stamp}.wav")
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
        hw_block = indata[:, 0].astype(np.float32)
        block = resample_block(hw_block, HOP)  # HW_SR -> model's 16kHz
        try:
            out_block = denoiser.process(block)
        except Exception as e:
            print(f"Processing error: {e}", file=sys.stderr)
            out_block = np.zeros(HOP, dtype=np.float32)
        hw_out = resample_block(out_block, len(outdata))  # 16kHz -> HW_SR
        outdata[:, 0] = hw_out
        if RECORD and state["is_recording"]:
            state["input"].append(block.copy())
            state["output"].append(out_block.copy())

    try:
        with sd.Stream(samplerate=HW_SR, blocksize=HW_HOP, channels=1, dtype="float32",
                        device=(input_idx, output_idx), callback=callback):
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
              "to see available input/output devices, and check that your USB "
              "headset is the default (or set its device index explicitly).")


if __name__ == "__main__":
    main()
