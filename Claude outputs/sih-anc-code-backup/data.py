"""
Turns raw speech + noise clips into (context, ideal_mask) training pairs.

Everything downstream (model.py, train.py, the offline test script, and the
Pi's real-time script) shares these exact constants -- SR/N_FFT/HOP/CONTEXT
must never drift between files, or the model trained here won't line up
with what's fed to it later.
"""
import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 512
HOP = 256
CONTEXT = 8  # how many recent frames of context the model gets to see
WIN = np.hanning(N_FFT).astype(np.float32)


def load_mono(path):
    """Load an audio file as float32 mono at SR. Raises if the sample rate
    doesn't match -- resample with ffmpeg first if you hit this."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"{path} is {sr}Hz -- resample to {SR}Hz first (ffmpeg -i in.wav -ar 16000 -ac 1 out.wav)"
    return audio


def mix_at_snr(speech, noise, snr_db):
    """Mix speech with a random segment of noise, scaled to hit snr_db."""
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    start = np.random.randint(0, len(noise) - len(speech) + 1)
    noise = noise[start:start + len(speech)]
    sig_p = np.mean(speech ** 2) + 1e-8
    noise_p = np.mean(noise ** 2) + 1e-8
    scale = np.sqrt(sig_p / (noise_p * 10 ** (snr_db / 10)))
    return speech + noise * scale, noise * scale


def spectrogram(audio):
    """Frame `audio` into overlapping N_FFT windows, hop HOP, and return the
    magnitude spectrogram as (n_frames, N_FFT//2+1)."""
    pad = np.zeros(N_FFT, dtype=np.float32)
    padded = np.concatenate([pad, audio])
    n = (len(padded) - N_FFT) // HOP
    mags = np.zeros((n, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(n):
        mags[i] = np.abs(np.fft.rfft(padded[i * HOP:i * HOP + N_FFT] * WIN))
    return mags


def make_training_example(speech, noise, snr_db):
    """One (context, target_mask) pair, sampled at a random point in a
    freshly generated mixture. Call this many times per (speech, noise)
    pair -- random SNR and random mix position means a small pool of source
    clips still produces huge variety."""
    mix, noise_scaled = mix_at_snr(speech, noise, snr_db)
    mix_mag = spectrogram(mix)
    clean_mag = spectrogram(speech)
    noise_mag = spectrogram(noise_scaled)
    n = min(len(mix_mag), len(clean_mag), len(noise_mag))
    if n <= CONTEXT:
        # extremely short clip -- pad by repeating the first frame
        pad_needed = CONTEXT + 1 - n
        mix_mag = np.concatenate([np.tile(mix_mag[:1], (pad_needed, 1)), mix_mag])
        clean_mag = np.concatenate([np.tile(clean_mag[:1], (pad_needed, 1)), clean_mag])
        noise_mag = np.concatenate([np.tile(noise_mag[:1], (pad_needed, 1)), noise_mag])
        n = len(mix_mag)
    t = np.random.randint(CONTEXT - 1, n)
    context = mix_mag[t - CONTEXT + 1: t + 1]  # (CONTEXT, n_freq)
    ideal_mask = np.clip(clean_mag[t] / (clean_mag[t] + noise_mag[t] + 1e-8), 0.0, 1.0)
    return context.astype(np.float32), ideal_mask.astype(np.float32)
