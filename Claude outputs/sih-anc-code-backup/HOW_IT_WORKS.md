# How the Noise Cancellation System Works

*A plain-language walkthrough of every file in this project, written so you can explain it confidently to the jury without needing to be a machine learning expert.*

---

## 1. The one-sentence version

The system listens to a mix of speech and noise, and every 16 milliseconds it decides, frequency by frequency, "how much of what I'm hearing right now is voice, and how much is noise" — then it keeps the voice part and throws away the noise part. It does this so fast and continuously that it sounds like real-time noise cancellation, not a delayed recording effect.

That decision-maker is a small neural network you trained yourself, not a copy of an existing tool like RNNoise.

---

## 2. The big picture: two phases

Everything in this project splits into two phases, and it helps to keep them mentally separate.

**Phase A — Training (done on your laptop).** This is where the neural network actually *learns*. You feed it thousands of examples of "noisy speech in, clean speech mask out" until it gets good at the pattern. This phase needs a lot of computing power and takes time (our 35-epoch run took about 1.5–2 hours), but it only needs to happen once. The result is a saved file of learned numbers — the trained model.

**Phase B — Deployment (runs on the Raspberry Pi, in real time).** This is where the *already-trained* model is used live. No more learning happens here — the model's knowledge is frozen. It just takes a stream of microphone audio in and produces a stream of cleaned-up audio out, continuously, with no noticeable delay. This is what the jury will actually see running on the day.

Training is slow and happens once. Deployment is fast and happens every time someone uses the device.

---

## 3. How the AI actually removes noise (the core idea)

Here's the mental model to explain to the jury:

Imagine the sound is split into many separate narrow "channels," like a very fine graphic equalizer with 257 sliders, each one controlling a tiny slice of the audio frequency range (from low rumbles to high hiss). Every 16 milliseconds, the AI looks at the current sound and decides, independently for each of those 257 sliders: "keep this slider open" (this frequency slice is mostly voice) or "close this slider" (this frequency slice is mostly noise). The values in between (partially open) are for frequencies where voice and noise overlap.

This set of 257 decisions is called a **mask**. The system doesn't invent or synthesize clean speech from scratch — it only ever decides what to keep and what to remove from what's actually there. That's an important distinction if the jury asks "is the AI generating fake audio?" — no, it's a filter, not a generator.

The technical name for this specific approach is an **Ideal Ratio Mask**, and it's the same general family of technique used by well-known tools like RNNoise and DeepFilterNet — the difference here is that this mask is predicted by a model *you trained from scratch* on defence-relevant noise conditions, rather than a pretrained general-purpose model.

---

## 4. What "training" actually means, concretely

Training is just repeatedly showing the network examples and correcting it when it's wrong, thousands of times over. Concretely, for every training example, the code:

1. Takes a real clean speech clip and a real noise clip.
2. Mixes them together at a random loudness ratio (random SNR — Signal-to-Noise Ratio) — sometimes the noise is faint, sometimes it's overwhelming. This forces the model to handle a wide range of real-world conditions, not just one.
3. Calculates the *correct answer* mathematically — since we still have the original clean speech and the original noise separately (before mixing), we know exactly what the "ideal" mask should have been for that mixture.
4. Shows the network the noisy mix and asks it to guess the mask.
5. Compares the network's guess to the known correct answer, and nudges the network's internal numbers slightly to make its next guess a bit closer.

Repeat that process millions of times (thousands of examples, many times over — that's what "epochs" means: one full pass through the training data) and the network gradually gets good at recognizing noise patterns it has seen — that's why the *variety* of noise clips you train on matters so much for how well it generalizes to sounds it hasn't seen before.

---

## 5. File-by-file walkthrough

These are listed in the order they're actually used, not alphabetically.

### `data.py` — turns raw audio files into training examples
This file doesn't train anything itself — it's the "kitchen" that prepares ingredients for training. Its two main jobs:
- **Mixing**: given a speech clip and a noise clip, blend them at a specified loudness ratio.
- **Framing (STFT)**: chop continuous audio into small overlapping 32-millisecond windows and convert each window from a raw waveform into frequency information (how much low pitch, mid pitch, high pitch, etc. is present in that instant). This frequency view is what the neural network actually looks at — not the raw waveform. This conversion step is called a **Short-Time Fourier Transform (STFT)**, and it's a completely standard, well-understood signal-processing technique — nothing experimental about it.

### `model.py` — the neural network's structure
This defines the actual "brain" — a small network with two layers:
- A **GRU** (Gated Recurrent Unit) — a type of neural network built specifically for sequences, meaning it has a short memory of what it just heard, not just the current instant. This matters because noise sounds different from speech partly *over time*, not just in a single frozen instant.
- A final layer that turns the GRU's internal understanding into the 257 mask values (one per frequency slider) between 0 and 1.

The whole network is deliberately small (a few hundred thousand numbers) — big enough to learn the pattern, small enough to run in real time on a Raspberry Pi with no GPU.

### `train.py` — the actual training loop
This is what you ran for 1.5–2 hours. It:
- Loads all your speech and noise files.
- Splits them into a *training* set (used to teach the model) and a *validation* set (files never used for teaching — only for checking whether the model is actually learning the general pattern, or just memorizing).
- Runs the training loop described in section 4 for 35 full passes (epochs) over the data.
- After every epoch, saves the model's numbers to `denoise_net_best.pt` **only if** this epoch is the best-scoring one so far on the validation set — so you always end up with the best version, not just the last one.
- Prints the loss (a number representing "how wrong the model's guesses are, on average" — lower is better) after every epoch so you can watch it improve.

### `check_data.py` and `fix_sample_rates.py` — housekeeping helpers
Not part of the AI itself. `check_data.py` checks whether your speech/noise files are actually usable (right format, right sample rate) before you waste time training on broken data. `fix_sample_rates.py` automatically repairs files that aren't in the right format (16,000 samples per second, mono).

### `enhance_test.py` — the offline listening test
Takes one speech clip and one noise clip, mixes them, runs the *already-trained* model over the whole thing at once, and saves two files (`test_noisy.wav`, `test_enhanced.wav`) so you can listen to before/after. This is how you validate the model actually works before trusting it in a live demo.

### `live_test.py` — the real-time microphone test
This is the important one to understand deeply, because it's a laptop rehearsal of exactly what will run on the Raspberry Pi tomorrow. Unlike `enhance_test.py` (which processes a whole file at once, with the luxury of seeing the entire clip), `live_test.py` processes audio in a continuous stream, one 16-millisecond chunk at a time, using only what it has already heard — never any future audio. This "no looking ahead" constraint is called **causal** processing, and it's the whole reason real-time noise cancellation is harder than cleaning up a pre-recorded file: the model has to make its best decision immediately, with limited context, because there's no time to wait and see what happens next.

It also includes two tuning knobs added after listening tests on real audio:
- **`MASK_GAMMA`** — sharpens the mask's decisions after the fact, without retraining. Raising it pushes "probably noise" values closer to fully-zero, for stronger suppression.
- **`MASK_FLOOR`** — a hard cutoff: any frequency the model is fairly confident is noise gets forced to complete silence, rather than just "turned down a lot."
- **Makeup gain (AGC)** — suppression can only ever make audio quieter, never louder, so this step measures how much loudness was lost and boosts the output back up afterward, so speech doesn't end up sounding weaker than it started.

### `export_onnx.py` — preparing the model for the Raspberry Pi
PyTorch (the framework used for training) is a large, heavy piece of software — installing it on a Raspberry Pi is possible but slow, and downloads a lot of unnecessary training-only functionality the Pi will never use. This script converts the trained model into a lightweight, portable format called **ONNX**, which can run using a much smaller, faster runtime (`onnxruntime`) — no full PyTorch installation needed on the Pi. This is standard practice for deploying trained AI models onto small embedded devices.

---

## 6. Suggested one-minute explanation for the jury

*"We built a small neural network that looks at incoming audio 16 milliseconds at a time, splits it into 257 frequency bands, and decides — based on patterns it learned from thousands of real speech and noise examples — how much of each band is voice versus noise. We trained it ourselves, from scratch, on defence-relevant noise conditions rather than using an existing pretrained tool, so it's tuned specifically for this use case. It only removes information that's confidently identified as noise, never generates or alters the speech itself, and runs continuously in real time with no meaningful delay — which we've verified both on recorded test files and live through a microphone."*

---

## 7. Known, honest limitations (good to have ready if asked)

Being upfront about these — and explaining *why* they happen — reads far better to a technical jury than pretending they don't exist:

- **Sudden, very sharp noises (a clap, a finger-snap) leak through slightly at the exact instant they start.** This is inherent to any real-time system that can't see into the future — the model needs a brief moment of hearing a sound before it can confidently classify it as noise. The same limitation exists in production tools like RNNoise.
- **Noise types the model hasn't specifically seen during training are suppressed less reliably** than noise types it has seen many examples of. This is a data-coverage question, not a fundamental flaw — suppression quality scales with training data variety, and this is easy to explain as a natural "next step" if asked about future work.
- **This is a single-microphone (single-channel) system.** A second microphone (for beamforming/reference-noise cancellation) is a documented possible extension, not something claimed as already built.
