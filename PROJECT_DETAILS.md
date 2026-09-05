# Project Details — Full Technical Reference

*A complete reference of every key number, dataset, and term used in this project, in plain language. Written so you can look up any specific fact quickly, and understand what it means without a technical background.*

---

## 1. The problem, restated simply

Build an AI system that takes in speech mixed with background noise (stationary noise like engine hum, changing noise, and sudden/impulsive noise) and outputs clearer, more understandable speech — running in real time, fully offline, on small embedded hardware, for defence communication use.

---

## 2. Basic definitions (read this section first if a term below is unfamiliar)

**Sample rate** — how many tiny snapshots of a sound are recorded every second. Measured in Hz (Hertz). This project uses **16,000 Hz (16kHz)** throughout — the same rate used in phone-quality speech systems. Every audio file used has to be at this exact rate, or the AI misreads it (like playing a record at the wrong speed).

**Frequency / spectrogram** — any sound is made of many pitches layered together. A spectrogram is a way of showing "how much of each pitch is present, moment by moment" instead of just the raw up-and-down wave. This project converts every sound into this form before the AI looks at it, because noise and speech tend to occupy different pitches, which is much easier for the AI to reason about than the raw wave.

**STFT (Short-Time Fourier Transform)** — the specific, standard mathematical technique used to do that raw-wave-to-frequency conversion. It works by chopping audio into small overlapping windows and analyzing the pitch content of each window separately. This is decades-old, well-established signal processing — nothing experimental.

**Frequency bin** — one "slot" in the frequency breakdown, representing a narrow range of pitch. This project splits sound into **257 frequency bins** per moment in time — think of it as a 257-slider graphic equalizer.

**Mask (specifically, Ideal Ratio Mask)** — the AI's actual output: 257 numbers, each between 0 and 1, one per frequency bin, representing "how much of this bin should be kept" (1 = fully keep, 0 = fully remove). The AI never generates new audio — it only ever decides what fraction of the real, existing sound to keep.

**GRU (Gated Recurrent Unit)** — the type of neural network used as the AI's "brain." A GRU has a short memory — it considers what it just heard a moment ago, not only the current instant — which matters because separating voice from noise often depends on how a sound changes over time.

**Epoch** — one full pass through all the training examples. "Training for 35 epochs" means the AI was shown the entire training dataset 35 times over, refining itself a little more each pass.

**SNR (Signal-to-Noise Ratio)** — a measure, in decibels (dB), of how loud the speech is compared to the noise. A high SNR (e.g. +15dB) means the noise is faint; a low or negative SNR (e.g. -5dB) means the noise is louder than the speech. This project trains on a wide random range of SNRs so the AI learns to handle both easy and very difficult noisy conditions.

**Causal / real-time processing** — processing audio using only what has already happened, with no ability to "peek" into the future. This is what makes real-time systems harder than cleaning up a pre-recorded file — the AI must decide instantly, with limited context, because there's no time to wait and see what comes next.

**ONNX** — a lightweight, portable file format for a trained AI model, designed so it can run using a small, fast program instead of the full (and much heavier) training software. Used specifically to prepare the model to run on the Raspberry Pi.

**Edge AI / edge processing** — running AI directly on the local device (like the Raspberry Pi) instead of sending data to a remote server or the cloud. Matters here because it means the system works fully offline, with no internet dependency — important for defence use.

**Validation set** — a portion of the data deliberately held back from training, used only to check whether the AI is learning the *general* pattern or just memorizing the exact examples it was shown. A model that does well on data it has never seen is more trustworthy than one only tested on data it was trained on.

---

## 3. The exact technical numbers used

| Setting | Value | What it means |
|---|---|---|
| Sample rate | 16,000 Hz | All audio standardized to this rate before use |
| FFT window size | 512 samples | The size of each small chunk of audio analyzed at once (32 milliseconds) |
| Hop size | 256 samples | How far the analysis window slides forward each step (16 milliseconds) — this is also how often the AI makes a fresh decision in real time |
| Frequency bins | 257 | Number of separate pitch "channels" the AI reasons about per moment |
| Context window | 8 frames (~144 ms) | How much recent history the AI considers before making its current decision |
| Model type | GRU (64 hidden units) + output layer | The AI's internal structure |
| Training epochs (latest run) | 35 | Full passes through the training data |
| Batch size | 32 | Number of examples processed together at once during training |
| Training/validation split | 85% / 15% (by file, not by example) | Ensures the AI is tested on speakers/noises it never trained on |
| SNR range used in training | -5 dB to +15 dB | Random loudness ratio of noise-to-speech per training example |
| Best validation error so far | 0.08451 (down from 0.08908 in an earlier, shorter run) | Lower is better; this is the model's average prediction error on unseen data |

---

## 4. Datasets used, in detail

**LibriSpeech (dev-clean subset)** — clean human speech recordings. Sourced from LibriVox public-domain audiobook readings. Part of the ~1,000-hour LibriSpeech corpus, natively 16kHz, licensed under **CC BY 4.0** (free to use with attribution). Source: openslr.org/12.

**CMU ARCTIC** — additional clean speech recordings, phonetically balanced (meaning the sentences were specifically chosen to cover a wide, even range of speech sounds), roughly 1,100+ utterances per speaker, across multiple speaker voices. Freely distributed for both commercial and non-commercial use. Source: festvox.org/cmu_arctic. Native sample rate 16kHz.

**ESC-50 (Environmental Sound Classification)** — the noise half of the training data. 2,000 total clips, 50 categories, 40 clips per category, each clip 5 seconds long, originally recorded at 44.1kHz (converted to 16kHz for this project). Licensed under **Creative Commons Attribution-NonCommercial**. Sourced from real field recordings via Freesound.org. Full category list:
- *Animals:* dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow
- *Nature/water:* rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm
- *Human non-speech:* crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking/sipping
- *Interior/domestic:* door knock, mouse click, keyboard typing, door creak, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking
- *Exterior/urban:* helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw

**Important, honest note:** ESC-50 does **not** include gunshots or explosions — the sounds most central to this project's stated problem focus. Fireworks is the closest available proxy (a sharp, loud, impulsive bang). This is why the project plan includes manually recording/sourcing real gunshot and explosion sound clips to add to training — a deliberate, identified next step, not an oversight being hidden.

**Custom recorded/added sounds (planned/in progress):** finger snaps, keyboard typing (your own mic/environment), claps, and defence-relevant impulsive sounds (gunshots, explosions) sourced from royalty-free sound-effect libraries, to directly close the gap above.

---

## 5. How training actually works, step by step

1. Take one clean speech clip and one noise clip.
2. Mix them at a randomly chosen loudness ratio (SNR) — sometimes barely noisy, sometimes very noisy.
3. Convert both the mixture and the original clean speech into frequency form (the 257-bin breakdown).
4. Calculate the mathematically "correct" mask — since the clean speech and noise are still known separately at this stage (before mixing), the ideal keep/remove decision for every bin can be computed exactly.
5. Show the AI only the noisy mixture, and have it guess the mask.
6. Compare its guess to the known correct answer, and adjust the AI's internal numbers slightly to make the next guess better.
7. Repeat this millions of times (thousands of examples, many epochs) until the AI's guesses consistently get close to correct.

---

## 6. Testing done and what it showed

- **Offline test** (`enhance_test.py`): mixes a speech + noise file, runs the trained model, saves before/after audio to listen to. Confirms the model audibly cleans up recordings.
- **Live microphone test** (`live_test.py`): runs the exact same real-time, moment-by-moment processing the Raspberry Pi will eventually run, using the laptop's own mic and headphones. This is what surfaced the real, honest findings noted throughout this project — some sounds are suppressed well (because they're in the training data), others aren't yet (because they're not) — and confirmed a small, expected delay at the very start of any sudden new sound, which is a property of real-time processing itself, not a flaw in this implementation specifically.
- **Tuning added after live testing, without retraining:** a "mask sharpening" setting (pushes low-confidence values more firmly toward zero), a "mask floor" setting (forces very-low-confidence bins to complete silence), and an automatic "make-up gain" step (restores overall loudness after suppression, since suppression can only remove volume, never add it).

---

## 7. Reference papers (verified directly from source)

**[1] Valin, J.-M. (2018).** *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement.* IEEE MMSP 2018. Combines a small neural network with classical pitch-based filtering for efficient real-time noise suppression (this is the paper behind the well-known RNNoise tool). Closest published relative to this project's overall approach — small model, real-time, frequency-domain — though this project trains its own model from scratch on its own data rather than using RNNoise directly.

**[2] Schröter, H., Escalante-B., A.N., Rosenkranz, T., Maier, A. (2022).** *DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio.* IWAENC 2022. Demonstrates a speech-enhancement model efficient enough to run about 25× faster than real-time on a standard laptop CPU. Evidence that lightweight, embedded-friendly speech enhancement is an active, credible research direction, supporting the feasibility of this project's approach.

**[3] Reddy, C.K.A., Beyrami, E., Pool, J., Cutler, R., Srinivasan, S., Gehrke, J. (2019).** *A Scalable Noisy Speech Dataset and Online Subjective Test Framework.* Interspeech 2019, pp. 1816–1820. Introduces a method (MS-SNSD) for generating noisy-speech training data at scale and evaluating speech-enhancement quality using real human listeners. **Note:** this project did not use the MS-SNSD dataset itself — cite this as related methodology (this project also builds its own synthetic noisy-speech mixtures, following the same general idea, from different source audio), not as a dataset actually used.

---

## 8. Known, honest limitations

- Sudden, very sharp sounds leak through slightly at the instant they begin — inherent to any real-time system with no ability to see the future, including production tools like RNNoise.
- Suppression quality depends on what noise types were included in training — sounds not represented in the training data are recognized less reliably, which is a data-coverage question, not a fundamental flaw. This is being actively addressed by adding real recorded and sourced sounds relevant to the actual problem (gunshots, explosions, finger snaps, keyboard sounds).
- This is a single-microphone system — a second reference microphone (for physical noise cancellation techniques like beamforming) is a possible future extension, not something currently implemented.
