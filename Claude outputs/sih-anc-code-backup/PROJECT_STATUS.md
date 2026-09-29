# Project Status — What We've Done So Far

*A plain-language summary of everything built and tested in this project, for your own reference and for explaining progress to your team or the jury.*

---

## The goal, in one sentence

Build an AI system that listens to speech mixed with background noise and outputs cleaner, more understandable speech — trained from scratch (not borrowed from an existing tool), light enough to run in real time on a Raspberry Pi with no internet connection, for defence communication use.

---

## The overall plan we followed

Everything splits into two phases:

- **Phase A — Training (on the laptop).** Teach an AI model to tell voice apart from noise, using thousands of example recordings.
- **Phase B — Deployment (on the Raspberry Pi).** Take that already-trained AI and run it live, continuously, on real microphone input, fully offline.

We've made real progress on Phase A. Phase B (the Raspberry Pi hardware side) is still ahead of us.

---

## 1. The code we built

All of this lives in your `D:\sih-anc` folder. In the order they're actually used:

- **`data.py`** — turns raw speech and noise recordings into the training examples the AI learns from (mixing them together, converting raw sound into frequency information).
- **`model.py`** — defines the AI's actual structure: a small neural network (a GRU, explained below) that outputs a decision for how much of the sound at each moment is voice vs. noise.
- **`train.py`** — the training loop itself. This is what actually teaches the AI, by showing it example after example and correcting it when it's wrong.
- **`check_data.py`** and **`fix_sample_rates.py`** — helper tools that check your audio files are in the right format before training, and automatically fix ones that aren't (more on this below).
- **`enhance_test.py`** — an offline test: mixes one speech clip and one noise clip, runs your trained AI on it, and saves before/after audio files so you can listen and judge the result.
- **`live_test.py`** — a real-time test using your laptop's microphone and headphones, so you can hear the AI working live, the same way it will eventually work on the Raspberry Pi.
- **`record_noise.py`** — lets you record new noise sounds (claps, keyboard typing, etc.) straight into the training data folder, using your own microphone.
- **`export_onnx.py`** — converts the finished, trained AI into a lightweight format that can run on the Raspberry Pi without needing the heavy training software. **Not run yet** — this is a Phase B step, planned for once the Pi hardware is ready.

---

## 2. Training results so far

We trained the AI twice:

- **First run: 15 rounds of practice (called "epochs").** Result: a measurable error score of 0.08908 (lower is better — this measures how far off the AI's guesses are on average).
- **Second run: 35 rounds of practice.** Result: improved to 0.08451 — a real, if modest, improvement. The improvement mostly leveled off after about round 20, which tells us the AI had learned close to what it could from the data available — more training rounds alone won't help much further; more/better data is the next lever.

**What it was trained on:** real human speech recordings (from two public datasets — LibriSpeech and CMU ARCTIC) mixed with background noise clips from a public dataset called ESC-50, at random loudness levels, so the AI sees a wide range of noisy conditions.

---

## 3. Testing we've done, and what we learned

We tested the trained AI two ways:

**Offline listening test.** Ran a noisy recording through the AI and listened to before/after. The AI does audibly clean up the recording.

**Live microphone test.** Ran the AI live through your laptop's mic and headphones, speaking with real background noise (claps, keyboard typing, finger snaps) happening around you. This surfaced some genuinely useful findings:

- The AI reduced clapping sounds noticeably — because clapping is one of the sound types in its training data.
- A finger-snap was **not** recognized well — because that specific sound isn't in the training data at all. This isn't a bug; the AI can only learn to recognize what it's actually been shown examples of.
- Sudden sounds always leak through slightly at the very first instant, because the AI reacts to what it just heard — it can't see into the future. This is a normal limitation of any real-time system, not unique to ours.
- Right after suppression, we noticed the cleaned-up voice sounded quieter than expected. This is because the AI can only ever *remove* sound, never *add* loudness back. We fixed this by adding an automatic "make-up gain" step, which restores the overall loudness after suppression — implemented and working.
- We also added two tuning settings that let you make suppression stronger *without retraining*: one sharpens how confidently the AI commits to "this is noise" (currently set to a value chosen after listening to both settings side-by-side), and one forces very-low-confidence noise to complete silence rather than just "turned down" (added after we noticed the make-up gain step was re-amplifying leftover noise).

**Important discovery:** the noise dataset we trained on (ESC-50) does **not** contain gunshots or explosions — the exact sounds the project's problem statement is most focused on. It does contain some related sounds (helicopter, siren, engine, fireworks), but not the real thing. This is the most important gap still open.

---

## 4. What's planned next (not done yet)

- **Add real gunshot, explosion, and other missing impulsive sounds to the training data**, then retrain. We already prepared the retraining process to be fast for this: instead of starting over from scratch (which took 1.5–2 hours), the training script now continues from the already-trained model, so adding new sounds should only take about 30–40 minutes. This step is planned to happen at college.
- **Convert the trained model to the lightweight ONNX format** (`export_onnx.py`) — quick, a couple of minutes, but only worth doing once the training above is finalized.
- **Get the Raspberry Pi hardware working** — recommended approach is a single USB headset (built-in mic + earphone in one device) for the simplest, most reliable setup, since the Pi has no built-in microphone input.
- **Run the exact same real-time processing on the Pi** that's already been tested and validated on the laptop.

---

## 5. The presentation (PPT)

- The **Technical Approach** slide was rebuilt to show the actual process as two flowcharts (technologies used, and the two-phase training → deployment pipeline) instead of plain bullet points — this is now saved and matches the real code.
- The rest of the deck was reviewed against what's actually been built. Three wording spots were flagged as slightly overstating what's currently done (the "transient-aware handling" claim, a reference to a dataset that wasn't actually used, and the "defence-specific" training claim) — updated wording was suggested for each, honest but still confident, ready for you to apply. Team ID and team name are still blank placeholders and need to be filled in before submission.

---

## In short

The core AI pipeline works, has been trained twice, and has been validated both offline and live on a real microphone — that's a genuinely working prototype, not just code that runs. The two things still ahead are: teaching it the specific defence-relevant impulsive sounds (gunshots, explosions) that are the heart of the problem statement, and getting it running live on the actual Raspberry Pi hardware.
