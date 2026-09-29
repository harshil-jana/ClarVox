# ClarVox — training pipeline and Raspberry Pi deployment

AI/ML-based adaptive noise cancellation for defence communications, built for SIH 2026.

The project has two phases, and this README is ordered the same way:

- **Part A — Train (on your laptop, Windows).** Teaches the model, offline. Needs PyTorch.
- **Part B — Deploy (on the Raspberry Pi, Linux).** Runs the already-trained model live, in
  real time, over a microphone. Needs `onnxruntime`, not PyTorch.

Run each numbered step in order the first time through — later steps assume the files
produced by earlier ones already exist.

---

## Part A — Train on your laptop

Run these from inside `D:\sih-anc`, in a terminal (PowerShell or Command Prompt).

### 0. One-time setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(`venv\Scripts\activate` needs to be run again every time you open a new terminal window.)

### 1. Check your data

```
python check_data.py
```

This tells you how many speech/noise files you have and flags any that aren't 16kHz mono
(the format everything else assumes). If it says "NOT 16kHz", fix everything in one go with:

```
python fix_sample_rates.py
```

This doesn't need ffmpeg -- it uses soundfile + scipy (already in requirements.txt) to
resample every non-conforming file in `data\speech` and `data\noise` to 16kHz mono, in
place, backing up the originals into a `_originals` subfolder first. Run `check_data.py`
again afterward to confirm it's clean.

(If you'd still rather use ffmpeg for some files: `winget install ffmpeg` installs it on
Windows, then `ffmpeg -i input.wav -ar 16000 -ac 1 output.wav` per file -- but
`fix_sample_rates.py` is faster since it's already installed and handles the whole folder
at once.)

Don't move on to training until `check_data.py` prints `READY? YES`.

### 2. (Optional) record your own noise clips

```
python record_noise.py finger_snap 20
python record_noise.py keyboard_typing 20
```

Records straight from your microphone into `data\noise\<name>.wav`, ready to be picked up
by the very next training run. Useful for closing gaps in the public noise dataset (ESC-50
doesn't include gunshots/explosions, for example) -- record 15-30 seconds per sound, making
it several times with natural pauses in between rather than once, so the training code has
real variety to sample from within one file.

### 3. (Optional) sanity-check the model shape

```
python model.py
```

No data needed -- just confirms the network builds and prints its parameter count (around
78,700 parameters -- a single GRU layer with 64 hidden units plus one linear output layer).
Should finish instantly and print "Self-test passed."

### 4. Train

```
python train.py
```

Prints one line per epoch (train loss, validation loss, learning rate, time taken). Takes
anywhere from a few minutes to a while depending on how much data you have and your CPU --
this is the part to kick off and leave running while you work on the hardware side.

**Note:** if `denoise_net_best.pt` already exists from a previous run, `train.py` fine-tunes
it by default (`FINE_TUNE_FROM_EXISTING = True`) rather than starting over from random
weights -- this is much faster for "teach it a few new noise types" (12 epochs, a gentler
learning rate) than a full retrain. If you want a genuine from-scratch run instead (e.g.
after a major change to the training data), delete or rename `denoise_net_best.pt` first.

When it finishes you'll have:
- `denoise_net_best.pt` -- the checkpoint with the lowest validation loss (this is the one that matters)
- `denoise_net_final.pt` -- the last epoch's weights, kept as a fallback
- `train_log.csv` -- loss per epoch, if you want to eyeball whether it was still improving

**What to look for:** `val_loss` should trend down across epochs. If it's flat from epoch 1,
something's wrong upstream (check data.py's assumptions against your actual files) -- paste
the printed output back and it can be debugged from there.

### 5. Listen to the result offline

```
python enhance_test.py
```

Picks a speech + noise file from your data folders, mixes them, runs the trained model, and
writes `test_noisy.wav` / `test_enhanced.wav` so you can compare by ear. This is the real
test -- loss numbers can look fine while the audio still sounds off, so always listen before
trusting the model.

You can also point it at specific files:

```
python enhance_test.py data\speech\some_clip.wav data\noise\some_clip.wav 0
```

(last number is the SNR in dB -- lower = noisier mix, harder test)

### 6. Rehearse live, through your own microphone

```
python live_test.py
```

**Wear headphones** (the enhanced output plays out live -- without headphones the mic will
pick its own output back up and howl). Unlike `enhance_test.py`, this processes audio
continuously, 16ms at a time, using only what it's already heard -- the same causal,
block-by-block pattern that will run on the Raspberry Pi. This is the closest laptop preview
of real-time behaviour there is, and worth doing before ever trusting the model live on the
Pi.

Recording is manual and off by default while it runs: press ENTER to start capturing a take,
ENTER again to stop and save it (as `live_input_take<N>_...wav` / `live_enhanced_take<N>_...wav`
next to the script) -- do this as many times as you like in one session, e.g. right as you
play a specific noise you want to check. `MASK_GAMMA` / `MASK_FLOOR` at the top of the file
control how aggressively noise is suppressed, without needing to retrain.

### 7. Export for the Raspberry Pi

```
python export_onnx.py
```

Produces `denoise_net.onnx` and checks that its output matches the PyTorch model closely (in
practice, to within about 1.64×10⁻⁷ -- effectively bit-identical). Copy this one file to the
Pi (over `scp` or a USB stick) -- the Pi only needs `onnxruntime`, not the full PyTorch
install.

---

## Part B — Deploy on the Raspberry Pi

Run these on the Pi itself, over SSH or a direct terminal. Copy `denoise_net.onnx` (from
step 7 above) into this same project folder on the Pi before starting.

### 8. One-time Pi setup

```
python3 -m venv ~/anc-venv
source ~/anc-venv/bin/activate
pip install onnxruntime numpy sounddevice
```

Deliberately not the full `requirements.txt` -- the Pi never needs PyTorch, scipy, or
soundfile, only what the real-time runtime scripts actually import.

### 9. Smoke-test the model, no audio hardware needed

```
python pi_model_smoke_test.py
```

Loads `denoise_net.onnx` through ONNX Runtime and runs 20 inference passes with fake input --
no microphone or speaker required. Confirms the model + onnxruntime + this specific Pi
combination actually works, and prints timing (expect roughly 0.27ms average / 1.52ms
maximum per 16ms frame -- well inside the 16ms-per-frame real-time budget it checks against).
Run this before any audio hardware even arrives; if it fails, that's a model/runtime problem,
not an audio problem.

### 10. Test the output hardware alone

```
python pi_output_test.py
```

No model involved -- just plays a plain 440Hz tone through the Pi's 3.5mm jack, to confirm
`sounddevice`/PortAudio can actually drive live playback on this Pi before adding the
denoiser into the mix. If you don't hear the tone, that's a hardware/driver problem to fix
here first, separate from anything AI-related.

### 11. Run the real-time denoiser

```
python pi_live_test.py
```

Plug in a USB microphone/headset for input; the Pi's own 3.5mm jack is used for output. Same
causal, block-by-block algorithm as `live_test.py`, same `MASK_GAMMA` / `MASK_FLOOR` tuning,
same manual ENTER-to-record takes (saved as `pi_input_take<N>_...wav` /
`pi_enhanced_take<N>_...wav`). Internally this runs the mic and the speaker as two fully
independent audio streams bridged by a small buffer, rather than one combined stream --
necessary because they're two separate physical sound cards with no shared hardware clock.

**Important:** WAV writes complete (and get printed as "Saved") the moment `save_wav()`
returns without error, but that only guarantees the data reached the OS's page cache, not
that it's physically on the SD card. If the Pi loses power abruptly right after a demo
(unplugged rather than `sudo shutdown -h now` or `sync`), a just-saved take can still be
lost. Run `sync` on the Pi before cutting power if a recording needs to survive.

### 12. Double-check a saved take numerically

```
python compare_takes.py pi_input_take1_20260908_112802.wav pi_enhanced_take1_20260908_112802.wav
```

Reports whether the input/enhanced pair are bit-for-bit identical, the max sample
difference, and the RMS (loudness) of each -- useful when a short or quiet noise burst is
hard to judge confidently by ear alone.

### 13. (Optional) run it automatically on boot

```
sudo cp sih-anc.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sih-anc.service
```

Edit the `User=`, `WorkingDirectory=`, and `ExecStart=` paths in `sih-anc.service` first to
match your actual Pi username and where the project folder actually lives. Once enabled,
`pi_live_test.py` starts automatically on boot and restarts itself on a crash -- no SSH
session required to keep the demo running. Note that a systemd service has no interactive
terminal attached, so the manual ENTER-to-record feature in `pi_live_test.py` won't do
anything useful when run this way; use step 11 directly (in a terminal) instead, whenever
you specifically need to capture takes.

---

## If something breaks

**On the laptop (Part A):**
- **"No speech/noise files found"** -- data.py/train.py expect `.wav` or `.flac` files
  directly inside `data\speech` and `data\noise` (not in subfolders).
- **"... is Xhz -- resample to 16000Hz first"** -- run `fix_sample_rates.py` (step 1) on
  that file.
- **Training loss is `nan`** -- almost always a silent or all-zero audio file in your data
  folders; check `check_data.py`'s per-file durations for anything that reads 0.0s.

**On the Raspberry Pi (Part B):**
- **`pi_model_smoke_test.py` reports average inference time above ~16ms** -- the model may
  not keep up with live audio on this particular Pi; worth knowing before a demo, not during
  one.
- **No live audio, but a saved WAV take looks correct** -- this was exactly the
  combined-duplex-stream bug `pi_live_test.py` is already written to avoid (see step 11);
  if it resurfaces, check that `pick_devices()` is actually finding your USB mic as the input
  and the Pi's own jack (`bcm2835`) as the output via
  `python -c "import sounddevice as sd; print(sd.query_devices())"`.
- **A take that should exist is missing after the Pi was power-cycled** -- see the note under
  step 11; the write likely never reached the SD card before power was cut.

**Either machine:**
- **Anything else** -- copy the full error message/traceback and the exact command you ran;
  that's enough to diagnose without needing to reproduce it locally.
