# Running sih-anc on Windows (PowerShell) — step by step

Project folder: `D:\Projects\SIH_2026\sih-anc`
The `venv` in that folder already has torch, numpy, scipy, soundfile, onnx,
onnxruntime and sounddevice installed, so setup is a one-liner.

---

## Step 0 — Open the project

```powershell
cd D:\Projects\SIH_2026\sih-anc
```

## Step 1 — Allow the venv to activate (PowerShell only)

PowerShell blocks activation scripts by default. This unblocks it for **this
window only** — it changes nothing permanently:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

## Step 2 — Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

The prompt should now start with `(venv)`. Confirm the right Python is active:

```powershell
Get-Command python | Select-Object Source
```

It must point inside `...\sih-anc\venv\Scripts\python.exe`.

> Repeat Steps 1 and 2 every time you open a new PowerShell window.

## Step 3 — Confirm the libraries load

```powershell
python -c "import torch, onnxruntime, sounddevice, soundfile; print('all libraries OK')"
```

---

# A. Demo path (model is already trained — use this on jury day)

`denoise_net_best.pt` and `denoise_net.onnx` are already in the folder, so you
can skip training entirely.

### A1. Offline before/after proof

```powershell
python enhance_test.py
```

Writes `test_noisy.wav` and `test_enhanced.wav` next to the script. Play them:

```powershell
start test_noisy.wav
start test_enhanced.wav
```

Point it at specific files and a specific SNR (last number is dB — lower is harder):

```powershell
python enhance_test.py data\speech\some_clip.wav data\noise\some_clip.wav 0
```

### A2. Live microphone demo on the laptop

**Wear headphones** — speaker output loops back into the mic and howls.

```powershell
python live_test.py
```

- Audio flows mic → model → headphones immediately.
- Press **ENTER** to start recording a take, **ENTER** again to stop and save it.
- Each take saves `live_input_take<N>_<timestamp>.wav` and
  `live_enhanced_take<N>_<timestamp>.wav` in the project folder.
- **Ctrl+C** ends the session (a running take is saved first).

If the stream fails to open, list your audio devices:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### A3. Measure a saved take

```powershell
python compare_takes.py live_input_take1_<timestamp>.wav live_enhanced_take1_<timestamp>.wav
```

Prints the RMS and peak reduction in dB — this is where the 5.2 dB average /
28.8 dB peak numbers come from.

---

# B. Full training path (only if you are retraining)

### B1. Check the dataset

```powershell
python check_data.py
```

Do not continue until it prints `READY? YES`. If anything reports "NOT 16kHz":

```powershell
python fix_sample_rates.py
python check_data.py
```

(`fix_sample_rates.py` resamples everything in `data\speech` and `data\noise`
to 16 kHz mono in place, backing the originals up into a `_originals` subfolder.
No ffmpeg needed.)

### B2. Record extra noise samples (optional)

```powershell
python record_noise.py gunshot 20
python record_noise.py engine_hum 30
```

`<name>` becomes `data\noise\<name>.wav`; the second number is seconds (default 15).
Make the sound several times, spread out, while it records.

### B3. Sanity-check the network

```powershell
python model.py
```

Instant. Should print the parameter count and `Self-test passed.`

### B4. Train

```powershell
python train.py
```

One line per epoch (train loss, val loss, LR, time). `val_loss` should trend
down. Note: `FINE_TUNE_FROM_EXISTING = True` in `train.py`, so if
`denoise_net_best.pt` exists it **continues** from it at a gentler learning rate
instead of starting over. Delete that file first if you want a fresh model.

Produces:

| File | What it is |
|---|---|
| `denoise_net_best.pt` | lowest validation loss — the one that matters |
| `denoise_net_final.pt` | last epoch, kept as a fallback |
| `train_log.csv` | loss per epoch |

### B5. Listen before trusting it

```powershell
python enhance_test.py
```

Loss numbers can look fine while the audio sounds wrong. Always listen.

### B6. Export for the Raspberry Pi

```powershell
python export_onnx.py
```

Produces `denoise_net.onnx` (~310 KB) and verifies it matches the PyTorch model.

---

# C. Raspberry Pi deployment (run from the same PowerShell window)

### C1. Copy the model across

```powershell
scp .\denoise_net.onnx harshil_jana@sih-anc.local:~/sih-anc-code/
```

If the hostname does not resolve, use the IP:

```powershell
scp .\denoise_net.onnx harshil_jana@10.76.219.8:~/sih-anc-code/
```

### C2. Log in and run it on the Pi

```powershell
ssh harshil_jana@sih-anc.local
```

Then, on the Pi (this is bash, not PowerShell):

```bash
cd ~/sih-anc-code
source ~/anc-venv/bin/activate
python pi_live_test.py
```

### C3. Pull the Pi's recordings back to the laptop

Run this in a **new** PowerShell window (leave the Pi session running). The
quotes matter — PowerShell would otherwise try to expand the `*` itself:

```powershell
scp "harshil_jana@10.76.219.8:~/sih-anc-code/*.wav" "D:\Projects\SIH_2026\sih-anc\recordings\"
```

---

# Quick reference

| Goal | Command |
|---|---|
| Activate env | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` then `.\venv\Scripts\Activate.ps1` |
| Check dataset | `python check_data.py` |
| Fix sample rates | `python fix_sample_rates.py` |
| Record noise | `python record_noise.py <name> [seconds]` |
| Train | `python train.py` |
| Offline A/B | `python enhance_test.py [speech.wav noise.wav snr_db]` |
| Live mic demo | `python live_test.py` |
| Measure a take | `python compare_takes.py <input.wav> <enhanced.wav>` |
| Export ONNX | `python export_onnx.py` |
| Leave the env | `deactivate` |

## If something breaks

- **`Activate.ps1 cannot be loaded`** — you skipped Step 1.
- **`No speech/noise files found`** — the scripts expect `.wav`/`.flac` directly
  inside `data\speech` and `data\noise`, not in subfolders.
- **Training loss is `nan`** — almost always a silent/all-zero audio file; look
  for a 0.0 s duration in `check_data.py`'s per-file output.
- **Howling during `live_test.py`** — you are not wearing headphones.
- **`denoise_net_best.pt not found`** — run `train.py` first.
