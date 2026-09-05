# sih-anc training pipeline

Run these in order, from inside `D:\sih-anc`, in a terminal (PowerShell or
Command Prompt).

## 0. One-time setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(`venv\Scripts\activate` needs to be run again every time you open a new
terminal window.)

## 1. Check your data

```
python check_data.py
```

This tells you how many speech/noise files you have and flags any that
aren't 16kHz mono (the format everything else assumes). If it says
"NOT 16kHz", fix everything in one go with:

```
python fix_sample_rates.py
```

This doesn't need ffmpeg -- it uses soundfile + scipy (already in
requirements.txt) to resample every non-conforming file in `data\speech`
and `data\noise` to 16kHz mono, in place, backing up the originals into a
`_originals` subfolder first. Run `check_data.py` again afterward to
confirm it's clean.

(If you'd still rather use ffmpeg for some files: `winget install ffmpeg`
installs it on Windows, then `ffmpeg -i input.wav -ar 16000 -ac 1
output.wav` per file -- but `fix_sample_rates.py` is faster since it's
already installed and handles the whole folder at once.)

Don't move on to training until `check_data.py` prints `READY? YES`.

## 2. (Optional) sanity-check the model shape

```
python model.py
```

No data needed -- just confirms the network builds and prints its
parameter count. Should finish instantly and print "Self-test passed."

## 3. Train

```
python train.py
```

Prints one line per epoch (train loss, validation loss, learning rate,
time taken). Takes anywhere from a few minutes to a while depending on how
much data you have and your CPU -- this is the part to kick off and leave
running while you work on the hardware side.

When it finishes you'll have:
- `denoise_net_best.pt` -- the checkpoint with the lowest validation loss (this is the one that matters)
- `denoise_net_final.pt` -- the last epoch's weights, kept as a fallback
- `train_log.csv` -- loss per epoch, if you want to eyeball whether it was still improving

**What to look for:** `val_loss` should trend down across epochs. If it's
flat from epoch 1, something's wrong upstream (check data.py's assumptions
against your actual files) -- paste the printed output back and it can be
debugged from there.

## 4. Listen to the result

```
python enhance_test.py
```

Picks a speech + noise file from your data folders, mixes them, runs the
trained model, and writes `test_noisy.wav` / `test_enhanced.wav` so you can
compare by ear. This is the real test -- loss numbers can look fine while
the audio still sounds off, so always listen before trusting the model.

You can also point it at specific files:

```
python enhance_test.py data\speech\some_clip.wav data\noise\some_clip.wav 0
```

(last number is the SNR in dB -- lower = noisier mix, harder test)

## 5. Export for the Raspberry Pi

```
python export_onnx.py
```

Produces `denoise_net.onnx` and checks that its output matches the
PyTorch model closely. Copy this one file to the Pi (over `scp` or a USB
stick) -- the Pi only needs `onnxruntime`, not the full PyTorch install.

## If something breaks

- **"No speech/noise files found"** -- data.py/train.py expect `.wav` or
  `.flac` files directly inside `data\speech` and `data\noise` (not in
  subfolders).
- **"... is Xhz -- resample to 16000Hz first"** -- run the `ffmpeg` command
  from step 1 on that file.
- **Training loss is `nan`** -- almost always a silent or all-zero audio
  file in your data folders; check `check_data.py`'s per-file durations for
  anything that reads 0.0s.
- **Anything else** -- copy the full error message/traceback and the exact
  command you ran; that's enough to diagnose without needing to reproduce
  it locally.
