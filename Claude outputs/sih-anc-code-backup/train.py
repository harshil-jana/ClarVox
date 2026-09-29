"""
Trains DenoiseNet on your speech + noise clips.

Usage:
    python train.py

Expects:
    data/speech/*.wav (or .flac)   -- clean speech clips, 16kHz mono
    data/noise/*.wav  (or .flac)   -- noise clips, 16kHz mono

Produces:
    denoise_net_best.pt   -- weights from the epoch with the lowest validation
                             loss (this is the one export_onnx.py uses)
    denoise_net_final.pt  -- weights from the last epoch (kept as a fallback)
    train_log.csv         -- per-epoch train/val loss

Run `python check_data.py` first if you're not sure your data folders are ready.
"""
import csv
import glob
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import load_mono, make_training_example
from model import DenoiseNet, N_FREQ

# ---------------------------------------------------------------------------
# Config -- tweak these if you have more time / data, otherwise leave as-is.
# ---------------------------------------------------------------------------
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SPEECH_DIR = os.path.join(DATA_DIR, "speech")
NOISE_DIR = os.path.join(DATA_DIR, "noise")

VAL_FRACTION = 0.15        # fraction of *files* (not examples) held out for validation
EXAMPLES_PER_EPOCH = 4000  # synthetic (speech, noise, snr) mixes generated per epoch
BATCH_SIZE = 32
EPOCHS = 12                 # a top-up run needs far fewer epochs than training from scratch
LR = 1e-3                   # used only when training a fresh model from scratch
STEP_SIZE = 8               # StepLR: shrink LR every STEP_SIZE epochs
GAMMA = 0.5                 # ...by this factor
SNR_RANGE_DB = (-5, 15)     # random SNR sampled uniformly from this range per example

# If denoise_net_best.pt already exists, continue training it instead of
# starting over from random weights. This is much faster for "teach it a
# few new noise types" than a full retrain, and uses a gentler learning
# rate so it doesn't undo what it already learned.
FINE_TUNE_FROM_EXISTING = True
FINE_TUNE_LR = 3e-4

AUDIO_EXTS = (".wav", ".flac")


def find_audio_files(folder):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def split_files(files, val_fraction, rng):
    files = list(files)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_fraction)) if len(files) > 1 else 0
    return files[n_val:], files[:n_val]


class MixtureDataset(Dataset):
    """Generates synthetic noisy/clean training pairs on the fly.

    Audio is loaded into RAM once at construction time (clips are short, so
    this is fast and avoids repeated disk reads / multiprocessing headaches).
    len(dataset) is a *virtual* size (n_examples), not the number of files --
    the same files get reused many times with different noise/SNR
    combinations across an epoch, which is what gives a small clip pool a
    lot of effective variety.
    """

    def __init__(self, speech_files, noise_files, n_examples, snr_range, seed):
        assert speech_files, "no speech files given to MixtureDataset"
        assert noise_files, "no noise files given to MixtureDataset"
        self.speech = [load_mono(f) for f in speech_files]
        self.noise = [load_mono(f) for f in noise_files]
        self.n_examples = n_examples
        self.snr_range = snr_range
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        speech = self.speech[self.rng.integers(0, len(self.speech))]
        noise = self.noise[self.rng.integers(0, len(self.noise))]
        snr_db = self.rng.uniform(*self.snr_range)
        context, mask = make_training_example(speech, noise, snr_db)
        return torch.from_numpy(context), torch.from_numpy(mask)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    with torch.set_grad_enabled(train):
        for context, mask in loader:
            context = context.to(device)
            mask = mask.to(device)
            pred = model(context)
            loss = criterion(pred, mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    speech_files = find_audio_files(SPEECH_DIR)
    noise_files = find_audio_files(NOISE_DIR)

    if not speech_files:
        sys.exit(f"No speech files found in {SPEECH_DIR}. Add .wav/.flac clips there first "
                  f"(run check_data.py for guidance).")
    if not noise_files:
        sys.exit(f"No noise files found in {NOISE_DIR}. Add .wav/.flac clips there first "
                  f"(run check_data.py for guidance).")

    print(f"Found {len(speech_files)} speech file(s), {len(noise_files)} noise file(s).")

    train_speech, val_speech = split_files(speech_files, VAL_FRACTION, rng)
    train_noise, val_noise = split_files(noise_files, VAL_FRACTION, rng)
    # If the split leaves either validation list empty (e.g. very few files),
    # fall back to reusing the training files for validation so training can
    # still proceed -- not ideal, but better than crashing right before a
    # deadline.
    if not val_speech:
        val_speech = train_speech
    if not val_noise:
        val_noise = train_noise

    print(f"Train files: {len(train_speech)} speech / {len(train_noise)} noise")
    print(f"Val files:   {len(val_speech)} speech / {len(val_noise)} noise")

    print("Loading audio into memory...")
    t0 = time.time()
    train_ds = MixtureDataset(train_speech, train_noise, EXAMPLES_PER_EPOCH, SNR_RANGE_DB, SEED)
    val_ds = MixtureDataset(val_speech, val_noise, max(500, EXAMPLES_PER_EPOCH // 8), SNR_RANGE_DB, SEED + 1)
    print(f"Done in {time.time() - t0:.1f}s.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    best_path = os.path.join(out_dir, "denoise_net_best.pt")
    final_path = os.path.join(out_dir, "denoise_net_final.pt")
    log_path = os.path.join(out_dir, "train_log.csv")

    model = DenoiseNet(n_freq=N_FREQ).to(device)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    fine_tuning = FINE_TUNE_FROM_EXISTING and os.path.exists(best_path)
    if fine_tuning:
        print(f"Fine-tuning from existing checkpoint: {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))
        # Baseline the current checkpoint's validation loss BEFORE any new
        # training, so a rough early epoch during fine-tuning can't
        # overwrite an already-good model with something worse.
        baseline_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
        best_val_loss = baseline_loss
        print(f"Current checkpoint's validation loss (baseline): {baseline_loss:.5f}")
        lr = FINE_TUNE_LR
    else:
        print("No existing checkpoint found (or FINE_TUNE_FROM_EXISTING=False) -- training from scratch.")
        lr = LR

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        flag = "  <- best so far, saved" if improved else ""
        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({dt:.1f}s){flag}")
        log_rows.append([epoch, train_loss, val_loss, scheduler.get_last_lr()[0], dt])

    torch.save(model.state_dict(), final_path)

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])
        w.writerows(log_rows)

    print()
    print(f"Done. Best val_loss = {best_val_loss:.5f}")
    print(f"  Best-checkpoint weights:  {best_path}   <- use this one for export_onnx.py")
    print(f"  Final-epoch weights:      {final_path}")
    print(f"  Per-epoch log:            {log_path}")


if __name__ == "__main__":
    main()
