# ClarVox / sih-anc — Full Execution Runbook

Team Ignivex · SIH 2026 · PS 26052
Laptop project folder: `D:\Projects\SIH_2026\sih-anc`
Pi login: `harshil_jana@sih-anc.local`

This is the single document to follow end to end. Parts 1–5 are the demo.
Part 6 is only for retraining. Appendices are for when something goes wrong.

---

## PART 0 — The day before (do this on working WiFi, at home)

Skipping this part is what makes demo day go badly. Each step takes a minute.

**0.1 — Confirm the trained model is present on the laptop**

```powershell
cd D:\Projects\SIH_2026\sih-anc
Get-ChildItem denoise_net_best.pt, denoise_net.onnx
```

Both must exist. `denoise_net.onnx` should be roughly 310 KB.

**0.2 — Confirm the Pi is reachable and has the model**

```powershell
ssh harshil_jana@sih-anc.local
```

On the Pi:

```bash
ls -la ~/sih-anc-code/denoise_net.onnx
```

If it is missing or older than the laptop's copy, exit the SSH session and push it:

```powershell
scp .\denoise_net.onnx harshil_jana@sih-anc.local:~/sih-anc-code/
```

**0.3 — Resolve the folder mismatch**

`sih-anc.service` points at `/home/harshil_jana/sih-anc`, but you run the script
manually from `~/sih-anc-code`. On the Pi:

```bash
ls -d ~/sih-anc ~/sih-anc-code 2>/dev/null
```

If both exist, decide which is real and make the service match. This mismatch is
the likeliest reason recorded takes went missing after the internal round —
the service was saving into one folder while you looked in the other.

**0.4 — Pre-save your phone hotspot on the Pi**

Do this while the Pi is still on working WiFi. It is the difference between a
two-minute setup at the venue and a lost demo. See **Appendix A**.

**0.5 — Do one full dry run**

Run Parts 2, 3 and 4 below, start to finish, exactly as you will on the day.

---

## PART 1 — Laptop setup (every new PowerShell window)

**1.1 — Open the project folder**

```powershell
cd D:\Projects\SIH_2026\sih-anc
```

**1.2 — Allow the virtual environment to activate**

PowerShell blocks activation scripts by default. This affects only this window
and changes nothing permanently.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

**1.3 — Activate the environment**

```powershell
.\venv\Scripts\Activate.ps1
```

The prompt must now begin with `(venv)`.

**1.4 — Verify the right Python is active**

```powershell
Get-Command python | Select-Object Source
```

The path must contain `...\sih-anc\venv\Scripts\python.exe`.

**1.5 — Verify the libraries load**

```powershell
python -c "import torch, onnxruntime, sounddevice, soundfile; print('all libraries OK')"
```

Nothing needs installing — the venv already has torch, numpy, scipy, soundfile,
onnx, onnxruntime and sounddevice.

---

## PART 2 — Laptop demo (offline proof + live microphone)

**2.1 — Produce the before/after audio pair**

```powershell
python enhance_test.py
```

Writes `test_noisy.wav` and `test_enhanced.wav` beside the script.

**2.2 — Play them for the jury**

```powershell
start test_noisy.wav
start test_enhanced.wav
```

**2.3 — (Optional) run a harder case on demand**

```powershell
python enhance_test.py data\speech\<clip>.wav data\noise\<clip>.wav 0
```

The last number is SNR in dB — lower is noisier and harder. Default is 5.0.

**2.4 — Start the live microphone demo**

Put headphones on first. Speaker output loops back into the mic and howls.

```powershell
python live_test.py
```

**2.5 — Record a take while it runs**

| Key | Effect |
|---|---|
| ENTER | start recording a take |
| ENTER again | stop and save that take |
| Ctrl+C | end the session (a running take is saved first) |

Each take saves two files in the project folder:
`live_input_take<N>_<timestamp>.wav` and `live_enhanced_take<N>_<timestamp>.wav`.

**2.6 — Put a number on it**

```powershell
python compare_takes.py live_input_take1_<timestamp>.wav live_enhanced_take1_<timestamp>.wav
```

Prints RMS and peak reduction in dB. This is where the 5.2 dB average /
28.8 dB peak figures on slide 3 come from.

---

## PART 3 — Connect to the Raspberry Pi

**3.1 — Power the Pi and wait about 30 seconds**

**3.2 — Check it is on the network**

```powershell
ping sih-anc.local
```

No reply → go to **Appendix A**.

**3.3 — Log in**

```powershell
ssh harshil_jana@sih-anc.local
```

**3.4 — Enter the project and activate the Pi's environment**

Everything from here is bash on the Pi, not PowerShell.

```bash
cd ~/sih-anc-code
source ~/anc-venv/bin/activate
```

**3.5 — Confirm the audio devices are seen**

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

The USB headset must appear with `max_input_channels > 0`. The script picks the
input and output devices automatically — you do not pass any `hw:` arguments.

---

## PART 4 — Pi demo

**4.1 — Start the denoiser**

```bash
python pi_live_test.py
```

**4.2 — Controls (identical to the laptop version)**

| Key | Effect |
|---|---|
| ENTER | start recording a take |
| ENTER again | stop and save that take |
| Ctrl+C | stop the session |

Takes are saved as `pi_input_take<N>_<timestamp>.wav` and
`pi_enhanced_take<N>_<timestamp>.wav` in `~/sih-anc-code`.

**4.3 — Confirm the files actually exist before shutting anything down**

```bash
ls -la ~/sih-anc-code/pi_*.wav
sync
```

Run `sync` before pulling power. Files written but not flushed to the microSD
are lost on an abrupt power-off — this is what happened at the internal round.

**4.4 — Leave the Pi session**

```bash
exit
```

---

## PART 5 — Collect the results on the laptop

**5.1 — Open a second PowerShell window** (leave the Pi session alone)

**5.2 — Pull the recordings across**

Keep the quotes — PowerShell would otherwise try to expand the `*` itself.

```powershell
scp "harshil_jana@sih-anc.local:~/sih-anc-code/pi_*.wav" "D:\Projects\SIH_2026\sih-anc\recordings\"
```

Use the hostname, not a hard-coded IP. `10.76.219.8` was valid on one network
only and changes every time the Pi joins a different one.

**5.3 — Measure a Pi take**

```powershell
cd D:\Projects\SIH_2026\sih-anc
python compare_takes.py recordings\pi_input_take1_<timestamp>.wav recordings\pi_enhanced_take1_<timestamp>.wav
```

---

## PART 6 — Retraining (only when you are adding new noise types)

Not part of the demo. Do all of this on the laptop with the venv active (Part 1).

**6.1 — Record new noise samples (optional)**

```powershell
python record_noise.py gunshot 20
python record_noise.py engine_hum 30
```

Arguments are `<name> [seconds]`, default 15 s. Saves to `data\noise\<name>.wav`.
Make the sound several times, spread out, during the recording.

**6.2 — Check the dataset**

```powershell
python check_data.py
```

Do not continue until it prints `READY? YES`.

**6.3 — Fix sample rates if anything is flagged**

```powershell
python fix_sample_rates.py
python check_data.py
```

Resamples everything in `data\speech` and `data\noise` to 16 kHz mono in place,
backing the originals up into a `_originals` subfolder. No ffmpeg required.

**6.4 — Sanity-check the network builds**

```powershell
python model.py
```

Instant. Prints the parameter count and `Self-test passed.`

**6.5 — Decide: continue training, or start fresh**

`train.py` has `FINE_TUNE_FROM_EXISTING = True`. If `denoise_net_best.pt` exists
it **continues** from it at a gentler learning rate (3e-4) instead of starting
over. That is what you want for "teach it a few new noise types".

For a genuinely fresh model, rename the existing checkpoint first:

```powershell
Rename-Item denoise_net_best.pt denoise_net_best_backup.pt
```

**6.6 — Train**

```powershell
python train.py
```

12 epochs by default. One line per epoch: train loss, validation loss, learning
rate, time. `val_loss` must trend downward. Flat from epoch 1 means something is
wrong upstream, not that it needs more epochs.

Outputs:

| File | What it is |
|---|---|
| `denoise_net_best.pt` | lowest validation loss — the one that matters |
| `denoise_net_final.pt` | last epoch, fallback only |
| `train_log.csv` | loss per epoch |

**6.7 — Listen before trusting it**

```powershell
python enhance_test.py
start test_noisy.wav
start test_enhanced.wav
```

Loss numbers can look fine while the audio sounds wrong. Always listen.

**6.8 — Export for the Pi**

```powershell
python export_onnx.py
```

Writes `denoise_net.onnx` and verifies its output matches the PyTorch model.

**6.9 — Push it to the Pi and re-test**

```powershell
scp .\denoise_net.onnx harshil_jana@sih-anc.local:~/sih-anc-code/
```

Then repeat Parts 3 and 4.

> If you change `MASK_GAMMA`, `MASK_FLOOR` or the AGC settings, change them in
> `live_test.py` **and** `pi_live_test.py`. They are currently in sync at
> `MASK_GAMMA = 2.0`, `MASK_FLOOR = 0.08`, `AGC_MAX_GAIN = 1.0`.

---

## APPENDIX A — Changing the Pi's network

**A.1 — Find out which network stack the Pi uses**

On the Pi:

```bash
systemctl is-active NetworkManager
```

- `active` → Raspberry Pi OS Bookworm. Use `nmcli` (A.2).
- anything else → Bullseye or older. Use `raspi-config` (A.3).

**A.2 — Bookworm: join a new network over SSH**

```bash
sudo nmcli device wifi list
sudo nmcli device wifi connect "SSID" password "thepassword"
nmcli connection show
```

**A.3 — Bullseye or older**

```bash
sudo raspi-config
```

System Options → Wireless LAN → enter SSID and password.

**A.4 — Pre-save the phone hotspot (do this before demo day)**

Set your phone's hotspot name and password to something fixed, then, on the Pi,
while it is still on working WiFi:

```bash
sudo nmcli connection add type wifi con-name "phone-hotspot" ifname wlan0 ssid "HOTSPOT_NAME"
sudo nmcli connection modify "phone-hotspot" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "thepassword"
sudo nmcli connection modify "phone-hotspot" connection.autoconnect-priority 20
```

Verify it saved:

```bash
nmcli connection show
```

At the venue: turn the hotspot on before powering the Pi. It joins by itself.

**A.5 — When you cannot reach the Pi at all**

Try in this order:

1. `ping sih-anc.local` from the laptop.
2. Turn the phone hotspot on, power-cycle the Pi, wait 60 s, ping again.
3. Plug in a monitor and keyboard, log in locally, run `hostname -I` for the IP,
   then `ssh harshil_jana@<that-ip>`.
4. Connect the Pi to the laptop or a router by Ethernet cable and try
   `sih-anc.local` again.

Editing the Pi's config offline from the laptop is **not** an option — Windows
cannot read the microSD card's Linux filesystem, and Bookworm ignores the old
`wpa_supplicant.conf` on the boot partition. Steps 2 and 3 are the real answers.

---

## APPENDIX B — Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Activate.ps1 cannot be loaded` | You skipped step 1.2. |
| `python` is not the venv one | Activation did not take. Re-run 1.2 and 1.3, check with 1.4. |
| Howling / feedback in `live_test.py` | You are not wearing headphones. |
| `Could not open audio stream` | Run `python -c "import sounddevice as sd; print(sd.query_devices())"` and check the default mic and output in Windows sound settings. |
| Pi sees no microphone | Re-seat the USB headset, then re-run the `query_devices` check in 3.5. |
| `denoise_net_best.pt not found` | Run `train.py` first (Part 6). |
| `No speech/noise files found` | Files must sit directly in `data\speech` and `data\noise`, not in subfolders. |
| Training loss is `nan` | Almost always a silent or all-zero audio file. Look for a 0.0 s duration in `check_data.py` output. |
| Recorded takes missing after the demo | Power was cut before the write flushed. Always run `sync` (step 4.3) before pulling power. |
| `scp` copies nothing | The `*` was unquoted, or the remote folder is `~/sih-anc` rather than `~/sih-anc-code`. |

---

## APPENDIX C — Command quick reference

**Laptop (PowerShell, venv active)**

| Goal | Command |
|---|---|
| Activate env | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` then `.\venv\Scripts\Activate.ps1` |
| Offline A/B | `python enhance_test.py [speech.wav noise.wav snr_db]` |
| Live mic demo | `python live_test.py` |
| Measure a take | `python compare_takes.py <input.wav> <enhanced.wav>` |
| Check dataset | `python check_data.py` |
| Fix sample rates | `python fix_sample_rates.py` |
| Record noise | `python record_noise.py <name> [seconds]` |
| Model self-test | `python model.py` |
| Train | `python train.py` |
| Export ONNX | `python export_onnx.py` |
| Push model to Pi | `scp .\denoise_net.onnx harshil_jana@sih-anc.local:~/sih-anc-code/` |
| Pull recordings | `scp "harshil_jana@sih-anc.local:~/sih-anc-code/pi_*.wav" "D:\Projects\SIH_2026\sih-anc\recordings\"` |
| Leave env | `deactivate` |

**Raspberry Pi (bash)**

| Goal | Command |
|---|---|
| Log in | `ssh harshil_jana@sih-anc.local` |
| Enter project | `cd ~/sih-anc-code` |
| Activate env | `source ~/anc-venv/bin/activate` |
| List audio devices | `python -c "import sounddevice as sd; print(sd.query_devices())"` |
| Run denoiser | `python pi_live_test.py` |
| Flush to card | `sync` |
| Show IP | `hostname -I` |
| Join a network | `sudo nmcli device wifi connect "SSID" password "pw"` |
| List saved networks | `nmcli connection show` |
| Log out | `exit` |
