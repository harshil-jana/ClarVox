# Raspberry Pi — clean reinstall checklist

Verified against `D:\Projects\SIH_2026\sih-anc` on 2026-09-29.

---

## 1. Do you have everything? Yes.

`pi_live_test.py` is fully self-contained — it imports only the standard
library plus `numpy`, `onnxruntime` and `sounddevice`. It does **not** import
`data.py` or `model.py`, and it does not need PyTorch. Everything the Pi runs
is on the laptop.

### Required on the Pi — only two files

| File | Size | On laptop? |
|---|---|---|
| `pi_live_test.py` | 12,251 B | yes |
| `denoise_net.onnx` | 317,104 B | yes |

### Useful on the Pi

| File | Purpose | On laptop? |
|---|---|---|
| `pi_model_smoke_test.py` | proves model + onnxruntime work before any audio hardware is attached | yes |
| `pi_output_test.py` | proves the 3.5 mm output works | yes |
| `sih-anc.service` | autostart on boot | yes (needs a path fix — see §5) |

### Deliberately NOT copied to the Pi

`train.py`, `model.py`, `data.py`, `export_onnx.py`, `live_test.py`,
`enhance_test.py`, `check_data.py`, `fix_sample_rates.py`, `record_noise.py`,
`compare_takes.py`, `requirements.txt`, the whole `data\` folder, and the
`.pt` checkpoints. These are laptop-only — the Pi needs none of them, and
`requirements.txt` lists torch, which must not be installed on the Pi.

### Backed up already

`recordings\` holds 7 take pairs from 7–8 Sep. One pair
(`*_20260908_114328.wav`) is 0 bytes — that is the take lost to the abrupt
power-off, not a transfer error.

### The only real risk

Anything that exists **only** on the Pi's SD card and was never copied back:

- Pi takes recorded after 8 Sep that were never `scp`-ed to `recordings\`.
- Any edit you made to `pi_live_test.py` directly on the Pi (device indices,
  `MASK_GAMMA`, AGC settings) that was never copied back to the laptop.
- The Pi's venv and whatever you installed into it — easy to rebuild.

There is also a GitHub remote configured:
`https://github.com/harshil-jana/SIH-2026.git`. Before wiping anything, check
it is current:

```powershell
cd D:\Projects\SIH_2026\sih-anc
git status
git log origin/main -1
```

Note `.gitignore` excludes `data/`, `venv/` and `*.wav`, so the dataset and
recordings are **not** on GitHub. The `.onnx` and `.pt` files are not ignored,
so they should be — confirm with `git log --stat`.

---

## 2. Before you wipe: don't wipe

**Use a different microSD card.** They cost very little and this removes all
risk. Keep the current card untouched in case something on it turns out to
matter.

If you must reuse the same card, image it first. On Windows, Win32DiskImager
("Read" to a `.img` file) or Rufus can do this. A Windows laptop cannot read
the Pi's Linux root partition directly, so a full card image is the only way
to get anything off it later.

---

## 3. Flash the card with Raspberry Pi Imager

### 3.0 — Prepare the phone hotspot first

Do this before flashing, because the SSID and password get baked into the
image and a later change means reflashing.

- Rename the hotspot to something plain: `sihdemo`. No spaces, no apostrophes,
  no emoji — an apostrophe in a name like `Harshil's iPhone` breaks the
  generated config file.
- Password: something simple and typo-proof, e.g. `sihdemo2026`.
- Band: **2.4 GHz** (iPhone: turn on *Maximize Compatibility*).
- Security: **WPA2-Personal**, not WPA3.
- Turn off "switch off hotspot automatically when no devices are connected".
- Write the exact SSID and password down. A single wrong character means the
  Pi boots and silently never joins, with no way in to check.

### 3.1 — Get the card into the laptop

Remove the microSD from the Pi and put it in your laptop's card slot or a USB
reader.

Windows will almost certainly pop up **"You need to format the disk before you
can use it"** — possibly two or three times, one per partition. **Click Cancel
every time.** That dialog appears because Windows cannot read the Pi's Linux
partition; it does not mean the card is broken. Formatting through that dialog
is not needed and can leave the card in a state Imager has to clean up anyway.

### 3.2 — Install Raspberry Pi Imager

Download from `raspberrypi.com/software` and install it. Use the current
version — older ones lack the customisation screen that fixes the WiFi.

### 3.3 — (Only if the card misbehaves) erase it first

Skip this normally; writing an image replaces the partition table anyway. If
Imager fails or the card has a strange partition layout:

Imager → **Choose OS** → scroll to the bottom → **Erase (Format card as FAT32)**
→ Choose Storage → Write.

Do not use Windows `diskpart` for this. It is easy to clean the wrong disk.

### 3.4 — Choose Device

**Choose Device** → **Raspberry Pi 4**. This filters the OS list to images
that actually boot on your board.

### 3.5 — Choose OS

**Choose OS** → **Raspberry Pi OS (other)** → **Raspberry Pi OS Lite (64-bit)**.

Two things matter here:

- **64-bit is mandatory.** `pip install onnxruntime` has no wheel for 32-bit
  Raspberry Pi OS and the install will fail. This is the most common way a
  rebuild goes wrong.
- **Lite** has no desktop, which means less CPU competing with the audio
  callback. Pick the full desktop version instead only if you want a screen to
  fall back on when the network misbehaves.

### 3.6 — Choose Storage

**Choose Storage** → select the card.

Check the reported capacity matches your card before continuing. This step
erases whatever you select, permanently, with no undo. If two removable drives
are listed and you are not certain which is which, unplug the other one.

### 3.7 — Edit Settings (do not skip this)

Click **Next**. When asked *"Would you like to apply OS customisation
settings?"*, click **Edit Settings**.

**General tab:**

| Field | Value |
|---|---|
| Set hostname | `sih-anc` — keeps `sih-anc.local` working in every note you have |
| Set username | `harshil_jana` — keeps every path in your notes and the service file valid |
| Password | your choice, written down |
| Configure wireless LAN — SSID | `sihdemo` (whatever you set in 3.0) |
| Configure wireless LAN — Password | as set in 3.0 |
| **Wireless LAN country** | **IN** |
| Locale — time zone | `Asia/Kolkata` |
| Locale — keyboard layout | `us` |

**Wireless LAN country = IN is the fix for your current problem.** With it
blank the radio is regulatory-blocked, which is exactly why networks never
appeared and the Pi never showed up in the phone's device list.

**Services tab:**

- Tick **Enable SSH** → **Use password authentication**.

**Options tab:** leave the defaults. "Eject media when finished" is fine.

Click **Save**, then **Yes** to apply the settings.

### 3.8 — Write

Confirm the "all existing data will be erased" warning. Imager writes, then
verifies — expect roughly 5–15 minutes depending on the card. Do not unplug
during the verify stage.

### 3.9 — Sanity check before ejecting

After writing, Windows should show a small FAT32 drive named `bootfs`
(or `boot` on older images). If it opens and you can see `config.txt`, the
write succeeded. Cancel any format prompt for the other partition as before.

Eject safely, put the card in the Pi.

### 3.10 — First boot

- Turn the hotspot **on first** and leave it on.
- Use the proper 5 V / 3 A USB-C supply, not a laptop port or phone charger.
- Power the Pi and wait 90 seconds. It expands the filesystem and reboots once
  on its own — that reboot is normal, don't interrupt it.
- Watch your phone's connected-devices list. `sih-anc` appearing there is your
  confirmation before you even try SSH.

---

## 4. First boot and setup

```powershell
ping sih-anc.local
ssh harshil_jana@sih-anc.local
```

On the Pi:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv libportaudio2

mkdir -p ~/sih-anc-code
python3 -m venv ~/anc-venv
source ~/anc-venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy sounddevice
```

`libportaudio2` is the system library `sounddevice` binds to. Without it the
import fails with an obscure error.

Verify:

```bash
python -c "import onnxruntime, numpy, sounddevice; print('pi libs OK')"
```

---

## 5. Copy the project files across

From PowerShell on the laptop:

```powershell
cd D:\Projects\SIH_2026\sih-anc
scp .\pi_live_test.py .\denoise_net.onnx .\pi_model_smoke_test.py .\pi_output_test.py harshil_jana@sih-anc.local:~/sih-anc-code/
```

---

## 6. Test in order — do not skip ahead

**6.1 Model only, no audio hardware:**

```bash
cd ~/sih-anc-code
source ~/anc-venv/bin/activate
python pi_model_smoke_test.py
```

**6.2 Plug in the USB headset, then check both devices are seen:**

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

The USB headset must show `max_input_channels > 0`.

**6.3 Output path:**

```bash
python pi_output_test.py
```

**6.4 The real thing:**

```bash
python pi_live_test.py
```

ENTER starts a take, ENTER again saves it, Ctrl+C stops. Run `sync` afterwards
before cutting power.

---

## 7. Autostart on boot — do this, it ends the WiFi dependency

Once the Pi starts the denoiser on power-up, you never need SSH or a network
for the demo itself.

The current `sih-anc.service` points at `/home/harshil_jana/sih-anc`, but every
one of your working notes uses `~/sih-anc-code`. Standardise on
`sih-anc-code`. Corrected file:

```ini
[Unit]
Description=SIH Adaptive Noise Cancellation - real-time denoiser
After=sound.target

[Service]
Type=simple
User=harshil_jana
WorkingDirectory=/home/harshil_jana/sih-anc-code
ExecStart=/home/harshil_jana/anc-venv/bin/python /home/harshil_jana/sih-anc-code/pi_live_test.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Install it:

```bash
sudo cp ~/sih-anc-code/sih-anc.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sih-anc.service
sudo systemctl start sih-anc.service
systemctl status sih-anc.service
```

Then reboot and confirm audio flows with nothing but power and the headset:

```bash
sudo reboot
```

Note: under systemd there is no terminal, so the ENTER-to-record feature does
nothing. That is fine for a live demo. Stop the service and run the script by
hand when you want to capture takes:

```bash
sudo systemctl stop sih-anc.service
```

---

## 8. Pre-save a second network while everything works

Add your college WiFi (or vice versa) so the Pi has two ways onto a network:

```bash
sudo nmcli device wifi list
sudo nmcli connection add type wifi con-name "phone-hotspot" ifname wlan0 ssid "HOTSPOT_NAME"
sudo nmcli connection modify "phone-hotspot" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "thepassword"
sudo nmcli connection modify "phone-hotspot" connection.autoconnect-priority 20
nmcli connection show
```

---

## 9. Final check before you pack up

- [ ] `pi_live_test.py` runs by hand and audio is audibly cleaner
- [ ] A take records and both WAVs exist after `sync`
- [ ] `scp` pulls a take back to `recordings\` on the laptop
- [ ] Reboot with power only — denoiser starts on its own
- [ ] Both networks saved, `nmcli connection show` lists them
- [ ] The old SD card is kept, or imaged
