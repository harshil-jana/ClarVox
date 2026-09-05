"""
Quick readiness check for your data/speech and data/noise folders.
Run this before train.py:

    python check_data.py
"""
import glob
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SPEECH_DIR = os.path.join(DATA_DIR, "speech")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
AUDIO_EXTS = (".wav", ".flac")

TARGET_SR = 16000


def find_audio_files(folder):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def check_folder(name, folder):
    if not os.path.isdir(folder):
        print(f"[MISSING] {name} folder does not exist: {folder}")
        return []
    files = find_audio_files(folder)
    print(f"[{name}] {folder}")
    print(f"  {len(files)} audio file(s) found (.wav / .flac)")
    return files


def main():
    print("Checking data folders...\n")
    speech_files = check_folder("speech", SPEECH_DIR)
    print()
    noise_files = check_folder("noise", NOISE_DIR)
    print()

    try:
        import soundfile as sf
    except ImportError:
        print("soundfile is not installed yet -- run: pip install -r requirements.txt")
        print("(skipping sample-rate checks until then)\n")
        sf = None

    problems = []
    if sf is not None:
        print("Checking sample rates (first 5 files per folder)...")
        for label, files in (("speech", speech_files), ("noise", noise_files)):
            for f in files[:5]:
                try:
                    info = sf.info(f)
                    flag = "" if info.samplerate == TARGET_SR else "  <-- NOT 16kHz, resample this!"
                    print(f"  [{label}] {os.path.basename(f)}: {info.samplerate}Hz, "
                          f"{info.channels}ch, {info.duration:.1f}s{flag}")
                    if info.samplerate != TARGET_SR:
                        problems.append(f)
                except Exception as e:
                    print(f"  [{label}] {os.path.basename(f)}: could not read ({e})")
                    problems.append(f)
        print()

    if not speech_files:
        print("READY? NO -- add speech clips to data/speech (e.g. LibriSpeech dev-clean, CMU ARCTIC).")
    elif not noise_files:
        print("READY? NO -- add noise clips to data/noise (e.g. ESC-50).")
    elif problems:
        print("READY? ALMOST -- some files aren't 16kHz mono. Fix all of them with:")
        print("  python fix_sample_rates.py")
        print("(no ffmpeg needed -- it uses soundfile + scipy, already in requirements.txt)")
    else:
        print(f"READY? YES -- {len(speech_files)} speech + {len(noise_files)} noise files, "
              f"looking good. Run: python train.py")


if __name__ == "__main__":
    main()
