"""
No-audio smoke test for the Raspberry Pi: confirms the exported ONNX model
loads and runs correctly on this machine's actual processor, WITHOUT needing
any microphone or speaker connected.

Run this on the Pi after copying denoise_net.onnx over, before any audio
hardware arrives. If this passes, you know the model + onnxruntime + Pi
combination works -- the only thing left to test once you have a mic/headset
is the audio plumbing itself (pi_live_test.py).

Usage:
    source ~/anc-venv/bin/activate
    python pi_model_smoke_test.py
"""
import os
import sys
import time

import numpy as np

CONTEXT = 8
N_FREQ = 257  # N_FFT // 2 + 1, with N_FFT=512 -- must match data.py / model.py


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(here, "denoise_net.onnx")

    if not os.path.exists(onnx_path):
        sys.exit(f"{onnx_path} not found -- copy it over from the laptop first "
                  f"(scp denoise_net.onnx <this Pi>:~/sih-anc/).")

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("onnxruntime is not installed. Run: pip install onnxruntime")

    print(f"Loading {onnx_path} ...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"Input:  name={input_name!r}  shape={input_shape}")
    print(f"Output: name={output_name!r}")

    # Fake input, same shape real audio would produce: (batch=1, CONTEXT, N_FREQ)
    dummy = np.random.rand(1, CONTEXT, N_FREQ).astype(np.float32)

    print("\nRunning 20 inference passes to check correctness and timing...")
    times = []
    for i in range(20):
        t0 = time.time()
        out = sess.run([output_name], {input_name: dummy})[0]
        times.append(time.time() - t0)

    out_shape_ok = out.shape == (1, N_FREQ)
    out_range_ok = bool(np.all(out >= 0.0) and np.all(out <= 1.0))
    avg_ms = 1000 * sum(times) / len(times)
    max_ms = 1000 * max(times)

    print(f"\nOutput shape: {out.shape}  (expected (1, {N_FREQ}))  {'OK' if out_shape_ok else 'MISMATCH'}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]  (expected within [0, 1])  "
          f"{'OK' if out_range_ok else 'OUT OF RANGE'}")
    print(f"Inference time: avg {avg_ms:.2f}ms, max {max_ms:.2f}ms per frame")
    print(f"Real-time budget per frame: {256/16000*1000:.1f}ms (HOP=256 samples @ 16kHz)")

    if avg_ms > 256 / 16000 * 1000:
        print("\nWARNING: average inference time exceeds the real-time budget -- "
              "the model may not keep up with live audio on this Pi. Worth flagging "
              "before the demo, not after.")
    elif not (out_shape_ok and out_range_ok):
        print("\nSomething is wrong with the model output -- do not proceed to the "
              "live audio test until this is fixed.")
    else:
        print("\nAll checks passed. The model runs correctly and fast enough on this "
              "Pi. Once you have a mic/headset, pi_live_test.py should just work.")


if __name__ == "__main__":
    main()
