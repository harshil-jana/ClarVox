"""
Exports the trained PyTorch model to ONNX so it can run on the Raspberry Pi
via onnxruntime, without needing the full PyTorch install there.

Run this AFTER train.py has produced denoise_net_best.pt.

Usage:
    python export_onnx.py
"""
import os

import numpy as np
import torch

from model import DenoiseNet, N_FREQ

CONTEXT = 8  # must match data.py's CONTEXT


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(out_dir, "denoise_net_best.pt")
    onnx_path = os.path.join(out_dir, "denoise_net.onnx")

    if not os.path.exists(weights_path):
        raise SystemExit(
            f"{weights_path} not found -- run train.py first (it saves this file "
            f"automatically as soon as the first epoch completes)."
        )

    model = DenoiseNet(n_freq=N_FREQ)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.zeros(1, CONTEXT, N_FREQ)  # (batch=1, CONTEXT, n_freq)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["context"],
        output_names=["mask"],
        opset_version=17,
        dynamic_axes=None,  # fixed shape is fine -- the Pi script always feeds (1, CONTEXT, N_FREQ)
        dynamo=False,  # force the legacy TorchScript-based exporter -- newer torch defaults to
                        # a dynamo/onnxscript-based exporter that requires an extra package
                        # (onnxscript) not in requirements.txt; this simple GRU+Linear model
                        # has no dynamic control flow, so the legacy exporter is a perfect fit
                        # and avoids the extra dependency entirely.
    )

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"Exported {onnx_path} ({size_kb:.1f} KB)")
    print("Sanity-checking the export with onnxruntime...")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_input = np.random.rand(1, CONTEXT, N_FREQ).astype(np.float32)
        onnx_out = sess.run(None, {"context": test_input})[0]

        with torch.no_grad():
            torch_out = model(torch.from_numpy(test_input)).numpy()

        max_diff = float(np.max(np.abs(onnx_out - torch_out)))
        print(f"Max difference between PyTorch and ONNX output: {max_diff:.2e}")
        if max_diff < 1e-4:
            print("Looks good -- ONNX export matches PyTorch closely.")
        else:
            print("WARNING: outputs differ more than expected -- double-check before relying on this.")
    except ImportError:
        print("(onnxruntime not installed, skipping the match-check -- "
              "`pip install onnxruntime` to enable it. The .onnx file was still created.)")


if __name__ == "__main__":
    main()
