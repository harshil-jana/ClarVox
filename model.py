"""
The neural network itself.

Small on purpose: this has to run in real time on a Raspberry Pi with no
GPU. A single GRU layer + one linear layer is roughly the same size class
as RNNoise's core network -- a few hundred thousand parameters, not
millions.

Input:  (batch, CONTEXT, n_freq)  -- a short rolling window of magnitude
        spectrogram frames (the "context" data.py builds).
Output: (batch, n_freq)           -- a mask in [0, 1] for the newest frame.
        Multiply the noisy magnitude spectrum by this mask and you get an
        estimate of the clean magnitude spectrum.
"""
import torch
import torch.nn as nn

N_FREQ = 257  # N_FFT // 2 + 1, with N_FFT=512 (must match data.py)
HIDDEN = 64


class DenoiseNet(nn.Module):
    def __init__(self, n_freq=N_FREQ, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size=n_freq, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_freq)

    def forward(self, x):
        # x: (batch, CONTEXT, n_freq)
        out, _ = self.gru(x)          # out: (batch, CONTEXT, hidden)
        last = out[:, -1, :]          # take the last timestep -> (batch, hidden)
        mask = torch.sigmoid(self.fc(last))  # (batch, n_freq), values in (0, 1)
        return mask


if __name__ == "__main__":
    # Quick self-test: run this file directly (`python model.py`) to confirm
    # the model builds and shapes flow correctly, with no dataset needed.
    m = DenoiseNet()
    dummy = torch.zeros(4, 8, N_FREQ)  # batch=4, CONTEXT=8, n_freq=257
    out = m(dummy)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"DenoiseNet OK. Output shape: {tuple(out.shape)} (expected (4, {N_FREQ}))")
    print(f"Total parameters: {n_params:,}")
    assert out.shape == (4, N_FREQ), "shape mismatch -- check N_FREQ / CONTEXT"
    print("Self-test passed.")
