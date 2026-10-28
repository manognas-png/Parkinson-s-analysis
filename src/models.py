"""Small neural-network models for gait strides, 8-bit quantization (simulated),
and microcontroller cost estimates."""

import copy

import numpy as np
import torch
from torch import nn


class GaitAutoencoder(nn.Module):
    """1D convolutional autoencoder for one stride (channels x 100 points).

    Encoder: 100 -> 50 -> 25 time steps; bottleneck of `latent` channels x 25
    (with latent=4: 100 numbers for a 600-number stride, 6x compression).
    The decoder mirrors it with transposed convolutions back to 100 steps.
    """

    def __init__(self, n_channels=6, hidden=16, latent=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, hidden, 5, stride=2, padding=2), nn.ReLU(),  # 100 -> 50
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.ReLU(),  # 50 -> 25
            nn.Conv1d(hidden, latent, 5, padding=2),  # 25 -> 25, bottleneck
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent, hidden, 4, stride=2, padding=1), nn.ReLU(),  # 25 -> 50
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),  # 50 -> 100
            nn.Conv1d(hidden, n_channels, 5, padding=2),  # 100 -> 100
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train, X_val, seed, max_epochs=200, patience=20, batch_size=128, lr=1e-3,
                      device="cpu"):
    """Fit on X_train (strides, channels, points) with mean squared error; early-stop on X_val.

    `device` can be "cpu" or "cuda" (a GPU); the returned model is always on the CPU.
    Returns the model at the epoch with the lowest validation loss, that epoch, and the loss.
    """
    torch.manual_seed(seed)
    model = GaitAutoencoder(n_channels=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_train = torch.from_numpy(X_train).float().to(device)
    x_val = torch.from_numpy(X_val).float().to(device)
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_epoch, best_state, waited = np.inf, -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(device)
        for i in range(0, len(x_train), batch_size):
            batch = x_train[order[i:i + batch_size]]
            optimizer.zero_grad()
            loss = ((model(batch) - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = ((model(x_val) - x_val) ** 2).mean().item()
        if val_loss < best_loss:
            best_loss, best_epoch, best_state, waited = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model, best_epoch, best_loss


def squared_errors(model, X):
    """Per-element squared reconstruction error, shape like X (strides, channels, points)."""
    with torch.no_grad():
        x = torch.from_numpy(X).float()
        return ((model(x) - x) ** 2).numpy()


# ---- Simulated 8-bit quantization ----


def quantize_weights(w, bits=8):
    """Symmetric per-tensor quantization: w -> round(w / s) * s, with |integers| <= 127."""
    levels = 2 ** (bits - 1) - 1
    scale = w.abs().max() / levels
    return w if scale == 0 else torch.round(w / scale).clamp(-levels, levels) * scale


def fake_quantize(x, low, high, bits=8):
    """Asymmetric quantization of activations to 2^bits levels between low and high."""
    scale = (high - low) / (2 ** bits - 1)
    if scale == 0:
        return x
    return torch.clamp(torch.round((x - low) / scale), 0, 2 ** bits - 1) * scale + low


class FakeQuantAutoencoder(nn.Module):
    """The trained model run as if in int8: 8-bit weights and 8-bit activations.

    Activation ranges come from calibration data (training controls). Biases stay in
    float here; on a device they are int32, which adds negligible error. This simulates
    int8 inference accuracy without needing a device or a quantization toolchain.
    """

    def __init__(self, model, calibration):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(list(model.encoder) + list(model.decoder)))
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, "weight"):
                    layer.weight.copy_(quantize_weights(layer.weight))
            x = torch.from_numpy(calibration).float()
            self.ranges = [(x.min().item(), x.max().item())]
            for layer in self.layers:
                x = layer(x)
                self.ranges.append((x.min().item(), x.max().item()))
        self.eval()

    def forward(self, x):
        x = fake_quantize(x, *self.ranges[0])
        for layer, value_range in zip(self.layers, self.ranges[1:]):
            x = fake_quantize(layer(x), *value_range)
        return x


# ---- Microcontroller cost estimates ----


def cost_report(model, n_channels=6, n_points=100, raw_samples_per_stride=200):
    """Parameters, multiply-accumulates (MACs) and memory for one stride.

    RAM counts the largest input + output activation pair of any layer (ReLU runs
    in place) plus a raw buffer of `raw_samples_per_stride` int16 readings per
    sensor (2 s at 100 Hz) for resampling.
    """
    layers = list(model.encoder) + list(model.decoder)
    length, channels = n_points, n_channels
    macs, weights, biases, peak_pair = 0, 0, 0, 0
    for layer in layers:
        if isinstance(layer, nn.Conv1d):
            k, s, p = layer.kernel_size[0], layer.stride[0], layer.padding[0]
            out_length = (length + 2 * p - k) // s + 1
            macs += layer.out_channels * out_length * layer.in_channels * k
        elif isinstance(layer, nn.ConvTranspose1d):
            k, s, p = layer.kernel_size[0], layer.stride[0], layer.padding[0]
            out_length = (length - 1) * s - 2 * p + k
            macs += length * layer.in_channels * layer.out_channels * k
        else:
            continue
        weights += layer.weight.numel()
        biases += layer.bias.numel()
        peak_pair = max(peak_pair, channels * length + layer.out_channels * out_length)
        length, channels = out_length, layer.out_channels

    raw_buffer = n_channels * raw_samples_per_stride * 2
    n_layers = sum(isinstance(l, (nn.Conv1d, nn.ConvTranspose1d)) for l in layers)
    return {
        "parameters": weights + biases,
        "macs_per_stride": macs,
        "flash_bytes_float32": 4 * (weights + biases),
        "flash_bytes_int8": weights + 4 * biases + 8 * (n_layers + 1),  # int8 weights, int32 biases, scales
        "ram_bytes_float32": 4 * peak_pair + raw_buffer,
        "ram_bytes_int8": peak_pair + raw_buffer,
    }
