from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SAE(nn.Module):
    """
    Simple linear SAE:
      - Encoder: x @ W_e + b_e, then ReLU
      - Decoder: h @ W_d + b_d
    Reconstruction loss: MSE(x, x_hat)
    Sparsity: L1 penalty on hidden activations (post-ReLU)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.enc = nn.Linear(input_dim, hidden_dim, bias=True)
        self.dec = nn.Linear(hidden_dim, input_dim, bias=True)
        self.act = nn.ReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.enc(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


def _set_torch_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    # CPU only pipeline; keep defaults for performance/determinism tradeoff.


def train_sae(
    acts: np.ndarray,
    hidden_dim: int,
    top_k: int,
    l1_lambda: float,
    seed: int,
    epochs: int,
    lr: float,
    device: str = "cpu",
) -> Tuple[SAE, Dict]:
    """
    Train SAE on activations with MSE + L1 on hidden activations.

    Args:
        acts: np.ndarray [N, D]
        hidden_dim: H
        top_k: not used in training (only in encoding); included for completeness/logging
        l1_lambda: coefficient for L1 penalty on hidden activations
        seed: RNG seed
        epochs: training epochs
        lr: learning rate
        device: 'cpu' (default)

    Returns:
        model: trained SAE
        stats: dict with loss curves
    """
    assert acts.ndim == 2, "acts must be 2D [N, D]"
    _set_torch_determinism(seed)
    torch_device = torch.device(device)

    X = torch.from_numpy(acts.astype(np.float32))
    N, D = X.shape

    model = SAE(input_dim=D, hidden_dim=hidden_dim).to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="mean")

    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=min(64, max(8, N // 16)), shuffle=True, drop_last=False)

    recon_curve = []
    l1_curve = []
    total_curve = []

    model.train()
    for _ in range(epochs):
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_total = 0.0
        count = 0

        for (xb,) in loader:
            xb = xb.to(torch_device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            x_hat, h = model(xb)
            recon = mse(x_hat, xb)
            l1 = h.abs().mean()
            loss = recon + l1_lambda * l1
            loss.backward()
            opt.step()

            bsz = xb.size(0)
            epoch_recon += recon.item() * bsz
            epoch_l1 += l1.item() * bsz
            epoch_total += loss.item() * bsz
            count += bsz

        recon_curve.append(epoch_recon / max(1, count))
        l1_curve.append(epoch_l1 / max(1, count))
        total_curve.append(epoch_total / max(1, count))

    stats = {
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "l1_lambda": l1_lambda,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "recon_curve": recon_curve,
        "l1_curve": l1_curve,
        "total_curve": total_curve,
    }
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, stats


@torch.no_grad()
def encode_topk(model: SAE, acts: np.ndarray, top_k: int, device: str = "cpu") -> np.ndarray:
    """
    Encode activations via SAE encoder and keep top-k per sample (zero out the rest).

    Args:
        model: trained SAE
        acts: np.ndarray [N, D]
        top_k: number of features to keep per sample (if >= H, keep all)
        device: 'cpu' (default)

    Returns:
        features: np.ndarray [N, H] sparse activations after top-k masking
    """
    X = torch.from_numpy(acts.astype(np.float32)).to(device)
    H = model.enc.out_features
    h = model.encode(X)  # [N, H]

    if top_k is None or top_k >= H:
        return h.cpu().numpy()

    k = max(0, int(top_k))
    if k == 0:
        return torch.zeros_like(h, device="cpu").numpy()

    # Vectorized top-k mask
    vals, idx = torch.topk(h, k=k, dim=1)
    mask = torch.zeros_like(h, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    out = torch.where(mask, h, torch.zeros_like(h))
    return out.cpu().numpy()