"""Stage 1: Flood segmentation head training scaffold.

This script provides a runnable training path with real optimization and metric
computation, and is intended to be swapped to true FloodNet dataloaders.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import mIoU


class FloodHead(nn.Module):
    def __init__(self, in_ch: int = 4, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_miou(logits: torch.Tensor, labels: torch.Tensor, n_classes: int = 10) -> float:
    preds = logits.argmax(dim=1)
    per_class = []
    for c in range(n_classes):
        pred_c = preds == c
        true_c = labels == c
        tp = (pred_c & true_c).sum().item()
        fp = (pred_c & ~true_c).sum().item()
        fn = (~pred_c & true_c).sum().item()
        per_class.append((tp, fp, fn))
    return mIoU(per_class)


def main() -> None:
    torch.manual_seed(42)

    n_total = 256
    n_test = 64
    n_train = n_total - n_test
    h, w = 64, 64

    # Synthetic scaffold tensors; replace with FloodNet dataloader in production.
    x = torch.randn(n_total, 4, h, w)
    y = torch.randint(0, 10, (n_total, h, w))

    train_ds = TensorDataset(x[:n_train], y[:n_train])
    test_ds = TensorDataset(x[n_train:], y[n_train:])
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = FloodHead()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3
    for epoch in range(epochs):
        model.train()
        losses = []
        for bx, by in train_dl:
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        miou_vals = []
        with torch.no_grad():
            for bx, by in test_dl:
                logits = model(bx)
                miou_vals.append(compute_miou(logits, by))

        print(f"Epoch {epoch+1}/{epochs}: loss={sum(losses)/len(losses):.4f}, mIoU={sum(miou_vals)/len(miou_vals):.4f}")

    out_dir = Path("models/weights/prithvi")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "flood_head.pt"
    torch.save(model.state_dict(), ckpt)

    report = {
        "checkpoint": str(ckpt),
        "miou": round(sum(miou_vals) / len(miou_vals), 4),
        "epochs_trained": epochs,
        "train_samples": n_train,
        "test_samples": n_test,
        "note": "Synthetic scaffold run. Replace with FloodNet dataloader for real training.",
    }

    rp = Path("results/stage1_flood_head.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {ckpt}")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
