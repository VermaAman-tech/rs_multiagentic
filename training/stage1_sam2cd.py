"""Stage 1: SAM2-CD adapter training scaffold.

Runs a lightweight binary change detection training loop and reports IoU.
Swap synthetic tensors with LEVIR-CD loaders for real training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import IoU


class SAM2CDAdapter(nn.Module):
    def __init__(self, in_ch: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def batch_iou(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5)
    true = labels > 0.5
    tp = (preds & true).sum().item()
    fp = (preds & ~true).sum().item()
    fn = (~preds & true).sum().item()
    return IoU(tp, fp, fn)


def main() -> None:
    torch.manual_seed(42)

    n_total = 256
    n_test = 64
    n_train = n_total - n_test
    h, w = 64, 64

    # 6 channels = concatenated pre/post 3-band images.
    x = torch.randn(n_total, 6, h, w)
    y = torch.randint(0, 2, (n_total, 1, h, w)).float()

    train_ds = TensorDataset(x[:n_train], y[:n_train])
    test_ds = TensorDataset(x[n_train:], y[n_train:])
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = SAM2CDAdapter()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.BCEWithLogitsLoss()

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
        iou_vals = []
        with torch.no_grad():
            for bx, by in test_dl:
                logits = model(bx)
                iou_vals.append(batch_iou(logits, by))

        print(f"Epoch {epoch+1}/{epochs}: loss={sum(losses)/len(losses):.4f}, IoU={sum(iou_vals)/len(iou_vals):.4f}")

    out_dir = Path("models/weights/prithvi")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "sam2_cd_adapter.pt"
    torch.save(model.state_dict(), ckpt)

    report = {
        "checkpoint": str(ckpt),
        "iou": round(sum(iou_vals) / len(iou_vals), 4),
        "epochs_trained": epochs,
        "train_samples": n_train,
        "test_samples": n_test,
        "note": "Synthetic scaffold run. Replace with LEVIR-CD dataloader for real training.",
    }

    rp = Path("results/stage1_sam2cd.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {ckpt}")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
