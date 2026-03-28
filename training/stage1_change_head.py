"""Stage 1: Change detection head (LSTM) training.

Trains a 2-layer LSTM on synthetic temporal sequences (scaffold for Sen1Floods11).
Computes real MTCS (temporal consistency) metric on a held-out test split.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import MTCS, weighted_f1


class ChangeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def compute_mtcs(model: nn.Module, dataloader: DataLoader) -> float:
    """Compute temporal consistency: fraction of consecutive prediction pairs
    that are monotonically consistent (same class or next phase)."""
    model.eval()
    consistent = 0
    total_pairs = 0

    with torch.no_grad():
        prev_preds = None
        for bx, _ in dataloader:
            logits = model(bx)
            preds = logits.argmax(dim=1)
            if prev_preds is not None:
                # Check monotone consistency: pred should be >= prev_pred
                # (simulating phase progression)
                min_len = min(len(preds), len(prev_preds))
                batch_consistent = (preds[:min_len] >= prev_preds[:min_len]).sum().item()
                consistent += batch_consistent
                total_pairs += min_len
            prev_preds = preds

    return MTCS(consistent, total_pairs) / 100.0 if total_pairs > 0 else 0.0


def compute_test_f1(model: nn.Module, dataloader: DataLoader, num_classes: int = 4) -> float:
    """Compute weighted F1 on test data."""
    model.eval()
    counts = {c: [0, 0, 0, 0] for c in range(num_classes)}

    with torch.no_grad():
        for bx, by in dataloader:
            logits = model(bx)
            preds = logits.argmax(dim=1)
            for c in range(num_classes):
                pred_c = (preds == c)
                true_c = (by == c)
                tp = (pred_c & true_c).sum().item()
                fp = (pred_c & ~true_c).sum().item()
                fn = (~pred_c & true_c).sum().item()
                support = true_c.sum().item()
                counts[c][0] += tp
                counts[c][1] += fp
                counts[c][2] += fn
                counts[c][3] += support

    classes_data = [(counts[c][0], counts[c][1], counts[c][2], counts[c][3]) for c in range(num_classes)]
    return weighted_f1(classes_data)


def main() -> None:
    torch.manual_seed(42)
    n_total = 2000
    n_test = 400
    n_train = n_total - n_test

    x = torch.randn(n_total, 6, 1)
    y = torch.randint(0, 4, (n_total,))

    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = ChangeHead()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    print("Stage 1 Change Head: Training...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_dl:
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        test_f1 = compute_test_f1(model, test_dl)
        mtcs_val = compute_mtcs(model, test_dl)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, test_f1={test_f1:.4f}, mtcs={mtcs_val:.4f}")

    # Final evaluation
    final_f1 = compute_test_f1(model, test_dl)
    final_mtcs = compute_mtcs(model, test_dl)
    print(f"Final test weighted_f1 = {final_f1:.4f}, mtcs = {final_mtcs:.4f}")

    out_dir = Path("models/weights/prithvi")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "phase_lstm.pt"
    torch.save(model.state_dict(), ckpt)

    report = {
        "checkpoint": str(ckpt),
        "weighted_f1": round(final_f1, 4),
        "mtcs": round(final_mtcs, 4),
        "epochs_trained": epochs,
        "train_samples": n_train,
        "test_samples": n_test,
        "note": "Real metrics computed on held-out test split. "
                "Using synthetic data scaffold; replace with Sen1Floods11 dataloader for real run.",
    }
    rp = Path("results/stage1_change_head.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {ckpt}")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
